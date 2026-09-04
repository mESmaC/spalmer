"""One-command path from a text corpus to a testable SPALMER checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.checkpoint import load_checkpoint, save_checkpoint
from spalmer.config import SPALMERConfig
from spalmer.data import JsonlAdapterConfig, prepare_approved_jsonl
from spalmer.experiment import ExplicitVocabularyPolicy, RecurrenceScaleConfig, plan_ladder
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.precision import detect_precision_capabilities
from spalmer.runtime import RecurrenceTrace, generate_tokens, train_token_stream
from spalmer.tokenizer import Encoder, Sample, TokenizerBackend, TrainerConfig, Vocab, train
from spalmer.tokenizer.backends import (
    HFTokenizerAdapter,
    RPDTokenizerAdapter,
    SpecialTokenIds,
)
from spalmer.training.recurrence import RecurrenceSampler

_SMOKE_TEXT = (
    "SPALMER routes each token through small experts predicted to be least surprised. "
    "Three KDA layers compress ordinary history and one global MLA layer restores exact "
    "causal access. The model learns next-token prediction from a versioned tokenizer. "
) * 96

_HYBRID_CYCLE = ("kda", "kda", "kda", "mla")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["plan"]:
        plan_arguments = arguments[1:]
        args = _plan_parser(device=_requested_device(plan_arguments)).parse_args(plan_arguments)
        _run_plan(args)
        return
    if arguments[:1] == ["prepare-data"]:
        args = _prepare_data_parser().parse_args(arguments[1:])
        _run_prepare_data(args)
        return
    if arguments[:1] == ["precision"]:
        args = _precision_parser().parse_args(arguments[1:])
        _run_precision(args)
        return
    if arguments[:1] == ["generate"]:
        args = _generate_parser().parse_args(arguments[1:])
        _run_generate(args)
        return

    parser = _training_parser(device=_requested_device(arguments))
    args = parser.parse_args(arguments)
    if args.smoke and args.text_file is not None:
        parser.error("use either a corpus path or --smoke, not both")
    if not args.smoke and args.text_file is None:
        parser.error("provide a UTF-8 corpus path or use --smoke")
    _run(args)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _requested_device(arguments: Sequence[str]) -> str:
    """Read ``--device`` before building device-specific precision choices."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=_default_device())
    namespace, _ = parser.parse_known_args(arguments)
    return str(namespace.device)


def _selectable_precision_cli_values(
    device: str | torch.device,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    capabilities = detect_precision_capabilities(device)
    pairs = capabilities.selectable_pairs
    pair_text = ", ".join(
        f"{pair.weight_format}/{pair.activation_format}" for pair in pairs
    ) or "none"
    return (
        capabilities.selectable_weight_formats,
        capabilities.selectable_activation_formats,
        pair_text,
    )


def _resolved_expert_activation(weight_format: str, activation_format: str | None) -> str:
    return activation_format or {
        "bfloat16": "bfloat16",
        "nvfp4": "nvfp4",
    }.get(weight_format, "mxfp8")


def _plan_parser(*, device: str | torch.device | None = None) -> argparse.ArgumentParser:
    resolved_device = _default_device() if device is None else str(device)
    weight_formats, activation_formats, pair_text = _selectable_precision_cli_values(
        resolved_device
    )
    parser = argparse.ArgumentParser(
        prog="spalmer plan",
        description="Plan scale-aware model shapes without constructing or training a model",
    )
    parser.add_argument(
        "targets",
        metavar="TARGET",
        nargs="+",
        type=_parameter_count,
        help="total parameter targets such as 10m 50m 100m",
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocab_sizes",
        metavar="N",
        nargs="+",
        type=int,
        required=True,
        help="one explicit vocabulary size per target, in the same order",
    )
    parser.add_argument(
        "--expert-weight-format",
        choices=weight_formats,
        default="bfloat16",
        help=f"device-verified routed-expert weight format; exact pairs: {pair_text}",
    )
    parser.add_argument(
        "--expert-activation-format",
        choices=activation_formats,
        help="real activation format (default is derived honestly from the weight format)",
    )
    parser.add_argument("--device", default=resolved_device)
    parser.add_argument(
        "--ple-backend",
        choices=("qr",),
        default="qr",
        help="BF16 quotient/remainder codebooks behind one exact input table",
    )
    parser.add_argument(
        "--recurrence",
        metavar=("PRELUDE", "CODA"),
        nargs=2,
        type=int,
        help="plan a depth-recurrent core with this many prelude and coda blocks",
    )
    parser.add_argument(
        "--recurrence-steps",
        type=int,
        default=8,
        help="default inference depth recorded on a recurrent plan",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _prepare_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spalmer prepare-data",
        description="Convert approved JSONL documents into immutable mmap token shards",
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    tokenizer = parser.add_mutually_exclusive_group(required=True)
    tokenizer.add_argument("--tokenizer-rpd", type=Path)
    tokenizer.add_argument("--tokenizer-hf")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--eod-token-id", type=int)
    parser.add_argument("--approved-field")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="document_id")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--kind-field", default="domain")
    parser.add_argument("--language-field", default="lang")
    parser.add_argument("--default-kind", choices=("prose", "code"), default="prose")
    parser.add_argument("--top-code-languages", type=int, default=6)
    parser.add_argument("--documents-per-shard", type=int, default=20_000)
    return parser


def _precision_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spalmer precision",
        description="Report only verified real expert training kernels for this system",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _training_parser(*, device: str | torch.device | None = None) -> argparse.ArgumentParser:
    resolved_device = _default_device() if device is None else str(device)
    weight_formats, activation_formats, pair_text = _selectable_precision_cli_values(
        resolved_device
    )
    capabilities = detect_precision_capabilities(resolved_device)
    promotion_formats = (
        ("mxfp8", "bfloat16")
        if capabilities.supports("mxfp8", "mxfp8")
        else ("bfloat16",)
    )
    parser = argparse.ArgumentParser(
        prog="spalmer",
        description="Train a small SPALMER prototype",
        epilog="Use 'spalmer generate --help' to generate from a saved checkpoint.",
    )
    parser.add_argument("text_file", type=Path, nargs="?")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="train on a built-in sample instead of a corpus file",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/spalmer-prototype.pt"))
    parser.add_argument("--kind", choices=("prose", "code"), default="prose")
    parser.add_argument("--prompt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--expert-width", type=int)
    parser.add_argument(
        "--expert-weight-format",
        choices=weight_formats,
        default="bfloat16",
        help=f"device-verified routed-expert weight format; exact pairs: {pair_text}",
    )
    parser.add_argument(
        "--expert-activation-format",
        choices=activation_formats,
        help="real activation format (default is derived honestly from the weight format)",
    )
    parser.add_argument(
        "--expert-qat-backend",
        choices=("auto", "native"),
        default="auto",
        help="select a verified real kernel or fail; emulation is never used",
    )
    parser.add_argument(
        "--expert-promotion-format",
        choices=promotion_formats,
        default="bfloat16",
        help="device-verified whole-expert promotion precision",
    )
    parser.add_argument("--potentiation-budget", type=int, default=0)
    parser.add_argument("--potentiation-warmup-steps", type=int, default=4)
    parser.add_argument("--potentiation-hold-steps", type=int, default=8)
    parser.add_argument("--surprise-loss-weight", type=float, default=0.05)
    parser.add_argument("--surprise-ema-decay", type=float, default=0.99)
    parser.add_argument("--residency-increment", type=int, default=2)
    parser.add_argument("--residency-min-gain", type=float, default=0.02)
    parser.add_argument("--ple-expansion", type=int, default=2)
    parser.add_argument(
        "--ple-backend",
        choices=("qr",),
        default="qr",
        help="quotient/remainder PLE; emulated checkpoint formats fail closed",
    )
    parser.add_argument(
        "--recurrent-layers",
        metavar=("PRELUDE", "CORE", "CODA"),
        nargs=3,
        type=int,
        help="split --layers into a prelude, an iterated latent core, and a coda",
    )
    parser.add_argument(
        "--mean-recurrence",
        type=float,
        help="expected sampled core iterations per optimizer step (required with "
        "--recurrent-layers)",
    )
    parser.add_argument("--backprop-depth", type=int, default=4)
    parser.add_argument("--recurrence-sigma", type=float, default=0.5)
    parser.add_argument(
        "--default-steps",
        type=int,
        help="inference depth recorded in the checkpoint (default: round(mean recurrence))",
    )
    parser.add_argument("--max-recurrence", type=int)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default=resolved_device)
    return parser


def _generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spalmer generate",
        description="Generate text from a saved SPALMER checkpoint",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--kind",
        choices=("mixed", "prose", "code"),
        help="override the checkpoint corpus kind used to route prompt regions",
    )
    parser.add_argument(
        "--dynamic-residency",
        action="store_true",
        help=(
            "use the experimental legacy prompt-global residency controller instead of "
            "quality-preserving full-pool expert paging"
        ),
    )
    parser.add_argument(
        "--expert-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "keep complete expert banks on CPU and page selected rows into bounded, "
            "layer-local inference caches (default: enabled for all CUDA generation)"
        ),
    )
    parser.add_argument(
        "--expert-cache-size",
        type=int,
        help="maximum expert identities staged per layer (default: checkpoint resident cap)",
    )
    parser.add_argument(
        "--recurrence-steps",
        type=int,
        help="latent-core depth per token (default: the checkpoint's default_steps)",
    )
    parser.add_argument(
        "--adaptive-exit",
        choices=("none", "latent_diff", "kl"),
        default="none",
        help="stop decode iterations early once the chosen criterion settles",
    )
    parser.add_argument("--exit-threshold", type=float)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--exit-state-policy", choices=("fill", "skip"), default="fill")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _run_plan(args: argparse.Namespace) -> None:
    if len(args.targets) != len(args.vocab_sizes):
        raise SystemExit("provide exactly one --vocab-size value for every target")
    if args.targets != sorted(args.targets) or len(set(args.targets)) != len(args.targets):
        raise SystemExit("targets must be unique and listed from smallest to largest")
    if any(size <= 0 for size in args.vocab_sizes):
        raise SystemExit("vocabulary sizes must be positive")
    if len(args.vocab_sizes) > 1 and any(
        right <= left for left, right in zip(args.vocab_sizes, args.vocab_sizes[1:])
    ):
        raise SystemExit("vocabulary size must increase at every larger model target")
    activation_format = _resolved_expert_activation(
        args.expert_weight_format,
        args.expert_activation_format,
    )
    detect_precision_capabilities(args.device).require(
        args.expert_weight_format,
        activation_format,
    )
    recurrence = None
    if args.recurrence is not None:
        prelude, coda = args.recurrence
        recurrence = RecurrenceScaleConfig(
            prelude_layers=prelude,
            coda_layers=coda,
            default_steps=args.recurrence_steps,
        )
    policy = ExplicitVocabularyPolicy(tuple(zip(args.targets, args.vocab_sizes, strict=True)))
    plans = plan_ladder(
        tuple(args.targets),
        policy,
        expert_weight_format=args.expert_weight_format,
        expert_activation_format=activation_format,
        recurrence=recurrence,
        ple_backend=args.ple_backend,
    )
    if args.json:
        print(json.dumps([plan.to_dict() for plan in plans], indent=2))
        return
    header = (
        f"{'target':>12} {'planned':>12} {'error':>9} {'vocab':>8} "
        f"{'d':>5} {'L':>3} {'core':>5} {'eff':>5} {'h':>3} {'E':>4} "
        f"{'expert':>7} {'PLE':>4} {'backend':>8} {'PLE params':>11} "
        f"{'BF16 GiB':>9} {'cached W4':>11}"
    )
    print(header)
    print("-" * len(header))
    for plan in plans:
        shape = plan.config
        memory = plan.memory
        core = "-" if shape.recurrence is None else f"{shape.core_layers:d}"
        print(
            f"{plan.target_parameters:12,d} {plan.parameters.total:12,d} "
            f"{plan.relative_error:+8.2%} {shape.vocab_size:8,d} {shape.d_model:5d} "
            f"{shape.n_layers:3d} {core:>5} {plan.effective_depth:5d} "
            f"{shape.num_heads:3d} {shape.num_experts:4d} "
            f"{shape.expert_width:7d} {shape.ple_expansion:4d} {shape.ple_backend:>8} "
            f"{plan.parameters.ple_total:11,d} "
            f"{memory.gib(memory.reference_training_bytes):9.3f} "
            f"{memory.gib(memory.packed_training_bytes):11.3f}"
        )


def _run_precision(args: argparse.Namespace) -> None:
    capabilities = detect_precision_capabilities(args.device)
    if args.json:
        print(json.dumps(capabilities.to_dict(), indent=2))
        return
    name = capabilities.device_name or "unknown device"
    print(f"device={capabilities.device} name={name}")
    for capability in capabilities.expert_precisions:
        state = "selectable" if capability.selectable else "unavailable"
        grouped = "grouped" if capability.grouped_available else "dense"
        print(
            f"{capability.weight_format}/{capability.activation_format} "
            f"provider={capability.provider_id} state={state} execution={grouped} "
            f"detail={capability.detail}"
        )
    for diagnostic in capabilities.diagnostics:
        print(f"diagnostic: {diagnostic}")


def _run_prepare_data(args: argparse.Namespace) -> None:
    if args.tokenizer_rpd is not None:
        content_adapter = RPDTokenizerAdapter.from_file(args.tokenizer_rpd)
        reserved_eod_id = (
            content_adapter.content_vocab_size
            if args.eod_token_id is None
            else args.eod_token_id
        )
        adapter = RPDTokenizerAdapter(
            content_adapter.vocab,
            special_tokens=SpecialTokenIds(eod_id=reserved_eod_id),
            source=content_adapter.identity.source,
        )
    else:
        adapter = HFTokenizerAdapter.from_pretrained(
            args.tokenizer_hf,
            revision=args.tokenizer_revision,
            local_files_only=args.local_files_only,
            trust_remote_code=False,
        )
    if args.eod_token_id is not None and isinstance(adapter, HFTokenizerAdapter):
        known_special_ids = set(adapter.special_tokens.assigned)
        known_special_ids.update(
            int(token_id)
            for token_id in (getattr(adapter.tokenizer, "all_special_ids", ()) or ())
        )
        if args.eod_token_id not in known_special_ids:
            raise SystemExit(
                "--eod-token-id must identify a token already registered as special by "
                "the Hugging Face tokenizer"
            )
        special = replace(adapter.special_tokens, eod_id=args.eod_token_id)
        adapter = HFTokenizerAdapter.from_tokenizer(
            adapter.tokenizer,
            source=adapter.identity.source,
            revision=adapter.identity.revision,
            artifact_fingerprint=adapter.identity.artifact_fingerprint,
            special_tokens=special,
        )
    prepared = prepare_approved_jsonl(
        [str(path) for path in args.inputs],
        args.output_directory,
        adapter,
        name=args.name,
        adapter_config=JsonlAdapterConfig(
            text_field=args.text_field,
            id_field=args.id_field,
            source_field=args.source_field,
            kind_field=args.kind_field,
            language_field=args.language_field,
            approved_field=args.approved_field,
            default_kind=args.default_kind,
        ),
        top_code_languages=args.top_code_languages,
        documents_per_shard=args.documents_per_shard,
        eod_token_id=args.eod_token_id,
    )
    print(json.dumps(prepared.to_dict(), indent=2))


def _parameter_count(value: str) -> int:
    text = value.strip().lower().replace("_", "").replace(",", "")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = text[-1:] if text else ""
    multiplier = multipliers.get(suffix, 1)
    if multiplier != 1:
        text = text[:-1]
    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid parameter count: {value!r}") from exc
    count = int(parsed * multiplier)
    if count <= 0 or parsed * multiplier != count:
        raise argparse.ArgumentTypeError("parameter counts must resolve to positive integers")
    return count


def _run(args: argparse.Namespace) -> None:
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")
    if args.experts < 2:
        raise ValueError("the SPALMER prototype requires at least two experts")
    if args.layers <= 0 or args.layers % len(_HYBRID_CYCLE):
        raise ValueError("layers must be a positive multiple of 4 for the 3:1 KDA/MLA cycle")
    recurrence_config, sampler = _resolve_training_recurrence(args)
    device = torch.device(args.device)
    activation_format = _resolved_expert_activation(
        args.expert_weight_format,
        args.expert_activation_format,
    )
    capability = detect_precision_capabilities(device).require(
        args.expert_weight_format,
        activation_format,
    )

    if args.smoke:
        text = _SMOKE_TEXT
        source = "built-in-smoke"
        corpus_name = "spalmer-smoke"
    else:
        if not args.text_file.is_file():
            raise SystemExit(
                f"Corpus file not found: {args.text_file}\n"
                "Pass an existing UTF-8 text file, or run with --smoke."
            )
        text = args.text_file.read_text(encoding="utf-8")
        source = str(args.text_file)
        corpus_name = args.text_file.stem
    vocab = train(
        [Sample(text=text, kind=args.kind)],
        TrainerConfig(),
        name=corpus_name,
    )
    encoder = Encoder(vocab)
    token_ids = torch.tensor(encoder.encode(text, kind=args.kind), dtype=torch.long)
    head_dim = args.d_model // args.heads

    model_fields: dict[str, object] = {
        "vocab_size": len(vocab),
        "d_model": args.d_model,
        "n_layers": args.layers,
        "tokenizer_version": vocab.version,
        "tokenizer_fingerprint": vocab.fingerprint,
        "ple_expansion_factor": args.ple_expansion,
        "token_mixer_pattern": _HYBRID_CYCLE,
        "surprise_ema_decay": args.surprise_ema_decay,
    }
    model_fields["ple_backend"] = args.ple_backend
    if recurrence_config is not None:
        model_fields["recurrence"] = recurrence_config
    config = SPALMERConfig(**model_fields)
    kda_config = KDAConfig(
        hidden_size=args.d_model,
        num_heads=args.heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        backend="auto",
    )
    mla_config = MLAConfig(
        hidden_size=args.d_model,
        num_heads=args.heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        q_latent_dim=max(head_dim, args.d_model // 2),
        kv_latent_dim=max(head_dim, args.d_model // 4),
    )
    experts_config = MicroExpertsConfig(
        d_model=args.d_model,
        num_experts=args.experts,
        expert_inter_dim=args.expert_width,
        active_experts=args.active_experts,
        max_active_experts=min(20, args.experts),
        expert_execution="grouped" if capability.grouped_available else "loop",
        expert_weight_format=args.expert_weight_format,
        expert_activation_format=activation_format,
        expert_master_dtype="bfloat16",
        expert_qat_backend=args.expert_qat_backend,
        expert_promotion_format=args.expert_promotion_format,
        potentiation_budget=args.potentiation_budget,
        potentiation_warmup_steps=args.potentiation_warmup_steps,
        potentiation_hold_steps=args.potentiation_hold_steps,
        residency_increment=args.residency_increment,
        residency_min_gain=args.residency_min_gain,
    )
    model = build_spalmer_model(config, kda_config, mla_config, experts_config)
    dtype = torch.bfloat16
    model.to(device=device, dtype=dtype)
    print(
        "expert precision: "
        f"weights={experts_config.expert_weight_format} "
        f"activations={experts_config.expert_activation_format} "
        f"master={experts_config.expert_master_dtype} "
        f"provider={capability.provider_id} "
        f"grouped={capability.grouped_available}",
        flush=True,
    )

    def report(step: int, loss: float) -> None:
        interval = max(1, args.steps // 10)
        if step == 1 or step == args.steps or step % interval == 0:
            print(f"step={step}/{args.steps} loss={loss:.4f}", flush=True)

    result = train_token_stream(
        model,
        token_ids,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        surprise_calibration_weight=args.surprise_loss_weight,
        log=report,
        recurrence=sampler,
    )
    destination = save_checkpoint(
        args.output,
        model,
        vocab,
        kda_config=kda_config,
        mla_config=mla_config,
        experts_config=experts_config,
        metadata={
            "tokens_seen": result.tokens_seen,
            "final_loss": result.losses[-1],
            "final_model_loss": result.final_model_loss,
            "final_auxiliary_loss": result.final_auxiliary_loss,
            "final_surprise_calibration_loss": result.final_surprise_calibration_loss,
            "final_predictive_entropy": result.final_predictive_entropy,
            "promoted_experts": list(result.promoted_experts),
            "average_surprise": result.average_surprise,
            "source": source,
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "kind": args.kind,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "learning_rate": args.learning_rate,
            "surprise_ema_decay": args.surprise_ema_decay,
            "residency_increment": args.residency_increment,
            "residency_min_gain": args.residency_min_gain,
            "seed": 0,
            "ple_backend": config.ple_backend,
            "recurrence": _recurrence_metadata(recurrence_config, sampler, result),
        },
    )

    prompt = args.prompt or text[: min(80, len(text))]
    prompt_ids = torch.tensor(encoder.encode(prompt, kind=args.kind), dtype=torch.long)
    generated_ids = generate_tokens(model, prompt_ids, max_new_tokens=args.new_tokens)
    payload = b"".join(vocab.get(int(token)).payload() for token in generated_ids[0])
    generated = payload.decode("utf-8", errors="replace")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = (
        f"saved={destination} parameters={parameters:,} vocab={len(vocab):,} "
        f"tokens_seen={result.tokens_seen:,} seconds={result.elapsed_seconds:.2f} "
        f"promoted={list(result.promoted_experts)} "
        f"average_surprise={result.average_surprise:.4f}"
    )
    if recurrence_config is not None and sampler is not None:
        summary += (
            f" effective_depth={config.effective_depth()} "
            f"mean_r={result.mean_recurrence_steps:.2f}"
        )
    print(summary)
    print(_terminal_safe_text(generated))


def _resolve_training_recurrence(args: argparse.Namespace) -> tuple[object | None, object | None]:
    """Turn ``--recurrent-layers`` and its companions into config + sampler."""

    if args.recurrent_layers is None:
        for name in ("mean_recurrence", "default_steps", "max_recurrence"):
            if getattr(args, name) is not None:
                raise ValueError(
                    f"--{name.replace('_', '-')} requires --recurrent-layers"
                )
        return None, None
    prelude, core, coda = args.recurrent_layers
    if prelude + core + coda != args.layers:
        raise ValueError(
            f"--recurrent-layers must sum to --layers ({prelude + core + coda} != {args.layers})"
        )
    if args.mean_recurrence is None:
        raise ValueError("--mean-recurrence is required with --recurrent-layers")
    from spalmer.config import RecurrenceConfig

    default_steps = args.default_steps or max(1, round(args.mean_recurrence))
    config = RecurrenceConfig(
        prelude_layers=prelude,
        core_layers=core,
        coda_layers=coda,
        default_steps=default_steps,
    )
    sampler = RecurrenceSampler(
        mean_recurrence=float(args.mean_recurrence),
        mean_backprop_depth=int(args.backprop_depth),
        sigma=float(args.recurrence_sigma),
        max_recurrence=args.max_recurrence,
    )
    return config, sampler


def _recurrence_metadata(
    config: object | None,
    sampler: object | None,
    result: object,
) -> dict[str, object] | None:
    if config is None or sampler is None:
        return None
    return {
        "prelude_layers": config.prelude_layers,
        "core_layers": config.core_layers,
        "coda_layers": config.coda_layers,
        "default_steps": config.default_steps,
        "latent_init_std": config.latent_init_std,
        "mean_recurrence": sampler.mean_recurrence,
        "mean_backprop_depth": sampler.mean_backprop_depth,
        "recurrence_sigma": sampler.sigma,
        "mean_recurrence_steps": result.mean_recurrence_steps,
    }


def _run_generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, map_location="cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # Cast on CPU first so enabling offload never transiently materializes the
    # complete expert pool on the accelerator.
    model.to(dtype=dtype)
    model.eval()
    use_expert_offload = (
        device.type == "cuda" if args.expert_offload is None else args.expert_offload
    )
    if use_expert_offload:
        if device.type == "cpu":
            raise ValueError("--expert-offload requires a non-CPU --device")
        model.enable_expert_offload(
            device,
            cache_size=args.expert_cache_size,
            paging=not args.dynamic_residency,
        )
    else:
        model.to(device=device)

    prompt_kind = args.kind or metadata.get("kind", "mixed")
    if isinstance(tokenizer, Vocab):
        encoded_prompt = Encoder(tokenizer).encode(args.prompt, kind=prompt_kind)
    elif isinstance(tokenizer, TokenizerBackend):
        encoded_prompt = tokenizer.encode(args.prompt, kind=prompt_kind)
    else:  # The checkpoint boundary should make this unreachable.
        raise TypeError(f"unsupported checkpoint tokenizer: {type(tokenizer).__name__}")
    prompt_ids = torch.tensor(encoded_prompt, dtype=torch.long)
    if prompt_ids.numel() == 0:
        raise ValueError("prompt must encode to at least one token")
    recurrence_kwargs, trace = _generate_recurrence_kwargs(args, model)
    generated_ids = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=args.new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        dynamic_residency=args.dynamic_residency,
        **recurrence_kwargs,
    )
    decoded_ids = generated_ids[0].detach().cpu().tolist()
    if isinstance(tokenizer, Vocab):
        payload = b"".join(tokenizer.get(token).payload() for token in decoded_ids)
        generated = payload.decode("utf-8", errors="replace")
    elif isinstance(tokenizer, RPDTokenizerAdapter):
        generated = tokenizer.decode(
            decoded_ids,
            skip_special_tokens=False,
            errors="replace",
        )
    else:
        generated = tokenizer.decode(decoded_ids, skip_special_tokens=False)
    print(_terminal_safe_text(generated))
    if trace is not None:
        print(
            f"recurrence: steps={trace.max_steps} prefill={trace.prefill_steps} "
            f"decode mean={trace.mean_decode_steps:.2f} min={trace.min_decode_steps} "
            f"max={trace.max_decode_steps} early_exits={trace.early_exits}"
        )


_RECURRENCE_GENERATE_FLAGS = (
    ("recurrence_steps", "--recurrence-steps"),
    ("exit_threshold", "--exit-threshold"),
)


def _generate_recurrence_kwargs(
    args: argparse.Namespace,
    model: object,
) -> tuple[dict[str, object], RecurrenceTrace | None]:
    """Build the recurrence keywords, refusing them on a flat checkpoint."""

    requested = [
        flag for name, flag in _RECURRENCE_GENERATE_FLAGS if getattr(args, name) is not None
    ]
    if args.adaptive_exit != "none":
        requested.append("--adaptive-exit")
    if args.warm_start:
        requested.append("--warm-start")
    if getattr(getattr(model, "config", None), "recurrence", None) is None:
        if requested:
            raise SystemExit(
                "checkpoint has no recurrent core; drop " + ", ".join(sorted(requested))
            )
        return {}, None
    kwargs: dict[str, object] = {
        "recurrence_steps": args.recurrence_steps,
        "warm_start": args.warm_start,
    }
    if args.adaptive_exit != "none":
        from spalmer.modeling import AdaptiveExit

        kwargs["adaptive_exit"] = AdaptiveExit(
            criterion=args.adaptive_exit,
            threshold=args.exit_threshold,
            min_steps=args.min_steps,
            state_policy=args.exit_state_policy,
        )
    trace = RecurrenceTrace()
    kwargs["trace"] = trace
    return kwargs, trace


def _terminal_safe_text(text: str, *, encoding: str | None = None) -> str:
    """Keep arbitrary byte-token generations printable on legacy Windows consoles."""

    target = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(target, errors="backslashreplace").decode(target)


if __name__ == "__main__":
    main()
