"""One-command path from a text corpus to a testable SPALMER checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.checkpoint import load_checkpoint, save_checkpoint
from spalmer.config import SPALMERConfig
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.runtime import generate_tokens, train_token_stream
from spalmer.tokenizer import Encoder, Sample, TrainerConfig, train

_SMOKE_TEXT = (
    "SPALMER routes each token through small experts predicted to be least surprised. "
    "Three KDA layers compress ordinary history and one global MLA layer restores exact "
    "causal access. The model learns next-token prediction from a versioned tokenizer. "
) * 96

_HYBRID_CYCLE = ("kda", "kda", "kda", "mla")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["generate"]:
        args = _generate_parser().parse_args(arguments[1:])
        _run_generate(args)
        return

    parser = _training_parser()
    args = parser.parse_args(arguments)
    if args.smoke and args.text_file is not None:
        parser.error("use either a corpus path or --smoke, not both")
    if not args.smoke and args.text_file is None:
        parser.error("provide a UTF-8 corpus path or use --smoke")
    _run(args)


def _training_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--expert-quant-bits", type=int, default=4)
    parser.add_argument("--potentiation-budget", type=int, default=2)
    parser.add_argument("--potentiation-warmup-steps", type=int, default=4)
    parser.add_argument("--potentiation-hold-steps", type=int, default=8)
    parser.add_argument("--surprise-loss-weight", type=float, default=0.05)
    parser.add_argument("--surprise-ema-decay", type=float, default=0.99)
    parser.add_argument("--residency-increment", type=int, default=2)
    parser.add_argument("--residency-min-gain", type=float, default=0.02)
    parser.add_argument("--ple-expansion", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
            "let the C13 residency controller expand the active expert count from the "
            "configured minimum while the prompt's surprise stays above average"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")
    if args.experts < 2:
        raise ValueError("the SPALMER prototype requires at least two experts")
    if args.layers <= 0 or args.layers % len(_HYBRID_CYCLE):
        raise ValueError("layers must be a positive multiple of 4 for the 3:1 KDA/MLA cycle")

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

    config = SPALMERConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.layers,
        tokenizer_version=vocab.version,
        tokenizer_fingerprint=vocab.fingerprint,
        ple_expansion_factor=args.ple_expansion,
        token_mixer_pattern=_HYBRID_CYCLE,
        surprise_ema_decay=args.surprise_ema_decay,
    )
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
        expert_quant_bits=args.expert_quant_bits,
        potentiation_budget=args.potentiation_budget,
        potentiation_warmup_steps=args.potentiation_warmup_steps,
        potentiation_hold_steps=args.potentiation_hold_steps,
        residency_increment=args.residency_increment,
        residency_min_gain=args.residency_min_gain,
    )
    model = build_spalmer_model(config, kda_config, mla_config, experts_config)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)

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
        },
    )

    prompt = args.prompt or text[: min(80, len(text))]
    prompt_ids = torch.tensor(encoder.encode(prompt, kind=args.kind), dtype=torch.long)
    generated_ids = generate_tokens(model, prompt_ids, max_new_tokens=args.new_tokens)
    payload = b"".join(vocab.get(int(token)).payload() for token in generated_ids[0])
    generated = payload.decode("utf-8", errors="replace")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"saved={destination} parameters={parameters:,} vocab={len(vocab):,} "
        f"tokens_seen={result.tokens_seen:,} seconds={result.elapsed_seconds:.2f} "
        f"promoted={list(result.promoted_experts)} "
        f"average_surprise={result.average_surprise:.4f}"
    )
    print(_terminal_safe_text(generated))


def _run_generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, vocab, metadata = load_checkpoint(args.checkpoint, map_location="cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()

    encoder = Encoder(vocab)
    prompt_kind = args.kind or metadata.get("kind", "mixed")
    prompt_ids = torch.tensor(encoder.encode(args.prompt, kind=prompt_kind), dtype=torch.long)
    if prompt_ids.numel() == 0:
        raise ValueError("prompt must encode to at least one token")
    generated_ids = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=args.new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        dynamic_residency=args.dynamic_residency,
    )
    payload = b"".join(vocab.get(int(token)).payload() for token in generated_ids[0])
    print(_terminal_safe_text(payload.decode("utf-8", errors="replace")))


def _terminal_safe_text(text: str, *, encoding: str | None = None) -> str:
    """Keep arbitrary byte-token generations printable on legacy Windows consoles."""

    target = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(target, errors="backslashreplace").decode(target)


if __name__ == "__main__":
    main()
