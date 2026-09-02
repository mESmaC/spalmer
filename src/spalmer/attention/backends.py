"""Execution backends for the KDA recurrence.

- :class:`ReferenceKdaBackend`: plain-PyTorch loop (CPU-friendly, fp32/fp64
  compute, differentiable, exact ledger semantics). Used for correctness and
  smoke tests.
- :class:`FlaKdaBackend`: guarded adapter around fla-core's ``chunk_kda``
  prefill kernel and its recurrent decode kernel when the head width is safe.
  It does
  NOT reimplement any kernel; if fla-core is unavailable it raises a clear
  error instead of silently falling back.

Both backends consume raw ``f_proj`` output and raw ``b_proj`` logits so the
gate math lives in exactly one place per backend (the fla kernels apply
``A_log``/``dt_bias``/sigmoid internally when ``use_gate_in_kernel`` and
``use_beta_sigmoid_in_kernel`` are set).
"""

from __future__ import annotations

from typing import Protocol

import torch

from spalmer.attention.recurrence import compute_forget_gate, kda_recurrent_reference

_MIN_FLA_VERSION_HINT = "pip install -U fla-core (>= 0.4)"


def fla_available() -> bool:
    """True if ``fla.ops.kda`` is importable in this environment."""

    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda  # noqa: F401
    except Exception:
        return False
    return True


class KdaBackend(Protocol):
    """Backend contract: sequence-level and single-step KDA recurrence."""

    name: str

    def sequence(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        beta_raw: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor | None,
        lower_bound: float | None,
        allow_neg_eigval: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def step(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        beta_raw: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor | None,
        lower_bound: float | None,
        allow_neg_eigval: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ReferenceKdaBackend:
    """Plain-PyTorch reference path (see :mod:`spalmer.attention.recurrence`)."""

    name = "reference"

    def sequence(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        beta_raw: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor | None,
        lower_bound: float | None,
        allow_neg_eigval: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = compute_forget_gate(f, A_log, dt_bias, lower_bound)
        return kda_recurrent_reference(
            q, k, v, g, beta_raw, state, allow_neg_eigval=allow_neg_eigval
        )

    step = sequence


class FlaKdaBackend:
    """Adapter around fla-core's optimized KDA kernels (no kernel rewrites).

    Requires the current fla-core KDA API (``fla-core >= 0.4``): the kernels
    take raw ``g``/``beta`` with ``use_gate_in_kernel`` /
    ``use_beta_sigmoid_in_kernel`` flags and ``A_log``/``dt_bias`` tensors.
    SPALMER keeps ``dt_bias`` as ``[H, K]`` for the reference implementation,
    while fla-core's fused kernels require the same values flattened to
    ``[H * K]``.  The adapter performs that shape-only conversion at the
    boundary so gradients still return to the canonical parameter shape.
    SPALMER keeps the ordinary ``[B, H, K, V]`` state layout, so
    ``state_v_first`` remains disabled at this boundary.
    """

    name = "fla"

    def __init__(self) -> None:
        try:
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
        except Exception as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "FlaKdaBackend requested but fla.ops.kda is not importable "
                f"({_MIN_FLA_VERSION_HINT}). Original error: {exc}"
            ) from exc
        self._chunk_kda = chunk_kda
        self._fused_recurrent_kda = fused_recurrent_kda

    def _run(self, kernel, **kwargs):
        try:
            return kernel(**kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "fla-core was imported but its KDA kernel signature differs "
                f"from the expected API ({_MIN_FLA_VERSION_HINT}). "
                f"Original error: {exc}"
            ) from exc

    def sequence(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        beta_raw: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor | None,
        lower_bound: float | None,
        allow_neg_eigval: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run(
            self._chunk_kda,
            q=q,
            k=k,
            v=v,
            g=f,
            beta=beta_raw,
            A_log=A_log,
            dt_bias=dt_bias.reshape(-1),
            initial_state=None if state is None else state.float(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
            lower_bound=lower_bound,
            state_v_first=False,
        )

    def step(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        beta_raw: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor | None,
        lower_bound: float | None,
        allow_neg_eigval: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_width = q.shape[-1]
        # fla-core 0.5.2's fused recurrent KDA kernel loads its power-of-two
        # padded gate lanes without a mask.  Non-power-of-two head widths can
        # therefore ingest out-of-bounds values and intermittently produce NaN
        # states.  chunk_kda accepts T=1 with the same FP32 state contract and
        # is the stable fallback; retain the faster fused path when no padded
        # lanes exist.
        kernel = (
            self._fused_recurrent_kda
            if key_width > 0 and key_width & (key_width - 1) == 0
            else self._chunk_kda
        )
        return self._run(
            kernel,
            q=q,
            k=k,
            v=v,
            g=f,
            beta=beta_raw,
            A_log=A_log,
            dt_bias=dt_bias.reshape(-1),
            initial_state=None if state is None else state.float(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
            lower_bound=lower_bound,
            state_v_first=False,
        )


def resolve_backend(name: str, device: torch.device) -> KdaBackend:
    """Resolve a backend name against the runtime environment.

    ``auto`` picks fla only when it is importable AND the tensors live on
    CUDA (fla kernels are Triton/CUDA); otherwise the reference path runs.
    """

    if name == "reference":
        return ReferenceKdaBackend()
    if name == "fla":
        return FlaKdaBackend()
    if name == "auto":
        if device.type == "cuda" and fla_available():
            return FlaKdaBackend()
        return ReferenceKdaBackend()
    raise ValueError(f"unknown backend {name!r}")
