"""Pure quotient/remainder layout helpers shared by planning and execution."""

from __future__ import annotations

import math


def qr_lane_moduli(vocab_size: int, lane_count: int) -> tuple[int, ...]:
    """Return deterministic pairwise-coprime QR moduli near ``sqrt(vocab_size)``.

    The layout uses the first ``lane_count`` ascending primes at or above
    ``ceil(sqrt(vocab_size))``. Distinct primes provide complementary sharing
    partitions while keeping both codebooks approximately square.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if lane_count <= 0:
        raise ValueError("lane_count must be positive")

    root = math.isqrt(vocab_size)
    candidate = root if root * root == vocab_size else root + 1
    candidate = max(candidate, 2)
    moduli: list[int] = []
    while len(moduli) < lane_count:
        if _is_prime(candidate):
            moduli.append(candidate)
        candidate += 1
    return tuple(moduli)


def qr_codebook_rows(vocab_size: int, lane_count: int) -> tuple[int, int]:
    """Return ``(remainder_rows, quotient_rows)`` for the canonical QR layout."""

    moduli = qr_lane_moduli(vocab_size, lane_count)
    return (
        sum(moduli),
        sum(_ceil_div(vocab_size, modulus) for modulus in moduli),
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


__all__ = ["qr_codebook_rows", "qr_lane_moduli"]
