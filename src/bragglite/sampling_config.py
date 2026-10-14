"""Dependency-light scalar sampler configuration helpers."""

from __future__ import annotations

import math


def resolve_nonnegative_scalar(
    name: str, value: float | None, default: float = 0.0
) -> float:
    scalar = float(default) if value is None else float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}.")
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0, got {scalar}.")
    return scalar


def resolve_positive_scalar(
    name: str, value: float | None, default: float
) -> float:
    scalar = float(default) if value is None else float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}.")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0, got {scalar}.")
    return scalar


def resolve_aa_rho_pair(
    aa_rho_coords: float | None,
    aa_rho_lattice: float | None,
) -> tuple[float, float]:
    return (
        resolve_nonnegative_scalar("aa_rho_coords", aa_rho_coords),
        resolve_nonnegative_scalar("aa_rho_lattice", aa_rho_lattice),
    )
