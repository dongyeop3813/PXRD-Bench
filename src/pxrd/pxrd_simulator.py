"""Validate the physical grid of stored PXRD profiles."""
from __future__ import annotations
import math
import torch

def validate_two_theta_grid(
    grid: torch.Tensor, *, name: str = "two_theta_grid"
) -> torch.Tensor:
    grid = torch.as_tensor(grid, dtype=torch.float32)
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError(f"{name} must be a 1-D grid with at least two points")
    if not torch.isfinite(grid).all():
        raise ValueError(f"{name} must contain only finite values")
    if not torch.all(grid[1:] > grid[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    return grid.contiguous()


def two_theta_grid_summary(grid: torch.Tensor) -> dict[str, float | int]:
    grid = validate_two_theta_grid(grid)
    return {
        "count": int(grid.numel()),
        "first": float(grid[0]),
        "last": float(grid[-1]),
        "step": float((grid[1:] - grid[:-1]).median()),
    }


def validate_two_theta_bounds(
    configured_min: float,
    configured_max: float,
    declared_bounds: tuple[float, float],
    *,
    name: str = "dataset PXRD",
) -> None:
    declared_min, declared_max = map(float, declared_bounds)
    configured_min = float(configured_min)
    configured_max = float(configured_max)
    if not (
        math.isclose(configured_min, declared_min, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(configured_max, declared_max, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(
            f"configured simulator two-theta bounds [{configured_min}, {configured_max}] "
            f"do not match {name} declared bounds [{declared_min}, {declared_max}]"
        )


