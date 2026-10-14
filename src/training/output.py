"""Dependency-light training output namespace guards."""

from __future__ import annotations

from pathlib import Path


def training_output_is_reusable(output_dir: str | Path) -> bool:
    root = Path(output_dir)
    if not root.exists():
        return True
    return root.is_dir() and next(root.iterdir(), None) is None


def guard_training_output(output_dir: str | Path) -> None:
    if not training_output_is_reusable(output_dir):
        raise FileExistsError(
            f"training output namespace is occupied: {output_dir}; "
            "use a fresh run.output_dir"
        )
