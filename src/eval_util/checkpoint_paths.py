"""Dependency-light checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint_path(
    checkpoint: str | Path | None,
    train_output_dir: str | Path | None = None,
    preference: str = "auto",
) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if not train_output_dir:
        raise ValueError("Provide either checkpoint or train_output_dir.")
    ckpt_dir = Path(train_output_dir) / "checkpoints"
    names = (
        ("best.pt", "final.pt", "step_latest.pt", "epoch_latest.pt")
        if preference == "auto"
        else (f"{preference}.pt",)
    )
    for name in names:
        path = ckpt_dir / name
        if path.exists():
            return path
    looked = ", ".join(str(ckpt_dir / name) for name in names)
    raise FileNotFoundError(f"Could not resolve checkpoint. Looked at: {looked}")
