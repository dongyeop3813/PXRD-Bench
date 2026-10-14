"""Shared training runtime setup and canonical output directories."""

from __future__ import annotations
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from src.training.output import guard_training_output

_CLAIM_FILE = ".bragglite-training-claim"
_HYDRA_METADATA = ("config.yaml", "hydra.yaml", "overrides.yaml")


@dataclass(frozen=True)
class TrainingPaths:
    root: Path
    checkpoints: Path
    hydra: Path
    logs: Path
    samples: Path
    artifacts: Path
    wandb: Path

    @classmethod
    def from_root(cls, output_dir: str | Path) -> "TrainingPaths":
        root = Path(output_dir)
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            hydra=root / "hydra",
            logs=root / "logs",
            samples=root / "samples",
            artifacts=root / "artifacts",
            wandb=root / "wandb",
        )

    @classmethod
    def create(cls, output_dir: str | Path) -> "TrainingPaths":
        guard_training_output(output_dir)
        paths = cls.from_root(output_dir)
        paths.root.mkdir(parents=True, exist_ok=True)
        try:
            with (paths.root / _CLAIM_FILE).open("x"):
                pass
        except FileExistsError as exc:
            raise FileExistsError(
                f"training output namespace is occupied: {output_dir}; use a fresh run.output_dir"
            ) from exc
        for path in (
            paths.checkpoints,
            paths.hydra,
            paths.logs,
            paths.samples,
            paths.artifacts,
            paths.wandb,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return paths


def copy_hydra_metadata(
    paths: TrainingPaths, source_dir: str | Path | None = None
) -> None:
    """Copy Hydra's pre-task metadata into an atomically claimed training run."""
    if source_dir is None:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        source_dir = Path(hydra_cfg.runtime.output_dir) / str(hydra_cfg.output_subdir)
    source = Path(source_dir)
    for name in _HYDRA_METADATA:
        shutil.copy2(source / name, paths.hydra / name)


def configure_device(args: Any, *, local_rank: int | None = None) -> torch.device:
    requested = str(args.device)
    use_cuda = torch.cuda.is_available() and requested.startswith("cuda")
    if use_cuda and local_rank is not None:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested if use_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not bool(getattr(args, "deterministic", False))
        torch.set_float32_matmul_precision("high")
    elif requested.startswith("cuda"):
        print("[device] CUDA unavailable; falling back to CPU.")
    return device


def maybe_compile_model(model: torch.nn.Module, args: Any) -> torch.nn.Module:
    enabled = bool(getattr(args, "compile", False))
    if not enabled or not any((parameter.is_cuda for parameter in model.parameters())):
        return model
    mode = str(getattr(args, "compile_mode", "default"))
    if mode == "max-autotune" and (
        int(getattr(args, "grad_accum_steps", 1)) > 1
        or (bool(False) and float(0.0) > 0.0)
    ):
        mode = "max-autotune-no-cudagraphs"
    print(f"[compile] torch.compile(model, mode={mode!r})")
    return torch.compile(model, mode=mode)
