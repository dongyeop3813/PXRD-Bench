"""Fine-tune ordered XDecomposer with additive mixture conservation."""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.two_phase.data import MixturePhasePXRDDataset
from src.data.pxrd_bench import _ShardBatchSampler, _TrainingShardBatchSampler
from src.utils.provenance import file_sha256
from src.two_phase.xdecomposer_adapter import (
    XDecomposerVariant,
    _load_module,
    load_official_model,
    resample_to_xdecomposer,
)
from src.utils.wandb_utils import init_wandb, log_metrics, log_summary


class OrderedXDecomposerDataset(Dataset):
    """Official four-slot target contract: two active followed by two inactive."""

    def __init__(self, root: str | Path, split: str) -> None:
        self.source = MixturePhasePXRDDataset(root, split=split)
        self.indices = [
            index
            for index, fraction in enumerate(self.source.rows["fraction_b"].tolist())
            if abs(float(fraction) - 0.5) > 1e-12
        ]
        self.rows = self.source.rows.iloc[self.indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple:
        item = self.source[self.indices[int(index)]]
        mixture = torch.from_numpy(resample_to_xdecomposer(item["mixture_xrd"].numpy()))
        active = np.stack(
            [
                resample_to_xdecomposer(profile.numpy())
                for profile in item["phase_xrd_contribution"]
            ]
        )
        targets = torch.zeros((4, 3500), dtype=torch.float32)
        targets[:2] = torch.from_numpy(active)
        result = (mixture, targets, int(float(item["fraction_b"]) > 0.5))
        return result


def _pairwise_correlations(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    with torch.autocast(predictions.device.type, enabled=False):
        predicted = predictions.float() - predictions.float().mean(dim=-1, keepdim=True)
        target = targets.float() - targets.float().mean(dim=-1, keepdim=True)
        return torch.bmm(
            F.normalize(predicted, dim=-1, eps=1e-08),
            F.normalize(target, dim=-1, eps=1e-08).transpose(1, 2),
        )


def _matched_correlations(
    predictions: torch.Tensor, activity_logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return Pearson correlations in target order after active-slot PIT alignment."""
    active = torch.argsort(activity_logits, dim=1, stable=True)[:, -2:]
    selected = torch.gather(
        predictions, 1, active.unsqueeze(-1).expand(-1, -1, predictions.shape[-1])
    )
    pairwise = _pairwise_correlations(selected, targets[:, :2])
    straight = torch.stack((pairwise[:, 0, 0], pairwise[:, 1, 1]), dim=1)
    swapped = torch.stack((pairwise[:, 1, 0], pairwise[:, 0, 1]), dim=1)
    use_swapped = swapped.sum(dim=1) > straight.sum(dim=1)
    return torch.where(use_swapped.unsqueeze(1), swapped, straight)


def _ordered_targets(
    targets: torch.Tensor, dominant_index: torch.Tensor
) -> torch.Tensor:
    """Put the phase-fraction-dominant target first and the minor target second."""
    order = torch.stack((dominant_index, 1 - dominant_index), dim=1)
    active = torch.gather(
        targets[:, :2], 1, order.unsqueeze(-1).expand(-1, -1, targets.shape[-1])
    )
    return torch.cat((active, targets[:, 2:]), dim=1)


def _ordered_correlations(
    predictions: torch.Tensor, targets: torch.Tensor, dominant_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct slot-0/slot-1 correlations and whether their order is preferred."""
    targets = _ordered_targets(targets, dominant_index)[:, :2]
    pairwise = _pairwise_correlations(predictions[:, :2], targets)
    direct = torch.stack((pairwise[:, 0, 0], pairwise[:, 1, 1]), dim=1)
    swapped = pairwise[:, 0, 1] + pairwise[:, 1, 0]
    return (direct, direct.sum(dim=1) > swapped)


def _loss(
    model: torch.nn.Module,
    losses: Any,
    mixture: torch.Tensor,
    targets: torch.Tensor,
    dominant_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_inputs = {}
    predictions, activity_logits = model(mixture.unsqueeze(1), **model_inputs)
    targets = _ordered_targets(targets, dominant_index)
    separation_loss = sum(
        (
            losses.calculate_pit_loss(
                predictions[:, slot : slot + 1],
                targets[:, slot : slot + 1],
                alpha=10.0,
                lambda_sisdr=0.1,
                lambda_geo=1.0,
            )[0]
            for slot in range(predictions.shape[1])
        )
    )
    separation_loss = (
        separation_loss + torch.abs((predictions**1).sum(dim=1) - mixture).mean()
    )
    pairwise = _pairwise_correlations(predictions[:, :2], targets[:, :2])
    direct = pairwise.diagonal(dim1=1, dim2=2)
    shape_loss = (1.0 - direct).square().mean()
    separation_loss = separation_loss + 2.0 * shape_loss
    aligned_activity = torch.tensor(
        [1.0, 1.0, 0.0, 0.0], device=targets.device
    ).expand_as(activity_logits)
    activity_loss = F.binary_cross_entropy_with_logits(
        activity_logits, aligned_activity
    )
    return (separation_loss + 2.0 * activity_loss, predictions, activity_logits)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    losses: Any,
    loader: DataLoader,
    device: torch.device,
    *,
    pearson_threshold: float,
    limit: int = 0,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    phase_correlations: list[torch.Tensor] = []
    dominant_correlations: list[torch.Tensor] = []
    minor_correlations: list[torch.Tensor] = []
    ordered_correlations: list[torch.Tensor] = []
    correct_order: list[torch.Tensor] = []
    exact_activity: list[torch.Tensor] = []
    reconstruction_error_sum = 0.0
    reconstruction_error_max = 0.0
    count = 0
    for mixture, targets, dominant_index, *shared in loader:
        if limit and count >= limit:
            break
        if limit and count + len(mixture) > limit:
            keep = limit - count
            mixture, targets, dominant_index = (
                mixture[:keep],
                targets[:keep],
                dominant_index[:keep],
            )
            shared = [mask[:keep] for mask in shared]
        mixture, targets = (mixture.to(device), targets.to(device))
        dominant_index = dominant_index.to(device=device, dtype=torch.long)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss, predictions, activity_logits = _loss(
                model, losses, mixture, targets, dominant_index
            )
        if not torch.isfinite(loss) or not torch.isfinite(predictions).all():
            raise FloatingPointError("non-finite validation loss or profiles")
        correlations = _matched_correlations(
            predictions.float(), activity_logits.float(), targets.float()
        )
        dominant = torch.gather(correlations, 1, dominant_index.unsqueeze(1)).squeeze(1)
        minor = torch.gather(
            correlations, 1, (1 - dominant_index).unsqueeze(1)
        ).squeeze(1)
        phase_correlations.append(correlations.cpu())
        dominant_correlations.append(dominant.cpu())
        minor_correlations.append(minor.cpu())
        direct, is_correct = _ordered_correlations(
            predictions.float(), targets.float(), dominant_index
        )
        ordered_correlations.append(direct.cpu())
        correct_order.append(is_correct.cpu())
        exact_activity.append(
            ((torch.sigmoid(activity_logits) > 0.5).sum(1) == 2).cpu()
        )
        error = (predictions.float().sum(dim=1) - mixture.float()).abs()
        reconstruction_error_sum += float(error.mean(dim=1).sum())
        reconstruction_error_max = max(reconstruction_error_max, float(error.max()))
        loss_sum += float(loss) * len(mixture)
        count += len(mixture)
    if count == 0:
        raise ValueError("validation split is empty")
    phase = torch.cat(phase_correlations)
    dominant = torch.cat(dominant_correlations)
    minor = torch.cat(minor_correlations)
    exact = torch.cat(exact_activity)
    threshold = float(pearson_threshold)
    metrics: dict[str, float | int] = {
        "num_mixtures": count,
        "loss": loss_sum / count,
        "matched_profile_pearson_mean": float(phase.mean()),
        "matched_profile_pearson_median": float(phase.median()),
        "dominant_profile_pearson_mean": float(dominant.mean()),
        "minor_profile_pearson_mean": float(minor.mean()),
        f"phase_profile_pearson@{threshold:g}": float(
            (phase >= threshold).float().mean()
        ),
        f"both_phase_profile_pearson@{threshold:g}": float(
            (phase >= threshold).all(dim=1).float().mean()
        ),
        "activity_count_exact_match": float(exact.float().mean()),
        "mixture_reconstruction_mae": reconstruction_error_sum / count,
        "mixture_reconstruction_max_abs": reconstruction_error_max,
    }
    direct = torch.cat(ordered_correlations)
    metrics.update(
        {
            "ordered_dominant_profile_pearson_mean": float(direct[:, 0].mean()),
            "ordered_minor_profile_pearson_mean": float(direct[:, 1].mean()),
            "ordered_profile_pearson_mean": float(direct.mean()),
            f"ordered_dominant_profile_pearson@{threshold:g}": float(
                (direct[:, 0] >= threshold).float().mean()
            ),
            f"ordered_minor_profile_pearson@{threshold:g}": float(
                (direct[:, 1] >= threshold).float().mean()
            ),
            f"ordered_both_phase_profile_pearson@{threshold:g}": float(
                (direct >= threshold).all(dim=1).float().mean()
            ),
            "dominant_minor_order_accuracy": float(
                torch.cat(correct_order).float().mean()
            ),
        }
    )
    return metrics


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _selection_key(
    metrics: dict[str, float | int], threshold: float
) -> tuple[float, float]:
    prefix = "ordered_"
    return (
        float(metrics[f"{prefix}both_phase_profile_pearson@{threshold:g}"]),
        float(metrics[f"{prefix}profile_pearson_mean"]),
    )


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    loss_options = {"mixture_power": 1, "correlation_weight": 2.0}
    model_options = {"shared_elements": False, "normalize_masks": True}
    if file_sha256(args.separation_checkpoint) != args.separation_sha256:
        raise ValueError("separation checkpoint SHA-256 mismatch")
    if file_sha256(args.mae_checkpoint) != args.mae_sha256:
        raise ValueError("MAE checkpoint SHA-256 mismatch")
    actual_commit = _git_commit(args.official_source)
    if actual_commit != args.official_commit:
        raise ValueError(f"official source commit mismatch: {actual_commit}")
    model, official_config = load_official_model(
        args.official_source,
        args.separation_checkpoint,
        args.mae_checkpoint,
        device=device,
    )
    initial = None
    source_is_variant = False
    if args.initial_checkpoint:
        if file_sha256(args.initial_checkpoint) != args.initial_sha256:
            raise ValueError("initial checkpoint SHA-256 mismatch")
        initial = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=False
        )
        source_options = initial["config"].get("model_options", {})
        source_is_variant = any(source_options.values())
        if source_is_variant and source_options != model_options:
            raise ValueError(
                "initial checkpoint architecture does not match this variant"
            )
        if not source_is_variant:
            model.load_state_dict(initial["model_state_dict"])
    if any(model_options.values()):
        model = XDecomposerVariant(model, **model_options).to(device)
    if source_is_variant:
        model.load_state_dict(initial["model_state_dict"])
    losses = _load_module(
        "xdecomposer_losses", Path(args.official_source) / "src/losses.py"
    )
    train_dataset = OrderedXDecomposerDataset(args.source_root, "train")
    val_dataset = OrderedXDecomposerDataset(args.source_root, "val")
    train_sampler = _TrainingShardBatchSampler(
        train_dataset.rows, args.batch_size, args.seed
    )
    val_sampler = _ShardBatchSampler(
        val_dataset.rows, args.val_batch_size, shuffle=False
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    if not args.resume and output.parent.exists() and any(output.parent.iterdir()):
        raise ValueError(f"output namespace is not empty: {output.parent}")
    latest = output.with_name("latest.pt")
    metrics_path = output.with_name("metrics.jsonl")
    config = {
        "source_root": str(Path(args.source_root).resolve()),
        "train_split": "train",
        "validation_split": "val",
        "test_used": False,
        "train_mixtures": len(train_dataset),
        "validation_mixtures": len(val_dataset),
        "dataset_files_sha256": {
            f"{split}/{name}": file_sha256(Path(args.source_root) / split / name)
            for split in ("train", "val")
            for name in ("samples.parquet", "metadata.json")
        },
        "tie_exclusion": "fraction_b != 0.5",
        "lr": args.lr,
        "min_lr": args.min_lr,
        "lr_patience": args.lr_patience,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "pearson_threshold": args.pearson_threshold,
        "selection_metric": f"ordered_both_phase_profile_pearson@{args.pearson_threshold:g}",
        "selection_tiebreaker": "ordered_profile_pearson_mean",
        "early_stopping_patience": args.patience,
        "active_slots": 2,
        "inactive_slots": 2,
        "ordered_dominant_minor": True,
        "dominance_definition": "phase fraction > 0.5",
        "loss_options": loss_options,
        "model_options": model_options,
        "element_conditioning": "none",
        "parameter_count": sum((parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": sum(
            (p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "scheduler_metric": "selection",
        "official_commit": actual_commit,
        "bragglite_commit": _git_commit(Path.cwd()),
        "source_checkpoint_sha256": args.separation_sha256,
        "mae_checkpoint_sha256": args.mae_sha256,
        "initial_checkpoint": str(Path(args.initial_checkpoint).resolve())
        if args.initial_checkpoint
        else None,
        "initial_checkpoint_sha256": args.initial_sha256,
        "official_config": official_config,
    }
    try:
        wandb_run = init_wandb(
            project="bragglite-xdecomposer",
            name=f"{output.parent.parent.name}-{output.parent.name}",
            config=config,
            enabled=bool(getattr(args, "wandb", False)),
            dir=str(output.parent),
        )
    except Exception as error:
        print(
            f"W&B unavailable; retaining local logs: {type(error).__name__}: {error}",
            flush=True,
        )
        wandb_run = init_wandb(
            project="bragglite-xdecomposer",
            config=config,
            enabled=False,
            dir=str(output.parent),
        )
    completed_steps = 0
    start_epoch = 0
    best_epoch = 0
    best_key = (-math.inf, -math.inf)
    validations_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume["config"] != config:
            raise ValueError("resume config does not match the current run")
        model.load_state_dict(
            resume.get("raw_model_state_dict", resume["model_state_dict"])
        )
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        completed_steps = int(resume["completed_steps"])
        start_epoch = int(resume["epoch"])
        best_epoch = int(resume["best_epoch"])
        best_key = tuple((float(value) for value in resume["best_key"]))
        validations_without_improvement = int(resume["validations_without_improvement"])
        history = list(resume["history"])
        random.setstate(resume["rng_state"]["python"])
        np.random.set_state(resume["rng_state"]["numpy"])
        torch.set_rng_state(resume["rng_state"]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["rng_state"]["cuda"])

    def checkpoint(epoch: int, weights: str = "raw") -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "raw_model_state_dict": model.state_dict(),
            "selected_weights": weights,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_steps": completed_steps,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_key": best_key,
            "validations_without_improvement": validations_without_improvement,
            "history": history,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        }

    if not history:
        initial = validate(
            model,
            losses,
            val_loader,
            device,
            pearson_threshold=args.pearson_threshold,
            limit=args.val_limit,
        )
        best_key = _selection_key(initial, args.pearson_threshold)
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "lr": args.lr,
                "validation": initial,
                "validation_raw": initial,
                "selected_weights": "raw",
            }
        )
        _atomic_save(checkpoint(0), output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(history[-1]) + "\n", encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)
        log_metrics(
            {f"xdecomposer_val_raw/{key}": value for key, value in initial.items()},
            step=0,
            enabled=wandb_run is not None,
        )
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        count = 0
        for mixture, targets, dominant_index, *shared in train_loader:
            mixture, targets = (mixture.to(device), targets.to(device))
            dominant_index = dominant_index.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss, _, _ = _loss(model, losses, mixture, targets, dominant_index)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at step {completed_steps}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(mixture)
            count += len(mixture)
            completed_steps += 1
            if completed_steps % args.log_every_steps == 0:
                log_metrics(
                    {
                        "xdecomposer_train_raw/loss": float(loss.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=completed_steps,
                    enabled=wandb_run is not None,
                )
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={completed_steps} loss={float(loss.detach()):.6f}",
                    flush=True,
                )
            if args.max_steps and completed_steps >= args.max_steps:
                break
        metrics = validate(
            model,
            losses,
            val_loader,
            device,
            pearson_threshold=args.pearson_threshold,
            limit=args.val_limit,
        )
        raw_metrics = metrics
        raw_key = _selection_key(metrics, args.pearson_threshold)
        scheduler.step(raw_key[0])
        selected_weights = "raw"
        current_key = raw_key
        improved = current_key > best_key
        if improved:
            best_key = current_key
            best_epoch = epoch + 1
            validations_without_improvement = 0
        else:
            validations_without_improvement += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(count, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "validation": metrics,
                "validation_raw": raw_metrics,
                "selected_weights": selected_weights,
                "selected_best": improved,
            }
        )
        _atomic_save(checkpoint(epoch + 1), latest)
        if improved:
            _atomic_save(checkpoint(epoch + 1, selected_weights), output)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")
        print(json.dumps(history[-1]), flush=True)
        log_metrics(
            {
                **{
                    f"xdecomposer_val_raw/{key}": value
                    for key, value in raw_metrics.items()
                },
            },
            step=completed_steps,
            enabled=wandb_run is not None,
        )
        if args.max_steps and completed_steps >= args.max_steps:
            stop_reason = "max_steps"
            break
        if validations_without_improvement >= args.patience:
            stop_reason = "early_stopping"
            break
    best = torch.load(output, map_location="cpu", weights_only=False)
    manifest = {
        "status": "complete",
        "stop_reason": stop_reason,
        "best_checkpoint": str(output.resolve()),
        "best_checkpoint_sha256": file_sha256(output),
        "latest_checkpoint": str(latest.resolve()),
        "latest_checkpoint_sha256": file_sha256(latest) if latest.exists() else None,
        "completed_steps": completed_steps,
        "completed_epochs": history[-1]["epoch"],
        "best_epoch": best["best_epoch"],
        "selected_weights": best["selected_weights"],
        "best_validation": best["history"][best["best_epoch"]]["validation"],
        "history": history,
        "resolved_config": config,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log_summary(
        {
            "status": "complete",
            "best_epoch": manifest["best_epoch"],
            "best_validation": manifest["best_validation"],
            "stop_reason": stop_reason,
        },
        enabled=wandb_run is not None,
    )
    if wandb_run is not None:
        wandb_run.finish()
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=3e-06)
    parser.add_argument("--min-lr", type=float, default=1e-07)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--pearson-threshold", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable optional W&B logging"
    )
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--protocol", type=Path)
    preliminary, _ = preliminary_parser.parse_known_args(argv)
    if preliminary.protocol is None:
        parser.parse_args(argv)  # Handle --help or report the missing protocol.
    from src.two_phase.predictions import load_protocol

    config = load_protocol(preliminary.protocol)
    allowed = {
        "lr",
        "min_lr",
        "lr_patience",
        "epochs",
        "patience",
        "seed",
        "batch_size",
        "val_batch_size",
        "pearson_threshold",
        "initial_checkpoint",
    }
    if set(config["training"]) - allowed:
        parser.error("unknown two-phase training option in protocol")
    parser.set_defaults(
        **{
            key: value
            for key, value in config["training"].items()
            if key != "initial_checkpoint"
        }
    )
    args = parser.parse_args(argv)
    args.source_root = Path(config["source_root"])
    args.official_source = Path(config["official_source"])
    args.official_commit = config["official_commit"]
    for prefix, key in (("separation", "release"), ("mae", "mae")):
        setattr(args, prefix + "_checkpoint", Path(config[key]["path"]))
        setattr(args, prefix + "_sha256", config[key]["sha256"])
    initial = config["training"].get("initial_checkpoint")
    args.initial_checkpoint = Path(initial["path"]) if initial else None
    args.initial_sha256 = initial["sha256"] if initial else None
    if args.epochs <= 0 or args.batch_size <= 0 or args.val_batch_size <= 0:
        parser.error("epochs and batch sizes must be positive")
    if args.patience <= 0 or args.lr_patience < 0 or args.log_every_steps <= 0:
        parser.error("patience and log interval are invalid")
    if not -1.0 <= args.pearson_threshold <= 1.0:
        parser.error("--pearson-threshold must be in [-1, 1]")
    if bool(args.initial_checkpoint) != bool(args.initial_sha256):
        parser.error(
            "--initial-checkpoint and --initial-sha256 must be provided together"
        )
    if not (
        math.isfinite(args.lr)
        and math.isfinite(args.min_lr)
        and (0 < args.min_lr <= args.lr)
    ):
        parser.error("learning rates must be finite and 0 < min_lr <= lr")
    return args


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2))


if __name__ == "__main__":
    main()
