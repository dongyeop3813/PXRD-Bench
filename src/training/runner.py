"""Shared step-based training lifecycle for Bragglite entrypoints."""

from __future__ import annotations
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import torch
from tqdm import tqdm
from src.training.loop import compute_train_metrics, run_validation, train_step
from src.eval_util.metric_names import monitoring_section
from src.utils.checkpoint import (
    BestCkptState,
    build_val_fallback_candidate,
    maybe_update_best_ckpt,
)
from src.utils.wandb_utils import log_metrics
from src.utils.seeding import capture_rng_state, restore_rng_state

SamplingEvaluationHook = Callable[
    [int, BestCkptState], bool | tuple[bool, float | None]
]
SamplingEvaluationDue = Callable[[int], bool]


def _metric_section(args: Any, split: str) -> str:
    return monitoring_section(
        getattr(args, "dataset_name", "dataset"),
        getattr(args, "real_pxrd_variant", None),
        split,
        weights="raw",
    )


def _qualify_train_losses(
    metrics: dict[str, float], *, section: str
) -> dict[str, float]:
    qualified = {}
    for key, value in metrics.items():
        if key.startswith("loss/"):
            qualified[f"{section}/loss_{key.removeprefix('loss/')}"] = value
        elif key == "gradient/norm":
            qualified[f"{section}/gradient_norm"] = value
        else:
            qualified[key] = value
    return qualified


def _validation_log_payload(
    values: dict[str, float], *, args: Any, weights: dict[str, float]
) -> dict[str, float]:
    section = _metric_section(args, "val")
    payload = {
        f"{section}/loss_total": values["loss_total"],
        f"{section}/loss_coord": values["loss_f"] * weights["F"],
        f"{section}/loss_lattice": values["loss_y"] * weights["Y"],
    }
    if not args.csp:
        payload[f"{section}/loss_type"] = values["loss_a"] * weights["A"]
    return payload


@dataclass
class TrainingRunContext:
    args: Any
    model: torch.nn.Module
    optimizer: Any
    scheduler: Any
    ema: Any
    type_encoding: Any
    weights: dict[str, float]
    device: torch.device
    bf16_dtype: torch.dtype | None
    data_iter: Any
    val_loader: Any
    has_split: bool
    steps_per_epoch: float
    ckpt_dir: Path
    build_ckpt: Callable[[int], dict]
    wandb_enabled: bool
    run_handle: Any = None
    progress_desc: str = "train"
    start_step: int = 0
    distributed: Any = None
    eval_model: torch.nn.Module | None = None
    train_loader: Any = None
    resume_state: dict | None = None
    best_state: BestCkptState = field(default_factory=BestCkptState)


def _build_resumable_checkpoint(ctx: TrainingRunContext, step: int) -> dict:
    checkpoint = ctx.build_ckpt(step)
    optimizer_state = getattr(ctx.optimizer, "state_dict", None)
    scheduler_state = getattr(ctx.scheduler, "state_dict", None)
    if callable(optimizer_state):
        checkpoint["optimizer_state_dict"] = optimizer_state()
    if callable(scheduler_state):
        checkpoint["scheduler_state_dict"] = scheduler_state()
    checkpoint["training_state"] = {
        "rng": capture_rng_state(),
        "best": ctx.best_state.state_dict(),
        "loader": getattr(ctx.train_loader, "checkpoint_state", None)
        if not getattr(ctx.train_loader, "persistent_workers", False)
        and (ctx.distributed is None or not ctx.distributed.enabled)
        else None,
    }
    return checkpoint


def _write_final_checkpoint(ctx: TrainingRunContext) -> None:
    epoch = math.ceil(ctx.args.max_steps / ctx.steps_per_epoch)
    epoch_path = ctx.ckpt_dir / "epoch_latest.pt"
    checkpoint = _build_resumable_checkpoint(ctx, ctx.args.max_steps)
    checkpoint["epoch"] = epoch
    torch.save(checkpoint, epoch_path)
    final_path = ctx.ckpt_dir / "final.pt"
    try:
        if final_path.exists() or final_path.is_symlink():
            final_path.unlink()
        os.link(epoch_path, final_path)
    except OSError:
        try:
            if final_path.exists() or final_path.is_symlink():
                final_path.unlink()
            os.symlink(epoch_path.name, final_path)
        except OSError:
            torch.save(checkpoint, final_path)


def run_training(
    ctx: TrainingRunContext,
    *,
    sampling_evaluation_hook: SamplingEvaluationHook | None = None,
    sampling_evaluation_due: SamplingEvaluationDue | None = None,
) -> None:
    """Run the explicit training lifecycle and optional sampling evaluation."""
    args = ctx.args
    keep_from_step = int(getattr(args, "ckpt_keep_from_step", 0))
    if keep_from_step < 0:
        raise ValueError("ckpt_keep_from_step must be nonnegative")
    distributed = ctx.distributed
    is_main = distributed is None or distributed.is_main
    world_size = 1 if distributed is None else int(distributed.world_size)
    if (
        world_size > 1
        and sampling_evaluation_hook is not None
        and (sampling_evaluation_due is None)
    ):
        raise ValueError(
            "Distributed training requires sampling_evaluation_due so non-main ranks can wait only on evaluation steps."
        )
    if ctx.resume_state:
        ctx.best_state = BestCkptState.from_state_dict(ctx.resume_state["best"])
        if ctx.train_loader is not None and ctx.resume_state.get("loader"):
            ctx.train_loader.resume_state = ctx.resume_state["loader"]
        restore_rng_state(ctx.resume_state["rng"])
    best_state = ctx.best_state
    ctx.model.train()
    perf_started = time.perf_counter()
    perf_step = ctx.start_step
    perf_samples = 0
    progress = tqdm(
        range(ctx.start_step + 1, args.max_steps + 1),
        desc=ctx.progress_desc,
        dynamic_ncols=True,
        disable=not is_main,
    )
    try:
        for step in progress:
            reset_performance_window = False
            performance_clock_synchronized = False
            grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
            microbatches = [next(ctx.data_iter) for _ in range(grad_accum_steps)]
            microbatch_sizes = [
                int(microbatch["A0"].shape[0])
                if isinstance(microbatch, dict) and "A0" in microbatch
                else int(getattr(args, "batch_size", 1))
                for microbatch in microbatches
            ]
            atom_slots = max(
                (
                    int(microbatch["A0"].shape[1])
                    for microbatch in microbatches
                    if isinstance(microbatch, dict) and "A0" in microbatch
                ),
                default=int(getattr(args, "nmax", 0)),
            )
            local_step_samples = sum(microbatch_sizes)
            perf_samples += local_step_samples * world_size
            step_batch = microbatches[0] if grad_accum_steps == 1 else microbatches
            losses, aux = train_step(
                step_batch,
                model=ctx.model,
                optimizer=ctx.optimizer,
                scheduler=ctx.scheduler,
                ema=ctx.ema,
                type_encoding=ctx.type_encoding,
                weights=ctx.weights,
                args=args,
                device=ctx.device,
                bf16_dtype=ctx.bf16_dtype,
                collect_metrics=step % args.log_every == 0 and is_main,
            )
            if step % args.log_every == 0 and is_main:
                if ctx.device.type == "cuda":
                    torch.cuda.synchronize(ctx.device)
                    performance_clock_synchronized = True
                now = time.perf_counter()
                measured_steps = max(1, step - perf_step)
                elapsed = max(now - perf_started, 1e-09)
                metrics = compute_train_metrics(
                    losses,
                    aux,
                    weights=ctx.weights,
                    optimizer=ctx.optimizer,
                    type_encoding=ctx.type_encoding,
                    csp=bool(args.csp),
                )
                metrics["epoch"] = math.ceil(step / ctx.steps_per_epoch)
                metrics["performance/it_s"] = measured_steps / elapsed
                metrics["performance/samples_s"] = perf_samples / elapsed
                metrics["performance/world_size"] = world_size
                metrics["performance/microbatch_size"] = (
                    local_step_samples / grad_accum_steps
                )
                metrics["performance/grad_accum_steps"] = grad_accum_steps
                metrics["performance/local_batch_size"] = local_step_samples
                metrics["performance/global_batch_size"] = (
                    local_step_samples * world_size
                )
                metrics["performance/atom_slots"] = atom_slots
                if ctx.device.type == "cuda":
                    metrics["performance/gpu_max_memory_gb"] = (
                        torch.cuda.max_memory_allocated(ctx.device) / 2**30
                    )
                log_metrics(
                    _qualify_train_losses(
                        metrics, section=_metric_section(args, "train")
                    ),
                    step=step,
                    enabled=ctx.wandb_enabled,
                )
                progress.set_postfix(loss=f"{metrics['loss/total']:.4f}")
                reset_performance_window = True
            if args.val_every > 0 and step % args.val_every == 0 and ctx.has_split:
                plateau_metric = None
                if is_main:
                    values = run_validation(
                        ctx.val_loader,
                        model=ctx.eval_model or ctx.model,
                        type_encoding=ctx.type_encoding,
                        weights=ctx.weights,
                        args=args,
                        device=ctx.device,
                        bf16_dtype=ctx.bf16_dtype,
                    )
                    validation_metrics = _validation_log_payload(
                        values, args=args, weights=ctx.weights
                    )
                    validation_metrics["epoch"] = math.ceil(step / ctx.steps_per_epoch)
                    log_metrics(
                        validation_metrics, step=step, enabled=ctx.wandb_enabled
                    )
                    progress.set_postfix(val_loss=f"{values['loss_total']:.4f}")
                    plateau_metric = float(values["loss_total"])
                    candidate = build_val_fallback_candidate(
                        step=step,
                        epoch=math.ceil(step / ctx.steps_per_epoch),
                        mode="csp" if args.csp else "dng",
                        val_loss=values["loss_total"],
                    )
                    if candidate is not None:
                        maybe_update_best_ckpt(
                            state=best_state,
                            candidate=candidate,
                            maximize=False,
                            ckpt_dir=ctx.ckpt_dir,
                            build_ckpt_fn=ctx.build_ckpt,
                            enabled=args.best_ckpt,
                        )
                    ctx.model.train()
                if (
                    isinstance(
                        ctx.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    )
                    and str(getattr(args, "plateau_metric", "val_loss")) == "val_loss"
                ):
                    if distributed is not None and distributed.enabled:
                        metric_tensor = torch.tensor(
                            [plateau_metric or 0.0], device=ctx.device
                        )
                        torch.distributed.broadcast(metric_tensor, src=0)
                        plateau_metric = float(metric_tensor.item())
                    ctx.scheduler.step(plateau_metric)
                if distributed is not None:
                    distributed.barrier()
                reset_performance_window = True
                performance_clock_synchronized = False
            sampling_due = sampling_evaluation_hook is not None and (
                world_size == 1 or bool(sampling_evaluation_due(step))
            )
            if sampling_due:
                hook_result = (
                    sampling_evaluation_hook(step, best_state)
                    if is_main
                    else (True, None)
                )
                if isinstance(hook_result, tuple):
                    did_run, plateau_metric = hook_result
                else:
                    did_run, plateau_metric = (bool(hook_result), None)
                if (
                    isinstance(
                        ctx.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    )
                    and str(getattr(args, "plateau_metric", "val_loss")) != "val_loss"
                    and did_run
                ):
                    if is_main and plateau_metric is None:
                        raise RuntimeError(
                            "sampling hook did not return a plateau metric"
                        )
                    if distributed is not None and distributed.enabled:
                        metric_tensor = torch.tensor(
                            [plateau_metric or 0.0], device=ctx.device
                        )
                        torch.distributed.broadcast(metric_tensor, src=0)
                        plateau_metric = float(metric_tensor.item())
                    ctx.scheduler.step(plateau_metric)
                    if is_main:
                        print(
                            f"[lr-plateau] step={step} metric={plateau_metric:.6f} lr={ctx.optimizer.param_groups[0]['lr']:.8g}"
                        )
                if distributed is not None:
                    distributed.barrier()
                reset_performance_window = reset_performance_window or did_run
                if did_run:
                    performance_clock_synchronized = False
            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                if is_main:
                    latest_only = args.ckpt_latest_only or step < keep_from_step
                    path = ctx.ckpt_dir / (
                        "step_latest.pt" if latest_only else f"step_{step:07d}.pt"
                    )
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(_build_resumable_checkpoint(ctx, step), temporary)
                    temporary.replace(path)
                if distributed is not None:
                    distributed.barrier()
                reset_performance_window = True
            if reset_performance_window:
                if ctx.device.type == "cuda" and (not performance_clock_synchronized):
                    torch.cuda.synchronize(ctx.device)
                perf_started = time.perf_counter()
                perf_step = step
                perf_samples = 0
        if is_main and bool(getattr(args, "save_final_checkpoint", True)):
            _write_final_checkpoint(ctx)
        if distributed is not None:
            distributed.barrier()
    finally:
        if is_main and ctx.run_handle is not None:
            ctx.run_handle.finish()
