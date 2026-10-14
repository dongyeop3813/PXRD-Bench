"""Optimizer + LR schedule for Bragglite training.

Extracted from ``train_bragglite.py``. Default schedule is linear warmup
followed by cosine decay to zero (legacy behavior). ``optim.lr_schedule=constant``
keeps the LR flat at ``optim.lr`` after warmup (no decay) — for very long runs.
"""

from __future__ import annotations

import math

import torch


def build_optimizer_and_scheduler(model, args, *, start_step: int = 0):
    """Build Adam/AdamW and the configured LR scheduler.

    Returns ``(optimizer, scheduler)``.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("optimizer requires at least one trainable parameter")
    fused = bool(getattr(args, "fused_optimizer", True)) and any(
        parameter.is_cuda for parameter in parameters
    )
    optimizer_name = str(getattr(args, "optimizer", "adamw")).strip().lower()
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(
        optimizer_name
    )
    if optimizer_cls is None:
        raise ValueError(
            f"optim.optimizer must be 'adam' or 'adamw', got {optimizer_name!r}."
        )
    optimizer = optimizer_cls(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(getattr(args, "betas", (0.9, 0.999))),
        eps=float(getattr(args, "eps", 1e-8)),
        fused=fused,
    )
    warmup_steps = max(0, args.lr_warmup_steps)
    warmup_start_step = getattr(args, "lr_warmup_start_step", None)
    warmup_start_lr = float(getattr(args, "lr_warmup_start_lr", 0.0))
    if warmup_start_step is not None:
        warmup_start_step = int(warmup_start_step)
        if warmup_start_step < 0:
            raise ValueError("optim.lr_warmup_start_step must be non-negative")
        if not 0.0 <= warmup_start_lr <= float(args.lr):
            raise ValueError("optim.lr_warmup_start_lr must be between 0 and optim.lr")
    max_steps = max(1, args.max_steps)
    if warmup_steps >= max_steps:
        warmup_steps = max_steps - 1

    schedule = str(getattr(args, "lr_schedule", "cosine")).strip().lower()
    if schedule == "plateau":
        plateau_metric = str(getattr(args, "plateau_metric", "val_loss"))
        return optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if plateau_metric == "val_loss" else "max",
            factor=float(getattr(args, "plateau_factor", 0.6)),
            patience=int(getattr(args, "plateau_patience", 30)),
            min_lr=float(getattr(args, "plateau_min_lr", 1e-4)),
        )
    if schedule not in {"cosine", "constant"}:
        raise ValueError(
            "optim.lr_schedule must be cosine, constant, or plateau, "
            f"got {schedule!r}."
        )

    cosine_decay_steps = getattr(args, "cosine_decay_steps", None)
    cosine_start_step = int(getattr(args, "cosine_start_step", 0))
    cosine_min_lr = float(getattr(args, "cosine_min_lr", 0.0))
    if cosine_decay_steps is not None:
        cosine_decay_steps = int(cosine_decay_steps)
        if cosine_decay_steps <= 0:
            raise ValueError("optim.cosine_decay_steps must be positive")
        if cosine_start_step < 0:
            raise ValueError("optim.cosine_start_step must be non-negative")
        if not 0.0 <= cosine_min_lr <= float(args.lr):
            raise ValueError("optim.cosine_min_lr must be between 0 and optim.lr")

    def lr_lambda(step: int) -> float:
        if (
            warmup_start_step is not None
            and warmup_steps > 0
            and step < warmup_start_step + warmup_steps
        ):
            progress = max(0.0, (step - warmup_start_step) / warmup_steps)
            start_factor = warmup_start_lr / float(args.lr)
            return start_factor + (1.0 - start_factor) * progress
        if warmup_start_step is None and warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if schedule == "constant":
            # Flat LR after warmup; no decay even past max_steps.
            return 1.0
        if cosine_decay_steps is not None:
            progress = min(
                1.0,
                max(0.0, (step - cosine_start_step) / cosine_decay_steps),
            )
            floor = cosine_min_lr / float(args.lr)
            return floor + (1.0 - floor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        if step >= max_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    start_step = max(0, int(start_step))
    if start_step:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=start_step - 1,
    )
    return optimizer, scheduler
