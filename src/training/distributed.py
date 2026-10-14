"""Minimal single-node distributed runtime for Bragglite training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and synchronization helpers for a torchrun job."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(args: Any) -> DistributedContext:
    """Initialize NCCL when launched by torchrun; otherwise stay single-process."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistributedContext()
    if not str(args.device).startswith("cuda"):
        raise SystemExit(
            "config error: multi-process training requires run.device=cuda."
        )
    if not torch.cuda.is_available():
        raise SystemExit("config error: torchrun requested but CUDA is unavailable.")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise SystemExit(
            "config error: LOCAL_RANK is outside the visible CUDA device range: "
            f"local_rank={local_rank}, visible={torch.cuda.device_count()}."
        )
    torch.cuda.set_device(local_rank)
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(
                seconds=float(getattr(args, "ddp_timeout_seconds", 600.0))
            ),
        )
        initialized_here = True
    context = DistributedContext(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        initialized_here=initialized_here,
    )
    if context.is_main:
        print(f"[ddp] initialized single-node NCCL: world_size={context.world_size}")
    return context


def wrap_distributed_model(
    model: torch.nn.Module,
    *,
    context: DistributedContext,
    args: Any,
) -> torch.nn.Module:
    """Wrap a CUDA model with throughput-oriented DDP defaults."""

    if not context.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=bool(getattr(args, "ddp_find_unused_parameters", False)),
        gradient_as_bucket_view=True,
        static_graph=bool(getattr(args, "ddp_static_graph", True)),
    )
