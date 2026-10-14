"""EMA weight selection and defensive token-to-crystal conversion."""

from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator
import numpy as np
import torch
from src.eval_util.crystal import Crystal
from src.data.mp20_tokens import tokens_to_eval_dict


def _safe_crystal_from_tokens(item: dict[str, Any], sample_idx: int) -> Crystal:
    try:
        return Crystal(tokens_to_eval_dict(item, sample_idx=sample_idx))
    except Exception:
        dummy = {
            "frac_coords": np.zeros((1, 3), dtype=np.float32),
            "atom_types": np.zeros((1,), dtype=np.int64),
            "lengths": np.ones(3, dtype=np.float32),
            "angles": np.ones(3, dtype=np.float32) * 90.0,
            "sample_idx": sample_idx,
        }
        return Crystal(dummy)


@contextmanager
def maybe_use_ema(
    model: torch.nn.Module,
    ema: Any | None,
    use_ema: bool,
) -> Iterator[None]:
    backup = None
    try:
        if use_ema and ema is not None:
            backup = ema.apply(model)
        yield
    finally:
        if backup is not None:
            ema.restore(model, backup)
