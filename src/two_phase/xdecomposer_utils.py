"""Shared XDecomposer model/inference helpers for released experiment paths.

No dataset generation, ID selection, splitting or training entrypoint lives here.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import torch
from pymatgen.core.periodic_table import Element
from src.data.mp20_tokens import VZ


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _shared_element_mask(system: str) -> torch.Tensor:
    mask = torch.zeros(VZ, dtype=torch.bool)
    for symbol in str(system).split("-"):
        mask[int(Element(symbol).Z) - 1] = True
    if not mask.any():
        raise ValueError(f"empty chemical system: {system!r}")
    return mask
