"""Shared PXRD and element-set conditioning for evaluation and sampling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import torch
import torch.nn.functional as F
from src.bragglite import mod1
from src.data.mp20_tokens import VZ
from src.bragglite.sampler import clamp_lattice_latent
from src.models.lattice_repr import lattice_latent_to_y1


@dataclass
class ConditionedBatch:
    """Tensors derived from target token items for conditioned CSP sampling."""

    source_items: Sequence[dict[str, Any]]
    a0_base: torch.Tensor
    pad_base: torch.Tensor
    a0: torch.Tensor
    pad_mask: torch.Tensor
    fixed_atom_types: torch.Tensor
    cond: dict[str, torch.Tensor] | None


def element_set_mask_from_a0(
    a0: torch.Tensor, pad_mask: torch.Tensor, *, vz: int = VZ
) -> torch.Tensor:
    """Return a per-sample allowed-element mask derived from padded A0 tokens."""
    real = ~pad_mask.bool()
    safe = torch.where(
        real, a0.long().clamp(1, int(vz)) - 1, torch.zeros_like(a0.long())
    )
    one_hot = F.one_hot(safe, num_classes=int(vz)).to(dtype=torch.bool)
    return (one_hot & real[..., None]).any(dim=1)


def condition_dict_from_batch(
    batch: dict[str, Any], args: Any
) -> dict[str, torch.Tensor] | None:
    """Model conditioning tensors from a collated training/validation batch."""
    cond: dict[str, torch.Tensor] = {}
    if getattr(args, "pxrd_cond", False):
        cond["pxrd_d"] = batch["pxrd_d"]
        cond["pxrd_i"] = batch["pxrd_i"]
        cond["pxrd_peak_pad_mask"] = batch["pxrd_peak_pad_mask"].bool()
        for key in ("pxrd_position", "pxrd_intensity", "pxrd_prominence", "pxrd_width"):
            if key in batch:
                cond[key] = batch[key]
    if getattr(args, "element_set_cond", False):
        cond["elem_set_mask"] = (
            batch["elem_set_mask"].bool()
            if "elem_set_mask" in batch
            else element_set_mask_from_a0(batch["A0"], batch["pad_mask"], vz=VZ)
        )
    return cond or None


def build_conditioned_batch(
    items: Sequence[dict[str, Any]],
    *,
    type_encoding,
    args: Any,
    device: torch.device,
    lattice_repr: str,
    repeat: int = 1,
    pxrd_cond: bool | None = None,
) -> ConditionedBatch:
    """Build fixed composition and optional PXRD/element conditions.

    ``repeat`` uses the candidate-major layout ``k * B + b``.
    """
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}.")
    a0_base = torch.stack([it["A0"] for it in items]).to(device)
    pad_base = torch.stack([it["pad_mask"] for it in items]).to(device).bool()
    a0 = a0_base.repeat(repeat, 1)
    pad_mask = pad_base.repeat(repeat, 1)
    fixed_atom_types = type_encoding.encode_from_A0(a0=a0, pad_mask=pad_mask)
    pxrd_enabled = (
        bool(getattr(args, "pxrd_cond", False))
        if pxrd_cond is None
        else bool(pxrd_cond)
    )
    cond: dict[str, torch.Tensor] = {}
    if pxrd_enabled:
        cond["pxrd_d"] = (
            torch.stack([it["pxrd_d"] for it in items]).to(device).repeat(repeat, 1)
        )
        cond["pxrd_i"] = (
            torch.stack([it["pxrd_i"] for it in items]).to(device).repeat(repeat, 1)
        )
        cond["pxrd_peak_pad_mask"] = (
            torch.stack([it["pxrd_peak_pad_mask"] for it in items])
            .to(device)
            .bool()
            .repeat(repeat, 1)
        )
        for key in ("pxrd_position", "pxrd_intensity", "pxrd_prominence", "pxrd_width"):
            if key in items[0]:
                cond[key] = (
                    torch.stack([it[key] for it in items]).to(device).repeat(repeat, 1)
                )
    if bool(getattr(args, "element_set_cond", False)):
        if all(("elem_set_mask" in item for item in items)):
            elem_set = torch.stack(
                [torch.as_tensor(item["elem_set_mask"]).bool() for item in items]
            ).to(device)
        else:
            elem_set = element_set_mask_from_a0(a0_base, pad_base, vz=VZ)
        cond["elem_set_mask"] = elem_set.repeat(repeat, 1)
    return ConditionedBatch(
        source_items=items,
        a0_base=a0_base,
        pad_base=pad_base,
        a0=a0,
        pad_mask=pad_mask,
        fixed_atom_types=fixed_atom_types,
        cond=cond or None,
    )


def sampler_output_to_token_items(
    samples: dict[str, torch.Tensor],
    source_items: Sequence[dict[str, Any]],
    *,
    pad_mask: torch.Tensor,
    lattice_repr: str,
    include_mp_id: bool = False,
) -> list[dict[str, Any]]:
    """Convert CSP sampler output tensors into token-item dictionaries."""
    pad_cpu = pad_mask.detach().to(device="cpu", dtype=torch.bool)
    if pad_cpu.shape != samples["frac"].shape[:2]:
        raise ValueError(
            f"pad_mask must match sampled atom dimensions, got {tuple(pad_cpu.shape)} vs {tuple(samples['frac'].shape[:2])}."
        )
    real_cpu = ~pad_cpu
    a0_cpu = torch.stack([it["A0"] for it in source_items]).to("cpu")
    atom_idx = torch.where(real_cpu, a0_cpu, torch.zeros_like(a0_cpu))
    frac = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
    frac = torch.where(real_cpu[..., None], frac, torch.zeros_like(frac))
    lat_y1 = lattice_latent_to_y1(
        clamp_lattice_latent(
            samples["lat"].detach().cpu().clone(), lattice_repr=lattice_repr
        ),
        lattice_repr=lattice_repr,
    )
    lat_y1 = clamp_lattice_latent(lat_y1, lattice_repr="y1")
    items_out: list[dict[str, Any]] = []
    for i, src in enumerate(source_items):
        item = {
            "A0": atom_idx[i],
            "F1": frac[i],
            "Y1": lat_y1[i],
            "pad_mask": pad_cpu[i],
        }
        if include_mp_id:
            item["mp_id"] = src.get("mp_id")
        items_out.append(item)
    return items_out
