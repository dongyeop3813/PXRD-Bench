"""Shared runtime for PXRD-conditioned joint atom/structure reconstruction."""

from __future__ import annotations
from collections import Counter
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from src.bragglite import mod1
from src.bragglite.sampler import clamp_lattice_latent, edm_sampler
from src.pxrd.model_runtime import _safe_crystal_from_tokens
from src.models.lattice_repr import (
    lattice_latent_to_y1,
)
from src.pxrd.conditioning import element_set_mask_from_a0
from src.pxrd.eval_metrics import score_crystal_batches


def formula_key(
    a0: torch.Tensor, pad_mask: torch.Tensor
) -> tuple[tuple[int, int], ...]:
    atoms = a0[~pad_mask.bool()].long().tolist()
    return tuple(sorted(Counter((int(z) for z in atoms if int(z) > 0)).items()))


def build_joint_condition(
    items: Sequence[dict[str, Any]], *, model_args: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build PXRD condition and the supplied or formula-derived element mask.

    The element mask is always returned for diagnostics and optional hard
    decoding, but it enters the model only for element-set-conditioned runs.
    """
    a0 = torch.stack([it["A0"] for it in items])
    pad_mask = torch.stack([it["pad_mask"] for it in items])
    supplied = ["elem_set_mask" in item for item in items]
    if any(supplied) and (not all(supplied)):
        raise ValueError("explicit element masks must be present for every item")
    elem_set_mask = (
        torch.stack([torch.as_tensor(item["elem_set_mask"]).bool() for item in items])
        if all(supplied)
        else element_set_mask_from_a0(a0, pad_mask)
    )
    cond = {
        "pxrd_d": torch.stack([it["pxrd_d"] for it in items]).to(device),
        "pxrd_i": torch.stack([it["pxrd_i"] for it in items]).to(device),
        "pxrd_peak_pad_mask": torch.stack([it["pxrd_peak_pad_mask"] for it in items])
        .to(device)
        .bool(),
    }
    for key in ("pxrd_position", "pxrd_intensity", "pxrd_prominence", "pxrd_width"):
        if key in items[0]:
            cond[key] = torch.stack([it[key] for it in items]).to(device)
    if bool(model_args.get("element_set_cond", False)):
        cond["elem_set_mask"] = elem_set_mask.to(device)
    return (cond, elem_set_mask)


def decode_joint_atom_types(
    type_values: torch.Tensor,
    pad_mask: torch.Tensor,
    elem_set_mask: torch.Tensor,
    *,
    type_encoding,
    use_element_set: bool,
) -> torch.Tensor:
    """Decode continuous type channels, optionally enforcing the element set."""
    decoded = type_encoding.decode_logits_to_A0(
        type_values, pad_mask, allowed_mask=elem_set_mask if use_element_set else None
    )
    return torch.where(~pad_mask.bool(), decoded, torch.zeros_like(decoded))


def prototype_assignment_diagnostics(
    type_values: torch.Tensor,
    pad_mask: torch.Tensor,
    elem_set_mask: torch.Tensor,
    *,
    type_encoding,
) -> dict[str, torch.Tensor]:
    """Compare unrestricted and element-set-constrained prototype assignments."""
    vz = int(type_encoding.vz)
    if hasattr(type_encoding, "snap_table"):
        normalized = F.normalize(type_values.float(), p=2, dim=-1, eps=1e-08)
        scores = normalized @ type_encoding.snap_table.to(type_values.device).T
        max_z = int(type_encoding.max_z)
    else:
        scores = type_values.float()[..., :vz]
        max_z = vz
    scores = scores[..., :vz]
    allowed = elem_set_mask.to(device=scores.device, dtype=torch.bool).clone()
    if max_z < vz:
        allowed[:, max_z:] = False
    valid_elements = torch.arange(vz, device=scores.device)[None, :] < max_z
    disallowed = valid_elements & ~allowed
    unrestricted_score, unrestricted_idx = scores.max(dim=-1)
    allowed_scores = scores.masked_fill(~allowed[:, None, :], -torch.inf)
    allowed_score, allowed_idx = allowed_scores.max(dim=-1)
    disallowed_scores = scores.masked_fill(~disallowed[:, None, :], -torch.inf)
    disallowed_score, disallowed_idx = disallowed_scores.max(dim=-1)
    real = ~pad_mask.to(device=scores.device, dtype=torch.bool)
    unrestricted_a0 = torch.where(real, unrestricted_idx + 1, 0)
    allowed_a0 = torch.where(real, allowed_idx + 1, 0)
    out_of_set = real & ~allowed.gather(1, unrestricted_idx)
    margin = torch.where(
        out_of_set,
        unrestricted_score - allowed_score,
        torch.zeros_like(unrestricted_score),
    )
    allowed_vs_disallowed_margin = torch.where(
        real, allowed_score - disallowed_score, torch.zeros_like(allowed_score)
    )
    return {
        "unrestricted_a0": unrestricted_a0,
        "allowed_a0": allowed_a0,
        "unrestricted_score": unrestricted_score,
        "allowed_score": allowed_score,
        "best_disallowed_a0": torch.where(real, disallowed_idx + 1, 0),
        "best_disallowed_score": disallowed_score,
        "out_of_set": out_of_set,
        "out_of_set_margin": margin,
        "allowed_vs_disallowed_margin": allowed_vs_disallowed_margin,
    }


def sample_joint_items(
    *,
    model,
    items: Sequence[dict[str, Any]],
    type_encoding,
    model_args: dict[str, Any],
    eval_args,
    device: torch.device,
    seed: int,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], float]:
    out: list[dict[str, Any]] = []
    token_correct = 0
    token_total = 0
    generator = torch.Generator(device=device).manual_seed(int(seed))
    bf16_dtype = torch.bfloat16 if bool(eval_args.bf16) else None
    use_element_set = bool(model_args.get("element_set_cond", False))
    with torch.no_grad():
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            pad_mask = torch.stack([it["pad_mask"] for it in chunk]).to(device).bool()
            cond, elem_set_mask = build_joint_condition(
                chunk, model_args=model_args, device=device
            )
            target_a0_device = torch.stack([it["A0"] for it in chunk]).to(device)
            fix_types = bool(getattr(eval_args, "fixed_atom_types", False))
            fixed_type_values = (
                type_encoding.encode_from_A0(a0=target_a0_device, pad_mask=pad_mask)
                if fix_types
                else None
            )
            fix_frac = bool(getattr(eval_args, "fixed_fractional_coords", False))
            fixed_fractional_coords = (
                torch.stack([it["F1"] for it in chunk]).to(device) - 0.5
                if fix_frac
                else None
            )
            samples = edm_sampler(
                model=model,
                pad_mask=pad_mask,
                type_dim=type_encoding.type_dim,
                num_steps=int(eval_args.sample_num_steps),
                solver=str(getattr(eval_args, "solver", "heun")),
                sigma_min=float(eval_args.sigma_min),
                sigma_max=float(eval_args.sigma_max),
                rho=float(eval_args.rho),
                S_churn=float(eval_args.S_churn),
                S_min=float(eval_args.S_min),
                S_max=float(eval_args.S_max),
                S_noise=float(eval_args.S_noise),
                sigma_data_type=float(model_args.get("sigma_data_type", 1.0)),
                sigma_data_coord=float(model_args.get("sigma_data_coord", 0.25)),
                sigma_data_lat=float(model_args.get("sigma_data_lattice", 1.0)),
                generator=generator,
                autocast_dtype=bf16_dtype,
                fixed_atom_types=fixed_type_values,
                fixed_fractional_coords=fixed_fractional_coords,
                skip_type_scaling=fix_types,
                aa_frac_max_scale=float(eval_args.aa_frac_max_scale),
                aa_rho_types=float(eval_args.aa_rho_types),
                aa_rho_coords=float(eval_args.aa_rho_coords),
                aa_rho_lattice=float(eval_args.aa_rho_lattice),
                sigma_min_types=getattr(eval_args, "sigma_min_types", None),
                sigma_max_types=getattr(eval_args, "sigma_max_types", None),
                rho_types=getattr(eval_args, "rho_types", None),
                sigma_min_coords=getattr(eval_args, "sigma_min_coords", None),
                sigma_max_coords=getattr(eval_args, "sigma_max_coords", None),
                rho_coords=getattr(eval_args, "rho_coords", None),
                sigma_min_lattice=getattr(eval_args, "sigma_min_lattice", None),
                sigma_max_lattice=getattr(eval_args, "sigma_max_lattice", None),
                rho_lattice=getattr(eval_args, "rho_lattice", None),
                lattice_repr=str(model_args.get("lattice_repr", "y1")),
                cond=cond,
            )
            pad_cpu = pad_mask.cpu()
            type_cpu = samples["type"].detach().cpu()
            elem_set_cpu = elem_set_mask.cpu()
            if fix_types:
                pred_a0 = target_a0_device.cpu()
                assignment = {
                    "unrestricted_a0": pred_a0,
                    "allowed_a0": pred_a0,
                    "unrestricted_score": torch.ones_like(pred_a0, dtype=torch.float32),
                    "allowed_score": torch.ones_like(pred_a0, dtype=torch.float32),
                    "best_disallowed_a0": torch.zeros_like(pred_a0),
                    "best_disallowed_score": torch.zeros_like(
                        pred_a0, dtype=torch.float32
                    ),
                    "out_of_set": torch.zeros_like(pred_a0, dtype=torch.bool),
                    "out_of_set_margin": torch.zeros_like(pred_a0, dtype=torch.float32),
                    "allowed_vs_disallowed_margin": torch.ones_like(
                        pred_a0, dtype=torch.float32
                    ),
                }
            else:
                assignment = prototype_assignment_diagnostics(
                    type_cpu, pad_cpu, elem_set_cpu, type_encoding=type_encoding
                )
                pred_a0 = (
                    assignment["allowed_a0"]
                    if use_element_set
                    else assignment["unrestricted_a0"]
                )
            real_cpu = ~pad_cpu
            target_a0 = torch.stack([it["A0"] for it in chunk]).cpu()
            token_correct += int(((pred_a0 == target_a0) & real_cpu).sum().item())
            token_total += int(real_cpu.sum().item())
            frac = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
            frac = torch.where(real_cpu[..., None], frac, torch.zeros_like(frac))
            lattice_repr = str(model_args.get("lattice_repr", "y1"))
            lat_latent = clamp_lattice_latent(
                samples["lat"].detach().cpu(), lattice_repr=lattice_repr
            )
            y1 = lattice_latent_to_y1(lat_latent, lattice_repr=lattice_repr)
            y1 = clamp_lattice_latent(y1, lattice_repr="y1")
            for i in range(pred_a0.shape[0]):
                out.append(
                    {
                        "A0": pred_a0[i],
                        "F1": frac[i],
                        "Y1": y1[i],
                        "pad_mask": pad_cpu[i],
                        "unrestricted_A0": assignment["unrestricted_a0"][i],
                        "prototype_unrestricted_score": assignment[
                            "unrestricted_score"
                        ][i],
                        "prototype_best_allowed_score": assignment["allowed_score"][i],
                        "prototype_best_disallowed_A0": assignment[
                            "best_disallowed_a0"
                        ][i],
                        "prototype_best_disallowed_score": assignment[
                            "best_disallowed_score"
                        ][i],
                        "prototype_out_of_set": assignment["out_of_set"][i],
                        "prototype_out_of_set_margin": assignment["out_of_set_margin"][
                            i
                        ],
                        "prototype_allowed_vs_disallowed_margin": assignment[
                            "allowed_vs_disallowed_margin"
                        ][i],
                    }
                )
    return (out, float(token_correct) / float(max(token_total, 1)))


_PROFILE_WORKER_GRID: np.ndarray | None = None
_PROFILE_WORKER_SIGMA = 0.08
_PROFILE_WORKER_CALCULATOR: XRDCalculator | None = None


def evaluate_joint_model(
    *,
    model,
    targets: Sequence[dict[str, Any]],
    type_encoding,
    model_args: dict[str, Any],
    eval_args,
    device: torch.device,
    seed: int,
    chunk_size: int,
    num_candidates: int,
    topk: Sequence[int],
    matcher_profile: str | None = "strict",
) -> dict[str, Any]:
    """Sample and score joint atom-type/coordinate/lattice reconstruction."""
    if not targets:
        raise ValueError("targets must not be empty.")
    matcher_profile = matcher_profile or "strict"
    topk_norm = sorted({int(k) for k in topk})
    if not topk_norm or topk_norm[0] <= 0:
        raise ValueError("topk must contain positive integers.")
    num_candidates = max(int(num_candidates), topk_norm[-1])
    gt_crys = [_safe_crystal_from_tokens(item, i) for i, item in enumerate(targets)]
    gt_formula = [formula_key(item["A0"], item["pad_mask"]) for item in targets]
    pred_batches = []
    formula_hits = {k: [False] * len(targets) for k in topk_norm}
    token_acc_first = 0.0
    for cand_idx in range(num_candidates):
        items, token_acc = sample_joint_items(
            model=model,
            items=targets,
            type_encoding=type_encoding,
            model_args=model_args,
            eval_args=eval_args,
            device=device,
            seed=int(seed) + 1000003 * cand_idx,
            chunk_size=max(1, int(chunk_size)),
        )
        if cand_idx == 0:
            token_acc_first = token_acc
        pred_batches.append(
            [_safe_crystal_from_tokens(item, i) for i, item in enumerate(items)]
        )
        for i, item in enumerate(items):
            hit = formula_key(item["A0"], item["pad_mask"]) == gt_formula[i]
            for k in topk_norm:
                if cand_idx < k:
                    formula_hits[k][i] = formula_hits[k][i] or hit
    scored, records = score_crystal_batches(
        pred_batches,
        gt_crys,
        topk=topk_norm,
        matcher_profile=matcher_profile,
        match_gate="matcher_only",
        return_per_target=True,
    )
    metrics = {"pxrd_joint_type_token_acc@1": float(token_acc_first)}
    for k, (rate, rmsd) in scored.items():
        metrics[f"pxrd_joint_match_rate@{k}"] = rate
        metrics[f"pxrd_joint_rmsd@{k}"] = rmsd if np.isfinite(rmsd) else None
        metrics[f"pxrd_joint_formula_recall@{k}"] = sum(formula_hits[k]) / len(targets)
    return {"metrics": metrics, "records": records}
