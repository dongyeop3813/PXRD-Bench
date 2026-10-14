"""In-training PXRD structure matching on fixed held-out targets.

The default strict matcher uses ltol=0.2, stol=0.3 and angle_tol=5 degrees.
Match@k counts any matcher success among the first k candidates, with no
additional structural or composition-validity gate. Target subsets and
sampling seeds stay fixed across training for comparable evaluations.
"""

from __future__ import annotations
from typing import Any
import torch
from src.bragglite.sampler import edm_sampler
from src.eval_util.metric_names import metric_key, PXRD, MATCH_RATE, RMSD
from src.pxrd.eval_metrics import (
    aggregate_lattice_parameter_metrics,
    score_crystal_batches,
)
from src.pxrd.model_runtime import _safe_crystal_from_tokens
from src.pxrd.conditioning import build_conditioned_batch, sampler_output_to_token_items


def select_eval_targets(
    eval_ds, num_targets: int, seed: int, target_ids_path: str | None = None
) -> torch.Tensor:
    """Deterministic target subset of the eval split (fixed across steps)."""
    if target_ids_path:
        from src.utils.benchmark_ids import manifest_indices

        selected = manifest_indices(eval_ds, target_ids_path)
        if len(selected) != int(num_targets):
            raise ValueError(
                "fixed monitor manifest count must equal requested samples"
            )
        return torch.tensor(selected, dtype=torch.long)
    n = len(eval_ds)
    if n == 0:
        raise ValueError("Evaluation dataset is empty.")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(n, generator=gen)[: min(int(num_targets), n)]


def _sample_candidate_set(
    *,
    model,
    eval_items: list[dict[str, Any]],
    type_encoding,
    args,
    device: torch.device,
    seed: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """One conditioned CSP sample per eval item; returns token-item dicts."""
    items_out: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    bf16_dtype = torch.bfloat16 if args.bf16 else None
    with torch.no_grad():
        for start in range(0, len(eval_items), chunk_size):
            chunk = eval_items[start : start + chunk_size]
            cond_batch = build_conditioned_batch(
                chunk,
                type_encoding=type_encoding,
                args=args,
                device=device,
                lattice_repr=args.lattice_repr,
            )
            samples = edm_sampler(
                model=model,
                pad_mask=cond_batch.pad_mask,
                type_dim=type_encoding.type_dim,
                num_steps=args.sample_num_steps,
                solver=getattr(args, "solver", "heun"),
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                generator=generator,
                autocast_dtype=bf16_dtype,
                fixed_atom_types=cond_batch.fixed_atom_types,
                skip_type_scaling=True,
                aa_frac_max_scale=args.aa_frac_max_scale,
                aa_rho_types=args.aa_rho_types,
                aa_rho_coords=args.aa_rho_coords,
                aa_rho_lattice=args.aa_rho_lattice,
                sigma_min_types=getattr(args, "sigma_min_types", None),
                sigma_max_types=getattr(args, "sigma_max_types", None),
                rho_types=getattr(args, "rho_types", None),
                sigma_min_coords=getattr(args, "sigma_min_coords", None),
                sigma_max_coords=getattr(args, "sigma_max_coords", None),
                rho_coords=getattr(args, "rho_coords", None),
                sigma_min_lattice=getattr(args, "sigma_min_lattice", None),
                sigma_max_lattice=getattr(args, "sigma_max_lattice", None),
                rho_lattice=getattr(args, "rho_lattice", None),
                lattice_repr=args.lattice_repr,
                cond=cond_batch.cond,
            )
            if "pad_mask" not in samples:
                samples["pad_mask"] = cond_batch.pad_mask
            items_out.extend(
                sampler_output_to_token_items(
                    samples,
                    chunk,
                    pad_mask=cond_batch.pad_mask,
                    lattice_repr=args.lattice_repr,
                )
            )
    return items_out


def run_pxrd_match_eval(
    *,
    model,
    eval_ds,
    target_indices: torch.Tensor,
    type_encoding,
    args,
    device: torch.device,
    return_per_target: bool = False,
) -> dict[str, float | None] | tuple[dict[str, float | None], list[dict[str, Any]]]:
    """PXRD match/RMSD and first-candidate physical metrics on fixed targets.

    The caller handles EMA application, W&B logging, and restoring
    ``model.train()``.
    """
    topk = sorted({int(k) for k in args.monitor_topk})
    if not topk or topk[0] <= 0:
        raise ValueError(
            f"monitor_topk must be positive ints, got {args.monitor_topk}."
        )
    num_candidates = max(int(args.monitor_num_candidates), topk[-1])
    chunk_size = int(args.monitor_chunk_size) or len(target_indices)
    chunk_size = max(1, chunk_size)
    was_training = model.training
    model.eval()
    try:
        eval_items = [eval_ds[int(i)] for i in target_indices]
        gt_crys = [
            _safe_crystal_from_tokens(item, i) for i, item in enumerate(eval_items)
        ]
        pred_batches = []
        for cand_idx in range(num_candidates):
            cand_seed = int(args.monitor_seed) + 1000003 * cand_idx
            cand_items = _sample_candidate_set(
                model=model,
                eval_items=eval_items,
                type_encoding=type_encoding,
                args=args,
                device=device,
                seed=cand_seed,
                chunk_size=chunk_size,
            )
            pred_batches.append(
                [
                    _safe_crystal_from_tokens(item, i)
                    for i, item in enumerate(cand_items)
                ]
            )
        scored, per_target = score_crystal_batches(
            pred_batches,
            gt_crys,
            topk=topk,
            matcher_profile=str(getattr(args, "matcher_profile", "strict")),
            match_gate=str(getattr(args, "match_gate", "matcher_only")),
            return_per_target=True,
        )
        for local_index, row in enumerate(per_target):
            row["target_index"] = int(target_indices[local_index])
        metrics: dict[str, float | None] = {}
        for k in topk:
            match_rate, rmsd = scored[k]
            metrics[metric_key(PXRD, MATCH_RATE, k)] = float(match_rate)
            metrics[metric_key(PXRD, RMSD, k)] = float(rmsd)
        metrics.update(
            aggregate_lattice_parameter_metrics(pred_batches[0], gt_crys, prefix=PXRD)
        )
        metrics.pop(f"{PXRD}_valid_lattice_rate@1", None)
        return (metrics, per_target) if return_per_target else metrics
    finally:
        if was_training:
            model.train()
