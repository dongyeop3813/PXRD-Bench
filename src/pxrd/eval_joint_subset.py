"""Metric helpers for aggregating repeated PXRD joint-evaluation subsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("target index artifact must contain a non-empty indices list")
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("target indices must be unique nonnegative integers")
    return indices


def _same_formula(left: Any, right: Any) -> bool:
    def normalize(formula):
        return sorted((int(z), int(count)) for z, count in formula)

    return normalize(left) == normalize(right)


def _is_collision(min_distance: Any, threshold: float) -> bool:
    """Treat unavailable geometry diagnostics as unknown, not a collision."""
    return min_distance is not None and float(min_distance) < float(threshold)


def _candidate_budget_metrics(
    records: list[dict[str, Any]], candidate_budget: int
) -> dict[str, float | None]:
    matched_rmsds: list[float] = []
    formula_count = 0
    for record in records:
        rmsds = record["candidate_rmsds"][:candidate_budget]
        valid_rmsds = [float(value) for value in rmsds if value is not None]
        if valid_rmsds:
            matched_rmsds.append(min(valid_rmsds))
        formulas = record["candidate_formulas"][:candidate_budget]
        formula_count += int(
            any(_same_formula(record["target_formula"], value) for value in formulas)
        )
    denominator = len(records)
    return {
        f"pxrd_joint_match_rate@{candidate_budget}": len(matched_rmsds) / denominator,
        f"pxrd_joint_formula_recall@{candidate_budget}": formula_count / denominator,
        f"pxrd_joint_rmsd@{candidate_budget}": (
            sum(matched_rmsds) / len(matched_rmsds) if matched_rmsds else None
        ),
    }


def aggregate_seed_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for payload in payloads for record in payload["records"]]
    if not records:
        raise ValueError("joint subset evaluation produced no records")
    denominator = len(records)
    num_atoms = sum(int(record["target_num_atoms"]) for record in records)
    matched_rmsds = [
        float(record["rmsd@1"])
        for record in records
        if record.get("matched@1") and record.get("rmsd@1") is not None
    ]
    out_of_set = [
        int(record.get("unconstrained_out_of_set_count", 0)) for record in records
    ]
    metrics: dict[str, float | None] = {
        "pxrd_joint_match_rate@1": sum(
            bool(record.get("matched@1")) for record in records
        )
        / denominator,
        "pxrd_joint_formula_recall@1": sum(
            _same_formula(record["target_formula"], record["pred_formula"])
            for record in records
        )
        / denominator,
        "pxrd_joint_collision_rate@1": sum(
            bool(record.get("collision@1")) for record in records
        )
        / denominator,
        "pxrd_joint_unconstrained_out_of_set_token_rate@1": sum(out_of_set)
        / max(1, num_atoms),
        "pxrd_joint_unconstrained_out_of_set_structure_rate@1": sum(
            value > 0 for value in out_of_set
        )
        / denominator,
        "pxrd_joint_rmsd@1": (
            sum(matched_rmsds) / len(matched_rmsds) if matched_rmsds else None
        ),
    }
    candidate_budget = int(payloads[0].get("num_candidates", 1))
    if candidate_budget > 1:
        metrics.update(_candidate_budget_metrics(records, candidate_budget))
    return {
        "num_trajectories": denominator,
        "num_target_atoms": num_atoms,
        "metrics": metrics,
        "records": records,
    }
