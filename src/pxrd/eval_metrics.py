"""Shared metrics for PXRD reconstruction tasks.

The scorer is deliberately independent of checkpoints and datasets: references
and candidates use one small JSON-compatible structure schema.  This keeps
generation separate from scoring and lets one prediction artifact be rescored
with a different matcher profile or validity gate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from src.eval_util.contracts import (
    MATCH_GATES,
    MATCHER_PROFILES,
    MatchGate,
    MatcherProfile,
)
from src.eval_util.bounded_matching import _MatcherTimedOut, _matcher_rmsd
from src.eval_util.crystal import smact_validity, structure_validity


def _valid_lattice(
    lengths: Any, angles: Any
) -> tuple[Lattice | None, np.ndarray | None, np.ndarray | None]:
    """Construct a physically admissible lattice independently of atom validity."""
    try:
        lengths_array = np.asarray(lengths, dtype=float).reshape(-1)
        angles_array = np.asarray(angles, dtype=float).reshape(-1)
        if lengths_array.shape != (3,) or angles_array.shape != (3,):
            return None, None, None
        if not np.isfinite(lengths_array).all() or not np.isfinite(angles_array).all():
            return None, None, None
        if (lengths_array <= 0).any() or (angles_array <= 0).any():
            return None, None, None
        if (angles_array >= 180).any():
            return None, None, None
        lattice = Lattice.from_parameters(
            *lengths_array.tolist(), *angles_array.tolist()
        )
        if not math.isfinite(lattice.volume) or lattice.volume <= 0.1:
            return None, None, None
        return lattice, lengths_array, angles_array
    except Exception:
        return None, None, None


def aggregate_lattice_parameter_metrics(
    predicted: Sequence[Any],
    references: Sequence[Any],
    *,
    prefix: str,
) -> dict[str, float | None]:
    """Aggregate first-candidate lattice errors; volume MAPE is percent."""
    count = min(len(predicted), len(references))
    valid_count = 0
    length_errors: list[float] = []
    angle_errors: list[float] = []
    volume_apes: list[float] = []

    def values(item: Any, name: str) -> Any:
        return item.get(name) if isinstance(item, dict) else getattr(item, name, None)

    for pred, ref in zip(predicted[:count], references[:count]):
        pred_lattice, pred_lengths, pred_angles = _valid_lattice(
            values(pred, "lengths"), values(pred, "angles")
        )
        ref_lattice, ref_lengths, ref_angles = _valid_lattice(
            values(ref, "lengths"), values(ref, "angles")
        )
        if pred_lattice is None:
            continue
        valid_count += 1
        if ref_lattice is None:
            continue
        length_errors.extend(np.abs(pred_lengths - ref_lengths).tolist())
        angle_errors.extend(np.abs(pred_angles - ref_angles).tolist())
        volume_apes.append(
            abs(pred_lattice.volume - ref_lattice.volume) / ref_lattice.volume * 100.0
        )
    return {
        f"{prefix}_lattice_length_mae@1": (
            float(np.mean(length_errors)) if length_errors else None
        ),
        f"{prefix}_angle_mae@1": (
            float(np.mean(angle_errors)) if angle_errors else None
        ),
        f"{prefix}_volume_mape@1": (
            float(np.mean(volume_apes)) if volume_apes else None
        ),
        f"{prefix}_valid_lattice_rate@1": (valid_count / count if count else 0.0),
    }


def structure_to_record(structure: Structure) -> dict[str, Any]:
    return {
        "atom_types": [int(site.specie.Z) for site in structure],
        "frac_coords": np.asarray(structure.frac_coords, dtype=float).tolist(),
        "lengths": [float(value) for value in structure.lattice.abc],
        "angles": [float(value) for value in structure.lattice.angles],
    }


def record_to_structure(record: dict[str, Any]) -> tuple[Structure | None, str | None]:
    required = {"atom_types", "frac_coords", "lengths", "angles"}
    missing = required - set(record)
    if missing:
        return None, f"missing structure fields: {sorted(missing)}"
    try:
        atom_types = np.asarray(record["atom_types"], dtype=int).reshape(-1)
        frac_coords = np.asarray(record["frac_coords"], dtype=float)
        lengths = np.asarray(record["lengths"], dtype=float).reshape(-1)
        angles = np.asarray(record["angles"], dtype=float).reshape(-1)
        if atom_types.size == 0:
            raise ValueError("atom_types is empty")
        if frac_coords.shape != (atom_types.size, 3):
            raise ValueError(
                f"frac_coords shape {frac_coords.shape} != ({atom_types.size}, 3)"
            )
        if lengths.shape != (3,) or angles.shape != (3,):
            raise ValueError("lengths and angles must have shape (3,)")
        if not np.isfinite(frac_coords).all() or not np.isfinite(lengths).all():
            raise ValueError("coordinates or lengths contain non-finite values")
        if not np.isfinite(angles).all():
            raise ValueError("angles contain non-finite values")
        if (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any():
            raise ValueError("lattice parameters are outside physical bounds")
        if (atom_types < 1).any() or (atom_types > 118).any():
            raise ValueError("atom_types contains an unsupported atomic number")
        structure = Structure(
            Lattice.from_parameters(*lengths.tolist(), *angles.tolist()),
            atom_types.tolist(),
            frac_coords % 1.0,
            coords_are_cartesian=False,
        )
        if not math.isfinite(structure.volume) or structure.volume <= 0:
            raise ValueError("structure has non-positive volume")
        return structure, None
    except Exception as exc:
        return None, repr(exc)


def formula_key(atom_types: Iterable[int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(Counter(int(z) for z in atom_types).items()))


def reduced_formula_key(atom_types: Iterable[int]) -> tuple[tuple[int, int], ...]:
    counts = Counter(int(z) for z in atom_types)
    divisor = math.gcd(*counts.values())
    return tuple(
        sorted((element, count // divisor) for element, count in counts.items())
    )


def _composition_valid(structure: Structure) -> bool:
    counts = Counter(int(site.specie.Z) for site in structure)
    elems = tuple(sorted(counts))
    values = np.asarray([counts[z] for z in elems], dtype=int)
    values //= np.gcd.reduce(values)
    try:
        return bool(smact_validity(elems, tuple(values.tolist())))
    except Exception:
        return False


def _passes_gate(structure: Structure, gate: MatchGate) -> bool:
    if gate == "matcher_only":
        return True
    struct_ok = bool(structure_validity(structure))
    if gate == "struct_valid":
        return struct_ok
    if gate == "full_valid":
        return struct_ok and _composition_valid(structure)
    raise ValueError(f"unknown match gate: {gate!r}")


def _gate_validity(structure: Structure | None) -> dict[str, bool]:
    if structure is None:
        return {gate: False for gate in MATCH_GATES}
    struct_ok = bool(structure_validity(structure))
    return {
        "matcher_only": True,
        "struct_valid": struct_ok,
        "full_valid": struct_ok and _composition_valid(structure),
    }


def _pdd_emd(reference: Structure, candidate: Structure, pdd_k: int) -> float:
    import amd
    from amd.io import periodicset_from_pymatgen_structure

    reference_pdd = amd.PDD(periodicset_from_pymatgen_structure(reference), pdd_k)
    candidate_pdd = amd.PDD(periodicset_from_pymatgen_structure(candidate), pdd_k)
    return float(amd.EMD(reference_pdd, candidate_pdd))


def score_prediction_row(
    row: dict[str, Any],
    *,
    topk: Sequence[int],
    matcher_profile: MatcherProfile,
    match_gate: MatchGate,
    pdd_k: int,
    include_formula: bool,
    match_only: bool = False,
    matcher_timeout_seconds: float = 0.0,
) -> dict[str, Any]:
    """Score one reference and its ordered candidate list."""
    if matcher_profile not in MATCHER_PROFILES:
        raise ValueError(f"unknown matcher profile: {matcher_profile!r}")
    if match_gate not in MATCH_GATES:
        raise ValueError(f"unknown match gate: {match_gate!r}")
    reference, reference_error = record_to_structure(row["reference"])
    if reference is None:
        raise ValueError(
            f"sample {row.get('sample_id')!r} has invalid reference: {reference_error}"
        )
    matcher = StructureMatcher(**MATCHER_PROFILES[matcher_profile])
    reference_validity = _gate_validity(reference)
    candidate_rows: list[dict[str, Any]] = []
    structures: list[Structure | None] = []
    for rank, candidate in enumerate(row["candidates"][: max(topk)]):
        structure, error = record_to_structure(candidate)
        structures.append(structure)
        candidate_validity = _gate_validity(structure)
        matcher_rmsd = None
        matcher_timed_out = False
        if structure is not None:
            try:
                result = _matcher_rmsd(
                    matcher, reference, structure, float(matcher_timeout_seconds)
                )
                matcher_rmsd = float(result[0]) if result is not None else None
            except _MatcherTimedOut:
                matcher_timed_out = True
            except Exception:
                matcher_rmsd = None
        matched_by_gate = {
            gate: bool(
                matcher_rmsd is not None
                and reference_validity[gate]
                and candidate_validity[gate]
            )
            for gate in MATCH_GATES
        }
        selected_match = matched_by_gate[match_gate]
        candidate_rows.append(
            {
                "rank": rank,
                "constructible": structure is not None,
                "struct_valid": candidate_validity["struct_valid"],
                "full_valid": candidate_validity["full_valid"],
                "gate_valid": candidate_validity[match_gate],
                "construction_error": error,
                "matcher_timed_out": matcher_timed_out,
                "matcher_matched": matcher_rmsd is not None,
                "matcher_rmsd": matcher_rmsd,
                "matched": selected_match,
                "rmsd": matcher_rmsd if selected_match else None,
            }
        )

    topk_rows: dict[str, Any] = {}
    target_formula = formula_key(int(site.specie.Z) for site in reference)
    for k in topk:
        matched_by_gate = {
            gate: any(
                candidate["matcher_matched"]
                and (
                    gate == "matcher_only"
                    or (reference_validity[gate] and candidate[gate])
                )
                for candidate in candidate_rows[:k]
            )
            for gate in MATCH_GATES
        }
        rms_values = [
            candidate["matcher_rmsd"]
            for candidate in candidate_rows[:k]
            if candidate["matched"]
        ]
        entry = {
            "matched": matched_by_gate[match_gate],
            "matched_by_gate": matched_by_gate,
            "rmsd": min(rms_values) if rms_values else None,
        }
        if include_formula:
            entry["formula_recalled"] = any(
                structure is not None
                and formula_key(int(site.specie.Z) for site in structure)
                == target_formula
                for structure in structures[:k]
            )
        topk_rows[str(k)] = entry

    result = {
        "sample_id": str(row["sample_id"]),
        "dataset_index": int(row.get("dataset_index", -1)),
        "strata": dict(row.get("strata", {})),
        "reference_validity": {
            gate: reference_validity[gate] for gate in ("struct_valid", "full_valid")
        },
        "topk": topk_rows,
        "candidates": candidate_rows,
    }
    if match_only:
        return result
    first = structures[0] if structures else None
    first_candidate = row["candidates"][0] if row["candidates"] else {}
    lattice_metrics = aggregate_lattice_parameter_metrics(
        [first_candidate], [row["reference"]], prefix="_row"
    )
    lattice_mae = None
    pdd_emd = None
    continuous_error = None
    if first is not None:
        lattice_mae = np.abs(
            np.asarray(first.lattice.abc) - np.asarray(reference.lattice.abc)
        ).tolist()
        try:
            pdd_emd = _pdd_emd(reference, first, pdd_k)
        except Exception as exc:
            continuous_error = repr(exc)
    result.update(
        {
            "pdd_emd@1": pdd_emd,
            "lattice_mae@1": lattice_mae,
            "lattice_length_mae@1": lattice_metrics["_row_lattice_length_mae@1"],
            "angle_mae@1": lattice_metrics["_row_angle_mae@1"],
            "volume_mape@1": lattice_metrics["_row_volume_mape@1"],
            "valid_lattice@1": lattice_metrics["_row_valid_lattice_rate@1"],
            "type_token_correct@1": int(first_candidate.get("type_token_correct", 0)),
            "type_token_total@1": int(first_candidate.get("type_token_total", 0)),
            "continuous_error": continuous_error,
        }
    )
    return result


def score_metre_group(
    payload: tuple[Sequence[dict[str, Any]], Sequence[int], MatcherProfile, MatchGate],
) -> dict[int, list[float | None]]:
    """Return best RMSDs for each reference against its composition-wide pool."""
    rows, topk, matcher_profile, match_gate = payload
    matcher = StructureMatcher(**MATCHER_PROFILES[matcher_profile])
    references = [record_to_structure(row["reference"])[0] for row in rows]
    output: dict[int, list[float | None]] = {}
    for k in topk:
        candidates = [
            record_to_structure(candidate)[0]
            for row in rows
            for candidate in row["candidates"][:k]
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate is not None
            and (match_gate == "matcher_only" or _passes_gate(candidate, match_gate))
        ]
        best: list[float | None] = []
        for reference in references:
            if reference is None or (
                match_gate != "matcher_only" and not _passes_gate(reference, match_gate)
            ):
                best.append(None)
                continue
            rmsds = []
            for candidate in candidates:
                try:
                    result = matcher.get_rms_dist(candidate, reference)
                except Exception:
                    result = None
                if result is not None:
                    rmsds.append(float(result[0]))
            best.append(min(rmsds) if rmsds else None)
        output[int(k)] = best
    return output


class PredictionScoreAccumulator:
    """Incrementally aggregate scored rows without retaining per-sample data."""

    def __init__(
        self,
        *,
        topk: Sequence[int],
        prefix: str,
        include_formula: bool | None = None,
        include_auxiliary: bool = False,
        include_rmsd: bool = False,
    ) -> None:
        self.topk = tuple(int(k) for k in topk)
        self.prefix = prefix
        self.include_auxiliary = bool(include_auxiliary)
        self.include_rmsd = bool(include_rmsd)
        self.include_formula = self.include_auxiliary and (
            prefix == "pxrd_elem" if include_formula is None else bool(include_formula)
        )
        self.count = 0
        self.match_counts = {k: 0 for k in self.topk}
        self.formula_counts = {k: 0 for k in self.topk}
        self.rms_sums = {k: 0.0 for k in self.topk}
        self.rms_counts = {k: 0 for k in self.topk}
        self.pdd_sum = 0.0
        self.pdd_count = 0
        self.lattice_sums = [0.0, 0.0, 0.0]
        self.lattice_counts = [0, 0, 0]
        self.type_correct = 0
        self.type_total = 0
        self.lattice_length_sum = 0.0
        self.lattice_length_count = 0
        self.angle_sum = 0.0
        self.angle_count = 0
        self.volume_mape_sum = 0.0
        self.volume_mape_count = 0
        self.valid_lattice_count = 0
        self.matcher_timeout_count = 0

    @staticmethod
    def _finite(value: Any) -> float | None:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def update(self, row: dict[str, Any]) -> None:
        self.count += 1
        self.matcher_timeout_count += sum(
            int(bool(candidate.get("matcher_timed_out", False)))
            for candidate in row.get("candidates", ())
        )
        for k in self.topk:
            entry = row["topk"][str(k)]
            self.match_counts[k] += int(bool(entry["matched"]))
            rmsd = self._finite(entry["rmsd"])
            if rmsd is not None:
                self.rms_sums[k] += rmsd
                self.rms_counts[k] += 1
            if self.include_formula:
                self.formula_counts[k] += int(bool(entry["formula_recalled"]))
        if not self.include_auxiliary:
            return
        pdd = self._finite(row["pdd_emd@1"])
        if pdd is not None:
            self.pdd_sum += pdd
            self.pdd_count += 1
        lattice = row["lattice_mae@1"]
        if lattice is not None:
            for axis, value in enumerate(lattice):
                finite = self._finite(value)
                if finite is not None:
                    self.lattice_sums[axis] += finite
                    self.lattice_counts[axis] += 1
        for key, sum_name, count_name in (
            ("lattice_length_mae@1", "lattice_length_sum", "lattice_length_count"),
            ("angle_mae@1", "angle_sum", "angle_count"),
            ("volume_mape@1", "volume_mape_sum", "volume_mape_count"),
        ):
            finite = self._finite(row.get(key))
            if finite is not None:
                setattr(self, sum_name, getattr(self, sum_name) + finite)
                setattr(self, count_name, getattr(self, count_name) + 1)
        self.valid_lattice_count += int(bool(row.get("valid_lattice@1", False)))
        self.type_correct += int(row["type_token_correct@1"])
        self.type_total += int(row["type_token_total@1"])

    def finalize(self) -> tuple[dict[str, float | None], dict[str, int]]:
        metrics: dict[str, float | None] = {}
        coverage: dict[str, int] = {}
        for k in self.topk:
            metrics[f"{self.prefix}_match_rate@{k}"] = (
                self.match_counts[k] / self.count if self.count else 0.0
            )
            if self.include_rmsd:
                rms_key = f"{self.prefix}_rmsd@{k}"
                metrics[rms_key] = (
                    self.rms_sums[k] / self.rms_counts[k]
                    if self.rms_counts[k]
                    else None
                )
                coverage[rms_key] = self.rms_counts[k]
            if self.include_formula:
                metrics[f"{self.prefix}_formula_recall@{k}"] = (
                    self.formula_counts[k] / self.count if self.count else 0.0
                )
        coverage[f"{self.prefix}_matcher_timeout_count"] = self.matcher_timeout_count
        if not self.include_auxiliary:
            return metrics, coverage
        pdd_key = f"{self.prefix}_pdd_emd@1"
        metrics[pdd_key] = self.pdd_sum / self.pdd_count if self.pdd_count else None
        coverage[pdd_key] = self.pdd_count
        for axis, name in enumerate("abc"):
            key = f"{self.prefix}_lattice_mae_{name}@1"
            metrics[key] = (
                self.lattice_sums[axis] / self.lattice_counts[axis]
                if self.lattice_counts[axis]
                else None
            )
            coverage[key] = self.lattice_counts[axis]
        lattice_key = f"{self.prefix}_lattice_mae@1"
        lattice_count = sum(self.lattice_counts)
        metrics[lattice_key] = (
            sum(self.lattice_sums) / lattice_count if lattice_count else None
        )
        coverage[lattice_key] = lattice_count
        for suffix, total, count in (
            (
                "lattice_length_mae@1",
                self.lattice_length_sum,
                self.lattice_length_count,
            ),
            ("angle_mae@1", self.angle_sum, self.angle_count),
            ("volume_mape@1", self.volume_mape_sum, self.volume_mape_count),
        ):
            key = f"{self.prefix}_{suffix}"
            metrics[key] = total / count if count else None
            coverage[key] = count
        metrics[f"{self.prefix}_valid_lattice_rate@1"] = (
            self.valid_lattice_count / self.count if self.count else 0.0
        )
        if self.include_formula:
            key = f"{self.prefix}_type_token_acc@1"
            metrics[key] = (
                self.type_correct / self.type_total if self.type_total else 0.0
            )
            coverage[key] = self.type_total
        return metrics, coverage


def aggregate_prediction_scores(
    samples: Iterable[dict[str, Any]], *, topk: Sequence[int], prefix: str
) -> tuple[dict[str, float | None], dict[str, int]]:
    accumulator = PredictionScoreAccumulator(topk=topk, prefix=prefix)
    for row in samples:
        accumulator.update(row)
    return accumulator.finalize()


def score_crystal_batches(
    pred_batches: Sequence[Sequence[Any]],
    gt_crystals: Sequence[Any],
    *,
    topk: Sequence[int],
    matcher_profile: MatcherProfile,
    match_gate: MatchGate,
    return_per_target: bool = False,
) -> (
    dict[int, tuple[float, float]]
    | tuple[dict[int, tuple[float, float]], list[dict[str, Any]]]
):
    """Training-time adapter for existing ``Crystal`` objects."""
    if match_gate not in MATCH_GATES:
        raise ValueError(f"unknown match gate: {match_gate!r}")
    matcher = StructureMatcher(**MATCHER_PROFILES[matcher_profile])
    all_rms = np.full((len(pred_batches), len(gt_crystals)), np.nan, dtype=float)

    def valid(crystal: Any, gate: MatchGate) -> bool:
        if not bool(getattr(crystal, "constructed", False)):
            return False
        if gate == "matcher_only":
            return True
        if gate == "struct_valid":
            return bool(crystal.struct_valid)
        if gate == "full_valid":
            return bool(crystal.valid)
        raise ValueError(f"unknown match gate: {gate!r}")

    gate_masks = {
        gate: np.asarray(
            [
                [
                    valid(gt, gate) and valid(pred, gate)
                    for gt, pred in zip(gt_crystals, batch)
                ]
                for batch in pred_batches
            ],
            dtype=bool,
        )
        for gate in MATCH_GATES
    }

    for target_index, gt in enumerate(gt_crystals):
        for candidate_index, batch in enumerate(pred_batches):
            pred = batch[target_index]
            if not gate_masks["matcher_only"][candidate_index, target_index]:
                continue
            try:
                result = matcher.get_rms_dist(pred.structure, gt.structure)
                if result is not None:
                    all_rms[candidate_index, target_index] = float(result[0])
            except Exception:
                pass
    output: dict[int, tuple[float, float]] = {}
    per_target = [{"target_index": int(index)} for index in range(len(gt_crystals))]
    for k in topk:
        subset = np.where(gate_masks[match_gate][:k], all_rms[:k], np.nan)
        matched_by_gate = {
            gate: (np.isfinite(all_rms[:k]) & mask[:k]).any(axis=0)
            for gate, mask in gate_masks.items()
        }
        matched = np.isfinite(subset).any(axis=0)
        best = np.where(np.isfinite(subset), subset, np.inf).min(axis=0)
        output[int(k)] = (
            float(matched.mean()) if matched.size else 0.0,
            float(best[matched].mean()) if matched.any() else float("nan"),
        )
        for target_index, is_match in enumerate(matched):
            per_target[target_index][f"matched_by_gate@{int(k)}"] = {
                gate: bool(values[target_index])
                for gate, values in matched_by_gate.items()
            }
            per_target[target_index][f"match@{int(k)}"] = bool(is_match)
            per_target[target_index][f"rmsd@{int(k)}"] = (
                float(best[target_index]) if bool(is_match) else None
            )
    return (output, per_target) if return_per_target else output
