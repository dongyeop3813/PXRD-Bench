"""Dataset construction shared by task-specific evaluation entrypoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.data.factory import DATASET_CLASSES, build_token_dataset
from src.eval_util.contracts import (
    CANONICAL_OPXRD_SPLITS as CANONICAL_OPXRD_SPLITS,
    validate_rruff_split_manifest as validate_rruff_split_manifest,
)


def load_target_indices_manifest(
    path: str | Path, *, dataset_size: int, split: str
) -> tuple[list[int], dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("target_indices"), list
    ):
        raise ValueError("target manifest must contain a target_indices list")
    if payload.get("split") != split:
        raise ValueError(
            f"target manifest split={payload.get('split')!r} does not match data split={split!r}"
        )
    values = payload["target_indices"]
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError("target manifest indices must be non-empty integers")
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError("target manifest indices must be unique")
    invalid = [value for value in indices if value < 0 or value >= dataset_size]
    if invalid:
        raise ValueError(
            f"target manifest index {invalid[0]} is outside dataset size {dataset_size}"
        )
    if int(payload.get("num_targets", len(indices))) != len(indices):
        raise ValueError("target manifest num_targets does not match target_indices")
    digest = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()
    ).hexdigest()
    if payload.get("target_indices_sha256", digest) != digest:
        raise ValueError("target manifest target_indices_sha256 mismatch")
    return indices, payload


def resolve_dataset_class(explicit: str | None, model_args: dict[str, Any]) -> str:
    """Apply evaluation override -> checkpoint -> legacy default precedence."""
    dataset_class = str(explicit or model_args.get("dataset_class", "mp20_tokens"))
    if dataset_class not in DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset_class {dataset_class!r}; expected one of {DATASET_CLASSES}."
        )
    return dataset_class


def resolve_split(
    *,
    dataset_class: str,
    root: str,
    requested: str,
    fallbacks: Iterable[str] = (),
) -> str:
    """Resolve CSV-backed MP20 aliases while preserving native dataset splits."""
    if dataset_class != "mp20_tokens":
        return requested
    raw = Path(root) / "raw"
    for candidate in (requested, *fallbacks):
        if (raw / f"{candidate}.csv").exists():
            return candidate
    return requested


def build_evaluation_dataset(
    *,
    dataset_class: str,
    root: str,
    split: str,
    nmax: int,
    model_args: dict[str, Any],
    augment_translate: bool = False,
    real_pxrd_variant: str = "all",
    structure_cache: str | None = None,
):
    """Build any token dataset supported by :mod:`src.data.factory`."""
    atom_ordering = (
        str(model_args.get("atom_ordering", "none"))
        if dataset_class == "mp20_tokens"
        else "none"
    )
    return build_token_dataset(
        dataset_class,
        root=root,
        split=split,
        nmax=nmax,
        augment_translate=augment_translate,
        real_pxrd_variant=real_pxrd_variant,
        real_pxrd_in_memory=False,
        structure_cache=structure_cache or model_args.get("structure_cache"),
        atom_ordering=atom_ordering,
        hierarchical_orbit_permutation=False,
        atom_ordering_symprec=float(model_args.get("atom_ordering_symprec", 0.1)),
    )
