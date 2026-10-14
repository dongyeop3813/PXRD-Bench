"""Dependency-light evaluation contracts and frozen split validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

MatcherProfile = Literal["loose", "strict"]
MatchGate = Literal["matcher_only", "struct_valid", "full_valid"]

MATCHER_PROFILES: dict[str, dict[str, float]] = {
    "loose": {"ltol": 0.3, "stol": 0.5, "angle_tol": 10.0},
    "strict": {"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0},
}
MATCH_GATES = ("matcher_only", "struct_valid", "full_valid")
CANONICAL_CANDIDATE_BUDGET = 20
CANONICAL_MATCHER_PROFILE = "strict"
CANONICAL_MATCH_GATE = "matcher_only"

RRUFF_SPLITS = {
    "validation": (
        48,
        "686927855e98534f56770a33a8ef5169608d88c1b11504b50a28c6f2e1a276b3",
    ),
    "test": (
        100,
        "cc47d1fbc157c3a745a9294aab3c86eaa1ae3fe4b95d11584352a05e80556726",
    ),
}
RRUFF_SOURCE_MANIFEST_SHA256 = (
    "0c5135cf7085d31a129506ee72fae3f7520333da360107137569ad9ef4aa9100"
)
CANONICAL_OPXRD_SPLITS = {
    "opXRD_experimental_v2_nmax20_val24": ("val", 20),
    "opXRD_experimental_v2_nmax20_test56": ("test", 20),
    "opXRD_experimental_v2_val40": ("val", 52),
    "opXRD_experimental_v2_test83": ("test", 52),
}


def validate_canonical_evaluation_contract(
    *,
    mode: str,
    topk: Sequence[int],
    matcher_profile: str,
    match_gate: str,
    num_candidates: int | None = None,
    allow_noncanonical_candidate_budget: bool = False,
    allow_noncanonical_evaluation_contract: bool = False,
) -> None:
    if mode == "score":
        return
    if (
        not allow_noncanonical_evaluation_contract
        and matcher_profile != CANONICAL_MATCHER_PROFILE
    ):
        raise ValueError("new evaluations require scoring.matcher_profile=strict")
    if (
        not allow_noncanonical_evaluation_contract
        and match_gate != CANONICAL_MATCH_GATE
    ):
        raise ValueError("new evaluations require scoring.match_gate=matcher_only")
    allow_noncanonical_candidate_budget = (
        allow_noncanonical_candidate_budget
        or allow_noncanonical_evaluation_contract
    )
    values = sorted({int(value) for value in topk})
    if not values or values[0] != 1 or (
        not allow_noncanonical_candidate_budget
        and values[-1] != CANONICAL_CANDIDATE_BUDGET
    ):
        raise ValueError(
            "new evaluations require scoring.topk to include 1 and 20 with max=20"
        )
    if (
        not allow_noncanonical_candidate_budget
        and num_candidates is not None
        and int(num_candidates) != CANONICAL_CANDIDATE_BUDGET
    ):
        raise ValueError("new evaluations require sampling.num_candidates=20")


def validate_rruff_split_manifest(payload: dict[str, Any]) -> None:
    logical_split = str(payload.get("logical_split", ""))
    expected = RRUFF_SPLITS.get(logical_split)
    actual = (payload.get("num_targets"), payload.get("target_indices_sha256"))
    if (
        payload.get("dataset") != "rruff_148_author_exact"
        or payload.get("split") != "test"
        or payload.get("source_manifest_sha256") != RRUFF_SOURCE_MANIFEST_SHA256
        or expected != actual
    ):
        raise ValueError(
            "RRUFF evaluation requires the frozen validation-48 or test-100 manifest"
        )


def validate_rruff_split_path(path: str | Path) -> None:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RRUFF target manifest must be a JSON object")
    validate_rruff_split_manifest(payload)


def validate_opxrd_split_path(path: str | Path, *, split: str, nmax: int) -> None:
    archive = Path(path).expanduser().resolve()
    payload = json.loads(
        archive.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    dataset_id = str(payload.get("dataset_id", ""))
    evaluation = payload.get("evaluation") or {}
    if (
        CANONICAL_OPXRD_SPLITS.get(dataset_id) != (split, int(nmax))
        or payload.get("archive") != archive.name
        or evaluation.get("dataset_class") != "experimental_pxrd"
        or (evaluation.get("split"), evaluation.get("nmax"))
        != (split, int(nmax))
    ):
        raise ValueError(
            "new opXRD evaluations require a frozen v2 validation/test archive"
        )
