"""Lightweight evaluation request identities shared by runtime and preflight."""

from __future__ import annotations

from src.bragglite.compat import CSP_SEED_DOMAIN, EVALUATION_SCHEMA

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = EVALUATION_SCHEMA


_NON_IDENTITY_RUNTIME_KEYS = {
    "cache_shard_size",
    "device",
    "num_shards",
    "parallel_devices",
    "parallel_timeout_seconds",
    "parallel_worker",
    "shard_index",
    "workers",
    "warmup_compile",
}
_NON_IDENTITY_SAMPLING_KEYS = {
    "batch_candidates",
    "candidate_chunk_size",
    "chunk_size",
    "sample_chunk_size",
}
_NON_IDENTITY_SCORING_KEYS = {"eval_jobs", "workers"}
_NON_IDENTITY_CHECKPOINT_KEYS = {
    "checkpoint",
    "checkpoint_preference",
    "path",
    "train_output_dir",
}


def _jsonable(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [_jsonable(value) for value in payload]
    if hasattr(payload, "item"):
        try:
            return payload.item()
        except (RuntimeError, TypeError, ValueError):
            pass
    return str(payload)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def config_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def normalized_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = _jsonable(request)
    if not isinstance(normalized, dict):
        raise ValueError("evaluation request must be a mapping")
    run = normalized.get("run")
    if isinstance(run, dict):
        normalized["run"] = {
            key: value
            for key, value in run.items()
            if key not in {"force", "mode", "output_dir"}
        }
    return normalized


def scoring_request(normalized: dict[str, Any], *, task: Any) -> dict[str, Any]:
    scoring = normalized.get("scoring", {})
    if not isinstance(scoring, dict):
        raise ValueError("evaluation scoring config must be a mapping")
    return {
        "task": task,
        "data": normalized.get("data", {}),
        "scoring": {
            key: value
            for key, value in scoring.items()
            if key not in _NON_IDENTITY_SCORING_KEYS
        },
    }


def evaluation_lineage(
    request: dict[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any]:
    """Build result-affecting sampling and scoring identities."""
    normalized = normalized_request(request)
    task = normalized.get("task")
    runtime = normalized.get("runtime", {})
    checkpoint = normalized.get("checkpoint", {})
    sampling = normalized.get("sampling", {})
    scoring = normalized.get("scoring", {})
    for name, value in (
        ("runtime", runtime),
        ("checkpoint", checkpoint),
        ("sampling", sampling),
        ("scoring", scoring),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"evaluation {name} config must be a mapping")
    non_identity_sampling_keys = set(_NON_IDENTITY_SAMPLING_KEYS)
    if task == "dng":
        non_identity_sampling_keys.discard("sample_chunk_size")
    sampling_identity = {
        "task": task,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": {
            key: value
            for key, value in checkpoint.items()
            if key not in _NON_IDENTITY_CHECKPOINT_KEYS
        },
        "data": normalized.get("data", {}),
        "sampling": {
            key: value
            for key, value in sampling.items()
            if key not in non_identity_sampling_keys
        },
        "runtime": {
            key: value
            for key, value in runtime.items()
            if key not in _NON_IDENTITY_RUNTIME_KEYS
        },
        "rng_scheme": normalized.get("rng_scheme", (
            "sequential_generator_v1"
            if task == "dng"
            else CSP_SEED_DOMAIN if task == "csp"
            else "per_target_candidate_v1"
        )),
    }
    if "external" in normalized:
        sampling_identity["external"] = normalized["external"]
    if task == "csp":
        topk = [int(value) for value in scoring.get("topk", ())]
        if topk:
            sampling_identity["candidate_budget"] = max(topk)
    scoring_identity = scoring_request(normalized, task=task)
    sampling_fingerprint = config_fingerprint(sampling_identity)
    scoring_fingerprint = config_fingerprint(scoring_identity)
    return {
        "request": normalized,
        "sampling_request": sampling_identity,
        "scoring_request": scoring_identity,
        "sampling_fingerprint": sampling_fingerprint,
        "scoring_fingerprint": scoring_fingerprint,
        "active_fingerprint": config_fingerprint(
            {
                "sampling_fingerprint": sampling_fingerprint,
                "scoring_fingerprint": scoring_fingerprint,
            }
        ),
    }
