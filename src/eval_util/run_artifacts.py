"""Canonical evaluation-run directories, manifests, and provenance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from src.data.mp20_tokens import tokens_to_eval_dict
from src.eval_util.lineage import (
    MANIFEST_SCHEMA,
    config_fingerprint,
    evaluation_lineage,
    normalized_request as _normalized_request,
    scoring_request as _scoring_request,
)
from src.utils.provenance import file_sha256

def _jsonable(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [_jsonable(value) for value in payload]
    if torch.is_tensor(payload):
        return payload.detach().cpu().tolist()
    if hasattr(payload, "item"):
        try:
            return payload.item()
        except (TypeError, ValueError):
            pass
    return str(payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            if on_row is not None:
                on_row(row)
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def structure_record_from_tokens(item: dict[str, Any], sample_index: int) -> dict[str, Any]:
    arrays = tokens_to_eval_dict(item, sample_idx=sample_index)
    return {
        "atom_types": [int(value) for value in arrays["atom_types"]],
        "frac_coords": torch.as_tensor(arrays["frac_coords"]).tolist(),
        "lengths": torch.as_tensor(arrays["lengths"]).reshape(-1).tolist(),
        "angles": torch.as_tensor(arrays["angles"]).reshape(-1).tolist(),
    }


def tokens_from_structure_record(
    record: dict[str, Any], *, nmax: int | None = None
) -> dict[str, torch.Tensor]:
    atom_types = torch.as_tensor(record["atom_types"], dtype=torch.long)
    count = int(atom_types.numel())
    width = max(count, int(nmax or count))
    a0 = torch.zeros(width, dtype=torch.long)
    frac = torch.zeros((width, 3), dtype=torch.float32)
    pad = torch.ones(width, dtype=torch.bool)
    a0[:count] = atom_types
    frac[:count] = torch.as_tensor(record["frac_coords"], dtype=torch.float32)
    pad[:count] = False
    y1 = torch.tensor(
        [*record["lengths"], *record["angles"]], dtype=torch.float32
    )
    return {"A0": a0, "F1": frac, "Y1": y1, "pad_mask": pad}


@dataclass(frozen=True)
class EvaluationPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def checkpoint_metadata(self) -> Path:
        return self.source / "checkpoint.json"

    @property
    def checkpoint_link(self) -> Path:
        return self.source / "checkpoint.pt"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def prediction_index(self) -> Path:
        return self.samples / "index.json"

    @property
    def shards(self) -> Path:
        return self.samples / "shards"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def metrics_json(self) -> Path:
        return self.metrics / "metrics.json"

    @property
    def per_sample(self) -> Path:
        return self.metrics / "per_sample.jsonl"

    @property
    def runtime_json(self) -> Path:
        return self.root / "runtime.json"

    def create(self) -> None:
        for directory in (
            self.source,
            self.samples,
            self.shards,
            self.metrics,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def checkpoint_provenance(checkpoint: Path, payload: dict[str, Any]) -> dict[str, Any]:
    resolved = checkpoint.expanduser().resolve()
    model_args = payload.get("model_args")
    training_run = payload.get("training_run")
    if not training_run and resolved.parent.name == "checkpoints":
        training_run = str(resolved.parent.parent)
    config = (
        payload.get("resolved_config")
        if isinstance(payload.get("resolved_config"), dict)
        else model_args if isinstance(model_args, dict) else {}
    )
    relative_path = None
    if training_run:
        try:
            relative_path = str(
                resolved.relative_to(Path(str(training_run)).expanduser().resolve())
            )
        except ValueError:
            relative_path = os.path.relpath(
                resolved, Path(str(training_run)).expanduser().resolve()
            )
    return {
        "path": str(resolved),
        "relative_path": relative_path,
        "sha256": file_sha256(resolved),
        "step": int(payload.get("step", -1)),
        "training_run": str(training_run or ""),
        "git_commit": payload.get("git_commit"),
        "has_resolved_config": isinstance(payload.get("resolved_config"), dict),
        "has_legacy_model_args": isinstance(model_args, dict),
        "resolved_config_hash": config_fingerprint(config),
        "config": config,
    }


def link_checkpoint(paths: EvaluationPaths, checkpoint: Path, provenance: dict[str, Any]) -> None:
    write_json(paths.checkpoint_metadata, provenance)
    link = paths.checkpoint_link
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(checkpoint.expanduser().resolve())


def prepare_run(
    *,
    paths: EvaluationPaths,
    request: dict[str, Any],
    checkpoint: Path | None,
    checkpoint_payload: dict[str, Any] | None,
    force: bool,
) -> tuple[dict[str, Any], bool]:
    """Prepare an evaluation run and return the manifest and reuse flag."""
    if checkpoint is None or checkpoint_payload is None:
        raise ValueError("run mode requires checkpoint provenance")
    provenance = checkpoint_provenance(checkpoint, checkpoint_payload)
    lineage = evaluation_lineage(
        request, checkpoint_sha256=str(provenance["sha256"])
    )
    existing: dict[str, Any] | None = None
    if paths.manifest.exists():
        try:
            candidate = json.loads(paths.manifest.read_text(encoding="utf-8"))
            existing = candidate if isinstance(candidate, dict) else None
        except (OSError, json.JSONDecodeError):
            existing = None
    if existing and not force:
        if (
            existing.get("schema_version") == MANIFEST_SCHEMA
            and existing.get("active_fingerprint")
            == lineage["active_fingerprint"]
            and existing.get("status") == "complete"
            and paths.prediction_index.exists()
            and paths.metrics_json.exists()
        ):
            return existing, True
        raise FileExistsError(
            f"evaluation run already exists with a different or incomplete request: {paths.root}"
        )

    paths.create()
    link_checkpoint(paths, checkpoint, provenance)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "running",
        "started_at": now,
        "updated_at": now,
        **lineage,
        "checkpoint": provenance,
        "scoring_history": [],
        "artifacts": {
            "predictions": str(paths.prediction_index.relative_to(paths.root)),
            "metrics": str(paths.metrics_json.relative_to(paths.root)),
            "per_sample": str(paths.per_sample.relative_to(paths.root)),
            "runtime": str(paths.runtime_json.relative_to(paths.root)),
        },
    }
    write_json(paths.manifest, manifest)
    return manifest, False


def update_manifest_scoring(
    manifest: dict[str, Any],
    *,
    request: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Activate a score-only config without changing sampling provenance."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported evaluation manifest schema: {manifest.get('schema_version')!r}"
        )
    normalized = _normalized_request(request)
    scoring_request = _scoring_request(
        normalized,
        task=manifest.get("sampling_request", {}).get("task"),
    )
    scoring_fingerprint = config_fingerprint(scoring_request)
    active_fingerprint = config_fingerprint(
        {
            "sampling_fingerprint": manifest["sampling_fingerprint"],
            "scoring_fingerprint": scoring_fingerprint,
        }
    )
    history = list(manifest.get("scoring_history", []))
    history.append(
        {
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "scoring_fingerprint": scoring_fingerprint,
            "scoring_request": scoring_request,
            "metrics": _jsonable(metrics),
        }
    )
    return {
        **manifest,
        "scoring_request": scoring_request,
        "scoring_fingerprint": scoring_fingerprint,
        "active_fingerprint": active_fingerprint,
        "scoring_history": history,
    }


def finish_run(paths: EvaluationPaths, manifest: dict[str, Any], *, status: str) -> None:
    if status not in {"complete", "sampled", "failed"}:
        raise ValueError(f"unsupported final evaluation status: {status}")
    updated = {
        **manifest,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths.manifest, updated)


def fail_run_if_running(
    paths: EvaluationPaths, *, previous_manifest: bytes | None = None
) -> None:
    """Mark a prepared run failed without masking the triggering exception."""
    try:
        current_manifest = paths.manifest.read_bytes()
        manifest = json.loads(current_manifest.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if previous_manifest is not None and current_manifest == previous_manifest:
        return
    if (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == MANIFEST_SCHEMA
        and manifest.get("status") == "running"
    ):
        finish_run(paths, manifest, status="failed")
