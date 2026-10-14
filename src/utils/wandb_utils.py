from __future__ import annotations

from src.bragglite.compat import METRICS_SCHEMA, RUN_SCHEMA

import json
import os
import platform
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.provenance import git_commit


_LOCAL_METRICS_PATH: Path | None = None
_LOCAL_RUN_METADATA_PATH: Path | None = None
_LOCAL_LOG_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_run_metadata(payload: dict[str, Any]) -> None:
    if _LOCAL_RUN_METADATA_PATH is None:
        return
    _LOCAL_RUN_METADATA_PATH.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _session_record(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "started_at",
        "command",
        "cwd",
        "hostname",
        "python",
        "platform",
        "git_commit",
        "cuda_visible_devices",
        "wandb",
        "config",
    )
    return {key: _jsonable(metadata[key]) for key in keys if key in metadata}


def _configure_local_logging(
    *,
    output_dir: str | None,
    project: str,
    name: str | None,
    config: dict[str, Any] | None,
    enabled: bool,
) -> dict[str, Any]:
    global _LOCAL_METRICS_PATH, _LOCAL_RUN_METADATA_PATH
    _LOCAL_METRICS_PATH = None
    _LOCAL_RUN_METADATA_PATH = None
    current_session = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "wandb": {"enabled": bool(enabled), "project": project, "name": name},
        "config": config or {},
    }
    metadata = {
        "schema_version": RUN_SCHEMA,
        **current_session,
        "last_started_at": current_session["started_at"],
        "resume_count": 0,
        "sessions": [_session_record(current_session)],
    }
    if output_dir:
        logs_dir = Path(output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        _LOCAL_METRICS_PATH = logs_dir / "metrics.jsonl"
        _LOCAL_RUN_METADATA_PATH = logs_dir / "run_metadata.json"
        if _LOCAL_RUN_METADATA_PATH.exists():
            try:
                previous = json.loads(
                    _LOCAL_RUN_METADATA_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                previous = None
            if isinstance(previous, dict):
                sessions = previous.get("sessions")
                if not isinstance(sessions, list) or not sessions:
                    sessions = [_session_record(previous)]
                metadata = {
                    **previous,
                    "schema_version": RUN_SCHEMA,
                    "last_started_at": current_session["started_at"],
                    "resume_count": len(sessions),
                    "sessions": [*sessions, _session_record(current_session)],
                    # Keep the top-level view useful for existing readers while
                    # the sessions list preserves every earlier invocation.
                    "wandb": current_session["wandb"],
                    "config": current_session["config"],
                }
        _write_run_metadata(metadata)
    return metadata


def init_wandb(
    project: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    enabled: bool = True,
    dir: str | None = None,
):
    metadata = _configure_local_logging(
        output_dir=dir,
        project=project,
        name=name,
        config=config,
        enabled=enabled,
    )
    if not enabled:
        return None
    import wandb

    run = wandb.init(
        project=project,
        config=config or {},
        name=name,
        entity=entity,
        group=group,
        tags=tags,
        notes=notes,
        mode=mode,
        dir=dir,
    )
    metadata["wandb"].update(
        {
            "id": getattr(run, "id", ""),
            "url": getattr(run, "url", ""),
            "mode": mode or "online",
        }
    )
    metadata["sessions"][-1]["wandb"] = dict(metadata["wandb"])
    _write_run_metadata(metadata)
    return run


def log_metrics(metrics: dict[str, Any], step: int | None = None, enabled: bool = True) -> None:
    if _LOCAL_METRICS_PATH is not None:
        record = {
            "schema_version": METRICS_SCHEMA,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "metrics": _jsonable(metrics),
        }
        with _LOCAL_LOG_LOCK:
            with _LOCAL_METRICS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not enabled:
        return
    import wandb

    wandb.log(metrics, step=step)


def log_summary(values: dict[str, Any], enabled: bool = True) -> None:
    """Persist run-level facts without creating W&B time-series metrics."""
    summary = _jsonable(values)
    if _LOCAL_RUN_METADATA_PATH is not None:
        with _LOCAL_LOG_LOCK:
            try:
                metadata = json.loads(
                    _LOCAL_RUN_METADATA_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                metadata = {}
            previous = metadata.get("summary")
            if not isinstance(previous, dict):
                previous = {}
            metadata["summary"] = {**previous, **summary}
            _write_run_metadata(metadata)
    if not enabled:
        return
    import wandb

    wandb.run.summary.update(summary)


def log_images(
    key: str,
    images: list[Any],
    step: int | None = None,
    enabled: bool = True,
) -> None:
    if not enabled or not images:
        return
    import wandb

    wandb.log({key: [wandb.Image(img) for img in images]}, step=step)


def save_artifact(path: str, enabled: bool = True) -> None:
    if not enabled:
        return
    import wandb

    wandb.save(path)
