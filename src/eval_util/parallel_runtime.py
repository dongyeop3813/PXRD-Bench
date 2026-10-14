"""Queue-independent multi-GPU orchestration for offline evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from src.eval_util.prediction_cache import (
    merge_prediction_cache_shards,
    remove_prediction_cache,
)
from src.eval_util.run_artifacts import EvaluationPaths, write_json


def parallel_devices(explicit: Any) -> list[str]:
    """Validate the caller-selected CUDA device list without probing availability."""
    values = explicit.split(",") if isinstance(explicit, str) else explicit or []
    devices = [str(value).strip() for value in values if str(value).strip()]
    if len(devices) < 2:
        raise ValueError(
            "parallel evaluation requires at least two runtime.parallel_devices; "
            "use runtime.device=cuda:N for one GPU"
        )
    if len(devices) != len(set(devices)):
        raise ValueError("runtime.parallel_devices contains duplicate devices")
    return devices


def resolve_parallel_devices(
    explicit: Any,
    inherited_visible_devices: str | None,
) -> list[str]:
    """Map logical shard indices through an inherited CUDA visibility mask."""
    devices = parallel_devices(explicit)
    if inherited_visible_devices is None:
        return devices
    visible = [
        value.strip() for value in inherited_visible_devices.split(",") if value.strip()
    ]
    resolved = []
    for device in devices:
        try:
            logical_index = int(device)
        except ValueError as exc:
            raise ValueError(
                "runtime.parallel_devices must contain logical integer indices "
                "when CUDA_VISIBLE_DEVICES is inherited"
            ) from exc
        if logical_index < 0 or logical_index >= len(visible):
            raise ValueError(
                f"logical GPU index {logical_index} is outside inherited "
                f"CUDA_VISIBLE_DEVICES={inherited_visible_devices}"
            )
        resolved.append(visible[logical_index])
    return resolved


def _launch_local_worker(
    *, command: list[str], device: str, log_path: Path, cwd: Path
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    return {
        "process": process,
        "pid": process.pid,
        "device": str(device),
        "log_path": str(log_path),
        "log_handle": log_handle,
    }


def wait_for_local_workers(
    workers: list[dict[str, Any]], *, timeout_seconds: float
) -> list[dict[str, Any]]:
    """Wait for local subprocesses and report shard logs on failure."""
    deadline = None if timeout_seconds <= 0 else time.monotonic() + timeout_seconds
    results = []
    try:
        for worker in workers:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                returncode = worker["process"].wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("parallel evaluation timed out") from exc
            result = {
                key: value
                for key, value in worker.items()
                if key not in {"process", "log_handle"}
            }
            result["returncode"] = int(returncode)
            results.append(result)
    except BaseException:
        for worker in workers:
            process = worker["process"]
            if process.poll() is None:
                process.terminate()
        for worker in workers:
            process = worker["process"]
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise
    finally:
        for worker in workers:
            worker["log_handle"].close()
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        detail = ", ".join(
            f"device {item['device']} exit={item['returncode']} log={item['log_path']}"
            for item in failed
        )
        raise RuntimeError(f"parallel evaluation shard failure: {detail}")
    return results


def _stop_started_workers(workers: list[dict[str, Any]]) -> None:
    """Best-effort cleanup when a later shard cannot be launched."""
    for worker in workers:
        process = worker.get("process")
        if process is not None and process.poll() is None:
            process.terminate()
    for worker in workers:
        process = worker.get("process")
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    for worker in workers:
        handle = worker.get("log_handle")
        if handle is not None and not handle.closed:
            handle.close()


def _worker_module(task: str) -> str:
    modules = {
        "stoich": "src.pxrd.eval_bragglite_pxrd_stoich_ckpt",
        "element_set": "src.pxrd.eval_bragglite_pxrd_element_set_ckpt",
    }
    try:
        return modules[task]
    except KeyError as exc:
        raise ValueError(f"unsupported parallel evaluation task: {task}") from exc


def _write_worker_config(
    raw_config: dict[str, Any], config_path: Path, checkpoint: Path
) -> None:
    config = dict(raw_config)
    checkpoint_key = "checkpoint" if config.get("task") == "csp" else "path"
    config["checkpoint"] = {
        **raw_config["checkpoint"],
        checkpoint_key: str(checkpoint),
    }
    if config.get("task") == "csp":
        config["run"] = {**raw_config["run"]}
    config["runtime"] = {
        **raw_config["runtime"],
        "parallel_devices": [],
    }
    config["hydra"] = {
        "job": {"chdir": False},
        "output_subdir": "hydra",
        "run": {"dir": "${run.output_dir}"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=config_path)


def run_parallel_offline_evaluation(
    *,
    task: str,
    raw_config: dict[str, Any],
    paths: EvaluationPaths,
    checkpoint: Path,
    launch_worker: Callable[..., dict[str, Any]] = _launch_local_worker,
    wait_for_workers: Callable[..., list[dict[str, Any]]] = wait_for_local_workers,
) -> dict[str, Any]:
    """Launch local sampling shards, merge their caches, and return metadata."""
    runtime = raw_config["runtime"]
    selected_devices = resolve_parallel_devices(
        runtime.get("parallel_devices"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    parallel_root = (paths.root / "hydra" / "parallel").resolve()
    worker_config = parallel_root / "worker.yaml"
    _write_worker_config(raw_config, worker_config, checkpoint)
    module = _worker_module(task)
    cwd = Path.cwd().resolve()
    workers = []
    worker_indices = []
    started = time.perf_counter()
    try:
        for shard_index, device in enumerate(selected_devices):
            worker_root = parallel_root / f"shard-{shard_index:03d}"
            command = [
                sys.executable,
                "-m",
                module,
                f"--config-path={parallel_root}",
                "--config-name=worker",
                f"run.output_dir={worker_root}",
                "run.force=true",
                "runtime.device=cuda",
                "runtime.parallel_worker=true",
                f"runtime.num_shards={len(selected_devices)}",
                f"runtime.shard_index={shard_index}",
                "runtime.parallel_devices=[]",
            ]
            workers.append(
                launch_worker(
                    command=command,
                    device=device,
                    log_path=worker_root / "worker.log",
                    cwd=cwd,
                )
            )
            worker_indices.append(worker_root / "samples" / "index.json")
    except BaseException:
        _stop_started_workers(workers)
        raise

    completed = wait_for_workers(
        workers,
        timeout_seconds=float(runtime.get("parallel_timeout_seconds", 0.0)),
    )
    missing = [path for path in worker_indices if not path.exists()]
    if missing:
        raise FileNotFoundError(f"parallel evaluation produced no cache: {missing[0]}")
    merge_prediction_cache_shards(worker_indices, paths.samples)
    for worker_index in worker_indices:
        remove_prediction_cache(worker_index)
        worker_manifest = worker_index.parent.parent / "manifest.json"
        if worker_manifest.exists():
            payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
            write_json(
                worker_manifest,
                {
                    **payload,
                    "status": "merged",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "merged_into": str(paths.prediction_index),
                },
            )
    return {
        "backend": "local_subprocess",
        "parallel_devices": selected_devices,
        "num_shards": len(selected_devices),
        "workers": completed,
        "worker_outputs": [str(path.parent.parent) for path in worker_indices],
        "sampling_and_merge_elapsed_seconds": time.perf_counter() - started,
    }
