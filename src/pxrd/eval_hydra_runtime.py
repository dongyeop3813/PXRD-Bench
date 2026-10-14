"""Hydra adapter shared by task-specific evaluation entrypoints."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from src.eval_util.run_artifacts import EvaluationPaths
from src.eval_util.config_validation import (
    TaskName,
    parse_topk,
    validate_pxrd_eval_config as validate_args,
)


def run_command(*args, **kwargs):
    from src.pxrd.eval_task_runtime import run_command as runtime_run_command

    return runtime_run_command(*args, **kwargs)


def _plain_config(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must resolve to a mapping")
    payload.pop("hydra", None)
    return payload


def pxrd_namespace(cfg: DictConfig) -> argparse.Namespace:
    raw = _plain_config(cfg)
    checkpoint = raw["checkpoint"]
    data = raw["data"]
    sampling = raw["sampling"]
    scoring = raw["scoring"]
    runtime = raw["runtime"]
    run = raw["run"]
    output_dir = Path(str(run["output_dir"]))
    return argparse.Namespace(
        command=str(run["mode"]),
        checkpoint=str(checkpoint["path"] or ""),
        sample_mode=str(checkpoint["weights"]),
        dataset_name=str(data["name"]),
        dataset_class=data["dataset_class"],
        data_root=str(data["root"]),
        split=str(data["split"]),
        nmax=data["nmax"],
        allow_nmax_extrapolation=bool(data.get("allow_nmax_extrapolation", False)),
        real_pxrd_variant=str(data["real_pxrd_variant"]),
        real_pxrd_peak_preprocessing=str(data["real_pxrd_peak_preprocessing"]),
        experimental_background_subtraction=bool(
            data["experimental_background_subtraction"]
        ),
        diagnostic_pxrd_features_path=data.get("diagnostic_pxrd_features_path"),
        diagnostic_pxrd_features_sha256=data.get("diagnostic_pxrd_features_sha256"),
        structure_cache=data["structure_cache"],
        simulated_profile_augmentation_seed=data["simulated_profile_augmentation_seed"],
        phase_reference_root=str(data["reference_root"]),
        num_targets=int(sampling["num_targets"]),
        sampling_backend=str(sampling["backend"]),
        target_indices_path=sampling["target_indices_path"],
        num_candidates=int(sampling["num_candidates"]),
        allow_noncanonical_candidate_budget=bool(
            sampling.get("allow_noncanonical_candidate_budget", False)
        ),
        allow_noncanonical_evaluation_contract=bool(
            run.get("allow_noncanonical_evaluation_contract", False)
        ),
        topk=parse_topk(scoring["topk"]),
        metre_topk=parse_topk(scoring["metre_topk"]) if scoring["metre_topk"] else (),
        seed=int(sampling["seed"]),
        workers=int(runtime["workers"]),
        out_dir=output_dir,
        matcher_profile=str(scoring["matcher_profile"]),
        match_gate=str(scoring["match_gate"]),
        matcher_timeout_seconds=float(scoring["matcher_timeout_seconds"]),
        pdd_k=int(scoring["pdd_k"]),
        match_only=bool(scoring["match_only"]),
        device=str(runtime["device"]),
        torch_num_threads=None
        if runtime["torch_num_threads"] is None
        else int(runtime["torch_num_threads"]),
        legacy_runtime_pca=bool(runtime["legacy_runtime_pca"]),
        chunk_size=int(sampling["chunk_size"]),
        compile=bool(runtime["compile"]),
        compile_mode=str(runtime["compile_mode"]),
        warmup_compile=bool(runtime["warmup_compile"]),
        profile_gpu=bool(runtime["profile_gpu"]),
        profile_phases=bool(runtime.get("profile_phases", False)),
        gpu_poll_interval_seconds=float(runtime["gpu_poll_interval_seconds"]),
        batch_candidates=bool(sampling["batch_candidates"]),
        candidate_chunk_size=int(sampling["candidate_chunk_size"]),
        rng_mode=str(sampling["rng_mode"]),
        num_shards=int(runtime["num_shards"]),
        shard_index=int(runtime["shard_index"]),
        cache_shard_size=int(runtime["cache_shard_size"]),
        parallel_devices=list(runtime["parallel_devices"] or []),
        parallel_worker=bool(runtime["parallel_worker"]),
        bf16=bool(runtime["bf16"]),
        sample_num_steps=sampling["num_steps"],
        solver=sampling["solver"],
        sigma_min=sampling["sigma_min"],
        sigma_max=sampling["sigma_max"],
        rho=sampling["rho"],
        S_churn=sampling["S_churn"],
        S_min=sampling["S_min"],
        S_max=sampling["S_max"],
        S_noise=sampling["S_noise"],
        aa_frac_max_scale=sampling["aa_frac_max_scale"],
        aa_rho_types=sampling["aa_rho_types"],
        aa_rho_coords=sampling["aa_rho_coords"],
        aa_rho_lattice=sampling["aa_rho_lattice"],
        sigma_min_types=sampling["sigma_min_types"],
        sigma_max_types=sampling["sigma_max_types"],
        rho_types=sampling["rho_types"],
        sigma_min_coords=sampling["sigma_min_coords"],
        sigma_max_coords=sampling["sigma_max_coords"],
        rho_coords=sampling["rho_coords"],
        sigma_min_lattice=sampling["sigma_min_lattice"],
        sigma_max_lattice=sampling["sigma_max_lattice"],
        rho_lattice=sampling["rho_lattice"],
        predictions=None,
    )


def _load_existing_manifest(paths: EvaluationPaths) -> dict[str, Any]:
    from src.eval_util.run_artifacts import MANIFEST_SCHEMA

    if not paths.manifest.exists():
        raise FileNotFoundError(f"missing evaluation manifest: {paths.manifest}")
    payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation manifest must be an object")
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported evaluation manifest schema: {payload.get('schema_version')!r}"
        )
    return payload


def _use_parallel_coordinator(args: argparse.Namespace) -> bool:
    if args.parallel_worker:
        return False
    if len(args.parallel_devices) == 1:
        raise ValueError(
            "runtime.parallel_devices requires at least two entries; use runtime.device=cuda:N for one GPU"
        )
    return len(args.parallel_devices) >= 2


def run_pxrd_hydra(task: TaskName, cfg: DictConfig) -> dict[str, Any] | None:
    from src.eval_util.checkpoint_runtime import load_checkpoint
    from src.eval_util.parallel_runtime import run_parallel_offline_evaluation
    from src.eval_util.prediction_cache import load_prediction_index
    from src.eval_util.run_artifacts import (
        EvaluationPaths,
        fail_run_if_running,
        finish_run,
        prepare_run,
        update_manifest_scoring,
        write_json,
    )

    raw = _plain_config(cfg)
    args = pxrd_namespace(cfg)
    paths = EvaluationPaths(Path(args.out_dir))
    previous_manifest = paths.manifest.read_bytes() if paths.manifest.exists() else None
    mode = str(args.command)
    if mode not in {"run", "score"}:
        raise ValueError("run.mode must be run or score")
    validate_args(args, cwd=Path.cwd())
    if mode == "run":
        if not args.checkpoint:
            raise ValueError("checkpoint.path is required for run.mode=run")
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
        request = {"task": task, **raw}
        request["run"] = {
            key: value for key, value in raw["run"].items() if key != "force"
        }
        manifest, reused = prepare_run(
            paths=paths,
            request=request,
            checkpoint=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
            force=bool(raw["run"]["force"]),
        )
        if reused:
            print(f"[eval] reusing complete run: {paths.root}")
            return json.loads(paths.metrics_json.read_text(encoding="utf-8"))
    else:
        manifest = _load_existing_manifest(paths)
        if not paths.prediction_index.exists():
            raise FileNotFoundError("score mode requires samples/index.json")
    started = time.perf_counter()
    thread_started = time.thread_time()
    previous_fast_sampler = os.environ.get("BRAGGLITE_FAST_SAMPLER")
    if bool(raw["runtime"]["fast_sampler"]):
        os.environ["BRAGGLITE_FAST_SAMPLER"] = "1"
    try:
        parallel_record = None
        if mode == "run" and _use_parallel_coordinator(args):
            parallel_record = run_parallel_offline_evaluation(
                task=task,
                raw_config=raw,
                paths=paths,
                checkpoint=Path(args.checkpoint).expanduser().resolve(),
            )
            args.command = "score"
            args.predictions = paths.prediction_index
        result = run_command(task, args)
        if result is None:
            _, worker_index = load_prediction_index(paths.prediction_index)
            worker_metadata = dict(worker_index.get("metadata", {}))
            write_json(
                paths.runtime_json,
                {
                    "mode": "sample",
                    "shard_index": int(args.shard_index),
                    "num_shards": int(args.num_shards),
                    "sampling_elapsed_seconds": worker_metadata.get(
                        "sampling_elapsed_seconds"
                    ),
                    "candidate_samples_per_second": worker_metadata.get(
                        "candidate_samples_per_second"
                    ),
                },
            )
            finish_run(paths, manifest, status="sampled")
            return None
        elapsed = time.perf_counter() - started
        main_thread_cpu = time.thread_time() - thread_started
        _, prediction_index = load_prediction_index(paths.prediction_index)
        prediction_metadata = dict(prediction_index.get("metadata", {}))
        sampling_elapsed = prediction_metadata.get("sampling_elapsed_seconds")
        sampling_rate = prediction_metadata.get("candidate_samples_per_second")
        runtime_record = {
            "mode": mode,
            "elapsed_seconds": elapsed,
            "sampling_elapsed_seconds": sampling_elapsed,
            "num_targets": int(result.get("num_targets", 0)),
            "num_candidates": int(result.get("num_candidates", 0)),
            "candidate_samples_per_second": sampling_rate,
            "nfe_per_sample": prediction_metadata.get("nfe_per_sample"),
            "denoiser_forward_calls_total": prediction_metadata.get(
                "denoiser_forward_calls_total"
            ),
            "end_to_end_candidate_samples_per_second": int(result.get("num_targets", 0))
            * int(result.get("num_candidates", 0))
            / max(elapsed, 1e-09),
        }
        phase_timings = dict(prediction_metadata.get("phase_timings", {}))
        phase_timings.update(result.get("_phase_timings", {}))
        if phase_timings:
            warmup_wall = float(phase_timings.get("compile_warmup_wall_seconds", 0.0))
            warmup_cpu = float(
                phase_timings.get("compile_warmup_main_thread_cpu_seconds", 0.0)
            )
            if not bool(raw["runtime"]["compile"]) or bool(
                raw["runtime"]["warmup_compile"]
            ):
                compile_excluded = max(0.0, elapsed - warmup_wall)
                compile_excluded_cpu = max(0.0, main_thread_cpu - warmup_cpu)
                phase_timings.update(
                    {
                        "compile_excluded_elapsed_seconds": compile_excluded,
                        "compile_excluded_main_thread_cpu_seconds": compile_excluded_cpu,
                        "compile_excluded_main_thread_wait_seconds": max(
                            0.0, compile_excluded - compile_excluded_cpu
                        ),
                        "compile_excluded_candidate_samples_per_second": int(
                            result.get("num_targets", 0)
                        )
                        * int(result.get("num_candidates", 0))
                        / max(compile_excluded, 1e-09),
                    }
                )
            runtime_record["phase_timings"] = phase_timings
        if parallel_record is not None:
            runtime_record["parallel"] = parallel_record
        gpu_profile = prediction_metadata.get("gpu_profile")
        if gpu_profile is not None:
            runtime_record["gpu_profile"] = gpu_profile
        write_json(paths.runtime_json, runtime_record)
        manifest = update_manifest_scoring(
            manifest,
            request={"task": task, **raw},
            metrics=dict(result.get("metrics", {})),
        )
        finish_run(paths, manifest, status="complete")
        return result
    except BaseException:
        fail_run_if_running(paths, previous_manifest=previous_manifest)
        raise
    finally:
        if previous_fast_sampler is None:
            os.environ.pop("BRAGGLITE_FAST_SAMPLER", None)
        else:
            os.environ["BRAGGLITE_FAST_SAMPLER"] = previous_fast_sampler


def print_metrics(result: dict[str, Any] | None) -> None:
    if result is not None:
        print(json.dumps(result.get("metrics", {}), indent=2))
