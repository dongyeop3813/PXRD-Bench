"""Shared generation and scoring runtime for PXRD evaluation tasks."""

from __future__ import annotations
from src.bragglite.compat import EVAL_SEED_DOMAIN
import argparse
import hashlib
import math
import resource
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Iterator
from typing import Any
import torch
from src.bragglite import mod1
from src.bragglite.sampler import clamp_lattice_latent, edm_sampler
from src.data.factory import build_token_dataset
from src.data.mp20_tokens import VZ
from src.eval_util.checkpoint_runtime import (
    SamplerConfig,
    build_model_from_checkpoint,
    checkpoint_type_encoding_name,
    checkpoint_model_args,
    cfg_value,
    load_checkpoint,
    maybe_apply_checkpoint_ema,
)
from src.eval_util.config_validation import (
    TaskName,
    validate_pxrd_eval_config as validate_args,
)
from src.eval_util.dataset import (
    CANONICAL_OPXRD_SPLITS,
    load_target_indices_manifest,
    validate_rruff_split_manifest,
)
from src.pxrd.eval_metrics import (
    MATCHER_PROFILES,
    MATCH_GATES,
    PredictionScoreAccumulator,
    reduced_formula_key,
    score_metre_group,
    score_prediction_row,
)
from src.eval_util.prediction_cache import (
    iter_prediction_cache,
    load_prediction_index,
    merge_prediction_caches,
    write_prediction_cache,
)
from src.eval_util.run_artifacts import (
    EvaluationPaths,
    structure_record_from_tokens,
    write_json,
    write_jsonl,
)
from src.models.lattice_repr import lattice_latent_to_y1
from src.models.type_encoding import build_type_encoding
from src.pxrd.conditioning import element_set_mask_from_a0
from src.utils.constants import DATASET_NMAX_DEFAULTS
from src.utils.provenance import file_sha256
from src.utils.seeding import stable_seed

SCHEMA_VERSION = "pxrd_task_predictions_npz_v1"
_token_item_to_record = structure_record_from_tokens


class _CudaPipelineMonitor:
    """Low-overhead NVML and allocator telemetry for prediction generation."""

    def __init__(
        self, device: torch.device, *, enabled: bool, interval_seconds: float
    ) -> None:
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda")
        self.interval_seconds = max(0.05, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilization: list[float] = []
        self._device_memory_utilization: list[float] = []
        self._poll_error: str | None = None

    def _poll_once(self) -> None:
        try:
            self._utilization.append(float(torch.cuda.utilization(self.device)))
            self._device_memory_utilization.append(
                float(torch.cuda.memory_usage(self.device))
            )
        except Exception as exc:
            self._poll_error = f"{type(exc).__name__}: {exc}"
            self._stop.set()

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._poll_once()

    def start(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._poll_once()
        if not self._stop.is_set():
            self._thread = threading.Thread(
                target=self._poll, name="bragglite-cuda-monitor", daemon=True
            )
            self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        torch.cuda.synchronize(self.device)
        if not self._stop.is_set():
            self._poll_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_seconds))
        total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
        record: dict[str, Any] = {
            "enabled": True,
            "poll_interval_seconds": self.interval_seconds,
            "poll_count": len(self._utilization),
            "gpu_utilization_mean_pct": sum(self._utilization) / len(self._utilization)
            if self._utilization
            else None,
            "gpu_utilization_min_pct": min(self._utilization)
            if self._utilization
            else None,
            "gpu_utilization_max_pct": max(self._utilization)
            if self._utilization
            else None,
            "device_memory_utilization_mean_pct": sum(self._device_memory_utilization)
            / len(self._device_memory_utilization)
            if self._device_memory_utilization
            else None,
            "device_memory_utilization_max_pct": max(self._device_memory_utilization)
            if self._device_memory_utilization
            else None,
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(self.device)
            / 2**30,
            "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(self.device)
            / 2**30,
            "device_total_memory_gb": total_memory / 2**30,
        }
        if self._poll_error is not None:
            record["poll_error"] = self._poll_error
        return record


def merge_prediction_shards(
    shard_paths: list[Path], output_root: Path, metadata_path: Path | None = None
) -> Path:
    """Compatibility name for the common NPZ cache merger."""
    del metadata_path
    return merge_prediction_caches(shard_paths, output_root)


def _sample_id(item: dict[str, Any], dataset_index: int) -> str:
    value = item.get(
        "sample_id", item.get("mp_id", item.get("material_id", dataset_index))
    )
    if torch.is_tensor(value):
        value = value.item() if value.numel() == 1 else value.tolist()
    return str(value)


def _sampler_config(args: argparse.Namespace, model_args: dict[str, Any]) -> Any:
    keys = (
        "sample_num_steps",
        "solver",
        "sigma_min",
        "sigma_max",
        "rho",
        "S_churn",
        "S_min",
        "S_max",
        "S_noise",
        "aa_frac_max_scale",
        "aa_rho_types",
        "aa_rho_coords",
        "aa_rho_lattice",
        "sigma_min_types",
        "sigma_max_types",
        "rho_types",
        "sigma_min_coords",
        "sigma_max_coords",
        "rho_coords",
        "sigma_min_lattice",
        "sigma_max_lattice",
        "rho_lattice",
    )
    overrides = {key: getattr(args, key, None) for key in keys}
    cfg = SamplerConfig.resolve(
        model_args,
        overrides=overrides,
        defaults={
            "sample_num_steps": 100,
            "S_churn": 20.0,
            "S_noise": 1.0,
            "sigma_data_type": 1.0,
            "sigma_data_coord": 0.25,
            "sigma_data_lattice": 1.0,
        },
    )

    def inherit(arg_key: str, default: Any, model_key: str | None = None) -> Any:
        return cfg_value(
            getattr(args, arg_key, None), model_args, model_key or arg_key, default
        )

    return SimpleNamespace(
        **vars(cfg),
        backend=str(getattr(args, "sampling_backend", "edm")),
        bf16=bool(args.bf16),
        rng_mode=str(getattr(args, "rng_mode", "generator_list")),
        nfe_per_sample=None,
        denoiser_forward_calls_total=0,
    )


def _condition_batch(
    items: list[dict[str, Any]], *, model_args: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    cond = {
        "pxrd_d": torch.stack([item["pxrd_d"] for item in items]).to(device),
        "pxrd_i": torch.stack([item["pxrd_i"] for item in items]).to(device),
        "pxrd_peak_pad_mask": torch.stack(
            [item["pxrd_peak_pad_mask"] for item in items]
        )
        .to(device)
        .bool(),
    }
    for key in ("pxrd_position", "pxrd_intensity", "pxrd_prominence", "pxrd_width"):
        if key in items[0]:
            cond[key] = torch.stack([item[key] for item in items]).to(device)
    return cond


def _decode_atom_types(
    *,
    task: TaskName,
    items: list[dict[str, Any]],
    samples: dict[str, torch.Tensor],
    pad_mask: torch.Tensor,
    element_mask: torch.Tensor,
    type_encoding: Any,
) -> torch.Tensor:
    if task == "stoich":
        return torch.stack([item["A0"] for item in items]).cpu()
    type_values = samples["type"].detach()
    return type_encoding.decode_logits_to_A0(
        type_values,
        pad_mask.to(type_values.device),
        allowed_mask=element_mask.to(type_values.device),
    ).cpu()


def _records_from_samples(
    *,
    task: TaskName,
    items: list[dict[str, Any]],
    batch_indices: list[int],
    samples: dict[str, torch.Tensor],
    sampler: Any,
    element_mask: torch.Tensor,
    type_encoding: Any,
    pad_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    pad_cpu = pad_mask.cpu()
    real = ~pad_cpu
    pred_a0 = _decode_atom_types(
        task=task,
        items=items,
        samples=samples,
        pad_mask=pad_cpu,
        element_mask=element_mask,
        type_encoding=type_encoding,
    )
    pred_a0 = torch.where(real, pred_a0, torch.zeros_like(pred_a0))
    frac = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
    frac = torch.where(real[..., None], frac, torch.zeros_like(frac))
    latent = clamp_lattice_latent(samples["lat"].detach().cpu(), sampler.lattice_repr)
    y1 = lattice_latent_to_y1(latent, lattice_repr=sampler.lattice_repr)
    y1 = clamp_lattice_latent(y1, lattice_repr="y1")
    targets = torch.stack([item["A0"] for item in items]).cpu()
    records = []
    for index, dataset_index in enumerate(batch_indices):
        candidate = {
            "A0": pred_a0[index],
            "F1": frac[index],
            "Y1": y1[index],
            "pad_mask": pad_cpu[index],
        }
        record = structure_record_from_tokens(candidate, dataset_index)
        if task != "stoich":
            correct = ((pred_a0[index] == targets[index]) & real[index]).sum()
            record["type_token_correct"] = int(correct.item())
            record["type_token_total"] = int(real[index].sum().item())
        records.append(record)
    return records


def _sample_items(
    *,
    task: TaskName,
    model: Any,
    type_encoding: Any,
    dataset: Any,
    indices: list[int],
    model_args: dict[str, Any],
    sampler: Any,
    device: torch.device,
    seed: int,
    chunk_size: int,
    sample_seeds: list[int] | None = None,
    pad_final_batch: bool = False,
    phase_timings: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    autocast_dtype = torch.bfloat16 if sampler.bf16 else None
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            batch_indices = indices[start : start + chunk_size]
            batch_seeds = (
                sample_seeds[start : start + len(batch_indices)]
                if sample_seeds is not None
                else None
            )
            real_batch_size = len(batch_indices)
            if pad_final_batch and real_batch_size < chunk_size:
                pad_count = chunk_size - real_batch_size
                batch_indices = [*batch_indices, *[batch_indices[-1]] * pad_count]
                if batch_seeds is not None:
                    batch_seeds = [*batch_seeds, *[batch_seeds[-1]] * pad_count]
            use_stateless_rng = (
                str(getattr(sampler, "rng_mode", "generator_list")) == "stateless_batch"
            )
            generators = (
                [
                    torch.Generator(device=device).manual_seed(item_seed)
                    for item_seed in batch_seeds
                ]
                if batch_seeds is not None and (not use_stateless_rng)
                else None
            )
            generator_batch_sizes = (
                [1] * len(generators) if generators is not None else None
            )
            input_started = time.perf_counter()
            input_cpu_started = time.thread_time()
            getitems = getattr(dataset, "__getitems__", None)
            items = (
                getitems(batch_indices)
                if getitems is not None
                else [dataset[index] for index in batch_indices]
            )
            a0 = torch.stack([item["A0"] for item in items]).to(device)
            pad_mask = (
                torch.stack([item["pad_mask"] for item in items]).to(device).bool()
            )
            cond = _condition_batch(items, model_args=model_args, device=device)
            if all(("elem_set_mask" in item for item in items)):
                elem_mask = torch.stack(
                    [torch.as_tensor(item["elem_set_mask"]).bool() for item in items]
                )
            else:
                elem_mask = element_set_mask_from_a0(
                    torch.stack([item["A0"] for item in items]),
                    torch.stack([item["pad_mask"] for item in items]),
                )
            if task != "stoich" and bool(model_args.get("element_set_cond", False)):
                cond["elem_set_mask"] = elem_mask.to(device)
            fixed_types = (
                type_encoding.encode_from_A0(a0=a0, pad_mask=pad_mask)
                if task == "stoich"
                else None
            )
            if phase_timings is not None:
                phase_timings["input_prep_h2d_wall_seconds"] = phase_timings.get(
                    "input_prep_h2d_wall_seconds", 0.0
                ) + (time.perf_counter() - input_started)
                phase_timings["input_prep_h2d_main_thread_cpu_seconds"] = (
                    phase_timings.get("input_prep_h2d_main_thread_cpu_seconds", 0.0)
                    + time.thread_time()
                    - input_cpu_started
                )
            sampler_stats: dict[str, int | float | str] = {}
            sampler_started = time.perf_counter()
            sampler_cpu_started = time.thread_time()
            gpu_started = gpu_finished = None
            if phase_timings is not None and device.type == "cuda":
                gpu_started = torch.cuda.Event(enable_timing=True)
                gpu_finished = torch.cuda.Event(enable_timing=True)
                gpu_started.record()
                torch.cuda.nvtx.range_push("pxrd_edm_sampler")
            samples = edm_sampler(
                model=model,
                pad_mask=pad_mask,
                type_dim=type_encoding.type_dim,
                num_steps=sampler.sample_num_steps,
                solver=getattr(sampler, "solver", "heun"),
                sigma_min=sampler.sigma_min,
                sigma_max=sampler.sigma_max,
                rho=sampler.rho,
                S_churn=sampler.S_churn,
                S_min=sampler.S_min,
                S_max=sampler.S_max,
                S_noise=sampler.S_noise,
                sigma_data_type=sampler.sigma_data_type,
                sigma_data_coord=sampler.sigma_data_coord,
                sigma_data_lat=sampler.sigma_data_lattice,
                generator=generators if generators is not None else generator,
                generator_batch_sizes=generator_batch_sizes,
                sample_seeds=batch_seeds if use_stateless_rng else None,
                autocast_dtype=autocast_dtype,
                fixed_atom_types=fixed_types,
                skip_type_scaling=task == "stoich",
                aa_frac_max_scale=sampler.aa_frac_max_scale,
                aa_rho_types=sampler.aa_rho_types,
                aa_rho_coords=sampler.aa_rho_coords,
                aa_rho_lattice=sampler.aa_rho_lattice,
                sigma_min_types=getattr(sampler, "sigma_min_types", None),
                sigma_max_types=getattr(sampler, "sigma_max_types", None),
                rho_types=getattr(sampler, "rho_types", None),
                sigma_min_coords=getattr(sampler, "sigma_min_coords", None),
                sigma_max_coords=getattr(sampler, "sigma_max_coords", None),
                rho_coords=getattr(sampler, "rho_coords", None),
                sigma_min_lattice=getattr(sampler, "sigma_min_lattice", None),
                sigma_max_lattice=getattr(sampler, "sigma_max_lattice", None),
                rho_lattice=getattr(sampler, "rho_lattice", None),
                lattice_repr=sampler.lattice_repr,
                cond=cond,
                sampler_stats=sampler_stats,
            )
            if gpu_finished is not None:
                torch.cuda.nvtx.range_pop()
                gpu_finished.record()
                gpu_finished.synchronize()
                phase_timings["gpu_sampler_seconds"] = (
                    phase_timings.get("gpu_sampler_seconds", 0.0)
                    + gpu_started.elapsed_time(gpu_finished) / 1000.0
                )
            if phase_timings is not None:
                phase_timings["sampler_wall_seconds"] = phase_timings.get(
                    "sampler_wall_seconds", 0.0
                ) + (time.perf_counter() - sampler_started)
                phase_timings["sampler_main_thread_cpu_seconds"] = (
                    phase_timings.get("sampler_main_thread_cpu_seconds", 0.0)
                    + time.thread_time()
                    - sampler_cpu_started
                )
            if sampler_stats:
                nfe = int(sampler_stats["nfe_per_sample"])
                existing = getattr(sampler, "nfe_per_sample", None)
                if existing is not None and int(existing) != nfe:
                    raise ValueError("sampler NFE changed across evaluation batches")
                sampler.nfe_per_sample = nfe
                sampler.denoiser_forward_calls_total += int(
                    sampler_stats["denoiser_calls"]
                )
            decode_started = time.perf_counter()
            decode_cpu_started = time.thread_time()
            if phase_timings is not None and device.type == "cuda":
                torch.cuda.nvtx.range_push("pxrd_d2h_decode")
            records = _records_from_samples(
                task=task,
                items=items,
                batch_indices=batch_indices,
                samples=samples,
                sampler=sampler,
                element_mask=elem_mask,
                type_encoding=type_encoding,
                pad_mask=pad_mask,
            )
            if phase_timings is not None:
                if device.type == "cuda":
                    torch.cuda.nvtx.range_pop()
                phase_timings["d2h_decode_wall_seconds"] = phase_timings.get(
                    "d2h_decode_wall_seconds", 0.0
                ) + (time.perf_counter() - decode_started)
                phase_timings["d2h_decode_main_thread_cpu_seconds"] = (
                    phase_timings.get("d2h_decode_main_thread_cpu_seconds", 0.0)
                    + time.thread_time()
                    - decode_cpu_started
                )
            output.extend(records[:real_batch_size])
    return output


def _build_dataset(
    args: argparse.Namespace, model_args: dict[str, Any]
) -> tuple[Any, str, int]:
    dataset_class = args.dataset_class or str(
        model_args.get("dataset_class", "mp20_tokens")
    )
    nmax = args.nmax
    if nmax is None:
        nmax = model_args.get("nmax") or DATASET_NMAX_DEFAULTS.get(
            args.dataset_name, 20
        )
    if model_args.get("pxrd_native_realpxrd_root"):
        raise ValueError(
            "Native baseline-input checkpoints are excluded from this Bragglite release"
        )
    representation = str(model_args.get("pxrd_representation", "legacy_di"))
    default_two_theta_max = (
        80.0 if representation in {"broadened_basic", "broadened_shape"} else 90.0
    )
    profile_seed = getattr(args, "simulated_profile_augmentation_seed", None)
    profile_augment = profile_seed is not None
    dataset = build_token_dataset(
        dataset_class,
        root=args.data_root,
        split=args.split,
        nmax=int(nmax),
        augment_translate=False,
        with_pxrd=False,
        real_pxrd_variant=args.real_pxrd_variant,
        real_pxrd_peak_preprocessing=getattr(
            args, "real_pxrd_peak_preprocessing", "official"
        ),
        experimental_background_subtraction=bool(
            getattr(args, "experimental_background_subtraction", True)
        ),
        real_pxrd_in_memory=False,
        structure_cache=getattr(args, "structure_cache", None)
        or model_args.get("structure_cache"),
        with_pxrd_peaks=True,
        pxrd_max_peaks=int(model_args.get("pxrd_max_peaks", 128)),
        pxrd_peak_selection=str(model_args.get("pxrd_peak_selection", "strongest")),
        pxrd_low_angle_fraction=float(model_args.get("pxrd_low_angle_fraction", 0.25)),
        pxrd_wavelength=float(model_args.get("pxrd_wavelength", 1.54184)),
        real_pxrd_two_theta_min=float(model_args.get("pxrd_two_theta_min", 5.0)),
        real_pxrd_two_theta_max=float(
            model_args.get("pxrd_two_theta_max", default_two_theta_max)
        ),
        pxrd_peak_augment=False,
        pxrd_profile_augment=profile_augment,
        pxrd_profile_augmentation_seed=profile_seed,
        pxrd_aug_jitter_deg=float(model_args.get("pxrd_aug_jitter_deg", 0.1)),
        pxrd_aug_int_scale_min=float(model_args.get("pxrd_aug_int_scale_min", 0.8)),
        pxrd_aug_int_scale_max=float(model_args.get("pxrd_aug_int_scale_max", 1.2)),
        pxrd_aug_width_scale_min=float(model_args.get("pxrd_aug_width_scale_min", 0.7)),
        pxrd_aug_width_scale_max=float(model_args.get("pxrd_aug_width_scale_max", 1.5)),
        pxrd_aug_drop_frac_max=float(model_args.get("pxrd_aug_drop_frac_max", 0.1)),
        pxrd_aug_zero_shift_deg=float(model_args.get("pxrd_aug_zero_shift_deg", 0.15)),
        pxrd_aug_eta_min=float(model_args.get("pxrd_aug_eta_min", 0.0)),
        pxrd_aug_eta_max=float(model_args.get("pxrd_aug_eta_max", 0.5)),
        pxrd_aug_background_max=float(model_args.get("pxrd_aug_background_max", 0.2)),
        pxrd_aug_noise_std_max=float(model_args.get("pxrd_aug_noise_std_max", 0.02)),
        pxrd_representation=representation,
        pxrd_peak_source=str(model_args.get("pxrd_peak_source", "bragg")),
        pxrd_peak_features=str(
            model_args.get("pxrd_peak_features", "position_intensity")
        ),
        pxrd_peak_position=str(model_args.get("pxrd_peak_position", "d")),
        pxrd_peak_position_normalized=bool(
            model_args.get("pxrd_peak_position_normalized", False)
        ),
        pxrd_peak_intensity_normalization=str(
            model_args.get("pxrd_peak_intensity_normalization", "selected_topk")
        ),
        pxrd_peak_min_prominence=float(model_args.get("pxrd_peak_min_prominence", 0.0)),
        pxrd_peak_min_relative_intensity=float(
            model_args.get("pxrd_peak_min_relative_intensity", 0.0)
        ),
        pxrd_peak_prominence_normalization=str(
            model_args.get("pxrd_peak_prominence_normalization", "none")
        ),
        pxrd_peak_width_unit=str(model_args.get("pxrd_peak_width_unit", "two_theta")),
        pxrd_peak_width_normalization=str(
            model_args.get("pxrd_peak_width_normalization", "none")
        ),
        pxrd_feature_source=str(model_args.get("pxrd_feature_source", "cached")),
        pxrd_profile_step=float(model_args.get("pxrd_profile_step", 0.1)),
        pxrd_caglioti_u=float(model_args.get("pxrd_caglioti_u", 0.1)),
        pxrd_caglioti_v=float(model_args.get("pxrd_caglioti_v", 0.01)),
        pxrd_caglioti_w=float(model_args.get("pxrd_caglioti_w", 0.1)),
        pxrd_pseudo_voigt_eta=float(model_args.get("pxrd_pseudo_voigt_eta", 0.1)),
        pxrd_dense_n_bins=int(model_args.get("pxrd_dense_n_bins", 8500)),
        pxrd_augmentation="profile_randomization" if profile_augment else "none",
        pxrd_dense_profile_augment=profile_augment,
        phase_reference_root=str(getattr(args, "phase_reference_root", "data/mp20")),
    )
    override = getattr(args, "diagnostic_pxrd_features_path", None)
    if override:
        from src.data.experimental_pxrd import apply_diagnostic_features

        apply_diagnostic_features(
            dataset, override, args.diagnostic_pxrd_features_sha256, representation
        )
    return (dataset, dataset_class, int(nmax))


def _validate_dataset_contract(
    dataset: Any,
    *,
    checkpoint_nmax: int,
    backend: str,
    allow_nmax_extrapolation: bool = False,
) -> None:
    provenance = getattr(dataset, "dataset_provenance", None)
    if not provenance:
        return
    evaluation = provenance.get("evaluation") or {}
    dataset_id = provenance.get("dataset_id", "dataset")
    if (
        str(dataset_id).startswith("opXRD_")
        and dataset_id not in CANONICAL_OPXRD_SPLITS
    ):
        raise ValueError(
            "new opXRD evaluations require a frozen v2 validation/test archive"
        )
    required_nmax = evaluation.get("nmax")
    if (
        required_nmax is not None
        and int(checkpoint_nmax) != int(required_nmax)
        and (
            not (allow_nmax_extrapolation and int(required_nmax) > int(checkpoint_nmax))
        )
    ):
        raise ValueError(
            f"{dataset_id} requires a checkpoint with nmax={required_nmax}, got nmax={checkpoint_nmax}"
        )
    supported_backend = evaluation.get("supported_backend")
    if supported_backend is not None and str(backend) != str(supported_backend):
        raise ValueError(
            f"{dataset_id} supports sampling.backend={supported_backend!r}, got {backend!r}"
        )


def _target_indices(
    args: argparse.Namespace, dataset_size: int, dataset_class: str
) -> list[int]:
    manifest_path = getattr(args, "target_indices_path", None)
    if manifest_path:
        indices, payload = load_target_indices_manifest(
            manifest_path, dataset_size=dataset_size, split=str(args.split)
        )
        if dataset_class in {"xrdiff_author_rruff", "xrdiff_author_rruff_simulated"}:
            validate_rruff_split_manifest(payload)
        if int(args.num_targets) > 0 and int(args.num_targets) != len(indices):
            raise ValueError(
                "sampling.num_targets must be 0 or equal the target manifest size"
            )
    else:
        if dataset_class in {"xrdiff_author_rruff", "xrdiff_author_rruff_simulated"}:
            raise ValueError(
                "RRUFF evaluation requires a frozen validation-48 or test-100 manifest"
            )
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        if args.num_targets > 0:
            indices = indices[: args.num_targets]
    start = len(indices) * int(args.shard_index) // int(args.num_shards)
    end = len(indices) * (int(args.shard_index) + 1) // int(args.num_shards)
    return indices[start:end]


def _candidate_seed(
    task: TaskName, args: argparse.Namespace, dataset_index: int, candidate_index: int
) -> int:
    return stable_seed(
        EVAL_SEED_DOMAIN, task, int(args.seed), int(dataset_index), int(candidate_index)
    )


def _rank_rng(
    task: TaskName,
    args: argparse.Namespace,
    dataset_indices: list[int],
    rank: int,
    rng_mode: str,
) -> tuple[int, list[int] | None]:
    if rng_mode == "historical_rank_stream":
        return (int(args.seed) + 1000003 * rank, None)
    return (
        int(args.seed),
        [
            _candidate_seed(task, args, dataset_index, rank)
            for dataset_index in dataset_indices
        ],
    )


def _prediction_rows(
    *,
    task: TaskName,
    args: argparse.Namespace,
    model: Any,
    type_encoding: Any,
    dataset: Any,
    indices: list[int],
    model_args: dict[str, Any],
    sampler: Any,
    device: torch.device,
    metadata: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    sampling_elapsed = 0.0
    phase_timings = metadata.get("phase_timings")
    if args.batch_candidates:
        target_block_size = max(
            1, int(args.candidate_chunk_size) // int(args.num_candidates)
        )
    else:
        target_block_size = max(1, int(args.chunk_size))
    for start in range(0, len(indices), target_block_size):
        block = indices[start : start + target_block_size]
        candidates_by_rank: list[list[dict[str, Any]]]
        block_started = time.perf_counter()
        if args.batch_candidates:
            repeated = [
                dataset_index
                for rank in range(args.num_candidates)
                for dataset_index in block
            ]
            sample_seeds = [
                _candidate_seed(task, args, dataset_index, rank)
                for rank in range(args.num_candidates)
                for dataset_index in block
            ]
            flat = _sample_items(
                task=task,
                model=model,
                type_encoding=type_encoding,
                dataset=dataset,
                indices=repeated,
                model_args=model_args,
                sampler=sampler,
                device=device,
                seed=int(args.seed),
                chunk_size=max(1, int(args.candidate_chunk_size)),
                sample_seeds=sample_seeds,
                pad_final_batch=bool(getattr(args, "compile", False)),
                phase_timings=phase_timings,
            )
            width = len(block)
            candidates_by_rank = [
                flat[rank * width : (rank + 1) * width]
                for rank in range(args.num_candidates)
            ]
        else:
            candidates_by_rank = []
            for rank in range(args.num_candidates):
                rank_seed, sample_seeds = _rank_rng(
                    task, args, block, rank, sampler.rng_mode
                )
                candidates_by_rank.append(
                    _sample_items(
                        task=task,
                        model=model,
                        type_encoding=type_encoding,
                        dataset=dataset,
                        indices=block,
                        model_args=model_args,
                        sampler=sampler,
                        device=device,
                        seed=rank_seed,
                        chunk_size=max(1, int(args.chunk_size)),
                        sample_seeds=sample_seeds,
                        pad_final_batch=bool(getattr(args, "compile", False)),
                        phase_timings=phase_timings,
                    )
                )
        sampling_elapsed += time.perf_counter() - block_started
        for position, dataset_index in enumerate(block):
            item = dataset[dataset_index]
            yield {
                "sample_id": _sample_id(item, dataset_index),
                "dataset_index": int(dataset_index),
                "reference": structure_record_from_tokens(item, dataset_index),
                "candidates": [
                    candidate_batch[position] for candidate_batch in candidates_by_rank
                ],
                "strata": {
                    "parent_sample_id": str(item["parent_sample_id"]),
                    "phase_role": str(item["phase_role"]),
                    "thermo_class": str(item["thermo_class"]),
                    "minor_fraction": f"{float(item['minor_fraction']):.1f}",
                }
                if "parent_sample_id" in item
                else {},
            }
    metadata["sampling_elapsed_seconds"] = sampling_elapsed
    metadata["candidate_samples_per_second"] = (
        len(indices) * args.num_candidates / max(sampling_elapsed, 1e-09)
    )
    metadata["nfe_per_sample"] = getattr(sampler, "nfe_per_sample", None)
    metadata["denoiser_forward_calls_total"] = getattr(
        sampler, "denoiser_forward_calls_total", None
    )
    if isinstance(phase_timings, dict):
        sampling_cpu = sum(
            (
                float(phase_timings.get(key, 0.0))
                for key in (
                    "input_prep_h2d_main_thread_cpu_seconds",
                    "sampler_main_thread_cpu_seconds",
                    "d2h_decode_main_thread_cpu_seconds",
                )
            )
        )
        phase_timings["sampling_main_thread_wait_seconds"] = max(
            0.0, sampling_elapsed - sampling_cpu
        )
        phase_timings["sampler_main_thread_wait_seconds"] = max(
            0.0,
            float(phase_timings.get("sampler_wall_seconds", 0.0))
            - float(phase_timings.get("sampler_main_thread_cpu_seconds", 0.0)),
        )
        phase_timings["sampling_unattributed_wall_seconds"] = max(
            0.0,
            sampling_elapsed
            - sum(
                (
                    float(phase_timings.get(key, 0.0))
                    for key in (
                        "input_prep_h2d_wall_seconds",
                        "sampler_wall_seconds",
                        "d2h_decode_wall_seconds",
                    )
                )
            ),
        )


def _profiled_prediction_rows(
    *,
    task: TaskName,
    args: argparse.Namespace,
    model: Any,
    type_encoding: Any,
    dataset: Any,
    indices: list[int],
    model_args: dict[str, Any],
    sampler: Any,
    device: torch.device,
    metadata: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    monitor = _CudaPipelineMonitor(
        device,
        enabled=bool(getattr(args, "profile_gpu", True)),
        interval_seconds=float(getattr(args, "gpu_poll_interval_seconds", 0.25)),
    )
    monitor.start()
    try:
        yield from _prediction_rows(
            task=task,
            args=args,
            model=model,
            type_encoding=type_encoding,
            dataset=dataset,
            indices=indices,
            model_args=model_args,
            sampler=sampler,
            device=device,
            metadata=metadata,
        )
    finally:
        metadata["gpu_profile"] = monitor.stop()


def generate_predictions(task: TaskName, args: argparse.Namespace) -> Path:
    paths = EvaluationPaths(Path(args.out_dir))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_args = checkpoint_model_args(checkpoint)
    if not bool(model_args.get("pxrd_cond", False)):
        raise ValueError("checkpoint was not trained with pxrd_cond=true")
    model, _ = build_model_from_checkpoint(checkpoint, device=device)
    maybe_apply_checkpoint_ema(
        model, checkpoint, args.sample_mode, log_prefix="[pxrd-task]"
    )
    model.eval()
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as exc:
            print(f"[pxrd-task] torch.compile disabled after failure: {exc}")
    if getattr(args, "torch_num_threads", None) is not None:
        torch.set_num_threads(int(args.torch_num_threads))
    type_encoding = build_type_encoding(
        checkpoint_type_encoding_name(checkpoint, model_args),
        vz=VZ,
        legacy_runtime_pca=bool(getattr(args, "legacy_runtime_pca", False)),
    )
    type_encoding_sha256 = hashlib.sha256(
        type_encoding.z_to_enc.contiguous().numpy().tobytes()
    ).hexdigest()
    dataset, dataset_class, nmax = _build_dataset(args, model_args)
    checkpoint_nmax = int(model_args.get("nmax") or 20)
    if nmax > checkpoint_nmax and (
        str(model_args.get("atom_positional_encoding", "none")) != "none"
    ):
        raise ValueError(
            "checkpoint has fixed atom slots; nmax extrapolation is unsupported"
        )
    indices = _target_indices(args, len(dataset), dataset_class)
    sampler = _sampler_config(args, model_args)
    _validate_dataset_contract(
        dataset,
        checkpoint_nmax=checkpoint_nmax,
        backend=sampler.backend,
        allow_nmax_extrapolation=bool(getattr(args, "allow_nmax_extrapolation", False)),
    )
    validate_args(args, cwd=Path.cwd())
    if sampler.rng_mode not in {
        "generator_list",
        "stateless_batch",
        "historical_rank_stream",
    }:
        raise ValueError(
            f"sampling.rng_mode must be generator_list|stateless_batch|historical_rank_stream, got {sampler.rng_mode!r}"
        )
    if sampler.rng_mode == "historical_rank_stream" and args.batch_candidates:
        raise ValueError(
            "historical_rank_stream requires sampling.batch_candidates=false"
        )
    warmup_wall = warmup_cpu = 0.0
    if args.compile and bool(getattr(args, "warmup_compile", False)) and indices:
        warmup_size = (
            min(len(indices) * args.num_candidates, args.candidate_chunk_size)
            if args.batch_candidates
            else min(len(indices), args.chunk_size)
        )
        warmup_indices = (indices * math.ceil(warmup_size / max(1, len(indices))))[
            :warmup_size
        ]
        warmup_started = time.perf_counter()
        warmup_cpu_started = time.thread_time()
        _sample_items(
            task=task,
            model=model,
            type_encoding=type_encoding,
            dataset=dataset,
            indices=warmup_indices,
            model_args=model_args,
            sampler=sampler,
            device=device,
            seed=int(args.seed),
            chunk_size=warmup_size,
        )
        warmup_wall = time.perf_counter() - warmup_started
        warmup_cpu = time.thread_time() - warmup_cpu_started
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "torch_num_threads": int(torch.get_num_threads()),
        "legacy_runtime_pca": bool(getattr(args, "legacy_runtime_pca", False)),
        "type_encoding_sha256": type_encoding_sha256,
        "dataset_name": args.dataset_name,
        "dataset_class": dataset_class,
        "dataset_provenance": getattr(dataset, "dataset_provenance", None),
        "pxrd_peak_min_prominence": float(
            model_args.get("pxrd_peak_min_prominence", 0.0)
        ),
        "data_root": args.data_root,
        "real_pxrd_peak_preprocessing": getattr(
            args, "real_pxrd_peak_preprocessing", "official"
        ),
        "split": args.split,
        "nmax": nmax,
        "checkpoint_training_nmax": checkpoint_nmax,
        "nmax_extrapolation": nmax > checkpoint_nmax,
        "oracle_num_atoms": True,
        "fixed_atom_types": task == "stoich",
        "element_set_source": "target_formula" if task == "element_set" else None,
        "num_candidates": args.num_candidates,
        "prediction_targets": len(indices),
        "seed": args.seed,
        "target_indices_path": str(
            Path(args.target_indices_path).expanduser().resolve()
        )
        if getattr(args, "target_indices_path", None)
        else None,
        "rng_scheme": "per_target_candidate_stateless_v1"
        if sampler.rng_mode == "stateless_batch"
        else "per_target_candidate_v1",
        "batch_candidates": bool(args.batch_candidates),
        "chunk_size": args.chunk_size,
        "candidate_chunk_size": args.candidate_chunk_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sampler": vars(sampler),
    }
    if bool(getattr(args, "profile_phases", False)):
        metadata["phase_timings"] = {
            "compile_warmup_wall_seconds": warmup_wall,
            "compile_warmup_main_thread_cpu_seconds": warmup_cpu,
        }
    return write_prediction_cache(
        paths.samples,
        _profiled_prediction_rows(
            task=task,
            args=args,
            model=model,
            type_encoding=type_encoding,
            dataset=dataset,
            indices=indices,
            model_args=model_args,
            sampler=sampler,
            device=device,
            metadata=metadata,
        ),
        metadata=metadata,
        shard_size=getattr(args, "cache_shard_size", 512),
    )


def _score_worker(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, options = payload
    return score_prediction_row(row, **options)


def score_predictions(task: TaskName, args: argparse.Namespace) -> dict[str, Any]:
    paths = EvaluationPaths(Path(args.out_dir))
    predictions_path = Path(args.predictions or paths.prediction_index)
    _, prediction_index = load_prediction_index(predictions_path)
    metadata = dict(prediction_index.get("metadata", {}))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prediction schema_version does not match evaluator")
    if metadata.get("task") != task:
        raise ValueError(
            f"prediction task {metadata.get('task')!r} does not match {task!r} evaluator"
        )
    candidate_budget = int(prediction_index.get("candidate_budget", 0))
    if candidate_budget < max(args.topk):
        raise ValueError("stored predictions do not contain the requested top-k")
    options = {
        "topk": args.topk,
        "matcher_profile": args.matcher_profile,
        "match_gate": args.match_gate,
        "matcher_timeout_seconds": float(getattr(args, "matcher_timeout_seconds", 0.0)),
        "pdd_k": args.pdd_k,
        "include_formula": task != "stoich",
        "match_only": bool(getattr(args, "match_only", True)),
    }
    metre_topk = tuple(getattr(args, "metre_topk", ()))
    if metre_topk and candidate_budget < max(metre_topk):
        raise ValueError("stored predictions do not contain the requested METRe top-k")
    metre_groups: dict[tuple[tuple[int, int], ...], list[dict[str, Any]]] = {}

    def payloads():
        for row in iter_prediction_cache(predictions_path):
            if len(row["candidates"]) < max(args.topk):
                raise ValueError(
                    "a prediction row has fewer candidates than requested top-k"
                )
            if metre_topk:
                key = reduced_formula_key(row["reference"]["atom_types"])
                metre_groups.setdefault(key, []).append(
                    {
                        "reference": row["reference"],
                        "candidates": row["candidates"][: max(metre_topk)],
                    }
                )
            yield (row, options)

    prefix = {
        "stoich": "pxrd",
        "element_set": "pxrd_elem",
    }[task]
    include_formula = task != "stoich"
    accumulator = PredictionScoreAccumulator(
        topk=args.topk,
        prefix=prefix,
        include_formula=include_formula,
        include_auxiliary=not bool(getattr(args, "match_only", True)),
        include_rmsd=not bool(getattr(args, "match_only", True)),
    )
    strata_accumulators: dict[str, PredictionScoreAccumulator] = {}

    def update_accumulators(row: dict[str, Any]) -> None:
        accumulator.update(row)
        for name, value in row.get("strata", {}).items():
            key = f"{name}={value}"
            group = strata_accumulators.setdefault(
                key,
                PredictionScoreAccumulator(
                    topk=args.topk,
                    prefix=prefix,
                    include_formula=include_formula,
                    include_auxiliary=not bool(getattr(args, "match_only", True)),
                    include_rmsd=not bool(getattr(args, "match_only", True)),
                ),
            )
            group.update(row)

    if args.workers == 1:
        scored_rows = (_score_worker(payload) for payload in payloads())
        write_jsonl(paths.per_sample, scored_rows, on_row=update_accumulators)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            scored_rows = executor.map(_score_worker, payloads(), chunksize=8)
            write_jsonl(paths.per_sample, scored_rows, on_row=update_accumulators)
    metrics, coverage = accumulator.finalize()
    if metre_topk:
        metre_payloads = (
            (rows, metre_topk, args.matcher_profile, args.match_gate)
            for rows in metre_groups.values()
        )
        if args.workers == 1:
            metre_results = map(score_metre_group, metre_payloads)
            metre_executor = None
        else:
            metre_executor = ProcessPoolExecutor(max_workers=args.workers)
            metre_results = metre_executor.map(
                score_metre_group, metre_payloads, chunksize=8
            )
        metre_rmsds = {k: [] for k in metre_topk}
        try:
            for group in metre_results:
                for k, values in group.items():
                    metre_rmsds[k].extend(values)
        finally:
            if metre_executor is not None:
                metre_executor.shutdown()
        for k, values in metre_rmsds.items():
            matched = [value for value in values if value is not None]
            metrics[f"{prefix}_metre@{k}"] = (
                len(matched) / len(values) if values else 0.0
            )
            metrics[f"{prefix}_metre_rmsd@{k}"] = (
                sum(matched) / len(matched) if matched else None
            )
            coverage[f"{prefix}_metre@{k}"] = len(matched)
            coverage[f"{prefix}_metre_rmsd@{k}"] = len(matched)
    stratified = {}
    for key, group in sorted(strata_accumulators.items()):
        group_metrics, group_coverage = group.finalize()
        stratified[key] = {
            "num_targets": group.count,
            "metrics": group_metrics,
            "coverage": group_coverage,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "dataset_name": metadata.get("dataset_name"),
        "split": metadata.get("split"),
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "checkpoint_step": metadata.get("checkpoint_step"),
        "prediction_cache": str(predictions_path),
        "num_targets": accumulator.count,
        "num_candidates": candidate_budget,
        "topk": list(args.topk),
        "metre_topk": list(metre_topk),
        "matcher_profile": args.matcher_profile,
        "matcher": MATCHER_PROFILES[args.matcher_profile],
        "match_gate": args.match_gate,
        "matcher_timeout_seconds": float(getattr(args, "matcher_timeout_seconds", 0.0)),
        "recorded_match_gates": list(MATCH_GATES),
        "pdd_k": args.pdd_k,
        "match_only": bool(getattr(args, "match_only", True)),
        "metrics": metrics,
        "coverage": coverage,
        "stratified": stratified,
    }
    write_json(paths.metrics_json, result)
    return result


def run_command(task: TaskName, args: argparse.Namespace) -> dict[str, Any] | None:
    validate_args(args, cwd=Path.cwd())
    if args.command == "run":
        predictions_path = generate_predictions(task, args)
        if args.num_shards != 1 or bool(getattr(args, "parallel_worker", False)):
            return None
        args.predictions = predictions_path
        scoring_started = time.perf_counter()
        scoring_cpu_started = time.thread_time()
        children_started = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = score_predictions(task, args)
    else:
        scoring_started = time.perf_counter()
        scoring_cpu_started = time.thread_time()
        children_started = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = score_predictions(task, args)
    if bool(getattr(args, "profile_phases", False)):
        children_finished = resource.getrusage(resource.RUSAGE_CHILDREN)
        scoring_wall = time.perf_counter() - scoring_started
        scoring_cpu = time.thread_time() - scoring_cpu_started
        result["_phase_timings"] = {
            "cpu_scoring_wall_seconds": scoring_wall,
            "cpu_scoring_main_thread_cpu_seconds": scoring_cpu,
            "cpu_scoring_main_thread_wait_seconds": max(
                0.0, scoring_wall - scoring_cpu
            ),
            "cpu_scoring_worker_cpu_seconds": children_finished.ru_utime
            + children_finished.ru_stime
            - children_started.ru_utime
            - children_started.ru_stime,
        }
    return result
