"""Dependency-light semantic validation for composed evaluation configs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Literal

from src.eval_util.contracts import (
    MATCH_GATES,
    MATCHER_PROFILES,
    validate_canonical_evaluation_contract,
    validate_opxrd_split_path,
    validate_rruff_split_path,
)

TaskName = Literal["stoich", "element_set"]
_DEVICE = re.compile(r"^(?:cpu|cuda)(?::\d+)?$")


def parse_topk(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    values = (
        value.replace("[", "").replace("]", "").split(",")
        if isinstance(value, str)
        else value
    )
    try:
        result = tuple(sorted({int(item) for item in values if str(item).strip()}))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "topk must be comma-separated positive ints"
        ) from exc
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("topk must contain positive ints")
    return result


def _validate_device(value: Any) -> None:
    if _DEVICE.fullmatch(str(value).strip()) is None:
        raise ValueError(f"invalid device: {value}")


def _validate_checkpoint_source(args: Any) -> None:
    checkpoint = bool(str(getattr(args, "checkpoint", "") or "").strip())
    train_output_dir = bool(str(getattr(args, "train_output_dir", "") or "").strip())
    if checkpoint == train_output_dir:
        raise ValueError("provide exactly one checkpoint source")
    if getattr(args, "checkpoint_preference", "auto") not in {
        "auto",
        "best",
        "final",
        "step_latest",
        "epoch_latest",
    }:
        raise ValueError("checkpoint_preference is invalid")
    if getattr(args, "sample_mode", "ema") not in {"regular", "ema"}:
        raise ValueError("sample_mode must be regular or ema")


def _validate_minimums(args: Any, fields: tuple[tuple[str, int, bool], ...]) -> None:
    for name, minimum, allow_none in fields:
        value = getattr(args, name, None)
        if value is None and allow_none:
            continue
        if value is None or int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")


def validate_csp_eval_config(args: Any, *, cwd: Path | None = None) -> None:
    mode = str(getattr(args, "mode", ""))
    if mode not in {"run", "score", "merge_score"}:
        raise ValueError("run.mode must be run, score, or merge_score")
    if not str(getattr(args, "output_dir", "")).strip():
        raise ValueError("run.output_dir must be nonempty")
    topk = [int(value) for value in getattr(args, "topk", ())]
    if not topk or any(value <= 0 for value in topk):
        raise ValueError("scoring.topk must contain positive integers")
    if getattr(args, "matcher_profile", None) not in MATCHER_PROFILES:
        raise ValueError("scoring.matcher_profile is invalid")
    if getattr(args, "match_gate", None) not in MATCH_GATES:
        raise ValueError("scoring.match_gate is invalid")
    validate_canonical_evaluation_contract(
        mode=mode,
        topk=topk,
        matcher_profile=str(args.matcher_profile),
        match_gate=str(args.match_gate),
        allow_noncanonical_evaluation_contract=bool(
            getattr(args, "allow_noncanonical_evaluation_contract", False)
        ),
    )
    dataset_class = str(getattr(args, "dataset_class", "") or "")
    if mode == "run" and dataset_class in {
        "xrdiff_author_rruff",
        "xrdiff_author_rruff_simulated",
    }:
        target = getattr(args, "target_indices_path", None)
        if not target:
            raise ValueError(
                "RRUFF evaluation requires a frozen validation-48 or test-100 manifest"
            )
        path = Path(str(target)).expanduser()
        validate_rruff_split_path(
            path if path.is_absolute() or cwd is None else cwd / path
        )
    for name in ("workers", "pdd_k"):
        if int(getattr(args, name, 0)) <= 0:
            raise ValueError(f"{name} must be positive")
    if mode == "score":
        return
    _validate_checkpoint_source(args)
    if (
        mode == "merge_score"
        and len(list(getattr(args, "shard_output_dirs", []) or [])) < 2
    ):
        raise ValueError("run.shard_output_dirs requires at least two shard outputs")
    if getattr(args, "seed_mode", None) not in {"fixed", "step_offset"}:
        raise ValueError("seed_mode must be fixed or step_offset")
    if getattr(args, "target_selection", None) not in {"sequential", "random"}:
        raise ValueError("target_selection must be sequential or random")
    _validate_device(getattr(args, "device", ""))
    _validate_minimums(
        args,
        (
            ("num_samples", 0, False),
            ("sample_chunk_size", 1, False),
            ("candidate_chunk_size", 1, False),
            ("sample_seed", 0, True),
            ("sample_num_steps", 1, True),
            ("nmax", 1, True),
            ("cache_shard_size", 1, False),
            ("num_shards", 1, False),
        ),
    )
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")
    if getattr(args, "rng_mode", None) not in {
        "generator_list",
        "stateless_batch",
    }:
        raise ValueError("sampling.rng_mode must be generator_list or stateless_batch")
    timeout = float(getattr(args, "parallel_timeout_seconds", 0.0))
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError("parallel_timeout_seconds must be finite and nonnegative")
    devices = list(getattr(args, "parallel_devices", []) or [])
    if len(devices) == 1:
        raise ValueError("parallel_devices requires zero or at least two entries")
    if len(devices) != len(set(devices)):
        raise ValueError("parallel_devices contains duplicate entries")


def validate_pxrd_eval_config(args: Any, *, cwd: Path | None = None) -> None:
    matcher_profile = getattr(args, "matcher_profile", None)
    if matcher_profile not in MATCHER_PROFILES:
        raise ValueError(
            f"scoring.matcher_profile must be one of {tuple(MATCHER_PROFILES)}"
        )
    match_gate = getattr(args, "match_gate", None)
    if match_gate not in MATCH_GATES:
        raise ValueError(f"scoring.match_gate must be one of {MATCH_GATES}")
    validate_canonical_evaluation_contract(
        mode=str(getattr(args, "command", "")),
        topk=args.topk,
        matcher_profile=str(matcher_profile),
        match_gate=str(match_gate),
        num_candidates=int(args.num_candidates),
        allow_noncanonical_candidate_budget=bool(
            getattr(args, "allow_noncanonical_candidate_budget", False)
        ),
        allow_noncanonical_evaluation_contract=bool(
            getattr(args, "allow_noncanonical_evaluation_contract", False)
        ),
    )
    if getattr(args, "command", None) == "run":
        dataset_class = str(getattr(args, "dataset_class", "") or "")
        target = getattr(args, "target_indices_path", None)
        if dataset_class in {
            "xrdiff_author_rruff",
            "xrdiff_author_rruff_simulated",
        }:
            if not target:
                raise ValueError(
                    "RRUFF evaluation requires a frozen validation-48 or test-100 manifest"
                )
            path = Path(str(target)).expanduser()
            validate_rruff_split_path(
                path if path.is_absolute() or cwd is None else cwd / path
            )
        if dataset_class == "frozen_phase_pxrd":
            if getattr(args, "nmax", None) is None:
                raise ValueError("frozen phase evaluation requires explicit data.nmax")
            from src.data.experimental_pxrd import frozen_phase_records

            path = Path(str(args.data_root)).expanduser()
            frozen_phase_records(
                path if path.is_absolute() or cwd is None else cwd / path,
                split=str(args.split),
                nmax=int(args.nmax),
            )
        if dataset_class == "experimental_pxrd":
            if getattr(args, "nmax", None) is None:
                raise ValueError(
                    "opXRD evaluation requires explicit data.nmax=20 or 52"
                )
            path = Path(str(args.data_root)).expanduser()
            validate_opxrd_split_path(
                path if path.is_absolute() or cwd is None else cwd / path,
                split=str(args.split),
                nmax=int(args.nmax),
            )
    override = getattr(args, "diagnostic_pxrd_features_path", None)
    override_hash = getattr(args, "diagnostic_pxrd_features_sha256", None)
    if override or override_hash:
        import hashlib

        if (
            not override
            or not override_hash
            or getattr(args, "dataset_class", None)
            not in {"experimental_pxrd", "xrdiff_author_rruff"}
            or getattr(args, "sampling_backend", None) != "edm"
            or not getattr(args, "allow_noncanonical_evaluation_contract", False)
        ):
            raise ValueError(
                "diagnostic PXRD features require path/hash, experimental_pxrd or xrdiff_author_rruff, edm, and explicit noncanonical opt-in"
            )
        path = Path(str(override)).expanduser()
        if not path.is_absolute() and cwd is not None:
            path = cwd / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != override_hash:
            raise ValueError("diagnostic PXRD feature SHA-256 mismatch")
    threads = getattr(args, "torch_num_threads", None)
    if threads is not None and int(threads) <= 0:
        raise ValueError("runtime.torch_num_threads must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    matcher_timeout = float(getattr(args, "matcher_timeout_seconds", 0.0))
    if not math.isfinite(matcher_timeout) or matcher_timeout < 0.0:
        raise ValueError("matcher_timeout_seconds must be finite and nonnegative")
    if args.pdd_k <= 0:
        raise ValueError("pdd_k must be positive")
    metre_topk = tuple(getattr(args, "metre_topk", ()))
    if metre_topk and (min(metre_topk) <= 0 or max(metre_topk) > args.num_candidates):
        raise ValueError("scoring.metre_topk must be within the candidate budget")
    if getattr(args, "command", None) != "run":
        return
    backend = str(getattr(args, "sampling_backend", "edm"))
    if backend != "edm":
        raise ValueError("sampling.backend must be edm")
    if args.num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if args.num_candidates < max(args.topk):
        raise ValueError(
            f"num_candidates={args.num_candidates} is smaller than max(topk)={max(args.topk)}"
        )
    if args.chunk_size <= 0 or args.candidate_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if getattr(args, "cache_shard_size", 512) <= 0:
        raise ValueError("cache_shard_size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
