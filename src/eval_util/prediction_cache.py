"""Lossless, sharded NPZ cache for evaluation references and predictions."""

from __future__ import annotations

from src.bragglite.compat import PREDICTION_CACHE_SCHEMA

import json
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from src.utils.provenance import file_sha256

CACHE_SCHEMA = PREDICTION_CACHE_SCHEMA
INDEX_NAME = "index.json"
_SHARD_LOCAL_METADATA_KEYS = {
    "candidate_samples_per_second",
    "denoiser_forward_calls_total",
    "gpu_profile",
    "phase_timings",
    "prediction_targets",
    "sampling_elapsed_seconds",
    "shard_index",
}


def _comparable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    comparable = {
        key: value
        for key, value in metadata.items()
        if key not in _SHARD_LOCAL_METADATA_KEYS
    }
    sampler = comparable.get("sampler")
    if isinstance(sampler, dict):
        comparable["sampler"] = {
            key: value
            for key, value in sampler.items()
            if key != "denoiser_forward_calls_total"
        }
    return comparable


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _structure_width(record: dict[str, Any]) -> int:
    return len(record.get("atom_types", []))


def _fill_structure(
    record: dict[str, Any],
    atom_types: np.ndarray,
    frac_coords: np.ndarray,
    lattice: np.ndarray,
    index: tuple[int, ...],
) -> int:
    count = _structure_width(record)
    if count == 0:
        return 0
    frac = np.asarray(record["frac_coords"], dtype=np.float32)
    lengths = np.asarray(record["lengths"], dtype=np.float32).reshape(3)
    angles = np.asarray(record["angles"], dtype=np.float32).reshape(3)
    if frac.shape != (count, 3):
        raise ValueError(f"fractional coordinates have shape {frac.shape}, expected {(count, 3)}")
    atom_types[index + (slice(0, count),)] = np.asarray(
        record["atom_types"], dtype=np.uint8
    )
    frac_coords[index + (slice(0, count), slice(None))] = frac
    lattice[index] = np.concatenate((lengths, angles))
    return count


def _rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[str]]:
    if not rows:
        raise ValueError("prediction cache shard cannot be empty")
    candidate_width = max(len(row.get("candidates", [])) for row in rows)
    if candidate_width <= 0:
        raise ValueError("every prediction cache shard must contain candidates")
    structures = [row.get("reference", {}) for row in rows]
    structures.extend(
        candidate for row in rows for candidate in row.get("candidates", [])
    )
    nmax = max(1, max(_structure_width(record) for record in structures))
    target_count = len(rows)

    reference_atom_types = np.zeros((target_count, nmax), dtype=np.uint8)
    reference_frac_coords = np.zeros((target_count, nmax, 3), dtype=np.float32)
    reference_lattice = np.zeros((target_count, 6), dtype=np.float32)
    reference_num_atoms = np.zeros(target_count, dtype=np.uint16)
    candidate_atom_types = np.zeros(
        (target_count, candidate_width, nmax), dtype=np.uint8
    )
    candidate_frac_coords = np.zeros(
        (target_count, candidate_width, nmax, 3), dtype=np.float32
    )
    candidate_lattice = np.zeros(
        (target_count, candidate_width, 6), dtype=np.float32
    )
    candidate_num_atoms = np.zeros(
        (target_count, candidate_width), dtype=np.uint16
    )
    candidate_count = np.zeros(target_count, dtype=np.uint16)
    type_token_correct = np.full((target_count, candidate_width), -1, dtype=np.int32)
    type_token_total = np.full((target_count, candidate_width), -1, dtype=np.int32)

    sample_ids: list[str] = []
    strata_json: list[str] = []
    dataset_indices = np.empty(target_count, dtype=np.int32)
    for target_index, row in enumerate(rows):
        sample_ids.append(str(row["sample_id"]))
        strata_json.append(
            json.dumps(row.get("strata", {}), sort_keys=True, separators=(",", ":"))
        )
        dataset_indices[target_index] = int(row.get("dataset_index", target_index))
        reference_num_atoms[target_index] = _fill_structure(
            row.get("reference", {}),
            reference_atom_types,
            reference_frac_coords,
            reference_lattice,
            (target_index,),
        )
        candidates = row.get("candidates", [])
        candidate_count[target_index] = len(candidates)
        for rank, candidate in enumerate(candidates):
            candidate_num_atoms[target_index, rank] = _fill_structure(
                candidate,
                candidate_atom_types,
                candidate_frac_coords,
                candidate_lattice,
                (target_index, rank),
            )
            if "type_token_correct" in candidate:
                type_token_correct[target_index, rank] = int(
                    candidate["type_token_correct"]
                )
            if "type_token_total" in candidate:
                type_token_total[target_index, rank] = int(
                    candidate["type_token_total"]
                )

    arrays = {
        "dataset_index": dataset_indices,
        "reference_num_atoms": reference_num_atoms,
        "reference_atom_types": reference_atom_types,
        "reference_frac_coords": reference_frac_coords,
        "reference_lattice": reference_lattice,
        "candidate_count": candidate_count,
        "candidate_num_atoms": candidate_num_atoms,
        "candidate_atom_types": candidate_atom_types,
        "candidate_frac_coords": candidate_frac_coords,
        "candidate_lattice": candidate_lattice,
        "type_token_correct": type_token_correct,
        "type_token_total": type_token_total,
        "strata_json": np.asarray(strata_json),
    }
    return arrays, sample_ids


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def write_prediction_cache(
    root: Path,
    rows: Iterable[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    shard_size: int = 512,
) -> Path:
    """Stream prediction rows into lossless compressed NPZ shards."""
    if shard_size <= 0:
        raise ValueError("prediction cache shard_size must be positive")
    root = Path(root)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    row_iter = iter(rows)
    seen_sample_ids: set[str] = set()
    generated_paths: list[Path] = []
    shard_records: list[dict[str, Any]] = []
    target_count = 0
    candidate_budget = 0
    try:
        for shard_index in range(2**31):
            chunk = list(islice(row_iter, shard_size))
            if not chunk:
                break
            chunk_ids = [str(row.get("sample_id", "")) for row in chunk]
            if any(not sample_id for sample_id in chunk_ids):
                raise ValueError("every prediction cache row requires a sample_id")
            if len(chunk_ids) != len(set(chunk_ids)) or seen_sample_ids.intersection(
                chunk_ids
            ):
                raise ValueError("prediction cache sample IDs must be unique")
            seen_sample_ids.update(chunk_ids)
            candidate_budget = max(
                candidate_budget,
                max(len(row.get("candidates", [])) for row in chunk),
            )
            write_started = time.perf_counter()
            write_cpu_started = time.thread_time()
            arrays, serialized_ids = _rows_to_arrays(chunk)
            path = (
                shard_dir
                / f"predictions-{generation}-{shard_index:05d}.npz"
            )
            _write_npz_atomic(path, arrays)
            generated_paths.append(path)
            shard_records.append(
                {
                    "path": str(path.relative_to(root)),
                    "target_start": target_count,
                    "target_count": len(chunk),
                    "sample_ids": serialized_ids,
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "nmax": int(arrays["candidate_atom_types"].shape[-1]),
                    "candidate_width": int(
                        arrays["candidate_atom_types"].shape[1]
                    ),
                }
            )
            phase_timings = metadata.get("phase_timings")
            if isinstance(phase_timings, dict):
                phase_timings["cache_write_wall_seconds"] = phase_timings.get(
                    "cache_write_wall_seconds", 0.0
                ) + (time.perf_counter() - write_started)
                phase_timings["cache_write_main_thread_cpu_seconds"] = (
                    phase_timings.get("cache_write_main_thread_cpu_seconds", 0.0)
                    + time.thread_time()
                    - write_cpu_started
                )
            target_count += len(chunk)
        if target_count == 0:
            raise ValueError("prediction cache cannot be empty")

        index = {
            "schema_version": CACHE_SCHEMA,
            "target_count": target_count,
            "candidate_budget": candidate_budget,
            "shard_size": shard_size,
            "metadata": metadata,
            "shards": shard_records,
        }
        index_path = root / INDEX_NAME
        _atomic_json(index_path, index)
    except BaseException:
        for path in generated_paths:
            path.unlink(missing_ok=True)
        raise

    published = set(generated_paths)
    for stale_shard in shard_dir.glob("predictions-*.npz"):
        if stale_shard not in published:
            stale_shard.unlink()
    return index_path


def load_prediction_index(path: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path)
    index_path = path / INDEX_NAME if path.is_dir() else path
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"unsupported prediction cache schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("shards"), list) or not payload["shards"]:
        raise ValueError("prediction cache index contains no shards")
    return index_path, payload


def _structure_record(
    atom_types: np.ndarray,
    frac_coords: np.ndarray,
    lattice: np.ndarray,
    count: int,
) -> dict[str, Any]:
    if count <= 0:
        return {}
    return {
        "atom_types": atom_types[:count].astype(int).tolist(),
        "frac_coords": frac_coords[:count].astype(np.float32).tolist(),
        "lengths": lattice[:3].astype(np.float32).tolist(),
        "angles": lattice[3:].astype(np.float32).tolist(),
    }


def iter_prediction_cache(path: Path, *, verify: bool = True) -> Iterator[dict[str, Any]]:
    """Yield cached rows shard-by-shard without loading the full run."""
    index_path, index = load_prediction_index(Path(path))
    root = index_path.parent
    yielded = 0
    for shard in index["shards"]:
        shard_path = root / shard["path"]
        if verify and file_sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"prediction cache checksum mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as archive:
            # Compressed NPZ members are decompressed on every __getitem__ call.
            # Materialize each shard once before iterating candidates.
            arrays = {name: archive[name] for name in archive.files}
            target_count = int(arrays["dataset_index"].shape[0])
            if target_count != int(shard["target_count"]):
                raise ValueError(f"prediction cache target count mismatch: {shard_path}")
            for local_index in range(target_count):
                reference = _structure_record(
                    arrays["reference_atom_types"][local_index],
                    arrays["reference_frac_coords"][local_index],
                    arrays["reference_lattice"][local_index],
                    int(arrays["reference_num_atoms"][local_index]),
                )
                candidates = []
                for rank in range(int(arrays["candidate_count"][local_index])):
                    candidate = _structure_record(
                        arrays["candidate_atom_types"][local_index, rank],
                        arrays["candidate_frac_coords"][local_index, rank],
                        arrays["candidate_lattice"][local_index, rank],
                        int(arrays["candidate_num_atoms"][local_index, rank]),
                    )
                    correct = int(arrays["type_token_correct"][local_index, rank])
                    total = int(arrays["type_token_total"][local_index, rank])
                    if correct >= 0:
                        candidate["type_token_correct"] = correct
                    if total >= 0:
                        candidate["type_token_total"] = total
                    candidates.append(candidate)
                yield {
                    "sample_id": str(shard["sample_ids"][local_index]),
                    "dataset_index": int(arrays["dataset_index"][local_index]),
                    "reference": reference,
                    "candidates": candidates,
                    "strata": json.loads(
                        str(arrays["strata_json"][local_index])
                    )
                    if "strata_json" in arrays
                    else {},
                }
                yielded += 1
    if yielded != int(index["target_count"]):
        raise ValueError(
            f"prediction cache yielded {yielded} targets, expected {index['target_count']}"
        )


def read_prediction_cache(
    path: Path, *, verify: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load cached rows and cache metadata."""
    _, index = load_prediction_index(Path(path))
    return list(iter_prediction_cache(path, verify=verify)), dict(index.get("metadata", {}))


def remove_prediction_cache(path: Path) -> None:
    """Remove one fully indexed cache after a verified merge."""
    index_path, index = load_prediction_index(Path(path))
    root = index_path.parent
    shard_parents: set[Path] = set()
    for shard in index["shards"]:
        shard_path = root / shard["path"]
        shard_path.unlink(missing_ok=True)
        shard_parents.add(shard_path.parent)
    index_path.unlink(missing_ok=True)
    for directory in sorted(shard_parents, key=lambda item: len(item.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def merge_prediction_caches(
    cache_paths: list[Path],
    output_root: Path,
    *,
    shard_size: int = 512,
) -> Path:
    """Merge compatible shard caches into one canonical target-ordered cache."""
    if not cache_paths:
        raise ValueError("no prediction caches were provided")
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for cache_path in cache_paths:
        cache_rows, cache_metadata = read_prediction_cache(cache_path)
        comparable = _comparable_metadata(cache_metadata)
        if metadata is None:
            metadata = comparable
        elif comparable != metadata:
            raise ValueError(f"incompatible prediction cache metadata: {cache_path}")
        rows.extend(cache_rows)
    rows.sort(key=lambda row: int(row["dataset_index"]))
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs across prediction caches")
    assert metadata is not None
    metadata["prediction_targets"] = len(rows)
    return write_prediction_cache(
        output_root, rows, metadata=metadata, shard_size=shard_size
    )


def merge_prediction_cache_shards(cache_paths: list[Path], output_root: Path) -> Path:
    """Merge worker caches by copying verified NPZ shards without decoding structures."""
    if not cache_paths:
        raise ValueError("no prediction caches were provided")
    output_root = Path(output_root)
    output_shards = output_root / "shards"
    output_shards.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex

    metadata: dict[str, Any] | None = None
    merged_shards: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_dataset_indices: set[int] = set()
    target_start = 0
    candidate_budget = 0
    for cache_path in cache_paths:
        index_path, index = load_prediction_index(Path(cache_path))
        comparable = _comparable_metadata(dict(index.get("metadata", {})))
        if metadata is None:
            metadata = comparable
        elif comparable != metadata:
            raise ValueError(f"incompatible prediction cache metadata: {cache_path}")
        candidate_budget = max(candidate_budget, int(index.get("candidate_budget", 0)))
        for source_shard in index["shards"]:
            source_path = index_path.parent / source_shard["path"]
            if file_sha256(source_path) != source_shard["sha256"]:
                raise ValueError(f"prediction cache checksum mismatch: {source_path}")
            with np.load(source_path, allow_pickle=False) as arrays:
                dataset_indices = [int(value) for value in arrays["dataset_index"]]
            if len(dataset_indices) != int(source_shard["target_count"]):
                raise ValueError(f"prediction cache target count mismatch: {source_path}")
            if len(dataset_indices) != len(
                set(dataset_indices)
            ) or seen_dataset_indices.intersection(dataset_indices):
                raise ValueError("duplicate dataset indices across prediction caches")
            seen_dataset_indices.update(dataset_indices)
            sample_ids = [str(value) for value in source_shard["sample_ids"]]
            if len(sample_ids) != len(dataset_indices):
                raise ValueError(f"prediction cache sample ID count mismatch: {source_path}")
            if len(sample_ids) != len(set(sample_ids)) or seen_sample_ids.intersection(
                sample_ids
            ):
                raise ValueError("duplicate sample IDs across prediction caches")
            seen_sample_ids.update(sample_ids)

            destination = (
                output_shards
                / f"predictions-{generation}-{len(merged_shards):05d}.npz"
            )
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, destination)
            target_count = int(source_shard["target_count"])
            merged_shards.append(
                {
                    **source_shard,
                    "path": str(destination.relative_to(output_root)),
                    "target_start": target_start,
                }
            )
            target_start += target_count

    assert metadata is not None
    metadata["prediction_targets"] = target_start
    payload = {
        "schema_version": CACHE_SCHEMA,
        "target_count": target_start,
        "candidate_budget": candidate_budget,
        "shard_size": max(int(shard["target_count"]) for shard in merged_shards),
        "metadata": metadata,
        "shards": merged_shards,
    }
    index_path = output_root / INDEX_NAME
    _atomic_json(index_path, payload)
    published = {
        output_root / str(shard["path"])
        for shard in merged_shards
    }
    for stale_shard in output_shards.glob("predictions-*.npz"):
        if stale_shard not in published:
            stale_shard.unlink()
    return index_path
