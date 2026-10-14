"""Compact, provenance-carrying cache for ragged Bragg reflection lists."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch


SCHEMA_VERSION = 2
_STRUCTURE_KEYS = ("A0", "F1", "Y1", "pad_mask")


@dataclass(frozen=True)
class BraggPeakCache:
    ids: tuple[str, ...]
    offsets: np.ndarray
    two_theta: np.ndarray
    intensity: np.ndarray
    manifest: dict

    def __post_init__(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Bragg cache ids must be unique")
        if self.offsets.shape != (len(self.ids) + 1,):
            raise ValueError("Bragg cache offsets do not match ids")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.two_theta):
            raise ValueError("Bragg cache offsets do not span the reflection arrays")
        if self.two_theta.shape != self.intensity.shape:
            raise ValueError("Bragg cache reflection arrays have different shapes")

    def items(self):
        for index, material_id in enumerate(self.ids):
            start, stop = map(int, self.offsets[index : index + 2])
            yield material_id, (self.two_theta[start:stop], self.intensity[start:stop])

    def as_dict(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return dict(self.items())


def build_bragg_cache(
    patterns: Mapping[str, tuple[np.ndarray, np.ndarray]]
    | Iterable[tuple[str, tuple[np.ndarray, np.ndarray]]],
    *,
    manifest: dict,
) -> BraggPeakCache:
    rows = list(patterns.items() if isinstance(patterns, Mapping) else patterns)
    ids: list[str] = []
    offsets = [0]
    positions: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    for material_id, (two_theta, intensity) in rows:
        tt = np.asarray(two_theta, dtype=np.float64).reshape(-1)
        it = np.asarray(intensity, dtype=np.float64).reshape(-1)
        if (
            tt.shape != it.shape
            or not np.isfinite(tt).all()
            or not np.isfinite(it).all()
        ):
            raise ValueError(f"invalid Bragg pattern for {material_id!r}")
        ids.append(str(material_id))
        positions.append(tt)
        intensities.append(it)
        offsets.append(offsets[-1] + len(tt))
    return BraggPeakCache(
        ids=tuple(ids),
        offsets=np.asarray(offsets, dtype=np.int64),
        two_theta=np.concatenate(positions) if positions else np.empty(0, np.float64),
        intensity=(
            np.concatenate(intensities) if intensities else np.empty(0, np.float64)
        ),
        manifest=dict(manifest),
    )


def structure_items_sha256(items: Iterable[Mapping]) -> str:
    """Hash the exact token fields consumed by the Bragg simulator."""
    digest = hashlib.sha256()
    for item in items:
        material_id = str(item["mp_id"]).encode("utf-8")
        digest.update(len(material_id).to_bytes(8, "big"))
        digest.update(material_id)
        for key in _STRUCTURE_KEYS:
            tensor = torch.as_tensor(item[key]).detach().cpu().contiguous()
            metadata = f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode("ascii")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_bragg_cache(path: str | Path, cache: BraggPeakCache) -> Path:
    """Write a new cache atomically; never replace an existing provenance artifact."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Bragg cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.", suffix=".part"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ids": list(cache.ids),
        "offsets": torch.from_numpy(cache.offsets),
        "two_theta": torch.from_numpy(cache.two_theta),
        "intensity": torch.from_numpy(cache.intensity),
        "manifest": cache.manifest,
    }
    try:
        torch.save(payload, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite Bragg cache: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_bragg_cache(
    path: str | Path,
    *,
    expected_manifest: Mapping | None = None,
    expected_ids: Iterable[str] | None = None,
) -> BraggPeakCache:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported Bragg cache schema: {path}")
    cache = BraggPeakCache(
        ids=tuple(map(str, payload["ids"])),
        offsets=torch.as_tensor(payload["offsets"]).numpy(),
        two_theta=torch.as_tensor(payload["two_theta"]).numpy(),
        intensity=torch.as_tensor(payload["intensity"]).numpy(),
        manifest=dict(payload["manifest"]),
    )
    if expected_ids is not None and cache.ids != tuple(map(str, expected_ids)):
        raise ValueError(f"Bragg cache ids do not match current dataset: {path}")
    mismatches = {
        key: (cache.manifest.get(key), value)
        for key, value in dict(expected_manifest or {}).items()
        if cache.manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Bragg cache manifest mismatch for {path}: {mismatches}")
    return cache
