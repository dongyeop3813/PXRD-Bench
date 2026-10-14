"""Read-only canonical structure sidecars shared by token datasets."""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

import numpy as np

from src.data.mp20_tokens import pack_structure_tokens


SCHEMA_VERSION = 1
_HEADER = struct.Struct("<H6f")


def canonical_cache_split(dataset_class: str, split: str) -> str:
    if (
        dataset_class == "pxrd_bench"
        and split == "val"
    ):
        return "iid_val"
    return split


def structure_cache_path(
    root: str | Path,
    tag: str,
    dataset_class: str,
    split: str,
    real_pxrd_variant: str = "all",
) -> Path:
    split = canonical_cache_split(dataset_class, split)
    parts = [dataset_class]
    if dataset_class == "real_pxrd":
        parts.append(real_pxrd_variant)
    parts.append(split)
    name = ".".join(re.sub(r"[^A-Za-z0-9_.-]", "_", part) for part in parts)
    return Path(root) / "structure_cache" / tag / f"{name}.lmdb"


def encode_structure(structure) -> bytes:
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=np.uint8)
    frac_coords = np.asarray(structure.frac_coords, dtype="<f4") % 1.0
    lattice = (*structure.lattice.lengths, *structure.lattice.angles)
    return (
        _HEADER.pack(len(atomic_numbers), *lattice)
        + atomic_numbers.tobytes()
        + frac_coords.tobytes()
    )


def decode_structure(value: bytes) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    if len(value) < _HEADER.size:
        raise ValueError("truncated canonical structure cache record")
    n, *lattice = _HEADER.unpack_from(value)
    expected = _HEADER.size + n + 12 * n
    if len(value) != expected:
        raise ValueError(
            f"invalid canonical structure cache record size: {len(value)} != {expected}"
        )
    offset = _HEADER.size
    atomic_numbers = np.frombuffer(value, dtype=np.uint8, count=n, offset=offset)
    frac_coords = np.frombuffer(
        value, dtype="<f4", count=3 * n, offset=offset + n
    ).reshape(n, 3)
    return atomic_numbers, frac_coords, tuple(lattice)


class StructureCache:
    """Lazy, fork-safe LMDB overlay keyed by the dataset's stable material ID."""

    def __init__(self, path: Path, *, dataset_class: str, split: str) -> None:
        self.path = path
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"complete structure cache not found: {path}")
        self.manifest = json.loads(manifest_path.read_text())
        expected = {
            "schema_version": SCHEMA_VERSION,
            "dataset_class": dataset_class,
            "split": split,
            "complete": True,
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"structure cache manifest mismatch at {path}: {mismatches}")
        self._env = None
        self._env_pid = None

    def _environment(self):
        pid = os.getpid()
        if self._env is None or self._env_pid != pid:
            import lmdb

            self._env = lmdb.open(
                str(self.path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1024,
            )
            self._env_pid = pid
        return self._env

    def apply(self, item: dict, *, nmax: int) -> dict:
        material_id = str(item["mp_id"])
        with self._environment().begin() as txn:
            value = txn.get(material_id.encode("utf-8"))
        if value is None:
            raise KeyError(f"structure cache {self.path} has no entry for {material_id!r}")
        atomic_numbers, frac_coords, lattice = decode_structure(value)
        if len(atomic_numbers) > nmax:
            raise ValueError(
                f"cached structure {material_id!r} has {len(atomic_numbers)} atoms > nmax={nmax}"
            )
        cached = pack_structure_tokens(
            atomic_numbers,
            frac_coords,
            lattice[:3],
            lattice[3:],
            nmax=nmax,
            num_atoms=len(atomic_numbers),
            mp_id=material_id,
        )
        out = dict(item)
        out.update({key: cached[key] for key in ("A0", "F1", "Y1", "pad_mask", "num_atoms")})
        return out


def load_structure_cache(
    *,
    root: str | Path,
    tag: str | None,
    dataset_class: str,
    split: str,
    real_pxrd_variant: str = "all",
) -> StructureCache | None:
    if not tag:
        return None
    split = canonical_cache_split(dataset_class, split)
    return StructureCache(
        structure_cache_path(root, tag, dataset_class, split, real_pxrd_variant),
        dataset_class=dataset_class,
        split=split,
    )
