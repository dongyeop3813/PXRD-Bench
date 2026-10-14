"""PyTorch Dataset/DataLoader for sharded dual-phase PXRD artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


MIXTURE_SCHEMA_VERSION = 3


class MixturePhasePXRDDataset(Dataset):
    """Read an existing frozen two-phase artifact split.

    Metadata stays in ``samples.parquet`` while dense tensors are grouped into
    small ``shards/*.pt`` files. Only the most recently accessed shard is kept
    in memory, which keeps worker memory bounded for the full dataset.
    """

    def __init__(self, root: str | Path, split: str = "test") -> None:
        self.root = Path(root)
        self.split = str(split)
        self.split_dir = self.root / self.split
        manifest_path = self.split_dir / "samples.parquet"
        metadata_path = self.split_dir / "metadata.json"
        if not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"missing mixture-phase dataset split under {self.split_dir}"
            )
        self.rows = pd.read_parquet(manifest_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.two_theta = torch.linspace(
            float(self.metadata["two_theta_min"]),
            float(self.metadata["two_theta_max"]),
            int(self.metadata["num_points"]),
            dtype=torch.float32,
        )
        self.pxrd_two_theta_grid = self.two_theta
        self.pxrd_n_bins = int(self.two_theta.numel())
        self.pxrd_two_theta_bounds = (
            float(self.metadata["two_theta_min"]),
            float(self.metadata["two_theta_max"]),
        )
        self._cached_shard_name: str | None = None
        self._cached_shard: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> dict[str, torch.Tensor]:
        if name != self._cached_shard_name:
            payload = torch.load(
                self.split_dir / "shards" / name,
                map_location="cpu",
                weights_only=False,
            )
            if payload.get("schema_version") != MIXTURE_SCHEMA_VERSION:
                raise ValueError(f"unsupported shard schema in {name}")
            self._cached_shard_name = name
            self._cached_shard = payload["tensors"]
        assert self._cached_shard is not None
        return self._cached_shard

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        shard = self._load_shard(str(row["shard"]))
        offset = int(row["offset"])
        out: dict[str, Any] = {
            key: value[offset].clone() for key, value in shard.items()
        }
        out.update(
            {
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "pairing_mode": str(row.get("pairing_mode", "thermo")),
                "chemical_system": str(row["chemical_system"]),
                "thermo_class": str(row["thermo_class"]),
                "endpoint_pair": bool(row["endpoint_pair"]),
                "phase_mp_ids": (str(row["phase_a_id"]), str(row["phase_b_id"])),
                "phase_formulas": (
                    str(row["phase_a_formula"]),
                    str(row["phase_b_formula"]),
                ),
                "bulk_formula": str(row["bulk_formula"]),
                "thermo_source": str(row["thermo_source"]),
                "constituent_ehull_policy": str(
                    row.get("constituent_ehull_policy", "unspecified")
                ),
                "fraction_b": float(row["fraction_b"]),
                "phase_x": (float(row["x_a"]), float(row["x_b"])),
                "bulk_x": float(row["x_bulk"]),
                "assemblage_energy_per_atom": float(
                    row["assemblage_energy_per_atom"]
                ),
                "hull_energy_per_atom": float(row["hull_energy_per_atom"]),
                "max_phase_e_above_hull": float(
                    row.get("max_phase_e_above_hull", float("nan"))
                ),
                "two_theta": self.two_theta.clone(),
            }
        )
        return out


def collate_mixture_phase_pxrd(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor/numeric columns and preserve human-readable metadata."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (bool, int, float)):
            out[key] = torch.as_tensor(values)
        else:
            out[key] = values
    return out


def make_mixture_phase_dataloader(
    root: str | Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MixturePhasePXRDDataset(root, split=split)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_mixture_phase_pxrd,
    )
