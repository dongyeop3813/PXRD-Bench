"""Frozen shared-element, oracle-profile inputs for two-phase fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.mp20_tokens import MP20Tokens, VZ


class SharedPhaseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        reference_root: str | Path,
        nmax: int = 20,
        augment_translate: bool = False,
        atom_ordering: str = "none",
        hierarchical_orbit_permutation: bool = False,
        atom_ordering_symprec: float = 0.1,
        pxrd_representation: str = "peaks_128x4",
    ) -> None:
        if nmax != 20:
            raise ValueError("the two-phase phase cache requires nmax=20")
        path = Path(root) / f"{split}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "oracle_phase_shared_cache_v1":
            raise ValueError(f"expected oracle_phase_shared_cache_v1: {path}")
        self.rows = list(payload["rows"])
        self.tensors = payload["tensors"]
        mask = self.tensors.get("elem_set_mask")
        counts = self.tensors.get("oracle_num_atoms")
        if (
            not self.rows
            or mask is None
            or mask.shape != (len(self.rows), VZ)
            or mask.dtype != torch.bool
            or not mask.any(-1).all()
            or counts is None
            or counts.shape != (len(self.rows),)
            or not ((counts >= 1) & (counts <= nmax)).all()
            or not torch.equal(counts, counts.long())
        ):
            raise ValueError("invalid oracle phase shared-element conditions")
        if pxrd_representation != "peaks_128x4":
            raise ValueError("oracle phase cache requires peaks_128x4")
        features = self.tensors.get("pxrd_peak128x4")
        if (
            features is None
            or features.shape != (len(self.rows), 512)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("invalid oracle phase PXRD features")
        parent_masks = {}
        for index, row in enumerate(self.rows):
            parent = row["parent_sample_id"]
            previous = parent_masks.setdefault(parent, index)
            if not torch.equal(mask[previous], mask[index]):
                raise ValueError("phase rows must share the mixture element superset")
        self.pxrd_representation = pxrd_representation
        if any(len(value) != len(self.rows) for value in self.tensors.values()):
            raise ValueError(f"shared phase cache length mismatch: {path}")
        self.reference = MP20Tokens(
            root=str(reference_root),
            split=split,
            nmax=nmax,
            augment_translate=augment_translate,
            atom_ordering=atom_ordering,
            hierarchical_orbit_permutation=hierarchical_orbit_permutation,
            atom_ordering_symprec=atom_ordering_symprec,
        )
        self._reference_index = {
            str(item["mp_id"]): index for index, item in enumerate(self.reference.items)
        }
        missing = {row["phase_id"] for row in self.rows} - set(self._reference_index)
        if missing:
            raise ValueError(
                f"phase targets missing from MP20 {split}: {sorted(missing)[:5]}"
            )
        self.dataset_provenance = payload.get("provenance")

    def __len__(self) -> int:
        return len(self.rows)

    def element_distribution(self) -> torch.Tensor | None:
        return self.reference.element_distribution()

    def allowed_elements(self) -> torch.Tensor | None:
        return self.reference.allowed_elements()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        target = dict(self.reference[self._reference_index[row["phase_id"]]])
        pxrd_d = self.tensors["pxrd_peak128x4"][index].clone()
        pxrd_i = torch.zeros_like(pxrd_d)
        peak_pad = torch.zeros_like(pxrd_d, dtype=torch.bool)
        target.update(
            {
                "sample_id": row["sample_id"],
                "parent_sample_id": row["parent_sample_id"],
                "phase_index": int(row["phase_index"]),
                "phase_role": row["phase_role"],
                "thermo_class": row["thermo_class"],
                "minor_fraction": float(row["minor_fraction"]),
                "pxrd_d": pxrd_d,
                "pxrd_i": pxrd_i,
                "pxrd_peak_pad_mask": peak_pad,
            }
        )
        if "elem_set_mask" in self.tensors:
            mask = self.tensors["elem_set_mask"][index].clone().bool()
            atoms = target["A0"][~target["pad_mask"].bool()].long()
            if not mask[atoms - 1].all():
                raise ValueError("shared set does not contain the target elements")
            target["elem_set_mask"] = mask
        if "oracle_num_atoms" in self.tensors:
            if int((~target["pad_mask"].bool()).sum()) != int(
                self.tensors["oracle_num_atoms"][index]
            ):
                raise ValueError("oracle atom count disagrees with reference target")
        return target
