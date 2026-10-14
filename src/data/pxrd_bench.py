"""Read frozen PXRD-Bench shards without selecting or splitting records."""

from __future__ import annotations
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from src.data.mp20_tokens import (
    ATOM_ORDERING_CHOICES,
    decode_Y1,
    mcflow_canonical_order,
    mcflow_hierarchical_permutation,
)
from src.data.pxrd_bench_schema import (
    SPLITS,
    MAX_SITES,
    DEFAULT_MAX_SITES,
    SHARD_SCHEMA_VERSION,
    PXRD,
)

DATASET_ID = "pxrd_bench"
_PXRD_GRID = PXRD["two_theta_min"] + np.arange(PXRD["bins"]) * PXRD["bin_width"]
_PXRD_GRID_TENSOR = torch.from_numpy(_PXRD_GRID)


def _bragg_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = payload["tensors"]
    material_ids = list(map(str, payload["material_ids"]))
    return [
        {
            "mp_id": material_id,
            "A0": tensors["A0"][index],
            "F1": tensors["F1"][index],
            "Y1": tensors["Y1"][index],
            "pad_mask": tensors["pad_mask"][index],
            "num_atoms": int(tensors["num_atoms"][index]),
        }
        for index, material_id in enumerate(material_ids)
    ]


def _bragg_sidecar_path(root: Path, shard_name: str, source_sha256: str) -> Path:
    return (
        root / "bragg_peaks" / f"{Path(shard_name).stem}.bragg-v2-src{source_sha256}.pt"
    )


def _bragg_manifest(
    shard_name: str, source_sha256: str, wavelength: float
) -> dict[str, Any]:
    return {
        "source_shard": shard_name,
        "source_structures_sha256": source_sha256,
        "wavelength_angstrom": float(wavelength),
        "two_theta_range_deg": [
            float(PXRD["two_theta_min"]),
            float(PXRD["two_theta_max"]),
        ],
    }


class PXRDBench(Dataset):
    """Lazy loader for one frozen final PXRD-Bench split."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        nmax: int = DEFAULT_MAX_SITES,
        augment_translate: bool = False,
        with_pxrd: bool = True,
        with_pxrd_peaks: bool = False,
        pxrd_max_peaks: int = 128,
        pxrd_wavelength: float = 1.54184,
        pxrd_two_theta_min: float = 5.0,
        pxrd_two_theta_max: float = 90.0,
        pxrd_peak_augment: bool = False,
        pxrd_profile_augment: bool = False,
        pxrd_aug_jitter_deg: float = 0.1,
        pxrd_aug_int_scale_min: float = 0.8,
        pxrd_aug_int_scale_max: float = 1.2,
        pxrd_aug_width_scale_min: float = 0.7,
        pxrd_aug_width_scale_max: float = 1.5,
        pxrd_aug_drop_frac_max: float = 0.1,
        pxrd_aug_zero_shift_deg: float = 0.15,
        pxrd_aug_eta_min: float = 0.0,
        pxrd_aug_eta_max: float = 0.5,
        pxrd_aug_background_max: float = 0.2,
        pxrd_aug_noise_std_max: float = 0.02,
        pxrd_representation: str = "peak_list",
        pxrd_peak_source: str = "bragg",
        pxrd_peak_features: str = "position_intensity",
        pxrd_peak_position: str = "d",
        pxrd_peak_position_normalized: bool = False,
        pxrd_peak_intensity_normalization: str = "selected_topk",
        pxrd_peak_prominence_normalization: str = "none",
        pxrd_peak_width_unit: str = "two_theta",
        pxrd_peak_width_normalization: str = "none",
        pxrd_peak_selection: str = "strongest",
        pxrd_low_angle_fraction: float = 0.25,
        pxrd_profile_step: float = 0.1,
        pxrd_caglioti_u: float = 0.1,
        pxrd_caglioti_v: float = 0.01,
        pxrd_caglioti_w: float = 0.1,
        pxrd_pseudo_voigt_eta: float = 0.1,
        pxrd_dense_n_bins: int = 8500,
        pxrd_augmentation: str = "none",
        pxrd_xrdiff_gamma_shape: float | None = None,
        pxrd_xrdiff_gamma_scale: float | None = None,
        pxrd_dense_profile_augment: bool | None = None,
        atom_ordering: str = "none",
        hierarchical_orbit_permutation: bool = False,
        atom_ordering_symprec: float = 0.1,
    ) -> None:
        manifest = json.loads(
            (Path(root) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("dataset_id") != DATASET_ID:
            raise ValueError("Expected PXRD-Bench dataset_id=pxrd_bench")
        if manifest.get("final_curated_splits") is not True:
            raise ValueError("PXRD-Bench requires frozen final curated split tables")
        if manifest.get("distribution") != "materialized":
            raise ValueError(
                "Reference-only release: reconstruct licensed tensors before loading"
            )
        requested_split = str(split)
        split = "iid_val" if requested_split == "val" else requested_split
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self.nmax = int(nmax)
        if self.nmax not in {DEFAULT_MAX_SITES, MAX_SITES}:
            raise ValueError(
                f"nmax must select a built view: {DEFAULT_MAX_SITES} or {MAX_SITES}"
            )
        self.root = Path(root)
        self.split = requested_split
        self.storage_split = split
        self.augment_translate = bool(augment_translate)
        self.atom_ordering = str(atom_ordering).strip().lower()
        if self.atom_ordering not in ATOM_ORDERING_CHOICES:
            raise ValueError(
                f"atom_ordering must be one of {ATOM_ORDERING_CHOICES}, got {atom_ordering!r}."
            )
        self.hierarchical_orbit_permutation = bool(hierarchical_orbit_permutation)
        if self.hierarchical_orbit_permutation and self.atom_ordering != "mcflow":
            raise ValueError(
                "hierarchical_orbit_permutation requires atom_ordering='mcflow'."
            )
        self.atom_ordering_symprec = float(atom_ordering_symprec)
        if self.atom_ordering_symprec <= 0.0:
            raise ValueError("atom_ordering_symprec must be positive.")
        self.with_pxrd = bool(with_pxrd)
        self.with_pxrd_peaks = bool(with_pxrd_peaks)
        self.pxrd_max_peaks = int(pxrd_max_peaks)
        self.pxrd_wavelength = float(pxrd_wavelength)
        self.pxrd_transform_two_theta_min = float(pxrd_two_theta_min)
        self.pxrd_transform_two_theta_max = float(pxrd_two_theta_max)
        self.pxrd_peak_augment = bool(pxrd_peak_augment)
        if pxrd_dense_profile_augment is not None:
            pxrd_profile_augment = bool(pxrd_dense_profile_augment)
        self.pxrd_profile_augment = bool(pxrd_profile_augment)
        self.pxrd_dense_profile_augment = self.pxrd_profile_augment
        self.pxrd_aug_jitter_deg = float(pxrd_aug_jitter_deg)
        self.pxrd_aug_int_scale_min = float(pxrd_aug_int_scale_min)
        self.pxrd_aug_int_scale_max = float(pxrd_aug_int_scale_max)
        self.pxrd_aug_width_scale_min = float(pxrd_aug_width_scale_min)
        self.pxrd_aug_width_scale_max = float(pxrd_aug_width_scale_max)
        self.pxrd_aug_drop_frac_max = float(pxrd_aug_drop_frac_max)
        self.pxrd_aug_zero_shift_deg = float(pxrd_aug_zero_shift_deg)
        self.pxrd_aug_eta_min = float(pxrd_aug_eta_min)
        self.pxrd_aug_eta_max = float(pxrd_aug_eta_max)
        self.pxrd_aug_background_max = float(pxrd_aug_background_max)
        self.pxrd_aug_noise_std_max = float(pxrd_aug_noise_std_max)
        from src.pxrd.representation import (
            PUBLIC_REPRESENTATIONS,
            legacy_representation,
            normalize_peak_features,
            resolve_representation,
        )

        requested_representation = str(pxrd_representation)
        semantic_representation, semantic_source = resolve_representation(
            requested_representation, peak_source=pxrd_peak_source
        )
        self.pxrd_peak_fields = normalize_peak_features(
            pxrd_peak_features, position=pxrd_peak_position
        )
        self.pxrd_public_api = requested_representation in PUBLIC_REPRESENTATIONS
        self.pxrd_semantic_representation = semantic_representation
        self.pxrd_peak_source = semantic_source
        if (
            not float(PXRD["two_theta_min"])
            <= self.pxrd_transform_two_theta_min
            < self.pxrd_transform_two_theta_max
            <= float(PXRD["two_theta_max"])
        ):
            raise ValueError("PXRD-Bench transform range must stay within 5-90 degrees")
        if semantic_source == "dense" and (
            self.pxrd_transform_two_theta_min != float(PXRD["two_theta_min"])
            or self.pxrd_transform_two_theta_max != float(PXRD["two_theta_max"])
        ):
            raise ValueError("stored PXRD-Bench dense profiles require 5-90 degrees")
        self.pxrd_representation = legacy_representation(
            requested_representation,
            peak_source=semantic_source,
            peak_features=self.pxrd_peak_fields,
            position_normalized=bool(pxrd_peak_position_normalized),
        )
        self._peak128_bragg_augmentation = bool(
            self.with_pxrd_peaks
            and self.pxrd_representation == "peaks_128x4"
            and (self.pxrd_profile_augment or self.pxrd_peak_augment)
        )
        self.pxrd_peak_position = str(pxrd_peak_position)
        self.pxrd_peak_position_normalized = bool(pxrd_peak_position_normalized)
        self.pxrd_peak_intensity_normalization = str(pxrd_peak_intensity_normalization)
        self.pxrd_peak_prominence_normalization = str(
            pxrd_peak_prominence_normalization
        )
        self.pxrd_peak_width_unit = str(pxrd_peak_width_unit)
        self.pxrd_peak_width_normalization = str(pxrd_peak_width_normalization)
        self.pxrd_peak_selection = str(pxrd_peak_selection)
        self.pxrd_low_angle_fraction = float(pxrd_low_angle_fraction)
        self.pxrd_profile_step = float(pxrd_profile_step)
        self.pxrd_caglioti_u = float(pxrd_caglioti_u)
        self.pxrd_caglioti_v = float(pxrd_caglioti_v)
        self.pxrd_caglioti_w = float(pxrd_caglioti_w)
        self.pxrd_pseudo_voigt_eta = float(pxrd_pseudo_voigt_eta)
        self.pxrd_dense_n_bins = int(pxrd_dense_n_bins)
        self.pxrd_augmentation = str(pxrd_augmentation)
        self.pxrd_xrdiff_gamma_shape = (
            None if pxrd_xrdiff_gamma_shape is None else float(pxrd_xrdiff_gamma_shape)
        )
        self.pxrd_xrdiff_gamma_scale = (
            None if pxrd_xrdiff_gamma_scale is None else float(pxrd_xrdiff_gamma_scale)
        )
        if self.pxrd_profile_augment:
            if self.pxrd_augmentation == "none":
                self.pxrd_augmentation = "profile_randomization"
        else:
            self.pxrd_augmentation = "none"
        self.pxrd_two_theta_grid = _PXRD_GRID_TENSOR
        self.pxrd_n_bins = int(_PXRD_GRID_TENSOR.numel())
        self.pxrd_two_theta_bounds = (
            float(PXRD["two_theta_min"]),
            float(PXRD["two_theta_max"]),
        )
        if (
            self.with_pxrd_peaks
            and (not self.pxrd_public_api)
            and (
                self.pxrd_representation != "peaks_128x4" or self.pxrd_max_peaks != 512
            )
        ):
            raise ValueError(
                "PXRD-Bench conditioning currently supports peaks_128x4 with pxrd_max_peaks=512"
            )
        split_root = (
            self.root / "splits"
            if self.nmax == MAX_SITES
            else self.root / "views" / f"nmax{self.nmax}" / "splits"
        )
        path = split_root / f"{self.storage_split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing built split: {path}")
        self.rows = pd.read_parquet(path)
        self._element_counts_cache: torch.Tensor | None = None
        self._cached_shard: str | None = None
        self._cached_payload: dict[str, torch.Tensor] | None = None
        self._cached_shard_ids: tuple[str, ...] | None = None
        self._cached_shard_source_sha256: str | None = None
        self._bragg_identities: dict[str, tuple[tuple[str, ...], str]] = {}
        self._cached_bragg_shard: str | None = None
        self._cached_bragg_patterns: dict[str, tuple[np.ndarray, np.ndarray]] | None = (
            None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> dict[str, torch.Tensor]:
        if name != self._cached_shard:
            payload = torch.load(
                str(self.root / "shards" / name),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
                raise ValueError(f"unsupported PXRD-Bench shard schema: {name}")
            self._cached_shard = name
            self._cached_payload = payload["tensors"]
            self._cached_shard_ids = None
            self._cached_shard_source_sha256 = None
            if self.with_pxrd_peaks and (
                self.pxrd_peak_source == "bragg" or self._peak128_bragg_augmentation
            ):
                from src.pxrd.bragg_cache import structure_items_sha256

                identity = self._bragg_identities.get(name)
                if identity is None:
                    items = _bragg_items(payload)
                    identity = (
                        tuple((item["mp_id"] for item in items)),
                        structure_items_sha256(items),
                    )
                    self._bragg_identities[name] = identity
                self._cached_shard_ids, self._cached_shard_source_sha256 = identity
        assert self._cached_payload is not None
        return self._cached_payload

    def _load_bragg_shard(self, name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if name != self._cached_bragg_shard:
            from src.pxrd.bragg_cache import load_bragg_cache

            if self._cached_shard != name:
                self._load_shard(name)
            assert self._cached_shard_ids is not None
            assert self._cached_shard_source_sha256 is not None
            path = _bragg_sidecar_path(
                self.root, name, self._cached_shard_source_sha256
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"missing PXRD-Bench Bragg sidecar: {path}; run build_bragg_sidecars(...) before using peak_source='bragg' or canonical dense_profile"
                )
            cache = load_bragg_cache(
                path,
                expected_manifest=_bragg_manifest(
                    name, self._cached_shard_source_sha256, self.pxrd_wavelength
                ),
                expected_ids=self._cached_shard_ids,
            )
            self._cached_bragg_patterns = cache.as_dict()
            self._cached_bragg_shard = name
        assert self._cached_bragg_patterns is not None
        return self._cached_bragg_patterns

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get_row(self.rows.iloc[int(index)])

    def _get_row(self, row) -> dict[str, Any]:
        tensors = self._load_shard(str(row["shard"]))
        offset = int(row["offset"])
        out: dict[str, Any] = {
            key: value[offset].clone()
            for key, value in tensors.items()
            if self.with_pxrd or self.with_pxrd_peaks or key != "xrd"
        }
        for key in ("A0", "F1", "pad_mask"):
            out[key] = out[key][: self.nmax]
        out.update(
            {
                "mp_id": str(row["material_id"]),
                "source": str(row["source"]),
                "source_id": str(row["source_id"]),
                "split": self.split,
                "composition_key": str(row["composition_key"]),
                "prototype_id": str(row["prototype_id"]),
                "structure_entity_id": str(row["structure_entity_id"]),
                "leakage_group_id": str(row["leakage_group_id"]),
                "energy_above_hull": float(row["energy_above_hull"]),
            }
        )
        structure_cache = getattr(self, "structure_cache", None)
        if structure_cache is not None:
            out = structure_cache.apply(out, nmax=self.nmax)
        if self.atom_ordering == "mcflow":
            real = ~out["pad_mask"].bool()
            lengths, angles = decode_Y1(out["Y1"])
            order, metadata = mcflow_canonical_order(
                out["A0"][real].numpy(),
                out["F1"][real].numpy(),
                lengths,
                angles,
                symprec=self.atom_ordering_symprec,
            )
            order = torch.as_tensor(
                [*order, *range(int(out["num_atoms"]), self.nmax)], dtype=torch.long
            )
            for key in ("A0", "F1", "pad_mask"):
                out[key] = out[key].index_select(0, order)
            out["_mcflow_orbits"] = metadata["orbits"]
            out["_mcflow_inter_orbit_groups"] = metadata["inter_orbit_groups"]
            orbit_ids = torch.full((self.nmax,), -1, dtype=torch.long)
            for orbit_id, orbit in enumerate(metadata["orbits"]):
                orbit_ids[list(orbit)] = orbit_id
            out["mcflow_orbit_ids"] = orbit_ids
            if self.hierarchical_orbit_permutation:
                permutation = mcflow_hierarchical_permutation(out)
                for key in ("A0", "F1", "pad_mask", "mcflow_orbit_ids"):
                    out[key] = out[key].index_select(0, permutation)
        if self.augment_translate:
            real = ~out["pad_mask"]
            out["F1"][real] = (out["F1"][real] + torch.rand(3)) % 1
        if self.with_pxrd_peaks:
            bragg_pattern = None
            if self.pxrd_public_api:
                self._attach_public_pxrd(
                    out, str(row["shard"]), bragg_pattern=bragg_pattern
                )
            else:
                if bragg_pattern is None and self._peak128_bragg_augmentation:
                    bragg_pattern = self._load_bragg_shard(str(row["shard"]))[
                        out["mp_id"]
                    ]
                self._attach_peak128(out, bragg_pattern=bragg_pattern)
            if not self.with_pxrd:
                out.pop("xrd", None)
        return out

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Load deterministic batches once per index and shard."""
        if (
            self.augment_translate
            or 0.0 > 0.0
            or self.pxrd_profile_augment
            or self.pxrd_peak_augment
        ):
            return [
                self._get_row(row) for row in self.rows.iloc[indices].to_dict("records")
            ]
        unique = list(dict.fromkeys(map(int, indices)))
        unique.sort(key=lambda index: str(self.rows.iloc[index]["shard"]))
        loaded = {index: self[index] for index in unique}
        return [loaded[int(index)] for index in indices]

    def _attach_public_pxrd(
        self,
        out: dict[str, Any],
        shard: str,
        *,
        bragg_pattern: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        from src.pxrd.representation import (
            PXRDTransformConfig,
            attach_pxrd_result,
            transform_bragg_pattern,
            transform_dense_profile,
        )

        needs_bragg = self.pxrd_peak_source == "bragg"
        dense_bins = self.pxrd_dense_n_bins if needs_bragg else int(out["xrd"].numel())
        config = PXRDTransformConfig(
            representation=self.pxrd_semantic_representation,
            peak_source=self.pxrd_peak_source,
            peak_features=self.pxrd_peak_fields,
            max_peaks=self.pxrd_max_peaks,
            wavelength=self.pxrd_wavelength,
            two_theta_min=self.pxrd_transform_two_theta_min,
            two_theta_max=self.pxrd_transform_two_theta_max,
            dense_n_bins=dense_bins,
            selection=self.pxrd_peak_selection,
            low_angle_fraction=self.pxrd_low_angle_fraction,
            caglioti_u=self.pxrd_caglioti_u,
            caglioti_v=self.pxrd_caglioti_v,
            caglioti_w=self.pxrd_caglioti_w,
            pseudo_voigt_eta=self.pxrd_pseudo_voigt_eta,
            position_normalized=self.pxrd_peak_position_normalized,
            intensity_normalization=self.pxrd_peak_intensity_normalization,
            prominence_normalization=self.pxrd_peak_prominence_normalization,
            width_unit=self.pxrd_peak_width_unit,
            width_normalization=self.pxrd_peak_width_normalization,
            augment_sticks=self.pxrd_peak_augment,
            augmentation=self.pxrd_augmentation,
            jitter_deg=self.pxrd_aug_jitter_deg,
            intensity_scale_min=self.pxrd_aug_int_scale_min,
            intensity_scale_max=self.pxrd_aug_int_scale_max,
            width_scale_min=self.pxrd_aug_width_scale_min,
            width_scale_max=self.pxrd_aug_width_scale_max,
            zero_shift_deg=self.pxrd_aug_zero_shift_deg,
            eta_min=self.pxrd_aug_eta_min,
            eta_max=self.pxrd_aug_eta_max,
            background_max=self.pxrd_aug_background_max,
            noise_std_max=self.pxrd_aug_noise_std_max,
            drop_fraction_max=self.pxrd_aug_drop_frac_max,
            xrdiff_gamma_shape=self.pxrd_xrdiff_gamma_shape,
            xrdiff_gamma_scale=self.pxrd_xrdiff_gamma_scale,
        )
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        if needs_bragg:
            two_theta, intensity = (
                bragg_pattern
                if bragg_pattern is not None
                else self._load_bragg_shard(shard)[out["mp_id"]]
            )
            result = transform_bragg_pattern(two_theta, intensity, config, rng=rng)
        else:
            result = transform_dense_profile(
                np.asarray(out["xrd"], dtype=np.float32), config, rng=rng
            )
        attach_pxrd_result(
            out, result, config, internal_representation=self.pxrd_representation
        )

    def _attach_peak128(
        self,
        out: dict[str, Any],
        *,
        bragg_pattern: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Build the shared Profile-Strong Peak128 features from stored PXRD."""
        from scipy.signal import find_peaks
        from src.pxrd.peaks import (
            augmented_peak128_feature_vector,
            xrdiff_feature_vector_from_profile,
        )

        profile = np.asarray(out["xrd"], dtype=np.float32)
        two_theta, intensity = bragg_pattern or (None, None)
        if two_theta is None:
            indices, _ = find_peaks(profile, prominence=0.0)
            two_theta = 5.0 + 0.1 * indices.astype(np.float32)
            intensity = profile[indices]
        if self.pxrd_profile_augment or self.pxrd_peak_augment:
            features = augmented_peak128_feature_vector(
                two_theta,
                intensity,
                jitter_deg=self.pxrd_aug_jitter_deg,
                int_scale_min=self.pxrd_aug_int_scale_min,
                int_scale_max=self.pxrd_aug_int_scale_max,
                width_scale_min=self.pxrd_aug_width_scale_min,
                width_scale_max=self.pxrd_aug_width_scale_max,
                drop_frac_max=self.pxrd_aug_drop_frac_max,
                zero_shift_deg=self.pxrd_aug_zero_shift_deg,
                eta_min=self.pxrd_aug_eta_min,
                eta_max=self.pxrd_aug_eta_max,
                background_max=self.pxrd_aug_background_max,
                noise_std_max=self.pxrd_aug_noise_std_max,
            )
        else:
            features = xrdiff_feature_vector_from_profile(profile, "peaks_128x4")
        out["pxrd_d"] = torch.from_numpy(features)
        out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
        out["pxrd_peak_pad_mask"] = torch.zeros_like(out["pxrd_d"], dtype=torch.bool)

    def _composition_counts(self) -> torch.Tensor:
        if self._element_counts_cache is not None:
            return self._element_counts_cache
        counts = torch.zeros(94, dtype=torch.float64)
        for value in self.rows["composition_key"].astype(str):
            for token in value.split("|"):
                atomic_number, amount = token.split(":")
                counts[int(atomic_number) - 1] += int(amount)
        self._element_counts_cache = counts
        return counts

    def training_batch_sampler(
        self,
        batch_size: int,
        seed: int,
        *,
        rank: int = 0,
        world_size: int = 1,
    ):
        return _TrainingShardBatchSampler(
            self.rows,
            batch_size,
            seed,
            rank=rank,
            world_size=world_size,
        )

    def element_distribution(self) -> torch.Tensor | None:
        counts = self._composition_counts()
        total = float(counts.sum())
        return counts / total if total > 0 else None

    def allowed_elements(self) -> torch.Tensor | None:
        mask = self._composition_counts() > 0
        return mask if bool(mask.any()) else None


def collate_pxrd_bench(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (bool, int, float)):
            out[key] = torch.as_tensor(values)
        else:
            out[key] = values
    return out


class _ShardBatchSampler(Sampler[list[int]]):
    def __init__(self, rows: pd.DataFrame, batch_size: int, shuffle: bool) -> None:
        self.groups = [
            np.asarray(indices, dtype=np.int64)
            for indices in rows.groupby("shard", sort=False).indices.values()
        ]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self) -> Iterator[list[int]]:
        group_order = (
            torch.randperm(len(self.groups)).tolist()
            if self.shuffle
            else range(len(self.groups))
        )
        for group_index in group_order:
            indices = self.groups[group_index]
            if self.shuffle:
                indices = indices[torch.randperm(len(indices)).numpy()]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum((math.ceil(len(group) / self.batch_size) for group in self.groups))


class _TrainingShardBatchSampler(Sampler[list[int]]):
    """Shuffle by shard while keeping every training batch full-sized."""

    def __init__(
        self,
        rows: pd.DataFrame,
        batch_size: int,
        seed: int,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.groups = [
            np.asarray(indices, dtype=np.int64)
            for indices in rows.groupby("shard", sort=False).indices.values()
        ]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                "rank must satisfy 0 <= rank < world_size with world_size > 0"
            )
        self.epoch = 0
        self.total = sum(map(len, self.groups))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pending = np.empty(0, dtype=np.int64)
        global_batch_index = 0
        usable_global_batches = len(self) * self.world_size
        for group_index in rng.permutation(len(self.groups)):
            group = self.groups[int(group_index)]
            values = rng.permutation(group)
            if pending.size:
                values = np.concatenate((pending, values))
            stop = values.size - values.size % self.batch_size
            for start in range(0, stop, self.batch_size):
                if global_batch_index >= usable_global_batches:
                    break
                if global_batch_index % self.world_size == self.rank:
                    yield values[start : start + self.batch_size].tolist()
                global_batch_index += 1
            pending = values[stop:]
        self.epoch += 1

    def __len__(self) -> int:
        return self.total // self.batch_size // self.world_size


def make_pxrd_bench_dataloader(
    root: str | Path,
    *,
    split: str = "train",
    batch_size: int = 512,
    shuffle: bool | None = None,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader:
    dataset = PXRDBench(root, split, **kwargs)
    sampler = _ShardBatchSampler(
        dataset.rows,
        int(batch_size),
        split == "train" if shuffle is None else bool(shuffle),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=collate_pxrd_bench,
    )
