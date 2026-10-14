"""Token loader for the frozen author-provided RRUFF-148 release."""

from __future__ import annotations
import csv
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial import Polynomial
from pymatgen.core import Structure
from scipy.signal import find_peaks
from torch.utils.data import Dataset
from src.data.mp20_tokens import VZ, pack_structure_tokens, translate_frac_coords
from src.pxrd.peaks import (
    augmented_peak128_feature_vector,
    augment_peak128_features,
    augment_peaks,
    peak_matrix_feature_vector_from_profile,
    peaks_to_di_tensors,
    peaks_to_pseudo_voigt_tensors,
    xrdiff_feature_vector,
    xrdiff_feature_vector_from_profile,
)
from src.utils.seeding import stable_seed

_REPRESENTATIONS = {
    "legacy_di",
    "full_850",
    "peaks_20x4",
    "peaks_128x4",
    "peaks_20x4_norm_prom",
    "peaks_128x4_norm_prom",
    "peak_matrix",
    "broadened_basic",
    "broadened_shape",
}


def _manifest(root: Path, expected_kind: str) -> tuple[dict, str]:
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != expected_kind:
        raise ValueError(f"{path} must declare kind={expected_kind!r}")
    return (value, hashlib.sha256(path.read_bytes()).hexdigest())


def _normalized_profile(value, bins: int = 850) -> np.ndarray:
    profile = np.asarray(value, dtype=np.float32).reshape(-1)
    if profile.shape != (bins,):
        raise ValueError(f"expected a {bins}-bin profile, got {profile.shape}")
    maximum = float(profile.max(initial=0.0))
    if maximum <= 0.0 or not np.isfinite(profile).all():
        raise ValueError("profile must be finite with a positive maximum")
    return profile / maximum


def _realpxrd_solver_official_peaks(
    profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the public RealPXRD-Solver experimental-profile preprocessing."""
    if profile.shape != (8500,):
        raise ValueError("RealPXRD-Solver preprocessing requires an 8500-bin profile")
    two_theta = 5.0 + 0.01 * np.arange(8500)
    smoothed = np.convolve(profile, np.ones(3) / 3, mode="same")
    corrected = np.maximum(
        0, smoothed - Polynomial.fit(two_theta, smoothed, 6)(two_theta)
    )
    if corrected.max() <= 0:
        raise ValueError("RealPXRD-Solver background subtraction found no signal")
    corrected = corrected / corrected.max() * 100
    indices, _ = find_peaks(
        corrected,
        height=5,
        width=3,
        prominence=(corrected.max() - corrected.min()) * 0.05,
    )
    if not len(indices):
        raise ValueError("RealPXRD-Solver peak search found no peaks")
    intensity = corrected[indices]
    return (
        two_theta[indices].astype(np.float32),
        (intensity / intensity.max() * 100).astype(np.float32),
    )


class _RRUFFProfileTokens(Dataset):
    def _configure_profile(
        self,
        *,
        with_pxrd: bool,
        with_pxrd_peaks: bool,
        pxrd_representation: str = "peak_list",
        pxrd_peak_source: str = "dense",
        pxrd_max_peaks: int = 128,
        pxrd_peak_augment: bool = False,
        pxrd_profile_augment: bool = False,
        pxrd_wavelength: float = 1.54184,
        pxrd_two_theta_min: float = 5.0,
        pxrd_two_theta_max: float = 90.0,
        pxrd_peak_features: str = "position_intensity",
        pxrd_peak_position: str = "d",
        pxrd_peak_position_normalized: bool = False,
        pxrd_peak_intensity_normalization: str = "selected_topk",
        pxrd_peak_min_relative_intensity: float = 0.0,
        pxrd_peak_min_prominence: float = 0.0,
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
        pxrd_dense_n_bins: int = 8500,
        pxrd_augmentation: str = "none",
        pxrd_xrdiff_gamma_shape: float | None = None,
        pxrd_xrdiff_gamma_scale: float | None = None,
        pxrd_dense_profile_augment: bool | None = None,
        dense_profile_source: str | None = None,
        real_pxrd_peak_preprocessing: str = "official",
    ) -> None:
        from src.pxrd.representation import (
            PUBLIC_REPRESENTATIONS,
            legacy_representation,
            normalize_peak_features,
            resolve_representation,
        )

        requested = str(pxrd_representation)
        if requested not in _REPRESENTATIONS | PUBLIC_REPRESENTATIONS:
            raise ValueError(
                f"unsupported XRDiff representation: {pxrd_representation}"
            )
        if pxrd_dense_profile_augment is not None:
            pxrd_profile_augment = bool(pxrd_dense_profile_augment)
        if requested == "dense_profile" and dense_profile_source is not None:
            pxrd_peak_source = dense_profile_source
        semantic, source = resolve_representation(
            requested, peak_source=pxrd_peak_source
        )
        stored_contract = getattr(self, "_stored_dense_profile_contract", None)
        if requested in PUBLIC_REPRESENTATIONS and source == "dense":
            if stored_contract is None:
                if semantic == "dense_profile" and pxrd_dense_n_bins != 8500:
                    raise ValueError("stored author dense profiles require 8500 bins")
                if (
                    float(pxrd_two_theta_min) != 5.0
                    or float(pxrd_two_theta_max) != 90.0
                ):
                    raise ValueError(
                        "stored author dense profiles require the 5-90 degree range"
                    )
            else:
                stored_min, stored_max, stored_bins = stored_contract
                step = (stored_max - stored_min) / stored_bins
                start = round((float(pxrd_two_theta_min) - stored_min) / step)
                stop = round((float(pxrd_two_theta_max) - stored_min) / step)
                requested_bins = int(pxrd_dense_n_bins)
                stored_window_bins = stop - start
                if (
                    start < 0
                    or stop > stored_bins
                    or stop <= start
                    or (not np.isclose(stored_min + start * step, pxrd_two_theta_min))
                    or (not np.isclose(stored_min + stop * step, pxrd_two_theta_max))
                    or (requested_bins <= 0)
                    or stored_window_bins % requested_bins
                ):
                    raise ValueError(
                        "stored author dense profile window must align with its grid and pxrd_dense_n_bins"
                    )
                self._stored_dense_profile_slice = slice(
                    start, stop, stored_window_bins // requested_bins
                )
        peak_fields = normalize_peak_features(
            pxrd_peak_features, position=pxrd_peak_position
        )
        internal = legacy_representation(
            requested,
            peak_source=source,
            peak_features=peak_fields,
            position_normalized=bool(pxrd_peak_position_normalized),
        )
        theoretical_legacy_profile_view = (
            getattr(self, "theoretical_profile", False) and requested == "peaks_128x4"
        )
        if (
            pxrd_profile_augment
            and requested not in PUBLIC_REPRESENTATIONS
            and (not theoretical_legacy_profile_view)
        ):
            raise ValueError(
                "legacy author representations do not support dense-profile augmentation"
            )
        if (
            pxrd_profile_augment
            and source == "dense"
            and (str(pxrd_augmentation) == "xrdiff_style")
        ):
            raise ValueError("xrdiff_style requires Bragg peaks before rendering")
        if pxrd_peak_augment and internal not in {
            "legacy_di",
            "broadened_basic",
            "broadened_shape",
            "peaks_128x4",
            "peak_matrix",
            "dense_profile",
        }:
            raise ValueError(
                "author-profile augmentation supports legacy_di, broadened, or peaks_128x4 representations"
            )
        self.with_pxrd = bool(with_pxrd)
        self.with_pxrd_peaks = bool(with_pxrd_peaks)
        self.pxrd_public_api = requested in PUBLIC_REPRESENTATIONS
        self.pxrd_semantic_representation = semantic
        self.pxrd_peak_source = source
        self.pxrd_peak_fields = peak_fields
        self.pxrd_representation = internal
        self.pxrd_max_peaks = int(pxrd_max_peaks)
        self.pxrd_peak_augment = bool(pxrd_peak_augment)
        self.pxrd_wavelength = float(pxrd_wavelength)
        self.pxrd_two_theta_min = float(pxrd_two_theta_min)
        self.pxrd_two_theta_max = float(pxrd_two_theta_max)
        self.pxrd_peak_features = str(pxrd_peak_features)
        self.pxrd_peak_position = str(pxrd_peak_position)
        self.pxrd_peak_position_normalized = bool(pxrd_peak_position_normalized)
        self.pxrd_peak_intensity_normalization = str(pxrd_peak_intensity_normalization)
        self.pxrd_peak_min_prominence = float(pxrd_peak_min_prominence)
        if not 0.0 <= self.pxrd_peak_min_prominence <= 1.0:
            raise ValueError("pxrd_peak_min_prominence must be in [0, 1]")
        if self.pxrd_peak_min_prominence and self.pxrd_representation != "peaks_128x4":
            raise ValueError("pxrd_peak_min_prominence requires peaks_128x4")
        self.pxrd_peak_min_relative_intensity = float(pxrd_peak_min_relative_intensity)
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
        self.pxrd_dense_n_bins = int(pxrd_dense_n_bins)
        self.pxrd_augmentation = str(pxrd_augmentation)
        self.pxrd_xrdiff_gamma_shape = (
            None if pxrd_xrdiff_gamma_shape is None else float(pxrd_xrdiff_gamma_shape)
        )
        self.pxrd_xrdiff_gamma_scale = (
            None if pxrd_xrdiff_gamma_scale is None else float(pxrd_xrdiff_gamma_scale)
        )
        if pxrd_profile_augment:
            if self.pxrd_augmentation == "none":
                self.pxrd_augmentation = "profile_randomization"
        else:
            self.pxrd_augmentation = "none"
        self.pxrd_dense_profile_augment = bool(pxrd_profile_augment)
        self.real_pxrd_peak_preprocessing = str(real_pxrd_peak_preprocessing)

    def _attach_bragg_cache(self) -> None:
        """Attach a non-overwriting flat Bragg cache for the theoretical view."""
        from src.pxrd.bragg_cache import load_bragg_cache, structure_items_sha256

        source_sha256 = structure_items_sha256(self.items)
        cache_path = (
            self.root
            / "pxrd_peaks"
            / f"author-bragg-v2_{self.split}_wl{self.pxrd_wavelength:.5f}_tt{self.pxrd_two_theta_min:g}-{self.pxrd_two_theta_max:g}_src{source_sha256[:12]}.pt"
        )
        expected_manifest = {
            "dataset_root": str(self.root.resolve()),
            "split": self.split,
            "manifest_sha256": self.dataset_provenance["manifest_sha256"],
            "source_structures_sha256": source_sha256,
            "wavelength_angstrom": self.pxrd_wavelength,
            "two_theta_range_deg": [self.pxrd_two_theta_min, self.pxrd_two_theta_max],
        }
        expected_ids = [item["mp_id"] for item in self.items]
        cache = load_bragg_cache(
            cache_path, expected_manifest=expected_manifest, expected_ids=expected_ids
        )
        peaks = cache.as_dict()
        for item in self.items:
            two_theta, intensity = peaks[item["mp_id"]]
            item["_pxrd_x"] = two_theta
            item["_pxrd_y"] = intensity

    def _element_counts(self) -> torch.Tensor:
        counts = torch.zeros(VZ, dtype=torch.float64)
        for item in self.items:
            values = item["A0"][~item["pad_mask"]]
            counts += torch.bincount(values, minlength=VZ + 1)[1:].double()
        return counts

    def element_distribution(self) -> torch.Tensor | None:
        counts = self._element_counts()
        return counts / counts.sum() if counts.sum() else None

    def allowed_elements(self) -> torch.Tensor | None:
        mask = self._element_counts() > 0
        return mask if bool(mask.any()) else None

    def _attach_profile(
        self,
        out: dict,
        profile: np.ndarray,
        source: dict,
        *,
        bragg_pattern: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if self.with_pxrd:
            out["xrd"] = torch.from_numpy(profile.copy())
        if not self.with_pxrd_peaks:
            return
        peak_count = (
            self.pxrd_max_peaks // 4
            if self.pxrd_representation == "peaks_128x4"
            else 128
        )
        if bragg_pattern is not None and self.pxrd_representation in {
            "full_850",
            "peaks_20x4",
            "peaks_128x4",
            "peaks_20x4_norm_prom",
            "peaks_128x4_norm_prom",
        }:
            features = (
                augmented_peak128_feature_vector(
                    bragg_pattern[0],
                    bragg_pattern[1],
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
                    min_prominence=self.pxrd_peak_min_prominence,
                    peak_count=peak_count,
                    rng=rng,
                )
                if self.pxrd_dense_profile_augment
                else xrdiff_feature_vector(
                    bragg_pattern[0],
                    bragg_pattern[1],
                    self.pxrd_representation,
                    min_prominence=self.pxrd_peak_min_prominence,
                    peak_count=peak_count,
                )
            )
            out["pxrd_d"] = torch.from_numpy(features)
            out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
            out["pxrd_peak_pad_mask"] = torch.zeros_like(
                out["pxrd_d"], dtype=torch.bool
            )
            return
        if self.pxrd_public_api or bragg_pattern is not None:
            from src.pxrd.representation import (
                PXRDTransformConfig,
                attach_pxrd_result,
                transform_bragg_pattern,
                transform_dense_profile,
            )

            dense_bins = (
                len(profile)
                if self.pxrd_peak_source == "dense"
                else self.pxrd_dense_n_bins
            )
            config = PXRDTransformConfig(
                representation=self.pxrd_semantic_representation,
                peak_source=self.pxrd_peak_source,
                peak_features=self.pxrd_peak_fields,
                max_peaks=self.pxrd_max_peaks,
                wavelength=self.pxrd_wavelength,
                two_theta_min=self.pxrd_two_theta_min,
                two_theta_max=self.pxrd_two_theta_max,
                dense_n_bins=dense_bins,
                selection=self.pxrd_peak_selection,
                low_angle_fraction=self.pxrd_low_angle_fraction,
                caglioti_u=self.pxrd_caglioti_u,
                caglioti_v=self.pxrd_caglioti_v,
                caglioti_w=self.pxrd_caglioti_w,
                pseudo_voigt_eta=self.pxrd_pseudo_voigt_eta,
                position_normalized=self.pxrd_peak_position_normalized,
                intensity_normalization=self.pxrd_peak_intensity_normalization,
                min_relative_intensity=self.pxrd_peak_min_relative_intensity,
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
            if bragg_pattern is not None:
                result = transform_bragg_pattern(
                    bragg_pattern[0], bragg_pattern[1], config, rng=rng
                )
            elif self.pxrd_peak_source == "bragg":
                if "_pxrd_x" not in source:
                    raise RuntimeError("Bragg-derived author PXRD is not cached")
                result = transform_bragg_pattern(
                    source["_pxrd_x"], source["_pxrd_y"], config, rng=rng
                )
            else:
                result = transform_dense_profile(profile, config, rng=rng)
            attach_pxrd_result(
                out, result, config, internal_representation=self.pxrd_representation
            )
            return
        representation = self.pxrd_representation
        if representation == "peak_matrix":
            features = peak_matrix_feature_vector_from_profile(
                profile,
                max_peaks=self.pxrd_max_peaks,
                features=self.pxrd_peak_features,
                position=self.pxrd_peak_position,
                normalize_position=self.pxrd_peak_position_normalized,
                width_normalization=self.pxrd_peak_width_normalization,
                selection=self.pxrd_peak_selection,
                low_angle_fraction=self.pxrd_low_angle_fraction,
                wavelength=self.pxrd_wavelength,
                two_theta_min=5.0,
                two_theta_max=90.0,
            )
        elif representation in {"broadened_basic", "broadened_shape", "legacy_di"}:
            if self.real_pxrd_peak_preprocessing == "realpxrd_solver_official":
                two_theta, intensity = _realpxrd_solver_official_peaks(profile)
            else:
                indices, _ = find_peaks(profile, prominence=0.0)
                two_theta = 5.0 + 0.1 * indices
                intensity = profile[indices]
            if self.pxrd_peak_augment:
                two_theta, intensity = augment_peaks(
                    two_theta,
                    intensity,
                    jitter_deg=self.pxrd_aug_jitter_deg,
                    int_scale_min=self.pxrd_aug_int_scale_min,
                    int_scale_max=self.pxrd_aug_int_scale_max,
                    drop_frac_max=self.pxrd_aug_drop_frac_max,
                )
            if representation == "legacy_di":
                d, intensity, pad = peaks_to_di_tensors(
                    two_theta,
                    intensity,
                    max_peaks=self.pxrd_max_peaks,
                    wavelength=self.pxrd_wavelength,
                    two_theta_min=5.0,
                    two_theta_max=90.0,
                    selection=self.pxrd_peak_selection,
                    low_angle_fraction=self.pxrd_low_angle_fraction,
                )
                out.update(
                    pxrd_d=torch.from_numpy(d),
                    pxrd_i=torch.from_numpy(intensity),
                    pxrd_peak_pad_mask=torch.from_numpy(pad),
                )
                return
            position, intensity, prominence, width, d, pad = (
                peaks_to_pseudo_voigt_tensors(
                    two_theta,
                    intensity,
                    max_peaks=self.pxrd_max_peaks,
                    wavelength=self.pxrd_wavelength,
                    two_theta_min=5.0,
                    two_theta_max=90.0,
                    step=self.pxrd_profile_step,
                    caglioti_u=self.pxrd_caglioti_u,
                    caglioti_v=self.pxrd_caglioti_v,
                    caglioti_w=self.pxrd_caglioti_w,
                    eta=self.pxrd_pseudo_voigt_eta,
                )
            )
            out.update(
                pxrd_position=torch.from_numpy(position),
                pxrd_intensity=torch.from_numpy(intensity),
                pxrd_prominence=torch.from_numpy(prominence),
                pxrd_width=torch.from_numpy(width),
                pxrd_d=torch.from_numpy(d),
                pxrd_i=torch.from_numpy(intensity),
                pxrd_peak_pad_mask=torch.from_numpy(pad),
            )
            return
        else:
            features = xrdiff_feature_vector_from_profile(
                profile,
                representation,
                min_prominence=self.pxrd_peak_min_prominence,
                peak_count=peak_count,
            )
            if self.pxrd_peak_augment:
                features = augment_peak128_features(
                    features,
                    jitter_deg=self.pxrd_aug_jitter_deg,
                    int_scale_min=self.pxrd_aug_int_scale_min,
                    int_scale_max=self.pxrd_aug_int_scale_max,
                    width_scale_min=self.pxrd_aug_width_scale_min,
                    width_scale_max=self.pxrd_aug_width_scale_max,
                    drop_frac_max=self.pxrd_aug_drop_frac_max,
                )
        out["pxrd_d"] = torch.from_numpy(np.asarray(features, dtype=np.float32))
        out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
        out["pxrd_peak_pad_mask"] = torch.zeros_like(out["pxrd_d"], dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        source = self.items[index]
        structure_cache = getattr(self, "structure_cache", None)
        if structure_cache is not None:
            source = structure_cache.apply(source, nmax=self.nmax)
        out = {key: value for key, value in source.items() if not key.startswith("_")}
        if self.augment_translate:
            out["F1"] = torch.from_numpy(
                translate_frac_coords(out["F1"], out["pad_mask"])
            )
        bragg_pattern = (
            (source["_pxrd_x"], source["_pxrd_y"])
            if getattr(self, "theoretical_profile", False)
            else None
        )
        profile_rng = (
            np.random.default_rng(
                stable_seed(
                    "rruff-simulated-profile",
                    int(self.profile_augmentation_seed),
                    index,
                )
            )
            if getattr(self, "profile_augmentation_seed", None) is not None
            else None
        )
        self._attach_profile(
            out,
            source["_profile"],
            source,
            bragg_pattern=bragg_pattern,
            rng=profile_rng,
        )
        if "_diagnostic_pxrd_d" in source:
            out["pxrd_d"] = source["_diagnostic_pxrd_d"].clone()
        return out


class RRUFF148Tokens(_RRUFFProfileTokens):
    """The author-provided 148-target RRUFF set."""

    def __init__(
        self,
        root: str,
        split: str = "test",
        nmax: int = 20,
        augment_translate: bool = False,
        with_pxrd: bool = False,
        with_pxrd_peaks: bool = False,
        prop_list: list[str] | None = None,
        **profile_kwargs,
    ):
        del prop_list
        self.theoretical_profile = bool(
            profile_kwargs.pop("theoretical_profile", False)
        )
        self.profile_augmentation_seed = profile_kwargs.pop(
            "profile_augmentation_seed", None
        )
        self._stored_dense_profile_contract = (5.0, 90.0, 8500)
        if split not in {"test", "all"}:
            raise ValueError("author RRUFF has one evaluation-only split: test")
        self.root = Path(root)
        self.split = split
        self.nmax = int(nmax)
        self.augment_translate = bool(augment_translate)
        manifest, manifest_sha256 = _manifest(self.root, "xrdiff_author_rruff_148")
        self.dataset_provenance = {
            "label": "XRDiff author-provided RRUFF-148",
            "exact_author_dataset": True,
            "release": manifest["release"],
            "manifest_sha256": manifest_sha256,
            "profile_adapter": "take every 10th bin from 8500 to 850, then normalize",
            "exact_author_benchmark_adapter": False,
        }
        self._configure_profile(
            with_pxrd=with_pxrd,
            with_pxrd_peaks=with_pxrd_peaks,
            dense_profile_source="dense",
            **{"pxrd_peak_source": "dense"} | profile_kwargs,
        )
        if self.pxrd_public_api and self.pxrd_peak_source == "dense":
            profile_slice = self._stored_dense_profile_slice
            profile_bins = len(
                range(profile_slice.start, profile_slice.stop, profile_slice.step or 1)
            )
            self.dataset_provenance["profile_adapter"] = (
                f"slice the stored experimental 5-90 degree/8500-bin profile to {self.pxrd_two_theta_min:g}-{self.pxrd_two_theta_max:g} degrees/{profile_bins} bins, then normalize"
            )
        if self.profile_augmentation_seed is not None:
            if not self.theoretical_profile:
                raise ValueError(
                    "profile_augmentation_seed requires a theoretical RRUFF profile"
                )
            if self.pxrd_representation != "peaks_128x4":
                raise ValueError(
                    "fixed simulated-profile augmentation supports peaks_128x4 only"
                )
        profile_augmented = bool(
            self.theoretical_profile and self.pxrd_dense_profile_augment
        )
        self.dataset_provenance["pxrd_input_view"] = (
            "reference_structure_theoretical_strong_augmented"
            if profile_augmented
            else "reference_structure_theoretical"
            if self.theoretical_profile
            else "experimental"
        )
        if profile_augmented:
            self.dataset_provenance["profile_augmentation"] = (
                "checkpoint_native_profile_randomization"
            )
        if self.profile_augmentation_seed is not None:
            self.dataset_provenance.update(
                profile_augmentation_seed=int(self.profile_augmentation_seed),
                profile_augmentation_seed_scheme="stable_seed('rruff-simulated-profile', base_seed, row_index)",
            )
        if self.theoretical_profile:
            self.dataset_provenance["profile_adapter"] = (
                "pymatgen XRDCalculator on the author-provided reference CIF, then the checkpoint-native dense-profile renderer"
            )
        official_adapter = (
            self.real_pxrd_peak_preprocessing == "realpxrd_solver_official"
        )
        if official_adapter and self.pxrd_representation != "legacy_di":
            raise ValueError(
                "RealPXRD-Solver preprocessing currently supports legacy_di only"
            )
        if official_adapter and (
            self.pxrd_two_theta_min != 5.0 or self.pxrd_two_theta_max != 90.0
        ):
            raise ValueError(
                "RealPXRD-Solver preprocessing requires the 5-90 degree grid"
            )
        if official_adapter:
            self.dataset_provenance.update(
                profile_adapter="RealPXRD-Solver public sample_flow.py: moving-average-3, polynomial-background-6, height-5, width-3, prominence-5%",
                exact_author_benchmark_adapter=True,
            )
        profiles = torch.load(
            self.root / "test_pv_xrd.pt", map_location="cpu", weights_only=True
        )
        with (self.root / "test.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        names = [row["material_id"] for row in rows]
        if len(names) != len(set(names)) or set(names) != set(profiles):
            raise ValueError(
                "author RRUFF CSV ids must uniquely match all profile keys"
            )
        items = []
        for row in rows:
            name = row["material_id"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = Structure.from_str(row["cif"], fmt="cif")
            atom_types = np.asarray(structure.atomic_numbers, dtype=np.int64)
            count = len(atom_types)
            if count > self.nmax or atom_types.max(initial=0) > VZ:
                raise ValueError(f"author RRUFF target {name!r} exceeds nmax/VZ")
            item = pack_structure_tokens(
                atom_types,
                structure.frac_coords,
                structure.lattice.abc,
                structure.lattice.angles,
                nmax=self.nmax,
                num_atoms=count,
                mp_id=name,
            )
            raw_profile = np.asarray(profiles[name], dtype=np.float32).reshape(-1)
            if raw_profile.shape != (8500,):
                raise ValueError(
                    f"author RRUFF profile {name!r} has shape {raw_profile.shape}"
                )
            if self.pxrd_public_api and self.pxrd_peak_source == "dense":
                profile = raw_profile[self._stored_dense_profile_slice]
            else:
                profile = (
                    raw_profile
                    if self.pxrd_public_api or official_adapter
                    else raw_profile[::10]
                )
            item["_profile"] = _normalized_profile(profile, bins=profile.size)
            item["mineral"] = name
            items.append(item)
        expected = int(manifest["counts"]["test"])
        if len(items) != expected:
            raise ValueError(f"author RRUFF has {len(items)} rows, expected {expected}")
        self.items = items
        if self.with_pxrd_peaks and (
            self.theoretical_profile
            or (self.pxrd_public_api and self.pxrd_peak_source == "bragg")
        ):
            self._attach_bragg_cache()
