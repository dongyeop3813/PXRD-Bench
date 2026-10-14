"""Canonical PXRD representations built from cached Bragg reflections.

New dataset code uses two public representations:

``peak_list``
    A padded list of selected peaks.  ``source='bragg'`` exposes ideal
    reflection positions/intensities; ``source='dense'`` detects peaks after
    pseudo-Voigt broadening and optional dense-profile augmentation.

``dense_profile``
    A uniformly sampled one-dimensional profile on ``[two_theta_min,
    two_theta_max)``.  The default is 8,500 points over 5--90 degrees.

Historical names remain dataset/model adapters; numerical work belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.pxrd.peaks import CU_KA_WAVELENGTH, augment_peaks, two_theta_to_d


PEAK_COORDINATES = {"d", "q", "theta", "two_theta"}
PEAK_FEATURES = {*PEAK_COORDINATES, "intensity", "prominence", "width"}
NORMALIZATIONS = {"none", "all_detected", "selected_topk"}
WIDTH_UNITS = {"bins", "two_theta", "position"}
PUBLIC_REPRESENTATIONS = {"peak_list", "dense_profile"}
PEAK_SOURCES = {"bragg", "dense"}
LEGACY_REPRESENTATIONS = {
    "legacy_di",
    "full_850",
    "peaks_20x4",
    "peaks_128x4",
    "peaks_20x4_norm_prom",
    "peaks_128x4_norm_prom",
    "broadened_basic",
    "broadened_shape",
    "peak_matrix",
}


@dataclass(frozen=True)
class PXRDTransformConfig:
    representation: str = "peak_list"
    peak_source: str = "bragg"
    peak_features: tuple[str, ...] = ("d", "intensity")
    max_peaks: int = 128
    wavelength: float = CU_KA_WAVELENGTH
    two_theta_min: float = 5.0
    two_theta_max: float = 90.0
    dense_n_bins: int = 8500
    selection: str = "strongest"
    low_angle_fraction: float = 0.25
    caglioti_u: float = 0.1
    caglioti_v: float = 0.01
    caglioti_w: float = 0.1
    pseudo_voigt_eta: float = 0.1
    position_normalized: bool = False
    intensity_normalization: str = "selected_topk"
    min_relative_intensity: float = 0.0
    prominence_normalization: str = "none"
    width_unit: str = "two_theta"
    width_normalization: str = "none"
    augment_sticks: bool = False
    augmentation: str = "none"
    jitter_deg: float = 0.1
    intensity_scale_min: float = 0.8
    intensity_scale_max: float = 1.2
    width_scale_min: float = 0.7
    width_scale_max: float = 1.5
    zero_shift_deg: float = 0.15
    eta_min: float = 0.0
    eta_max: float = 0.5
    background_max: float = 0.2
    noise_std_max: float = 0.02
    drop_fraction_max: float = 0.1
    xrdiff_gamma_shape: float | None = None
    xrdiff_gamma_scale: float | None = None

    def validate(self) -> None:
        if self.representation not in PUBLIC_REPRESENTATIONS:
            raise ValueError(f"unsupported PXRD representation: {self.representation}")
        if self.peak_source not in PEAK_SOURCES:
            raise ValueError(f"unsupported peak-list source: {self.peak_source}")
        if self.representation == "peak_list":
            if (
                not self.peak_features
                or not set(self.peak_features) <= PEAK_FEATURES
            ):
                raise ValueError(
                    f"unsupported peak-list features: {self.peak_features}"
                )
            coordinates = set(self.peak_features) & PEAK_COORDINATES
            if len(coordinates) != 1:
                raise ValueError("peak_list requires exactly one position coordinate")
            if "intensity" not in self.peak_features:
                raise ValueError("peak_list requires intensity")
            if self.peak_source == "bragg" and {"prominence", "width"} & set(
                self.peak_features
            ):
                raise ValueError(
                    "Bragg-derived peak lists do not define prominence or width; "
                    "use peak_source='dense'"
                )
        if self.max_peaks <= 0 or self.dense_n_bins <= 0:
            raise ValueError("max_peaks and dense_n_bins must be positive")
        if self.two_theta_max <= self.two_theta_min or self.wavelength <= 0:
            raise ValueError("invalid two-theta range or wavelength")
        if self.selection not in {"strongest", "prominence", "low_angle", "hybrid"}:
            raise ValueError(f"unsupported peak selection: {self.selection}")
        if (
            self.representation == "peak_list"
            and self.peak_source == "bragg"
            and self.selection == "prominence"
        ):
            raise ValueError("prominence selection requires peak_source='dense'")
        if not 0.0 <= self.low_angle_fraction <= 1.0:
            raise ValueError("low_angle_fraction must be in [0, 1]")
        if not 0.0 <= self.pseudo_voigt_eta <= 1.0:
            raise ValueError("pseudo_voigt_eta must be in [0, 1]")
        if self.intensity_normalization not in NORMALIZATIONS:
            raise ValueError("unsupported intensity normalization")
        if not 0.0 <= self.min_relative_intensity <= 1.0:
            raise ValueError("min_relative_intensity must be in [0, 1]")
        if self.prominence_normalization not in NORMALIZATIONS:
            raise ValueError("unsupported prominence normalization")
        if (
            "prominence" not in self.peak_features
            and self.prominence_normalization != "none"
        ):
            raise ValueError("peak list has no prominence to normalize")
        if self.width_unit not in WIDTH_UNITS:
            raise ValueError("unsupported peak-width unit")
        if self.width_normalization not in NORMALIZATIONS:
            raise ValueError("unsupported width normalization")
        if "width" not in self.peak_features and self.width_normalization != "none":
            raise ValueError("peak list has no width to normalize")
        if not 0.0 <= self.eta_min <= self.eta_max <= 1.0:
            raise ValueError("augmentation eta bounds must be in [0, 1]")
        if self.jitter_deg < 0 or self.zero_shift_deg < 0:
            raise ValueError("augmentation position bounds must be non-negative")
        if not 0.0 <= self.intensity_scale_min <= self.intensity_scale_max:
            raise ValueError("invalid augmentation intensity-scale bounds")
        if not 0.0 < self.width_scale_min <= self.width_scale_max:
            raise ValueError("invalid augmentation width-scale bounds")
        if not 0.0 <= self.drop_fraction_max <= 1.0:
            raise ValueError("drop_fraction_max must be in [0, 1]")
        if self.background_max < 0 or self.noise_std_max < 0:
            raise ValueError("dense-profile noise bounds must be non-negative")
        if self.augmentation not in {"none", "profile_randomization", "xrdiff_style"}:
            raise ValueError(f"unsupported PXRD augmentation: {self.augmentation}")
        if (self.xrdiff_gamma_shape is None) != (self.xrdiff_gamma_scale is None):
            raise ValueError("XRDiff Gamma shape and scale must be set together")
        if self.xrdiff_gamma_shape is not None and (
            self.xrdiff_gamma_shape <= 0 or self.xrdiff_gamma_scale <= 0
        ):
            raise ValueError("XRDiff Gamma shape and scale must be positive")


def normalize_peak_features(
    value: str | Iterable[str], *, position: str = "d"
) -> tuple[str, ...]:
    """Resolve one position, mandatory intensity, and optional shape fields."""
    if position not in PEAK_COORDINATES:
        raise ValueError(f"unsupported peak position coordinate: {position}")
    if isinstance(value, str):
        schemas = {
            "position_intensity": (position, "intensity"),
            "prominence_intensity_position": (
                "prominence",
                "intensity",
                position,
            ),
            "width_intensity_position": ("width", "intensity", position),
            "width_prominence_intensity_position": (
                "width",
                "prominence",
                "intensity",
                position,
            ),
        }
        if value not in schemas:
            raise ValueError(f"unsupported peak-list feature schema: {value}")
        fields = schemas[value]
    else:
        fields = tuple(str(part).strip() for part in value)
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate peak-list features: {fields}")
    coordinates = set(fields) & PEAK_COORDINATES
    if len(coordinates) != 1 or "intensity" not in fields:
        raise ValueError(
            "peak-list features require exactly one position and intensity"
        )
    if not set(fields) <= PEAK_FEATURES:
        raise ValueError(f"unsupported peak-list features: {fields}")
    return fields


def resolve_representation(
    representation: str,
    *,
    peak_source: str = "bragg",
) -> tuple[str, str]:
    """Return ``(public representation, peak source)`` for new or legacy names."""
    representation = str(representation)
    if representation == "peak_list":
        return representation, str(peak_source)
    if representation == "dense_profile":
        return representation, str(peak_source)
    if representation == "full_850":
        return "dense_profile", "dense"
    if representation == "legacy_di":
        return "peak_list", "bragg"
    if representation in LEGACY_REPRESENTATIONS:
        return "peak_list", "dense"
    raise ValueError(f"unsupported PXRD representation: {representation}")


def legacy_representation(
    representation: str,
    *,
    peak_source: str,
    peak_features: tuple[str, ...],
    position_normalized: bool = False,
    feature_source: str = "auto",
) -> str:
    """Map the public schema onto the unchanged model/checkpoint implementations."""
    if representation in LEGACY_REPRESENTATIONS:
        return representation
    if representation == "dense_profile":
        return "dense_profile"
    if representation != "peak_list":
        raise ValueError(f"unsupported PXRD representation: {representation}")
    if peak_source == "bragg" and {"prominence", "width"} & set(peak_features):
        raise ValueError("Bragg-derived peak lists cannot contain width/prominence")
    if (
        peak_source == "bragg"
        and feature_source != "online"
        and not position_normalized
        and len(peak_features) == 2
        and set(peak_features) == {"d", "intensity"}
    ):
        return "legacy_di"
    return "peak_matrix"


def model_pxrd_representation(model_args: dict) -> str:
    """Internal representation stored by new checkpoints, with legacy fallback."""
    return str(
        model_args.get(
            "pxrd_representation_internal",
            model_args.get("pxrd_representation", "legacy_di"),
        )
    )


def two_theta_grid(
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    n_bins: int = 8500,
) -> np.ndarray:
    if n_bins <= 0 or two_theta_max <= two_theta_min:
        raise ValueError("invalid dense-profile grid")
    step = (float(two_theta_max) - float(two_theta_min)) / int(n_bins)
    return float(two_theta_min) + step * np.arange(int(n_bins), dtype=np.float64)


def bragg_to_dense_profile(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    n_bins: int = 8500,
    caglioti_u: float = 0.1,
    caglioti_v: float = 0.01,
    caglioti_w: float = 0.1,
    eta: float = 0.1,
) -> np.ndarray:
    """Simulate a dense profile using pseudo-Voigt peak broadening."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    grid = two_theta_grid(two_theta_min, two_theta_max, n_bins)
    tt = np.asarray(two_theta, dtype=np.float64)
    it = np.asarray(intensity, dtype=np.float64)
    if tt.ndim != 1 or tt.shape != it.shape:
        raise ValueError("two_theta/intensity must be matching 1-D arrays")
    keep = (
        np.isfinite(tt)
        & np.isfinite(it)
        & (it > 0)
        & (tt >= two_theta_min)
        & (tt < two_theta_max)
    )
    tt, it = tt[keep], it[keep]
    profile = np.zeros(int(n_bins), dtype=np.float64)
    if tt.size:
        tangent = np.tan(np.deg2rad(tt * 0.5))
        fwhm = np.sqrt(
            np.clip(
                caglioti_u * tangent**2 + caglioti_v * tangent + caglioti_w,
                1e-8,
                None,
            )
        )
        for position, peak_intensity, peak_fwhm in zip(tt, it, fwhm, strict=True):
            scaled = (grid - position) / peak_fwhm
            squared = scaled**2
            # exp(-750) is exactly zero in float64. Retain every nonzero
            # Gaussian value and the full Lorentzian tail, in the same sum order.
            radius = peak_fwhm * np.sqrt(750.0 / (4.0 * np.log(2.0)))
            left, right = np.searchsorted(grid, [position - radius, position + radius])
            gaussian = np.zeros_like(scaled)
            gaussian[left:right] = np.exp(-4.0 * np.log(2.0) * squared[left:right])
            lorentzian = 1.0 / (1.0 + 4.0 * squared)
            profile += peak_intensity * (
                (1.0 - eta) * gaussian + eta * lorentzian
            )
    maximum = float(profile.max(initial=0.0))
    if maximum > 0:
        profile /= maximum
    return profile.astype(np.float32)


def augment_dense_profile(
    profile: np.ndarray,
    *,
    background_max: float = 0.2,
    noise_std_max: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply post-broadening background/noise and max-normalize."""
    rng = np.random.default_rng() if rng is None else rng
    out = np.asarray(profile, dtype=np.float32).copy()
    x = np.linspace(-1.0, 1.0, out.size, dtype=np.float32)
    if background_max > 0:
        background = rng.uniform(0.0, background_max) * (
            rng.uniform(0.0, 1.0)
            + rng.uniform(-0.5, 0.5) * x
            + rng.uniform(0.0, 0.5) * x**2
        )
        out += np.clip(background, 0.0, None)
    if noise_std_max > 0:
        out += rng.normal(0.0, rng.uniform(0.0, noise_std_max), size=out.shape)
    out = np.clip(out, 0.0, None)
    maximum = float(out.max(initial=0.0))
    if maximum > 0:
        out /= maximum
    return out.astype(np.float32)


def _coordinate_values(
    two_theta: np.ndarray,
    coordinate: str,
    wavelength: float,
    *,
    normalized: bool = False,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
) -> np.ndarray:
    if coordinate == "two_theta":
        values = np.asarray(two_theta, dtype=np.float32)
    elif coordinate == "theta":
        values = (0.5 * np.asarray(two_theta)).astype(np.float32)
    else:
        d = two_theta_to_d(two_theta, wavelength=wavelength)
        values = d if coordinate == "d" else (2.0 * np.pi / d).astype(np.float32)
    if not normalized:
        return values.astype(np.float32)
    bounds = _coordinate_values(
        np.asarray([two_theta_min, two_theta_max]),
        coordinate,
        wavelength,
    )
    low, high = float(bounds.min()), float(bounds.max())
    return ((values - low) / max(high - low, 1e-8)).astype(np.float32)


def _normalization_scale(
    values: np.ndarray, selected: np.ndarray, normalization: str
) -> float:
    if normalization == "none":
        return 1.0
    candidates = values if normalization == "all_detected" else values[selected]
    return max(float(candidates.max(initial=0.0)), 1e-8)


def _width_values(
    two_theta: np.ndarray,
    widths_in_bins: np.ndarray,
    *,
    config: PXRDTransformConfig,
) -> np.ndarray:
    if config.width_unit == "bins":
        return widths_in_bins.astype(np.float32)
    step = (config.two_theta_max - config.two_theta_min) / config.dense_n_bins
    widths_deg = widths_in_bins * step
    if config.width_unit == "two_theta":
        return widths_deg.astype(np.float32)
    lower = two_theta - 0.5 * widths_deg
    upper = two_theta + 0.5 * widths_deg
    return np.abs(
        _coordinate_values(
            upper,
            next(field for field in config.peak_features if field in PEAK_COORDINATES),
            config.wavelength,
            normalized=config.position_normalized,
            two_theta_min=config.two_theta_min,
            two_theta_max=config.two_theta_max,
        )
        - _coordinate_values(
            lower,
            next(field for field in config.peak_features if field in PEAK_COORDINATES),
            config.wavelength,
            normalized=config.position_normalized,
            two_theta_min=config.two_theta_min,
            two_theta_max=config.two_theta_max,
        )
    ).astype(np.float32)


def _select_indices(
    positions: np.ndarray,
    intensity: np.ndarray,
    *,
    max_peaks: int,
    selection: str,
    low_angle_fraction: float,
    prominence: np.ndarray | None = None,
) -> np.ndarray:
    count = len(positions)
    if count <= max_peaks:
        return np.arange(count)
    scores = prominence if selection == "prominence" else intensity
    if selection == "low_angle":
        return np.argsort(positions, kind="stable")[:max_peaks]
    if selection == "hybrid":
        n_low = min(max(int(round(max_peaks * low_angle_fraction)), 1), max_peaks)
        low = np.argsort(positions, kind="stable")[:n_low]
        chosen = set(map(int, low))
        ranked = [
            int(index)
            for index in np.argsort(scores, kind="stable")[::-1]
            if int(index) not in chosen
        ]
        return np.asarray([*low, *ranked[: max_peaks - n_low]], dtype=np.int64)
    return np.argsort(scores, kind="stable")[::-1][:max_peaks]


def _pack_peak_list(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    config: PXRDTransformConfig,
    prominence: np.ndarray | None = None,
    width: np.ndarray | None = None,
) -> dict[str, np.ndarray | str]:
    maximum = float(intensity.max(initial=0.0))
    if maximum > 0.0 and config.min_relative_intensity > 0.0:
        keep = intensity >= config.min_relative_intensity * maximum
        two_theta, intensity = two_theta[keep], intensity[keep]
        if prominence is not None:
            prominence = prominence[keep]
        if width is not None:
            width = width[keep]
    selected = _select_indices(
        two_theta,
        intensity,
        max_peaks=config.max_peaks,
        selection=config.selection,
        low_angle_fraction=config.low_angle_fraction,
        prominence=prominence,
    )
    selected = selected[np.argsort(two_theta[selected], kind="stable")]
    tt, it = two_theta[selected], intensity[selected]
    intensity_scale = _normalization_scale(
        intensity, selected, config.intensity_normalization
    )
    it = it / intensity_scale
    fields: dict[str, np.ndarray | str] = {
        "representation": "peak_list",
        "source": config.peak_source,
        "pad_mask": np.ones(config.max_peaks, dtype=np.bool_),
    }
    n = len(selected)
    fields["pad_mask"][:n] = False
    for feature in config.peak_features:
        values = np.zeros(config.max_peaks, dtype=np.float32)
        if feature in PEAK_COORDINATES:
            values[:n] = _coordinate_values(
                tt,
                feature,
                config.wavelength,
                normalized=config.position_normalized,
                two_theta_min=config.two_theta_min,
                two_theta_max=config.two_theta_max,
            )
        elif feature == "intensity":
            values[:n] = it
        elif feature == "prominence":
            assert prominence is not None
            p = prominence[selected]
            values[:n] = p / _normalization_scale(
                prominence, selected, config.prominence_normalization
            )
        else:
            assert width is not None
            values[:n] = width[selected] / _normalization_scale(
                width, selected, config.width_normalization
            )
        fields[feature] = values
    return fields


def bragg_to_peak_list(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    config: PXRDTransformConfig,
) -> dict[str, np.ndarray | str]:
    config.validate()
    if config.peak_source != "bragg":
        raise ValueError("bragg_to_peak_list requires peak_source='bragg'")
    tt = np.asarray(two_theta, dtype=np.float32)
    it = np.asarray(intensity, dtype=np.float32)
    keep = (
        np.isfinite(tt)
        & np.isfinite(it)
        & (it > 0)
        & (tt >= config.two_theta_min)
        & (tt < config.two_theta_max)
    )
    return _pack_peak_list(tt[keep], it[keep], config=config)


def dense_profile_to_peak_list(
    profile: np.ndarray,
    config: PXRDTransformConfig,
    *,
    rng: np.random.Generator | None = None,
    apply_stick_augmentation: bool = True,
) -> dict[str, np.ndarray | str]:
    from scipy.signal import find_peaks, peak_widths

    config.validate()
    if config.peak_source != "dense":
        raise ValueError("dense_profile_to_peak_list requires peak_source='dense'")
    dense = np.asarray(profile, dtype=np.float32).reshape(-1)
    if dense.shape != (config.dense_n_bins,):
        raise ValueError(
            f"expected {config.dense_n_bins} dense-profile points, got {dense.shape}"
        )
    maximum = float(dense.max(initial=0.0))
    dense = dense / maximum if maximum > 0 else dense
    indices, properties = find_peaks(dense, prominence=0.0)
    step = (config.two_theta_max - config.two_theta_min) / config.dense_n_bins
    positions = config.two_theta_min + step * indices.astype(np.float32)
    widths = _width_values(
        positions,
        peak_widths(dense, indices, rel_height=0.5)[0].astype(np.float32),
        config=config,
    )
    prominences = np.asarray(properties["prominences"], dtype=np.float32)
    intensities = dense[indices]
    if config.augment_sticks and apply_stick_augmentation and indices.size:
        rng = np.random.default_rng() if rng is None else rng
        if config.drop_fraction_max > 0.0 and indices.size > 1:
            drop_count = min(
                int(rng.uniform(0.0, config.drop_fraction_max) * indices.size),
                indices.size - 1,
            )
            if drop_count:
                keep = np.ones(indices.size, dtype=np.bool_)
                keep[np.argsort(intensities)[:drop_count]] = False
                positions, intensities, prominences, widths = (
                    values[keep]
                    for values in (positions, intensities, prominences, widths)
                )
        positions = positions + rng.uniform(
            -config.jitter_deg, config.jitter_deg, size=positions.shape
        ).astype(np.float32)
        positions += rng.uniform(-config.zero_shift_deg, config.zero_shift_deg)
        scales = rng.uniform(
            config.intensity_scale_min,
            config.intensity_scale_max,
            size=intensities.shape,
        ).astype(np.float32)
        intensities = intensities * scales
        prominences = prominences * scales
    return _pack_peak_list(
        positions,
        intensities,
        config=config,
        prominence=prominences,
        width=widths,
    )


def transform_dense_profile(
    profile: np.ndarray,
    config: PXRDTransformConfig,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | str]:
    """Transform an already sampled dense profile without re-rendering it."""
    config.validate()
    if config.peak_source != "dense":
        raise ValueError("transform_dense_profile requires peak_source='dense'")
    rng = np.random.default_rng() if rng is None else rng
    dense = np.asarray(profile, dtype=np.float32).reshape(-1)
    if dense.shape != (config.dense_n_bins,):
        raise ValueError(
            f"expected {config.dense_n_bins} dense-profile points, got {dense.shape}"
        )
    if config.augmentation == "xrdiff_style":
        raise ValueError("xrdiff_style requires Bragg peaks before profile rendering")
    if config.augmentation == "profile_randomization":
        dense = augment_dense_profile(
            dense,
            background_max=config.background_max,
            noise_std_max=config.noise_std_max,
            rng=rng,
        )
    if config.representation == "dense_profile":
        return {
            "representation": "dense_profile",
            "source": "dense",
            "profile": dense.copy(),
        }
    return dense_profile_to_peak_list(dense, config, rng=rng)


def attach_pxrd_result(
    out: dict,
    result: dict[str, np.ndarray | str],
    config: PXRDTransformConfig,
    *,
    internal_representation: str,
) -> None:
    """Attach one public PXRD result plus the legacy model-input adapter."""
    import torch

    if config.representation == "dense_profile":
        profile = torch.from_numpy(result["profile"])
        out["pxrd_dense_profile"] = profile
        out["xrd"] = profile
        out["pxrd_d"] = profile
        out["pxrd_i"] = torch.zeros_like(profile)
        out["pxrd_peak_pad_mask"] = torch.zeros_like(profile, dtype=torch.bool)
        return

    pad = torch.from_numpy(result["pad_mask"])
    out["pxrd_peak_list_pad_mask"] = pad
    out["pxrd_peak_list_source"] = str(result["source"])
    for field in config.peak_features:
        out[f"pxrd_peak_list_{field}"] = torch.from_numpy(result[field])
    position_field = next(
        field for field in config.peak_features if field in PEAK_COORDINATES
    )
    position = torch.from_numpy(result[position_field])
    intensity = torch.from_numpy(result["intensity"])
    out["pxrd_position"], out["pxrd_intensity"] = position, intensity
    if internal_representation == "legacy_di":
        out["pxrd_d"], out["pxrd_i"] = position, intensity
        out["pxrd_peak_pad_mask"] = pad
        return
    if "prominence" in result:
        out["pxrd_prominence"] = torch.from_numpy(result["prominence"])
    if "width" in result:
        out["pxrd_width"] = torch.from_numpy(result["width"])
    matrix = torch.stack(
        [torch.from_numpy(result[field]) for field in config.peak_features], dim=-1
    ).flatten()
    out["pxrd_d"] = matrix
    out["pxrd_i"] = torch.zeros_like(matrix)
    out["pxrd_peak_pad_mask"] = torch.zeros_like(matrix, dtype=torch.bool)


def transform_bragg_pattern(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    config: PXRDTransformConfig,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | str]:
    """Apply augmentation, broadening, and the requested public representation."""
    config.validate()
    rng = np.random.default_rng() if rng is None else rng
    tt = np.asarray(two_theta, dtype=np.float32)
    it = np.asarray(intensity, dtype=np.float32)
    u, v, w, eta = (
        config.caglioti_u,
        config.caglioti_v,
        config.caglioti_w,
        config.pseudo_voigt_eta,
    )

    if config.augment_sticks:
        tt, it = augment_peaks(
            tt,
            it,
            jitter_deg=config.jitter_deg,
            int_scale_min=config.intensity_scale_min,
            int_scale_max=config.intensity_scale_max,
            drop_frac_max=config.drop_fraction_max,
            rng=rng,
        )
        if tt.size:
            tt += rng.uniform(-config.zero_shift_deg, config.zero_shift_deg)
    if config.augmentation == "profile_randomization":
        width_scale = rng.uniform(config.width_scale_min, config.width_scale_max)
        u, v, w = u * width_scale**2, v * width_scale**2, w * width_scale**2
        eta = rng.uniform(config.eta_min, config.eta_max)
    elif config.augmentation == "xrdiff_style":
        variants = (
            (0.05, -0.06, 0.07),
            (0.05, -0.01, 0.01),
            (0.0, 0.0, 0.1),
            (0.0, 0.0, rng.uniform(0.001, 0.1)),
        )
        variant = int(rng.integers(0, len(variants)))
        u, v, w = variants[variant]
        if variant != 1 and config.xrdiff_gamma_shape is not None:
            it = it * rng.gamma(
                config.xrdiff_gamma_shape,
                config.xrdiff_gamma_scale,
                size=it.shape,
            )

    if config.representation == "peak_list" and config.peak_source == "bragg":
        return bragg_to_peak_list(tt, it, config)

    profile = bragg_to_dense_profile(
        tt,
        it,
        two_theta_min=config.two_theta_min,
        two_theta_max=config.two_theta_max,
        n_bins=config.dense_n_bins,
        caglioti_u=u,
        caglioti_v=v,
        caglioti_w=w,
        eta=eta,
    )
    if config.augmentation == "profile_randomization":
        profile = augment_dense_profile(
            profile,
            background_max=config.background_max,
            noise_std_max=config.noise_std_max,
            rng=rng,
        )
    if config.representation == "dense_profile":
        return {
            "representation": "dense_profile",
            "source": "dense",
            "profile": profile,
        }
    return dense_profile_to_peak_list(
        profile,
        config,
        rng=rng,
        apply_stick_augmentation=False,
    )
