"""Discrete d-I (interplanar spacing - intensity) peak-list utilities.

RealPXRD LMDB records carry sparse Bragg peak lists as ``pxrd_x`` (2theta, deg,
Cu Ka) and ``pxrd_y`` (intensities, max-normalized to 100). Following
RealPXRD-Solver (related_works/real_pxrd_solver.pdf, Methods 4.1-4.3), the
PXRD-conditioned model consumes these as a fixed-length d-I list:

  * 2theta -> d via Bragg's law  d = lambda / (2 sin(theta)),
  * intensities max-normalized to 1,
  * truncated (keeping the strongest peaks) or zero-padded to ``max_peaks``.

Training-time augmentation mirrors the paper's experiment-mimicking
perturbations (Methods 4.2 / 4.8): 2theta peak-position jitter, per-peak
intensity scaling, and weak-peak dropping, all applied in 2theta space before
the d conversion. All functions are pure numpy so they can run inside
DataLoader workers (seeded per worker via ``np.random``).
"""

from __future__ import annotations
import numpy as np

CU_KA_WAVELENGTH = 1.54184
XRDIFF_PROFILE_BINS = 850


def two_theta_to_d(
    two_theta_deg: np.ndarray, wavelength: float = CU_KA_WAVELENGTH
) -> np.ndarray:
    """Bragg's law: d = lambda / (2 sin(theta)); 2theta in degrees."""
    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64)) * 0.5
    sin_theta = np.clip(np.sin(theta), 1e-08, None)
    return (wavelength / (2.0 * sin_theta)).astype(np.float32)


def d_to_two_theta(d: np.ndarray, wavelength: float = CU_KA_WAVELENGTH) -> np.ndarray:
    """Inverse Bragg conversion; d-spacing in Angstrom to 2theta in degrees."""
    d = np.asarray(d, dtype=np.float64)
    theta = np.arcsin(np.clip(wavelength / (2.0 * np.clip(d, 1e-08, None)), -1.0, 1.0))
    return np.rad2deg(2.0 * theta).astype(np.float32)


def augment_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    jitter_deg: float = 0.1,
    int_scale_min: float = 0.8,
    int_scale_max: float = 1.2,
    drop_frac_max: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Experiment-mimicking peak-list perturbation (paper Methods 4.2).

    (i) per-peak 2theta jitter ~ U(-jitter_deg, +jitter_deg); (ii) per-peak
    intensity scale ~ U(int_scale_min, int_scale_max); (iii) drop a random
    fraction ~ U(0, drop_frac_max) of the weakest peaks (at least one peak is
    always kept). Intensities are re-max-normalized to the input max afterwards.
    """
    tt = np.asarray(two_theta, dtype=np.float32).copy()
    it = np.asarray(intensity, dtype=np.float32).copy()
    if tt.size == 0:
        return (tt, it)
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    peak_max = float(it.max()) if it.size else 1.0
    if drop_frac_max > 0.0 and tt.size > 1:
        frac = float(rng.uniform(0.0, drop_frac_max))
        n_drop = min(int(frac * tt.size), tt.size - 1)
        if n_drop > 0:
            weakest = np.argsort(it)[:n_drop]
            keep = np.ones(tt.size, dtype=bool)
            keep[weakest] = False
            tt, it = (tt[keep], it[keep])
    if jitter_deg > 0.0:
        tt = tt + rng.uniform(-jitter_deg, jitter_deg, size=tt.shape).astype(np.float32)
    if int_scale_max > int_scale_min or int_scale_min != 1.0:
        it = it * rng.uniform(int_scale_min, int_scale_max, size=it.shape).astype(
            np.float32
        )
    m = float(it.max()) if it.size else 0.0
    if m > 0:
        it = it * (peak_max / m)
    return (tt, it)


def pack_di_peaks(
    d: np.ndarray,
    intensity: np.ndarray,
    max_peaks: int,
    *,
    selection: str = "strongest",
    low_angle_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-length packing of a d-I list (paper Methods 4.1).

    Keeps the ``max_peaks`` strongest peaks when over-long, then sorts by
    descending d (low-angle first) for determinism; pads with zeros. Returns
    ``(d_pad (P,), i_pad (P,), pad_mask (P,) bool)`` with True where padded.
    Intensities are max-normalized to 1.
    """
    d = np.asarray(d, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    if d.shape != intensity.shape or d.ndim != 1:
        raise ValueError(
            f"d/intensity must be matching 1-D arrays, got {d.shape} vs {intensity.shape}"
        )
    if selection not in {"strongest", "low_angle", "hybrid"}:
        raise ValueError(f"Unknown peak selection strategy: {selection}")
    if not 0.0 <= low_angle_fraction <= 1.0:
        raise ValueError("low_angle_fraction must be in [0, 1]")
    if d.size > max_peaks:
        if selection == "strongest":
            keep = np.argsort(intensity)[::-1][:max_peaks]
        elif selection == "low_angle":
            keep = np.argsort(d)[::-1][:max_peaks]
        else:
            n_low = min(max(int(round(max_peaks * low_angle_fraction)), 1), max_peaks)
            low = np.argsort(d)[::-1][:n_low]
            chosen = set((int(i) for i in low))
            strong = [
                int(i) for i in np.argsort(intensity)[::-1] if int(i) not in chosen
            ]
            keep = np.asarray(
                [*low.tolist(), *strong[: max_peaks - n_low]], dtype=np.int64
            )
        d, intensity = (d[keep], intensity[keep])
    order = np.argsort(d)[::-1]
    d, intensity = (d[order], intensity[order])
    m = float(intensity.max()) if intensity.size else 0.0
    if m > 0:
        intensity = intensity / m
    n = d.size
    d_pad = np.zeros(max_peaks, dtype=np.float32)
    i_pad = np.zeros(max_peaks, dtype=np.float32)
    pad_mask = np.ones(max_peaks, dtype=np.bool_)
    d_pad[:n] = d
    i_pad[:n] = intensity
    pad_mask[:n] = False
    return (d_pad, i_pad, pad_mask)


def peaks_to_di_tensors(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    max_peaks: int,
    wavelength: float = CU_KA_WAVELENGTH,
    two_theta_min: float | None = None,
    two_theta_max: float | None = None,
    selection: str = "strongest",
    low_angle_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2theta/intensity peak list -> packed (d, I, pad_mask) numpy triplet."""
    tt = np.asarray(two_theta, dtype=np.float32)
    it = np.asarray(intensity, dtype=np.float32)
    if two_theta_min is not None or two_theta_max is not None:
        lo = -np.inf if two_theta_min is None else float(two_theta_min)
        hi = np.inf if two_theta_max is None else float(two_theta_max)
        keep = (tt >= lo) & (tt <= hi)
        tt, it = (tt[keep], it[keep])
    d = two_theta_to_d(tt, wavelength=wavelength)
    return pack_di_peaks(
        d,
        it,
        max_peaks=max_peaks,
        selection=selection,
        low_angle_fraction=low_angle_fraction,
    )


def pseudo_voigt_profile(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    step: float = 0.1,
    caglioti_u: float = 0.1,
    caglioti_v: float = 0.01,
    caglioti_w: float = 0.1,
    eta: float = 0.1,
) -> np.ndarray:
    """XRDiff pseudo-Voigt profile on ``[two_theta_min, two_theta_max)``."""
    if step <= 0 or two_theta_max <= two_theta_min:
        raise ValueError("invalid PXRD profile range or step")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1]")
    grid = np.arange(two_theta_min, two_theta_max, step, dtype=np.float64)
    tt = np.asarray(two_theta, dtype=np.float64)
    it = np.asarray(intensity, dtype=np.float64)
    if tt.shape != it.shape or tt.ndim != 1:
        raise ValueError("two_theta/intensity must be matching 1-D arrays")
    keep = (
        np.isfinite(tt)
        & np.isfinite(it)
        & (it > 0)
        & (tt >= two_theta_min)
        & (tt < two_theta_max)
    )
    tt, it = (tt[keep], it[keep])
    profile = np.zeros_like(grid)
    if tt.size:
        tan_theta = np.tan(np.deg2rad(tt * 0.5))
        fwhm = np.sqrt(
            np.clip(
                caglioti_u * tan_theta**2 + caglioti_v * tan_theta + caglioti_w,
                1e-08,
                None,
            )
        )
        scaled = (grid[None, :] - tt[:, None]) / fwhm[:, None]
        np.square(scaled, out=scaled)
        gaussian = scaled * (-4 * np.log(2))
        underflow = gaussian < -746.0
        np.exp(gaussian, out=gaussian, where=~underflow)
        gaussian[underflow] = 0.0
        gaussian *= 1 - eta
        scaled *= 4
        scaled += 1
        np.reciprocal(scaled, out=scaled)
        scaled *= eta
        gaussian += scaled
        gaussian *= it[:, None]
        profile = gaussian.sum(axis=0)
        if profile.max() > 0:
            profile /= profile.max()
    return profile.astype(np.float32)


def peaks_to_pseudo_voigt_tensors(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    max_peaks: int,
    wavelength: float = CU_KA_WAVELENGTH,
    two_theta_min: float = 5.0,
    two_theta_max: float = 80.0,
    step: float = 0.1,
    caglioti_u: float = 0.1,
    caglioti_v: float = 0.01,
    caglioti_w: float = 0.1,
    eta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Historical broadened peak fields used by broadened_basic/shape checkpoints."""
    from scipy.signal import find_peaks, peak_widths

    if max_peaks <= 0:
        raise ValueError("max_peaks must be positive")
    profile = pseudo_voigt_profile(
        two_theta,
        intensity,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
        step=step,
        caglioti_u=caglioti_u,
        caglioti_v=caglioti_v,
        caglioti_w=caglioti_w,
        eta=eta,
    )
    grid = np.arange(two_theta_min, two_theta_max, step, dtype=np.float64).astype(
        np.float32
    )
    peak_idx, properties = find_peaks(profile, prominence=0.0)
    if peak_idx.size == 0 and bool(np.any(profile > 0.0)):
        peak_idx = np.asarray([int(np.argmax(profile))], dtype=np.int64)
        prominences = np.asarray([float(profile[peak_idx[0]])], dtype=np.float64)
        widths = np.asarray([1.0], dtype=np.float64)
    elif peak_idx.size:
        prominences = np.asarray(properties["prominences"], dtype=np.float64)
        widths = np.asarray(
            peak_widths(profile, peak_idx, rel_height=0.5)[0], dtype=np.float64
        )
    else:
        prominences = np.empty(0, dtype=np.float64)
        widths = np.empty(0, dtype=np.float64)
    if peak_idx.size > max_peaks:
        selected = np.argsort(profile[peak_idx], kind="stable")[::-1][:max_peaks]
        peak_idx, prominences, widths = (
            peak_idx[selected],
            prominences[selected],
            widths[selected],
        )
    order = np.argsort(grid[peak_idx], kind="stable")
    peak_idx, prominences, widths = (peak_idx[order], prominences[order], widths[order])
    outputs = [np.zeros(max_peaks, dtype=np.float32) for _ in range(5)]
    position, peak_intensity, prominence, width, d_spacing = outputs
    pad_mask = np.ones(max_peaks, dtype=np.bool_)
    count = int(peak_idx.size)
    if count:
        tt = grid[peak_idx]
        position[:count] = (tt - two_theta_min) / (two_theta_max - two_theta_min)
        peak_intensity[:count] = profile[peak_idx]
        prominence[:count] = prominences
        if widths.max(initial=0.0) > 0:
            width[:count] = widths / widths.max()
        d_spacing[:count] = two_theta_to_d(tt, wavelength=wavelength)
        pad_mask[:count] = False
    return (position, peak_intensity, prominence, width, d_spacing, pad_mask)


def xrdiff_feature_vector(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    representation: str,
    *,
    min_prominence: float = 0.0,
    peak_count: int = 128,
) -> np.ndarray:
    """Return XRDiff's 850-profile or flattened top-20 peak descriptors."""
    profile = pseudo_voigt_profile(two_theta, intensity)
    return xrdiff_feature_vector_from_profile(
        profile, representation, min_prominence=min_prominence, peak_count=peak_count
    )


def xrdiff_feature_vector_from_profile(
    profile: np.ndarray,
    representation: str,
    *,
    wavelength: float = CU_KA_WAVELENGTH,
    min_prominence: float = 0.0,
    peak_count: int = 128,
) -> np.ndarray:
    """Encode a normalized 850-bin experimental or simulated profile."""
    from scipy.signal import find_peaks, peak_widths

    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (XRDIFF_PROFILE_BINS,):
        raise ValueError(f"expected an 850-bin profile, got {profile.shape}")
    if not 0.0 <= min_prominence <= 1.0:
        raise ValueError("min_prominence must be finite and in [0, 1]")
    if min_prominence and representation != "peaks_128x4":
        raise ValueError("min_prominence requires peaks_128x4")
    if (
        not isinstance(peak_count, int)
        or not 1 <= peak_count <= (XRDIFF_PROFILE_BINS - 1) // 2
    ):
        raise ValueError("peak_count must be an integer in [1, 424]")
    if peak_count != 128 and representation != "peaks_128x4":
        raise ValueError("variable peak_count requires peaks_128x4")
    maximum = float(profile.max(initial=0))
    profile = profile / maximum if maximum > 0 else profile
    if representation == "full_850":
        return profile
    normalized_peak_counts = {"peaks_20x4_norm_prom": 20, "peaks_128x4_norm_prom": 128}
    if representation in normalized_peak_counts:
        max_peaks = normalized_peak_counts[representation]
        peaks, props = find_peaks(profile, prominence=0.0)
        descriptors = np.zeros((max_peaks, 4), dtype=np.float32)
        if not peaks.size:
            return descriptors.ravel()
        prominences = props["prominences"]
        widths = peak_widths(profile, peaks, rel_height=0.5)[0]
        width_scale = widths.max(initial=0.0)
        selected = np.argsort(prominences, kind="stable")[::-1][:max_peaks]
        selected = selected[np.argsort(peaks[selected], kind="stable")]
        n = len(selected)
        descriptors[:n, 0] = profile[peaks[selected]]
        descriptors[:n, 1] = peaks[selected] / (XRDIFF_PROFILE_BINS - 1)
        descriptors[:n, 2] = prominences[selected]
        if width_scale > 0:
            descriptors[:n, 3] = widths[selected] / width_scale
        return descriptors.ravel()
    if representation == "peaks_128x4":
        peaks, props = find_peaks(profile, prominence=min_prominence)
        descriptors = np.zeros((peak_count, 4), dtype=np.float32)
        if not peaks.size:
            return descriptors.ravel()
        widths = peak_widths(profile, peaks, rel_height=0.5)[0] * 0.1
        selected = np.argsort(profile[peaks], kind="stable")[::-1][:peak_count]
        peaks = peaks[selected]
        order = np.argsort(peaks, kind="stable")
        peaks = peaks[order]
        n = len(peaks)
        descriptors[:n, 0] = widths[selected][order]
        descriptors[:n, 1] = props["prominences"][selected][order]
        descriptors[:n, 2] = profile[peaks]
        descriptors[:n, 3] = two_theta_to_d(5.0 + 0.1 * peaks, wavelength=wavelength)
        return descriptors.ravel()
    if representation != "peaks_20x4":
        raise ValueError(f"unsupported XRDiff representation: {representation}")
    peaks, props = find_peaks(profile, prominence=0.0)
    if not peaks.size:
        return np.zeros(80, dtype=np.float32)
    prominences = props["prominences"]
    widths = peak_widths(profile, peaks, rel_height=0.5)[0]
    selected = np.argsort(profile[peaks], kind="stable")[::-1][:20]
    peaks, prominences, widths = (
        peaks[selected],
        prominences[selected],
        widths[selected],
    )
    order = np.argsort(peaks, kind="stable")
    peaks, prominences, widths = (peaks[order], prominences[order], widths[order])
    descriptors = np.zeros((20, 4), dtype=np.float32)
    n = len(peaks)
    descriptors[:n, 0] = profile[peaks]
    descriptors[:n, 1] = peaks / (XRDIFF_PROFILE_BINS - 1)
    descriptors[:n, 2] = prominences
    if widths.max(initial=0) > 0:
        descriptors[:n, 3] = widths / widths.max()
    return descriptors.ravel()


def peak_matrix_feature_vector(
    two_theta: np.ndarray, intensity: np.ndarray, **kwargs
) -> np.ndarray:
    """Build a configurable peak matrix from a simulated reflection list."""
    two_theta_min = float(kwargs.get("two_theta_min", 5.0))
    two_theta_max = float(kwargs.get("two_theta_max", 90.0))
    return peak_matrix_feature_vector_from_profile(
        pseudo_voigt_profile(
            two_theta,
            intensity,
            two_theta_min=two_theta_min,
            two_theta_max=two_theta_max,
            step=(two_theta_max - two_theta_min) / XRDIFF_PROFILE_BINS,
        ),
        **kwargs,
    )


def peak_matrix_feature_vector_from_profile(
    profile: np.ndarray,
    *,
    max_peaks: int,
    features: str = "width_prominence_intensity_position",
    position: str = "d",
    normalize_position: bool = False,
    width_normalization: str = "none",
    selection: str = "prominence",
    low_angle_fraction: float = 0.25,
    wavelength: float = CU_KA_WAVELENGTH,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
) -> np.ndarray:
    """Configurable flattened peak list extracted from an 850-bin profile."""
    from scipy.signal import find_peaks, peak_widths

    if features not in {"width_prominence_intensity_position", "position_intensity"}:
        raise ValueError(f"unsupported peak feature schema: {features}")
    if position not in {"d", "q", "theta", "two_theta"}:
        raise ValueError(f"unsupported peak position coordinate: {position}")
    if width_normalization not in {"none", "all_detected", "selected_topk"}:
        raise ValueError(f"unsupported width normalization: {width_normalization}")
    if selection not in {"strongest", "prominence", "low_angle", "hybrid"}:
        raise ValueError(f"unsupported peak selection: {selection}")
    if max_peaks <= 0 or not 0.0 <= low_angle_fraction <= 1.0:
        raise ValueError("max_peaks must be positive and low_angle_fraction in [0, 1]")
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (XRDIFF_PROFILE_BINS,):
        raise ValueError(f"expected an 850-bin profile, got {profile.shape}")
    maximum = float(profile.max(initial=0.0))
    profile = profile / maximum if maximum > 0 else profile
    peaks, props = find_peaks(profile, prominence=0.0)
    feature_dim = 4 if features == "width_prominence_intensity_position" else 2
    descriptors = np.zeros((max_peaks, feature_dim), dtype=np.float32)
    if not peaks.size:
        return descriptors.ravel()
    prominences = props["prominences"]
    widths = peak_widths(profile, peaks, rel_height=0.5)[0].astype(np.float32)
    if width_normalization == "all_detected" and widths.max(initial=0.0) > 0:
        widths /= widths.max()
    scores = prominences if selection == "prominence" else profile[peaks]
    if peaks.size > max_peaks:
        if selection == "low_angle":
            selected = np.argsort(peaks, kind="stable")[:max_peaks]
        elif selection == "hybrid":
            n_low = min(max(int(round(max_peaks * low_angle_fraction)), 1), max_peaks)
            low = np.argsort(peaks, kind="stable")[:n_low]
            chosen = set((int(i) for i in low))
            ranked = [
                int(i)
                for i in np.argsort(scores, kind="stable")[::-1]
                if int(i) not in chosen
            ]
            selected = np.asarray([*low, *ranked[: max_peaks - n_low]])
        else:
            selected = np.argsort(scores, kind="stable")[::-1][:max_peaks]
        peaks, prominences, widths = (
            peaks[selected],
            prominences[selected],
            widths[selected],
        )
    if width_normalization == "selected_topk" and widths.max(initial=0.0) > 0:
        widths /= widths.max()
    order = np.argsort(peaks, kind="stable")
    peaks, prominences, widths = (peaks[order], prominences[order], widths[order])
    two_theta = (
        two_theta_min + (two_theta_max - two_theta_min) * peaks / XRDIFF_PROFILE_BINS
    )
    position_values = _peak_positions(
        two_theta,
        coordinate=position,
        normalized=normalize_position,
        wavelength=wavelength,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
    )
    n = len(peaks)
    if feature_dim == 4:
        descriptors[:n, 0] = (
            widths
            if width_normalization != "none"
            else widths * ((two_theta_max - two_theta_min) / XRDIFF_PROFILE_BINS)
        )
        descriptors[:n, 1] = prominences
        descriptors[:n, 2] = profile[peaks]
        descriptors[:n, 3] = position_values
    else:
        descriptors[:n, 0] = position_values
        descriptors[:n, 1] = profile[peaks]
    return descriptors.ravel()


def _peak_positions(
    two_theta: np.ndarray,
    *,
    coordinate: str,
    normalized: bool,
    wavelength: float,
    two_theta_min: float,
    two_theta_max: float,
) -> np.ndarray:
    two_theta = np.asarray(two_theta, dtype=np.float32)
    if coordinate == "two_theta":
        values = two_theta
    elif coordinate == "theta":
        values = 0.5 * two_theta
    else:
        d = two_theta_to_d(two_theta, wavelength=wavelength)
        values = d if coordinate == "d" else 2.0 * np.pi / d
    if not normalized:
        return values.astype(np.float32)
    bounds = _peak_positions(
        np.asarray([two_theta_min, two_theta_max], dtype=np.float32),
        coordinate=coordinate,
        normalized=False,
        wavelength=wavelength,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
    )
    lo, hi = (float(bounds.min()), float(bounds.max()))
    return ((values - lo) / max(hi - lo, 1e-08)).astype(np.float32)


def _two_theta_from_peak_positions(
    values: np.ndarray,
    *,
    coordinate: str,
    normalized: bool,
    wavelength: float,
    two_theta_min: float,
    two_theta_max: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if normalized:
        bounds = _peak_positions(
            np.asarray([two_theta_min, two_theta_max], dtype=np.float32),
            coordinate=coordinate,
            normalized=False,
            wavelength=wavelength,
            two_theta_min=two_theta_min,
            two_theta_max=two_theta_max,
        )
        lo, hi = (float(bounds.min()), float(bounds.max()))
        values = lo + values * (hi - lo)
    if coordinate == "two_theta":
        return values
    if coordinate == "theta":
        return 2.0 * values
    d = values if coordinate == "d" else 2.0 * np.pi / np.clip(values, 1e-08, None)
    return d_to_two_theta(d, wavelength=wavelength)


def augment_peak_matrix_features(
    values: np.ndarray,
    *,
    max_peaks: int,
    features: str,
    position: str,
    normalize_position: bool,
    width_normalization: str = "none",
    wavelength: float,
    two_theta_min: float,
    two_theta_max: float,
    jitter_deg: float = 0.1,
    int_scale_min: float = 0.8,
    int_scale_max: float = 1.2,
    width_scale_min: float = 0.7,
    width_scale_max: float = 1.5,
    drop_frac_max: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cheap per-sample augmentation of a cached generic peak matrix."""
    feature_dim = 4 if features == "width_prominence_intensity_position" else 2
    out = np.asarray(values, dtype=np.float32).reshape(max_peaks, feature_dim).copy()
    intensity_column = 2 if feature_dim == 4 else 1
    position_column = 3 if feature_dim == 4 else 0
    valid = np.flatnonzero(out[:, intensity_column] > 0)
    rng = np.random.default_rng() if rng is None else rng
    if valid.size > 1 and drop_frac_max > 0:
        n_drop = min(int(rng.uniform(0, drop_frac_max) * valid.size), valid.size - 1)
        if n_drop:
            weakest = valid[np.argsort(out[valid, intensity_column])[:n_drop]]
            out[weakest] = 0
            valid = np.flatnonzero(out[:, intensity_column] > 0)
    if valid.size:
        scales = rng.uniform(int_scale_min, int_scale_max, size=valid.size)
        out[valid, intensity_column] *= scales
        if feature_dim == 4:
            if width_normalization != "all_detected":
                out[valid, 0] *= rng.uniform(
                    width_scale_min, width_scale_max, size=valid.size
                )
                if width_normalization == "selected_topk":
                    maximum_width = out[valid, 0].max(initial=0.0)
                    if maximum_width > 0:
                        out[valid, 0] /= maximum_width
            out[valid, 1] *= scales
        maximum = out[valid, intensity_column].max(initial=0.0)
        if maximum > 0:
            out[valid, intensity_column] /= maximum
            if feature_dim == 4:
                out[valid, 1] /= maximum
                out[valid, 1] = np.minimum(out[valid, 1], out[valid, intensity_column])
        if jitter_deg > 0:
            two_theta = _two_theta_from_peak_positions(
                out[valid, position_column],
                coordinate=position,
                normalized=normalize_position,
                wavelength=wavelength,
                two_theta_min=two_theta_min,
                two_theta_max=two_theta_max,
            )
            two_theta += rng.uniform(-jitter_deg, jitter_deg, size=valid.size)
            out[valid, position_column] = _peak_positions(
                np.clip(two_theta, two_theta_min, two_theta_max),
                coordinate=position,
                normalized=normalize_position,
                wavelength=wavelength,
                two_theta_min=two_theta_min,
                two_theta_max=two_theta_max,
            )
    return out.ravel()


def augment_peak128_features(
    features: np.ndarray,
    *,
    jitter_deg: float = 0.1,
    int_scale_min: float = 0.8,
    int_scale_max: float = 1.2,
    width_scale_min: float = 0.7,
    width_scale_max: float = 1.5,
    drop_frac_max: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cheap train-only domain randomization of cached 128x4 peak descriptors."""
    out = np.asarray(features, dtype=np.float32).reshape(-1, 4).copy()
    valid = out[:, 2] > 0
    if rng is None:
        rng = np.random.default_rng()
    indices = np.flatnonzero(valid)
    if indices.size > 1 and drop_frac_max > 0:
        n_drop = min(
            int(rng.uniform(0, drop_frac_max) * indices.size), indices.size - 1
        )
        if n_drop:
            weakest = indices[np.argsort(out[indices, 2])[:n_drop]]
            out[weakest] = 0
            valid[weakest] = False
            indices = np.flatnonzero(valid)
    if indices.size:
        if jitter_deg > 0:
            two_theta = d_to_two_theta(out[indices, 3])
            two_theta += rng.uniform(-jitter_deg, jitter_deg, size=indices.size)
            out[indices, 3] = two_theta_to_d(two_theta)
        out[indices, 0] *= rng.uniform(width_scale_min, width_scale_max)
        scales = rng.uniform(int_scale_min, int_scale_max, size=indices.size)
        out[indices, 1:3] *= scales[:, None]
        maximum = out[indices, 2].max(initial=0)
        if maximum > 0:
            out[indices, 1:3] /= maximum
        out[indices, 1] = np.minimum(out[indices, 1], out[indices, 2])
    return out.ravel()


def augmented_peak128_feature_vector(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    jitter_deg: float = 0.1,
    int_scale_min: float = 0.8,
    int_scale_max: float = 1.2,
    width_scale_min: float = 0.7,
    width_scale_max: float = 1.5,
    drop_frac_max: float = 0.1,
    zero_shift_deg: float = 0.15,
    eta_min: float = 0.0,
    eta_max: float = 0.5,
    background_max: float = 0.2,
    noise_std_max: float = 0.02,
    min_prominence: float = 0.0,
    peak_count: int = 128,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Extract peak128 after experiment-like dense-profile randomization."""
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    tt, it = augment_peaks(
        two_theta,
        intensity,
        jitter_deg=jitter_deg,
        int_scale_min=int_scale_min,
        int_scale_max=int_scale_max,
        drop_frac_max=drop_frac_max,
        rng=rng,
    )
    if tt.size:
        tt = tt + rng.uniform(-zero_shift_deg, zero_shift_deg)
    width_scale = rng.uniform(width_scale_min, width_scale_max)
    profile = pseudo_voigt_profile(
        tt,
        it,
        caglioti_u=0.1 * width_scale**2,
        caglioti_v=0.01 * width_scale**2,
        caglioti_w=0.1 * width_scale**2,
        eta=rng.uniform(eta_min, eta_max),
    )
    x = np.linspace(-1.0, 1.0, profile.size, dtype=np.float32)
    if background_max > 0:
        background = rng.uniform(0, background_max) * (
            rng.uniform(0, 1) + rng.uniform(-0.5, 0.5) * x + rng.uniform(0, 0.5) * x**2
        )
        profile += np.clip(background, 0, None)
    if noise_std_max > 0:
        profile += rng.normal(0, rng.uniform(0, noise_std_max), size=profile.shape)
    profile = np.clip(profile, 0, None)
    maximum = profile.max(initial=0)
    if maximum > 0:
        profile /= maximum
    return xrdiff_feature_vector_from_profile(
        profile, "peaks_128x4", min_prominence=min_prominence, peak_count=peak_count
    )
