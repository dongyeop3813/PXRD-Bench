"""Fast raw Bragg simulation compatible with pymatgen's XRDCalculator."""

from __future__ import annotations

import math

import numpy as np
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS


def fast_xrd_pattern(
    structure,
    *,
    wavelength: float,
    two_theta_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return only the peak positions/intensities needed by the Bragg cache."""
    wavelength = float(wavelength)
    tt_min, tt_max = map(float, two_theta_range)
    min_r, max_r = (
        2 * math.sin(math.radians(value / 2)) / wavelength
        for value in (tt_min, tt_max)
    )
    reciprocal = structure.lattice.reciprocal_lattice_crystallographic
    points = sorted(
        (
            point
            for point in reciprocal.get_points_in_sphere(
                [[0, 0, 0]], [0, 0, 0], max_r
            )
            if point[1] >= min_r and point[1] != 0
        ),
        key=lambda point: (
            point[1],
            -point[0][0],
            -point[0][1],
            -point[0][2],
        ),
    )
    if not points:
        raise ValueError("empty diffraction pattern")

    hkls = np.rint([point[0] for point in points]).astype(np.int64)
    g_hkls = np.asarray([point[1] for point in points], dtype=np.float64)
    theta = np.arcsin(wavelength * g_hkls / 2)
    s2 = np.square(g_hkls / 2)

    zs: list[int] = []
    coeffs: list[list[list[float]]] = []
    frac_coords: list[np.ndarray] = []
    occupancies: list[float] = []
    for site in structure:
        for specie, occupancy in site.species.items():
            zs.append(specie.Z)
            try:
                coeffs.append(ATOMIC_SCATTERING_PARAMS[specie.symbol])
            except KeyError as exc:
                raise ValueError(
                    f"missing XRD scattering coefficients for {specie.symbol}"
                ) from exc
            frac_coords.append(site.frac_coords)
            occupancies.append(float(occupancy))

    zs_array = np.asarray(zs, dtype=np.float64)
    coeffs_array = np.asarray(coeffs, dtype=np.float64)
    coords_array = np.asarray(frac_coords, dtype=np.float64)
    occupancy_array = np.asarray(occupancies, dtype=np.float64)
    intensities = np.empty(len(hkls), dtype=np.float64)

    # Bound the temporary hkl x atom x coefficient arrays for unusually large cells.
    for start in range(0, len(hkls), 4096):
        stop = min(start + 4096, len(hkls))
        chunk_s2 = s2[start:stop]
        scattering = zs_array[None, :] - 41.78214 * chunk_s2[:, None] * np.sum(
            coeffs_array[None, :, :, 0]
            * np.exp(-coeffs_array[None, :, :, 1] * chunk_s2[:, None, None]),
            axis=2,
        )
        phase = np.exp(
            2j * np.pi * (hkls[start:stop] @ coords_array.T)
        )
        structure_factor = np.sum(
            scattering * occupancy_array[None, :] * phase, axis=1
        )
        chunk_theta = theta[start:stop]
        lorentz = (1 + np.square(np.cos(2 * chunk_theta))) / (
            np.square(np.sin(chunk_theta)) * np.cos(chunk_theta)
        )
        intensities[start:stop] = np.square(np.abs(structure_factor)) * lorentz

    two_theta = np.degrees(2 * theta)
    grouped_x: list[float] = []
    grouped_y: list[float] = []
    tolerance = AbstractDiffractionPatternCalculator.TWO_THETA_TOL
    for position, intensity in zip(two_theta, intensities, strict=True):
        if grouped_x and abs(grouped_x[-1] - position) < tolerance:
            grouped_y[-1] += float(intensity)
        else:
            grouped_x.append(float(position))
            grouped_y.append(float(intensity))

    y = np.asarray(grouped_y, dtype=np.float64)
    maximum = float(y.max(initial=0))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("empty or non-finite diffraction pattern")
    y *= 100 / maximum
    keep = y > AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL
    return np.asarray(grouped_x, dtype=np.float64)[keep], y[keep]
