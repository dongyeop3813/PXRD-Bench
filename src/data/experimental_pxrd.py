"""Local CSV/CIF pairs or versioned opXRD ZIPs as Bragglite evaluation items."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure
from scipy.signal import savgol_filter
from torch.utils.data import Dataset

from src.data.mp20_tokens import lattice_to_Y
from src.pxrd.peaks import (
    peak_matrix_feature_vector_from_profile,
    peaks_to_di_tensors,
    peaks_to_pseudo_voigt_tensors,
    xrdiff_feature_vector_from_profile,
)


_ELEMENT = re.compile(r"^[A-Z][a-z]?")


def _decode(value):
    return json.loads(value) if isinstance(value, str) else value


def _canonicalize_wavelength(theta, intensity, source_wavelength, target_wavelength):
    """Map measured 2theta onto the checkpoint wavelength through d-spacing."""
    d_spacing = source_wavelength / (2 * np.sin(np.deg2rad(theta / 2)))
    argument = target_wavelength / (2 * d_spacing)
    valid = np.isfinite(argument) & (argument > 0) & (argument < 1)
    return np.rad2deg(2 * np.arcsin(argument[valid])), intensity[valid]


def _opxrd_records(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        if not names:
            raise ValueError(f"opXRD archive contains no JSON records: {path}")
        for name in names:
            payload = json.loads(archive.read(name))
            label = _decode(payload["label"])
            if (
                label.get("is_simulated") is not False
                or len(label.get("phases") or []) != 1
            ):
                raise ValueError(f"expected experimental single-phase opXRD record: {name}")
            phase = _decode(label["phases"][0])
            lattice = Lattice.from_parameters(*ast.literal_eval(phase["lattice"]))
            basis = [_decode(site) for site in _decode(phase["basis"])]
            species, coords = [], []
            for site in basis:
                match = _ELEMENT.match(site["symbol"])
                if not match:
                    raise ValueError(
                        f"invalid element symbol in {name}: {site['symbol']!r}"
                    )
                species.append(
                    {Element(match.group()): float(site.get("occupancy", 1.0))}
                )
                coords.append([float(site[key]) for key in ("x", "y", "z")])
            structure = Structure(lattice, species, coords).get_primitive_structure(
                tolerance=0.1
            )
            xray = _decode(label.get("xray_info") or {})
            wavelength = float(xray["primary_wavelength"])
            if not np.isfinite(wavelength) or wavelength <= 0:
                raise ValueError(f"invalid primary wavelength in {name}: {wavelength}")
            yield (
                name.removesuffix(".json"),
                structure,
                np.asarray(payload["two_theta_values"], dtype=float),
                np.asarray(payload["intensities"], dtype=float),
                wavelength,
            )


def frozen_phase_records(path: Path, *, split: str, nmax: int):
    """Read hashed separated profiles without relabeling them as measurements."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != "frozen_phase_pxrd_v1"
            or manifest.get("split") != split or manifest.get("nmax") != nmax):
        raise ValueError("frozen phase manifest schema/split/nmax mismatch")
    archive = path.parent / manifest["profiles"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["profiles_sha256"]:
        raise ValueError("frozen phase profile SHA-256 mismatch")
    with np.load(archive, allow_pickle=False) as arrays:
        theta, profiles = arrays["two_theta"], arrays["profiles"]
    rows = manifest["rows"]
    ids = [row["sample_id"] for row in rows]
    if (not rows or not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(rows)
            or theta.ndim != 1 or len(theta) < 2 or not np.all(np.diff(theta) > 0)
            or profiles.shape != (len(rows), len(theta))
            or not np.isfinite(theta).all() or not np.isfinite(profiles).all()
            or (profiles < 0).any() or not (profiles.max(axis=1) > 0).all()):
        raise ValueError("invalid frozen phase IDs or profile arrays")
    wavelength = float(manifest["wavelength_angstrom"])
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("invalid frozen phase wavelength")
    records = []
    for row, profile in zip(rows, profiles, strict=True):
        structure = Structure.from_dict(row["reference_structure"])
        if not structure.is_ordered or not 1 <= len(structure) <= nmax:
            raise ValueError("frozen phase reference must be ordered and within nmax")
        # Preserve the explicitly audited primitive cell, including its tolerance.
        records.append((row["sample_id"], structure, theta, profile, wavelength))
    return records, {**manifest, "manifest_sha256": hashlib.sha256(raw).hexdigest()}


def apply_diagnostic_features(dataset, path: str, expected_sha256: str, representation: str) -> None:
    """Replace declared conditioning only; preserve frozen structures and membership."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("diagnostic PXRD feature SHA-256 mismatch")
    payload = json.loads(raw)
    provenance = dataset.dataset_provenance
    parents = [(key, payload["parent_" + key]) for key in ("archive_sha256", "manifest_sha256")
               if "parent_" + key in payload]
    if (payload.get("schema_version") != 1 or representation != "peaks_128x4"
            or payload.get("representation") != representation
            or len(parents) != 1 or not parents[0][1]
            or parents[0][1] != provenance.get(parents[0][0])):
        raise ValueError("diagnostic PXRD features must identify the frozen parent and peaks_128x4 representation")
    ids = payload["sample_ids"]
    items = {item["mp_id"]: item for item in dataset.items}
    indices = {item["mp_id"]: index for index, item in enumerate(dataset.items)}
    if (not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids)
            or len(ids) != len(set(ids)) or not set(ids) <= items.keys()):
        raise ValueError("diagnostic PXRD target IDs must be unique members of the frozen split")
    features = torch.tensor(payload["features"], dtype=torch.float32)
    if (features.shape != (len(ids), 512) or not torch.isfinite(features).all()
            or (features < 0).any()):
        raise ValueError("diagnostic PXRD features must be finite nonnegative N x 512 arrays")
    peaks = features.reshape(len(ids), 128, 4)
    allow_empty = payload.get("allow_empty_pattern", False)
    if not isinstance(allow_empty, bool):
        raise ValueError("allow_empty_pattern must be a boolean")
    valid = peaks[:, :, 2].max(dim=1).values > 0
    if allow_empty:
        valid |= (features == 0).all(dim=1)
    if not valid.all():
        raise ValueError("diagnostic PXRD features must contain a nonzero peak per target")
    if ((peaks[:, :, 2] > 0) & (peaks[:, :, 3] <= 0)).any():
        raise ValueError("diagnostic PXRD peaks require positive d-spacing")
    if any((items[sample_id] if "pxrd_d" in items[sample_id] else dataset[indices[sample_id]])["pxrd_d"].shape != (512,) for sample_id in ids):
        raise ValueError("diagnostic PXRD feature shape differs from the native input")
    for sample_id, feature in zip(ids, features, strict=True):
        key = "_diagnostic_pxrd_d" if "_profile" in items[sample_id] else "pxrd_d"
        items[sample_id][key] = feature.clone()
    provenance["diagnostic_pxrd_features"] = {
        "path": str(path), "sha256": expected_sha256, "sample_ids": ids,
        "purpose": "post-hoc input diagnostic; not an experimental benchmark result",
    }
    if allow_empty:
        provenance["diagnostic_pxrd_features"]["allow_empty_pattern"] = True


class ExperimentalPXRDTokens(Dataset):
    def __init__(
        self,
        root: str,
        *,
        split: str = "test",
        nmax: int = 20,
        pxrd_representation: str,
        pxrd_max_peaks: int = 128,
        pxrd_peak_min_prominence: float = 0.0,
        pxrd_wavelength: float = 1.54184,
        pxrd_peak_features: str = "width_prominence_intensity_position",
        pxrd_peak_position: str = "d",
        pxrd_peak_position_normalized: bool = False,
        pxrd_peak_width_normalization: str = "none",
        pxrd_peak_selection: str = "prominence",
        pxrd_profile_step: float = 0.1,
        pxrd_caglioti_u: float = 0.1,
        pxrd_caglioti_v: float = 0.01,
        pxrd_caglioti_w: float = 0.1,
        pxrd_pseudo_voigt_eta: float = 0.1,
        background_subtraction: bool = True,
        background_order: int = 5,
        pxrd_dense_n_bins: int = 850,
        real_pxrd_peak_preprocessing: str = "official",
        with_pxrd: bool = False,
        **_ignored,
    ) -> None:
        self.pxrd_peak_min_prominence = float(pxrd_peak_min_prominence)
        if not 0.0 <= self.pxrd_peak_min_prominence <= 1.0:
            raise ValueError("pxrd_peak_min_prominence must be in [0, 1]")
        if self.pxrd_peak_min_prominence and pxrd_representation != "peaks_128x4":
            raise ValueError("pxrd_peak_min_prominence requires peaks_128x4")
        self.items = []
        target_wavelength = float(pxrd_wavelength)
        if not np.isfinite(target_wavelength) or target_wavelength <= 0:
            raise ValueError(f"invalid target PXRD wavelength: {pxrd_wavelength}")
        root_path = Path(root)
        manifest = None
        manifest_path = root_path.with_suffix(".manifest.json")
        if root_path.suffix == ".zip":
            if not manifest_path.exists():
                raise ValueError(
                    f"versioned opXRD archive requires sibling manifest: "
                    f"{manifest_path}"
                )
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            actual_sha256 = hashlib.sha256(root_path.read_bytes()).hexdigest()
            if actual_sha256 != manifest.get("archive_sha256"):
                raise ValueError(
                    f"opXRD archive SHA-256 does not match {manifest_path}"
                )
            try:
                evaluation = manifest["evaluation"]
                expected_records = int(manifest["records"])
                expected_nmax = int(evaluation["nmax"])
                expected_split = str(evaluation["split"])
                dataset_id = str(manifest["dataset_id"])
                supported_backend = str(evaluation["supported_backend"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid opXRD manifest: {manifest_path}") from exc
            if evaluation.get("dataset_class") != "experimental_pxrd":
                raise ValueError(
                    f"{dataset_id} requires dataset_class=experimental_pxrd"
                )
            if not supported_backend:
                raise ValueError(f"invalid opXRD manifest: {manifest_path}")
            if int(nmax) != expected_nmax:
                raise ValueError(
                    f"{dataset_id} requires nmax={expected_nmax}, got {nmax}"
                )
            if str(split) != expected_split:
                raise ValueError(
                    f"{dataset_id} requires split={expected_split!r}, got {split!r}"
                )
            self.dataset_provenance = {
                **manifest,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "profile_wavelength_canonicalization": {
                    "target_angstrom": target_wavelength,
                    "method": "measured 2theta -> d-spacing -> target 2theta before fixed-grid preprocessing",
                },
                "profile_background_subtraction": {
                    "enabled": bool(background_subtraction),
                    "polynomial_order": background_order if background_subtraction else None,
                },
            }

        def csv_records():
            for csv_path in sorted(root_path.glob("*.csv")):
                cif_path = csv_path.with_suffix(".cif")
                if not cif_path.exists():
                    continue
                raw = np.genfromtxt(csv_path, delimiter=",", names=True)
                yield (
                    csv_path.stem,
                    Structure.from_file(cif_path).get_primitive_structure(),
                    np.asarray(raw[raw.dtype.names[0]], dtype=float),
                    np.asarray(raw[raw.dtype.names[1]], dtype=float),
                    target_wavelength,
                )

        if root_path.suffix == ".json":
            records, provenance = frozen_phase_records(root_path, split=split, nmax=nmax)
            self.dataset_provenance = {
                **provenance,
                "profile_background_subtraction": {"enabled": bool(background_subtraction)},
                "profile_wavelength_canonicalization": {"target_angstrom": target_wavelength},
            }
        else:
            records = _opxrd_records(root_path) if root_path.suffix == ".zip" else csv_records()
        realpxrd_official = (
            real_pxrd_peak_preprocessing == "realpxrd_solver_official"
        )
        if realpxrd_official and pxrd_dense_n_bins != 8500:
            raise ValueError(
                "RealPXRD-Solver preprocessing requires pxrd_dense_n_bins=8500"
            )
        dense_profile = pxrd_representation == "dense_profile"
        n_bins = pxrd_dense_n_bins if dense_profile or realpxrd_official else 850
        grid = np.linspace(5.0, 90.0, n_bins, endpoint=False)
        self.pxrd_two_theta_grid = grid
        for mp_id, structure, theta, intensity, wavelength in records:
            if len(structure) > nmax:
                if manifest is not None:
                    raise ValueError(
                        f"{mp_id} has {len(structure)} atoms, exceeding frozen "
                        f"dataset nmax={nmax}"
                    )
                continue
            if theta.ndim != 1 or theta.shape != intensity.shape:
                raise ValueError(
                    f"invalid PXRD arrays for {mp_id}: "
                    f"{theta.shape} vs {intensity.shape}"
                )
            valid = (
                np.isfinite(theta)
                & np.isfinite(intensity)
                & (theta > 0)
                & (theta < 180)
            )
            theta, intensity = theta[valid], intensity[valid]
            theta, intensity = _canonicalize_wavelength(
                theta, intensity, wavelength, target_wavelength
            )
            wavelength = target_wavelength
            if not len(theta):
                raise ValueError(f"no finite PXRD points for {mp_id}")
            order = np.argsort(theta)
            theta, intensity = theta[order], intensity[order]
            profile = np.interp(grid, theta, intensity, left=0.0, right=0.0)
            if not realpxrd_official and len(profile) >= 7:
                profile = savgol_filter(profile, 7, 3)
            present = (grid >= theta.min()) & (grid <= theta.max())
            if (
                not realpxrd_official
                and background_subtraction
                and present.sum() > background_order
            ):
                coeff = np.polyfit(grid[present], profile[present], background_order)
                profile[present] -= np.polyval(coeff, grid[present])
            profile = np.clip(profile, 0, None)
            if profile.max() > 0:
                profile /= profile.max()
            if pxrd_representation == "legacy_di":
                from scipy.signal import find_peaks

                peak_indices, _ = find_peaks(profile, prominence=0.0)
                pxrd_d, pxrd_i, pxrd_pad = peaks_to_di_tensors(
                    grid[peak_indices],
                    profile[peak_indices],
                    max_peaks=pxrd_max_peaks,
                    wavelength=wavelength,
                )
                pxrd_d = torch.from_numpy(pxrd_d)
                pxrd_i = torch.from_numpy(pxrd_i)
                pxrd_pad = torch.from_numpy(pxrd_pad)
                extra_pxrd = {}
            elif dense_profile:
                pxrd_d = torch.from_numpy(profile.astype(np.float32))
                pxrd_i = torch.zeros(len(profile), dtype=torch.float32)
                pxrd_pad = torch.zeros(len(profile), dtype=torch.bool)
                extra_pxrd = {"pxrd_dense_profile": pxrd_d}
            elif pxrd_representation in {"broadened_basic", "broadened_shape"}:
                from scipy.signal import find_peaks

                peak_indices, _ = find_peaks(profile, prominence=0.0)
                position, peak_intensity, prominence, width, d_spacing, peak_pad = (
                    peaks_to_pseudo_voigt_tensors(
                        grid[peak_indices],
                        profile[peak_indices],
                        max_peaks=pxrd_max_peaks,
                        wavelength=wavelength,
                        two_theta_min=5.0,
                        two_theta_max=90.0,
                        step=pxrd_profile_step,
                        caglioti_u=pxrd_caglioti_u,
                        caglioti_v=pxrd_caglioti_v,
                        caglioti_w=pxrd_caglioti_w,
                        eta=pxrd_pseudo_voigt_eta,
                    )
                )
                pxrd_d = torch.from_numpy(d_spacing)
                pxrd_i = torch.from_numpy(peak_intensity)
                pxrd_pad = torch.from_numpy(peak_pad)
                extra_pxrd = {
                    "pxrd_position": torch.from_numpy(position),
                    "pxrd_intensity": torch.from_numpy(peak_intensity),
                    "pxrd_prominence": torch.from_numpy(prominence),
                    "pxrd_width": torch.from_numpy(width),
                }
            elif pxrd_representation == "peak_matrix":
                features = peak_matrix_feature_vector_from_profile(
                    profile.astype(np.float32),
                    max_peaks=pxrd_max_peaks,
                    features=pxrd_peak_features,
                    position=pxrd_peak_position,
                    normalize_position=pxrd_peak_position_normalized,
                    width_normalization=pxrd_peak_width_normalization,
                    selection=pxrd_peak_selection,
                    wavelength=wavelength,
                )
                pxrd_d = torch.from_numpy(features)
                pxrd_i = torch.zeros(len(features), dtype=torch.float32)
                pxrd_pad = torch.zeros(len(features), dtype=torch.bool)
                extra_pxrd = {}
            else:
                features = xrdiff_feature_vector_from_profile(
                    profile.astype(np.float32),
                    pxrd_representation,
                    wavelength=wavelength,
                    min_prominence=self.pxrd_peak_min_prominence,
                    peak_count=pxrd_max_peaks // 4 if pxrd_representation == "peaks_128x4" else 128,
                )
                pxrd_d = torch.from_numpy(features)
                pxrd_i = torch.zeros(len(features), dtype=torch.float32)
                pxrd_pad = torch.zeros(len(features), dtype=torch.bool)
                extra_pxrd = {}

            z = np.asarray(structure.atomic_numbers, dtype=np.int64)
            a0 = np.zeros(nmax, dtype=np.int64)
            frac = np.zeros((nmax, 3), dtype=np.float32)
            pad = np.ones(nmax, dtype=bool)
            a0[: len(z)] = z
            frac[: len(z)] = structure.frac_coords
            pad[: len(z)] = False
            self.items.append(
                {
                    "mp_id": mp_id,
                    "A0": torch.from_numpy(a0),
                    "F1": torch.from_numpy(frac),
                    "Y1": torch.from_numpy(
                        lattice_to_Y(structure.lattice.lengths, structure.lattice.angles)
                    ),
                    "pad_mask": torch.from_numpy(pad),
                    "num_atoms": len(z),
                    "pxrd_d": pxrd_d,
                    "pxrd_i": pxrd_i,
                    "pxrd_peak_pad_mask": pxrd_pad,
                    **(
                        {"xrd": torch.from_numpy(profile.astype(np.float32))}
                        if with_pxrd
                        else {}
                    ),
                    **extra_pxrd,
                }
            )
        if manifest is not None and len(self.items) != expected_records:
            raise ValueError(
                f"{dataset_id} manifest declares {expected_records} records, "
                f"but loaded {len(self.items)}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]
