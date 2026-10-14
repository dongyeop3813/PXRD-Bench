"""Fixed source decoding and tensor serialization for selected IDs only."""
from __future__ import annotations
import bz2
import hashlib
import json
import math
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Element, Lattice, Structure
from src.data.mp20_tokens import lattice_to_Y
from src.data.pxrd_bench_schema import MAX_SITES, PXRD

MAX_ATOMIC_NUMBER = 94
PRIMITIVE_TOLERANCE = 0.01
_XRD_CALCULATOR = None
_PXRD_GRID = PXRD["two_theta_min"] + np.arange(PXRD["bins"]) * PXRD["bin_width"]

def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_and_primitive(structure: Structure) -> Structure:
    if not structure.is_ordered:
        raise ValueError("disordered_or_partially_occupied")
    if len(structure) < 1:
        raise ValueError("empty_source_structure")
    if not math.isfinite(float(structure.volume)) or structure.volume <= 1e-6:
        raise ValueError("invalid_cell_volume")
    structure = structure.get_primitive_structure(tolerance=PRIMITIVE_TOLERANCE)
    if not 1 <= len(structure) <= MAX_SITES:
        raise ValueError("primitive_site_count_out_of_range")
    atomic_numbers = [int(site.specie.Z) for site in structure]
    if not atomic_numbers or max(atomic_numbers) > MAX_ATOMIC_NUMBER:
        raise ValueError("unsupported_atomic_number")
    if not np.isfinite(structure.lattice.matrix).all():
        raise ValueError("non_finite_lattice")
    if not np.isfinite(structure.frac_coords).all():
        raise ValueError("non_finite_coordinates")
    return structure


def _tokenize(structure: Structure, nmax: int = MAX_SITES) -> dict[str, torch.Tensor | int]:
    n = len(structure)
    if n > nmax:
        raise ValueError(f"structure has {n} sites, exceeds nmax={nmax}")
    atom_types = np.asarray([site.specie.Z for site in structure], dtype=np.int64)
    a0 = np.zeros(nmax, dtype=np.int64)
    f1 = np.zeros((nmax, 3), dtype=np.float32)
    pad_mask = np.ones(nmax, dtype=np.bool_)
    a0[:n] = atom_types
    f1[:n] = np.asarray(structure.frac_coords, dtype=np.float32) % 1.0
    pad_mask[:n] = False
    return {
        "A0": torch.from_numpy(a0),
        "F1": torch.from_numpy(f1),
        "Y1": torch.from_numpy(
            lattice_to_Y(structure.lattice.lengths, structure.lattice.angles)
        ),
        "pad_mask": torch.from_numpy(pad_mask),
        "num_atoms": n,
    }


def simulate_pxrd(structure: Structure) -> torch.Tensor:
    global _XRD_CALCULATOR
    if _XRD_CALCULATOR is None:
        _XRD_CALCULATOR = XRDCalculator(wavelength=PXRD["wavelength"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pattern = _XRD_CALCULATOR.get_pattern(
            structure,
            two_theta_range=(PXRD["two_theta_min"], PXRD["two_theta_max"]),
        )
    profile = np.zeros(PXRD["bins"], dtype=np.float64)
    for position, intensity in zip(pattern.x, pattern.y, strict=True):
        theta = math.radians(float(position) / 2)
        tangent = math.tan(theta)
        fwhm = math.sqrt(
            max(
                PXRD["U"] * tangent**2 + PXRD["V"] * tangent + PXRD["W"],
                np.finfo(float).eps,
            )
        )
        delta = _PXRD_GRID - float(position)
        gaussian = np.exp(-4 * math.log(2) * delta**2 / fwhm**2)
        lorentzian = 1 / (1 + 4 * delta**2 / fwhm**2)
        profile += float(intensity) * (
            PXRD["eta"] * lorentzian + (1 - PXRD["eta"]) * gaussian
        )
    maximum = float(profile.max(initial=0))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("empty_or_non_finite_pxrd")
    return torch.from_numpy((profile / maximum).astype(np.float32))


def _iter_bz2_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the top-level Alexandria ``entries`` array using stdlib only."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            elif not buffer:
                break
            if not started:
                marker = buffer.find('"entries"')
                if marker < 0:
                    if chunk:
                        buffer = buffer[-32:]
                        continue
                    raise ValueError(f"missing entries array in {path}")
                opening = buffer.find("[", marker)
                if opening < 0:
                    if chunk:
                        continue
                    raise ValueError(f"unterminated entries array in {path}")
                buffer = buffer[opening + 1 :]
                started = True
            while True:
                buffer = buffer.lstrip(" \r\n\t,")
                if buffer.startswith("]"):
                    return
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"non-object Alexandria entry in {path}")
                yield value
                buffer = buffer[end:]
            if not chunk:
                raise ValueError(f"truncated JSON in {path}")
            if len(buffer) > 256 * 1024 * 1024:
                raise ValueError(f"single Alexandria entry is implausibly large in {path}")


def _build_tensor_item(task: tuple[str, str]) -> tuple[str, dict[str, Any], str | None]:
    material_id, structure_json = task
    try:
        structure = Structure.from_dict(json.loads(structure_json))
        item = _tokenize(structure)
        item["xrd"] = simulate_pxrd(structure)
        item = {
            key: value.numpy() if torch.is_tensor(value) else value
            for key, value in item.items()
        }
        return material_id, item, None
    except Exception as exc:
        return material_id, {}, f"{type(exc).__name__}: {exc}"


AFLOW_REQUIRED_FIELDS = (
    "auid",
    "species",
    "composition",
    "natoms",
    "geometry",
    "positions_fractional",
)


def _validate_aflow_record(record: dict[str, Any]) -> None:
    if any(record.get(field) is None for field in AFLOW_REQUIRED_FIELDS):
        raise ValueError("missing_or_null_required_field")
    auid = str(record["auid"])
    species = [str(value) for value in record["species"]]
    amounts = [float(value) for value in record["composition"]]
    if not species or len(species) != len(amounts) or len(set(species)) != len(species):
        raise ValueError("invalid_species_or_composition")
    if any(not math.isfinite(value) or value <= 0 for value in amounts):
        raise ValueError("invalid_composition_amount")
    if any(Element(symbol).Z > MAX_ATOMIC_NUMBER for symbol in species):
        raise ValueError("unsupported_atomic_number")
    natoms = int(record["natoms"])
    if natoms < 1:
        raise ValueError("source_site_count_out_of_range")
    rounded = [int(round(value)) for value in amounts]
    if any(not math.isclose(value, rounded_value, abs_tol=1e-6) for value, rounded_value in zip(amounts, rounded, strict=True)):
        raise ValueError("non_integer_composition")
    if sum(rounded) != natoms or len(record["positions_fractional"]) != natoms:
        raise ValueError("composition_site_count_mismatch")
    geometry = np.asarray(record["geometry"], dtype=float)
    if geometry.shape != (6,) or not np.isfinite(geometry).all():
        raise ValueError("invalid_geometry")
    if not auid:
        raise ValueError("missing_or_null_required_field")


def _structure_from_aflow_record(record: dict[str, Any]) -> Structure:
    species = [str(value) for value in record["species"]]
    counts = [int(round(float(value))) for value in record["composition"]]
    site_species = [symbol for symbol, count in zip(species, counts, strict=True) for _ in range(count)]
    coordinates = np.asarray(record["positions_fractional"], dtype=float)
    if len(site_species) != len(coordinates):
        raise ValueError("composition_site_count_mismatch")
    geometry = [float(value) for value in record["geometry"]]
    lattice = Lattice.from_parameters(*geometry)
    return Structure(lattice, site_species, coordinates, to_unit_cell=True)


