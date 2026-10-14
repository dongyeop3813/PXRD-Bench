"""Transport-independent integrity checks for the frozen PXRD-Bench release."""
from __future__ import annotations

import hashlib
import gzip
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element

from src.data.mp20_tokens import tokens_to_structure
from src.data.pxrd_bench_schema import PXRD, SPLITS
from src.data.structure_cache import encode_structure

TENSOR_KEYS = ("A0", "F1", "Y1", "pad_mask", "num_atoms", "xrd")
IDENTITY_COLUMNS = ("material_id", "source", "source_id", "source_release")
METADATA_COLUMNS = (*IDENTITY_COLUMNS, "energy_above_hull", "energy_filter_basis",
                    "composition_key", "prototype_id", "leakage_group_id",
                    "num_sites", "structure_entity_id")
HASH_COLUMNS = ("primitive_sha256", "tensors_sha256", "canonical_sha256", "bragg_sha256")
REGISTRY_COLUMNS = (*METADATA_COLUMNS, "source_resource", *HASH_COLUMNS)
PROCESSING_PACKAGES = ("numpy", "scipy", "pymatgen", "spglib", "torch")
_ELEMENT_NUMBERS = {Element.from_Z(z).symbol: z for z in range(1, 95)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def primitive_sha256(structure) -> str:
    # Exclude serializer version/site properties; preserve cell, order and precision.
    return json_sha256({"Z": list(map(int, structure.atomic_numbers)),
                       "lattice": structure.lattice.matrix.tolist(),
                       "frac_coords": structure.frac_coords.tolist()})


def primitive_json_sha256(content: str) -> str:
    """Same hash directly from the ordered representative's frozen serialization."""
    value = json.loads(content)
    return json_sha256({"Z": [_ELEMENT_NUMBERS[site["species"][0]["element"]] for site in value["sites"]],
                       "lattice": value["lattice"]["matrix"],
                       "frac_coords": [site["abc"] for site in value["sites"]]})


def array_sha256(values) -> str:
    digest = hashlib.sha256()
    for key, value in values:
        array = np.asarray(value)
        metadata = json.dumps([key, array.dtype.str, list(array.shape)]).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def tensor_sha256(item, *, nmax=52) -> str:
    values = []
    for key in TENSOR_KEYS:
        value = torch.as_tensor(item[key]).detach().cpu().numpy()
        if key in ("A0", "F1", "pad_mask"):
            value = value[:nmax]
        values.append((key, value))
    return array_sha256(values)


def bragg_sha256(two_theta, intensity) -> str:
    return array_sha256((('two_theta', np.asarray(two_theta, dtype=np.float64)),
                         ('intensity', np.asarray(intensity, dtype=np.float64))))


def canonical_bytes(item) -> bytes:
    structure = tokens_to_structure(item).get_primitive_structure(tolerance=0.01)
    return encode_structure(structure.get_reduced_structure(reduction_algo="niggli"))


def membership_sha256(ids) -> str:
    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative release path: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes release root: {relative}")
    return result


def validate_catalog(catalog: dict) -> None:
    versions = catalog.get("processing_versions", {})
    if set(PROCESSING_PACKAGES) - versions.keys():
        raise ValueError("Source catalog must freeze all processing package versions")
    for name, source in catalog["sources"].items():
        for key in ("source_release", "license_url", "attribution"):
            if not source.get(key):
                raise ValueError(f"Missing {name}.{key}")
        if name in ("materials_project", "aflow"):
            continue
        for key, resource in source["resources"].items():
            for required in ("url", "sha256", "filename"):
                if not resource.get(required):
                    raise ValueError(f"Missing {name}.{key}.{required}")
            digest = resource["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid resource hash: {name}/{key}")
            contained_path(Path("."), resource["filename"])


def check_processing_versions(catalog: dict) -> None:
    for package, expected in catalog["processing_versions"].items():
        observed = version(package)
        # CPU/CUDA local wheel tags do not change the stated PyTorch release.
        if observed.split("+")[0] != expected.split("+")[0]:
            raise ValueError(f"Processing dependency differs: {package}={observed}, expected {expected}")


def read_registry(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("dataset_id") != "pxrd_bench"
            or manifest.get("schema_version") != 2
            or manifest.get("distribution") != "references"
            or manifest.get("final_curated_splits") is not True
            or set(manifest.get("splits", {})) != set(SPLITS)
            or manifest.get("pxrd") != PXRD):
        raise ValueError("Expected a complete frozen schema-v2 PXRD-Bench reference release")
    catalog_path = root / "sources.json"
    if file_sha256(catalog_path) != manifest["source_catalog_sha256"]:
        raise ValueError("Source catalog checksum differs")
    catalog = json.loads(catalog_path.read_text())
    validate_catalog(catalog)
    return manifest, catalog


def read_split(root: Path, manifest: dict, split: str) -> pd.DataFrame:
    contract = manifest["splits"][split]
    parts = contract.get("files") or [contract]
    frames = []
    for part in parts:
        path = contained_path(root, part["file"])
        if file_sha256(path) != part["sha256"]:
            raise ValueError(f"Registry file checksum differs: {split}/{path.name}")
        chunk = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "rows" in part and len(chunk) != part["rows"]:
            raise ValueError(f"Registry fragment count differs: {split}/{path.name}")
        frames.append(chunk)
    frame = pd.concat(frames, ignore_index=True)
    if set(REGISTRY_COLUMNS) - set(frame):
        raise ValueError(f"Registry is missing required fields: {split}")
    if len(frame) != contract["rows"] or frame.material_id.duplicated().any():
        raise ValueError(f"Registry count/uniqueness differs: {split}")
    if membership_sha256(frame.material_id) != contract["ordered_membership_sha256"]:
        raise ValueError(f"Registry membership/order differs: {split}")
    if "id_list" in contract:
        spec = contract["id_list"]
        path = contained_path(root, spec["file"])
        if file_sha256(path) != spec["sha256"] or spec["rows"] != len(frame):
            raise ValueError(f"Ordered ID list checksum/count differs: {split}")
        with gzip.open(path, "rb") as stream:
            digest = hashlib.sha256()
            count = 0
            for line in stream:
                digest.update(line)
                count += 1
        if count != len(frame) or digest.hexdigest() != contract["ordered_membership_sha256"]:
            raise ValueError(f"Ordered ID list membership differs: {split}")
    for column in (*IDENTITY_COLUMNS, "source_resource", *HASH_COLUMNS):
        if frame[column].eq("").any():
            raise ValueError(f"Empty {column} in {split}")
    for column in HASH_COLUMNS:
        if not frame[column].str.fullmatch("[0-9a-f]{64}").all():
            raise ValueError(f"Invalid {column} in {split}")
    return frame
