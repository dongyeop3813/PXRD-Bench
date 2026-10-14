"""Resolve a frozen monitoring manifest by identity, never by row position."""
import json
from pathlib import Path

import torch

from src.utils.provenance import file_sha256


def fixed_monitor_indices(dataset, path, count):
    manifest = json.loads(Path(path).read_text())
    wanted = list(map(str, manifest["material_ids"]))
    if len(wanted) != count or len(set(wanted)) != count:
        raise ValueError("monitor manifest count/duplicate IDs mismatch")
    base = getattr(dataset, "dataset", dataset)
    if getattr(base, "storage_split", None) != manifest["storage_split"]:
        raise ValueError("monitor manifest is for another split")
    root = Path(base.root)
    split_root = root / "splits" if base.nmax == 208 else root / "views" / f"nmax{base.nmax}" / "splits"
    if file_sha256(split_root / f"{base.storage_split}.parquet") != manifest["split_sha256"]:
        raise ValueError("monitor source split checksum mismatch")
    ids = dataset.rows.material_id.astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("validation dataset has duplicate material IDs")
    lookup = {material_id: i for i, material_id in enumerate(ids)}
    if set(wanted) - lookup.keys():
        raise ValueError("monitor IDs missing from actual validation dataset")
    return torch.tensor([lookup[material_id] for material_id in wanted])
