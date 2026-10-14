"""Material-ID contracts shared by training monitors and dataset adapters."""
import hashlib
import json
from pathlib import Path

from torch.utils.data import Subset


def ordered_ids_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def material_ids(dataset):
    if isinstance(dataset, Subset):
        source = material_ids(dataset.dataset)
        ids = [source[int(i)] for i in dataset.indices]
    elif getattr(dataset, "rows", None) is not None:
        ids = dataset.rows["material_id"].astype(str).tolist()
    elif hasattr(dataset, "items"):
        ids = [str(row["mp_id"]) for row in dataset.items]
    elif hasattr(dataset, "dataset"):
        ids = material_ids(dataset.dataset)
    else:
        ids = [str(dataset[i]["mp_id"]) for i in range(len(dataset))]
    if len(ids) != len(dataset) or len(set(ids)) != len(ids):
        raise ValueError("dataset material IDs must be unique and cover every row")
    return ids


def manifest_indices(dataset, path):
    payload = json.loads(Path(path).read_text())
    ids = payload["ids"]
    if not ids or any(not isinstance(value, str) for value in ids):
        raise ValueError("manifest requires nonempty string material IDs")
    if len(set(ids)) != len(ids) or payload.get("count", len(ids)) != len(ids):
        raise ValueError("manifest count or uniqueness mismatch")
    if payload.get("ordered_ids_sha256", ordered_ids_sha256(ids)) != ordered_ids_sha256(ids):
        raise ValueError("manifest ordered ID hash mismatch")
    lookup = {value: i for i, value in enumerate(material_ids(dataset))}
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(f"manifest IDs missing from dataset: {sorted(missing)[:5]}")
    return [lookup[value] for value in ids]
