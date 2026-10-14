from __future__ import annotations

from pathlib import Path

import torch
from src.data.mp20_tokens import VZ, tokens_to_structure
from src.models.type_encoding import TypeEncoding

def compute_dataset_element_distribution(dataset) -> torch.Tensor | None:
    counts = torch.zeros(VZ + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        item = dataset[i]
        atom_types = item["A0"]
        pad_mask = item["pad_mask"].bool()
        vals = atom_types[~pad_mask]
        if vals.numel() == 0:
            continue
        bincount = torch.bincount(vals, minlength=VZ + 1)
        counts += bincount.to(dtype=torch.float64)
    counts = counts[1 : VZ + 1]
    total = float(counts.sum().item())
    if total <= 0:
        return None
    return counts / total


def build_count_distribution(dataset, nmax: int) -> torch.Tensor:
    counts = torch.zeros(nmax + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        count = int(dataset[i]["num_atoms"])
        if 1 <= count <= nmax:
            counts[count] += 1
    if counts[1:].sum() == 0:
        raise ValueError("No valid num_atoms found for count distribution.")
    return counts[1:] / counts[1:].sum()

def dataset_to_structures(dataset) -> list:
    """Convert token dataset items to pymatgen Structures, skipping failures."""
    structs = []
    for i in range(len(dataset)):
        item = dataset[i]
        try:
            struct = tokens_to_structure(
                {
                    "A0": item["A0"],
                    "F1": item["F1"],
                    "Y1": item["Y1"],
                    "pad_mask": item["pad_mask"],
                }
            )
            structs.append(struct)
        except Exception:
            continue
    return structs

def ensure_dataset_splits(data_root: str | Path, dataset_name: str) -> bool:
    """Require already frozen paper splits; no automatic download or selection."""
    root = Path(data_root)
    if dataset_name == "mp20_polymorph_split":
        from src.data.mp20_polymorph_split import verify_prepared_dataset
        verify_prepared_dataset(root)
    for split in ("train", "val"):
        path = root / "raw" / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen split: {path}; provide the released dataset files")
    return True


def compute_allowed_elements(dataset) -> torch.Tensor | None:
    allowed = torch.zeros(VZ, dtype=torch.bool)
    for i in range(len(dataset)):
        item = dataset[i]
        atom_types = item["A0"]
        pad_mask = item["pad_mask"].bool()
        vals = atom_types[~pad_mask]
        if vals.numel() == 0:
            continue
        bincount = torch.bincount(vals, minlength=VZ + 1)
        allowed |= bincount[1 : VZ + 1].bool()
    if allowed.any():
        return allowed
    return None

def sample_input_stats(
    dataset, sample_size: int, type_encoding: TypeEncoding
) -> dict[str, torch.Tensor]:
    """
    Draw a random subset of dataset items and report mean/std of the
    normalized inputs that feed the model:
      - centered fractional coords (F1 - 0.5)
      - encoded atom types with padding removed
      - lattice encodings (Y1: log lengths, cos angles)
    """
    if sample_size <= 0 or len(dataset) == 0:
        return {}

    n = min(sample_size, len(dataset))
    idx = torch.randperm(len(dataset))[:n]

    type_sum = torch.zeros(type_encoding.type_dim, dtype=torch.float64)
    type_sq_sum = torch.zeros(type_encoding.type_dim, dtype=torch.float64)
    coord_sum = torch.zeros(3, dtype=torch.float64)
    coord_sq_sum = torch.zeros(3, dtype=torch.float64)
    lat_sum = torch.zeros(6, dtype=torch.float64)
    lat_sq_sum = torch.zeros(6, dtype=torch.float64)

    total_atoms = 0.0
    total_lat = 0.0

    for i in idx:
        item = dataset[int(i)]
        pad_mask = item["pad_mask"].bool()
        real_mask = ~pad_mask
        if real_mask.any():
            type_oh = type_encoding.encode_from_A0(
                a0=item["A0"].unsqueeze(0), pad_mask=pad_mask.unsqueeze(0)
            ).squeeze(0)
            type_oh = type_oh.to(dtype=torch.float64)
            # Keep only real atoms for stats.
            type_oh = type_oh[real_mask]
            type_sum += type_oh.sum(dim=0)
            type_sq_sum += (type_oh**2).sum(dim=0)

            frac_c = (item["F1"] - 0.5).to(dtype=torch.float64)
            frac_c = frac_c[real_mask]
            coord_sum += frac_c.sum(dim=0)
            coord_sq_sum += (frac_c**2).sum(dim=0)
            total_atoms += float(real_mask.sum().item())

        lat = item["Y1"].to(dtype=torch.float64)
        lat_sum += lat
        lat_sq_sum += lat**2
        total_lat += 1.0

    stats = {}
    if total_atoms > 0:
        stats["type_mean"] = type_sum / total_atoms
        stats["type_std"] = torch.sqrt(
            type_sq_sum / total_atoms - stats["type_mean"] ** 2 + 1e-12
        )
        stats["frac_mean"] = coord_sum / total_atoms
        stats["frac_std"] = torch.sqrt(
            coord_sq_sum / total_atoms - stats["frac_mean"] ** 2 + 1e-12
        )
    if total_lat > 0:
        stats["lat_mean"] = lat_sum / total_lat
        stats["lat_std"] = torch.sqrt(
            lat_sq_sum / total_lat - stats["lat_mean"] ** 2 + 1e-12
        )
    return stats
