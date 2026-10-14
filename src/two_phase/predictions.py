"""Cache native ordered XDecomposer slots for two-phase reconstruction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.data.mp20_tokens import MP20Tokens
from src.two_phase.data import MixturePhasePXRDDataset
from src.two_phase.xdecomposer_adapter import (
    XDecomposerVariant,
    load_official_model,
    profile_to_peak128x4,
    resample_to_xdecomposer,
)
from src.two_phase.xdecomposer_utils import _atomic_torch_save, _shared_element_mask
from src.utils.provenance import file_sha256

SCHEMA = "xdecomposer_frozen_slots_v1"


def checked_path(record: dict) -> Path:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_protocol(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "pxrd_bench_two_phase_v1":
        raise ValueError("expected pxrd_bench_two_phase_v1 pipeline config")
    expected = {
        "num_candidates": 20,
        "topk": [1, 20],
        "num_steps": 150,
        "solver": "heun",
        "rng_mode": "stateless_batch",
        "weights": "ema",
    }
    if config["sampling"] != expected:
        raise ValueError("ordered recovery requires K20/Heun150/EMA")
    if (
        config["scoring"]["matcher_profile"] != "strict"
        or config["scoring"]["match_gate"] != "matcher_only"
    ):
        raise ValueError("strict matcher-only scoring is required")
    if config["test_used"]:
        raise ValueError("this protocol is validation-only")
    if not Path(config["output_root"]).is_absolute():
        raise ValueError("output_root must be absolute")
    config["protocol_sha256"] = file_sha256(path)
    return config


def read_cache(config: dict, arm: str, split: str) -> dict:
    if arm != "conservation" or split != "val":
        raise ValueError(
            "only ordered conservation validation predictions are released"
        )
    path = Path(config["output_root"]) / arm / f"{split}_slots.pt"
    manifest = json.loads(path.with_suffix(".json").read_text())
    if (
        manifest["status"] != "complete"
        or file_sha256(path) != manifest["cache_sha256"]
    ):
        raise ValueError("cache is incomplete or has changed")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if (
        cache["schema_version"] != SCHEMA
        or cache["protocol_sha256"] != config["protocol_sha256"]
        or cache["arm"] != arm
        or cache["split"] != split
    ):
        raise ValueError("cache protocol mismatch")
    return cache


@torch.inference_mode()
def cache_slots(config: dict, device: torch.device) -> Path:
    """Predict slots 0/1 directly; references supply labels and counts only."""
    output = Path(config["output_root"]) / "conservation/val_slots.pt"
    if output.exists():
        raise FileExistsError(output)
    for record in config["data"]["val"]["files"]:
        checked_path(record)
    source = Path(config["official_source"])
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["official_commit"]:
        raise ValueError("official source commit mismatch")
    backbone, _ = load_official_model(
        source,
        checked_path(config["release"]),
        checked_path(config["mae"]),
        device=device,
    )
    payload = torch.load(
        checked_path(config["decomposition_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    options = payload["config"].get("model_options", {})
    if options != {"shared_elements": False, "normalize_masks": True}:
        raise ValueError("expected the ordered conservation checkpoint")
    model = XDecomposerVariant(backbone, **options).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().requires_grad_(False)
    dataset = MixturePhasePXRDDataset(config["source_root"], split="val")
    indices = [
        i
        for i, value in enumerate(dataset.rows["fraction_b"].tolist())
        if abs(float(value) - 0.5) > 1e-12
    ]
    if len(indices) != config["data"]["val"]["num_mixtures"]:
        raise ValueError("frozen non-tie coverage changed")
    reference = MP20Tokens(root=config["reference_root"], split="val", nmax=20)
    counts_by_id = {
        str(item["mp_id"]): int(item["num_atoms"]) for item in reference.items
    }
    rows, peaks, shared, counts = [], [], [], []
    batch_size = int(config["cache_batch_size"])
    for start in range(0, len(indices), batch_size):
        items = [dataset[i] for i in indices[start : start + batch_size]]
        mixture = torch.from_numpy(
            np.stack(
                [resample_to_xdecomposer(item["mixture_xrd"].numpy()) for item in items]
            )
        ).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            profiles, _ = model(mixture[:, None])
        if not torch.isfinite(profiles).all():
            raise ValueError("nonfinite decomposition")
        slots = profiles[:, :2].float().cpu().numpy()
        peaks.append(
            torch.from_numpy(
                np.stack([profile_to_peak128x4(p) for p in slots.reshape(-1, 3500)])
            ).reshape(len(items), 2, -1)
        )
        for item in items:
            dominant = int(float(item["fraction_b"]) > 0.5)
            phase_ids = [str(item["phase_mp_ids"][i]) for i in (dominant, 1 - dominant)]
            rows.append(
                {
                    "parent_sample_id": str(item["sample_id"]),
                    "phase_ids_dominant_minor": phase_ids,
                    "chemical_system": item["chemical_system"],
                    "fraction_b": float(item["fraction_b"]),
                }
            )
            shared.append(_shared_element_mask(item["chemical_system"]))
            counts.append([counts_by_id[p] for p in phase_ids])
    cache = {
        "schema_version": SCHEMA,
        "protocol_sha256": config["protocol_sha256"],
        "arm": "conservation",
        "split": "val",
        "rows": rows,
        "peaks": torch.cat(peaks),
        "shared": torch.stack(shared),
        "target_counts": torch.tensor(counts),
        "oracle_slot_alignment": False,
    }
    _atomic_torch_save(cache, output)
    write_json(
        output.with_suffix(".json"),
        {"status": "complete", "cache_sha256": file_sha256(output)},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache_slots(load_protocol(args.protocol), torch.device(args.device))


if __name__ == "__main__":
    main()
