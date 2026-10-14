"""Ordered recovery with shared mixture elements and oracle atom counts."""

from __future__ import annotations
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import torch
from src.data.mp20_tokens import MP20Tokens, VZ
from src.eval_util.checkpoint_runtime import (
    build_model_from_checkpoint,
    checkpoint_model_args,
    checkpoint_type_encoding_name,
    load_checkpoint,
    maybe_apply_checkpoint_ema,
)
from src.eval_util.prediction_cache import iter_prediction_cache, write_prediction_cache
from src.eval_util.run_artifacts import structure_record_from_tokens
from src.models.type_encoding import build_type_encoding
from src.pxrd.eval_metrics import score_prediction_row
from src.pxrd.eval_task_runtime import _sample_items, _sampler_config
from src.two_phase.predictions import (
    checked_path,
    load_protocol,
    read_cache,
    write_json,
)
from src.utils.provenance import file_sha256
from src.utils.seeding import stable_seed


class RecoverySlots:
    """Only slot PXRD, element masks and atom counts reach the sampler."""

    def __init__(
        self, peaks: torch.Tensor, elements: torch.Tensor, num_atoms: torch.Tensor
    ):
        if elements.shape == (len(peaks), VZ):
            elements = elements[:, None, :].expand(-1, 2, -1)
        if (
            peaks.ndim != 3
            or peaks.shape[1:] != (2, 512)
            or peaks.shape[:2] != num_atoms.shape
            or num_atoms.ndim != 2
            or num_atoms.shape[1] != 2
            or (elements.shape != (len(peaks), 2, VZ))
            or (not torch.isfinite(peaks).all())
            or (not elements.any(-1).all())
            or (not ((num_atoms >= 1) & (num_atoms <= 20)).all())
            or (not (num_atoms == num_atoms.long()).all())
        ):
            raise ValueError("invalid slot conditions")
        if not torch.equal(elements[:, 0], elements[:, 1]):
            raise ValueError("both phases must use the same mixture element superset")
        self.peaks = peaks.flatten(0, 1).float()
        self.elements = elements.flatten(0, 1).bool()
        self.num_atoms = num_atoms.flatten().long()

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, index):
        peaks = self.peaks[index]
        return {
            "A0": torch.zeros(20, dtype=torch.long),
            "pad_mask": torch.arange(20) >= self.num_atoms[index],
            "elem_set_mask": self.elements[index],
            "pxrd_d": peaks,
            "pxrd_i": torch.zeros_like(peaks),
            "pxrd_peak_pad_mask": torch.zeros_like(peaks, dtype=torch.bool),
        }


def reference_records(config: dict, cache: dict) -> dict:
    reference = MP20Tokens(root=config["reference_root"], split="val", nmax=20)
    by_id = {str(item["mp_id"]): i for i, item in enumerate(reference.items)}
    return {
        p: structure_record_from_tokens(reference[by_id[p]], by_id[p])
        for p in {p for row in cache["rows"] for p in row["phase_ids_dominant_minor"]}
    }


def oracle_conditions(
    cache: dict, references: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference role supplies N + unique elements; never align PXRD slots."""
    counts = torch.zeros(len(cache["rows"]), 2, dtype=torch.long)
    elements = torch.zeros(len(cache["rows"]), 2, VZ, dtype=torch.bool)
    for i, row in enumerate(cache["rows"]):
        if len(row["phase_ids_dominant_minor"]) != 2:
            raise ValueError("expected dominant and minor phase IDs")
        for slot, phase_id in enumerate(row["phase_ids_dominant_minor"]):
            atoms = torch.as_tensor(references[phase_id]["atom_types"])
            if (
                atoms.ndim != 1
                or not 1 <= len(atoms) <= 20
                or (not ((atoms >= 1) & (atoms <= VZ)).all())
                or (not (atoms == atoms.long()).all())
            ):
                raise ValueError(f"invalid oracle atoms: {phase_id}")
            counts[i, slot] = len(atoms)
            elements[i, slot, atoms.long() - 1] = True
    if not torch.equal(counts, cache["target_counts"].long()):
        raise ValueError("oracle reference counts disagree with frozen role order")
    return (counts, elements)


def score_mixture(task: tuple) -> dict:
    parent, references, slot_candidates = task
    if (
        len(references) != 2
        or len(slot_candidates) != 2
        or any((len(c) != 20 for c in slot_candidates))
    ):
        raise ValueError(
            "all-phase recovery requires two distinct slots with K=20 each"
        )
    pairs = [
        [
            score_prediction_row(
                {
                    "sample_id": f"{parent}::slot{s}::ref{r}",
                    "reference": references[r],
                    "candidates": slot_candidates[s],
                },
                topk=[1, 20],
                matcher_profile="strict",
                match_gate="matcher_only",
                pdd_k=100,
                include_formula=False,
                match_only=True,
            )
            for r in range(2)
        ]
        for s in range(2)
    ]
    result = {"sample_id": parent, "topk": {}}
    for k in (1, 20):
        result["topk"][str(k)] = {}
        for gate in ("matcher_only",):
            edge = [
                [pairs[s][r]["topk"][str(k)]["matched_by_gate"][gate] for r in range(2)]
                for s in range(2)
            ]
            result["topk"][str(k)][gate] = {
                "dominant": bool(edge[0][0]),
                "minor": bool(edge[1][1]),
                "ordered_all_phase": bool(edge[0][0] and edge[1][1]),
                "all_phase": bool(
                    edge[0][0] and edge[1][1] or (edge[0][1] and edge[1][0])
                ),
                "phase_fraction": max(
                    int(edge[0][0]) + int(edge[1][1]), int(edge[0][1]) + int(edge[1][0])
                )
                / 2,
            }
    result["pairwise_scores"] = pairs
    return result


def score_cache(output: Path, config: dict, cache: dict) -> dict:
    references = reference_records(config, cache)
    write_json(output / "references.json", references)

    def tasks():
        predictions = iter(iter_prediction_cache(output / "samples"))
        for row in cache["rows"]:
            slots = [next(predictions), next(predictions)]
            parent = row["parent_sample_id"]
            if [item["sample_id"] for item in slots] != [
                f"{parent}::slot0",
                f"{parent}::slot1",
            ]:
                raise ValueError("slot coverage/order mismatch")
            refs = [references[p] for p in row["phase_ids_dominant_minor"]]
            yield (parent, refs, [item["candidates"] for item in slots])
        if next(predictions, None) is not None:
            raise ValueError("unexpected extra prediction rows")

    totals = {
        f"{prefix}{kind}_match_rate@{k}": 0.0
        for prefix in ("",)
        for kind in ("all_phase", "phase", "dominant", "minor", "ordered_all_phase")
        for k in (1, 20)
    }
    output.joinpath("metrics").mkdir(exist_ok=True)
    count = 0
    with (
        ProcessPoolExecutor(
            max_workers=config["scoring"]["workers"],
            mp_context=multiprocessing.get_context("spawn"),
            initializer=torch.set_num_threads,
            initargs=(1,),
        ) as pool,
        (output / "metrics/per_mixture.jsonl").open("w") as handle,
    ):
        pending = tasks()

        def results():
            while batch := list(islice(pending, 128)):
                yield from pool.map(score_mixture, batch, chunksize=8)

        for result in results():
            handle.write(json.dumps(result) + "\n")
            count += 1
            for k in (1, 20):
                for gate, prefix in (("matcher_only", ""),):
                    entry = result["topk"][str(k)][gate]
                    totals[f"{prefix}all_phase_match_rate@{k}"] += entry["all_phase"]
                    totals[f"{prefix}phase_match_rate@{k}"] += entry["phase_fraction"]
                    for kind in ("dominant", "minor", "ordered_all_phase"):
                        totals[f"{prefix}{kind}_match_rate@{k}"] += entry[kind]
            if count % 256 == 0:
                print(f"[scoring] {count}/{len(cache['rows'])}", flush=True)
    if count != len(cache["rows"]) or count != config["data"]["val"]["num_mixtures"]:
        raise ValueError("incomplete full-validation coverage")
    result = {
        "num_mixtures": count,
        "num_slots": 2 * count,
        "num_candidates": 40 * count,
        **{key: value / count for key, value in totals.items()},
    }
    write_json(output / "metrics/metrics.json", result)
    return result


def recover(
    config: dict, arm: str, device: torch.device, *, output_root: Path | None = None
) -> dict:
    if config["implementation_smoke"]:
        raise ValueError("recovery cannot use an implementation-smoke protocol")
    os.environ["BRAGGLITE_FAST_SAMPLER"] = "1"
    cache = read_cache(config, arm, "val")
    if len(cache["rows"]) != config["data"]["val"]["num_mixtures"]:
        raise ValueError("recovery requires the frozen full validation set")
    if output_root is not None and (
        not output_root.is_absolute() or "/worktree/" in str(output_root.resolve())
    ):
        raise ValueError("outputs must be absolute and outside worktrees")
    if (
        output_root is None
        or output_root.resolve() == Path(config["output_root"]).resolve()
    ):
        raise ValueError("recovery requires a separate output root")
    output = Path(output_root or config["output_root"]) / arm / "recovery"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    elements = cache["shared"]
    if cache.get("oracle_slot_alignment", False):
        raise ValueError("reference-aligned slots are outside ordered recovery")
    counts, _ = oracle_conditions(cache, reference_records(config, cache))
    dataset = RecoverySlots(cache["peaks"], elements, counts)
    checkpoint_path = checked_path(config["structure_checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_args = checkpoint_model_args(checkpoint)
    if (
        not model_args.get("element_set_cond")
        or not model_args.get("pxrd_cond")
        or (model_args.get("pxrd_representation") != "peaks_128x4")
        or (int(model_args.get("nmax", 20)) != 20)
    ):
        raise ValueError(
            "checkpoint must be PXRD + element-set only, peaks_128x4, Nmax20"
        )
    model, _ = build_model_from_checkpoint(checkpoint, device=device)
    maybe_apply_checkpoint_ema(model, checkpoint, "ema", log_prefix="[blind-recovery]")
    model.eval()
    encoding = build_type_encoding(
        checkpoint_type_encoding_name(checkpoint, model_args), vz=VZ
    )
    del checkpoint
    if config["runtime"]["compile"]:
        model = torch.compile(model, mode=config["runtime"]["compile_mode"])
    sampler = _sampler_config(
        SimpleNamespace(
            sample_num_steps=150, solver="heun", bf16=True, rng_mode="stateless_batch"
        ),
        model_args,
    )
    metadata = {
        "schema_version": "xdecomposer_fixed_slot_recovery_v2",
        "status": "running",
        "protocol": config,
        "output_root": str(output_root or config["output_root"]),
        "arm": arm,
        "split": "val",
        "slot_order": ["dominant", "minor"],
        "oracle_num_atoms": True,
        "primary_metric": "ordered_all_phase_match_rate@20",
        "oracle_phase_elements": False,
        "oracle_slot_alignment": bool(cache.get("oracle_slot_alignment", False)),
        "element_set_source": "shared_mixture_chemical_system",
        "atom_count_source": "oracle_phase_num_atoms",
        "candidate_order": "seeded_generation_order_no_reranker",
        "num_candidates_per_slot": 20,
        "checkpoint_sha256": config["structure_checkpoint"]["sha256"],
        "sampler": {
            k: v
            for k, v in vars(sampler).items()
            if isinstance(v, (str, bool, int, float, type(None)))
            and (not (isinstance(v, float) and (not torch.isfinite(torch.tensor(v)))))
        },
        "conditioning_shift": config.get(
            "conditioning_shift", "fine-tuned with the shared mixture element superset"
        ),
    }
    write_json(output / "manifest.json", metadata)
    write_json(output / "source/checkpoint.json", config["structure_checkpoint"])
    chunk_size = config["runtime"]["candidate_chunk_size"]
    slots_per_chunk = max(1, chunk_size // 20)

    def rows(sampling_model):
        started = time.monotonic()
        for start in range(0, len(dataset), slots_per_chunk):
            slot_ids = list(range(start, min(start + slots_per_chunk, len(dataset))))
            indices = [i for i in slot_ids for _ in range(20)]
            seeds = [
                stable_seed(
                    config["seed"],
                    cache["rows"][i // 2]["parent_sample_id"],
                    i % 2,
                    candidate,
                )
                for i in slot_ids
                for candidate in range(20)
            ]
            candidates = _sample_items(
                task="element_set",
                model=sampling_model,
                type_encoding=encoding,
                dataset=dataset,
                indices=indices,
                model_args=model_args,
                sampler=sampler,
                device=device,
                seed=config["seed"],
                chunk_size=chunk_size,
                sample_seeds=seeds,
                pad_final_batch=True,
            )
            for offset, i in enumerate(slot_ids):
                selected = candidates[offset * 20 : (offset + 1) * 20]
                for record in selected:
                    record.pop("type_token_correct", None)
                    record.pop("type_token_total", None)
                yield {
                    "sample_id": f"{cache['rows'][i // 2]['parent_sample_id']}::slot{i % 2}",
                    "dataset_index": i,
                    "reference": {},
                    "candidates": selected,
                    "strata": {
                        "slot": i % 2,
                        "phase_role": ("dominant", "minor")[i % 2],
                        "oracle_n": int(dataset.num_atoms[i]),
                    },
                }
            finished = min(start + slots_per_chunk, len(dataset))
            progress = {
                "completed_slots": finished,
                "total_slots": len(dataset),
                "sampling_seconds": time.monotonic() - started,
            }
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)

    index = write_prediction_cache(
        output / "samples", rows(model), metadata=metadata, shard_size=512
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = score_cache(output, config, cache)
    write_json(
        output / "manifest.json",
        {
            **metadata,
            "status": "complete",
            "metrics": metrics,
            "sample_index_sha256": file_sha256(index),
            "reference_sha256": file_sha256(output / "references.json"),
        },
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    recover(
        load_protocol(args.protocol),
        "conservation",
        torch.device(args.device),
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
