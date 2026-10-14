"""Dataset + DataLoader construction for Bragglite training.

Extracted from ``train_bragglite.py`` so the entrypoint's ``run()`` stays an
orchestrator. Functions are pure (no hidden state); ``args`` is the validated
namespace produced by ``config.validate_cfg``.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from src.data.factory import build_token_dataset, dataset_needs_split_setup
from src.data.mp20_tokens import VZ, collate_mp20_tokens
from src.utils.dataset import (
    compute_allowed_elements,
    compute_dataset_element_distribution,
    ensure_dataset_splits,
)
from src.utils.seeding import seed_dataloader_worker


def _parse_overfit_subset_indices(value: Any) -> list[int] | None:
    """Normalize an optional exact train-index subset override."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null"}:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        if not stripped:
            return None
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise TypeError(
        f"overfit_subset_indices must be null, an int, a comma-separated string, or a list of ints; got {type(value).__name__}."
    )


def _load_excluded_train_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text())
    values = payload.get("excluded_train_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("train exclusion manifest must contain excluded_train_ids")
    if isinstance(payload, dict):
        expected_count = payload.get("excluded_train_id_count")
        if expected_count is not None and int(expected_count) != len(values):
            raise ValueError("train exclusion manifest ID count mismatch")
        expected_sha256 = payload.get("excluded_train_ids_sha256")
        observed_sha256 = hashlib.sha256(
            "\n".join(map(str, values)).encode()
        ).hexdigest()
        if expected_sha256 is not None and expected_sha256 != observed_sha256:
            raise ValueError("train exclusion manifest ID checksum mismatch")
    ids = {str(value) for value in values}
    if len(ids) != len(values):
        raise ValueError("train exclusion manifest contains duplicate IDs")
    return ids


def _dataset_material_ids(dataset: Dataset):
    """Yield stable IDs without decoding large lazy dataset payloads."""
    items = getattr(dataset, "items", None)
    if items is not None:
        yield from (str(item["mp_id"]) for item in items)
        return
    keys = getattr(dataset, "keys", None)
    if keys is not None:
        yield from (
            key.decode() if isinstance(key, bytes) else str(key) for key in keys
        )
        return
    rows = getattr(dataset, "rows", None)
    if rows is not None and "material_id" in rows:
        yield from rows["material_id"].astype(str)
        return
    yield from (
        str(_dataset_raw_item(dataset, index)["mp_id"]) for index in range(len(dataset))
    )


def _dataset_element_counts(
    dataset: Dataset, indices: list[int] | None = None
) -> torch.Tensor:
    """Count elements in the dataset's native cached-stat convention."""
    if indices is None:
        if hasattr(dataset, "_element_counts"):
            return torch.as_tensor(dataset._element_counts(), dtype=torch.float64)
        if hasattr(dataset, "_kept_elem_counts"):
            return torch.as_tensor(dataset._kept_elem_counts(), dtype=torch.float64)
        if hasattr(dataset, "_composition_counts"):
            return torch.as_tensor(dataset._composition_counts(), dtype=torch.float64)
        indices = list(range(len(dataset)))
    rows = getattr(dataset, "rows", None)
    if rows is not None and "composition_key" in rows:
        counts = torch.zeros(VZ, dtype=torch.float64)
        for value in rows.iloc[indices]["composition_key"].astype(str):
            for token in value.split("|"):
                atomic_number, amount = token.split(":")
                counts[int(atomic_number) - 1] += int(amount)
        return counts
    counts = torch.zeros(VZ, dtype=torch.float64)
    for index in indices:
        item = _dataset_raw_item(dataset, index)
        values = item["A0"][~item["pad_mask"].bool()]
        counts += torch.bincount(values, minlength=VZ + 1)[1:].double()
    return counts


def _stats_from_counts(counts: torch.Tensor):
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("filtered element counts must be finite and nonnegative")
    total = float(counts.sum())
    distribution = counts / total if total > 0 else None
    allowed = counts > 0
    return (distribution, allowed if bool(allowed.any()) else None)


def _select_dataset_indices(
    dataset: Dataset, indices: list[int], *, element_counts: torch.Tensor | None = None
) -> Dataset:
    """Filter built-in datasets in place so their custom samplers remain visible."""
    items = getattr(dataset, "items", None)
    if items is not None:
        dataset.items = [items[index] for index in indices]
        return dataset
    keys = getattr(dataset, "keys", None)
    if keys is not None and hasattr(dataset, "ns"):
        dataset.keys = [keys[index] for index in indices]
        dataset.ns = np.asarray(dataset.ns)[indices]
        if element_counts is not None and hasattr(dataset, "_kept_elem_counts"):
            dataset._filtered_elem_counts = np.asarray(
                element_counts.cpu(), dtype=np.int64
            )
        if getattr(dataset, "_mem", None) is not None:
            dataset._mem = [dataset._mem[index] for index in indices]
        dataset.n_kept = len(indices)
        return dataset
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        dataset.rows = rows.iloc[indices].reset_index(drop=True)
        if hasattr(dataset, "_element_counts_cache"):
            dataset._element_counts_cache = None
        return dataset
    return Subset(dataset, indices)


class _RepeatToLength(Dataset):
    """Repeat a small dataset view without caching its returned samples."""

    def __init__(self, dataset: Dataset, length: int) -> None:
        if len(dataset) <= 0:
            raise ValueError("Cannot repeat an empty dataset.")
        if length < len(dataset):
            raise ValueError(
                f"repeat length {length} must be >= dataset length {len(dataset)}."
            )
        self.dataset = dataset
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.dataset[int(index) % len(self.dataset)]


def make_token_dataset(
    *, args, nmax, root, split, augment_translate, has_split, augment_pxrd
):
    """Build a token dataset of ``args.dataset_class``.

    For ``mp20_tokens`` the split may collapse to ``"all"`` when no train/val
    CSVs exist; ``dao``/``real_pxrd`` always ship their own splits.

    PXRD-conditioned runs (``args.pxrd_cond``, see ``configs/train_pxrd.yaml``)
    additionally attach the d-I peak list. ``augment_pxrd`` is independent of
    fractional-coordinate translation.
    """
    resolved_split = split if has_split else "all"
    pxrd_kwargs = {}
    if getattr(args, "pxrd_cond", False):
        pxrd_representation = getattr(args, "pxrd_representation", "legacy_di")
        apply_pxrd_augmentation = bool(augment_pxrd)
        stick_augment = bool(apply_pxrd_augmentation and args.pxrd_augment)
        dense_augment = bool(
            apply_pxrd_augmentation
            and getattr(args, "pxrd_dense_profile_augment", False)
        )
        pxrd_kwargs = {
            "with_pxrd_peaks": True,
            "pxrd_max_peaks": args.pxrd_max_peaks,
            "pxrd_peak_selection": args.pxrd_peak_selection,
            "pxrd_low_angle_fraction": args.pxrd_low_angle_fraction,
            "pxrd_wavelength": args.pxrd_wavelength,
            "pxrd_peak_augment": stick_augment,
            "pxrd_aug_jitter_deg": args.pxrd_aug_jitter_deg,
            "pxrd_aug_int_scale_min": args.pxrd_aug_int_scale_min,
            "pxrd_aug_int_scale_max": args.pxrd_aug_int_scale_max,
            "pxrd_aug_width_scale_min": getattr(args, "pxrd_aug_width_scale_min", 0.7),
            "pxrd_aug_width_scale_max": getattr(args, "pxrd_aug_width_scale_max", 1.5),
            "pxrd_profile_augment": dense_augment,
            "pxrd_dense_profile_augment": dense_augment,
            "pxrd_profile_augment_probability": float(
                getattr(args, "pxrd_profile_augment_probability", 1.0)
            ),
            "pxrd_aug_zero_shift_deg": getattr(args, "pxrd_aug_zero_shift_deg", 0.15),
            "pxrd_aug_eta_min": getattr(args, "pxrd_aug_eta_min", 0.0),
            "pxrd_aug_eta_max": getattr(args, "pxrd_aug_eta_max", 0.5),
            "pxrd_aug_background_max": getattr(args, "pxrd_aug_background_max", 0.2),
            "pxrd_aug_noise_std_max": getattr(args, "pxrd_aug_noise_std_max", 0.02),
            "pxrd_aug_drop_frac_max": args.pxrd_aug_drop_frac_max,
            "pxrd_representation": pxrd_representation,
            "pxrd_peak_source": getattr(args, "pxrd_peak_source", "bragg"),
            "pxrd_peak_features": getattr(
                args, "pxrd_peak_features", "position_intensity"
            ),
            "pxrd_peak_position": getattr(args, "pxrd_peak_position", "d"),
            "pxrd_peak_position_normalized": bool(
                getattr(args, "pxrd_peak_position_normalized", False)
            ),
            "pxrd_peak_intensity_normalization": getattr(
                args, "pxrd_peak_intensity_normalization", "selected_topk"
            ),
            "pxrd_peak_min_prominence": getattr(args, "pxrd_peak_min_prominence", 0.0),
            "pxrd_peak_min_relative_intensity": getattr(
                args, "pxrd_peak_min_relative_intensity", 0.0
            ),
            "pxrd_peak_prominence_normalization": getattr(
                args, "pxrd_peak_prominence_normalization", "none"
            ),
            "pxrd_peak_width_unit": getattr(args, "pxrd_peak_width_unit", "two_theta"),
            "pxrd_peak_width_normalization": getattr(
                args, "pxrd_peak_width_normalization", "none"
            ),
            "pxrd_feature_source": getattr(args, "pxrd_feature_source", "auto"),
            "pxrd_profile_step": getattr(args, "pxrd_profile_step", 0.1),
            "pxrd_caglioti_u": getattr(args, "pxrd_caglioti_u", 0.1),
            "pxrd_caglioti_v": getattr(args, "pxrd_caglioti_v", 0.01),
            "pxrd_caglioti_w": getattr(args, "pxrd_caglioti_w", 0.1),
            "pxrd_pseudo_voigt_eta": getattr(args, "pxrd_pseudo_voigt_eta", 0.1),
            "pxrd_dense_n_bins": getattr(args, "pxrd_dense_n_bins", 8500),
            "pxrd_xrdiff_gamma_shape": getattr(args, "pxrd_xrdiff_gamma_shape", None),
            "pxrd_xrdiff_gamma_scale": getattr(args, "pxrd_xrdiff_gamma_scale", None),
            "real_pxrd_two_theta_min": getattr(args, "pxrd_two_theta_min", 5.0),
            "real_pxrd_two_theta_max": getattr(
                args,
                "pxrd_two_theta_max",
                80.0
                if pxrd_representation in {"broadened_basic", "broadened_shape"}
                else 90.0,
            ),
            "pxrd_augmentation": getattr(
                args, "pxrd_augmentation", "profile_randomization"
            )
            if dense_augment
            else "none",
        }
    sg_kwargs = {}
    return build_token_dataset(
        args.dataset_class,
        root=root,
        split=resolved_split,
        nmax=nmax,
        augment_translate=augment_translate,
        real_pxrd_variant=args.real_pxrd_variant,
        real_pxrd_in_memory=args.real_pxrd_in_memory,
        structure_cache=getattr(args, "structure_cache", None),
        atom_ordering=getattr(args, "atom_ordering", "none"),
        hierarchical_orbit_permutation=bool(
            augment_translate and getattr(args, "hierarchical_orbit_permutation", False)
        ),
        atom_ordering_symprec=float(getattr(args, "atom_ordering_symprec", 0.1)),
        phase_reference_root=str(getattr(args, "phase_reference_root", "data/mp20")),
        **pxrd_kwargs,
        **sg_kwargs,
    )


@dataclass
class DatasetBundle:
    """Train/val/ref splits plus train-split element statistics."""

    ds: Any
    val_ds: Any
    ref_ds: Any
    has_split: bool
    train_element_dist: Any
    train_allowed_mask: Any
    train_subset_indices: list[int] | None = None


def setup_datasets(*, args, nmax) -> DatasetBundle:
    """Build the train/val/ref datasets and train element distribution/mask.

    Prefers a dataset's own O(1) element stats (RealPXRDTokens caches these in
    its index) over the generic full-pass helpers, which would scan millions
    of lazily-decoded records on the real_pxrd train split.
    """
    if dataset_needs_split_setup(args.dataset_class):
        has_split = ensure_dataset_splits(args.data_root, args.dataset_name)
    else:
        has_split = True

    def make_ds(split, augment_translate, augment_pxrd):
        return make_token_dataset(
            args=args,
            nmax=nmax,
            root=args.data_root,
            split=split,
            augment_translate=augment_translate,
            augment_pxrd=augment_pxrd,
            has_split=has_split,
        )

    disable_train_augmentation = bool(getattr(args, "overfit_no_augment", False))
    train_translate = bool(getattr(args, "augment_translate", True)) and (
        not disable_train_augmentation
    )
    train_pxrd = not disable_train_augmentation
    ds = make_ds("train", train_translate, train_pxrd)
    val_ds = make_ds("val", False, False)
    ref_ds = make_ds("train", False, False)
    train_subset_indices = None
    excluded_ids = _load_excluded_train_ids(
        getattr(args, "train_exclude_ids_path", None)
    )
    exclusion_stats = None
    if excluded_ids:
        excluded_indices_by_id = {}
        for index, material_id in enumerate(_dataset_material_ids(ds)):
            if material_id not in excluded_ids:
                continue
            if material_id in excluded_indices_by_id:
                raise ValueError(f"duplicate loaded train ID: {material_id}")
            excluded_indices_by_id[material_id] = index
        missing = excluded_ids - excluded_indices_by_id.keys()
        if missing:
            raise ValueError(
                f"train exclusion IDs not found in loaded split: {sorted(missing)[:5]}"
            )
        excluded_indices = sorted(excluded_indices_by_id.values())
        excluded_index_set = set(excluded_indices)
        retained_indices = [
            index for index in range(len(ds)) if index not in excluded_index_set
        ]
        filtered_element_counts = _dataset_element_counts(ds) - _dataset_element_counts(
            ds, excluded_indices
        )
        exclusion_stats = _stats_from_counts(filtered_element_counts)
        ds = _select_dataset_indices(
            ds, retained_indices, element_counts=filtered_element_counts
        )
        ref_ds = _select_dataset_indices(
            ref_ds, retained_indices, element_counts=filtered_element_counts
        )
        print(
            f"[data] excluded {len(excluded_ids)} train IDs; remaining={len(retained_indices)}"
        )
    train_subset_size = int(getattr(args, "train_subset_size", 0) or 0)
    explicit_indices = _parse_overfit_subset_indices(
        getattr(args, "overfit_subset_indices", None)
    )
    overfit_subset_size = int(getattr(args, "overfit_subset_size", 0) or 0)
    if train_subset_size > 0:
        if explicit_indices or overfit_subset_size > 0:
            raise ValueError(
                "train_subset_size cannot be combined with overfit subset options"
            )
        if train_subset_size > len(ds):
            raise ValueError(
                f"train_subset_size={train_subset_size} exceeds effective train split size {len(ds)}."
            )
        generator = torch.Generator(device="cpu").manual_seed(
            int(getattr(args, "train_subset_seed", args.seed))
        )
        train_subset_indices = torch.randperm(len(ds), generator=generator)[
            :train_subset_size
        ].tolist()
        ds = _select_dataset_indices(ds, train_subset_indices)
        ref_ds = _select_dataset_indices(ref_ds, train_subset_indices)
        exclusion_stats = None
    elif explicit_indices:
        if excluded_ids:
            raise ValueError(
                "train_exclude_ids_path cannot be combined with overfit_subset_indices"
            )
        bad = [idx for idx in explicit_indices if idx < 0 or idx >= len(ds)]
        if bad:
            raise ValueError(
                f"overfit_subset_indices contains out-of-range train indices for split size {len(ds)}: {bad}"
            )
        train_subset_indices = explicit_indices
        ds = Subset(ds, train_subset_indices)
        ref_ds = Subset(ref_ds, train_subset_indices)
    else:
        subset_size = overfit_subset_size
        if subset_size > 0:
            if excluded_ids:
                raise ValueError(
                    "train_exclude_ids_path cannot be combined with overfit_subset_size"
                )
            if subset_size > len(ds):
                raise ValueError(
                    f"overfit_subset_size={subset_size} exceeds train split size {len(ds)}."
                )
            generator = torch.Generator(device="cpu").manual_seed(
                int(getattr(args, "overfit_subset_seed", args.seed))
            )
            train_subset_indices = torch.randperm(len(ds), generator=generator)[
                :subset_size
            ].tolist()
            ds = Subset(ds, train_subset_indices)
            ref_ds = Subset(ref_ds, train_subset_indices)
    if exclusion_stats is not None:
        train_element_dist, train_allowed_mask = exclusion_stats
    elif isinstance(ds, Subset):
        train_element_dist, train_allowed_mask = _raw_element_stats(ds)
    elif hasattr(ds, "element_distribution"):
        train_element_dist = ds.element_distribution()
        train_allowed_mask = ds.allowed_elements()
    else:
        train_element_dist = compute_dataset_element_distribution(ds)
        train_allowed_mask = compute_allowed_elements(ds)
    return DatasetBundle(
        ds=ds,
        val_ds=val_ds,
        ref_ds=ref_ds,
        has_split=has_split,
        train_element_dist=train_element_dist,
        train_allowed_mask=train_allowed_mask,
        train_subset_indices=train_subset_indices,
    )


def infinite_loader(dl, *, distributed_sampler=None):
    """Yield batches from ``dl`` forever (step-based training)."""
    from src.utils.seeding import isolated_rng

    resume = getattr(dl, "resume_state", None)
    epoch = resume["epoch"] if resume else 0
    if resume:
        if dl.persistent_workers or dl.generator is None:
            raise ValueError(
                "Loader resume requires an explicit generator and nonpersistent workers"
            )
        dl.generator.set_state(resume["generator"].cpu())
    while True:
        if distributed_sampler is not None:
            distributed_sampler.set_epoch(epoch)
        state = {
            "epoch": epoch,
            "batches": 0,
            "generator": dl.generator.get_state() if dl.generator is not None else None,
        }
        iterator = iter(dl)
        if resume:
            with isolated_rng():
                for _ in range(resume["batches"]):
                    next(iterator)
            state["batches"] = resume["batches"]
            resume = None
        for batch in iterator:
            state["batches"] += 1
            dl.checkpoint_state = state
            yield batch
        epoch += 1


def _dataset_raw_item(dataset: Dataset, index: int) -> dict:
    """Read cached metadata without triggering stochastic train augmentation."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    items = getattr(dataset, "items", None)
    if items is None:
        return dataset[index]
    return items[index]


def _raw_element_stats(dataset: Dataset):
    """Compute subset stats without invoking stochastic sample augmentation."""
    counts = torch.zeros(VZ, dtype=torch.float64)
    for index in range(len(dataset)):
        item = _dataset_raw_item(dataset, index)
        values = item["A0"][~item["pad_mask"].bool()]
        counts += torch.bincount(values, minlength=VZ + 1)[1:].double()
    total = float(counts.sum())
    distribution = counts / total if total > 0 else None
    allowed = counts > 0
    return (distribution, allowed if bool(allowed.any()) else None)


def build_dataloaders(*, args, ds, val_ds, device, rank: int = 0, world_size: int = 1):
    """Construct train/val DataLoaders and an infinite train iterator.

    Returns ``(loader, val_loader, data_iter, steps_per_epoch)``.
    """
    train_loader_gen = torch.Generator()
    train_loader_gen.manual_seed(int(args.seed) + int(rank))
    loader_extra = (
        {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
        if args.num_workers > 0
        else {}
    )
    if (
        args.num_workers > 0
        and world_size == 1
        and (getattr(args, "dataset_name", None) == "mp20_polymorph_split")
        and (not getattr(args, "pxrd_cond", False))
    ):
        loader_extra["persistent_workers"] = False
    overfit_enabled = bool(
        _parse_overfit_subset_indices(getattr(args, "overfit_subset_indices", None))
        or int(getattr(args, "overfit_subset_size", 0) or 0) > 0
    )
    train_ds = ds
    if overfit_enabled and len(ds) < int(args.batch_size):
        train_ds = _RepeatToLength(ds, int(args.batch_size))
    shard_sampler = None
    distributed_sampler = None
    if hasattr(train_ds, "training_batch_sampler"):
        shard_sampler = train_ds.training_batch_sampler(
            int(args.batch_size),
            int(args.seed),
            rank=int(rank),
            world_size=int(world_size),
        )
    elif world_size > 1:
        distributed_sampler = DistributedSampler(
            train_ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            seed=int(args.seed),
            drop_last=True,
        )
    common_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=train_loader_gen,
        worker_init_fn=seed_dataloader_worker,
        **loader_extra,
    )
    if shard_sampler is not None:
        loader = DataLoader(
            train_ds,
            batch_sampler=shard_sampler,
            collate_fn=collate_mp20_tokens,
            **common_loader_kwargs,
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=distributed_sampler is None,
            sampler=distributed_sampler or None,
            collate_fn=collate_mp20_tokens,
            drop_last=True,
            **common_loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mp20_tokens,
        pin_memory=device.type == "cuda",
        drop_last=False,
        **loader_extra,
    )
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    steps_per_epoch = max(1.0, len(loader) / grad_accum_steps)
    data_iter = infinite_loader(
        loader, distributed_sampler=None or shard_sampler or distributed_sampler
    )
    return (loader, val_loader, data_iter, steps_per_epoch)
