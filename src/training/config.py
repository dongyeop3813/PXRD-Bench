"""Hydra/OmegaConf config layer for Bragglite training.

The grouped YAML configs under ``configs/`` are for readability only. The flat
dict produced here uses the EXACT legacy argparse attribute names so that
``build_train_checkpoint()`` (which saves ``{"model_args": vars(args), ...}``)
round-trips and the sample/eval/pxrd entrypoints can read
``model_args.get("<name>")``.

This module replaced the previous argparse layer (``build_parser`` /
``validate_args``); ``validate_cfg`` is the single source of config validation.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf, open_dict
from src.bragglite.sampling_config import (
    resolve_aa_rho_pair,
    resolve_nonnegative_scalar,
)
from src.data.factory import DATASET_CLASSES
from src.utils.constants import BEST_CKPT_SELECTOR_CHOICES, DATASET_NMAX_DEFAULTS

DEFAULT_NMAX = DATASET_NMAX_DEFAULTS["mp20"]
_GROUP_KEYS = (
    "run",
    "data",
    "model",
    "optim",
    "edm",
    "sampling",
    "monitor",
    "data_augmentation",
    "logging",
    "loss",
    "ckpt",
    "pxrd",
    "cond",
    "augmentation",
)


def _normalize_topk_list(values: list[int]) -> list[int]:
    if not values:
        return []
    out = sorted({int(v) for v in values})
    if any((v <= 0 for v in out)):
        raise ValueError("csp_precise_topk_list values must be positive integers.")
    return out


def _compute_topk_target_count(total_targets: int, requested_targets: int) -> int:
    total_targets = max(0, int(total_targets))
    requested_targets = int(requested_targets)
    if requested_targets <= 0:
        return 0
    return min(total_targets, requested_targets)


def _selected_config_choices(cfg: DictConfig) -> dict[str, str]:
    choices = OmegaConf.select(cfg, "hydra.runtime.choices")
    if choices is None:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            choices = HydraConfig.get().runtime.choices
    if choices is None:
        return {}
    return {
        group: str(choice)
        for group in ("model", "pxrd")
        if (choice := choices.get(group)) is not None
    }


def flatten_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Merge every nested config group into one flat dict keyed by leaf arg names.

    Asserts no leaf-name collisions across groups. Returns plain python types
    (no DictConfig). ``loss_weights`` -> tuple[float, float, float];
    ``csp_precise_topk_list`` -> list[int].
    """
    choices = _selected_config_choices(cfg)
    task_cfg = cfg.copy()
    if "hydra" in task_cfg:
        with open_dict(task_cfg):
            del task_cfg["hydra"]
    container = OmegaConf.to_container(task_cfg, resolve=True, throw_on_missing=True)
    flat: dict[str, Any] = {}
    for key, value in container.items():
        if key in _GROUP_KEYS and isinstance(value, dict):
            for leaf, leaf_value in value.items():
                if leaf in flat:
                    raise ValueError(
                        f"flatten_cfg: duplicate leaf key '{leaf}' (collision while merging group '{key}')."
                    )
                flat[leaf] = leaf_value
        else:
            if key in flat:
                raise ValueError(f"flatten_cfg: duplicate top-level key '{key}'.")
            flat[key] = value
    if "loss_weights" in flat and flat["loss_weights"] is not None:
        lw = [float(v) for v in flat["loss_weights"]]
        if len(lw) != 3:
            raise ValueError(f"loss_weights must have exactly 3 values, got {len(lw)}.")
        flat["loss_weights"] = tuple(lw)
    if "csp_precise_topk_list" in flat and flat["csp_precise_topk_list"] is not None:
        flat["csp_precise_topk_list"] = [int(v) for v in flat["csp_precise_topk_list"]]
    for group, choice in choices.items():
        key = f"{group}_preset"
        if key in flat:
            raise ValueError(
                f"{group}.{key} is derived from the selected {group}= preset and cannot be overridden."
            )
        flat[key] = choice
    return flat


def to_namespace(cfg: DictConfig) -> argparse.Namespace:
    """Bridge cfg -> argparse.Namespace so the training entrypoint reads args.<name>."""
    return argparse.Namespace(**flatten_cfg(cfg))


def validate_cfg(cfg: DictConfig, *, cwd: Path | None = None) -> argparse.Namespace:
    """Validate + normalize a composed config, returning the resolved Namespace.

    Raises ``SystemExit("config error: ...")`` on invalid combinations.
    """
    ns = to_namespace(cfg)
    removed_options = (
        "train_sampling_manifest",
        "train_sampling_manifest_sha256",
        "train_sampling_group_fraction",
        "train_encoder_only",
        "atom_query_init_std",
        "atom_token_repeats",
        "collision_loss_weight",
        "collision_min_distance_angstrom",
        "complexity_importance_beta",
        "complexity_sampling_alpha",
        "complexity_sampling_max_weight",
        "direction_loss_weight",
        "extra_attn_head_bias",
        "formula_count_loss_weight",
        "formula_count_temperature",
        "lattice_cond",
        "lattice_cond_v",
        "lattice_token_repeats",
        "length_bucket_boundaries",
        "length_bucket_max_batch_size",
        "length_bucket_max_tokens",
        "n_space_groups",
        "online_lattice_aug_prob",
        "online_lattice_aug_sigma",
        "orbit_type_loss_weight",
        "persistent_atom_queries",
        "pxrd_sparse_bank",
        "pxrd_sparse_bank_path",
        "pxrd_sparse_bank_sha256",
        "pxrd_sparse_mode",
        "recycle_cross_attention",
        "recycle_cross_attention_layers",
        "recycle_frac",
        "recycle_lat",
        "recycle_mask",
        "recycle_memory",
        "recycle_memory_pad_mask",
        "recycle_present",
        "recycling_cross_attention_layers",
        "recycling_enabled",
        "recycling_fusion",
        "recycling_gate_bias",
        "recycling_gate_hidden_dim",
        "recycling_mode",
        "recycling_pair_bias_init",
        "recycling_probability",
        "recycling_sampling_mode",
        "recycling_widened_layers",
        "set_matching_coord_cost",
        "set_matching_iterations",
        "set_matching_loss",
        "set_matching_method",
        "set_matching_temperature",
        "set_matching_type_cost",
        "sg_cond",
        "sg_cond_dropout",
        "sg_fusion",
        "sg_symprec",
        "skip_lat_scaling",
        "thermo_ppd_mp",
        "thermo_stability_check",
        "with_sg",
    )
    for key in vars(ns):
        if (
            key in removed_options
            or key.startswith("flow_")
            or key.startswith("thermo_")
            or key.startswith("nequip_")
        ):
            raise SystemExit(f"config error: unsupported option {key}")

    def _err(msg: str) -> None:
        raise SystemExit(f"config error: {msg}")

    if str(ns.dataset_class) not in DATASET_CLASSES:
        _err(
            f"unsupported dataset_class {ns.dataset_class!r}; expected {DATASET_CLASSES}"
        )
    dataset_name = str(ns.dataset_name)
    dataset_class = str(ns.dataset_class)
    data_root = Path(str(ns.data_root)).as_posix().rstrip("/")
    if (
        dataset_class == "real_pxrd"
        or dataset_name in {"alex_mp20", "mpts_52"}
        or data_root.endswith("/alex_mp20/base")
        or data_root.endswith("/mpts_52")
    ):
        _err(
            "Alex-MP-20, MPTS-52, and RealPXRD-Solver are retired for new training; use MP-20 or PXRD-Bench."
        )
    legacy_dense_augment = getattr(ns, "pxrd_profile_augment", None)
    dense_augment = bool(getattr(ns, "pxrd_dense_profile_augment", False))
    if legacy_dense_augment is not None:
        if dense_augment and dense_augment != bool(legacy_dense_augment):
            _err("pxrd_dense_profile_augment and legacy pxrd_profile_augment disagree.")
        dense_augment = bool(legacy_dense_augment)
    ns.pxrd_augmentation = str(getattr(ns, "pxrd_augmentation", "none"))
    if ns.pxrd_augmentation not in {"none", "profile_randomization", "xrdiff_style"}:
        _err(
            "pxrd.pxrd_augmentation must be none, profile_randomization, or xrdiff_style."
        )
    dense_augment = dense_augment or ns.pxrd_augmentation != "none"
    ns.pxrd_dense_profile_augment = dense_augment
    ns.pxrd_profile_augment = dense_augment
    if not 0.0 <= float(ns.pxrd_profile_augment_probability) <= 1.0:
        _err("data_augmentation.pxrd_profile_augment_probability must be in [0, 1].")
    gamma_shape = getattr(ns, "pxrd_xrdiff_gamma_shape", None)
    gamma_scale = getattr(ns, "pxrd_xrdiff_gamma_scale", None)
    if (gamma_shape is None) != (gamma_scale is None):
        _err("XRDiff Gamma shape and scale must be set together.")
    if gamma_shape is not None and (
        float(gamma_shape) <= 0.0 or float(gamma_scale) <= 0.0
    ):
        _err("XRDiff Gamma shape and scale must be positive.")
    if dense_augment and ns.pxrd_augmentation == "none":
        ns.pxrd_augmentation = "profile_randomization"
    ns.formulation = str(ns.formulation).strip().lower()
    if ns.formulation != "edm":
        _err("Only the paper EDM formulation is supported.")
    ns.atom_positional_encoding = str(ns.atom_positional_encoding).strip().lower()
    if ns.atom_positional_encoding not in {"none", "sinusoidal"}:
        _err("model.atom_positional_encoding must be none or sinusoidal.")
    ns.atom_ordering = str(ns.atom_ordering).strip().lower()
    if ns.atom_ordering not in {"none", "mcflow"}:
        _err("data.atom_ordering must be none or mcflow.")
    if bool(ns.hierarchical_orbit_permutation) and ns.atom_ordering != "mcflow":
        _err(
            "data_augmentation.hierarchical_orbit_permutation requires data.atom_ordering=mcflow."
        )
    if ns.atom_ordering == "mcflow" and str(ns.dataset_class) not in {
        "mp20_tokens",
        "pxrd_bench",
    }:
        _err(
            "data.atom_ordering=mcflow requires dataset_class=mp20_tokens, or pxrd_bench."
        )
    if float(ns.atom_ordering_symprec) <= 0.0:
        _err("data.atom_ordering_symprec must be > 0.")
    checkpoint_modes = [
        name
        for name in ("init_checkpoint", "resume_checkpoint", "transfer_checkpoint")
        if getattr(ns, name, None)
    ]
    if len(checkpoint_modes) > 1:
        _err(
            "use only one of pxrd.init_checkpoint, pxrd.resume_checkpoint, or pxrd.transfer_checkpoint."
        )
    for name in checkpoint_modes:
        path = Path(getattr(ns, name))
        checked_path = path if path.is_absolute() or cwd is None else cwd / path
        if not checked_path.is_file():
            _err(f"pxrd.{name} does not exist or is not a file: {path}")
    if hasattr(ns, "monitor_selection_split"):
        ns.monitor_selection_split = str(ns.monitor_selection_split).strip().lower()
        if ns.monitor_selection_split != "val":
            _err(
                "monitor.monitor_selection_split must be 'val'; selecting PXRD checkpoints on the test split is test-set leakage."
            )
    if hasattr(ns, "monitor_split"):
        ns.monitor_split = str(ns.monitor_split).strip().lower()
    if ns.type_encoding == "subatomic_tokenizer_pca":
        ns.type_encoding = "subatomic_tokenizer_pca_24"
    if hasattr(ns, "csp_precise_topk_list"):
        try:
            ns.csp_precise_topk_list = _normalize_topk_list(ns.csp_precise_topk_list)
        except ValueError as exc:
            _err(str(exc))
    if hasattr(ns, "csp_precise_topk_samples") and ns.csp_precise_topk_samples < 0:
        _err("--csp_precise_topk_samples must be >= 0.")
    if int(ns.monitor_every) < 0:
        _err("monitor.monitor_every must be >= 0.")
    if int(ns.log_every) <= 0:
        _err("monitor.log_every must be > 0.")
    if int(ns.val_every) < 0:
        _err("monitor.val_every must be >= 0.")
    if int(ns.val_batches) <= 0:
        _err("monitor.val_batches must be > 0.")
    if int(ns.monitor_num_samples) < 0:
        _err("monitor.monitor_num_samples must be >= 0.")
    if ns.best_ckpt_selector == "csp_match_rate@1" and (
        not getattr(ns, "monitor_ids_path", None)
        or ns.monitor_mode != "ema"
        or ns.monitor_every <= 0
        or (not ns.csp)
    ):
        _err(
            "csp_match_rate@1 checkpoint selection requires fixed-ID EMA CSP monitoring"
        )
    if getattr(ns, "monitor_ids_path", None):
        if ns.monitor_resample_targets:
            _err("a frozen monitor ID manifest requires monitor_resample_targets=false")
        if ns.csp and (ns.monitor_num_candidates != 1 or list(ns.monitor_topk) != [1]):
            _err("fixed-ID CSP checkpoint monitoring requires K=1")
    ns.monitor_mode = str(ns.monitor_mode).strip().lower()
    if ns.monitor_mode not in {"regular", "ema", "both"}:
        _err("monitor.monitor_mode must be regular, ema, or both.")
    if hasattr(ns, "pxrd_cond") and bool(ns.pxrd_cond) and (ns.monitor_mode == "both"):
        _err("PXRD training does not support monitor.monitor_mode=both.")
    if hasattr(ns, "pxrd_cond") and bool(ns.pxrd_cond):
        ns.monitor_topk = sorted({int(k) for k in ns.monitor_topk})
        if not ns.monitor_topk or any((k <= 0 for k in ns.monitor_topk)):
            _err("monitor.monitor_topk must contain positive integers.")
        if int(ns.monitor_num_candidates) < ns.monitor_topk[-1]:
            _err(
                "monitor.monitor_num_candidates must be at least max(monitor.monitor_topk)."
            )
    ns.plateau_metric = str(getattr(ns, "plateau_metric", "val_loss")).strip().lower()
    if (
        str(ns.lr_schedule).strip().lower() == "plateau"
        and ns.plateau_metric != "val_loss"
    ):
        prefix = "csp_match_rate@"
        try:
            plateau_k = int(ns.plateau_metric.removeprefix(prefix))
        except ValueError:
            plateau_k = 0
        if (
            not bool(ns.csp)
            or not ns.plateau_metric.startswith(prefix)
            or plateau_k <= 0
        ):
            _err("optim.plateau_metric must be val_loss or csp_match_rate@K.")
        if int(ns.monitor_every) <= 0 or int(ns.monitor_num_samples) <= 0:
            _err("CSP match-rate plateau scheduling requires periodic monitoring.")
        if ns.monitor_mode == "both":
            _err("CSP match-rate plateau scheduling requires one monitor weight mode.")
        if plateau_k > 1 and (
            plateau_k not in ns.csp_precise_topk_list
            or int(ns.csp_precise_topk_samples) < int(ns.monitor_num_samples)
        ):
            _err(
                "optim.plateau_metric requires matching csp_precise_topk_list and csp_precise_topk_samples >= monitor_num_samples."
            )
    if int(ns.max_steps) <= 0:
        _err("run.max_steps must be > 0.")
    if float(ns.grad_clip_norm) <= 0.0:
        _err("optim.grad_clip_norm must be > 0.")
    ns.grad_clip_mode = str(ns.grad_clip_mode).strip().lower()
    if ns.grad_clip_mode not in {"norm", "value"}:
        _err("optim.grad_clip_mode must be norm or value.")
    if int(ns.grad_accum_steps) <= 0:
        _err("optim.grad_accum_steps must be > 0.")
    if ns.seed < 0:
        _err("--seed must be >= 0.")
    try:
        aa_rho_coords, aa_rho_lattice = resolve_aa_rho_pair(
            aa_rho_coords=ns.aa_rho_coords, aa_rho_lattice=ns.aa_rho_lattice
        )
        aa_rho_types = resolve_nonnegative_scalar("aa_rho_types", ns.aa_rho_types)
    except ValueError as exc:
        _err(str(exc))
    ns.aa_rho_coords = float(aa_rho_coords)
    ns.aa_rho_lattice = float(aa_rho_lattice)
    ns.aa_rho_types = float(aa_rho_types)
    ns.aa_rho_by_target = {
        "types": float(ns.aa_rho_types),
        "coords": float(ns.aa_rho_coords),
        "lattice": float(ns.aa_rho_lattice),
    }
    if ns.d_model % ns.n_heads != 0:
        _err("--d_model must be divisible by --n_heads.")
    if ns.edge_bias_n_freqs <= 0:
        _err("--edge_bias_n_freqs must be > 0.")
    if ns.edge_bias_hidden_dim <= 0:
        _err("--edge_bias_hidden_dim must be > 0.")
    if ns.edge_bias_n_rbf <= 0:
        _err("--edge_bias_n_rbf must be > 0.")
    if ns.edge_bias_rbf_max <= 0.0:
        _err("--edge_bias_rbf_max must be > 0.")
    dataset_default_nmax = DATASET_NMAX_DEFAULTS.get(ns.dataset_name, DEFAULT_NMAX)
    nmax = int(ns.nmax) if ns.nmax is not None else int(dataset_default_nmax)
    if nmax <= 0:
        _err("--nmax must be a positive integer.")
    ns.nmax = nmax
    assert ns.best_ckpt_selector in BEST_CKPT_SELECTOR_CHOICES, (
        f"--best_ckpt_selector={ns.best_ckpt_selector} not in {BEST_CKPT_SELECTOR_CHOICES}."
    )
    return ns
