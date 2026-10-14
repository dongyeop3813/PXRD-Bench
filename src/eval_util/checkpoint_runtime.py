"""Shared checkpoint/runtime helpers for Bragglite eval and sampling entrypoints."""

from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from src.bragglite.factory import build_bragglite_from_model_args

_ORIG_PREFIX = "_orig_mod."


@dataclass(frozen=True)
class SamplerConfig:
    sample_num_steps: int
    solver: str
    sigma_min: float
    sigma_max: float
    rho: float
    S_churn: float
    S_min: float
    S_max: float
    S_noise: float
    sigma_data_type: float
    sigma_data_coord: float
    sigma_data_lattice: float
    aa_frac_max_scale: float
    aa_rho_types: float
    aa_rho_coords: float
    aa_rho_lattice: float
    lattice_repr: str
    sigma_min_types: float | None = None
    sigma_max_types: float | None = None
    rho_types: float | None = None
    sigma_min_coords: float | None = None
    sigma_max_coords: float | None = None
    rho_coords: float | None = None
    sigma_min_lattice: float | None = None
    sigma_max_lattice: float | None = None
    rho_lattice: float | None = None

    @classmethod
    def resolve(
        cls,
        model_args: dict[str, Any],
        *,
        overrides: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> "SamplerConfig":
        """Resolve sampler values with override > checkpoint > legacy default precedence."""
        overrides = overrides or {}
        base = {
            "sample_num_steps": 150,
            "solver": "heun",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "S_churn": 60.0,
            "S_min": 0.0,
            "S_max": 999.0,
            "S_noise": 1.003,
            "sigma_data_type": 0.3,
            "sigma_data_coord": 0.3,
            "sigma_data_lattice": 0.3,
            "aa_frac_max_scale": 0.0,
            "aa_rho_types": 0.0,
            "aa_rho_coords": 10.0,
            "aa_rho_lattice": 10.0,
            "lattice_repr": "y1",
        }
        base.update(defaults or {})
        values = {
            key: cfg_value(overrides.get(key), model_args, key, default)
            for key, default in base.items()
        }
        optional_keys = (
            "sigma_min_types",
            "sigma_max_types",
            "rho_types",
            "sigma_min_coords",
            "sigma_max_coords",
            "rho_coords",
            "sigma_min_lattice",
            "sigma_max_lattice",
            "rho_lattice",
        )
        optional_values = {
            key: cfg_value(overrides.get(key), model_args, key, None)
            for key in optional_keys
        }

        def optional_float(key: str) -> float | None:
            value = optional_values[key]
            return None if value is None else float(value)

        return cls(
            sample_num_steps=int(values["sample_num_steps"]),
            solver=str(values["solver"]).strip().lower(),
            sigma_min=float(values["sigma_min"]),
            sigma_max=float(values["sigma_max"]),
            rho=float(values["rho"]),
            S_churn=float(values["S_churn"]),
            S_min=float(values["S_min"]),
            S_max=float(values["S_max"]),
            S_noise=float(values["S_noise"]),
            sigma_data_type=float(values["sigma_data_type"]),
            sigma_data_coord=float(values["sigma_data_coord"]),
            sigma_data_lattice=float(values["sigma_data_lattice"]),
            aa_frac_max_scale=float(values["aa_frac_max_scale"]),
            aa_rho_types=float(values["aa_rho_types"]),
            aa_rho_coords=float(values["aa_rho_coords"]),
            aa_rho_lattice=float(values["aa_rho_lattice"]),
            lattice_repr=str(values["lattice_repr"]),
            sigma_min_types=optional_float("sigma_min_types"),
            sigma_max_types=optional_float("sigma_max_types"),
            rho_types=optional_float("rho_types"),
            sigma_min_coords=optional_float("sigma_min_coords"),
            sigma_max_coords=optional_float("sigma_max_coords"),
            rho_coords=optional_float("rho_coords"),
            sigma_min_lattice=optional_float("sigma_min_lattice"),
            sigma_max_lattice=optional_float("sigma_max_lattice"),
            rho_lattice=optional_float("rho_lattice"),
        )


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a torch checkpoint with compatibility for older torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def claim_output_dir(path: str | Path) -> Path:
    """Atomically claim a fresh output namespace."""
    output = Path(path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"output namespace already exists: {output}; choose a fresh output_dir"
        ) from exc
    return output


def strip_orig_prefix(state: Any) -> Any:
    """Remove torch.compile's ``_orig_mod.`` prefix from a state dict if present."""
    if isinstance(state, dict) and any((k.startswith(_ORIG_PREFIX) for k in state)):
        return {
            k[len(_ORIG_PREFIX) :] if k.startswith(_ORIG_PREFIX) else k: v
            for k, v in state.items()
        }
    return state


def normalize_compiled_state_prefixes(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Strip compile prefixes from model/EMA state dicts in-place and return ``ckpt``."""
    ckpt["model_state_dict"] = strip_orig_prefix(ckpt.get("model_state_dict"))
    if ckpt.get("ema_state_dict") is not None:
        ckpt["ema_state_dict"] = strip_orig_prefix(ckpt["ema_state_dict"])
    return ckpt


def checkpoint_model_args(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Resolve checkpoint config with resolved_config over legacy model_args."""
    legacy = ckpt.get("model_args")
    resolved = ckpt.get("resolved_config")
    values = dict(legacy) if isinstance(legacy, dict) else {}
    if isinstance(resolved, dict):
        values.update(resolved)
    return values


def build_model_from_checkpoint(
    ckpt: dict[str, Any], *, device: torch.device, strip_compile_prefixes: bool = True
):
    """Build a Bragglite model from ``ckpt`` and return ``(model, model_args)``."""
    if strip_compile_prefixes:
        normalize_compiled_state_prefixes(ckpt)
    merged = checkpoint_model_args(ckpt)
    compatible = {**ckpt, "model_args": merged}
    return build_bragglite_from_model_args(ckpt=compatible, device=device)


def cfg_value(cli_val: Any, model_args: dict[str, Any], key: str, default: Any) -> Any:
    """Return CLI/config value if set, else checkpoint ``model_args`` value, else default."""
    if isinstance(cli_val, str) and cli_val == "":
        cli_val = None
    if cli_val is not None:
        return cli_val
    if key in model_args:
        value = model_args[key]
        return default if value is None else value
    return default


def cfg_or_model_value(
    cfg: Any, cfg_key: str, model_args: dict[str, Any], model_key: str, default: Any
) -> Any:
    """Return ``cfg[cfg_key]`` when non-null, otherwise ``model_args[model_key]``."""
    cfg_value_raw = cfg.get(cfg_key, None) if hasattr(cfg, "get") else None
    return cfg_value(cfg_value_raw, model_args, model_key, default)


def apply_ema_state_dict(
    model: torch.nn.Module, ema_state: dict[str, torch.Tensor]
) -> None:
    """Copy EMA parameters into ``model`` in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                src = ema_state[name].to(device=param.device, dtype=param.dtype)
                param.copy_(src)


def maybe_apply_checkpoint_ema(
    model: torch.nn.Module,
    ckpt: dict[str, Any],
    sample_mode: str,
    *,
    log_prefix: str = "[ckpt]",
) -> bool:
    """Apply checkpoint EMA weights when requested and available.

    Returns True when EMA weights were applied.
    """
    if str(sample_mode).lower() != "ema":
        print(f"{log_prefix} Using regular model weights.")
        return False
    ema_state = ckpt.get("ema_state_dict", None)
    if ema_state is None:
        print(
            f"{log_prefix} EMA state not found; falling back to regular model weights."
        )
        return False
    apply_ema_state_dict(model, ema_state)
    print(f"{log_prefix} Using EMA weights.")
    return True


def checkpoint_type_encoding_name(
    ckpt: dict[str, Any], model_args: dict[str, Any], default: str = "atomic_number"
) -> str:
    return str(ckpt.get("type_encoding", model_args.get("type_encoding", default)))


def build_pxrd_eval_args(cfg: Any, model_args: dict[str, Any]) -> Namespace:
    """Assemble the Namespace expected by ``run_pxrd_match_eval``.

    Schedule and conditioning fields default to checkpoint ``model_args`` so
    offline eval reproduces in-training PXRD match@k unless cfg explicitly
    overrides a nullable field.
    """

    def pick(cfg_key: str, model_key: str, default: Any, cast):
        return cast(cfg_or_model_value(cfg, cfg_key, model_args, model_key, default))

    def ma(key: str, default: Any) -> Any:
        value = model_args.get(key, default)
        return default if value is None else value

    def cfg_get(key: str, default: Any = None) -> Any:
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def optional_schedule(key: str) -> float | None:
        value = cfg_get(key)
        if value is None:
            value = model_args.get(key)
        return None if value is None else float(value)

    return Namespace(
        monitor_topk=[int(k) for k in cfg_get("topk")],
        monitor_num_candidates=int(cfg_get("num_candidates")),
        monitor_chunk_size=int(cfg_get("chunk_size")),
        monitor_seed=int(cfg_get("seed")),
        sample_num_steps=pick("sample_num_steps", "sample_num_steps", 100, int),
        solver=str(cfg_get("solver", ma("solver", "heun"))).strip().lower(),
        sigma_min=pick("sigma_min", "sigma_min", 0.002, float),
        sigma_max=pick("sigma_max", "sigma_max", 80.0, float),
        rho=pick("rho", "rho", 7.0, float),
        S_churn=pick("S_churn", "S_churn", 20.0, float),
        S_min=pick("S_min", "S_min", 0.0, float),
        S_max=pick("S_max", "S_max", 999.0, float),
        S_noise=pick("S_noise", "S_noise", 1.0, float),
        aa_frac_max_scale=pick("aa_frac_max_scale", "aa_frac_max_scale", 0.0, float),
        aa_rho_types=pick("aa_rho_types", "aa_rho_types", 0.0, float),
        aa_rho_coords=pick("aa_rho_coords", "aa_rho_coords", 0.0, float),
        aa_rho_lattice=pick("aa_rho_lattice", "aa_rho_lattice", 0.0, float),
        sigma_min_types=optional_schedule("sigma_min_types"),
        sigma_max_types=optional_schedule("sigma_max_types"),
        rho_types=optional_schedule("rho_types"),
        sigma_min_coords=optional_schedule("sigma_min_coords"),
        sigma_max_coords=optional_schedule("sigma_max_coords"),
        rho_coords=optional_schedule("rho_coords"),
        sigma_min_lattice=optional_schedule("sigma_min_lattice"),
        sigma_max_lattice=optional_schedule("sigma_max_lattice"),
        rho_lattice=optional_schedule("rho_lattice"),
        sigma_data_type=float(ma("sigma_data_type", 1.0)),
        sigma_data_coord=float(ma("sigma_data_coord", 0.25)),
        sigma_data_lattice=float(ma("sigma_data_lattice", 1.0)),
        lattice_repr=str(ma("lattice_repr", "y1")),
        pxrd_cond=bool(ma("pxrd_cond", False)),
        bf16=bool(cfg_get("bf16")),
        matcher_ltol=float(cfg_get("matcher_ltol", 0.3)),
        matcher_stol=float(cfg_get("matcher_stol", 0.5)),
        matcher_angle_tol=float(cfg_get("matcher_angle_tol", 10.0)),
    )
