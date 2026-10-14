from __future__ import annotations
import math
import os
import struct
from collections.abc import Callable, Sequence
import torch
from src.bragglite.sampling_config import (
    resolve_aa_rho_pair,
    resolve_nonnegative_scalar,
    resolve_positive_scalar,
)
from src.bragglite.edm_utils import karras_sigma_steps, denoise_edm
from src.bragglite.model import BraggliteModel


def wrap_frac(delta: torch.Tensor) -> torch.Tensor:
    return delta - torch.round(delta)


def _logical_right_shift_64(values: torch.Tensor, bits: int) -> torch.Tensor:
    return values >> bits & (1 << 64 - bits) - 1


def _splitmix64(values: torch.Tensor) -> torch.Tensor:
    """Vectorized integer mixer with identical CPU/CUDA overflow semantics."""
    mixed = values - 7046029254386353131
    mixed = (mixed ^ _logical_right_shift_64(mixed, 30)) * -4658895280553007687
    mixed = (mixed ^ _logical_right_shift_64(mixed, 27)) * -7723592293110705685
    return mixed ^ _logical_right_shift_64(mixed, 31)


def _stateless_normal_bundle(
    seeds: torch.Tensor, width: int, *, stream: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generate one deterministic standard-normal row per seed without loops."""
    counters = torch.arange(width, device=seeds.device, dtype=torch.int64)[None]
    keys = seeds[:, None] ^ 2 * counters + 2000003 * int(stream)
    first = _logical_right_shift_64(_splitmix64(keys), 40).float()
    second = _logical_right_shift_64(_splitmix64(keys + 1), 40).float()
    uniform_1 = (first + 0.5) * 2.0 ** (-24)
    uniform_2 = (second + 0.5) * 2.0 ** (-24)
    normal = torch.sqrt(-2.0 * torch.log(uniform_1)) * torch.cos(
        2.0 * math.pi * uniform_2
    )
    return normal.to(dtype=dtype)


def clamp_lattice_latent(lat: torch.Tensor, lattice_repr: str) -> torch.Tensor:
    rep = lattice_repr.lower()
    if rep == "y1":
        lat[..., :3] = lat[..., :3].clamp(min=-5.0, max=5.0)
        lat[..., 3:] = lat[..., 3:].clamp(min=-0.999, max=0.999)
        return lat
    if rep == "ltri":
        lat[..., 0] = lat[..., 0].clamp(min=-5.0, max=5.0)
        lat[..., 2] = lat[..., 2].clamp(min=-5.0, max=5.0)
        lat[..., 5] = lat[..., 5].clamp(min=-5.0, max=5.0)
        lat[..., 1] = lat[..., 1].clamp(min=-10.0, max=10.0)
        lat[..., 3] = lat[..., 3].clamp(min=-10.0, max=10.0)
        lat[..., 4] = lat[..., 4].clamp(min=-10.0, max=10.0)
        return lat
    raise ValueError(f"Unsupported lattice representation: {lattice_repr}")


def project_type_values_to_allowed_prototypes(
    type_values: torch.Tensor,
    prototypes: torch.Tensor,
    allowed_mask: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Snap each real atom to its nearest allowed continuous type prototype."""
    if prototypes.ndim != 2 or prototypes.shape[1] != type_values.shape[-1]:
        raise ValueError(
            f"prototypes must be (V, type_dim) matching type_values; got {tuple(prototypes.shape)} and {tuple(type_values.shape)}."
        )
    if allowed_mask.shape != (type_values.shape[0], prototypes.shape[0]):
        raise ValueError(
            f"allowed_mask must be (B, V); got {tuple(allowed_mask.shape)} for B={type_values.shape[0]}, V={prototypes.shape[0]}."
        )
    allowed = allowed_mask.to(device=type_values.device, dtype=torch.bool)
    if not allowed.any(dim=1).all():
        raise ValueError("Each sample must allow at least one type prototype.")
    table = prototypes.to(device=type_values.device, dtype=type_values.dtype)
    scores = torch.einsum("bnd,vd->bnv", type_values, table)
    scores = scores.masked_fill(~allowed[:, None, :], -torch.inf)
    indices = scores.argmax(dim=-1)
    projected = table[indices]
    return torch.where(~pad_mask[..., None], projected, torch.zeros_like(projected))


@torch.no_grad()
@torch.no_grad()
def edm_sampler(
    model: BraggliteModel,
    pad_mask: torch.Tensor,
    type_dim: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    S_churn: float,
    S_min: float,
    S_max: float,
    S_noise: float,
    sigma_data_type: float,
    sigma_data_coord: float,
    sigma_data_lat: float,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
    generator_batch_sizes: Sequence[int] | None = None,
    sample_seeds: Sequence[int] | torch.Tensor | None = None,
    autocast_dtype: torch.dtype | None = None,
    fixed_atom_types: torch.Tensor | None = None,
    fixed_atom_type_mask: torch.Tensor | None = None,
    fixed_fractional_coords: torch.Tensor | None = None,
    fixed_fractional_coord_mask: torch.Tensor | None = None,
    skip_type_scaling: bool = False,
    aa_frac_max_scale: float = 0.0,
    aa_rho_coords: float = 0.0,
    aa_rho_lattice: float = 0.0,
    aa_rho_types: float = 0.0,
    lattice_repr: str = "y1",
    cond: dict[str, torch.Tensor] | None = None,
    cfg_scale: float = 1.0,
    cond_null: dict[str, torch.Tensor] | None = None,
    uncond_model: "BraggliteModel | None" = None,
    step_callback: Callable[[dict[str, object]], None] | None = None,
    formulation: str | None = None,
    solver: str = "heun",
    sigma_min_types: float | None = None,
    sigma_max_types: float | None = None,
    rho_types: float | None = None,
    sigma_min_coords: float | None = None,
    sigma_max_coords: float | None = None,
    rho_coords: float | None = None,
    sigma_min_lattice: float | None = None,
    sigma_max_lattice: float | None = None,
    rho_lattice: float | None = None,
    sampler_stats: dict[str, int | float | str] | None = None,
) -> dict[str, torch.Tensor]:
    base_model = getattr(model, "_orig_mod", model)
    resolved_formulation = (
        getattr(base_model, "formulation", "edm")
        if formulation is None
        else formulation
    )
    if str(resolved_formulation).strip().lower() != "edm":
        raise ValueError("Only the paper EDM sampler is supported.")
    device = pad_mask.device
    bsz, nmax = pad_mask.shape
    real_mask = ~pad_mask
    solver = str(solver).strip().lower()
    if solver not in {"euler", "heun"}:
        raise ValueError(f"solver must be euler|heun, got {solver!r}.")
    stateless_seeds = None
    if sample_seeds is not None:
        if generator_batch_sizes is not None:
            raise ValueError(
                "sample_seeds and generator_batch_sizes are mutually exclusive"
            )
        stateless_seeds = torch.as_tensor(
            sample_seeds, device=device, dtype=torch.int64
        ).reshape(-1)
        if stateless_seeds.numel() != bsz:
            raise ValueError("sample_seeds must contain exactly one seed per batch row")
    noise_widths = {"type": nmax * int(type_dim), "frac": nmax * 3, "lat": 6}

    def _stateless_noise(
        stream: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if stateless_seeds is None:
            raise RuntimeError("stateless noise requested without sample seeds")
        width = sum(noise_widths.values())
        flat = _stateless_normal_bundle(
            stateless_seeds, width, stream=stream, dtype=torch.float32
        )
        type_end = noise_widths["type"]
        frac_end = type_end + noise_widths["frac"]
        type_noise = flat[:, :type_end].reshape(bsz, nmax, type_dim)
        frac_noise = flat[:, type_end:frac_end].reshape(bsz, nmax, 3)
        lat_noise = flat[:, frac_end:].reshape(bsz, 6)
        return (type_noise, frac_noise, lat_noise)

    def _randn(
        shape: tuple[int, ...] | torch.Size, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if generator_batch_sizes is None:
            if isinstance(generator, Sequence):
                raise ValueError(
                    "generator_batch_sizes is required for generator lists"
                )
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)
        if not isinstance(generator, Sequence):
            raise ValueError(
                "generator must be a sequence when generator_batch_sizes is set"
            )
        sizes = [int(size) for size in generator_batch_sizes]
        if len(generator) != len(sizes) or sum(sizes) != int(shape[0]):
            raise ValueError(
                "generator list and batch sizes must cover the batch exactly"
            )
        tail = tuple(shape[1:])
        return torch.cat(
            [
                torch.randn((size, *tail), device=device, dtype=dtype, generator=item)
                for item, size in zip(generator, sizes, strict=True)
            ],
            dim=0,
        )

    def _resolve_fixed_slot_mask(values, mask, name):
        if values is None:
            if mask is not None:
                raise ValueError(f"{name} requires corresponding fixed values.")
            return None
        if values.shape[:2] != pad_mask.shape:
            raise ValueError(
                f"{name} values must start with shape {tuple(pad_mask.shape)}."
            )
        if mask is None:
            return real_mask
        if mask.shape != pad_mask.shape:
            raise ValueError(
                f"{name} must have shape {tuple(pad_mask.shape)}, got {tuple(mask.shape)}."
            )
        return mask.to(device=device, dtype=torch.bool) & real_mask

    fixed_type_mask = _resolve_fixed_slot_mask(
        fixed_atom_types, fixed_atom_type_mask, "fixed_atom_type_mask"
    )
    fixed_frac_mask = _resolve_fixed_slot_mask(
        fixed_fractional_coords,
        fixed_fractional_coord_mask,
        "fixed_fractional_coord_mask",
    )

    def _pin_slots(values, fixed_values, fixed_mask):
        if fixed_values is None:
            return values
        return torch.where(
            fixed_mask[..., None],
            fixed_values.to(device=device, dtype=values.dtype),
            values,
        )

    use_cfg = abs(float(cfg_scale) - 1.0) > 1e-09 and (
        uncond_model is not None or cond_null is not None
    )
    null_model = uncond_model if uncond_model is not None else model
    null_cond = None if uncond_model is not None else cond_null

    def _denoise(type_noisy, frac_noisy, lat_noisy, sigma):
        d_cond = denoise_edm(
            model=model,
            type_noisy=type_noisy,
            frac_noisy=frac_noisy,
            lat_noisy=lat_noisy,
            pad_mask=pad_mask,
            sigma=sigma,
            sigma_data_type=sigma_data_type,
            sigma_data_coord=sigma_data_coord,
            sigma_data_lat=sigma_data_lat,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            autocast_dtype=autocast_dtype,
            skip_type_scaling=skip_type_scaling,
            cond=cond,
        )
        if use_cfg:
            d_null = denoise_edm(
                model=null_model,
                type_noisy=type_noisy,
                frac_noisy=frac_noisy,
                lat_noisy=lat_noisy,
                pad_mask=pad_mask,
                sigma=sigma,
                sigma_data_type=sigma_data_type,
                sigma_data_coord=sigma_data_coord,
                sigma_data_lat=sigma_data_lat,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                autocast_dtype=autocast_dtype,
                skip_type_scaling=skip_type_scaling,
                cond=null_cond,
            )
            w = float(cfg_scale)
            result = {
                "type": d_cond["type"],
                "frac": d_null["frac"] + w * (d_cond["frac"] - d_null["frac"]),
                "lat": d_null["lat"] + w * (d_cond["lat"] - d_null["lat"]),
            }
        else:
            result = d_cond
        return result

    if stateless_seeds is None:
        type_x = _randn((bsz, nmax, type_dim))
        frac_x = _randn((bsz, nmax, 3))
        lat_x = _randn((bsz, 6))
    else:
        type_x, frac_x, lat_x = _stateless_noise(0)
    type_x = torch.where(real_mask[..., None], type_x, torch.zeros_like(type_x))
    frac_x = torch.where(real_mask[..., None], frac_x, torch.zeros_like(frac_x))
    shared_schedule = {
        "sigma_min": resolve_positive_scalar("sigma_min", sigma_min, 0.002),
        "sigma_max": resolve_positive_scalar("sigma_max", sigma_max, 80.0),
        "rho": resolve_positive_scalar("rho", rho, 7.0),
    }
    if shared_schedule["sigma_min"] > shared_schedule["sigma_max"]:
        raise ValueError("sigma_min must be <= sigma_max.")
    channel_overrides = {
        "types": (sigma_min_types, sigma_max_types, rho_types),
        "coords": (sigma_min_coords, sigma_max_coords, rho_coords),
        "lattice": (sigma_min_lattice, sigma_max_lattice, rho_lattice),
    }
    channel_schedules: dict[str, dict[str, float]] = {}
    t_steps_by_target: dict[str, torch.Tensor] = {}
    for target, (target_min, target_max, target_rho) in channel_overrides.items():
        schedule = {
            "sigma_min": resolve_positive_scalar(
                f"sigma_min_{target}", target_min, shared_schedule["sigma_min"]
            ),
            "sigma_max": resolve_positive_scalar(
                f"sigma_max_{target}", target_max, shared_schedule["sigma_max"]
            ),
            "rho": resolve_positive_scalar(
                f"rho_{target}", target_rho, shared_schedule["rho"]
            ),
        }
        if schedule["sigma_min"] > schedule["sigma_max"]:
            raise ValueError(f"sigma_min_{target} must be <= sigma_max_{target}.")
        channel_schedules[target] = schedule
        t_steps_by_target[target] = karras_sigma_steps(
            num_steps=num_steps,
            sigma_min=schedule["sigma_min"],
            sigma_max=schedule["sigma_max"],
            rho=schedule["rho"],
            device=device,
        )
    channel_schedule_groups: list[tuple[str, ...]] = []
    grouped_targets: dict[tuple[float, float, float], list[str]] = {}
    for target, schedule in channel_schedules.items():
        key = tuple((schedule[name] for name in ("sigma_min", "sigma_max", "rho")))
        grouped_targets.setdefault(key, []).append(target)
    channel_schedule_groups = [tuple(targets) for targets in grouped_targets.values()]
    aa_rho_coords_eff, aa_rho_lattice_eff = resolve_aa_rho_pair(
        aa_rho_coords=aa_rho_coords, aa_rho_lattice=aa_rho_lattice
    )
    aa_rho_types_eff = resolve_nonnegative_scalar("aa_rho_types", aa_rho_types)
    aa_rho_by_target = {
        "types": float(aa_rho_types_eff),
        "coords": float(aa_rho_coords_eff),
        "lattice": float(aa_rho_lattice_eff),
    }
    aa_scale_steps: dict[str, torch.Tensor] = {}
    for aa_target, aa_rho_target in aa_rho_by_target.items():
        if aa_rho_target <= 0.0:
            continue
        base_schedule = channel_schedules[aa_target]
        base_steps = t_steps_by_target[aa_target]
        aa_steps = karras_sigma_steps(
            num_steps=num_steps,
            sigma_min=base_schedule["sigma_min"],
            sigma_max=base_schedule["sigma_max"],
            rho=aa_rho_target,
            device=device,
        )
        t_main = base_steps[:-1]
        aa_main = aa_steps[:-1]
        if t_main.numel() >= 2:
            delta_base = t_main[1:] - t_main[:-1]
            delta_aa = aa_main[1:] - aa_main[:-1]
            scale_steps = (
                delta_aa.abs() / delta_base.abs().clamp_min(1e-12)
            ).clamp_min(1.0)
            scale_steps = torch.cat([scale_steps, scale_steps[-1:]], dim=0)
        else:
            scale_steps = torch.ones_like(t_main)
        if aa_target == "coords" and aa_frac_max_scale > 0.0:
            scale_steps = scale_steps.clamp(max=aa_frac_max_scale)
        aa_scale_steps[aa_target] = scale_steps
    type_next = type_x * t_steps_by_target["types"][0]
    frac_next = frac_x * t_steps_by_target["coords"][0]
    lat_next = lat_x * t_steps_by_target["lattice"][0]
    type_next = _pin_slots(type_next, fixed_atom_types, fixed_type_mask)
    frac_next = _pin_slots(frac_next, fixed_fractional_coords, fixed_frac_mask)
    _fast = os.environ.get("BRAGGLITE_FAST_SAMPLER", "0") == "1"
    _sqrt2m1 = math.sqrt(2.0) - 1.0
    t_sched_by_target = (
        {
            target: steps.detach().to("cpu", dtype=torch.float64).tolist()
            for target, steps in t_steps_by_target.items()
        }
        if _fast
        else None
    )

    def _f32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", float(value)))[0]

    def _step_values(target: str, index: int) -> dict[str, float | torch.Tensor]:
        if _fast:
            target_schedule = t_sched_by_target[target]
            t_cur_target = target_schedule[index]
            t_next_target = target_schedule[index + 1]
            in_range_target = S_min <= t_cur_target <= S_max
        else:
            target_steps = t_steps_by_target[target]
            t_cur_target = target_steps[index]
            t_next_target = target_steps[index + 1]
            in_range_target = bool(t_cur_target >= S_min and t_cur_target <= S_max)
        gamma_target = min(S_churn / num_steps, _sqrt2m1) if in_range_target else 0.0
        if _fast:
            t_hat_target = _f32(t_cur_target + _f32(_f32(gamma_target) * t_cur_target))
            step_delta_target = _f32(t_next_target - t_hat_target)
            noise_squared_target = _f32(
                _f32(t_hat_target * t_hat_target) - _f32(t_cur_target * t_cur_target)
            )
            noise_scale_target = _f32(math.sqrt(max(noise_squared_target, 0.0)))
        else:
            t_hat_target = t_cur_target + gamma_target * t_cur_target
            step_delta_target = t_next_target - t_hat_target
            noise_scale_target = (t_hat_target**2 - t_cur_target**2).sqrt()
        return {
            "current": t_cur_target,
            "hat": t_hat_target,
            "next": t_next_target,
            "delta": step_delta_target,
            "noise": noise_scale_target,
        }

    def _denoise_at_channel_sigmas(
        type_state: torch.Tensor,
        frac_state: torch.Tensor,
        lat_state: torch.Tensor,
        sigmas: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        combined: dict[str, torch.Tensor] = {}
        for targets in channel_schedule_groups:
            result = _denoise(type_state, frac_state, lat_state, sigmas[targets[0]])
            if "types" in targets:
                combined["type"] = result["type"]
            if "coords" in targets:
                combined["frac"] = result["frac"]
            if "lattice" in targets:
                combined["lat"] = result["lat"]
        return combined

    for i in range(num_steps):
        step_values = {
            target: _step_values(target, i) for target in ("types", "coords", "lattice")
        }
        if stateless_seeds is None:
            type_noise = _randn(type_next.shape, dtype=type_next.dtype)
            frac_noise = _randn(frac_next.shape, dtype=frac_next.dtype)
            lat_noise = _randn(lat_next.shape, dtype=lat_next.dtype)
        else:
            type_noise, frac_noise, lat_noise = _stateless_noise(i + 1)
        type_hat = type_next + step_values["types"]["noise"] * S_noise * type_noise
        frac_hat = frac_next + step_values["coords"]["noise"] * S_noise * frac_noise
        lat_hat = lat_next + step_values["lattice"]["noise"] * S_noise * lat_noise
        type_hat = _pin_slots(type_hat, fixed_atom_types, fixed_type_mask)
        frac_hat = _pin_slots(frac_hat, fixed_fractional_coords, fixed_frac_mask)
        sigma_hat = {
            target: torch.full((bsz,), float(values["hat"]), device=device)
            for target, values in step_values.items()
        }
        denoised = _denoise_at_channel_sigmas(type_hat, frac_hat, lat_hat, sigma_hat)
        type_d = denoised["type"]
        frac_d = denoised["frac"]
        lat_d = denoised["lat"]
        type_dx = (type_hat - type_d) / step_values["types"]["hat"]
        frac_dx = wrap_frac(frac_hat - frac_d) / step_values["coords"]["hat"]
        lat_dx = (lat_hat - lat_d) / step_values["lattice"]["hat"]
        scale_type_i = aa_scale_steps["types"][i] if "types" in aa_scale_steps else None
        scale_frac_i = (
            aa_scale_steps["coords"][i] if "coords" in aa_scale_steps else None
        )
        scale_lat_i = (
            aa_scale_steps["lattice"][i] if "lattice" in aa_scale_steps else None
        )
        if scale_type_i is not None:
            type_dx = scale_type_i * type_dx
        if scale_frac_i is not None:
            frac_dx = scale_frac_i * frac_dx
        if scale_lat_i is not None:
            lat_dx = scale_lat_i * lat_dx
        type_next = type_hat + step_values["types"]["delta"] * type_dx
        type_next = _pin_slots(type_next, fixed_atom_types, fixed_type_mask)
        frac_next = frac_hat + step_values["coords"]["delta"] * frac_dx
        frac_next = _pin_slots(frac_next, fixed_fractional_coords, fixed_frac_mask)
        lat_next = lat_hat + step_values["lattice"]["delta"] * lat_dx
        frac_next = frac_next - torch.round(frac_next)
        denoised_next = None
        if solver == "heun" and i < num_steps - 1:
            sigma_next = {
                target: torch.full((bsz,), float(values["next"]), device=device)
                for target, values in step_values.items()
            }
            denoised_next = _denoise_at_channel_sigmas(
                type_next, frac_next, lat_next, sigma_next
            )
            type_d2 = denoised_next["type"]
            frac_d2 = denoised_next["frac"]
            lat_d2 = denoised_next["lat"]
            type_dx2 = (type_next - type_d2) / step_values["types"]["next"]
            frac_dx2 = wrap_frac(frac_next - frac_d2) / step_values["coords"]["next"]
            lat_dx2 = (lat_next - lat_d2) / step_values["lattice"]["next"]
            if scale_type_i is not None:
                type_dx2 = scale_type_i * type_dx2
            if scale_frac_i is not None:
                frac_dx2 = scale_frac_i * frac_dx2
            if scale_lat_i is not None:
                lat_dx2 = scale_lat_i * lat_dx2
            type_next = type_hat + step_values["types"]["delta"] * (
                0.5 * type_dx + 0.5 * type_dx2
            )
            type_next = _pin_slots(type_next, fixed_atom_types, fixed_type_mask)
            frac_next = frac_hat + step_values["coords"]["delta"] * (
                0.5 * frac_dx + 0.5 * frac_dx2
            )
            frac_next = _pin_slots(frac_next, fixed_fractional_coords, fixed_frac_mask)
            lat_next = lat_hat + step_values["lattice"]["delta"] * (
                0.5 * lat_dx + 0.5 * lat_dx2
            )
            frac_next = frac_next - torch.round(frac_next)
        if step_callback is not None:
            step_callback(
                {
                    "step": i,
                    "num_steps": num_steps,
                    "sigma_schedule": float(step_values["types"]["current"]),
                    "sigma_hat": float(step_values["types"]["hat"]),
                    "sigma_next": float(step_values["types"]["next"]),
                    "channel_sigma_schedule": {
                        target: float(values["current"])
                        for target, values in step_values.items()
                    },
                    "channel_sigma_hat": {
                        target: float(values["hat"])
                        for target, values in step_values.items()
                    },
                    "channel_sigma_next": {
                        target: float(values["next"])
                        for target, values in step_values.items()
                    },
                    "solver": solver,
                    "x0_type": type_d,
                    "x0_frac": frac_d,
                    "x0_lat": lat_d,
                    "state_type": type_next,
                    "state_frac": frac_next,
                    "state_lat": lat_next,
                }
            )
    type_next = torch.where(
        real_mask[..., None], type_next, torch.zeros_like(type_next)
    )
    frac_next = torch.where(
        real_mask[..., None], frac_next, torch.zeros_like(frac_next)
    )
    lat_next = torch.nan_to_num(lat_next, nan=0.0, posinf=10.0, neginf=-10.0)
    type_next = torch.nan_to_num(type_next, nan=0.0, posinf=10.0, neginf=-10.0)
    frac_next = torch.nan_to_num(frac_next, nan=0.0, posinf=1.0, neginf=0.0)
    lat_next = clamp_lattice_latent(lat_next, lattice_repr=lattice_repr)
    type_next = type_next.clamp(min=-50.0, max=50.0)
    return {"type": type_next, "frac": frac_next, "lat": lat_next}
