from __future__ import annotations
import torch
from contextlib import nullcontext
from src.bragglite.model import BraggliteModel, mod1
from src.models.lattice_repr import lattice_latent_to_gram, y1_to_loss_features


def sample_sigma(
    bsz: int,
    device: torch.device,
    P_mean: float,
    P_std: float,
    sigma_min: float,
    sigma_max: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    log_sigma = (
        torch.randn((bsz,), device=device, generator=generator, dtype=torch.float32)
        * P_std
        + P_mean
    )
    sigma = log_sigma.exp()
    if sigma_min > 0:
        sigma = sigma.clamp_min(sigma_min)
    if sigma_max > 0:
        sigma = sigma.clamp_max(sigma_max)
    return sigma


def wrap_centered(value: torch.Tensor) -> torch.Tensor:
    """Wrap fractional values or displacements to [-0.5, 0.5)."""
    return value - torch.round(value)


def sigma_to_cnoise(sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return sigma.clamp_min(eps).log() * 0.25


def karras_sigma_steps(
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0 for edm_sampler.")
    if num_steps == 1:
        steps = torch.tensor([float(sigma_max)], device=device, dtype=torch.float32)
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
    return torch.cat([steps, torch.zeros_like(steps[:1])])


def denoise_edm(
    model: BraggliteModel,
    type_noisy: torch.Tensor,
    frac_noisy: torch.Tensor,
    lat_noisy: torch.Tensor,
    pad_mask: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data_type: float,
    sigma_data_coord: float,
    sigma_data_lat: float,
    sigma_min: float,
    sigma_max: float,
    autocast_dtype: torch.dtype | None = None,
    skip_type_scaling: bool = False,
    cond: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    EDM preconditioning for all MP20 variables.

    ``cond`` is forwarded to the model as extra keyword inputs (e.g. the PXRD
    d-I peak list ``pxrd_d``/``pxrd_i``/``pxrd_peak_pad_mask``); ``None``
    preserves the unconditional call exactly.
    """
    type_noisy = type_noisy.to(dtype=torch.float32)
    frac_noisy = frac_noisy.to(dtype=torch.float32)
    lat_noisy = lat_noisy.to(dtype=torch.float32)
    sigma = sigma.to(dtype=torch.float32)
    bsz = type_noisy.shape[0]
    sigma_b = sigma.view(bsz, 1, 1)
    sigma_lat = sigma.view(bsz, 1)

    def coeffs(sig: torch.Tensor, sigma_data: float):
        c_skip = sigma_data**2 / (sig**2 + sigma_data**2)
        c_out = sig * sigma_data / torch.sqrt(sig**2 + sigma_data**2)
        c_in = 1.0 / torch.sqrt(sig**2 + sigma_data**2)
        return (c_skip, c_out, c_in)

    c_skip_t, c_out_t, c_in_t = coeffs(sigma_b, sigma_data_type)
    c_skip_f, c_out_f, _ = coeffs(sigma_b, sigma_data_coord)
    c_skip_y, c_out_y, c_in_y = coeffs(sigma_lat, sigma_data_lat)
    type_in = type_noisy if skip_type_scaling else c_in_t * type_noisy
    lat_in = c_in_y * lat_noisy
    del sigma_min, sigma_max
    t_sigma = sigma_to_cnoise(sigma).to(dtype=torch.float32)
    frac_in = mod1(frac_noisy + 0.5)
    device_type = next(model.parameters()).device.type
    amp_ctx = (
        torch.autocast("cuda", dtype=autocast_dtype)
        if autocast_dtype is not None and device_type == "cuda"
        else nullcontext()
    )
    model_kwargs = dict(cond or {})
    with amp_ctx:
        raw = model(
            type_in,
            frac_in,
            lat_in,
            pad_mask,
            t_sigma,
            lattice_bias_feats=lat_noisy,
            **model_kwargs,
        )
    type_raw = raw["type_logits"]
    frac_raw = raw["coord_vel"]
    lat_raw = raw["lattice_vel"]
    D_type = c_skip_t * type_noisy + c_out_t * type_raw
    D_frac = c_skip_f * frac_noisy + c_out_f * frac_raw
    D_lat = c_skip_y * lat_noisy + c_out_y * lat_raw
    return {"type": D_type, "frac": D_frac, "lat": D_lat, "raw": raw}


def compute_edm_loss(
    denoised: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor],
    frac_noisy: torch.Tensor,
    sigma: torch.Tensor,
    pad_mask: torch.Tensor,
    sigma_data_type: float,
    sigma_data_coord: float,
    sigma_data_lat: float,
    loss_weights: dict[str, float],
    coord_loss_mode: str = "cart_metric_vnorm_com",
    lattice_repr: str = "y1",
    formulation: str = "edm",
    edm_P_mean: float = -1.2,
    edm_P_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> dict[str, torch.Tensor]:
    device = sigma.device
    real_mask = ~pad_mask
    sigma_b = sigma.view(-1, 1, 1)
    sigma_lat = sigma.view(-1, 1)
    atoms_per_sample_y = real_mask.float().sum(dim=1).clamp_min(1.0)

    def masked_token_reduce(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sum feature errors per real token, matching upstream Bragglite."""
        if values.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=values.dtype)
        denom = mask.sum()
        return (values * mask).sum() / denom.clamp_min(1.0)

    def weight(sig: torch.Tensor, sigma_data: float):
        return (sig**2 + sigma_data**2) / (sig * sigma_data) ** 2

    weight_t = weight(sigma_b, sigma_data_type)
    weight_f = weight(sigma_b, sigma_data_coord)
    weight_y = weight(sigma_lat, sigma_data_lat)
    zero = torch.tensor(0.0, device=device)
    clean_type = clean["type"]
    clean_frac = clean["frac_c"]
    mask_exp = real_mask[..., None].float()
    if real_mask.any():
        endpoint_error_type = denoised["type"] - clean_type
        err_t = endpoint_error_type.square()
        endpoint_diff = denoised["frac"] - clean_frac
        endpoint_diff = endpoint_diff - torch.round(endpoint_diff)
        loss_t = masked_token_reduce(weight_t * err_t, mask_exp)
        if coord_loss_mode == "frac_mse":
            err_f = endpoint_diff.square()
            loss_f = masked_token_reduce(weight_f * err_f, mask_exp)
        elif coord_loss_mode == "cart_metric_vnorm_com":
            atom_mask = real_mask.float()
            atoms_per_sample = atom_mask.sum(dim=1).clamp_min(1.0)
            gram = lattice_latent_to_gram(clean["lat"], lattice_repr=lattice_repr).to(
                dtype=endpoint_diff.dtype
            )
            gram = torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
            vol = torch.sqrt(torch.linalg.det(gram).clamp_min(1e-12))
            scale = (vol / atoms_per_sample).clamp_min(1e-12).pow(1.0 / 3.0)
            weight_f_atom = weight_f[:, 0, 0]

            def cart_metric_loss(delta: torch.Tensor) -> torch.Tensor:
                mean_delta = (delta * atom_mask[..., None]).sum(
                    dim=1
                ) / atoms_per_sample[:, None]
                centered = (delta - mean_delta[:, None, :]) * atom_mask[..., None]
                dist2_cart = torch.einsum("bni,bij,bnj->bn", centered, gram, centered)
                dist2_cart = torch.nan_to_num(
                    dist2_cart, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp_min(0.0)
                values = (
                    weight_f_atom[:, None]
                    * dist2_cart
                    / scale[:, None].square()
                    * atom_mask
                )
                return values.sum() / atom_mask.sum().clamp_min(1.0)

            loss_f = cart_metric_loss(endpoint_diff)
        else:
            raise ValueError(f"Unsupported coord_loss_mode: {coord_loss_mode}")
    else:
        loss_t = zero
        loss_f = zero
    if lattice_repr.lower() == "y1":
        denoised_y = y1_to_loss_features(denoised["lat"], num_atoms=atoms_per_sample_y)
        clean_y = y1_to_loss_features(clean["lat"], num_atoms=atoms_per_sample_y)
        endpoint_err_y = (denoised_y - clean_y).square()
    else:
        endpoint_err_y = (denoised["lat"] - clean["lat"]).square()
    loss_y = (
        (weight_y * endpoint_err_y).mean(dim=1).mean()
        if endpoint_err_y.numel() > 0
        else zero
    )
    total = (
        loss_weights["A"] * loss_t
        + loss_weights["F"] * loss_f
        + loss_weights["Y"] * loss_y
    )
    return {"loss_total": total, "loss_a": loss_t, "loss_f": loss_f, "loss_y": loss_y}
