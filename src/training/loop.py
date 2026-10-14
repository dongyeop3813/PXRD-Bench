"""Per-step Bragglite training and validation helpers."""

from __future__ import annotations
from contextlib import nullcontext
import torch
from src.pxrd.conditioning import condition_dict_from_batch
from src.bragglite.edm_utils import sample_sigma, denoise_edm, compute_edm_loss
from src.bragglite.sampler import wrap_frac
from src.models.lattice_repr import y1_to_lattice_latent


def _batch_to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _cond_from_batch(batch, args):
    """PXRD and element-set inputs from a collated batch."""
    return condition_dict_from_batch(batch, args)


def _apply_cond_dropout(cond, pxrd_p, elem_set_p=0.0):
    """Per-sample classifier-free-guidance condition dropout (training only).

    With probability ``pxrd_p`` each sample's PXRD peak list is replaced by the
    null (empty, all-padded) condition; with probability ``sg_p`` its space
    group is replaced by the null index 0. The model thereby learns a genuine
    unconditional branch alongside the conditional one, enabling proper CFG at
    inference. Probabilities ``<=0`` leave that signal untouched (bit-identical
    to no-dropout training). PXRD, element-set, and SG are dropped independently.
    """
    if cond is None:
        return cond
    out = dict(cond)
    if "pxrd_d" in out and float(pxrd_p) > 0.0:
        bsz = out["pxrd_d"].shape[0]
        drop = torch.rand(bsz, device=out["pxrd_d"].device) < float(pxrd_p)
        d2 = drop[:, None]
        out["pxrd_d"] = torch.where(d2, torch.zeros_like(out["pxrd_d"]), out["pxrd_d"])
        out["pxrd_i"] = torch.where(d2, torch.zeros_like(out["pxrd_i"]), out["pxrd_i"])
        for key in ("pxrd_position", "pxrd_intensity", "pxrd_prominence", "pxrd_width"):
            if key in out:
                out[key] = torch.where(d2, torch.zeros_like(out[key]), out[key])
        out["pxrd_peak_pad_mask"] = torch.where(
            d2, torch.ones_like(out["pxrd_peak_pad_mask"]), out["pxrd_peak_pad_mask"]
        )
    if "elem_set_mask" in out and float(elem_set_p) > 0.0:
        bsz = out["elem_set_mask"].shape[0]
        drop = torch.rand(bsz, device=out["elem_set_mask"].device) < float(elem_set_p)
        out["elem_set_mask"] = torch.where(
            drop[:, None], torch.zeros_like(out["elem_set_mask"]), out["elem_set_mask"]
        )
    return out


def _predict_clean(*, formulation: str, noise_level: torch.Tensor, **kwargs):
    return denoise_edm(sigma=noise_level, **kwargs)


def _train_microbatch(
    batch,
    *,
    model,
    type_encoding,
    weights,
    args,
    device,
    bf16_dtype,
    loss_scale: float = 1.0,
):
    """Run one microbatch forward/backward pass without an optimizer update."""
    batch = _batch_to_device(batch, device)
    formulation = str(getattr(args, "formulation", "edm")).strip().lower()
    sigma = sample_sigma(
        bsz=batch["A0"].shape[0],
        device=device,
        P_mean=args.edm_P_mean,
        P_std=args.edm_P_std,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    pad_mask = batch["pad_mask"].bool()
    real_mask = ~pad_mask
    type_clean = type_encoding.encode_from_A0(batch["A0"], pad_mask)
    frac_clean = batch["F1"]
    frac_clean_c = frac_clean - 0.5
    lat_clean = y1_to_lattice_latent(batch["Y1"], args.lattice_repr)
    g_type = torch.randn_like(type_clean)
    g_frac = torch.randn_like(frac_clean_c)
    g_lat = torch.randn_like(lat_clean)
    if args.csp:
        type_noisy = type_clean
    else:
        type_noisy = type_clean + sigma[:, None, None] * g_type
    frac_noisy = frac_clean_c + sigma[:, None, None] * g_frac
    frac_noisy = torch.where(
        real_mask[..., None], frac_noisy, torch.zeros_like(frac_noisy)
    )
    lat_noisy = lat_clean + sigma[:, None] * g_lat
    train_cond = _apply_cond_dropout(
        _cond_from_batch(batch, args),
        getattr(args, "pxrd_cond_dropout", 0.0),
        getattr(args, "element_set_cond_dropout", 0.0),
    )
    predict_kwargs = dict(
        model=model,
        type_noisy=type_noisy,
        frac_noisy=frac_noisy,
        lat_noisy=lat_noisy,
        pad_mask=pad_mask,
        sigma_data_type=args.sigma_data_type,
        sigma_data_coord=args.sigma_data_coord,
        sigma_data_lat=args.sigma_data_lattice,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        autocast_dtype=bf16_dtype,
        skip_type_scaling=args.csp,
        cond=train_cond,
    )
    denoised = _predict_clean(
        formulation=formulation, noise_level=sigma, **predict_kwargs
    )
    losses = compute_edm_loss(
        denoised=denoised,
        clean={"type": type_clean, "frac_c": frac_clean_c, "lat": lat_clean},
        frac_noisy=frac_noisy,
        sigma=sigma,
        pad_mask=pad_mask,
        sigma_data_type=args.sigma_data_type,
        sigma_data_coord=args.sigma_data_coord,
        sigma_data_lat=args.sigma_data_lattice,
        loss_weights=weights,
        coord_loss_mode=args.coord_loss_mode,
        lattice_repr=args.lattice_repr,
        formulation=formulation,
        edm_P_mean=args.edm_P_mean,
        edm_P_std=args.edm_P_std,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    (losses["loss_total"] * float(loss_scale)).backward()
    aux = {
        "denoised": denoised,
        "sigma": sigma,
        "pad_mask": pad_mask,
        "real_mask": real_mask,
        "frac_clean_c": frac_clean_c,
        "lat_clean": lat_clean,
        "A0": batch["A0"],
        "formulation": formulation,
    }
    return (losses, aux)


def train_step(
    batch,
    *,
    model,
    optimizer,
    scheduler,
    ema,
    type_encoding,
    weights,
    args,
    device,
    bf16_dtype,
    collect_metrics: bool = False,
):
    """Run one optimizer step over one or more accumulated microbatches.

    Losses are normalized by the number of microbatches. Gradient clipping,
    optimizer/scheduler stepping, and EMA update happen exactly once. Under
    DDP, non-final microbatches use ``no_sync`` to avoid redundant all-reduces.
    """
    microbatches = list(batch) if isinstance(batch, (list, tuple)) else [batch]
    if not microbatches:
        raise ValueError("train_step requires at least one microbatch.")
    optimizer.zero_grad(set_to_none=True)
    metric_sums: dict[str, float] = {}
    losses = aux = None
    loss_scale = 1.0 / len(microbatches)
    no_sync = getattr(model, "no_sync", None)
    for index, microbatch in enumerate(microbatches):
        should_sync = index + 1 == len(microbatches)
        sync_context = (
            no_sync() if not should_sync and callable(no_sync) else nullcontext()
        )
        with sync_context:
            losses, aux = _train_microbatch(
                microbatch,
                model=model,
                type_encoding=type_encoding,
                weights=weights,
                args=args,
                device=device,
                bf16_dtype=bf16_dtype,
                loss_scale=loss_scale,
            )
        if collect_metrics:
            micro_metrics = compute_train_metrics(
                losses,
                aux,
                weights=weights,
                optimizer=optimizer,
                type_encoding=type_encoding,
                csp=bool(args.csp),
            )
            for key, value in micro_metrics.items():
                if key != "gradient/norm":
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
    parameters = list(model.parameters())
    clip_mode = str(getattr(args, "grad_clip_mode", "norm")).strip().lower()
    if clip_mode == "value":
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        grad_norm = torch.nn.utils.get_total_norm(
            gradients, foreach=bool(getattr(args, "foreach_grad_clip", True))
        )
        torch.nn.utils.clip_grad_value_(
            parameters,
            clip_value=float(getattr(args, "grad_clip_norm", 1.0)),
            foreach=bool(getattr(args, "foreach_grad_clip", True)),
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float(getattr(args, "grad_clip_norm", 1.0)),
            foreach=bool(getattr(args, "foreach_grad_clip", True)),
        )
    optimizer.step()
    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    if ema is not None:
        ema.update(model)
    assert losses is not None and aux is not None
    aux["grad_norm"] = grad_norm.detach()
    if collect_metrics:
        aux["_accumulated_metrics"] = {
            key: value / len(microbatches) for key, value in metric_sums.items()
        }
        aux["_accumulated_metrics"]["gradient/norm"] = float(grad_norm.item())
    return (losses, aux)


def compute_train_metrics(losses, aux, *, weights, optimizer, type_encoding, csp: bool):
    """Train-time scalar metrics (loss components, type accuracy, coord/lattice L2)."""
    accumulated = aux.get("_accumulated_metrics")
    if accumulated is not None:
        return dict(accumulated)
    denoised = aux["denoised"]
    sigma = aux["sigma"]
    pad_mask = aux["pad_mask"]
    real_mask = aux["real_mask"]
    frac_clean_c = aux["frac_clean_c"]
    lat_clean = aux["lat_clean"]
    metrics = {
        "loss/total": float(losses["loss_total"].item()),
        "loss/coord": float(losses["loss_f"].item() * weights["F"]),
        "loss/lattice": float(losses["loss_y"].item() * weights["Y"]),
        "lr": optimizer.param_groups[0]["lr"],
        "sigma/mean": float(sigma.mean().item()),
        "sigma/std": float(sigma.std().item()),
    }
    if "grad_norm" in aux:
        metrics["gradient/norm"] = float(aux["grad_norm"].item())
    if not csp:
        metrics["loss/type"] = float(losses["loss_a"].item() * weights["A"])
    if real_mask.any():
        if not csp:
            type_pred = type_encoding.decode_logits_to_A0(
                type_logits=denoised["type"], pad_mask=pad_mask
            )
            target_zero = aux["A0"]
            correct = (type_pred == target_zero) & real_mask
            metrics["stats/type_acc"] = float(
                correct.float().sum().item() / real_mask.sum().item()
            )
        frac_delta = wrap_frac(denoised["frac"] - frac_clean_c)
        mask_exp = real_mask[..., None].float()
        metrics["stats/coord_l2"] = float(
            (frac_delta**2 * mask_exp).sum().item()
            / mask_exp.sum().clamp_min(1.0).item()
        )
    else:
        if not csp:
            metrics["stats/type_acc"] = 0.0
        metrics["stats/coord_l2"] = 0.0
    metrics["stats/lattice_l2"] = float(
        ((denoised["lat"] - lat_clean) ** 2).mean().item()
    )
    return metrics


def run_validation(
    val_loader, *, model, type_encoding, weights, args, device, bf16_dtype
):
    """Average EDM losses over up to ``args.val_batches`` validation batches.

    Sets ``model.eval()``; the caller restores ``model.train()`` afterwards.
    """
    model.eval()
    val_losses = {"loss_total": 0.0, "loss_a": 0.0, "loss_f": 0.0, "loss_y": 0.0}
    num = 0
    formulation = str(getattr(args, "formulation", "edm")).strip().lower()
    with torch.no_grad():
        for v_idx, v_batch in enumerate(val_loader):
            if v_idx >= args.val_batches:
                break
            v_batch = _batch_to_device(v_batch, device)
            sigma_v = sample_sigma(
                bsz=v_batch["A0"].shape[0],
                device=device,
                P_mean=args.edm_P_mean,
                P_std=args.edm_P_std,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
            )
            pad_v = v_batch["pad_mask"].bool()
            real_v = ~pad_v
            type_clean_v = type_encoding.encode_from_A0(v_batch["A0"], pad_v)
            frac_clean_v = v_batch["F1"]
            frac_clean_v_c = frac_clean_v - 0.5
            lat_clean_v = y1_to_lattice_latent(v_batch["Y1"], args.lattice_repr)
            type_noise_v = torch.randn_like(type_clean_v)
            frac_noise_v = torch.randn_like(frac_clean_v_c)
            lat_noise_v = torch.randn_like(lat_clean_v)
            if args.csp:
                type_noisy_v = type_clean_v
            else:
                type_noisy_v = type_clean_v + sigma_v[:, None, None] * type_noise_v
            frac_noisy_v = frac_clean_v_c + sigma_v[:, None, None] * frac_noise_v
            frac_noisy_v = torch.where(
                real_v[..., None], frac_noisy_v, torch.zeros_like(frac_noisy_v)
            )
            lat_noisy_v = lat_clean_v + sigma_v[:, None] * lat_noise_v
            predict_kwargs_v = dict(
                model=model,
                type_noisy=type_noisy_v,
                frac_noisy=frac_noisy_v,
                lat_noisy=lat_noisy_v,
                pad_mask=pad_v,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                autocast_dtype=bf16_dtype,
                skip_type_scaling=args.csp,
                cond=_cond_from_batch(v_batch, args),
            )
            first_pass_v = _predict_clean(
                formulation=formulation, noise_level=sigma_v, **predict_kwargs_v
            )

            def validation_losses(prediction):
                losses = compute_edm_loss(
                    denoised=prediction,
                    clean={
                        "type": type_clean_v,
                        "frac_c": frac_clean_v_c,
                        "lat": lat_clean_v,
                    },
                    frac_noisy=frac_noisy_v,
                    sigma=sigma_v,
                    pad_mask=pad_v,
                    sigma_data_type=args.sigma_data_type,
                    sigma_data_coord=args.sigma_data_coord,
                    sigma_data_lat=args.sigma_data_lattice,
                    loss_weights=weights,
                    coord_loss_mode=args.coord_loss_mode,
                    lattice_repr=args.lattice_repr,
                    formulation=formulation,
                    edm_P_mean=args.edm_P_mean,
                    edm_P_std=args.edm_P_std,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                )
                return losses

            v_losses = validation_losses(first_pass_v)
            for k in ("loss_total", "loss_a", "loss_f", "loss_y"):
                val_losses[k] += float(v_losses[k].item())
            num += 1
    if num > 0:
        for k in val_losses:
            val_losses[k] /= num
    return val_losses
