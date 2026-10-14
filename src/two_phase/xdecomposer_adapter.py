"""Thin adapter around the official XDecomposer V_0.0.1 implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks

from src.pxrd.peaks import peaks_to_di_tensors, xrdiff_feature_vector_from_profile


CRYSTAL_GRID = np.linspace(5.0, 80.0, 7500, dtype=np.float32)
XDECOMPOSER_GRID = np.linspace(5.0, 90.0, 3500, dtype=np.float32)
PROFILE_STRONG_GRID = np.arange(5.0, 90.0, 0.1, dtype=np.float32)


def conserve_mixture(
    profiles: torch.Tensor, mixture: torch.Tensor, *, minimum_total: float = 1e-12
) -> torch.Tensor:
    """Normalize nonnegative slot contributions, including zero-signal bins."""
    with torch.autocast(profiles.device.type, enabled=False):
        profiles = profiles.float()
        total = profiles.sum(dim=1, keepdim=True)
        # Subnormal Gaussian tails can underflow the division's backward square.
        # Treat negligible bins like zero bins while still conserving the mixture.
        positive = total > minimum_total
        weights = profiles / torch.where(positive, total, torch.ones_like(total))
        weights = torch.where(
            positive, weights, torch.full_like(weights, 1 / profiles.shape[1])
        )
        return weights * mixture.float()


class XDecomposerVariant(torch.nn.Module):
    """Ordered XDecomposer with additive mixture conservation.

    Element-superset conditioning belongs to BraggliteModel. The selected
    XDecomposer checkpoint uses shared_elements=False, normalize_masks=True.
    """

    def __init__(
        self, backbone: torch.nn.Module, *, shared_elements: bool, normalize_masks: bool
    ):
        super().__init__()
        self.backbone = backbone
        self.normalize_masks = normalize_masks
        if shared_elements or not normalize_masks:
            raise ValueError(
                "the released XDecomposer uses unconditioned, normalized masks"
            )

    def forward(
        self, mixture: torch.Tensor, *, shared_elements: torch.Tensor | None = None
    ):
        if shared_elements is not None:
            raise ValueError("this model does not use shared elements")
        profiles, activity = self.backbone(mixture)
        profiles = conserve_mixture(profiles, mixture)
        return profiles, activity


def resample_to_xdecomposer(profile: np.ndarray) -> np.ndarray:
    """Map Bragglite's 5--80 degree profile to 5--90, padding the tail."""
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (7500,):
        raise ValueError(f"expected a 7500-bin profile, got {profile.shape}")
    return np.interp(
        XDECOMPOSER_GRID, CRYSTAL_GRID, profile, left=0.0, right=0.0
    ).astype(np.float32)


def resample_from_xdecomposer(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (3500,):
        raise ValueError(f"expected a 3500-bin profile, got {profile.shape}")
    return np.interp(CRYSTAL_GRID, XDECOMPOSER_GRID, profile).astype(np.float32)


def profile_to_di(
    profile: np.ndarray, *, max_peaks: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract local maxima and reuse Bragglite's canonical top-k d--I packer."""
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (7500,) or not np.isfinite(profile).all():
        raise ValueError("profile must be a finite 7500-bin vector")
    maximum = float(profile.max(initial=0.0))
    normalized = profile / maximum if maximum > 0.0 else profile
    indices, _ = find_peaks(normalized, prominence=0.0)
    if not indices.size and maximum > 0.0:
        indices = np.asarray([int(np.argmax(normalized))])
    return peaks_to_di_tensors(
        CRYSTAL_GRID[indices],
        normalized[indices],
        max_peaks=max_peaks,
        two_theta_min=5.0,
        two_theta_max=80.0,
    )


def profile_to_peak128x4(
    profile: np.ndarray, *, two_theta_max: float = 80.0
) -> np.ndarray:
    """Encode an XDecomposer slot with the checkpoint's native Profile Strong input."""
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (3500,) or not np.isfinite(profile).all():
        raise ValueError("profile must be a finite 3500-bin vector")
    if not 5.0 < float(two_theta_max) <= 90.0:
        raise ValueError("two_theta_max must be in (5, 90]")
    cropped = profile.copy()
    cropped[XDECOMPOSER_GRID > float(two_theta_max)] = 0.0
    profile_850 = np.interp(PROFILE_STRONG_GRID, XDECOMPOSER_GRID, cropped)
    return xrdiff_feature_vector_from_profile(
        profile_850.astype(np.float32), "peaks_128x4"
    )


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_model(
    source_root: str | Path,
    separation_checkpoint: str | Path,
    mae_checkpoint: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load official source and release weights without vendoring either."""
    source_root = Path(source_root)
    transformer = _load_module(
        "xdecomposer_xrd_transformer",
        source_root / "src/models/xrd_transformer.py",
    )
    hybrid = _load_module(
        "xdecomposer_hybrid_film", source_root / "src/models/hybrid_film.py"
    )
    separation = torch.load(
        separation_checkpoint, map_location="cpu", weights_only=False
    )
    mae_payload = torch.load(mae_checkpoint, map_location="cpu", weights_only=False)
    config = dict(separation["config"])
    mae_config = dict(mae_payload.get("config", {}))
    mae = transformer.XRDMaskedAutoencoder(
        xrd_length=config["xrd_length"],
        d_model=mae_config.get("d_model", 768),
        n_layers=mae_config.get("n_layers", 4),
        n_heads=mae_config.get("n_heads", 12),
        decoder_d_model=mae_config.get("decoder_dim", 512),
        decoder_n_layers=mae_config.get("decoder_layers", 4),
    )
    model = hybrid.build_hybrid_model(
        mae,
        num_sources=config["num_phases"],
        cnn_channels=config.get("cnn_channels", [64, 128, 256, 512]),
        cnn_kernels=config.get("cnn_kernels"),
        cnn_strides=config.get("cnn_strides"),
    )
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in separation["model_state_dict"].items()
    }
    model.load_state_dict(state)
    return model.to(device).eval(), config
