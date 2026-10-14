"""Universal XRD encoder over discrete d-I peak lists.

Faithful to RealPXRD-Solver Methods 4.3 (related_works/real_pxrd_solver.pdf):
each peak's continuous d-spacing and intensity scalars are projected by two
independent MLPs and summed into one peak feature; a learnable [CLS] token is
prepended; a multi-layer Transformer encoder models contextual dependencies
among all peaks; the final [CLS] hidden state is the holistic pattern latent,
projected to the conditioning width (added to the time embedding feeding adaLN
in :class:`src.bragglite.model.BraggliteModel`).

Peak order is not encoded. The flat q-I transformer optionally adds an RBF
or Fourier embedding of physical q; all peak encoders still treat peaks as a set.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.embeddings import sinusoidal_position_embedding


_D_FEATURE_MODES = {"raw", "metric", "fourier_q", "metric_fourier_q"}


def _d_feature_width(mode: str, n_freqs: int) -> int:
    if mode == "raw":
        return 1
    if mode == "metric":
        return 4
    base = 2 if mode == "fourier_q" else 5
    return base + 2 * int(n_freqs)


def _encode_d_features(
    pxrd_d: torch.Tensor,
    *,
    mode: str,
    wavelength: float,
    freqs: torch.Tensor | None,
) -> torch.Tensor:
    """Continuous d-spacing features; callers mask padded peaks afterwards."""
    if mode == "raw":
        return pxrd_d[..., None]

    safe_d = torch.where(pxrd_d > 0.0, pxrd_d, torch.ones_like(pxrd_d))
    reciprocal = safe_d.reciprocal()
    if mode == "metric":
        return torch.stack(
            [safe_d, reciprocal, reciprocal.square(), safe_d.log()], dim=-1
        )

    if freqs is None:
        raise RuntimeError(f"Fourier frequencies are missing for mode={mode!r}.")
    q_norm = (0.5 * float(wavelength) * reciprocal).clamp(0.0, 1.0)
    args = (
        2.0
        * math.pi
        * q_norm[..., None]
        * freqs.to(device=pxrd_d.device, dtype=pxrd_d.dtype)
    )
    fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if mode == "fourier_q":
        base = torch.stack([safe_d, q_norm], dim=-1)
    else:
        base = torch.stack(
            [safe_d, reciprocal, reciprocal.square(), safe_d.log(), q_norm], dim=-1
        )
    return torch.cat([base, fourier], dim=-1)


class PXRDPeakEncoder(nn.Module):
    def __init__(
        self,
        d_out: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        d_feature_mode: str = "raw",
        d_fourier_n_freqs: int = 16,
        wavelength: float = 1.54184,
        representation: str = "legacy_di",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"pxrd encoder d_model={d_model} not divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.representation = str(representation)
        if self.representation not in {
            "legacy_di",
            "broadened_basic",
            "broadened_shape",
        }:
            raise ValueError(f"unsupported PXRD representation: {representation}")
        if d_feature_mode not in _D_FEATURE_MODES:
            raise ValueError(f"Unsupported d_feature_mode={d_feature_mode!r}")
        self.d_feature_mode = d_feature_mode
        self.d_fourier_n_freqs = int(d_fourier_n_freqs)
        if self.d_fourier_n_freqs <= 0:
            raise ValueError("d_fourier_n_freqs must be positive.")
        self.wavelength = float(wavelength)
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive.")
        d_fourier_freqs = (
            torch.arange(1, self.d_fourier_n_freqs + 1, dtype=torch.float32)
            if "fourier_q" in self.d_feature_mode
            else None
        )
        self.register_buffer("d_fourier_freqs", d_fourier_freqs, persistent=False)

        def scalar_mlp(in_features: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, d_model, bias=True),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
            )

        self.mlp_d = scalar_mlp(
            _d_feature_width(self.d_feature_mode, self.d_fourier_n_freqs)
        )
        self.mlp_i = scalar_mlp()
        self.mlp_prominence = (
            scalar_mlp() if self.representation != "legacy_di" else None
        )
        self.mlp_width = scalar_mlp() if self.representation != "legacy_di" else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out, bias=True)

    def encode_tokens(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token encoder states (for cross-attended condition pooling).

        Args:
            pxrd_d: (B, P) d-spacings (Angstrom), 0 in padded slots.
            pxrd_i: (B, P) max-normalized intensities in [0, 1], 0 in padded slots.
            pxrd_peak_pad_mask: (B, P) bool, True where padded.
        Returns:
            tokens: (B, 1+P, d_out) projected hidden states ([CLS] first).
            pad_seq: (B, 1+P) bool key-padding mask (True where padded; [CLS]
                is never padded).
        """
        if pxrd_d.dim() != 2 or pxrd_d.shape != pxrd_i.shape:
            raise RuntimeError(
                f"pxrd_d/pxrd_i must be matching (B, P), got {tuple(pxrd_d.shape)} vs {tuple(pxrd_i.shape)}"
            )
        pad = pxrd_peak_pad_mask.bool()
        if pad.shape != pxrd_d.shape:
            raise RuntimeError(
                f"pxrd_peak_pad_mask must be (B, P)={tuple(pxrd_d.shape)}, got {tuple(pad.shape)}"
            )
        d_features = _encode_d_features(
            pxrd_d,
            mode=self.d_feature_mode,
            wavelength=self.wavelength,
            freqs=self.d_fourier_freqs,
        )
        f_peak = self.mlp_d(d_features) + self.mlp_i(pxrd_i[..., None])  # (B, P, D)
        if self.representation != "legacy_di":
            if pxrd_prominence is None or pxrd_width is None:
                raise RuntimeError(
                    f"PXRD representation {self.representation} requires prominence/width inputs"
                )
            shape = self.mlp_prominence(pxrd_prominence[..., None]) + self.mlp_width(
                pxrd_width[..., None]
            )
            f_peak = f_peak + (
                shape if self.representation == "broadened_shape" else shape * 0.0
            )
        f_peak = f_peak.masked_fill(pad[..., None], 0.0)

        bsz = f_peak.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1).to(dtype=f_peak.dtype)
        x = torch.cat([cls, f_peak], dim=1)  # (B, 1+P, D)
        pad_seq = torch.cat(
            [torch.zeros((bsz, 1), device=pad.device, dtype=torch.bool), pad], dim=1
        )
        h = self.encoder(x, src_key_padding_mask=pad_seq)
        # LayerNorm + Linear act per token, so tokens[:, 0] equals the legacy
        # CLS-only path bit-for-bit.
        return self.out_proj(self.norm_out(h)), pad_seq

    def forward(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, d_out) conditioning vector (the projected [CLS] hidden state)."""
        tokens, _ = self.encode_tokens(
            pxrd_d, pxrd_i, pxrd_peak_pad_mask, pxrd_prominence, pxrd_width
        )
        return tokens[:, 0]


class PXRDPeakTokenEncoder(nn.Module):
    """Encode each d-I peak independently as a DiT context token.

    Unlike :class:`PXRDPeakEncoder`, this module does not add a CLS token and
    does not contextualize peaks with a PXRD-side Transformer. The returned
    sequence is meant to be inserted directly into the Bragglite DiT token
    stream, where atom/lattice tokens can attend to the peak tokens in-context.
    """

    def __init__(
        self,
        d_out: int,
        d_model: int = 256,
        dropout: float = 0.0,
        d_feature_mode: str = "raw",
        d_fourier_n_freqs: int = 16,
        wavelength: float = 1.54184,
        representation: str = "legacy_di",
    ) -> None:
        super().__init__()
        self.representation = str(representation)
        if self.representation not in {
            "legacy_di",
            "broadened_basic",
            "broadened_shape",
        }:
            raise ValueError(f"unsupported PXRD representation: {representation}")

        if d_feature_mode not in _D_FEATURE_MODES:
            raise ValueError(f"Unsupported d_feature_mode={d_feature_mode!r}")
        self.d_feature_mode = str(d_feature_mode)
        self.d_fourier_n_freqs = int(d_fourier_n_freqs)
        if self.d_fourier_n_freqs <= 0:
            raise ValueError("d_fourier_n_freqs must be positive.")
        self.wavelength = float(wavelength)
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive.")
        d_fourier_freqs = (
            torch.arange(1, self.d_fourier_n_freqs + 1, dtype=torch.float32)
            if "fourier_q" in self.d_feature_mode
            else None
        )
        self.register_buffer("d_fourier_freqs", d_fourier_freqs, persistent=False)

        def scalar_mlp(in_features: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, d_model, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_out, bias=True),
            )

        self.mlp_d = scalar_mlp(
            _d_feature_width(self.d_feature_mode, self.d_fourier_n_freqs)
        )
        self.mlp_i = scalar_mlp()
        self.mlp_prominence = (
            scalar_mlp() if self.representation != "legacy_di" else None
        )
        self.mlp_width = scalar_mlp() if self.representation != "legacy_di" else None

    def forward(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-peak context tokens and their key-padding mask.

        Args:
            pxrd_d: (B, P) d-spacings (Angstrom), 0 in padded slots.
            pxrd_i: (B, P) max-normalized intensities in [0, 1], 0 in padded slots.
            pxrd_peak_pad_mask: (B, P) bool, True where padded.
        Returns:
            tokens: (B, P, d_out), zero in padded slots.
            pad: (B, P) bool key-padding mask.
        """
        if pxrd_d.dim() != 2 or pxrd_d.shape != pxrd_i.shape:
            raise RuntimeError(
                f"pxrd_d/pxrd_i must be matching (B, P), got {tuple(pxrd_d.shape)} vs {tuple(pxrd_i.shape)}"
            )
        pad = pxrd_peak_pad_mask.bool()
        if pad.shape != pxrd_d.shape:
            raise RuntimeError(
                f"pxrd_peak_pad_mask must be (B, P)={tuple(pxrd_d.shape)}, got {tuple(pad.shape)}"
            )
        d_features = _encode_d_features(
            pxrd_d,
            mode=self.d_feature_mode,
            wavelength=self.wavelength,
            freqs=self.d_fourier_freqs,
        )
        tokens = self.mlp_d(d_features) + self.mlp_i(pxrd_i[..., None])
        if self.representation != "legacy_di":
            if pxrd_prominence is None or pxrd_width is None:
                raise RuntimeError(
                    f"PXRD representation {self.representation} requires prominence/width inputs"
                )
            shape = self.mlp_prominence(pxrd_prominence[..., None]) + self.mlp_width(
                pxrd_width[..., None]
            )
            tokens = tokens + (
                shape if self.representation == "broadened_shape" else shape * 0.0
            )
        return tokens.masked_fill(pad[..., None], 0.0), pad


class PXRDFeatureTransformerEncoder(nn.Module):
    """Transformer over full-profile bins or 4-feature XRDiff peaks."""

    def __init__(
        self,
        d_out: int,
        representation: str,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        peak_count: int = 128,
        peak_feature_dim: int = 4,
        num_cls_tokens: int = 1,
        encoder_peak_features: str = "full",
        width_coordinate: str = "two_theta",
        pooling: str = "cls",
        position_encoding: str = "none",
        fourier_frequencies: list[float] | None = None,
        mask_padding: bool = True,
        integer_two_theta: bool = False,
        wavelength: float = 1.54184,
    ) -> None:
        super().__init__()
        if representation not in {
            "full_850",
            "peaks_20x4",
            "peaks_128x4",
            "peaks_20x4_norm_prom",
            "peaks_128x4_norm_prom",
            "peak_matrix",
        }:
            raise ValueError(f"unsupported flat PXRD representation: {representation}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"pxrd encoder d_model={d_model} not divisible by n_heads={n_heads}"
            )
        self.num_cls_tokens = int(num_cls_tokens)
        if self.num_cls_tokens <= 0:
            raise ValueError("num_cls_tokens must be positive")
        self.representation = representation
        self.n_tokens = {
            "full_850": 850,
            "peaks_20x4": 20,
            "peaks_128x4": 128,
            "peaks_20x4_norm_prom": 20,
            "peaks_128x4_norm_prom": 128,
            "peak_matrix": int(peak_count),
        }[representation]
        in_features = (
            2
            if representation == "full_850"
            else (int(peak_feature_dim) if representation == "peak_matrix" else 4)
        )
        if self.n_tokens <= 0 or in_features not in {2, 3, 4}:
            raise ValueError(
                "peak_matrix requires positive peak_count and feature dim 2, 3, or 4"
            )
        self.peak_feature_dim = in_features
        widths = {
            "full": in_features,
            "d_i": 2,
            "q_i": 2,
            "two_theta_i": 2,
            "d_i_width": 3,
            "q_i_prominence": 3,
            "q_i_width_prominence": 4,
            "d_i_width_prominence": 4,
        }
        if encoder_peak_features not in widths:
            raise ValueError(
                f"unsupported encoder_peak_features: {encoder_peak_features}"
            )
        if encoder_peak_features != "full" and representation != "peaks_128x4":
            raise ValueError("encoder feature selection requires peaks_128x4")
        if not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("PXRD wavelength must be finite and positive")
        self.encoder_peak_features = encoder_peak_features
        if width_coordinate not in {"two_theta", "q", "d"}:
            raise ValueError("width_coordinate must be two_theta, q, or d")
        if (
            width_coordinate != "two_theta"
            and encoder_peak_features != f"{width_coordinate}_i_width_prominence"
        ):
            raise ValueError(
                "coordinate width requires matching q/d width-prominence features"
            )
        if pooling not in {"cls", "mean"} or (
            pooling == "mean" and (not mask_padding or self.num_cls_tokens != 1)
        ):
            raise ValueError("mean pooling requires masked peaks and one CLS token")
        self.width_coordinate = width_coordinate
        self.pooling = pooling
        if integer_two_theta and (representation != "peak_matrix" or in_features != 2):
            raise ValueError("integer two_theta requires two-feature peak_matrix")
        self.integer_two_theta = bool(integer_two_theta)
        self.mask_padding = bool(mask_padding)
        self.wavelength = float(wavelength)
        in_features = widths[encoder_peak_features]
        if position_encoding not in {"none", "rbf", "fourier"}:
            raise ValueError("position_encoding must be none, rbf, or fourier")
        if position_encoding != "none" and (
            representation != "peaks_128x4"
            or encoder_peak_features
            not in {
                "q_i",
                "q_i_prominence",
                "q_i_width_prominence",
                "d_i_width_prominence",
            }
        ):
            raise ValueError(
                "position encoding requires q-based or d-full peaks_128x4 transformer"
            )
        if fourier_frequencies is not None:
            if position_encoding != "fourier":
                raise ValueError(
                    "Fourier frequencies require fourier position encoding"
                )
            if (
                len(fourier_frequencies) != 16
                or any(not math.isfinite(f) or f <= 0 for f in fourier_frequencies)
                or any(
                    a >= b for a, b in zip(fourier_frequencies, fourier_frequencies[1:])
                )
            ):
                raise ValueError(
                    "Fourier frequencies must be 16 finite positive increasing values"
                )
        self.position_encoding = position_encoding
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.register_buffer(
            "cls_position_embed",
            (
                sinusoidal_position_embedding(self.num_cls_tokens, d_model)[None]
                if self.num_cls_tokens > 1
                else None
            ),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)
        self.register_buffer(
            "profile_positions",
            torch.linspace(0.0, 1.0, self.n_tokens)
            if representation == "full_850"
            else None,
            persistent=False,
        )
        self.position_proj = None
        if position_encoding != "none":
            # Preserve the RNG stream used to initialize the shared backbone.
            with torch.random.fork_rng(devices=[]):
                self.position_proj = nn.Linear(32, d_model, bias=False)
        self.register_buffer(
            "position_basis_grid",
            torch.linspace(0.0, 1.0, 32)
            if position_encoding == "rbf"
            else (
                torch.tensor(
                    fourier_frequencies
                    if fourier_frequencies is not None
                    else list(range(1, 17)),
                    dtype=torch.float32,
                )
                if position_encoding == "fourier"
                else None
            ),
            persistent=False,
        )

    def position_features(self, q: torch.Tensor) -> torch.Tensor:
        # x = sin(theta), using the physical q range [0, 4*pi/lambda].
        x = q[..., None] * (self.wavelength / (4.0 * math.pi))
        grid = self.position_basis_grid.to(dtype=q.dtype)
        if self.position_encoding == "rbf":
            basis = torch.exp(-0.5 * ((x - grid) * 31.0).square())
        else:
            phase = 2.0 * math.pi * x * grid
            basis = torch.cat([phase.sin(), phase.cos()], dim=-1)
        # Match basis RMS between localized RBFs and dense Fourier features.
        return torch.nn.functional.normalize(basis, dim=-1) * math.sqrt(32.0)

    def encode_tokens(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The historical name identifies the descriptor schema; rows may vary.
        n_tokens = self.n_tokens
        if self.representation == "peaks_128x4":
            if (
                pxrd_d.ndim != 2
                or not 4 <= pxrd_d.shape[1] <= 1696
                or pxrd_d.shape[1] % 4
            ):
                raise RuntimeError(
                    "peaks_128x4 expects 1..424 flattened four-feature rows"
                )
            n_tokens = pxrd_d.shape[1] // 4
        expected = (
            n_tokens
            if self.representation == "full_850"
            else n_tokens * self.peak_feature_dim
        )
        if pxrd_d.ndim != 2 or pxrd_d.shape[1] != expected:
            raise RuntimeError(
                f"{self.representation} Transformer expects (B, {expected}), got {tuple(pxrd_d.shape)}"
            )
        if pxrd_i.shape != pxrd_d.shape or pxrd_peak_pad_mask.shape != pxrd_d.shape:
            raise RuntimeError("PXRD feature tensors must have matching shapes")
        if self.representation == "full_850":
            position = self.profile_positions.to(dtype=pxrd_d.dtype).expand(
                pxrd_d.shape[0], -1
            )
            features = torch.stack([position, pxrd_d], dim=-1)
            pad = torch.zeros_like(pxrd_d, dtype=torch.bool)
        else:
            features = pxrd_d.reshape(pxrd_d.shape[0], n_tokens, self.peak_feature_dim)
            pad = features.abs().sum(dim=-1) == 0
        if self.encoder_peak_features != "full":
            # Select only after the shared peak extraction, preserving intensity and padding.
            d = features[..., 3].masked_fill(pad, 1.0)
            position = d
            if self.encoder_peak_features in {
                "q_i",
                "q_i_prominence",
                "q_i_width_prominence",
            }:
                position = 2.0 * math.pi / d
            elif self.encoder_peak_features == "two_theta_i":
                position = torch.rad2deg(
                    2.0 * torch.asin((self.wavelength / (2.0 * d)).clamp(max=1.0))
                )
            selected = [position.masked_fill(pad, 0.0), features[..., 2]]
            if self.encoder_peak_features in {
                "d_i_width",
                "q_i_width_prominence",
                "d_i_width_prominence",
            }:
                width = features[..., 0]
                if self.width_coordinate != "two_theta":
                    # ponytail: local Jacobian width; use measured endpoints if broad-peak accuracy is needed.
                    sin_theta = (self.wavelength / (2.0 * d)).clamp(max=1.0)
                    cos_theta = (1.0 - sin_theta.square()).clamp(min=0.0).sqrt()
                    slope = (2.0 * math.pi / self.wavelength) * cos_theta
                    if self.width_coordinate == "d":
                        slope = d.square() * cos_theta / self.wavelength
                    width = torch.deg2rad(width) * slope
                selected.append(width.masked_fill(pad, 0.0))
            if self.encoder_peak_features in {
                "q_i_prominence",
                "q_i_width_prominence",
                "d_i_width_prominence",
            }:
                selected.append(features[..., 1])
            features = torch.stack(selected, dim=-1)
        if self.integer_two_theta:
            features = torch.stack((features[..., 0].trunc(), features[..., 1]), dim=-1)
        peak_pad = pad
        if not self.mask_padding:
            pad = torch.zeros_like(pad)
        tokens = self.input_proj(features)
        if self.position_proj is not None:
            q = features[..., 0]
            if self.encoder_peak_features == "d_i_width_prominence":
                q = 2.0 * math.pi / q.masked_fill(peak_pad, 1.0)
            position_features = self.position_features(q)
            position_features = position_features.masked_fill(peak_pad[..., None], 0.0)
            tokens = tokens + self.position_proj(position_features)
        cls = self.cls_token.expand(tokens.shape[0], self.num_cls_tokens, -1).to(
            dtype=tokens.dtype
        )
        if self.cls_position_embed is not None:
            cls = cls + self.cls_position_embed.to(dtype=tokens.dtype)
        x = torch.cat([cls, tokens], dim=1)
        pad_seq = torch.cat(
            [
                torch.zeros(
                    (tokens.shape[0], self.num_cls_tokens),
                    device=pad.device,
                    dtype=torch.bool,
                ),
                pad,
            ],
            dim=1,
        )
        return self.out_proj(
            self.norm_out(self.encoder(x, src_key_padding_mask=pad_seq))
        ), pad_seq

    def forward(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, pad = self.encode_tokens(pxrd_d, pxrd_i, pxrd_peak_pad_mask)
        if self.pooling == "mean":
            # Keep the same sequence/backbone as CLS; only change the readout.
            peaks, peak_pad = (
                tokens[:, self.num_cls_tokens :],
                pad[:, self.num_cls_tokens :],
            )
            return peaks.masked_fill(peak_pad[..., None], 0.0).sum(1) / (~peak_pad).sum(
                1, keepdim=True
            ).clamp(min=1)
        return tokens[:, 0]


class XRDiffMLPEncoder(nn.Module):
    """Same-depth MLP used for either flattened XRDiff representation."""

    def __init__(
        self,
        d_out: int,
        input_dim: int,
        hidden_dim: int = 1101,
        n_layers: int = 4,
        dropout: float = 0.0,
        concat_di: bool = False,
        representation: str | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or n_layers < 2:
            raise ValueError("invalid XRDiff MLP dimensions")
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_dim, d_out)
        self.concat_di = bool(concat_di)
        self.representation = representation

    def forward(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pxrd_d.ndim != 2 or pxrd_i.shape != pxrd_d.shape:
            raise RuntimeError("XRDiff feature tensors must be matching (B, F)")
        if self.representation in {"broadened_basic", "broadened_shape"}:
            if pxrd_prominence is None or pxrd_width is None:
                raise RuntimeError(
                    f"PXRD representation {self.representation} requires prominence/width inputs"
                )
            shape = (
                [torch.zeros_like(pxrd_prominence), torch.zeros_like(pxrd_width)]
                if self.representation == "broadened_basic"
                else [pxrd_prominence, pxrd_width]
            )
            features = torch.stack([pxrd_d, pxrd_i, *shape], dim=-1).flatten(1)
        else:
            features = torch.cat([pxrd_d, pxrd_i], dim=-1) if self.concat_di else pxrd_d
        if features.shape[1] != self.mlp[0].in_features:
            raise RuntimeError(
                f"XRDiff MLP expects {self.mlp[0].in_features} features, "
                f"got {features.shape[1]}"
            )
        return self.out_proj(self.mlp(features))
