from __future__ import annotations
import torch
from torch import nn
from src.models.embeddings import (
    FourierCoordEmbedder,
    LatticeEmbedder,
    TimeEmbedder,
    sinusoidal_position_embedding,
)
from src.models.pxrd_encoder import (
    PXRDFeatureTransformerEncoder,
    PXRDPeakEncoder,
    PXRDPeakTokenEncoder,
    XRDiffMLPEncoder,
)
from src.models.transformer import TransformerTrunk
from src.models.heads import CrystalHeads


def mod1(x: torch.Tensor) -> torch.Tensor:
    """Wrap values into [0, 1)."""
    return x - torch.floor(x)


class BraggliteModel(nn.Module):
    """
    Transformer backbone for Bragglite EDM training on tokenized crystals.

    This keeps the original trunk/head architecture but swaps the token
    embedding to accept continuous (noisy) features:
      - type_feats: (B, N, type_dim) relaxed type features
      - frac_coords: (B, N, 3) fractional coords (wrapped to [0,1) internally)
      - lattice_feats: (B, 6)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vz: int,
        type_dim: int | None = None,
        n_freqs: int = 32,
        coord_embed_mode: str = "rff",
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        lattice_embed_mode: str = "rff",
        lattice_rff_dim: int = 256,
        lattice_rff_sigma: float = 5.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_distance_bias: bool = False,
        use_edge_bias: bool = False,
        edge_bias_n_freqs: int = 8,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        pbc_radius: int = 1,
        lattice_repr: str = "y1",
        dist_slope_init: float = -1.0,
        use_noise_gate: bool = True,
        gem_per_layer: bool = False,
        coord_head_mode: str = "direct",
        pxrd_cond: bool = False,
        pxrd_enc_d_model: int = 256,
        pxrd_enc_layers: int = 4,
        pxrd_enc_heads: int = 8,
        pxrd_enc_dropout: float = 0.0,
        pxrd_d_feature_mode: str = "raw",
        pxrd_d_fourier_n_freqs: int = 16,
        pxrd_wavelength: float = 1.54184,
        pxrd_encoder_arch: str = "transformer",
        pxrd_encoder_peak_features: str = "full",
        pxrd_encoder_width_coordinate: str = "two_theta",
        pxrd_encoder_pooling: str = "cls",
        pxrd_encoder_position_encoding: str = "none",
        pxrd_encoder_fourier_frequencies: list[float] | None = None,
        pxrd_encoder_mask_padding: bool = True,
        pxrd_encoder_integer_two_theta: bool = False,
        pxrd_representation: str = "legacy_di",
        pxrd_peak_features: str = "width_prominence_intensity_position",
        pxrd_enc_mlp_hidden_dim: int = 1101,
        pxrd_max_peaks: int = 128,
        pxrd_fusion: str = "sum",
        pxrd_in_context_mode: str = "peak_tokens",
        pxrd_context_repeats: int = 8,
        pxrd_context_learned_positions: bool = False,
        element_set_cond: bool = False,
        max_atoms: int = 20,
        atom_positional_encoding: str = "none",
        formulation: str = "edm",
    ) -> None:
        super().__init__()
        self.formulation = str(formulation).strip().lower()
        if self.formulation != "edm":
            raise ValueError("Only the paper EDM formulation is supported.")
        self.type_dim = vz + 1 if type_dim is None else int(type_dim)
        self.type_proj = nn.Sequential(
            nn.Linear(self.type_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.coord_embed = FourierCoordEmbedder(
            d_model=d_model,
            n_freqs=n_freqs,
            mode=coord_embed_mode,
            rff_dim=coord_rff_dim,
            rff_sigma=coord_rff_sigma,
        )
        self.lattice_embed = LatticeEmbedder(
            d_model=d_model,
            mode=lattice_embed_mode,
            rff_dim=lattice_rff_dim,
            rff_sigma=lattice_rff_sigma,
        )
        self.segment_embed = nn.Embedding(2, d_model)
        self.atom_positional_encoding = str(atom_positional_encoding).strip().lower()
        if self.atom_positional_encoding not in {"none", "sinusoidal"}:
            raise ValueError("atom_positional_encoding must be 'none' or 'sinusoidal'.")
        atom_position_embed = (
            sinusoidal_position_embedding(int(max_atoms), d_model)
            if self.atom_positional_encoding == "sinusoidal"
            else None
        )
        self.register_buffer(
            "atom_position_embed", atom_position_embed, persistent=False
        )
        self.time = TimeEmbedder(d_model=d_model)
        self.pxrd_fusion = str(pxrd_fusion).strip().lower()
        if pxrd_encoder_peak_features != "full" and (
            pxrd_representation != "peaks_128x4" or pxrd_encoder_arch != "transformer"
        ):
            raise ValueError(
                "encoder peak feature selection requires peaks_128x4 transformer"
            )
        if pxrd_encoder_position_encoding != "none" and (
            not pxrd_cond
            or pxrd_encoder_arch != "transformer"
            or pxrd_representation != "peaks_128x4"
            or (
                pxrd_encoder_peak_features
                not in {
                    "q_i",
                    "q_i_prominence",
                    "q_i_width_prominence",
                    "d_i_width_prominence",
                }
            )
        ):
            raise ValueError(
                "position encoding requires q-based or d-full peaks_128x4 transformer"
            )
        if pxrd_encoder_width_coordinate != "two_theta" and (
            not pxrd_cond
            or pxrd_representation != "peaks_128x4"
            or pxrd_encoder_arch != "transformer"
        ):
            raise ValueError("coordinate width requires peaks_128x4 transformer")
        if pxrd_encoder_pooling != "cls" and (
            not pxrd_cond
            or pxrd_encoder_arch != "transformer"
            or pxrd_representation != "peaks_128x4"
            or (pxrd_fusion != "in_context")
            or (pxrd_in_context_mode != "global_repeat")
        ):
            raise ValueError(
                "mean pooling requires peaks_128x4 transformer global_repeat"
            )
        self.pxrd_encoder_arch = str(pxrd_encoder_arch)
        if self.pxrd_encoder_arch not in {"transformer", "xrdiff_mlp"}:
            raise ValueError(f"Unsupported pxrd_encoder_arch: {pxrd_encoder_arch}")
        self.pxrd_in_context = bool(pxrd_cond and self.pxrd_fusion == "in_context")
        self.pxrd_in_context_mode = str(pxrd_in_context_mode).strip().lower()
        if self.pxrd_in_context_mode not in {
            "peak_tokens",
            "global_repeat",
            "multi_cls",
        }:
            raise ValueError(
                f"Unsupported pxrd_in_context_mode: {pxrd_in_context_mode}. Use peak_tokens|global_repeat|multi_cls."
            )
        self.pxrd_context_repeats = int(pxrd_context_repeats)
        if pxrd_context_learned_positions and (
            not self.pxrd_in_context or self.pxrd_in_context_mode != "global_repeat"
        ):
            raise ValueError(
                "learned PXRD context positions require in_context global_repeat"
            )
        if self.pxrd_context_repeats <= 0:
            raise ValueError(
                f"pxrd_context_repeats must be positive, got {pxrd_context_repeats}."
            )
        if self.pxrd_in_context_mode == "multi_cls" and (not self.pxrd_in_context):
            raise ValueError("multi_cls requires pxrd_fusion=in_context")
        self.pxrd_representation = str(pxrd_representation)
        if self.pxrd_representation not in {
            "legacy_di",
            "full_850",
            "peaks_20x4",
            "peaks_128x4",
            "peaks_20x4_norm_prom",
            "peaks_128x4_norm_prom",
            "broadened_basic",
            "broadened_shape",
            "peak_matrix",
            "dense_profile",
        }:
            raise ValueError(f"Unsupported pxrd_representation: {pxrd_representation}")
        feature_name = str(pxrd_peak_features)
        peak_feature_dim = {
            "width_prominence_intensity_position": 4,
            "prominence_intensity_position": 3,
            "width_intensity_position": 3,
            "position_intensity": 2,
        }.get(feature_name)
        if not peak_feature_dim:
            raise ValueError(f"Unsupported pxrd_peak_features: {pxrd_peak_features}")
        flat_input_dim = (
            pxrd_max_peaks * peak_feature_dim
            if self.pxrd_representation == "peak_matrix"
            else pxrd_max_peaks * 4
            if self.pxrd_representation in {"broadened_basic", "broadened_shape"}
            else pxrd_max_peaks
        )
        if self.pxrd_encoder_arch == "xrdiff_mlp" and (
            self.pxrd_fusion == "cross_attn"
            or (self.pxrd_in_context and self.pxrd_in_context_mode != "global_repeat")
        ):
            raise ValueError(
                "xrdiff_mlp supports only global PXRD fusion or global_repeat"
            )
        if (
            self.pxrd_in_context
            and self.pxrd_in_context_mode == "multi_cls"
            and (
                self.pxrd_representation
                not in {
                    "full_850",
                    "peaks_20x4",
                    "peaks_128x4",
                    "peaks_20x4_norm_prom",
                    "peaks_128x4_norm_prom",
                    "peak_matrix",
                }
            )
        ):
            raise ValueError("multi_cls requires a flat-profile Transformer encoder")
        if (
            self.pxrd_representation == "dense_profile"
            and self.pxrd_encoder_arch != "xrdiff_mlp"
        ):
            raise ValueError(
                "dense_profile requires xrdiff_mlp; dense-bin self-attention is unsupported"
            )
        if self.pxrd_representation in {
            "full_850",
            "peaks_20x4",
            "peaks_128x4",
            "peaks_20x4_norm_prom",
            "peaks_128x4_norm_prom",
            "peak_matrix",
            "dense_profile",
        } and (self.pxrd_in_context and self.pxrd_in_context_mode == "peak_tokens"):
            raise ValueError(
                "flat XRDiff representations require global_repeat in-context conditioning"
            )
        self.pxrd_encoder = (
            (
                XRDiffMLPEncoder(
                    d_out=d_model,
                    input_dim=flat_input_dim
                    * (2 if self.pxrd_representation == "legacy_di" else 1),
                    hidden_dim=pxrd_enc_mlp_hidden_dim,
                    n_layers=pxrd_enc_layers,
                    dropout=pxrd_enc_dropout,
                    concat_di=self.pxrd_representation == "legacy_di",
                    representation=self.pxrd_representation,
                )
                if self.pxrd_encoder_arch == "xrdiff_mlp"
                else PXRDPeakTokenEncoder(
                    d_out=d_model,
                    d_model=pxrd_enc_d_model,
                    dropout=pxrd_enc_dropout,
                    d_feature_mode=pxrd_d_feature_mode,
                    d_fourier_n_freqs=pxrd_d_fourier_n_freqs,
                    wavelength=pxrd_wavelength,
                    representation=self.pxrd_representation,
                )
                if self.pxrd_in_context and self.pxrd_in_context_mode == "peak_tokens"
                else PXRDPeakEncoder(
                    d_out=d_model,
                    d_model=pxrd_enc_d_model,
                    n_layers=pxrd_enc_layers,
                    n_heads=pxrd_enc_heads,
                    dropout=pxrd_enc_dropout,
                    d_feature_mode=pxrd_d_feature_mode,
                    d_fourier_n_freqs=pxrd_d_fourier_n_freqs,
                    wavelength=pxrd_wavelength,
                    representation=self.pxrd_representation,
                )
                if self.pxrd_representation
                in {"legacy_di", "broadened_basic", "broadened_shape"}
                else PXRDFeatureTransformerEncoder(
                    d_out=d_model,
                    encoder_peak_features=pxrd_encoder_peak_features,
                    width_coordinate=pxrd_encoder_width_coordinate,
                    pooling=pxrd_encoder_pooling,
                    position_encoding=pxrd_encoder_position_encoding,
                    fourier_frequencies=pxrd_encoder_fourier_frequencies,
                    mask_padding=pxrd_encoder_mask_padding,
                    integer_two_theta=pxrd_encoder_integer_two_theta,
                    wavelength=pxrd_wavelength,
                    representation=self.pxrd_representation,
                    d_model=pxrd_enc_d_model,
                    n_layers=pxrd_enc_layers,
                    n_heads=pxrd_enc_heads,
                    dropout=pxrd_enc_dropout,
                    peak_count=pxrd_max_peaks,
                    peak_feature_dim=peak_feature_dim,
                    num_cls_tokens=self.pxrd_context_repeats
                    if self.pxrd_in_context_mode == "multi_cls"
                    else 1,
                )
            )
            if pxrd_cond
            else None
        )
        if pxrd_encoder_integer_two_theta and (
            not isinstance(self.pxrd_encoder, PXRDFeatureTransformerEncoder)
            or self.pxrd_representation != "peak_matrix"
        ):
            raise ValueError("integer two_theta requires a peak_matrix transformer")
        if not pxrd_encoder_mask_padding and (
            not isinstance(self.pxrd_encoder, PXRDFeatureTransformerEncoder)
            or self.pxrd_representation == "full_850"
        ):
            raise ValueError(
                "disabling PXRD padding mask requires a flat peak transformer"
            )
        self.pxrd_context_segment = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if self.pxrd_in_context else None
        )
        if pxrd_cond and self.pxrd_fusion not in {"sum", "in_context"}:
            raise ValueError("PXRD fusion must be sum or in_context.")
        self.element_set_cond = bool(element_set_cond)
        self.elem_set_embed = (
            nn.Sequential(
                nn.Linear(vz, d_model, bias=True),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
            )
            if self.element_set_cond
            else None
        )
        self.trunk = TransformerTrunk(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_distance_bias=use_distance_bias,
            use_edge_bias=use_edge_bias,
            edge_bias_n_freqs=edge_bias_n_freqs,
            edge_bias_hidden_dim=edge_bias_hidden_dim,
            edge_bias_n_rbf=edge_bias_n_rbf,
            edge_bias_rbf_max=edge_bias_rbf_max,
            pbc_radius=pbc_radius,
            lattice_repr=lattice_repr,
            dist_slope_init=dist_slope_init,
            use_noise_gate=use_noise_gate,
            gem_per_layer=gem_per_layer,
        )
        self.heads = CrystalHeads(
            d_model=d_model,
            vz=vz,
            type_out_dim=self.type_dim,
            coord_head_mode=coord_head_mode,
        )
        self.pxrd_context_positions = (
            nn.Parameter(torch.empty(1, self.pxrd_context_repeats, d_model))
            if pxrd_context_learned_positions
            else None
        )
        if self.pxrd_context_positions is not None:
            nn.init.normal_(self.pxrd_context_positions, std=0.02)

    def _fuse_pxrd(
        self,
        t_emb: torch.Tensor,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse the PXRD latent into the adaLN conditioning vector (see __init__)."""
        p = self.pxrd_encoder(
            pxrd_d, pxrd_i, pxrd_peak_pad_mask, pxrd_prominence, pxrd_width
        )
        return t_emb + p

    def _pxrd_context_tokens(
        self,
        pxrd_d: torch.Tensor,
        pxrd_i: torch.Tensor,
        pxrd_peak_pad_mask: torch.Tensor,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PXRD context tokens for the in-context conditioning path."""
        if self.pxrd_in_context_mode == "global_repeat":
            token = self.pxrd_encoder(
                pxrd_d, pxrd_i, pxrd_peak_pad_mask, pxrd_prominence, pxrd_width
            )[:, None, :]
            tokens = token.expand(-1, self.pxrd_context_repeats, -1)
            tok_pad = torch.zeros(
                (tokens.shape[0], self.pxrd_context_repeats),
                device=tokens.device,
                dtype=torch.bool,
            )
            tokens = tokens + self.pxrd_context_segment.to(dtype=tokens.dtype)
            if self.pxrd_context_positions is not None:
                tokens = tokens + self.pxrd_context_positions.to(dtype=tokens.dtype)
            return (tokens, tok_pad)
        if self.pxrd_in_context_mode == "multi_cls":
            tokens, tok_pad = self.pxrd_encoder.encode_tokens(
                pxrd_d, pxrd_i, pxrd_peak_pad_mask, pxrd_prominence, pxrd_width
            )
            tokens = tokens[:, : self.pxrd_context_repeats]
            tok_pad = tok_pad[:, : self.pxrd_context_repeats]
        else:
            tokens, tok_pad = self.pxrd_encoder(
                pxrd_d, pxrd_i, pxrd_peak_pad_mask, pxrd_prominence, pxrd_width
            )
        tokens = tokens + self.pxrd_context_segment.to(dtype=tokens.dtype)
        tokens = tokens.masked_fill(tok_pad[..., None], 0.0)
        return (tokens, tok_pad)

    def forward(
        self,
        type_feats: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        t_sigma: torch.Tensor,
        lattice_bias_feats: torch.Tensor | None = None,
        pxrd_d: torch.Tensor | None = None,
        pxrd_i: torch.Tensor | None = None,
        pxrd_peak_pad_mask: torch.Tensor | None = None,
        pxrd_position: torch.Tensor | None = None,
        pxrd_intensity: torch.Tensor | None = None,
        pxrd_prominence: torch.Tensor | None = None,
        pxrd_width: torch.Tensor | None = None,
        elem_set_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            type_feats: (B, N, type_dim) relaxed type features with padding zeroed.
            frac_coords: (B, N, 3) fractional coords (not necessarily wrapped).
            lattice_feats: (B, 6)
            pad_mask: (B, N) bool where True denotes padding.
            t_sigma: (B,) scalar noise embedding; shared for t_g/t_a.
            lattice_bias_feats: optional (B, 6) lattice features used only for
                geometry-aware attention bias. If None, lattice_feats is used.
            pxrd_d / pxrd_i / pxrd_peak_pad_mask: (B, P) d-I peak list +
                padding mask; required iff the model was built with pxrd_cond.
            elem_set_mask: (B, VZ) bool/float mask of allowed elements; required
                iff the model was built with element_set_cond.
        """
        frac_mod = mod1(frac_coords)
        physical_time = t_sigma.to(dtype=torch.float32)
        time_input = physical_time
        t_emb = self.time(time_input, time_input)
        h_type = (
            self.type_proj(type_feats)
            + self.coord_embed(frac_mod)
            + self.segment_embed.weight[0]
        )
        if self.atom_position_embed is not None:
            num_slots = int(type_feats.shape[1])
            if num_slots > int(self.atom_position_embed.shape[0]):
                raise RuntimeError(
                    f"Received {num_slots} atom slots, but fixed positional encoding supports only {self.atom_position_embed.shape[0]}."
                )
            position = self.atom_position_embed[:num_slots].to(dtype=h_type.dtype)
            position = position[None, :, :].masked_fill(pad_mask[..., None], 0.0)
            h_type = h_type + position
        h_lat = self.lattice_embed(lattice_feats) + self.segment_embed.weight[1]
        current_atom_pad = pad_mask.bool()
        h_lat = h_lat[:, None, :]
        current_tokens = torch.cat([h_type, h_lat], dim=1)
        current_pad = torch.cat(
            [
                current_atom_pad,
                torch.zeros(
                    (pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool
                ),
            ],
            dim=1,
        )
        x = current_tokens
        pad_seq = current_pad
        atom_token_offset = 0
        if self.pxrd_encoder is not None:
            if self.pxrd_representation in {"broadened_basic", "broadened_shape"}:
                peak_position, peak_intensity = (pxrd_position, pxrd_intensity)
            else:
                peak_position, peak_intensity = (pxrd_d, pxrd_i)
            if (
                peak_position is None
                or peak_intensity is None
                or pxrd_peak_pad_mask is None
            ):
                raise RuntimeError(
                    f"Model built with pxrd_representation={self.pxrd_representation} did not receive its required peak inputs."
                )
            if self.pxrd_in_context:
                pxrd_tokens, pxrd_pad = self._pxrd_context_tokens(
                    peak_position,
                    peak_intensity,
                    pxrd_peak_pad_mask,
                    pxrd_prominence,
                    pxrd_width,
                )
                atom_token_offset = pxrd_tokens.shape[1]
                x = torch.cat([pxrd_tokens, x], dim=1)
                pad_seq = torch.cat([pxrd_pad, pad_seq], dim=1)
            else:
                t_emb = self._fuse_pxrd(
                    t_emb,
                    peak_position,
                    peak_intensity,
                    pxrd_peak_pad_mask,
                    pxrd_prominence,
                    pxrd_width,
                )
        elif pxrd_d is not None or pxrd_position is not None:
            raise RuntimeError(
                "pxrd_* inputs given but model was built with pxrd_cond=False."
            )
        if self.elem_set_embed is not None:
            if elem_set_mask is None:
                raise RuntimeError(
                    "Model built with element_set_cond=True requires elem_set_mask."
                )
            t_emb = t_emb + self.elem_set_embed(elem_set_mask.to(dtype=t_emb.dtype))
        elif elem_set_mask is not None:
            raise RuntimeError(
                "elem_set_mask input given but model was built with element_set_cond=False."
            )
        sigma_for_gem = torch.exp(4.0 * t_sigma.to(dtype=torch.float32))
        lattice_for_bias = (
            lattice_feats if lattice_bias_feats is None else lattice_bias_feats
        )
        h = self.trunk(
            x,
            t_emb,
            pad_mask=pad_seq,
            coords=frac_mod,
            lattice=lattice_for_bias,
            t_sigma=sigma_for_gem,
            atom_token_offset=atom_token_offset,
        )
        h_out = h[:, atom_token_offset:, :]
        return self.heads(h_out, frac_coords=frac_mod, pad_mask=pad_mask.bool())


MP20EDMModel = BraggliteModel
