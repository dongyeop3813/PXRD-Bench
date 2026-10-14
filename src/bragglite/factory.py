"""Shared Bragglite model factory.

Single source of truth for mapping the flat training-arg surface to
``BraggliteModel`` constructor kwargs:

- ``build_bragglite`` — strict construction from a complete flat arg dict
  (training path: every key must exist, missing config fails loudly).
- ``build_bragglite_from_model_args`` — inference path: fills legacy defaults
  for old checkpoints, then constructs, loads weights, and switches to eval.
"""

from __future__ import annotations
import torch
from src.data.mp20_tokens import VZ
from src.bragglite import BraggliteModel

_LEGACY_MODEL_ARG_DEFAULTS: dict = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 18,
    "coord_embed_mode": "fourier",
    "coord_head_mode": "direct",
    "coord_rff_dim": None,
    "coord_rff_sigma": 1.0,
    "lattice_embed_mode": "mlp",
    "lattice_rff_dim": 256,
    "lattice_rff_sigma": 5.0,
    "lattice_repr": "y1",
    "dropout": 0.0,
    "attn_dropout": 0.0,
    "use_distance_bias": False,
    "use_edge_bias": False,
    "edge_bias_n_freqs": 8,
    "edge_bias_hidden_dim": 128,
    "edge_bias_n_rbf": 16,
    "edge_bias_rbf_max": 2.0,
    "pbc_radius": 1,
    "dist_slope_init": -1.0,
    "use_noise_gate": True,
    "gem_per_layer": False,
    "pxrd_cond": False,
    "pxrd_enc_d_model": 256,
    "pxrd_enc_layers": 4,
    "pxrd_enc_heads": 8,
    "pxrd_enc_dropout": 0.0,
    "pxrd_d_feature_mode": "raw",
    "pxrd_d_fourier_n_freqs": 16,
    "pxrd_wavelength": 1.54184,
    "pxrd_encoder_arch": "transformer",
    "pxrd_encoder_peak_features": "full",
    "pxrd_encoder_width_coordinate": "two_theta",
    "pxrd_encoder_pooling": "cls",
    "pxrd_encoder_position_encoding": "none",
    "pxrd_encoder_fourier_frequencies": None,
    "pxrd_encoder_mask_padding": True,
    "pxrd_encoder_integer_two_theta": False,
    "pxrd_representation": "legacy_di",
    "pxrd_peak_source": "bragg",
    "pxrd_peak_features": "width_prominence_intensity_position",
    "pxrd_dense_n_bins": 8500,
    "pxrd_enc_mlp_hidden_dim": 1101,
    "pxrd_max_peaks": 128,
    "pxrd_fusion": "sum",
    "pxrd_in_context_mode": "peak_tokens",
    "pxrd_context_repeats": 8,
    "pxrd_context_learned_positions": False,
    "element_set_cond": False,
    "atom_positional_encoding": "none",
    "formulation": "edm",
}


def reject_unsupported_model_options(model_args: dict) -> None:
    """Accept disabled historical metadata, reject unsupported architectures."""
    disabled_flags = (
        "modern_dit",
        "resformer_value_residual",
        "residual_attention",
        "recycling_enabled",
        "persistent_atom_queries",
        "sg_cond",
        "lattice_cond",
    )
    for name in disabled_flags:
        if model_args.get(name, False):
            raise ValueError(f"Unsupported checkpoint architecture: {name}")
    for name in ("atom_token_repeats", "lattice_token_repeats"):
        if int(model_args.get(name, 1)) != 1:
            raise ValueError(f"Unsupported checkpoint architecture: {name}")
    if str(model_args.get("recycling_sampling_mode", "off")) != "off":
        raise ValueError("Recycling is not supported.")
    if str(model_args.get("formulation", "edm")) != "edm":
        raise ValueError("Only the paper EDM formulation is supported.")


def build_bragglite(
    model_args: dict, *, type_dim: int, device: torch.device
) -> BraggliteModel:
    """Construct a ``BraggliteModel`` from a complete flat arg dict.

    Strict on purpose: architecture keys are indexed directly, so a missing
    field raises ``KeyError`` instead of silently falling back to a default.
    Legacy-checkpoint default filling belongs in
    ``build_bragglite_from_model_args``.

    Exception: ``pxrd_cond`` is an opt-in feature flag read via ``.get`` so the
    older checkpoint metadata stays compatible; when the flag is set the
    encoder hyperparameters ARE indexed strictly.
    """
    for flag in ("modern_dit", "resformer_value_residual", "residual_attention"):
        if model_args.get(flag, False):
            raise ValueError(
                f"Unsupported experimental checkpoint architecture: {flag}"
            )
    reject_unsupported_model_options(model_args)
    pxrd_cond = bool(model_args.get("pxrd_cond", False))
    from src.pxrd.representation import model_pxrd_representation

    pxrd_representation = model_pxrd_representation(model_args)
    pxrd_input_width = (
        int(model_args.get("pxrd_dense_n_bins", 8500))
        if pxrd_representation == "dense_profile"
        else int(model_args.get("pxrd_max_peaks", 128))
    )
    if model_args.get("pxrd_encoder_integer_two_theta", False) and (
        not pxrd_cond
        or pxrd_representation != "peak_matrix"
        or model_args.get("pxrd_peak_position") != "two_theta"
        or (model_args.get("pxrd_peak_features") != "position_intensity")
        or model_args.get("pxrd_peak_position_normalized", False)
    ):
        raise ValueError(
            "integer two_theta requires raw two_theta/intensity peak_matrix"
        )
    pxrd_kwargs = (
        {
            "pxrd_cond": True,
            "pxrd_enc_d_model": int(model_args["pxrd_enc_d_model"]),
            "pxrd_enc_layers": int(model_args["pxrd_enc_layers"]),
            "pxrd_enc_heads": int(model_args["pxrd_enc_heads"]),
            "pxrd_enc_dropout": float(model_args["pxrd_enc_dropout"]),
            "pxrd_d_feature_mode": str(model_args.get("pxrd_d_feature_mode", "raw")),
            "pxrd_d_fourier_n_freqs": int(model_args.get("pxrd_d_fourier_n_freqs", 16)),
            "pxrd_wavelength": float(model_args.get("pxrd_wavelength", 1.54184)),
            "pxrd_encoder_arch": str(
                model_args.get("pxrd_encoder_arch", "transformer")
            ),
            "pxrd_representation": pxrd_representation,
            "pxrd_encoder_peak_features": str(
                model_args.get("pxrd_encoder_peak_features", "full")
            ),
            "pxrd_encoder_width_coordinate": str(
                model_args.get("pxrd_encoder_width_coordinate", "two_theta")
            ),
            "pxrd_encoder_pooling": str(model_args.get("pxrd_encoder_pooling", "cls")),
            "pxrd_encoder_position_encoding": str(
                model_args.get("pxrd_encoder_position_encoding", "none")
            ),
            "pxrd_encoder_fourier_frequencies": model_args.get(
                "pxrd_encoder_fourier_frequencies"
            ),
            "pxrd_encoder_mask_padding": bool(
                model_args.get("pxrd_encoder_mask_padding", True)
            ),
            "pxrd_encoder_integer_two_theta": bool(
                model_args.get("pxrd_encoder_integer_two_theta", False)
            ),
            "pxrd_peak_features": str(
                model_args.get(
                    "pxrd_peak_features", "width_prominence_intensity_position"
                )
            ),
            "pxrd_enc_mlp_hidden_dim": int(
                model_args.get("pxrd_enc_mlp_hidden_dim", 1101)
            ),
            "pxrd_max_peaks": pxrd_input_width,
            "pxrd_fusion": str(model_args.get("pxrd_fusion", "sum")),
            "pxrd_in_context_mode": str(
                model_args.get("pxrd_in_context_mode", "peak_tokens")
            ),
            "pxrd_context_repeats": int(model_args.get("pxrd_context_repeats", 8)),
            "pxrd_context_learned_positions": bool(
                model_args.get("pxrd_context_learned_positions", False)
            ),
        }
        if pxrd_cond
        else {"pxrd_cond": False}
    )
    element_set_kwargs = {
        "element_set_cond": bool(model_args.get("element_set_cond", False))
    }
    atom_query_kwargs = {
        "max_atoms": int(model_args.get("nmax", 20)),
        "atom_positional_encoding": str(
            model_args.get("atom_positional_encoding", "none")
        ),
    }
    return BraggliteModel(
        d_model=int(model_args["d_model"]),
        n_heads=int(model_args["n_heads"]),
        n_layers=int(model_args["n_layers"]),
        vz=VZ,
        type_dim=int(type_dim),
        n_freqs=int(model_args["coord_n_freqs"]),
        coord_embed_mode=str(model_args["coord_embed_mode"]),
        coord_head_mode=str(model_args["coord_head_mode"]),
        coord_rff_dim=model_args["coord_rff_dim"],
        coord_rff_sigma=float(model_args["coord_rff_sigma"]),
        lattice_embed_mode=str(model_args["lattice_embed_mode"]),
        lattice_rff_dim=int(model_args["lattice_rff_dim"]),
        lattice_rff_sigma=float(model_args["lattice_rff_sigma"]),
        lattice_repr=str(model_args["lattice_repr"]),
        dropout=float(model_args["dropout"]),
        attn_dropout=float(model_args["attn_dropout"]),
        use_distance_bias=bool(model_args["use_distance_bias"]),
        use_edge_bias=bool(model_args["use_edge_bias"]),
        edge_bias_n_freqs=int(model_args["edge_bias_n_freqs"]),
        edge_bias_hidden_dim=int(model_args["edge_bias_hidden_dim"]),
        edge_bias_n_rbf=int(model_args["edge_bias_n_rbf"]),
        edge_bias_rbf_max=float(model_args["edge_bias_rbf_max"]),
        pbc_radius=int(model_args["pbc_radius"]),
        dist_slope_init=float(model_args["dist_slope_init"]),
        use_noise_gate=bool(model_args["use_noise_gate"]),
        gem_per_layer=bool(model_args["gem_per_layer"]),
        formulation=str(model_args.get("formulation", "edm")),
        **pxrd_kwargs,
        **element_set_kwargs,
        **atom_query_kwargs,
    ).to(device)


def build_bragglite_from_model_args(
    ckpt: dict, device: torch.device
) -> tuple[BraggliteModel, dict]:
    """Build a ``BraggliteModel`` from a checkpoint's stored ``model_args``.

    Fills legacy defaults for checkpoints that predate newer fields, loads
    ``model_state_dict`` strictly, and returns the model in eval mode together
    with the original (unfilled) ``model_args`` dict.
    """
    model_args = dict(ckpt.get("model_args", {}))
    if str(model_args.get("attn_type", "mha")).strip().lower() != "mha":
        raise ValueError(
            "This codebase no longer supports simplicial attention checkpoints. The checkpoint was configured with attn_type != 'mha'."
        )
    type_dim = int(ckpt.get("type_dim", model_args.get("type_dim", VZ + 1)))
    resolved = dict(model_args)
    resolved.setdefault("coord_n_freqs", resolved.get("n_freqs", 32))
    for key, default in _LEGACY_MODEL_ARG_DEFAULTS.items():
        resolved.setdefault(key, default)
    model = build_bragglite(resolved, type_dim=type_dim, device=device)
    model_state = ckpt.get("model_state_dict", None)
    if model_state is None:
        raise KeyError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return (model, model_args)
