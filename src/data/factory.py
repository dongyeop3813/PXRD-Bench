"""Load frozen Bragglite datasets using a shared token schema.

Paper loaders cover MP-20, PXRD-Bench, RRUFF-148, opXRD and two-phase
artifacts. Historical RRUFF dataset identifiers are retained for frozen
manifest/checkpoint compatibility; no external baseline datasets are loaded.
"""

from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset

DATASET_CLASSES = (
    "mp20_tokens",
    "real_pxrd",
    "xrdiff_author_rruff",
    "xrdiff_author_rruff_simulated",
    "shared_phase",
    "experimental_pxrd",
    "frozen_phase_pxrd",
    "pxrd_bench",
)


def dataset_needs_split_setup(dataset_class: str) -> bool:
    """Whether ``ensure_dataset_splits`` applies (only the CSV-based MP20 path).

    Other datasets ship their own on-disk splits, so callers skip split setup.
    """
    return dataset_class == "mp20_tokens"


def _experimental_representation(
    representation: str,
    *,
    peak_source: str,
    peak_features: str,
    peak_position: str,
    position_normalized: bool,
    feature_source: str,
) -> str:
    from src.pxrd.representation import (
        legacy_representation,
        normalize_peak_features,
        resolve_representation,
    )

    _, source = resolve_representation(representation, peak_source=peak_source)
    return legacy_representation(
        representation,
        peak_source=source,
        peak_features=normalize_peak_features(peak_features, position=peak_position),
        position_normalized=position_normalized,
        feature_source=feature_source,
    )


def build_token_dataset(
    dataset_class: str,
    *,
    root: str,
    split: str,
    nmax: int,
    augment_translate: bool = False,
    with_pxrd: bool = False,
    prop_list: list[str] | None = None,
    real_pxrd_variant: str = "all",
    real_pxrd_in_memory: bool = False,
    real_pxrd_peak_preprocessing: str = "official",
    experimental_background_subtraction: bool = True,
    structure_cache: str | None = None,
    real_pxrd_n_bins: int = 8500,
    real_pxrd_two_theta_min: float = 5.0,
    real_pxrd_two_theta_max: float = 90.0,
    real_pxrd_sigma: float = 0.05,
    with_pxrd_peaks: bool = False,
    pxrd_max_peaks: int = 128,
    pxrd_wavelength: float = 1.54184,
    pxrd_peak_augment: bool = False,
    pxrd_aug_jitter_deg: float = 0.1,
    pxrd_aug_int_scale_min: float = 0.8,
    pxrd_aug_int_scale_max: float = 1.2,
    pxrd_aug_width_scale_min: float = 0.7,
    pxrd_aug_width_scale_max: float = 1.5,
    pxrd_profile_augment: bool = False,
    pxrd_profile_augmentation_seed: int | None = None,
    pxrd_profile_augment_probability: float = 1.0,
    pxrd_aug_zero_shift_deg: float = 0.15,
    pxrd_aug_eta_min: float = 0.0,
    pxrd_aug_eta_max: float = 0.5,
    pxrd_aug_background_max: float = 0.2,
    pxrd_aug_noise_std_max: float = 0.02,
    pxrd_aug_drop_frac_max: float = 0.1,
    pxrd_peak_selection: str = "strongest",
    pxrd_low_angle_fraction: float = 0.25,
    pxrd_representation: str = "peak_list",
    pxrd_peak_source: str = "bragg",
    pxrd_peak_features: str = "position_intensity",
    pxrd_peak_position: str = "d",
    pxrd_peak_position_normalized: bool = False,
    pxrd_peak_intensity_normalization: str = "selected_topk",
    pxrd_peak_min_relative_intensity: float = 0.0,
    pxrd_peak_min_prominence: float = 0.0,
    pxrd_peak_prominence_normalization: str = "none",
    pxrd_peak_width_unit: str = "two_theta",
    pxrd_peak_width_normalization: str = "none",
    pxrd_feature_source: str = "auto",
    pxrd_profile_step: float = 0.1,
    pxrd_caglioti_u: float = 0.1,
    pxrd_caglioti_v: float = 0.01,
    pxrd_caglioti_w: float = 0.1,
    pxrd_pseudo_voigt_eta: float = 0.1,
    pxrd_dense_n_bins: int = 8500,
    pxrd_augmentation: str = "none",
    pxrd_xrdiff_gamma_shape: float | None = None,
    pxrd_xrdiff_gamma_scale: float | None = None,
    pxrd_dense_profile_augment: bool | None = None,
    atom_ordering: str = "none",
    hierarchical_orbit_permutation: bool = False,
    atom_ordering_symprec: float = 0.1,
    phase_reference_root: str = "data/mp20",
) -> Dataset:
    """Construct the requested token Dataset. Class-specific kwargs are ignored
    by the classes that don't use them."""
    if dataset_class not in DATASET_CLASSES:
        raise ValueError(
            f"Unsupported dataset_class {dataset_class!r}; expected {DATASET_CLASSES}"
        )
    if pxrd_peak_min_prominence and (
        pxrd_representation != "peaks_128x4"
        or dataset_class
        not in {
            "mp20_tokens",
            "xrdiff_author_rruff",
            "xrdiff_author_rruff_simulated",
            "experimental_pxrd",
            "frozen_phase_pxrd",
        }
    ):
        raise ValueError("pxrd_peak_min_prominence requires supported peaks_128x4 data")
    if pxrd_profile_augmentation_seed is not None and dataset_class not in {
        "mp20_tokens",
        "xrdiff_author_rruff_simulated",
    }:
        raise ValueError(
            "pxrd_profile_augmentation_seed requires dataset_class=mp20_tokens or xrdiff_author_rruff_simulated"
        )

    def finish(dataset: Dataset) -> Dataset:
        if structure_cache:
            from src.data.structure_cache import load_structure_cache

            dataset.structure_cache = load_structure_cache(
                root=root,
                tag=structure_cache,
                dataset_class="xrdiff_author_rruff"
                if dataset_class == "xrdiff_author_rruff_simulated"
                else dataset_class,
                split=split,
                real_pxrd_variant=real_pxrd_variant,
            )
        return dataset

    if dataset_class == "mp20_tokens":
        from src.data.mp20_tokens import MP20Tokens

        if structure_cache and (
            str(atom_ordering).strip().lower() != "none"
            or hierarchical_orbit_permutation
        ):
            raise ValueError(
                "canonical structure caches cannot reuse MCFlow orbit metadata"
            )
        return finish(
            MP20Tokens(
                root=root,
                split=split,
                nmax=nmax,
                augment_translate=augment_translate,
                with_pxrd=with_pxrd,
                prop_list=prop_list,
                with_pxrd_peaks=with_pxrd_peaks,
                pxrd_max_peaks=pxrd_max_peaks,
                pxrd_wavelength=pxrd_wavelength,
                pxrd_two_theta_min=real_pxrd_two_theta_min,
                pxrd_two_theta_max=real_pxrd_two_theta_max,
                pxrd_peak_augment=pxrd_peak_augment,
                pxrd_aug_jitter_deg=pxrd_aug_jitter_deg,
                pxrd_aug_int_scale_min=pxrd_aug_int_scale_min,
                pxrd_aug_int_scale_max=pxrd_aug_int_scale_max,
                pxrd_aug_width_scale_min=pxrd_aug_width_scale_min,
                pxrd_aug_width_scale_max=pxrd_aug_width_scale_max,
                pxrd_profile_augment=pxrd_profile_augment,
                pxrd_profile_augmentation_seed=pxrd_profile_augmentation_seed,
                pxrd_profile_augment_probability=pxrd_profile_augment_probability,
                pxrd_aug_zero_shift_deg=pxrd_aug_zero_shift_deg,
                pxrd_aug_eta_min=pxrd_aug_eta_min,
                pxrd_aug_eta_max=pxrd_aug_eta_max,
                pxrd_aug_background_max=pxrd_aug_background_max,
                pxrd_aug_noise_std_max=pxrd_aug_noise_std_max,
                pxrd_aug_drop_frac_max=pxrd_aug_drop_frac_max,
                pxrd_peak_selection=pxrd_peak_selection,
                pxrd_low_angle_fraction=pxrd_low_angle_fraction,
                pxrd_representation=pxrd_representation,
                pxrd_peak_source=pxrd_peak_source,
                pxrd_peak_features=pxrd_peak_features,
                pxrd_peak_position=pxrd_peak_position,
                pxrd_peak_position_normalized=pxrd_peak_position_normalized,
                pxrd_peak_intensity_normalization=pxrd_peak_intensity_normalization,
                pxrd_peak_min_relative_intensity=pxrd_peak_min_relative_intensity,
                pxrd_peak_min_prominence=pxrd_peak_min_prominence,
                pxrd_peak_prominence_normalization=pxrd_peak_prominence_normalization,
                pxrd_peak_width_unit=pxrd_peak_width_unit,
                pxrd_peak_width_normalization=pxrd_peak_width_normalization,
                pxrd_feature_source=pxrd_feature_source,
                pxrd_profile_step=pxrd_profile_step,
                pxrd_caglioti_u=pxrd_caglioti_u,
                pxrd_caglioti_v=pxrd_caglioti_v,
                pxrd_caglioti_w=pxrd_caglioti_w,
                pxrd_pseudo_voigt_eta=pxrd_pseudo_voigt_eta,
                pxrd_dense_n_bins=pxrd_dense_n_bins,
                pxrd_augmentation=pxrd_augmentation,
                pxrd_xrdiff_gamma_shape=pxrd_xrdiff_gamma_shape,
                pxrd_xrdiff_gamma_scale=pxrd_xrdiff_gamma_scale,
                pxrd_dense_profile_augment=pxrd_dense_profile_augment,
                atom_ordering=atom_ordering,
                hierarchical_orbit_permutation=hierarchical_orbit_permutation,
                atom_ordering_symprec=atom_ordering_symprec,
            )
        )
    if dataset_class in {"xrdiff_author_rruff", "xrdiff_author_rruff_simulated"}:
        if str(atom_ordering).strip().lower() != "none" or bool(
            hierarchical_orbit_permutation
        ):
            raise ValueError(
                "RRUFF-148 uses its frozen atom order; MCFlow ordering is not supported"
            )
        from src.data.rruff_tokens import RRUFF148Tokens

        return finish(
            RRUFF148Tokens(
                root=root,
                split="all"
                if dataset_class == "xrdiff_author_rruff_simulated"
                and split in {"train", "val"}
                else split,
                nmax=nmax,
                augment_translate=augment_translate,
                prop_list=prop_list,
                with_pxrd=with_pxrd,
                with_pxrd_peaks=with_pxrd_peaks,
                pxrd_max_peaks=pxrd_max_peaks,
                pxrd_wavelength=pxrd_wavelength,
                pxrd_two_theta_min=real_pxrd_two_theta_min,
                pxrd_two_theta_max=real_pxrd_two_theta_max,
                pxrd_peak_augment=pxrd_peak_augment,
                pxrd_aug_jitter_deg=pxrd_aug_jitter_deg,
                pxrd_aug_int_scale_min=pxrd_aug_int_scale_min,
                pxrd_aug_int_scale_max=pxrd_aug_int_scale_max,
                pxrd_aug_width_scale_min=pxrd_aug_width_scale_min,
                pxrd_aug_width_scale_max=pxrd_aug_width_scale_max,
                pxrd_profile_augment=pxrd_profile_augment,
                pxrd_aug_drop_frac_max=pxrd_aug_drop_frac_max,
                pxrd_aug_zero_shift_deg=pxrd_aug_zero_shift_deg,
                pxrd_aug_eta_min=pxrd_aug_eta_min,
                pxrd_aug_eta_max=pxrd_aug_eta_max,
                pxrd_aug_background_max=pxrd_aug_background_max,
                pxrd_aug_noise_std_max=pxrd_aug_noise_std_max,
                pxrd_peak_selection=pxrd_peak_selection,
                pxrd_low_angle_fraction=pxrd_low_angle_fraction,
                pxrd_representation=pxrd_representation,
                pxrd_peak_source="dense"
                if dataset_class
                in {"xrdiff_author_rruff", "xrdiff_author_rruff_simulated"}
                else pxrd_peak_source,
                pxrd_peak_features=pxrd_peak_features,
                pxrd_peak_position=pxrd_peak_position,
                pxrd_peak_position_normalized=pxrd_peak_position_normalized,
                pxrd_peak_intensity_normalization=pxrd_peak_intensity_normalization,
                pxrd_peak_min_relative_intensity=pxrd_peak_min_relative_intensity,
                pxrd_peak_min_prominence=pxrd_peak_min_prominence,
                pxrd_peak_prominence_normalization=pxrd_peak_prominence_normalization,
                pxrd_peak_width_unit=pxrd_peak_width_unit,
                pxrd_peak_width_normalization=pxrd_peak_width_normalization,
                pxrd_profile_step=pxrd_profile_step,
                pxrd_caglioti_u=pxrd_caglioti_u,
                pxrd_caglioti_v=pxrd_caglioti_v,
                pxrd_caglioti_w=pxrd_caglioti_w,
                pxrd_pseudo_voigt_eta=pxrd_pseudo_voigt_eta,
                pxrd_dense_n_bins=pxrd_dense_n_bins,
                pxrd_augmentation=pxrd_augmentation,
                pxrd_xrdiff_gamma_shape=pxrd_xrdiff_gamma_shape,
                pxrd_xrdiff_gamma_scale=pxrd_xrdiff_gamma_scale,
                pxrd_dense_profile_augment=pxrd_dense_profile_augment,
                real_pxrd_peak_preprocessing=real_pxrd_peak_preprocessing,
                **{
                    "theoretical_profile": True,
                    "profile_augmentation_seed": pxrd_profile_augmentation_seed,
                }
                if dataset_class == "xrdiff_author_rruff_simulated"
                else {},
            )
        )
    if dataset_class == "shared_phase":
        from src.two_phase.data import SharedPhaseDataset

        return SharedPhaseDataset(
            root=root,
            split=split,
            reference_root=phase_reference_root,
            nmax=nmax,
            augment_translate=augment_translate,
            atom_ordering=atom_ordering,
            hierarchical_orbit_permutation=hierarchical_orbit_permutation,
            atom_ordering_symprec=atom_ordering_symprec,
            pxrd_representation=pxrd_representation,
        )
    if dataset_class == "pxrd_bench":
        from src.data.pxrd_bench import PXRDBench as dataset_type

        return finish(
            dataset_type(
                root=root,
                split=split,
                nmax=nmax,
                augment_translate=augment_translate,
                with_pxrd=with_pxrd,
                with_pxrd_peaks=with_pxrd_peaks,
                pxrd_max_peaks=pxrd_max_peaks,
                pxrd_wavelength=pxrd_wavelength,
                pxrd_two_theta_min=real_pxrd_two_theta_min,
                pxrd_two_theta_max=real_pxrd_two_theta_max,
                pxrd_peak_augment=pxrd_peak_augment,
                pxrd_profile_augment=pxrd_profile_augment,
                pxrd_aug_jitter_deg=pxrd_aug_jitter_deg,
                pxrd_aug_int_scale_min=pxrd_aug_int_scale_min,
                pxrd_aug_int_scale_max=pxrd_aug_int_scale_max,
                pxrd_aug_width_scale_min=pxrd_aug_width_scale_min,
                pxrd_aug_width_scale_max=pxrd_aug_width_scale_max,
                pxrd_aug_drop_frac_max=pxrd_aug_drop_frac_max,
                pxrd_aug_zero_shift_deg=pxrd_aug_zero_shift_deg,
                pxrd_aug_eta_min=pxrd_aug_eta_min,
                pxrd_aug_eta_max=pxrd_aug_eta_max,
                pxrd_aug_background_max=pxrd_aug_background_max,
                pxrd_aug_noise_std_max=pxrd_aug_noise_std_max,
                pxrd_representation=pxrd_representation,
                pxrd_peak_source=pxrd_peak_source,
                pxrd_peak_features=pxrd_peak_features,
                pxrd_peak_position=pxrd_peak_position,
                pxrd_peak_position_normalized=pxrd_peak_position_normalized,
                pxrd_peak_intensity_normalization=pxrd_peak_intensity_normalization,
                pxrd_peak_prominence_normalization=pxrd_peak_prominence_normalization,
                pxrd_peak_width_unit=pxrd_peak_width_unit,
                pxrd_peak_width_normalization=pxrd_peak_width_normalization,
                pxrd_peak_selection=pxrd_peak_selection,
                pxrd_low_angle_fraction=pxrd_low_angle_fraction,
                pxrd_profile_step=pxrd_profile_step,
                pxrd_caglioti_u=pxrd_caglioti_u,
                pxrd_caglioti_v=pxrd_caglioti_v,
                pxrd_caglioti_w=pxrd_caglioti_w,
                pxrd_pseudo_voigt_eta=pxrd_pseudo_voigt_eta,
                pxrd_dense_n_bins=pxrd_dense_n_bins,
                pxrd_augmentation=pxrd_augmentation,
                pxrd_xrdiff_gamma_shape=pxrd_xrdiff_gamma_shape,
                pxrd_xrdiff_gamma_scale=pxrd_xrdiff_gamma_scale,
                pxrd_dense_profile_augment=pxrd_dense_profile_augment,
                atom_ordering=atom_ordering,
                hierarchical_orbit_permutation=hierarchical_orbit_permutation,
                atom_ordering_symprec=atom_ordering_symprec,
            )
        )
    if str(atom_ordering).strip().lower() != "none" or bool(
        hierarchical_orbit_permutation
    ):
        raise ValueError(
            "MCFlow atom ordering is currently supported only for dataset_class='mp20_tokens'."
        )
    if dataset_class == "real_pxrd":
        from src.data.real_pxrd_tokens import RealPXRDTokens

        experimental_representation = _experimental_representation(
            pxrd_representation,
            peak_source=pxrd_peak_source,
            peak_features=pxrd_peak_features,
            peak_position=pxrd_peak_position,
            position_normalized=pxrd_peak_position_normalized,
            feature_source=pxrd_feature_source,
        )
        return finish(
            RealPXRDTokens(
                root=root,
                split=split,
                nmax=nmax,
                augment_translate=augment_translate,
                variant=real_pxrd_variant,
                in_memory=real_pxrd_in_memory,
                with_pxrd=with_pxrd,
                prop_list=prop_list,
                pxrd_n_bins=real_pxrd_n_bins,
                pxrd_two_theta_min=real_pxrd_two_theta_min,
                pxrd_two_theta_max=real_pxrd_two_theta_max,
                pxrd_sigma=real_pxrd_sigma,
                with_pxrd_peaks=with_pxrd_peaks,
                pxrd_max_peaks=pxrd_max_peaks,
                pxrd_wavelength=pxrd_wavelength,
                pxrd_peak_augment=pxrd_peak_augment,
                pxrd_peak_preprocessing=real_pxrd_peak_preprocessing,
                pxrd_aug_jitter_deg=pxrd_aug_jitter_deg,
                pxrd_aug_int_scale_min=pxrd_aug_int_scale_min,
                pxrd_aug_int_scale_max=pxrd_aug_int_scale_max,
                pxrd_aug_drop_frac_max=pxrd_aug_drop_frac_max,
                pxrd_peak_selection=pxrd_peak_selection,
                pxrd_low_angle_fraction=pxrd_low_angle_fraction,
                pxrd_representation=experimental_representation,
                pxrd_peak_features=pxrd_peak_features,
                pxrd_peak_position=pxrd_peak_position,
                pxrd_peak_position_normalized=pxrd_peak_position_normalized,
                pxrd_peak_width_normalization=pxrd_peak_width_normalization,
                pxrd_profile_step=pxrd_profile_step,
                pxrd_caglioti_u=pxrd_caglioti_u,
                pxrd_caglioti_v=pxrd_caglioti_v,
                pxrd_caglioti_w=pxrd_caglioti_w,
                pxrd_pseudo_voigt_eta=pxrd_pseudo_voigt_eta,
            )
        )
    if dataset_class in {"experimental_pxrd", "frozen_phase_pxrd"}:
        from src.data.experimental_pxrd import ExperimentalPXRDTokens

        if dataset_class == "frozen_phase_pxrd" and Path(root).suffix != ".json":
            raise ValueError("frozen_phase_pxrd requires its hashed JSON manifest")
        experimental_representation = _experimental_representation(
            pxrd_representation,
            peak_source=pxrd_peak_source,
            peak_features=pxrd_peak_features,
            peak_position=pxrd_peak_position,
            position_normalized=pxrd_peak_position_normalized,
            feature_source=pxrd_feature_source,
        )
        return ExperimentalPXRDTokens(
            pxrd_peak_min_prominence=pxrd_peak_min_prominence,
            root=root,
            split=split,
            nmax=nmax,
            pxrd_representation=experimental_representation,
            pxrd_max_peaks=pxrd_max_peaks,
            pxrd_wavelength=pxrd_wavelength,
            pxrd_peak_features=pxrd_peak_features,
            pxrd_peak_position=pxrd_peak_position,
            pxrd_peak_position_normalized=pxrd_peak_position_normalized,
            pxrd_peak_width_normalization=pxrd_peak_width_normalization,
            pxrd_peak_selection=pxrd_peak_selection,
            pxrd_profile_step=pxrd_profile_step,
            pxrd_caglioti_u=pxrd_caglioti_u,
            pxrd_caglioti_v=pxrd_caglioti_v,
            pxrd_caglioti_w=pxrd_caglioti_w,
            pxrd_pseudo_voigt_eta=pxrd_pseudo_voigt_eta,
            background_subtraction=experimental_background_subtraction,
            pxrd_dense_n_bins=pxrd_dense_n_bins,
            real_pxrd_peak_preprocessing=real_pxrd_peak_preprocessing,
            with_pxrd=with_pxrd,
        )
    raise ValueError(
        f"Unknown dataset_class {dataset_class!r}; expected one of {DATASET_CLASSES}."
    )
