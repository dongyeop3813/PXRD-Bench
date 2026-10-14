from __future__ import annotations
import os
import math
import re
import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from pymatgen.io.cif import CifWriter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from src.utils.seeding import stable_seed

NMAX = 20
VZ = 94
ATOM_ORDERING_CHOICES = ("none", "mcflow")
MCFLOW_ORDERING_CACHE_VERSION = 1


def lattice_to_Y(lengths, angles_deg):
    """
    Encode lattice parameters to Y representation.
    """
    a, b, c = lengths
    alpha, beta, gamma = angles_deg
    return np.array(
        [
            np.log(a),
            np.log(b),
            np.log(c),
            np.cos(np.deg2rad(alpha)),
            np.cos(np.deg2rad(beta)),
            np.cos(np.deg2rad(gamma)),
        ],
        dtype=np.float32,
    )


def pack_structure_tokens(
    atom_types, frac_coords, lengths, angles, *, nmax, num_atoms, mp_id
):
    """Pad one structure into the token schema shared by structure datasets."""
    n = int(num_atoms)
    a0 = np.zeros((nmax,), dtype=np.int64)
    a0[:n] = np.asarray(atom_types, dtype=np.int64)[:n]
    f1 = np.zeros((nmax, 3), dtype=np.float32)
    f1[:n] = np.asarray(frac_coords, dtype=np.float32)[:n] % 1.0
    pad_mask = np.ones((nmax,), dtype=np.bool_)
    pad_mask[:n] = False
    return {
        "mp_id": mp_id,
        "A0": torch.from_numpy(a0),
        "F1": torch.from_numpy(f1),
        "Y1": torch.from_numpy(lattice_to_Y(lengths, angles)),
        "pad_mask": torch.from_numpy(pad_mask),
        "num_atoms": n,
    }


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _wyckoff_letter(symbol: str) -> str:
    """Return the Wyckoff letter from pymatgen symbols such as ``"4c"``."""
    match = re.search("([a-zA-Z])$", str(symbol).strip())
    if match is None:
        raise ValueError(f"Invalid Wyckoff symbol: {symbol!r}.")
    return match.group(1).lower()


def _electronegativity_sort_key(atomic_number: int) -> tuple[int, float]:
    """Ascending Pauling electronegativity with a deterministic noble-gas fallback."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        value = float(Element.from_Z(int(atomic_number)).X)
    if math.isfinite(value):
        return (0, value)
    return (1, float(atomic_number))


def mcflow_canonical_order(
    atom_types, frac_coords, lengths, angles, *, symprec: float = 0.1
) -> tuple[np.ndarray, dict]:
    """MCFlow orbit ordering and metadata for hierarchical augmentation.

    Orbits are sorted by ascending Pauling electronegativity and then by
    Wyckoff letter. Atom order inside an orbit is deliberately left unresolved;
    MCFlow handles that residual symmetry with intra-orbit permutation.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    atom_types = np.asarray(atom_types, dtype=np.int64)
    frac_coords = np.asarray(frac_coords, dtype=np.float32)
    lattice = Lattice.from_parameters(
        *list(np.asarray(lengths, dtype=float)) + list(np.asarray(angles, dtype=float))
    )
    structure = Structure(
        lattice=lattice,
        species=[Element.from_Z(int(z)) for z in atom_types],
        coords=frac_coords,
        coords_are_cartesian=False,
    )
    symmetrized = SpacegroupAnalyzer(
        structure, symprec=float(symprec)
    ).get_symmetrized_structure()
    equivalent_indices = list(symmetrized.equivalent_indices)
    wyckoff_symbols = list(symmetrized.wyckoff_symbols)
    if len(equivalent_indices) != len(wyckoff_symbols):
        raise ValueError(
            f"Symmetry analysis returned mismatched orbit and Wyckoff metadata: {len(equivalent_indices)} != {len(wyckoff_symbols)}."
        )
    entries = []
    seen = []
    for indices, symbol in zip(equivalent_indices, wyckoff_symbols, strict=True):
        indices = tuple((int(i) for i in indices))
        if not indices:
            raise ValueError("Symmetry analysis returned an empty orbit.")
        orbit_types = {int(atom_types[i]) for i in indices}
        if len(orbit_types) != 1:
            raise ValueError(
                f"Symmetry orbit mixes atom types: indices={indices}, types={orbit_types}."
            )
        atomic_number = orbit_types.pop()
        normalized_symbol = str(symbol).strip().lower()
        letter = _wyckoff_letter(normalized_symbol)
        en_key = _electronegativity_sort_key(atomic_number)
        entries.append(
            {
                "indices": indices,
                "atomic_number": atomic_number,
                "wyckoff_letter": letter,
                "wyckoff_symbol": normalized_symbol,
                "sort_key": (*en_key, letter, normalized_symbol, atomic_number),
            }
        )
        seen.extend(indices)
    if sorted(seen) != list(range(len(atom_types))):
        raise ValueError(
            f"Symmetry orbits do not partition all atoms exactly once: seen={sorted(seen)}, expected={list(range(len(atom_types)))}."
        )
    entries.sort(key=lambda entry: entry["sort_key"])
    order = np.asarray(
        [index for entry in entries for index in entry["indices"]], dtype=np.int64
    )
    orbit_slices = []
    inter_keys: dict[tuple[int, str], list[int]] = {}
    offset = 0
    for orbit_id, entry in enumerate(entries):
        size = len(entry["indices"])
        orbit_slices.append(tuple(range(offset, offset + size)))
        offset += size
        inter_keys.setdefault(
            (entry["atomic_number"], entry["wyckoff_symbol"]), []
        ).append(orbit_id)
    inter_groups = [tuple(group) for group in inter_keys.values() if len(group) > 1]
    for group in inter_groups:
        sizes = {len(orbit_slices[orbit_id]) for orbit_id in group}
        if len(sizes) != 1:
            raise ValueError(
                f"Tied MCFlow orbits must have equal multiplicity for inter-orbit permutation, got sizes={sorted(sizes)}."
            )
    return (
        order,
        {"orbits": tuple(orbit_slices), "inter_orbit_groups": tuple(inter_groups)},
    )


def mcflow_hierarchical_permutation(item: dict, rng=None) -> torch.Tensor:
    """Return a symmetry-preserving permutation over real and padded slots."""
    if rng is None:
        rng = np.random
    orbits = [tuple((int(i) for i in orbit)) for orbit in item["_mcflow_orbits"]]
    replacements = list(range(len(orbits)))
    for group in item["_mcflow_inter_orbit_groups"]:
        group = [int(i) for i in group]
        shuffled = list(rng.permutation(group))
        for destination, source in zip(group, shuffled, strict=True):
            replacements[destination] = source
    real_order = []
    for source in replacements:
        real_order.extend((int(i) for i in rng.permutation(orbits[source])))
    num_atoms = int(item["num_atoms"])
    if sorted(real_order) != list(range(num_atoms)):
        raise ValueError(
            f"MCFlow hierarchical permutation did not preserve real atom slots: order={real_order}, num_atoms={num_atoms}."
        )
    nmax = int(item["pad_mask"].numel())
    return torch.tensor(real_order + list(range(num_atoms, nmax)), dtype=torch.long)


def decode_Y1(y1):
    y1 = _to_numpy(y1)
    lengths = np.exp(y1[:3])
    angles = np.degrees(np.arccos(np.clip(y1[3:], -1.0, 1.0)))
    return (lengths.astype(np.float32), angles.astype(np.float32))


def tokens_to_eval_dict(item: dict, sample_idx: int = 0) -> dict:
    mask = ~_to_numpy(item["pad_mask"]).astype(bool)
    atom_types = _to_numpy(item["A0"])[mask]
    frac_coords = _to_numpy(item["F1"])[mask]
    lengths, angles = decode_Y1(item["Y1"])
    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
        "sample_idx": sample_idx,
    }


def tokens_to_structure(item: dict):
    mask = ~_to_numpy(item["pad_mask"]).astype(bool)
    atom_types = _to_numpy(item["A0"])[mask]
    if (atom_types <= 0).any():
        raise ValueError("Found non-positive atomic number in unpadded positions.")
    species = [Element.from_Z(int(z)) for z in atom_types]
    frac_coords = _to_numpy(item["F1"])[mask]
    lengths, angles = decode_Y1(item["Y1"])
    lattice = Lattice.from_parameters(*lengths.tolist() + angles.tolist())
    return Structure(
        lattice=lattice, species=species, coords=frac_coords, coords_are_cartesian=False
    )


def _simulate_one_peak(item, wavelength, tt_min, tt_max):
    """Worker: token item -> (mp_id, two_theta, intensity).

    Module-level (picklable) so it runs under joblib/loky. Failures yield empty
    arrays so the record is dropped downstream rather than killing the pool.
    """
    from src.pxrd.bragg_simulator import fast_xrd_pattern

    mid = item["mp_id"]
    try:
        struct = tokens_to_structure(item)
        two_theta, intensity = fast_xrd_pattern(
            struct, wavelength=wavelength, two_theta_range=(tt_min, tt_max)
        )
        return (mid, two_theta.astype(np.float32), intensity.astype(np.float32))
    except Exception:
        return (mid, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))


def _augment_lattice_and_simulate(
    item: dict, *, sigma: float, wavelength: float, tt_min: float, tt_max: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Keep perturbed lattice lengths only when PXRD resimulation succeeds."""
    candidate = dict(item)
    candidate["Y1"] = item["Y1"].clone()
    candidate["Y1"][:3] += torch.randn(3).clamp_(-3.0, 3.0) * float(sigma)
    _, two_theta, intensity = _simulate_one_peak(
        candidate, float(wavelength), float(tt_min), float(tt_max)
    )
    if not two_theta.size:
        return None
    item["Y1"] = candidate["Y1"]
    return (two_theta, intensity)


def tokens_batch_to_structures(batch: dict):
    structs = []
    batch_size = _to_numpy(batch["A0"]).shape[0]
    for i in range(batch_size):
        item = {
            "A0": batch["A0"][i],
            "F1": batch["F1"][i],
            "Y1": batch["Y1"][i],
            "pad_mask": batch["pad_mask"][i],
        }
        mask = ~_to_numpy(item["pad_mask"]).astype(bool)
        atom_types = _to_numpy(item["A0"])[mask]
        if (atom_types <= 0).any():
            raise ValueError("Found non-positive atomic number in unpadded positions.")
        species = [Element.from_Z(int(z)) for z in atom_types]
        frac_coords = _to_numpy(item["F1"])[mask]
        lengths, angles = decode_Y1(item["Y1"])
        lattice = Lattice.from_parameters(*lengths.tolist() + angles.tolist())
        structs.append(
            Structure(
                lattice=lattice,
                species=species,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
        )
    return structs


def tokens_batch_to_cif_strings(batch: dict) -> list[str]:
    from pymatgen.io.cif import CifWriter

    return [str(CifWriter(struct)) for struct in tokens_batch_to_structures(batch)]


def tokens_to_cif_string(item: dict) -> str:
    struct = tokens_to_structure(item)
    return str(CifWriter(struct))


def translate_frac_coords(frac_coords, pad_mask, rng=None):
    coords = _to_numpy(frac_coords).copy()
    mask = ~_to_numpy(pad_mask).astype(bool)
    if mask.any():
        if rng is None:
            delta = np.random.random(3).astype(np.float32)
        else:
            try:
                delta = rng.random(3, dtype=np.float32)
            except TypeError:
                delta = np.asarray(rng.random(3), dtype=np.float32)
        coords[mask] = (coords[mask] + delta) % 1.0
    return coords


def collate_mp20_tokens(batch):
    """Simple collate function for MP20Tokens dataset."""
    out = {
        "mp_id": [b["mp_id"] for b in batch],
        "A0": torch.stack([b["A0"] for b in batch], dim=0),
        "F1": torch.stack([b["F1"] for b in batch], dim=0),
        "Y1": torch.stack([b["Y1"] for b in batch], dim=0),
        "pad_mask": torch.stack([b["pad_mask"] for b in batch], dim=0),
        "num_atoms": torch.tensor([b["num_atoms"] for b in batch], dtype=torch.long),
    }
    if "sg_number" in batch[0]:
        out["sg_number"] = torch.stack(
            [torch.as_tensor(b["sg_number"], dtype=torch.long) for b in batch], dim=0
        )
    if "mcflow_orbit_ids" in batch[0]:
        out["mcflow_orbit_ids"] = torch.stack(
            [b["mcflow_orbit_ids"] for b in batch], dim=0
        )
    if "elem_set_mask" in batch[0]:
        out["elem_set_mask"] = torch.stack(
            [torch.as_tensor(b["elem_set_mask"]).bool() for b in batch], dim=0
        )
    if "complexity_importance_weight" in batch[0]:
        out["complexity_importance_weight"] = torch.stack(
            [torch.as_tensor(b["complexity_importance_weight"]) for b in batch], dim=0
        ).float()
    if "xrd" in batch[0]:
        out["xrd"] = torch.stack([b["xrd"] for b in batch], dim=0)
    if "pxrd_dense_profile" in batch[0]:
        out["pxrd_dense_profile"] = torch.stack(
            [b["pxrd_dense_profile"] for b in batch], dim=0
        )
    if "pxrd_d" in batch[0]:
        out["pxrd_d"] = torch.stack([b["pxrd_d"] for b in batch], dim=0)
        out["pxrd_i"] = torch.stack([b["pxrd_i"] for b in batch], dim=0)
        out["pxrd_peak_pad_mask"] = torch.stack(
            [b["pxrd_peak_pad_mask"] for b in batch], dim=0
        )
        for key in ("pxrd_position", "pxrd_intensity", "pxrd_prominence", "pxrd_width"):
            if key in batch[0]:
                out[key] = torch.stack([b[key] for b in batch], dim=0)
        for key in batch[0]:
            if key.startswith("pxrd_peak_list_") and torch.is_tensor(batch[0][key]):
                out[key] = torch.stack([b[key] for b in batch], dim=0)
        if "pxrd_peak_list_source" in batch[0]:
            out["pxrd_peak_list_source"] = [b["pxrd_peak_list_source"] for b in batch]
    return out


class MP20Tokens(Dataset):
    """
    Plain PyTorch dataset returning dicts with:
      A0: (NMAX,) long
      F1: (NMAX,3) float
      Y1: (6,) float
      pad_mask: (NMAX,) bool   True where padded
    """

    def __init__(
        self,
        root: str,
        force_reprocess: bool = False,
        augment_translate: bool = False,
        prop_list: list[str] | None = None,
        split: str = "all",
        nmax: int = NMAX,
        with_pxrd: bool = False,
        pxrd_trim_last: bool = True,
        pxrd_dtype: torch.dtype = torch.float32,
        with_pxrd_peaks: bool = False,
        pxrd_max_peaks: int = 128,
        pxrd_wavelength: float = 1.54184,
        pxrd_two_theta_min: float = 5.0,
        pxrd_two_theta_max: float = 90.0,
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
    ):
        self.root = root
        self.force_reprocess = bool(force_reprocess)
        self.augment_translate = augment_translate
        self.atom_ordering = str(atom_ordering).strip().lower()
        if self.atom_ordering not in ATOM_ORDERING_CHOICES:
            raise ValueError(
                f"atom_ordering must be one of {ATOM_ORDERING_CHOICES}, got {atom_ordering!r}."
            )
        self.hierarchical_orbit_permutation = bool(hierarchical_orbit_permutation)
        if self.hierarchical_orbit_permutation and self.atom_ordering != "mcflow":
            raise ValueError(
                "hierarchical_orbit_permutation requires atom_ordering='mcflow'."
            )
        self.atom_ordering_symprec = float(atom_ordering_symprec)
        if self.atom_ordering_symprec <= 0.0:
            raise ValueError("atom_ordering_symprec must be positive.")
        self.prop_list = prop_list or []
        self.split = split
        self.nmax = int(nmax)
        self.with_pxrd = with_pxrd
        self.pxrd_trim_last = pxrd_trim_last
        self.pxrd_dtype = pxrd_dtype
        self.with_pxrd_peaks = bool(with_pxrd_peaks)
        self.pxrd_max_peaks = int(pxrd_max_peaks)
        self.pxrd_wavelength = float(pxrd_wavelength)
        self.pxrd_two_theta_min = float(pxrd_two_theta_min)
        self.pxrd_two_theta_max = float(pxrd_two_theta_max)
        self.pxrd_peak_augment = bool(pxrd_peak_augment)
        self.pxrd_aug_jitter_deg = float(pxrd_aug_jitter_deg)
        self.pxrd_aug_int_scale_min = float(pxrd_aug_int_scale_min)
        self.pxrd_aug_int_scale_max = float(pxrd_aug_int_scale_max)
        self.pxrd_aug_width_scale_min = float(pxrd_aug_width_scale_min)
        self.pxrd_aug_width_scale_max = float(pxrd_aug_width_scale_max)
        if (
            pxrd_dense_profile_augment is not None
            and bool(pxrd_profile_augment) != bool(pxrd_dense_profile_augment)
            and bool(pxrd_profile_augment)
        ):
            raise ValueError(
                "pxrd_profile_augment and pxrd_dense_profile_augment disagree"
            )
        self.pxrd_profile_augment = bool(
            pxrd_profile_augment
            if pxrd_dense_profile_augment is None
            else pxrd_dense_profile_augment
        )
        self.pxrd_dense_profile_augment = self.pxrd_profile_augment
        self.pxrd_profile_augment_probability = float(pxrd_profile_augment_probability)
        if not 0.0 <= self.pxrd_profile_augment_probability <= 1.0:
            raise ValueError("pxrd_profile_augment_probability must be in [0, 1]")
        self.profile_augmentation_seed = (
            None
            if pxrd_profile_augmentation_seed is None
            else int(pxrd_profile_augmentation_seed)
        )
        if self.profile_augmentation_seed is not None and (
            not self.pxrd_profile_augment
        ):
            raise ValueError(
                "pxrd_profile_augmentation_seed requires pxrd_profile_augment=true"
            )
        self.pxrd_aug_zero_shift_deg = float(pxrd_aug_zero_shift_deg)
        self.pxrd_aug_eta_min = float(pxrd_aug_eta_min)
        self.pxrd_aug_eta_max = float(pxrd_aug_eta_max)
        self.pxrd_aug_background_max = float(pxrd_aug_background_max)
        self.pxrd_aug_noise_std_max = float(pxrd_aug_noise_std_max)
        self.pxrd_aug_drop_frac_max = float(pxrd_aug_drop_frac_max)
        self.pxrd_peak_selection = str(pxrd_peak_selection)
        self.pxrd_low_angle_fraction = float(pxrd_low_angle_fraction)
        from src.pxrd.representation import (
            PUBLIC_REPRESENTATIONS,
            legacy_representation,
            normalize_peak_features,
            resolve_representation,
        )

        requested_representation = str(pxrd_representation)
        semantic_representation, semantic_source = resolve_representation(
            requested_representation, peak_source=pxrd_peak_source
        )
        self.pxrd_peak_fields = normalize_peak_features(
            pxrd_peak_features, position=pxrd_peak_position
        )
        self.pxrd_public_api = requested_representation in PUBLIC_REPRESENTATIONS
        self.pxrd_semantic_representation = semantic_representation
        self.pxrd_peak_source = semantic_source
        self.pxrd_representation = legacy_representation(
            requested_representation,
            peak_source=semantic_source,
            peak_features=self.pxrd_peak_fields,
            position_normalized=bool(pxrd_peak_position_normalized),
            feature_source=str(pxrd_feature_source),
        )
        self.pxrd_peak_features = str(pxrd_peak_features)
        self.pxrd_peak_position = str(pxrd_peak_position)
        self.pxrd_peak_position_normalized = bool(pxrd_peak_position_normalized)
        self.pxrd_peak_intensity_normalization = str(pxrd_peak_intensity_normalization)
        self.pxrd_peak_min_prominence = float(pxrd_peak_min_prominence)
        if not 0.0 <= self.pxrd_peak_min_prominence <= 1.0:
            raise ValueError("pxrd_peak_min_prominence must be in [0, 1]")
        if self.pxrd_peak_min_prominence and self.pxrd_representation != "peaks_128x4":
            raise ValueError("pxrd_peak_min_prominence requires peaks_128x4")
        self.pxrd_peak_min_relative_intensity = float(pxrd_peak_min_relative_intensity)
        self.pxrd_peak_prominence_normalization = str(
            pxrd_peak_prominence_normalization
        )
        self.pxrd_peak_width_unit = str(pxrd_peak_width_unit)
        self.pxrd_peak_width_normalization = str(pxrd_peak_width_normalization)
        self.pxrd_feature_source = str(pxrd_feature_source)
        self.pxrd_profile_step = float(pxrd_profile_step)
        self.pxrd_caglioti_u = float(pxrd_caglioti_u)
        self.pxrd_caglioti_v = float(pxrd_caglioti_v)
        self.pxrd_caglioti_w = float(pxrd_caglioti_w)
        self.pxrd_pseudo_voigt_eta = float(pxrd_pseudo_voigt_eta)
        self.pxrd_dense_n_bins = int(pxrd_dense_n_bins)
        self.pxrd_augmentation = str(pxrd_augmentation)
        self.pxrd_xrdiff_gamma_shape = (
            None if pxrd_xrdiff_gamma_shape is None else float(pxrd_xrdiff_gamma_shape)
        )
        self.pxrd_xrdiff_gamma_scale = (
            None if pxrd_xrdiff_gamma_scale is None else float(pxrd_xrdiff_gamma_scale)
        )
        if self.pxrd_profile_augment:
            if self.pxrd_augmentation == "none":
                self.pxrd_augmentation = "profile_randomization"
        else:
            self.pxrd_augmentation = "none"
        if self.pxrd_feature_source == "auto":
            self.pxrd_feature_source = (
                "online" if self.pxrd_profile_augment else "cached"
            )
        expected_features = {
            "full_850": 850,
            "peaks_20x4": 80,
            "peaks_128x4": 512,
            "peaks_20x4_norm_prom": 80,
            "peaks_128x4_norm_prom": 512,
        }
        if self.pxrd_representation not in {
            "legacy_di",
            "broadened_basic",
            "broadened_shape",
            "peak_matrix",
            "dense_profile",
            *expected_features,
        }:
            raise ValueError(f"unsupported PXRD representation: {pxrd_representation}")
        if self.pxrd_representation in expected_features:
            expected = expected_features[self.pxrd_representation]
            if self.pxrd_representation == "peaks_128x4":
                if not 4 <= self.pxrd_max_peaks <= 1696 or self.pxrd_max_peaks % 4:
                    raise ValueError("peaks_128x4 requires 1..424 four-feature rows")
                expected = self.pxrd_max_peaks
            if self.pxrd_max_peaks != expected:
                raise ValueError(
                    f"{self.pxrd_representation} requires pxrd_max_peaks={expected}"
                )
            if self.pxrd_peak_augment and self.pxrd_representation != "peaks_128x4":
                raise ValueError("XRDiff representations require pxrd_augment=false")
        if (
            self.pxrd_representation in {"broadened_basic", "broadened_shape"}
            and self.pxrd_peak_augment
        ):
            raise ValueError(
                "broadened PXRD representations require pxrd_augment=false"
            )
        if self.pxrd_profile_augment and self.pxrd_representation != "peaks_128x4":
            if self.pxrd_representation not in {"peak_matrix", "dense_profile"}:
                raise ValueError(
                    "pxrd_dense_profile_augment requires dense-derived peak_list or dense_profile"
                )
        if self.pxrd_representation == "peak_matrix":
            if not self.pxrd_public_api and self.pxrd_peak_features not in {
                "width_prominence_intensity_position",
                "position_intensity",
            }:
                raise ValueError(
                    f"unsupported peak features: {self.pxrd_peak_features}"
                )
            if self.pxrd_peak_position not in {"d", "q", "theta", "two_theta"}:
                raise ValueError(
                    f"unsupported peak position: {self.pxrd_peak_position}"
                )
            if self.pxrd_peak_width_normalization not in {
                "none",
                "all_detected",
                "selected_topk",
            }:
                raise ValueError(
                    f"unsupported width normalization: {self.pxrd_peak_width_normalization}"
                )
            if self.pxrd_feature_source not in {"cached", "online"}:
                raise ValueError(
                    f"unsupported feature source: {self.pxrd_feature_source}"
                )
        if self.pxrd_dense_n_bins <= 0:
            raise ValueError("pxrd_dense_n_bins must be positive")
        if self.nmax <= 0:
            raise ValueError(f"nmax must be > 0, got {self.nmax}.")
        split_name = "all" if split is None else split
        self.split_name = split_name
        self.cache_suffix = "" if self.nmax == NMAX else f"_nmax{self.nmax}"
        ordering_suffix = ""
        if self.atom_ordering == "mcflow":
            symprec_token = f"{self.atom_ordering_symprec:g}".replace(".", "p")
            ordering_suffix = (
                f"_order_mcflow_v{MCFLOW_ORDERING_CACHE_VERSION}_symprec{symprec_token}"
            )
        self.processed_cache_suffix = self.cache_suffix + ordering_suffix
        self.raw_csv = os.path.join(root, "raw", f"{split_name}.csv")
        self.proc_pt = os.path.join(
            root,
            "processed",
            f"mp20_tokens_{split_name}{self.processed_cache_suffix}.pt",
        )
        if force_reprocess:
            raise ValueError(
                "Dataset generation is not included; provide frozen MP-20 token caches"
            )
        if not os.path.isfile(self.proc_pt):
            raise FileNotFoundError(f"Missing frozen MP-20 token cache: {self.proc_pt}")
        self.items = torch.load(self.proc_pt, weights_only=False)
        stored_dense_source = bool(
            self.with_pxrd_peaks
            and self.pxrd_public_api
            and (self.pxrd_peak_source == "dense")
        )
        if self.with_pxrd or stored_dense_source:
            self._attach_pxrd()
        if self.with_pxrd_peaks and (not stored_dense_source):
            self._attach_pxrd_peaks()

    def _attach_pxrd_peaks(self):
        """Simulate discrete (2theta, intensity) Bragg peaks per structure via
        pymatgen ``XRDCalculator`` (CuKa, [tmin, tmax] window) and attach the raw
        ``_pxrd_x``/``_pxrd_y`` arrays; ``__getitem__`` packs (and optionally
        augments) them into the d-I tensors the encoder consumes.

        The simulation is cached per structure digest + split + diffraction
        settings so the ~minutes-long XRDCalculator pass runs once.
        """
        import numpy as np

        cache_dir = os.path.join(self.root, "pxrd_peaks")
        from src.pxrd.bragg_cache import structure_items_sha256

        source_sha256 = structure_items_sha256(self.items)
        cache_path = os.path.join(
            cache_dir,
            f"bragg-v2_{self.split_name}{self.cache_suffix}_wl{self.pxrd_wavelength:.5f}_tt{self.pxrd_two_theta_min:g}-{self.pxrd_two_theta_max:g}_src{source_sha256[:12]}.pt",
        )
        from src.pxrd.bragg_cache import load_bragg_cache

        manifest = {
            "split": self.split_name,
            "wavelength_angstrom": self.pxrd_wavelength,
            "two_theta_range_deg": [self.pxrd_two_theta_min, self.pxrd_two_theta_max],
            "source_structures_sha256": source_sha256,
        }
        expected_ids = [item["mp_id"] for item in self.items]
        cache = load_bragg_cache(
            cache_path, expected_manifest=manifest, expected_ids=expected_ids
        )
        peaks = cache.as_dict()
        features = None
        cache_features = (
            not getattr(self, "pxrd_public_api", False)
            and self.pxrd_representation != "legacy_di"
            and (
                self.pxrd_representation == "peak_matrix"
                and self.pxrd_feature_source == "cached"
                or (
                    self.pxrd_representation != "peak_matrix"
                    and (not self.pxrd_profile_augment)
                )
            )
        )
        if cache_features:

            def token(value):
                return f"{value:g}".replace("-", "m").replace(".", "p")

            common_tag = f"_k{self.pxrd_max_peaks}_wl{self.pxrd_wavelength:.5f}_tt{token(self.pxrd_two_theta_min)}-{token(self.pxrd_two_theta_max)}_src{source_sha256[:12]}"
            representation_tag = (
                f"_prom{token(self.pxrd_peak_min_prominence)}"
                if self.pxrd_peak_min_prominence
                else ""
            )
            if self.pxrd_representation == "peak_matrix":
                representation_tag = f"_{self.pxrd_peak_features}_{self.pxrd_peak_position}_posnorm{int(self.pxrd_peak_position_normalized)}_width{self.pxrd_peak_width_normalization}_select{self.pxrd_peak_selection}_low{token(self.pxrd_low_angle_fraction)}"
            elif self.pxrd_representation in {"broadened_basic", "broadened_shape"}:
                representation_tag = f"_step{token(self.pxrd_profile_step)}_uvw{token(self.pxrd_caglioti_u)}-{token(self.pxrd_caglioti_v)}-{token(self.pxrd_caglioti_w)}_eta{token(self.pxrd_pseudo_voigt_eta)}"
            feature_path = os.path.join(
                cache_dir,
                f"xrdiff_{self.pxrd_representation}{common_tag}{representation_tag}_{self.split_name}{self.cache_suffix}.pt",
            )
            if os.path.exists(feature_path):
                features = torch.load(feature_path, weights_only=False)
            else:
                raise FileNotFoundError(
                    f"Missing frozen PXRD feature cache: {feature_path}"
                )
        kept = []
        dropped = 0
        for item in self.items:
            entry = peaks.get(item["mp_id"])
            if entry is None or np.asarray(entry[0]).size == 0:
                dropped += 1
                continue
            new_item = dict(item)
            if features is None:
                new_item["_pxrd_x"] = np.asarray(entry[0], dtype=np.float32)
                new_item["_pxrd_y"] = np.asarray(entry[1], dtype=np.float32)
            elif self.pxrd_representation in {"broadened_basic", "broadened_shape"}:
                new_item["_broadened_features"] = features[item["mp_id"]]
            else:
                new_item["_xrdiff_features"] = np.asarray(
                    features[item["mp_id"]], dtype=np.float32
                )
            kept.append(new_item)
        self.items = kept
        self.pxrd_peaks_dropped = dropped

    def _finalize_peaks(
        self, out: dict, *, rng: np.random.Generator | None = None
    ) -> None:
        """Replace raw ``_pxrd_x/_pxrd_y`` with packed (augmented) d-I tensors."""
        from src.pxrd.peaks import (
            augment_peak_matrix_features,
            augment_peak128_features,
            augment_peaks,
            augmented_peak128_feature_vector,
            peak_matrix_feature_vector_from_profile,
            peaks_to_di_tensors,
            pseudo_voigt_profile,
            xrdiff_feature_vector,
        )

        if rng is None:
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        profile_augment_probability = getattr(
            self, "pxrd_profile_augment_probability", 1.0
        )
        apply_profile_augment = bool(
            self.pxrd_profile_augment
            and (
                profile_augment_probability >= 1.0
                or (
                    profile_augment_probability > 0.0
                    and rng.random() < profile_augment_probability
                )
            )
        )
        apply_peak_augment = bool(
            self.pxrd_peak_augment
            and (not self.pxrd_profile_augment or apply_profile_augment)
        )
        public_online_profile = bool(
            getattr(self, "pxrd_public_api", False)
            and self.pxrd_peak_source == "bragg"
            and (self.pxrd_feature_source == "online")
            and (self.pxrd_representation == "peak_matrix")
        )
        exact_peak128_di = bool(
            public_online_profile
            and self.pxrd_peak_features == "position_intensity"
            and (self.pxrd_peak_position == "d")
            and (not self.pxrd_peak_position_normalized)
            and (self.pxrd_peak_selection == "strongest")
            and (self.pxrd_max_peaks == 128)
        )
        if exact_peak128_di:
            features = (
                (
                    augmented_peak128_feature_vector(
                        out.pop("_pxrd_x"),
                        out.pop("_pxrd_y"),
                        jitter_deg=self.pxrd_aug_jitter_deg,
                        int_scale_min=self.pxrd_aug_int_scale_min,
                        int_scale_max=self.pxrd_aug_int_scale_max,
                        width_scale_min=self.pxrd_aug_width_scale_min,
                        width_scale_max=self.pxrd_aug_width_scale_max,
                        drop_frac_max=self.pxrd_aug_drop_frac_max,
                        zero_shift_deg=self.pxrd_aug_zero_shift_deg,
                        eta_min=self.pxrd_aug_eta_min,
                        eta_max=self.pxrd_aug_eta_max,
                        background_max=self.pxrd_aug_background_max,
                        noise_std_max=self.pxrd_aug_noise_std_max,
                        rng=rng,
                    )
                    if apply_profile_augment
                    else xrdiff_feature_vector(
                        out.pop("_pxrd_x"), out.pop("_pxrd_y"), "peaks_128x4"
                    )
                )
                .reshape(128, 4)[:, [3, 2]]
                .copy()
                .ravel()
            )
            out["pxrd_d"] = torch.from_numpy(features)
            out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
            out["pxrd_peak_pad_mask"] = torch.zeros_like(
                out["pxrd_d"], dtype=torch.bool
            )
            return
        if getattr(self, "pxrd_public_api", False) and (not public_online_profile):
            from src.pxrd.representation import (
                PXRDTransformConfig,
                attach_pxrd_result,
                transform_bragg_pattern,
                transform_dense_profile,
            )

            config = PXRDTransformConfig(
                representation=self.pxrd_semantic_representation,
                peak_source=self.pxrd_peak_source,
                peak_features=self.pxrd_peak_fields,
                max_peaks=self.pxrd_max_peaks,
                wavelength=self.pxrd_wavelength,
                two_theta_min=self.pxrd_two_theta_min,
                two_theta_max=self.pxrd_two_theta_max,
                dense_n_bins=self.pxrd_dense_n_bins,
                selection=self.pxrd_peak_selection,
                low_angle_fraction=self.pxrd_low_angle_fraction,
                caglioti_u=self.pxrd_caglioti_u,
                caglioti_v=self.pxrd_caglioti_v,
                caglioti_w=self.pxrd_caglioti_w,
                pseudo_voigt_eta=self.pxrd_pseudo_voigt_eta,
                position_normalized=self.pxrd_peak_position_normalized,
                intensity_normalization=self.pxrd_peak_intensity_normalization,
                min_relative_intensity=self.pxrd_peak_min_relative_intensity,
                prominence_normalization=self.pxrd_peak_prominence_normalization,
                width_unit=self.pxrd_peak_width_unit,
                width_normalization=self.pxrd_peak_width_normalization,
                augment_sticks=apply_peak_augment,
                augmentation=self.pxrd_augmentation
                if apply_profile_augment
                else "none",
                jitter_deg=self.pxrd_aug_jitter_deg,
                intensity_scale_min=self.pxrd_aug_int_scale_min,
                intensity_scale_max=self.pxrd_aug_int_scale_max,
                width_scale_min=self.pxrd_aug_width_scale_min,
                width_scale_max=self.pxrd_aug_width_scale_max,
                zero_shift_deg=self.pxrd_aug_zero_shift_deg,
                eta_min=self.pxrd_aug_eta_min,
                eta_max=self.pxrd_aug_eta_max,
                background_max=self.pxrd_aug_background_max,
                noise_std_max=self.pxrd_aug_noise_std_max,
                drop_fraction_max=self.pxrd_aug_drop_frac_max,
                xrdiff_gamma_shape=self.pxrd_xrdiff_gamma_shape,
                xrdiff_gamma_scale=self.pxrd_xrdiff_gamma_scale,
            )
            if self.pxrd_peak_source == "bragg":
                result = transform_bragg_pattern(
                    out.pop("_pxrd_x"), out.pop("_pxrd_y"), config, rng=rng
                )
            else:
                result = transform_dense_profile(out["xrd"].numpy(), config, rng=rng)
            attach_pxrd_result(
                out, result, config, internal_representation=self.pxrd_representation
            )
            return
        if self.pxrd_representation in {"broadened_basic", "broadened_shape"}:
            position, intensity, prominence, width, d_spacing, peak_pad = out.pop(
                "_broadened_features"
            )
            out["pxrd_position"] = torch.from_numpy(position)
            out["pxrd_intensity"] = torch.from_numpy(intensity)
            out["pxrd_prominence"] = torch.from_numpy(prominence)
            out["pxrd_width"] = torch.from_numpy(width)
            out["pxrd_d"] = torch.from_numpy(d_spacing)
            out["pxrd_i"] = torch.from_numpy(intensity)
            out["pxrd_peak_pad_mask"] = torch.from_numpy(peak_pad)
            return
        if self.pxrd_representation == "peak_matrix":
            if self.pxrd_feature_source == "cached":
                features = out.pop("_xrdiff_features")
                if apply_peak_augment:
                    features = augment_peak_matrix_features(
                        features,
                        max_peaks=self.pxrd_max_peaks,
                        features=self.pxrd_peak_features,
                        position=self.pxrd_peak_position,
                        normalize_position=self.pxrd_peak_position_normalized,
                        width_normalization=self.pxrd_peak_width_normalization,
                        wavelength=self.pxrd_wavelength,
                        two_theta_min=self.pxrd_two_theta_min,
                        two_theta_max=self.pxrd_two_theta_max,
                        jitter_deg=self.pxrd_aug_jitter_deg,
                        int_scale_min=self.pxrd_aug_int_scale_min,
                        int_scale_max=self.pxrd_aug_int_scale_max,
                        width_scale_min=self.pxrd_aug_width_scale_min,
                        width_scale_max=self.pxrd_aug_width_scale_max,
                        drop_frac_max=self.pxrd_aug_drop_frac_max,
                        rng=rng,
                    )
            else:
                px, py = (out.pop("_pxrd_x"), out.pop("_pxrd_y"))
                if apply_peak_augment:
                    px, py = augment_peaks(
                        px,
                        py,
                        jitter_deg=self.pxrd_aug_jitter_deg,
                        int_scale_min=self.pxrd_aug_int_scale_min,
                        int_scale_max=self.pxrd_aug_int_scale_max,
                        drop_frac_max=self.pxrd_aug_drop_frac_max,
                        rng=rng,
                    )
                    px = px + rng.uniform(
                        -self.pxrd_aug_zero_shift_deg, self.pxrd_aug_zero_shift_deg
                    )
                width_scale = (
                    rng.uniform(
                        self.pxrd_aug_width_scale_min, self.pxrd_aug_width_scale_max
                    )
                    if apply_profile_augment
                    else 1.0
                )
                profile = pseudo_voigt_profile(
                    px,
                    py,
                    two_theta_min=self.pxrd_two_theta_min,
                    two_theta_max=self.pxrd_two_theta_max,
                    step=(self.pxrd_two_theta_max - self.pxrd_two_theta_min) / 850,
                    caglioti_u=0.1 * width_scale**2,
                    caglioti_v=0.01 * width_scale**2,
                    caglioti_w=0.1 * width_scale**2,
                    eta=rng.uniform(self.pxrd_aug_eta_min, self.pxrd_aug_eta_max)
                    if apply_profile_augment
                    else 0.1,
                )
                if apply_profile_augment:
                    x = np.linspace(-1.0, 1.0, profile.size, dtype=np.float32)
                    profile += rng.uniform(0, self.pxrd_aug_background_max) * (
                        rng.uniform(0, 1)
                        + rng.uniform(-0.5, 0.5) * x
                        + rng.uniform(0, 0.5) * x**2
                    )
                    profile += rng.normal(
                        0,
                        rng.uniform(0, self.pxrd_aug_noise_std_max),
                        size=profile.shape,
                    )
                    profile = np.clip(profile, 0, None)
                features = peak_matrix_feature_vector_from_profile(
                    profile,
                    max_peaks=self.pxrd_max_peaks,
                    features=self.pxrd_peak_features,
                    position=self.pxrd_peak_position,
                    normalize_position=self.pxrd_peak_position_normalized,
                    width_normalization=self.pxrd_peak_width_normalization,
                    selection=self.pxrd_peak_selection,
                    low_angle_fraction=self.pxrd_low_angle_fraction,
                    wavelength=self.pxrd_wavelength,
                    two_theta_min=self.pxrd_two_theta_min,
                    two_theta_max=self.pxrd_two_theta_max,
                )
            out["pxrd_d"] = torch.from_numpy(features)
            out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
            out["pxrd_peak_pad_mask"] = torch.zeros_like(
                out["pxrd_d"], dtype=torch.bool
            )
            return
        if self.pxrd_representation != "legacy_di":
            if apply_profile_augment:
                features = augmented_peak128_feature_vector(
                    out.pop("_pxrd_x"),
                    out.pop("_pxrd_y"),
                    jitter_deg=self.pxrd_aug_jitter_deg,
                    int_scale_min=self.pxrd_aug_int_scale_min,
                    int_scale_max=self.pxrd_aug_int_scale_max,
                    width_scale_min=self.pxrd_aug_width_scale_min,
                    width_scale_max=self.pxrd_aug_width_scale_max,
                    drop_frac_max=self.pxrd_aug_drop_frac_max,
                    zero_shift_deg=self.pxrd_aug_zero_shift_deg,
                    eta_min=self.pxrd_aug_eta_min,
                    eta_max=self.pxrd_aug_eta_max,
                    background_max=self.pxrd_aug_background_max,
                    noise_std_max=self.pxrd_aug_noise_std_max,
                    min_prominence=self.pxrd_peak_min_prominence,
                    peak_count=self.pxrd_max_peaks // 4,
                    rng=rng,
                )
            elif self.pxrd_profile_augment:
                features = xrdiff_feature_vector(
                    out.pop("_pxrd_x"),
                    out.pop("_pxrd_y"),
                    self.pxrd_representation,
                    min_prominence=self.pxrd_peak_min_prominence,
                    peak_count=self.pxrd_max_peaks // 4
                    if self.pxrd_representation == "peaks_128x4"
                    else 128,
                )
            else:
                features = out.pop("_xrdiff_features")
            if apply_peak_augment and (not self.pxrd_profile_augment):
                features = augment_peak128_features(
                    features,
                    jitter_deg=self.pxrd_aug_jitter_deg,
                    int_scale_min=self.pxrd_aug_int_scale_min,
                    int_scale_max=self.pxrd_aug_int_scale_max,
                    width_scale_min=self.pxrd_aug_width_scale_min,
                    width_scale_max=self.pxrd_aug_width_scale_max,
                    drop_frac_max=self.pxrd_aug_drop_frac_max,
                    rng=rng,
                )
            out["pxrd_d"] = torch.from_numpy(features)
            out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
            out["pxrd_peak_pad_mask"] = torch.zeros_like(
                out["pxrd_d"], dtype=torch.bool
            )
            return
        px = out.pop("_pxrd_x")
        py = out.pop("_pxrd_y")
        if apply_peak_augment:
            px, py = augment_peaks(
                px,
                py,
                jitter_deg=self.pxrd_aug_jitter_deg,
                int_scale_min=self.pxrd_aug_int_scale_min,
                int_scale_max=self.pxrd_aug_int_scale_max,
                drop_frac_max=self.pxrd_aug_drop_frac_max,
                rng=rng,
            )
        d_pad, i_pad, peak_pad = peaks_to_di_tensors(
            px,
            py,
            max_peaks=self.pxrd_max_peaks,
            wavelength=self.pxrd_wavelength,
            two_theta_min=self.pxrd_two_theta_min,
            two_theta_max=self.pxrd_two_theta_max,
            selection=self.pxrd_peak_selection,
            low_angle_fraction=self.pxrd_low_angle_fraction,
        )
        out["pxrd_d"] = torch.from_numpy(d_pad)
        out["pxrd_i"] = torch.from_numpy(i_pad)
        out["pxrd_peak_pad_mask"] = torch.from_numpy(peak_pad)
        out["pxrd_peak_list_d"] = out["pxrd_d"]
        out["pxrd_peak_list_intensity"] = out["pxrd_i"]
        out["pxrd_peak_list_pad_mask"] = out["pxrd_peak_pad_mask"]
        out["pxrd_peak_list_source"] = "bragg"

    def _attach_pxrd(self):
        """Attach `xrd` tensor to each item by matching `mp_id` against
        `data/mp20/pxrd/{split}.pt` produced from the PXRDgen dataset.

        Items whose `mp_id` is missing in the PXRD source are dropped.
        """
        pxrd_path = os.path.join(self.root, "pxrd", f"{self.split_name}.pt")
        if not os.path.exists(pxrd_path):
            raise FileNotFoundError(
                f"PXRD cache not found: {pxrd_path}. Build it from the PXRDgen dataset before enabling with_pxrd=True."
            )
        cache = torch.load(pxrd_path, weights_only=False)
        ids: list[str] = cache["material_id"]
        xrd: torch.Tensor = cache["xrd_array"]
        if self.pxrd_trim_last:
            xrd = xrd[:, :-1]
        if xrd.dtype != self.pxrd_dtype:
            xrd = xrd.to(self.pxrd_dtype)
        self.pxrd_meta = cache.get("meta", {})
        self.pxrd_n_bins = int(xrd.shape[1])
        two_theta_min = float(self.pxrd_meta.get("two_theta_min", 5.0))
        two_theta_max = float(self.pxrd_meta.get("two_theta_max", 80.0))
        self.pxrd_two_theta_bounds = (two_theta_min, two_theta_max)
        if (
            self.with_pxrd_peaks
            and self.pxrd_public_api
            and (self.pxrd_peak_source == "dense")
        ):
            from src.pxrd.pxrd_simulator import validate_two_theta_bounds

            validate_two_theta_bounds(
                self.pxrd_two_theta_min,
                self.pxrd_two_theta_max,
                self.pxrd_two_theta_bounds,
                name="stored MP20 PXRD",
            )
            if self.pxrd_dense_n_bins != self.pxrd_n_bins:
                raise ValueError(
                    f"stored MP20 PXRD has {self.pxrd_n_bins} bins, configured {self.pxrd_dense_n_bins}"
                )
        stored_bins = int(
            self.pxrd_meta.get("n_bins", self.pxrd_n_bins + int(self.pxrd_trim_last))
        )
        self.pxrd_two_theta_grid = torch.from_numpy(
            np.linspace(two_theta_min, two_theta_max, stored_bins, dtype=np.float32)
        )[: self.pxrd_n_bins]
        id_to_row = {mid: i for i, mid in enumerate(ids)}
        kept = []
        dropped = 0
        for item in self.items:
            row = id_to_row.get(item["mp_id"])
            if row is None:
                dropped += 1
                continue
            x = xrd[row]
            if torch.isnan(x).any():
                dropped += 1
                continue
            new_item = dict(item)
            new_item["xrd"] = x
            kept.append(new_item)
        self.items = kept
        self.pxrd_dropped = dropped

    def __len__(self):
        return len(self.items)

    def _element_counts(self) -> torch.Tensor:
        counts = torch.zeros(VZ, dtype=torch.float64)
        for item in self.items:
            values = item["A0"][~item["pad_mask"].bool()]
            counts += torch.bincount(values, minlength=VZ + 1)[1:].double()
        return counts

    def element_distribution(self) -> torch.Tensor | None:
        counts = self._element_counts()
        total = float(counts.sum())
        return counts / total if total > 0 else None

    def allowed_elements(self) -> torch.Tensor | None:
        mask = self._element_counts() > 0
        return mask if bool(mask.any()) else None

    def __getitem__(self, idx):
        item = self.items[idx]
        structure_cache = getattr(self, "structure_cache", None)
        if structure_cache is not None:
            item = structure_cache.apply(item, nmax=self.nmax)
        if not (
            self.augment_translate
            or self.with_pxrd_peaks
            or False
            or self.hierarchical_orbit_permutation
            or (self.atom_ordering == "mcflow")
        ):
            return item
        out = dict(item)
        if "_mcflow_orbits" in item:
            orbit_ids = torch.full((self.nmax,), -1, dtype=torch.long)
            for orbit_id, orbit in enumerate(item["_mcflow_orbits"]):
                orbit_ids[list(orbit)] = int(orbit_id)
            out["mcflow_orbit_ids"] = orbit_ids
        if self.hierarchical_orbit_permutation:
            permutation = mcflow_hierarchical_permutation(item)
            out["A0"] = item["A0"].index_select(0, permutation)
            out["F1"] = item["F1"].index_select(0, permutation)
            out["pad_mask"] = item["pad_mask"].index_select(0, permutation)
            out["mcflow_orbit_ids"] = out["mcflow_orbit_ids"].index_select(
                0, permutation
            )
        if self.augment_translate:
            out["F1"] = torch.from_numpy(
                translate_frac_coords(out["F1"], out["pad_mask"])
            )
        if self.with_pxrd_peaks:
            profile_seed = getattr(self, "profile_augmentation_seed", None)
            self._finalize_peaks(
                out,
                rng=np.random.default_rng(
                    stable_seed("mp20-profile", int(profile_seed), idx)
                )
                if profile_seed is not None
                else None,
            )
            if (
                not self.with_pxrd
                and self.pxrd_public_api
                and (self.pxrd_peak_source == "dense")
            ):
                out.pop("xrd", None)
        return out

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Prepare deterministic repeated evaluation inputs once per target."""
        if (
            self.augment_translate
            or 0.0 > 0.0
            or self.pxrd_peak_augment
            or self.pxrd_profile_augment
            or self.hierarchical_orbit_permutation
        ):
            return [self[index] for index in indices]
        loaded = {index: self[index] for index in dict.fromkeys(map(int, indices))}
        return [loaded[int(index)] for index in indices]
