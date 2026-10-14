"""Load fixed opXRD evaluation artifacts in the RealPXRD LMDB format.

Emits Bragglite's shared ``A0/F1/Y1/pad_mask/num_atoms/mp_id`` token schema.
The historical class/format identifiers are retained for frozen cache and
checkpoint compatibility. New training on this corpus is rejected by the
training configuration validator.

RealPXRD data layout
--------------------
The download (Google Drive folder ``1WHXuSx9LD5...``) stores structures + powder
patterns as **LMDB** files under
``data/real_pxrd_data/alex_aflow_oqmd_mp/datasets/``::

    pxrd_all_{train,valid,test}.lmdb       # full corpus (train ~6.09M); mostly
                                           #   small cells (~99% have <=25 atoms,
                                           #   mean ~5).
    pxrd_atom25_{train,valid,test}.lmdb    # >=25-atom subset (train ~132k); LARGE
                                           #   cells only (min 25, mean ~46, up to
                                           #   444). NOT a <=25 subset -- pairing it
                                           #   with nmax=25 keeps only the ~0.9% at
                                           #   exactly 25 atoms.

Each LMDB value is a ``gzip``-compressed ``pickle`` dict::

    {
      "p_atom_type":     list[str]  element symbols, e.g. ['Al','Sr','Sr','Y']
      "p_atom_pos":      (N,3) float fractional coords
      "p_lattice_matrix":(3,3) float lattice (rows = a,b,c cartesian vectors)
      "pxrd_x":          (P,) float  sparse peak positions  (2theta, deg, Cu Ka)
      "pxrd_y":          (P,) float  sparse peak intensities (max-normalized 100)
    }

Loading behavior
----------------
* **Lazy** by default: the full ``pxrd_all_train`` is ~6M structures and will not
  fit in RAM as padded tensors. We build a small **on-disk index** once (keys +
  per-record atom count + max Z + global element counts), then decode one record
  per ``__getitem__``. Set ``in_memory=True`` for small splits (e.g. the 10k
  ``test``) to preload everything like the other token datasets.
* Atom types arrive as **element symbols** -> converted to atomic numbers (Z).
* Lattice arrives as a **3x3 matrix** -> converted to (lengths, angles) -> ``Y1``.
* Optional ``with_pxrd=True`` rasterizes the sparse ``(2theta, I)`` peak list into
  a fixed-length ``xrd`` vector (Gaussian-broadened) so the same item also feeds
  PXRD-conditioned models; ``collate_mp20_tokens`` already stacks an ``xrd`` key.

LMDB + ``num_workers`` note: the environment is opened **lazily per process** (in
``__getitem__``), never in ``__init__`` — an ``lmdb.Environment`` is not safe to
share across forked DataLoader workers.
"""

from __future__ import annotations
import gzip
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.mp20_tokens import (
    NMAX,
    VZ,
    collate_mp20_tokens,
    pack_structure_tokens,
    translate_frac_coords,
)

collate_real_pxrd_tokens = collate_mp20_tokens
_SPLIT_TO_FILE = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
_SYMBOL_Z: dict[str, int] = {}


def _symbol_to_z(sym: str) -> int:
    z = _SYMBOL_Z.get(sym)
    if z is None:
        try:
            from pymatgen.core.periodic_table import Element

            z = int(Element(sym).Z)
        except Exception:
            z = 0
        _SYMBOL_Z[sym] = z
    return z


def _decode(raw: bytes) -> dict:
    return pickle.loads(gzip.decompress(raw))


def _preprocess_official_pxrd(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    augment: bool,
    jitter_deg: float = 0.1,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the official RealPXRD-Solver dataset preprocessing."""
    px = np.asarray(two_theta, dtype=np.float32)
    py = np.asarray(intensity, dtype=np.float32)
    keep = py > 5.0
    px, py = (px[keep], py[keep])
    if not augment or py.size == 0:
        return (px, py)
    py_aug = py + np.random.normal(0.0, 0.05 * py.max(), size=py.shape)
    px_aug = px + np.random.uniform(-jitter_deg, jitter_deg, size=px.shape)
    py_aug *= np.random.uniform(scale_min, scale_max)
    py_aug = np.where(py_aug < 0.0, py, py_aug)
    py_aug = py_aug / py_aug.max() * 100.0
    return (px_aug.astype(np.float32), py_aug.astype(np.float32))


def _preprocess_real_pxrd_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    mode: str,
    augment: bool,
    jitter_deg: float = 0.1,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the current official or pre-August-2026 checkpoint protocol."""
    mode = str(mode).strip().lower()
    if mode == "official":
        return _preprocess_official_pxrd(
            two_theta,
            intensity,
            augment=augment,
            jitter_deg=jitter_deg,
            scale_min=scale_min,
            scale_max=scale_max,
        )
    if mode != "historical_unfiltered":
        raise ValueError(
            "pxrd_peak_preprocessing must be official|historical_unfiltered"
        )
    if augment:
        raise ValueError("historical_unfiltered is evaluation-only")
    return (
        np.asarray(two_theta, dtype=np.float32),
        np.asarray(intensity, dtype=np.float32),
    )


def _lattice_params(matrix) -> tuple[np.ndarray, np.ndarray]:
    """3x3 lattice matrix (rows a,b,c) -> (lengths(3,), angles_deg(3,))."""
    m = np.asarray(matrix, dtype=np.float64)
    a, b, c = np.linalg.norm(m, axis=1)
    lengths = np.array([a, b, c], dtype=np.float32)

    def _ang(u, v):
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom == 0:
            return 90.0
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))

    angles = np.array(
        [_ang(m[1], m[2]), _ang(m[0], m[2]), _ang(m[0], m[1])], dtype=np.float32
    )
    return (lengths, angles)


def _rasterize_pxrd(
    px: np.ndarray, py: np.ndarray, n_bins: int, tmin: float, tmax: float, sigma: float
) -> np.ndarray:
    """Sparse (2theta, intensity) peaks -> fixed-length Gaussian-broadened curve.

    Returns a ``(n_bins,)`` float32 vector over [tmin, tmax), max-normalized.
    """
    from src.pxrd.representation import two_theta_grid

    grid = two_theta_grid(tmin, tmax, n_bins).astype(np.float32)
    out = np.zeros(n_bins, dtype=np.float32)
    if px.size:
        inv2s2 = 1.0 / (2.0 * sigma * sigma)
        for x0, i0 in zip(px, py):
            if x0 < tmin or x0 >= tmax:
                continue
            out += float(i0) * np.exp(-((grid - float(x0)) ** 2) * inv2s2)
        m = out.max()
        if m > 0:
            out /= m
    return out


class RealPXRDTokens(Dataset):
    """PyTorch dataset over a RealPXRD LMDB split, emitting Bragglite tokens.

    Each item is a dict::

        mp_id:     str               LMDB key (string)
        A0:        (nmax,)   long     atomic numbers, 0-padded
        F1:        (nmax, 3) float    fractional coords in [0,1), 0-padded
        Y1:        (6,)      float    log-lengths + cos-angles lattice encoding
        pad_mask:  (nmax,)   bool     True where padded
        num_atoms: int
        xrd:       (pxrd_n_bins,) float   only when ``with_pxrd=True``

    Parameters
    ----------
    root : str
        Directory holding the ``*.lmdb`` files, e.g.
        ``data/real_pxrd_data/alex_aflow_oqmd_mp/datasets``.
    split : str
        ``train`` | ``val``/``valid`` | ``test``.
    nmax : int
        Max atoms; structures with more are dropped (must match the model).
    augment_translate : bool
        Random fractional-coordinate translation per ``__getitem__`` (train only).
    variant : str
        ``"all"`` (full ~6.09M corpus, ~99% <=25 atoms) or ``"atom25"`` (the
        ~132k subset of cells with **>=25** atoms, mean ~46 -- not a <=25 subset).
        Selects the ``pxrd_{variant}_{split}.lmdb`` file. Ignored if
        ``lmdb_prefix`` is given.
    lmdb_prefix : str | None
        Override the file prefix (default ``f"pxrd_{variant}"``).
    in_memory : bool
        Preload every kept record into RAM (fine for ``test``/``valid``; avoid for
        the multi-million ``train`` split).
    prop_list : list[str] | None
        Extra scalar keys to carry through from the source dict, if present.
    with_pxrd : bool
        Attach a rasterized ``xrd`` curve built from ``pxrd_x``/``pxrd_y``.
    with_pxrd_peaks : bool
        Attach the discrete d-I peak list (RealPXRD-Solver style) as
        ``pxrd_d``/``pxrd_i``/``pxrd_peak_pad_mask`` fixed-length tensors of
        size ``pxrd_max_peaks`` (strongest peaks kept when over-long). Following
        the official loader, peaks with intensity <= 5 are removed. When
        ``pxrd_peak_augment=True`` (train only), Gaussian intensity noise,
        2theta jitter, and global intensity scaling are re-sampled on every
        ``__getitem__``.
    force_reindex : bool
        Rebuild the cached index even if present.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        nmax: int = NMAX,
        augment_translate: bool = False,
        *,
        variant: str = "all",
        lmdb_prefix: str | None = None,
        in_memory: bool = False,
        prop_list: list[str] | None = None,
        with_pxrd: bool = False,
        pxrd_n_bins: int = 8500,
        pxrd_two_theta_min: float = 5.0,
        pxrd_two_theta_max: float = 90.0,
        pxrd_sigma: float = 0.05,
        with_pxrd_peaks: bool = False,
        pxrd_max_peaks: int = 128,
        pxrd_wavelength: float = 1.54184,
        pxrd_peak_augment: bool = False,
        pxrd_peak_preprocessing: str = "official",
        pxrd_aug_jitter_deg: float = 0.1,
        pxrd_aug_int_scale_min: float = 0.8,
        pxrd_aug_int_scale_max: float = 1.2,
        pxrd_aug_drop_frac_max: float = 0.1,
        pxrd_peak_selection: str = "strongest",
        pxrd_low_angle_fraction: float = 0.25,
        pxrd_representation: str = "legacy_di",
        pxrd_peak_features: str = "width_prominence_intensity_position",
        pxrd_peak_position: str = "d",
        pxrd_peak_position_normalized: bool = False,
        pxrd_peak_width_normalization: str = "none",
        pxrd_profile_step: float = 0.1,
        pxrd_caglioti_u: float = 0.1,
        pxrd_caglioti_v: float = 0.01,
        pxrd_caglioti_w: float = 0.1,
        pxrd_pseudo_voigt_eta: float = 0.1,
        force_reindex: bool = False,
    ):
        self.root = root
        self.split = split
        self.nmax = int(nmax)
        self.augment_translate = augment_translate
        self.variant = variant
        self.in_memory = in_memory
        self.prop_list = prop_list or []
        self.with_pxrd = with_pxrd
        self.pxrd_n_bins = int(pxrd_n_bins)
        self.pxrd_two_theta_min = float(pxrd_two_theta_min)
        self.pxrd_two_theta_max = float(pxrd_two_theta_max)
        self.pxrd_two_theta_bounds = (self.pxrd_two_theta_min, self.pxrd_two_theta_max)
        self.pxrd_sigma = float(pxrd_sigma)
        from src.pxrd.representation import two_theta_grid

        self.pxrd_two_theta_grid = torch.from_numpy(
            two_theta_grid(
                self.pxrd_two_theta_min, self.pxrd_two_theta_max, self.pxrd_n_bins
            ).astype(np.float32)
        )
        self.with_pxrd_peaks = bool(with_pxrd_peaks)
        self.pxrd_max_peaks = int(pxrd_max_peaks)
        self.pxrd_wavelength = float(pxrd_wavelength)
        self.pxrd_peak_augment = bool(pxrd_peak_augment)
        self.pxrd_peak_preprocessing = str(pxrd_peak_preprocessing).strip().lower()
        if self.pxrd_peak_preprocessing not in {"official", "historical_unfiltered"}:
            raise ValueError(
                "pxrd_peak_preprocessing must be official|historical_unfiltered"
            )
        if (
            self.pxrd_peak_preprocessing == "historical_unfiltered"
            and self.pxrd_peak_augment
        ):
            raise ValueError("historical_unfiltered is evaluation-only")
        self.pxrd_aug_jitter_deg = float(pxrd_aug_jitter_deg)
        self.pxrd_aug_int_scale_min = float(pxrd_aug_int_scale_min)
        self.pxrd_aug_int_scale_max = float(pxrd_aug_int_scale_max)
        self.pxrd_peak_selection = str(pxrd_peak_selection)
        self.pxrd_low_angle_fraction = float(pxrd_low_angle_fraction)
        self.pxrd_representation = str(pxrd_representation)
        self.pxrd_peak_features = str(pxrd_peak_features)
        self.pxrd_peak_position = str(pxrd_peak_position)
        self.pxrd_peak_position_normalized = bool(pxrd_peak_position_normalized)
        self.pxrd_peak_width_normalization = str(pxrd_peak_width_normalization)
        self.pxrd_profile_step = float(pxrd_profile_step)
        self.pxrd_caglioti_u = float(pxrd_caglioti_u)
        self.pxrd_caglioti_v = float(pxrd_caglioti_v)
        self.pxrd_caglioti_w = float(pxrd_caglioti_w)
        self.pxrd_pseudo_voigt_eta = float(pxrd_pseudo_voigt_eta)
        supported_representations = {
            "legacy_di",
            "full_850",
            "peaks_20x4",
            "peaks_128x4",
            "peaks_20x4_norm_prom",
            "peaks_128x4_norm_prom",
            "broadened_basic",
            "broadened_shape",
            "peak_matrix",
        }
        if self.pxrd_representation not in supported_representations:
            raise ValueError(f"unsupported PXRD representation: {pxrd_representation}")
        if (
            self.pxrd_representation == "legacy_di"
            and self.pxrd_peak_selection == "prominence"
        ):
            raise ValueError("legacy_di does not support prominence peak selection")
        if self.nmax <= 0:
            raise ValueError(f"nmax must be > 0, got {self.nmax}.")
        file_split = _SPLIT_TO_FILE.get(split)
        if file_split is None:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {sorted(_SPLIT_TO_FILE)}."
            )
        prefix = lmdb_prefix or f"pxrd_{variant}"
        self.prefix = prefix
        self.lmdb_path = os.path.join(root, f"{prefix}_{file_split}.lmdb")
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(
                f"RealPXRD LMDB not found: {self.lmdb_path}. Expected '{prefix}_{file_split}.lmdb' under {root}."
            )
        self._env = None
        self._env_pid = None
        idx_dir = os.path.join(root, "processed")
        self.index_path = os.path.join(
            idx_dir, f"realpxrd_index_{prefix}_{file_split}.pt"
        )
        if force_reindex:
            raise ValueError(
                "Dataset index generation is not included in the public loader"
            )
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"Missing frozen RealPXRD index: {self.index_path}")
        idx = torch.load(self.index_path, weights_only=False)
        if "elem_counts_by_n" not in idx:
            raise ValueError("Frozen RealPXRD index lacks elem_counts_by_n")
        all_keys = idx["keys"]
        all_ns = np.asarray(idx["ns"], dtype=np.int64)
        all_ok = np.asarray(idx["ok"], dtype=bool)
        self._elem_counts_by_n = np.asarray(idx["elem_counts_by_n"], dtype=np.int64)
        keep = all_ok & (all_ns >= 1) & (all_ns <= self.nmax)
        self.keys = [all_keys[i] for i in np.nonzero(keep)[0]]
        self.ns = all_ns[keep]
        self.n_total_raw = len(all_keys)
        self.n_kept = len(self.keys)
        self._mem: list[dict] | None = None
        if self.in_memory:
            self._mem = [self._load_item(k) for k in self.keys]

    def _get_env(self):
        pid = os.getpid()
        if self._env is None or self._env_pid != pid:
            import lmdb

            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                subdir=False,
                readahead=False,
                meminit=False,
            )
            self._env_pid = pid
        return self._env

    def _load_item(self, key: bytes) -> dict:
        with self._get_env().begin() as txn:
            rec = _decode(txn.get(key))
        return self._record_to_item(key, rec)

    def _record_to_item(self, key: bytes, rec: dict) -> dict:
        syms = list(rec["p_atom_type"])
        atom_types = np.array([_symbol_to_z(s) for s in syms], dtype=np.int64)
        frac_coords = np.asarray(rec["p_atom_pos"], dtype=np.float32)
        n = len(atom_types)
        lengths, angles = _lattice_params(rec["p_lattice_matrix"])
        item = pack_structure_tokens(
            atom_types,
            frac_coords,
            lengths,
            angles,
            nmax=self.nmax,
            num_atoms=n,
            mp_id=key.decode() if isinstance(key, bytes) else str(key),
        )
        if self.with_pxrd:
            px = np.asarray(rec.get("pxrd_x", []), dtype=np.float32)
            py = np.asarray(rec.get("pxrd_y", []), dtype=np.float32)
            item["xrd"] = torch.from_numpy(
                _rasterize_pxrd(
                    px,
                    py,
                    self.pxrd_n_bins,
                    self.pxrd_two_theta_min,
                    self.pxrd_two_theta_max,
                    self.pxrd_sigma,
                )
            )
        if self.with_pxrd_peaks:
            item["_pxrd_x"] = np.asarray(rec.get("pxrd_x", []), dtype=np.float32)
            item["_pxrd_y"] = np.asarray(rec.get("pxrd_y", []), dtype=np.float32)
        for k in self.prop_list:
            if k in rec:
                try:
                    item[k] = float(rec[k])
                except (TypeError, ValueError):
                    pass
        return item

    def _kept_elem_counts(self) -> np.ndarray:
        """Element counts (Z=1..VZ) over exactly the kept records (n<=nmax, valid),
        summed from the per-atom-count buckets in the cached index."""
        filtered = getattr(self, "_filtered_elem_counts", None)
        if filtered is not None:
            return filtered
        rows = self._elem_counts_by_n[1 : self.nmax + 1]
        return rows.sum(axis=0)[1 : VZ + 1]

    def element_distribution(self) -> torch.Tensor | None:
        """Normalized element frequency over Z=1..VZ for the kept set (from the
        cached index). Matches ``compute_dataset_element_distribution(self)`` but
        O(1) — use it to skip the slow full pass on the multi-million train split.
        """
        counts = torch.from_numpy(self._kept_elem_counts().astype(np.float64))
        total = float(counts.sum().item())
        return None if total <= 0 else counts / total

    def allowed_elements(self) -> torch.Tensor | None:
        mask = torch.from_numpy(self._kept_elem_counts() > 0)
        return mask if bool(mask.any()) else None

    def count_distribution(self) -> torch.Tensor | None:
        """Empirical atom-count distribution over n=1..nmax for the kept set,
        computed O(1) from the cached per-record counts (``self.ns``). Matches
        ``src.utils.dataset.build_count_distribution`` (length nmax,
        index 0 -> n=1) but skips the full-pass decode of every record."""
        counts = np.bincount(self.ns, minlength=self.nmax + 1).astype(np.float64)
        counts = counts[1 : self.nmax + 1]
        total = float(counts.sum())
        return None if total <= 0 else torch.from_numpy(counts / total)

    def __len__(self) -> int:
        return self.n_kept

    def __getitem__(self, idx: int) -> dict:
        if self._mem is not None:
            item = self._mem[idx]
        else:
            item = self._load_item(self.keys[idx])
        structure_cache = getattr(self, "structure_cache", None)
        if structure_cache is not None:
            item = structure_cache.apply(item, nmax=self.nmax)
        if not (self.augment_translate or self.with_pxrd_peaks):
            return item
        out = dict(item)
        if self.augment_translate:
            out["F1"] = torch.from_numpy(
                translate_frac_coords(out["F1"], out["pad_mask"])
            )
        if self.with_pxrd_peaks:
            self._finalize_peaks(out)
        return out

    def _finalize_peaks(self, out: dict) -> None:
        """Replace raw ``_pxrd_x/_pxrd_y`` with packed (augmented) d-I tensors."""
        from src.pxrd.peaks import (
            peak_matrix_feature_vector,
            peaks_to_di_tensors,
            peaks_to_pseudo_voigt_tensors,
            xrdiff_feature_vector,
        )

        px = out.pop("_pxrd_x")
        py = out.pop("_pxrd_y")
        px, py = _preprocess_real_pxrd_peaks(
            px,
            py,
            mode=getattr(self, "pxrd_peak_preprocessing", "official"),
            augment=self.pxrd_peak_augment,
            jitter_deg=self.pxrd_aug_jitter_deg,
            scale_min=self.pxrd_aug_int_scale_min,
            scale_max=self.pxrd_aug_int_scale_max,
        )
        if self.pxrd_representation == "legacy_di":
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
            return
        if self.pxrd_representation in {"broadened_basic", "broadened_shape"}:
            position, intensity, prominence, width, d_spacing, peak_pad = (
                peaks_to_pseudo_voigt_tensors(
                    px,
                    py,
                    max_peaks=self.pxrd_max_peaks,
                    wavelength=self.pxrd_wavelength,
                    two_theta_min=self.pxrd_two_theta_min,
                    two_theta_max=self.pxrd_two_theta_max,
                    step=self.pxrd_profile_step,
                    caglioti_u=self.pxrd_caglioti_u,
                    caglioti_v=self.pxrd_caglioti_v,
                    caglioti_w=self.pxrd_caglioti_w,
                    eta=self.pxrd_pseudo_voigt_eta,
                )
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
            features = peak_matrix_feature_vector(
                px,
                py,
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
        else:
            features = xrdiff_feature_vector(
                px,
                py,
                self.pxrd_representation,
                peak_count=self.pxrd_max_peaks // 4
                if self.pxrd_representation == "peaks_128x4"
                else 128,
            )
        out["pxrd_d"] = torch.from_numpy(features)
        out["pxrd_i"] = torch.zeros_like(out["pxrd_d"])
        out["pxrd_peak_pad_mask"] = torch.zeros_like(out["pxrd_d"], dtype=torch.bool)
