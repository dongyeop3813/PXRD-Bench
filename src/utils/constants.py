DATASET_NMAX_DEFAULTS = {
    "mp20": 20,
    "mp20_polymorph_split": 20,
    "rruff_148_author_exact": 20,
    "pxrd_bench": 52,
}

BEST_CKPT_SELECTOR_CHOICES = (
    "auto",
    "legacy",
    "csp_val_ratio",
    "csp_precise_val_ratio",
    "csp_sample_val_ratio",
    "csp_match_rate@1",
)

AA_TARGET_ORDER = ("coords", "lattice")
DIAGNOSTIC_SECTION_KEYS = (
    "struct_min_dist_frac_lt_cutoff",
    "struct_volume_frac_lt_cutoff",
    "lattice_length_frac_lt_1",
    "comp_missing_ox_frac",
    "invalid_atom_frac",
    "num_elems_frac_single",
)

# Backward-compatible alias for older imports during the refactor.
_DIAGNOSTIC_SECTION_KEYS = DIAGNOSTIC_SECTION_KEYS
