"""Frozen PXRD-Bench storage and profile schema."""
SPLITS = (
    "train",
    "iid_val",
    "iid_test",
    "polymorph_ood_val",
    "polymorph_ood_test",
    "prototype_ood_val",
    "prototype_ood_test",
    "composition_ood_val",
    "composition_ood_test",
)


MAX_SITES = 208


DEFAULT_MAX_SITES = 52


SHARD_SCHEMA_VERSION = 2


PXRD = {
    "wavelength": "CuKa",
    "two_theta_min": 5.0,
    "two_theta_max": 90.0,
    "bin_width": 0.1,
    "bins": 850,
    "U": 0.1,
    "V": 0.01,
    "W": 0.1,
    "eta": 0.1,
}


