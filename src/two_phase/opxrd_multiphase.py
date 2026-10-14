"""Read and normalize fixed opXRD profiles; no sample selection or dataset construction."""
import hashlib
from pathlib import Path
import numpy as np
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_sha256(two_theta: np.ndarray, intensity: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(two_theta.astype("<f8", copy=False).tobytes())
    digest.update(intensity.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def prepare_profile(
    two_theta: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return an increasing grid, averaging repeated source-axis positions."""
    two_theta = np.asarray(two_theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if np.all(np.diff(two_theta) > 0):
        return two_theta, intensity, "identity"
    order = np.argsort(two_theta, kind="stable")
    sorted_theta = two_theta[order]
    sorted_intensity = intensity[order]
    unique_theta, first, counts = np.unique(
        sorted_theta, return_index=True, return_counts=True
    )
    mean_intensity = np.add.reduceat(sorted_intensity, first) / counts
    return unique_theta, mean_intensity, "stable_sort_and_mean_duplicate_two_theta"


