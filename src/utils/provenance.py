"""Small provenance helpers shared by training and evaluation entrypoints."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the checked-out commit, or ``None`` outside a Git checkout."""
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if cwd is None else str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value or None
