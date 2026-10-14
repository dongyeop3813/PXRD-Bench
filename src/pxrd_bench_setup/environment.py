"""Lightweight installation check, usable before importing scientific packages."""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


def check(requirements: Path) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The frozen reconstruction environment requires Python 3.12")
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==")
        actual = version(package).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{package}: need {expected}, found {actual}")
    # A newer split-out distribution overwrites the pinned pymatgen module.
    try:
        version("pymatgen-core")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError("pymatgen-core conflicts with the frozen pymatgen version; use a clean environment")
    for module in ("numpy", "scipy", "pandas", "pyarrow", "pymatgen.core", "spglib", "torch", "lmdb", "joblib"):
        importlib.import_module(module)


def oqmd_buffer_pool_size(root: Path = Path("/")) -> int:
    """Use one eighth of free memory, capped at 8 GiB, for our private DB only."""
    mib = 1024 * 1024
    limits = []
    try:
        for line in (root / "proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError):
        pass
    for directory, maximum, current in (
        ("sys/fs/cgroup", "memory.max", "memory.current"),
        ("sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes"),
    ):
        try:
            path = root / directory
            capacity = int((path / maximum).read_text())
            used = int((path / current).read_text())
            limits.append(max(0, capacity - used))
        except (OSError, ValueError):
            pass
    if not limits:
        return 1024 * mib
    return max(8 * mib, min(8192 * mib, min(limits) // 8) // mib * mib)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("data/requirements-reconstruction.txt"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--oqmd-buffer-pool", action="store_true", help="Print the safe local MariaDB cache size in bytes")
    args = parser.parse_args()
    if args.oqmd_buffer_pool:
        print(oqmd_buffer_pool_size())
        return
    try:
        check(args.requirements)
    except Exception as exc:
        if not args.quiet:
            print(f"Reconstruction environment is not ready: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
