"""Reproduce canonical caches using bounded, checksum-pinned libm profiles.

The scientific package versions alone do not fix last-bit trigonometry. These
profiles recompute structures; a candidate is usable only when its frozen
reference hash matches. They never patch NumPy, Python math, or system libraries.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import math
import os
from pathlib import Path
import tarfile
import types

import numpy as np
from pymatgen.core import Lattice, Structure

from src.data.mp20_tokens import decode_Y1, tokens_to_structure
from src.data.structure_cache import encode_structure

PROFILES = {
    "2.31": {
        "archive_sha256": "1b2281aac4935dfea1f89dfc19e445fdbb45303202af60679ccb8bf035f081a0",
        "member": "./lib/x86_64-linux-gnu/libm-2.31.so",
        "library_sha256": "a0e70f48f02a399e8dd37f861a99e226c285a9daf50a25d19fff7ccf18b97f9c",
    },
    "2.35": {
        "archive_sha256": "79e35256227e16a607c154cdeb8d76ff12d20e31de286ea7fd9ad3b96fe0452d",
        "member": "./lib/x86_64-linux-gnu/libm.so.6",
        "library_sha256": "3dd5511ae94785c9f921429b0f2b2f7aabb461b6f0e6de6dfdbef15f24bdfee6",
    },
}
_CLASSES = {}
_PARSERS = {}
_AFLOW_FACTORIES = {}
_DENSE_SIMULATORS = {}


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def unpack(profile: str, archive: Path, destination: Path) -> None:
    """Extract only the known library and its license, without installing libc."""
    contract = PROFILES[profile]
    content = archive.read_bytes()
    if _digest(content) != contract["archive_sha256"]:
        raise ValueError("Canonical math archive checksum differs")
    if not content.startswith(b"!<arch>\n"):
        raise ValueError("Expected an official Debian package archive")
    offset, payload, name = 8, None, None
    while offset < len(content):
        header = content[offset : offset + 60]
        if len(header) != 60 or header[-2:] != b"`\n":
            raise ValueError("Invalid Debian archive member")
        size = int(header[48:58])
        name = header[:16].decode("ascii").strip().rstrip("/")
        start = offset + 60
        if name.startswith("data.tar."):
            payload = content[start : start + size]
            break
        offset = start + size + size % 2
    if payload is None:
        raise ValueError("Debian archive has no data tar member")
    if name.endswith(".zst"):
        import zstandard
        payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=64 * 1024 * 1024)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive_tar:
        member = archive_tar.getmember(contract["member"])
        if not member.isfile():
            raise ValueError("Expected a regular canonical math library")
        library = archive_tar.extractfile(member).read()
        if _digest(library) != contract["library_sha256"]:
            raise ValueError("Canonical math library checksum differs")
        copyright_member = archive_tar.getmember("./usr/share/doc/libc6/copyright")
        copyright_content = archive_tar.extractfile(copyright_member).read()
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / "libm.so.6.part"
    partial.write_bytes(library)
    (destination / "COPYRIGHT").write_bytes(copyright_content)
    partial.replace(destination / "libm.so.6")


class _MathProxy:
    """Override scalar float64 trigonometry only in the frozen lattice class."""

    def __init__(self, module, library):
        self.module, self.library = module, library
        self.functions = {}

    def __getattr__(self, name):
        symbol = {"arccos": "acos", "arcsin": "asin"}.get(name, name)
        if symbol not in ("sin", "cos", "acos", "asin", "sqrt", "tan", "log", "exp"):
            return getattr(self.module, name)
        if name in self.functions:
            return self.functions[name]
        function = getattr(self.library, symbol)
        function.argtypes = [ctypes.c_double]
        function.restype = ctypes.c_double
        if self.module is math:
            self.functions[name] = function
            return function
        original = getattr(self.module, name)

        def call(value, *args, **kwargs):
            array = np.asarray(value)
            if array.dtype != np.float64:
                return original(value, *args, **kwargs)
            if args or kwargs:
                raise ValueError("Unexpected ufunc options in the frozen lattice recipe")
            result = np.fromiter((function(float(x)) for x in array.flat),
                                 dtype=np.float64, count=array.size).reshape(array.shape)
            return result[()] if result.ndim == 0 else result

        self.functions[name] = call
        return call


def _lattice_class(profile: str):
    if profile in _CLASSES:
        return _CLASSES[profile]
    root = os.environ.get("PXRD_BENCH_MATH_ROOT")
    if not root:
        raise RuntimeError("Use data/setup_pxrd_bench.sh to prepare frozen canonical math")
    path = Path(root) / profile / "libm.so.6"
    if _digest(path.read_bytes()) != PROFILES[profile]["library_sha256"]:
        raise ValueError("Frozen canonical math library checksum differs")
    library = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_LOCAL)
    proxies = {"np": _MathProxy(np, library), "math": _MathProxy(math, library)}

    def bind(function):
        # Retain pinned pymatgen code/defaults, with private math bindings. This
        # avoids duplicating its algorithms or modifying global module objects.
        bound = types.FunctionType(function.__code__, {**function.__globals__, **proxies},
                                   function.__name__, function.__defaults__, function.__closure__)
        bound.__kwdefaults__ = function.__kwdefaults__
        return bound

    class FrozenLattice(Lattice):
        pass

    FrozenLattice.from_parameters = classmethod(bind(Lattice.from_parameters.__func__))
    FrozenLattice.angles = property(bind(Lattice.angles.fget))
    FrozenLattice.get_niggli_reduced_lattice = bind(Lattice.get_niggli_reduced_lattice)
    FrozenLattice.find_all_mappings = bind(Lattice.find_all_mappings)
    _CLASSES[profile] = FrozenLattice
    return FrozenLattice


def canonical_candidates(item):
    structure = tokens_to_structure(item)
    lengths, angles = decode_Y1(item["Y1"])
    for profile in ("2.35", "2.31"):
        lattice = _lattice_class(profile).from_parameters(*(lengths.tolist() + angles.tolist()))
        candidate = Structure(lattice, structure.species, structure.frac_coords,
                              coords_are_cartesian=False)
        candidate = candidate.get_primitive_structure(tolerance=0.01)
        yield encode_structure(candidate.get_reduced_structure(reduction_algo="niggli"))


def cif_candidates(content: str):
    """Parse the same source CIF with the two frozen lattice-math bindings."""
    from pymatgen.io.cif import CifParser

    for profile in ("2.35", "2.31"):
        if profile not in _PARSERS:
            class FrozenParser(CifParser):
                pass
            function = CifParser.get_lattice_no_exception
            bound = types.FunctionType(function.__code__,
                {**function.__globals__, "Lattice": _lattice_class(profile)},
                function.__name__, function.__defaults__, function.__closure__)
            bound.__kwdefaults__ = function.__kwdefaults__
            FrozenParser.get_lattice_no_exception = staticmethod(bound)
            _PARSERS[profile] = FrozenParser
        # Match Structure.from_str(..., fmt='cif'): no new primitive/reduction
        # settings or occupancy/symmetry tolerances are introduced here.
        yield _PARSERS[profile].from_str(content).parse_structures(primitive=False)[0]


def aflow_candidates(record):
    """Recompute the same native AFLOW geometry with the frozen math profiles."""
    from src.pxrd_bench_setup.serialization import _structure_from_aflow_record

    for profile in ("2.35", "2.31"):
        if profile not in _AFLOW_FACTORIES:
            function = _structure_from_aflow_record
            bound = types.FunctionType(function.__code__,
                {**function.__globals__, "Lattice": _lattice_class(profile)},
                function.__name__, function.__defaults__, function.__closure__)
            bound.__kwdefaults__ = function.__kwdefaults__
            _AFLOW_FACTORIES[profile] = bound
        yield _AFLOW_FACTORIES[profile](record)


def dense_pxrd_candidates(structure):
    """Recompute the pinned XRD/broadening algorithms with original libm bindings."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from src.pxrd_bench_setup.serialization import simulate_pxrd
    from src.data.pxrd_bench_schema import PXRD

    for profile in ("2.35", "2.31"):
        if profile not in _DENSE_SIMULATORS:
            root = os.environ.get("PXRD_BENCH_MATH_ROOT")
            if not root:
                raise RuntimeError("Use data/setup_pxrd_bench.sh to prepare frozen canonical math")
            path = Path(root) / profile / "libm.so.6"
            if _digest(path.read_bytes()) != PROFILES[profile]["library_sha256"]:
                raise ValueError("Frozen canonical math library checksum differs")
            library = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_LOCAL)
            proxies = {"np": _MathProxy(np, library), "math": _MathProxy(math, library)}

            def bind(function, extra=None):
                bound = types.FunctionType(function.__code__,
                    {**function.__globals__, **proxies, **(extra or {})},
                    function.__name__, function.__defaults__, function.__closure__)
                bound.__kwdefaults__ = function.__kwdefaults__
                return bound

            class Calculator(XRDCalculator):
                pass

            Calculator.get_pattern = bind(XRDCalculator.get_pattern)
            _DENSE_SIMULATORS[profile] = bind(simulate_pxrd,
                {"_XRD_CALCULATOR": Calculator(wavelength=PXRD["wavelength"])})
        yield _DENSE_SIMULATORS[profile](structure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    unpack(args.profile, args.archive, args.destination)


if __name__ == "__main__":
    main()
