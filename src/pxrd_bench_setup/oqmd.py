"""Export OQMD v1.5 structures from the frozen local MariaDB snapshot."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import multiprocessing
import os
import subprocess
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from pymatgen.core import Structure

from src.pxrd_bench_setup.serialization import _json_dump, _sha256


OQMD_QUERY = """
SELECT fe.id, fe.entry_id, fe.stability, s.id,
       s.x1, s.x2, s.x3, s.y1, s.y2, s.y3, s.z1, s.z2, s.z3,
       a.id, a.site_id, a.element_id, a.x, a.y, a.z, a.occupancy
FROM formation_energies AS fe
JOIN entries AS e ON e.id = fe.entry_id
JOIN calculations AS c ON c.id = fe.calculation_id
JOIN structures AS s ON s.id = c.output_id
JOIN atoms AS a ON a.structure_id = s.id
WHERE fe.fit_id = 'standard'
  AND e.id = e.duplicate_of_id
ORDER BY fe.id
""".strip()


@contextmanager
def _gzip_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                yield handle


def _none(value: str) -> str | None:
    return None if value in {"NULL", r"\N"} else value


def _convert(rows: list[list[str]]) -> tuple[dict[str, Any] | None, str | None]:
    warnings.filterwarnings("ignore")
    try:
        first = rows[0]
        cell = [
            [float(value) for value in first[start : start + 3]]
            for start in (4, 7, 10)
        ]
        sites: dict[str, list[tuple[str, float, list[float]]]] = defaultdict(list)
        for row in sorted(
            rows,
            key=lambda value: (int(_none(value[14]) or value[13]), int(value[13])),
        ):
            site_id = _none(row[14]) or f"atom:{row[13]}"
            sites[site_id].append(
                (
                    row[15],
                    float(row[19]),
                    [float(row[16]), float(row[17]), float(row[18])],
                )
            )
        species: list[str | dict[str, float]] = []
        coordinates: list[list[float]] = []
        for atoms in sites.values():
            composition: dict[str, float] = defaultdict(float)
            for element, occupancy, _ in atoms:
                composition[element] += occupancy
            species.append(
                next(iter(composition))
                if len(composition) == 1
                and abs(next(iter(composition.values())) - 1) < 1e-8
                else dict(composition)
            )
            coordinates.append(atoms[0][2])
        return (
            {
                "material_id": f"oqmd-fe-{first[0]}",
                "oqmd_formation_energy_id": int(first[0]),
                "oqmd_entry_id": int(first[1]),
                "oqmd_structure_id": int(first[3]),
                "source": "oqmd-v1.5",
                "oqmd_stability": float(first[2]) if _none(first[2]) else None,
                "cif": Structure(
                    cell, species, coordinates, coords_are_cartesian=False
                ).to(fmt="cif"),
            },
            None,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def export_oqmd(
    *,
    mysql_bin: Path,
    mysql_user: str,
    socket: Path,
    database: str,
    dump_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if not database.replace("_", "").isalnum() or workers < 1:
        raise ValueError("invalid database name or worker count")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "oqmd_v1_5_auxiliary.csv.gz"
    excluded_path = output_dir / "oqmd_v1_5_export_excluded.csv.gz"
    table_partial = table_path.with_suffix(table_path.suffix + ".part")
    excluded_partial = excluded_path.with_suffix(excluded_path.suffix + ".part")
    process = subprocess.Popen(
        [
            str(mysql_bin),
            "--no-defaults",
            "--protocol=SOCKET",
            f"--user={mysql_user}",
            f"--socket={socket}",
            "--batch",
            "--raw",
            "--quick",
            "--skip-column-names",
            database,
            "-e",
            OQMD_QUERY,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None and process.stderr is not None
    reader = csv.reader(process.stdout, delimiter="\t")
    current_key: str | None = None
    current_rows: list[list[str]] = []
    pending: list[list[list[str]]] = []
    retained = excluded = 0

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor,
        _gzip_writer(table_partial) as table_handle,
        _gzip_writer(excluded_partial) as excluded_handle,
    ):
        fields = [
            "material_id",
            "oqmd_formation_energy_id",
            "oqmd_entry_id",
            "oqmd_structure_id",
            "source",
            "oqmd_stability",
            "cif",
        ]
        table_writer = csv.DictWriter(table_handle, fieldnames=fields)
        excluded_writer = csv.DictWriter(
            excluded_handle, fieldnames=["formationenergy_id", "reason"]
        )
        table_writer.writeheader()
        excluded_writer.writeheader()

        def flush() -> None:
            nonlocal retained, excluded
            for rows, (record, error) in zip(
                pending,
                executor.map(_convert, pending, chunksize=8),
                strict=True,
            ):
                if record is not None:
                    table_writer.writerow(record)
                    retained += 1
                else:
                    excluded_writer.writerow(
                        {"formationenergy_id": rows[0][0], "reason": error}
                    )
                    excluded += 1
                if (retained + excluded) % 100_000 == 0:
                    print(f"exported {retained:,}; excluded {excluded:,}", flush=True)
            pending.clear()

        for row in reader:
            if len(row) != 20:
                process.kill()
                raise RuntimeError(f"unexpected OQMD SQL row width: {len(row)}")
            if current_key is not None and row[0] != current_key:
                pending.append(current_rows)
                current_rows = []
                if len(pending) >= workers * 32:
                    flush()
            current_key = row[0]
            current_rows.append(row)
        if current_rows:
            pending.append(current_rows)
        flush()

    stderr = process.stderr.read()
    if process.wait():
        raise RuntimeError(f"mysql export failed: {stderr[-4000:]}")
    table_partial.replace(table_path)
    excluded_partial.replace(excluded_path)
    manifest = {
        "source": "OQMD v1.5 frozen MySQL snapshot",
        "dump_path": str(dump_path.resolve()),
        "dump_sha256": _sha256(dump_path),
        "query": OQMD_QUERY,
        "selection": {
            "fit": "standard",
            "canonical_entries_only": True,
            "source_site_filter": None,
            "stability_filter": "deferred to dataset ingestion",
        },
        "retained_rows": retained,
        "export_excluded_rows": excluded,
        "table": str(table_path.resolve()),
        "table_sha256": _sha256(table_path),
        "table_columns": fields,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    _json_dump(output_dir / "oqmd_source_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mysql-bin", type=Path, required=True)
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--database", default="qmdb_migrate")
    parser.add_argument("--dump-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    print(export_oqmd(**vars(args)))


if __name__ == "__main__":
    main()
