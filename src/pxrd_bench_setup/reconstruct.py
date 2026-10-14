"""Download selected upstream records and materialize a frozen PXRD-Bench registry.

Invoked by data/setup_pxrd_bench.sh. This CPU builder never reselects or resplits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import http.client
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from src.data.mp20_tokens import _simulate_one_peak
from src.data.pxrd_bench_schema import PXRD, SHARD_SCHEMA_VERSION, SPLITS
from src.pxrd_bench_setup.serialization import (
    _build_tensor_item, _iter_bz2_entries, _validate_and_primitive,
    _structure_from_aflow_record, _validate_aflow_record,
)
from src.pxrd_bench_setup.numeric import PROFILES, canonical_candidates, cif_candidates, aflow_candidates
from src.data.pxrd_bench import _bragg_items, _bragg_manifest, _bragg_sidecar_path
from src.data.pxrd_bench_integrity import (
    METADATA_COLUMNS, bragg_sha256, canonical_bytes, check_processing_versions,
    contained_path, file_sha256, membership_sha256, primitive_sha256,
    read_registry, read_split, tensor_sha256,
)
from src.pxrd.bragg_cache import (
    build_bragg_cache, load_bragg_cache, save_bragg_cache, structure_items_sha256,
)
from src.pxrd.peaks import CU_KA_WAVELENGTH


NUMPY_DISABLED_FEATURES = {"AVX512F", "AVX512CD", "AVX512_SKX", "AVX512_CLX", "AVX512_CNL", "AVX512_ICL"}


def check_numeric_profile() -> dict:
    from numpy.core._multiarray_umath import __cpu_features__
    disabled = set(os.environ.get("NPY_DISABLE_CPU_FEATURES", "").split(","))
    if disabled != NUMPY_DISABLED_FEATURES or os.environ.get("OPENBLAS_CORETYPE") != "Haswell":
        raise ValueError("Use data/setup_pxrd_bench.sh to select the frozen NumPy AVX2 / OpenBLAS Haswell math path")
    if any(__cpu_features__.get(feature) for feature in NUMPY_DISABLED_FEATURES):
        raise ValueError("Set the frozen CPU profile before importing NumPy; use data/setup_pxrd_bench.sh")
    if not __cpu_features__.get("AVX2") or not __cpu_features__.get("FMA3"):
        raise ValueError("Fresh reconstruction requires an x86-64 CPU supporting AVX2 and FMA3")
    return {"numpy_disabled_cpu_features": sorted(disabled), "openblas_coretype": "Haswell"}


def request_json(url: str, *, headers=None, interval=1.0):
    for attempt in range(5):
        try:
            started = time.monotonic()
            request = urllib.request.Request(url, headers={"User-Agent": "PXRD-Bench/1.0", **(headers or {})})
            with urllib.request.urlopen(request, timeout=180) as response:
                chunks = []
                # read1 returns available bytes without waiting for the whole
                # body; a trickling connection must not wait indefinitely.
                while chunk := response.read1(64 * 1024):
                    if time.monotonic() - started > 180:
                        raise TimeoutError("Provider response deadline exceeded")
                    chunks.append(chunk)
                result = json.loads(b"".join(chunks))
            time.sleep(interval)
            return result
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                # Do not expose an API key or response body in diagnostics.
                raise RuntimeError(f"Provider request failed with HTTP {exc.code}") from None
            time.sleep(max(interval, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            if attempt == 4:
                raise RuntimeError("Provider request failed after bounded retries") from None
            time.sleep(max(interval, 2 ** attempt))
    raise RuntimeError("Provider request did not return data")


def download_file(url: str, path: Path, expected: str, offline=False) -> Path:
    if path.exists():
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen resource checksum differs: {path}")
        print(f"[download] Verified existing frozen file: {path}", flush=True)
        return path
    if offline:
        raise FileNotFoundError(f"Offline resource missing: {path}")
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Only HTTPS upstream downloads are supported")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "PXRD-Bench/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    print(f"[download] {path.name}: requesting original source (local partial: {start:,} bytes)", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
        # Only append an exact Content-Range response; a full response restarts.
        append = (response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {start}-"))
        received = start if append else 0
        progress_at = received + 256 * 1024 * 1024
        with partial.open("ab" if append else "wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                received += len(chunk)
                if received >= progress_at:
                    print(f"[download] {path.name}: {received:,} bytes received", flush=True)
                    progress_at = received + 256 * 1024 * 1024
    if file_sha256(partial) != expected:
        raise ValueError(f"Downloaded snapshot differs from the paper snapshot: {path}")
    partial.replace(path)
    print(f"[download] Frozen SHA-256 passed: {path.name}", flush=True)
    return path


def resource_url(resource: dict) -> str:
    if not resource.get("link_filename"):
        return resource["url"]
    links = []
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                href = dict(attrs).get("href", "")
                if Path(urllib.parse.urlparse(href).path).name == resource["link_filename"]:
                    links.append(urllib.parse.urljoin(resource["url"], href))
    # The dynamic download page can be unavailable while official static files work.
    # Only the exact paper snapshot has a known fallback; download_file still
    # requires its frozen SHA-256 before accepting a single byte as input.
    fallback = None
    if (urllib.parse.urlparse(resource["url"]).hostname in {"oqmd.org", "www.oqmd.org"}
            and resource["link_filename"] == "qmdb__v1_5__102021.sql.gz"):
        fallback = "https://static.oqmd.org/static/downloads/qmdb__v1_5__102021.sql.gz"
    try:
        with urllib.request.urlopen(resource["url"], timeout=20) as response:
            Links().feed(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError):
        if not fallback:
            raise
        print("[download] OQMD page unavailable; using the official frozen static snapshot", file=sys.stderr, flush=True)
        return fallback
    if not links and fallback:
        print("[download] Frozen link absent from page; using the official OQMD static snapshot", file=sys.stderr, flush=True)
        return fallback
    if len(set(links)) != 1:
        raise ValueError("Official source page does not identify a unique frozen snapshot link")
    return links[0]


def open_index(cache: Path) -> sqlite3.Connection:
    cache.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache / "selected.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS selected (source TEXT, source_id TEXT, resource TEXT, "
               "release TEXT, primitive_sha256 TEXT, structure_json TEXT, PRIMARY KEY(source,source_id))")
    db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
    return db


def register(registry: Path, cache: Path) -> tuple[dict, dict]:
    manifest, catalog = read_registry(registry)
    check_processing_versions(catalog)
    db = open_index(cache)
    try:
        fingerprint = file_sha256(registry / "manifest.json")
        old = db.execute("SELECT value FROM state WHERE key='registry_sha256'").fetchone()
        if old and old[0] != fingerprint:
            raise ValueError("Source cache belongs to another registry; choose a separate --cache")
        db.execute("INSERT OR REPLACE INTO state VALUES ('registry_sha256',?)", (fingerprint,))
        indexed = db.execute("SELECT value FROM state WHERE key='registration_complete'").fetchone() == ("yes",)
        seen: set[str] = set()
        for split in SPLITS:
            frame = read_split(registry, manifest, split)
            if seen.intersection(frame.material_id):
                raise ValueError(f"Cross-split overlap in final registry: {split}")
            seen.update(frame.material_id)
            if not frame.structure_entity_id.equals(frame.material_id):
                raise ValueError(f"Structure identity differs from material ID: {split}")
            for name, rows in frame.groupby("source"):
                if name not in catalog["sources"] or set(rows.source_release) != {catalog["sources"][name]["source_release"]}:
                    raise ValueError(f"Source version differs: {name}/{split}")
                if name not in ("aflow", "materials_project"):
                    if set(rows.source_resource) - catalog["sources"][name]["resources"].keys():
                        raise ValueError(f"Unknown frozen source resource: {name}/{split}")
            if indexed:
                continue
            for row in frame.itertuples(index=False):
                prior = db.execute("SELECT resource,release,primitive_sha256 FROM selected "
                                   "WHERE source=? AND source_id=?", (row.source, row.source_id)).fetchone()
                expected = (row.source_resource, row.source_release, row.primitive_sha256)
                if prior is not None and prior != expected:
                    raise ValueError(f"Conflicting source identity: {row.source}/{row.source_id}")
                db.execute("INSERT OR IGNORE INTO selected VALUES (?,?,?,?,?,NULL)",
                           (row.source, row.source_id, *expected))
            db.commit()
        db.execute("INSERT OR REPLACE INTO state VALUES ('registration_complete','yes')")
        db.commit()
    finally:
        db.close()
    return manifest, catalog


def put_structure(db, source: str, source_id: str, structure, *, alternatives=()) -> None:
    expected = db.execute("SELECT primitive_sha256 FROM selected WHERE source=? AND source_id=?",
                          (source, source_id)).fetchone()
    if expected is None:
        return  # Never retain an unselected entry from an upstream archive.
    primitive = _validate_and_primitive(structure)
    if primitive_sha256(primitive) != expected[0]:
        for alternative in alternatives:
            primitive = _validate_and_primitive(alternative)
            if primitive_sha256(primitive) == expected[0]:
                break
        else:
            raise ValueError(f"Source structure content differs: {source}/{source_id}")
    db.execute("UPDATE selected SET structure_json=? WHERE source=? AND source_id=?",
               (json.dumps(primitive.as_dict(), separators=(",", ":")), source, source_id))


def fetch_sources(registry: Path, cache: Path, *, offline=False, interval=1.0,
                  oqmd_table: Path | None = None) -> None:
    os.environ.setdefault("PXRD_BENCH_MATH_ROOT", str(cache / "tools/canonical-math"))
    _, catalog = register(registry, cache)
    check_numeric_profile()
    db = open_index(cache)
    try:
        # Finish independent snapshot sources before the live AFLOW API so a
        # transient API outage does not prevent those downloads from resuming.
        for source in sorted(catalog["sources"], key=lambda name: name == "aflow"):
            data = catalog["sources"][source]
            resources = [row[0] for row in db.execute(
                "SELECT DISTINCT resource FROM selected WHERE source=? AND structure_json IS NULL", (source,))]
            if source == "alexandria":
                for key in resources:
                    resource = data["resources"][key]
                    path = contained_path(cache / source, resource["filename"])
                    download_file(resource["url"], path, resource["sha256"], offline)
                    selected = {row[0] for row in db.execute(
                        "SELECT source_id FROM selected WHERE source=? AND resource=? AND structure_json IS NULL", (source, key))}
                    for entry in _iter_bz2_entries(path):
                        sid = str(entry.get("data", {}).get("mat_id", ""))
                        if sid in selected:
                            put_structure(db, source, sid, Structure.from_dict(entry["structure"]))
                    db.commit()
            elif source == "oqmd" and resources:
                if oqmd_table is None:
                    raise FileNotFoundError("OQMD selected records require --oqmd-table generated from the frozen SQL snapshot")
                expected = data["table_sha256"]
                if file_sha256(oqmd_table) != expected:
                    raise ValueError("OQMD exported table differs from the frozen paper source")
                selected = {row[0] for row in db.execute(
                    "SELECT source_id FROM selected WHERE source='oqmd' AND structure_json IS NULL")}
                normalized = 0
                for part in pd.read_csv(oqmd_table, dtype={"material_id": str}, chunksize=10000):
                    for row in part.loc[part.material_id.isin(selected)].itertuples(index=False):
                        content = str(row.cif)
                        put_structure(db, source, str(row.material_id), Structure.from_str(content, fmt="cif"),
                                      alternatives=cif_candidates(content))
                        normalized += 1
                    db.commit()
                    print(f"[fetch] oqmd: {normalized:,}/{len(selected):,} pending entries complete", flush=True)
            elif source == "aflow" and resources:
                pending = [row[0] for row in db.execute(
                    "SELECT source_id FROM selected WHERE source='aflow' AND structure_json IS NULL")]
                directory = cache / source
                directory.mkdir(parents=True, exist_ok=True)
                if len(pending) > 4096 or (directory / "native-prefix-pages").exists():
                    from src.pxrd_bench_setup.aflow import fetch_prefixes
                    fetch_prefixes(db, directory, pending, request_json, put_structure,
                                   offline=offline, interval=interval)
                    continue
                for start in range(0, len(pending), 64):
                    ids = pending[start:start + 64]
                    if any(not re.fullmatch(r"aflow:[0-9a-f]{16}", sid) for sid in ids):
                        raise ValueError("Invalid AFLOW AUID in final registry")
                    missing = [sid for sid in ids if not (directory / f"{sid.split(':')[1]}.json").exists()]
                    if missing:
                        if offline:
                            raise FileNotFoundError(f"Offline selected AFLOW records missing: {missing[:3]}")
                        # AFLUX colon is OR within a property. Limit the request to selected AUIDs.
                        selector = ":".join(f"'{sid}'" for sid in missing)
                        query = (f"auid({selector}),species,composition,natoms,geometry,positions_fractional,"
                                 f"paging(1,{len(missing)})")
                        payload = request_json("https://aflow.org/API/aflux/?" + urllib.parse.quote(query, safe=""), interval=interval)
                        records = list(payload.values()) if isinstance(payload, dict) else []
                        returned = [record.get("auid") for record in records]
                        if len(returned) != len(missing) or set(returned) != set(missing):
                            raise ValueError("Missing or ambiguous exact selected AFLOW AUIDs")
                        for record in records:
                            sid = record["auid"]
                            _validate_aflow_record(record)
                            put_structure(db, source, sid, _structure_from_aflow_record(record),
                                          alternatives=aflow_candidates(record))
                            (directory / f"{sid.split(':')[1]}.json").write_text(json.dumps({sid: record}))
                    for sid in ids:
                        payload = json.loads((directory / f"{sid.split(':')[1]}.json").read_text())
                        records = list(payload.values()) if isinstance(payload, dict) else []
                        if len(records) != 1 or records[0].get("auid") != sid:
                            raise ValueError(f"Cached AFLOW identity differs: {sid}")
                        _validate_aflow_record(records[0])
                        put_structure(db, source, sid, _structure_from_aflow_record(records[0]),
                                      alternatives=aflow_candidates(records[0]))
                    db.commit()
            elif source == "materials_project" and resources:
                pending = [row[0] for row in db.execute(
                    "SELECT source_id FROM selected WHERE source='materials_project' AND structure_json IS NULL")]
                for start in range(0, len(pending), 100):
                    ids = pending[start:start + 100]
                    cache_dir = cache / source
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    missing = [sid for sid in ids if not (cache_dir / f"{sid}.json").exists()]
                    if any(not re.fullmatch(r"mp-[a-z0-9]+", sid) for sid in ids):
                        raise ValueError("Invalid Materials Project ID in registry")
                    if missing:
                        if offline:
                            raise FileNotFoundError(f"Offline MP records missing: {missing[:3]}")
                        key = os.environ.get("MP_API_KEY")
                        if not key:
                            raise ValueError("Set MP_API_KEY to download selected Materials Project entries")
                        query = urllib.parse.urlencode({"material_ids": ",".join(missing),
                            "_fields": "material_id,structure,origins,theoretical,builder_meta", "_limit": len(missing)})
                        payload = request_json("https://api.materialsproject.org/materials/summary/?" + query,
                                               headers={"X-API-KEY": key}, interval=interval)
                        docs = payload.get("data", [])
                        if len(docs) != len(missing) or {str(doc["material_id"]) for doc in docs} != set(missing):
                            raise ValueError("Missing or ambiguous selected Materials Project entries")
                        for doc in docs:
                            sid = str(doc["material_id"])
                            metadata = doc.get("builder_meta", {})
                            if metadata.get("database_version") != data["source_release"]:
                                raise ValueError(f"Materials Project source release differs: {sid}")
                            put_structure(db, source, sid, Structure.from_dict(doc["structure"]))
                            (cache_dir / f"{sid}.json").write_text(json.dumps(doc))
                    for sid in ids:
                        if sid in missing:
                            continue
                        doc = json.loads((cache_dir / f"{sid}.json").read_text())
                        if str(doc["material_id"]) != sid:
                            raise ValueError(f"Cached MP identity differs: {sid}")
                        if doc.get("builder_meta", {}).get("database_version") != data["source_release"]:
                            raise ValueError(f"Cached MP source release differs: {sid}")
                        put_structure(db, source, sid, Structure.from_dict(doc["structure"]))
                    db.commit()
                    print(f"[fetch] materials_project: {start + len(ids):,}/{len(pending):,} pending entries complete", flush=True)
            elif resources:
                raise ValueError(f"Unsupported upstream source: {source}")
        missing = db.execute("SELECT source,COUNT(*) FROM selected WHERE structure_json IS NULL GROUP BY source").fetchall()
        if missing:
            raise ValueError(f"Selected source records remain missing: {missing}")
    finally:
        db.close()


def create_item(row: dict, structure_json: str):
    primitive = Structure.from_dict(json.loads(structure_json))
    if primitive_sha256(primitive) != row["primitive_sha256"]:
        raise ValueError(f"Cached source fingerprint differs: {row['material_id']}")
    mid, item, error = _build_tensor_item((row["material_id"], structure_json))
    if error is not None:
        raise ValueError(f"Tensor/PXRD generation failed for {mid}: {error}")
    item = {key: torch.as_tensor(value) for key, value in item.items()}
    item["num_atoms"] = item["num_atoms"].to(torch.long)
    if int(item["num_atoms"]) > 52:
        raise ValueError(f"Processed structure/dense PXRD differs: {mid}")
    if tensor_sha256(item) != row["tensors_sha256"]:
        from src.pxrd_bench_setup.numeric import dense_pxrd_candidates
        for dense in dense_pxrd_candidates(primitive):
            candidate = {**item, "xrd": dense}
            if tensor_sha256(candidate) == row["tensors_sha256"]:
                item = candidate
                break
        else:
            raise ValueError(f"Processed structure/dense PXRD differs: {mid}")
    item["mp_id"] = mid
    canonical = canonical_bytes(item)
    if hashlib.sha256(canonical).hexdigest() != row["canonical_sha256"]:
        # Recompute with the bounded original libm variants. Every candidate
        # must still match the exact frozen bytes; there is no tolerance gate.
        for candidate in canonical_candidates(item):
            if hashlib.sha256(candidate).hexdigest() == row["canonical_sha256"]:
                canonical = candidate
                break
        else:
            raise ValueError(f"Canonical structure differs: {mid}")
    _, tt, ii = _simulate_one_peak(item, CU_KA_WAVELENGTH, PXRD["two_theta_min"], PXRD["two_theta_max"])
    if len(tt) == 0 or bragg_sha256(tt, ii) != row["bragg_sha256"]:
        raise ValueError(f"Bragg PXRD differs: {mid}")
    return item, canonical, (tt, ii)


def _create_item_task(task):
    item, canonical, pattern = create_item(*task)
    # Transfer ordinary arrays, avoiding thousands of shared Torch file descriptors.
    arrays = {key: value.numpy() if isinstance(value, torch.Tensor) else value for key, value in item.items()}
    return arrays, canonical, pattern


def _worker_init() -> None:
    torch.set_num_threads(1)


def verify_output(registry: Path, output: Path) -> dict:
    manifest, catalog = read_registry(registry)
    check_processing_versions(catalog)
    local = json.loads((output / "manifest.json").read_text())
    if (local.get("dataset_id") != "pxrd_bench" or local.get("distribution") != "materialized"
            or local.get("final_curated_splits") is not True
            or local.get("registry_sha256") != file_sha256(registry / "manifest.json")):
        raise ValueError("Reconstructed dataset identity differs")
    report = {"registry_sha256": local["registry_sha256"], "splits": {}, "passed": False}
    for split in SPLITS:
        expected = read_split(registry, manifest, split)
        actual = pd.read_parquet(output / f"views/nmax52/splits/{split}.parquet")
        if len(actual) != len(expected) or list(actual.material_id) != list(expected.material_id):
            raise ValueError(f"Final membership/order differs: {split}")
        for column in METADATA_COLUMNS:
            if column in ("energy_above_hull", "num_sites"):
                # Parse shortest-roundtrip CSV decimals without pd.to_numeric's precision loss.
                a = actual[column].to_numpy(dtype=np.float64)
                b = expected[column].replace("", np.nan).astype(np.float64).to_numpy()
                if not np.array_equal(a, b, equal_nan=True):
                    raise ValueError(f"Metadata differs: {split}/{column}")
            else:
                left = actual[column].fillna("").astype(str).reset_index(drop=True)
                right = expected[column].fillna("").astype(str).reset_index(drop=True)
                if not left.equals(right):
                    raise ValueError(f"Metadata differs: {split}/{column}")
        env = lmdb.open(str(output / "structure_cache/primitive_niggli_v1" / f"pxrd_bench.{split}.lmdb"),
                        readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                if txn.stat()["entries"] != len(expected):
                    raise ValueError(f"Canonical cache has extra/missing entries: {split}")
                for shard, rows in actual.groupby("shard", sort=False):
                    payload = torch.load(output / "shards" / shard, weights_only=False, map_location="cpu", mmap=True)
                    if payload.get("schema_version") != SHARD_SCHEMA_VERSION or payload["material_ids"] != rows.material_id.tolist():
                        raise ValueError(f"Shard contains extra/missing or unordered entries: {shard}")
                    source_hash = structure_items_sha256(_bragg_items(payload))
                    bragg = load_bragg_cache(_bragg_sidecar_path(output, shard, source_hash),
                        expected_ids=payload["material_ids"],
                        expected_manifest=_bragg_manifest(shard, source_hash, CU_KA_WAVELENGTH))
                    for row, ref in zip(rows.itertuples(index=False), expected.loc[rows.index].itertuples(index=False), strict=True):
                        offset = int(row.offset)
                        if payload["material_ids"][offset] != row.material_id:
                            raise ValueError(f"Tensor offset identity differs: {row.material_id}")
                        item = {key: values[offset] for key, values in payload["tensors"].items()}
                        if tensor_sha256(item) != ref.tensors_sha256:
                            raise ValueError(f"Tensor content differs: {row.material_id}")
                        value = txn.get(str(row.material_id).encode())
                        if value is None or hashlib.sha256(value).hexdigest() != ref.canonical_sha256:
                            raise ValueError(f"Canonical content differs: {row.material_id}")
                        start, stop = map(int, bragg.offsets[offset:offset + 2])
                        if bragg_sha256(bragg.two_theta[start:stop], bragg.intensity[start:stop]) != ref.bragg_sha256:
                            raise ValueError(f"Bragg content differs: {row.material_id}")
        finally:
            env.close()
        report["splits"][split] = {"rows": len(actual), "ordered_membership_sha256": membership_sha256(actual.material_id),
                                   "all_tensor_canonical_bragg_hashes_equal": True}
        print(f"[verify] {split}: all {len(actual):,} rows passed", flush=True)
    report["passed"] = True
    return report


def build(registry: Path, cache: Path, output: Path, *, shard_size=2048, workers=1) -> dict:
    os.environ.setdefault("PXRD_BENCH_MATH_ROOT", str(cache / "tools/canonical-math"))
    numeric_profile = check_numeric_profile()
    manifest, catalog = register(registry, cache)
    check_processing_versions(catalog)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if shard_size <= 0 or workers <= 0:
        raise ValueError("Shard size and worker count must be positive")
    temporary = output.with_name(output.name + ".building")
    temporary.mkdir(parents=True, exist_ok=True)
    fingerprint = file_sha256(registry / "manifest.json")
    state_path = temporary / "build-state.json"
    state = {"registry_sha256": fingerprint, "shard_size": shard_size,
             "mode": "upstream_recalculation",
             "reused_manifest_sha256": None}
    if numeric_profile is not None:
        state["numeric_profile"] = numeric_profile
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError("Unfinished output belongs to different registry or shard size")
    state_path.write_text(json.dumps(state))
    (temporary / "shards").mkdir(exist_ok=True)
    (temporary / "splits").mkdir(exist_ok=True)
    split_root = temporary / "views/nmax52/splits"
    split_root.mkdir(parents=True, exist_ok=True)
    db = open_index(cache)
    pool = None
    if workers > 1:
        # Spawn after selecting the frozen CPU profile, with one math thread per worker.
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    local = {**manifest, "distribution": "materialized", "registry_sha256": fingerprint,
             "splits": {}, "selection": {"available_nmax_views": [52]}, "views": {},
             "reconstruction": state}
    local["reconstruction_math"] = {
        "canonical_profile_selection": "exact_frozen_sha256",
        "source_profile_selection": "exact_frozen_primitive_sha256",
        "dense_pxrd_profile_selection": "exact_frozen_tensors_sha256",
        "canonical_libm_sha256": {key: value["library_sha256"] for key, value in PROFILES.items()},
    }
    try:
        for split in SPLITS:
            refs = read_split(registry, manifest, split)
            metadata_parts = []
            cache_path = temporary / "structure_cache/primitive_niggli_v1" / f"pxrd_bench.{split}.lmdb"
            cache_path.mkdir(parents=True, exist_ok=True)
            env = lmdb.open(str(cache_path), map_size=max(1 << 26, len(refs) * 4096))
            try:
                for start in range(0, len(refs), shard_size):
                    rows = refs.iloc[start:start + shard_size]
                    shard = f"{split}-{start // shard_size:05d}.pt"
                    path = temporary / "shards" / shard
                    marker = path.with_suffix(".complete.json")
                    if marker.exists():
                        done = json.loads(marker.read_text())
                        if file_sha256(path) != done["sha256"]:
                            raise ValueError(f"Resumed tensor shard checksum differs: {shard}")
                        bragg_path = contained_path(temporary, done["bragg_file"])
                        if file_sha256(bragg_path) != done["bragg_sha256"]:
                            raise ValueError(f"Resumed Bragg cache checksum differs: {shard}")
                    else:
                        if path.exists():
                            path.unlink()  # Only an unfinished shard in our .building output.
                        items, patterns, canonical_rows = [], [], []
                        tasks = []
                        for row in rows.to_dict("records"):
                            record = db.execute("SELECT structure_json FROM selected WHERE source=? AND source_id=?",
                                                (row["source"], row["source_id"])).fetchone()
                            if record is None or record[0] is None:
                                raise ValueError(f"Selected source was not downloaded: {row['material_id']}")
                            tasks.append((row, record[0]))
                        calculated = pool.map(_create_item_task, tasks, chunksize=8) if pool else map(_create_item_task, tasks)
                        for (row, _), (arrays, canonical, pattern) in zip(tasks, calculated, strict=True):
                            item = {key: torch.as_tensor(value) if key != "mp_id" else value for key, value in arrays.items()}
                            items.append(item)
                            patterns.append((row["material_id"], pattern))
                            canonical_rows.append((row["material_id"].encode(), canonical))
                        payload = {"schema_version": SHARD_SCHEMA_VERSION,
                                   "material_ids": rows.material_id.tolist(),
                                   "tensors": {key: torch.stack([item[key] for item in items])
                                               for key in ("A0", "F1", "Y1", "pad_mask", "num_atoms", "xrd")}}
                        part_path = path.with_suffix(".part")
                        torch.save(payload, part_path)
                        part_path.replace(path)
                        source_hash = structure_items_sha256(_bragg_items(payload))
                        bragg_path = _bragg_sidecar_path(temporary, shard, source_hash)
                        cache_manifest = _bragg_manifest(shard, source_hash, CU_KA_WAVELENGTH)
                        if not bragg_path.exists():
                            save_bragg_cache(bragg_path, build_bragg_cache(patterns, manifest=cache_manifest))
                        else:
                            load_bragg_cache(bragg_path, expected_manifest=cache_manifest, expected_ids=payload["material_ids"])
                        with env.begin(write=True) as txn:
                            for key, value in canonical_rows:
                                txn.put(key, value)
                        env.sync()
                        marker.write_text(json.dumps({"sha256": file_sha256(path),
                            "bragg_file": str(bragg_path.relative_to(temporary)), "bragg_sha256": file_sha256(bragg_path)}))
                    part = rows.loc[:, list(METADATA_COLUMNS)].copy()
                    part["energy_above_hull"] = part.energy_above_hull.replace("", np.nan).astype(np.float64)
                    part["num_sites"] = part.num_sites.astype(int)
                    part["split"], part["shard"] = split, shard
                    part["offset"] = np.arange(len(part), dtype=np.int32)
                    metadata_parts.append(part)
                    print(f"[build] {split}/{shard}: {start + len(rows):,}/{len(refs):,}", flush=True)
            finally:
                env.close()
            (cache_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "dataset_class": "pxrd_bench",
                "split": split, "complete": True, "entry_count": len(refs), "tag": "primitive_niggli_v1",
                "transform": [{"name": "get_primitive_structure", "tolerance": 0.01},
                              {"name": "get_reduced_structure", "reduction_algo": "niggli"}]}))
            metadata = pd.concat(metadata_parts, ignore_index=True) if metadata_parts else pd.DataFrame(
                columns=[*METADATA_COLUMNS, "split", "shard", "offset"])
            metadata.to_parquet(split_root / f"{split}.parquet", index=False)
            metadata.to_parquet(temporary / "splits" / f"{split}.parquet", index=False)
            local["splits"][split] = {"rows": len(metadata), "file": f"views/nmax52/splits/{split}.parquet",
                "sha256": file_sha256(split_root / f"{split}.parquet"),
                "ordered_membership_sha256": membership_sha256(metadata.material_id)}
        local["views"]["nmax52"] = {"shard_counts": {split: spec["rows"] for split, spec in local["splits"].items()}}
        (temporary / "manifest.json").write_text(json.dumps(local, indent=2) + "\n")
        report = verify_output(registry, temporary)
        (temporary / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
        return report
    finally:
        db.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "fetch", "build", "verify", "download-resource", "oqmd-info", "mp-info"))
    parser.add_argument("--registry", type=Path, default=Path("data/pxrd_bench_registry"))
    parser.add_argument("--cache", type=Path, default=Path("data/sources"))
    parser.add_argument("--output", type=Path, default=Path("data/pxrd_bench"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--oqmd-table", type=Path)
    parser.add_argument("--resource-source")
    parser.add_argument("--resource-key")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.interval < 1.0:
        parser.error("--interval must be at least one second")
    if args.action == "check":
        check_numeric_profile()
        register(args.registry, args.cache)
        print("Registry integrity, CPU profile and processing versions passed")
    elif args.action == "fetch":
        fetch_sources(args.registry, args.cache, offline=args.offline, interval=args.interval, oqmd_table=args.oqmd_table)
    elif args.action == "build":
        print(json.dumps(build(args.registry, args.cache, args.output, shard_size=args.shard_size, workers=args.workers), indent=2))
    elif args.action == "verify":
        print(json.dumps(verify_output(args.registry, args.output), indent=2))
    elif args.action == "download-resource":
        _, catalog = read_registry(args.registry)
        resource = catalog["sources"][args.resource_source]["resources"][args.resource_key]
        path = contained_path(args.cache / args.resource_source, resource["filename"])
        url = resource["url"] if path.exists() or args.offline else resource_url(resource)
        download_file(url, path, resource["sha256"], args.offline)
        print(path)
    elif args.action == "mp-info":
        db = open_index(args.cache)
        try:
            # Prompt only when a selected MP record actually requires a request.
            pending = db.execute("SELECT source_id FROM selected WHERE source='materials_project' AND structure_json IS NULL")
            if any(not (args.cache / "materials_project" / f"{sid}.json").is_file() for (sid,) in pending):
                print("MP_API_KEY required")
        finally:
            db.close()
    elif args.action == "oqmd-info":
        _, catalog = read_registry(args.registry)
        db = open_index(args.cache)
        try:
            count = db.execute("SELECT COUNT(*) FROM selected WHERE source='oqmd' AND structure_json IS NULL").fetchone()[0]
            if count:
                source = catalog["sources"]["oqmd"]
                if len(source["resources"]) != 1:
                    raise ValueError("Expected one frozen OQMD snapshot")
                key, resource = next(iter(source["resources"].items()))
                database = source["database"]
                if not re.fullmatch(r"[A-Za-z0-9_]+", database):
                    raise ValueError("Invalid snapshot database name")
                print(key)
                print(contained_path(args.cache / "oqmd", resource["filename"]))
                print(database)
        finally:
            db.close()


if __name__ == "__main__":
    main()
