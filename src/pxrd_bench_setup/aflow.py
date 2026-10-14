"""Retrieve large selected AFLOW sets through bounded, complete AUID-prefix pages.

The official AFLUX paging(0) directive returns all matches. Only the exact
registered AUIDs enter the source index; native structure hashes remain strict.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import urllib.parse

from src.pxrd_bench_setup.numeric import aflow_candidates
from src.pxrd_bench_setup.serialization import _structure_from_aflow_record, _validate_aflow_record


def fetch_prefixes(db, directory: Path, pending, request_json, put_structure, *, offline=False, interval=1.0):
    groups = defaultdict(list)
    for sid in pending:
        if not re.fullmatch(r"aflow:[0-9a-f]{16}", sid):
            raise ValueError("Invalid AFLOW AUID in final registry")
        groups[sid[6:8]].append(sid)
    pages = directory / "native-prefix-pages"
    pages.mkdir(parents=True, exist_ok=True)
    completed = 0
    for prefix, ids in sorted(groups.items()):
        native_paths = {sid: directory / f"{sid[6:]}.json" for sid in ids}
        missing = [sid for sid in ids if not native_paths[sid].exists()]
        records = {}
        if missing:
            page = pages / f"{prefix}.json"
            if page.exists():
                payload = json.loads(page.read_text())
            else:
                if offline:
                    raise FileNotFoundError(f"Offline selected AFLOW prefix missing: {prefix}")
                query = (f"auid('aflow:{prefix}'*),species,composition,natoms,geometry,"
                         "positions_fractional,paging(0)")
                url = "https://aflow.org/API/aflux/?" + urllib.parse.quote(query, safe="(),:'*")
                payload = request_json(url, interval=interval)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid complete AFLOW prefix page: {prefix}")
            values = list(payload.values())
            returned = [record.get("auid") if isinstance(record, dict) else None for record in values]
            if (any(not isinstance(sid, str) or not re.fullmatch(r"aflow:[0-9a-f]{16}", sid)
                    or not sid.startswith(f"aflow:{prefix}") for sid in returned)):
                raise ValueError(f"Ambiguous or out-of-prefix AFLOW identities: {prefix}")
            for sid, record in zip(returned, values, strict=True):
                if sid in records and json.dumps(records[sid], sort_keys=True) != json.dumps(record, sort_keys=True):
                    raise ValueError(f"Ambiguous or out-of-prefix AFLOW identities: {prefix}")
                # AFLUX sometimes repeats the exact same native record. Only
                # identical duplicates may collapse; conflicting records fail.
                records[sid] = record
            absent = set(missing) - records.keys()
            if absent:
                raise ValueError(f"Selected AFLOW AUIDs missing from complete prefix {prefix}: {sorted(absent)[:3]}")
            if not page.exists():
                temporary = page.with_suffix(".json.part")
                temporary.write_text(json.dumps(payload, separators=(",", ":")))
                temporary.replace(page)
        for sid in ids:
            path = native_paths[sid]
            if path.exists():
                payload = json.loads(path.read_text())
                values = list(payload.values()) if isinstance(payload, dict) else []
                if len(values) != 1 or not isinstance(values[0], dict) or values[0].get("auid") != sid:
                    raise ValueError(f"Cached AFLOW identity differs: {sid}")
                record = values[0]
            else:
                record = records[sid]
            _validate_aflow_record(record)
            put_structure(db, "aflow", sid, _structure_from_aflow_record(record),
                          alternatives=aflow_candidates(record))
            # The atomic native page is the resumable raw source. Keep only
            # selected normalized structures in SQLite, avoiding millions of
            # redundant individual JSON files on a full setup.
        db.commit()
        completed += len(ids)
        print(f"[fetch] aflow: {completed:,}/{len(pending):,} pending entries complete (prefix {prefix})", flush=True)
