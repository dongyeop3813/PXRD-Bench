"""Verify existing frozen MP-20 polymorph split files; never generate splits."""
import hashlib
import json
from pathlib import Path


REPO_ID = "colabfit/MP-20-polymorph-split"
REVISION = "82d98554d7a33713a6567c711cfca94c6a1fdbf8"
EXPECTED_FILES = {
    "train": {
        "rows": 27_137,
        "sha256": "fca4cfd2feb34d7096a98600488f7769ef8f0fd058e358d1337fb438905e9f00",
    },
    "val": {
        "rows": 9_060,
        "sha256": "331e42d254e85c27b3e0ede8b34e1d7e0d7c3223668db52b782648fad580b7ca",
    },
    "test": {
        "rows": 9_032,
        "sha256": "a9ac5c7c7a1e43a4d219b21ac6b640019dd48db47dc4468cf030427bb347c030",
    },
}

SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prepared_dataset(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset") != "mp20_polymorph_split"
        or manifest.get("nmax") != 20
        or manifest.get("source") != {"repo_id": REPO_ID, "revision": REVISION}
    ):
        raise ValueError("MP-20 polymorph split identity mismatch")
    for split in SPLITS:
        expected = EXPECTED_FILES[split]
        record = manifest["files"][split]
        if record["source"] != f"source/{split}.parquet":
            raise ValueError(f"{split} source path mismatch")
        if record["raw"] != f"raw/{split}.csv":
            raise ValueError(f"{split} raw path mismatch")
        if record["source_sha256"] != expected["sha256"]:
            raise ValueError(f"{split} manifest source checksum mismatch")
        source = root / record["source"]
        raw = root / record["raw"]
        if record["rows"] != expected["rows"]:
            raise ValueError(f"{split} manifest row count mismatch")
        if sha256(source) != expected["sha256"]:
            raise ValueError(f"{split} source checksum mismatch")
        if sha256(raw) != record["raw_sha256"]:
            raise ValueError(f"{split} raw CSV checksum mismatch")
    return manifest


