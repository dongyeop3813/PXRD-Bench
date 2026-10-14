# PXRD-Bench setup

PXRD-Bench contains **4,262,691 entries**. The repository includes source
identifiers and checksums in [`pxrd_bench_registry/`](pxrd_bench_registry/).
The setup script downloads provider data and builds the dataset locally.

## One-command setup

From the repository root on Linux x86-64, run:

```bash
bash data/setup_pxrd_bench.sh
```

The script prepares Python 3.12 in `.venv`, installs the frozen scientific
packages and CPU-only PyTorch, obtains selected upstream entries, and writes
parquet splits, tensor shards, canonical LMDB caches and Bragg sidecars. It
verifies every entry against the frozen registry before declaring success.
No environment activation, separate pip commands, sudo or database service
configuration are needed. Existing compatible environments are reused.

If selected Materials Project records are missing, the script asks for your
[Materials Project API key](https://next-gen.materialsproject.org/api) with
hidden terminal input before downloading the large source snapshots. The key
is kept only in the process environment, never saved in the dataset or log.
For unattended runs, set `MP_API_KEY` in the environment. A fully populated
source cache and verification of a completed dataset need no key.

Downloads, private tools and progress logs live under `data/sources`; the
verified dataset appears at `data/pxrd_bench`. Logs are in
`data/sources/logs/setup-*.log`. The first setup needs internet access, standard
Linux utilities (`bash`, `curl`, `tar`, `gzip`, `sha256sum`) and enough free disk
space for the provider archives, OQMD SQL database, source index and
4.26-million-entry dataset. The script downloads pinned
[uv](https://docs.astral.sh/uv/guides/install-python/) locally when needed;
Python is downloaded automatically if no suitable interpreter is available.
Fresh calculation requires AVX2 and FMA3 CPU instructions. OQMD tools are also
installed locally if missing, with their official archive checksum checked.
A system lacking MariaDB's required shared libraries needs those installed or
an existing verified OQMD export supplied via `--oqmd-table PATH`.

Useful options:

```bash
# Prepare only the dataset reconstruction environment.
bash data/setup_pxrd_bench.sh --environment-only
# Use different storage locations, including paths containing spaces.
bash data/setup_pxrd_bench.sh --cache /path/to/sources --output /path/to/pxrd_bench
# Compute/export with more CPU workers; results remain checked entry by entry.
bash data/setup_pxrd_bench.sh --workers 16
# Require an existing environment instead of installing packages/tools.
bash data/setup_pxrd_bench.sh --no-bootstrap
```

This prepares the CPU dataset reconstruction environment. Model training needs
the additional dependencies in `pyproject.toml` and its own runtime validation.
An existing incompatible Python `.venv` is not deleted: move it aside and rerun.
Scientific processing is implemented under `src/pxrd_bench_setup/`; runtime
loaders live under `src/data/`. Only registered AFLOW/MP records are fetched;
Alexandria and OQMD use the frozen snapshots in `sources.json`.

Numerical reconstruction uses fixed package versions, AVX2/FMA3 CPU dispatch
and checksum-pinned math libraries in the local tool cache. No system libraries
are replaced. Every record must match the registered source and output hashes;
missing or changed records stop the build.

Source versions and usage conditions are listed in
[DATA_LICENSES.md](../DATA_LICENSES.md). OQMD setup imports the checksum-verified
SQL snapshot into a private MariaDB directory and Unix socket, with networking
disabled. It stops that private instance on exit. An existing verified CSV
export can instead be supplied with `--oqmd-table PATH`.

## Resume, verify and use

```bash
bash data/setup_pxrd_bench.sh --offline
bash data/setup_pxrd_bench.sh --output data/pxrd_bench --verify-only
bash data/setup_pxrd_bench.sh --help
```

Use the same registry, cache, output and shard size when resuming. Completed
shards and Bragg files are checked before reuse. An unfinished build lives in
`<output>.building`; the final destination appears only after full verification.
Re-running the same command automatically verifies a completed output
instead of downloading or overwriting it. `--offline` forbids network access
for Python, package/tool installation and provider data alike; missing local
inputs cause a clear failure. Completed shards and source caches are reused. An interrupted OQMD SQL import
restarts in the script's own private snapshot database.

After successful verification, `data/pxrd_bench` contains `manifest.json`, nine
parquet split tables, tensor shards, canonical LMDB caches and Bragg sidecars.
The public loader consumes these tables. Train with `data=pxrd_bench data.data_root=data/pxrd_bench`; evaluation
uses `data.dataset_class=pxrd_bench data.root=data/pxrd_bench`.

The standalone loader defaults to the final nmax52 view:

```python
from src.data.pxrd_bench import make_pxrd_bench_dataloader

loader = make_pxrd_bench_dataloader(
    "data/pxrd_bench", split="train", batch_size=16, shuffle=False
)
batch = next(iter(loader))
```

`energy_above_hull` is provenance metadata, not a model input. AFLOW does not
supply it, so its entries retain NaN. Validate model-input tensors separately
from this optional metadata.

## Verified reconstruction

A clean setup was verified on **2026-10-14**. Starting without a `.venv`,
source cache or generated dataset, `bash data/setup_pxrd_bench.sh --workers 64`
finished in about **7 hours 8 minutes** on the validation host. The public shell
ran once, with no source edits or restarts. All **4,262,691 final entries**
were rebuilt from downloaded upstream data:

| Source | Final entries |
| --- | ---: |
| AFLOW | 1,879,759 |
| Alexandria | 2,139,337 |
| Materials Project | 77,493 |
| OQMD | 166,102 |

Every source structure, tensor, dense PXRD, canonical cache and Bragg hash
matches the frozen references. All nine splits preserve their IDs, order and
metadata. The complete OQMD SQL download, import and regenerated CSV also
passed their checksums. Original research payloads and caches were not used
as reconstruction inputs.

All nine splits passed real standalone batches and two-worker batches with
canonical caches and Bragg inputs. The executed setup and loader files match
the current public release. Earlier implementation checks also passed
**107 CPU regression tests**.

## Licenses

See [DATASET_LICENSE.md](../DATASET_LICENSE.md) and
[DATA_LICENSES.md](../DATA_LICENSES.md). Our own covered contributions use
CC BY-NC 4.0; upstream data retain their respective terms. OQMD's calculated
data notice uses [CC BY 4.0](https://oqmd.org/api/); original ICSD inputs have
separate restrictions. We do not mirror AFLOW structures or derived payloads.
Code and ID lists do not grant permission to redistribute a locally built
combined dataset. 

