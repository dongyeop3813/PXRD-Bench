# PXRD-Bench

[![arXiv](https://img.shields.io/badge/arXiv-2610.16480-b31b1b.svg)](https://arxiv.org/abs/2610.16480)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2610.16480-blue.svg)](https://doi.org/10.48550/arXiv.2610.16480)

Bragglite predicts crystal structures from powder X-ray diffraction (PXRD)
and composition. PXRD-Bench provides benchmark data for PXRD-based structure
prediction.

## Installation

Use Python 3.12 and install the environment from the repository root:

```bash
uv sync --locked
```

Optional `torch.compile` requires a compiler and matching Python development
headers, such as `build-essential` and `python3.12-dev` on Ubuntu.

## PXRD-Bench setup

On Linux x86-64, download and prepare the dataset with:

```bash
bash data/setup_pxrd_bench.sh
```

The dataset is written to `data/pxrd_bench`. To verify an existing installation:

```bash
bash data/setup_pxrd_bench.sh --verify-only
```

See [data/README.md](data/README.md) for credentials, storage requirements,
resume behavior and command options.

## Training

Both tasks use `[q, I, q-width, prominence]`, 16 log-spaced Fourier frequencies
from 1 to 16, in-context conditioning with learnable positional embeddings,
and full PXRD augmentation during training only. Batch size is 512 per GPU;
gradient accumulation is 1. Element-set training uses MCFlow atom ordering
and sinusoidal positional encoding on atom tokens.

The training config selects the dataset and training budget: PXRD-Bench
uses 4,000,000 steps; MP-20 uses 800,000 steps. Select the appropriate config
with `--config-name` below. Data presets supply the default paths
`data/pxrd_bench` and `data/mp20`; override `data.data_root` only when your
files are elsewhere. Prepare the selected dataset before training.

MP-20 requires prepared token and PXRD caches under `data/mp20`. Element-set
training also needs MCFlow-ordered tokens and their matching PXRD caches;
see [the MP-20 input layout](data/mp20.md).

PXRD-Bench, known stoichiometry

```bash
.venv/bin/python -m src.pxrd.train_bragglite \
  --config-name=train_pxrd \
  run.output_dir=outputs/training/pxrd_bench_stoichiometry
```

MP-20, known stoichiometry

```bash
.venv/bin/python -m src.pxrd.train_bragglite \
  --config-name=train_pxrd_mp20 \
  run.output_dir=outputs/training/mp20_stoichiometry
```

PXRD-Bench, element set

```bash
.venv/bin/python -m src.pxrd.train_bragglite_pxrd_joint \
  --config-name=train_pxrd_element_set \
  run.output_dir=outputs/training/pxrd_bench_element_set
```

MP-20, element set

```bash
.venv/bin/python -m src.pxrd.train_bragglite_pxrd_joint \
  --config-name=train_pxrd_element_set_mp20 \
  run.output_dir=outputs/training/mp20_element_set
```

Local metrics are saved automatically. W&B is disabled by default; enable it
with `logging.no_wandb=false`. `logging.wandb_name=my-run` is optional.

For reproducibility checks, use `run.compile=false` during training and
`runtime.compile=false` during evaluation. Compiled results may differ even
with fixed seeds.

## Checkpoint download

> [!NOTE]
> **TBA.** Pretrained Bragglite and fine-tuned two-phase checkpoints will be
> released separately. This release contains code only.

## Evaluation

The same `data=...` preset supplies dataset name, loader, root and atom-count
limit. Evaluation defaults to validation (`iid_val` for PXRD-Bench, `val` for
MP-20), EMA weights, K=20, seed 777 and Match@1/20.

PXRD-Bench, known stoichiometry

```bash
.venv/bin/python -m src.pxrd.eval_bragglite \
  data=pxrd_bench \
  checkpoint.path=/path/to/checkpoint.pt \
  run.output_dir=outputs/evaluation/pxrd_bench_stoichiometry
```

MP-20, known stoichiometry

```bash
.venv/bin/python -m src.pxrd.eval_bragglite \
  data=mp20 \
  checkpoint.path=/path/to/checkpoint.pt \
  run.output_dir=outputs/evaluation/mp20_stoichiometry
```

PXRD-Bench, element set

```bash
.venv/bin/python -m src.pxrd.eval_bragglite_pxrd_element_set_ckpt \
  data=pxrd_bench \
  checkpoint.path=/path/to/checkpoint.pt \
  run.output_dir=outputs/evaluation/pxrd_bench_element_set
```

MP-20, element set

```bash
.venv/bin/python -m src.pxrd.eval_bragglite_pxrd_element_set_ckpt \
  data=mp20 \
  checkpoint.path=/path/to/checkpoint.pt \
  run.output_dir=outputs/evaluation/mp20_element_set
```

Select the model and sampler on validation, then evaluate test with both fixed:
add `data.split=iid_test` for PXRD-Bench or `data.split=test` for MP-20, using
a separate output directory.

Report `match_rate@k` using the strict StructureMatcher tolerances
`ltol=0.2`, `stol=0.3`, `angle_tol=5°`, without an additional validity gate.
Set `scoring.match_only=false` for additional diagnostics. For RRUFF/opXRD,
evaluate validation and test separately with seeds 777–781.

## Two-phase pipeline

XDecomposer supplies phase decomposition in the Bragglite two-phase pipeline.
Code and loaders live in `src.two_phase`; one ordered/shared-element workflow
is documented in [the two-phase guide](src/two_phase/README.md).
Provide the mixture artifacts under `data/two_phase`, MP-20 reference caches
and any required predicted-phase caches. XDecomposer also requires its official
V_0.0.1 source, weights and dependencies.

```bash
.venv/bin/python -m src.two_phase.finetune_xdecomposer --help
.venv/bin/python -m src.two_phase.finetune_bragglite --help
```

See [the two-phase guide](src/two_phase/README.md) for all entrypoints.

## License and attribution

Original Bragglite contributions are licensed under MIT; see [LICENSE.md](LICENSE.md).
The implementation builds on [Crystalite](https://github.com/joshrosie/crystalite).
Its copyright and permission notice are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Retained FlowMM-derived
helpers are licensed under CC BY-NC 4.0; their noncommercial restriction applies
when using those components. CDVAE-derived helpers retain their MIT notice.

PXRD-Bench contributors' own covered contributions use
[CC BY-NC 4.0](DATASET_LICENSE.md). Source data and third-party components retain
their original terms; see [source licenses and attribution](DATA_LICENSES.md).

### Citation

If you use PXRD-Bench or Bragglite, please cite our arXiv preprint:

```bibtex
@article{woo2026pxrdbench,
  title={PXRD-Bench: How Close Are We to Determining Crystal Structures from Powder Diffraction?},
  author={Woo, Dongyeop and Segal, Nofit and Kim, Nayoung and Kim, Minkyu and Gomez-Bombarelli, Rafael and Ahn, Sungsoo},
  journal={arXiv preprint arXiv:2610.16480},
  year={2026},
  doi={10.48550/arXiv.2610.16480},
  url={https://arxiv.org/abs/2610.16480}
}
```
