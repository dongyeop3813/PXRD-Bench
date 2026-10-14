# Two-phase training and recovery

The released pipeline has one setting: ordered XDecomposer (dominant, minor),
followed by one Bragglite model conditioned on the same mixture element superset
for both phases and a supplied atom count for each phase. No reference-based
slot alignment or phase-specific element sets are used at inference.

## Inputs

Provide official XDecomposer V_0.0.1 source at commit
`53c33dc41df53d26c3fc011ec270e34acc0d01c7`, its release/MAE weights and its
separate dependencies. Source and weights retain their authors' licenses.
Provide these frozen data assets; the repository does not generate their IDs
or splits:

| Location | Format and use |
| --- | --- |
| `data/two_phase/mixtures/{train,val}` | `samples.parquet`, `metadata.json`, `shards/*.pt`; mixture shard schema **3**, used by ordered XDecomposer. |
| `data/two_phase/shared_phase/{train,val}.pt` | `oracle_phase_shared_cache_v1`: rows, `pxrd_peak128x4` `[N,512]`, shared boolean `elem_set_mask` `[N,94]`, integer `oracle_num_atoms` `[N]`; used by Bragglite fine-tuning. |
| `data/mp20` | Frozen reference tokens matching each phase ID, used for fine-tuning targets and recovery scoring. |

Bragglite is fine-tuned on **oracle single-phase profiles**, then evaluated on
XDecomposer's predicted profiles. Mixture data, training phase caches and
predicted slot caches describe different stages; they are not alternative
experiment settings. Only the paper's input formats are accepted. Each phase
row identifies its parent mixture, phase ID and dominant/minor role. Both rows
of a parent must carry the same mixture element union.

## Configure once

```bash
cp configs/two_phase/pipeline.example.json two_phase.json
```

Replace local paths and SHA256 values in `two_phase.json`. The example records
the selected research checkpoint hashes; after new training, set each produced
checkpoint's own path/hash. Keep the fixed sampler and matcher settings.
`training.initial_checkpoint` is the recorded ordered XDecomposer continuation;
set it to `null` to start from the official release weights instead. This changes
the initialization and does not reproduce that continuation.

## 1. Fine-tune ordered XDecomposer

```bash
.venv/bin/python -m src.two_phase.finetune_xdecomposer \
  --protocol two_phase.json \
  --output outputs/two_phase/xdecomposer/best.pt
```

The config supplies seed 42, batch 16, validation batch 32, LR 3e-6, up to
25 epochs and patience 8. Additive mask conservation and correlation loss
(weight 2) preserve ordered phase prediction. Validation selects ordered
Pearson success at 0.5, then mean correlation. W&B is off unless `--wandb`
is provided. Use a new output directory for each run.

## 2. Fine-tune Bragglite with shared element supersets

```bash
.venv/bin/python -m src.two_phase.finetune_bragglite \
  pxrd.transfer_checkpoint=/path/to/pretrained_element_set.pt \
  run.output_dir=outputs/two_phase/bragglite
```

The default data path is `data/two_phase/shared_phase`. The paper checkpoint
recipe is 768 width / 24 layers / 24 heads, per-layer GEM, batch 128, 50,000
steps, LR 1e-5 with warmup 500 and cosine decay, seed 123. Its atom-token PE
is sinusoidal. The frozen phase cache/reference order is retained; MCFlow is
not reapplied during this fine-tuning. This recorded two-phase architecture
has **context learnable PE disabled** and no PXRD augmentation. The single-phase
training defaults have context learnable PE enabled and MCFlow ordering for
element-set training. Use a pretrained checkpoint compatible with the two-phase
architecture. `torch.compile` is enabled here; use `run.compile=false` for
reproducibility checks.

Update `decomposition_checkpoint` and `structure_checkpoint` in `two_phase.json`
with your selected checkpoints before recovery.

## 3. Predict ordered profiles and recover structures

```bash
.venv/bin/python -m src.two_phase.predictions --protocol two_phase.json
.venv/bin/python -m src.two_phase.recover \
  --protocol two_phase.json \
  --output-root /absolute/path/to/two_phase_recovery
```

The first command creates an ordered prediction cache at `output_root` from the
config. The second writes structure candidates and metrics to a separate,
empty output directory. EMA, K=20, Heun150, seed 777 and strict matcher-only
scoring are fixed. Slot order stays dominant/minor; both slots receive the same
mixture element mask. Oracle per-phase atom counts are supplied by the benchmark
references. The primary metric is `ordered_all_phase_match_rate@20`.

These commands reproduce the frozen synthetic **validation** workflow, excluding
50/50 ties. They are not a general inference CLI for arbitrary measured mixtures
or unknown phase atom counts. Experimental-profile loaders remain available in
`src.data.experimental_pxrd` and `src.two_phase.opxrd_multiphase`. Official
XDecomposer assets, frozen datasets and final trained checkpoints must be supplied
separately. Compilation does not guarantee bit-identical results.
