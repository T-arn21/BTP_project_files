# Controlled FP32 continuation of the preserved Kaggle run

No full training has been launched. This prepares a short diagnostic Run All.
Original files under `C:\Users\Arnav Tiku\Downloads\results\btp_outputs` are
read-only inputs to the local verification; they were not edited or rewritten.

## Compatibility findings

Session: `20261028T063245Z_f710283f`, source commit `5e10109`.

| Check | Superclass | Subclass |
| --- | --- | --- |
| last.pth saved epoch | 3 | 5 |
| Checkpoint config equals adjacent config.json | Passed | Passed |
| Checkpoint metadata equals corresponding fields of data_manifest.json | Passed | Passed |
| Model/optimizer tensors finite | Passed | Passed |
| AdamW state restores on CPU; moment shapes match all 188 parameters | Passed | Passed |
| Scheduler/scaler state roundtrip on CPU | Passed | Passed |
| Saved scaler scale | 32768 | 131072 |
| Saved optimizer LR | 0.0003 | 0.00024371172537788062 |
| Legacy config accepted with original AMP/settings | Passed | Passed |
| Strict resume with AMP disabled | **Rejected** | **Rejected** |

The Mamba architecture source is unchanged from the failed run. The Git LF
preprocessing source hash matches the saved Kaggle identity. Saved metadata,
normalization and cache agree with the checkpoint provenance; prior full-cache
inspection verified checksum and finite values. No original PTB-XL raw files
or live Kaggle runtime were available locally. Actual Mamba strict state_dict
loading and GPU execution still require Kaggle. CPU state restoration does not
prove CUDA compatibility or that a later epoch will stay finite.

An exact resume would require the original AMP=True, LR=3e-4, batch=8,
epochs=10 and matching data/model/provenance settings. It would restore the
optimizer, scheduler, scaler and RNG, starting at epochs 4 and 6. Turning AMP
off changes config; the strict gate correctly refuses it. Do not edit checkpoint
configs or remove provenance checks to get around that gate.

## Selected procedure: a new weights-only run

Use the original epoch-3/epoch-5 **last.pth** weights in NEW experiments with
FP32 forward/BCE, conservative LR, and fresh AdamW/scheduler/scaler/RNG/early
stopping state. This intentionally resets optimization; it is not exact epoch
continuation, fresh training, or a reproduction of the original AMP run.
New epoch 1 follows weights learned through original epoch 3 or 5; budgets and
metrics must be described that way. Original artifacts are never used as the
new output directories.

The weights-only path now requires:

- Same model configuration, task, class order, seed and data settings, except
  cache directory/worker count (storage/loader execution only).
- Exact saved normalization, metadata hashes, dataset root/release, raw-file
  signatures, preprocessing implementation/version identity and fit record IDs.
- Exact full split IDs for a full-data run. Diagnostics require the exact
  declared train/validation prefixes of the source splits; no test subset.
- Finite model weights and strict parameter key/shape loading on the real model.

Diagnostic cache containers have fewer records and different payload hashes;
their source identity and normalization must still match, and the data loader
validates the new cache independently. Full-data cache payloads must match
when both runs use caching. No release/path/package-version rebinding occurs.
If any comparison fails, execution stops with `checkpoint_failure.json` and
`status.json`; there is no fallback to random initialization or relaxed checks.

## Prepare Kaggle inputs before starting

1. Import the latest `05_btp_kaggle_final.ipynb` from this branch.
2. Attach the same PTB-XL 1.0.1 dataset. Its recorded dataset root is:

   `/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`

3. Upload/attach the preserved outputs as a private Kaggle dataset, keeping
   session/experiment folder names. Only the two checkpoints are required as
   weight inputs because they embed config/metadata. Retain their adjacent
   manifests/history for inspection. Compatible saved wheels can be attached too.
4. Enable GPU, Internet and the `GITHUB_TOKEN` secret as before.

The notebook automatically finds this unique session below `/kaggle/input`:

```text
.../20261028T063245Z_f710283f/experiments/benchmark/
  mamba_superclass_20261028T080512Z_13735164/last.pth
  mamba_subclass_20261028T083047Z_9f335dd9/last.pth
```

It pins their audited SHA256 values:

```text
superclass f9b28d43d3373c12075be6bcfa3fc769dbe511477c3e1de2c8a7511d18d81312
subclass   f6f7d5c9278415f070257c1cfb8551993a9aebb5f1ed075ac4749f2c4e03b54c
```

Set `CONTINUATION_ARTIFACT_ROOT` to the parent containing that session if
multiple copies are attached. Missing/ambiguous sessions or hash mismatches
abort. Source inputs remain read-only. Exact mount names for the uploaded
outputs need not be guessed; the fixed session/experiment names identify them.

## Exact first-cell settings for the next run

These are the notebook's prepared defaults:

```python
CONTINUATION_SESSION_ID = "20261028T063245Z_f710283f"
CONTINUATION_ARTIFACT_ROOT = ""  # auto-discovery under /kaggle/input
# Keep CONTINUATION_CHECKPOINT_SHA256 as supplied above.
INITIAL_CHECKPOINTS = {"superclass": "", "subclass": ""}
RESUME_RUN_DIRS = {"superclass": "", "subclass": ""}
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = True
NUMERICAL_STABILITY_MODE = "conservative"
MIXED_PRECISION = False
FULL_EPOCHS = 1
SUPERCLASS_EPOCHS = SUBCLASS_EPOCHS = None
FULL_BATCH_SIZE = SMOKE_BATCH_SIZE = 2
DIAGNOSTIC_MODE = True
DIAGNOSTIC_BATCHES = 2
LOG_EVERY_BATCHES = 1
NUM_WORKERS = 0
DATASET_VERSION = "1.0.1"
NORMALIZATION_MODE = "sampled"
NORMALIZATION_SAMPLES = 1000
NORMALIZATION_SEED = SEED = 42
APPLY_FILTER = AUGMENT = CACHE_SIGNALS = True
OUTPUT_ROOT = "/kaggle/working/btp_outputs"
CACHE_DIR = "/kaggle/working/btp_outputs/cache"
OPTIMIZER_SETTINGS = dict(learning_rate=1e-4, weight_decay=5e-4,
    warmup_epochs=3, min_lr=1e-6, label_smoothing=0.05,
    use_pos_weight=True, pos_weight_max=8.0, grad_clip_norm=1.0)
```

For one epoch the existing scheduler clamps effective warmup to zero. The
first optimizer step uses LR 1e-4. No forward autocast or loss scaling is enabled.
Architecture remains 384-wide, eight bidirectional blocks, d_state=16,
d_conv=4, expand=2, dropout=.45, mean_attention pooling. Both tasks train on
four records and validate on four as an explicitly labeled diagnostic, not a
benchmark. The independent tiny smoke uses freshly initialized models; the
diagnostic training itself loads and tests the selected checkpoint weights.

The prior preprocessing identity includes NumPy 2.1.3, SciPy 1.16.3 and WFDB
4.3.1. Its Mamba environment was Torch 2.11.0+cu128 / Python 3.13.15 / T4,
Mamba 2.3.2.post1 and causal-conv1d 1.7.0. The installer keeps working packages
or tries matching attached wheels, then verifies CUDA backward. A changed
preprocessing identity fails continuation even if installation succeeds. Do
not erase version/path/hash differences; use a verified matching environment
or explicitly choose fresh training instead. The notebook never restarts itself.

## Outputs and decision after the diagnostic

New files go only under:

```text
/kaggle/working/btp_outputs/sessions/<NEW_session_id>/
  final_summary.json
  experiments/diagnostic/mamba_superclass_<NEW_run_id>/
  experiments/diagnostic/mamba_subclass_<NEW_run_id>/
```

Each new task directory records `initial_checkpoint.json` (source epoch,
SHA256, provenance check, changed settings and reset states), config,
`numerical_settings.json`, history, new best/last checkpoints, diagnostic
validation predictions/metrics and figures. On NaN/Inf it saves
`numerical_failure.json` and `diagnostics/epoch_*_<phase>.json` with ECG IDs,
epoch/batch, tensor finite flags/ranges, gradients, LR, scaler and parameter
state. Inputs, labels, logits, loss, gradients and updates are guarded;
nonfinite clipping rejects the batch. Nothing is skipped or silently retried.

Only after reviewing both diagnostics should a separate run be configured
with `DIAGNOSTIC_MODE=False`, one full-data epoch, and the same conservative
settings. Keep the original full-run checkpoints as the source; do not promote
a diagnostic-prefix checkpoint into a full-split run. A later full budget is
another explicit choice, not part of the next prepared Run All.

If provenance cannot be matched, deliberate fresh training requires clearing
CONTINUATION_SESSION_ID and both checkpoint mappings in the first cell before
Run All. Keep diagnostic mode and FP32 settings for that first fresh check.
Do not mislabel this alternative as continuation of the old experiments.
