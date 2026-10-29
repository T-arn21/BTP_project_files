# SSL-pretrained Mamba classification fine-tuning

Current notebook defaults now select [explicit SSL-cache-backed GPU smoke tests](ssl_cache_backed_finetuning.md).
This guide describes the separate independent audit/migration workflow.

This path initializes the existing 5-superclass or 23-subclass Mamba classifier
from the completed 20-epoch reconstruction backbone. Both tasks use the existing
patient-disjoint folds, labels, waveform preprocessing, normalization, weighted
BCE trainer, validation-only threshold selection and classification metrics.
All three fine-tuning flags default **False**. No expensive training is launched
by implementing this feature or by a smoke-only configuration.

## Required source files and pinned source

Attach the user's [checkpoint Kaggle dataset](https://www.kaggle.com/datasets/arnavtiku/checkpoint),
plus the same PTB-XL dataset and compatible CUDA wheels as needed.
Keep these **three files together** in the pretraining export directory:

```text
encoder.pth
config.json
summary.json
```

The pinned encoder path is already in the first notebook cell:

```text
/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/sessions/20261028T191718Z_e9445a9d/ssl/pretraining/ssl_mamba_20261028T201304Z_e1341eb8/encoder.pth
```

Expected SHA256 supplied by the user:

```text
8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4
```

Do not select the encoder in `ssl/smoke/` (two ECGs). The classifier checkpoints
`best.pth`/`last.pth`, reconstruction `last.pth` and wheels are not replacements
for this encoder export. Pretraining optimizer state is not needed or loaded.
If present, `history.csv`, `status.json` and `split_provenance.json` are also
validated and must agree; they are optional duplicates/completion evidence.
The authenticated encoder itself contains the required source split provenance.

Initial remote access returned HTTP 403. Subsequently supplied local artifacts
were audited read-only: the epoch-20 encoder SHA256, finite weights and completion
metadata are verified, and their hashes match the failed run's recorded source
verification. See [the artifact audit](ssl_provenance_artifact_audit.md) for the
raw-file-stat rejection; live provenance checks remain mandatory. This is saved
execution evidence, not 20 epochs rerun locally.

## Transfer policy

| State | Fine-tuning initialization |
| --- | --- |
| input_proj, all Mamba blocks, final_norm | Copy verified SSL backbone tensors |
| attn_pool | Fresh classifier initialization; ignore SSL's unused pool weights |
| head | Fresh 5-output or 23-output classifier |
| SSL reconstruction head | Never loaded |
| optimizer, scheduler, scaler, RNG, early stopping | New supervised experiment |

All parameters, including the backbone, remain trainable. No architecture,
hidden width, layer count or supervised checkpoint key is changed. Initialization
is seeded separately for each task/run using the configured seed. Smoke updates
are discarded: each full task starts again from the pinned SSL backbone with a
fresh pool/head. This is fine-tuning, not continuation of a supervised baseline.

## Strict validation before updates

1. Require a 64-character expected SHA256. Compare it to file bytes **before**
   `torch.load(weights_only=True)`; never automatically trust a newly computed hash.
2. Require the encoder export format, epoch exactly 20, `smoke_only=False`,
   `pooling_pretrained=False`, matching config/model/data fields, and a summary
   confirming completion of 20 full epochs over all eligible training ECGs.
   Optional sidecars must agree; inconsistent or partial history is rejected.
3. Check every encoder tensor is finite. Require the preserved model config,
   exact expected encoder keys, shapes and dtypes, and trained-parameter names
   covering exactly the transferred backbone. Build a full state dictionary and
   load it strictly; no unverified missing-key load is used.
4. Audit live official folds/patients. The SSL eligible and used IDs must equal
   all live fold-1-8 IDs before diagnostic-label exclusions. Its excluded fold-9
   and fold-10 IDs and training patient IDs must match exactly. SSL provenance
   must confirm no diagnostic labels or held-out training waveforms.
5. Recompute the **training-only** raw waveform signature using the same helper
   as SSL and compare its entire preprocessing identity: metadata/SCP hashes,
   root, release, source signatures, code hash and NumPy/SciPy/WFDB versions.
   The ordinary supervised fingerprint covers all folds, so directly comparing
   it to the training-only SSL fingerprint would compare different populations.
6. Require exact normalization statistics, fit IDs and fit configuration. Training
   augmentation, label exclusion and cache/worker storage controls intentionally
   differ between label-free SSL and classification; waveform semantics cannot.

No path/version/statistics rebinding or provenance bypass occurs. If a different
runtime/data attachment changes a checked identity, stop and supply a genuinely
matching environment/input. Never edit the source checkpoint/manifest to force
acceptance. Missing files, hash mismatch and incompatible provenance fail clearly;
there is no fallback to fresh weights, a smoke encoder or supervised checkpoints.

Field-by-field expected/actual values now persist in ssl_provenance_comparison.json,
including after failed initialization. A path spelling can differ only if both
existing names are proven to be the same directory via filesystem identity and
every other field is exact. Missing mounts/different copies are still rejected.
See [provenance diagnosis and zero-step audit mode](ssl_provenance_diagnostics.md)
for the reported failed session and the safe next-run switches.

## Later run after the CPU audit passes: both GPU smokes only

Import the latest `05_btp_kaggle_final.ipynb`, enable GPU and Internet, attach
the checkpoint and matching PTB-XL inputs, and enable the existing GITHUB_TOKEN
secret. Edit the existing first-cell assignments/dictionary before its final
SETTINGS line, then use **Run All / Save & Run All**:

```python
# Disable other workflows for this fine-tuning-only run:
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False

SSL_FINETUNE_PROVENANCE_ONLY = False  # explicitly leave the current CPU audit mode
RUN_SSL_FINETUNE_SMOKE = True
RUN_SSL_FINETUNE_SUPERCLASS = False
RUN_SSL_FINETUNE_SUBCLASS = False

# Already pinned in the supplied first cell:
SSL_FINETUNE_ENCODER_PATH = "/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/sessions/20261028T191718Z_e9445a9d/ssl/pretraining/ssl_mamba_20261028T201304Z_e1341eb8/encoder.pth"
SSL_FINETUNE_ENCODER_SHA256 = "8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4"

SSL_FINETUNE_SETTINGS = dict(
    epochs=20, batch_size=2, learning_rate=1e-4, mixed_precision=False,
    numerical_stability_mode="conservative", diagnostic_batches=2,
    log_every_batches=1, weight_decay=5e-4, warmup_epochs=3, min_lr=1e-6,
    label_smoothing=0.05, use_pos_weight=True, pos_weight_max=8.0,
    grad_clip_norm=1.0, patience=15,
)
```

Keep MODEL_SETTINGS equal to the source encoder configuration. Keep the matching
PTB-XL release/root, normalization settings, filter and seed in the shared data
controls. Supervised training augmentation follows AUGMENT; validation never
augments. The original source used the pipeline's training-only statistics;
reading/fitting them may take longer than the few smoke optimizer steps.

The two smokes each force one epoch, four training records and four validation
records (batch 2 x two diagnostic batches), without test evaluation or threshold
tuning. They use the actual preserved Mamba on CUDA. BCE is FP32 outside autocast;
AMP is disabled. Each backward checks finite **and nonzero** backbone,
attention-pool and classifier gradient groups. Standard finite input/target/logit/
loss/gradient/state guards and nonfinite-rejecting clipping remain enabled.
Neither the `epochs=20` setting nor global FULL_EPOCHS starts full fine-tuning
while both full-task flags are False.

## Later full fine-tuning

Only after reviewing both real GPU smokes, use a separate explicitly configured
Run All (same source pin, AMP off and the other workflow flags still off):

```python
SSL_FINETUNE_PROVENANCE_ONLY = False  # explicitly leave the current CPU audit mode
RUN_SSL_FINETUNE_SMOKE = True
RUN_SSL_FINETUNE_SUPERCLASS = True  # full superclass experiment
RUN_SSL_FINETUNE_SUBCLASS = True    # full subclass experiment; independently selectable
SSL_FINETUNE_SETTINGS["epochs"] = 20  # explicitly choose full budget (or 1 first)
```

Both same-execution smokes must pass before either full task starts. Full tasks
are separate, sequential new experiments with fresh pooling/heads/optimizers.
Either task may be disabled to run the other alone; both smokes still run first.
Global DIAGNOSTIC_MODE does not truncate full fine-tuning: these flags explicitly
select full data with all labeled training records and existing early stopping.
Global FULL_EPOCHS/MIXED_PRECISION/OPTIMIZER_SETTINGS do not override the separate
fine-tuning dictionary. A failure stops this workflow and records remaining full
tasks as blocked/not run; no automatic retry, precision change or batch skipping.

Full runs retain original folds 1-8/9/10, five/all-23 target columns, validation
model selection/threshold tuning, and held-out test AUROC/AP/F1/precision/recall
and confusion counts. Test data never selects the model/thresholds. Neither
source SSL nor baseline classification artifacts are modified.

## Output paths and evidence

```text
/kaggle/working/btp_outputs/sessions/<NEW_session_id>/
  ssl_finetune_source.json
  final_summary.json
  ssl_finetuning/
    smoke/diagnostic/mamba_<superclass|subclass>_<NEW_run_id>/
    full/benchmark/mamba_<superclass|subclass>_<NEW_run_id>/
```

Each task folder includes `ssl_initialization.json` with source SHA/epoch,
source sidecar hashes, verified SSL provenance, transferred/fresh parameter
names, class order and initialization seed. `config.json` and both classifier
checkpoints carry the source encoder path/hash. Other files reuse the existing
`data_manifest.json`, class statistics, numerical settings, best/last checkpoints,
history, plots and status/logs. Smokes additionally write
`ssl_smoke_gradients.json` and diagnostic validation metrics/predictions.
Full runs write validation/test/test-at-.5 metrics/prediction CSVs and thresholds.
Failures preserve `checkpoint_failure.json` or numerical diagnostics as applicable.

Source inputs are read-only; outputs always use new run folders. Baseline and
reconstruction directories are preserved. This feature does not automatically
resume an old supervised experiment or inject the SSL weights into baseline runs.

Local tests use clearly identified synthetic checkpoint evidence and CPU model
doubles to verify strict gates, fresh pools/heads, both labels/tasks, export
metrics and workflow ordering. Conditional real Mamba GPU smoke tests remain
available. The original encoder's integrity is now verified from local artifacts;
live Kaggle provenance and both real GPU smokes still must pass before full runs.
No fine-tuning scores are claimed here.

See [full training content equivalence](ssl_content_identity.md) for the opt-in,
checkpoint-authenticated full-population proof required for a sole stat-signature mismatch.

[Normalizer diagnosis and explicit authenticated source reuse](ssl_normalizer_migration.md)
documents independent full content reporting when mean/std also mismatch, current
CPU-audit settings and the additional exact reproducibility/authorization checks.
