# Fully automated Kaggle execution

Use the latest **05_btp_kaggle_final.ipynb** from `dev/unified-ecg-pipeline`.
The notebook is a self-contained entry point; it synchronizes source from GitHub.
`04` and `ecg-xl-btp` remain development/attempt references, not execution prerequisites.

## One-time preparation before Run All

1. Import the latest final notebook into Kaggle. You do **not** need a Kaggle
   source dataset or a manually cloned checkout.
2. Attach extracted PTB-XL data containing `ptbxl_database.csv`,
   `scp_statements.csv` and `records100/` with matching `.hea`/`.dat` files.
   Attach compatible Mamba wheels if available.
3. Enable a CUDA GPU and Internet. GitHub synchronization needs Internet even
   when package wheels are attached. The reported successful hardware was T4.
4. In Kaggle Secrets, create **GITHUB_TOKEN** and grant this notebook access.
   Its GitHub permissions must allow reading T-arn21/BTP_project_files
   (e.g. Contents: Read for that repository). Never paste the value into code,
   URLs, notebook settings, output or logs.
5. Configure the **first cell only**, then choose Run All / Save & Run All.
   No manual execution/restarts/edits/input are needed between stages.

## Initial settings

```python
RUN_SUPERCLASS = True
RUN_SUBCLASS = True
RUN_SMOKE = True
RUN_METADATA_AUDIT = True
FULL_EPOCHS = 10
FULL_BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 15
DIAGNOSTIC_MODE = False
NUM_WORKERS = 0
NORMALIZATION_MODE = "sampled"
NORMALIZATION_SAMPLES = 1000
```

The first cell also exposes per-task epoch overrides, optimizer parameters,
normalization seed, smoke budget, source-build/wheel options, filters,
augmentation and resume paths. Use task flags to skip one model. The reference
384-wide/eight-block architecture is preserved. Batch 8 is an explicit memory
adjustment from reference batch 64, not a reduction of the training dataset.

Default source checkout is `/kaggle/working/BTP_project_files`, branch
`dev/unified-ecg-pipeline`. OUTPUT_ROOT is `/kaggle/working/btp_outputs`; the
shared cache is OUTPUT_ROOT/cache. Dataset root is auto-detected under Kaggle
inputs; set DATASET_ROOT only if multiple candidates need disambiguation.

Set DATASET_VERSION honestly. Default **1.0.1** matches the user's reported
successful smoke input; use **1.0.3** only when that release is actually attached.
Root, metadata hashes and operator-declared release are recorded; CSV schema
alone does not certify version. Do not pool different releases as one benchmark.

## What happens automatically

1. Retrieve GITHUB_TOKEN using `kaggle_secrets.UserSecretsClient`.
2. Clone the private repository or fetch and fast-forward the requested branch.
   Dirty tracked/untracked files, a wrong origin, or local-only/divergent commits
   cause a refusal. No reset/stash/clean or forced local branch overwrite occurs.
   The token is never in a URL, argv, Git config, helper contents or logs.
   Only Git clone/fetch receive it in temporary child environments.
3. Verify source imports resolve to the synchronized checkout, not old inputs.
   Start a clean standard-library controller process for dependency setup.
4. Install missing base dependencies without installing/replacing Torch.
   Keep already-working CUDA packages; try attached compatible wheels first;
   use source builds only if ALLOW_SOURCE_BUILD=True. Build with the successful
   KEEP_CUDA=True, MAX_JOBS=2 and --no-build-isolation/-v recipe. Save wheels.
5. Verify actual Mamba CUDA forward/backward, then launch a **fresh scientific
   worker**. Notebook-kernel caches are irrelevant; no kernel restart is assumed.
6. Auto-discover PTB-XL and audit both label sets and patient separation.
   Five superclasses and all 23 mapped subclasses retain multi-label diagnoses
   and rare columns. Folds remain train 1–8, validation 9, test 10.
7. Run a small real-data full-architecture GPU/AMP smoke **before full caching**.
   It uses a tiny training-only fit and train/validation batches; no test access
   or benchmark metrics. Smoke weights are discarded.
8. Build/validate a shared deterministic float32 waveform memory map once,
   fit/cache training-only normalization and reuse it across tasks and epochs.
9. Train superclass Mamba, evaluate/export it, release model memory, then do
   the same for subclass Mamba. Full modes use all training records and
   validation-only model/threshold selection, followed by held-out test metrics.
10. Verify exports and persist/print a concise summary of passed checks,
    completed/skipped/failed experiments and result directories.

A token/clone/setup/GPU/smoke failure stops clearly with a saved failure summary.
Task-level training failures preserve their run logs and completed artifacts;
by default the other task is attempted, then the final status is failed.
No missing-variable dependency on later cells is needed to save failure evidence.

## Progress, cache and interruptions

The notebook launcher, stage logger and installer emit progress/heartbeats at
most 30 seconds apart while waiting. Logs identify signature scans, cache record
counts, normalization, first-batch wait, GPU forward/backward, epochs and checkpoint
writes. Data-wait timing is recorded in epoch history. Initial cache creation
still pays real I/O once; measured Kaggle throughput remains unverified.

The cache stores filtered, center-padded **unnormalized/unaugmented** float32
signals (~1.05 GB at 100 Hz). Samples normalize and augment only during training;
WFDB/filtering is not repeated each epoch. Default normalization samples up to
1000 official-training IDs deterministically; full mode computes exact training
statistics. Both tasks share the same identity/configuration/IDs/means/stds.

Cache keys include dataset metadata/root/raw-file stats, preprocessing code,
settings and numerical package versions. Shape/size/SHA256 validation protects
completed arrays; invalid/incomplete arrays rebuild. Use a new cache directory
for timestamp-preserving raw source changes. Do not run concurrent cache writers.

Completed epoch checkpoints, history and JSON summaries are atomically replaced;
an interrupted write leaves the previous complete file. Each passed check/task
updates final_summary.json immediately. Retain saved Kaggle notebook outputs:
Kaggle may discard working files when a session is destroyed. Interrupted cache
builds are validated/rebuilt rather than trusted as complete training data.

To resume, restore prior outputs to writable working storage and set
RESUME_RUN_DIRS in the first cell. Keep last.pth, best.pth and the original
config/manifests together. Match epoch budget, release, data/cache paths, model,
seed and preprocessing. Resume restores optimizer/scheduler/RNG state and rejects
incompatible settings. CPU zero-worker continuation is tested; fused CUDA kernels
and multi-worker restarts need not be bitwise reproducible.

## Diagnostic versus benchmark runs

For a short end-to-end plumbing check, set DIAGNOSTIC_MODE=True,
DIAGNOSTIC_BATCHES=2 and FULL_EPOCHS=1 before starting. Reduce
NORMALIZATION_SAMPLES explicitly if an even smaller diagnostic is desired.
These runs cap train/validation records, never evaluate fold 10, and have
benchmark_result=False in separate diagnostic directories. Full benchmarks never
silently truncate the dataset. Early stopping within the configured full budget
is expected completion; failed or batch-limited runs are not complete benchmarks.

## Dependency compatibility and wheel preservation

User-reported working setup: Python 3.13.15, Torch 2.11.0+cu128, toolkit 12.8,
Tesla T4, ABI=True; causal-conv1d 1.7.0, mamba-ssm 2.3.2.post1, WFDB 4.3.1.
These are the known recipe, not a guarantee for every Kaggle image. Wheels must
match Python ABI, OS/architecture, Torch minor, CUDA and CXX11 ABI. Fresh sessions
need packages installed again. Save session mamba_wheels/ as a private Kaggle
input for future runs. Source builds need nvcc/compiler, disk/RAM and Internet;
their full logs and waiting heartbeats are preserved. Import success alone is
not considered working CUDA. A failed compatibility probe stops execution.

## Output locations and summary

```text
/kaggle/working/btp_outputs/
  cache/                              # shared reusable waveform/statistics cache
  sessions/<timestamp_id>/
    execution_config.json             # explicitly whitelisted nonsecret controls
    launcher.log, pipeline.log, worker.log, pipeline_status.json
    final_summary.json                # checks + experiment status + completed paths/errors
    environment.json, session_config.json, setup_report.json
    metadata_audit.json, normalization.json, *_smoke.json, *_class_statistics.csv
    setup_logs/, mamba_wheels/
    experiments/benchmark/<task_run>/
    experiments/diagnostic/<task_run>/
```

Task folders have atomic best/last checkpoints, config/environment/data manifests,
class counts, history, loss/confusion figures, run.log/status.json and summaries.
Benchmark folders additionally have validation thresholds and validation/test/
fixed-.5 metrics and prediction CSVs. Diagnostic outputs cannot be labeled test
benchmarks. The notebook prints the concise execution summary even on bootstrap
or setup failure; completed checks/tasks are not lost when another stage fails.

## Validation limits

Local tests exercise Git safety with disposable local repositories and fake
secrets, worker/import isolation, summaries, atomic writes and CPU synthetic
training/evaluation. They do not authenticate to Kaggle or execute real Mamba
GPU training. Actual secret access, GitHub clone inside Kaggle, pinned CUDA build,
real cache throughput/VRAM/AMP and complete superclass/subclass benchmarks must
still be executed there. No new GPU classification results are claimed here.
