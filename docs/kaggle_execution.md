# Run 05_btp_kaggle_final.ipynb end to end

This is the primary execution notebook. `04_btp_experiments.ipynb` is kept as a
development reference. `ecg-xl-btp.ipynb` preserves the user's Kaggle attempts
and temporary fixes. Those fixes are now in modules; **do not patch source or
reload modules manually** in the final notebook. Start from a fresh session.

## 1. Prepare attached inputs

1. Download the latest **dev/unified-ecg-pipeline** branch source from GitHub,
   extract it, and create/upload a private Kaggle source dataset. Include
   `src/`, both requirements files and `05_btp_kaggle_final.ipynb`. Exclude
   `.git/`, `outputs/`, datasets, local caches, credentials and virtualenvs.
2. Attach an extracted PTB-XL dataset. Its root must contain
   `ptbxl_database.csv`, `scp_statements.csv`, and `records100/` with matching
   `.hea` and `.dat` files. Nested input directories are supported.
3. Attach compatible Mamba wheels if available. Both causal-conv1d and
   mamba-ssm wheels are needed, plus missing runtime dependencies offline.
4. Select a GPU (the reported working hardware was Tesla T4), and enable
   Internet if source build or base package installation will need downloads.
5. Import/open `05_btp_kaggle_final.ipynb` from the repository.

## 2. Set only the initial configuration cell

The default controls are:

```python
RUN_SUPERCLASS = True
RUN_SUBCLASS = True
RUN_SMOKE = True
RUN_METADATA_AUDIT = True
FULL_EPOCHS = 10
FULL_BATCH_SIZE = 8
NUM_WORKERS = 0
NORMALIZATION_MODE = "sampled"
NORMALIZATION_SAMPLES = 1000
NORMALIZATION_SEED = 42
CACHE_SIGNALS = True
DIAGNOSTIC_MODE = False
```

Set `SUPERCLASS_EPOCHS`/`SUBCLASS_EPOCHS` to override the shared epoch budget,
otherwise leave None. Set a task flag False to skip that full experiment.
`RUN_SMOKE` runs both heads' one-step smoke; metadata integrity checks remain
mandatory in the loader even when the extra metadata-report flag is False.
Full experiments initialize fresh models and run **sequentially**.

Set `SOURCE_ROOT`/`DATASET_ROOT` if discovery has zero/multiple candidates.
They may instead come from BTP_REPO_DIR/PTBXL_DATA_DIR. Use exact extracted
roots, not ZIP paths. Example only (replace with your actual inputs):

```python
SOURCE_ROOT = "/kaggle/input/datasets/<you>/btp-source/BTP_project_files"
DATASET_ROOT = "/kaggle/input/datasets/<owner>/ptb-xl/ptb-xl-1.0.1"
DATASET_VERSION = "1.0.1"
```

The default declared version is **1.0.1**, matching the user's successful smoke
input. Change it to **1.0.3 only if that is the actual attached release**.
The record is explicitly operator-declared, not independently certified from
CSV schema. Root and metadata hashes are saved. Release differences affect
cohort/splits/counts; compare models using the same release and preprocessing.
Do not combine 1.0.1 and 1.0.3 results as the same benchmark.

## 3. Choose smoke, fast diagnostic or full benchmark

- **Smoke only:** set both full task flags False, keep RUN_SMOKE=True. It still
  prepares the shared cache/statistics; each head gets one train step and one
  validation forward, without test evaluation or benchmark metrics.
- **Fast diagnostic:** set DIAGNOSTIC_MODE=True and DIAGNOSTIC_BATCHES=2,
  FULL_EPOCHS=1. Enabled tasks use at most two train/validation batches, never
  test, and write `diagnostic/` artifacts with benchmark_result=False. This
  quickly tests first-batch, optimization, checkpoint and evaluation/export
  plumbing. Normalization still samples up to 1000 training ECGs (reduce
  NORMALIZATION_SAMPLES explicitly for an even smaller diagnostic).
- **Full benchmark:** DIAGNOSTIC_MODE=False, desired task flags True. All
  labeled training records are used, including the final partial batch. The
  small normalization fit is a declared preprocessing strategy, **not** a
  hidden reduction of the training dataset. Exact statistics are available
  with NORMALIZATION_MODE="full". A full one-epoch run is not a smoke test.

Then choose Run All / Save & Run All. No intermediate input or changes are
required. Default full runs train both tasks up to 10 epochs with early stopping.
Use smaller batches on OOM; architecture is not silently reduced.

## 4. Dependency lifecycle

The reported successful build used Python 3.13.15, Torch 2.11.0+cu128,
CUDA toolkit 12.8, Tesla T4, CXX11 ABI=True, causal-conv1d==1.7.0,
mamba-ssm==2.3.2.post1 and wfdb==4.3.1. These versions are preserved as the
known working recipe, not a guarantee for every Kaggle image.

The setup is standard-library-only, before scientific imports:

1. Install missing base requirements if INSTALL_BASE=True. Torch is not in
   `requirements-kaggle.txt` and is never explicitly installed/replaced.
2. Probe an already installed Mamba using actual CUDA forward/backward in a
   fresh Python process; skip CUDA-package installation if it works.
3. If WHEEL_DIR is given, use it. Otherwise look for attached mamba_ssm and
   causal_conv1d wheels under `/kaggle/input`. Pip must find compatible tags;
   packages install with --no-deps so they cannot replace Torch implicitly.
4. If no working wheels, source build is allowed only when the initial
   ALLOW_SOURCE_BUILD=True (default explicit fallback). Both pinned packages
   build with `MAMBA_KEEP_CUDA_BUILD=TRUE`, `MAX_JOBS=2`, --no-build-isolation
   and -v. `pip wheel` preserves wheel artifacts before installing them.
5. The actual CUDA probe must pass after installation. Failed command exit
   status and logs are retained, including ABI/runtime information.

A source build can take a long time and requires nvcc/compiler and enough
working disk/RAM. Installation logs stream with periodic waiting messages.
Current upstream installation details: https://github.com/state-spaces/mamba#installation.
Do not force a CPU Torch wheel onto Kaggle. A core-only Mamba install is not
proof of working selective-scan CUDA kernels.

Keep session `mamba_wheels/` and its `environment.json`/setup report as Kaggle
outputs. Upload wheels as a private dataset for the next session. Match Python
ABI (e.g. cp313), Linux architecture, Torch minor, CUDA and CXX11 ABI. **Packages
do not survive a fresh session**; attach/install wheels again. Run All works
without a manual kernel restart when started fresh and setup precedes imports.
If you have already imported/replaced binary packages interactively, restart
once before Run All to avoid stale modules.

## 5. Cache and normalization behavior

The first build reads WFDB and filters each needed 100 Hz recording once.
It writes a compact float32 memory map, usually ~1.05 GB for the full dataset,
with atomic completion manifest. It does not cache augmentation or store a
second normalized full array. Training samples normalize from the saved stats
and apply stochastic augmentation afterward; validation/test remain deterministic.

Default sampled normalization uses up to 1000 reproducibly chosen official
training IDs, before label exclusions. Full mode uses every training record.
Cached JSON includes the selected IDs, mode, seed, sample count, mean/std and
preprocessing identity. Superclass/subclass runs share these exact statistics.

`CACHE_DIR` defaults to `/kaggle/working/btp_cache`. Metadata/raw-file-stat,
preprocessing code/settings and numerical package versions key the cache;
arrays have shape/dtype/size/SHA256 checks. Corrupt/incomplete caches rebuild.
Changing model/task/augmentation need not rebuild deterministic signals.
Changing sampling/filter/length/data identity does. Rebuilding remains
single-writer; do not run two cache-building notebooks against the same folder.
For timestamp-preserving source changes, use a new cache directory/rebuild.

Progress is explicit during metadata/signature checks, preprocessing,
normalization, loader/trainer initialization, first batch/step, batches/epochs
and checkpoint writes. Optional workers have timeout/bounded prefetch/persistent
workers. Start with NUM_WORKERS=0; only increase after profiling.

## 6. Find outputs and diagnose failure

Each execution creates:

```text
/kaggle/working/btp_outputs/sessions/<timestamp_id>/
  pipeline.log, pipeline_status.json, final_summary.json
  environment.json, session_config.json, setup_report.json
  metadata_audit.json, normalization.json, *_class_statistics.csv
  *_smoke.json
  setup_logs/, mamba_wheels/
  experiments/benchmark/<task_run>/      # complete-data runs
  experiments/diagnostic/<task_run>/     # limited diagnostic runs
```

Task folders contain `run.log`, `status.json`, config/environment/data manifests,
class statistics, `best.pth`, `last.pth`, history and loss/confusion figures.
Benchmark folders also contain validation thresholds, validation/test/fixed-.5
metrics and predictions. Diagnostics only export diagnostic_validation/. They
cannot silently create test results or be labeled successful benchmarks.

The notebook verifies required exports. If a task fails, its stage/log and
completed artifacts survive. By default the other task is still attempted;
final_summary records errors and the notebook raises at the end to signal
failure. An earlier setup/cache failure stops immediately with stage logs.
Save the entire session directory even after failure.

For a stall, inspect the last log: metadata/stat scan, cache record count,
normalization count, loader first-batch wait, model forward/backward, metrics,
or checkpoint write. History contains train/validation data-wait timing. Near-zero
GPU usage alone is insufficient to prove a deadlock. Initial cache creation
still pays I/O once; measured Kaggle speedups/throughput remain unverified.

## 7. Resume an interrupted run

Retain best.pth, last.pth, config and data manifests together. Copy a previous
Kaggle output run folder to writable `/kaggle/working` if needed. Reattach the
same source dataset/release and reuse/rebuild the exact preprocessing cache.
Set the initial `RESUME_RUN_DIRS` mapping to the original writable task run
folder. Keep the original epoch budget (e.g. interrupted epoch 3 of 10 resumes
through 10), seed, task and data/model settings. The notebook checks them and
restores the old output_root instead of silently changing schedules.

Incompatible pre-Task-3 checkpoints/settings require a new run. Zero-worker CPU
RNG continuation is tested; CUDA fused kernels/multi-worker restarts may not be
bitwise reproducible. Avoid changing diagnostic mode or class order on resume.

## Remaining real-GPU validation

The user reports prior real CUDA/AMP smoke success, not full benchmark results.
Local tests execute CPU synthetic data and explicit CUDA/model/setup test doubles.
Actual cached PTB-XL throughput, GPU utilization/memory, full Mamba epochs,
AMP stability and test metrics on either release must be measured in Kaggle.
Do not report a complete benchmark unless its real run and exports succeeded.
