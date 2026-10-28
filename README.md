# Self-Supervised Learning for ECG Classification Using Mamba

The repository supports supervised Mamba on PTB-XL and **Milestone 2 masked
waveform reconstruction SSL**, using the same encoder and preprocessing.
Use **`05_btp_kaggle_final.ipynb`** as the primary Kaggle execution notebook.
No extra encoder architectures are added. SSL pretraining is opt-in and full
pretraining has not been executed here. See [SSL settings and outputs](docs/ssl_pretraining.md).

- `04_btp_experiments.ipynb`: development reference, preserved.
- `ecg-xl-btp.ipynb`: user's actual Kaggle attempt code, preserved. It has no
  saved outputs; successful CUDA/AMP smoke is user-reported, not a new benchmark.
- The three original preprocessing/training/evaluation notebooks are unchanged.
- Targets: five superclasses `NORM, MI, STTC, CD, HYP`, or all **23** official
  diagnostic subclasses (not 24), sorted by their official mapping names.

## Fully automated Kaggle execution

The final notebook is self-contained as an entry point. It securely retrieves
`GITHUB_TOKEN` from Kaggle Secrets, clones/fast-forwards this branch into
`/kaggle/working/BTP_project_files`, and runs the reusable modules from that
synchronized checkout. **No attached source dataset is required.**

Before starting, import the latest `05_btp_kaggle_final.ipynb`, enable a GPU and
Internet, attach PTB-XL (plus compatible wheels if available), and enable a
Kaggle secret named `GITHUB_TOKEN` with read access to this repository. Configure
only the first cell: release/data paths if needed, task flags, epochs/batch,
early-stopping patience and optimizer settings. Then **Run All / Save & Run All**.
There are no intermediate prompts, source edits, cell reruns or kernel restarts.
See [the execution guide](docs/kaggle_execution.md) for the exact sequence.

The current notebook is prepared for **controlled FP32 weights-only continuation**
from preserved session `20261028T063245Z_f710283f`: superclass epoch 3 and subclass
epoch 5. Attach those outputs too; the notebook finds and hashes both last.pth
files, checks provenance, and writes NEW diagnostic runs. See
[checkpoint compatibility and exact next-run settings](docs/checkpoint_continuation.md).
For deliberate fresh training, clear CONTINUATION_SESSION_ID and both checkpoint
mappings in the first cell before Run All. Missing source artifacts never trigger
an automatic fresh-training fallback.

Git credentials never enter URLs, argv, Git configuration, saved settings or
logs. A transient askpass script contains only environment lookup; the token
is passed only to clone/fetch child environments and removed afterward.
Existing uncommitted files, wrong origins or local-only/divergent commits are
refused; there is no reset, stash or clean. Python source precedence is verified.

Dependency setup runs in a clean controller interpreter, and scientific work
runs in a second fresh interpreter after installation. This avoids relying
on restarting a kernel that may already have older modules/binaries loaded.
Source, dependencies, CUDA, patient audit and short real-data GPU smoke are
verified before the full deterministic waveform cache is prepared. The two
full experiments then run sequentially with all requested training records.

Defaults now verify numerical stability: both tasks, smoke and audit, **FP32
(AMP off), LR 1e-4, one epoch, batch 2, DIAGNOSTIC_MODE=True**, workers 0,
and sampled 1000-training-record normalization. DIAGNOSTIC_MODE=True selects
explicitly limited train/validation-only diagnostics, not benchmark completion.
Early stopping and full budgets are first-cell settings. Every stage has
progress/30-second heartbeats and a persisted pass/fail summary.

All results, cache, logs, wheel artifacts and checkpoints are under
`/kaggle/working/btp_outputs/`. Epoch checkpoints, history and JSON summaries
use atomic replacement so an interrupted write preserves the preceding complete
file. Completed tasks/checks are recorded immediately. Kaggle can discard
working storage when destroying a session: retain saved notebook outputs for
future-session resume; local checkpoints cannot override platform retention.

## Dataset structure and release consistency

```text
<DATASET_ROOT>/
  ptbxl_database.csv
  scp_statements.csv
  records100/00000/00001_lr.hea
  records100/00000/00001_lr.dat
  ...
```

Use [official PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and record the
actual release used. Do not relabel 1.0.1 as 1.0.3 or mix their metrics as the
same experiment. Metadata/raw-file signatures identify caches; manifests
record the operator-declared version, source root, metadata hashes, class order,
record IDs and exact preprocessing/normalization settings. CSV schema alone
cannot certify the release. No waveforms/patient metadata are committed.

Both tasks use code presence (including confidence zero), multi-hot binary
labels, and no frequency cutoff. Unlabeled supervised records are counted and
excluded (or rejected with `unlabeled="error"`). Rare class columns remain.
Official folds 1–8 / 9 / 10 are patient-audited before exclusions; invalid folds,
missing patients and pairwise overlaps fail. Waveform errors identify ECG/path.
100 Hz, 10 seconds, 12 leads is the default `(B,12,1000)` layout. The lower-level
API also supports time-first tensors and 500 Hz via DataConfig.

## Fixes and performance

The encoder now returns pooled features; the classifier applies its head once.
The original 384-wide/eight-bidirectional-block architecture and parameter names
are retained. The user's local fix is permanently integrated. See
[the attempt analysis](docs/kaggle_attempt_analysis.md) for evidence and limits.

`DataConfig` defaults to sampled training-only normalization: up to **1000**
training ECGs chosen deterministically by `normalization_seed`. Set
`normalization_mode="full"` for exact full-training statistics. Both tasks fit
from the same official training population before label exclusions. IDs,
seed, means/stds and configuration are saved and cached across tasks/models.

With `cache_signals=True`, deterministic waveform reading/filtering/center
pad-crop happens once. One float32 NumPy memory map stores **unaugmented,
unnormalized** signals (~1.05 GB for 21,800 100 Hz ECGs). Samples read the local
cache, normalize with saved statistics, then augment only for training.
This avoids duplicate normalized arrays and repeated WFDB/filter work per epoch.
Default cache location is `/kaggle/working/btp_outputs/cache`.
The first build still pays real waveform I/O/filtering costs, with progress.

Cache identity includes root, declared release, both metadata hashes, source
waveform size/mtime signatures, sampling/filter/length settings, preprocessing
code and numerical/WFDB versions. Task, augmentation, batch size and model are
excluded so experiments can share deterministic data. Array header, size and
SHA256 are checked; incomplete/corrupt caches rebuild. Normalization settings
have their own key. Read-only Kaggle inputs are assumed immutable during a run;
for timestamp-preserving raw changes, choose a new cache directory/rebuild.
Keep enough disk for ~1.05 GB per preprocessing variant plus checkpoints/wheels.
No full signal stack is held in RAM. Concurrent cache writers are not supported.

The final default is zero DataLoader workers to avoid the observed ambiguous
startup path. Optional workers reopen maps independently, with timeout,
bounded prefetch and persistent workers. Progress covers metadata/signatures,
cache/normalization, initialization, first batch/optimizer step, subsequent
batches, epochs and checkpoint writes. Epoch history includes data-wait timing.
A measured Kaggle speedup is **not** claimed without a new Kaggle run.

## Successful installation procedure and wheels

User-reported working build: Python 3.13.15, Torch 2.11.0+cu128, toolkit 12.8,
Tesla T4, ABI=True; causal-conv1d **1.7.0**, mamba-ssm **2.3.2.post1**,
WFDB **4.3.1**. The final setup retains working installations, prefers attached
wheels and explicitly falls back to source only if `ALLOW_SOURCE_BUILD=True`.
It uses `MAMBA_KEEP_CUDA_BUILD=TRUE`, `MAX_JOBS=2`, `--no-build-isolation`,
verbose logs, and builds saved wheels before installing them. Torch is never
requested by Kaggle requirements or CUDA-package installation commands.

Wheel compatibility includes Python tag, Linux/architecture, Torch minor,
CUDA and CXX11 ABI. Built wheels and setup logs live in the session directory;
turn `mamba_wheels/` into a private Kaggle dataset for reuse. New Kaggle sessions
need packages installed again. The setup tests actual Mamba CUDA backward,
not just import success. Base/CUDA failures retain commands, logs and stage.
See the guide for offline wheels, source fallback and failure handling.

## Training, evaluation, outputs and resume

Reference optimization remains AdamW 3e-4, weight decay 5e-4, positive BCE
weights capped at 8, training-only smoothing .05, clipping 1, optional AMP,
three warmup epochs and cosine floor 1e-6, with validation-AUROC early stopping.
Reference batch was 64; verification default 2 is an explicit memory adjustment.
Reduce batch size on OOM and record the change; do not silently shrink Mamba.
Separate task epoch overrides are available in the first notebook cell.

BCE operands and loss are explicitly float32 outside autocast in training,
validation and smoke. Both numerical modes reject invalid inputs, targets,
logits, loss, gradients and optimizer/model state. Gradient clipping rejects
nonfinite norms. Failures save ECG IDs, epoch/batch, ranges and AMP/LR/state
diagnostics under the experiment folder; no batch is skipped or retried.
See [numerical stability and recovery](docs/numerical_stability.md) for the
safe first run, confirmed findings versus suspected causes, and checkpoint reuse.

Each benchmark run exports best/last checkpoints, config/environment/data
manifest, class counts, history, loss/confusion figures, validation thresholds,
validation/test/fixed-.5 predictions, per-class/overall metrics, timings and logs.
AUROC/AP/F1/precision/recall and support/confusion counts use one shared evaluator.
AUPRC is average precision. Undefined ranking scores become JSON null/blank CSV;
valid-class counts are reported, with zero-denominator F1/precision/recall=0.
Thresholds use validation only; test never selects models or thresholds.

The session has `pipeline.log`, stage status, setup logs/wheels, smoke reports,
normalization and `final_summary.json`. Runs have `run.log` and `status.json`;
failures preserve partial and completed-task artifacts. By default a failed
superclass run does not prevent attempting subclass; the final summary marks
failure and the notebook exits failed after saving it.

Resume by setting `RESUME_RUN_DIRS["superclass"]` or `["subclass"]` to the
original directory with `last.pth`, `best.pth`, and saved config. Keep the
original epoch budget, settings, dataset/cache paths and release; the final
notebook checks them. Restore/copy prior Kaggle outputs to writable working
storage before resume. Data/checkpoints from before these new normalization
settings require a new run; incompatible configs fail rather than silently
continue. Zero-worker CPU RNG continuation is tested; fused CUDA kernels and
multi-worker restarts need not be bitwise identical.

## Local checks

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

For Windows sandbox temp restrictions, create `outputs/` and add a fresh
`--basetemp=outputs/pytest-local` path. Local tests do not require Mamba/CUDA.
CPU head regression uses a clearly identified block test double; actual CUDA
compatibility remains covered by conditional GPU tests. The final notebook is
tested sequentially with explicit CPU installation/CUDA/model test doubles,
as well as syntax/schema checks. This is not evidence of GPU benchmark success.
See [test results](docs/test_results.md).

Core modules: `config`, `data`, `cache`, `signal`, `augmentation`, `models/mamba`,
`training`, `evaluation`, `experiments`, `kaggle`, `kaggle_setup`, secure standard-library
`bootstrap`, and the `unattended` controller/worker. Future architectures may use the registry; none are added now.
`MambaEncoder.forward_tokens` exposes temporal backbone features for the separate
`ssl` / `ssl_data` masked-reconstruction path; pooled `forward_features` remains
available for supervised classification. Both SSL flags are disabled by default.

## References

Wagner et al., [PTB-XL v1.0.3](https://doi.org/10.13026/kfzx-aw45), CC BY 4.0.
The only dataset fixture committed is the small official SCP mapping;
see `tests/fixtures/README.md`. Original notebook analysis is in
`docs/notebook_analysis.md`; historical stored scores are not new results.
