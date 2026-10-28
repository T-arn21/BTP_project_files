# Milestone 2 masked reconstruction SSL - 2026-10-28

Initial tree clean on dev/unified-ecg-pipeline. No real GPU smoke or full SSL
pretraining was launched. Existing supervised checkpoints and original notebook
artifacts were not modified. SSL adds no encoder architecture: a temporal
feature method exposes the same Mamba operations and a separate linear head
reconstructs 12-lead waveforms. Supervised state keys and forward results remain
covered by regression checks.

```powershell
$env:PYTHONPATH = (Resolve-Path 'outputs/wfdb-check').Path
python -m pytest -q --basetemp=outputs/pytest-milestone2-release --tb=short
```

**87 passed, 4 skipped, 5 warnings in 36.15s.** Four skips require actual CUDA
Mamba (three existing supervised tests, one new reconstruction test). Warnings
are the existing SequentialLR transition notice. Local Torch remains CPU-only;
Windows disposable Git integration tests ran outside the sandbox for MSYS.

Focused evidence:

- Deterministic contiguous masks share a window across all 12 leads, retain
  exactly the requested rounded/clamped coverage and depend on seed/epoch/ECG ID,
  independently of batch ordering. Targets are not mutated.
- FP32 masked-only MSE ignores arbitrarily large unmasked errors and produces
  zero gradients outside the masked region under ambient CPU autocast.
- SSL data preparation works with diagnostic columns absent; it retains every
  training ECG, reads no held-out waveform, fits training-only normalization,
  rejects patient leakage and performs zero repeated raw reads across cached
  epochs and repeated preparation.
- Synthetic CPU encoder-double smoke and full-budget plumbing exercise actual
  masked backward/optimizer updates, finite checks, checkpoints/encoder reload,
  complete-record accounting, loss histories, split/config provenance, curves
  and 12-lead reconstruction PNG/NPZ exports. These are not real Mamba results.
- Injected NaN reconstruction aborts without exporting a successful encoder,
  saving epoch/batch/ECG/tensor finite diagnostics.
- Sequential worker tests use explicit CPU/CUDA/model doubles: SSL-only mode
  does not invoke diagnostic-label audits; full mode is blocked without a smoke
  or after a failed smoke. A passing smoke precedes explicit synthetic full mode.
- Existing supervised tests, controlled checkpoint continuation, finite guards,
  cache/split/normalization, secure bootstrap and notebook controls remain passing.

Compileall, notebook schema/syntax, unchanged embedded bootstrap, disabled SSL
defaults and Git whitespace checks pass. A generated synthetic 12-lead plot was
visually inspected: original/reconstruction curves, lead labels and shared mask
shading are readable. No generated waveform/checkpoint artifacts are committed.

Actual Kaggle must execute the preserved full-size CUDA Mamba SSL smoke first.
Full PTB-XL pretraining remains an explicit later choice via RUN_SSL_PRETRAIN;
both SSL flags currently default False. See ssl_pretraining.md for exact switches,
loss semantics, training-only data boundaries, export scope and output paths.

---

# Controlled checkpoint continuation - 2026-10-28

Read-only verification of the user's preserved real Kaggle checkpoints:
superclass epoch 3 and subclass epoch 5. Checkpoint configs and metadata match
adjacent config/manifests. Finite AdamW state and moment shapes restore on CPU
for all 188 parameters; scheduler/scaler states roundtrip exactly. Original
legacy configs pass the latest strict comparison; changing AMP to False fails
that comparison as intended. Architecture source is unchanged and preprocessing
Git source SHA256 matches the saved Kaggle identity. Original artifact hashes
are preserved. No real Mamba GPU execution or full training was launched.

```powershell
$env:PYTHONPATH = (Resolve-Path 'outputs/wfdb-check').Path
python -m pytest -q --basetemp=outputs/pytest-controlled-continuation --tb=short
```

**75 passed, 3 skipped, 5 warnings in 25.15s.** Skips require actual CUDA Mamba;
warnings are the existing scheduler transition notice. Full suite includes:

- Reject changed model, class order, filtering, normalization, raw-file identity,
  split order and missing checkpoint provenance.
- Permit declared FP32/LR changes only as an explicit new weights-only run,
  with exact original source identity/normalization and full or diagnostic-prefix
  split comparison. Diagnostic cache containers are checked independently.
- Resolve a unique preserved session/checkpoint, reject ambiguity and wrong SHA256.
- Run the full worker on CPU synthetic model/setup/CUDA test doubles, then load
  its full-run checkpoints into NEW diagnostic runs for both tasks. Verify
  provenance/reset-state reports and unchanged source checkpoint hashes.
- Existing NaN/Inf guards, exact resume, cache, splits, bootstrap/import isolation,
  notebook sequential execution, metrics and export tests continue to pass.

Compileall, notebook schema/syntax, unchanged embedded bootstrap and Git whitespace
checks pass. The final notebook selects the audited session and hashes, AMP off,
LR 1e-4, batch 2, one epoch, two diagnostic batches, and per-batch logging. Actual
Kaggle must recheck its live data/preprocessing identity, strict Mamba state load,
compatible wheels, FP32 GPU backward and both checkpoint-initialized diagnostics.
CPU restoration is not proof of CUDA compatibility or subsequent epoch stability.
See checkpoint_continuation.md for precise settings, paths and failure policy.

---

# Numerical stability release — 2026-10-28

Initial checkout was clean on dev/unified-ecg-pipeline; origin was fetched.
The user's prior real Kaggle run reportedly passed sync/install/audit, both
smokes and cache, then both tasks failed nonfinite BCE during substantial
full-data training. No failed-run artifacts were available for root-cause
confirmation. Corrected GPU training has not been executed here.

```powershell
$env:PYTHONPATH = (Resolve-Path 'outputs/wfdb-check').Path
python -m pytest -q --basetemp=outputs/pytest-stability-final --tb=short
```

**66 passed, 3 skipped, 5 warnings in 24.08s.** The three skips require real
CUDA Mamba; warnings are the existing SequentialLR transition notice. Runtime:
Python 3.12.7, Torch 2.7.1+cpu, isolated WFDB 4.1.2. Windows disposable local Git
integration tests ran outside the sandbox for MSYS transport; no fake-secret
test contacted GitHub.

New focused coverage (17 cases) deliberately injects NaN/Inf inputs, targets
and logits, finite-logit/nonfinite-loss failures, nonfinite gradients, finite
gradient elements with overflowed norm, and corrupt parameter/optimizer updates.
It verifies epoch/batch/stage, ECG IDs, finite flags/ranges, AMP/LR context,
JSON persistence, rejection before unsafe updates, and no invalid checkpoints.
A CPU autocast spy verifies BCE operands are FP32 with autocast disabled.
Validation-batch diagnostics, conservative-mode constraints, invalid class
weight rejection, nonfinite checkpoint rejection, legacy exact restore and
corrupt restored optimizer rejection pass. Explicit same-task weights-only
initialization runs a complete synthetic diagnostic with recorded provenance.

Existing sequential notebook/controller/worker tests still pass for disabled,
diagnostic and full modes using explicit CPU model/CUDA/setup test doubles.
The embedded bootstrap is unchanged and matches its tested module. Existing
cache, patient split, normalization, checkpoint interruption, deterministic
resume and export checks pass. Compileall, notebook schema/syntax and Git
whitespace checks pass. The Mamba architecture, signal/cache preprocessing,
04, the three original notebooks and the user's attempt notebook are unchanged.

The new notebook defaults are conservative numerical verification: AMP off,
LR 1e-4, batch 2, one epoch, two diagnostic batches for each task. A short
synthetic or real-data smoke pass cannot establish later full-data stability.
Kaggle still must execute corrected real Mamba FP32 backward, both diagnostics,
then full-data epochs/evaluation. No new GPU scores or benchmark completion
are claimed. See numerical_stability.md for safe Run All progression and
checkpoint/wheel reuse, and the verified-versus-suspected cause distinction.

---

# Unattended Kaggle entry-point release — 2026-10-28

```powershell
$env:PYTHONPATH = (Resolve-Path 'outputs/wfdb-check').Path
python -m pytest -q --basetemp=outputs/pytest-unattended-final --tb=short
```

**49 passed, 3 skipped, 5 warnings in 24.90s.** The three skipped checks require
real CUDA Mamba. Warnings remain the existing PyTorch SequentialLR transition
notice. No new real Kaggle authentication or GPU benchmark was executed here.

The local Git integration tests ran outside the Windows sandbox because its
MSYS local-transport startup was denied. They use disposable local/bare
repositories and a fake secret only; they make no GitHub authentication calls.
Initial CPU fixtures were updated to inject the synthetic reader into the new
pre-cache smoke path. The complete release suite then passed.

## New checks executed

- Clone and fast-forward a real disposable Git branch; verify synchronized
  local/remote commit IDs and a clean checkout after updates.
- Refuse tracked and untracked local edits before requesting a secret;
  preserve local-only/diverged commits without reset, stash or clean.
- Verify a fake secret never enters command arguments, helper-file contents,
  Git configuration or displayed output; helper files are removed.
- Kaggle Secrets provider failure details and Git authentication output are
  suppressed rather than leaking their raw messages. Real Kaggle Secrets
  access still requires execution on the platform.
- Verify source precedence in a fresh interpreter despite an intentionally
  preloaded stale src package from another directory.
- Verify the controller launches a fresh worker, removes GitHub-token
  environment variables from worker execution, and fails if the worker
  exits without a completed execution summary.
- Verify bootstrap/setup failure summaries retain their correct stage and
  success summaries identify completed experiments/output paths.
- Simulate an interrupted checkpoint write and verify the preceding completed
  checkpoint remains byte-for-byte intact.
- Execute all notebook cells sequentially with flags off. Verify the embedded
  bootstrap exactly matches its tested standard-library module source.
- Execute both complete worker orchestration branches (full-data and diagnostic)
  with explicit CPU CUDA/model test doubles, synthetic reader, actual training,
  checkpointing, evaluation, cache reuse and export checks. The new short smoke
  occurs before the full cache build. These are not real Mamba GPU results.
- Existing head, normalization, cache, labels, patient folds, generated WFDB
  file roundtrip and deterministic CPU resume regressions remain passing.

Compileall, notebook schema/syntax, and Git whitespace checks also pass.
All three original reference notebooks, 04, and the user's attempt notebook
remain unchanged. Final notebook controls now live in the first cell; the
embedded bootstrap needs no attached source dataset. Git/pip execution is
noninteractive, long stages have 30-second progress, and scientific imports
occur in a fresh process after installation, without a kernel restart.

## Real execution still required

Kaggle secret permission/read access, GitHub synchronization from that runtime,
wheel/source-build compatibility, real data/GPU/AMP smoke, cache throughput,
VRAM feasibility and full superclass/subclass training/test scores remain
unverified locally. The final notebook prints and persists passed checks,
completed/skipped/failed experiments and results under
`/kaggle/working/btp_outputs/sessions/<run_id>/final_summary.json`.

---

# Task 3 release checks — 2026-10-28

Repository: existing dev/unified-ecg-pipeline. Remote fetched before changes.
The initial working tree included the user's one-line encoder fix and new
`ecg-xl-btp.ipynb`; both are preserved and included in this milestone.

```powershell
$env:PYTHONPATH = (Resolve-Path 'outputs/wfdb-check').Path
python -m pytest -q --basetemp=outputs/pytest-task3-release --tb=short
```

**41 passed, 3 skipped, 5 warnings in 26.27s.** The three skips require real
Mamba CUDA kernels; warnings remain PyTorch SequentialLR transition deprecation.
Local runtime remains Python 3.12.7, Torch 2.7.1+cpu, NumPy 1.26.4, pandas
2.2.2, SciPy 1.13.1, scikit-learn 1.5.1, with isolated WFDB 4.1.2 from the
previous milestone. The user's reported Python 3.13/Torch 2.11/WFDB 4.3.1 Kaggle
stack is documented, **not locally reproduced**.

## New regression evidence

- Actual encoder/pooling/head code runs on CPU with a test double only for
  the CUDA-only block. Both 5- and 23-output models return pooled features
  from the encoder, call the head once, and backpropagate finite gradients.
- Deterministic sampled training-only normalization, dedicated sampling seed,
  exact full mode, shared superclass/subclass cache/statistics and retained
  original split/rare-class behavior pass.
- Cached normalized tensors match the uncached path exactly. Changing filter,
  metadata or source-file stats invalidates caches; incomplete completion
  markers and same-size corrupted payload bytes trigger rebuilds.
- A cached DataLoader reaches its first batch and an actual optimizer update.
  Two training passes use **zero additional waveform-reader calls** after
  initial cache creation. Augmentation remains outside deterministic caching.
- A synthetic end-to-end diagnostic performs training/checkpoint/evaluation
  exports without test predictions or tuned thresholds. Diagnostic limits
  cannot be used as benchmark data.
- Every final notebook code cell executes sequentially with all flags disabled.
  Explicit CPU model/CUDA/setup test doubles also exercise Run All with both
  tasks, smoke, full training/evaluation/export branches, and diagnostic branches.
  These tests do not claim real Mamba/GPU execution.
- Setup tests confirm working installations skip install and source-build
  commands retain pinned versions, KEEP_CUDA=True, MAX_JOBS=2, verbose logging
  and --no-build-isolation. No actual CUDA installation was attempted locally.
- Existing label, data, metric, checkpoint, resume, and generated WFDB roundtrip
  tests continue to pass.

## Small synthetic performance check (not a Kaggle benchmark)

36 generated waveforms, cached float32 100 Hz data, CPU test-only linear adapter:

| Measurement | Observed |
| --- | --- |
| Metadata/cache/normalization preparation | 0.1484 s |
| Cached first batch (4 records) | 0.00173 s |
| BCE backward + optimizer step | 0.00076 s |
| Reader calls after preparation across 48 training samples | 0 |

These numbers measure local synthetic plumbing, **not** WFDB storage throughput
or preserved Mamba training speed. Their useful evidence is reaching optimization
and eliminating repeated reader/filter work. Real initial cache creation still
reads and filters every required record once; its Kaggle wall time is unknown.

Additional checks: compileall, notebook schema/syntax and Git whitespace checks
pass. The original three reference notebooks are byte-for-byte unchanged.
`04_btp_experiments.ipynb` has no Git diff (Windows text newline normalization
is accounted for). The user's attempt notebook is preserved as supplied and has
zero saved outputs, so its reported smoke success is attributed to the user.

## Still needs real Kaggle execution

- Pinned source-build/supplied-wheel compatibility on the actual fresh runtime.
- Real PTB-XL release/data/hash verification, full deterministic-cache build
  duration and space use, first-batch time, data-wait timing and GPU utilization.
- Both full-architecture CUDA/AMP smoke runs after the permanent head fix.
- Full superclass/subclass epochs, validation/test performance and complete
  benchmark outputs. No new real classification scores are claimed.
- Actual GPU memory budget and fused-kernel/multi-worker reproducibility.

See README and kaggle_execution.md for the Run All controls and resume procedure.

---

# Executed checks — latest 2026-10-27

## Kaggle preflight preparation milestone

The repository was clean on `dev/unified-ecg-pipeline` before this milestone.
The existing commit was pushed successfully to GitHub. New work prepares real
Kaggle execution; it does **not** claim real PTB-XL waveform or CUDA validation.

Final local command (PowerShell, repository root):

```powershell
$env:PYTHONPATH = (Resolve-Path 'outputs/wfdb-check').Path
python -m pytest -q --basetemp=outputs/pytest-kaggle-validation-1 --tb=short
```

Result: **29 passed, 3 skipped, 5 warnings in 45.28s**. The three skipped tests
need actual Mamba CUDA kernels. The warnings remain PyTorch's internal
SequentialLR transition deprecation. Python/Torch/scientific package versions
are unchanged from the baseline below.

WFDB 4.1.2 was installed with `--target outputs/wfdb-check --no-deps` to avoid
changing the global Python environment. That ignored folder is not committed;
normal users install WFDB through requirements. The passing suite includes:

- Unique, absent, ambiguous and invalid explicit dataset-root diagnostics.
- Both official target vocabularies and metadata audit exports/hashes.
- Bounded smoke normalization from training IDs only; no test-fold waveform
  reads, with one tiny train/validation batch for both tasks.
- A real WFDB `.hea`/`.dat` write/read roundtrip using **locally generated ECG
  waveforms**, including sampling-frequency validation. These are not PTB-XL.
- Finite BCE, finite nonzero gradients, actual optimizer weight changes and
  validation forward using a **test-only CPU linear adapter**. This verifies
  smoke machinery, not Mamba or GPU execution.
- Wrong raw waveform shape and missing GPU diagnostics.
- All updated notebook code cells execute safely with default flags False;
  no installation, dataset read or training is triggered. Missing smoke/audit
  evidence blocks the full-run control path. Epochs are centrally configured.
- All existing pipeline, metrics, synthetic experiment and checkpoint-resume
  tests continue to pass.

`python -m compileall -q src tests`, notebook schema/syntax validation and
`git diff --check` also pass. Original reference notebooks remain untouched.

Still requires **actual Kaggle execution**: discovering the user's attached
real v1.0.3 dataset, auditing its patient metadata, reading its real 100 Hz
waveforms, matching CUDA wheels, initializing the full 384-wide/eight-block
Mamba, passing real-data forward/backward/AMP smoke for both heads, fitting
full-training normalization, completing superclass/subclass training and test
evaluation, validating GPU memory use, and inspecting/exporting those real-run
artifacts. See `kaggle_execution.md` and the notebook's explicit run controls.

## Baseline milestone — 2026-10-26

Final automated command, run from `D:\Coding\BTP\BTP_project_files`:

```text
python -m pytest -q --basetemp=outputs/pytest-final --tb=short
21 passed, 3 skipped, 5 warnings in 19.46s
```

The warnings are PyTorch's internal SequentialLR scheduler transition deprecation
warning. They do not indicate failed optimization. Earlier test runs encountered
Windows sandbox temporary-directory permissions; selecting a repository-local
ignored output directory resolved those environmental failures. A test-fixture
syntax typo was also fixed before the successful runs.

Runtime: Python 3.12.7, PyTorch 2.7.1+cpu, NumPy 1.26.4, pandas 2.2.2,
SciPy 1.13.1, scikit-learn 1.5.1. No CUDA device and no mamba-ssm installed.
Tests do not depend on wfdb being installed: a deterministic injected reader
generates synthetic ECGs; a mocked WFDB reader separately checks transpose and
frequency validation. The only downloaded dataset file is the official 9.5 KB
SCP mapping fixture. No real patient metadata or waveforms were downloaded.

| Required check | Executed evidence |
| --- | --- |
| Superclass labels | Five-class order, multiple diagnoses and confidence-zero code presence |
| Subclass labels | All **23 actual official** subclasses covered; no rare-class cutoff; absent columns retained |
| Encoding/order | Binary multi-hot vectors; mapping row shuffle leaves deterministic order unchanged |
| Patient separation | Official folds, pairwise overlaps rejected, fractional/invalid folds rejected |
| Waveform tensors/loaders | Both layouts, 100 Hz `(12,1000)` and 500 Hz `(12,5000)`, no dropped training tail |
| Training-only normalization | Streamed statistics agree with direct processed-training calculations; held-out perturbations do not change fit; both tasks agree |
| Corrupt/unlabeled records | Explicit waveform failures; unlabeled drop/error policies; excluded IDs recorded |
| Augmentations | All nine operations preserve shape and finiteness; seeded reproducibility and stochastic repeated calls |
| Mamba real forward | **3 skipped tests:** 5/23/24 output widths require CUDA and compatible mamba-ssm |
| BCE and common training | Finite weighted multilabel BCE losses, backward updates and validation for 5/23/24 output widths using a test-only linear adapter |
| Early stopping | Deliberately unreachable improvement tolerance stops at expected patience |
| Checkpoint serialization | Strict weights-only reload; reordered vocabulary rejected; state/history restored |
| Interrupted resume | Three-epoch CPU run matches interrupted-and-resumed weights bit-for-bit with zero workers |
| Metrics | Macro/micro AUROC/AP/F1, precision/recall, counts/support; all-negative/all-positive edge cases; strict JSON null for missing ranking scores |
| Threshold leakage | Validation-only threshold tuning; explicit test split rejected; absent/single-valued validation columns keep 0.5 |
| Exports | Per-class/overall metrics, confusion counts and predictions CSV/JSON files verified |
| Full orchestration | One-epoch synthetic superclass and subclass experiments created unique output directories, best/last checkpoints, manifests, thresholds, figures and summaries |
| Notebook | nbformat validation and Python syntax parsing passed |

Additional executed checks:

- Every default executable cell in `04_btp_experiments.ipynb` ran successfully
  in a fresh namespace, without a dataset, package installation or training.
- `python -m compileall -q src tests` passed.
- `git diff --check` passed.
- All three original notebook files were compared **byte-for-byte** with
  `git show HEAD:<filename>` and are unchanged.
- Generated loss curves and the 23-column confusion grid were visually inspected.
  These figures contain **synthetic adapter results only**, not PTB-XL performance.

## Still requires Kaggle / real data

- Real WFDB waveform loading and v1.0.3 metadata integrity on the attached dataset.
- Mamba CUDA installation, forward/backward, AMP and exact source-model/checkpoint
  compatibility. Optional tests compare the extracted classifier to the original
  executable architecture with strict state_dict loading when CUDA is available.
- Real training, GPU memory requirements for the 384-wide/eight-block reference,
  actual classification metrics and measured Mamba inference/training time.
- Multi-worker reproducibility and binary wheel compatibility with Kaggle's
  actual Python/Torch/CUDA/CXX11 ABI combination.

No performance results are claimed for the new pipeline. The saved results in
the original notebooks use a different dataset version and, for evaluation, a
different model size. See `notebook_analysis.md` before reusing old checkpoints.
