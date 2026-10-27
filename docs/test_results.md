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
