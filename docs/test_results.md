# Executed checks — 2026-10-26

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
