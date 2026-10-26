# Self-Supervised Learning for ECG Classification Using Mamba

This repository now provides a shared **supervised** baseline pipeline for PTB-XL
v1.0.3. Start with `04_btp_experiments.ipynb`. The three original notebooks are
unchanged reference implementations. No new benchmark architectures or SSL
pretraining are included.

**Specification correction:** the official v1.0.3 `scp_statements.csv` has
**23 distinct diagnostic subclasses**, not 24. Both tasks use every official
mapping: five superclasses or all 23 subclasses. The classifier accepts any
output width (including 24 for a future explicitly defined vocabulary), but
this pipeline does not fabricate a 24th PTB-XL label. See the checked-in mapping
fixture and `docs/notebook_analysis.md`.

## Local setup and checks

Use Python 3.10+ and install PyTorch for your platform (CPU is sufficient for
data/evaluation/tests). Run commands from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

If your environment restricts temporary directories, first create `outputs/`,
then use `python -m pytest -q --basetemp=outputs/pytest-local` with a fresh
temporary directory name. Pytest clears its selected basetemp before each run.
No real dataset or Mamba installation is needed for CPU tests. Waveform I/O uses
wfdb; it is imported lazily. Mamba imports only when that model is requested,
with a clear error if its CUDA dependencies are missing or incompatible.

Versions are bounded in requirements rather than claiming a universal CUDA
lockfile. Each actual experiment writes exact runtime package versions,
Python/platform, Torch CUDA version and GPU name to `environment.json`.

## Dataset and labels

Attach/download PTB-XL **v1.0.3** separately using
[the official dataset](https://physionet.org/content/ptb-xl/1.0.3/).
Do not commit waveforms or patient metadata. Set `PTBXL_DATA_DIR` or pass
`data_dir` to the folder with this structure:

```text
<PTBXL_DATA_DIR>/
  ptbxl_database.csv
  scp_statements.csv
  records100/00000/00001_lr.hea
  records100/00000/00001_lr.dat
  ...
  records500/...                 # only needed for sampling_rate=500
```

The loader records metadata hashes, original record count, normalization fit
IDs and supervised split IDs. It records the **requested** version as 1.0.3;
the CSV files do not themselves prove the release number. Verify the attached
dataset's provenance. Original notebook paths/outputs refer to v1.0.1.

```python
from src.data import load_ptbxl
from src.config import DataConfig

data = load_ptbxl(task="superclass", data_dir="/path/to/ptb-xl-1.0.3")
subclasses = load_ptbxl(task="subclass", data_dir="/path/to/ptb-xl-1.0.3")
loaders = data.loaders(batch_size=64)
print(data.class_names, data.statistics)
```

Defaults: 100 Hz, 10 seconds, 12 leads, `(B,12,1000)` tensors. Use
`DataConfig(layout="time_first")` for `(B,1000,12)`; the common trainer/inference
adapter converts to `(B,12,T)` before model calls. `sampling_rate=500` loads
500 Hz records and uses 5000 samples, without pretending they are 100 Hz.

Superclasses have order `NORM, MI, STTC, CD, HYP`. Subclasses have this sorted order:

```text
AMI, CLBBB, CRBBB, ILBBB, IMI, IRBBB, ISCA, ISCI, ISC_, IVCD,
LAFB/LPFB, LAO/LAE, LMI, LVH, NORM, NST_, PMI, RAO/RAE, RVH,
SEHYP, STTC, WPW, _AVB
```

SCP code presence, including confidence zero, is mapped through diagnostic rows
of the official CSV. Labels are binary multi-hot, with multiple diagnoses
retained and no minimum class frequency. Records with no diagnostic labels are
counted, their IDs exported, and excluded from supervised tasks; set
`DataConfig(unlabeled="error")` to reject them instead. Unknown diagnoses are
never silently treated as known negative examples.

Folds 1–8 train, 9 validate, 10 test. Pairwise patient separation is checked
before excluding labels. Missing patient IDs, invalid folds, empty splits and
duplicate ECG IDs raise errors. Missing/corrupt/nonfinite waveforms raise with
ECG ID and path; no replacement zero waveforms are supplied.

Filtering uses the reference 4th-order 0.5–40 Hz Butterworth bandpass, zero phase,
in stable SOS form; a 50 Hz notch is applied only below Nyquist. Center
pad/crop makes length deterministic. Per-lead population mean/std is streamed
over **processed training-fold signals only**, before task label exclusions,
so both tasks use the same normalization. Every load reads the complete training
waveform set once to fit it. Augmentation applies only to training. Set
`DataConfig(augment=False)` to disable it or `apply_filter=False` for a declared
preprocessing ablation. Do not mix these settings across benchmark models.

## Kaggle GPU execution

1. Create a Kaggle GPU notebook and attach a verified PTB-XL **v1.0.3** dataset.
   Select a CUDA GPU. No long training has been run by this repository setup.
2. Download this development branch as a source archive and attach its extracted
   files as a private Kaggle dataset, keeping `src/`, requirements, tests and the
   main notebook together. Import `04_btp_experiments.ipynb` into the editor.
   The source can stay read-only under `/kaggle/input`; outputs go to working.
3. Set paths before the main notebook bootstrap (replace these example paths
   with your actual attached folder names):

   ```python
   import os
   os.environ["BTP_REPO_DIR"] = "/kaggle/input/btp-source/BTP_project_files"
   os.environ["PTBXL_DATA_DIR"] = "/kaggle/input/ptbxl-v103/ptb-xl-1.0.3"
   from pathlib import Path
   assert (Path(os.environ["BTP_REPO_DIR"]) / "src").is_dir()
   assert (Path(os.environ["PTBXL_DATA_DIR"]) / "ptbxl_database.csv").is_file()
   ```

4. Install the base requirements without unnecessarily replacing Kaggle's
   existing Torch/CUDA runtime:

   ```python
   import subprocess, sys
   subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                   os.path.join(os.environ["BTP_REPO_DIR"], "requirements.txt")], check=True)
   import torch
   print(torch.__version__, torch.version.cuda)
   print("CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
   assert torch.cuda.is_available()
   ```

5. Install CUDA dependencies separately. With Internet and a compatible build
   environment, following the [official Mamba setup](https://github.com/state-spaces/mamba#installation):

   ```python
   subprocess.run([sys.executable, "-m", "pip", "install", "packaging", "ninja"], check=True)
   cuda_build_env = os.environ.copy()
   cuda_build_env["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"
   subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                   "causal-conv1d>=1.4.0", "mamba-ssm"], check=True, env=cuda_build_env)
   ```

   Current upstream Mamba defaults to a core install without `selective_scan_cuda`;
   `MAMBA_KEEP_CUDA_BUILD=TRUE` opts into its CUDA extension. Older compatible
   wheels may already bundle it. The forward test below is the deciding check.

   If source builds fail or Internet is disabled, attach **compatible prebuilt
   wheels** matching Python, Linux, Torch, CUDA and CXX11 ABI, then use:

   ```python
   subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                   "--find-links=/kaggle/input/mamba-wheels", "causal-conv1d", "mamba-ssm"], check=True)
   ```

   There is no universal wheel compatible with every Kaggle runtime. Restart
   the kernel after installing binary packages if needed and rerun the path
   setup. Verify **an actual CUDA forward**, rather than only successful import:

   ```python
   import sys, torch
   sys.path.insert(0, os.environ["BTP_REPO_DIR"])
   from src.models.registry import create_model
   from src.config import ModelConfig
   m = create_model("mamba", 5, ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1)).cuda().eval()
   with torch.no_grad():
       assert m(torch.randn(2, 12, 1000, device="cuda")).shape == (2, 5)
   del m
   torch.cuda.empty_cache()
   ```

6. Run the notebook bootstrap/config cells, set `RUN_TRAINING = True`, and run
   the superclass/subclass cells. Equivalent calls:

   ```python
   from src.config import ExperimentConfig, DataConfig
   from src.experiments import run_experiment
   cfg = ExperimentConfig(
       data=DataConfig(data_dir=os.environ["PTBXL_DATA_DIR"], num_workers=2),
       output_root="/kaggle/working/btp_outputs",
   )
   a = run_experiment(model_name="mamba", task="superclass", epochs=20, config=cfg)
   b = run_experiment(model_name="mamba", task="subclass", epochs=20, config=cfg)
   ```

   Reference batch size is 64. The large reference model may exceed the GPU's
   memory; pass `batch_size=8` or 16, record that change, and compare with the
   same configuration in future experiments. This GPU budget is not verified
   locally. Save the contents of `/kaggle/working/btp_outputs` as notebook output.

## Training, checkpoints and evaluation

The reference model uses eight independent forward/backward Mamba layers,
embedding/hidden width 384, d_state 16, d_conv 4, expand 2, dropout .45 and
mean-plus-attention pooling. It retains the original parameter names. The
training source's budget was 50 epochs; `ExperimentConfig(epochs=50)` restores
that budget. Other reference settings are retained in `src/config.py`.

Training uses AdamW, weighted `BCEWithLogitsLoss`, training-only .05 target
smoothing, positive weights capped at 8, gradient clipping, optional CUDA AMP,
linear warmup and cosine decay. BCE validation loss has unsmoothed targets.
Early stopping maximizes validation macro AUROC. If no validation class has
both outcomes, the run consistently falls back to negative validation loss.
History loss is averaged by sample count, including the final partial batch.

Each unique run directory contains:

- `config.json`, `environment.json`, `data_manifest.json`, `class_statistics.csv`.
- `best.pth`, `last.pth`, `history.csv`, and `thresholds.json` with validation
  provenance and class order. Checkpoints include model/optimizer/scheduler/AMP,
  early stopping/history, RNG state and fitted data preprocessing/split metadata.
- `validation/`, `test/`, `test_at_0.5/`: overall JSON, per-class CSV, binary
  confusion matrices, and ECG IDs / labels / sigmoid probabilities / predictions.
- `figures/loss_curves.png`, `figures/confusion_matrices.png`, `summary.json`.

F1 thresholds are selected on validation only using .05–.95 grid points. Ties
prefer the value nearest .5, then lower. Single-valued validation columns retain
.5. Test is never used for early stopping or threshold tuning. Both frozen
validation thresholds and the fixed .5 baseline are reported on the same test
predictions, without new model fitting.

Metrics are macro/micro AUROC, average precision (**AUPRC is AP, not trapezoidal
PR area**), F1, precision/recall, per-class support and TN/FP/FN/TP. AUROC is
undefined without both outcomes. AP is undefined with no positives and equals
1 for all-positive targets. Undefined ranking metrics export as JSON `null` /
empty CSV fields; macro ranking scores average only defined classes and report
the valid count. F1/precision/recall use zero for zero denominators, retaining
all class columns in macro F1. Micro ranking operates on flattened targets.

Summary records trainable/total parameters, training duration, epoch duration,
and validation/test inference duration. Inference timing includes loader I/O
and device transfers, with CUDA synchronization; it is not kernel-only latency.

Resume an interrupted run with its unchanged config, `run_dir=<original dir>`
and `resume_from=<original dir>/last.pth`. Keep `best.pth` with it. Class order,
config and split/preprocessing provenance must match. Exact CPU RNG continuation
is tested with zero workers; multi-worker augmentation restarts and fused CUDA
kernels do not guarantee bitwise reproducibility.

`load_checkpoint(path, model, class_names=...)` uses strict weights and validates
the ordered vocabulary. To reuse an **original** notebook checkpoint, reconstruct
its actual frozen architecture, call with `class_names=None`, and independently
verify original class order and preprocessing: old files lack those fields.
Their weights have not been validated in this environment. Do not assume a
five-class head transfers to subclasses.

## Modules and future benchmarking

| File | Purpose |
| --- | --- |
| `src/labels.py` | Official mapping, multi-hot labels and class statistics |
| `src/signal.py`, `src/augmentation.py` | Shared preprocessing, streaming normalization and extracted augmentations |
| `src/data.py` | WFDB loading, patient checks, datasets and seeded loaders |
| `src/models/mamba.py` | Extracted reference Mamba encoder/classifier |
| `src/models/registry.py` | Lazy factories and future lightweight adapters |
| `src/config.py`, `src/utils.py` | Central settings, seeds, environment and JSON |
| `src/training.py` | Common BCE trainer and checkpoint loading |
| `src/evaluation.py` | Common metrics, thresholds, inference and exports |
| `src/experiments.py` | `run_experiment` orchestration |
| `04_btp_experiments.ipynb` | Main future experiment notebook |
| `docs/notebook_analysis.md` | Original notebook analysis, reuse and known issues |
| `docs/notebook_inventory.json` | Original executable cells and output inventory |
| `docs/test_results.md` | Executed checks and environment limitations |

Future models register `register_model(name, factory)`, where
`factory(n_classes, config)` returns a Torch module accepting `(B,12,T)` and
returning `(B,C)` **logits**. Keep preprocessing, label order, folds, loss and
metrics shared. CNN/ResNet/BiLSTM/Transformer implementations are intentionally
left for later. `MambaEncoder` has no classification head and
`MambaClassifier.forward_features` exposes that same encoder path; future SSL
can reuse it without changing the supervised forward computation.

## References

Wagner et al., PTB-XL v1.0.3, [DOI 10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45),
CC BY 4.0. The only dataset file committed here is the small official SCP mapping
test fixture; its source and attribution are in `tests/fixtures/README.md`.
[Official benchmark](https://github.com/helme/ecg_ptbxl_benchmarking) documents
the recommended folds and the distinction between diagnostic tasks.
