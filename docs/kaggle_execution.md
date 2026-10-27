# Real PTB-XL validation: execution sequence

Use `04_btp_experiments.ipynb` from branch `dev/unified-ecg-pipeline`.
No Kaggle GPU run is claimed by preparing this notebook. The three reference
notebooks remain unchanged. This milestone uses Mamba only, without SSL.

## Attach sources and select a GPU

1. Download the current branch's source archive from GitHub, extract it, and
   upload it as a **private** Kaggle dataset. Keep `src/`,
   `requirements-kaggle.txt`, and `04_btp_experiments.ipynb` together. Import the
   notebook into Kaggle. The source may remain read-only under `/kaggle/input`.
2. Attach a separate, extracted PTB-XL **v1.0.3** Kaggle dataset with both
   `ptbxl_database.csv`, `scp_statements.csv`, and `records100/` waveform files.
   Do not reuse the originals' v1.0.1 dataset without explicitly declaring it
   as a different experiment. CSV schema alone does not certify release version.
3. Choose a CUDA GPU in Kaggle's session settings. Enable Internet if package
   installation needs it. Outputs are saved under `/kaggle/working/btp_outputs`.

## Source and dataset discovery

The first cell needs only Python's standard library. It finds a source root by
`src/kaggle.py` and `requirements-kaggle.txt`. Section 5 locates a dataset by
both CSV files plus `records100/`, including nested Kaggle input folders.
Zero/multiple candidates stop with a diagnostic. A bad explicit override never
falls back silently. To resolve ambiguity set these overrides in section 1:

```python
REPO_OVERRIDE = "/kaggle/input/btp-source/BTP_project_files"  # your actual path
DATA_OVERRIDE = "/kaggle/input/ptbxl-v103/ptb-xl-1.0.3"       # your actual path
```

Alternatively set `BTP_REPO_DIR` / `PTBXL_DATA_DIR` before running the notebook.
Use the directory containing the files, not the parent input slug or a ZIP file.

## Dependency and CUDA setup: sections 2–4

All install flags default False. The base file is `requirements-kaggle.txt`;
it does not request Torch, so it avoids intentionally replacing Kaggle's CUDA
runtime. The local `requirements.txt` remains available for local development.

If scientific imports/wfdb fail, set `INSTALL_BASE=True` and execute section 2.
Its exact command is:

```python
subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                str(REPO_DIR / "requirements-kaggle.txt")], check=True)
```

Section 3 prints Python, Torch, compiled CUDA, CXX11 ABI, GPU capability and
package versions. CPU-only Torch or no visible GPU must be fixed before Mamba
smoke. After any binary package installation restart the kernel and rerun with
installation flags False; otherwise already imported libraries can be stale.

For Mamba install set `INSTALL_MAMBA=True` in section 1. Section 4 requires
CUDA before installing. With Internet/build tools, it runs:

```python
build_env = os.environ.copy()
build_env["MAMBA_KEEP_CUDA_BUILD"] = "TRUE"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                "causal-conv1d>=1.4.0", "mamba-ssm"], check=True, env=build_env)
```

Current upstream defaults may omit `selective_scan_cuda`; the flag opts into
the CUDA extension. Follow [the official install guidance](https://github.com/state-spaces/mamba#installation).
The base setup includes packaging/ninja; compiling still needs a compatible
CUDA toolkit/compiler. If a core-only Mamba package already satisfies pip,
replace it with a matching CUDA wheel or explicitly rebuild it with the CUDA
opt-in, then restart. Imports alone do not prove a working kernel.

Offline, attach matching **Linux/Python/Torch/CUDA/CXX11 ABI** wheel files and
set `WHEEL_DIR` to their directory. Section 4 uses:

```python
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                f"--find-links={WHEEL_DIR}", "causal-conv1d", "mamba-ssm"], check=True)
```

Include transitive dependency wheels if absent from Kaggle. No universal wheel
is assumed. Pip logs are streamed, with the failed command and exit status;
keep the log when diagnosing installation failures.

## Run a short real-data smoke: sections 5–7

Set these controls in section 1, then run through section 7:

```python
RUN_METADATA_AUDIT = True
RUN_SMOKE = True
RUN_SUPERCLASS = False
RUN_SUBCLASS = False
SMOKE_FIT_RECORDS = 16
SMOKE_BATCH_SIZE = 2
```

Section 5 reads metadata only. It verifies:

- Complete five-superclass vocabulary (NORM, MI, STTC, CD, HYP) and all **23**
  official subclasses from the diagnostic mapping. No rare classes drop.
- Binary multi-hot targets for both tasks, counts of unlabeled exclusions,
  class support/imbalance for every supervised split.
- Official folds 1–8 / 9 / 10, pairwise patient separation before exclusions,
  unique ECG IDs, present patient IDs and valid folds.
- Metadata hashes and a release-provenance reminder. Audits are exported to
  `metadata_audit.json` and both task class-statistics CSVs.

Section 6 freezes the reference architecture and full settings. In section 7,
each task reads 16 labeled **training** ECGs for a smoke-only normalizer, then
one tiny training batch and one validation batch. Overlapping reads may occur.
WFDB checks header frequency at 100 Hz; the original raw arrays must be
`(12,1000)`. Nonfinite/corrupt/missing files fail with ECG ID and path.
The shared filter/normalization pipeline is used. Augmentation is off for smoke.

The next explicit cell initializes a fresh preserved Mamba for each task and
performs one AdamW step using multilabel BCE, checks finite
nonzero gradients, clips gradients, and performs a validation forward. AMP is
enabled by the experiment setting. It writes:

```text
btp_outputs/smoke/
  status.json                # started/failed/passed; benchmark_result=false
  environment.json
  config.json
  superclass_smoke.json      # shapes, loss, IDs, normalization, gradient evidence
  subclass_smoke.json
```

Smoke BCE is unweighted and unsmoothed to isolate kernel/backward validation;
full training retains reference class weighting and training-only smoothing.
**No fold-10 waveforms are read, no thresholds are tuned, and no benchmark
accuracy is produced by smoke.** Metadata includes all folds for separation
checks. A pass covers only the sampled files and one step, not a full data sweep
or stable multi-epoch convergence. Smoke weights are discarded.

## Full runs: sections 8–10

After smoke passes in the current session, enable `RUN_SUPERCLASS=True` and/or
`RUN_SUBCLASS=True`. The full cells are independent and require their task's
current CUDA smoke report. If model/data settings change, rerun smoke.
For a repeatable Run All, leave metadata and smoke controls True as well.

```python
FULL_EPOCHS = 20
FULL_BATCH_SIZE = 8
RUN_SUPERCLASS = True
RUN_SUBCLASS = True
```

The reference architecture is unchanged: embed/hidden 384, 8 bidirectional
blocks, d_state 16, d_conv 4, expand 2, dropout .45, mean-attention pooling.
Batch 8 is an explicit memory adjustment from reference batch 64. Epochs/batch
are read from `CONFIG`; cells do not silently override the central settings.

Each full run initializes fresh weights, fits normalization on **all** official
training records before label exclusions, trains on full folds 1–8, validates
on 9, selects its best checkpoint/thresholds on validation, and evaluates held-out
fold 10. A full run with `FULL_EPOCHS=1` is a one-epoch full-dataset run, **not**
the short sampled smoke. These calls are in the explicit task cells:

```python
superclass_result = run_experiment(model_name="mamba", task="superclass", config=CONFIG)
subclass_result = run_experiment(model_name="mamba", task="subclass", config=CONFIG)
```

Each unique output directory contains config, environment, data manifest,
class statistics, best/last checkpoints, training history, validation thresholds,
summary timing/parameters, and validation/test/fixed-.5 predictions and metrics.
Section 10 verifies all expected files exist and displays loss curves. It
checks only enabled full runs and fails on incomplete exports. Save these
folders as Kaggle notebook output for later comparison.

## Common failures and what to retain

| Failure | Next step |
| --- | --- |
| No/ambiguous source or data root | Inspect candidates; attach extracted folders and set explicit overrides |
| Missing `records100`/`.hea`/`.dat` | Attach the low-resolution waveforms; preserve the failing ECG ID/path |
| WFDB sampling/shape mismatch | Confirm 100 Hz filenames and original `(1000,12)` WFDB arrays; do not silently relabel 500 Hz data |
| CUDA unavailable | Enable Kaggle GPU, restart, inspect Torch CUDA build and session GPU availability |
| `undefined symbol`, extension/import/build error | Match wheel ABI/Torch/CUDA/Python, keep pip/runtime logs, restart after replacement |
| CUDA out of memory | Restart to release failed allocations; smoke batch 1, smaller declared full batch; retain the reference architecture |
| Nonfinite loss/gradients | Stop; retain task, ECG IDs, config and traceback. Disable AMP as an explicitly recorded diagnostic and rerun smoke |
| Patient overlap/invalid labels | Stop and verify metadata release/integrity; do not replace official folds with random splits |
| Missing export | Retain output folder and traceback; do not claim a completed run |

Bring back metadata audit/hash files, smoke reports, runtime/install logs and
complete full-run output folders. Full dataset integrity, real Mamba CUDA
forward/backward, GPU memory feasibility, AMP stability and both full training
results still require actual Kaggle execution. Synthetic local tests and a
generated WFDB roundtrip do not establish real PTB-XL performance.
