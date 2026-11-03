# ECG classification with SSL-pretrained Mamba

PTB-XL experiments using one shared Mamba implementation: supervised BCE,
masked-waveform reconstruction, SSL classifier fine-tuning, completed-run epoch
extension, and ASL second-stage optimization. The notebooks below explain and
select existing `src/` execution paths; they contain no copied model/trainer code.

## Experiment order

| Order | Notebook | Inputs and purpose | Default |
|---|---|---|---|
| 1 | [Supervised baseline](01_supervised_baseline.ipynb) | PTB-XL; fresh five/23-class BCE classifiers | Smoke only, execution off |
| 2 | [SSL pretraining](02_ssl_pretraining.ipynb) | Training-fold ECGs only; masked reconstruction, no diagnostic labels | Smoke only, execution off |
| 3 | [SSL fine-tuning](03_ssl_finetuning.ipynb) | Completed encoder + original cache/normalizer; fresh pooling/heads | Both task smokes, execution off |
| 4 | [Extended fine-tuning](04_ssl_extended_finetuning.ipynb) | Completed full classifiers; restore optimizer/scheduler/RNG/history and extend total epochs | Epoch5 to15, execution off |
| 5 | [ASL fine-tuning](05_asl_finetuning.ipynb) | Epoch15 validation-selected best classifiers; all weights, new AdamW/cosine | Five epochs, execution off |
| 6 | [Evaluation and plots](06_evaluation_and_plots.ipynb) | Saved complete classification prediction exports; CPU re-analysis | Execution off |

Baseline 01 is a comparison, not an input dependency of SSL. Stage02 feeds03,
03 feeds04, and04 feeds05. Stage06 can review any completed classification stage.
Defaults in03-05 reference the confirmed historical Kaggle exports. A newly
trained chain requires replacing paths/hashes with its own authenticated outputs;
notebook numbering alone does not connect artifacts automatically.

Start with [the reproduction guide](docs/reproduction.md). Each notebook includes
methodology, required inputs, settings, execution and output descriptions.
All six start with `EXECUTE=False`, so Run All first permits safe review. For
01-03, set `EXECUTE=True, MODE="smoke"` before opting into `MODE="full"`.
04-05 support full mode only and require explicit execution opt-in. No GPU
training was run as part of this repository organization.

## Kaggle execution

Import a notebook from `refactor/project-organization` while reviewing this branch.
Its `GIT_BRANCH` selects that branch; after merging, explicitly select the reviewed
`main`. Enable Internet, GPU for01-05, and a Kaggle `GITHUB_TOKEN` secret with
repository read permission. Attach the required data/checkpoints and compatible
Mamba wheels when available. Change the first configuration cell, then Run All.
Never print the token or put it in a URL/configuration file.

The standard-library bootstrap securely synchronizes the clean checkout, verifies
source precedence, resolves the stage settings, installs/checks dependencies and
starts fresh controller/worker interpreters. No kernel restart is assumed.
Uncommitted or divergent checkouts reject instead of being overwritten. Profiles
in [src/notebook_workflows.py](src/notebook_workflows.py) select the existing
worker; [configs/kaggle_defaults.json](configs/kaggle_defaults.json) retains the
unified configuration snapshot. Resolved settings and source commit are exported.

[05_btp_kaggle_final.ipynb](05_btp_kaggle_final.ipynb) remains the fully configurable
unified entry point for established workflows, recovery and provenance-only
investigations. All training, smoke, resume and diagnostic flags now default to False.
Manually select the desired experiment before Run All; the inactive default does no GPU work.
See [unified execution](docs/kaggle_execution.md).

Stage06 needs no GPU or CUDA extensions. It recalculates metrics from saved
predictions rather than rerunning inference; its output is explicitly labelled
prediction re-analysis. Local execution uses `requirements.txt`, a local checkout
and `SYNC_REPOSITORY=False`.

## Model and data

The preserved architecture accepts `(batch, 12, 1000)` ECGs: 12-lead projection to
384 features, eight pre-norm residual bidirectional Mamba blocks (forward/reverse
scans concatenated and projected), final LayerNorm, mean plus learned attention
pooling, dropout and an MLP head returning five or 23 logits. State dimension16,
convolution width4, expansion2, hidden width384 and dropout.45 remain unchanged.
SSL uses the same temporal encoder with a linear waveform reconstruction head;
only masked positions contribute to MSE. New classifiers receive fresh pooling
and heads in03; ASL loads the complete existing classifier in05.

Preprocessing is shared: real 100Hz WFDB waveforms, 0.5-40Hz bandpass, center
pad/crop to10seconds and training-derived lead normalization. Unaugmented,
unnormalized processed float32 signals are memory-mapped and cached. Official
folds1-8 /9 /10 define train/validation/test, with patient separation audited
before supervised label exclusions. Labels are multi-hot five superclasses
`NORM, MI, STTC, CD, HYP` or 23 official diagnostic subclasses, including rare
classes and code presence at confidence zero. No test-derived fitting is allowed.

Checkpoint hashes, class order, preprocessing parameters/versions, original
normalizer, cache authentication and split IDs remain enforced. SSL-cache-backed
runs use the authenticated original training cache and normalizer, not independently
processed training waveforms. The ASL validator's narrow held-out stat/path
compatibility rule still requires matching verified processed-signal identity.
See [ASL provenance](docs/asl_second_stage.md) and [cache-backed data](docs/ssl_cache_backed_finetuning.md).

## Repository structure

```text
01_...ipynb through 06_...ipynb    stage-specific execution and explanation
05_btp_kaggle_final.ipynb        retained unified Kaggle entry point
src/                            shared data, model, SSL, training, metrics, safeguards
configs/kaggle_defaults.json     unified settings snapshot for stage selectors
scripts/                        deterministic notebook builder and validation
archive/notebooks/              byte-preserved development/original notebooks
docs/                          current stage/recovery guides and test records
docs/history/                  historical analysis, inventory and legacy execution guide
 tests/                         CPU checks and conditional CUDA checks
```

See [documentation index](docs/README.md) and [archive inventory](archive/notebooks/README.md).
Dataset attribution and fixture licensing remain in [tests/fixtures/README.md](tests/fixtures/README.md).

No trained weights, PTB-XL waveforms, patient metadata or full historical Kaggle
outputs are stored in Git. Existing experiment configurations in code/docs,
artifact pins, provenance policies and Git history are preserved; no history was
rewritten. Downloaded/input artifacts are never cleanup targets.

## Outputs and evidence

Training writes new session folders under `/kaggle/working/btp_outputs/`, with
logs/status, source revision, configuration/environment, provenance, checkpoints,
history, metrics, predictions, thresholds and figures. Preserve full output and
cache exports for later stages. Check `final_summary.json` and each task's
`status.json`; smoke/diagnostic completion is not benchmark completion.

Historical outputs/analyses are evidence from their recorded runs. This cleanup
does not reproduce historical GPU results or certify current Kaggle compatibility.
Exact reproductions depend on external artifacts, dataset release, dependency
versions, source paths and CUDA availability; strict mismatches fail closed.

## Local validation

```bash
python -m pip install -r requirements.txt
python scripts/validate_notebooks.py
python -m pytest -q
```

The validator checks schema, syntax, safe-default execution, notebook bootstrap
synchronization and archived notebook hashes; it does not train. Tests use
synthetic CPU data/model doubles for execution paths and conditionally skip real
CUDA kernels. [Test records](docs/test_results.md) distinguish these checks from
GPU evidence. To regenerate maintained stage notebooks, run
`python scripts/build_experiment_notebooks.py`; archived notebooks are never rebuilt.
