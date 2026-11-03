# Resume completed SSL-cache-backed runs from epoch 5 to 10

Current notebook task defaults now select [ASL second-stage fine-tuning](asl_second_stage.md).
Turn both RUN_ASL_* flags off before selecting the workflow in this guide.

This is an explicitly selected **full-state resume into new experiment folders**.
It loads the trained classifier from last.pth, including its trained pooling/head;
it does not reinitialize from the SSL encoder or run new diagnostic smokes.
Original benchmark files and epoch-5 best/last checkpoints remain unchanged.

## Strict boundary and scheduler policy

Only total epochs may increase. Output_root is additionally relocated because
new output folders are required. All other ExperimentConfig fields (including
architecture, AMP/LR/batch size/augmentation/seed/patience, data/cache paths and
preprocessing settings) must match. The notebook inherits config.json from each
source run instead of substituting current smoke/full settings. Current live
cache-backed loading still authenticates the original epoch-20 encoder, original
training cache/normalizer, metadata, labels and patient separation. Checkpoint
metadata (including held-out cache hashes/signatures), class order and full split
IDs must then match **exactly**. Changed mount paths, file-stat signatures or
held-out preprocessing payloads reject; no provenance migration is applied.
Keep the original input mounts/version and all saved preprocessing settings.
If a new Kaggle environment changes those values, preserve the rejection report;
this feature does not relax them to make a resume pass.

The source must be completed, non-diagnostic ssl_cache_backed_finetuning at its
original epoch limit, with consistent config.json, summary/status, best.pth,
last.pth and round-trip history.csv. Missing states, inconsistent best checkpoint,
nonfinite model values, incomplete epoch history, exhausted early stopping and
unsupported worker-local augmentation RNG reject. Model keys/shapes load strictly;
optimizer/scaler state is checked before any update. Torch, CUDA, Python, NumPy,
shuffle-generator and applicable augmentation RNG states restore. Worker-local
augmentation from persistent workers cannot be reconstructed from old checkpoints;
source runs with augment=True and num_workers>0 are rejected.

`SSL_FINETUNE_RESUME_SCHEDULER_POLICY="hold_completed_lr"` is required explicitly.
The original warmup/cosine schedule has already finished at epoch 5. Its complete
saved scheduler state and optimizer LR are restored unchanged. For epochs 6-10,
the terminal LR is **held**, not reset, stretched or stepped past the old cosine
horizon (which would increase LR again). For both supplied checkpoints this is
**1e-6**. AdamW moments/step counts continue; no new warmup is introduced. The
scheduler policy/state persists in every new checkpoint, including an exact
resume after interruption. This extension is not equivalent to planning a
ten-epoch cosine schedule at the beginning. Early-stopping history/counter and
best score are preserved, so an inherited stopping condition is never reset.

## Required Kaggle inputs and exact checkpoint locations

Keep the original PTB-XL 1.0.1 and `arnavtiku/checkpoint` inputs used by epoch 5:
original epoch-20 encoder/config/summary and full SSL training cache/manifest.
Their saved absolute paths are inherited unchanged. Encoder pin remains
`8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4`.

Attach the downloaded **results_4/btp_outputs** as an additional Kaggle dataset,
retaining its directory tree. The user confirmed its mount is `/kaggle/input/datasets/arnavvt/checkpoint-2`. The unique
completed session ID is `20261029T180106Z_e154ce34`. Under that input mount:

```text
btp_outputs/sessions/20261029T180106Z_e154ce34/ssl_finetuning/full/benchmark/
  mamba_superclass_20261029T180304Z_bb3fcff5/
  mamba_subclass_20261029T200603Z_991377cb/
```

**Each folder must contain:** last.pth, best.pth, config.json, summary.json,
status.json and history.csv. Preserve all other original results in the dataset.
The original run directories were inspected locally at
`C:/Users/Arnav Tiku/Downloads/results_4/btp_outputs/sessions/20261029T180106Z_e154ce34/`.
Both source checkpoints passed the new offline completion/config/history/best/
scheduler validation against their saved metadata. This does not assert the
next Kaggle environment will reproduce live metadata exactly.

Verified original checkpoint SHA256s:

| Task | File | SHA256 |
|---|---|---|
| superclass | last.pth | fce647c16ac76aa358a5986cd137fc6af0643ededd8b1b45c975142e6780edb6 |
| superclass | best.pth | a26c91c630bcbc40d0f4c3687d43f7c06ea7c15b927fd84faa3fb939354fd923 |
| subclass | last.pth | 095309f697583b50554f66e2be505e36a7775a7240957f93bd1386a330ddfde6 |
| subclass | best.pth | 6c30bbd18e4b81c8de5208fbe72cbe8b16c7de4f7dd9287c18aa11ea20308332 |

## Exact notebook configuration

Update/import the latest 05_btp_kaggle_final.ipynb. Before Run All, edit the
existing assignments before SETTINGS is collected:

```python
RUN_ASL_SUPERCLASS = RUN_ASL_SUBCLASS = False
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False
RUN_SSL_FINETUNE_SMOKE = False
RUN_SSL_FINETUNE_SUPERCLASS = RUN_SSL_FINETUNE_SUBCLASS = False
SSL_FINETUNE_PROVENANCE_ONLY = False
RUN_SSL_FINETUNE_RESUME = True
SSL_FINETUNE_RESUME_SESSION_ID = "20261029T180106Z_e154ce34"
SSL_FINETUNE_RESUME_ARTIFACT_ROOT = "/kaggle/input/datasets/arnavvt/checkpoint-2"
SSL_FINETUNE_RESUME_RUN_DIRS = {
    "superclass": "/kaggle/input/datasets/arnavvt/checkpoint-2/btp_outputs/sessions/20261029T180106Z_e154ce34/ssl_finetuning/full/benchmark/mamba_superclass_20261029T180304Z_bb3fcff5",
    "subclass": "/kaggle/input/datasets/arnavvt/checkpoint-2/btp_outputs/sessions/20261029T180106Z_e154ce34/ssl_finetuning/full/benchmark/mamba_subclass_20261029T200603Z_991377cb",
}
SSL_FINETUNE_RESUME_TO_EPOCHS = 10
SSL_FINETUNE_RESUME_SCHEDULER_POLICY = "hold_completed_lr"
```

Select GPU, Internet and GITHUB_TOKEN access, and attach the inputs above.
A unique session and exactly one full benchmark folder per task are required.
If two attached datasets contain the same session, set ARTIFACT_ROOT to the
specific input mount (or the exact session directory). Alternatively set both
RUN_DIRS values to their full `<input-mount>/btp_outputs/sessions/...` folders;
The supplied notebook now pins both confirmed directories above; clear both
RUN_DIRS values to opt back into discovery. Explicit run directories take precedence
over session discovery. Task mismatches
and missing files reject. The arnavvt/checkpoint-2 mount is user-confirmed; no dataset slug is guessed.

`SSL_FINETUNE_SETTINGS`, FULL_EPOCHS and other smoke/training overrides do not
change resumed settings: the source config is authoritative. Changing only the
resume target to 10 requests five additional epochs, not ten additional epochs.
The default resume flag remains False: no expensive run starts just by importing
the updated notebook. Resume is exclusive with all smoke/new-training/audit flags.

## New outputs and interruption safety

Each task writes under:
`/kaggle/working/btp_outputs/sessions/<NEW>/ssl_finetuning/resumed/benchmark/mamba_<task>_<id>/`.

- `resume_source/` contains hash-verified copies of the six required source files.
  The original epoch-5 best.pth is preserved even if a later epoch improves.
- Working best.pth initially copies that original best. It is replaced only on
  a real improvement; if epoch 5 remains best, that exact checkpoint is retained.
- last.pth and history.csv append epochs 6-10 to the original epoch 1-5 history,
  without duplicates. All normal intermediate checkpoints/results are saved.
- `resume_extension.json` records source file hashes/config, original and target
  epoch limits, restored states and the explicit scheduler policy.
- New summaries identify full-state continuation, final completed epoch, source
  directory/pin and scheduler policy. Metrics/figures/predictions are recomputed
  in the new folder using the best across the entire preserved history.

No original results are moved or edited. An interruption after a new epoch has
saved leaves the new last/best/history and immutable source snapshot. The existing
exact-resume API can continue that new working directory with unchanged config;
it inherits the stored terminal-LR policy. Source runs that already exhausted
patience are rejected instead of silently resetting it.

## Python API

```python
from src.resume_extension import config_from_completed_run
from src.experiments import run_experiment
cfg = config_from_completed_run(source_run_directory, epochs=10, output_root=new_output_root)
run_experiment(config=cfg, resume_completed_from=source_run_directory,
               scheduler_policy="hold_completed_lr")
```

This document is execution preparation only. No Kaggle/GPU training was launched
while implementing or testing resume.
