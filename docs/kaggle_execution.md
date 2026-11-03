# Unified Kaggle execution

[05_btp_kaggle_final.ipynb](../05_btp_kaggle_final.ipynb) remains the established
fully configurable Run All entry point. For a single explained stage, prefer the
[six experiment notebooks](../README.md) and [handoff guide](reproduction.md).
The unified notebook defaults all training, smoke, resume and diagnostic flags
to False; the six new notebooks default EXECUTE=False. Each interface requires
explicit manual opt-in before it performs work.

## Setup and source

Use `refactor/project-organization` while reviewing this cleanup branch. The
notebook's GIT_BRANCH selects that branch; after merging select reviewed main.
Enable Internet and a Kaggle GITHUB_TOKEN secret with repository read access.
Attach PTB-XL and required stage checkpoints/caches; compatible attached CUDA
wheels are preferred. Secure synchronization clones/fast-forwards only clean
checkouts, verifies imports from that checkout, never prints/saves credentials
and never resets/stashes/cleans local work.

Base dependencies are checked/installed without replacing Torch. Mamba and
causal-conv1d compatibility includes Python/platform tags, Torch minor, CUDA and
CXX11 ABI. WHEEL_DIR and ALLOW_SOURCE_BUILD control wheel discovery and explicit
source fallback. Setup records logs/wheels; incompatible GPU/binary environments
fail clearly. Fresh controller and worker interpreters avoid a kernel restart.

## Select exactly one workflow

Edit only the first configuration cell, disable unrelated run controls, and Run
All. The explicit flag sets are documented in the detailed guide for each stage:

| Workflow | Flags/guide |
|---|---|
| Fresh baseline | RUN_SMOKE, RUN_METADATA_AUDIT, RUN_SUPERCLASS, RUN_SUBCLASS; clear CONTINUATION_SESSION_ID and checkpoint mappings |
| Masked reconstruction | RUN_SSL_SMOKE; optionally RUN_SSL_PRETRAIN after same-run smoke; [SSL](ssl_pretraining.md) |
| SSL classifier transfer | RUN_SSL_FINETUNE_SMOKE; optionally both full task flags; [fine-tuning](ssl_finetuning.md) |
| Completed epoch extension | RUN_SSL_FINETUNE_RESUME exclusively; [resume](ssl_completed_resume.md) |
| ASL | RUN_ASL_SUPERCLASS and RUN_ASL_SUBCLASS exclusively; [ASL](asl_second_stage.md) |
| CPU provenance-only | RUN_SSL_FINETUNE_SMOKE + SSL_FINETUNE_PROVENANCE_ONLY, all other run flags False; [diagnostics](ssl_provenance_diagnostics.md) |

The stage notebooks select these flags programmatically through
src/notebook_workflows.py. They do not copy the worker or change provenance.
Historical weights-only continuation is a separate deliberate operation; see
[checkpoint continuation](checkpoint_continuation.md). It never falls back to
fresh training when a required source is missing.

## Failures, outputs and recovery

Review pipeline.log, worker.log, stage setup logs, final_summary.json and each
experiment's run.log/status.json. Training exports configurations, environment,
source revision, data/split/normalizer provenance, best/last checkpoints, epoch
history, figures, predictions and metrics. Failures preserve diagnostic files and
already completed artifacts. CUDA/installation failure is not training success;
smoke success is not a completed benchmark.

All new output lives below `/kaggle/working/btp_outputs/`. Atomic intermediate
checkpoint/history writes limit interruption losses, but a destroyed Kaggle
session can lose working storage. Save/download full outputs and the original
cache before another session. Exact resume requires its original settings and
provenance; epoch-extension and new-optimizer ASL are explicitly separate modes.

Historical low-level setup/recovery details remain in
[the archived previous guide](history/kaggle_execution_legacy.md). They are
historical configuration examples, not current defaults. No new GPU benchmark
was run during repository organization.
