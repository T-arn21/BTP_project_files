# Numerical stability: verification and recovery

Update: the failed-run artifacts have now been inspected read-only. Checkpoints
are finite at superclass epoch 3 and subclass epoch 5, while failures occurred in
epoch 4 training and epoch 6 validation. The next notebook defaults select these
weights explicitly with strict provenance checks. See
[controlled continuation](checkpoint_continuation.md) for current settings;
the fresh-run diagnostic settings below apply only after clearing the automatic
CONTINUATION_SESSION_ID selection.

The user reports the prior Kaggle run passed synchronization, installation,
metadata/patient audit, both smoke tests and caching, then both tasks failed
with `RuntimeError: Nonfinite BCE loss` after substantial full-data GPU training.
That is failed training, not a completed benchmark. No failed-run logs/checkpoints
were supplied when the guards were first implemented; subsequent inspection
still cannot establish the originating numerical operation. It is **unverified**.

## Findings and scope

Confirmed by source inspection: the previous trainer computed BCE inside
autocast, checked only the resulting loss, and clipped without
`error_if_nonfinite=True`. It did not identify whether invalid inputs/logits,
gradients or an earlier update caused failure. CUDA autocast already promotes
[BCE-with-logits to float32](https://docs.pytorch.org/docs/2.11/amp.html#cuda-ops-that-can-autocast-to-float32).
Therefore its previous location alone is not evidence that BCE ran in FP16.
Casting an already nonfinite forward output to FP32 cannot repair it.

Suspected causes include FP16 overflow in the forward/backward path, an invalid
earlier gradient/update, or unusually large normalized signals. None is confirmed
for the user's run. The new reports distinguish the first invalid boundary;
they do not identify the exact internal Mamba operation.

The preserved Mamba architecture, head, parameter names and forward path are
unchanged. Preprocessing/cache implementations are unchanged: raw waveforms are
checked for finite values, SOS filtering and streaming FP64 training-only
normalization remain, with the existing 1e-8 standard-deviation floor. Positive
class weights still use the reference negative/positive ratio capped at 8 and
neutral weight for zero-positive columns. Invalid/nonbinary class-weight inputs
are now rejected. There is no amplitude clipping, NaN replacement, skipped ECG,
batch retry, automatic precision fallback or changed-setting restart.

## Guards and evidence

- Forward uses the explicitly selected AMP setting. Training, validation and
  real-data smoke compute BCE with float32 operands and autocast disabled.
- Every train/validation batch checks inputs, binary targets, smoothed targets,
  logits and loss. Gradients are checked after scaler unscale, then
  [clipped with nonfinite rejection](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html).
  AMP gradient overflow aborts rather than silently skipping an update.
- Model parameters, optimizer tensors, scaler and learning rates are checked
  on initialization/restoration and immediately after each optimizer update.
  A corrupt update aborts before the next batch or checkpoint; it is not rolled
  back in memory. Previous completed epoch checkpoints remain available.
- `numerical_settings.json` records effective AMP, configured LR, FP32 loss,
  clipping, AdamW epsilon, class weights, smoothing and abort policy.
- On failure, `numerical_failure.json` and
  `diagnostics/epoch_XXXX_<train|validation>_batch_XXXXXX_<phase>.json` record
  epoch/batch (one-based during training), ECG IDs and affected IDs, tensor finite
  status/counts, input/logit bounds, gradient/parameter/optimizer status, scaler
  scale, actual LR, normalization statistics and whether an update made parameters
  nonfinite. Unavailable values are marked; partially nonfinite tensors use
  `finite_min`/`finite_max` for their finite subset. `run.log`, `status.json` and
  the session's `final_summary.json` retain the error and completed work.

Both `strict` and `conservative` enable all guards. `conservative` additionally
requires explicit `MIXED_PRECISION=False` and LR <= 1e-4. Conflicting settings
raise an error; mode selection never changes them silently.

## Safest first Kaggle run

Import the updated **05_btp_kaggle_final.ipynb**, enable GPU/Internet/secret,
attach the same PTB-XL release and compatible wheels, and configure only its
first cell. The new defaults are:

```python
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = True
NUMERICAL_STABILITY_MODE = "conservative"
MIXED_PRECISION = False
FULL_BATCH_SIZE = SMOKE_BATCH_SIZE = 2
FULL_EPOCHS = 1
DIAGNOSTIC_MODE = True
DIAGNOSTIC_BATCHES = 2
NUM_WORKERS = 0
RESUME_RUN_DIRS = {"superclass": "", "subclass": ""}
INITIAL_CHECKPOINTS = {"superclass": "", "subclass": ""}
# In the existing OPTIMIZER_SETTINGS dictionary: learning_rate=1e-4
```

Keep other preprocessing/normalization/class-weight/augmentation settings and
the full architecture unchanged. Run All still executes unattended. This first
run performs both smoke tests and short diagnostic training/validation with
`benchmark_result=False`, no test evaluation. A short pass cannot rule out a
later failure on another ECG or epoch.

After that passes, make a **second explicit Run All** with
`DIAGNOSTIC_MODE=False`, `FULL_EPOCHS=1` and the same FP32/LR/batch settings to
exercise every training/validation ECG. This is a deliberately one-epoch budget,
not a claim that a longer benchmark finished. Only then choose the intended
full epoch budget and early stopping. No training-data truncation occurs in full
mode. FP32 may require more GPU memory; if it OOMs, explicitly lower both batch
sizes to 1 and record the change. Do not shrink the architecture.

Review each run's `numerical_settings.json`, history and session
`final_summary.json`. Both tasks must be complete with their exports; a failed
or diagnostic run is not a complete benchmark. Copy/save Kaggle output artifacts
before a session is destroyed. Corrected GPU training is not executed locally.

## Existing checkpoints, cache and wheels

Exact resume uses `RESUME_RUN_DIRS[task]` with its original writable run folder,
`last.pth`, `best.pth`, configs/manifests and identical settings/provenance.
Legacy configs missing only the new mode and initialization fields are compared
as `strict` and `None`. To resume an old AMP run exactly, explicitly restore its
original AMP/LR/budget/batch settings and select strict mode. This can reproduce
its instability; prefer the conservative checks first. Loaded model tensors and
restored optimizer/scaler state must be finite.

To reuse useful weights **while changing AMP/LR**, keep that task's
RESUME_RUN_DIRS empty and set, for example:

```python
INITIAL_CHECKPOINTS["superclass"] = "/kaggle/working/restored/<superclass_run>/best.pth"
INITIAL_CHECKPOINTS["subclass"] = "/kaggle/working/restored/<subclass_run>/best.pth"
```

These are explicit NEW experiments: strict same-task/class-order/state-shape
loading, matching source/preprocessing/normalization/split provenance and finite
weights required, fresh optimizer/scheduler/scaler, epoch count
restarts at one. `initial_checkpoint.json` records the source epoch and old/new
settings. Scores must be labeled as initialized from that previous training;
this is not a fresh-training benchmark or exact optimizer continuation.
Use original data/preprocessing/normalization for meaningful weight reuse.
Nonfinite checkpoints are refused. A finite checkpoint can still be unstable,
so run the same short diagnostic with those selected weights before a full run.

The deterministic cache key is unchanged; reuse CACHE_DIR and the same dataset
root/version/file signatures/settings/package versions. For a new Kaggle session,
restore cache/checkpoints into writable working storage before Run All (the
notebook never writes attached read-only inputs). The existing installer keeps
working CUDA packages, auto-detects attached compatible wheels or uses WHEEL_DIR,
and retains source-build wheels under sessions/<id>/mamba_wheels. Python/Torch/
CUDA/ABI compatibility and actual CUDA backward are still checked; no restart
is assumed. Save matching wheels as an attached Kaggle dataset for later runs.
