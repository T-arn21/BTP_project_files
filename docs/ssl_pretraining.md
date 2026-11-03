# Milestone 2: masked ECG reconstruction pretraining

The SSL path reuses `MambaEncoder` and the current waveform reading, filtering,
length handling, normalization and cache helpers. It adds a per-timestep linear
reconstruction head, not another encoder architecture. No full real-data SSL
training has been run during implementation. Both SSL notebook switches default
to **False**, preserving the existing supervised workflow.

## Objective and split isolation

Official folds **1-8 only** supply SSL training waveforms, including ECGs without
diagnostic labels. The loader reads only ECG ID, patient ID, fold and waveform
filenames from metadata. It never parses `scp_codes`, builds diagnostic targets,
uses class weights, selects by diagnosis or loads supervised checkpoint weights.
Patient separation is audited across all official folds using metadata; folds 9
and 10 never supply waveform reads, normalization samples or reconstruction loss.
The shared cache identity hashes the metadata/SCP CSV bytes for source provenance
only; these hashes do not supply labels to the learning algorithm.

100 Hz / 10-second / 12-lead input uses the same filtering and center pad/crop.
The target is the **unmasked, preprocessed, training-normalized waveform**,
not an inversion of the raw filtering operation. Other supervised augmentations
are explicitly disabled for SSL. Full-mode normalization uses the configured
training-only sampled/full policy; smoke uses at most `smoke_fit_records`
deterministically selected training ECGs. All fit IDs and mean/std are saved.

One contiguous temporal window per ECG is zeroed on all 12 leads together.
Default mask fraction .25 gives 250 of 1000 samples at 100 Hz. Window length is
rounded and clamped to [1, T-1]. Zero represents the training-normalized mean.
SHA256(seed, epoch, ECG ID) seeds a local PyTorch CPU generator for window starts,
making masks independent of batch order and DataLoader worker scheduling.
Masks repeat for the same seed/epoch/ID and change across epochs; a new random
draw may occasionally choose the same window. No global RNG state is consumed
by masking. Seeded shuffle is reproducible; fused CUDA execution is not promised
to be bitwise reproducible across hardware/runtime versions.

`MambaEncoder.forward_tokens()` exposes the original projection, bidirectional
blocks and final normalization before pooling. The supervised forward calls the
same operations, then its original pooling/head path. Parameter names, dimensions
and supervised checkpoint keys remain unchanged. The reconstruction head maps
each 384-dimensional token to 12 leads. Only masked lead/time values contribute:

```text
loss = sum((prediction - original_normalized_waveform)^2 on masked values)
       / number_of_masked_values
```

The loss operands and reduction run in float32 with autocast disabled. Default
SSL forward is also FP32. Unmasked positions contribute zero loss gradient.
Pooling/attention-pool and supervised classifier weights are not pretrained
by this objective; exports mark `pooling_pretrained=False` and list the encoder
parameters that actually received gradients. The reconstruction head is separate
from any supervised classification head.

## First run: GPU smoke only

Use the updated `05_btp_kaggle_final.ipynb`. Enable GPU/Internet and GitHub secret
access, attach PTB-XL and compatible Mamba wheels if available. Configure the
**first cell only** before Run All:

```python
RUN_SUPERCLASS = False
RUN_SUBCLASS = False
RUN_SMOKE = False              # this is the older supervised smoke switch
RUN_METADATA_AUDIT = False     # SSL performs its own label-free patient/fold audit
RUN_SSL_SMOKE = True
RUN_SSL_PRETRAIN = False

# First-cell SSL_SETTINGS defaults:
SSL_SETTINGS = dict(
    epochs=20,                 # full-mode budget only; smoke always uses one epoch
    batch_size=2,
    learning_rate=1e-4,
    weight_decay=5e-4,
    mask_fraction=0.25,
    grad_clip_norm=1.0,
    mixed_precision=False,
    smoke_batches=1,
    smoke_fit_records=16,
    log_every_batches=20,
    checkpoint_every_batches=500,
)
```

Keep the existing `MODEL_SETTINGS` (384 width, eight bidirectional blocks),
`DATASET_VERSION` matching the attached release, normalization settings and
`SEED`. The full budget `epochs=20` in the dictionary does **not** trigger full
pretraining while RUN_SSL_PRETRAIN=False. Global FULL_EPOCHS, DIAGNOSTIC_MODE,
MIXED_PRECISION and OPTIMIZER_SETTINGS belong to supervised experiments; SSL uses
its separate dictionary. CONTINUATION_SESSION_ID and supervised checkpoint maps
are ignored with both supervised task flags off. SSL starts fresh, without
diagnostic-label knowledge from the supervised checkpoints.

The smoke selects at most batch_size * smoke_batches training records and a
small training-only normalization fit, runs the full preserved Mamba encoder,
masked MSE, backward, nonzero encoder AND reconstruction-head gradient checks,
clipping and real optimizer updates. Input, output, loss, gradients and updated
model/optimizer state must be finite. Clipping uses `error_if_nonfinite=True`.
There is no automatic precision change, skipped batch or retry. Smoke outputs
and encoder weights are marked `smoke_only=True`; they are not full pretraining.

## Later run: explicitly enable full pretraining

After inspecting the real GPU smoke, configure a separate Run All:

```python
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = True
RUN_SSL_PRETRAIN = True
SSL_SETTINGS["epochs"] = 20    # choose the budget explicitly; start with 1 if desired
SSL_SETTINGS["batch_size"] = 2
SSL_SETTINGS["mixed_precision"] = False
```

A fresh smoke must pass in **that execution** before any full SSL training.
Missing/failed smoke blocks full mode and is recorded in the execution summary.
Full mode initializes a fresh encoder/head with the configured seed; smoke
updates are discarded. It visits **all** official training-fold ECGs each epoch,
keeps the final incomplete batch and does not exclude unlabeled diagnoses or
apply hidden batch limits. It runs exactly the configured epoch budget with
fixed AdamW LR; no validation/test waveform is used for early stopping or model
selection. Training masked MSE is not a classification benchmark or a held-out
generalization estimate. Changing masks/dropout also affect epoch-loss comparison.

The default pretraining LR is restricted to <=1e-4 for this conservative path.
Batch/epoch budgets, mask fraction and checkpoint/log intervals are explicit
settings. Choose memory/budget settings before starting; no architecture change,
automatic data reduction or fallback occurs.

## Output files and interruptions

```text
/kaggle/working/btp_outputs/
  cache/ssl/                              # separate training-only signal/statistics caches
  sessions/<NEW_session_id>/
    final_summary.json                    # SSL smoke/full passed, disabled, blocked or failed
    ssl/
      smoke/ssl_mamba_<NEW_run_id>/
      pretraining/ssl_mamba_<NEW_run_id>/
        config.json, environment.json, split_provenance.json
        run.log, status.json, summary.json
        history.csv, loss_curve.png
        encoder.pth                      # encoder-only state + config/provenance/scope
        last.pth                         # full reconstruction model + optimizer/scaler state
        reconstruction_epoch_001.png     # original vs reconstruction, mask shading, 12 leads
        reconstruction_example.npz       # target, prediction, mask, ECG ID from latest epoch
        numerical_failure.json           # on nonfinite failure
        diagnostics/epoch_*_batch_*_<phase>.json
```

Each call creates a new directory; supervised run folders and original downloaded
checkpoints are never outputs. Signal preprocessing is cached once per SSL
variant using only training IDs. Epochs read the cache and apply new masks in the
trainer, without repeated WFDB/filtering. Normalization/caches can be reused when
identity/settings match. Smoke/full scopes may have different cache populations
and normalization budgets, so their cache keys can differ intentionally.

Epoch checkpoints/encoder exports and CSV history use atomic replacement.
`last.pth` also saves intermediate progress every checkpoint_every_batches steps,
with `completed_epoch=False`, epoch/global step, optimizer/scaler and RNG/loader
state. No automatic resume is implemented for SSL in this milestone; a partial
checkpoint is preserved evidence, not a claim that an epoch or budget finished.
Retain Kaggle saved outputs because session storage can be discarded. Long stages
have existing 30-second worker heartbeats plus timed cache/batch progress.

Figures show the first shuffled training example's forward prediction before
that batch's optimizer update, with normalized amplitudes. They are training
reconstruction diagnostics, not validation examples. The raw NPZ permits closer
inspection of exactly the plotted mask/target. Pooling/classification results
are not exported by SSL.

To load an encoder export into a separately instantiated matching MambaEncoder:

```python
state = torch.load(path_to_encoder, map_location="cpu", weights_only=True)
assert not state["smoke_only"]             # require an actual completed pretraining export
# Verify model/data configuration, normalization and split provenance for your experiment.
encoder.load_state_dict(state["encoder_state_dict"], strict=True)
```

This milestone does not automatically inject SSL weights into supervised runs.
Existing supervised checkpoints/configuration and the controlled continuation
procedure remain separate. No classification improvement is claimed before a
future explicitly configured fine-tuning experiment.

## Local verification limits

Tests cover masking reproducibility/contiguity/shared leads, masked-only FP32
loss and zero unmasked gradients, training-only waveform/normalization/cache
access with no diagnostic columns, patient leakage rejection, finite smoke and
nonfinite diagnostics, serialization/exports, and notebook smoke/full gating.
CPU tests use an explicitly identified tiny encoder double for unavailable CUDA
blocks. Supervised backbone/pooling/head formulas are checked for unchanged
outputs and state keys. A conditional real Mamba GPU test remains available.
Actual PTB-XL CUDA smoke and full SSL pretraining are still unexecuted locally;
no generated synthetic figure/loss is a real reconstruction benchmark.
