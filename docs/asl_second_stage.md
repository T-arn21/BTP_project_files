# ASL second-stage SSL Mamba fine-tuning

This is a new five-epoch optimization stage initialized from the **entire**
validation-selected classifier in each completed epoch-15 experiment. It loads
backbone, attention pooling and classification head with strict tensor keys/shapes.
It does not perform an SSL-only transfer, create fresh pooling/heads, resume old
AdamW moments or inherit the completed/frozen BCE scheduler.

## Verified source inputs

The source best checkpoints and completion sidecars were inspected read-only
from local results_5. Source epochs are 15, selection epochs are 7 and 14, and
canonical source vocabulary sizes are 5 and 23. The source history's original
validation macro AUROC/min_delta selection is replayed before any training.

The user confirmed these exact Kaggle paths:

```python
ASL_SUPERCLASS_CHECKPOINT = (
    "/kaggle/input/datasets/arnavvt/checkpoint-2/"
    "btp_outputs/sessions/20261030T060028Z_a420e362/"
    "ssl_finetuning/resumed/benchmark/"
    "mamba_superclass_20261030T060143Z_006ea4fa/best.pth"
)
ASL_SUBCLASS_CHECKPOINT = (
    "/kaggle/input/datasets/arnavvt/checkpoint-2/"
    "btp_outputs/sessions/20261030T060028Z_a420e362/"
    "ssl_finetuning/resumed/benchmark/"
    "mamba_subclass_20261030T095134Z_8047172d/best.pth"
)
```

Each parent folder also needs config.json, summary.json, status.json and
history.csv. Do not select resume_source/best.pth: this is explicitly rejected.
Model SHA256 is checked before weights_only deserialization:

| Task | Best epoch | SHA256 |
|---|---:|---|
| superclass | 7 | 4f2fba212204d399a1f8dc79ae9e22d9d0d879eba88471d39e19b0a61d9c9c94 |
| subclass | 14 | 85abb89239b9c7a9a15119cc93f11e66308324d3f800f8d2374db4770e028380 |

Keep the original inputs required by the saved source configs attached:

- PTB-XL 1.0.1 at
  `/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`.
- Original SSL input `arnavtiku/checkpoint`, with encoder.pth/config.json/summary.json at
  `btp_outputs/sessions/20261028T191718Z_e9445a9d/ssl/pretraining/ssl_mamba_20261028T201304Z_e1341eb8/`;
  encoder SHA256 remains 8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4.
- The same input's original full training cache at
  `btp_outputs/cache/ssl/signals_fa14baaf8ce23a18d5778e2d/signals.npy`, with adjacent
  manifest.json, payload SHA a507c0a89f768485ffb13ff2cd3a538867669933baa2bc1861d35a95d987b9c3.
- Optional compatible Mamba/causal-conv1d wheels to avoid source-build time.

`arnavvt/checkpoint-2` is the full-classifier source. The original SSL input is
still required for data/normalizer provenance, but its encoder weights never
overwrite the loaded classifier. Do not rewrite the source configs' input paths
to point at unrelated exports. Missing/mismatched inputs fail clearly.

## Loss and training semantics

ExperimentConfig.loss_type is explicit: `bce` remains the legacy default;
`asl` enables multilabel ASL. Configuration rejects ASL unless BCE pos_weight
is explicitly disabled, label_smoothing is zero and mixed_precision is False.
Trainer also rejects a supplied BCE positive-weight tensor on this path.

ASL uses gamma_pos=0, gamma_neg=4, clip=0.05. Positive log probabilities use
logsigmoid(logits); negative probabilities use sigmoid(-logits), add the clip
and cap at 1 before taking the log. With zero clip, negative logs use
logsigmoid(-logits), avoiding log(sigmoid) underflow. Asymmetric focusing weights
are detached, matching the authors' standard AsymmetricLoss variant with
`disable_torch_grad_focal_loss=True`; reduction is mean over batch and labels.
The implementation never changes global autograd mode. See the
[authors' loss implementation](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py).
This variant/reduction is recorded in numerical_settings.json. Raw BCE and ASL
loss values are different objectives; compare AUROC/AUPRC/F1, not cross-loss magnitudes.

Every loss computation disables autocast and uses float32. Existing finite
input/target/logit/loss/gradient/optimizer checks, diagnostic ECG IDs and
nonfinite-rejecting gradient clipping remain active. The existing weighted BCE
computation and smoothing behavior are unchanged for loss_type=bce. Legacy
checkpoint configs gain only documented default values; BCE/ASL configs cannot
silently compare equal during exact resume.

The new recipe is five full-data epochs, batch2, FP32, a NEW AdamW optimizer
at LR 1e-5, a NEW cosine schedule with T_max=5 and eta_min=1e-6, and no warmup.
Weight decay, augmentation (False in these source runs), architecture, seed and
other non-recipe settings are inherited unchanged. Optimizer/scaler/RNG/history,
best-score and early-stopping state are fresh for this separate stage. All model
parameters remain trainable. Validation macro AUROC selects the stage's best
checkpoint with the inherited min_delta; test metrics never select checkpoints.

## Provenance without redundant expensive audits

Full-classifier initialization requires an explicit mode plus classifier hash,
expected epoch and the retained SSL encoder/cache pins. The old prohibition on
mixing arbitrary initial_checkpoint and ssl_encoder_checkpoint remains active
for ordinary weights-only initialization. Only `ssl_full_classifier` permits the
verified combination, with SSL-cache-backed data and all remaining checks intact.

A standard-library path/sidecar preflight checks both classifier exports and
the saved SSL/PTB-XL input paths before CUDA installation or probes. Binary
hash, vocabulary and validation selection checks follow once dependencies are
ready, before either task trains. Model/data/seed
configuration and exact class order, metadata, original normalizer, official
patient-disjoint folds, complete training/held-out IDs and cache provenance must
match the source checkpoint. Only explicitly declared optimization/loss and
initialization/output fields may differ. A non-recipe change rejects; no
provenance audit or compatibility waiver is requested from the user.

Training ECGs come directly from the original authenticated SSL cache; only
held-out ECGs need the existing preprocessing/cache path. No 17,441-record
waveform reconstruction/equivalence check or normalizer refit occurs.

A fresh execution scope shares the successful cache SHA across both tasks **only
on a verified read-only filesystem**, keyed by file identity/stat, manifest byte
hash, payload pin and expected shape. Manifest/ID/normalizer checks still run.
A changed file/manifest invalidates the memo. Writable/local files are rehashed:
stat metadata alone is not proof that a writable memory map was unchanged.
No authentication result persists across executions, and saved reports are not
accepted as authentication tokens. This change also avoids repeated payload
hashes within a normal experiment on immutable Kaggle inputs.

## Exact notebook configuration

The updated first cell is preconfigured for these two explicit task flags; no
GPU work occurs until the user starts Run All / Save & Run All:

```python
RUN_ASL_SUPERCLASS = True
RUN_ASL_SUBCLASS = True
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False
RUN_SSL_FINETUNE_SMOKE = False
RUN_SSL_FINETUNE_SUPERCLASS = RUN_SSL_FINETUNE_SUBCLASS = False
RUN_SSL_FINETUNE_RESUME = False
SSL_FINETUNE_PROVENANCE_ONLY = False
DIAGNOSTIC_MODE = False
ASL_BEST_CHECKPOINTS = {"superclass": ASL_SUPERCLASS_CHECKPOINT, "subclass": ASL_SUBCLASS_CHECKPOINT}
ASL_SOURCE_BEST_EPOCHS = {"superclass": 7, "subclass": 14}
ASL_STAGE_SETTINGS = dict(
    epochs=5, learning_rate=1e-5, min_lr=1e-6, batch_size=2,
    mixed_precision=False, warmup_epochs=0, loss_type="asl",
    label_smoothing=0., use_pos_weight=False,
    asl_gamma_pos=0., asl_gamma_neg=4., asl_clip=.05, diagnostic=False,
)
```

Paths/hashes above are already pinned in the cell. Source model/data settings
are inherited, not replaced by unrelated legacy smoke/full settings. Set both
RUN_ASL flags False to opt out. Other training/diagnostic flags cannot be mixed
with this stage. Select a GPU, enable Internet and GITHUB_TOKEN, attach inputs,
and review the cell before Run All. Setup may probe CUDA/Mamba functionality;
no diagnostic ECG experiment or separate GPU smoke run is launched.

## Outputs, comparisons and duration

Exactly one experiment directory per task is created under:
`/kaggle/working/btp_outputs/sessions/<NEW>/asl_stage2/benchmark/mamba_<task>_<id>/`.

Outputs include best.pth/last.pth, config/environment/data_manifest, history.csv,
full_classifier_initialization.json with source pins/all transferred names/reset
states, numerical settings, training curves, thresholds.json and summary.json.
There are validation and test metrics, per-class CSVs, predictions and confusion
matrices at both fixed 0.5 and validation-tuned thresholds. Thresholds are fitted
on validation only. threshold_comparison.csv places the four evaluations side by
side (macro/micro AUROC, AUPRC, F1 and other reported metrics). Loss curves are
labeled ASL. Session final_summary.json reports each task's completion and path.
Source test metrics are recorded as historical references, not selection inputs.
Old results and checkpoints are not changed.

Reported reference test results to compare after execution:

| Task | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|
| superclass | 0.9156 | 0.8053 | 0.7536 |
| subclass | 0.8525 | 0.4375 | 0.4324 |

The epoch-15 source session used a Tesla T4 and took about 13,130 seconds for ten
additional epochs per task, about 21.9 minutes/epoch. Ten ASL epochs total across
the two tasks therefore estimate **3.5-4.5 GPU hours**, plus dependency setup
(particularly any CUDA source build). This estimate is based on the prior run,
not an executed ASL benchmark. ASL may improve or worsen metrics; evaluate the new
validation-selected checkpoints against preserved baselines. Library/input
provenance differences still fail closed. No real GPU smoke or GPU training was
launched while preparing this implementation.

## November 2 ASL provenance repair and rerun

Session `20261102T164449Z_fb365bf6` stopped before epoch 1 because commit
7716f19 used recursive exact equality for the entire full-classifier metadata.
`cache.preprocessing_identity()` hashes each held-out `.hea`/`.dat` relative
name, size **and `st_mtime_ns`** into `raw_file_stats_sha256`; it is not a
waveform content checksum. `prepare_cache_backed_data()` stores that identity
twice (top-level and heldout_preprocessing_identity). `ensure_signal_cache()`
includes the complete identity in its cache key, so a timestamp change also
changes the resulting cache directory. The three reported differences can
therefore have one environmental cause. They do not themselves establish that
only timestamps changed, since file sizes also participate.

Read-only inspection of both original pinned best.pth files established:

- Held-out raw-stat digest: `b702c9e88e9c769efadb046e54242356e34a47024f78521d7c612a1c4d5f2eff`.
- Held-out cache: `/kaggle/working/btp_outputs/cache/signals_3dc5daab9a69e1210fdc127d/signals.npy`.
- Processed held-out payload SHA256: `0c2c7eef0b23c55d456a94a48499f12c20b5bcd2a937359c9b7eb625a1dc0f42`, with 4,319 held-out record IDs.

The failed session was not available locally during this repair. Its reported
exhaustive three-field difference list is consistent with the payload hash
matching; its current data_manifest.json is needed to confirm the actual hash,
path and stat digest independently. Per-file old/new sizes and timestamps would
be needed to attribute the raw-stat change specifically to timestamps. No
claim of raw-file byte equivalence follows from a stat digest.

The new rule is confined to ASL full-classifier initialization and exactly:

1. `provenance.preprocessing_identity.raw_file_stats_sha256`
2. `provenance.ssl_cache_backed.heldout_preprocessing_identity.raw_file_stats_sha256`
3. `provenance.ssl_cache_backed.heldout_cache.path`

Those differences are accepted together or separately ONLY if the processed
held-out cache hash matches the hash inside the SHA-pinned classifier. The
current loader must have verified/built that cache in this execution, with the
same identity, complete ordered held-out IDs, dataset cache bindings and no
file-stat change since verification. Both duplicate identities must agree.
Missing payload proof or an altered payload rejects. The path exception does
not require the old ephemeral working path to exist, and does not apply to the
original training cache, dataset root or arbitrary metadata paths.

All other metadata must match exactly, including dataset version, metadata CSV
hashes, preprocessing source and dependency versions, filter/shape/reader
parameters, original normalizer and fit IDs, training cache payload/manifest,
class order and patient-disjoint split IDs. Source model/config/hash validation
is unchanged. PTB-XL release_verified remains False: a matching declared version
alone is not treated as immutable dataset proof. The authenticated original
training cache plus matching verified processed held-out payload provide signal
identity; current raw files need not be claimed byte-identical.

The verification receipt is runtime-only; a saved receipt/report cannot authorize
compatibility. The rule consumes ensure_signal_cache's existing verification,
without any extra full-cache SHA256 pass, training-waveform reconstruction or
expensive equivalence audit. Existing one-per-execution original-cache
verification and read-only scoped sharing are unchanged. Raw digests and all
original artifacts remain preserved. Exact resume and other validators retain
their existing policies.

For the rerun, refresh `05_btp_kaggle_final.ipynb` from this branch, attach the
same three inputs listed above and compatible wheels if available, select the
same GPU, enable Internet and the GITHUB_TOKEN secret, and use Run All. Retain
**all** configuration values in the Exact notebook configuration section:
RUN_ASL_SUPERCLASS=True, RUN_ASL_SUBCLASS=True, all unrelated training/smoke/
diagnostic flags False; five epochs, batch2, FP32, LR1e-5/min1e-6 and unchanged
ASL gamma/clip settings. Both source checkpoint paths, epochs and SHA256 pins
remain unchanged. The synchronized checkout should show the repair commit
before the worker starts. No manual provenance waiver or extra audit is needed.

Each new task folder saves `asl_classifier_provenance_comparison.json` on
success or rejection, with expected/actual mismatches, accepted flags, guard
results and both held-out payload hashes. Success is `passed_exact` or
`passed_verified_heldout_cache_relocation`; the latter has zero blocking
mismatches. Any other mismatch or missing content proof stops before epoch 1.
The success report is also embedded in full_classifier_initialization.json.
If a rerun rejects, retain that JSON plus data_manifest.json and
checkpoint_failure.json from the failed task folder; no new GPU experiment is
needed to interpret them. Output roots and the two-task duration estimate
above are unchanged. No GPU training was executed while preparing this repair.
