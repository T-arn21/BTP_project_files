# Explicit SSL-cache-backed Mamba fine-tuning

Current notebook task defaults now select [ASL second-stage fine-tuning](asl_second_stage.md).
Turn both RUN_ASL_* flags off before selecting the workflow in this guide.

## Experiment definition and strict boundaries

`SSL_FINETUNE_DATA_MODE="ssl_cache_backed"` explicitly selects original SSL
artifacts as the training input source. This resolves the independently processed
CPU waveform drift by **changing the declared experiment**, not by declaring
those differing waveforms equivalent. The existing `independent` mode, strict
normalizer checks, content proof and optional verified normalizer migration
remain separate and unchanged.

The completed epoch-20 encoder is hashed before weights-only CPU deserialization;
config.json and summary.json must agree with its architecture, full completion
and train-only provenance. Optional history/status/split provenance must agree
when present. The original cache manifest must exactly match the original
identity, sorted complete eligible/used training IDs and embedded payload pin.
The actual NPY file must match that SHA256, shape and float32 dtype. Smoke/subset
caches, corrupted or unauthenticated artifacts fail without fallback.

The model is unchanged. Only trained Mamba backbone tensors transfer; attention
pooling and 5/23-class heads initialize fresh. All parameters remain trainable.
The original finite normalizer is loaded exactly from the authenticated encoder.
It is never estimated from a new sample, validation or test data.

The current dataset root/version, metadata CSV hashes, preprocessing source hash,
reader, NumPy/SciPy/WFDB versions, data/model configuration, all official training
IDs (including unlabeled SSL ECGs), excluded validation/test IDs, training patient
IDs, training folds, normalization fit IDs/configuration and source train-only
assertions must match. Official patient separation is audited before exclusions.
Labels are built from the same hashed metadata/SCP statements and the existing
canonical 5-superclass/23-subclass mapping, and checked against each dataset's
labels. Supervised IDs must be the full official labeled splits, or the exact
configured diagnostic prefixes. Arbitrary reductions/mixed-patient splits reject.

Training datasets read only the original cache, with the authenticated row-to-ID
mapping. Validation/test datasets process their respective current raw ECGs with
the original checked settings and original normalizer. They never read the SSL
cache and never fit normalization. Optional deterministic held-out caches cover
only requested held-out records. Training augmentation is separately controlled
and recorded; it is off for this smoke. No 17,441-record WFDB loading/rebuild,
normalizer fitting or independent content-equivalence audit is run. Authentication
still streams the original ~0.84 GB NPY checksum; that integrity check is required.

The original raw-file stat digest is retained in the authenticated source identity.
Current held-out raw signatures are recorded with their own population scope;
they are not compared as though they were training signatures. Original-training
identity is checked against the source manifest, while non-stat current
preprocessing configuration/metadata/version fields remain exact checks.

## Required Kaggle inputs

1. The same PTB-XL 1.0.1 input used for SSL, including metadata, SCP statements
   and 100 Hz raw files for validation/test. Default expected root:
   `/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`.
2. `arnavtiku/checkpoint`, with the epoch-20 export:
   `/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/sessions/20261028T191718Z_e9445a9d/ssl/pretraining/ssl_mamba_20261028T201304Z_e1341eb8/encoder.pth`,
   plus adjacent **config.json and summary.json** (retain optional sidecars too).
   Encoder SHA256:
   `8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4`.
3. The original full cache in that input:
   `/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/cache/ssl/signals_fa14baaf8ce23a18d5778e2d/signals.npy`,
   plus adjacent **manifest.json**. Payload SHA256 bound inside the encoder:
   `a507c0a89f768485ffb13ff2cd3a538867669933baa2bc1861d35a95d987b9c3`.
   The file is 837,168,128 bytes and covers 17,441 training ECGs. Never use ssl/smoke.
4. Optional compatible Mamba/causal-conv1d wheels. This is a **GPU** smoke; select
   a Kaggle GPU, enable Internet and the GITHUB_TOKEN secret. The existing setup
   probes CUDA/Mamba and uses compatible wheels or the explicitly configured
   source-build fallback without replacing Kaggle's Torch.

No separate normalization JSON is trusted or required: its authoritative exact
values are inside the pinned encoder. All source files remain read-only.

## Exact first-cell smoke configuration

All unified-notebook execution flags default to False. To opt into this smoke
workflow, manually set the following first-cell configuration:

```python
RUN_ASL_SUPERCLASS = RUN_ASL_SUBCLASS = False
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False
RUN_SSL_FINETUNE_SMOKE = True
RUN_SSL_FINETUNE_SUPERCLASS = RUN_SSL_FINETUNE_SUBCLASS = False
SSL_FINETUNE_PROVENANCE_ONLY = False
SSL_FINETUNE_DATA_MODE = "ssl_cache_backed"
SSL_FINETUNE_NORMALIZER_POLICY = "strict"  # remaining fields exact; source normalizer mandatory in this mode
SSL_FINETUNE_REFERENCE_CACHE = "/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/cache/ssl/signals_fa14baaf8ce23a18d5778e2d/signals.npy"
SSL_FINETUNE_ENCODER_SHA256 = "8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4"
AUGMENT = False
SSL_FINETUNE_SETTINGS = dict(
    epochs=20, batch_size=2, diagnostic_batches=2, mixed_precision=False,
    learning_rate=1e-4, numerical_stability_mode="conservative", log_every_batches=1,
    weight_decay=5e-4, warmup_epochs=3, min_lr=1e-6, label_smoothing=0.05,
    use_pos_weight=True, pos_weight_max=8.0, grad_clip_norm=1.0, patience=15,
)
```

The shared encoder path is already pinned as above. Leave source model/data
settings unchanged. `epochs=20` is a later full-run budget only: each smoke forces
**one epoch**, four training records (two batches) and four validation records
(two batches), with no test evaluation. Each backward validates finite/nonzero
backbone, attention pooling and head gradients, FP32 BCE, nonfinite-rejecting
clipping and finite parameters/optimizer state. AMP remains disabled.

Use Run All / Save & Run All. Both task smokes run sequentially. A failure aborts
rather than changing data sources, precision or training settings. No full task
is enabled. Do not turn on either full flag until the real GPU smoke outputs
have been reviewed. No Kaggle training has been launched from this implementation.

## Evidence and outputs

New outputs are isolated under:
`/kaggle/working/btp_outputs/sessions/<NEW>/ssl_finetuning/smoke/diagnostic/mamba_<task>_<id>/`.

`ssl_provenance_comparison.json` must say
`status="passed_authenticated_ssl_cache_backed"` and
`experiment_kind="ssl_cache_backed_finetuning"`. Its explicit
`independent_training_waveform_equivalence` flags are **performed=False,
verified=False**; there is no byte-reproduction claim. `ssl_cache_backed.json`,
`ssl_initialization.json`, data_manifest.json, config.json, new checkpoints and
summary.json record the artifact-backed source, original normalizer, original
training identity/payload pin, held-out identity/IDs, fresh head/pooling and mode.
`ssl_smoke_gradients.json` records gradient checks. Existing diagnostics, loss
curves, history and diagnostic-validation predictions/metrics remain available.
The session final_summary.json lists each smoke's completion/failure and path;
`benchmark_result=False` remains explicit. Old benchmarks and SSL artifacts are
never overwritten.

To return to independent CPU auditing, explicitly set
`SSL_FINETUNE_DATA_MODE="independent"`, `SSL_FINETUNE_PROVENANCE_ONLY=True`,
leave the smoke selector True and all full/baseline/SSL flags False, and select
Accelerator None. The old byte-equivalence requirements remain in force there.

Completed full runs can separately [extend their epoch limit with strict full-state resume](ssl_completed_resume.md).
The default notebook resume flag is False; it is mutually exclusive with these smoke flags.
