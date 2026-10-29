# Full training content equivalence for SSL fine-tuning

Current notebook defaults now select [explicit SSL-cache-backed GPU smoke tests](ssl_cache_backed_finetuning.md).
This guide describes the separate independent audit/migration workflow.

The historical `raw_file_stats_sha256` is SHA256 over consecutive UTF-8 strings,
in metadata row order, for each record's `.hea` then `.dat`:
`{relative_filename}{suffix}:{st_size}:{st_mtime_ns}` or
`{relative_filename}{suffix}:missing`. There are no delimiters between records.
It contains filenames, existence, byte sizes and filesystem modification times
(in nanoseconds); it does **not** hash file contents. A timestamp change can
change it without any waveform change. The current failure's exact historical
per-file cause remains unverified.

## Strict, opt-in compatibility gate

`ExperimentConfig.ssl_reference_cache` (notebook `SSL_FINETUNE_REFERENCE_CACHE`)
defaults to None in the Python API. The notebook currently selects the supplied full reference for an audit-only run. Existing behavior remains strict when it is unset.
A content audit is reportable when differences are limited to the stat digest
and/or normalizer values. It does not by itself approve different normalizers.
See [normalizer diagnosis and explicit source reuse](ssl_normalizer_migration.md). Dataset root/version, metadata
hashes, preprocessing source, reader and package versions, model settings,
normalization configuration/fit IDs, every training/held-out ID and
patient separation must all match first. Directory relocation is not combined
with this exception.

The reference must be the original full `signals.npy` plus adjacent
`manifest.json`. Accepted path forms are the `signals.npy` file, its specific
`signals_<key>` entry directory, or the parent `cache/ssl` directory. The parent
form searches immediate `signals_*` entries and requires exactly one manifest
whose IDs, original identity and payload pin match authenticated encoder metadata.
It rejects zero/multiple matches; it never selects the first arbitrary cache.
Each form still checks both files exist and authenticates the payload itself.
The report records the requested path, layout and resolved file/manifest paths.

The previous implementation accepted only the file path. Calling `with_name`
on an entry-directory path looked for `manifest.json` in the parent directory;
this source-level reproduction explains the reported missing `cache/ssl/manifest.json` in session
`20261029T124001Z_81cda7c8`. Directory inputs are now resolved before file access. The manifest's complete training IDs and identity must exactly
match the authenticated encoder provenance. The actual NPY file SHA256 must
match the payload pin **inside the independently SHA256-pinned encoder**; an
untrusted manifest checksum alone cannot authorize acceptance. Smoke/subset
caches, altered arrays, missing files and incompatible manifests fail.

The audit constructs an independent deterministic dataset and prepares/reuses a
checksummed cache for **every official folds 1-8 ECG, including unlabeled ECGs**.
No held-out waveforms, diagnostic limits, augmentation or normalization are used
in this content comparison. Each float32 (12,1000) waveform must be finite and
byte-identical to its reference record. It computes separate SHA256s over a
versioned canonical header (ordered IDs, shape, dtype) and concatenated
little-endian float32 C-order waveform bytes. Every record is explicitly compared;
partial overlap cannot pass. Normalization remains an independent check; only an explicit, fully authenticated
source-normalizer migration can resolve value differences after complete content
proof and an exact same-runtime live-fit reproducibility check.

This proves identical *processed training inputs*, not byte-identical original
WFDB headers or files, nor that timestamps were the only underlying difference.
Existing normal preprocessing caches are keyed by current identity and complete
IDs, with payload SHA256 validation on reuse. Preprocessing prints progress at
least every 100 ECGs/10 seconds; the launcher also emits heartbeats. No old
acceptance report is used as an authorization token: the proof is rechecked at
each task initialization. Data inputs must remain immutable during execution.

The existing comparison report preserves expected/actual stat hashes and marks
that mismatch accepted only after complete proof. It records
`passed_verified_training_content` for strict/stat-only compatibility, or
`passed_verified_content_original_normalizer` for the explicit original-normalizer
policy; checked count, all IDs, canonical hashes,
reference/current payload hashes and identities; failure retains `rejected`,
error and the first mismatched ECG ID when applicable. Both success and failure
are written to `ssl_provenance_comparison.json`. Successful classifier
initialization also records evidence in `ssl_initialization.json`. Source
checkpoint, reference files and old outputs are never rewritten.

## Required Kaggle artifacts and audit-only execution

The downloaded successful run contains the required full reference:
`btp_outputs/cache/ssl/signals_fa14baaf8ce23a18d5778e2d/signals.npy`
(837,168,128 bytes) and its adjacent `manifest.json`.
Its authenticated payload SHA256 is
`a507c0a89f768485ffb13ff2cd3a538867669933baa2bc1861d35a95d987b9c3`.
Keep the epoch-20 encoder/config/summary at the already configured checkpoint
path, with encoder SHA256
`8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4`.

If the Kaggle `arnavtiku/checkpoint` input preserves the downloaded `btp_outputs`
tree including `cache/ssl`, use:

```python
SSL_FINETUNE_REFERENCE_CACHE = "/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/cache/ssl/signals_fa14baaf8ce23a18d5778e2d/signals.npy"
```

The user confirmed both reference files exist at this Kaggle location after the
failed session. This environment has not independently mounted the Kaggle input.
If using a different input layout, attach these **two original cache files**
unchanged and set their actual path. No additional historical raw ECG dataset is required when this full
checkpoint-authenticated processed cache is available alongside current PTB-XL.
Encoder/config/summary plus the 1,004-record overlap are insufficient by themselves.

Edit the first configuration cell before its SETTINGS dictionary is collected:

```python
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False
RUN_SSL_FINETUNE_SMOKE = True
SSL_FINETUNE_PROVENANCE_ONLY = True
RUN_SSL_FINETUNE_SUPERCLASS = RUN_SSL_FINETUNE_SUBCLASS = False
# Set SSL_FINETUNE_REFERENCE_CACHE as above; retain the pinned encoder settings.
```

Set **Kaggle Accelerator = None**, Internet on, and Run All. Exclusive
provenance-only execution now uses CPU for checkpoint loading and data checks;
it does not probe, import, install or build Mamba/causal-conv1d CUDA extensions.
Torch is still required for safe `weights_only=True`, `map_location="cpu"`
checkpoint deserialization. Only missing base scientific packages are installed;
Torch is not replaced. Exact numpy/scipy/WFDB/source-version checks remain active:
if the CPU image differs, the report rejects it rather than relaxing compatibility.
`RUN_SSL_FINETUNE_SMOKE=True` is solely the existing audit selector while
`SSL_FINETUNE_PROVENANCE_ONLY=True`; no GPU smoke is run. Combining this CPU mode
with any baseline, reconstruction or full fine-tuning action is rejected before
setup. The worker derives device requirements from the actual flags.

Run All performs full training-content verification with **zero model/optimizer
steps**. It may preprocess all 17,441 training ECGs (about 0.84 GB additional
cache, plus diagnostic/norm caches); this is a complete content audit, not a
four-record smoke. Results are under
`/kaggle/working/btp_outputs/sessions/<NEW>/ssl_finetuning/provenance_audit/<task>_<id>/`.
For a truly equivalent input, strict/stat-only success is
`passed_verified_training_content` with `checked_records=17441`; normalizer
differences still reject until an explicit verified source-normalizer policy is selected
for each task. A 1,004-record overlap or partial count cannot authorize transfer.
After both audits pass, select a GPU for a later GPU smoke and use
`SSL_FINETUNE_PROVENANCE_ONLY=False` with both full flags still False, AMP off.
No full training was launched or content equivalence of the remaining 16,437
real records claimed during local implementation.

Current notebook defaults set SSL_FINETUNE_NORMALIZER_POLICY="strict" for diagnosis.
The supplied CPU-session waveforms have confirmed differences and must remain rejected.
Read the normalizer migration guide above for its additional gates and preserved evidence.
