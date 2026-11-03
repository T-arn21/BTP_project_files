# Read-only audit of the real SSL and failed fine-tuning artifacts

Successful SSL session: `20261028T191718Z_e9445a9d`.
Failed fine-tuning session: `20261029T052515Z_51f75e15`.

The original local roots are Downloads/results/btp_outputs and
Downloads/results (1)/btp_outputs. All 88 original files were SHA256-hashed,
size/mtime recorded, and rechecked after inspection. None changed.

## Findings

The actual completed encoder has the pinned SHA256:
`8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4`.
It loads with weights_only=True; all 184 encoder tensors are finite.
The original config/summary/checkpoint and optional history/status/provenance
agree on epoch 20, 17,441 training records and full pretraining completion.
The failed run's saved source-verification hashes match these exact encoder,
config and summary files. It did not accidentally load the two-record smoke export.

The saved population-independent identity fields are ALL equal: root, declared
version 1.0.1, both metadata hashes, preprocessing code hash, sampling/filter/
duration/leads, NumPy/SciPy/WFDB versions, reader and schema. The complete saved
environments match too. Model configuration matches. Normalization statistics,
all 1,000 fit IDs and normalization configuration match exactly.

Therefore the failed field is deduced by elimination to be
`preprocessing_identity.raw_file_stats_sha256`: the old code failed on its freshly
computed training-only identity, and every other field is invariant to that
population selection and equal in the saved artifacts. This is NOT a root/mount
alias difference, library mismatch or normalization mismatch.

| Signature | Value | Scope |
| --- | --- | --- |
| Original SSL training-only | `4c9e5b74d0ad83dab5bc232fa92554933359dc2216cd62c990bf1575f79bf04b` | 17,441 official training ECGs |
| Failed run's recomputed training-only | **Not persisted** | 17,441 ECGs; logged recomputation then rejection |
| Failed run's saved cache identity | `02668e5cb8252b23872215a6700f1411702495d86ffaf6b75d58c28b7c3cf689` | All metadata folds, NOT the training-only comparison |

The two printed hashes must not be presented as an expected/actual same-scope
pair. The old error and cache manifest do not contain the actual failing
training-only digest. Hash preimage recovery is not possible from these artifacts.
The new provenance-only audit records that value in its new execution.

## Cached waveform evidence

Both source/current cached arrays match their manifest SHA256s; the full SSL
array also matches the payload SHA embedded in the authenticated encoder.
All **1,004 overlapping training ECGs** are bit-for-bit equal before normalization
(maximum absolute difference 0). This includes all 1,000 normalization fit ECGs.
The failed cache's other four records are validation IDs 8, 10, 26 and 28 and
are correctly absent from the SSL training cache.

The remaining **16,437** pretraining ECGs have no current-run cached counterpart.
Thus complete training-population equivalence is NOT established. Stat signatures
include names/presence, sizes and nanosecond mtimes; timestamp-only drift is
plausible, but per-file historical stats were not saved, so its precise cause
cannot be proved. Partial overlap and equal normalization do not justify accepting
a different signature globally.

## Decision

Retain strict rejection. Do not rewrite the encoder/provenance or remove the
raw signature check. The previously added filesystem-alias exception is irrelevant
here because the root strings already match. A future compatibility policy would
need complete trusted source/live content-equivalence evidence, not just matching
names, library versions, metadata or a 1,004-record overlap.

No GPU model, optimizer, training or preprocessing of raw ECGs was executed in
this audit. The numeric operations were read-only comparisons of saved cache
arrays and saved statistics. See findings.json for full metadata/evidence and
artifact_hashes.json for original file integrity records.

A subsequent implementation adds an opt-in [full training content proof](../ssl_content_identity.md).
The historical decision above remains in force until that complete proof actually passes.
