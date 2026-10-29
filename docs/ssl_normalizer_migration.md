# Normalizer diagnosis and authenticated source reuse

## Evidence and limits

The supplied CPU audit session `20261029T135756Z_bcc506df` has exactly three
blocking fields: `normalizer.mean`, `normalizer.std`, and training-only raw stat
digest. Its comparison reports, normalization cache and waveform payload have
now been inspected read-only from `Downloads/results_3/btp_outputs`.

**Confirmed cause of the normalizer mismatch: the processed inputs differ.**
Across 1,004 overlapping training ECGs, 80 ECGs have 86 different float32 values;
all 80 are among the exact 1,000 normalization fit IDs. The largest absolute
waveform difference is 2.9802322387695312e-08; the largest representational
separation is 21 float32 ULPs (near small magnitudes). This is not just a stat
hash or an arithmetic difference in reducing identical inputs.

Re-fitting the original reference reproduces the original saved mean/std
**exactly**. Re-fitting the CPU run's cached waveforms reproduces its own saved
mean/std **exactly**, on the same local runtime (NumPy 1.26.4). The mean maximum
difference between runs is 1.8160874280109818e-14; std maximum difference is
2.906008766956347e-14. Fit IDs and configurations match exactly; the current
payload checksum passes. This reproduces both normalizers from their respective
inputs and ties their divergence to the changed processed samples. The CPU cache
is not shown to contain incorrect fit statistics.

The original epoch-20 encoder pin and complete reference payload pin were
independently verified. The preserved original
`normalization_200263f9774bfae9af1851df.json` has the same statistics, identity and
configuration as the encoder. Original values are recoverable losslessly from
`provenance.normalizer` inside that pinned encoder.

Original/current Python, NumPy (2.1.3), SciPy (1.16.3), pandas (2.3.3), WFDB
(4.3.1), source hash, reader and metadata versions match. Torch differs as
expected between GPU and CPU images; preprocessing uses NumPy/SciPy/WFDB, not
Torch. The underlying reason for processed float differences is **not verified**:
CPU/SIMD/build-dependent preprocessing arithmetic is plausible, but historical
raw waveform bytes, CPU features and compiled build details were not saved here.
No tolerance or CPU-image exception is justified by this evidence.

The original training-only stat signature is
`4c9e5b74d0ad83dab5bc232fa92554933359dc2216cd62c990bf1575f79bf04b`;
this session's actual training-only value is
`5e11fda53e23dd5d6986e830d4a4e88d7952b438a4a72e359eceebc2be357a73`.
These are preserved. All 24 files in the newly supplied download were hashed,
size/mtime recorded and rechecked unchanged. Full local evidence:
`outputs/normalizer_forensics/cpu_session_findings.json` and
`cpu_session_artifact_hashes.json`; the original-reference replay is
`reference_refit.json` in the same ignored directory.

## Source-level findings

Both SSL and classification choose normalization IDs from the complete official
training folds before label exclusions. `select_normalization_ids` sorts IDs,
uses NumPy RNG seed 42 by default to sample 1,000, then sorts the sample again.
Configuration, ordered IDs and train/held-out patient separation remain exact
gates. `SignalNormalizer.fit` promotes deterministic float32 cached ECGs to
float64, using ordered streaming population mean/variance updates and a minimum
standard deviation of 1e-8. Augmentation and held-out waveforms are not used.

Normalization cache keys cover the entire preprocessing identity plus the fit
configuration/ordered IDs. They therefore include raw stats (relative filename,
presence, size, nanosecond mtime), metadata/preprocessing hashes and versions.
Additionally, SSL keys use a training-only identity while supervised keys use
all metadata folds. These keys intentionally differ; a miss causes fitting,
not reuse of historical statistics. JSON is round-trippable to the saved binary64
values. Statistics checksums detect cache inconsistency but an arbitrary JSON's
self-reported checksum is not authentication of original SSL normalization.

The former comparison required a sole stat mismatch before content verification.
That gate hid the very waveform evidence needed to investigate a normalizer
mismatch. It has been replaced by independent content reporting when the only
differences are stat digest and/or normalizer values. All other fields (including
normalization fit IDs/configuration, splits, code, reader, versions and model)
must match first. They remain blocking and prevent migration.

## Separate evidence from approval

With a configured reference cache, the complete processed-waveform proof runs
across every original training ECG, including unlabeled training records. The
reference payload must still match the checksum embedded in the SHA256-pinned
encoder. No 1,004-record overlap, tolerance, rounded hash or skipped batch can
replace this proof. A content failure rejects and reports the first differing
ECG, finite-status failure where applicable, and waveform difference magnitude.

Once the authenticated reference and complete current cache are available,
`normalization_analysis` independently re-fits both caches (even if their waveform
comparison fails) using the saved fit IDs/order in the **same
runtime**. It records full values, exact equality and maximum absolute mean/std
differences, original/current cache keys and the source of the original values.
The current normalizer must reproduce **exactly** from this live fit. A stale,
corrupt or otherwise irreproducible current normalizer remains a rejection even
when waveform equivalence passes. This avoids hiding a genuinely wrong fit by
simply substituting historical statistics. Exact equality between reference and
current same-runtime re-fits is also required; no tolerance authorizes migration.
The original refit-versus-encoder difference is reported, not misrepresented as
proof of a historical rounding cause. Diagnostic fitting on unequal inputs can
never authorize migration; `normalization_analysis.verified` also requires the
complete byte-level content proof.

`SSL_FINETUNE_NORMALIZER_POLICY` selects the explicit next step:

- `"strict"` (Python API and next-audit notebook default): reports content/normalization evidence but
  **does not accept** different normalizer values. `content_equivalence.verified`
  may be True while the comparison status remains rejected with two blocking
  normalizer fields. Complete content proof can resolve only the stat difference.
- `"authenticated_source"` (later explicit opt-in, not the next audit default):
  requires the pinned/completed export, full content equivalence, reproducible
  current fit and every other gate passing. Then the new data bundle uses the
  original encoder's exact normalizer for every train/validation/test dataset.
  It never substitutes a newly computed approximation of the original values.

Both modes preserve expected/actual statistics and stat hashes in the report.
The latter records `normalizer_migration.approved=True`, original/current
statistics, the encoder pin, refit evidence and the selected policy. The original
files/caches are not rewritten. Only the new in-memory bundle and new output
manifest/checkpoints use the authenticated source normalizer. Fine-tuning
metadata is refreshed before Trainer creation so future checkpoints record the
normalizer actually used. After approval, training reads the **current** full
cache that passed byte verification, preventing a different diagnostic cache or
new preprocessing pass from supplying unchecked training values. This never
substitutes the historical reference payload for live data. The separate
ssl_verified_training_cache provenance records the effective current path,
checksum and all training IDs; held-out caches remain separate. A previous audit report alone cannot authorize a new
run: full content/authentication/reproducibility checks run again.

## Next Kaggle CPU audit

Use the updated `05_btp_kaggle_final.ipynb`, **Accelerator None**, Internet on,
and the same checkpoint and PTB-XL inputs. Current first-cell defaults are:

```python
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False
RUN_SSL_FINETUNE_SMOKE = True  # selector only while provenance-only is True
RUN_SSL_FINETUNE_SUPERCLASS = RUN_SSL_FINETUNE_SUBCLASS = False
SSL_FINETUNE_PROVENANCE_ONLY = True
SSL_FINETUNE_NORMALIZER_POLICY = "strict"  # report-only investigation
SSL_FINETUNE_REFERENCE_CACHE = "/kaggle/input/datasets/arnavtiku/checkpoint/btp_outputs/cache/ssl/signals_fa14baaf8ce23a18d5778e2d/signals.npy"
SSL_FINETUNE_ENCODER_SHA256 = "8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4"
```

No model/optimizer/GPU smoke/full training runs. Both task reports are attempted.
The audit can preprocess/cache the full population and re-fit only the 1,000
normalization IDs, with visible progress and launcher heartbeats. CUDA extensions
are not installed or probed. Existing waveform caches are reused only under
normal exact-identity and payload-checksum validation.

**For the supplied CPU artifacts, rejection is expected**: there are already
80 proved unequal ECGs. The next run will expose the first live content mismatch
(the earliest known overlapping mismatch is ECG 120; earlier uncached records
were not inspected), its magnitude, and diagnostic fit comparisons. It must not
adopt the source normalizer. The remaining 16,437 records are not assumed equal.
There is no safe normalizer-only fix for this input under the requested byte policy.
To proceed, reproduce matching preprocessing/input data and prove all 17,441
records; matching version strings alone is insufficient. Running in a matching
original environment is a candidate investigation, not a guaranteed fix. Future
canonical preprocessing would need its own fully verified migration, not an
unverified substitution of the historical source cache for current raw data.

For a truly equivalent dataset, strict mode still rejects differing normalizer values
while reporting full content equivalence. Only a later explicit switch to
SSL_FINETUNE_NORMALIZER_POLICY="authenticated_source" requests original-normalizer
reuse; it rechecks every proof and never trusts an old report alone. In that
mode, success requires `content_equivalence.verified=True`, `checked_records=17441`,
`normalization_analysis.current_fit_reproduced=True`,
`normalizer_migration.source_authenticated=True`, `approved=True`, and status
`passed_verified_content_original_normalizer`. This is an audit completion,
not a classification benchmark. `data_manifest.json` contains the effective
normalizer plus `normalizer_before_ssl_validation`; the comparison retains all
original differences. `summary.json` reports zero optimizer steps.

If waveforms differ, expect `rejected` with the first mismatched ECG and no
normalizer activation. If waveforms match but current statistics do not reproduce,
expect `normalization_analysis.error` and continued rejection; preserve the output
and investigate the current cache/runtime rather than deleting evidence or
claiming rounding. Any other mismatched provenance field remains blocking.

Reports are saved under:
`/kaggle/working/btp_outputs/sessions/<NEW>/ssl_finetuning/provenance_audit/<task>_<id>/`.
Keep both full task flags False after this audit; later GPU smoke/fine-tuning is
an independently configured run, never started automatically by an audit success.

Independent JSON artifacts beside the comparison report: content_equivalence.json,
normalization_analysis.json, normalizer_migration.json and numerical_runtime.json.
The last records NumPy/SciPy build and CPU/SIMD runtime information for the next
investigation without changing acceptance rules.
