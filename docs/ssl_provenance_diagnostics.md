# Diagnosing the SSL fine-tuning provenance rejection

Reported session: `20261029T052515Z_51f75e15`, superclass smoke, before training.
The artifacts were initially unavailable, motivating the diagnostic release.
They have since been supplied and inspected read-only: see the
[actual artifact audit](history/ssl_provenance_artifact_audit.md). The root/version,
metadata/code hashes, environment and normalization match exactly. The rejected
field is deduced to be the raw file-stat signature; the actual historical
training-only digest was not persisted. This is not a mount-alias mismatch.
Timestamp-only drift remains unproven; a sole stat difference is rejected unless the full checkpoint-authenticated content audit succeeds.

## What source inspection establishes

In the failing release the check was one dictionary equality:

```python
if source["preprocessing_identity"] != current_training_only_identity:
    raise ValueError("SSL training-only dataset/preprocessing identity mismatch (root, hashes, signatures or versions)")
```

Reaching that exception means source SHA/completion/config checks, model config,
eligible/used pretraining IDs, held-out IDs and training patient checks already
passed. At least one of these identity fields differed:

- schema, root, declared_version;
- metadata_sha256 for ptbxl_database.csv and scp_statements.csv;
- raw_file_stats_sha256;
- sampling_rate, duration, apply_filter, leads;
- preprocessing_source_sha256;
- numpy, scipy, wfdb, reader.

The earlier failure did **not** establish whether normalization, its fit IDs or
waveform settings matched: those checks occurred later and were never reached.
The old fine-tuning path also saved its live data_manifest.json after successful
initialization, leaving no complete live comparison when initialization failed.
This is a confirmed diagnostic limitation, not proof of which data field differs.

## New comparison report

`ssl_provenance_comparison.json` is now saved before rejection or transfer.
It contains all comparisons in a single pass instead of stopping at the first
identity difference:

- `expected`: original pinned encoder metadata;
- `actual`: recomputed live metadata using the same **training-only** waveform
  signature scope as SSL, before label exclusions;
- `source_export`: source config.json and summary.json snapshots, checkpoint
  SHA/epoch, sidecar hashes and verified completion evidence;
- `mismatches`: each differing field's exact expected/actual value, presence flags,
  category and acceptance/evidence status;
- full normalization mean/std, fit settings/IDs, eligible/used pretraining IDs,
  excluded validation/test IDs and training patient IDs;
- current supervised split selections, separately from the original pretraining
  population (the bounded smoke selection is not mistaken for pretraining scope);
- intentional differences such as supervised augmentation and cache/worker controls.

Full ID lists and statistics are retained in JSON without truncation. Console
errors include field names and expected/actual values; large arrays/dictionaries
use counts and hashes there, pointing to the complete JSON. Missing fields remain
distinct from explicit null, and boolean assertions reject numeric 0/1 substitutes.
Fine-tuning failures also retain current config.json, environment.json and
data_manifest.json, plus structured mismatches in checkpoint_failure.json.

Source checksum and completion checks remain mandatory before trusting the source
metadata. A corrupt/missing encoder fails those gates before comparison.

## Narrow directory-alias compatibility

Only these two fields can receive a representation exception:

```text
data_config.data_dir
preprocessing_identity.root
```

It is accepted only when:

1. They are the **entire** mismatch: every other data, code/version/hash,
   normalization, split/patient and source-policy check passes exactly.
2. Both expected and actual values are absolute directory names that currently
   exist and are readable.
3. `Path.samefile` proves that they name the **same filesystem directory**.

The report records canonical resolved paths and device/inode evidence, with
status `passed_verified_directory_alias`. Original metadata is retained; nothing
is rewritten in the encoder or sidecars. Exact matches use `passed_exact`.

Different existing directories, an absent old mount, a matching basename, dataset
name or matching metadata hashes alone do not establish this alias. Failed
filesystem verification remains a rejection. A path alias plus any other mismatch
is also rejected; no unrelated field is ignored or normalized.

This policy is justified by actual filesystem identity. The real artifact audit
finds identical root strings in this session, so it does not apply to that failure.

## What must still match exactly

Dataset version, metadata/SCP hashes, raw source signature, filter/rate/length,
preprocessing source hash, library versions/reader, all normalization statistics
and fit settings/IDs, model configuration, official pretraining/held-out IDs,
patients and source scope assertions remain strict. Labels, metrics and the
supervised patient-disjoint folds are unchanged.

`raw_file_stats_sha256` hashes ordered filenames plus file sizes and nanosecond
mtimes. A mismatch by itself cannot distinguish timestamp-only changes from
different files/order/sizes. It is not a waveform content checksum. This fix
does not remove timestamps or accept a new signature merely because the release
name or normalization looks similar. Nor does it round statistics, sort mismatched
IDs, rewrite package versions or ignore metadata/code hashes. A genuinely changed
input/runtime must be restored to a verified matching one; a future content-based
migration would require separate trusted source/live evidence and validation.

## Next Kaggle run: provenance only, no fine-tuning steps

Use the updated final notebook and the same pinned pretraining encoder input.
Edit the existing first-cell assignments before its final SETTINGS line:

```python
RUN_SUPERCLASS = RUN_SUBCLASS = RUN_SMOKE = RUN_METADATA_AUDIT = False
RUN_SSL_SMOKE = RUN_SSL_PRETRAIN = False
RUN_SSL_FINETUNE_SMOKE = True
RUN_SSL_FINETUNE_SUPERCLASS = False
RUN_SSL_FINETUNE_SUBCLASS = False
SSL_FINETUNE_PROVENANCE_ONLY = True
```

Keep the actual dataset/root/version/normalization/model settings from the
failed run to diagnose them faithfully. Do not edit settings solely to conceal
a mismatch. Leave the supplied epoch-20 encoder path and SHA256 pinned.

Select Kaggle Accelerator None. The launcher synchronizes GitHub and checks base
scientific dependencies plus CPU checkpoint loading; CUDA/Mamba extension probes
and installation are skipped in exclusive provenance-only mode.
The fine-tuning audit itself creates **no classifier or optimizer**, performs
zero fine-tuning steps and does not evaluate/predict test waveforms. It may fit
the configured training-only normalization and build/reuse bounded diagnostic
caches to reproduce the current preprocessing exactly. Both label-task audits
are attempted even if the first rejects. Each report is written under:

```text
/kaggle/working/btp_outputs/sessions/<NEW_session_id>/
  ssl_finetuning/provenance_audit/<task>_<NEW_run_id>/
    ssl_provenance_comparison.json
    config.json, environment.json, data_manifest.json, status.json
```

The session final_summary.json distinguishes provenance audits from GPU smokes;
successful audits have `mode=provenance_only`, `optimizer_steps=0`, and no classifier
checkpoints. Any mismatch marks the audit/session failed while preserving both
reports. Mixing this mode with either full fine-tuning flag is refused.

After reviewing compatible reports, set SSL_FINETUNE_PROVENANCE_ONLY=False for a
separate bounded GPU smoke. Keep both full flags False. Full fine-tuning remains
an independent later opt-in; no full training was launched for this fix.

Normal smoke/full paths also always save the same comparison JSON in their task
folder, including when initialization rejects before the first batch.

See [full training content equivalence](ssl_content_identity.md) for the opt-in,
checkpoint-authenticated full-population proof required for a sole stat-signature mismatch.

Current notebook defaults run only the CPU provenance audit (Kaggle Accelerator None).
The reference accepts a signals.npy file, its entry directory, or the parent cache/ssl directory;
see the content-equivalence guide for exact pinned paths and strict resolution rules.

[Normalizer diagnosis and explicit authenticated source reuse](ssl_normalizer_migration.md)
documents independent full content reporting when mean/std also mismatch, current
CPU-audit settings and the additional exact reproducibility/authorization checks.
