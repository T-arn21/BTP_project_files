# Documentation index

Start with [README](../README.md) and [reproduction / stage handoffs](reproduction.md).
These replace the overlapping current-default narratives. Detailed operational
and provenance guides are retained because they document distinct implemented
checks, recovery policies and artifact evidence.

| Topic | Guide |
|---|---|
| Unified entry point, secrets, installation, failure handling | [Kaggle execution](kaggle_execution.md) |
| Reconstruction pretraining/masking/artifacts | [SSL pretraining](ssl_pretraining.md) |
| Encoder transfer and separate classifier initialization | [SSL fine-tuning](ssl_finetuning.md) |
| Original training cache + original normalizer | [Cache-backed fine-tuning](ssl_cache_backed_finetuning.md) |
| Completed-run full-state epoch extension | [Completed resume](ssl_completed_resume.md) |
| Full-classifier ASL and narrow held-out compatibility | [ASL stage](asl_second_stage.md) |
| NaN/Inf detection and FP32 recovery | [Numerical stability](numerical_stability.md) |
| Earlier weights-only checkpoint continuation | [Checkpoint continuation](checkpoint_continuation.md) |
| Provenance-only investigation and reports | [Provenance diagnostics](ssl_provenance_diagnostics.md) |
| Independent waveform equivalence policy | [Content identity](ssl_content_identity.md) |
| Original normalizer migration policy | [Normalizer migration](ssl_normalizer_migration.md) |
| Validation records and limitations | [Test results](test_results.md) |

Detailed older guides describe the unified notebook's flag interface, not the
new stage notebooks' EXECUTE/MODE interface. First-cell settings in the selected
notebook are authoritative; examples for one stage must not enable unrelated
modes. Historical investigations are in [history/](history/README.md).
