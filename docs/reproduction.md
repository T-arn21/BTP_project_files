# Reproduction and experiment handoffs

The six root experiment notebooks are execution entry points for shared `src/`.
No historical experiment is reproduced merely by importing them. No expensive
run is enabled by default. The retained unified notebook is a separate advanced
entry point with all execution flags disabled by default.

## Stage controls

| Notebook | EXECUTE=True with MODE=smoke | EXECUTE=True with MODE=full |
|---|---|---|
| 01_supervised_baseline | Metadata audit + both real GPU forward/backward checks | Same checks then both full supervised tasks |
| 02_ssl_pretraining | Tiny training-fold masked reconstruction check | Same check then20-epoch pretraining |
| 03_ssl_finetuning | Both two-batch task diagnostics, batch2, FP32 | Same checks then both full classifiers, default5epochs |
| 04_ssl_extended_finetuning | Unsupported; rejects | Restore completed source runs; increase total epoch budget only |
| 05_asl_finetuning | Unsupported; rejects | Both pinned full classifiers;5 new ASL epochs |
| 06_evaluation_and_plots | Not applicable | EXECUTE=True regenerates metrics/plots on CPU; no MODE variable |

For01-05 the first cell holds EXECUTE, MODE, branch, checkout/output roots and
OVERRIDES. The synchronized `workflow_settings()` loads the unified configuration
snapshot, disables unrelated modes, selects the documented stage and saves the
resolved settings with the commit. Unknown/competing mode overrides reject.
Nested setting dictionaries merge with the snapshot, so unspecified optimizer
and architecture values remain visible and recorded. Profiles never weaken
checkpoint or data provenance checks. Stage06 has its own explicit input mapping.

## Before Run All

1. Use notebooks from `refactor/project-organization` during review. Set GIT_BRANCH
   to that branch; after merge use reviewed main. Reimport updated notebooks when
   bootstrap/profile code changes. An old notebook can select stale source code.
2. Attach PTB-XL with its CSVs and 100Hz `.hea`/`.dat` records. DATASET_ROOT is
   auto-detected when empty. Set the actual declared release (historical pinned
   runs used1.0.1); metadata schema alone cannot prove release identity.
3. Enable Internet and Kaggle GITHUB_TOKEN for secure read access. Enable GPU
   for01-05; prefer compatible precompiled mamba-ssm/causal-conv1d wheels. Source
   fallback remains an explicit setting; ABI/Python/Torch/CUDA incompatibility
   rejects clearly. No unattended kernel restart is required.
4. Review required stage artifacts and their exact mount paths in each first
   cell and detailed guide. Keep a clean checkout: synchronization never resets,
   stashes, cleans or overwrites uncommitted work.
5. First use smoke mode for01-03. After reviewing successful smokes, explicitly
   select full. Full02/03 still require fresh same-execution smoke checks.
6. Preserve saved Kaggle outputs and caches. Working storage alone is not durable.

## Artifact handoffs

| Consumer | Required source export |
|---|---|
| 03 | Completed20-epoch encoder.pth, independently verified SHA256, config/summary and available completion/provenance sidecars; original full training signals.npy plus manifest.json |
| 04 | Each completed03 full folder: last.pth, best.pth, config/summary/status JSON, history.csv; all saved-config SSL/data artifacts |
| 05 | Each completed15-epoch extension's validation-selected best.pth plus config/summary/status/history; original SSL/cache/data artifacts retained |
| 06 | Completed classifier config/summary/status/data_manifest/history/thresholds and validation/test predictions.csv (or historical test/test_predictions.csv) |

Historical defaults in03 use `arnavtiku/checkpoint` and encoder SHA
`8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4`.
The authenticated original training cache covers17,441 ECGs. The historical04
source is `/kaggle/input/datasets/arnavvt/checkpoint-2/` session20261029T180106Z_e154ce34.
The historical05/06 source is `arnavvt/checkpoint-2`,
session20261030T060028Z_a420e362: superclass best7, subclass best14. Exact paths
and classifier pins are in the notebook and [ASL guide](asl_second_stage.md).

These defaults intentionally support existing artifacts, not automatic discovery
of whatever a previous notebook most recently produced. For new experiments,
export/attach those artifacts and update BOTH task paths plus applicable pins.
Do not relabel a new encoder as the historical one or use a smoke checkpoint.
Current transfer validation requires a completed20-epoch SSL export. ASL's
historical recipe explicitly requires completed15-epoch BCE source classifiers.

## Extension and ASL are different operations

Notebook04 restores optimizer, completed scheduler, RNG, history and selection
state. `hold_completed_lr` explicitly holds the prior terminal LR, so increasing
total epochs does not silently reset or stretch the cosine schedule. All settings
except total epoch limit/new output location remain exact. Defaults extend5to15;
change the target only after checking source completion.

Notebook05 transfers all classifier model weights and deliberately starts new
AdamW/cosine/history/selection state. Its recipe is five epochs, LR1e-5,
min_lr1e-6, batch2, FP32, gamma_pos0/gamma_neg4/clip.05, no BCE positive weights or
smoothing. Existing narrow held-out stat/path compatibility requires the same
verified processed cache payload. No additional huge training-waveform audit is
introduced. Both operations preserve source best.pth and previous results.

## Reporting and remaining limits

All runs use new timestamped directories. Ranking metrics are AUROC and average
precision (AUPRC); undefined scores are reported with valid-class counts.
Validation chooses model checkpoints and F1 thresholds. Full classifiers export
both tuned and fixed0.5 test metrics; ASL and06 also compare both validation
variants. No test-derived model or threshold selection is allowed.

Stage06 authenticates which exported files it reviewed using SHA256, verifies
recorded ECG split separation, schema, class order and validation thresholds,
then writes a separate review folder. It does not independently re-audit raw
patient metadata, verify model inference or run CUDA. It rejects diagnostic/SSL
folders, missing exports, class/ID mismatches and threshold inconsistencies.

Fresh GPU execution has not been attempted in this reorganization. Exact CUDA
numerics, wheel availability and Kaggle mount/dependency compatibility remain
external requirements. Original strict provenance can reject changed paths or
versions; organization is not permission to relax those rules. Historical
results require the original exported artifacts and environments, which are
not committed here. See [test evidence](test_results.md).

Completed-run extension retains exact cache/stat metadata validation. It may reject a new session with changed source signatures or cache paths even when other stages accept a documented relocation. This organization does not broaden resume compatibility.
