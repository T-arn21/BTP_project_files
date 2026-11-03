"""Build cleared experiment notebooks from the shared secure bootstrap; no execution."""
import json
import hashlib
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'refactor/project-organization'
DEFAULTS = json.loads((ROOT / 'configs/kaggle_defaults.json').read_text())
BOOTSTRAP = (ROOT / 'src/bootstrap.py').read_text(encoding='utf-8')

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text, tag): return nbf.v4.new_code_cell(text, metadata={'tags':[tag]})
def write(name, cells):
    for index, cell in enumerate(cells):
        cell.id = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12]
    book=nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}, 'language_info':{'name':'python'}})
    nbf.validate(book)
    nbf.write(book, ROOT/name)

STAGES = [
('01_supervised_baseline.ipynb','supervised_baseline','Supervised baseline','smoke',
'''Train the preserved Mamba classifier from fresh weights on five superclasses and 23 diagnostic subclasses. Official folds 1-8 train, 9 validate and 10 test; patient separation is audited before label exclusions. Training-only normalization, deterministic waveform caching and weighted BCE with training-only smoothing reuse src/. Both tasks use the same evaluator. Validation macro AUROC selects checkpoints; validation alone selects thresholds.''',
{'FULL_EPOCHS':20,'FULL_BATCH_SIZE':2,'MIXED_PRECISION':False},
'''PTB-XL 1.0.1 (`ptbxl_database.csv`, `scp_statements.csv`, `records100/`), compatible CUDA/Mamba installation or wheels, Internet and Kaggle GITHUB_TOKEN. No pretrained checkpoint is used. DATASET_ROOT can be empty for automatic discovery; the declared DATASET_VERSION must match the attached release.''',
'''`MODE="smoke"` runs metadata auditing and one real forward/backward smoke per task, without full waveform caching or benchmark training. `MODE="full"` runs those checks then two full-data BCE experiments. FULL_EPOCHS is a new run budget, not a claim to reproduce a particular historical checkpoint. Existing optimizer defaults (LR1e-4, warmup3, smoothing.05, pos_weight capped8) are visible in configs/kaggle_defaults.json.''',
'''Smoke reports or `experiments/benchmark/mamba_<task>_<id>/`: best/last.pth, history, config/environment/data_manifest, curves, validation/test predictions, per-class metrics and validation-tuned/fixed-0.5 test evaluations.'''),
('02_ssl_pretraining.ipynb','ssl_pretraining','Masked ECG reconstruction pretraining','smoke',
'''Pretrain the same temporal Mamba encoder using only official training-fold waveforms, including training records without diagnostic labels. Diagnostic labels are not used. Reproducible contiguous temporal masks cover the same time positions across all 12 leads. A reconstruction head predicts the original normalized waveform; only masked positions contribute to MSE. Validation/test patients are excluded from SSL data and normalization.''',
{'SSL_SETTINGS':DEFAULTS['SSL_SETTINGS']},
'''PTB-XL 1.0.1 and the GPU/bootstrap inputs from notebook 01. No classifier or encoder checkpoint is needed for fresh pretraining. Cache and normalization provenance are saved for downstream reuse; retain the entire full-pretraining export and original signals.npy/manifest.json, not just encoder.pth.''',
'''`MODE="smoke"` runs a tiny finite forward/backward reconstruction check. `MODE="full"` requires and runs that same-execution smoke before the 20-epoch full training-fold stage. SSL_SETTINGS controls masking, budget, batch2, FP32, LR and checkpoint cadence. The smoke encoder is explicitly ineligible for downstream full fine-tuning.''',
'''`ssl/smoke/` and, only in full mode, `ssl/pretraining/ssl_mamba_<id>/`: encoder.pth, last.pth, config, split_provenance, history, summary/status, loss curve and masked reconstruction visualizations. Preserve `cache/ssl/` with its manifest for cache-backed fine-tuning.'''),
('03_ssl_finetuning.ipynb','ssl_finetuning','SSL encoder fine-tuning','smoke',
'''Load the SHA-pinned completed 20-epoch encoder. Transfer the trained Mamba backbone and initialize NEW attention pooling and classification heads. Use the authenticated original SSL training waveform cache and original training-derived normalizer; preprocess held-out records separately using the same configuration. Both official classification tasks use patient-disjoint folds and the existing BCE trainer/evaluator. This is explicitly ssl_cache_backed, not a claim of independent waveform reproduction.''',
{k:DEFAULTS[k] for k in ('SSL_FINETUNE_ENCODER_PATH','SSL_FINETUNE_ENCODER_SHA256','SSL_FINETUNE_REFERENCE_CACHE')} | {'SSL_FINETUNE_SETTINGS':dict(DEFAULTS['SSL_FINETUNE_SETTINGS'],epochs=5)},
'''PTB-XL 1.0.1 plus `arnavtiku/checkpoint`: completed encoder.pth, config.json, summary.json (and available completion sidecars); original cache signals.npy and adjacent manifest.json. The pinned historical encoder and paths below select the completed export, never ssl/smoke. A newly generated notebook-02 export requires its own independently verified hash and matching artifacts.''',
'''`MODE="smoke"` runs both task diagnostics with batch2, two diagnostic batches per task and AMP off. `MODE="full"` first runs both smokes then enables both full tasks. The initial five-epoch budget can feed notebook04 after completion. All checkpoint, class-order, data/normalizer/split and preprocessing gates remain active; no automatic fallback or expensive independent training-content audit is selected.''',
'''`ssl_finetuning/smoke/` or `ssl_finetuning/full/benchmark/mamba_<task>_<id>/`: new classifiers, SSL transfer/provenance reports, history, evaluation exports and configurations. Preserve the complete full task directories for extension.'''),
('04_ssl_extended_finetuning.ipynb','ssl_extended_finetuning','Completed SSL classifier epoch extension','full',
'''Extend completed cache-backed classifier runs with ALL saved model, optimizer, scheduler, RNG and history state. Only the total epoch limit and new output location may change. Source best.pth and its epoch remain preserved if later epochs do not improve validation macro AUROC. The explicit hold_completed_lr policy retains the completed scheduler and terminal learning rate; it does not reset a cosine schedule.''',
{k:DEFAULTS[k] for k in ('SSL_FINETUNE_RESUME_RUN_DIRS','SSL_FINETUNE_RESUME_SCHEDULER_POLICY')} | {'SSL_FINETUNE_RESUME_TO_EPOCHS':15},
'''Both completed source task directories: last.pth, best.pth, config.json, summary.json, status.json and history.csv; plus every SSL/PTB-XL artifact referenced by their saved configs. Defaults identify the historical epoch-5 session in the verified arnavvt/checkpoint-2 input. For newly completed notebook03 runs, replace BOTH source directories with the saved full exports. Preserve required source-config mount paths; substantive provenance drift must reject.''',
'''Only `MODE="full"` is supported; EXECUTE=False prevents any work until reviewed. The example extends epoch5 to epoch15. There is no new smoke optimization stage or new optimizer. An already completed epoch15 source requires a larger explicit target, not a restart. Incompatible settings/data fail closed; do not reinterpret a provenance failure as permission to initialize fresh weights.''',
'''`ssl_finetuning/resumed/benchmark/mamba_<task>_<id>/`: source snapshot, inherited history plus new epochs, preserved prior best or a newly validation-selected best, checkpoints, configuration and complete evaluation exports.'''),
('05_asl_finetuning.ipynb','asl_finetuning','ASL second-stage classifier fine-tuning','full',
'''Initialize ALL classifier weights (backbone, pooling, head) from the SHA-pinned validation-selected best.pth of each completed epoch15 BCE extension: superclass best7, subclass best14. This is a NEW five-epoch optimization stage with fresh AdamW and cosine, LR1e-5/min1e-6, batch2, FP32, ASL gamma_pos0/gamma_neg4/clip.05. BCE pos_weight and label smoothing are disabled. Validation macro AUROC selects the new best checkpoint.''',
{k:DEFAULTS[k] for k in ('ASL_BEST_CHECKPOINTS','ASL_CHECKPOINT_SHA256','ASL_SOURCE_BEST_EPOCHS','ASL_STAGE_SETTINGS')},
'''PTB-XL 1.0.1, original arnavtiku/checkpoint encoder/cache/normalizer artifacts, and arnavvt/checkpoint-2 containing BOTH epoch15 best classifiers plus config/summary/status/history sidecars. Do not use resume_source/best.pth. The exact confirmed paths and pins below are checked before training. Model parameters are never overwritten by an SSL-only transfer.''',
'''Only `MODE="full"` is supported; EXECUTE=False is the initial guard. No unrelated training, diagnostic or smoke stage is selected. Preserve the ASL recipe. Only the three documented held-out stat/cache-path fields may relocate after the already-verified processed payload matches; all substantive provenance stays strict. See docs/asl_second_stage.md for the evidence-backed rule.''',
'''`asl_stage2/benchmark/mamba_<task>_<id>/`: one new directory per task, full_classifier_initialization and provenance comparison reports, best/last checkpoints, history, curves, per-class/prediction CSVs, and fixed0.5 versus validation-tuned validation/test threshold_comparison.csv.''')]

for filename, profile, title, mode, method, overrides, inputs, settings, outputs in STAGES:
    configuration=f'''# Review this cell; nothing runs until EXECUTE=True.
EXECUTE = False
MODE = {mode!r}  # smoke/full for 01-03; full only for 04-05
GIT_BRANCH = {BRANCH!r}  # after merge, select the reviewed main branch explicitly
CHECKOUT_DIR = "/kaggle/working/BTP_project_files"
OUTPUT_ROOT = "/kaggle/working/btp_outputs"
OVERRIDES = {overrides!r}
REQUEST = dict(WORKFLOW_PROFILE={profile!r}, WORKFLOW_MODE=MODE,
    WORKFLOW_OVERRIDES=OVERRIDES, CHECKOUT_DIR=CHECKOUT_DIR,
    OUTPUT_ROOT=OUTPUT_ROOT, GIT_BRANCH=GIT_BRANCH, GIT_TIMEOUT_SECONDS=600)
'''
    # Pretty configuration values, with valid Python booleans rather than JSON booleans.
    import pprint
    configuration=configuration.replace(repr(overrides),pprint.pformat(overrides,width=110,sort_dicts=False))
    execution=BOOTSTRAP+'''

if EXECUTE:
    EXECUTION_SUMMARY = execute_notebook(REQUEST)
    if EXECUTION_SUMMARY["status"] != "complete":
        raise RuntimeError("Workflow failed; inspect the printed summary and preserved session logs")
else:
    EXECUTION_SUMMARY = dict(status="not_executed", completed={}, session_dir=None)
    print("Configuration review only. Set EXECUTE=True to run this stage; no secret, network, installation or GPU used.")
'''
    write(filename,[md(f'# {title}\n\nStage-specific executable entry point using shared `src/` modules. This notebook contains no trained outputs and does not establish historical reproduction. See [execution order](README.md) and [stage guide](docs/reproduction.md).'),md('## Methodology\n\n'+method),md('## Required inputs\n\n'+inputs),md('## Key settings and execution\n\n'+settings+'\n\nOn Kaggle enable GPU, Internet and the GITHUB_TOKEN secret. Review the configuration, set EXECUTE=True, then Run All. Secure synchronization and dependency setup run in fresh processes; no kernel restart is assumed. Settings are resolved from the synchronized repository snapshot and saved with its commit. See configs/kaggle_defaults.json for inherited settings.'),code(configuration,'stage-config'),code(execution,'stage-execution'),md('## Outputs and interpretation\n\nAll session files are below OUTPUT_ROOT/sessions/<new_session>/. '+outputs+'\n\nInspect final_summary.json and task status.json before reporting completion. Smoke/diagnostic outputs are not full benchmarks. Inputs and prior outputs are never overwritten.'),code('print("Status:", EXECUTION_SUMMARY["status"])\nprint("Completed:", list(EXECUTION_SUMMARY["completed"]))\nprint("Session:", EXECUTION_SUMMARY["session_dir"])','stage-summary')])

review_cfg=f'''from pathlib import Path
EXECUTE = False
SYNC_REPOSITORY = True  # False uses an existing local checkout (no network/secret)
GIT_BRANCH = {BRANCH!r}
REPO_ROOT = "/kaggle/working/BTP_project_files"  # local: absolute clone path
EXPERIMENTS = {str({task:str(Path(path).parent).replace(chr(92),'/') for task,path in DEFAULTS['ASL_BEST_CHECKPOINTS'].items()})}
OUTPUT_ROOT = "/kaggle/working/btp_outputs/evaluation_reviews"
'''
review_exec=BOOTSTRAP+'''

REVIEWS = {}
if EXECUTE:
    if SYNC_REPOSITORY:
        synchronized = sync_repository(REPO_ROOT, branch=GIT_BRANCH)
        select_import_root(synchronized["checkout"])
    else:
        select_import_root(REPO_ROOT)
    from src.result_review import review_experiment
    review_root = Path(OUTPUT_ROOT) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    if not EXPERIMENTS:
        raise ValueError("Set EXPERIMENTS to completed classifier export directories")
    for name, directory in EXPERIMENTS.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Review names must be single directory names")
        REVIEWS[name] = review_experiment(directory, review_root / name)
    print("CPU prediction re-analysis saved:", review_root)
else:
    print("Configuration review only; no files, secrets, network, installation or GPU used.")
'''
write('06_evaluation_and_plots.ipynb',[md('# Evaluation and plots from saved predictions\n\nCPU-only re-analysis using the shared evaluator. This does not rerun inference, pretraining or classification training and is not a new independently reproduced benchmark.'),md('## Methodology\n\nRead complete validation/test prediction exports with exact recorded class order and ECG split IDs. Recompute validation-only F1 thresholds and verify them against thresholds.json. Export macro/micro AUROC, average precision (AUPRC), F1, per-class results and fixed0.5/tuned comparisons. Regenerate the recorded training loss and test confusion plots. Test never tunes thresholds. Recorded ECG separation is checked; raw patient metadata/provenance is not re-audited.'),md('## Inputs and settings\n\nEach completed classification directory needs config.json, summary.json, status.json, data_manifest.json, history.csv, thresholds.json and validation/test predictions.csv (historical test/test_predictions.csv is also supported; duplicate candidates reject). Defaults select the historical epoch15 classifier exports from arnavvt/checkpoint-2; replace EXPERIMENTS with newly completed stage outputs to compare them. Diagnostic-only and SSL reconstruction folders are rejected. SSL loss/reconstruction visualizations are already exported by notebook02.\n\nKaggle accelerator None is sufficient. With SYNC_REPOSITORY=True enable Internet/GITHUB_TOKEN; no CUDA extensions are installed. Local: install requirements.txt, set REPO_ROOT and SYNC_REPOSITORY=False. Set EXECUTE=True and Run All.'),code(review_cfg,'stage-config'),code(review_exec,'stage-execution'),md('## Outputs\n\nA new timestamped folder per invocation and experiment holds threshold_comparison.csv, per-class/overall metrics and prediction CSVs, loss/confusion figures and review_provenance.json with source export SHA256s. Source experiments are never edited. These hashes document the reviewed files, not independent authentication of raw ECGs or model inference.'),code('print("Reviewed experiments:", list(REVIEWS))\nfor name, proof in REVIEWS.items():\n    print(name, proof["output"])','stage-summary')])

# Keep the established unified entry point and its first-cell configuration intact,
# except selecting the review branch. Keep its tested bootstrap in sync.
p=ROOT/'05_btp_kaggle_final.ipynb'
n=nbf.read(p,as_version=4)
n.cells[1].source=n.cells[1].source.replace('GIT_BRANCH = "dev/unified-ecg-pipeline"', f'GIT_BRANCH = "{BRANCH}"')
old=n.cells[3].source
suffix=old[old.index('\n\nif ACTIVE:'):]
n.cells[3].source=BOOTSTRAP+suffix
nbf.write(n,p)
