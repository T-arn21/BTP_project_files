# Original notebook analysis

The three originals are reference artifacts, preserved byte-for-byte. Every code
cell was inspected, including configurations, helper definitions, calls and saved
text outputs. Saved figures cover waveform/augmentation views, losses, ROC/PR,
confusion matrices, embeddings and attribution. Those are historical outputs;
none establishes reproducibility of the new implementation. See the companion
cell inventory for executable definitions and stored output types/counts.

## Preprocessing reference

`archive/notebooks/preprocessing-reusable-data-pipeline.ipynb`, code cells 3–46:

- Installs wfdb and checks a broad scientific stack (including torchvision,
  plotly, torchmetrics and optional psutil). Configuration uses 100 Hz, 1000
  samples, 12 leads; optional 500 Hz loads 5000 samples.
- Uses WFDB `rdsamp` and transposes `(T,12)` to `(12,T)`. Metadata SCP strings
  are parsed with `ast.literal_eval`. Rows with diagnostic==1 map through
  diagnostic_class; code presence is used even at confidence zero. Multi-hot
  superclass order is NORM, MI, STTC, CD, HYP. Unlabeled supervised rows drop.
- Provides signal validation, pooled statistics, fitted training z-score and
  per-record minmax/robust transforms. Bandpass is 4th-order 0.5–40 Hz,
  zero-phase; the 50 Hz notch is skipped at 100 Hz (Nyquist).
- Offers nine augmentations with independent probabilities: crop, stretch,
  circular shift, lead dropout, masking, amplitude scaling, per-lead scaling,
  noise, jitter. They apply to training classification samples only. SSL
  two-view dataset helpers exist but are not used for this task.
- Uses official folds 1–8, 9, 10, with pairwise patient overlap assertions;
  Dataset returns signal, label and ECG ID. DataLoader uses pinned memory,
  worker prefetch, persistent workers and drops the last training batch.
- Outputs enriched metadata, labels, class balance, signal statistics, QA
  results and demo figures. Default fit uses only 300 raw training samples
  selected from a 3000-record validation sweep.

Reused: label presence semantics, fold assignment, channel-first WFDB loading,
filter settings, padding/cropping and the complete augmentation class.
Normalizer changed to a streaming full-training fit on **filtered and length
adjusted** signals; the same fit IDs apply to both tasks before label exclusions.
Both tasks use a single deterministic waveform path.

## Training reference

`archive/notebooks/ptb-xl-ecg-supervised-mamba-trainin-v2-improved.ipynb`, cells 3–32:

- Explicit CUDA package setup (`mamba-ssm`, `causal-conv1d`), TensorBoard,
  torchmetrics, AdamW, AMP, cosine/linear schedules. It executes preprocessing
  notebook cells into globals and re-synchronizes seed to preprocessing seed.
- Architecture (cell 17) is preserved: per-timepoint Linear(12,384), eight
  prenorm residual bidirectional Mamba blocks, separate forward/backward
  scans, concatenation and linear merge, LayerNorm, concatenated mean and
  additive attention pooling (64-hidden scorer), dropout, then a two-layer
  GELU MLP head (hidden 384). No positional encoding or softmax output.
- Reference source hyperparameters: dropout .45, AdamW lr 3e-4 / weight decay
  5e-4, batch 64, 50 epochs, 3 warmup epochs, cosine floor 1e-6, clipping 1,
  AMP, smoothed training targets .05 and training negative/positive BCE
  weights capped at 8. Validation labels are not smoothed.
- Early stopping maximizes validation macro AUROC, patience 15, delta 1e-4.
  New default budget is 20 epochs, with 50 still configurable.
- Checkpoint keys: epoch, model_state_dict, optimizer_state_dict,
  scheduler_state_dict, scaler_state_dict, early_stopping_best_score and
  early_stopping_counter. History/config are separate CSV/JSON files.
- Validation-only per-class threshold grid .05–.95 (19 candidates) maximizes F1.

Reused: exact architecture computations and state_dict names, training-only
smoothing, BCE weighting, AdamW, clipping, AMP, warmup/cosine, validation model
selection and threshold grid. `MambaEncoder` exposes the head-free path and
`MambaClassifier` adds the original head. Source key compatibility is covered
by the optional real-CUDA test; actual old checkpoint files are absent.

## Evaluation reference

`archive/notebooks/ptb-xl-ecg-supervised-mamba-evaluation-version2.ipynb`, cells 3–55:

- Loads frozen config/checkpoints, reconstructs the architecture and adds a
  return_embedding option. Executes preprocessing notebook again, re-fitting
  normalization rather than loading saved training statistics.
- Inference applies sigmoid and optional validation thresholds. Exports ECG
  IDs, truth, probabilities, decisions and confidence to CSV.
- Computes per-class/overall AUROC, average precision, F1, precision, recall,
  specificity, balanced accuracy, hamming loss and exact-match accuracy.
  New unified evaluator focuses on the metrics requested for this task.
- Generates ROC/PR, binary confusion grids, probability histograms,
  confidence/calibration plots, PCA/t-SNE/UMAP. Captum Integrated Gradients
  and saliency analyze selected examples; error analysis finds FP/FN and
  high-confidence/rare-class errors. These optional research plots remain in
  the reference notebook and are not duplicated in the shared core.
- Profiles parameters, model size, latency, throughput, GPU memory and a
  hand-derived FLOP estimate, then writes an 18-artifact manifest.

Reused: sigmoid inference, classwise metric definitions, binary confusion
counts, CSV prediction structure and validation thresholds. New evaluation
also guards micro metrics, records valid class counts, writes strict JSON null
for undefined ranking metrics, and measures full-loader inference time.
It provides loss curves and confusion grids for any class count.

## Saved outputs and critical limitations

1. Saved preprocessing and training outputs identify **v1.0.1**, 21,837 rows,
   18,885 patients, with 407 unlabeled records; supervised splits 17,111 /
   2,156 / 2,163. These are not verified v1.0.3 counts. The new loader records
   actual metadata SHA256 hashes and IDs; the operator must attach v1.0.3.
2. Training source/output describes 18,117,638 parameters (384-wide, 8 blocks).
   Saved evaluation config instead uses width 128, hidden 256, 6 blocks,
   dropout .35, weight decay 3e-4, patience 5 and 1,673,990 parameters.
   Its thresholds differ too. Evaluation is therefore from a different run.
3. Historical training text reports best validation AUROC .9251 at epoch 11;
   evaluation text reports test AUROC .9196 for the smaller checkpoint.
   These values were read, **not reproduced**, and must not be attributed to
   this implementation or compared as the same experiment.
4. Notebook-wide exec imports run installation, validation sweeps, plots and
   exports, suppress arbitrary exceptions and can leave partially defined state.
   New modules have no notebook execution or install-at-import behavior.
5. Missing files silently become zero waveforms; signal validation's overall
   validity omits length/frequency flags, and its summary counts pass flags as
   failures. New waveform failures raise with ECG ID/path and sampling rate is
   checked. Length handling is an explicit deterministic preprocessing step.
6. Raw normalization fit / filtered inference is inconsistent. Private RNGs
   copied to workers can repeat augmentation streams; an unseeded crop generator
   impairs reproducibility. New worker initialization reseeds augmentation and
   base crop/pad is deterministic. Exact continuation is supported with zero
   workers; workers and CUDA kernels remain platform-dependent.
7. The final QA patient check intersects all three sets rather than checking
   all pairs (the earlier cell correctly checks pairs). New integrity checks
   run before any label exclusion, and reject missing patient IDs/invalid folds.
8. Source scheduler has a zero-length cosine segment for epochs=1; new trainer
   explicitly supports a one-epoch synthetic check. New losses average by
   sample count so the final batch has the right contribution.
9. Evaluation palette indexes tab10 directly and fails for 23+ classes.
   New confusion figures dynamically size their grid. Ranking metrics with
   absent positives/negatives are explicitly documented and tested.
10. The official mapping file contains **23**, not 24, diagnostic subclasses.
    Vocabulary is derived from diagnostic_subclass and never from record
    frequencies. No rare subclass or unlabeled synthetic extra class is added.

Full data training, original checkpoint reconstruction and GPU performance
remain unverified until executed with the actual dataset and compatible CUDA.
