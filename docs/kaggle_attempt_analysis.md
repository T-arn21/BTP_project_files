# Task 3: Kaggle attempt evidence and bottleneck analysis

`ecg-xl-btp.ipynb` is preserved as supplied. Its 40 cells contain the working
source-copy head patch, source-build commands, wheel preservation, isolated
CUDA probe, main smoke cells, and temporary 1000-record normalization patch.
The supplied file has **zero saved outputs**. It therefore documents code and
intent, not independently inspectable successful CUDA results or epoch timings.

The user reports successful source builds on Python 3.13.15, Torch 2.11.0+cu128,
CUDA toolkit 12.8, Tesla T4, CXX11 ABI=True, with causal-conv1d 1.7.0,
mamba-ssm 2.3.2.post1 and WFDB 4.3.1. They also report passing real five-class
and 23-class `(2,12,1000)` forward/backward/optimizer/AMP smoke on an input
directory labeled **1.0.1**. These are user-reported smoke observations, not
newly executed training benchmarks. The research target 1.0.3 remains distinct.

## Confirmed code issues

1. The committed MambaEncoder returned `self.head(pooled)`. The subclass's
   forward then invoked the head a second time, producing the reported
   `(2x5) @ (768x384)` mismatch. The user's local `return pooled` patch is now
   retained permanently. CPU tests run actual encoder/pooling/head code with
   a test double only for the unavailable CUDA block, count exactly one head
   call, and backpropagate for both 5/23 outputs. Real kernel tests remain
   conditional on CUDA.
2. Normalization read/filtered all training waveforms separately on each load.
   Default sampled fit now selects up to 1000 official-training records using
   a dedicated seed; exact full fit remains configurable. Both modes save
   selected IDs, seed, preprocessing identity, mean/std and their configuration.
3. Dataset `__getitem__` repeated WFDB I/O and zero-phase filtering for every
   sample every epoch. A single task-independent float32 memory map now stores
   deterministic filtered, center-padded, **unnormalized/unaugmented** signals.
   Normalization and stochastic augmentation happen after reading that array.
4. Default two-worker startup could obscure an I/O or multiprocessing stall,
   and logging did not start until after a whole epoch. Final defaults use
   workers=0. Optional workers get independent memory-map handles, timeout,
   bounded prefetch and persistent workers. Metadata, source-signature scans,
   cache creation/verification, normalization, trainer/loaders, first batch,
   first optimizer step, batch/epoch metrics and checkpoints now log progress.
   Training history records data-wait time to separate I/O from GPU compute.
5. Best checkpoint loading previously kept duplicate checkpoint tensors on
   the GPU during evaluation. It now loads on CPU, copies strict model weights
   and releases the checkpoint state before inference.

## What the available evidence cannot establish

No saved profiler traces identify the exact point of the reported 30-minute
stall. Near-zero GPU utilization alone cannot distinguish waiting on WFDB,
filtering, a DataLoader worker, initialization, or a transient monitor sample.
It would be misleading to claim a proven deadlock or a measured Kaggle speedup.
The repeated expensive code paths and logging gaps are confirmed from source;
the new first-batch/step and data-wait logs let the next run locate remaining
latency. Initial cache creation still reads/filters each required ECG once and
can be I/O-bound. Its progress is visible and subsequent tasks/epochs reuse it.

## Local measurable validation

Synthetic regression tests count exactly one reader call per required record
during cache creation and **zero additional reader calls for two cached train
passes**. A tiny first batch performs a finite BCE backward and an actual
optimizer update; timing is exported by the profile test. Cached/uncached
normalized samples agree exactly. Modified settings, metadata, source-file
stats, missing completion manifests and corrupted array bytes invalidate cache.
Sequential execution of every final notebook cell is tested in both benchmark
and diagnostic branches using explicit installation/CUDA/model test doubles.
Those CPU mechanics do not measure real Kaggle I/O or Mamba throughput.
