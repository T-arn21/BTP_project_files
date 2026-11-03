# Archived notebooks

Historical/reference notebooks, moved byte-for-byte from the repository root:

- 04_btp_experiments.ipynb - development implementation/execution reference.
- ecg-xl-btp.ipynb - supplied Kaggle attempt code.
- preprocessing-reusable-data-pipeline.ipynb - original preprocessing reference.
- ptb-xl-ecg-supervised-mamba-trainin-v2-improved.ipynb - original architecture/training reference.
- ptb-xl-ecg-supervised-mamba-evaluation-version2.ipynb - original evaluation reference.

Their outputs, configurations and original source are preserved; they are not
maintained Run All entry points and can contain obsolete paths/environment setup.
Use the six root experiment notebooks or the retained unified Kaggle runner.
Some archived code assumes repository-root working directory. Compatibility and
syntax tests still inspect these files at their new locations.

`manifest.json` records their SHA256s at the merged-main base dc59970. The
validator checks that the archived bytes remain unchanged. Git history retains
their original root locations. [Historical analysis](../../docs/history/README.md)
explains which stored outputs were evidence and which outcomes were only reported.
