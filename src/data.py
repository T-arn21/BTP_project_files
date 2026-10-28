"""Task-independent waveform processing and official patient-disjoint folds."""
import hashlib
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from time import perf_counter
from torch.utils.data import Dataset, DataLoader
from .config import DataConfig
from .labels import build_labels, class_statistics
from .signal import preprocess_signal, SignalNormalizer
from .augmentation import ECGAugmentation
from .cache import preprocessing_identity, select_normalization_ids, ensure_signal_cache, get_normalizer
from .utils import log_progress


def build_official_splits(metadata):
    if metadata.index.has_duplicates or metadata.patient_id.isna().any():
        raise ValueError("ECG IDs must be unique and patient IDs must be present")
    if not metadata.strat_fold.isin(range(1, 11)).all():
        raise ValueError("strat_fold must contain integers 1..10")
    splits = {"train": metadata.index[metadata.strat_fold <= 8],
              "val": metadata.index[metadata.strat_fold == 9],
              "test": metadata.index[metadata.strat_fold == 10]}
    for a, b in combinations(splits, 2):
        if set(metadata.loc[splits[a], "patient_id"]) & set(metadata.loc[splits[b], "patient_id"]):
            raise ValueError(f"Patient leakage between {a} and {b}")
    if any(len(ids) == 0 for ids in splits.values()):
        raise ValueError("Every split must contain records")
    return splits


def read_waveform(path, sampling_rate):
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install wfdb to load actual PTB-XL records") from exc
    signal, fields = wfdb.rdsamp(str(path))
    if fields.get("fs") != sampling_rate:
        raise ValueError(f"Wrong sampling frequency for {path}: {fields.get('fs')}")
    return np.asarray(signal, dtype=np.float32).T


class PTBXLDataset(Dataset):
    def __init__(self, metadata, labels, ids, config, normalizer, split,
                 reader=read_waveform, seed=42):
        self.metadata, self.labels, self.ids = metadata, labels, list(ids)
        self.config, self.normalizer, self.reader = config, normalizer, reader
        self.augmenter = ECGAugmentation(seed=seed) if split == "train" and config.augment else None
        self.cache_path, self.cache_rows, self._cached_signals = None, {}, None
        self._label_values = labels.loc[self.ids].to_numpy(dtype=np.float32) if all(i in labels.index for i in self.ids) else None

    def __len__(self):
        return len(self.ids)

    def processed_signal(self, ecg_id):
        if self.cache_path is not None:
            if self._cached_signals is None:
                self._cached_signals = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
            return np.array(self._cached_signals[self.cache_rows[int(ecg_id)]], copy=True)
        column = "filename_lr" if self.config.sampling_rate == 100 else "filename_hr"
        path = Path(self.config.data_dir) / self.metadata.loc[ecg_id, column]
        try:
            return preprocess_signal(self.reader(path, self.config.sampling_rate),
                                     self.config.sampling_rate,
                                     self.config.sampling_rate * self.config.duration,
                                     self.config.apply_filter)
        except Exception as exc:
            raise RuntimeError(f"Failed ECG {ecg_id} at {path}: {exc}") from exc

    def __getitem__(self, index):
        ecg_id = self.ids[index]
        x = self.normalizer.transform(self.processed_signal(ecg_id))
        if self.augmenter:
            x = self.augmenter(x)
        if self.config.layout == "time_first":
            x = x.T
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self._label_values[index].copy()), int(ecg_id))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_signals"] = None  # workers independently reopen, never pickle a GB array
        return state


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if info.dataset.augmenter:
        info.dataset.augmenter.rng = np.random.default_rng(seed)


@dataclass
class PTBXLData:
    metadata: pd.DataFrame
    labels: pd.DataFrame
    splits: dict
    datasets: dict
    normalizer: SignalNormalizer
    statistics: pd.DataFrame
    dropped_ids: list
    provenance: dict
    config: DataConfig
    seed: int

    @property
    def class_names(self):
        return list(self.labels.columns)

    def loaders(self, batch_size=64):
        start = perf_counter()
        log_progress(f"DataLoader initialization: workers={self.config.num_workers}, batch={batch_size}")
        workers = self.config.num_workers
        extra = dict(persistent_workers=True, prefetch_factor=2, timeout=60) if workers > 0 else {}
        result = {name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train",
                                 num_workers=self.config.num_workers,
                                 pin_memory=torch.cuda.is_available(), drop_last=False,
                                 worker_init_fn=seed_worker,
                                 generator=torch.Generator().manual_seed(self.seed), **extra)
                for name, ds in self.datasets.items()}
        log_progress(f"DataLoaders ready in {perf_counter() - start:.2f}s: " + str({k: len(v) for k, v in result.items()}))
        return result


def load_ptbxl(task="superclass", data_dir=None, config=None, seed=42, reader=read_waveform, diagnostic_limits=None):
    """Read metadata and fit shared normalization. Does not download any files.

    Stats use all official training records before task-specific exclusions, so
    exactly the same processed waveform is returned by either classification task.
    """
    from dataclasses import replace
    config = replace(config or DataConfig())
    config.data_dir = str(data_dir or config.data_dir or os.environ.get("PTBXL_DATA_DIR", ""))
    if not config.data_dir:
        raise ValueError("Set data_dir or PTBXL_DATA_DIR to the PTB-XL v1.0.3 root")
    start = perf_counter()
    root = Path(config.data_dir).resolve()
    config.data_dir = str(root)
    log_progress(f"Loading {task} metadata from {root}; declared release={config.dataset_version}")
    metadata = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    original_splits = build_official_splits(metadata)  # audit before exclusions
    labels = build_labels(metadata, statements, task)
    missing = labels.index[labels.sum(axis=1).eq(0)].tolist()
    if missing and config.unlabeled == "error":
        raise ValueError(f"{len(missing)} records have no diagnostic labels")
    labels = labels.loc[labels.sum(axis=1).gt(0)]
    splits = {k: ids.intersection(labels.index, sort=False) for k, ids in original_splits.items()}
    if any(len(ids) == 0 for ids in splits.values()):
        raise ValueError("A supervised split is empty after excluding unlabeled records")
    if diagnostic_limits is not None:
        if set(diagnostic_limits) != {"train", "val"} or any(v < 1 for v in diagnostic_limits.values()):
            raise ValueError("Diagnostic limits must include positive train/val counts only")
        splits = {k: splits[k][:limit] for k, limit in diagnostic_limits.items()}
        log_progress(f"DIAGNOSTIC ONLY: limited records { {k: len(v) for k, v in splits.items()} }; no test evaluation")
    log_progress(f"Metadata/labels/patient audit ready in {perf_counter() - start:.2f}s")
    identity = preprocessing_identity(root, metadata, config, reader)
    fit_ids = select_normalization_ids(original_splits["train"], config)
    normalizer = SignalNormalizer()
    fit_ds = PTBXLDataset(metadata, labels, original_splits["train"], config, normalizer, "val", reader)
    cache_path, cache_manifest = None, None
    if config.cache_signals:
        needed = set(fit_ids).union(*(set(ids) for ids in splits.values()))
        cache_path, cache_manifest = ensure_signal_cache(identity, needed, config, fit_ds.processed_signal)
        fit_ds.cache_path = cache_path
        fit_ds.cache_rows = {ecg_id: row for row, ecg_id in enumerate(cache_manifest["ids"])}
    normalizer, normalization = get_normalizer(identity, fit_ids, config, fit_ds.processed_signal)
    datasets = {k: PTBXLDataset(metadata, labels, ids, config, normalizer, k, reader, seed)
                for k, ids in splits.items()}
    for ds in datasets.values():
        ds.cache_path = cache_path
        ds.cache_rows = fit_ds.cache_rows
    provenance = {"declared_dataset_version": config.dataset_version, "release_verified": False,
                  "dataset_root": str(root), "metadata_sha256": identity["metadata_sha256"],
                  "records_before_exclusions": len(metadata),
                  "normalization_record_ids": fit_ids,
                  "normalization_configuration": normalization["configuration"],
                  "preprocessing_identity": identity,
                  "signal_cache": {"path": cache_path, "payload_sha256": cache_manifest["payload_sha256"]} if cache_manifest else None,
                  "diagnostic_only": diagnostic_limits is not None,
                  "diagnostic_limits": diagnostic_limits}
    log_progress(f"{task} data ready in {perf_counter() - start:.2f}s")
    return PTBXLData(metadata, labels, splits, datasets, normalizer,
                     class_statistics(labels, splits), missing, provenance, config, seed)
