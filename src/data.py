"""Task-independent waveform processing and official patient-disjoint folds."""
import hashlib
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .config import DataConfig
from .labels import build_labels, class_statistics
from .signal import preprocess_signal, SignalNormalizer
from .augmentation import ECGAugmentation


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

    def __len__(self):
        return len(self.ids)

    def processed_signal(self, ecg_id):
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
                torch.tensor(self.labels.loc[ecg_id].to_numpy(), dtype=torch.float32), int(ecg_id))


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
        return {name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train",
                                 num_workers=self.config.num_workers,
                                 pin_memory=torch.cuda.is_available(), drop_last=False,
                                 worker_init_fn=seed_worker,
                                 generator=torch.Generator().manual_seed(self.seed))
                for name, ds in self.datasets.items()}


def load_ptbxl(task="superclass", data_dir=None, config=None, seed=42, reader=read_waveform):
    """Read metadata and fit shared normalization. Does not download any files.

    Stats use all official training records before task-specific exclusions, so
    exactly the same processed waveform is returned by either classification task.
    """
    from dataclasses import replace
    config = replace(config or DataConfig())
    config.data_dir = str(data_dir or config.data_dir or os.environ.get("PTBXL_DATA_DIR", ""))
    if not config.data_dir:
        raise ValueError("Set data_dir or PTBXL_DATA_DIR to the PTB-XL v1.0.3 root")
    root = Path(config.data_dir)
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
    normalizer = SignalNormalizer()
    fit_ds = PTBXLDataset(metadata, labels, original_splits["train"], config, normalizer, "val", reader)
    normalizer.fit(fit_ds.processed_signal(i) for i in original_splits["train"])
    datasets = {k: PTBXLDataset(metadata, labels, ids, config, normalizer, k, reader, seed)
                for k, ids in splits.items()}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("ptbxl_database.csv", "scp_statements.csv")}
    provenance = {"requested_version": "1.0.3", "metadata_sha256": hashes,
                  "records_before_exclusions": len(metadata),
                  "normalization_record_ids": list(map(int, original_splits["train"]))}
    return PTBXLData(metadata, labels, splits, datasets, normalizer,
                     class_statistics(labels, splits), missing, provenance, config, seed)
