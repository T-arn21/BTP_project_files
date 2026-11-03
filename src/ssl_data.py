"""Label-free, official-training-fold waveform access for reconstruction SSL."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .data import PTBXLDataset, read_waveform, build_official_splits, seed_worker
from .signal import SignalNormalizer
from .cache import preprocessing_identity, select_normalization_ids, ensure_signal_cache, get_normalizer


class SSLWaveforms(Dataset):
    def __init__(self, processed):
        self.processed = processed
        self.ids = processed.ids
        self.augmenter = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ecg_id = self.ids[index]
        x = self.processed.normalizer.transform(self.processed.processed_signal(ecg_id))
        return torch.from_numpy(np.ascontiguousarray(x)), int(ecg_id)


@dataclass
class SSLData:
    dataset: SSLWaveforms
    config: object
    provenance: dict

    def loader(self, batch_size, seed):
        workers = self.config.num_workers
        extra = dict(persistent_workers=True, prefetch_factor=2, timeout=60) if workers else {}
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                          num_workers=workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(seed), **extra)


def prepare_ssl_data(config, *, seed=42, record_limit=None, reader=read_waveform):
    # Never parse scp_codes or build labels; unlabeled training ECGs are retained.
    cfg = replace(config, data_dir=str(Path(config.data_dir).resolve()), augment=False, layout="channels_first")
    metadata = pd.read_csv(Path(cfg.data_dir) / "ptbxl_database.csv", index_col="ecg_id",
                           usecols=["ecg_id", "patient_id", "strat_fold", "filename_lr", "filename_hr"])
    splits = build_official_splits(metadata)  # audit all patient folds, read no held-out waveform
    all_train = list(map(int, splits["train"]))
    if record_limit is not None and record_limit < 1:
        raise ValueError("SSL record_limit must be positive or None")
    ids = all_train if record_limit is None else all_train[:record_limit]
    # Signatures and cache are restricted to training waveforms as well.
    identity = preprocessing_identity(Path(cfg.data_dir), metadata.loc[all_train], cfg, reader)
    fit_ids = select_normalization_ids(all_train, cfg)
    normalizer = SignalNormalizer()
    empty_labels = pd.DataFrame(index=metadata.index)
    processed = PTBXLDataset(metadata, empty_labels, ids, cfg, normalizer, "val", reader, seed)
    cache_path, manifest = None, None
    if cfg.cache_signals:
        cache_path, manifest = ensure_signal_cache(identity, set(ids) | set(fit_ids), cfg, processed.processed_signal)
        processed.cache_path = cache_path
        processed.cache_rows = {int(ecg_id): i for i, ecg_id in enumerate(manifest["ids"])}
    normalizer, normalization = get_normalizer(identity, fit_ids, cfg, processed.processed_signal)
    processed.normalizer = normalizer
    provenance = dict(objective="masked_waveform_reconstruction", diagnostic_labels_used=False,
                      training_folds=list(range(1, 9)), eligible_train_ids=all_train, used_train_ids=ids,
                      excluded_validation_ids=list(map(int, splits["val"])),
                      excluded_test_ids=list(map(int, splits["test"])),
                      train_patient_ids=sorted(map(int, metadata.loc[all_train, "patient_id"].unique())),
                      patient_separation_verified=True, smoke_only=record_limit is not None,
                      preprocessing_identity=identity, normalizer=normalizer.state_dict(),
                      normalization_configuration=normalization["configuration"],
                      normalization_record_ids=fit_ids,
                      signal_cache=dict(path=cache_path, payload_sha256=manifest["payload_sha256"]) if manifest else None)
    return SSLData(SSLWaveforms(processed), cfg, provenance)
