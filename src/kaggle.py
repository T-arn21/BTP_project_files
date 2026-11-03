"""Small, explicit preflight checks for attached PTB-XL data and CUDA Mamba.

No installation, downloads, GPU work or training occurs at import time.
The smoke path never reads test-fold waveforms or produces benchmark metrics.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .config import DataConfig
from .data import PTBXLDataset, build_official_splits, read_waveform
from .labels import build_labels, class_statistics
from .signal import SignalNormalizer
from .numerics import fp32_bce, named_finite, optimizer_tensors


def find_ptbxl_root(explicit=None, search_roots=("/kaggle/input",)):
    """Require one unambiguous dataset; an invalid explicit path never falls back."""
    required = ("ptbxl_database.csv", "scp_statements.csv", "records100")
    def valid(path):
        return all((path / name).exists() for name in required)
    if explicit:
        root = Path(explicit).expanduser().resolve()
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(f"PTBXL_DATA_DIR={root} is incomplete; missing {missing}. "
                                    "Choose the extracted folder containing both CSVs and records100/.")
        return root
    candidates = set()
    for base in map(Path, search_roots):
        if base.is_dir():
            for csv in base.rglob("ptbxl_database.csv"):
                if valid(csv.parent):
                    candidates.add(csv.parent.resolve())
    if len(candidates) != 1:
        listing = "\n".join(str(p) for p in sorted(candidates)) or "(none)"
        raise FileNotFoundError("Expected exactly one extracted PTB-XL dataset. "
                                "Attach v1.0.3 with records100/, or set PTBXL_DATA_DIR explicitly. "
                                f"Candidates:\n{listing}")
    return candidates.pop()


def audit_metadata(root, dataset_version="unknown"):
    """Verify both official tasks and all patient folds without reading waveforms."""
    root = Path(root)
    try:
        metadata = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
        statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read PTB-XL metadata at {root}: {exc}") from exc
    required = {"scp_codes", "patient_id", "strat_fold", "filename_lr"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"ptbxl_database.csv missing columns: {sorted(required - set(metadata.columns))}")
    required_mapping = {"diagnostic", "diagnostic_class", "diagnostic_subclass"}
    if not required_mapping.issubset(statements.columns):
        raise ValueError(f"scp_statements.csv missing columns: {sorted(required_mapping - set(statements.columns))}")
    splits = build_official_splits(metadata)
    targets = {task: build_labels(metadata, statements, task) for task in ("superclass", "subclass")}
    if metadata.filename_lr.isna().any():
        raise ValueError("Some records have no filename_lr for 100 Hz loading")
    for labels in targets.values():
        if not np.isin(labels.to_numpy(), [0, 1]).all():
            raise ValueError("Diagnostic targets are not binary")
    summary = {"records": len(metadata), "patients": metadata.patient_id.nunique(),
               "declared_dataset_version": dataset_version, "release_verified": False,
               "dataset_root": str(root.resolve()),
               "version_note": "Operator-declared release; CSV schema alone cannot certify the version.",
               "metadata_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in ("ptbxl_database.csv", "scp_statements.csv")},
               "splits": {name: {"records": len(ids), "patients": metadata.loc[ids, "patient_id"].nunique()}
                          for name, ids in splits.items()},
               "tasks": {task: {"classes": list(labels.columns),
                                "unlabeled_records": int(labels.sum(axis=1).eq(0).sum())}
                         for task, labels in targets.items()}, "patient_overlap": False}
    statistics = {}
    for task, labels in targets.items():
        labeled_splits = {name: ids[labels.loc[ids].sum(axis=1).gt(0)] for name, ids in splits.items()}
        if any(len(ids) == 0 for ids in labeled_splits.values()):
            raise ValueError(f"Empty supervised split for {task}")
        statistics[task] = class_statistics(labels, labeled_splits)
    return {"root": root, "metadata": metadata, "targets": targets,
            "splits": splits, "summary": summary, "statistics": statistics}


def prepare_smoke_data(audit, task, *, fit_records=16, batch_size=2, config=None, reader=read_waveform):
    """Fit a tiny training-only normalizer and return one train/val batch.

    This normalization is smoke-only; the full experiment independently fits all
    training records. Unmodified 100 Hz records must actually have (12,1000).
    """
    if fit_records < 1 or batch_size < 1:
        raise ValueError("fit_records and batch_size must be positive")
    metadata, labels = audit["metadata"], audit["targets"][task]
    train_ids = audit["splits"]["train"]
    train_ids = train_ids[labels.loc[train_ids].sum(axis=1).gt(0)]
    val_ids = audit["splits"]["val"]
    val_ids = val_ids[labels.loc[val_ids].sum(axis=1).gt(0)]
    config = replace(config or DataConfig(), data_dir=str(audit["root"]), augment=False, num_workers=0)
    if config.sampling_rate != 100 or config.duration != 10 or config.layout != "channels_first":
        raise ValueError("Kaggle smoke preflight requires 100 Hz, 10 s, channels_first data")
    def checked_reader(path, fs):
        x = reader(path, fs)
        if x.shape != (12, 1000):
            raise ValueError(f"Expected an actual 100 Hz, 10-second, 12-lead record at {path}; got {x.shape}")
        return x
    norm = SignalNormalizer()
    fit_ds = PTBXLDataset(metadata, labels, train_ids[:fit_records], config, norm, "val", checked_reader)
    norm.fit(fit_ds.processed_signal(i) for i in fit_ds.ids)
    batches = {}
    ids_used = {}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        ds = PTBXLDataset(metadata, labels, ids[:batch_size], config, norm, split, checked_reader)
        batches[split] = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False)))
        x, y, _ = batches[split]
        if not torch.isfinite(x).all() or y.shape[1] != len(labels.columns):
            raise ValueError("Invalid preprocessed smoke batch")
        ids_used[split] = list(map(int, ds.ids))
    return {"batches": batches, "class_names": list(labels.columns),
            "normalizer": norm.state_dict(), "normalization_record_ids": list(map(int, fit_ds.ids)),
            "record_ids": ids_used, "task": task, "purpose": "smoke_only_no_test_evaluation"}


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. In Kaggle choose Settings > Accelerator > GPU, "
                           "restart the session, and verify torch.version.cuda and nvidia-smi. "
                           "Do not install a CPU Torch wheel over Kaggle's GPU runtime.")


def cached_smoke_data(data, batch_size=2):
    """Use the same cached signals/statistics as training, without augmentation/test access."""
    config = replace(data.config, augment=False, num_workers=0)
    batches, record_ids = {}, {}
    for split in ("train", "val"):
        source = data.datasets[split]
        ds = PTBXLDataset(data.metadata, data.labels, source.ids[:batch_size],
                         config, data.normalizer, "val", source.reader)
        ds.cache_path, ds.cache_rows = source.cache_path, source.cache_rows
        batches[split] = next(iter(DataLoader(ds, batch_size=batch_size)))
        record_ids[split] = list(map(int, ds.ids))
    return dict(batches=batches, class_names=data.class_names,
                normalizer=data.normalizer.state_dict(),
                normalization_record_ids=data.provenance["normalization_record_ids"],
                record_ids=record_ids, task="superclass" if len(data.class_names) == 5 else "subclass",
                purpose="smoke_only_no_test_evaluation")


def smoke_forward_backward(model, batch, *, device="cuda", mixed_precision=True, require_gpu=True, learning_rate=3e-4):
    """One real optimization step followed by one validation forward, no metrics tuning.

    Caller supplies the preserved Mamba model. CPU mode exists only to test this
    machinery locally with a test adapter; it does not validate Mamba kernels.
    """
    if require_gpu:
        require_cuda()
        if torch.device(device).type != "cuda":
            raise ValueError("GPU smoke test requires a CUDA device")
    model.to(device).train()
    x, y, _ = batch["batches"]["train"]
    x, y = x.to(device), y.to(device)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise RuntimeError("Invalid smoke inputs/targets; no optimizer step performed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and torch.device(device).type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(torch.device(device).type, enabled=scaler.is_enabled()):
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError(f"Logits {logits.shape} do not match targets {y.shape}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite smoke logits before FP32 BCE")
    loss = fp32_bce(torch.nn.BCEWithLogitsLoss(), logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite smoke BCE loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Missing or nonfinite smoke gradients")
    if not any(torch.count_nonzero(g).item() for g in grads):
        raise RuntimeError("All smoke gradients are zero")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Nonfinite smoke gradient norm")
    scaler.step(optimizer)
    scaler.update()
    if not named_finite(list(model.named_parameters())) or not named_finite(optimizer_tensors(model, optimizer)):
        raise RuntimeError("Nonfinite smoke model/optimizer state after update")
    model.eval()
    xv, yv, _ = batch["batches"]["val"]
    with torch.no_grad(), torch.amp.autocast(torch.device(device).type, enabled=scaler.is_enabled()):
        val_logits = model(xv.to(device))
        if val_logits.shape != yv.shape or not torch.isfinite(val_logits).all():
            raise RuntimeError("Invalid validation smoke logits")
    val_loss = fp32_bce(torch.nn.BCEWithLogitsLoss(), val_logits, yv.to(device))
    if not torch.isfinite(val_loss):
        raise RuntimeError("Nonfinite validation smoke loss")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return {"purpose": batch["purpose"], "task": batch["task"], "device": str(device),
            "class_names": batch["class_names"], "train_shape": list(x.shape),
            "train_bce": loss.item(), "validation_bce": val_loss.item(),
            "grad_norm_before_clip": grad_norm.item(), "finite_gradients": True,
            "optimizer_steps": 1, "amp_enabled": scaler.is_enabled(),
            "learning_rate": learning_rate, "loss_computation": "float32_autocast_disabled",
            "parameters": sum(p.numel() for p in model.parameters()),
            "normalization_record_ids": batch["normalization_record_ids"],
            "record_ids": batch["record_ids"], "normalizer": batch["normalizer"]}
