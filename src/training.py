"""Architecture-independent weighted BCE training adapted from reference cells 20-24."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import math
import random
import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from .evaluation import evaluate
from .utils import log_progress


def atomic_checkpoint(state, path):
    """Keep the previous completed checkpoint intact if writing is interrupted."""
    path = Path(path)
    partial = path.with_suffix(".partial.pth")
    torch.save(state, partial)
    os.replace(partial, path)


def make_pos_weight(labels, maximum=8.0):
    labels = np.asarray(labels)
    positives = labels.sum(axis=0)
    # No positives: no useful positive loss term, retain finite neutral weight.
    ratio = np.divide(len(labels) - positives, positives, out=np.ones_like(positives, dtype=float), where=positives > 0)
    return torch.tensor(np.minimum(ratio, maximum), dtype=torch.float32)


def load_checkpoint(path, model, class_names=None, device="cpu"):
    """Load only trusted, locally produced weight-only-compatible checkpoints.

    Original notebook checkpoints also contain model_state_dict. Supply their
    correct architecture and explicitly verify their class order before reuse.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    if class_names is not None and state.get("class_names") != list(class_names):
        raise ValueError("Checkpoint class order/task mismatch (or missing vocabulary)")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return state


class Trainer:
    def __init__(self, model, config, class_names, directory, pos_weight=None, metadata=None):
        init_start = perf_counter()
        log_progress("Trainer initialization: moving model, creating BCE/AdamW/scheduler")
        self.config, self.class_names = config, list(class_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
        self.model = model.to(self.device)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device) if pos_weight is not None else None)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        warmup = min(max(config.warmup_epochs, 0), config.epochs - 1)
        if warmup:
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [
                torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs - warmup, eta_min=config.min_lr)], [warmup])
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=config.min_lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and self.device.type == "cuda")
        self.history, self.best_score, self.counter = [], -math.inf, 0
        log_progress(f"Trainer ready on {self.device} in {perf_counter() - init_start:.2f}s")

    def _epoch(self, loader, train):
        self.model.train(train)
        total, count, ys, ps = 0.0, 0, [], []
        stage = "train" if train else "validation"
        epoch_start, last_log = perf_counter(), perf_counter()
        wait_start, data_wait = perf_counter(), 0.0
        log_progress(f"{stage}: waiting for first batch ({len(loader)} batches, workers={loader.num_workers})")
        for batch_index, (x, y, _) in enumerate(loader, 1):
            batch_wait = perf_counter() - wait_start
            data_wait += batch_wait
            if batch_index == 1:
                log_progress(f"{stage}: first batch loaded in {batch_wait:.3f}s; shape={tuple(x.shape)}; starting forward{'/backward' if train else ''}")
            x, y = x.to(self.device), y.to(self.device).float()
            if self.config.data.layout == "time_first":
                x = x.transpose(1, 2).contiguous()
            targets = y * (1 - self.config.label_smoothing) + self.config.label_smoothing / 2 if train else y
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    logits = self.model(x)
                    if logits.shape != y.shape:
                        raise ValueError("Expected logits matching multilabel target shape")
                    loss = self.criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite BCE loss")
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if batch_index == 1 or batch_index % self.config.log_every_batches == 0 or perf_counter() - last_log >= 15 or batch_index == len(loader):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                log_progress(f"{stage} batch {batch_index}/{len(loader)}: BCE={loss.item():.4f}, elapsed={perf_counter() - epoch_start:.1f}s, data_wait={data_wait:.2f}s" + ("; optimizer step completed" if train else ""))
                last_log = perf_counter()
            total += loss.item() * len(y)
            count += len(y)
            ys.append(y.detach().cpu().numpy())
            ps.append(logits.detach().sigmoid().float().cpu().numpy())
            wait_start = perf_counter()
        if not count:
            raise ValueError("Training and validation loaders must be nonempty")
        log_progress(f"{stage}: computing epoch metrics for {count} records")
        metrics, _ = evaluate(np.concatenate(ys), np.concatenate(ps), self.class_names)
        self.last_data_wait_seconds = data_wait
        return total / count, metrics["macro_auroc"]

    def fit(self, train_loader, val_loader, resume_from=None):
        start_epoch = 1
        if resume_from:
            state = load_checkpoint(resume_from, self.model, self.class_names, self.device)
            if state["config"] != asdict(self.config) or state["metadata"] != self.metadata:
                raise ValueError("Resume requires identical config, preprocessing and split provenance")
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            self.best_score, self.counter, self.history = state["best_score"], state["counter"], state["history"]
            start_epoch = state["epoch"] + 1
            if not (self.directory / "best.pth").exists():
                raise ValueError("Resume in the original run directory with best.pth present")
            torch.set_rng_state(state["rng_cpu"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_cuda"]])
            random.setstate(state["rng_python"])
            np.random.set_state((state["rng_numpy"][0], np.array(state["rng_numpy"][1], dtype=np.uint32), *state["rng_numpy"][2:]))
            if train_loader.generator is not None and state["loader_rng"] is not None:
                train_loader.generator.set_state(state["loader_rng"].cpu())
            aug = getattr(train_loader.dataset, "augmenter", None)
            if aug is not None and state["augmentation_rng"] is not None:
                aug.rng.bit_generator.state = state["augmentation_rng"]
        if self.counter >= self.config.patience:
            return pd.DataFrame(self.history)
        for epoch in range(start_epoch, self.config.epochs + 1):
            start = perf_counter()
            log_progress(f"Epoch {epoch}/{self.config.epochs} starting")
            train_loss, train_auc = self._epoch(train_loader, True)
            train_data_wait = self.last_data_wait_seconds
            val_loss, val_auc = self._epoch(val_loader, False)
            self.scheduler.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                          train_auroc=train_auc, val_auroc=val_auc,
                          learning_rate=self.optimizer.param_groups[0]["lr"], epoch_seconds=perf_counter() - start,
                          train_data_wait_seconds=train_data_wait, val_data_wait_seconds=self.last_data_wait_seconds)
            self.history.append(record)
            # If every validation AUROC is undefined, consistently monitor -val_loss.
            score = val_auc if np.isfinite(val_auc) else -val_loss
            improved = score > self.best_score + self.config.min_delta
            self.best_score = score if improved else self.best_score
            self.counter = 0 if improved else self.counter + 1
            np_rng = np.random.get_state()
            aug = getattr(train_loader.dataset, "augmenter", None)
            state = dict(epoch=epoch, model_state_dict=self.model.state_dict(),
                         optimizer_state_dict=self.optimizer.state_dict(), scheduler_state_dict=self.scheduler.state_dict(),
                         scaler_state_dict=self.scaler.state_dict(), best_score=self.best_score, counter=self.counter,
                         history=self.history, config=asdict(self.config), class_names=self.class_names, metadata=self.metadata,
                         rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                         rng_python=random.getstate(), rng_numpy=(np_rng[0], np_rng[1].tolist(), *np_rng[2:]),
                         loader_rng=train_loader.generator.get_state() if train_loader.generator is not None else None,
                         augmentation_rng=aug.rng.bit_generator.state if aug is not None else None)
            save_start = perf_counter()
            log_progress(f"Saving checkpoint epoch {epoch} to {self.directory}")
            if improved:
                atomic_checkpoint(state, self.directory / "best.pth")
            atomic_checkpoint(state, self.directory / "last.pth")
            history_partial = self.directory / "history.partial.csv"
            pd.DataFrame(self.history).to_csv(history_partial, index=False)
            os.replace(history_partial, self.directory / "history.csv")
            log_progress(f"Checkpoint saved in {perf_counter() - save_start:.2f}s")
            log_progress(f"Epoch {epoch}/{self.config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_AUROC={val_auc:.4f}")
            if self.counter >= self.config.patience:
                break
        return pd.DataFrame(self.history)
