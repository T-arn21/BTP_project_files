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
from .utils import write_json
from .config import checkpoint_config_matches
from .cache import digest_file
from .numerics import fp32_bce, tensor_finite, named_finite, optimizer_tensors, failure_report


def atomic_checkpoint(state, path):
    """Keep the previous completed checkpoint intact if writing is interrupted."""
    path = Path(path)
    partial = path.with_suffix(".partial.pth")
    torch.save(state, partial)
    os.replace(partial, path)


def make_pos_weight(labels, maximum=8.0):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Class weights require nonempty binary finite training targets")
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Positive-class weight cap must be finite and positive")
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
    bad = [name for name, value in state["model_state_dict"].items()
           if torch.is_tensor(value) and not tensor_finite(value)]
    if bad:
        raise ValueError(f"Checkpoint contains nonfinite model tensors: {bad}")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return state


class Trainer:
    def __init__(self, model, config, class_names, directory, pos_weight=None, metadata=None):
        init_start = perf_counter()
        log_progress(f"Trainer initialization: moving model, creating {config.loss_type.upper()}/AdamW/scheduler")
        self.config, self.class_names = config, list(class_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
        self.model = model.to(self.device)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        if config.loss_type == "asl":
            from .losses import AsymmetricLoss
            if pos_weight is not None or config.use_pos_weight or config.label_smoothing != 0 or config.mixed_precision:
                raise ValueError("ASL must not receive BCE pos_weight/smoothing or mixed precision")
            self.criterion = AsymmetricLoss(config.asl_gamma_pos, config.asl_gamma_neg, config.asl_clip)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device).float() if pos_weight is not None else None)
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
        self.scheduler_continuation = None
        self.current_epoch, self.batch_index, self.stage = 0, 0, "initialization"
        self._batch = dict(record_ids=[], inputs=None, targets=None, loss_targets=None, logits=None, loss=None)
        weights = getattr(self.criterion, "pos_weight", None)
        if weights is not None and (weights.shape != (len(self.class_names),) or not tensor_finite(weights) or (weights < 0).any()):
            self._fail("Invalid positive-class weights", "class_weights")
        self._check_state("initialization")
        write_json(self.directory / "numerical_settings.json", dict(
            loss_type=config.loss_type, asl=dict(gamma_pos=config.asl_gamma_pos, gamma_neg=config.asl_gamma_neg,
                clip=config.asl_clip, focal_weight_gradient="detached", reduction="mean_batch_and_labels") if config.loss_type == "asl" else None,
            numerical_stability_mode=config.numerical_stability_mode, mixed_precision_requested=config.mixed_precision,
            amp_enabled=self.scaler.is_enabled(), learning_rate=config.learning_rate,
            loss_computation="float32_autocast_disabled", grad_clip_norm=config.grad_clip_norm,
            error_if_nonfinite=True, optimizer="AdamW", optimizer_eps=self.optimizer.defaults["eps"],
            pos_weight=weights.detach().cpu().tolist() if weights is not None else None,
            label_smoothing=config.label_smoothing, invalid_batch_policy="abort_no_skip_no_retry"))
        log_progress(f"Trainer ready on {self.device} in {perf_counter() - init_start:.2f}s")

    def _fail(self, reason, phase, detail=None):
        failure_report(self.directory, reason=reason, phase=phase, epoch=self.current_epoch,
                       batch_index=self.batch_index, stage=self.stage, **self._batch,
                       model=self.model, optimizer=self.optimizer, scaler=self.scaler, config=self.config,
                       pos_weight=getattr(self.criterion, "pos_weight", None), normalization=self.metadata.get("normalizer"), detail=detail)

    def _check_state(self, phase):
        if not named_finite(list(self.model.named_parameters())):
            self._fail("Nonfinite model parameters", phase)
        if not named_finite(optimizer_tensors(self.model, self.optimizer)):
            self._fail("Nonfinite optimizer state", phase)
        if not math.isfinite(self.scaler.get_scale()) or self.scaler.get_scale() <= 0:
            self._fail("Invalid AMP scaler scale", phase)
        if any(not math.isfinite(g["lr"]) or g["lr"] <= 0 for g in self.optimizer.param_groups):
            self._fail("Invalid optimizer learning rate", phase)

    def _epoch(self, loader, train):
        self.model.train(train)
        total, count, ys, ps = 0.0, 0, [], []
        stage = "train" if train else "validation"
        self.stage, self.batch_index = stage, 0
        self._batch = dict(record_ids=[], inputs=None, targets=None, loss_targets=None, logits=None, loss=None)
        self._check_state("epoch_start")
        epoch_start, last_log = perf_counter(), perf_counter()
        wait_start, data_wait = perf_counter(), 0.0
        log_progress(f"{stage}: waiting for first batch ({len(loader)} batches, workers={loader.num_workers})")
        for batch_index, (x, y, ecg_ids) in enumerate(loader, 1):
            self.batch_index = batch_index
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            batch_wait = perf_counter() - wait_start
            data_wait += batch_wait
            if batch_index == 1:
                log_progress(f"{stage}: first batch loaded in {batch_wait:.3f}s; shape={tuple(x.shape)}; starting forward{'/backward' if train else ''}")
            x, y = x.to(self.device).float(), y.to(self.device).float()
            if self.config.data.layout == "time_first":
                x = x.transpose(1, 2).contiguous()
            targets = y * (1 - self.config.label_smoothing) + self.config.label_smoothing / 2 if train else y
            self._batch = dict(record_ids=ecg_ids, inputs=x, targets=y, loss_targets=targets, logits=None, loss=None)
            if not tensor_finite(x):
                self._fail("Nonfinite ECG inputs", "inputs")
            if not tensor_finite(y) or not tensor_finite(targets):
                self._fail("Nonfinite targets", "targets")
            if not bool(((y == 0) | (y == 1)).all().item()):
                self._fail("Nonbinary multilabel targets", "targets")
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    logits = self.model(x)
                self._batch["logits"] = logits
                if not tensor_finite(logits):
                    self._fail("Nonfinite model logits", "logits")
                if logits.shape != y.shape:
                    raise ValueError("Expected logits matching multilabel target shape")
                loss = fp32_bce(self.criterion, logits, targets)
                self._batch["loss"] = loss
                if not tensor_finite(loss):
                    self._fail(f"Nonfinite {self.config.loss_type.upper()} loss", "loss")
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    gradients = [(name, p.grad) for name, p in self.model.named_parameters() if p.grad is not None]
                    if not gradients or not named_finite(gradients):
                        self._fail("Nonfinite or missing gradients", "gradients")
                    if self.config.ssl_encoder_checkpoint and self.config.diagnostic:
                        from .finetuning import audit_smoke_gradients
                        audit_smoke_gradients(self, gradients)
                    try:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True)
                    except RuntimeError as exc:
                        self._fail("Nonfinite gradient norm", "gradient_clipping", detail=str(exc))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self._check_state("post_optimizer")

            if batch_index == 1 or batch_index % self.config.log_every_batches == 0 or perf_counter() - last_log >= 15 or batch_index == len(loader):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                log_progress(f"{stage} batch {batch_index}/{len(loader)}: {self.config.loss_type.upper()}={loss.item():.4f}, elapsed={perf_counter() - epoch_start:.1f}s, data_wait={data_wait:.2f}s" + ("; optimizer step completed" if train else ""))
                last_log = perf_counter()
            total += loss.item() * len(y)
            count += len(y)
            ys.append(y.detach().cpu().numpy())
            ps.append(logits.detach().float().sigmoid().cpu().numpy())
            self._batch = dict(record_ids=[], inputs=None, targets=None, loss_targets=None, logits=None, loss=None)
            wait_start = perf_counter()
        if not count:
            raise ValueError("Training and validation loaders must be nonempty")
        log_progress(f"{stage}: computing epoch metrics for {count} records")
        metrics, _ = evaluate(np.concatenate(ys), np.concatenate(ps), self.class_names)
        self.last_data_wait_seconds = data_wait
        return total / count, metrics["macro_auroc"]

    def fit(self, train_loader, val_loader, resume_from=None, completed_extension=None):
        start_epoch = 1
        if resume_from:
            state = load_checkpoint(resume_from, self.model, self.class_names, self.device)
            matches = checkpoint_config_matches(state["config"], self.config)
            if completed_extension:
                from .resume_extension import extension_config_matches
                matches = extension_config_matches(state["config"], self.config)
                if (completed_extension["source_files"]["last.pth"] != digest_file(resume_from)
                        or completed_extension["source_epoch"] != state["epoch"]):
                    raise ValueError("Completed extension checkpoint differs from validated snapshot")
            if not matches or state["metadata"] != self.metadata:
                raise ValueError("Resume requires identical config, preprocessing and split provenance")
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            self.scheduler_continuation = (completed_extension["scheduler_continuation"] if completed_extension
                                           else state.get("scheduler_continuation"))
            if self.scheduler_continuation:
                if (self.scheduler_continuation["policy"] != "hold_completed_lr" or
                        self.scheduler_continuation["held_lrs"] != [g["lr"] for g in self.optimizer.param_groups]):
                    raise ValueError("Invalid scheduler continuation policy/LR")
            self._check_state("checkpoint_restore")
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
            self.current_epoch = epoch
            start = perf_counter()
            log_progress(f"Epoch {epoch}/{self.config.epochs} starting")
            train_loss, train_auc = self._epoch(train_loader, True)
            train_data_wait = self.last_data_wait_seconds
            val_loss, val_auc = self._epoch(val_loader, False)
            if self.scheduler_continuation is None:
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
            if self.scheduler_continuation:
                state["scheduler_continuation"] = self.scheduler_continuation
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
