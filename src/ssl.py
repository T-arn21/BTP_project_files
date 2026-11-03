"""Masked ECG reconstruction using the preserved Mamba temporal backbone."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import hashlib
import math
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from .config import DataConfig, ModelConfig
from .ssl_data import prepare_ssl_data
from .data import read_waveform
from .training import atomic_checkpoint
from .experiments import Tee
from .numerics import tensor_finite, named_finite, named_report, tensor_report, optimizer_tensors
from .utils import seed_everything, write_json, environment_info, log_progress


@dataclass
class SSLConfig:
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    mask_fraction: float = 0.25
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = False
    smoke_batches: int = 1
    smoke_fit_records: int = 16
    log_every_batches: int = 20
    checkpoint_every_batches: int = 500
    data: DataConfig = field(default_factory=lambda: DataConfig(augment=False))
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if any(x < 1 for x in (self.epochs, self.batch_size, self.smoke_batches, self.smoke_fit_records,
                                self.log_every_batches, self.checkpoint_every_batches)):
            raise ValueError("SSL budgets, batch size and logging/checkpoint intervals must be positive")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("SSL mask_fraction must be between zero and one")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1e-4:
            raise ValueError("SSL conservative learning_rate must be finite, positive and <= 1e-4")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("SSL grad_clip_norm must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("SSL weight_decay must be finite and nonnegative")


class MaskedECGReconstructor(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.reconstruction_head = nn.Linear(embed_dim, 12)

    def forward(self, x):
        return self.reconstruction_head(self.encoder.forward_tokens(x)).transpose(1, 2)


def create_reconstruction_model(config):
    from .models.mamba import MambaEncoder
    encoder = MambaEncoder(n_leads=12, n_classes=0, **asdict(config))
    return MaskedECGReconstructor(encoder, config.embed_dim)


def temporal_mask(x, record_ids, *, fraction, seed, epoch):
    """One random contiguous window per ECG, shared across all 12 leads.

    RNG is keyed by seed/epoch/record ID, independent of shuffle and workers.
    Zero denotes the training-normalized mean, not an observed target value.
    """
    if x.ndim != 3 or x.shape[1] != 12 or x.shape[2] < 2 or len(record_ids) != len(x):
        raise ValueError("Masking expects (batch,12,time>=2) and one ECG ID per sample")
    if not 0 < fraction < 1:
        raise ValueError("Mask fraction must be between zero and one")
    length = min(x.shape[-1] - 1, max(1, round(x.shape[-1] * fraction)))
    starts = []
    for ecg_id in record_ids:
        key = hashlib.sha256(f"{seed}:{epoch}:{int(ecg_id)}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "little") % (2**63 - 1))
        starts.append(torch.randint(x.shape[-1] - length + 1, (), generator=generator).item())
    starts = torch.tensor(starts, device=x.device).view(-1, 1, 1)
    times = torch.arange(x.shape[-1], device=x.device).view(1, 1, -1)
    mask = (times >= starts) & (times < starts + length)
    return x.masked_fill(mask, 0), mask


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or prediction.ndim != 3 or target.shape[1] != 12:
        raise ValueError("Reconstruction and target must have the same (batch,12,time) shape")
    if mask.dtype != torch.bool or mask.shape != (len(target), 1, target.shape[-1]) or not mask.any():
        raise ValueError("Expected a nonempty boolean (batch,1,time) mask")
    with torch.amp.autocast(prediction.device.type, enabled=False):
        selected = mask.expand_as(target)
        residual = prediction.float().masked_select(selected) - target.float().masked_select(selected)
        return residual.square().mean()


def plot_reconstruction(path, target, prediction, mask, record_id, sampling_rate, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(target.shape[-1]) / sampling_rate
    fig, axes = plt.subplots(12, 1, figsize=(12, 18), sharex=True)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    masked = np.flatnonzero(mask)
    for i, ax in enumerate(axes):
        ax.plot(t, target[i], color="0.5", label="Original normalized ECG", linewidth=.8)
        ax.plot(t, prediction[i], color="tab:blue", label="Reconstruction", linewidth=.7)
        ax.axvspan(t[masked[0]], (masked[-1] + 1) / sampling_rate, alpha=.2, color="tab:orange")
        ax.set_ylabel(leads[i])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s); shaded window masked on all 12 leads")
    fig.suptitle(f"{title}; training ECG {record_id}; normalized amplitudes")
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_ssl(config, output_root, *, smoke=True, reader=read_waveform, model_factory=create_reconstruction_model):
    """A bounded smoke OR an explicitly requested full training-fold budget.

    No labels, held-out waveforms, supervised checkpoint loading or auto resume.
    """
    cfg = replace(config, data=replace(config.data, augment=False, layout="channels_first"))
    if smoke:
        cfg = replace(cfg, epochs=1, data=replace(cfg.data, normalization_mode="sampled",
                                               normalization_samples=cfg.smoke_fit_records))
    purpose = "smoke" if smoke else "pretraining"
    out = Path(output_root) / purpose / f"ssl_mamba_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            stage = "training-fold-only data preparation"
            try:
                write_json(out / "status.json", dict(status="running", stage=stage, smoke_only=smoke))
                seed_everything(cfg.seed)
                device = torch.device(cfg.device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("SSL requires a CUDA GPU; local CPU adapters are test-only")
                data = prepare_ssl_data(cfg.data, seed=cfg.seed,
                    record_limit=cfg.batch_size * cfg.smoke_batches if smoke else None, reader=reader)
                cfg = replace(cfg, data=data.config)
                write_json(out / "config.json", asdict(cfg))
                write_json(out / "split_provenance.json", data.provenance)
                write_json(out / "environment.json", environment_info())
                model = model_factory(cfg.model).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
                loader = data.loader(cfg.batch_size, cfg.seed)
                history, global_step = [], 0
                epoch, batch_index, context = 0, 0, {}
                trained_names = set()

                def fail(reason, phase):
                    report = dict(reason=reason, phase=phase, epoch=epoch, batch_index=batch_index,
                                  ecg_record_ids=context.get("ids", []), smoke_only=smoke,
                                  amp_enabled=scaler.is_enabled(), scaler_scale=scaler.get_scale(),
                                  learning_rates=[g["lr"] for g in optimizer.param_groups],
                                  parameters=named_report(list(model.named_parameters())),
                                  gradients=named_report([(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]),
                                  optimizer_state=named_report(optimizer_tensors(model, optimizer)),
                                  **{k: tensor_report(context.get(k)) for k in ("inputs", "targets", "reconstruction", "mask", "loss")})
                    (out / "diagnostics").mkdir(exist_ok=True)
                    write_json(out / "diagnostics" / f"epoch_{epoch:04d}_batch_{batch_index:06d}_{phase}.json", report)
                    write_json(out / "numerical_failure.json", report)
                    raise RuntimeError(f"{reason}; epoch={epoch}, batch={batch_index}, ECG IDs={context.get('ids')}; {out / 'numerical_failure.json'}")

                def check_state(phase):
                    if not named_finite(list(model.named_parameters())) or not named_finite(optimizer_tensors(model, optimizer)):
                        fail("Nonfinite SSL model/optimizer state", phase)
                    if not math.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0:
                        fail("Invalid SSL scaler", phase)

                def checkpoint(completed_epoch):
                    atomic_checkpoint(dict(format="btp_masked_reconstruction_v1", epoch=epoch,
                        completed_epoch=completed_epoch, global_step=global_step, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                        config=asdict(cfg), provenance=data.provenance, history=history, smoke_only=smoke,
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        loader_rng=loader.generator.get_state()), out / "last.pth")

                check_state("initialization")
                stage = "masked reconstruction smoke" if smoke else "full training-fold pretraining"
                write_json(out / "status.json", dict(status="running", stage=stage, smoke_only=smoke))
                for epoch in range(1, cfg.epochs + 1):
                    model.train()
                    start, last_log = perf_counter(), perf_counter()
                    total, values, records = 0., 0, 0
                    for batch_index, (target, ids) in enumerate(loader, 1):
                        target = target.to(device).float()
                        optimizer.zero_grad(set_to_none=True)
                        context = dict(ids=ids.tolist(), inputs=target, targets=target)
                        if not tensor_finite(target):
                            fail("Nonfinite SSL waveform/normalization", "inputs")
                        masked, mask = temporal_mask(target, ids, fraction=cfg.mask_fraction, seed=cfg.seed, epoch=epoch)
                        context.update(inputs=masked, mask=mask)
                        with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                            prediction = model(masked)
                        context["reconstruction"] = prediction
                        if not tensor_finite(prediction):
                            fail("Nonfinite SSL reconstruction", "forward")
                        loss = masked_mse(prediction, target, mask)
                        context["loss"] = loss
                        if not tensor_finite(loss):
                            fail("Nonfinite masked MSE", "loss")
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        gradients = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
                        if not gradients or not named_finite(gradients):
                            fail("Missing/nonfinite SSL gradients", "gradients")
                        # Smoke must prove the backbone and reconstruction head both learn.
                        if smoke and any(not any(n.startswith(prefix) and bool(g.abs().max().item() > 0) for n, g in gradients)
                                         for prefix in ("encoder.", "reconstruction_head.")):
                            fail("Zero/missing encoder or reconstruction-head smoke gradients", "gradients")
                        try:
                            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True)
                        except RuntimeError:
                            fail("Nonfinite SSL gradient norm", "gradient_clipping")
                        trained_names.update(n for n, _ in gradients)
                        scaler.step(optimizer)
                        scaler.update()
                        check_state("post_optimizer")
                        global_step += 1
                        selected = int(mask.sum().item()) * 12
                        total += loss.item() * selected
                        values += selected
                        records += len(target)
                        if batch_index == 1:
                            example = (target[0].detach().cpu().numpy(), prediction[0].detach().float().cpu().numpy(),
                                       mask[0, 0].cpu().numpy(), int(ids[0]))
                        if batch_index == 1 or batch_index % cfg.log_every_batches == 0 or perf_counter() - last_log >= 15 or batch_index == len(loader):
                            log_progress(f"SSL {purpose} epoch {epoch}/{cfg.epochs}, batch {batch_index}/{len(loader)}, masked_MSE={loss.item():.6f}, records={records}, elapsed={perf_counter()-start:.1f}s")
                            last_log = perf_counter()
                        if global_step % cfg.checkpoint_every_batches == 0:
                            log_progress(f"Saving SSL intermediate checkpoint at step {global_step}")
                            checkpoint(False)
                        context = {}
                    if not records:
                        raise ValueError("SSL training loader is empty")
                    history.append(dict(epoch=epoch, masked_mse=total / values, masked_values=values,
                                        records=records, seconds=perf_counter()-start, learning_rate=cfg.learning_rate))
                    checkpoint(True)
                    atomic_checkpoint(dict(format="btp_masked_ecg_encoder_v1", encoder_state_dict=model.encoder.state_dict(),
                        model_config=asdict(cfg.model), data_config=asdict(cfg.data), provenance=data.provenance,
                        epoch=epoch, smoke_only=smoke, trained_parameter_names=sorted(n.removeprefix("encoder.") for n in trained_names if n.startswith("encoder.")),
                        pooling_pretrained=False), out / "encoder.pth")
                    temporary = out / "history.partial.csv"
                    pd.DataFrame(history).to_csv(temporary, index=False)
                    temporary.replace(out / "history.csv")
                    plot_reconstruction(out / f"reconstruction_epoch_{epoch:03d}.png", *example,
                                        cfg.data.sampling_rate, f"SSL {purpose} epoch {epoch}")
                    np.savez_compressed(out / "reconstruction_example.npz", target=example[0], prediction=example[1],
                                        mask=example[2], ecg_id=example[3])
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot([h["epoch"] for h in history], [h["masked_mse"] for h in history], marker="o")
                ax.set(xlabel="Epoch", ylabel="Training masked-region MSE", title=f"SSL {purpose}; training folds only")
                fig.savefig(out / "loss_curve.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                result = dict(status="complete", smoke_only=smoke, full_pretraining_completed=not smoke,
                              benchmark_result=False, objective="masked_waveform_reconstruction", device=str(device),
                              epochs_completed=len(history), train_records=len(data.dataset), optimizer_steps=global_step,
                              finite_forward_backward=True, amp_enabled=scaler.is_enabled(),
                              masked_mse=history[-1]["masked_mse"], diagnostic_labels_used=False,
                              pooling_pretrained=False, output_dir=str(out.resolve()))
                write_json(out / "summary.json", result)
                write_json(out / "status.json", dict(status="complete", stage=stage, smoke_only=smoke))
                return result
            except Exception as exc:
                write_json(out / "status.json", dict(status="failed", stage=stage, smoke_only=smoke, error=str(exc)))
                raise
