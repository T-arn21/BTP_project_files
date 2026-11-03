"""Focused finite checks and JSON diagnostics, with no model/data transformations."""
from pathlib import Path
import torch
from .utils import write_json


class NumericalStabilityError(RuntimeError):
    pass


def fp32_bce(criterion, logits, targets):
    # Explicitly disable any enclosing autocast, including during validation.
    with torch.amp.autocast(logits.device.type, enabled=False):
        return criterion(logits.float(), targets.float())


def tensor_finite(value):
    return bool(torch.isfinite(value).all().item())


def named_finite(named):
    """One host synchronization per device, rather than one per model parameter."""
    groups = {}
    for _, tensor in named:
        if tensor.is_floating_point() or tensor.is_complex():
            groups.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())
    return all(bool(torch.stack(flags).all().item()) for flags in groups.values())


def optimizer_tensors(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [(f"{names.get(id(p), 'parameter')}.{key}", value)
            for p, state in optimizer.state.items() for key, value in state.items() if torch.is_tensor(value)]


def tensor_report(value):
    if value is None:
        return {"available": False, "all_finite": None}
    value = value.detach()
    finite = torch.isfinite(value)
    finite_values = value[finite].float()
    all_finite = bool(finite.all().item())
    bounds = (float(finite_values.min().item()), float(finite_values.max().item())) if finite_values.numel() else (None, None)
    return dict(available=True, shape=list(value.shape), dtype=str(value.dtype), all_finite=all_finite,
                nonfinite_count=int((~finite).sum().item()), min=bounds[0] if all_finite else None,
                max=bounds[1] if all_finite else None, finite_min=bounds[0], finite_max=bounds[1])


def named_report(named):
    names = [name for name, value in named if not tensor_finite(value)]
    return {"all_finite": not names, "nonfinite_names": names, "checked_tensors": len(named)}


def failure_report(directory, *, reason, phase, epoch, batch_index, stage, record_ids,
                   inputs, targets, loss_targets, logits, loss, model, optimizer,
                   scaler, config, pos_weight=None, normalization=None, detail=None):
    ids = record_ids.detach().cpu().tolist() if torch.is_tensor(record_ids) else list(record_ids or [])
    params = named_report(list(model.named_parameters()))
    grads = named_report([(name, p.grad) for name, p in model.named_parameters() if p.grad is not None])
    report = dict(reason=reason, phase=phase, epoch=int(epoch), batch_index=int(batch_index), stage=stage,
                  ecg_record_ids=ids, inputs=tensor_report(inputs), targets=tensor_report(targets),
                  loss_targets=tensor_report(loss_targets), logits=tensor_report(logits), loss=tensor_report(loss),
                  parameters=params, gradients=grads,
                  optimizer_state=named_report(optimizer_tensors(model, optimizer)),
                  parameters_became_nonfinite=phase == "post_optimizer" and not params["all_finite"],
                  mixed_precision_requested=config.mixed_precision, amp_enabled=scaler.is_enabled(),
                  scaler_scale=float(scaler.get_scale()), learning_rates=[float(g["lr"]) for g in optimizer.param_groups],
                  numerical_stability_mode=config.numerical_stability_mode, loss_computation="float32_autocast_disabled",
                  pos_weight=tensor_report(pos_weight), normalization=normalization,
                  label_smoothing=config.label_smoothing, grad_clip_norm=config.grad_clip_norm, detail=detail)
    affected = {}
    for name, value in (("inputs", inputs), ("targets", targets), ("logits", logits)):
        if value is not None and value.ndim and value.shape[0] == len(ids):
            bad = ~torch.isfinite(value.detach()).reshape(len(ids), -1).all(dim=1)
            affected[name] = [ids[i] for i, flag in enumerate(bad.cpu().tolist()) if flag]
    report["nonfinite_record_ids"] = affected
    out = Path(directory)
    diagnostic = out / "diagnostics"
    diagnostic.mkdir(parents=True, exist_ok=True)
    path = diagnostic / f"epoch_{epoch:04d}_{stage}_batch_{batch_index:06d}_{phase}.json"
    write_json(path, report)
    write_json(out / "numerical_failure.json", report)
    raise NumericalStabilityError(f"{reason} at epoch {epoch}, {stage} batch {batch_index}; ECG IDs={ids}; diagnostics: {path}")
