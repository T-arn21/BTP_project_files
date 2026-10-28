"""Deliberate numerical failures must be identified, persisted and never skipped."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import ExperimentConfig, checkpoint_config_matches
from src.training import Trainer, make_pos_weight, load_checkpoint
from src.numerics import NumericalStabilityError, fp32_bce


def model(width=5):
    return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, width))


def loader(x=None, y=None):
    x = torch.randn(4, 12, 64) if x is None else x
    y = torch.tensor([[0., 1., 0., 1., 0.]] * 4) if y is None else y
    return DataLoader(TensorDataset(x, y, torch.tensor([101, 102, 103, 104])), batch_size=2)


def trainer(tmp_path, net=None):
    torch.set_num_threads(1)
    return Trainer(net if net is not None else model(),
                   ExperimentConfig(epochs=1, device="cpu", mixed_precision=False),
                   ["a", "b", "c", "d", "e"], tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("location", ["inputs", "targets", "logits"])
def test_nonfinite_batch_diagnostics_and_no_update(tmp_path, bad, location):
    net = model()
    x, y = torch.randn(4, 12, 64), torch.tensor([[0., 1., 0., 1., 0.]] * 4)
    if location == "inputs":
        x[1, 0, 0] = bad
    elif location == "targets":
        y[1, 0] = bad
    else:
        def corrupt(module, args, output):
            result = output.clone()
            result[1, 0] = bad
            return result
        net.register_forward_hook(corrupt)
    t = trainer(tmp_path, net)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with pytest.raises(NumericalStabilityError, match="ECG IDs"):
        t.fit(loader(x, y), loader())
    report = json.loads((tmp_path / "numerical_failure.json").read_text())
    assert report["phase"] == location and report["epoch"] == 1 and report["batch_index"] == 1
    assert report["ecg_record_ids"] == [101, 102]
    assert report["nonfinite_record_ids"][location] == [102]
    assert report[location]["all_finite"] is False
    assert report[location]["min"] is None and report[location]["finite_min"] is not None
    assert report["parameters"]["all_finite"] is True
    assert report["amp_enabled"] is False and report["learning_rates"][0] > 0
    for k, value in net.state_dict().items():
        torch.testing.assert_close(value, before[k], atol=0, rtol=0)
    assert not (tmp_path / "last.pth").exists()
    assert len(list((tmp_path / "diagnostics").glob("*.json"))) == 1


def test_validation_failure_records_epoch_and_stage(tmp_path):
    t = trainer(tmp_path)
    x = torch.randn(4, 12, 64)
    x[2, 0, 0] = float("inf")
    with pytest.raises(NumericalStabilityError):
        t.fit(loader(), loader(x))
    report = json.loads((tmp_path / "numerical_failure.json").read_text())
    assert report["stage"] == "validation" and report["batch_index"] == 2
    assert report["ecg_record_ids"] == [103, 104] and report["epoch"] == 1


@pytest.mark.parametrize("kind", ["nonfinite_elements", "overflowed_norm"])
def test_bad_gradients_cannot_reach_optimizer(tmp_path, monkeypatch, kind):
    t = trainer(tmp_path)
    value = float("nan") if kind == "nonfinite_elements" else 1e30
    next(t.model.parameters()).register_hook(lambda grad: torch.full_like(grad, value))
    monkeypatch.setattr(t.optimizer, "step", lambda *a, **kw: pytest.fail("invalid gradients reached optimizer"))
    with pytest.raises(NumericalStabilityError):
        t.fit(loader(), loader())
    report = json.loads((tmp_path / "numerical_failure.json").read_text())
    assert report["phase"] == ("gradients" if kind == "nonfinite_elements" else "gradient_clipping")
    assert report["parameters"]["all_finite"]


@pytest.mark.parametrize("kind", ["parameters", "optimizer_state"])
def test_corrupt_optimizer_update_is_not_silent_or_checkpointed(tmp_path, monkeypatch, kind):
    t = trainer(tmp_path)
    original = t.optimizer.step
    def corrupt(*a, **kw):
        result = original(*a, **kw)
        if kind == "parameters":
            with torch.no_grad():
                next(t.model.parameters()).fill_(float("inf"))
        else:
            next(iter(t.optimizer.state.values()))["exp_avg"].fill_(float("inf"))
        return result
    monkeypatch.setattr(t.optimizer, "step", corrupt)
    with pytest.raises(NumericalStabilityError):
        t.fit(loader(), loader())
    report = json.loads((tmp_path / "numerical_failure.json").read_text())
    assert report["phase"] == "post_optimizer" and not report[kind]["all_finite"]
    assert report["parameters_became_nonfinite"] is (kind == "parameters")
    assert not (tmp_path / "best.pth").exists()


def test_fp32_loss_explicitly_disables_autocast():
    class Spy(nn.BCEWithLogitsLoss):
        def forward(self, logits, targets):
            assert logits.dtype == targets.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            return super().forward(logits, targets)
    logits = torch.tensor([[1000., -1000.]], dtype=torch.float16, requires_grad=True)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = fp32_bce(Spy(), logits, torch.tensor([[0., 1.]]))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_nonfinite_loss_has_context_even_when_logits_finite(tmp_path):
    class BadLoss(nn.BCEWithLogitsLoss):
        def forward(self, logits, targets):
            return super().forward(logits, targets) * float("nan")
    t = trainer(tmp_path)
    t.criterion = BadLoss()
    with pytest.raises(NumericalStabilityError, match="Nonfinite BCE loss"):
        t.fit(loader(), loader())
    report = json.loads((tmp_path / "numerical_failure.json").read_text())
    assert report["phase"] == "loss" and report["logits"]["all_finite"]
    assert not report["loss"]["all_finite"]


def test_conservative_mode_weights_and_legacy_checkpoint_guards(tmp_path):
    safe = ExperimentConfig(mixed_precision=False, learning_rate=1e-4, numerical_stability_mode="conservative")
    assert not safe.mixed_precision
    with pytest.raises(ValueError, match="Conservative"):
        ExperimentConfig(numerical_stability_mode="conservative")
    with pytest.raises(ValueError, match="binary finite"):
        make_pos_weight(np.array([[0., float("nan")]]))
    regular = ExperimentConfig()
    old = asdict(regular)
    old.pop("numerical_stability_mode")
    old.pop("initial_checkpoint")
    assert checkpoint_config_matches(old, regular)
    assert not checkpoint_config_matches(old, safe)
    net = model()
    state = net.state_dict()
    state["2.weight"][0, 0] = float("nan")
    path = tmp_path / "invalid.pth"
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(ValueError, match="nonfinite model"):
        load_checkpoint(path, model())


def test_explicit_weights_only_start_preserves_single_run(monkeypatch, data_config, tmp_path):
    import src.experiments as experiments
    original = experiments.load_ptbxl
    monkeypatch.setattr(experiments, "load_ptbxl", lambda *a, **kw: original(*a, **kw, reader=__import__('conftest').synthetic_reader))
    monkeypatch.setattr(experiments, "create_model", lambda *a, **kw: model())
    path = tmp_path / "prior_best.pth"
    source = experiments.load_ptbxl(config=data_config)
    source_config = ExperimentConfig(data=source.config)
    source_metadata = dict(task="superclass", normalizer=source.normalizer.state_dict(), provenance=source.provenance,
                           splits={k: list(map(int, ids)) for k, ids in source.splits.items()})
    torch.save({"model_state_dict": model().state_dict(), "class_names": source.class_names, "epoch": 3,
                "config": asdict(source_config), "metadata": source_metadata}, path)
    cfg = ExperimentConfig(data=data_config, epochs=1, batch_size=2, device="cpu", mixed_precision=False,
                           learning_rate=1e-4, numerical_stability_mode="conservative", initial_checkpoint=str(path),
                           diagnostic=True, diagnostic_batches=1, output_root=str(tmp_path / "new_runs"))
    result = experiments.run_experiment(config=cfg)
    out = Path(result["output_dir"])
    assert (out / "initial_checkpoint.json").is_file() and (out / "numerical_settings.json").is_file()
    assert result["summary"]["benchmark_result"] is False


@pytest.mark.parametrize("corrupt_optimizer", [False, True])
def test_legacy_exact_resume_checks_loaded_optimizer(tmp_path, corrupt_optimizer):
    first = trainer(tmp_path)
    first.fit(loader(), loader())
    path = tmp_path / "last.pth"
    state = torch.load(path, weights_only=True)
    state["config"].pop("numerical_stability_mode")
    state["config"].pop("initial_checkpoint")
    if corrupt_optimizer:
        next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"].fill_(float("nan"))
    torch.save(state, path)
    resumed = trainer(tmp_path)
    if corrupt_optimizer:
        with pytest.raises(NumericalStabilityError, match="optimizer state"):
            resumed.fit(loader(), loader(), resume_from=path)
        report = json.loads((tmp_path / "numerical_failure.json").read_text())
        assert report["phase"] == "checkpoint_restore"
        assert not report["optimizer_state"]["all_finite"]
    else:
        history = resumed.fit(loader(), loader(), resume_from=path)
        assert len(history) == 1
        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[name], atol=0, rtol=0)
