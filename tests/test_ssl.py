from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from conftest import synthetic_reader
from src.config import ModelConfig
from src.ssl import SSLConfig, MaskedECGReconstructor, temporal_mask, masked_mse, run_ssl
from src.ssl_data import prepare_ssl_data


class TinyEncoder(nn.Module):
    """CPU plumbing double; no claim of Mamba/CUDA validation."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(12, 8, 3, padding=1)

    def forward_tokens(self, x):
        return self.conv(x).transpose(1, 2)


def factory(config):
    return MaskedECGReconstructor(TinyEncoder(), 8)


@pytest.fixture(autouse=True)
def cpu_test_threads():
    torch.set_num_threads(1)


def test_contiguous_masks_reproducible_per_record_and_epoch():
    x = torch.ones(4, 12, 100)
    masked, mask = temporal_mask(x, [1, 2, 3, 4], fraction=.25, seed=42, epoch=1)
    again, same = temporal_mask(x, [1, 2, 3, 4], fraction=.25, seed=42, epoch=1)
    assert torch.equal(masked, again) and torch.equal(mask, same)
    assert torch.equal(x, torch.ones_like(x))
    assert (mask.sum(dim=-1) == 25).all()
    for row in mask[:, 0]:
        locations = row.nonzero().flatten()
        assert (locations.diff() == 1).all()
    assert (masked.masked_select(mask.expand_as(masked)) == 0).all()
    assert (masked.masked_select(~mask.expand_as(masked)) == 1).all()
    perm = [2, 0, 3, 1]
    _, reordered = temporal_mask(x[perm], [3, 1, 4, 2], fraction=.25, seed=42, epoch=1)
    assert torch.equal(reordered, mask[perm])
    _, next_epoch = temporal_mask(x, [1, 2, 3, 4], fraction=.25, seed=42, epoch=2)
    assert not torch.equal(mask, next_epoch)


def test_masked_only_fp32_loss_and_gradients():
    target = torch.zeros(1, 12, 20)
    prediction = torch.ones_like(target, dtype=torch.float16, requires_grad=True)
    mask = torch.zeros(1, 1, 20, dtype=torch.bool)
    mask[..., 5:10] = True
    with torch.no_grad():
        prediction[..., :5] = 1000
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = masked_mse(prediction, target, mask)
    assert loss.dtype == torch.float32 and loss.item() == 1
    loss.backward()
    assert (prediction.grad.masked_select(~mask.expand_as(target)) == 0).all()
    assert torch.isfinite(prediction.grad).all() and (prediction.grad.masked_select(mask.expand_as(target)) > 0).all()
    with pytest.raises(ValueError, match="nonempty"):
        masked_mse(prediction, target, torch.zeros_like(mask))


def test_ssl_no_labels_or_heldout_waveform_reads_and_cache_reuse(data_config):
    root = Path(data_config.data_dir)
    table = pd.read_csv(root / "ptbxl_database.csv")
    table.drop(columns="scp_codes").to_csv(root / "ptbxl_database.csv", index=False)
    calls = []
    def training_only_reader(path, fs):
        ecg_id = int(path.name.rsplit("_", 1)[-1])
        assert ecg_id <= 24, "SSL attempted held-out waveform access"
        calls.append(ecg_id)
        return synthetic_reader(path, fs)
    cfg = replace(data_config, normalization_samples=5)
    data = prepare_ssl_data(cfg, reader=training_only_reader)
    assert data.dataset.ids == list(range(1, 25))  # no labels available; no records excluded
    assert not data.provenance["diagnostic_labels_used"]
    assert set(data.provenance["normalization_record_ids"]) <= set(range(1, 25))
    assert len(calls) == 24
    for _ in range(2):
        for x, ids in data.loader(7, 42):
            assert x.shape[1:] == (12, 1000) and len(ids) == len(x)
    prepare_ssl_data(cfg, reader=training_only_reader)
    assert len(calls) == 24  # zero repeat WFDB/filter reads across epochs/preparation


def test_ssl_patient_leakage_rejected_before_waveforms(data_config):
    root = Path(data_config.data_dir)
    metadata = pd.read_csv(root / "ptbxl_database.csv")
    metadata.loc[24, "patient_id"] = metadata.loc[0, "patient_id"]
    metadata.to_csv(root / "ptbxl_database.csv", index=False)
    with pytest.raises(ValueError, match="Patient leakage"):
        prepare_ssl_data(data_config, reader=lambda *a: pytest.fail("waveforms read before fold audit"))


@pytest.mark.parametrize("smoke", [True, False])
def test_ssl_smoke_and_explicit_full_cpu_plumbing_exports(data_config, tmp_path, smoke):
    cfg = SSLConfig(data=data_config, device="cpu", epochs=2, batch_size=8, smoke_fit_records=5,
                    checkpoint_every_batches=2, model=ModelConfig(embed_dim=8, hidden_dim=8, n_layers=1))
    result = run_ssl(cfg, tmp_path / "ssl_outputs", smoke=smoke, reader=synthetic_reader, model_factory=factory)
    out = Path(result["output_dir"])
    assert result["smoke_only"] is smoke and result["full_pretraining_completed"] is not smoke
    assert result["train_records"] == (8 if smoke else 24)
    assert result["optimizer_steps"] == (1 if smoke else 6)
    assert result["finite_forward_backward"] and not result["diagnostic_labels_used"]
    for name in ["encoder.pth", "last.pth", "history.csv", "config.json", "split_provenance.json",
                 "loss_curve.png", "reconstruction_epoch_001.png", "reconstruction_example.npz", "summary.json"]:
        assert (out / name).is_file()
    state = torch.load(out / "encoder.pth", weights_only=True)
    TinyEncoder().load_state_dict(state["encoder_state_dict"], strict=True)
    assert state["trained_parameter_names"] and state["smoke_only"] is smoke
    checkpoint = torch.load(out / "last.pth", weights_only=True)
    assert checkpoint["completed_epoch"] and checkpoint["global_step"] == result["optimizer_steps"]
    frame = pd.read_csv(out / "history.csv")
    assert frame.records.tolist() == ([8] if smoke else [24, 24])


def test_ssl_nonfinite_forward_aborts_with_record_diagnostics(data_config, tmp_path):
    def invalid_factory(config):
        model = factory(config)
        model.reconstruction_head.register_forward_hook(lambda module, args, output: output * float("nan"))
        return model
    cfg = SSLConfig(data=data_config, device="cpu", smoke_fit_records=2)
    with pytest.raises(RuntimeError, match="Nonfinite SSL reconstruction"):
        run_ssl(cfg, tmp_path / "ssl", smoke=True, reader=synthetic_reader, model_factory=invalid_factory)
    report = json.loads(next((tmp_path / "ssl").rglob("numerical_failure.json")).read_text())
    assert report["epoch"] == report["batch_index"] == 1 and report["ecg_record_ids"]
    assert report["targets"]["all_finite"] and not report["reconstruction"]["all_finite"]
    assert not list((tmp_path / "ssl").rglob("encoder.pth"))


def test_encoder_tokens_preserve_supervised_forward_and_state(monkeypatch):
    class BlockDouble(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)
    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace(Mamba=BlockDouble))
    path = Path(__file__).parents[1] / "src/models/mamba.py"
    spec = importlib.util.spec_from_file_location("ssl_encoder_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MambaClassifier(12, 5, 16, 16, 1, 16, 4, 2, 0).eval()
    x = torch.randn(2, 12, 100)
    keys = list(model.state_dict())
    legacy = model.input_proj(x.transpose(1, 2))
    for block in model.blocks:
        legacy = block(legacy)
    legacy = model.final_norm(legacy)
    pooled = model.dropout(torch.cat([legacy.mean(dim=1), model.attn_pool(legacy)], dim=-1))
    torch.testing.assert_close(model.forward_tokens(x), legacy, atol=0, rtol=0)
    torch.testing.assert_close(model(x), model.head(pooled), atol=0, rtol=0)
    assert keys == list(model.state_dict())
    ssl_model = MaskedECGReconstructor(model, 16)
    assert ssl_model(x).shape == x.shape


@pytest.mark.parametrize("mode", ["smoke_only", "full_after_smoke", "no_smoke", "smoke_failure"])
def test_ssl_unattended_controls_without_supervised_labels(monkeypatch, synthetic_root, tmp_path, mode):
    import nbformat
    import src.ssl as ssl_module
    import src.kaggle as kaggle
    from src.unattended import run_worker
    root = Path(__file__).parents[1]
    (synthetic_root / "records100").mkdir()
    namespace = {}
    exec(nbformat.read(root / "05_btp_kaggle_final.ipynb", as_version=4).cells[1].source, namespace)
    settings = namespace["SETTINGS"]
    session = tmp_path / "worker_session"
    session.mkdir()
    settings.update(CHECKOUT_DIR=str(root), DATASET_ROOT=str(synthetic_root), SESSION_DIR=str(session),
                    CACHE_DIR=str(tmp_path / "worker_cache"), RUN_SUPERCLASS=False, RUN_SUBCLASS=False,
                    RUN_SMOKE=False, RUN_METADATA_AUDIT=False, RUN_SSL_SMOKE=mode != "no_smoke",
                    RUN_SSL_PRETRAIN=mode != "smoke_only")
    settings["SSL_SETTINGS"].update(epochs=1, batch_size=8, smoke_fit_records=3)
    monkeypatch.setattr(kaggle, "require_cuda", lambda: None)
    monkeypatch.setattr(kaggle, "audit_metadata", lambda *a, **kw: pytest.fail("SSL-only path parsed diagnostic labels"))
    calls = []
    original_run = ssl_module.run_ssl
    def cpu_run(config, output_root, *, smoke):
        calls.append(smoke)
        if mode == "smoke_failure":
            raise RuntimeError("injected smoke failure")
        return original_run(replace(config, device="cpu"), output_root, smoke=smoke,
                            reader=synthetic_reader, model_factory=factory)
    monkeypatch.setattr(ssl_module, "run_ssl", cpu_run)
    outcome = run_worker(settings)
    final = json.loads((session / "final_summary.json").read_text())
    if mode == "smoke_only":
        assert outcome == 0 and calls == [True]
        assert set(final["completed"]) == {"ssl_smoke"}
        assert final["experiments"]["ssl_pretraining"]["status"] == "disabled_by_configuration"
    elif mode == "full_after_smoke":
        assert outcome == 0 and calls == [True, False]
        assert final["completed"]["ssl_pretraining"]["train_records"] == 24
    elif mode == "no_smoke":
        assert outcome == 1 and calls == []
        assert final["experiments"]["ssl_pretraining"]["status"] == "blocked_missing_smoke"
    else:
        assert outcome == 1 and calls == [True]
        assert final["experiments"]["ssl_pretraining"]["status"] == "blocked_by_smoke_failure"


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Requires actual CUDA Mamba; CPU adapter tests are not GPU evidence")
def test_actual_ssl_mamba_gpu_smoke():
    from src.ssl import create_reconstruction_model
    model = create_reconstruction_model(ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, dropout=0)).cuda()
    x = torch.randn(2, 12, 64, device="cuda")
    masked, mask = temporal_mask(x, [1, 2], fraction=.25, seed=42, epoch=1)
    loss = masked_mse(model(masked), x, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
