"""Synthetic source evidence tests; no assertion that 20 real SSL epochs ran here."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import torch
from torch import nn
import pandas as pd
import pytest
from conftest import synthetic_reader
from src.config import ModelConfig, ExperimentConfig
from src.ssl import SSLConfig
from src.ssl_data import prepare_ssl_data
from src.data import load_ptbxl
from src.cache import digest_file
from src.finetuning import BACKBONE_PREFIXES, load_completed_ssl_export, initialize_ssl_backbone
from src.utils import write_json


class ClassifierDouble(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_proj = nn.Linear(12, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8)])
        self.final_norm = nn.LayerNorm(8)
        self.attn_pool = nn.Linear(8, 1)
        self.head = nn.Linear(16, width)

    def forward(self, x):
        tokens = self.final_norm(self.blocks[0](self.input_proj(x.transpose(1, 2))).tanh())
        attention = self.attn_pool(tokens).softmax(dim=1)
        return self.head(torch.cat([tokens.mean(dim=1), (tokens * attention).sum(dim=1)], dim=1))


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def make_export(data_config, folder, model, model_config):
    """Fabricate coherent epoch-20 evidence for tests, never real training results."""
    data = prepare_ssl_data(data_config, reader=synthetic_reader)
    folder.mkdir()
    path = folder / "encoder.pth"
    encoder = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("head.")}
    for key in encoder:
        if key.startswith("attn_pool."):
            encoder[key].fill_(99)  # must never overwrite the fresh fine-tuning pool
    state = dict(format="btp_masked_ecg_encoder_v1", encoder_state_dict=encoder,
                 model_config=asdict(model_config), data_config=asdict(data.config), provenance=data.provenance,
                 epoch=20, smoke_only=False, pooling_pretrained=False,
                 trained_parameter_names=[k for k in encoder if k.startswith(BACKBONE_PREFIXES)])
    torch.save(state, path)
    write_json(folder / "config.json", asdict(SSLConfig(epochs=20, data=data.config, model=model_config)))
    write_json(folder / "summary.json", dict(status="complete", smoke_only=False, full_pretraining_completed=True,
               epochs_completed=20, diagnostic_labels_used=False, train_records=len(data.dataset)))
    return path, digest_file(path)


@pytest.fixture
def source(data_config, tmp_path):
    model_config = ModelConfig(embed_dim=8, hidden_dim=8, n_layers=1, dropout=0)
    path, sha = make_export(data_config, tmp_path / "pretraining", ClassifierDouble(5), model_config)
    cfg = ExperimentConfig(data=data_config, model=model_config, epochs=1, batch_size=2,
        device="cpu", mixed_precision=False, numerical_stability_mode="conservative", learning_rate=1e-4,
        diagnostic=True, diagnostic_batches=1, ssl_encoder_checkpoint=str(path), ssl_encoder_sha256=sha,
        output_root=str(tmp_path / "fine_outputs"))
    return path, cfg


def test_source_hash_before_deserialization(source, monkeypatch):
    path, cfg = source
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("deserialized before checksum validation"))
    with pytest.raises(ValueError, match="SHA256"):
        load_completed_ssl_export(path, "0" * 64)


@pytest.mark.parametrize("task,width", [("superclass", 5), ("subclass", 23)])
def test_only_trained_backbone_transferred_pool_and_head_fresh(source, task, width):
    path, cfg = source
    cfg.task = task
    data = load_ptbxl(task, config=cfg.data, reader=synthetic_reader)
    model = ClassifierDouble(width)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    proof = initialize_ssl_backbone(model, cfg, data)
    original = torch.load(path, weights_only=True)["encoder_state_dict"]
    for name, value in model.state_dict().items():
        expected = original[name] if name.startswith(BACKBONE_PREFIXES) else before[name]
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    assert proof["provenance_verified"] and len(proof["class_names"]) == width
    assert all(p.requires_grad for p in model.parameters())
    assert digest_file(path) == cfg.ssl_encoder_sha256


@pytest.mark.parametrize("damage", ["smoke", "epoch", "normalizer", "raw_signature", "heldout",
    "patient", "nonfinite", "shape", "pool_shape", "extra_head", "untrained", "incomplete"])
def test_invalid_source_or_provenance_refused(source, damage):
    path, cfg = source
    state = torch.load(path, weights_only=True)
    if damage == "smoke": state["smoke_only"] = True
    elif damage == "epoch": state["epoch"] = 19
    elif damage == "normalizer": state["provenance"]["normalizer"]["std"][0][0] *= 2
    elif damage == "raw_signature": state["provenance"]["preprocessing_identity"]["raw_file_stats_sha256"] = "changed"
    elif damage == "heldout": state["provenance"]["used_train_ids"][-1] = 25
    elif damage == "patient": state["provenance"]["train_patient_ids"][0] = -1
    elif damage == "nonfinite": state["encoder_state_dict"]["input_proj.weight"][0, 0] = float("nan")
    elif damage == "shape": state["encoder_state_dict"]["input_proj.weight"] = torch.zeros(1)
    elif damage == "pool_shape": state["encoder_state_dict"]["attn_pool.weight"] = torch.zeros(1)
    elif damage == "extra_head": state["encoder_state_dict"]["head.weight"] = torch.zeros(1)
    elif damage == "untrained": state["trained_parameter_names"].pop()
    else:
        summary = json.loads(path.with_name("summary.json").read_text())
        summary["full_pretraining_completed"] = False
        write_json(path.with_name("summary.json"), summary)
    torch.save(state, path)
    cfg.ssl_encoder_sha256 = digest_file(path)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    with pytest.raises(ValueError):
        initialize_ssl_backbone(ClassifierDouble(5), cfg, data)


def test_optional_source_history_cannot_contradict_completion(source):
    path, cfg = source
    pd.DataFrame(dict(epoch=[1], masked_mse=[.5], records=[24])).to_csv(path.with_name("history.csv"), index=False)
    with pytest.raises(ValueError, match="history"):
        load_completed_ssl_export(path, cfg.ssl_encoder_sha256)


@pytest.mark.parametrize("task", ["superclass", "subclass"])
def test_smoke_uses_existing_trainer_metrics_and_saves_transfer_proof(source, monkeypatch, task):
    import src.experiments as experiments
    path, cfg = source
    cfg.task = task
    original = experiments.load_ptbxl
    monkeypatch.setattr(experiments, "load_ptbxl", lambda *a, **k: original(*a, **k, reader=synthetic_reader))
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    result = experiments.run_experiment(config=cfg)
    out = Path(result["output_dir"])
    assert result["summary"]["initialization"] == "ssl_backbone_fresh_pool_and_head"
    assert not result["summary"]["benchmark_result"]
    assert (out / "ssl_initialization.json").is_file()
    checks = json.loads((out / "ssl_smoke_gradients.json").read_text())
    assert set(checks["groups"]) == {"backbone", "attention_pool", "classifier"}
    assert (out / "diagnostic_validation/predictions.csv").is_file()
    assert digest_file(path) == cfg.ssl_encoder_sha256


@pytest.mark.parametrize("mode", ["smoke_only", "full_after_smokes", "missing_smoke", "failed_smoke",
                                   "provenance_only", "provenance_mismatch", "audit_conflict"])
def test_finetuning_workflow_opt_in_and_stage_order(source, monkeypatch, tmp_path, mode):
    import nbformat
    import src.config as config_module
    import src.experiments as experiments
    import src.kaggle as kaggle
    import src.unattended as unattended
    path, cfg = source
    if mode == "provenance_mismatch":
        state = torch.load(path, weights_only=True)
        state["provenance"]["preprocessing_identity"]["raw_file_stats_sha256"] = "changed"
        torch.save(state, path)
        cfg.ssl_encoder_sha256 = digest_file(path)
    root = Path(__file__).parents[1]
    (Path(cfg.data.data_dir) / "records100").mkdir()
    ns = {}
    exec(nbformat.read(root / "05_btp_kaggle_final.ipynb", as_version=4).cells[1].source, ns)
    settings = ns["SETTINGS"]
    session = tmp_path / "worker_session"
    session.mkdir()
    settings.update(CHECKOUT_DIR=str(root), DATASET_ROOT=cfg.data.data_dir, SESSION_DIR=str(session),
        CACHE_DIR=cfg.data.cache_dir, RUN_SUPERCLASS=False, RUN_SUBCLASS=False, RUN_SMOKE=False,
        RUN_METADATA_AUDIT=False, RUN_SSL_FINETUNE_SMOKE=mode != "missing_smoke",
        RUN_SSL_FINETUNE_SUPERCLASS=mode in {"full_after_smokes", "missing_smoke", "failed_smoke", "audit_conflict"},
        RUN_SSL_FINETUNE_SUBCLASS=mode in {"full_after_smokes", "missing_smoke", "failed_smoke", "audit_conflict"},
        SSL_FINETUNE_PROVENANCE_ONLY=mode in {"provenance_only", "provenance_mismatch", "audit_conflict"},
        SSL_FINETUNE_ENCODER_PATH=str(path), SSL_FINETUNE_ENCODER_SHA256=cfg.ssl_encoder_sha256,
        MODEL_SETTINGS=asdict(cfg.model), AUGMENT=cfg.data.augment, DATASET_VERSION=cfg.data.dataset_version)
    settings["SSL_FINETUNE_SETTINGS"].update(epochs=1, diagnostic_batches=1)
    original_cfg = ExperimentConfig
    monkeypatch.setattr(config_module, "ExperimentConfig", lambda **k: original_cfg(**{**k, "device": "cpu"}))
    monkeypatch.setattr(kaggle, "require_cuda", lambda: None)
    original_load = experiments.load_ptbxl
    monkeypatch.setattr(experiments, "load_ptbxl", lambda *a, **k: original_load(*a, **k, reader=synthetic_reader))
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    if mode in {"provenance_only", "provenance_mismatch", "audit_conflict"}:
        import src.data as data_module
        original_data_load = data_module.load_ptbxl
        monkeypatch.setattr(data_module, "load_ptbxl", lambda *a, **k: original_data_load(*a, **k, reader=synthetic_reader))
        monkeypatch.setattr(experiments, "create_model", lambda *a, **k: pytest.fail("audit constructed a fine-tuning model"))
        monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("audit constructed an optimizer"))
    original_run = experiments.run_experiment
    calls = []
    def run(*a, **k):
        item = k["config"]
        calls.append((item.task, item.diagnostic))
        if mode == "failed_smoke" and len(calls) == 2:
            raise RuntimeError("injected subclass smoke failure")
        return original_run(*a, **k)
    monkeypatch.setattr(experiments, "run_experiment", run)
    code = unattended.run_worker(settings)
    final = json.loads((session / "final_summary.json").read_text())
    if mode in {"missing_smoke", "audit_conflict"}:
        assert code == 1 and calls == []
    elif mode in {"provenance_only", "provenance_mismatch"}:
        assert calls == [] and code == (0 if mode == "provenance_only" else 1)
        reports = list(session.rglob("ssl_provenance_comparison.json"))
        assert len(reports) == 2
        assert not list(session.rglob("*.pth"))
        for task in ("superclass", "subclass"):
            item = final["experiments"][f"ssl_finetune_{task}_provenance_audit"]
            assert item["status"] == ("complete" if code == 0 else "failed")
    elif mode == "failed_smoke":
        assert code == 1 and calls == [("superclass", True), ("subclass", True)]
        assert final["experiments"]["ssl_finetune_superclass"]["status"] == "blocked_by_smoke_failure"
    else:
        assert code == 0
        expected = [("superclass", True), ("subclass", True)]
        if mode == "full_after_smokes":
            expected += [("superclass", False), ("subclass", False)]
            for task in ("superclass", "subclass"):
                out = Path(final["completed"][f"ssl_finetune_{task}"]["output_dir"])
                assert (out / "test/metrics.json").is_file() and (out / "thresholds.json").is_file()
        assert calls == expected
    assert digest_file(path) == cfg.ssl_encoder_sha256


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Requires actual Mamba CUDA; source evidence is synthetic, not 20 executed SSL epochs")
@pytest.mark.parametrize("task", ["superclass", "subclass"])
def test_real_gpu_transfer_and_classification_smoke(data_config, tmp_path, task):
    from src.models.registry import create_model
    from src.experiments import run_experiment
    small = ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, dropout=0)
    path, checksum = make_export(data_config, tmp_path / "gpu_source", create_model("mamba", 5, small), small)
    cfg = ExperimentConfig(task=task, model=small, data=data_config, device="cuda", epochs=1, batch_size=2,
        mixed_precision=False, learning_rate=1e-4, numerical_stability_mode="conservative", diagnostic=True,
        diagnostic_batches=1, ssl_encoder_checkpoint=str(path), ssl_encoder_sha256=checksum,
        output_root=str(tmp_path / "gpu_fine"))
    data = load_ptbxl(task, config=data_config, diagnostic_limits={"train": 2, "val": 2}, reader=synthetic_reader)
    result = run_experiment(config=cfg, prepared_data=data)
    assert (Path(result["output_dir"]) / "ssl_smoke_gradients.json").is_file()
