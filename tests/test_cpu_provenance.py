"""CPU-only setup/controller/worker coverage: no model, optimizer or CUDA probe."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from test_finetuning import source
from test_content_identity import changed_stats
from conftest import synthetic_reader


def notebook_settings():
    nb = json.loads((Path(__file__).parents[1] / "05_btp_kaggle_final.ipynb").read_text())
    namespace = {}
    exec("".join(nb["cells"][1]["source"]), namespace)
    namespace["SETTINGS"].update(SSL_FINETUNE_PROVENANCE_ONLY=True, SSL_FINETUNE_DATA_MODE="independent")
    return namespace


def test_notebook_cpu_audit_defaults():
    from src.unattended import cpu_provenance_only
    namespace = notebook_settings()
    # CPU mode is explicitly configured; notebook defaults now request GPU smokes.
    settings = namespace["SETTINGS"]
    assert cpu_provenance_only(settings)
    assert settings["SSL_FINETUNE_NORMALIZER_POLICY"] == "strict"
    assert settings["SSL_FINETUNE_REFERENCE_CACHE"].endswith("signals_fa14baaf8ce23a18d5778e2d/signals.npy")
    assert settings["SSL_FINETUNE_ENCODER_SHA256"] == "8d99bd77ad9c65e15519cc744c57a1e683e0accd34fa8724c6a5382cdf1e38d4"


@pytest.mark.parametrize("flag", ["RUN_SUPERCLASS", "RUN_SUBCLASS", "RUN_SMOKE", "RUN_METADATA_AUDIT",
    "RUN_SSL_SMOKE", "RUN_SSL_PRETRAIN", "RUN_SSL_FINETUNE_SUPERCLASS", "RUN_SSL_FINETUNE_SUBCLASS", "RUN_SSL_FINETUNE_SMOKE"])
def test_cpu_mode_rejects_conflicting_actions(flag):
    from src.unattended import cpu_provenance_only
    settings = notebook_settings()["SETTINGS"]
    settings[flag] = not settings[flag]
    with pytest.raises(ValueError, match="all other run flags False"):
        cpu_provenance_only(settings)


def test_cpu_dependency_setup_never_probes_or_installs_extensions(tmp_path, monkeypatch):
    import src.kaggle_setup as setup
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(setup, "probe_mamba", lambda: pytest.fail("CUDA probe in CPU audit"))
    monkeypatch.setattr(setup, "preserve_wheels", lambda *a: pytest.fail("CUDA wheel search in CPU audit"))
    monkeypatch.setattr(setup, "run_logged", lambda *a, **k: pytest.fail("unexpected installation"))
    result = setup.setup_dependencies(tmp_path, tmp_path, require_gpu=False)
    assert result["cuda"] == "not_required" and result["mamba_extensions"] == "not_required"


def test_cpu_setup_installs_only_missing_scientific_package(tmp_path, monkeypatch):
    import src.kaggle_setup as setup
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None if name == "wfdb" else object())
    monkeypatch.setattr(setup, "probe_mamba", lambda: pytest.fail("CUDA probe"))
    calls = []
    monkeypatch.setattr(setup, "run_logged", lambda command, *a, **k: calls.append(command))
    setup.setup_dependencies(tmp_path, tmp_path, require_gpu=False)
    assert len(calls) == 1 and calls[0][3:] == ["install", "wfdb==4.3.1"]


def test_cpu_controller_worker_completes_both_real_proof_paths_without_model(source, tmp_path, monkeypatch):
    import torch
    import src.data as data
    import src.kaggle as kaggle
    import src.models.registry as registry
    import src.training as training
    import src.unattended as pipeline
    path, cfg, _, _ = changed_stats(source)
    settings = notebook_settings()["SETTINGS"]
    session = tmp_path / "cpu_session"
    session.mkdir()
    settings.update(SESSION_DIR=str(session), CHECKOUT_DIR=str(Path(__file__).parents[1]),
        DATASET_ROOT=cfg.data.data_dir, DATASET_VERSION=cfg.data.dataset_version,
        CACHE_DIR=cfg.data.cache_dir, MODEL_SETTINGS=asdict(cfg.model),
        SSL_FINETUNE_REFERENCE_CACHE=cfg.ssl_reference_cache,
        SSL_FINETUNE_ENCODER_PATH=str(path), SSL_FINETUNE_ENCODER_SHA256=cfg.ssl_encoder_sha256)
    monkeypatch.setattr(kaggle, "require_cuda", lambda: pytest.fail("GPU required in CPU audit"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("CUDA cache call in CPU audit"))
    monkeypatch.setattr(registry, "create_model", lambda *a, **k: pytest.fail("model constructed"))
    monkeypatch.setattr(training.Trainer, "__init__", lambda *a, **k: pytest.fail("trainer constructed"))
    original = data.load_ptbxl
    monkeypatch.setattr(data, "load_ptbxl", lambda *a, **k: original(*a, **k, reader=synthetic_reader))
    monkeypatch.setattr(kaggle, "find_ptbxl_root", lambda *a: Path(cfg.data.data_dir))
    def setup(*a, **kw):
        assert kw["require_gpu"] is False
        return {"installation": "cpu_setup_test_double"}
    monkeypatch.setattr(pipeline, "setup_dependencies", setup)
    monkeypatch.setattr(pipeline, "run_logged", lambda *a, **k: pipeline.run_worker(settings))
    assert pipeline.controller(settings, session / "execution_config.json") == 0
    summary = json.loads((session / "final_summary.json").read_text())
    assert summary["checks"]["cuda_mamba"] == "not_required_provenance_only"
    for task in ("superclass", "subclass"):
        result = summary["completed"][f"ssl_finetune_{task}_provenance_audit"]
        assert result["summary"]["optimizer_steps"] == 0
        report = json.loads((Path(result["output_dir"]) / "ssl_provenance_comparison.json").read_text())
        assert report["content_equivalence"]["checked_records"] == 24
        assert report["status"] == "passed_verified_training_content"
