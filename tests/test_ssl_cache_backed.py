"""Artifact-backed mode: train reads are cache-only; all other gates stay strict."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from test_finetuning import source, ClassifierDouble
from conftest import synthetic_reader
from src.cache import digest_file
from src.ssl_cache_backed import load_cache_backed_data
from src.finetuning import initialize_ssl_backbone


def cache_config(source):
    path, cfg = source
    state = torch.load(path, weights_only=True)
    cfg.ssl_data_mode = "ssl_cache_backed"
    cfg.ssl_reference_cache = state["provenance"]["signal_cache"]["path"]
    cfg.batch_size, cfg.diagnostic_batches = 2, 2
    return path, cfg, state


@pytest.mark.parametrize("task,width", [("superclass", 5), ("subclass", 23)])
def test_train_is_original_cache_and_no_fit_or_independent_audit(source, tmp_path, monkeypatch, task, width):
    from src.signal import SignalNormalizer
    import src.content_identity as content
    import src.data as data_module
    import src.ssl_cache_backed as backed
    path, cfg, state = cache_config(source)
    cfg.task = task
    reads = []
    original = data_module.PTBXLDataset.processed_signal
    def guard(self, ecg_id):
        if self.cache_path is None:
            reads.append(ecg_id)
            assert ecg_id > 24, "training waveform read independently"
        return original(self, ecg_id)
    monkeypatch.setattr(data_module.PTBXLDataset, "processed_signal", guard)
    monkeypatch.setattr(SignalNormalizer, "fit", lambda *a, **k: pytest.fail("normalizer refit"))
    monkeypatch.setattr(content, "verify_training_content", lambda *a, **k: pytest.fail("full equivalence audit"))
    data = load_cache_backed_data(cfg, diagnostic_limits={"train": 4, "val": 4}, reader=synthetic_reader)
    assert reads == [25, 26, 27, 28] and "test" not in data.datasets
    assert data.normalizer.state_dict() == state["provenance"]["normalizer"]
    assert data.datasets["train"].cache_path == str(Path(cfg.ssl_reference_cache).resolve())
    x, y, ecg_id = data.datasets["train"][0]
    array = np.load(cfg.ssl_reference_cache, mmap_mode="r")
    expected = data.normalizer.transform(array[data.datasets["train"].cache_rows[ecg_id]])
    np.testing.assert_array_equal(x.numpy(), expected)
    model = ClassifierDouble(width)
    fresh = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(("head.", "attn_pool."))}
    proof = initialize_ssl_backbone(model, cfg, data, report_dir=tmp_path)
    report = json.loads((tmp_path / "ssl_provenance_comparison.json").read_text())
    assert report["status"] == "passed_authenticated_ssl_cache_backed"
    assert proof["experiment_kind"] == "ssl_cache_backed_finetuning"
    assert report["ssl_cache_backed"]["independent_training_waveform_equivalence"] == {"performed": False, "verified": False}
    assert "content_equivalence" not in report
    for key, value in fresh.items(): torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    assert digest_file(path) == cfg.ssl_encoder_sha256


@pytest.mark.parametrize("damage", ["encoder", "cache_bytes", "manifest_ids", "manifest_identity", "missing", "normalizer", "patient", "train_ids", "version", "model", "metadata", "labels"])
def test_cache_backed_rejects_invalid_artifacts_or_provenance(source, tmp_path, monkeypatch, damage):
    path, cfg, state = cache_config(source)
    if damage == "encoder": cfg.ssl_encoder_sha256 = "0" * 64
    if damage == "cache_bytes":
        arr = np.load(cfg.ssl_reference_cache, mmap_mode="r+")
        arr[0, 0, 0] += 1; arr.flush(); del arr
    if damage in {"manifest_ids", "manifest_identity"}:
        p = Path(cfg.ssl_reference_cache).with_name("manifest.json")
        value = json.loads(p.read_text())
        if damage == "manifest_ids": value["ids"][0] = 25
        else: value["identity"]["numpy"] = "wrong"
        p.write_text(json.dumps(value))
    if damage == "missing": cfg.ssl_reference_cache = str(tmp_path / "missing" / "signals.npy")
    if damage == "version": cfg.data.dataset_version = "wrong"
    if damage == "model": cfg.model.dropout = .123
    if damage in {"patient", "metadata", "train_ids"}:
        import pandas as pd
        p = Path(cfg.data.data_dir) / "ptbxl_database.csv"
        table = pd.read_csv(p)
        if damage == "patient": table.loc[24, "patient_id"] = table.loc[0, "patient_id"]
        elif damage == "train_ids": table.loc[0, "strat_fold"] = 9
        else: table.loc[0, "scp_codes"] = "{'NORM': 100}"
        table.to_csv(p, index=False)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        data = load_cache_backed_data(cfg, diagnostic_limits={"train": 4, "val": 4}, reader=synthetic_reader)
        if damage == "normalizer": data.normalizer.std[0] *= 2
        if damage == "labels": data.datasets["train"]._label_values[0, 0] = 1 - data.datasets["train"]._label_values[0, 0]
        initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)


def test_full_data_keeps_test_patients_held_out_and_normalizer_fixed(source, monkeypatch):
    _, cfg, state = cache_config(source)
    from src.signal import SignalNormalizer
    monkeypatch.setattr(SignalNormalizer, "fit", lambda *a, **k: pytest.fail("supervised fitting used held-out data"))
    data = load_cache_backed_data(cfg, reader=synthetic_reader)
    assert len(data.splits["train"]) == 24 and len(data.splits["val"]) == len(data.splits["test"]) == 6
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(data.metadata.loc[data.splits[a], "patient_id"]) & set(data.metadata.loc[data.splits[b], "patient_id"])
    assert all(ds.normalizer.state_dict() == state["provenance"]["normalizer"] for ds in data.datasets.values())


def test_smoke_exports_record_mode_and_original_normalizer(source, monkeypatch):
    import src.data as data_module
    import src.experiments as experiments
    _, cfg, state = cache_config(source)
    monkeypatch.setattr(data_module, "read_waveform", synthetic_reader)
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    result = experiments.run_experiment(config=cfg)
    out = Path(result["output_dir"])
    ckpt = torch.load(out / "last.pth", weights_only=True)
    assert ckpt["metadata"]["normalizer"] == state["provenance"]["normalizer"]
    assert ckpt["metadata"]["provenance"]["ssl_cache_backed"]["mode"] == "ssl_cache_backed_finetuning"
    assert result["summary"]["experiment_kind"] == "ssl_cache_backed_finetuning"
    assert result["summary"]["benchmark_result"] is False
    assert (out / "ssl_smoke_gradients.json").is_file() and not (out / "test").exists()


def test_explicit_legacy_gpu_smoke_configuration():
    nb = json.loads((Path(__file__).parents[1] / "05_btp_kaggle_final.ipynb").read_text())
    namespace = {}; exec("".join(nb["cells"][1]["source"]), namespace)
    settings = namespace["SETTINGS"]
    settings.update(RUN_ASL_SUPERCLASS=False, RUN_ASL_SUBCLASS=False, RUN_SSL_FINETUNE_SMOKE=True)
    assert namespace["NEED_GPU"] and settings["RUN_SSL_FINETUNE_SMOKE"]
    assert not settings["SSL_FINETUNE_PROVENANCE_ONLY"]
    assert settings["SSL_FINETUNE_DATA_MODE"] == "ssl_cache_backed"
    assert settings["SSL_FINETUNE_NORMALIZER_POLICY"] == "strict"
    assert settings["SSL_FINETUNE_SETTINGS"]["batch_size"] == 2
    assert settings["SSL_FINETUNE_SETTINGS"]["diagnostic_batches"] == 2
    assert settings["SSL_FINETUNE_SETTINGS"]["mixed_precision"] is False
    assert not any(settings[k] for k in ("RUN_SUPERCLASS", "RUN_SUBCLASS", "RUN_SMOKE", "RUN_METADATA_AUDIT", "RUN_SSL_SMOKE", "RUN_SSL_PRETRAIN", "RUN_SSL_FINETUNE_SUPERCLASS", "RUN_SSL_FINETUNE_SUBCLASS"))


def test_worker_runs_only_both_cache_backed_smokes(source, tmp_path, monkeypatch):
    from dataclasses import asdict
    import src.config as config_module
    import src.data as data_module
    import src.kaggle as kaggle
    import src.experiments as experiments
    import src.unattended as pipeline
    path, cfg, _ = cache_config(source)
    root = Path(__file__).parents[1]
    nb = json.loads((root / "05_btp_kaggle_final.ipynb").read_text())
    namespace = {}; exec("".join(nb["cells"][1]["source"]), namespace)
    settings = namespace["SETTINGS"]
    settings.update(RUN_ASL_SUPERCLASS=False, RUN_ASL_SUBCLASS=False, RUN_SSL_FINETUNE_SMOKE=True)
    session = tmp_path / "smoke_worker"; session.mkdir()
    settings.update(SESSION_DIR=str(session), CHECKOUT_DIR=str(root), DATASET_ROOT=cfg.data.data_dir,
        CACHE_DIR=cfg.data.cache_dir, DATASET_VERSION=cfg.data.dataset_version, MODEL_SETTINGS=asdict(cfg.model),
        AUGMENT=False, SSL_FINETUNE_ENCODER_PATH=str(path), SSL_FINETUNE_ENCODER_SHA256=cfg.ssl_encoder_sha256,
        SSL_FINETUNE_REFERENCE_CACHE=cfg.ssl_reference_cache)
    original_config = config_module.ExperimentConfig
    monkeypatch.setattr(config_module, "ExperimentConfig", lambda **kw: original_config(**{**kw, "device": "cpu"}))
    monkeypatch.setattr(kaggle, "require_cuda", lambda: None)  # CPU adapter test, not real GPU validation
    monkeypatch.setattr(kaggle, "find_ptbxl_root", lambda *a: Path(cfg.data.data_dir))
    monkeypatch.setattr(data_module, "read_waveform", synthetic_reader)
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    assert pipeline.run_worker(settings) == 0
    summary = json.loads((session / "final_summary.json").read_text())
    assert set(summary["completed"]) == {"ssl_finetune_superclass_smoke", "ssl_finetune_subclass_smoke"}
    for task in ("superclass", "subclass"):
        result = summary["completed"][f"ssl_finetune_{task}_smoke"]
        assert result["summary"]["train_records"] == result["summary"]["validation_records"] == 4
        assert not result["summary"]["benchmark_result"]
        assert result["summary"]["experiment_kind"] == "ssl_cache_backed_finetuning"
        assert not (Path(result["output_dir"]) / "test").exists()
