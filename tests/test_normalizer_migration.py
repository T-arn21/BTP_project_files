"""Synthetic rounding/provenance cases; no claim about the missing Kaggle run."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_finetuning import source, ClassifierDouble
from test_content_identity import changed_stats
from conftest import synthetic_reader
from src.cache import digest_file
from src.data import load_ptbxl
from src.finetuning import initialize_ssl_backbone, compare_ssl_provenance
from src.provenance import ProvenanceMismatchError


@pytest.fixture
def historical_normalizer(source):
    path, cfg, state, _ = changed_stats(source)
    # Model a one-ULP historical fit difference. This is synthetic, not evidence
    # that the real CPU session's differences are rounding errors.
    for key in ("mean", "std"):
        value = np.asarray(state["provenance"]["normalizer"][key])
        state["provenance"]["normalizer"][key] = np.nextafter(value, np.inf).tolist()
    torch.save(state, path)
    cfg.ssl_encoder_sha256 = digest_file(path)
    return path, cfg, state


def test_full_content_report_runs_with_normalizer_mismatch_without_approval(historical_normalizer, tmp_path, monkeypatch):
    _, cfg, _ = historical_normalizer
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    before = data.normalizer.state_dict()
    model = ClassifierDouble(5)
    monkeypatch.setattr(model, "load_state_dict", lambda *a, **k: pytest.fail("strict policy transferred weights"))
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(model, cfg, data, report_dir=tmp_path)
    report = caught.value.report
    assert report["content_equivalence"]["verified"] and report["content_equivalence"]["checked_records"] == 24
    assert report["blocking_mismatch_count"] == 2
    assert {d["field"] for d in report["mismatches"] if not d["accepted"]} == {"normalizer.mean", "normalizer.std"}
    assert report["normalization_analysis"]["current_fit_reproduced"]
    assert not report["normalizer_migration"]["approved"] and data.normalizer.state_dict() == before


@pytest.mark.parametrize("task", ["superclass", "subclass"])
def test_explicit_source_migration_uses_exact_authenticated_values_for_every_split(historical_normalizer, tmp_path, task):
    path, cfg, state = historical_normalizer
    cfg.ssl_normalizer_policy = "authenticated_source"
    cfg.task = task
    data = load_ptbxl(task, config=cfg.data, reader=synthetic_reader)
    previous = data.normalizer.state_dict()
    proof = initialize_ssl_backbone(ClassifierDouble(5 if task == "superclass" else 23), cfg, data, report_dir=tmp_path)
    report = json.loads((tmp_path / "ssl_provenance_comparison.json").read_text())
    assert report["status"] == "passed_verified_content_original_normalizer"
    assert report["mismatch_count"] == 3 and not report["blocking_mismatch_count"]
    assert report["actual"]["normalizer"] == previous and previous != state["provenance"]["normalizer"]
    assert report["normalizer_migration"]["source_authenticated"]
    assert proof["normalizer_migration"]["approved"]
    assert data.normalizer.state_dict() == state["provenance"]["normalizer"]
    assert all(ds.normalizer is data.normalizer for ds in data.datasets.values())
    assert data.datasets["train"].cache_path == report["content_equivalence"]["current_cache_path"]
    assert data.provenance["ssl_verified_training_cache"]["payload_sha256"] == report["content_equivalence"]["current_payload_sha256"]
    analysis = report["normalization_analysis"]
    assert analysis["verified"] and not analysis["reference_refit_vs_original"]["exact"]
    assert analysis["same_runtime_reference_vs_current"]["exact"]
    assert digest_file(path) == cfg.ssl_encoder_sha256


@pytest.mark.parametrize("damage", ["statistics", "fit_ids", "preprocessing", "split", "signals", "unauthenticated"])
def test_source_policy_never_approves_invalid_normalization_or_data(historical_normalizer, tmp_path, monkeypatch, damage):
    _, cfg, state = historical_normalizer
    cfg.ssl_normalizer_policy = "authenticated_source"
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    if damage == "statistics": data.normalizer.std[0] *= 2
    if damage == "fit_ids": data.provenance["normalization_record_ids"][0] = -1
    if damage == "preprocessing": cfg.model.dropout = .123
    if damage == "split": data.metadata.loc[1, "strat_fold"] = 9
    if damage == "signals":
        import src.content_identity as content
        original = content.PTBXLDataset.processed_signal
        def altered(self, ecg_id):
            x = original(self, ecg_id)
            if ecg_id == 24: x[0, 0] += 1
            return x
        monkeypatch.setattr(content.PTBXLDataset, "processed_signal", altered)
    if damage == "unauthenticated":
        report = compare_ssl_provenance(state, cfg, data)  # no authenticated loader proof
    else:
        with pytest.raises(ProvenanceMismatchError) as caught:
            initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
        report = caught.value.report
    assert report["blocking_mismatch_count"] and report["status"] == "rejected"
    assert not report.get("normalizer_migration", {}).get("approved", False)
    if damage == "statistics":
        assert report["content_equivalence"]["verified"]
        assert not report["normalization_analysis"]["current_fit_reproduced"]
    if damage == "signals": assert report["content_equivalence"]["first_mismatched_ecg_id"] == 24


def test_effective_normalizer_is_saved_in_training_checkpoints(historical_normalizer, monkeypatch):
    import src.experiments as experiments
    _, cfg, state = historical_normalizer
    cfg.ssl_normalizer_policy = "authenticated_source"
    original = experiments.load_ptbxl
    monkeypatch.setattr(experiments, "load_ptbxl", lambda *a, **k: original(*a, **k, reader=synthetic_reader))
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    result = experiments.run_experiment(config=cfg)
    out = Path(result["output_dir"])
    manifest = json.loads((out / "data_manifest.json").read_text())
    checkpoint = torch.load(out / "last.pth", weights_only=True)
    assert manifest["normalizer"] == state["provenance"]["normalizer"]
    assert checkpoint["metadata"]["normalizer"] == manifest["normalizer"]
    assert manifest["provenance"]["ssl_normalizer_migration"]["approved"]


def test_explicit_reference_failure_rejects_even_when_other_fields_match(source, tmp_path):
    _, cfg = source
    cfg.ssl_reference_cache = str(tmp_path / "missing" / "signals.npy")
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    assert caught.value.report["blocking_mismatch_count"] == 1


def test_timestamp_only_change_can_use_authenticated_normalizer(data_config, tmp_path):
    import os
    from test_finetuning import make_export
    from src.config import ModelConfig, ExperimentConfig
    raw = Path(data_config.data_dir) / "record_1.dat"
    raw.write_bytes(b"unchanged physical bytes for the synthetic waveform reader")
    model_config = ModelConfig(embed_dim=8, hidden_dim=8, n_layers=1, dropout=0)
    path, sha = make_export(data_config, tmp_path / "original", ClassifierDouble(5), model_config)
    state = torch.load(path, weights_only=True)
    cfg = ExperimentConfig(data=data_config, model=model_config, ssl_encoder_checkpoint=str(path),
        ssl_encoder_sha256=sha, ssl_reference_cache=state["provenance"]["signal_cache"]["path"],
        ssl_normalizer_policy="authenticated_source")
    before = raw.read_bytes()
    stat = raw.stat()
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    proof = initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    report = json.loads((tmp_path / "ssl_provenance_comparison.json").read_text())
    assert report["mismatch_count"] == 1 and report["content_equivalence"]["checked_records"] == 24
    assert report["normalization_analysis"]["reference_refit_vs_original"]["exact"]
    assert proof["normalizer_migration"]["approved"] and raw.read_bytes() == before


def test_one_ulp_processed_drift_with_correct_current_fit_is_still_rejected(source, tmp_path, monkeypatch):
    import src.content_identity as content
    _, cfg, _, _ = changed_stats(source)
    cfg.ssl_normalizer_policy = "authenticated_source"
    original = content.PTBXLDataset.processed_signal
    def one_ulp(self, ecg_id):
        x = original(self, ecg_id)
        if ecg_id == 24 and self.cache_path is None:
            x[0, 0] = np.nextafter(x[0, 0], np.float32(np.inf))
        return x
    monkeypatch.setattr(content.PTBXLDataset, "processed_signal", one_ulp)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    report = caught.value.report
    assert report["content_equivalence"]["first_mismatched_ecg_id"] == 24
    assert report["content_equivalence"]["waveform_difference"]["unequal_values"] == 1
    assert report["normalization_analysis"]["current_fit_reproduced"]
    assert not report["normalization_analysis"]["same_runtime_reference_vs_current"]["exact"]
    assert not report["normalizer_migration"]["approved"] and report["status"] == "rejected"
