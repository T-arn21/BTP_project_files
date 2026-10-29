"""Exact mismatch diagnostics and filesystem-proven aliases, with no real training."""
import json
from pathlib import Path
import torch
import pytest
from test_finetuning import source, ClassifierDouble, make_export
from conftest import synthetic_reader
from src.cache import digest_file
from src.data import load_ptbxl
from src.finetuning import initialize_ssl_backbone
from src.provenance import ProvenanceMismatchError, comparison_report
from src.utils import write_json


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def rewrite_fixture(path, cfg, state):
    """Mutate only synthetic test input, never the user's pinned encoder."""
    torch.save(state, path)
    config = json.loads(path.with_name("config.json").read_text())
    config["data"], config["model"] = state["data_config"], state["model_config"]
    write_json(path.with_name("config.json"), config)
    cfg.ssl_encoder_sha256 = digest_file(path)


def test_collect_all_mismatches_and_exact_values_before_weight_updates(source, tmp_path, monkeypatch):
    path, cfg = source
    state = torch.load(path, weights_only=True)
    p = state["provenance"]
    identity = p["preprocessing_identity"]
    identity.update(root=str(tmp_path / "missing_mount"), declared_version="wrong-version", numpy="wrong-numpy",
                    raw_file_stats_sha256="wrong-signature", preprocessing_source_sha256="wrong-code")
    identity["metadata_sha256"]["ptbxl_database.csv"] = "wrong-metadata"
    state["data_config"].update(data_dir=identity["root"], dataset_version="wrong-version")
    p["normalizer"]["std"][0][0] *= 2
    p["normalization_record_ids"][0] = -1
    p["normalization_configuration"]["record_ids"][0] = -1
    p["used_train_ids"][-1] = 25
    p["excluded_validation_ids"][-1] = -1
    p["train_patient_ids"][-1] = -1
    rewrite_fixture(path, cfg, state)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    model = ClassifierDouble(5)
    monkeypatch.setattr(model, "load_state_dict", lambda *a, **k: pytest.fail("invalid provenance reached weight transfer"))
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(model, cfg, data, report_dir=tmp_path)
    report = json.loads((tmp_path / "ssl_provenance_comparison.json").read_text())
    fields = {d["field"]: d for d in report["mismatches"]}
    required = {"data_config.data_dir", "data_config.dataset_version", "preprocessing_identity.root",
                "preprocessing_identity.declared_version", "preprocessing_identity.numpy",
                "preprocessing_identity.raw_file_stats_sha256", "preprocessing_identity.preprocessing_source_sha256",
                "preprocessing_identity.metadata_sha256.ptbxl_database.csv", "normalizer.std",
                "normalization_record_ids", "normalization_configuration.record_ids", "used_train_ids",
                "excluded_validation_ids", "train_patient_ids"}
    assert required <= set(fields)
    assert report["status"] == "rejected" and all(not d["accepted"] for d in report["mismatches"])
    assert fields["normalizer.std"]["expected"] == p["normalizer"]["std"]
    assert fields["normalizer.std"]["actual"] == data.normalizer.state_dict()["std"]
    assert fields["used_train_ids"]["actual"] == list(range(1, 25))
    assert fields["used_train_ids"]["expected"][-1] == 25
    assert "expected=" in str(caught.value) and "actual=" in str(caught.value)
    assert report["source_export"]["source_config"]["epochs"] == 20
    assert report["source_export"]["source_summary"]["epochs_completed"] == 20
    assert digest_file(path) == cfg.ssl_encoder_sha256


def path_fixture(source, destination):
    path, cfg = source
    state = torch.load(path, weights_only=True)
    state["data_config"]["data_dir"] = destination
    state["provenance"]["preprocessing_identity"]["root"] = destination
    return path, cfg, state


def test_only_verified_same_directory_alias_is_accepted(source, tmp_path):
    root = Path(source[1].data.data_dir)
    alias = str(root / ".." / root.name)  # existing, different spelling; same filesystem object
    path, cfg, state = path_fixture(source, alias)
    rewrite_fixture(path, cfg, state)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    proof = initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    report = json.loads((tmp_path / "ssl_provenance_comparison.json").read_text())
    assert report["status"] == "passed_verified_directory_alias"
    assert {d["field"] for d in report["mismatches"]} == {"data_config.data_dir", "preprocessing_identity.root"}
    assert all(d["accepted"] and d["compatibility_evidence"]["verified"] for d in report["mismatches"])
    assert len(proof["accepted_path_aliases"]) == 2
    assert digest_file(path) == cfg.ssl_encoder_sha256


@pytest.mark.parametrize("kind", ["missing", "different_directory"])
def test_unverified_mount_relocation_is_rejected(source, tmp_path, kind):
    destination = tmp_path / "other_mount"
    if kind == "different_directory":
        destination.mkdir()
    path, cfg, state = path_fixture(source, str(destination))
    rewrite_fixture(path, cfg, state)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    assert caught.value.report["status"] == "rejected"
    assert all(not d["accepted"] for d in caught.value.report["mismatches"])
    assert all(not d["compatibility_evidence"]["verified"] for d in caught.value.report["mismatches"])


@pytest.mark.parametrize("field", ["numpy", "raw_file_stats_sha256", "preprocessing_source_sha256"])
def test_alias_exception_never_hides_other_mismatches(source, tmp_path, field):
    root = Path(source[1].data.data_dir)
    path, cfg, state = path_fixture(source, str(root / ".." / root.name))
    state["provenance"]["preprocessing_identity"][field] = "incompatible"
    rewrite_fixture(path, cfg, state)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    assert all(not d["accepted"] for d in caught.value.report["mismatches"])
    assert f"preprocessing_identity.{field}" in str(caught.value)


def test_report_preserves_full_id_lists_missing_fields_and_boolean_types():
    expected = {"ids": list(range(2000)), "missing": None, "flag": False}
    actual = {"ids": list(range(1999)), "flag": 0}
    report = comparison_report(expected, actual, {}, {})
    fields = {d["field"]: d for d in report["mismatches"]}
    assert fields["ids"]["expected"] == list(range(2000))
    assert fields["ids"]["actual"] == list(range(1999))
    assert not fields["missing"]["actual_present"]
    assert fields["flag"]["expected"] is False and fields["flag"]["actual"] == 0


def test_failed_experiment_saves_current_metadata_and_structured_comparison(source, tmp_path, monkeypatch):
    import src.experiments as experiments
    path, cfg = source
    state = torch.load(path, weights_only=True)
    state["provenance"]["preprocessing_identity"]["raw_file_stats_sha256"] = "changed"
    rewrite_fixture(path, cfg, state)
    original = experiments.load_ptbxl
    monkeypatch.setattr(experiments, "load_ptbxl", lambda *a, **k: original(*a, **k, reader=synthetic_reader))
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    monkeypatch.setattr(experiments.Trainer, "fit", lambda *a, **k: pytest.fail("training began despite mismatch"))
    with pytest.raises(ProvenanceMismatchError):
        experiments.run_experiment(config=cfg)
    out = next(Path(cfg.output_root).glob("diagnostic/mamba_*"))
    for name in ("config.json", "data_manifest.json", "environment.json", "checkpoint_failure.json", "ssl_provenance_comparison.json"):
        assert (out / name).is_file()
    failure = json.loads((out / "checkpoint_failure.json").read_text())
    assert failure["mismatches"][0]["field"] == "preprocessing_identity.raw_file_stats_sha256"
    assert not (out / "last.pth").exists()
    assert digest_file(path) == cfg.ssl_encoder_sha256


def test_all_fold_cache_signature_is_not_compared_to_ssl_training_signature(data_config, tmp_path):
    """The real failed artifacts save an all-fold digest, not its lost train-only digest."""
    import pandas as pd
    from src.config import ModelConfig, ExperimentConfig
    from src.finetuning import compare_ssl_provenance
    metadata_path = Path(data_config.data_dir) / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.loc[0, "scp_codes"] = "{}"  # retain for SSL; exclude from labeled classification
    metadata.to_csv(metadata_path, index=False)
    model_config = ModelConfig(embed_dim=8, hidden_dim=8, n_layers=1, dropout=0)
    path, sha = make_export(data_config, tmp_path / "source", ClassifierDouble(5), model_config)
    state = torch.load(path, weights_only=True)
    cfg = ExperimentConfig(data=data_config, model=model_config, ssl_encoder_checkpoint=str(path), ssl_encoder_sha256=sha)
    data = load_ptbxl(config=data_config, reader=synthetic_reader)
    original_train = state["provenance"]["preprocessing_identity"]["raw_file_stats_sha256"]
    saved_all_fold = data.provenance["preprocessing_identity"]["raw_file_stats_sha256"]
    assert original_train != saved_all_fold
    assert len(data.splits["train"]) == 23
    report = compare_ssl_provenance(state, cfg, data)
    assert report["status"] == "passed_exact"
    assert report["actual"]["preprocessing_identity"]["raw_file_stats_sha256"] == original_train
    assert len(report["actual"]["used_train_ids"]) == 24
    assert len(report["current_supervised_splits"]["train"]) == 23
