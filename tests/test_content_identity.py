"""Full-population proof gates, using synthetic checkpoints and deterministic ECGs."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from test_finetuning import source, ClassifierDouble
from conftest import synthetic_reader
from src.cache import preprocessing_identity
from src.data import load_ptbxl
from src.finetuning import initialize_ssl_backbone
from src.provenance import ProvenanceMismatchError
from src.content_identity import resolve_reference_cache


def changed_stats(source):
    path, cfg = source
    state = torch.load(path, weights_only=True)
    cfg.ssl_reference_cache = state["provenance"]["signal_cache"]["path"]
    # Physical files are irrelevant to the injected synthetic reader, but their
    # existence/mtime contributes to the historical stat signature.
    raw = Path(cfg.data.data_dir) / "record_24.dat"
    raw.write_bytes(b"synthetic physical record")
    return path, cfg, state, raw


def test_stat_signature_includes_mtime_not_content(source):
    path, cfg, state, raw = changed_stats(source)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    metadata = data.metadata.loc[state["provenance"]["eligible_train_ids"]]
    before = preprocessing_identity(raw.parent, metadata, cfg.data, synthetic_reader)
    stat = raw.stat()
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    after = preprocessing_identity(raw.parent, metadata, cfg.data, synthetic_reader)
    assert before["raw_file_stats_sha256"] != after["raw_file_stats_sha256"]
    before.pop("raw_file_stats_sha256"); after.pop("raw_file_stats_sha256")
    assert before == after


def test_complete_content_proof_preserves_mismatch_and_fresh_heads(source, tmp_path):
    path, cfg, state, _ = changed_stats(source)
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    model = ClassifierDouble(5)
    head = model.head.weight.clone()
    proof = initialize_ssl_backbone(model, cfg, data, report_dir=tmp_path)
    report = json.loads((tmp_path / "ssl_provenance_comparison.json").read_text())
    assert report["status"] == "passed_verified_training_content"
    assert report["mismatch_count"] == 1 and report["blocking_mismatch_count"] == 0
    difference = report["mismatches"][0]
    assert difference["expected"] != difference["actual"] and difference["accepted"]
    evidence = proof["content_equivalence"]
    assert evidence["checked_records"] == 24 and evidence["training_ids"] == list(range(1, 25))
    assert evidence["expected_content_sha256"] == evidence["actual_content_sha256"]
    assert evidence["reference_payload_sha256"] == state["provenance"]["signal_cache"]["payload_sha256"]
    assert torch.equal(model.head.weight, head)


@pytest.mark.parametrize("layout", ["file", "entry", "parent"])
def test_reference_layouts_authenticate_full_population(source, tmp_path, layout):
    _, cfg, state, _ = changed_stats(source)
    file = Path(cfg.ssl_reference_cache)
    cfg.ssl_reference_cache = str({"file": file, "entry": file.parent, "parent": file.parent.parent}[layout])
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    proof = initialize_ssl_backbone(ClassifierDouble(5), cfg, data, report_dir=tmp_path)
    evidence = proof["content_equivalence"]
    assert evidence["verified"] and evidence["checked_records"] == 24
    assert evidence["path_resolution"]["resolved_signals_path"] == str(file.resolve())
    assert evidence["reference_payload_sha256"] == state["provenance"]["signal_cache"]["payload_sha256"]


@pytest.mark.parametrize("layout", ["file", "entry", "parent"])
@pytest.mark.parametrize("missing", ["signals.npy", "manifest.json"])
def test_missing_reference_files_report_actionable_paths(source, layout, missing):
    _, cfg = source
    state = torch.load(source[0], weights_only=True)
    file = Path(state["provenance"]["signal_cache"]["path"])
    requested = {"file": file, "entry": file.parent, "parent": file.parent.parent}[layout]
    file.with_name(missing).unlink()  # synthetic fixture only
    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        resolve_reference_cache(requested, state["provenance"])
    assert str(requested) in str(caught.value)
    assert "signals.npy" in str(caught.value) and "manifest.json" in str(caught.value)


def test_parent_rejects_ambiguous_or_unrelated_entries(source, tmp_path):
    import shutil
    state = torch.load(source[0], weights_only=True)
    file = Path(state["provenance"]["signal_cache"]["path"])
    duplicate = file.parent.parent / "signals_duplicate"
    shutil.copytree(file.parent, duplicate)
    with pytest.raises(ValueError, match="found 2"):
        resolve_reference_cache(file.parent.parent, state["provenance"])
    manifest = json.loads((duplicate / "manifest.json").read_text())
    manifest["payload_sha256"] = "wrong"
    (duplicate / "manifest.json").write_text(json.dumps(manifest))
    resolved, _ = resolve_reference_cache(file.parent.parent, state["provenance"])
    assert resolved == file
    with pytest.raises(ValueError, match="found 0"):
        resolve_reference_cache(tmp_path / "pretraining", state["provenance"])


@pytest.mark.parametrize("damage", ["last_ecg", "nonfinite", "reference_bytes", "manifest_ids", "manifest_identity", "missing", "other_gate", "disabled"])
def test_full_proof_rejects_damage_without_transfer(source, tmp_path, monkeypatch, damage):
    path, cfg, state, _ = changed_stats(source)
    if damage == "disabled": cfg.ssl_reference_cache = None
    if damage == "missing": cfg.ssl_reference_cache = str(tmp_path / "absent" / "signals.npy")
    if damage == "reference_bytes":
        ref = np.load(cfg.ssl_reference_cache, mmap_mode="r+")
        ref[-1, 0, 0] += 1
        ref.flush()
        del ref
    if damage in {"manifest_ids", "manifest_identity"}:
        manifest_path = Path(cfg.ssl_reference_cache).with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if damage == "manifest_ids": manifest["ids"] = manifest["ids"][:-1]
        else: manifest["identity"]["numpy"] = "wrong"
        manifest_path.write_text(json.dumps(manifest))
    data = load_ptbxl(config=cfg.data, reader=synthetic_reader)
    if damage == "other_gate": data.normalizer.std[0] *= 2
    if damage in {"last_ecg", "nonfinite"}:
        import src.content_identity as content
        original = content.PTBXLDataset.processed_signal
        def altered(self, ecg_id):
            x = original(self, ecg_id)
            if ecg_id == 24: x[0, 0] = np.nan if damage == "nonfinite" else x[0, 0] + 1
            return x
        monkeypatch.setattr(content.PTBXLDataset, "processed_signal", altered)
    model = ClassifierDouble(5)
    monkeypatch.setattr(model, "load_state_dict", lambda *a, **k: pytest.fail("invalid proof reached transfer"))
    with pytest.raises(ProvenanceMismatchError) as caught:
        initialize_ssl_backbone(model, cfg, data, report_dir=tmp_path)
    report = caught.value.report
    assert report["status"] == "rejected" and report["blocking_mismatch_count"]
    if damage != "other_gate":
        assert not report.get("content_equivalence", {}).get("verified", False)
    if damage == "last_ecg":
        assert report["content_equivalence"]["first_mismatched_ecg_id"] == 24
        assert report["content_equivalence"]["checked_records"] == 23
    if damage == "other_gate":
        assert report["content_equivalence"]["verified"]
        assert not report["normalization_analysis"]["current_fit_reproduced"]
