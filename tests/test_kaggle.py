"""Local preflight mechanics; these are not evidence of real PTB-XL/GPU training."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.kaggle import find_ptbxl_root, audit_metadata, prepare_smoke_data, smoke_forward_backward, require_cuda
from conftest import synthetic_reader


def test_discovery_and_diagnostics(synthetic_root, tmp_path):
    (synthetic_root / "records100").mkdir()
    assert find_ptbxl_root(search_roots=[synthetic_root]) == synthetic_root.resolve()
    assert find_ptbxl_root(synthetic_root) == synthetic_root.resolve()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        find_ptbxl_root(tmp_path / "wrong", search_roots=[synthetic_root])
    other = tmp_path / "other"
    other.mkdir()
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        (other / name).write_text("placeholder")
    (other / "records100").mkdir()
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_ptbxl_root(search_roots=[synthetic_root])
    with pytest.raises(FileNotFoundError, match="none"):
        find_ptbxl_root(search_roots=[tmp_path / "absent"])


def test_metadata_audit_both_tasks(synthetic_root):
    audit = audit_metadata(synthetic_root)
    assert audit["targets"]["superclass"].shape == (36, 5)
    assert audit["targets"]["subclass"].shape == (36, 23)
    assert audit["summary"]["patient_overlap"] is False
    assert audit["summary"]["release_verified"] is False
    assert set(audit["statistics"]) == {"superclass", "subclass"}
    assert len(audit["summary"]["metadata_sha256"]["ptbxl_database.csv"]) == 64


@pytest.mark.parametrize("task,width", [("superclass", 5), ("subclass", 23)])
def test_smoke_reads_only_tiny_train_val_data_and_steps(synthetic_root, task, width):
    audit = audit_metadata(synthetic_root)
    seen = []
    def reader(path, fs):
        seen.append(int(path.name.split("_")[-1]))
        return synthetic_reader(path, fs)
    batch = prepare_smoke_data(audit, task, fit_records=4, batch_size=2, reader=reader)
    assert len(seen) == 8  # 4 fit records + 2 train + 2 validation
    assert not set(seen) & set(audit["splits"]["test"])
    assert set(batch["normalization_record_ids"]) <= set(audit["splits"]["train"])
    # Test-only tiny adapter exercises the machinery; it does not stand in for Mamba validation.
    model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, width))
    before = model[-1].weight.detach().clone()
    report = smoke_forward_backward(model, batch, device="cpu", mixed_precision=False, require_gpu=False)
    assert report["optimizer_steps"] == 1 and report["finite_gradients"]
    assert report["purpose"] == "smoke_only_no_test_evaluation"
    assert report["train_shape"] == [2, 12, 1000]
    assert np.isfinite(report["train_bce"]) and np.isfinite(report["validation_bce"])
    assert not torch.equal(model[-1].weight, before)


def test_smoke_rejects_wrong_actual_waveform_shape(synthetic_root):
    with pytest.raises(RuntimeError, match="actual 100 Hz"):
        prepare_smoke_data(audit_metadata(synthetic_root), "superclass",
                           reader=lambda p, fs: np.ones((12, 900)))


def test_gpu_requirement_message(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Accelerator"):
        require_cuda()


def test_notebook_safe_default_execution(monkeypatch):
    import nbformat
    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("BTP_REPO_DIR", str(root))
    monkeypatch.delenv("PTBXL_DATA_DIR", raising=False)
    book = nbformat.read(root / "04_btp_experiments.ipynb", as_version=4)
    namespace = {}
    for i, cell in enumerate(book.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"notebook:{i}", "exec"), namespace)
    for control in ("INSTALL_BASE", "INSTALL_MAMBA", "RUN_METADATA_AUDIT", "RUN_SMOKE", "RUN_SUPERCLASS", "RUN_SUBCLASS"):
        assert namespace[control] is False
    assert namespace["CONFIG"].epochs == namespace["FULL_EPOCHS"]
    with pytest.raises(RuntimeError, match="audit"):
        namespace["require_smoke_pass"]("superclass")
    # Full cells must honor CONFIG, not a second hardcoded epoch count.
    for cell in book.cells:
        if cell.cell_type == "code" and any(t in cell.metadata.get("tags", []) for t in ("full-superclass", "full-subclass")):
            assert "epochs=20" not in cell.source


def test_wfdb_file_roundtrip(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    from src.data import read_waveform
    x = synthetic_reader(Path("record_1"), 100).T
    wfdb.wrsamp("tiny", fs=100, units=["mV"] * 12,
                sig_name=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                p_signal=x, fmt=["16"] * 12, write_dir=str(tmp_path))
    loaded = read_waveform(tmp_path / "tiny", 100)
    assert loaded.shape == (12, 1000) and np.isfinite(loaded).all()
    np.testing.assert_allclose(loaded, x.T, atol=1e-3)
    with pytest.raises(ValueError, match="sampling frequency"):
        read_waveform(tmp_path / "tiny", 500)
