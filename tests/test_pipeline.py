from dataclasses import replace
from pathlib import Path
import ast
import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.labels import build_labels, SUPERCLASSES
from src.data import load_ptbxl, build_official_splits, read_waveform
from src.config import ExperimentConfig, ModelConfig, DataConfig
from src.signal import SignalNormalizer, preprocess_signal, pad_or_crop
from src.augmentation import ECGAugmentation, DEFAULT_PARAMS
from src.training import Trainer, make_pos_weight, load_checkpoint
from src.evaluation import evaluate, tune_thresholds, predict, export_evaluation, plot_results
from src.models.registry import create_model, register_model
from conftest import synthetic_reader


@pytest.mark.parametrize("task,width", [("superclass", 5), ("subclass", 23)])
def test_official_vocabulary_and_multilabel(official_statements, task, width):
    meta = pd.DataFrame({"scp_codes": [{"NORM": 100, "IMI": 0, "CRBBB": 100}, {"SR": 100}, {"IMI": 100}]})
    labels = build_labels(meta, official_statements, task)
    assert labels.shape == (3, width)
    assert set(np.unique(labels)) <= {0, 1}
    assert labels.iloc[0].sum() == 3 and labels.iloc[1].sum() == 0
    assert tuple(labels.columns) == (SUPERCLASSES if task == "superclass" else tuple(sorted(labels.columns)))
    assert (labels == build_labels(meta, official_statements.sample(frac=1, random_state=2), task)).all().all()
    assert labels["NORM"].sum() == 1  # multi-diagnosis record includes NORM


def test_all_23_official_subclasses(official_statements):
    diagnostic = official_statements.loc[official_statements.diagnostic.eq(1)]
    meta = pd.DataFrame({"scp_codes": [{code: 100} for code in diagnostic.index]})
    labels = build_labels(meta, official_statements, "subclass")
    assert labels.shape[1] == 23 and (labels.sum() > 0).all()
    for i, (_, row) in enumerate(diagnostic.iterrows()):
        assert labels.iloc[i][row.diagnostic_subclass] == 1


def test_bad_mapping(official_statements):
    scp = official_statements.copy()
    scp.loc["NORM", "diagnostic_subclass"] = np.nan
    with pytest.raises(ValueError, match="mapping"):
        build_labels(pd.DataFrame({"scp_codes": [{}]}), scp, "subclass")
    with pytest.raises(ValueError, match="task"):
        build_labels(pd.DataFrame(), scp, "other")


def test_split_integrity():
    m = pd.DataFrame({"patient_id": [1, 2, 3], "strat_fold": [1, 9, 10]})
    assert [list(x) for x in build_official_splits(m).values()] == [[0], [1], [2]]
    m.loc[1, "patient_id"] = 1
    with pytest.raises(ValueError, match="Patient leakage"):
        build_official_splits(m)
    m.loc[1, "patient_id"] = 2
    m["strat_fold"] = m["strat_fold"].astype(float)
    m.loc[0, "strat_fold"] = 1.5
    with pytest.raises(ValueError, match="strat_fold"):
        build_official_splits(m)


@pytest.mark.parametrize("layout,shape", [("channels_first", (12, 1000)), ("time_first", (1000, 12))])
def test_loaders_shared_preprocessing(data_config, layout, shape):
    cfg = replace(data_config, layout=layout)
    a = load_ptbxl(config=cfg, reader=synthetic_reader)
    b = load_ptbxl("subclass", config=cfg, reader=synthetic_reader)
    assert a.normalizer.state_dict() == b.normalizer.state_dict()
    assert torch.equal(a.datasets["val"][0][0], b.datasets["val"][0][0])
    x, y, ids = next(iter(a.loaders(5)["val"]))
    assert tuple(x.shape[1:]) == shape and y.shape[1] == 5
    assert y.dtype == x.dtype == torch.float32
    assert b.labels.shape[1] == 23 and (b.labels.sum() == 0).any()  # absent rare classes retained
    assert sum(len(batch[0]) for batch in a.loaders(5)["train"]) == 24  # no dropped tail
    expected = np.concatenate([a.datasets["train"].processed_signal(i) for i in a.splits["train"]], axis=1)
    np.testing.assert_allclose(a.normalizer.mean[:, 0], expected.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(a.normalizer.std[:, 0], expected.std(axis=1), rtol=1e-5)
    assert np.isfinite(a.statistics.prevalence).all()


def test_no_normalization_leakage(data_config):
    baseline = load_ptbxl(config=data_config, reader=synthetic_reader)
    def shifted_test_reader(path, fs):
        x = synthetic_reader(path, fs)
        return x * 1000 if int(path.name.split("_")[-1]) > 24 else x
    changed = load_ptbxl(config=data_config, reader=shifted_test_reader)
    assert baseline.normalizer.state_dict() == changed.normalizer.state_dict()


def test_unlabeled_policy(data_config):
    root = Path(data_config.data_dir)
    m = pd.read_csv(root / "ptbxl_database.csv")
    m.loc[0, "scp_codes"] = "{'SR': 100}"
    m.to_csv(root / "ptbxl_database.csv", index=False)
    a = load_ptbxl(config=data_config, reader=synthetic_reader)
    assert a.dropped_ids == [1] and 1 not in a.splits["train"]
    assert 1 in a.provenance["normalization_record_ids"]
    with pytest.raises(ValueError, match="no diagnostic labels"):
        load_ptbxl(config=replace(data_config, unlabeled="error"), reader=synthetic_reader)


def test_corrupt_waveform_fails(data_config):
    def broken_reader(path, fs):
        raise FileNotFoundError("missing")
    with pytest.raises(RuntimeError, match="Failed ECG"):
        load_ptbxl(config=data_config, reader=broken_reader)
    for x in [np.zeros((12, 1000)), np.full((12, 1000), np.nan), np.ones((11, 1000))]:
        with pytest.raises(ValueError):
            preprocess_signal(x)


def test_wfdb_layout_and_sampling_rate(monkeypatch):
    import sys
    from types import SimpleNamespace
    waveform = np.random.default_rng(1).normal(size=(1000, 12))
    fake = SimpleNamespace(rdsamp=lambda path: (waveform, {"fs": 100}))
    monkeypatch.setitem(sys.modules, "wfdb", fake)
    np.testing.assert_array_equal(read_waveform("record", 100), waveform.astype(np.float32).T)
    with pytest.raises(ValueError, match="sampling frequency"):
        read_waveform("record", 500)


def test_500hz_data(data_config):
    data = load_ptbxl(config=replace(data_config, sampling_rate=500), reader=synthetic_reader)
    assert data.datasets["test"][0][0].shape == (12, 5000)


def test_normalization_padding_and_augmentation():
    rng = np.random.default_rng(8)
    signals = [rng.normal(size=(12, 1000)).astype("float32") for _ in range(3)]
    norm = SignalNormalizer().fit(iter(signals))
    restored = SignalNormalizer.from_state_dict(norm.state_dict())
    np.testing.assert_array_equal(norm.transform(signals[0]), restored.transform(signals[0]))
    for n in [800, 1000, 1200]:
        assert pad_or_crop(signals[0][:, :n] if n <= 1000 else np.pad(signals[0], ((0, 0), (0, 200))), 1000).shape == (12, 1000)
    a, b = ECGAugmentation(seed=4), ECGAugmentation(seed=4)
    assert np.array_equal(a(signals[0]), b(signals[0]))
    assert not np.array_equal(a(signals[0]), a(signals[0]))
    for op in DEFAULT_PARAMS:
        x = getattr(a, op)(signals[0])
        assert x.shape == (12, 1000) and np.isfinite(x).all()


def tiny_model(width):
    # Test-only linear adapter, not a new benchmark architecture.
    return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, width))


@pytest.mark.parametrize("width", [5, 23, 24])
def test_training_checkpoint_metrics_exports(tmp_path, width):
    torch.manual_seed(6)
    torch.set_num_threads(1)
    x = torch.randn(16, 12, 64)
    y = torch.randint(0, 2, (16, width)).float()
    ds = TensorDataset(x, y, torch.arange(16))
    train = DataLoader(ds, batch_size=7, shuffle=True, generator=torch.Generator().manual_seed(3))
    val = DataLoader(ds, batch_size=7)
    cfg = ExperimentConfig(epochs=2, mixed_precision=False, device="cpu")
    names = [f"class_{i}" for i in range(width)]
    model = tiny_model(width)
    before = model[-1].weight.detach().clone()
    trainer = Trainer(model, cfg, names, tmp_path, make_pos_weight(y.numpy()))
    history = trainer.fit(train, val)
    assert len(history) == 2 and np.isfinite(history.val_loss).all()
    assert not torch.equal(before, model[-1].weight)
    state = load_checkpoint(tmp_path / "best.pth", model, names)
    assert state["epoch"] in [1, 2]
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(tmp_path / "best.pth", model, names[::-1])
    yt, p, ids, elapsed = predict(model, val)
    assert p.shape == (16, width) and elapsed > 0
    t = tune_thresholds(yt, p)
    result = export_evaluation(tmp_path / "evaluation", yt, p, ids, names, t)
    assert result["records"] == 16
    assert set(["macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc", "macro_f1", "micro_f1"]) <= result.keys()
    assert len(pd.read_csv(tmp_path / "evaluation" / "predictions.csv")) == 16
    assert json.loads((tmp_path / "evaluation" / "metrics.json").read_text())["classes"] == width
    _, table = evaluate(yt, p, names, t)
    plot_results(history, table, tmp_path)
    assert (tmp_path / "loss_curves.png").stat().st_size > 1000
    resumed = Trainer(tiny_model(width), cfg, names, tmp_path, make_pos_weight(y.numpy()))
    # A completed checkpoint can be loaded without running extra epochs.
    resumed.fit(train, val, tmp_path / "last.pth")
    assert len(resumed.history) == 2


def test_interrupted_training_resume(tmp_path):
    from src.utils import seed_everything
    cfg = ExperimentConfig(epochs=3, device="cpu", mixed_precision=False)
    seed_everything(11)
    ds = TensorDataset(torch.randn(12, 12, 64), torch.tensor([[0., 1.], [1., 0.]] * 6), torch.arange(12))
    def loaders():
        return (DataLoader(ds, batch_size=5, shuffle=True, generator=torch.Generator().manual_seed(9)),
                DataLoader(ds, batch_size=5))
    seed_everything(15)
    full = Trainer(tiny_model(2), cfg, ["a", "b"], tmp_path / "full")
    full.fit(*loaders())
    seed_everything(15)
    partial = Trainer(tiny_model(2), cfg, ["a", "b"], tmp_path / "partial")
    original_epoch = partial._epoch
    calls = 0
    def interrupt(loader, train):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_epoch(loader, train)
    partial._epoch = interrupt
    with pytest.raises(RuntimeError, match="simulated interruption"):
        partial.fit(*loaders())
    assert len(partial.history) == 1
    resumed = Trainer(tiny_model(2), cfg, ["a", "b"], tmp_path / "partial")
    resumed.fit(*loaders(), resume_from=tmp_path / "partial" / "last.pth")
    for k, v in full.model.state_dict().items():
        torch.testing.assert_close(v, resumed.model.state_dict()[k], atol=0, rtol=0)
    assert len(resumed.history) == 3


def test_metrics_edge_cases_and_validation_only(tmp_path):
    y = np.array([[0, 1, 0], [0, 1, 1]])
    p = np.array([[0.2, 0.9, 0.1], [0.3, 0.8, 0.9]])
    overall, table = evaluate(y, p, ["none", "all", "mixed"])
    assert np.isnan(table.auroc.iloc[0]) and np.isnan(table.auroc.iloc[1])
    assert np.isnan(table.auprc.iloc[0]) and table.auprc.iloc[1] == 1
    assert overall["valid_auroc_classes"] == 1 and overall["macro_auroc"] == 1
    assert overall["macro_f1"] == pytest.approx(2 / 3)
    assert table.loc[2, ["tn", "fp", "fn", "tp"]].tolist() == [1, 0, 0, 1]
    np.testing.assert_array_equal(tune_thresholds(y, p)[:2], [0.5, 0.5])
    with pytest.raises(ValueError, match="validation"):
        tune_thresholds(y, p, split="test")
    for bad in [p * np.nan, p + 2]:
        with pytest.raises(ValueError):
            evaluate(y, bad, ["none", "all", "mixed"])
    result = export_evaluation(tmp_path, np.zeros((2, 2)), np.zeros((2, 2)), [1, 2], ["a", "b"])
    parsed = json.loads((tmp_path / "metrics.json").read_text())
    assert parsed["macro_auroc"] is None and parsed["micro_auprc"] is None
    assert result["macro_f1"] == 0


def test_early_stopping(tmp_path):
    ds = TensorDataset(torch.randn(8, 12, 64), torch.tensor([[0., 1.], [1., 0.]] * 4), torch.arange(8))
    cfg = ExperimentConfig(epochs=8, patience=1, min_delta=10, device="cpu", mixed_precision=False)
    trainer = Trainer(tiny_model(2), cfg, ["a", "b"], tmp_path)
    assert len(trainer.fit(DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4))) == 2


def test_experiment_integration(monkeypatch, data_config, tmp_path):
    import src.experiments as exp
    original = exp.load_ptbxl
    monkeypatch.setattr(exp, "load_ptbxl", lambda *a, **kw: original(*a, **kw, reader=synthetic_reader))
    register_model("test_linear", lambda n_classes, config: tiny_model(n_classes))
    cfg = ExperimentConfig(model_name="test_linear", epochs=1, device="cpu", mixed_precision=False,
                           output_root=str(tmp_path / "runs"), data=data_config)
    for task in ["superclass", "subclass"]:
        result = exp.run_experiment(task=task, config=cfg)
        assert result["summary"]["classes"] == (5 if task == "superclass" else 23)
        out = Path(result["output_dir"])
        assert (out / "best.pth").exists() and (out / "thresholds.json").exists()
        assert result["summary"]["test_inference_seconds"] > 0


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Requires CUDA and compatible mamba-ssm")
@pytest.mark.parametrize("width", [5, 23, 24])
def test_real_mamba_forward_and_reference_compatibility(width):
    from src.models.mamba import MambaClassifier, MambaEncoder
    cfg = ModelConfig(embed_dim=16, hidden_dim=16, n_layers=1, dropout=0)
    model = create_model("mamba", width, cfg).cuda().eval()
    x = torch.randn(2, 12, 64, device="cuda")
    with torch.no_grad():
        assert model(x).shape == (2, width)
        encoder = MambaEncoder(n_leads=12, n_classes=width, **cfg.__dict__).cuda().eval()
        encoder.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("head.")})
        torch.testing.assert_close(model.forward_features(x), encoder(x))
        # Instantiate untouched original executable architecture, not a mock kernel.
        n = json.loads((Path(__file__).parents[1] / "ptb-xl-ecg-supervised-mamba-trainin-v2-improved.ipynb").read_text(encoding="utf-8"))
        from mamba_ssm import Mamba
        namespace = dict(nn=nn, torch=torch, Mamba=Mamba)
        exec("".join(n["cells"][17]["source"]), namespace)
        original = namespace["MambaClassifier"](12, width, **cfg.__dict__).cuda().eval()
        original.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(original(x), model(x))


def test_notebook_validity():
    import nbformat
    path = Path(__file__).parents[1] / "04_btp_experiments.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code" and not cell.source.startswith("%"):
            ast.parse(cell.source)
