"""Regressions for the observed Kaggle failures; no CPU substitute is a GPU result."""
from dataclasses import replace, asdict
import gc
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from src.config import ExperimentConfig
from src.data import load_ptbxl
from src.experiments import run_experiment, diagnostic_limits
from src.cache import select_normalization_ids
from src.utils import write_json
from conftest import synthetic_reader


@pytest.mark.parametrize("width", [5, 23])
def test_single_head_forward_backward_cpu(monkeypatch, width):
    # Only the unavailable CUDA block is replaced by a test-only differentiable
    # linear operation. All real encoder/pooling/head code is executed unchanged.
    class BlockDouble(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.projection = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.projection(x)
    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace(Mamba=BlockDouble))
    path = Path(__file__).parents[1] / "src/models/mamba.py"
    spec = importlib.util.spec_from_file_location("task3_model_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = dict(n_leads=12, n_classes=width, embed_dim=16, hidden_dim=16,
                n_layers=1, d_state=16, d_conv=4, expand=2, dropout=0)
    encoder = module.MambaEncoder(**args)
    model = module.MambaClassifier(**args)
    x = torch.randn(2, 12, 1000)
    assert encoder(x).shape == (2, 32)
    count = []
    handle = model.head.register_forward_hook(lambda *args: count.append(1))
    logits = model(x)
    assert logits.shape == (2, width) and len(count) == 1
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.randint(0, 2, logits.shape).float()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    handle.remove()


def test_sampled_normalization_shared_and_cache_hits(data_config):
    cfg = replace(data_config, normalization_samples=5, normalization_seed=123)
    calls = []
    def reader(path, fs):
        calls.append(path.name)
        return synthetic_reader(path, fs)
    a = load_ptbxl(config=cfg, reader=reader)
    assert len(calls) == 36  # raw reads once; normalization reads only the memory map
    ids = a.provenance["normalization_record_ids"]
    assert len(ids) == 5 and set(ids) <= set(a.metadata.index[a.metadata.strat_fold <= 8])
    assert ids == select_normalization_ids(list(range(24, 0, -1)), cfg)
    b = load_ptbxl("subclass", config=cfg, reader=reader)
    assert len(calls) == 36
    assert a.normalizer.state_dict() == b.normalizer.state_dict()
    assert a.provenance["signal_cache"] == b.provenance["signal_cache"]
    torch.testing.assert_close(a.datasets["val"][0][0], b.datasets["val"][0][0])
    old_ids = ids
    changed = load_ptbxl(config=replace(cfg, normalization_seed=124), reader=reader)
    assert changed.provenance["normalization_record_ids"] != old_ids
    assert len(calls) == 36  # changing normalization does not reread/filter raw signals
    full = load_ptbxl(config=replace(cfg, normalization_mode="full"), reader=reader)
    assert len(full.provenance["normalization_record_ids"]) == 24
    assert len(calls) == 36
    assert len(list(Path(cfg.cache_dir).glob("normalization_*.json"))) == 3


def test_cache_correctness_and_invalidation(data_config):
    a = load_ptbxl(config=data_config, reader=synthetic_reader)
    reference = load_ptbxl(config=replace(data_config, cache_signals=False), reader=synthetic_reader)
    torch.testing.assert_close(a.datasets["val"][0][0], reference.datasets["val"][0][0], atol=0, rtol=0)
    different = load_ptbxl(config=replace(data_config, apply_filter=False), reader=synthetic_reader)
    assert a.provenance["signal_cache"]["path"] != different.provenance["signal_cache"]["path"]
    # Metadata changes invalidate even if record IDs, shape and labels stay the same.
    root = Path(data_config.data_dir)
    csv = root / "ptbxl_database.csv"
    csv.write_bytes(csv.read_bytes() + b"\n")
    changed = load_ptbxl(config=data_config, reader=synthetic_reader)
    assert changed.provenance["signal_cache"]["path"] != a.provenance["signal_cache"]["path"]


def test_corrupt_or_partial_cache_rebuilt(data_config):
    calls = []
    def reader(path, fs):
        calls.append(1)
        return synthetic_reader(path, fs)
    data = load_ptbxl(config=data_config, reader=reader)
    path = Path(data.provenance["signal_cache"]["path"])
    expected = path.read_bytes()
    del data
    gc.collect()
    with path.open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(b"BAD!")  # same file size; checksum must detect corruption
    rebuilt = load_ptbxl(config=data_config, reader=reader)
    assert len(calls) == 72 and path.read_bytes() == expected
    del rebuilt
    gc.collect()
    path.with_name("manifest.json").unlink()
    load_ptbxl(config=data_config, reader=reader)
    assert len(calls) == 108


def test_waveform_file_stats_invalidate(data_config):
    root = Path(data_config.data_dir)
    (root / "record_1.hea").write_text("source signature only; injected reader")
    a = load_ptbxl(config=data_config, reader=synthetic_reader)
    (root / "record_1.hea").write_text("changed source waveform header signature")
    b = load_ptbxl(config=data_config, reader=synthetic_reader)
    assert a.provenance["signal_cache"]["path"] != b.provenance["signal_cache"]["path"]


def test_cached_first_batch_optimizer_and_io_profile(data_config, tmp_path):
    torch.set_num_threads(1)
    reads = []
    def reader(path, fs):
        reads.append(path.name)
        return synthetic_reader(path, fs)
    start = perf_counter()
    data = load_ptbxl(config=data_config, reader=reader)
    preparation_seconds = perf_counter() - start
    before = len(reads)
    start = perf_counter()
    loader = data.loaders(4)["train"]
    x, y, _ = next(iter(loader))
    first_batch_seconds = perf_counter() - start
    model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, 5))
    optimizer = torch.optim.AdamW(model.parameters())
    weight_before = model[-1].weight.detach().clone()
    start = perf_counter()
    loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    optimizer.step()
    step_seconds = perf_counter() - start
    for _ in range(2):
        list(loader)
    assert len(reads) == before == 36
    assert not torch.equal(weight_before, model[-1].weight)
    profile = dict(preparation_seconds=preparation_seconds, first_batch_seconds=first_batch_seconds,
                   optimizer_step_seconds=step_seconds, raw_reads_after_preparation=len(reads) - before,
                   two_epochs_records=48, purpose="synthetic_CPU_profile_not_Kaggle")
    write_json(tmp_path / "profile.json", profile)
    print("PROFILE", profile)


def test_diagnostic_limits_and_synthetic_end_to_end(monkeypatch, data_config, tmp_path):
    import src.experiments as experiments
    original_load = experiments.load_ptbxl
    monkeypatch.setattr(experiments, "load_ptbxl", lambda *a, **kw: original_load(*a, **kw, reader=synthetic_reader))
    monkeypatch.setattr(experiments, "create_model", lambda name, n_classes, config:
                        nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, n_classes)))
    config = ExperimentConfig(epochs=1, batch_size=4, data=data_config, device="cpu",
                              diagnostic=True, diagnostic_batches=1, output_root=str(tmp_path / "runs"))
    assert diagnostic_limits(config) == {"train": 4, "val": 4}
    assert diagnostic_limits(replace(config, diagnostic=False)) is None
    result = run_experiment(config=config)
    out = Path(result["output_dir"])
    assert result["summary"]["benchmark_result"] is False
    assert result["summary"]["train_records"] == 4
    assert (out / "diagnostic_validation/predictions.csv").is_file()
    assert not (out / "test").exists() and not (out / "thresholds.json").exists()
    manifest = json.loads((out / "data_manifest.json").read_text())
    assert "test" not in manifest["splits"]
    assert manifest["provenance"]["declared_dataset_version"] == "unknown"
    assert "optimizer step completed" in (out / "run.log").read_text()


def test_final_notebook_order_and_disabled_execution(monkeypatch):
    import nbformat
    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    book = nbformat.read(root / "05_btp_kaggle_final.ipynb", as_version=4)
    nbformat.validate(book)
    namespace = {}
    tags = []
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        tags += cell.metadata.tags
        source = cell.source
        if "final-config" in cell.metadata.tags:
            source = source.replace("RUN_SUPERCLASS = True", "RUN_SUPERCLASS = False").replace("RUN_SUBCLASS = True", "RUN_SUBCLASS = False")
            source = source.replace("RUN_SMOKE = True", "RUN_SMOKE = False").replace("RUN_METADATA_AUDIT = True", "RUN_METADATA_AUDIT = False")
        exec(compile(source, "05_btp_kaggle_final", "exec"), namespace)
    assert tags == ["final-config", "final-bootstrap", "final-summary"]
    assert not namespace["ACTIVE"] and namespace["EXECUTION_SUMMARY"]["status"] == "skipped"
    embedded = next(c.source for c in book.cells if "final-bootstrap" in c.metadata.get("tags", []))
    assert embedded.startswith((root / "src/bootstrap.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("diagnostic", [False, True])
def test_final_pipeline_sequential_cpu_adapters(monkeypatch, synthetic_root, tmp_path, diagnostic):
    """Exercise fresh-worker stage order with explicit CUDA/model test doubles."""
    import nbformat
    import src.data as data_module
    import src.kaggle as kaggle
    import src.models.registry as registry
    import src.experiments as experiments
    from src.unattended import run_worker
    root = Path(__file__).parents[1]
    (synthetic_root / "records100").mkdir()
    original_load = data_module.load_ptbxl
    original_smoke = kaggle.smoke_forward_backward
    original_prepare = kaggle.prepare_smoke_data
    factory = lambda name, n_classes, config: nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, n_classes))
    monkeypatch.setattr(data_module, "load_ptbxl", lambda *a, **kw: original_load(*a, **kw, reader=synthetic_reader))
    monkeypatch.setattr(registry, "create_model", factory)
    monkeypatch.setattr(experiments, "create_model", factory)
    original_config = ExperimentConfig
    import src.config as config_module
    monkeypatch.setattr(config_module, "ExperimentConfig", lambda **kwargs: original_config(**{**kwargs, "device": "cpu"}))
    monkeypatch.setattr(kaggle, "require_cuda", lambda: None)
    monkeypatch.setattr(kaggle, "prepare_smoke_data", lambda *a, **kw: original_prepare(*a, **kw, reader=synthetic_reader))
    monkeypatch.setattr(kaggle, "smoke_forward_backward", lambda model, batch, **kw:
                        original_smoke(model, batch, device="cpu", mixed_precision=False, require_gpu=False))
    namespace = {}
    book = nbformat.read(root / "05_btp_kaggle_final.ipynb", as_version=4)
    exec(book.cells[1].source, namespace)
    settings = namespace["SETTINGS"]
    session = tmp_path / "session_outputs"
    session.mkdir()
    settings.update(CHECKOUT_DIR=str(root), DATASET_ROOT=str(synthetic_root), SESSION_DIR=str(session),
                    CACHE_DIR=str(tmp_path / "signals_cache"), FULL_EPOCHS=1, FULL_BATCH_SIZE=4,
                    CONTINUATION_SESSION_ID="",
                    NORMALIZATION_SAMPLES=4, SMOKE_FIT_RECORDS=4, DIAGNOSTIC_MODE=diagnostic)
    assert run_worker(settings) == 0
    final = json.loads((session / "final_summary.json").read_text())
    assert final["errors"] == {} and set(final["completed"]) == {"superclass", "subclass"}
    assert final["diagnostic_only"] is diagnostic
    for task, result in final["completed"].items():
        assert result["summary"]["benchmark_result"] is not diagnostic
        assert (Path(result["output_dir"]) / "test").exists() is not diagnostic
    if not diagnostic:
        # Attach the preserved full run read-only in spirit: all writes go to a new session.
        import hashlib
        checkpoints = [Path(v["output_dir"]) / "last.pth" for v in final["completed"].values()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
        next_session = tmp_path / "controlled_continuation"
        next_session.mkdir()
        settings.update(SESSION_DIR=str(next_session), CONTINUATION_SESSION_ID=session.name,
                        CONTINUATION_ARTIFACT_ROOT=str(tmp_path), CONTINUATION_CHECKPOINT_SHA256={}, DIAGNOSTIC_MODE=True)
        assert run_worker(settings) == 0
        continued = json.loads((next_session / "final_summary.json").read_text())
        for result in continued["completed"].values():
            folder = Path(result["output_dir"])
            proof = json.loads((folder / "initial_checkpoint.json").read_text())
            assert proof["provenance_check"]["status"] == "passed"
            assert proof["reset_states"] == ["optimizer", "scheduler", "scaler", "rng", "early_stopping"]
            assert result["summary"]["initialization"] == "checkpoint_weights_only"
            assert not result["summary"]["benchmark_result"]
        assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints} == before


def test_setup_skips_working_install_and_preserves_source_flags(monkeypatch, tmp_path):
    import src.kaggle_setup as setup
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(setup, "probe_mamba", lambda: (True, "working test probe"))
    commands = []
    monkeypatch.setattr(setup, "run_logged", lambda command, log_path, env=None: commands.append((command, env)))
    result = setup.setup_dependencies(tmp_path, tmp_path / "working")
    assert result["installation"] == "already_working" and commands == []
    probes = iter([(False, "not installed"), (True, "test build probe")])
    monkeypatch.setattr(setup, "probe_mamba", lambda: next(probes))
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    result = setup.setup_dependencies(tmp_path, tmp_path / "build", allow_source_build=True, max_jobs=2)
    builds = [(command, env) for command, env in commands if "wheel" in command]
    assert len(builds) == 2
    assert all(env["MAMBA_KEEP_CUDA_BUILD"] == "TRUE" and env["MAX_JOBS"] == "2" for _, env in builds)
    assert all("--no-build-isolation" in command and "-v" in command for command, _ in builds)
    assert "causal-conv1d==1.7.0" in builds[0][0]
    assert "mamba-ssm==2.3.2.post1" in builds[1][0]
    assert result["installation"] == "source_built_saved_wheels"
