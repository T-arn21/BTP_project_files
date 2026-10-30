"""CPU state restoration tests; synthetic checkpoints, no Kaggle or Mamba run."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import shutil
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import ExperimentConfig, DataConfig
from src.training import Trainer
from src.resume_extension import (validate_completed_extension, snapshot_completed_source,
    resolve_completed_runs, validate_resume_settings)
from src.cache import digest_file
from src.utils import seed_everything, write_json
from test_finetuning import source, ClassifierDouble


def model(width):
    return nn.Sequential(nn.Flatten(), nn.Linear(96, 12), nn.Dropout(.2), nn.Linear(12, width))


def loaders(width):
    x = torch.randn(8, 12, 8, generator=torch.Generator().manual_seed(11))
    y = (torch.arange(8)[:, None] + torch.arange(width)[None, :]) % 2
    ds = TensorDataset(x, y.float(), torch.arange(8))
    return (DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(21)),
            DataLoader(ds, batch_size=4, generator=torch.Generator().manual_seed(31)))


@pytest.fixture(scope="module", params=[("superclass", 5), ("subclass", 23)])
def completed(request, tmp_path_factory):
    torch.set_num_threads(1)
    task, width = request.param
    directory = tmp_path_factory.mktemp(task)
    cfg = ExperimentConfig(task=task, epochs=5, batch_size=4, device="cpu", mixed_precision=False,
        ssl_data_mode="ssl_cache_backed", ssl_encoder_checkpoint="synthetic/encoder.pth", ssl_encoder_sha256="a"*64,
        ssl_reference_cache="synthetic/signals.npy", data=DataConfig(augment=False), output_root=str(directory.parent),
        patience=30)
    metadata = dict(task=task, normalizer={"mean": [[0.]]*12, "std": [[1.]]*12},
                    provenance={"training_cache": "authenticated-synthetic", "training_ids": [1,2,3]},
                    splits={"train": [1,2,3], "val": [4], "test": [5]})
    names = [f"label_{i}" for i in range(width)]
    seed_everything(41)
    trainer = Trainer(model(width), cfg, names, directory, metadata=metadata)
    trainer.fit(*loaders(width))
    best = torch.load(directory / "best.pth", weights_only=True)
    write_json(directory / "config.json", asdict(cfg))
    write_json(directory / "summary.json", dict(task=task, benchmark_result=True,
        experiment_kind="ssl_cache_backed_finetuning", best_epoch=best["epoch"]))
    write_json(directory / "status.json", dict(status="complete", benchmark_result=True))
    return directory, cfg, metadata, names, width


def test_full_state_extension_and_interrupted_continuation(completed, tmp_path):
    source, original, metadata, names, width = completed
    pins = {p.name: digest_file(p) for p in source.iterdir() if p.is_file()}
    state = torch.load(source / "last.pth", weights_only=True)
    cfg = replace(original, epochs=10, output_root=str(tmp_path))
    proof = validate_completed_extension(source, cfg, metadata, names, "hold_completed_lr")
    out = tmp_path / "complete"; out.mkdir()
    resume = snapshot_completed_source(proof, out)
    trainer = Trainer(model(width), cfg, names, out, metadata=metadata)
    original_epoch = trainer._epoch
    first = True
    def check_restored(loader, train):
        nonlocal first
        if first:
            first = False
            assert trainer.current_epoch == 6
            for key, value in trainer.model.state_dict().items(): torch.testing.assert_close(value, state["model_state_dict"][key], atol=0, rtol=0)
            assert trainer.best_score == state["best_score"] and trainer.counter == state["counter"]
            assert trainer.scheduler.state_dict() == state["scheduler_state_dict"]
            assert trainer.scaler.state_dict() == state["scaler_state_dict"]
            for key, values in trainer.optimizer.state_dict()["state"].items():
                for name, value in values.items(): torch.testing.assert_close(value, state["optimizer_state_dict"]["state"][key][name], atol=0, rtol=0)
            assert torch.equal(torch.get_rng_state(), state["rng_cpu"])
            assert torch.equal(loader.generator.get_state(), state["loader_rng"])
            assert random.getstate() == state["rng_python"]
            np.testing.assert_array_equal(np.random.get_state()[1], state["rng_numpy"][1])
        return original_epoch(loader, train)
    trainer._epoch = check_restored
    trainer.fit(*loaders(width), resume_from=resume, completed_extension=proof)
    final = torch.load(out / "last.pth", weights_only=True)
    assert [row["epoch"] for row in final["history"]] == list(range(1,11))
    assert final["history"][:5] == state["history"]
    assert all(row["learning_rate"] == original.min_lr for row in final["history"][5:])
    assert final["scheduler_state_dict"] == state["scheduler_state_dict"]
    assert digest_file(out / "resume_source/best.pth") == pins["best.pth"]
    assert digest_file(out / "resume_source/last.pth") == pins["last.pth"]
    # A session interrupted after epoch 6 resumes its inherited frozen policy.
    interrupted = tmp_path / "interrupted"; interrupted.mkdir()
    snapshot = snapshot_completed_source(proof, interrupted)
    partial = Trainer(model(width), cfg, names, interrupted, metadata=metadata)
    epoch = partial._epoch
    def fail(loader, train):
        if partial.current_epoch == 7: raise RuntimeError("session interruption")
        return epoch(loader, train)
    partial._epoch = fail
    with pytest.raises(RuntimeError, match="session interruption"):
        partial.fit(*loaders(width), resume_from=snapshot, completed_extension=proof)
    restored = Trainer(model(width), cfg, names, interrupted, metadata=metadata)
    restored.fit(*loaders(width), resume_from=interrupted / "last.pth")
    for key, value in trainer.model.state_dict().items(): torch.testing.assert_close(value, restored.model.state_dict()[key], atol=0, rtol=0)
    assert pins == {p.name: digest_file(p) for p in source.iterdir() if p.is_file()}


@pytest.mark.parametrize("damage", ["lr", "model", "preprocessing", "labels", "split", "epochs", "policy", "history", "best", "incomplete"])
def test_extension_rejects_changes_other_than_epochs(completed, tmp_path, damage):
    source, cfg, metadata, names, _ = completed
    source_copy = tmp_path / "source"; shutil.copytree(source, source_copy)
    cfg = copy.deepcopy(replace(cfg, epochs=10, output_root=str(tmp_path)))
    metadata, names, policy = copy.deepcopy(metadata), list(names), "hold_completed_lr"
    if damage == "lr": cfg.learning_rate *= 2
    if damage == "model": cfg.model.n_layers += 1
    if damage == "preprocessing": cfg.data.apply_filter = not cfg.data.apply_filter
    if damage == "labels": names.reverse()
    if damage == "split": metadata["splits"]["train"] = [4]
    if damage == "epochs": cfg.epochs = 5
    if damage == "policy": policy = "reset_cosine"
    if damage == "history": (source_copy / "history.csv").write_text("epoch,train_loss\n5,0\n")
    if damage == "best": (source_copy / "best.pth").unlink()
    if damage == "incomplete": write_json(source_copy / "status.json", {"status": "failed"})
    with pytest.raises((ValueError, KeyError)):
        validate_completed_extension(source_copy, cfg, metadata, names, policy)


def test_discovery_requires_unique_full_runs(tmp_path):
    root = tmp_path / "input" / "btp_outputs/sessions/session-id"
    settings = dict(SSL_FINETUNE_RESUME_SESSION_ID="session-id", SSL_FINETUNE_RESUME_SCHEDULER_POLICY="hold_completed_lr")
    for task in ("superclass", "subclass"):
        folder = root / "ssl_finetuning/full/benchmark" / f"mamba_{task}_original"
        folder.mkdir(parents=True); (folder / "last.pth").touch()
    found = resolve_completed_runs(settings, (tmp_path,))
    assert set(found) == {"superclass", "subclass"}
    validate_resume_settings(settings)
    with pytest.raises(ValueError): validate_resume_settings(dict(settings, RUN_SSL_FINETUNE_SMOKE=True))
    duplicate = root / "ssl_finetuning/full/benchmark/mamba_superclass_duplicate"
    duplicate.mkdir(); (duplicate / "last.pth").touch()
    with pytest.raises(ValueError, match="one full completed"):
        resolve_completed_runs(settings, (tmp_path,))


def test_resume_settings_validation_is_standard_library_only():
    import subprocess
    import sys
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'pandas', 'torch'}:
        raise AssertionError('scientific import before dependency setup: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.resume_extension import validate_resume_settings
validate_resume_settings({'SSL_FINETUNE_RESUME_SCHEDULER_POLICY': 'hold_completed_lr'})
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], check=True, capture_output=True)


def test_cache_backed_experiment_extension_preserves_original_results(source, tmp_path, monkeypatch):
    import src.data as data_module
    import src.experiments as experiments
    from conftest import synthetic_reader
    path, cfg = source
    state = torch.load(path, weights_only=True)
    cfg = replace(cfg, epochs=5, diagnostic=False, ssl_data_mode="ssl_cache_backed",
                  ssl_reference_cache=state["provenance"]["signal_cache"]["path"])
    monkeypatch.setattr(data_module, "read_waveform", synthetic_reader)
    monkeypatch.setattr(experiments, "create_model", lambda name, width, config: ClassifierDouble(width))
    old = experiments.run_experiment(config=cfg)
    directory = Path(old["output_dir"])
    pins = {str(p.relative_to(directory)): digest_file(p) for p in directory.rglob("*") if p.is_file()}
    new = experiments.run_experiment(config=replace(cfg, epochs=10, output_root=str(tmp_path / "new")),
        resume_completed_from=directory, scheduler_policy="hold_completed_lr")
    target = Path(new["output_dir"])
    last = torch.load(target / "last.pth", weights_only=True)
    assert target != directory and last["epoch"] == 10 and len(last["history"]) == 10
    assert new["summary"]["initialization"] == "completed_checkpoint_full_state_resume"
    assert new["summary"]["resumed_from_epoch"] == 5
    assert new["summary"]["best_epoch"] == torch.load(target / "best.pth", weights_only=True)["epoch"]
    assert digest_file(target / "resume_source/best.pth") == pins["best.pth"]
    assert pins == {str(p.relative_to(directory)): digest_file(p) for p in directory.rglob("*") if p.is_file()}
