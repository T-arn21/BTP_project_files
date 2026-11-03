from copy import deepcopy
from dataclasses import asdict, replace
import pytest
from conftest import synthetic_reader
from src.config import ExperimentConfig
from src.data import load_ptbxl
from src.continuation import verify_initial_provenance, resolve_continuation_checkpoints


@pytest.fixture
def continuation(data_config):
    data = load_ptbxl(config=data_config, reader=synthetic_reader)
    cfg = ExperimentConfig(data=data.config)
    metadata = dict(task=cfg.task, normalizer=data.normalizer.state_dict(), provenance=data.provenance,
                    splits={k: list(map(int, v)) for k, v in data.splits.items()})
    state = dict(config=asdict(cfg), metadata=metadata, class_names=data.class_names)
    safe = replace(cfg, mixed_precision=False, numerical_stability_mode="conservative", learning_rate=1e-4)
    return state, safe, deepcopy(metadata), data.class_names


@pytest.mark.parametrize("damage", ["normalizer", "identity", "splits", "classes", "model", "config", "missing"])
def test_continuation_rejects_provenance_changes(continuation, damage):
    state, cfg, metadata, classes = continuation
    if damage == "normalizer":
        metadata["normalizer"]["std"][0][0] *= 2
    elif damage == "identity":
        metadata["provenance"]["preprocessing_identity"]["raw_file_stats_sha256"] = "different"
    elif damage == "splits":
        metadata["splits"]["train"] = list(reversed(metadata["splits"]["train"]))
    elif damage == "classes":
        classes = list(reversed(classes))
    elif damage == "model":
        cfg.model = replace(cfg.model, n_layers=4)
    elif damage == "config":
        cfg.data = replace(cfg.data, apply_filter=False)
    else:
        state.pop("metadata")
    with pytest.raises(ValueError, match="[Cc]ontinuation|[Cc]heckpoint"):
        verify_initial_provenance(state, cfg, metadata, classes)


def test_continuation_allows_only_declared_diagnostic_prefix(continuation):
    state, cfg, metadata, classes = continuation
    proof = verify_initial_provenance(state, cfg, metadata, classes)
    assert proof["changed_settings"]["mixed_precision"] == {"previous": True, "current": False}
    cfg = replace(cfg, diagnostic=True, batch_size=2, diagnostic_batches=1)
    metadata["splits"] = {k: v[:2] for k, v in metadata["splits"].items() if k in ("train", "val")}
    metadata["provenance"].update(diagnostic_only=True, diagnostic_limits={"train": 2, "val": 2},
                                  signal_cache={"path": "new-subset-cache", "payload_sha256": "different-subset"})
    cfg.data = replace(cfg.data, cache_dir="new-writable-cache")
    assert verify_initial_provenance(state, cfg, metadata, classes)["diagnostic_prefix"]
    metadata["splits"]["train"] = metadata["splits"]["train"][::-1]
    with pytest.raises(ValueError, match="prefix"):
        verify_initial_provenance(state, cfg, metadata, classes)


def test_artifact_resolver_requires_unique_session_and_checkpoint(tmp_path):
    session = "preserved_session"
    for task in ("superclass", "subclass"):
        folder = tmp_path / "input_a" / session / "experiments/benchmark" / f"mamba_{task}_original"
        folder.mkdir(parents=True)
        (folder / "last.pth").write_bytes(b"resolver test only")
    result = resolve_continuation_checkpoints(session, tmp_path)
    assert set(result) == {"superclass", "subclass"}
    assert all(p.endswith("last.pth") for p in result.values())
    with pytest.raises(ValueError, match="SHA256"):
        resolve_continuation_checkpoints(session, tmp_path, expected_sha256={"superclass": "wrong"})
    duplicate = tmp_path / "input_b" / session / "experiments/benchmark"
    duplicate.mkdir(parents=True)
    with pytest.raises(ValueError, match="found 2"):
        resolve_continuation_checkpoints(session, tmp_path)
    assert resolve_continuation_checkpoints(session, tmp_path / "input_a") == result
