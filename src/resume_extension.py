"""Completed artifact-backed runs: fork outputs, extend only the epoch limit."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import shutil
from .config import ExperimentConfig, DataConfig, ModelConfig, checkpoint_config_matches
from .provenance import field_differences


POLICY = "hold_completed_lr"
FILES = ("last.pth", "best.pth", "config.json", "summary.json", "status.json", "history.csv")


def validate_resume_settings(settings):
    competing = [key for key in ("RUN_SUPERCLASS", "RUN_SUBCLASS", "RUN_SMOKE", "RUN_METADATA_AUDIT",
        "RUN_SSL_SMOKE", "RUN_SSL_PRETRAIN", "RUN_SSL_FINETUNE_SMOKE", "RUN_SSL_FINETUNE_SUPERCLASS",
        "RUN_SSL_FINETUNE_SUBCLASS", "SSL_FINETUNE_PROVENANCE_ONLY") if settings.get(key, False)]
    if competing or settings.get("SSL_FINETUNE_RESUME_SCHEDULER_POLICY") != POLICY:
        raise ValueError(f"Completed resume must be explicit and exclusive, with scheduler policy {POLICY}; competing={competing}")


def resolve_completed_runs(settings, search_roots=("/kaggle/input",)):
    explicit = settings.get("SSL_FINETUNE_RESUME_RUN_DIRS", {})
    if any(explicit.values()):
        if not all(explicit.get(task) for task in ("superclass", "subclass")):
            raise ValueError("Specify both completed task directories or use session discovery")
        return {task: str(Path(explicit[task])) for task in ("superclass", "subclass")}
    session_id = settings.get("SSL_FINETUNE_RESUME_SESSION_ID", "")
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("Set the exact completed SSL fine-tuning session ID")
    hint = settings.get("SSL_FINETUNE_RESUME_ARTIFACT_ROOT", "")
    sessions = set()
    for root in map(Path, [hint] if hint else search_roots):
        if root.is_dir() and root.name == session_id:
            sessions.add(root.resolve())
        sessions.update(p.resolve() for p in root.rglob(session_id) if p.is_dir())
    if len(sessions) != 1:
        raise ValueError(f"Expected one attached completed session {session_id}; found {len(sessions)}. Set explicit task directories if ambiguous")
    session = sessions.pop()
    result = {}
    for task in ("superclass", "subclass"):
        candidates = list(session.glob(f"ssl_finetuning/full/benchmark/mamba_{task}_*/last.pth"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one full completed {task} experiment under {session}; found {len(candidates)}")
        result[task] = str(candidates[0].parent)
    return result


def extension_config_matches(saved, current):
    """Only epochs and the required output relocation may differ."""
    current = asdict(current) if isinstance(current, ExperimentConfig) else dict(current)
    if type(current.get("epochs")) is not int or current["epochs"] <= saved.get("epochs", current["epochs"]):
        return False
    current.update(epochs=saved["epochs"], output_root=saved["output_root"])
    return checkpoint_config_matches(saved, current)


def config_from_completed_run(directory, epochs, output_root):
    saved = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
    cfg = dict(saved, epochs=epochs, output_root=str(output_root))
    cfg["data"], cfg["model"] = DataConfig(**cfg["data"]), ModelConfig(**cfg["model"])
    return ExperimentConfig(**cfg)


def validate_completed_extension(directory, config, metadata, class_names, scheduler_policy):
    import pandas as pd
    import torch
    from .cache import digest_file
    source = Path(directory).resolve()
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Completed resume requires source experiment files: missing={missing} at {source}")
    if scheduler_policy != POLICY:
        raise ValueError(f"Explicit scheduler policy must be {POLICY!r}; no schedule reset or silent horizon change")
    state = torch.load(source / "last.pth", map_location="cpu", weights_only=True)
    best = torch.load(source / "best.pth", map_location="cpu", weights_only=True)
    saved = json.loads((source / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if (saved.get("ssl_data_mode") != "ssl_cache_backed" or saved.get("diagnostic") is not False or
            saved.get("model_name") != "mamba" or state["epoch"] != saved["epochs"] or
            summary.get("experiment_kind") != "ssl_cache_backed_finetuning" or
            summary.get("benchmark_result") is not True or status.get("status") != "complete" or
            status.get("benchmark_result") is not True or summary.get("task") != saved["task"]):
        raise ValueError("Source must be a completed full ssl_cache_backed experiment at its original epoch limit")
    if not extension_config_matches(saved, config) or not checkpoint_config_matches(state["config"], saved):
        raise ValueError("Completed resume permits only increased epochs and output relocation; all other config must match")
    differences = field_differences(state["metadata"], metadata)
    if differences or state["class_names"] != list(class_names):
        raise ValueError(f"Completed resume class labels/preprocessing/split provenance mismatch: fields={[d['field'] for d in differences]}")
    if saved["data"]["num_workers"] and saved["data"]["augment"]:
        raise ValueError("Worker-local augmentation RNG was not checkpointed; exact extension requires a source run without worker-local augmentation")
    if not math.isfinite(state["best_score"]) or type(state["counter"]) is not int or state["counter"] < 0:
        raise ValueError("Invalid best-score/early-stopping state")
    if state["counter"] >= saved["patience"]:
        raise ValueError("Source already exhausted early stopping; increasing only epochs cannot reset patience")
    required = ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "history",
                "rng_cpu", "rng_cuda", "rng_python", "rng_numpy", "loader_rng", "augmentation_rng")
    if any(key not in state for key in required):
        raise ValueError("Source last.pth lacks required optimizer/scheduler/scaler/history/RNG resume state")
    if (state["loader_rng"] is None or (saved["data"]["augment"] and state["augmentation_rng"] is None)
            or (saved["device"].startswith("cuda") and not state["rng_cuda"])):
        raise ValueError("Source lacks the RNG states required by its saved loader/augmentation/device configuration")
    if [row["epoch"] for row in state["history"]] != list(range(1, state["epoch"] + 1)):
        raise ValueError("Source checkpoint history does not cover every completed epoch")
    try:
        pd.testing.assert_frame_equal(pd.DataFrame(state["history"]),
            pd.read_csv(source / "history.csv", float_precision="round_trip"), check_dtype=False, check_exact=True)
        pd.testing.assert_frame_equal(pd.DataFrame(best["history"]),
            pd.DataFrame(state["history"][:best["epoch"]]), check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Source history.csv/best.pth disagree with last.pth history") from exc
    if (not 1 <= best["epoch"] <= state["epoch"] or best["epoch"] != summary["best_epoch"] or
            best["best_score"] != state["best_score"] or best["class_names"] != state["class_names"] or
            best["metadata"] != state["metadata"] or not (checkpoint_config_matches(best["config"], saved)
                or extension_config_matches(best["config"], saved))):
        raise ValueError("Original best checkpoint/config/classes/provenance disagree with completed run")
    for checkpoint in (state, best):
        if any(not torch.isfinite(t).all() for t in checkpoint["model_state_dict"].values()):
            raise ValueError("Source model checkpoint contains nonfinite tensors")
    lrs = [group["lr"] for group in state["optimizer_state_dict"]["param_groups"]]
    if (not all(math.isfinite(lr) and lr > 0 for lr in lrs) or
            state["scheduler_state_dict"].get("_last_lr") != lrs):
        raise ValueError("Source scheduler/optimizer learning rates disagree or are nonfinite/nonpositive")
    if not state.get("scheduler_continuation") and (state["scheduler_state_dict"].get("last_epoch") != state["epoch"]
            or lrs != [saved["min_lr"]] * len(lrs)):
        raise ValueError("Source scheduler did not complete its original horizon at the configured minimum LR")
    continuation = state.get("scheduler_continuation") or dict(policy=POLICY,
        origin_total_epochs=saved["epochs"], origin_completed_epoch=state["epoch"], held_lrs=lrs,
        semantics="Restore original completed scheduler unchanged; hold its terminal optimizer LR for extra epochs")
    if continuation["policy"] != POLICY or continuation["held_lrs"] != lrs:
        raise ValueError("Unsupported or inconsistent inherited scheduler continuation")
    return dict(mode="completed_ssl_cache_backed_epoch_extension", source_dir=str(source),
        source_epoch=state["epoch"], target_epochs=config.epochs, source_config=saved,
        source_files={name: digest_file(source / name) for name in FILES},
        scheduler_continuation=continuation, restored_states=["model", "optimizer", "scheduler", "scaler",
            "history", "best_score", "early_stopping", "torch/python/numpy/loader/augmentation_rng"],
        config_exceptions=["epochs (increase only)", "output_root (new folder only)"])


def snapshot_completed_source(proof, destination):
    from .cache import digest_file
    from .utils import write_json
    out, source = Path(destination), Path(proof["source_dir"])
    snapshot = out / "resume_source"
    snapshot.mkdir(exist_ok=False)
    for name, pin in proof["source_files"].items():
        shutil.copy2(source / name, snapshot / name)
        if digest_file(snapshot / name) != pin:
            raise ValueError(f"Source artifact changed while snapshotting: {name}")
    # Working files must remain replaceable even when the input snapshot is read-only.
    shutil.copyfile(snapshot / "best.pth", out / "best.pth")
    shutil.copyfile(snapshot / "history.csv", out / "history.csv")
    write_json(out / "resume_extension.json", proof)
    return str(snapshot / "last.pth")
