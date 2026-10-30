"""Opt-in ASL stage from validation-selected full classifiers, never SSL-only weights."""
from dataclasses import asdict
import json
import math
from pathlib import Path
from .config import ExperimentConfig, DataConfig, ModelConfig, checkpoint_config_matches


RECIPE = dict(epochs=5, learning_rate=1e-5, min_lr=1e-6, batch_size=2, mixed_precision=False,
              warmup_epochs=0, loss_type="asl", label_smoothing=0., use_pos_weight=False,
              asl_gamma_pos=0., asl_gamma_neg=4., asl_clip=.05, diagnostic=False)


def validate_asl_settings(settings):
    other = [key for key in ("RUN_SUPERCLASS", "RUN_SUBCLASS", "RUN_SMOKE", "RUN_METADATA_AUDIT",
        "RUN_SSL_SMOKE", "RUN_SSL_PRETRAIN", "RUN_SSL_FINETUNE_SMOKE", "RUN_SSL_FINETUNE_SUPERCLASS",
        "RUN_SSL_FINETUNE_SUBCLASS", "RUN_SSL_FINETUNE_RESUME", "SSL_FINETUNE_PROVENANCE_ONLY", "DIAGNOSTIC_MODE")
        if settings.get(key, False)]
    if other:
        raise ValueError(f"ASL second-stage runs are exclusive with other training/diagnostic modes: {other}")


def resolve_asl_sources(settings, roots=("/kaggle/input",)):
    explicit = settings.get("ASL_BEST_CHECKPOINTS", {})
    if all(explicit.get(task) for task in ("superclass", "subclass")):
        return dict(explicit)
    if any(explicit.values()):
        raise ValueError("Specify both ASL best checkpoint paths or use unique-session discovery")
    session_id = settings["ASL_SOURCE_SESSION_ID"]
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("Invalid ASL source session ID")
    hint = settings.get("ASL_SOURCE_ROOT", "")
    sessions = set()
    for root in map(Path, [hint] if hint else roots):
        if root.is_dir() and root.name == session_id:
            sessions.add(root.resolve())
        sessions.update(p.resolve() for p in root.rglob(session_id) if p.is_dir())
    if len(sessions) != 1:
        raise ValueError(f"Expected one attached epoch-15 ASL source session {session_id}; found {len(sessions)}")
    session = sessions.pop()
    result = {}
    for task in ("superclass", "subclass"):
        matches = list(session.glob(f"ssl_finetuning/resumed/benchmark/mamba_{task}_*/best.pth"))
        if len(matches) != 1:
            raise ValueError(f"Expected one validation-selected {task} best.pth; found {len(matches)}")
        result[task] = str(matches[0])
    return result


def preflight_asl_paths(settings):
    """Standard-library-only availability checks before CUDA installation/probes."""
    paths = resolve_asl_sources(settings)
    result = {}
    for task in ("superclass", "subclass"):
        if not settings.get(f"RUN_ASL_{task.upper()}", False):
            continue
        path = Path(paths[task])
        files = [path, *(path.with_name(name) for name in ("config.json", "summary.json", "status.json", "history.csv"))]
        if path.name != "best.pth" or "resume_source" in path.parts or any(not p.is_file() for p in files):
            raise ValueError(f"Missing/invalid ASL best checkpoint export for {task}: {[str(p) for p in files if not p.is_file()]}")
        cfg = json.loads(path.with_name("config.json").read_text())
        summary = json.loads(path.with_name("summary.json").read_text())
        status = json.loads(path.with_name("status.json").read_text())
        if (cfg["task"] != task or cfg["epochs"] != 15 or summary["best_epoch"] != settings["ASL_SOURCE_BEST_EPOCHS"][task]
                or summary.get("epochs_completed") != 15 or status.get("status") != "complete"):
            raise ValueError(f"ASL {task} completion sidecars/expected epoch disagree")
        encoder, cache = Path(cfg["ssl_encoder_checkpoint"]), Path(cfg["ssl_reference_cache"])
        data_root = Path(cfg["data"]["data_dir"])
        required = [encoder, encoder.with_name("config.json"), encoder.with_name("summary.json"), cache,
                    data_root / "ptbxl_database.csv", data_root / "scp_statements.csv", data_root / "records100"]
        if cache.is_file():
            required.append(cache.with_name("manifest.json"))
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise ValueError(f"ASL {task} saved SSL/PTB-XL input paths unavailable: {missing}")
        result[task] = dict(path=str(path), best_epoch=settings["ASL_SOURCE_BEST_EPOCHS"][task],
                            availability="passed", binary_hash_and_vocabulary="checked after dependencies, before training")
    return result


def load_asl_source(path, expected_sha256, expected_epoch):
    import pandas as pd
    import torch
    from .cache import digest_file
    path = Path(path)
    if "resume_source" in path.parts:
        raise ValueError("ASL requires the epoch-15 run's selected best.pth, never resume_source/best.pth")
    required = (path, *(path.with_name(name) for name in ("config.json", "summary.json", "status.json", "history.csv")))
    missing = [str(p) for p in required if not p.is_file()]
    if path.name != "best.pth" or missing:
        raise ValueError(f"ASL requires validation-selected best.pth and completion sidecars; missing={missing}")
    actual_hash = digest_file(path)
    if not isinstance(expected_sha256, str) or actual_hash != expected_sha256.lower():
        raise ValueError("ASL classifier checkpoint SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    saved = json.loads(path.with_name("config.json").read_text())
    summary = json.loads(path.with_name("summary.json").read_text())
    status = json.loads(path.with_name("status.json").read_text())
    history = pd.read_csv(path.with_name("history.csv"), float_precision="round_trip")
    if (saved.get("epochs") != 15 or saved.get("diagnostic") is not False or
            saved.get("ssl_data_mode") != "ssl_cache_backed" or saved.get("model_name") != "mamba" or
            saved.get("loss_type", "bce") != "bce" or summary.get("epochs_completed") != 15 or
            summary.get("benchmark_result") is not True or status.get("status") != "complete" or
            status.get("benchmark_result") is not True or summary.get("experiment_kind") != "ssl_cache_backed_finetuning" or
            summary.get("task") != saved["task"] or summary.get("best_epoch") != expected_epoch or
            state["epoch"] != expected_epoch or not checkpoint_config_matches(state["config"], saved)):
        raise ValueError("ASL source is not the specified best checkpoint from a completed epoch-15 BCE extension")
    names = state["class_names"]
    if len(names) != (5 if saved["task"] == "superclass" else 23) or len(set(names)) != len(names):
        raise ValueError("ASL source classifier vocabulary is invalid")
    if history.epoch.tolist() != list(range(1,16)):
        raise ValueError("ASL source history must cover all 15 epochs")
    pd.testing.assert_frame_equal(pd.DataFrame(state["history"]), history.iloc[:expected_epoch].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    score, selected = -math.inf, None
    for row in history.itertuples():
        value = row.val_auroc if math.isfinite(row.val_auroc) else -row.val_loss
        if not math.isfinite(value):
            raise ValueError("Nonfinite ASL source validation monitor")
        if value > score + saved["min_delta"]:
            score, selected = value, row.epoch
    if selected != expected_epoch or state["best_score"] != score:
        raise ValueError("ASL source best.pth is not the validation macro-AUROC selected checkpoint")
    if not state["model_state_dict"] or any(not torch.isfinite(t).all() for t in state["model_state_dict"].values()):
        raise ValueError("ASL source model tensors are missing/nonfinite")
    return state, dict(path=str(path.resolve()), sha256=actual_hash, epoch=expected_epoch,
        source_config=saved, source_loss_type=saved.get("loss_type", "bce"), completion_epochs=15, selection="validation_macro_auroc_with_original_min_delta",
        source_reported_test_metrics={key: summary.get(key) for key in ("macro_auroc", "macro_auprc", "macro_f1")},
        class_names=names, source_sidecar_sha256={p.name: digest_file(p) for p in required[1:]})


def asl_config(path, pin, best_epoch, output_root, settings=None):
    state, proof = load_asl_source(path, pin, best_epoch)
    saved = dict(proof["source_config"])
    if settings and set(settings) - set(RECIPE):
        raise ValueError("ASL stage settings may change only declared optimization/loss recipe fields")
    recipe = dict(RECIPE, **(settings or {}))
    if recipe["loss_type"] != "asl" or recipe["warmup_epochs"] != 0 or recipe["diagnostic"]:
        raise ValueError("ASL stage requires ASL, a fresh cosine without warmup, and full official splits")
    saved.update(recipe, output_root=str(output_root), initial_checkpoint=str(path),
        initial_checkpoint_mode="ssl_full_classifier", initial_checkpoint_sha256=pin,
        initial_checkpoint_epoch=best_epoch)
    saved["data"], saved["model"] = DataConfig(**saved["data"]), ModelConfig(**saved["model"])
    del state
    return ExperimentConfig(**saved)


def initialize_full_classifier(model, config, data, metadata):
    from .provenance import field_differences
    state, proof = load_asl_source(config.initial_checkpoint, config.initial_checkpoint_sha256, config.initial_checkpoint_epoch)
    old, current = proof["source_config"], asdict(config)
    allowed = set(RECIPE) | {"output_root", "initial_checkpoint", "initial_checkpoint_mode",
                            "initial_checkpoint_sha256", "initial_checkpoint_epoch"}
    for key in allowed:
        current.pop(key, None)
    expected = {key: value for key, value in old.items() if key not in allowed}
    if not checkpoint_config_matches(expected, current):
        raise ValueError("ASL full-classifier initialization changed an unapproved source configuration field")
    differences = field_differences(state["metadata"], metadata)
    if differences or state["class_names"] != data.class_names:
        raise ValueError(f"ASL classifier class/preprocessing/split provenance mismatch: {[d['field'] for d in differences]}")
    targets, weights = model.state_dict(), state["model_state_dict"]
    if set(targets) != set(weights) or any(targets[key].shape != weights[key].shape or
            targets[key].dtype != weights[key].dtype for key in targets):
        raise ValueError("ASL classifier tensor keys/shapes/dtypes differ from the preserved model")
    model.load_state_dict(state["model_state_dict"], strict=True)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("ASL stage must fine-tune all classifier parameters")
    proof.update(initialization="verified_full_classifier_all_weights", transferred_parameter_names=sorted(state["model_state_dict"]),
        restored_states=["all_model_weights_including_pooling_and_head"],
        reset_states=["AdamW", "cosine_scheduler", "scaler", "RNG", "history", "best_score", "early_stopping"],
        changed_settings={key: dict(source=old.get(key), current=asdict(config).get(key)) for key in sorted(allowed)},
        cache_authentication="fresh execution scope; reuse only immutable read-only filesystem with unchanged file/manifest/payload pin", loss_type=config.loss_type)
    return proof
