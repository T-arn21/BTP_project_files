"""Strict SSL encoder transfer; reuse the supervised trainer/evaluator unchanged."""
from dataclasses import asdict
from pathlib import Path
import hmac
import json
import math
import pandas as pd
import torch
from .cache import digest_file, preprocessing_identity
from .data import build_official_splits
from .signal import SignalNormalizer
from .numerics import named_finite
from .utils import write_json


BACKBONE_PREFIXES = ("input_proj.", "blocks.", "final_norm.")
REQUIRED_FILES = ("encoder.pth", "config.json", "summary.json")


def load_completed_ssl_export(path, expected_sha256):
    """Hash before safe deserialization; never infer a trusted hash from this file."""
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path.parent / name).is_file()]
    if path.name != "encoder.pth" or missing:
        raise ValueError(f"Expected completed SSL export folder with {REQUIRED_FILES}; missing={missing}")
    actual = digest_file(path)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or not hmac.compare_digest(actual, expected_sha256.lower()):
        raise ValueError("SSL encoder SHA256 mismatch; supply the independently verified original checksum")
    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        provenance = state["provenance"]
        if state["format"] != "btp_masked_ecg_encoder_v1" or state["smoke_only"] is not False or state["epoch"] != 20:
            raise ValueError("SSL source must be the completed epoch-20 encoder, not a smoke/partial/classification checkpoint")
        if state["pooling_pretrained"] is not False:
            raise ValueError("Unexpected SSL pooling provenance")
        if config["epochs"] != 20 or config["model"] != state["model_config"] or config["data"] != state["data_config"]:
            raise ValueError("SSL config and encoder export disagree")
        if (path.parent / "split_provenance.json").is_file():
            if json.loads((path.parent / "split_provenance.json").read_text(encoding="utf-8")) != provenance:
                raise ValueError("SSL split_provenance.json and encoder export disagree")
        if (summary["status"] != "complete" or summary["smoke_only"] is not False or
                summary["full_pretraining_completed"] is not True or summary["epochs_completed"] != 20 or
                summary["diagnostic_labels_used"] is not False):
            raise ValueError("SSL completion evidence is missing or incomplete")
        count = len(provenance["eligible_train_ids"])
        if count < 1 or summary["train_records"] != count:
            raise ValueError("SSL summary does not account for every training-fold ECG")
        if (path.parent / "status.json").is_file():
            status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
            if status["status"] != "complete" or status["smoke_only"] is not False:
                raise ValueError("SSL status reports incomplete training")
        if (path.parent / "history.csv").is_file():
            history = pd.read_csv(path.parent / "history.csv")
            if (history.epoch.tolist() != list(range(1, 21)) or history.records.tolist() != [count] * 20 or
                    not all(math.isfinite(x) and x >= 0 for x in history.masked_mse)):
                raise ValueError("SSL history must account for 20 complete finite epochs over all training-fold ECGs")
        weights = state["encoder_state_dict"]
        if not weights or not all(torch.is_tensor(x) and x.is_floating_point() for x in weights.values()) or not named_finite(list(weights.items())):
            raise ValueError("SSL encoder has invalid/nonfinite weights")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"SSL export lacks required completion evidence: {exc}") from exc
    return state, dict(path=str(path.resolve()), sha256=actual, epoch=20, completion_verified=True,
                       source_config_sha256=digest_file(path.parent / "config.json"),
                       source_summary_sha256=digest_file(path.parent / "summary.json"),
                       additional_evidence_verified=[name for name in ("history.csv", "status.json", "split_provenance.json")
                                                     if (path.parent / name).is_file()])


def initialize_ssl_backbone(model, config, data):
    state, proof = load_completed_ssl_export(config.ssl_encoder_checkpoint, config.ssl_encoder_sha256)
    try:
        if state["model_config"] != asdict(config.model):
            raise ValueError("SSL model configuration mismatch")
        source = state["provenance"]
        folds = build_official_splits(data.metadata)
        train = list(map(int, folds["train"]))
        if (source["objective"] != "masked_waveform_reconstruction" or source["diagnostic_labels_used"] is not False or
                source["smoke_only"] is not False or source["patient_separation_verified"] is not True or
                source["training_folds"] != list(range(1, 9)) or source["eligible_train_ids"] != train or
                source["used_train_ids"] != train or len(set(train)) != len(train)):
            raise ValueError("SSL provenance must include every official training ECG, no diagnostic labels or held-out waveforms")
        for key, split in (("excluded_validation_ids", "val"), ("excluded_test_ids", "test")):
            if source[key] != list(map(int, folds[split])):
                raise ValueError(f"SSL held-out split mismatch: {key}")
        patients = sorted(map(int, data.metadata.loc[train, "patient_id"].unique()))
        if source["train_patient_ids"] != patients:
            raise ValueError("SSL training patient provenance mismatch")
        # SSL used training-only waveform signatures; comparing the full supervised
        # signature would compare different populations. Recompute the exact source scope.
        identity = preprocessing_identity(Path(data.config.data_dir), data.metadata.loc[train], data.config,
                                          data.datasets["train"].reader)
        if source["preprocessing_identity"] != identity:
            raise ValueError("SSL training-only dataset/preprocessing identity mismatch (root, hashes, signatures or versions)")
        SignalNormalizer.from_state_dict(source["normalizer"])
        if source["normalizer"] != data.normalizer.state_dict():
            raise ValueError("SSL normalization statistics mismatch")
        for key in ("normalization_record_ids", "normalization_configuration"):
            if source[key] != data.provenance[key]:
                raise ValueError(f"SSL normalization provenance mismatch: {key}")
        # Intentional differences: supervised augmentation, label-exclusion policy,
        # worker count and cache storage/population. Waveform semantics must agree.
        ignored = {"augment", "unlabeled", "num_workers", "cache_dir", "cache_signals"}
        before = {k: v for k, v in state["data_config"].items() if k not in ignored}
        after = {k: v for k, v in asdict(data.config).items() if k not in ignored}
        if before != after or state["data_config"]["augment"] is not False:
            raise ValueError("SSL waveform configuration mismatch")
        targets = model.state_dict()
        transferred = {k for k in targets if k.startswith(BACKBONE_PREFIXES)}
        fresh = {k for k in targets if k.startswith(("attn_pool.", "head."))}
        weights = state["encoder_state_dict"]
        expected_encoder = transferred | {k for k in fresh if k.startswith("attn_pool.")}
        if not transferred or set(targets) != transferred | fresh or set(weights) != expected_encoder:
            raise ValueError("SSL encoder tensor keys do not match the preserved Mamba backbone")
        if set(state["trained_parameter_names"]) != transferred:
            raise ValueError("SSL trained_parameter_names must cover exactly the transferred backbone")
        for key in expected_encoder:
            if weights[key].shape != targets[key].shape or weights[key].dtype != targets[key].dtype:
                raise ValueError(f"SSL encoder tensor shape/dtype mismatch: {key}")
        # Fill a complete strict state dictionary; no permissive missing-key load.
        model.load_state_dict({k: weights[k] if k in transferred else v for k, v in targets.items()}, strict=True)
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Fine-tuning requires all backbone/pooling/head parameters to be trainable")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"SSL export lacks required transfer provenance: {exc}") from exc
    proof.update(mode="ssl_backbone_finetuning", task=config.task, class_names=data.class_names,
                 verified_ssl_provenance=source, source_model_config=state["model_config"],
                 provenance_verified=True, transferred_parameter_names=sorted(transferred),
                 freshly_initialized_parameter_names=sorted(fresh),
                 ignored_ssl_pooling_parameter_names=sorted(k for k in weights if k.startswith("attn_pool.")),
                 fresh_initialization_seed=config.seed, all_parameters_trainable=True,
                 optimizer_scheduler_scaler="fresh", classification_labels="existing_official_supervised_mapping")
    return proof


def audit_smoke_gradients(trainer, gradients):
    groups = dict(backbone=BACKBONE_PREFIXES, classifier=("head.",))
    if trainer.config.model.pooling == "mean_attention":
        groups["attention_pool"] = ("attn_pool.",)
    report = dict(epoch=trainer.current_epoch, batch_index=trainer.batch_index, groups={})
    for name, prefixes in groups.items():
        selected = [(key, grad) for key, grad in gradients if key.startswith(prefixes)]
        nonzero = any(bool(grad.abs().max().item() > 0) for _, grad in selected)
        if not selected or not named_finite(selected) or not nonzero:
            trainer._fail(f"SSL fine-tuning smoke has missing/nonfinite/zero {name} gradients", "ssl_smoke_gradients")
        report["groups"][name] = dict(finite=True, nonzero=True, gradient_tensors=len(selected))
    write_json(trainer.directory / "ssl_smoke_gradients.json", report)
