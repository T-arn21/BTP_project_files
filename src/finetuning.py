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
from .provenance import comparison_report, comparison_message, ProvenanceMismatchError


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
                       source_config=config, source_summary=summary,
                       source_config_sha256=digest_file(path.parent / "config.json"),
                       source_summary_sha256=digest_file(path.parent / "summary.json"),
                       additional_evidence_verified=[name for name in ("history.csv", "status.json", "split_provenance.json")
                                                     if (path.parent / name).is_file()])


def compare_ssl_provenance(state, config, data, export_evidence=None):
    """Collect all differences, including checks formerly hidden by the first failure."""
    source = state["provenance"]
    folds = build_official_splits(data.metadata)
    train = list(map(int, folds["train"]))
    ignored = {"augment", "unlabeled", "num_workers", "cache_dir", "cache_signals"}
    expected = dict(model_config=state["model_config"],
                    data_config={k: v for k, v in state["data_config"].items() if k not in ignored})
    actual = dict(model_config=asdict(config.model),
                  data_config={k: v for k, v in asdict(data.config).items() if k not in ignored},
                  eligible_train_ids=train, used_train_ids=train,
                  excluded_validation_ids=list(map(int, folds["val"])),
                  excluded_test_ids=list(map(int, folds["test"])),
                  train_patient_ids=sorted(map(int, data.metadata.loc[train, "patient_id"].unique())),
                  normalizer=data.normalizer.state_dict())
    keys = ("eligible_train_ids", "used_train_ids", "excluded_validation_ids", "excluded_test_ids",
            "train_patient_ids", "normalizer", "preprocessing_identity", "normalization_record_ids", "normalization_configuration")
    expected.update({k: source[k] for k in keys if k in source})
    actual.update({k: data.provenance[k] for k in ("normalization_record_ids", "normalization_configuration") if k in data.provenance})
    policy_expected = dict(objective="masked_waveform_reconstruction", diagnostic_labels_used=False,
                           smoke_only=False, patient_separation_verified=True, training_folds=list(range(1, 9)),
                           normalizer_valid=True, current_identity_computed=True, augment=False)
    policy_actual = {k: source[k] for k in ("objective", "diagnostic_labels_used", "smoke_only", "patient_separation_verified", "training_folds") if k in source}
    if "augment" in state["data_config"]:
        policy_actual["augment"] = state["data_config"]["augment"]
    errors = {}
    try:
        SignalNormalizer.from_state_dict(source["normalizer"])
        policy_actual["normalizer_valid"] = True
    except (KeyError, TypeError, ValueError) as exc:
        policy_actual["normalizer_valid"] = False
        errors["normalizer"] = str(exc)
    try:
        # Exact original SSL scope: all official training IDs BEFORE label exclusions.
        actual["preprocessing_identity"] = preprocessing_identity(Path(data.config.data_dir), data.metadata.loc[train],
                                                                  data.config, data.datasets["train"].reader)
        policy_actual["current_identity_computed"] = True
    except (OSError, ValueError) as exc:
        policy_actual["current_identity_computed"] = False
        errors["current_preprocessing_identity"] = str(exc)
    report = comparison_report(expected, actual, policy_expected, policy_actual,
        source_export=export_evidence or {}, computation_errors=errors,
        current_supervised_splits={k: list(map(int, v)) for k, v in data.splits.items()},
        current_diagnostic_only=data.provenance["diagnostic_only"], class_names=data.class_names,
        population_scope="SSL ID/signature comparison uses all official training folds before label exclusions; supervised subsets are recorded separately",
        intentionally_different_controls={k: dict(expected_ssl=state["data_config"].get(k), actual_supervised=asdict(data.config).get(k)) for k in sorted(ignored)})

    # Content migration is opt-in and only eligible for the sole stat-digest
    # difference. Never mask a root/version/code/split/normalization mismatch.
    if config.ssl_reference_cache:
        field = "preprocessing_identity.raw_file_stats_sha256"
        eligible = len(report["mismatches"]) == 1 and report["mismatches"][0]["field"] == field
        if eligible:
            from .content_identity import verify_training_content
            evidence = verify_training_content(state, data, actual["preprocessing_identity"], config.ssl_reference_cache)
            report["content_equivalence"] = evidence
            if evidence["verified"]:
                report["mismatches"][0].update(accepted=True, compatibility_evidence=evidence)
                report.update(status="passed_verified_training_content", blocking_mismatch_count=0,
                              compatibility_policy="sole raw-file stat difference; entire authenticated SSL training waveform content verified; all other checks exact")
        else:
            report["content_equivalence"] = dict(verified=False, attempted=False,
                reason="Requires exactly one mismatch: preprocessing_identity.raw_file_stats_sha256; all other checks must pass")
    return report



def initialize_ssl_backbone(model, config, data, report_dir=None):
    state, proof = load_completed_ssl_export(config.ssl_encoder_checkpoint, config.ssl_encoder_sha256)
    report = compare_ssl_provenance(state, config, data, proof)
    report_path = Path(report_dir) / "ssl_provenance_comparison.json" if report_dir is not None else None
    if report_path:
        write_json(report_path, report)
    if report["blocking_mismatch_count"]:
        print(comparison_message(report, report_path), flush=True)
        raise ProvenanceMismatchError(report, report_path)
    print(f"SSL provenance comparison: {report['status']}; report={report_path}", flush=True)
    source = state["provenance"]
    try:
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
                 provenance_comparison_status=report["status"],
                 accepted_path_aliases=[d for d in report["mismatches"] if d["accepted"] and d["field"].endswith((".root", ".data_dir"))],
                 content_equivalence=report.get("content_equivalence"),
                 verified_ssl_provenance=source, source_model_config=state["model_config"],
                 provenance_verified=True, transferred_parameter_names=sorted(transferred),
                 freshly_initialized_parameter_names=sorted(fresh),
                 ignored_ssl_pooling_parameter_names=sorted(k for k in weights if k.startswith("attn_pool.")),
                 fresh_initialization_seed=config.seed, all_parameters_trainable=True,
                 optimizer_scheduler_scaler="fresh", classification_labels="existing_official_supervised_mapping")
    return proof


def run_provenance_audit(config):
    """Metadata/preprocessing audit only: no classifier, optimizer or GPU training."""
    from datetime import datetime, timezone
    from uuid import uuid4
    from .data import load_ptbxl
    from .utils import environment_info
    out = Path(config.output_root) / f"{config.task}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "ssl_provenance_comparison.json"
    write_json(out / "status.json", dict(status="running", mode="provenance_only", optimizer_steps=0))
    try:
        state, proof = load_completed_ssl_export(config.ssl_encoder_checkpoint, config.ssl_encoder_sha256)
        limits = {k: config.batch_size * config.diagnostic_batches for k in ("train", "val")}
        data = load_ptbxl(config.task, config=config.data, seed=config.seed, diagnostic_limits=limits)
        write_json(out / "config.json", asdict(config))
        write_json(out / "environment.json", environment_info())
        write_json(out / "data_manifest.json", dict(task=config.task, provenance=data.provenance,
            normalizer=data.normalizer.state_dict(), splits={k: list(map(int, ids)) for k, ids in data.splits.items()},
            class_names=data.class_names))
        report = compare_ssl_provenance(state, config, data, proof)
        write_json(report_path, report)
        if report["blocking_mismatch_count"]:
            raise ProvenanceMismatchError(report, report_path)
        summary = dict(status="complete", mode="provenance_only", optimizer_steps=0, benchmark_result=False,
                       provenance_comparison_status=report["status"], output_dir=str(out.resolve()))
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", summary)
        return {"output_dir": str(out.resolve()), "summary": summary}
    except Exception as exc:
        write_json(out / "status.json", dict(status="failed", mode="provenance_only", optimizer_steps=0, error=str(exc),
                                            provenance_report=str(report_path) if report_path.is_file() else None))
        raise


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
