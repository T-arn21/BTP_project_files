"""Unattended controller and fresh scientific worker for the final Kaggle notebook."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from .kaggle_setup import stage_log, setup_dependencies, run_logged


def cpu_provenance_only(settings):
    """Derive device requirements from actual actions, never a stale NEED_GPU flag."""
    if not settings.get("SSL_FINETUNE_PROVENANCE_ONLY", False):
        return False
    forbidden = ("RUN_SUPERCLASS", "RUN_SUBCLASS", "RUN_SMOKE", "RUN_METADATA_AUDIT",
                 "RUN_SSL_SMOKE", "RUN_SSL_PRETRAIN", "RUN_SSL_FINETUNE_SUPERCLASS", "RUN_SSL_FINETUNE_SUBCLASS", "RUN_SSL_FINETUNE_RESUME", "RUN_ASL_SUPERCLASS", "RUN_ASL_SUBCLASS")
    enabled = [key for key in forbidden if settings.get(key, False)]
    if enabled or not settings.get("RUN_SSL_FINETUNE_SMOKE", False):
        raise ValueError(f"CPU provenance-only execution requires RUN_SSL_FINETUNE_SMOKE=True as the audit selector and all other run flags False; enabled={enabled}")
    return True


def save_summary(path, summary):
    temporary = Path(path).with_suffix(".partial.json")
    temporary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def initial_summary(settings):
    return dict(status="running", stage="initialization", checks={"repository_sync": "passed"},
                completed={}, experiments={}, errors={}, session_dir=settings["SESSION_DIR"],
                source_commit=settings.get("SOURCE_REVISION"), diagnostic_only=settings["DIAGNOSTIC_MODE"])


def controller(settings, config_path):
    """Never import scientific libraries before installing/checking dependencies."""
    session = Path(settings["SESSION_DIR"])
    summary = initial_summary(settings)
    summary_path = session / "final_summary.json"
    save_summary(summary_path, summary)
    try:
        if any(settings.get(key, False) for key in ("RUN_ASL_SUPERCLASS", "RUN_ASL_SUBCLASS")):
            from .asl_stage import validate_asl_settings, preflight_asl_paths
            validate_asl_settings(settings)
            summary["stage"] = "ASL input path preflight"
            save_summary(summary_path, summary)
            save_summary(session / "asl_path_preflight.json", preflight_asl_paths(settings))
        if settings.get("RUN_SSL_FINETUNE_RESUME", False):
            from .resume_extension import validate_resume_settings
            validate_resume_settings(settings)
        cpu_only = cpu_provenance_only(settings)
        summary["stage"] = "CPU provenance dependency setup" if cpu_only else "dependency setup and CUDA verification"
        save_summary(summary_path, summary)
        with stage_log(session, summary["stage"]):
            setup = setup_dependencies(settings["CHECKOUT_DIR"], session,
                       wheel_dir=settings["WHEEL_DIR"], allow_source_build=settings["ALLOW_SOURCE_BUILD"],
                       install_base=settings["INSTALL_BASE"], max_jobs=settings["MAX_JOBS"], require_gpu=not cpu_only)
            save_summary(session / "setup_report.json", setup)
            summary["checks"]["dependencies_cpu" if cpu_only else "dependencies_cuda"] = "passed"
            if cpu_only:
                summary["checks"]["cuda_mamba"] = "not_required_provenance_only"
            summary["stage"] = "fresh training worker"
            save_summary(summary_path, summary)
        env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "BTP_GITHUB_TOKEN"}}
        env["PYTHONPATH"] = str(Path(settings["CHECKOUT_DIR"]).resolve())
        env["PYTHONUNBUFFERED"] = "1"
        run_logged([sys.executable, "-u", "-m", "src.unattended", "--worker", "--config", str(config_path)],
                   session / "worker.log", env=env)
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError("Worker exited without a completed execution summary")
        return 0
    except Exception as exc:
        # Worker writes its more precise failure stage before exiting.
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["errors"].setdefault(summary["stage"], str(exc))
        save_summary(summary_path, summary)
        print(f"FAILED {summary['stage']}: {exc}. Artifacts: {session}", flush=True)
        return 1


def run_worker(settings):
    """All stages run sequentially; every completion/failure updates summary on disk."""
    session = Path(settings["SESSION_DIR"])
    summary_path = session / "final_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else initial_summary(settings)
    summary["errors"] = {}
    stage = "source import verification"
    def passed(name):
        summary["checks"][name] = "passed"
        save_summary(summary_path, summary)
    try:
        if any(settings.get(key, False) for key in ("RUN_ASL_SUPERCLASS", "RUN_ASL_SUBCLASS")):
            from .asl_stage import validate_asl_settings
            validate_asl_settings(settings)
        if settings.get("RUN_SSL_FINETUNE_RESUME", False):
            from .resume_extension import validate_resume_settings
            validate_resume_settings(settings)
        cpu_only = cpu_provenance_only(settings)
        with stage_log(session, stage):
            import src
            checkout = Path(settings["CHECKOUT_DIR"]).resolve()
            if Path(src.__file__).resolve().parent != checkout / "src":
                raise RuntimeError("Worker imported source outside the synchronized checkout")
            import gc
            from dataclasses import asdict, replace
            import torch
            from .config import DataConfig, ModelConfig, ExperimentConfig, checkpoint_config_matches
            from .continuation import resolve_continuation_checkpoints
            from .data import load_ptbxl
            from .kaggle import find_ptbxl_root, audit_metadata, prepare_smoke_data, require_cuda, smoke_forward_backward
            from .models.registry import create_model
            from .experiments import run_experiment, diagnostic_limits
            from .utils import environment_info, write_json, seed_everything, log_progress
            for name, module in list(sys.modules.items()):
                if name.startswith("src.") and getattr(module, "__file__", None):
                    if not Path(module.__file__).resolve().is_relative_to(checkout / "src"):
                        raise RuntimeError("A worker module resolved outside the synchronized checkout")
            if not cpu_only:
                require_cuda()
            write_json(session / "environment.json", environment_info())
            passed("synchronized_imports")
        stage = "dataset discovery and patient metadata audit"
        summary["stage"] = stage
        save_summary(summary_path, summary)
        with stage_log(session, stage):
            root = find_ptbxl_root(settings["DATASET_ROOT"] or None)
            ft_requested = any(settings.get(k, False) for k in ("RUN_SSL_FINETUNE_SMOKE", "RUN_SSL_FINETUNE_SUPERCLASS", "RUN_SSL_FINETUNE_SUBCLASS"))
            supervised_requested = ft_requested or any(settings[k] for k in ("RUN_SUPERCLASS", "RUN_SUBCLASS", "RUN_SMOKE", "RUN_METADATA_AUDIT"))
            audit = audit_metadata(root, dataset_version=settings["DATASET_VERSION"]) if supervised_requested else None
            summary.update(dataset_root=str(root), declared_dataset_version=settings["DATASET_VERSION"])
            if settings["RUN_METADATA_AUDIT"]:
                write_json(session / "metadata_audit.json", audit["summary"])
                for task, table in audit["statistics"].items():
                    table.to_csv(session / f"{task}_class_statistics.csv", index=False)
            if audit is not None:
                passed("metadata_patient_audit")  # SSL audits folds separately without labels.
            data_config = DataConfig(data_dir=str(root), sampling_rate=100, duration=10,
                dataset_version=settings["DATASET_VERSION"], num_workers=settings["NUM_WORKERS"],
                normalization_mode=settings["NORMALIZATION_MODE"], normalization_samples=settings["NORMALIZATION_SAMPLES"],
                normalization_seed=settings["NORMALIZATION_SEED"], cache_dir=settings["CACHE_DIR"],
                cache_signals=settings["CACHE_SIGNALS"], apply_filter=settings["APPLY_FILTER"], augment=settings["AUGMENT"])
            config = ExperimentConfig(epochs=settings["FULL_EPOCHS"], batch_size=settings["FULL_BATCH_SIZE"],
                device="cpu" if cpu_only else "cuda", seed=settings["SEED"], mixed_precision=settings["MIXED_PRECISION"],
                numerical_stability_mode=settings.get("NUMERICAL_STABILITY_MODE", "strict"),
                model=ModelConfig(**settings["MODEL_SETTINGS"]), data=data_config,
                diagnostic=settings["DIAGNOSTIC_MODE"], diagnostic_batches=settings["DIAGNOSTIC_BATCHES"],
                log_every_batches=settings["LOG_EVERY_BATCHES"], patience=settings["EARLY_STOPPING_PATIENCE"],
                output_root=str(session / "experiments"), **settings.get("OPTIMIZER_SETTINGS", {}))
            write_json(session / "session_config.json", {"experiment": asdict(config), "source_commit": settings.get("SOURCE_REVISION")})
        # Catch incompatibilities on tiny real batches before building the full cache.
        if settings["RUN_SMOKE"]:
            stage = "short real-data CUDA smoke before full cache"
            summary["stage"] = stage
            save_summary(summary_path, summary)
            with stage_log(session, stage):
                for task in ("superclass", "subclass"):
                    model = None
                    try:
                        seed_everything(config.seed)
                        batch = prepare_smoke_data(audit, task, fit_records=settings["SMOKE_FIT_RECORDS"],
                                                   batch_size=settings["SMOKE_BATCH_SIZE"], config=data_config)
                        log_progress(f"Initializing preserved {task} Mamba for smoke")
                        model = create_model("mamba", len(batch["class_names"]), config.model)
                        report = smoke_forward_backward(model, batch, device="cuda", mixed_precision=config.mixed_precision,
                                                        learning_rate=config.learning_rate)
                        write_json(session / f"{task}_smoke.json", report)
                        passed(f"{task}_gpu_smoke")
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
        else:
            summary["checks"]["real_data_gpu_smoke"] = "skipped_by_configuration"
        ssl_smoke = settings.get("RUN_SSL_SMOKE", False)
        ssl_full = settings.get("RUN_SSL_PRETRAIN", False)
        for key, requested in (("ssl_smoke", ssl_smoke), ("ssl_pretraining", ssl_full)):
            summary["experiments"][key] = {"status": "pending" if requested else "disabled_by_configuration"}
        if ssl_full and not ssl_smoke:
            summary["experiments"]["ssl_pretraining"] = {"status": "blocked_missing_smoke"}
            raise ValueError("RUN_SSL_PRETRAIN requires RUN_SSL_SMOKE=True in the same execution; full SSL never starts without a passed smoke")
        if ssl_smoke:
            from .ssl import SSLConfig, run_ssl
            ssl_config = SSLConfig(data=replace(data_config, cache_dir=str(Path(data_config.cache_dir) / "ssl")),
                                   model=config.model, seed=config.seed, device="cuda", **settings.get("SSL_SETTINGS", {}))
            for key, smoke in (("ssl_smoke", True), ("ssl_pretraining", False)):
                if not smoke and not ssl_full:
                    continue
                stage = key
                summary["stage"] = stage
                summary["experiments"][key] = {"status": "running"}
                save_summary(summary_path, summary)
                try:
                    with stage_log(session, stage):
                        result = run_ssl(ssl_config, session / "ssl", smoke=smoke)
                    summary["completed"][key] = result
                    summary["experiments"][key] = result
                    passed("ssl_gpu_smoke" if smoke else "ssl_pretraining_exports")
                except Exception as exc:
                    summary["experiments"][key] = {"status": "failed", "error": str(exc)}
                    if smoke and ssl_full:
                        summary["experiments"]["ssl_pretraining"] = {"status": "blocked_by_smoke_failure"}
                    raise
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
        asl_tasks = [task for task in ("superclass", "subclass") if settings.get(f"RUN_ASL_{task.upper()}", False)]
        if asl_tasks:
            from .asl_stage import resolve_asl_sources, asl_config
            from .ssl_cache_backed import cache_authentication_scope
            paths = resolve_asl_sources(settings)
            configs = {}
            # Preflight both selected checkpoints before any classifier training.
            stage = "ASL full-classifier source preflight"
            summary["stage"] = stage
            with stage_log(session, stage):
                for task in asl_tasks:
                    configs[task] = asl_config(paths[task], settings["ASL_CHECKPOINT_SHA256"][task],
                        settings["ASL_SOURCE_BEST_EPOCHS"][task], session / "asl_stage2", settings.get("ASL_STAGE_SETTINGS"))
                    if configs[task].task != task:
                        raise ValueError("ASL source task/vocabulary mismatch")
                write_json(session / "asl_source_configs.json", {task: asdict(cfg) for task, cfg in configs.items()})
            with cache_authentication_scope():
                for task, task_config in configs.items():
                    stage = f"asl_{task}"
                    summary["stage"] = stage
                    summary["experiments"][stage] = {"status": "running"}
                    save_summary(summary_path, summary)
                    try:
                        with stage_log(session, stage):
                            result = run_experiment(config=task_config)
                            directory = Path(result["output_dir"])
                            required = ["best.pth", "last.pth", "history.csv", "full_classifier_initialization.json",
                                "config.json", "data_manifest.json", "summary.json", "thresholds.json", "threshold_comparison.csv"]
                            required += [f"{split}/{name}" for split in ("validation", "validation_at_0.5", "test", "test_at_0.5")
                                         for name in ("metrics.json", "per_class.csv", "predictions.csv")]
                            if any(not (directory / name).is_file() for name in required):
                                raise RuntimeError("Incomplete ASL outputs")
                        summary["completed"][stage] = {"output_dir": result["output_dir"], "summary": result["summary"]}
                        summary["experiments"][stage] = {"status": "complete", "output_dir": result["output_dir"]}
                        passed(stage)
                    except Exception as exc:
                        summary["experiments"][stage] = {"status": "failed", "error": str(exc)}
                        raise
                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()
        if settings.get("RUN_SSL_FINETUNE_RESUME", False):
            from .resume_extension import resolve_completed_runs, config_from_completed_run
            directories = resolve_completed_runs(settings)
            for task, directory in directories.items():
                stage = f"ssl_finetune_{task}_completed_resume"
                summary["stage"] = stage
                save_summary(summary_path, summary)
                resumed_config = config_from_completed_run(directory, settings["SSL_FINETUNE_RESUME_TO_EPOCHS"],
                    session / "ssl_finetuning" / "resumed")
                if resumed_config.task != task:
                    raise ValueError("Completed resume source task mismatch")
                with stage_log(session, stage):
                    result = run_experiment(config=resumed_config, resume_completed_from=directory,
                        scheduler_policy=settings["SSL_FINETUNE_RESUME_SCHEDULER_POLICY"])
                summary["completed"][stage] = {"output_dir": result["output_dir"], "summary": result["summary"]}
                summary["experiments"][stage] = dict(status="complete", output_dir=result["output_dir"])
                passed(stage)
                gc.collect()
                torch.cuda.empty_cache()
        if ft_requested:
            stage = "SSL fine-tuning source validation"
            summary["stage"] = stage
            full_tasks = [task for task in ("superclass", "subclass") if settings.get(f"RUN_SSL_FINETUNE_{task.upper()}", False)]
            provenance_only = settings.get("SSL_FINETUNE_PROVENANCE_ONLY", False)
            if provenance_only and full_tasks:
                raise ValueError("SSL_FINETUNE_PROVENANCE_ONLY requires both full fine-tuning flags False")
            for task in ("superclass", "subclass"):
                summary["experiments"][f"ssl_finetune_{task}"] = {"status": "pending" if task in full_tasks else "disabled_by_configuration"}
            if not settings.get("RUN_SSL_FINETUNE_SMOKE", False):
                for pending in full_tasks:
                    summary["experiments"][f"ssl_finetune_{pending}"] = {"status": "blocked_missing_smoke"}
                raise ValueError("Full SSL fine-tuning requires RUN_SSL_FINETUNE_SMOKE=True in the same execution")
            path, checksum = settings.get("SSL_FINETUNE_ENCODER_PATH", ""), settings.get("SSL_FINETUNE_ENCODER_SHA256", "")
            if not path or not checksum:
                raise ValueError("Set SSL_FINETUNE_ENCODER_PATH and independently verified SSL_FINETUNE_ENCODER_SHA256; no fallback to fresh/baseline weights")
            from .finetuning import load_completed_ssl_export
            with stage_log(session, stage):
                source, proof = load_completed_ssl_export(path, checksum)
                del source
                write_json(session / "ssl_finetune_source.json", proof)
            ft_config = replace(config, ssl_encoder_checkpoint=path, ssl_encoder_sha256=checksum,
                                ssl_reference_cache=settings.get("SSL_FINETUNE_REFERENCE_CACHE") or None,
                                ssl_normalizer_policy=settings.get("SSL_FINETUNE_NORMALIZER_POLICY", "strict"),
                                ssl_data_mode=settings.get("SSL_FINETUNE_DATA_MODE", "independent"),
                                **settings.get("SSL_FINETUNE_SETTINGS", {}))
            # Both actual GPU smokes must finish before either full task starts.
            stages = [(task, True) for task in ("superclass", "subclass")] + [(task, False) for task in full_tasks]
            for task, smoke in stages:
                key = f"ssl_finetune_{task}" + ("_provenance_audit" if provenance_only else ("_smoke" if smoke else ""))
                stage = key
                summary["stage"] = stage
                summary["experiments"][key] = {"status": "running"}
                save_summary(summary_path, summary)
                task_config = replace(ft_config, task=task, diagnostic=smoke,
                    epochs=1 if smoke else ft_config.epochs,
                    output_root=str(session / "ssl_finetuning" / ("provenance_audit" if provenance_only else ("smoke" if smoke else "full"))))
                try:
                    with stage_log(session, stage):
                        if provenance_only:
                            from .finetuning import run_provenance_audit
                            result = run_provenance_audit(task_config)
                        else:
                            result = run_experiment(config=task_config)
                        directory = Path(result["output_dir"])
                        required = ["ssl_initialization.json", "ssl_provenance_comparison.json", "best.pth", "last.pth", "history.csv", "summary.json",
                                    "config.json", "data_manifest.json", "numerical_settings.json", "figures/loss_curves.png"]
                        required += ["ssl_smoke_gradients.json"] if smoke else ["thresholds.json", "validation/predictions.csv", "test/metrics.json", "test/predictions.csv"]
                        if provenance_only:
                            required = ["ssl_provenance_comparison.json", "config.json", "data_manifest.json", "summary.json", "status.json"]
                        missing = [name for name in required if not (directory / name).is_file()]
                        if missing:
                            raise RuntimeError(f"Incomplete SSL fine-tuning exports: {missing}")
                    summary["completed"][key] = {"output_dir": result["output_dir"], "summary": result["summary"]}
                    summary["experiments"][key] = dict(status="complete", output_dir=result["output_dir"], benchmark_result=not smoke)
                    passed(key)
                except Exception as exc:
                    summary["experiments"][key] = {"status": "failed", "error": str(exc)}
                    if provenance_only:
                        summary["errors"][key] = str(exc)
                        save_summary(summary_path, summary)
                        continue  # audit both tasks; never initialize a fine-tuning model
                    if smoke:
                        for pending in full_tasks:
                            summary["experiments"][f"ssl_finetune_{pending}"] = {"status": "blocked_by_smoke_failure"}
                    else:
                        for pending in full_tasks:
                            if summary["experiments"][f"ssl_finetune_{pending}"]["status"] == "pending":
                                summary["experiments"][f"ssl_finetune_{pending}"] = {"status": "not_run_after_task_failure"}
                    raise
                finally:
                    gc.collect()
                    if not cpu_only:
                        torch.cuda.empty_cache()
        enabled = [task for task, flag in (("superclass", "RUN_SUPERCLASS"), ("subclass", "RUN_SUBCLASS")) if settings[flag]]
        initial_checkpoints = dict(settings.get("INITIAL_CHECKPOINTS", {}))
        if enabled and settings.get("CONTINUATION_SESSION_ID"):
            if any(initial_checkpoints.get(task) or settings["RESUME_RUN_DIRS"].get(task) for task in enabled):
                raise ValueError("Select automatic CONTINUATION_SESSION_ID or explicit checkpoint/resume paths, not both")
            initial_checkpoints.update(resolve_continuation_checkpoints(settings["CONTINUATION_SESSION_ID"],
                settings.get("CONTINUATION_ARTIFACT_ROOT", ""), tasks=enabled,
                expected_sha256=settings.get("CONTINUATION_CHECKPOINT_SHA256")))
            summary["continuation_checkpoints"] = initial_checkpoints
        for task in ("superclass", "subclass"):
            summary["experiments"][task] = {"status": "pending" if task in enabled else "skipped_by_configuration"}
        save_summary(summary_path, summary)
        for task in enabled:
            override = settings[f"{task.upper()}_EPOCHS"]
            task_config = replace(config, task=task, epochs=override if override is not None else settings["FULL_EPOCHS"],
                                  initial_checkpoint=initial_checkpoints.get(task) or None)
            stage = f"{task} data cache / training / evaluation"
            summary["stage"] = stage
            summary["experiments"][task] = {"status": "running"}
            save_summary(summary_path, summary)
            try:
                with stage_log(session, stage):
                    data = load_ptbxl(task, config=data_config, seed=config.seed, diagnostic_limits=diagnostic_limits(task_config))
                    write_json(session / "normalization.json", {"statistics": data.normalizer.state_dict(),
                               "configuration": data.provenance["normalization_configuration"]})
                    run_dir = settings["RESUME_RUN_DIRS"][task] or None
                    if run_dir:
                        if task_config.initial_checkpoint:
                            raise ValueError("Choose RESUME_RUN_DIRS or INITIAL_CHECKPOINTS, not both for the same task")
                        saved = json.loads((Path(run_dir) / "config.json").read_text())
                        candidate = asdict(task_config)
                        candidate["output_root"] = saved["output_root"]
                        candidate["initial_checkpoint"] = saved.get("initial_checkpoint")
                        if not checkpoint_config_matches(saved, candidate):
                            raise ValueError("Exact resume requires original settings. To change AMP/LR, use INITIAL_CHECKPOINTS for a NEW weights-only run")
                        task_config = replace(task_config, output_root=saved["output_root"], initial_checkpoint=saved.get("initial_checkpoint"))
                    result = run_experiment(config=task_config, prepared_data=data, run_dir=run_dir,
                        resume_from=str(Path(run_dir) / "last.pth") if run_dir else None)
                    directory = Path(result["output_dir"])
                    required = ["best.pth", "last.pth", "config.json", "environment.json", "data_manifest.json",
                                "class_statistics.csv", "history.csv", "summary.json", "run.log", "status.json",
                                "figures/loss_curves.png", "figures/confusion_matrices.png"]
                    splits = ("diagnostic_validation",) if config.diagnostic else ("validation", "test", "test_at_0.5")
                    required += [f"{split}/{file}" for split in splits for file in
                                 ("metrics.json", "per_class.csv", "confusion_matrices.csv", "predictions.csv")]
                    if not config.diagnostic:
                        required += ["thresholds.json"]
                    missing = [name for name in required if not (directory / name).is_file()]
                    if missing:
                        raise RuntimeError(f"Incomplete {task} exports: {missing}")
                    summary["completed"][task] = {"output_dir": result["output_dir"], "summary": result["summary"]}
                    summary["experiments"][task] = {"status": "complete", "output_dir": result["output_dir"],
                                                    "benchmark_result": result["summary"]["benchmark_result"]}
                    passed(f"{task}_exports")
            except Exception as exc:
                summary["experiments"][task] = {"status": "failed", "error": str(exc)}
                summary["errors"][task] = str(exc)
                save_summary(summary_path, summary)
                if not settings["CONTINUE_OTHER_TASK_ON_FAILURE"]:
                    raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        summary["status"] = "failed" if summary["errors"] else "complete"
        summary["stage"] = "finished"
    except Exception as exc:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["stage"] = stage
        summary["errors"].setdefault(stage, str(exc))
    finally:
        save_summary(summary_path, summary)
        save_summary(session / "pipeline_status.json", dict(status=summary["status"], stage=summary["stage"]))
        print("WORKER SUMMARY:", json.dumps({"status": summary["status"], "checks": summary["checks"],
              "experiments": summary["experiments"], "results": str(session)}, indent=2), flush=True)
    return 0 if summary["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    return run_worker(settings) if args.worker else controller(settings, path)


if __name__ == "__main__":
    sys.exit(main())
