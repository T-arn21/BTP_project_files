"""Unattended controller and fresh scientific worker for the final Kaggle notebook."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from .kaggle_setup import stage_log, setup_dependencies, run_logged


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
        summary["stage"] = "dependency setup and CUDA verification"
        save_summary(summary_path, summary)
        with stage_log(session, "dependency setup and CUDA verification"):
            setup = setup_dependencies(settings["CHECKOUT_DIR"], session,
                       wheel_dir=settings["WHEEL_DIR"], allow_source_build=settings["ALLOW_SOURCE_BUILD"],
                       install_base=settings["INSTALL_BASE"], max_jobs=settings["MAX_JOBS"])
            save_summary(session / "setup_report.json", setup)
            summary["checks"]["dependencies_cuda"] = "passed"
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
        with stage_log(session, stage):
            import src
            checkout = Path(settings["CHECKOUT_DIR"]).resolve()
            if Path(src.__file__).resolve().parent != checkout / "src":
                raise RuntimeError("Worker imported source outside the synchronized checkout")
            import gc
            from dataclasses import asdict, replace
            import torch
            from .config import DataConfig, ModelConfig, ExperimentConfig
            from .data import load_ptbxl
            from .kaggle import find_ptbxl_root, audit_metadata, prepare_smoke_data, require_cuda, smoke_forward_backward
            from .models.registry import create_model
            from .experiments import run_experiment, diagnostic_limits
            from .utils import environment_info, write_json, seed_everything, log_progress
            for name, module in list(sys.modules.items()):
                if name.startswith("src.") and getattr(module, "__file__", None):
                    if not Path(module.__file__).resolve().is_relative_to(checkout / "src"):
                        raise RuntimeError("A worker module resolved outside the synchronized checkout")
            require_cuda()
            write_json(session / "environment.json", environment_info())
            passed("synchronized_imports")
        stage = "dataset discovery and patient metadata audit"
        summary["stage"] = stage
        save_summary(summary_path, summary)
        with stage_log(session, stage):
            root = find_ptbxl_root(settings["DATASET_ROOT"] or None)
            audit = audit_metadata(root, dataset_version=settings["DATASET_VERSION"])
            summary.update(dataset_root=str(root), declared_dataset_version=settings["DATASET_VERSION"])
            if settings["RUN_METADATA_AUDIT"]:
                write_json(session / "metadata_audit.json", audit["summary"])
                for task, table in audit["statistics"].items():
                    table.to_csv(session / f"{task}_class_statistics.csv", index=False)
            passed("metadata_patient_audit")  # Integrity is never optional for training.
            data_config = DataConfig(data_dir=str(root), sampling_rate=100, duration=10,
                dataset_version=settings["DATASET_VERSION"], num_workers=settings["NUM_WORKERS"],
                normalization_mode=settings["NORMALIZATION_MODE"], normalization_samples=settings["NORMALIZATION_SAMPLES"],
                normalization_seed=settings["NORMALIZATION_SEED"], cache_dir=settings["CACHE_DIR"],
                cache_signals=settings["CACHE_SIGNALS"], apply_filter=settings["APPLY_FILTER"], augment=settings["AUGMENT"])
            config = ExperimentConfig(epochs=settings["FULL_EPOCHS"], batch_size=settings["FULL_BATCH_SIZE"],
                device="cuda", seed=settings["SEED"], mixed_precision=settings["MIXED_PRECISION"],
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
                        report = smoke_forward_backward(model, batch, device="cuda", mixed_precision=config.mixed_precision)
                        write_json(session / f"{task}_smoke.json", report)
                        passed(f"{task}_gpu_smoke")
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
        else:
            summary["checks"]["real_data_gpu_smoke"] = "skipped_by_configuration"
        enabled = [task for task, flag in (("superclass", "RUN_SUPERCLASS"), ("subclass", "RUN_SUBCLASS")) if settings[flag]]
        for task in ("superclass", "subclass"):
            summary["experiments"][task] = {"status": "pending" if task in enabled else "skipped_by_configuration"}
        save_summary(summary_path, summary)
        for task in enabled:
            override = settings[f"{task.upper()}_EPOCHS"]
            task_config = replace(config, task=task, epochs=override if override is not None else settings["FULL_EPOCHS"])
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
                        saved = json.loads((Path(run_dir) / "config.json").read_text())
                        candidate = asdict(task_config)
                        candidate["output_root"] = saved["output_root"]
                        if candidate != saved:
                            raise ValueError("Resume configuration differs; match original budget/data/model settings")
                        task_config = replace(task_config, output_root=saved["output_root"])
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
