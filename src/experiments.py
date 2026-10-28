"""Common full-benchmark and explicitly separated diagnostic orchestration."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from time import perf_counter
import sys
import traceback
import torch
from .config import ExperimentConfig, checkpoint_config_matches
from .data import load_ptbxl
from .models.registry import create_model
from .training import Trainer, make_pos_weight, load_checkpoint
from .continuation import verify_initial_provenance
from .cache import digest_file
from .evaluation import predict, tune_thresholds, evaluate, export_evaluation, plot_results
from .utils import seed_everything, write_json, environment_info, log_progress


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.console.flush()
        self.log.flush()


def diagnostic_limits(config):
    if not config.diagnostic:
        return None
    return {"train": config.diagnostic_batches * config.batch_size,
            "val": config.diagnostic_batches * config.batch_size}


def run_experiment(model_name=None, task=None, epochs=None, batch_size=None,
                   data_dir=None, config=None, *, run_dir=None, resume_from=None,
                   prepared_data=None):
    overrides = {k: v for k, v in dict(model_name=model_name, task=task, epochs=epochs,
                                     batch_size=batch_size).items() if v is not None}
    cfg = replace(config or ExperimentConfig(), **overrides)
    if data_dir:
        cfg.data = replace(cfg.data, data_dir=str(data_dir))
    if resume_from and run_dir is None:
        raise ValueError("Pass the original run_dir when resuming")
    purpose = "diagnostic" if cfg.diagnostic else "benchmark"
    if run_dir is None:
        run_dir = Path(cfg.output_root) / purpose / f"{cfg.model_name}_{cfg.task}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=bool(resume_from))
    stage = "data preparation"
    with (out / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                seed_everything(cfg.seed)
                log_progress(f"Starting {purpose} {cfg.task}; output={out.resolve()}")
                write_json(out / "status.json", dict(status="running", stage=stage, purpose=purpose, benchmark_result=False))
                if prepared_data is None:
                    data = load_ptbxl(cfg.task, data_dir=data_dir, config=cfg.data,
                                     seed=cfg.seed, diagnostic_limits=diagnostic_limits(cfg))
                else:
                    data = prepared_data
                    expected = replace(cfg.data, data_dir=str(Path(cfg.data.data_dir).resolve()))
                    if asdict(expected) != asdict(data.config) or data.seed != cfg.seed or len(data.class_names) != (5 if cfg.task == "superclass" else 23):
                        raise ValueError("Prepared data does not match experiment configuration/task")
                    if data.provenance["diagnostic_only"] != cfg.diagnostic:
                        raise ValueError("Cannot use diagnostic subsets as benchmark data")
                    if data.provenance["diagnostic_limits"] != diagnostic_limits(cfg):
                        raise ValueError("Prepared diagnostic record limits do not match the requested batch limits")
                cfg.data = data.config
                metadata = {"task": cfg.task, "normalizer": data.normalizer.state_dict(),
                            "provenance": data.provenance,
                            "splits": {k: list(map(int, ids)) for k, ids in data.splits.items()}}
                stage = "model and DataLoader initialization"
                log_progress(stage)
                model = create_model(cfg.model_name, len(data.class_names), cfg.model)
                loaders = data.loaders(cfg.batch_size)
                pos_weight = make_pos_weight(data.labels.loc[data.splits["train"]].to_numpy(), cfg.pos_weight_max) if cfg.use_pos_weight else None
                if cfg.initial_checkpoint and not resume_from:
                    log_progress("Loading explicitly selected finite checkpoint weights into a NEW run; optimizer/scheduler/scaler reset")
                    try:
                        initial = load_checkpoint(cfg.initial_checkpoint, model, data.class_names, "cpu")
                        verification = verify_initial_provenance(initial, cfg, metadata, data.class_names)
                    except ValueError as exc:
                        write_json(out / "checkpoint_failure.json", {"checkpoint": cfg.initial_checkpoint, "error": str(exc)})
                        raise
                    write_json(out / "initial_checkpoint.json", dict(path=cfg.initial_checkpoint,
                               epoch=initial.get("epoch"), mode="weights_only_new_optimizer",
                               sha256=digest_file(cfg.initial_checkpoint), provenance_check=verification,
                               restored_states=["model"], reset_states=["optimizer", "scheduler", "scaler", "rng", "early_stopping"],
                               original_training_config=initial.get("config"), new_training_config=asdict(cfg)))
                    del initial
                if resume_from:
                    previous = torch.load(resume_from, map_location="cpu", weights_only=True)
                    if not checkpoint_config_matches(previous.get("config", {}), cfg) or previous.get("metadata") != metadata or previous.get("class_names") != data.class_names:
                        raise ValueError("Exact resume requires identical training settings, classes and preprocessing. Use initial_checkpoint for an explicit NEW weights-only run when changing AMP/LR")
                    if not (out / "best.pth").is_file():
                        raise ValueError("Resume requires the original run directory and best.pth")
                    del previous
                trainer = Trainer(model, cfg, data.class_names, out, pos_weight, metadata)
                write_json(out / "config.json", asdict(cfg))
                write_json(out / "environment.json", environment_info())
                write_json(out / "data_manifest.json", {**metadata, "class_names": data.class_names,
                                                       "excluded_unlabeled_ids": data.dropped_ids})
                data.statistics.to_csv(out / "class_statistics.csv", index=False)
                stage = "training"
                write_json(out / "status.json", dict(status="running", stage=stage, purpose=purpose, benchmark_result=False))
                start = perf_counter()
                history = trainer.fit(loaders["train"], loaders["val"], resume_from)
                training_seconds = perf_counter() - start
                stage = "validation and evaluation exports"
                state = load_checkpoint(out / "best.pth", model, data.class_names, "cpu")
                best_epoch = state["epoch"]
                del state  # never retain duplicate model/optimizer checkpoint tensors on the GPU
                yv, pv, iv, val_seconds = predict(model, loaders["val"], trainer.device, cfg.data.layout)
                if cfg.diagnostic:
                    metrics = export_evaluation(out / "diagnostic_validation", yv, pv, iv, data.class_names, 0.5)
                    _, table = evaluate(yv, pv, data.class_names, 0.5)
                    thresholds = None
                    test_seconds = None
                else:
                    thresholds = tune_thresholds(yv, pv, split="val")
                    write_json(out / "thresholds.json", {"source_split": "val", "class_names": data.class_names,
                                                        "values": thresholds.tolist(), "best_epoch": best_epoch})
                    export_evaluation(out / "validation", yv, pv, iv, data.class_names, thresholds)
                    stage = "held-out test evaluation and exports"
                    yt, pt, it, test_seconds = predict(model, loaders["test"], trainer.device, cfg.data.layout)
                    metrics = export_evaluation(out / "test", yt, pt, it, data.class_names, thresholds)
                    export_evaluation(out / "test_at_0.5", yt, pt, it, data.class_names, 0.5)
                    _, table = evaluate(yt, pt, data.class_names, thresholds)
                plot_results(history, table, out / "figures")
                summary = dict(model_name=cfg.model_name, task=cfg.task, best_epoch=best_epoch,
                               initialization="checkpoint_weights_only" if cfg.initial_checkpoint else "fresh",
                               source_checkpoint=cfg.initial_checkpoint,
                               purpose=purpose, benchmark_result=not cfg.diagnostic,
                               train_records=len(data.splits["train"]), validation_records=len(data.splits["val"]),
                               total_parameters=sum(p.numel() for p in model.parameters()),
                               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                               training_seconds_this_call=training_seconds,
                               total_epoch_seconds=float(history.epoch_seconds.sum()),
                               validation_inference_seconds=val_seconds, test_inference_seconds=test_seconds,
                               inference_timing_scope="full loader pass including I/O and device transfers", **metrics)
                write_json(out / "summary.json", summary)
                write_json(out / "status.json", dict(status="complete", stage="complete", purpose=purpose,
                                                     benchmark_result=not cfg.diagnostic))
                log_progress(f"Completed {purpose} {cfg.task}: {out.resolve()}")
                return {"output_dir": str(out.resolve()), "summary": summary, "history": history,
                        "class_names": data.class_names, "thresholds": thresholds}
            except Exception as exc:
                traceback.print_exc()
                write_json(out / "status.json", dict(status="failed", stage=stage, purpose=purpose,
                                                     benchmark_result=False, error=str(exc)))
                raise
