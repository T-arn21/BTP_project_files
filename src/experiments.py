"""Public experiment interface used by 04_btp_experiments.ipynb."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from time import perf_counter
import torch
from .config import ExperimentConfig
from .data import load_ptbxl
from .models.registry import create_model
from .training import Trainer, make_pos_weight, load_checkpoint
from .evaluation import predict, tune_thresholds, evaluate, export_evaluation, plot_results
from .utils import seed_everything, write_json, environment_info


def run_experiment(model_name=None, task=None, epochs=None, batch_size=None,
                   data_dir=None, config=None, *, run_dir=None, resume_from=None):
    """Train one adapter, choose best checkpoint on validation, evaluate test once.

    Defaults: mamba, superclass, 20 epochs. Resume requires original run_dir,
    unchanged configuration and original best.pth. Dataset remains external.
    """
    overrides = {k: v for k, v in dict(model_name=model_name, task=task, epochs=epochs,
                                     batch_size=batch_size).items() if v is not None}
    cfg = replace(config or ExperimentConfig(), **overrides)
    if data_dir:
        cfg.data = replace(cfg.data, data_dir=str(data_dir))
    seed_everything(cfg.seed)
    if resume_from and run_dir is None:
        raise ValueError("Pass the original run_dir when resuming")
    if run_dir is None:
        run_dir = Path(cfg.output_root) / f"{cfg.model_name}_{cfg.task}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    out = Path(run_dir)
    if not resume_from:
        out.mkdir(parents=True, exist_ok=False)
    data = load_ptbxl(cfg.task, data_dir=data_dir, config=cfg.data, seed=cfg.seed)
    cfg.data = data.config  # freeze resolved env paths too
    loaders = data.loaders(cfg.batch_size)
    model = create_model(cfg.model_name, len(data.class_names), cfg.model)
    pos_weight = make_pos_weight(data.labels.loc[data.splits["train"]].to_numpy(), cfg.pos_weight_max) if cfg.use_pos_weight else None
    metadata = {"task": cfg.task, "normalizer": data.normalizer.state_dict(),
                "provenance": data.provenance,
                "splits": {k: list(map(int, ids)) for k, ids in data.splits.items()}}
    if resume_from:
        # Validate before replacing manifests in an existing run directory.
        previous = torch.load(resume_from, map_location="cpu", weights_only=True)
        if previous.get("config") != asdict(cfg) or previous.get("metadata") != metadata or previous.get("class_names") != data.class_names:
            raise ValueError("Resume requires identical config, classes and preprocessing/split provenance")
        if not (out / "best.pth").is_file():
            raise ValueError("Resume requires the original run directory and best.pth")
        del previous
    trainer = Trainer(model, cfg, data.class_names, out, pos_weight, metadata)
    write_json(out / "config.json", asdict(cfg))
    write_json(out / "environment.json", environment_info())
    write_json(out / "data_manifest.json", {**metadata, "class_names": data.class_names,
                                           "excluded_unlabeled_ids": data.dropped_ids})
    data.statistics.to_csv(out / "class_statistics.csv", index=False)
    start = perf_counter()
    history = trainer.fit(loaders["train"], loaders["val"], resume_from)
    training_seconds_this_call = perf_counter() - start
    state = load_checkpoint(out / "best.pth", model, data.class_names, trainer.device)
    yv, pv, iv, val_seconds = predict(model, loaders["val"], trainer.device, cfg.data.layout)
    thresholds = tune_thresholds(yv, pv, split="val")
    write_json(out / "thresholds.json", {"source_split": "val", "class_names": data.class_names,
                                        "values": thresholds.tolist(), "best_epoch": state["epoch"]})
    export_evaluation(out / "validation", yv, pv, iv, data.class_names, thresholds)
    yt, pt, it, test_seconds = predict(model, loaders["test"], trainer.device, cfg.data.layout)
    metrics = export_evaluation(out / "test", yt, pt, it, data.class_names, thresholds)
    export_evaluation(out / "test_at_0.5", yt, pt, it, data.class_names, 0.5)
    _, table = evaluate(yt, pt, data.class_names, thresholds)
    plot_results(history, table, out / "figures")
    summary = dict(model_name=cfg.model_name, task=cfg.task, best_epoch=state["epoch"],
                   total_parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   training_seconds_this_call=training_seconds_this_call,
                   total_epoch_seconds=float(history.epoch_seconds.sum()),
                   validation_inference_seconds=val_seconds, test_inference_seconds=test_seconds,
                   inference_timing_scope="full loader pass including I/O and device transfers",
                   **metrics)
    write_json(out / "summary.json", summary)
    return {"output_dir": str(out.resolve()), "summary": summary, "history": history,
            "class_names": data.class_names, "thresholds": thresholds}
