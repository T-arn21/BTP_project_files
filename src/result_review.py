"""CPU-only re-evaluation of exported predictions; never performs inference/training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .cache import digest_file
from .evaluation import export_evaluation, evaluate, plot_results, tune_thresholds
from .utils import write_json


def review_experiment(source, output):
    """Regenerate fixed/tuned metrics with exact recorded split IDs and vocabulary.

    Outputs are explicitly prediction re-analysis, not a reproduced benchmark or
    independent verification of raw data, patient metadata or model inference.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Review outputs must be separate from the source experiment")
    prediction_paths = {}
    for label in ("validation", "test"):
        candidates = [source / label / "predictions.csv"]
        if label == "test":
            # Observed in the downloaded historical epoch-15 subclass export.
            candidates.append(source / label / "test_predictions.csv")
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            raise ValueError(f"Ambiguous {label} prediction exports: {existing}")
        prediction_paths[label] = existing[0] if existing else candidates[0]
    files = [source / name for name in ("config.json", "summary.json", "status.json", "data_manifest.json",
        "history.csv", "thresholds.json")] + list(prediction_paths.values())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"Completed classification export required; missing={missing}")
    config, summary, status, manifest = [json.loads(path.read_text()) for path in files[:4]]
    thresholds = json.loads(files[5].read_text())
    names = manifest["class_names"]
    if (config.get("diagnostic") is not False or summary.get("benchmark_result") is not True or
            status.get("status") != "complete" or thresholds.get("source_split") != "val" or
            thresholds.get("class_names") != names or thresholds.get("best_epoch") != summary.get("best_epoch")):
        raise ValueError("Export completion, vocabulary or validation-threshold provenance mismatch")
    splits = manifest["splits"]
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Recorded ECG splits overlap")
    arrays = {}
    for label, key in (("validation", "val"), ("test", "test")):
        frame = pd.read_csv(prediction_paths[label], float_precision="round_trip")
        expected = ["ecg_id"] + [f"{name}_{suffix}" for name in names for suffix in ("true", "prob", "pred")]
        if frame.columns.tolist() != expected or frame.ecg_id.tolist() != splits[key] or frame.ecg_id.duplicated().any():
            raise ValueError(f"{label} prediction schema, class order or record IDs differ from manifest")
        arrays[label] = (frame[[f"{name}_true" for name in names]].to_numpy(),
            frame[[f"{name}_prob" for name in names]].to_numpy(dtype=np.float32), frame.ecg_id.to_numpy())
    y, p, _ = arrays["validation"]
    selected = tune_thresholds(y, p, split="val")
    if not np.array_equal(selected, np.asarray(thresholds["values"])):
        raise ValueError("Saved thresholds do not reproduce from the validation predictions")
    # Validate both arrays before creating outputs, and never overwrite a prior review.
    for y, p, _ in arrays.values():
        evaluate(y, p, names, selected)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, (y, p, ids) in arrays.items():
        for policy, values in (("fixed_0.5", .5), ("validation_tuned", selected)):
            metrics = export_evaluation(output / label / policy, y, p, ids, names, values)
            rows.append(dict(split=label, threshold_policy=policy, **metrics))
    table = evaluate(*arrays["test"][:2], names, selected)[1]
    plot_results(pd.read_csv(source / "history.csv"), table, output / "figures",
                 loss_label="ASL loss" if config.get("loss_type", "bce") == "asl" else "BCE loss")
    pd.DataFrame(rows).to_csv(output / "threshold_comparison.csv", index=False)
    proof = dict(kind="exported_prediction_reanalysis", source=str(source), output=str(output),
        model_inference_reproduced=False, raw_data_patient_audit_repeated=False,
        prediction_files={label: path.relative_to(source).as_posix() for label, path in prediction_paths.items()},
        source_sha256={path.relative_to(source).as_posix(): digest_file(path) for path in files})
    write_json(output / "review_provenance.json", proof)
    return proof
