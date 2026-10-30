"""One metric implementation for every architecture and diagnostic task."""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from .utils import write_json, log_progress


def validate_arrays(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if y.ndim != 2 or y.shape != p.shape or 0 in y.shape:
        raise ValueError("Expected nonempty matching (records, classes) arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Labels must be binary and probabilities finite in [0, 1]")
    return y.astype(int), p


def tune_thresholds(labels, probabilities, *, split="val", grid=None):
    """Maximize per-class F1 on validation only. Single-valued classes retain 0.5."""
    if split != "val":
        raise ValueError("Threshold selection requires validation data, never test data")
    y, p = validate_arrays(labels, probabilities)
    grid = np.linspace(0.05, 0.95, 19) if grid is None else np.asarray(grid)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("Threshold grid must contain finite values in [0,1]")
    thresholds = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            continue
        scores = [f1_score(y[:, j], p[:, j] >= t, zero_division=0) for t in grid]
        # Deterministic tie rule: prefer the threshold closest to 0.5, then lower.
        candidates = np.flatnonzero(np.isclose(scores, max(scores)))
        best = min(candidates, key=lambda k: (abs(grid[k] - 0.5), grid[k]))
        thresholds[j] = grid[best]
    return thresholds


def _ranking(y, p):
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "auprc": float(average_precision_score(y, p)) if y.sum() else np.nan}


def _counts(y, pred):
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
    return dict(tn=tn, fp=fp, fn=fn, tp=tp,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def evaluate(labels, probabilities, class_names, thresholds=0.5):
    y, p = validate_arrays(labels, probabilities)
    names = list(class_names)
    t = np.broadcast_to(np.asarray(thresholds), (y.shape[1],))
    if len(names) != y.shape[1] or len(set(names)) != len(names):
        raise ValueError("Class names must uniquely identify every probability column")
    if not np.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("Thresholds must be finite and in [0,1]")
    pred = p >= t
    table = pd.DataFrame([{"class": name, "support": int(y[:, j].sum()),
                           "negative_support": int((y[:, j] == 0).sum()), "threshold": float(t[j]),
                           **_ranking(y[:, j], p[:, j]), **_counts(y[:, j], pred[:, j])}
                          for j, name in enumerate(names)])
    overall = {f"macro_{key}": float(table[key].mean())
               for key in ("auroc", "auprc", "f1", "precision", "recall")}
    overall.update({f"micro_{key}": val for key, val in _ranking(y.ravel(), p.ravel()).items()})
    overall.update({f"micro_{key}": val for key, val in _counts(y.ravel(), pred.ravel()).items()
                    if key in ("f1", "precision", "recall")})
    overall.update(records=len(y), classes=len(names), valid_auroc_classes=int(table.auroc.notna().sum()),
                   valid_auprc_classes=int(table.auprc.notna().sum()))
    return overall, table


@torch.no_grad()
def predict(model, loader, device="cpu", layout="channels_first"):
    device = torch.device(device)
    model.eval()
    ys, ps, ids = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    log_progress(f"Inference starting: {len(loader)} batches")
    last_log = perf_counter()
    for batch_index, (x, y, ecg_ids) in enumerate(loader, 1):
        x = x.to(device)
        if layout == "time_first":
            x = x.transpose(1, 2).contiguous()
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError("Model must return (batch, classes) logits matching targets")
        ys.append(y.numpy())
        ps.append(logits.sigmoid().float().cpu().numpy())
        ids.append(np.asarray(ecg_ids))
        if batch_index == 1 or batch_index % 50 == 0 or perf_counter() - last_log >= 15:
            log_progress(f"Inference batch {batch_index}/{len(loader)}; elapsed={perf_counter() - start:.1f}s")
            last_log = perf_counter()
    if not ys:
        raise ValueError("Cannot predict an empty loader")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ids), perf_counter() - start


def export_evaluation(directory, labels, probabilities, ids, class_names, thresholds=0.5):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    overall, table = evaluate(labels, probabilities, class_names, thresholds)
    if len(ids) != len(labels):
        raise ValueError("Record IDs must align with predictions")
    write_json(out / "metrics.json", overall)
    table.to_csv(out / "per_class.csv", index=False)
    table[["class", "threshold", "tn", "fp", "fn", "tp"]].to_csv(out / "confusion_matrices.csv", index=False)
    frame = pd.DataFrame({"ecg_id": ids})
    for j, name in enumerate(class_names):
        frame[f"{name}_true"] = labels[:, j]
        frame[f"{name}_prob"] = probabilities[:, j]
        frame[f"{name}_pred"] = (probabilities[:, j] >= table.threshold.iloc[j]).astype(int)
    frame.to_csv(out / "predictions.csv", index=False)
    return overall


def plot_results(history, table, directory, loss_label="BCE loss"):
    import matplotlib.pyplot as plt
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("train_loss", "val_loss"):
        ax.plot(history["epoch"], history[column], label=column)
    ax.set(xlabel="Epoch", ylabel=loss_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)
    cols, rows = min(4, len(table)), (len(table) + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax, (_, row) in zip(axes.flat, table.iterrows()):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, str(count), ha="center", va="center")
        ax.set(title=f"{row['class']} (t={row.threshold:.2f})", xlabel="Predicted", ylabel="Actual",
               xticks=[0, 1], yticks=[0, 1])
    for ax in list(axes.flat)[len(table):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)
