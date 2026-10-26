"""Official diagnostic mappings; vocabulary never depends on observed record counts."""
import ast
import numpy as np
import pandas as pd

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def build_labels(metadata, statements, task="superclass"):
    if task not in {"superclass", "subclass"}:
        raise ValueError("task must be superclass or subclass")
    column = "diagnostic_class" if task == "superclass" else "diagnostic_subclass"
    diagnostic = statements.loc[statements["diagnostic"].eq(1)]
    if diagnostic.index.has_duplicates or diagnostic[column].isna().any():
        raise ValueError("Invalid official diagnostic mapping")
    mapping = diagnostic[column].to_dict()
    classes = SUPERCLASSES if task == "superclass" else tuple(sorted(set(mapping.values())))
    if set(mapping.values()) != set(classes) or len(classes) != (5 if task == "superclass" else 23):
        raise ValueError(f"Expected official {task} vocabulary; got {classes}")
    positions = {c: j for j, c in enumerate(classes)}
    matrix = np.zeros((len(metadata), len(classes)), dtype=np.float32)
    for i, codes in enumerate(metadata["scp_codes"]):
        codes = ast.literal_eval(codes) if isinstance(codes, str) else codes
        if not isinstance(codes, dict):
            raise ValueError("scp_codes must be a dictionary")
        # Presence, including confidence zero, matches the original PTB-XL aggregation.
        for code in codes:
            if code in mapping:
                matrix[i, positions[mapping[code]]] = 1
    return pd.DataFrame(matrix, index=metadata.index, columns=classes)


def class_statistics(labels, splits):
    rows = []
    for split, ids in splits.items():
        subset = labels.loc[ids]
        for cls in labels.columns:
            positive = int(subset[cls].sum())
            negative = len(subset) - positive
            rows.append(dict(split=split, **{"class": cls}, positive=positive,
                             negative=negative, prevalence=positive / len(subset),
                             negative_positive_ratio=negative / positive if positive else np.nan))
    return pd.DataFrame(rows)
