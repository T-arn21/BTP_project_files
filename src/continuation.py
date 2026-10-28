"""Explicit weights-only continuation with dataset/preprocessing provenance checks."""
from dataclasses import asdict
from pathlib import Path
import hashlib


def resolve_continuation_checkpoints(session_id, artifact_root="", tasks=("superclass", "subclass"), expected_sha256=None):
    """Find a unique preserved session under attached inputs; never write to it."""
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("Continuation session ID must be a single directory name")
    root = Path(artifact_root or "/kaggle/input")
    if not root.is_dir():
        raise FileNotFoundError(f"Continuation artifact root missing: {root}")
    sessions = ([root] if root.name == session_id else []) + list(root.rglob(session_id))
    sessions = sorted({p.resolve() for p in sessions if (p / "experiments/benchmark").is_dir()})
    if len(sessions) != 1:
        raise ValueError(f"Expected one attached continuation session {session_id} under {root}; found {len(sessions)}. Set CONTINUATION_ARTIFACT_ROOT explicitly")
    result = {}
    for task in tasks:
        paths = sorted(sessions[0].glob(f"experiments/benchmark/mamba_{task}_*/last.pth"))
        if len(paths) != 1:
            raise ValueError(f"Expected one preserved {task} last.pth; found {len(paths)}")
        if expected_sha256 and expected_sha256.get(task):
            with paths[0].open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected_sha256[task]:
                raise ValueError(f"Preserved {task} checkpoint SHA256 mismatch; original checkpoint required")
        result[task] = str(paths[0])
    return result


def verify_initial_provenance(state, config, metadata, class_names):
    """Permit explicit training-setting changes, never unknown/mismatched data.

    Storage-only cache paths and diagnostic cache containers can differ. All
    source identities, normalization and exact full/prefix split IDs must match.
    """
    current = asdict(config)
    try:
        old, previous = state["config"], state["metadata"]
        if state["class_names"] != list(class_names):
            raise ValueError("Continuation class order mismatch")
        for key in ("model_name", "model", "task", "seed"):
            if old[key] != current[key]:
                raise ValueError(f"Continuation mismatch: {key}")
        # These fields control storage/loader execution only, not preprocessing.
        storage_fields = {"cache_dir", "num_workers"}
        before = {k: v for k, v in old["data"].items() if k not in storage_fields}
        after = {k: v for k, v in current["data"].items() if k not in storage_fields}
        if before != after:
            raise ValueError("Continuation data configuration/preprocessing mismatch")
        for key in ("task", "normalizer"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Continuation mismatch: {key}")
        original, active = previous["provenance"], metadata["provenance"]
        required = ("declared_dataset_version", "release_verified", "dataset_root", "metadata_sha256",
                    "records_before_exclusions", "normalization_record_ids", "normalization_configuration",
                    "preprocessing_identity")
        for key in required:
            if original[key] != active[key]:
                raise ValueError(f"Continuation provenance mismatch: {key}")
        # Missing identity components must not compare equal as empty dicts.
        for key in ("root", "metadata_sha256", "raw_file_stats_sha256", "preprocessing_source_sha256",
                    "sampling_rate", "duration", "apply_filter", "numpy", "scipy", "wfdb", "reader"):
            if key not in active["preprocessing_identity"]:
                raise ValueError(f"Continuation provenance missing identity field: {key}")
        source_splits = previous["splits"]
        if config.diagnostic:
            limits = {k: config.batch_size * config.diagnostic_batches for k in ("train", "val")}
            expected = {k: source_splits[k][:limit] for k, limit in limits.items()}
            if metadata["splits"] != expected or active["diagnostic_limits"] != limits or not active["diagnostic_only"]:
                raise ValueError("Continuation diagnostic split is not the exact declared source prefix")
        elif metadata["splits"] != source_splits or original["diagnostic_only"] or active["diagnostic_only"]:
            raise ValueError("Continuation full split provenance mismatch")
        elif original["signal_cache"] and active["signal_cache"]:
            if original["signal_cache"]["payload_sha256"] != active["signal_cache"]["payload_sha256"]:
                raise ValueError("Continuation full cache payload mismatch")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Checkpoint lacks required continuation provenance: {exc}") from exc
    return dict(status="passed", policy="identical_source_normalization_and_full_or_diagnostic_prefix_splits",
                diagnostic_prefix=config.diagnostic,
                cache_container_policy="verified independently by data loader; diagnostic subsets have different payloads",
                changed_settings={k: {"previous": old.get(k), "current": value}
                                  for k, value in current.items() if old.get(k) != value})
