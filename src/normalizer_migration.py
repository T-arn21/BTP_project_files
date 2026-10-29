"""Evidence for explicit reuse of a pinned SSL normalizer after full content proof."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import scipy
from .cache import fingerprint
from .signal import SignalNormalizer
from .utils import log_progress


def numerical_runtime_info():
    """Informational CPU/SIMD/build evidence; never an acceptance exception."""
    evidence = dict(numpy=np.__version__, scipy=scipy.__version__)
    for name, callback in (("numpy_build", np.show_config), ("numpy_runtime", np.show_runtime),
                           ("scipy_build", scipy.show_config)):
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                callback()
            evidence[name] = buffer.getvalue()
        except Exception as exc:
            evidence[name] = f"Unavailable: {type(exc).__name__}: {exc}"
    return evidence


def state_difference(expected, actual):
    result = {}
    for key in ("mean", "std"):
        left, right = np.asarray(expected[key], dtype=np.float64), np.asarray(actual[key], dtype=np.float64)
        result[key] = dict(exact=expected[key] == actual[key],
                           max_absolute=float(np.max(np.abs(left - right))))
    return dict(exact=expected == actual, fields=result, expected=expected, actual=actual)


def audit_normalization(state, data, content):
    """No tolerances authorize migration; preserve original and fresh-fit values.

    The current statistics must reproduce exactly from the verified live fit IDs.
    This rejects corrupt/stale normalization caches, even if waveforms match.
    The historical values are recovered from the pinned encoder, never refitted.
    """
    evidence = dict(verified=False, current_fit_reproduced=False)
    try:
        if not content.get("reference_authenticated") or not content.get("current_cache_path"):
            raise ValueError("Authenticated reference and complete current cache required for diagnostic refits")
        source = state["provenance"]
        ids = list(map(int, source["normalization_record_ids"]))
        if (not ids or len(ids) != len(set(ids)) or not set(ids) <= set(content["training_ids"]) or
                source["normalization_configuration"]["record_ids"] != ids or
                data.provenance["normalization_record_ids"] != ids or
                data.provenance["normalization_configuration"] != source["normalization_configuration"]):
            raise ValueError("Normalization fit IDs/configuration must match authenticated training-only provenance")
        SignalNormalizer.from_state_dict(source["normalizer"])
        rows = {ecg_id: row for row, ecg_id in enumerate(content["training_ids"])}
        fits = {}
        paths = dict(reference=content["path_resolution"]["resolved_signals_path"], current=content["current_cache_path"])
        for name, path in paths.items():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            def signals():
                for n, ecg_id in enumerate(ids, 1):
                    x = np.array(array[rows[ecg_id]], copy=True)
                    if not np.isfinite(x).all():
                        raise ValueError(f"Nonfinite normalization input at ECG {ecg_id}")
                    yield x
                    if n == 1 or n % 100 == 0:
                        log_progress(f"Normalization reproducibility ({name}): {n}/{len(ids)} training ECGs")
            fits[name] = SignalNormalizer().fit(signals()).state_dict()
        current_state = data.normalizer.state_dict()
        identity = data.provenance["preprocessing_identity"]
        settings = source["normalization_configuration"]
        cache_path = Path(data.config.cache_dir) / f"normalization_{fingerprint({'identity': identity, 'normalization': settings})}.json"
        original_key = fingerprint({"identity": source["preprocessing_identity"], "normalization": settings})
        evidence.update(original=source["normalizer"], current=current_state,
            original_sha256=hashlib.sha256(json.dumps(source["normalizer"], sort_keys=True).encode()).hexdigest(),
            source_normalizer_origin="SHA256-pinned encoder provenance.normalizer; no re-estimation",
            fit_ids=ids, configuration=settings, accumulator_dtype="float64", input_dtype="float32",
            fit_order="authenticated normalization_record_ids in saved order; C-order cache rows",
            current_normalization_cache_path=str(cache_path),
            original_normalization_cache_key=original_key,
            current_normalization_cache_key=cache_path.stem.removeprefix("normalization_"),
            cache_key_scope="original SSL identity covers training folds; supervised identity covers all metadata folds; raw stats include filesystem mtimes",
            reference_refit_vs_original=state_difference(source["normalizer"], fits["reference"]),
            current_refit_vs_current=state_difference(current_state, fits["current"]),
            same_runtime_reference_vs_current=state_difference(fits["reference"], fits["current"]),
            current_fit_reproduced=current_state == fits["current"])
        evidence["content_equivalence_verified"] = content["verified"]
        evidence["verified"] = content["verified"] and evidence["current_fit_reproduced"] and fits["reference"] == fits["current"]
        if not evidence["verified"]:
            evidence["error"] = ("Processed waveform equivalence failed; diagnostic refits cannot authorize normalizer migration"
                if not content["verified"] else
                "Current normalization does not reproduce exactly from verified fit waveforms; reject stale/corrupt statistics or investigate fit implementation/layout")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        evidence["error"] = str(exc)
    return evidence


def activate_verified_ssl_data(state, data, report):
    """Bind training to the verified LIVE cache; optionally adopt pinned statistics."""
    migration = report.get("normalizer_migration", {})
    content = report.get("content_equivalence", {})
    if report["blocking_mismatch_count"] or not content.get("verified"):
        return
    # Use the actual current waveform payload just proven equal, rather than a
    # different diagnostic/all-fold cache or another raw preprocessing pass.
    train = data.datasets["train"]
    train.cache_path = content["current_cache_path"]
    train.cache_rows = {ecg_id: row for row, ecg_id in enumerate(content["training_ids"])}
    train._cached_signals = None
    data.provenance["ssl_verified_training_cache"] = dict(path=train.cache_path,
        payload_sha256=content["current_payload_sha256"], record_ids=content["training_ids"],
        source="current live preprocessing, complete byte equivalence to authenticated SSL cache")
    if migration.get("approved"):
        normalizer = SignalNormalizer.from_state_dict(state["provenance"]["normalizer"])
        data.normalizer = normalizer
        for dataset in data.datasets.values():
            dataset.normalizer = normalizer
        data.provenance["ssl_normalizer_migration"] = migration
