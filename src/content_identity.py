"""Full training-population equivalence against a checkpoint-authenticated cache.

This verifies deterministic pre-normalization waveform content, not original WFDB
file byte identity. Normalization, splits and implementation remain separate gates.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from .cache import digest_file, ensure_signal_cache
from .data import PTBXLDataset
from .signal import SignalNormalizer
from .utils import log_progress


def verify_training_content(state, data, identity, reference_path):
    """Never trust an external manifest/checksum unless bound to the pinned encoder."""
    proof = dict(schema=1, verified=False, method="full_training_processed_float32_bytes_v1",
                 reference_path=str(reference_path), checked_records=0,
                 scope="all official SSL training IDs, including unlabeled ECGs; no held-out waveforms")
    try:
        source = state["provenance"]
        ids = sorted(map(int, source["eligible_train_ids"]))
        if not ids or len(set(ids)) != len(ids) or sorted(source["used_train_ids"]) != ids:
            raise ValueError("Reference must cover the complete unique SSL training population")
        path = Path(reference_path)
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        pin = source["signal_cache"]["payload_sha256"]
        if (path.name != "signals.npy" or manifest["ids"] != ids or
                manifest["identity"] != source["preprocessing_identity"] or manifest["payload_sha256"] != pin):
            raise ValueError("Reference manifest does not match authenticated SSL identity, IDs or payload pin")
        log_progress(f"Authenticating full SSL reference cache: {path}")
        if digest_file(path) != pin:
            raise ValueError("Reference cache SHA256 differs from pinned encoder provenance")
        reference = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = (len(ids), 12, data.config.sampling_rate * data.config.duration)
        if reference.shape != shape or reference.dtype != np.dtype("float32"):
            raise ValueError("Reference cache shape/dtype mismatch")
        # Independent deterministic dataset: never reuse augmented/normalized samples
        # or the diagnostic subset. Cache creation reads every missing training ECG.
        processed = PTBXLDataset(data.metadata, pd.DataFrame(index=data.metadata.index), ids,
                                 data.config, SignalNormalizer(), "val", data.datasets["train"].reader)
        live_path, live_manifest = ensure_signal_cache(identity, ids, data.config, processed.processed_signal)
        current = np.load(live_path, mmap_mode="r", allow_pickle=False)
        expected_hash, actual_hash = hashlib.sha256(), hashlib.sha256()
        header = json.dumps(dict(schema=1, ids=ids, shape=list(shape), dtype="<f4"), sort_keys=True).encode()
        expected_hash.update(header)
        actual_hash.update(header)
        start = last_log = perf_counter()
        for row, ecg_id in enumerate(ids):
            left, right = reference[row], current[row]
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f"Nonfinite content at ECG {ecg_id}")
            a = np.asarray(left, dtype="<f4", order="C").tobytes()
            b = np.asarray(right, dtype="<f4", order="C").tobytes()
            expected_hash.update(a)
            actual_hash.update(b)
            if a != b:
                proof["first_mismatched_ecg_id"] = ecg_id
                raise ValueError(f"Waveform content mismatch at ECG {ecg_id}")
            proof["checked_records"] = row + 1
            if row == 0 or (row + 1) % 100 == 0 or perf_counter() - last_log >= 10:
                log_progress(f"Content equivalence {row + 1}/{len(ids)} training ECGs; elapsed {perf_counter() - start:.1f}s")
                last_log = perf_counter()
        proof.update(verified=True, expected_content_sha256=expected_hash.hexdigest(),
                     actual_content_sha256=actual_hash.hexdigest(), training_ids=ids,
                     reference_payload_sha256=pin, current_payload_sha256=live_manifest["payload_sha256"],
                     current_cache_path=live_path, expected_identity=source["preprocessing_identity"], actual_identity=identity)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        proof["error"] = str(exc)
    return proof
