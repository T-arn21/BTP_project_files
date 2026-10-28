"""Task-independent deterministic float32 waveform cache and training-only statistics.

Each signal array is ~1.05 GB for 21,800 x 12 x 1000 records. It is memory
mapped, never fully stacked in RAM. Random augmentations are never cached.
"""
import hashlib
import json
import os
import shutil
import importlib.metadata
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from .signal import SignalNormalizer
from .utils import log_progress, write_json


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def preprocessing_identity(root, metadata, config, reader):
    """Metadata bytes, source location, raw-file stats and preprocessing implementation.

    File stats detect replaced waveforms; they are not a cryptographic audit of
    every source waveform. Cache payloads themselves are checked with SHA256.
    """
    start = perf_counter()
    root = Path(root).resolve()
    column = "filename_lr" if config.sampling_rate == 100 else "filename_hr"
    raw = hashlib.sha256()
    last_log = perf_counter()
    log_progress(f"Checking dataset source signatures for {len(metadata)} records")
    for record_number, name in enumerate(metadata[column], 1):
        for suffix in (".hea", ".dat"):
            path = root / f"{name}{suffix}"
            if path.is_file():
                stat = path.stat()
                raw.update(f"{name}{suffix}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            else:
                # Injected synthetic readers need no physical files; real reads
                # will fail with the exact ECG ID/path during cache creation.
                raw.update(f"{name}{suffix}:missing".encode())
        if record_number == 1 or record_number % 2000 == 0 or perf_counter() - last_log >= 10:
            log_progress(f"Source signatures {record_number}/{len(metadata)}; elapsed {perf_counter() - start:.1f}s")
            last_log = perf_counter()
    try:
        wfdb_version = importlib.metadata.version("wfdb")
    except importlib.metadata.PackageNotFoundError:
        wfdb_version = None
    identity = dict(schema=1, root=str(root), declared_version=config.dataset_version,
                    metadata_sha256={name: digest_file(root / name) for name in ("ptbxl_database.csv", "scp_statements.csv")},
                    raw_file_stats_sha256=raw.hexdigest(), sampling_rate=config.sampling_rate,
                    duration=config.duration, apply_filter=config.apply_filter, leads=12,
                    preprocessing_source_sha256=digest_file(Path(__file__).with_name("signal.py")),
                    numpy=np.__version__, scipy=scipy.__version__, wfdb=wfdb_version,
                    reader=f"{reader.__module__}.{reader.__qualname__}")
    log_progress(f"Dataset/preprocessing identity checked in {perf_counter() - start:.2f}s")
    return identity


def select_normalization_ids(train_ids, config):
    ids = np.array(sorted(map(int, train_ids)), dtype=np.int64)
    if config.normalization_mode == "sampled" and len(ids) > config.normalization_samples:
        ids = np.sort(np.random.default_rng(config.normalization_seed).choice(
            ids, config.normalization_samples, replace=False))
    return ids.tolist()


def ensure_signal_cache(identity, ids, config, process):
    """Return an immutable completed cache; partial/invalid files are rebuilt."""
    ids = sorted(map(int, ids))
    key = fingerprint({"identity": identity, "ids": ids})
    directory = Path(config.cache_dir) / f"signals_{key}"
    directory.mkdir(parents=True, exist_ok=True)
    data_path, manifest_path = directory / "signals.npy", directory / "manifest.json"
    shape = (len(ids), 12, config.sampling_rate * config.duration)
    start = perf_counter()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or manifest["ids"] != ids:
            raise ValueError("identity/record IDs mismatch")
        array = np.load(data_path, mmap_mode="r", allow_pickle=False)
        valid = tuple(array.shape) == shape and array.dtype == np.float32
        del array
        if not valid or data_path.stat().st_size != manifest["file_bytes"]:
            raise ValueError("shape/dtype/size mismatch")
        log_progress(f"Validating cached payload SHA256: {data_path}")
        if digest_file(data_path) != manifest["payload_sha256"]:
            raise ValueError("payload checksum mismatch")
        log_progress(f"Signal cache HIT: {len(ids)} ECGs, verified in {perf_counter() - start:.2f}s")
        return str(data_path), manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log_progress(f"Signal cache MISS/invalid ({exc}); building {len(ids)} deterministic ECGs")
    bytes_needed = int(np.prod(shape)) * 4 + 1024
    if shutil.disk_usage(directory).free < bytes_needed + 64 * 1024 * 1024:
        raise OSError(f"Insufficient cache disk space: need ~{bytes_needed / 1e9:.2f} GB at {directory}")
    partial = directory / "signals.partial.npy"
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    last_log = perf_counter()
    try:
        for row, ecg_id in enumerate(ids):
            x = process(ecg_id)
            if x.shape != shape[1:] or not np.isfinite(x).all():
                raise ValueError(f"Invalid deterministic waveform for ECG {ecg_id}: {x.shape}")
            array[row] = x
            if row == 0 or (row + 1) % 100 == 0 or perf_counter() - last_log >= 10:
                log_progress(f"Preprocessing cache {row + 1}/{len(ids)} ECGs; elapsed {perf_counter() - start:.1f}s; last ECG {ecg_id}")
                last_log = perf_counter()
        array.flush()
    finally:
        del array
    os.replace(partial, data_path)
    manifest = dict(identity=identity, ids=ids, shape=list(shape), dtype="float32",
                    file_bytes=data_path.stat().st_size, payload_sha256=digest_file(data_path),
                    creation_seconds=perf_counter() - start, content="filtered_center_padded_float32_unnormalized_no_augmentation")
    temp_manifest = directory / "manifest.partial.json"
    write_json(temp_manifest, manifest)
    os.replace(temp_manifest, manifest_path)  # completion marker written last
    log_progress(f"Signal cache ready in {perf_counter() - start:.2f}s ({bytes_needed / 1e9:.2f} GB)")
    return str(data_path), manifest


def get_normalizer(identity, ids, config, process):
    settings = dict(mode=config.normalization_mode, sample_size=config.normalization_samples,
                    seed=config.normalization_seed, record_ids=list(map(int, ids)))
    key = fingerprint({"identity": identity, "normalization": settings})
    directory = Path(config.cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"normalization_{key}.json"
    start = perf_counter()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["identity"] != identity or state["configuration"] != settings:
            raise ValueError("normalization cache mismatch")
        if state["statistics_checksum"] != fingerprint(state["statistics"]):
            raise ValueError("normalization statistics checksum mismatch")
        norm = SignalNormalizer.from_state_dict(state["statistics"])
        log_progress(f"Normalization cache HIT: {len(ids)} training ECGs ({perf_counter() - start:.2f}s)")
        return norm, state
    except (OSError, KeyError, ValueError, TypeError) as exc:
        log_progress(f"Fitting {config.normalization_mode} normalization on {len(ids)} training ECGs ({exc})")
    def signals():
        last_log = perf_counter()
        for n, ecg_id in enumerate(ids, 1):
            yield process(ecg_id)
            if n == 1 or n % 100 == 0 or perf_counter() - last_log >= 10:
                log_progress(f"Normalization {n}/{len(ids)}; elapsed {perf_counter() - start:.1f}s")
                last_log = perf_counter()
    norm = SignalNormalizer().fit(signals())
    state = dict(identity=identity, configuration=settings, statistics=norm.state_dict(),
                 statistics_checksum=fingerprint(norm.state_dict()),
                 fit_seconds=perf_counter() - start)
    temporary = path.with_suffix(".partial.json")
    write_json(temporary, state)
    os.replace(temporary, path)
    log_progress(f"Normalization saved in {perf_counter() - start:.2f}s: {path}")
    return norm, state
