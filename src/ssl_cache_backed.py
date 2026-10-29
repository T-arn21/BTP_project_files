"""Explicit artifact-backed SSL fine-tuning; no live training-waveform reproduction."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .cache import digest_file, preprocessing_identity, select_normalization_ids, ensure_signal_cache
from .content_identity import resolve_reference_cache
from .signal import SignalNormalizer
from .utils import log_progress


def authenticate_cache(state, reference_path):
    """Authenticate manifest semantics and NPY bytes against the pinned export."""
    source = state["provenance"]
    ids = sorted(map(int, source["eligible_train_ids"]))
    if not ids or len(set(ids)) != len(ids) or sorted(source["used_train_ids"]) != ids:
        raise ValueError("SSL cache must cover exactly the complete original training population")
    path, resolution = resolve_reference_cache(reference_path, source)
    manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
    pin = source["signal_cache"]["payload_sha256"]
    if (manifest["ids"] != ids or manifest["identity"] != source["preprocessing_identity"] or
            manifest["payload_sha256"] != pin):
        raise ValueError("SSL cache manifest identity, IDs or payload pin differs from authenticated encoder")
    log_progress(f"Authenticating original SSL training cache SHA256: {path}")
    if digest_file(path) != pin:
        raise ValueError("Original SSL training cache SHA256 mismatch")
    shape = (len(ids), 12, state["data_config"]["sampling_rate"] * state["data_config"]["duration"])
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != np.dtype("float32"):
        raise ValueError("Original SSL cache shape/dtype mismatch")
    del array
    SignalNormalizer.from_state_dict(source["normalizer"])
    return dict(path=str(path.resolve()), manifest=manifest, resolution=resolution,
                payload_sha256=pin, authenticated=True)


def load_cache_backed_data(experiment, *, diagnostic_limits=None, reader=None):
    """Load pinned artifacts before data construction; never fit a new normalizer."""
    if experiment.ssl_data_mode != "ssl_cache_backed":
        raise ValueError("Original SSL artifact loading requires explicit ssl_data_mode='ssl_cache_backed'")
    from .finetuning import load_completed_ssl_export
    from .data import load_ptbxl, read_waveform
    state, proof = load_completed_ssl_export(experiment.ssl_encoder_checkpoint, experiment.ssl_encoder_sha256)
    source = dict(state=state, proof=proof, experiment=experiment,
                  cache=authenticate_cache(state, experiment.ssl_reference_cache))
    return load_ptbxl(experiment.task, config=experiment.data, seed=experiment.seed,
        diagnostic_limits=diagnostic_limits, reader=reader or read_waveform, ssl_cache_source=source)


def prepare_cache_backed_data(metadata, labels, splits, missing, config, seed, reader, diagnostic_limits, context):
    from .data import PTBXLDataset, PTBXLData, build_official_splits
    from .labels import class_statistics
    from .finetuning import compare_ssl_provenance
    from .provenance import ProvenanceMismatchError
    source, cache = context["state"]["provenance"], context["cache"]
    official = build_official_splits(metadata)
    fit_ids = select_normalization_ids(official["train"], config)
    normalizer = SignalNormalizer.from_state_dict(source["normalizer"])
    # Hash only held-out source-file stats. Training values come solely from the
    # authenticated artifact; no independent training reads or fit are performed.
    heldout_ids = sorted(set().union(*(set(ids) for name, ids in splits.items() if name != "train")))
    identity = preprocessing_identity(Path(config.data_dir), metadata.loc[heldout_ids], config, reader)
    normalization = dict(mode=config.normalization_mode, sample_size=config.normalization_samples,
                         seed=config.normalization_seed, record_ids=fit_ids)
    datasets = {name: PTBXLDataset(metadata, labels, ids, config, normalizer, name, reader, seed)
                for name, ids in splits.items()}
    datasets["train"].cache_path = cache["path"]
    datasets["train"].cache_rows = {ecg_id: row for row, ecg_id in enumerate(cache["manifest"]["ids"])}
    provenance = dict(declared_dataset_version=config.dataset_version, release_verified=False,
        dataset_root=config.data_dir, metadata_sha256=identity["metadata_sha256"], records_before_exclusions=len(metadata),
        normalization_record_ids=fit_ids, normalization_configuration=normalization,
        preprocessing_identity=identity, preprocessing_identity_scope="current held-out waveform source files only",
        signal_cache=None, diagnostic_only=diagnostic_limits is not None, diagnostic_limits=diagnostic_limits,
        ssl_cache_backed=dict(mode="ssl_cache_backed_finetuning", source_encoder_sha256=context["proof"]["sha256"],
            original_training_cache=cache, normalizer_source="pinned encoder provenance.normalizer",
            original_training_preprocessing_identity=source["preprocessing_identity"],
            independent_training_waveform_equivalence=dict(performed=False, verified=False,
                reason="Explicit artifact-backed experiment; live training waveforms are not used or claimed identical"),
            heldout_preprocessing_identity=identity, heldout_record_ids=heldout_ids))
    data = PTBXLData(metadata, labels, splits, datasets, normalizer, class_statistics(labels, splits),
                     missing, provenance, config, seed)
    # Check model/config/metadata/labels/splits/fit IDs BEFORE reading even a held-out ECG.
    report = compare_ssl_provenance(context["state"], context["experiment"], data, context["proof"])
    if report["blocking_mismatch_count"]:
        raise ProvenanceMismatchError(report)
    if config.cache_signals and heldout_ids:
        processor = PTBXLDataset(metadata, labels, heldout_ids, config, normalizer, "val", reader, seed)
        heldout_path, manifest = ensure_signal_cache(identity, heldout_ids, config, processor.processed_signal)
        for name, dataset in datasets.items():
            if name != "train":
                dataset.cache_path = heldout_path
                dataset.cache_rows = {ecg_id: row for row, ecg_id in enumerate(manifest["ids"])}
        provenance["ssl_cache_backed"]["heldout_cache"] = dict(path=heldout_path, payload_sha256=manifest["payload_sha256"])
    log_progress(f"SSL-cache-backed data: {len(splits['train'])} training ECGs from original cache; "
                 f"{len(heldout_ids)} held-out ECGs; no training preprocessing/normalizer fit")
    return data


def cache_backed_comparison_inputs(state, config, data, expected, actual, proof):
    """Separate authenticated training identity from current held-out file stats."""
    from .labels import build_labels
    from .data import build_official_splits
    source = state["provenance"]
    cache = authenticate_cache(state, config.ssl_reference_cache)
    source_identity = source["preprocessing_identity"]
    current = data.provenance["preprocessing_identity"]
    expected["preprocessing_identity"] = {k: v for k, v in source_identity.items() if k != "raw_file_stats_sha256"}
    actual["preprocessing_identity"] = {k: v for k, v in current.items() if k != "raw_file_stats_sha256"}
    expected["original_training_preprocessing_identity"] = source_identity
    actual["original_training_preprocessing_identity"] = cache["manifest"]["identity"]
    canonical = build_labels(data.metadata, pd.read_csv(Path(data.config.data_dir) / "scp_statements.csv", index_col=0), config.task)
    canonical = canonical.loc[canonical.sum(axis=1).gt(0)]
    valid_labels = data.labels.equals(canonical)
    official = build_official_splits(data.metadata)
    expected_splits = {name: ids.intersection(canonical.index, sort=False).tolist() for name, ids in official.items()}
    if data.provenance["diagnostic_only"]:
        limits = data.provenance["diagnostic_limits"]
        expected_splits = {name: expected_splits[name][:limit] for name, limit in limits.items()}
    split_membership = {name: list(ids) for name, ids in data.splits.items()} == expected_splits
    for name, dataset in data.datasets.items():
        valid_labels = valid_labels and dataset.ids == list(data.splits[name]) and np.array_equal(
            dataset._label_values, canonical.loc[dataset.ids].to_numpy(dtype=np.float32))
    binding = data.provenance.get("ssl_cache_backed", {})
    checks = dict(source_authenticated=bool(proof and proof.get("completion_verified") is True and
                  proof.get("sha256") == config.ssl_encoder_sha256.lower()),
        explicit_cache_backed_data=binding.get("mode") == "ssl_cache_backed_finetuning",
        canonical_label_mapping=bool(valid_labels),
        exact_supervised_split_membership=split_membership,
        training_cache_bound=Path(data.datasets["train"].cache_path or "").resolve() == Path(cache["path"]),
        training_cache_rows=data.datasets["train"].cache_rows == {i: row for row, i in enumerate(cache["manifest"]["ids"])},
        all_datasets_use_original_normalizer=all(ds.normalizer.state_dict() == source["normalizer"] for ds in data.datasets.values()),
        dataset_configuration=all(asdict(ds.config) == asdict(data.config) for ds in data.datasets.values()),
        heldout_excluded_from_ssl_cache=all((ds.cache_path is None or Path(ds.cache_path).resolve() != Path(cache["path"]))
            and ds.augmenter is None for name, ds in data.datasets.items() if name != "train"))
    return checks, dict(mode="ssl_cache_backed_finetuning", authenticated_cache=cache,
        normalizer=source["normalizer"], heldout_preprocessing_identity=current,
        independent_training_waveform_equivalence=dict(performed=False, verified=False),
        statement="Original SSL training cache and normalizer are authoritative for this explicit experiment; no separately processed waveform reproduction is claimed")
