"""Short CPU ASL/whole-classifier/provenance checks; no real GPU execution."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from test_finetuning import source, ClassifierDouble
from conftest import synthetic_reader
from src.losses import AsymmetricLoss
from src.config import ExperimentConfig, checkpoint_config_matches
from src.cache import digest_file
from src.ssl_cache_backed import load_cache_backed_data, authenticate_cache, cache_authentication_scope
from src.asl_stage import asl_config, load_asl_source, initialize_full_classifier
from src.training import Trainer
from src.utils import write_json


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("width", [5, 23])
@pytest.mark.parametrize("target", [0., 1.])
def test_asl_extreme_fp32_finite_loss_and_gradients(width, target):
    x = torch.linspace(-1e4, 1e4, width).repeat(2, 1).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = AsymmetricLoss()(x, torch.full_like(x, target))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_asl_reference_and_bce_limit_and_grad_mode():
    x = torch.tensor([[-3., -.1, .7, 3.]], requires_grad=True)
    y = torch.tensor([[0., 1., 0., 1.]])
    p, q = x.sigmoid(), ((-x).sigmoid() + .05).clamp(max=1)
    reference = -(y * p.log() + (1-y) * (1-q).detach().pow(4) * q.log()).mean()
    torch.testing.assert_close(AsymmetricLoss()(x, y), reference)
    torch.testing.assert_close(torch.autograd.grad(AsymmetricLoss()(x, y), x)[0], torch.autograd.grad(reference, x)[0])
    bce = F.binary_cross_entropy_with_logits(x, y)
    torch.testing.assert_close(AsymmetricLoss(0, 0, 0)(x, y), bce)
    with torch.no_grad():
        AsymmetricLoss()(x, y)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("bad", ["pos", "smoothing", "amp", "nan", "target"])
def test_asl_does_not_silently_accept_bce_controls_or_invalid_values(bad):
    if bad in {"nan", "target"}:
        with pytest.raises(ValueError):
            AsymmetricLoss()(torch.tensor([[float("nan") if bad == "nan" else 1.]]), torch.tensor([[.5 if bad == "target" else 1.]]))
    else:
        values = dict(loss_type="asl", use_pos_weight=False, label_smoothing=0, mixed_precision=False)
        values.update({"pos": {"use_pos_weight": True}, "smoothing": {"label_smoothing": .05}, "amp": {"mixed_precision": True}}[bad])
        with pytest.raises(ValueError): ExperimentConfig(**values)


def make_classifier_export(source, tmp_path, task, width, epoch):
    _, base = source
    encoder = torch.load(base.ssl_encoder_checkpoint, weights_only=True)
    base = replace(base, task=task, epochs=15, diagnostic=False, ssl_data_mode="ssl_cache_backed",
        ssl_reference_cache=encoder["provenance"]["signal_cache"]["path"])
    data = load_cache_backed_data(base, reader=synthetic_reader)
    metadata = dict(task=task, normalizer=data.normalizer.state_dict(), provenance=data.provenance,
                    splits={k: list(map(int,v)) for k,v in data.splits.items()})
    trained = ClassifierDouble(width)
    with torch.no_grad():
        for parameter in trained.parameters(): parameter.add_(.03)
    history = [dict(epoch=i, val_auroc=.5 + .01*i if i <= epoch else .4, val_loss=.7) for i in range(1,16)]
    folder = tmp_path / "classifier";folder.mkdir()
    path = folder / "best.pth"
    state = dict(epoch=epoch, model_state_dict=trained.state_dict(), config=asdict(base), metadata=metadata,
                 class_names=data.class_names, history=history[:epoch], best_score=history[epoch-1]["val_auroc"])
    torch.save(state,path)
    write_json(folder / "config.json", asdict(base))
    write_json(folder / "summary.json", dict(epochs_completed=15, benchmark_result=True,
        experiment_kind="ssl_cache_backed_finetuning", task=task, best_epoch=epoch))
    write_json(folder / "status.json", dict(status="complete", benchmark_result=True))
    pd.DataFrame(history).to_csv(folder / "history.csv",index=False)
    pin = digest_file(path)
    cfg = asl_config(path, pin, epoch, tmp_path / "new_stage")
    return path, pin, epoch, cfg, data, metadata, state


@pytest.fixture(params=[("superclass", 5, 7), ("subclass", 23, 14)])
def classified(source, tmp_path, request):
    return make_classifier_export(source, tmp_path, *request.param)


def test_all_classifier_tensors_load_and_optimizer_schedule_are_new(classified, tmp_path):
    path, pin, epoch, cfg, data, metadata, state = classified
    target = ClassifierDouble(len(data.class_names))
    proof = initialize_full_classifier(target, cfg, data, metadata)
    for key,value in target.state_dict().items(): torch.testing.assert_close(value,state["model_state_dict"][key],atol=0,rtol=0)
    assert any(key.startswith("head.") for key in proof["transferred_parameter_names"])
    assert any(key.startswith("attn_pool.") for key in proof["transferred_parameter_names"])
    trainer = Trainer(target, cfg, data.class_names, tmp_path / "trainer", metadata=metadata)
    assert not trainer.optimizer.state and trainer.scheduler_continuation is None
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-5
    assert trainer.scheduler.T_max == 5 and trainer.scheduler.eta_min == 1e-6
    assert cfg.label_smoothing == 0 and not cfg.use_pos_weight and isinstance(trainer.criterion,AsymmetricLoss)
    with pytest.raises(ValueError,match="pos_weight"):
        Trainer(target,cfg,data.class_names,tmp_path / "bad",pos_weight=torch.ones(len(data.class_names)))
    assert digest_file(path) == pin
    with pytest.raises(ValueError, match="keys/shapes/dtypes"):
        initialize_full_classifier(ClassifierDouble(len(data.class_names)).double(), cfg, data, metadata)


@pytest.mark.parametrize("damage", ["hash", "vocabulary", "preprocessing", "config", "completion", "selection"])
def test_classifier_source_mismatches_reject_before_transfer(classified, monkeypatch, damage):
    path, pin, epoch, cfg, data, metadata, state = classified
    target = ClassifierDouble(len(data.class_names))
    monkeypatch.setattr(target,"load_state_dict",lambda *a,**k:pytest.fail("invalid source reached weight transfer"))
    if damage == "hash": cfg.initial_checkpoint_sha256 = "0"*64
    if damage == "vocabulary": data.labels.columns = list(reversed(data.class_names))
    if damage == "preprocessing": metadata["normalizer"]["std"][0][0] *= 2
    if damage == "config": cfg.model.dropout = .2
    if damage == "completion": write_json(path.with_name("status.json"),{"status":"failed"})
    if damage == "selection": cfg.initial_checkpoint_epoch = epoch-1
    with pytest.raises((ValueError,AssertionError)):
        initialize_full_classifier(target,cfg,data,metadata)


def test_legacy_bce_checkpoint_defaults_and_cross_loss_resume_rejection():
    current = ExperimentConfig()
    saved = asdict(current)
    for key in ("loss_type","asl_gamma_pos","asl_gamma_neg","asl_clip","initial_checkpoint_mode","initial_checkpoint_sha256","initial_checkpoint_epoch"):
        saved.pop(key)
    assert checkpoint_config_matches(saved,current) and checkpoint_config_matches(saved,saved)
    assert not checkpoint_config_matches(saved,replace(current,loss_type="asl",use_pos_weight=False,label_smoothing=0,mixed_precision=False))


def test_lightweight_path_preflight_checks_inputs_and_missing_sidecars(classified):
    from src.asl_stage import preflight_asl_paths
    path, pin, epoch, cfg, data, metadata, state = classified
    (Path(cfg.data.data_dir) / "records100").mkdir(exist_ok=True)
    settings = dict(ASL_BEST_CHECKPOINTS={"superclass": str(path), "subclass": str(path)},
        ASL_SOURCE_BEST_EPOCHS={cfg.task: epoch}, **{f"RUN_ASL_{cfg.task.upper()}": True})
    assert preflight_asl_paths(settings)[cfg.task]["availability"] == "passed"
    path.with_name("history.csv").unlink()  # synthetic test export only
    with pytest.raises(ValueError, match="Missing/invalid"):
        preflight_asl_paths(settings)


def test_cache_hash_is_scoped_and_mutation_never_reuses_authentication(source, monkeypatch):
    import src.ssl_cache_backed as backed
    _,cfg = source
    state=torch.load(cfg.ssl_encoder_checkpoint,weights_only=True)
    path=state["provenance"]["signal_cache"]["path"]
    count=[]; original=backed.digest_file
    monkeypatch.setattr(backed,"digest_file",lambda p: (count.append(str(p)),original(p))[1])
    monkeypatch.setattr(backed,"_read_only_cache",lambda path:True)  # simulate immutable Kaggle input mount
    with cache_authentication_scope():
        authenticate_cache(state,path)
        with cache_authentication_scope(): authenticate_cache(state,path)
        assert len(count)==1
    with cache_authentication_scope(): authenticate_cache(state,path)
    assert len(count)==2
    with cache_authentication_scope():
        authenticate_cache(state,path)
        monkeypatch.setattr(backed,"_read_only_cache",lambda path:False)
        a=np.load(path,mmap_mode="r+");a[0,0,0]+=1;a.flush();del a
        with pytest.raises(ValueError,match="SHA256"):
            authenticate_cache(state,path)


def test_stage_exports_both_threshold_variants_without_ssl_weight_transfer(classified, monkeypatch):
    import src.experiments as experiments
    import src.finetuning as finetuning
    import src.data as data_module
    path,pin,epoch,cfg,data,metadata,state=classified
    monkeypatch.setattr(data_module,"read_waveform",synthetic_reader)
    monkeypatch.setattr(experiments,"create_model",lambda name,width,config:ClassifierDouble(width))
    monkeypatch.setattr(finetuning,"initialize_ssl_backbone",lambda *a,**k:pytest.fail("SSL-only transfer called"))
    source_hashes={p.name:digest_file(p) for p in path.parent.iterdir() if p.is_file()}
    result=experiments.run_experiment(config=cfg)
    out=Path(result["output_dir"])
    assert result["summary"]["initialization"]=="verified_full_classifier_all_weights"
    assert result["summary"]["loss_type"]=="asl" and len(result["history"])==5
    checkpoint=torch.load(out / "last.pth",weights_only=True)
    assert checkpoint["config"]["initial_checkpoint_sha256"]==pin
    assert checkpoint["config"]["label_smoothing"]==0 and not checkpoint["config"]["use_pos_weight"]
    assert "scheduler_continuation" not in checkpoint
    for split in ("validation","validation_at_0.5","test","test_at_0.5"):
        assert (out/split/"metrics.json").is_file() and (out/split/"per_class.csv").is_file()
        assert (out/split/"predictions.csv").is_file()
    assert source_hashes=={p.name:digest_file(p) for p in path.parent.iterdir() if p.is_file()}


def test_worker_runs_only_two_asl_tasks_with_one_immutable_cache_hash(source, tmp_path, monkeypatch):
    import src.data as data_module
    import src.experiments as experiments
    import src.finetuning as finetuning
    import src.kaggle as kaggle
    import src.ssl_cache_backed as backed
    import src.unattended as pipeline
    root=Path(__file__).parents[1]
    paths={};pins={}
    for task,width,epoch in (("superclass",5,7),("subclass",23,14)):
        folder=tmp_path/task;folder.mkdir()
        result=make_classifier_export(source,folder,task,width,epoch)
        paths[task],pins[task]=str(result[0]),result[1]
    nb=json.loads((root/"05_btp_kaggle_final.ipynb").read_text())
    namespace={};exec("".join(nb["cells"][1]["source"]),namespace)
    settings=namespace["SETTINGS"]
    session=tmp_path/"session";session.mkdir()
    settings.update(SESSION_DIR=str(session),CHECKOUT_DIR=str(root),DATASET_ROOT=source[1].data.data_dir,
        ASL_BEST_CHECKPOINTS=paths,ASL_CHECKPOINT_SHA256=pins)
    monkeypatch.setattr(kaggle,"require_cuda",lambda:None)  # CPU routing test, no CUDA execution
    monkeypatch.setattr(kaggle,"find_ptbxl_root",lambda *a:Path(source[1].data.data_dir))
    monkeypatch.setattr(data_module,"read_waveform",synthetic_reader)
    monkeypatch.setattr(experiments,"create_model",lambda name,width,config:ClassifierDouble(width))
    monkeypatch.setattr(finetuning,"initialize_ssl_backbone",lambda *a,**k:pytest.fail("SSL-only transfer"))
    monkeypatch.setattr(backed,"_read_only_cache",lambda p:True)  # immutable input contract
    hashes=[];original=backed.digest_file
    monkeypatch.setattr(backed,"digest_file",lambda p:(hashes.append(str(p)),original(p))[1])
    assert pipeline.run_worker(settings)==0
    result=json.loads((session/"final_summary.json").read_text())
    assert set(result["completed"])=={"asl_superclass","asl_subclass"}
    assert len(hashes)==1
    assert len(list((session/"asl_stage2/benchmark").iterdir()))==2
    for task in ("superclass","subclass"):
        out=Path(result["completed"][f"asl_{task}"]["output_dir"])
        assert pd.read_csv(out/"history.csv").epoch.tolist()==[1,2,3,4,5]
        assert len(pd.read_csv(out/"threshold_comparison.csv"))==4

def relocated_data(classified):
    """Fresh load after filesystem-only changes; synthetic waveform bytes unchanged."""
    from src.experiments import diagnostic_limits
    path, pin, epoch, cfg, old_data, old_metadata, state = classified
    ecg_id = old_data.provenance['ssl_cache_backed']['heldout_record_ids'][0]
    name = old_data.metadata.loc[ecg_id, 'filename_lr']
    header = Path(cfg.data.data_dir) / f'{name}.hea'
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text('Synthetic stat change; waveform reader remains identical')
    data = load_cache_backed_data(cfg, reader=synthetic_reader)
    metadata = dict(task=cfg.task, normalizer=data.normalizer.state_dict(), provenance=data.provenance,
                    splits={k:list(map(int,v)) for k,v in data.splits.items()})
    return data, metadata


def test_environmental_stats_and_cache_key_relocation_require_identical_payload(classified, tmp_path, monkeypatch):
    import src.cache as cache
    import src.ssl_cache_backed as backed
    path, pin, epoch, cfg, old_data, old_metadata, state = classified
    data, metadata = relocated_data(classified)
    # Compatibility must consume existing verification, never rehash a waveform cache.
    original = cache.digest_file
    def no_waveform_rehash(p):
        assert Path(p).suffix != '.npy'
        return original(p)
    monkeypatch.setattr(cache, 'digest_file', no_waveform_rehash)
    monkeypatch.setattr(backed, 'digest_file', no_waveform_rehash)
    proof = initialize_full_classifier(ClassifierDouble(len(data.class_names)), cfg, data, metadata, tmp_path)
    report = proof['provenance_comparison']
    assert report['status'] == 'passed_verified_heldout_cache_relocation'
    assert len(report['mismatches']) == 3 and all(d['accepted'] for d in report['mismatches'])
    assert all(report['compatibility_checks'].values())
    assert metadata['provenance']['ssl_cache_backed']['heldout_cache']['payload_sha256'] == state['metadata']['provenance']['ssl_cache_backed']['heldout_cache']['payload_sha256']
    assert (tmp_path / 'asl_classifier_provenance_comparison.json').is_file()
    assert digest_file(path) == pin


@pytest.mark.parametrize('damage', ['payload', 'normalizer', 'splits', 'version', 'preprocessing',
    'training_identity', 'class_order', 'missing_receipt', 'missing_payload', 'inconsistent_stats', 'cache_modified'])
def test_relocation_never_waives_substantive_mismatches(classified, monkeypatch, tmp_path, damage):
    path, pin, epoch, cfg, old_data, old_metadata, state = classified
    data, metadata = relocated_data(classified)
    p = metadata['provenance']; b = p['ssl_cache_backed']
    if damage == 'payload': b['heldout_cache']['payload_sha256'] = '0'*64
    elif damage == 'normalizer': metadata['normalizer']['std'][0][0] *= 2
    elif damage == 'splits': metadata['splits']['train'] = metadata['splits']['train'][:-1]
    elif damage == 'version': p['declared_dataset_version'] = 'different'
    elif damage == 'preprocessing': p['preprocessing_identity']['apply_filter'] = False
    elif damage == 'training_identity': b['original_training_preprocessing_identity']['raw_file_stats_sha256'] = '0'*64
    elif damage == 'class_order': data.labels.columns = list(reversed(data.class_names))
    elif damage == 'missing_receipt': del data._verified_heldout_cache
    elif damage == 'missing_payload': del b['heldout_cache']['payload_sha256']
    elif damage == 'inconsistent_stats':
        from copy import deepcopy
        b['heldout_preprocessing_identity'] = deepcopy(b['heldout_preprocessing_identity'])
        b['heldout_preprocessing_identity']['raw_file_stats_sha256'] = '1'*64
    else:
        with open(b['heldout_cache']['path'], 'ab') as f: f.write(b'changed')
    target = ClassifierDouble(len(data.class_names))
    monkeypatch.setattr(target, 'load_state_dict', lambda *a,**k:pytest.fail('rejected data reached weight transfer'))
    with pytest.raises(ValueError, match='provenance mismatch'):
        initialize_full_classifier(target, cfg, data, metadata, tmp_path)
    report = json.loads((tmp_path / 'asl_classifier_provenance_comparison.json').read_text())
    assert report['status'] == 'rejected' and report['blocking_mismatch_count'] > 0
    assert not any(d['accepted'] for d in report['mismatches'])

def test_path_only_relocation_is_content_bound(classified):
    from copy import deepcopy
    from src.asl_stage import classifier_metadata_report
    path, pin, epoch, cfg, data, metadata, state = classified
    saved = deepcopy(state)
    saved['metadata']['provenance']['ssl_cache_backed']['heldout_cache']['path'] = '/old/session/signals.npy'
    report = classifier_metadata_report(saved, data, metadata)
    assert report['status'] == 'passed_verified_heldout_cache_relocation'
    assert [d['field'] for d in report['mismatches']] == ['provenance.ssl_cache_backed.heldout_cache.path']


def test_real_processed_signal_change_rejected_even_with_identical_reader_signature(classified, tmp_path):
    path, pin, epoch, cfg, old_data, old_metadata, state = classified
    relocated_data(classified)  # change stat-based key so a new held-out cache is built
    # Force another cache miss while preserving the implementation identity.
    ecg_id = old_data.provenance['ssl_cache_backed']['heldout_record_ids'][0]
    header = Path(cfg.data.data_dir) / f"{old_data.metadata.loc[ecg_id, 'filename_lr']}.hea"
    header.write_text('Different waveform data, not just timestamps')
    def changed_reader(path, sampling_rate):
        return synthetic_reader(path, sampling_rate) * 2
    changed_reader.__module__ = synthetic_reader.__module__
    changed_reader.__qualname__ = synthetic_reader.__qualname__
    data = load_cache_backed_data(cfg, reader=changed_reader)
    metadata = dict(task=cfg.task, normalizer=data.normalizer.state_dict(), provenance=data.provenance,
                    splits={k:list(map(int,v)) for k,v in data.splits.items()})
    with pytest.raises(ValueError, match='heldout_cache.payload_sha256'):
        initialize_full_classifier(ClassifierDouble(len(data.class_names)), cfg, data, metadata, tmp_path)
