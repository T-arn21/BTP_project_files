"""Stage selectors and actual notebook entry paths with explicit CPU adapters."""
from copy import deepcopy
from pathlib import Path
import ast
import importlib.util
import json
import subprocess
import sys
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from src.notebook_workflows import PROFILES, workflow_settings
from src.result_review import review_experiment
from src.evaluation import export_evaluation, tune_thresholds
from src.utils import write_json
from src.cache import digest_file

ROOT=Path(__file__).parents[1]
NAMES=['01_supervised_baseline.ipynb','02_ssl_pretraining.ipynb','03_ssl_finetuning.ipynb',
       '04_ssl_extended_finetuning.ipynb','05_asl_finetuning.ipynb','06_evaluation_and_plots.ipynb']


def request(profile,mode='full',**overrides):
    return dict(WORKFLOW_PROFILE=profile,WORKFLOW_MODE=mode,WORKFLOW_OVERRIDES=overrides,
        CHECKOUT_DIR=str(ROOT),GIT_BRANCH='refactor/project-organization',
        GIT_TIMEOUT_SECONDS=600,OUTPUT_ROOT='/kaggle/working/btp_outputs')


def test_all_notebooks_validate_and_safe_defaults_preserve_archive():
    spec=importlib.util.spec_from_file_location('validator',ROOT/'scripts/validate_notebooks.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate()


def test_unified_default_execution_is_inactive_and_resume_mounts_are_consistent(monkeypatch):
    book=nbformat.read(ROOT/'05_btp_kaggle_final.ipynb',as_version=4)
    ns={}
    exec(book.cells[1].source,ns)
    assert all(value is False for key,value in ns['SETTINGS'].items() if key.startswith('RUN_'))
    assert ns['ACTIVE'] is False and ns['NEED_GPU'] is False
    assert ns['DIAGNOSTIC_MODE'] is False and ns['SSL_FINETUNE_PROVENANCE_ONLY'] is False
    prefix='/kaggle/input/datasets/arnavvt/checkpoint-2'
    assert ns['SSL_FINETUNE_RESUME_ARTIFACT_ROOT']==prefix
    for task,path in ns['SSL_FINETUNE_RESUME_RUN_DIRS'].items():
        assert path.startswith(prefix+'/btp_outputs/sessions/20261029T180106Z_e154ce34/')
        assert Path(path).name.startswith('mamba_'+task+'_')
    extension=nbformat.read(ROOT/NAMES[3],as_version=4)
    stage={};exec(next(c.source for c in extension.cells if 'stage-config' in c.metadata.get('tags',[])),stage)
    assert stage['EXECUTE'] is False
    assert stage['OVERRIDES']['SSL_FINETUNE_RESUME_RUN_DIRS']==ns['SSL_FINETUNE_RESUME_RUN_DIRS']
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:pytest.fail('inactive notebook launched a subprocess'))
    for cell in book.cells[2:]:
        if cell.cell_type=='code':exec(cell.source,ns)
    assert ns['EXECUTION_SUMMARY']['status']=='skipped'


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('mode',['smoke','full'])
def test_profiles_select_only_documented_worker_modes(profile,mode):
    if mode=='smoke' and profile in PROFILES[3:]:
        with pytest.raises(ValueError): workflow_settings(request(profile,mode))
        return
    s=workflow_settings(request(profile,mode))
    flags={k for k,v in s.items() if k.startswith('RUN_') and v}
    expected={
      'supervised_baseline': {'RUN_SMOKE','RUN_METADATA_AUDIT'} | ({'RUN_SUPERCLASS','RUN_SUBCLASS'} if mode=='full' else set()),
      'ssl_pretraining': {'RUN_SSL_SMOKE'} | ({'RUN_SSL_PRETRAIN'} if mode=='full' else set()),
      'ssl_finetuning': {'RUN_SSL_FINETUNE_SMOKE'} | ({'RUN_SSL_FINETUNE_SUPERCLASS','RUN_SSL_FINETUNE_SUBCLASS'} if mode=='full' else set()),
      'ssl_extended_finetuning': {'RUN_SSL_FINETUNE_RESUME'},
      'asl_finetuning': {'RUN_ASL_SUPERCLASS','RUN_ASL_SUBCLASS'}}[profile]
    assert flags==expected
    assert not s['CONTINUATION_SESSION_ID'] and not s['DIAGNOSTIC_MODE']
    assert not s['SSL_FINETUNE_PROVENANCE_ONLY'] and not s['MIXED_PRECISION']
    assert s['MODEL_SETTINGS']['embed_dim']==384 and s['MODEL_SETTINGS']['n_layers']==8
    if profile=='asl_finetuning':
        from src.asl_stage import RECIPE,validate_asl_settings
        validate_asl_settings(s)
        assert s['ASL_STAGE_SETTINGS']==RECIPE
    if profile=='ssl_extended_finetuning':
        from src.resume_extension import validate_resume_settings
        validate_resume_settings(s)


@pytest.mark.parametrize('overrides',[{'RUN_ASL_SUPERCLASS':True},{'DIAGNOSTIC_MODE':True}, {'TYPO':1}, {'GIT_BRANCH':'oops'}])
def test_profiles_reject_hidden_competing_modes(overrides):
    with pytest.raises(ValueError):workflow_settings(request('supervised_baseline',**overrides))


def test_nested_settings_and_unified_snapshot_keep_scientific_defaults():
    s=workflow_settings(request('ssl_pretraining',SSL_SETTINGS={'epochs':20}))
    assert s['SSL_SETTINGS']['mask_fraction']==.25 and s['SSL_SETTINGS']['batch_size']==2
    ns={};exec(nbformat.read(ROOT/'05_btp_kaggle_final.ipynb',as_version=4).cells[1].source,ns)
    saved=json.loads((ROOT/'configs/kaggle_defaults.json').read_text())
    old,current=deepcopy(saved),deepcopy(ns['SETTINGS'])
    old.pop('GIT_BRANCH');current.pop('GIT_BRANCH')
    assert old==current


@pytest.mark.parametrize('index',range(5))
@pytest.mark.parametrize('mode',['smoke','full'])
def test_enabled_notebook_entry_routes_to_shared_launcher(index,mode):
    book=nbformat.read(ROOT/NAMES[index],as_version=4)
    ns={}
    cell=next(c for c in book.cells if 'stage-config' in c.metadata.get('tags',[]))
    exec(cell.source,ns);ns.update(EXECUTE=True,MODE=mode)
    ns['REQUEST']['WORKFLOW_MODE']=mode
    execution=next(c.source for c in book.cells if 'stage-execution' in c.metadata.get('tags',[]))
    prefix,tail=execution.rsplit('\n\nif EXECUTE:',1)
    exec(prefix,ns)
    seen=[]
    def launch(req):
        settings=workflow_settings(req);seen.append(settings)
        return dict(status='complete',completed={'cpu_adapter':{}},session_dir='test')
    ns['execute_notebook']=launch
    if mode=='smoke' and index>=3:
        with pytest.raises(ValueError):exec('if EXECUTE:'+tail,ns)
    else:
        exec('if EXECUTE:'+tail,ns)
        assert seen[0]['WORKFLOW_PROFILE']==PROFILES[index]
        assert ns['EXECUTION_SUMMARY']['status']=='complete'


def test_bootstrap_resolves_profiles_only_after_sync_and_saves_resolved_settings(tmp_path,monkeypatch):
    import src.bootstrap as bootstrap
    seen=[]
    def sync(checkout,**kw):
        seen.append('sync');return dict(checkout=str(ROOT),commit='a'*40,branch=kw['branch'])
    monkeypatch.setattr(bootstrap,'sync_repository',sync)
    monkeypatch.setattr(bootstrap,'select_import_root',lambda p:str(ROOT/'src/__init__.py'))
    def launch(checkout,config_path,session):
        seen.append('launch');saved=json.loads(config_path.read_text())
        assert saved['RUN_SSL_PRETRAIN'] and not saved['RUN_ASL_SUPERCLASS']
        assert saved['SOURCE_REVISION']=='a'*40
        assert saved['SSL_SETTINGS']['mask_fraction']==.25
        write_json(session/'final_summary.json',dict(status='complete',completed={},checks={},errors={},session_dir=str(session)))
        return 0
    monkeypatch.setattr(bootstrap,'launch_pipeline',launch)
    req=request('ssl_pretraining');req['OUTPUT_ROOT']=str(tmp_path/'outputs')
    assert bootstrap.execute_notebook(req)['status']=='complete'
    assert seen==['sync','launch']


@pytest.fixture
def exported(tmp_path):
    source=tmp_path/'source';source.mkdir()
    names=['NORM','MI'];ids={'train':[1,2],'val':[3,4,5,6],'test':[7,8,9,10]}
    y=np.array([[0,1],[1,0],[1,1],[0,0]])
    p=np.array([[.2,.7],[.8,.3],[.6,.8],[.3,.1]],dtype=np.float32)
    t=tune_thresholds(y,p,split='val')
    export_evaluation(source/'validation',y,p,ids['val'],names,t)
    export_evaluation(source/'test',y,p,ids['test'],names,t)
    write_json(source/'config.json',dict(diagnostic=False,loss_type='bce'))
    write_json(source/'summary.json',dict(benchmark_result=True,best_epoch=1))
    write_json(source/'status.json',dict(status='complete'))
    write_json(source/'data_manifest.json',dict(class_names=names,splits=ids))
    write_json(source/'thresholds.json',dict(source_split='val',class_names=names,values=t.tolist(),best_epoch=1))
    pd.DataFrame([dict(epoch=1,train_loss=.8,val_loss=.7)]).to_csv(source/'history.csv',index=False)
    return source


def test_cpu_evaluation_notebook_runs_end_to_end_on_saved_predictions(exported,tmp_path):
    before={str(p):digest_file(p) for p in exported.rglob('*') if p.is_file()}
    # A fresh process tests real source selection without invalidating pytest's
    # already-imported src modules. No network, CUDA or model adapters are used.
    child=f'''import nbformat
book=nbformat.read({str(ROOT/NAMES[5])!r},as_version=4)
ns={{}}
for cell in book.cells:
    if cell.cell_type!='code': continue
    exec(cell.source,ns)
    if 'stage-config' in cell.metadata.get('tags',[]):
        ns.update(EXECUTE=True,SYNC_REPOSITORY=False,REPO_ROOT={str(ROOT)!r},
            EXPERIMENTS={{'model':{str(exported)!r}}},OUTPUT_ROOT={str(tmp_path/'reviews')!r})
'''
    completed=subprocess.run([sys.executable,'-c',child],capture_output=True,text=True,timeout=120)
    assert completed.returncode==0,completed.stdout+completed.stderr
    proof_path=next((tmp_path/'reviews').glob('*/model/review_provenance.json'))
    result=json.loads(proof_path.read_text());out=Path(result['output'])
    assert not result['model_inference_reproduced']
    assert len(pd.read_csv(out/'threshold_comparison.csv'))==4
    assert (out/'figures/loss_curves.png').exists() and (out/'figures/confusion_matrices.png').exists()
    assert before=={str(p):digest_file(p) for p in exported.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage',['class_order','record_ids','probabilities','thresholds','diagnostic','overlap'])
def test_reanalysis_rejects_inconsistent_exports_before_output_creation(exported,tmp_path,damage):
    if damage in {'class_order','record_ids','probabilities'}:
        p=exported/'test/predictions.csv';frame=pd.read_csv(p)
        if damage=='class_order':frame=frame[frame.columns.tolist()[::-1]]
        elif damage=='record_ids':frame.loc[0,'ecg_id']=999
        else:frame.loc[0,'MI_prob']=float('inf')
        frame.to_csv(p,index=False)
    else:
        filename={'thresholds':'thresholds.json','diagnostic':'config.json','overlap':'data_manifest.json'}[damage]
        p=exported/filename;value=json.loads(p.read_text())
        if damage=='thresholds':value['values']=[.1,.1]
        elif damage=='diagnostic':value['diagnostic']=True
        else:value['splits']['test'][0]=value['splits']['train'][0]
        write_json(p,value)
    output=tmp_path/'review'
    with pytest.raises(ValueError):review_experiment(exported,output)
    assert not output.exists()


def test_documented_historical_test_prediction_filename_and_ambiguity(exported,tmp_path):
    canonical=exported/'test/predictions.csv'
    alternate=canonical.with_name('test_predictions.csv')
    alternate.write_bytes(canonical.read_bytes())
    with pytest.raises(ValueError,match='Ambiguous'):
        review_experiment(exported,tmp_path/'ambiguous')
    canonical.unlink()  # synthetic fixture only
    proof=review_experiment(exported,tmp_path/'review')
    assert proof['prediction_files']['test']=='test/test_predictions.csv'
    assert 'test/test_predictions.csv' in proof['source_sha256']
