"""Validate maintained notebooks safely; does not install, sync, infer or train."""
import ast
import hashlib
import json
import re
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
STAGES = ['01_supervised_baseline.ipynb','02_ssl_pretraining.ipynb',
    '03_ssl_finetuning.ipynb','04_ssl_extended_finetuning.ipynb',
    '05_asl_finetuning.ipynb','06_evaluation_and_plots.ipynb']


def validate():
    bootstrap = (ROOT/'src/bootstrap.py').read_text(encoding='utf-8')
    for name in STAGES + ['05_btp_kaggle_final.ipynb']:
        book=nbformat.read(ROOT/name,as_version=4)
        nbformat.validate(book)
        namespace={}
        for i,cell in enumerate(book.cells):
            if cell.cell_type != 'code': continue
            ast.parse(cell.source,filename=f'{name}:{i}')
            if name in STAGES:
                assert not cell.outputs and cell.execution_count is None
                if 'stage-execution' in cell.metadata.get('tags',[]):
                    assert cell.source.startswith(bootstrap),name
                exec(compile(cell.source,f'{name}:{i}','exec'),namespace)
                if 'stage-config' in cell.metadata.get('tags',[]):
                    assert namespace['EXECUTE'] is False,name
            elif 'final-config' in cell.metadata.get('tags',[]):
                exec(compile(cell.source,f'{name}:{i}','exec'),namespace)
                assert not any(v for k,v in namespace['SETTINGS'].items() if k.startswith('RUN_'))
                assert not namespace['ACTIVE'] and not namespace['NEED_GPU']
                assert not namespace['DIAGNOSTIC_MODE'] and not namespace['SSL_FINETUNE_PROVENANCE_ONLY']
            else:
                exec(compile(cell.source,f'{name}:{i}','exec'),namespace)
        if name in STAGES:
            assert namespace['EXECUTE'] is False
        else:
            assert book.cells[3].source.startswith(bootstrap)
            assert namespace['EXECUTION_SUMMARY']['status']=='skipped'
    manifest=json.loads((ROOT/'archive/notebooks/manifest.json').read_text())
    for name,pin in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/'archive/notebooks'/name).read_bytes()).hexdigest()==pin,name
    for path in [ROOT/'README.md', *(ROOT/'docs').rglob('*.md'), *(ROOT/'archive').rglob('*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent/link.split('#')[0]).exists(),(path,link)
    print('Validated 7 maintained notebook schemas/code and safe defaults, plus 5 exact archived blobs. No GPU execution.')


if __name__=='__main__':
    validate()
