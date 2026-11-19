"""Regression checks for standalone delivery, ZIP reuse and ordered P0 prefetch."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import nbformat
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def sources(stage, prefix):
    path = next((ROOT / 'notebooks' / stage).glob(prefix + '*.ipynb'))
    return [''.join(c['source']) for c in json.loads(path.read_bytes())['cells'] if c['cell_type'] == 'code']


def test_standalone_notebooks_match_python():
    subprocess.run([sys.executable, str(ROOT / 'tools/build_notebooks.py'), '--check'], check=True)
    for p in (ROOT / 'workflows').glob('*/*.py'):
        target = ROOT / 'notebooks' / p.relative_to(ROOT / 'workflows').with_suffix('.ipynb')
        doc = nbformat.read(target, as_version=4)
        nbformat.validate(doc)
        assert all(not c.outputs for c in doc.cells if c.cell_type == 'code')


def test_zip_resume_and_direct_directory(tmp_path):
    namespace = {'Path': Path}
    exec(next(s for s in sources('dmip', '05') if 'def stage_archive' in s), namespace)
    stage = namespace['stage_archive']
    archive = tmp_path / 'folds.zip'
    with zipfile.ZipFile(archive, 'w') as f:
        f.writestr('wrapped/split.csv', 'label\n3\n')
    extracted = stage(archive, tmp_path / 'work')
    split = extracted / 'wrapped' / 'split.csv'
    stamp = split.stat().st_mtime_ns
    assert stage(archive, tmp_path / 'work') == extracted and split.stat().st_mtime_ns == stamp
    assert stage(extracted, tmp_path / 'work') == extracted
    with zipfile.ZipFile(archive, 'w') as f:
        f.writestr('wrapped/split.csv', 'label\n1\n')
    assert stage(archive, tmp_path / 'work') != extracted
    with zipfile.ZipFile(archive, 'w') as f:
        f.writestr('../outside.txt', 'invalid')
    with pytest.raises(ValueError, match='Invalid ZIP member'):
        stage(archive, tmp_path / 'work')


def test_p0_prefetch_order_and_failures():
    code = next(s for s in sources('mlwa', '00') if 'def prepared_rows(' in s)
    ns = {}
    exec(code[code.index('def prepared_rows('):code.index('def infer_split(')], ns)
    frame = pd.DataFrame({'image_id': ['a','b','c']}, index=[8,4,9])
    ns['prepare_image'] = lambda row, pooled: (row.image_id, pooled)
    for workers in (1,2):
        rows = list(ns['prepared_rows'](frame, True, workers))
        assert [(i, arrays) for i, _, arrays in rows] == [(8,('a', True)), (4,('b', True)), (9,('c', True))]
    def fail(row, pooled):
        raise OSError('unreadable input')
    ns['prepare_image'] = fail
    with pytest.raises(OSError, match='unreadable input'):
        list(ns['prepared_rows'](frame, False, 2))

