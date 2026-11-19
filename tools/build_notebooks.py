"""Render standalone Colab notebooks from percent-cell Python scripts using the standard library."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_cells(source):
    chunks = re.split(r'^# %%(\s*\[markdown\])?[ \t]*\n', source, flags=re.MULTILINE)
    for marker, body in zip(chunks[1::2], chunks[2::2]):
        if marker:
            text = '\n'.join(line.removeprefix('# ').rstrip() for line in body.strip('\n').splitlines())
            yield 'markdown', text
        else:
            yield 'code', body.strip('\n')


def render(script, check=False):
    source = script.read_text(encoding='utf-8-sig')
    cells = list(split_cells(source))
    identity = hashlib.sha256('\n'.join(body for kind, body in cells
        if kind == 'code' and not body.startswith('NOTEBOOK_CODE_SHA256 =')).encode()).hexdigest()
    source = re.sub(r"^NOTEBOOK_CODE_SHA256 = '[0-9a-f]{64}'$", f"NOTEBOOK_CODE_SHA256 = '{identity}'", source, flags=re.M)
    output = []
    for index, (kind, body) in enumerate(split_cells(source)):
        entry = dict(cell_type=kind, source=body.splitlines(keepends=True), metadata={},
                     id=hashlib.sha256(f'{index}:{body}'.encode()).hexdigest()[:12])
        if kind == 'code':
            compile(body, f'{script.name}:cell-{index}', 'exec')
            entry.update(execution_count=None, outputs=[])
        output.append(entry)
    metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    language_info=dict(name='python'), thesis=dict(code_sha256=identity))
    if script.parent.name == 'mlwa':
        metadata['mlwa'] = dict(code_sha256=identity)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=output)
    dest = ROOT / 'notebooks' / script.relative_to(ROOT / 'workflows').with_suffix('.ipynb')
    content = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    updates = [(script, source), (dest, content)]
    for path, value in updates:
        if path.is_file() and path.read_text(encoding='utf-8-sig') == value:
            continue
        if check:
            raise ValueError(f'Out of date: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if a notebook or cache fingerprint is stale.')
    options = parser.parse_args()
    workflows = sorted((ROOT / 'workflows').glob('*/*.py'))
    for workflow in workflows:
        render(workflow, check=options.check)
    print(f'Checked {len(workflows)} Python/notebook pairs.')

