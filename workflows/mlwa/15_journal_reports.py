# %% [markdown]
# # 15 · MLWA Journal Verified-Only Reports, Coverage Audit & Publication Figures

# %%
NOTEBOOK_CODE_SHA256 = '9134b13e445b51f4174cb3469a10e01e40b90f960275ea731fade6b637ddeed8'

# %%
import os
import sys
from pathlib import Path

IN_COLAB: bool = (
    'COLAB_RELEASE_TAG' in os.environ
    or 'COLAB_BACKEND_VERSION' in os.environ
    or 'google.colab' in sys.modules
)


def _running_in_notebook() -> bool:
    if 'ipykernel' not in sys.modules and 'google.colab' not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


def _bootstrap_colab_repo() -> Path:
    import importlib.util
    import subprocess
    import zipfile

    if IN_COLAB:
        deps = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'pyarrow': 'pyarrow'}
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')

    dr_root = Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()
    bundle_zip = Path(os.environ.get('DR_MLWA_BUNDLE', dr_root / 'inputs' / 'mlwa_journal_bundle.zip'))
    local_pkg = Path(os.environ.get('DR_WORK', '/content/dr_thesis_work' if IN_COLAB else dr_root / 'work')).resolve() / 'mlwa_pkg'
    if (dr_root / 'experiments' / 'mlwa').is_dir():
        repo = dr_root
    elif bundle_zip.is_file():
        local_pkg.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip) as zf:
            zf.extractall(local_pkg)
        repo = local_pkg
    else:
        repo = Path.cwd().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _bootstrap_colab_repo()
    P0_DIR = None
    E0_DIR = None
    E1_DIR = None
    JOURNAL_RUNS_DIR = None
    OUTPUT_DIR = None
    SMOKE = False

# %%
import argparse
import json
from typing import Optional, Sequence

from experiments.mlwa.constants import resolve_runtime_paths
from experiments.mlwa.reporting import generate_journal_reports


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    paths = resolve_runtime_paths()
    res_root = paths['repo_root'] / 'result'
    parser = argparse.ArgumentParser(description='15: Generate verified-only MLWA journal tables, plots, and coverage audit.')
    parser.add_argument('--p0-dir', type=Path, default=paths['p0_run'])
    parser.add_argument('--e0-dir', type=Path, default=res_root / 'p0_fp32_main_lodo_v1_e0_resume_v2')
    parser.add_argument('--e1-dir', type=Path, default=res_root / 'p0_fp32_main_lodo_v1_e1_resume_v2')
    parser.add_argument('--journal-runs-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=paths['outputs'] / 'mlwa_journal_reports')
    parser.add_argument('--smoke', action='store_true', help='Generate coverage-only report into output-dir with no fabrication.')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        manifest = generate_journal_reports(
            output_dir=args.output_dir,
            p0_dir=None,
            e0_dir=None,
            e1_dir=None,
            journal_runs_dir=None,
        )
    else:
        manifest = generate_journal_reports(
            output_dir=args.output_dir,
            p0_dir=args.p0_dir,
            e0_dir=args.e0_dir,
            e1_dir=args.e1_dir,
            journal_runs_dir=args.journal_runs_dir,
        )
    print(json.dumps({
        'output_dir': manifest['output_dir'],
        'generated_tables': list(manifest['generated_tables'].keys()),
        'generated_figures': list(manifest['generated_figures'].keys()),
        'coverage_table': manifest['coverage_table'],
        'missing_runs_table': manifest['missing_runs_table'],
    }, indent=2))
    return 0


# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _res_root = _paths['repo_root'] / 'result'
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_journal_reports')
    _nb_manifest = generate_journal_reports(
        output_dir=_out,
        p0_dir=None if SMOKE else (P0_DIR if P0_DIR is not None else _paths['p0_run']),
        e0_dir=None if SMOKE else (E0_DIR if E0_DIR is not None else (_res_root / 'p0_fp32_main_lodo_v1_e0_resume_v2')),
        e1_dir=None if SMOKE else (E1_DIR if E1_DIR is not None else (_res_root / 'p0_fp32_main_lodo_v1_e1_resume_v2')),
        journal_runs_dir=None if SMOKE else JOURNAL_RUNS_DIR,
    )
    print(json.dumps({'output_dir': _nb_manifest['output_dir'], 'tables': list(_nb_manifest['generated_tables'].keys())}, indent=2))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
