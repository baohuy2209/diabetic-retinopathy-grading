# %% [markdown]
# # 12 · E3 Gated Bounded Residual Correction (H3 Gate & Controls)

# %%
NOTEBOOK_CODE_SHA256 = '2f862b36fd69734c83c90013681eaf3588f894bd6402f652f612f3ebb55a49e6'

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


if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _bootstrap_colab_repo()
    MODE = 'gate_check'
    SMOKE = True

# %%
import argparse
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.mlwa.brc import evaluate_h3_gate, run_e3_with_controls
from experiments.mlwa.constants import PROB_COLS, STATUS_SKIPPED_GATE_NOT_MET, resolve_runtime_paths
from experiments.mlwa.data_contract import atomic_write_json


def run_e3_smoke(output_dir: Path) -> Dict[str, Any]:
    """Run synthetic E3 BRC + 7 controls verification cleanly on CPU (TEST_ONLY)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)

    def _mk(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        z = rng.normal(size=(n, 16)).astype(np.float32)
        s = z[:, 0]
        raw = rng.normal(size=(n, 5))
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        p = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        y = rng.integers(0, 5, size=n)
        y[:5] = np.arange(5)
        df = pd.DataFrame({
            'y_true': y,
            'patient_key': [f'pat_{i // 2}' for i in range(n)],
            'cohort': ['aptos' if i % 2 == 0 else 'idrid' for i in range(n)],
        })
        return z, s, p, df

    z_tr, s_tr, p_tr, df_tr = _mk(90)
    z_va, s_va, p_va, df_va = _mk(50)
    z_tg, s_tg, p_tg, df_tg = _mk(50)

    ctrl_df = run_e3_with_controls(
        z_train=z_tr, s_train=s_tr, p_train=p_tr, source_train_df=df_tr,
        z_val=z_va, s_val=s_va, p_val=p_va, y_val=df_va['y_true'].to_numpy(),
        z_target=z_tg, s_target=s_tg, p_target=p_tg, y_target=df_tg['y_true'].to_numpy(),
        beta_grid=(0.1, 0.5), seed=42,
    )
    ctrl_df.to_csv(out / 'e3_smoke_controls.csv', index=False)

    closed_gate = evaluate_h3_gate(
        pd.DataFrame([
            {'fold_id': 0, 'representation': 'z_v3', 'recall_g3': 0.51, 'macro_recall': 0.50},
            {'fold_id': 0, 'representation': 's_v3', 'recall_g3': 0.54, 'macro_recall': 0.52},
        ]),
        evidence_type='retrospective_legacy',
    )
    status = {'status': 'COMPLETE_SMOKE_TEST_ONLY', 'closed_gate_check': closed_gate, 'n_controls': len(ctrl_df)}
    atomic_write_json(out / 'status.json', status)
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    paths = resolve_runtime_paths()
    parser = argparse.ArgumentParser(description='12: Gated E3 Bounded Residual Correction (H3 gate enforced).')
    parser.add_argument('--probe-csv', type=Path, default=None, help='Path to E1 probe metrics CSV to evaluate H3 gate.')
    parser.add_argument('--evidence-type', type=str, default='retrospective_legacy', choices=['retrospective_legacy', 'journal_multiseed'])
    parser.add_argument('--output-dir', type=Path, default=paths['outputs'] / 'mlwa_e3_brc')
    parser.add_argument('--force-smoke-test-only', action='store_true', help='Run synthetic E3+controls check (TEST_ONLY).')
    parser.add_argument('--smoke', action='store_true', help='Alias for --force-smoke-test-only.')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.force_smoke_test_only or args.smoke:
        res = run_e3_smoke(args.output_dir)
        print(json.dumps(res, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.probe_csv is not None and Path(args.probe_csv).is_file():
        probe_df = pd.read_csv(args.probe_csv)
        gate = evaluate_h3_gate(probe_df, evidence_type=args.evidence_type)
    else:
        gate = {
            'gate_open': False,
            'status': STATUS_SKIPPED_GATE_NOT_MET,
            'reason': 'H3 gate closed by default (legacy E1 s_v3 > z_v3; multi-seed E2 probe CSV not supplied).',
            'evidence_type': args.evidence_type,
        }
    atomic_write_json(args.output_dir / 'h3_gate_status.json', gate)
    print(json.dumps(gate, indent=2))
    return 0


# %%
if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _out = resolve_runtime_paths()['outputs'] / 'mlwa_e3_brc_nb'
    print(run_e3_smoke(_out) if globals().get('SMOKE', True) else {'status': STATUS_SKIPPED_GATE_NOT_MET})

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
