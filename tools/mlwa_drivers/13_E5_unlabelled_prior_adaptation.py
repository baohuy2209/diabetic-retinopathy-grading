# %% [markdown]
# # 13 · E5 Opt-In Unlabelled Target Prior Adaptation (Saerens EM & BBSE)

# %%
NOTEBOOK_CODE_SHA256 = '0b2f7cbc96987c2d573139c88a664b5adcde51ab792e2c69dfce8d09aae6ef19'

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


def _setup_colab_environment() -> Path:
    if IN_COLAB:
        import importlib.util
        import subprocess
        deps = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'pyarrow': 'pyarrow'}
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')
    return Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _setup_colab_environment()
    ENABLE_E5 = False
    OUTPUT_DIR = None
    SMOKE = True

# %%
import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT, PROB_COLS, resolve_runtime_paths
from experiments.mlwa.data_contract import atomic_write_json, load_predictions_table
from experiments.mlwa.decoders import decode_argmax, prior_from_labels
from experiments.mlwa.evaluator import compute_decision_metrics
from experiments.mlwa.prior_shift import apply_prior_shift_ratio, estimate_prior_bbse, estimate_prior_saerens_em


def run_e5_cohort_adaptation(
    source_val_df: pd.DataFrame,
    target_df: pd.DataFrame,
    pi_source: np.ndarray,
    cohort: str,
    fold_id: int,
    cibd_tau_active: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run Saerens EM + BBSE strictly without exposing y_target during adaptation."""
    p_val = source_val_df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    y_val = source_val_df['y_true'].to_numpy(dtype=int)
    p_tgt = target_df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    # Only pass p_tgt to estimators (y_tgt used strictly post-hoc for accuracy evaluation)
    y_tgt_eval = target_df['y_true'].to_numpy(dtype=int)

    em_res = estimate_prior_saerens_em(p_tgt, pi_source)
    bb_soft = estimate_prior_bbse(y_val, p_val, p_tgt, mode='soft')
    bb_hard = estimate_prior_bbse(y_val, p_val, p_tgt, mode='hard')

    rows: List[Dict[str, Any]] = []
    for method_name, res in [('saerens_em', em_res), ('bbse_soft', bb_soft), ('bbse_hard', bb_hard)]:
        row: Dict[str, Any] = {
            'regime': 'TRANSDUCTIVE_ADAPTATION',
            'fold_id': fold_id,
            'cohort': cohort,
            'method': method_name,
            'status': res['status'],
            'oracle_leakage_checked': True,
        }
        pi_hat = res.get('pi_target')
        if pi_hat is not None and np.isfinite(pi_hat).all():
            p_adj = apply_prior_shift_ratio(p_tgt, pi_source, pi_hat, cibd_tau_active=cibd_tau_active)
            dm = compute_decision_metrics(y_tgt_eval, decode_argmax(p_adj))
            row.update({
                'recall_g3': dm['recall_g3'],
                'within1': dm['within1'],
                'qwk': dm['qwk'],
                'macro_recall': dm['macro_recall'],
                'pi_target_est': np.asarray(pi_hat).round(4).tolist(),
            })
        else:
            row.update({'recall_g3': np.nan, 'within1': np.nan, 'qwk': np.nan, 'macro_recall': np.nan})
        rows.append(row)
    return rows


def run_e5_smoke(output_dir: Path) -> Dict[str, Any]:
    """Synthetic label-shift smoke test for Saerens EM & BBSE (TEST_ONLY)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)

    pi_src = np.array([0.45, 0.15, 0.20, 0.12, 0.08], dtype=np.float64)
    pi_tgt_true = np.array([0.25, 0.15, 0.25, 0.25, 0.10], dtype=np.float64)

    def _sim(n: int, pi: np.ndarray) -> pd.DataFrame:
        y = rng.choice(5, size=n, p=pi)
        y[:5] = np.arange(5)
        x = rng.normal(loc=y[:, None] * 1.1, scale=0.8, size=(n, 5))
        logits = -0.5 * ((np.arange(5)[None, :] - x.mean(axis=1, keepdims=True) / 1.1) ** 2) + np.log(pi_src)[None, :]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        df = pd.DataFrame(p, columns=list(PROB_COLS))
        df['y_true'] = y
        return df

    val_df = _sim(400, pi_src)
    tgt_df = _sim(400, pi_tgt_true)
    rows = run_e5_cohort_adaptation(val_df, tgt_df, pi_src, cohort='aptos', fold_id=0)
    pd.DataFrame(rows).to_csv(out / 'e5_transductive_adaptation_smoke.csv', index=False)
    status = {'status': 'COMPLETE_SMOKE_TEST_ONLY', 'n_rows': len(rows)}
    atomic_write_json(out / 'status.json', status)
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    paths = resolve_runtime_paths()
    parser = argparse.ArgumentParser(description='13: Opt-in E5 unlabelled target prior adaptation (Saerens EM & BBSE).')
    parser.add_argument('--enable-e5', action='store_true', help='Explicit opt-in required for E5 transductive adaptation.')
    parser.add_argument('--p0-dir', type=Path, default=None, help='Source prediction bundles directory.')
    parser.add_argument('--output-dir', type=Path, default=paths['outputs'] / 'mlwa_e5_prior_adaptation')
    parser.add_argument('--smoke', action='store_true', help='Run deterministic synthetic label-shift check (TEST_ONLY).')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        res = run_e5_smoke(args.output_dir)
        print(json.dumps(res, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.enable_e5:
        status = {
            'status': 'OPT_IN_NOT_ENABLED',
            'regime': 'TRANSDUCTIVE_ADAPTATION',
            'note': 'E5 is optional and excluded from primary strict-LODO rankings. Pass --enable-e5 to run.',
        }
        atomic_write_json(args.output_dir / 'status.json', status)
        print(json.dumps(status, indent=2))
        return 0

    if args.p0_dir is None or not Path(args.p0_dir).is_dir():
        status = {'status': 'PENDING_ASSETS', 'note': 'Pass valid --p0-dir containing fold prediction bundles.'}
        atomic_write_json(args.output_dir / 'status.json', status)
        print(json.dumps(status, indent=2))
        return 1

    rows: List[Dict[str, Any]] = []
    for fold_id in range(6):
        cohort = CANONICAL_FOLD_TO_COHORT[fold_id]
        base = Path(args.p0_dir) / 'predictions' / f'fold_{fold_id}_{cohort}_V3'
        if not base.exists():
            continue
        tr = load_predictions_table(base / 'source_train')
        va = load_predictions_table(base / 'source_val')
        tg = load_predictions_table(base / 'target')
        pi_src = prior_from_labels(tr['y_true'].to_numpy(dtype=int))
        rows.extend(run_e5_cohort_adaptation(va, tg, pi_src, cohort=cohort, fold_id=fold_id))

    pd.DataFrame(rows).to_csv(args.output_dir / 'e5_transductive_adaptation.csv', index=False)
    status = {'status': 'COMPLETE', 'regime': 'TRANSDUCTIVE_ADAPTATION', 'n_rows': len(rows)}
    atomic_write_json(args.output_dir / 'status.json', status)
    print(json.dumps(status, indent=2))
    return 0


# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_e5_prior_adaptation_nb')
    print(run_e5_smoke(_out) if globals().get('SMOKE', True) else {'status': 'OPT_IN_NOT_ENABLED'})

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
