# %% [markdown]
# # 11 · Journal Decoders (E0 / CIBD) & Frozen-Encoder Representation Probes (E1)

# %%
NOTEBOOK_CODE_SHA256 = 'c372b9d6d726584dfcf972aafc4682b8404c43d04b88ee4619a5aebc180d738d'

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
    MODE = 'smoke'  # 'smoke' | 'run'
    SMOKE = True

# %%
import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.mlwa.constants import (
    CANONICAL_FOLD_TO_COHORT,
    PROB_COLS,
    THETA_COLS,
    resolve_runtime_paths,
)
from experiments.mlwa.data_contract import atomic_write_json, load_predictions_table, validate_p0_cache
from experiments.mlwa.decoders import (
    bootstrap_cibd_selector_stability,
    decode_argmax,
    decode_cibd,
    decode_cutpoint_latent,
    decode_native_legacy_round,
    decode_posterior_median,
    decode_rounded_expectation_bayes,
    prior_from_labels,
    select_cibd,
)
from experiments.mlwa.evaluator import (
    calibrate_rdr_threshold,
    compute_decision_metrics,
    compute_posterior_metrics,
    evaluate_rdr_screening,
)
from experiments.mlwa.probes import run_source_selected_probe

DECODERS_LIST = (
    'native',
    'argmax',
    'posterior_median',
    'rounded_expectation_bayes',
    'cibd_legacy_1pp',
    'cibd_dual_guardrail',
    'cibd_within1_05pp',
)


def _extract_probs_and_preds(df: pd.DataFrame, head_kind: str = 'clm', allow_cutpoint: bool = True) -> Dict[str, np.ndarray]:
    probs = df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    out: Dict[str, np.ndarray] = {
        'native': decode_native_legacy_round(probs),
        'argmax': decode_argmax(probs),
        'posterior_median': decode_posterior_median(probs),
        'rounded_expectation_bayes': decode_rounded_expectation_bayes(probs),
    }
    if allow_cutpoint and head_kind == 'clm' and set(THETA_COLS).issubset(df.columns):
        s_col = 's' if 's' in df.columns else ('latent_score' if 'latent_score' in df.columns else None)
        if s_col is not None:
            thetas = df[list(THETA_COLS)].to_numpy(dtype=np.float64)
            latent = df[s_col].to_numpy(dtype=np.float64)
            if np.isfinite(thetas).all() and np.isfinite(latent).all():
                out['cutpoint_latent'] = decode_cutpoint_latent(latent, thetas)
    return out


def run_decoder_evaluation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fold_id: int,
    cohort: str,
    unit_id: str,
    head_kind: str = 'clm',
    baseline_col: str = 'pred_native',
    stability_reps: int = 1000,
    seed_resampling: int = 42,
    include_exploratory_grid: bool = True,
) -> Dict[str, Any]:
    """Evaluate all fixed-posterior decoders, source-selected CIBD variants, RDR screening, and stability."""
    prior = prior_from_labels(train_df['y_true'].to_numpy(dtype=int))
    y_val = val_df['y_true'].to_numpy(dtype=int)
    p_val = val_df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    y_tgt = target_df['y_true'].to_numpy(dtype=int)
    p_tgt = target_df[list(PROB_COLS)].to_numpy(dtype=np.float64)

    # Calibrate RDR threshold strictly on source_val (95% specificity target)
    try:
        rdr_thr = calibrate_rdr_threshold(y_val, p_val, target_specificity=0.95)
        rdr_tgt = evaluate_rdr_screening(y_tgt, p_tgt, threshold=float(rdr_thr))
    except ValueError:
        rdr_thr = float('nan')
        rdr_tgt = {'rdr_sensitivity': float('nan'), 'rdr_specificity': float('nan')}
    post_tgt = compute_posterior_metrics(y_tgt, p_tgt)

    # Select CIBD strictly on source_val under versioned selectors
    sel_legacy, _ = select_cibd(val_df, prior, selector_id='legacy_within1_1pp_v1', baseline_col=baseline_col)
    sel_dual, _ = select_cibd(val_df, prior, selector_id='journal_dual_guardrail_v1', baseline_col=baseline_col)
    sel_05, _ = select_cibd(val_df, prior, selector_id='journal_within1_05pp_v1', baseline_col=baseline_col)
    selectors = {
        'cibd_legacy_1pp': sel_legacy,
        'cibd_dual_guardrail': sel_dual,
        'cibd_within1_05pp': sel_05,
    }

    stability = bootstrap_cibd_selector_stability(
        val_df,
        fixed_prior=prior,
        selector_id='journal_dual_guardrail_v1',
        reps=int(stability_reps),
        seed=int(seed_resampling) + fold_id,
    )

    tgt_preds = _extract_probs_and_preds(target_df, head_kind=head_kind, allow_cutpoint=(head_kind == 'clm'))
    for dec_name, sel in selectors.items():
        if sel.get('fallback'):
            tgt_preds[dec_name] = tgt_preds['native']
        else:
            tgt_preds[dec_name] = decode_cibd(p_tgt, prior, sel['alpha'], sel['tau'], sel['q'])

    decision_rows: List[Dict[str, Any]] = []
    for dec_name, pred in tgt_preds.items():
        dm = compute_decision_metrics(y_tgt, pred)
        dm.pop('confusion_matrix', None)
        decision_rows.append({
            'unit_id': unit_id,
            'fold_id': fold_id,
            'cohort': cohort,
            'decoder': dec_name,
            **{k: v for k, v in post_tgt.items()},
            'rdr_threshold_source_val': rdr_thr,
            'rdr_target_sens': rdr_tgt['rdr_sensitivity'],
            'rdr_target_spec': rdr_tgt['rdr_specificity'],
            **dm,
        })

    # Exploratory upper-bound grid on target (strictly labeled EXPLORATORY_NOT_SELECTED)
    exploratory_rows: List[Dict[str, Any]] = []
    if include_exploratory_grid:
        for alpha in (0.0, 0.5, 1.0):
            for tau in (0.0, 0.25, 0.5, 0.75):
                for q in (1, 2):
                    pred_exp = decode_cibd(p_tgt, prior, alpha=alpha, tau=tau, q=q)
                    dm = compute_decision_metrics(y_tgt, pred_exp)
                    exploratory_rows.append({
                        'unit_id': unit_id,
                        'fold_id': fold_id,
                        'cohort': cohort,
                        'label': 'EXPLORATORY_TARGET_GRID_NOT_SELECTED',
                        'alpha': alpha,
                        'tau': tau,
                        'q': q,
                        'recall_g3': dm['recall_g3'],
                        'within1': dm['within1'],
                        'qwk': dm['qwk'],
                        'macro_recall': dm['macro_recall'],
                    })

    return {
        'decision_rows': decision_rows,
        'selectors': selectors,
        'stability': stability,
        'exploratory_rows': exploratory_rows,
    }


def run_synthetic_smoke(output_dir: Path) -> Dict[str, Any]:
    """Run synthetic E0+E1 smoke verification cleanly on CPU (TEST_ONLY)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    all_dec_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    exp_rows: List[Dict[str, Any]] = []
    stability_summary: Dict[str, Any] = {}

    for fold_id in range(2):
        cohort = CANONICAL_FOLD_TO_COHORT[fold_id]

        def _make_df(n: int, shift: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
            z = rng.normal(loc=shift, size=(n, 16)).astype(np.float32)
            logits = rng.normal(size=(n, 5))
            logits[:, 3] += z[:, 0] * 1.1
            logits[:, 2] += z[:, 1] * 0.8
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            y = np.array([rng.choice(5, p=p) for p in probs], dtype=int)
            # Ensure all 5 classes appear
            y[:5] = np.arange(5)
            df = pd.DataFrame(probs, columns=list(PROB_COLS))
            df['y_true'] = y
            df['image_key'] = [f'img_{fold_id}_{i}' for i in range(n)]
            df['patient_key'] = [f'pat_{fold_id}_{i}' for i in range(n)]
            df['cohort'] = [CANONICAL_FOLD_TO_COHORT[(fold_id + 1 + i) % 6] for i in range(n)]
            df['pred_native'] = decode_argmax(probs)
            df['latent_score'] = z[:, 0]
            for j, tc in enumerate(THETA_COLS):
                df[tc] = -1.5 + j * 1.0
            return df, z

        tr_df, z_tr = _make_df(120, shift=0.0)
        va_df, z_va = _make_df(60, shift=0.05)
        tg_df, z_tg = _make_df(60, shift=0.15)

        dec_res = run_decoder_evaluation(
            tr_df, va_df, tg_df, fold_id=fold_id, cohort=cohort,
            unit_id=f'smoke_clm_f{fold_id}', head_kind='clm',
            stability_reps=50, seed_resampling=42,
        )
        all_dec_rows.extend(dec_res['decision_rows'])
        exp_rows.extend(dec_res['exploratory_rows'])
        stability_summary[f'fold_{fold_id}'] = dec_res['stability']

        for rep_id, x_tr, x_va, x_tg in [
            ('z_v3', z_tr, z_va, z_tg),
            ('s_v3', z_tr[:, :1], z_va[:, :1], z_tg[:, :1]),
        ]:
            pr = run_source_selected_probe(
                x_tr, tr_df['y_true'].to_numpy(),
                x_va, va_df['y_true'].to_numpy(),
                x_tg, tg_df['y_true'].to_numpy(),
                representation_id=rep_id, fold_id=fold_id, cohort=cohort,
                c_grid=(0.1, 1.0, 10.0), max_iter=300,
            )
            pr_ctrl = run_source_selected_probe(
                x_tr, tr_df['y_true'].to_numpy(),
                x_va, va_df['y_true'].to_numpy(),
                x_tg, tg_df['y_true'].to_numpy(),
                representation_id=f'{rep_id}_perm_ctrl', fold_id=fold_id, cohort=cohort,
                c_grid=(0.1, 1.0), max_iter=300,
                control_permutation_seed=777 + fold_id,
                source_train_cohorts=tr_df['cohort'].tolist(),
            )
            for row in (pr, pr_ctrl):
                dm = row.get('target_decision_metrics', {})
                probe_rows.append({
                    'status': row['status'],
                    'representation': row['representation'],
                    'fold_id': fold_id,
                    'cohort': cohort,
                    'selected_C': row.get('selected_C'),
                    'recall_g3': dm.get('recall_g3'),
                    'macro_recall': dm.get('macro_recall'),
                    'within1': dm.get('within1'),
                })

    pd.DataFrame(all_dec_rows).to_csv(out / 'e0_decision_metrics.csv', index=False)
    pd.DataFrame(exp_rows).to_csv(out / 'e0_exploratory_target_grid.csv', index=False)
    pd.DataFrame(probe_rows).to_csv(out / 'e1_probe_metrics.csv', index=False)
    status = {'status': 'COMPLETE_SMOKE_TEST_ONLY', 'stability_summary': stability_summary}
    atomic_write_json(out / 'status.json', status)
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    paths = resolve_runtime_paths()
    parser = argparse.ArgumentParser(description='11: Refreshed MLWA decoders (E0) and frozen-encoder probes (E1).')
    parser.add_argument('--p0-dir', type=Path, default=None, help='Optional cached P0 directory.')
    parser.add_argument('--journal-runs-dir', type=Path, default=None, help='Completed journal training units directory.')
    parser.add_argument('--output-dir', type=Path, default=paths['outputs'] / 'mlwa_e0_e1_refreshed')
    parser.add_argument('--stability-reps', type=int, default=1000)
    parser.add_argument('--smoke', action='store_true', help='Run deterministic synthetic CPU verification (TEST_ONLY).')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        res = run_synthetic_smoke(args.output_dir)
        print(json.dumps({'mode': 'smoke', 'status': res['status'], 'output_dir': str(args.output_dir)}, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    p0_dir = args.p0_dir
    if p0_dir is not None and Path(p0_dir).is_dir():
        audit = validate_p0_cache(Path(p0_dir), require_features=True)
        atomic_write_json(args.output_dir / 'p0_cache_audit.json', audit)
        if not audit.get('ok'):
            print(json.dumps({'status': 'P0_CACHE_INCOMPLETE', 'audit': audit}, indent=2))
            return 1
        rows: List[Dict[str, Any]] = []
        for fold_id in range(6):
            cohort = CANONICAL_FOLD_TO_COHORT[fold_id]
            for variant in ('V2', 'V3'):
                base = Path(p0_dir) / 'predictions' / f'fold_{fold_id}_{cohort}_{variant}'
                if not base.exists():
                    continue
                tr = load_predictions_table(base / 'source_train')
                va = load_predictions_table(base / 'source_val')
                tg = load_predictions_table(base / 'target')
                res = run_decoder_evaluation(
                    tr, va, tg, fold_id=fold_id, cohort=cohort,
                    unit_id=f'p0_{variant}_f{fold_id}', head_kind='clm' if variant == 'V3' else 'softmax',
                    stability_reps=args.stability_reps,
                )
                rows.extend(res['decision_rows'])
        if rows:
            pd.DataFrame(rows).to_csv(args.output_dir / 'p0_refreshed_decoder_metrics.csv', index=False)
        atomic_write_json(args.output_dir / 'status.json', {'status': 'COMPLETE', 'n_rows': len(rows)})
        print(json.dumps({'status': 'COMPLETE', 'n_rows': len(rows)}, indent=2))
        return 0

    status = {'status': 'NO_RUN_INPUTS_SUPPLIED', 'note': 'Pass --p0-dir or --journal-runs-dir, or --smoke for verification.'}
    atomic_write_json(args.output_dir / 'status.json', status)
    print(json.dumps(status, indent=2))
    return 0


# %%
if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _out = resolve_runtime_paths()['outputs'] / 'mlwa_e0_e1_refreshed_nb'
    print(run_synthetic_smoke(_out) if globals().get('SMOKE', True) else {'status': 'SET_INPUTS_FOR_RUN'})

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
