# %% [markdown]
# # Thesis Supplementary CPU Analyses (A–G)
#
# Post-hoc CPU verification and sensitivity analyses supporting Chapters 4, 5, 6, and the Appendices.
# All primary P0, E0, and E1 artifacts and archived predictions are treated as read-only inputs;
# outputs are written to `result/thesis_supplementary_v1/` by default.

# %%
NOTEBOOK_CODE_SHA256 = 'acc2565e07e4b096a8d41479bbcc47e9a3dd22cf0b2d931866c52a8e10111168'

# %%
import os
import sys
from pathlib import Path

# Edit these defaults in Colab, or set DR_* environment variables before running Python.
IN_COLAB: bool = (
    'COLAB_RELEASE_TAG' in os.environ
    or 'COLAB_BACKEND_VERSION' in os.environ
    or 'google.colab' in sys.modules
)


def _running_in_notebook() -> bool:
    """Return True only when executing inside an interactive IPython/Jupyter/Colab kernel."""
    if 'ipykernel' not in sys.modules and 'google.colab' not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


if __name__ == '__main__' and _running_in_notebook():
    import importlib.util
    import subprocess

    if IN_COLAB:
        _deps = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy'}
        _missing = [pkg for mod, pkg in _deps.items() if importlib.util.find_spec(mod) is None]
        if _missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *_missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')

    # Same workspace conventions as 01_E0_decision_optimization_resume_v2 and 02_E1_representation_probes_resume_v2.
    THESIS_ROOT = Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()
    INPUTS = Path(os.environ.get('DR_INPUTS', THESIS_ROOT / 'inputs')).resolve()
    OUTPUTS = Path(os.environ.get('DR_OUTPUTS', THESIS_ROOT / 'outputs')).resolve()
    WORK = Path(os.environ.get('DR_WORK', '/content/dr_thesis_work' if IN_COLAB else THESIS_ROOT / 'work')).resolve()

    # Override any path below in Colab (or leave None to auto-resolve DR_* / Drive / repo result/).
    P0_RUN = None  # e.g. Path('/content/drive/MyDrive/mlwa_journals_p0_e0_e1/p0_fp32_main_lodo_v1')
    E0_RUN = None
    E1_RUN = None
    REFERENCE_PATH = None
    RAW_PREDS = None
    FOLD_SOURCE = None
    OUTPUT_ID = os.environ.get('DR_ANALYSIS_ID', 'thesis_supplementary_v1')
    OUTPUT_DIR = None  # Defaults to OUTPUTS/mlwa/OUTPUT_ID on Colab or result/OUTPUT_ID locally.
    SEED = int(os.environ.get('DR_SEED', 42))
    BOOTSTRAP_REPS = int(os.environ.get('DR_BOOTSTRAP_REPS', 2000))
    ANALYSES_TO_RUN = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# %%
import argparse
import hashlib
import importlib.metadata as package_metadata
import json
import subprocess
import warnings
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

COHORTS: Tuple[str, ...] = ('aptos', 'idrid', 'messidor2', 'ddr', 'eyepacs', 'deepdrid')
COHORT_LABELS: Dict[str, str] = {
    'aptos': 'APTOS 2019',
    'idrid': 'IDRiD',
    'messidor2': 'Messidor-2',
    'ddr': 'DDR',
    'eyepacs': 'EyePACS',
    'deepdrid': 'DeepDRiD',
}
VARIANT_NAMES: Dict[str, str] = {
    'V2': 'V2_PlusGlobalContext_H2',
    'V3': 'V3_PlusOrdinalHead_H3_Champion',
}
PROB_COLS: List[str] = [f'p{k}' for k in range(5)]
C_GRID: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)
ALL_ANALYSES: Tuple[str, ...] = ('A', 'B', 'C', 'D', 'E', 'F', 'G')


def find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from start (or cwd/file) to locate the repository root, or None in standalone Colab."""
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    if '__file__' in globals():
        candidates.append(Path(__file__).resolve().parent)
    for base in candidates:
        for cur in (base, *base.parents):
            if (cur / 'pyproject.toml').is_file() and (cur / 'workflows' / 'mlwa').is_dir():
                return cur
    return None


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _first_existing(candidates: Sequence[Optional[Path]]) -> Optional[Path]:
    for cand in candidates:
        if cand is not None and Path(cand).expanduser().exists():
            return Path(cand).expanduser().resolve()
    return None


def _maybe_stage_zip(source: Optional[Path], work_dir: Path, expected_suffix: str) -> Optional[Path]:
    """Return directory as-is or extract a ZIP archive safely into work_dir."""
    if source is None:
        return None
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == '.zip':
        dest = work_dir / source.stem
        marker = dest / '.archive-complete'
        with zipfile.ZipFile(source) as zf:
            sig = hashlib.sha256(
                json.dumps([(m.filename, m.file_size, m.CRC) for m in zf.infolist()]).encode()
            ).hexdigest()
            if marker.is_file() and marker.read_text(encoding='utf-8') == sig:
                return dest
            dest.mkdir(parents=True, exist_ok=True)
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError(f'Unsafe ZIP member: {member.filename}')
            zf.extractall(dest)
            marker.write_text(sig, encoding='utf-8')
        matches = list(dest.rglob(f'*{expected_suffix}'))
        return matches[0].parent if matches else dest
    return source


def capture_code_provenance(
    repo_root: Optional[Path],
    execution_mode: str = 'module_api',
) -> Dict[str, Any]:
    """Capture code path, start SHA-256, and Git metadata at execution startup."""
    code_path: Optional[Path] = None
    script_path: Optional[Path] = Path(__file__).resolve() if '__file__' in globals() else None
    if execution_mode == 'notebook' and repo_root is not None:
        nb_cand = repo_root / 'notebooks' / 'mlwa' / '03_thesis_supplementary_analyses.ipynb'
        if nb_cand.is_file():
            code_path = nb_cand.resolve()
    if code_path is None and script_path is not None and script_path.is_file():
        code_path = script_path

    code_sha_start = hashlib.sha256(code_path.read_bytes()).hexdigest() if (code_path and code_path.is_file()) else NOTEBOOK_CODE_SHA256
    script_sha_start = hashlib.sha256(script_path.read_bytes()).hexdigest() if (script_path and script_path.is_file()) else None

    git_commit: Optional[str] = None
    git_dirty_relevant: Optional[bool] = None
    git_dirty_paths: List[str] = []
    if repo_root is not None and (repo_root / '.git').exists():
        try:
            rev = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(repo_root), capture_output=True, text=True, timeout=5, check=True,
            )
            git_commit = rev.stdout.strip()
            st = subprocess.run(
                [
                    'git', 'status', '--porcelain', '--',
                    'workflows/mlwa/03_thesis_supplementary_analyses.py',
                    'notebooks/mlwa/03_thesis_supplementary_analyses.ipynb',
                ],
                cwd=str(repo_root), capture_output=True, text=True, timeout=5, check=True,
            )
            lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
            git_dirty_relevant = len(lines) > 0
            git_dirty_paths = lines
        except Exception:
            pass

    return {
        'execution_mode': execution_mode,
        'code_path': str(code_path) if code_path else None,
        'code_sha256_start': code_sha_start,
        'script_sha256_start': script_sha_start,
        'notebook_embedded_code_sha256': NOTEBOOK_CODE_SHA256,
        'git_commit': git_commit,
        'git_dirty_relevant_files': git_dirty_relevant,
        'git_dirty_paths': git_dirty_paths,
    }


@dataclass
class AnalysisContext:
    repo_root: Optional[Path]
    p0_dir: Path
    e0_dir: Path
    e1_dir: Path
    output_dir: Path
    reference_json: Optional[Path]
    archived_preds_dir: Optional[Path]
    folds_dir: Optional[Path]
    seed: int = 42
    bootstrap_reps: int = 2000
    execution_mode: str = 'module_api'
    code_provenance: Dict[str, Any] = field(default_factory=dict)


def build_context(
    repo_root: Optional[Path] = None,
    p0_dir: Optional[Path] = None,
    e0_dir: Optional[Path] = None,
    e1_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    reference_json: Optional[Path] = None,
    archived_preds_dir: Optional[Path] = None,
    folds_dir: Optional[Path] = None,
    seed: int = 42,
    bootstrap_reps: int = 2000,
    execution_mode: str = 'module_api',
) -> AnalysisContext:
    """Resolve read-only input roots and dedicated output root using repo paths, DR_* env vars, or Colab Drive."""
    root = find_repo_root(repo_root)
    thesis_root = _env_path('DR_ROOT') or (Path('/content/drive/MyDrive/dr_thesis') if IN_COLAB else (root or Path.cwd()))
    inputs_root = _env_path('DR_INPUTS') or (thesis_root / 'inputs')
    outputs_root = _env_path('DR_OUTPUTS') or (thesis_root / 'outputs')
    work_root = _env_path('DR_WORK') or (Path('/content/dr_thesis_work') if IN_COLAB else (thesis_root / 'work'))
    colab_drive_mlwa = Path('/content/drive/MyDrive/mlwa_journals_p0_e0_e1')
    ext_mlwa = Path(r'D:\Github\deep-learning-for-computer-vision\notebooks\mlwa-journals\outputs')
    ext_study = Path(r'D:\Github\DR-evaluation-study2026\result')
    ext_folds = Path(r'D:\Github\deep-learning-for-computer-vision\data\processed\lodo_folds')

    resolved_p0 = _first_existing([
        p0_dir,
        _env_path('DR_P0_RUN'),
        (root / 'result' / 'p0_fp32_main_lodo_v1') if root else None,
        outputs_root / 'mlwa' / 'p0_fp32_main_lodo_v1',
        outputs_root / 'mlwa' / 'p0_thesis_fp32',
        colab_drive_mlwa / 'p0_fp32_main_lodo_v1',
        ext_mlwa / 'p0_fp32_main_lodo_v1',
    ])
    p0_parent = resolved_p0.parent if resolved_p0 is not None else None
    p0_name = resolved_p0.name if resolved_p0 is not None else 'p0_fp32_main_lodo_v1'

    resolved_e0 = _first_existing([
        e0_dir,
        _env_path('DR_E0_RUN'),
        (root / 'result' / 'p0_fp32_main_lodo_v1_e0_resume_v2') if root else None,
        (p0_parent / f'{p0_name}_e0_resume_v2') if p0_parent else None,
        (p0_parent / f'{p0_name}_e0_thesis_resume_v2') if p0_parent else None,
        outputs_root / 'mlwa' / f'{p0_name}_e0_resume_v2',
        outputs_root / 'mlwa' / f'{p0_name}_e0_thesis_resume_v2',
        colab_drive_mlwa / 'p0_fp32_main_lodo_v1_e0_resume_v2',
        ext_mlwa / 'p0_fp32_main_lodo_v1_e0_resume_v2',
    ])
    resolved_e1 = _first_existing([
        e1_dir,
        _env_path('DR_E1_RUN'),
        (root / 'result' / 'p0_fp32_main_lodo_v1_e1_resume_v2') if root else None,
        (p0_parent / f'{p0_name}_e1_resume_v2') if p0_parent else None,
        (p0_parent / f'{p0_name}_e1_thesis_resume_v2') if p0_parent else None,
        outputs_root / 'mlwa' / f'{p0_name}_e1_resume_v2',
        outputs_root / 'mlwa' / f'{p0_name}_e1_thesis_resume_v2',
        colab_drive_mlwa / 'p0_fp32_main_lodo_v1_e1_resume_v2',
        ext_mlwa / 'p0_fp32_main_lodo_v1_e1_resume_v2',
    ])
    if resolved_p0 is None or resolved_e0 is None or resolved_e1 is None:
        raise FileNotFoundError(
            f'Missing required P0/E0/E1 caches: p0={resolved_p0}, e0={resolved_e0}, e1={resolved_e1}. '
            'Set DR_P0_RUN / DR_E0_RUN / DR_E1_RUN or edit P0_RUN / E0_RUN / E1_RUN in the Colab config cell.'
        )

    resolved_ref = _first_existing([
        reference_json,
        _env_path('DR_REFERENCE'),
        (root / 'result' / 'reference' / 'lodo_summary_with_calibrated_screening.json') if root else None,
        inputs_root / 'reference' / 'lodo_summary_with_calibrated_screening.json',
        ext_study / 'reference' / 'lodo_summary_with_calibrated_screening.json',
    ])
    raw_cand = _first_existing([
        archived_preds_dir,
        _env_path('DR_RAW_PREDS'),
        (root / 'result' / 'raw_preds') if root else None,
        inputs_root / 'raw_preds',
        inputs_root / 'raw_preds.zip',
        ext_study / 'raw_preds',
    ])
    resolved_raw = _maybe_stage_zip(raw_cand, work_root / 'supplementary_archives', '.npz')
    folds_cand = _first_existing([
        folds_dir,
        _env_path('DR_FOLDS'),
        (root / 'result' / 'manifests' / 'lodo_folds') if root else None,
        inputs_root / 'lodo_folds',
        inputs_root / 'lodo_folds.zip',
        ext_folds,
    ])
    resolved_folds = _maybe_stage_zip(folds_cand, work_root / 'supplementary_archives', '_test.csv')

    output_id = os.environ.get('DR_ANALYSIS_ID', 'thesis_supplementary_v1')
    env_out = _env_path('DR_SUPP_OUTPUT')
    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
    elif env_out is not None:
        out = env_out.resolve()
    elif IN_COLAB or os.environ.get('DR_OUTPUTS'):
        out = (outputs_root / 'mlwa' / output_id).resolve()
    elif root is not None:
        out = (root / 'result' / output_id).resolve()
    else:
        out = (outputs_root / 'mlwa' / output_id).resolve()

    for src in (resolved_p0, resolved_e0, resolved_e1):
        if out == src or out.is_relative_to(src):
            raise ValueError(f'Output directory {out} must not overlap read-only cache {src}')
    out.mkdir(parents=True, exist_ok=True)
    prov_start = capture_code_provenance(root, execution_mode=execution_mode)
    return AnalysisContext(
        repo_root=root,
        p0_dir=resolved_p0,
        e0_dir=resolved_e0,
        e1_dir=resolved_e1,
        output_dir=out,
        reference_json=resolved_ref,
        archived_preds_dir=resolved_raw,
        folds_dir=resolved_folds,
        seed=int(seed),
        bootstrap_reps=int(bootstrap_reps),
        execution_mode=execution_mode,
        code_provenance=prov_start,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute exact five-grade metrics on 0-1 unit scale matching E0/E1 definitions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y, p, labels=np.arange(5))
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.full(5, np.nan), where=support > 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        qwk = cohen_kappa_score(y, p, labels=np.arange(5), weights='quadratic')
    n3 = int(support[3])
    pred3 = int(cm[:, 3].sum())
    out: Dict[str, Any] = {
        'n': int(len(y)),
        'qwk': float(qwk),
        'accuracy': float(np.mean(y == p)),
        'within1': float(np.mean(np.abs(y - p) <= 1)),
        'mae': float(np.mean(np.abs(y - p))),
        'macro_recall': float(np.mean(recall)),
        'g3_precision': float(cm[3, 3] / pred3) if pred3 > 0 else np.nan,
        'g3_precision_defined': bool(pred3 > 0),
        'true_g3': n3,
        'predicted_g3': pred3,
        'predicted_true_g3_ratio': float(pred3 / n3) if n3 > 0 else np.nan,
    }
    for k in range(5):
        out[f'recall_g{k}'] = float(recall[k])
        out[f'support_g{k}'] = int(support[k])
        out[f'correct_g{k}'] = int(cm[k, k])
    return out


def _read_p0_predictions(ctx: AnalysisContext, variant: str, fold: int, split: str) -> pd.DataFrame:
    cohort = COHORTS[fold]
    path = ctx.p0_dir / 'lodo' / f'{variant}_fold_{fold}_{cohort}' / split / 'predictions.csv'
    df = pd.read_csv(path, dtype={'image_key': str, 'image_id': str, 'patient_key': str})
    if df['image_key'].duplicated().any():
        raise ValueError(f'Duplicate image_key in {path}')
    return df


def _merge_on_image_key(left: pd.DataFrame, right: pd.DataFrame, right_cols: Sequence[str], label: str) -> pd.DataFrame:
    if left['image_key'].duplicated().any() or right['image_key'].duplicated().any():
        raise ValueError(f'Duplicate image_key detected before merge ({label})')
    merged = left.merge(right[['image_key', 'y_true', *right_cols]], on='image_key', how='inner', suffixes=('', '_rhs'))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f'Coverage mismatch in {label}: left={len(left)}, right={len(right)}, merged={len(merged)}')
    if not np.array_equal(merged['y_true'].to_numpy(int), merged['y_true_rhs'].to_numpy(int)):
        raise ValueError(f'Ground-truth label mismatch after image_key merge ({label})')
    return merged.drop(columns=['y_true_rhs'])


def verify_baselines(ctx: AnalysisContext) -> pd.DataFrame:
    """Recompute P0, E0, and E1 reference checks from fold-level files."""
    rows: List[Dict[str, Any]] = []
    g0_df = pd.read_csv(ctx.p0_dir / 'g0_comparison.csv')
    n_pass = int((g0_df['status'] == 'PASS').sum())
    n_disc = int((g0_df['status'] == 'DISCREPANCY').sum())
    rows.append(dict(check='P0 G0 audit counts', observed=f'{n_pass} PASS / {n_disc} DISCREPANCY (total {len(g0_df)})',
                     expected='118 PASS / 74 DISCREPANCY (total 192)', matched=(n_pass == 118 and n_disc == 74 and len(g0_df) == 192)))

    # Recompute V3 native and V3 CIBD from E0 target_decisions.csv
    nat_rows, cibd_rows = [], []
    for fold, cohort in enumerate(COHORTS):
        p0_df = _read_p0_predictions(ctx, 'V3', fold, 'target')
        e0_df = pd.read_csv(ctx.e0_dir / 'lodo' / 'units' / f'V3_fold_{fold}_{cohort}' / 'fold_report' / 'target_decisions.csv',
                            dtype={'image_key': str, 'image_id': str, 'patient_key': str})
        m_df = _merge_on_image_key(p0_df, e0_df, ['native_replay', 'cibd_selected'], f'baseline_V3_{cohort}')
        nat_rows.append(compute_metrics(m_df['y_true'], m_df['native_replay']))
        cibd_rows.append(compute_metrics(m_df['y_true'], m_df['cibd_selected']))
    nat_g3 = float(np.mean([r['recall_g3'] * 100 for r in nat_rows]))
    nat_w1 = float(np.mean([r['within1'] * 100 for r in nat_rows]))
    cibd_g3 = float(np.mean([r['recall_g3'] * 100 for r in cibd_rows]))
    cibd_w1 = float(np.mean([r['within1'] * 100 for r in cibd_rows]))
    rows.append(dict(check='V3 native target G3 / Within-1 (%)', observed=f'{nat_g3:.4f}% / {nat_w1:.4f}%',
                     expected='32.22% / 92.34%', matched=abs(nat_g3 - 32.22) < 0.01 and abs(nat_w1 - 92.34) < 0.01))
    rows.append(dict(check='V3 CIBD target G3 / Within-1 (%)', observed=f'{cibd_g3:.4f}% / {cibd_w1:.4f}%',
                     expected='41.07% / 91.18%', matched=abs(cibd_g3 - 41.07) < 0.01 and abs(cibd_w1 - 91.18) < 0.01))

    # Recompute E1 probe G3 recalls from unit predictions.csv
    for rep, exp_val in [('z_v2', 44.64), ('z_v3', 51.09), ('s_v3', 54.09)]:
        rep_rows = []
        for fold, cohort in enumerate(COHORTS):
            var = 'V2' if rep == 'z_v2' else 'V3'
            p0_df = _read_p0_predictions(ctx, var, fold, 'target')
            pr_df = pd.read_csv(ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / rep / 'predict_target' / 'predictions.csv',
                                dtype={'image_key': str, 'patient_key': str})
            m_df = _merge_on_image_key(p0_df, pr_df, ['prediction'], f'baseline_{rep}_{cohort}')
            rep_rows.append(compute_metrics(m_df['y_true'], m_df['prediction']))
        val = float(np.mean([r['recall_g3'] * 100 for r in rep_rows]))
        rows.append(dict(check=f'E1 probe {rep} target G3 recall (%)', observed=f'{val:.4f}%',
                         expected=f'{exp_val:.2f}%', matched=abs(val - exp_val) < 0.01))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ctx.output_dir / 'baseline_verification.csv', index=False)
    return df_out


# %% [markdown]
# ## Analysis A — Shared-decoder comparison of $V_2$ and $V_3$ on held-out targets

# %%
def run_analysis_a(ctx: AnalysisContext) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compare V2 and V3 under identical decoding rules on the 6 held-out target cohorts."""
    decoders = [
        ('argmax', 'Argmax (common rule; = V2 native)', True, True),
        ('posterior_median', 'Posterior median (common rule)', True, True),
        ('rounded_expectation', 'Rounded expectation (common rule; = V3 native)', True, True),
        ('cibd_selected', 'CIBD (same source-only selection protocol; model-specific params)', True, True),
        ('cutpoint_latent', 'Latent cutpoint (V3 CLM only)', False, True),
    ]
    per_fold: List[Dict[str, Any]] = []
    for fold, cohort in enumerate(COHORTS):
        p0_v2 = _read_p0_predictions(ctx, 'V2', fold, 'target')
        p0_v3 = _read_p0_predictions(ctx, 'V3', fold, 'target')
        e0_v2 = pd.read_csv(ctx.e0_dir / 'lodo' / 'units' / f'V2_fold_{fold}_{cohort}' / 'fold_report' / 'target_decisions.csv',
                            dtype={'image_key': str, 'image_id': str, 'patient_key': str})
        e0_v3 = pd.read_csv(ctx.e0_dir / 'lodo' / 'units' / f'V3_fold_{fold}_{cohort}' / 'fold_report' / 'target_decisions.csv',
                            dtype={'image_key': str, 'image_id': str, 'patient_key': str})
        sel_v2 = json.loads((ctx.e0_dir / 'lodo' / 'units' / f'V2_fold_{fold}_{cohort}' / 'fold_report' / 'selected_params.json').read_text())['selected']
        sel_v3 = json.loads((ctx.e0_dir / 'lodo' / 'units' / f'V3_fold_{fold}_{cohort}' / 'fold_report' / 'selected_params.json').read_text())['selected']

        m_v2 = _merge_on_image_key(p0_v2, e0_v2, ['argmax', 'posterior_median', 'rounded_expectation', 'cibd_selected'], f'A_V2_{cohort}')
        m_v3 = _merge_on_image_key(p0_v3, e0_v3, ['argmax', 'posterior_median', 'rounded_expectation', 'cibd_selected', 'cutpoint_latent'], f'A_V3_{cohort}')
        # Verify V2 and V3 align on the exact same target image_key set
        _merge_on_image_key(m_v2[['image_key', 'y_true']], m_v3[['image_key', 'y_true']], [], f'A_pair_{cohort}')

        for rule, desc, v2_ok, v3_ok in decoders:
            mv2 = compute_metrics(m_v2['y_true'], m_v2[rule]) if v2_ok else None
            mv3 = compute_metrics(m_v3['y_true'], m_v3[rule]) if v3_ok else None
            row: Dict[str, Any] = {
                'fold_id': fold,
                'cohort': cohort,
                'cohort_label': COHORT_LABELS[cohort],
                'decoder': rule,
                'decoder_note': desc,
                'v2_cibd_params': f"({sel_v2['alpha']},{sel_v2['tau']},{sel_v2['q']})" if rule == 'cibd_selected' else '',
                'v3_cibd_params': f"({sel_v3['alpha']},{sel_v3['tau']},{sel_v3['q']})" if rule == 'cibd_selected' else '',
            }
            for metric, scale in [('recall_g3', 100.0), ('g3_precision', 100.0), ('within1', 100.0), ('qwk', 1.0), ('mae', 1.0)]:
                unit = 'pct' if scale == 100.0 else 'unit'
                diff_unit = 'pp' if scale == 100.0 else 'unit'
                v2_val = mv2[metric] * scale if mv2 is not None else np.nan
                v3_val = mv3[metric] * scale if mv3 is not None else np.nan
                row[f'v2_{metric}_{unit}'] = v2_val
                row[f'v3_{metric}_{unit}'] = v3_val
                row[f'diff_v3_minus_v2_{metric}_{diff_unit}'] = (v3_val - v2_val) if (mv2 is not None and mv3 is not None) else np.nan
            per_fold.append(row)

    df_fold = pd.DataFrame(per_fold)
    summary_rows: List[Dict[str, Any]] = []
    for rule, desc, v2_ok, v3_ok in decoders:
        sub = df_fold[df_fold['decoder'] == rule]
        srow: Dict[str, Any] = {'decoder': rule, 'decoder_note': desc, 'aggregation': 'equal-cohort mean (1/6)'}
        for metric, scale in [('recall_g3', 100.0), ('g3_precision', 100.0), ('within1', 100.0), ('qwk', 1.0), ('mae', 1.0)]:
            unit = 'pct' if scale == 100.0 else 'unit'
            diff_unit = 'pp' if scale == 100.0 else 'unit'
            srow[f'v2_{metric}_{unit}'] = float(sub[f'v2_{metric}_{unit}'].mean()) if v2_ok else np.nan
            srow[f'v3_{metric}_{unit}'] = float(sub[f'v3_{metric}_{unit}'].mean()) if v3_ok else np.nan
            if v2_ok and v3_ok:
                # Compute difference from unrounded fold means (equal to mean of fold differences)
                srow[f'diff_v3_minus_v2_{metric}_{diff_unit}'] = float(sub[f'diff_v3_minus_v2_{metric}_{diff_unit}'].mean())
                srow[f'n_cohorts_v3_gt_v2_{metric}'] = int((sub[f'diff_v3_minus_v2_{metric}_{diff_unit}'] > 0).sum())
            else:
                srow[f'diff_v3_minus_v2_{metric}_{diff_unit}'] = np.nan
                srow[f'n_cohorts_v3_gt_v2_{metric}'] = np.nan
        summary_rows.append(srow)

    df_summary = pd.DataFrame(summary_rows)
    df_fold.to_csv(ctx.output_dir / 'analysis_a_per_fold.csv', index=False)
    df_summary.to_csv(ctx.output_dir / 'analysis_a_summary.csv', index=False)
    status = dict(
        analysis_id='A',
        title='Shared-decoder comparison of V2 and V3 on held-out targets',
        status='COMPLETE',
        reason='All 6 folds and both variants verified against E0 target_decisions.csv.',
        output_files='analysis_a_per_fold.csv; analysis_a_summary.csv',
        source_paths=f'{ctx.p0_dir}; {ctx.e0_dir}',
    )
    return df_fold, df_summary, status


# %% [markdown]
# ## Analysis B — Five-grade recall, Grade 3 prediction ratio, and paired image transitions

# %%
def run_analysis_b(ctx: AnalysisContext) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compute G0-G4 recall, G3 precision, G3 predicted/true ratio, and paired G3 transitions."""
    methods = [
        ('native_v2', 'V2', 'native', 'V2'),
        ('cibd_v2', 'V2', 'e0_cibd', 'V2'),
        ('native_v3', 'V3', 'native', 'V3'),
        ('cibd_v3', 'V3', 'e0_cibd', 'V3'),
        ('probe_z_v2', 'V2', 'e1_z_v2', 'V2'),
        ('probe_z_v3', 'V3', 'e1_z_v3', 'V3'),
        ('probe_s_v3', 'V3', 'e1_s_v3', 'V3'),
    ]
    fold_rows: List[Dict[str, Any]] = []
    trans_rows: List[Dict[str, Any]] = []

    for fold, cohort in enumerate(COHORTS):
        p0_v2 = _read_p0_predictions(ctx, 'V2', fold, 'target')
        p0_v3 = _read_p0_predictions(ctx, 'V3', fold, 'target')
        e0_v2 = pd.read_csv(ctx.e0_dir / 'lodo' / 'units' / f'V2_fold_{fold}_{cohort}' / 'fold_report' / 'target_decisions.csv',
                            dtype={'image_key': str})
        e0_v3 = pd.read_csv(ctx.e0_dir / 'lodo' / 'units' / f'V3_fold_{fold}_{cohort}' / 'fold_report' / 'target_decisions.csv',
                            dtype={'image_key': str})
        z2_pr = pd.read_csv(ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / 'z_v2' / 'predict_target' / 'predictions.csv',
                            dtype={'image_key': str})
        z3_pr = pd.read_csv(ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / 'z_v3' / 'predict_target' / 'predictions.csv',
                            dtype={'image_key': str})
        s3_pr = pd.read_csv(ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / 's_v3' / 'predict_target' / 'predictions.csv',
                            dtype={'image_key': str})

        base_v2 = _merge_on_image_key(p0_v2[['image_key', 'y_true', 'pred_native']], e0_v2, ['cibd_selected'], f'B_e0_v2_{cohort}')
        base_v2 = _merge_on_image_key(base_v2, z2_pr, ['prediction'], f'B_e1_z2_{cohort}').rename(columns={'prediction': 'probe_z_v2', 'cibd_selected': 'cibd_v2', 'pred_native': 'native_v2'})
        base_v3 = _merge_on_image_key(p0_v3[['image_key', 'y_true', 'pred_native']], e0_v3, ['cibd_selected'], f'B_e0_v3_{cohort}')
        base_v3 = _merge_on_image_key(base_v3, z3_pr, ['prediction'], f'B_e1_z3_{cohort}').rename(columns={'prediction': 'probe_z_v3'})
        base_v3 = _merge_on_image_key(base_v3, s3_pr, ['prediction'], f'B_e1_s3_{cohort}').rename(columns={'prediction': 'probe_s_v3', 'cibd_selected': 'cibd_v3', 'pred_native': 'native_v3'})

        for method, var, _, baseline_var in methods:
            df_src = base_v2 if var == 'V2' else base_v3
            y = df_src['y_true'].to_numpy(int)
            pred = df_src[method].to_numpy(int)
            m = compute_metrics(y, pred)
            row: Dict[str, Any] = {
                'fold_id': fold,
                'cohort': cohort,
                'cohort_label': COHORT_LABELS[cohort],
                'method': method,
                'variant': var,
                'recall_g0_pct': m['recall_g0'] * 100.0,
                'recall_g1_pct': m['recall_g1'] * 100.0,
                'recall_g2_pct': m['recall_g2'] * 100.0,
                'recall_g3_pct': m['recall_g3'] * 100.0,
                'recall_g4_pct': m['recall_g4'] * 100.0,
                'macro_recall_pct': m['macro_recall'] * 100.0,
                'g3_precision_pct': m['g3_precision'] * 100.0 if np.isfinite(m['g3_precision']) else np.nan,
                'within1_pct': m['within1'] * 100.0,
                'qwk': m['qwk'],
                'mae': m['mae'],
                'true_g3': m['true_g3'],
                'predicted_g3': m['predicted_g3'],
                'predicted_true_g3_ratio': m['predicted_true_g3_ratio'],
            }
            fold_rows.append(row)

            if not method.startswith('native_'):
                nat_col = 'native_v2' if baseline_var == 'V2' else 'native_v3'
                nat_pred = df_src[nat_col].to_numpy(int)
                newly_g3 = (pred == 3) & (nat_pred != 3)
                lost_g3 = (pred != 3) & (nat_pred == 3)
                kept_g3 = (pred == 3) & (nat_pred == 3)
                trow: Dict[str, Any] = {
                    'fold_id': fold,
                    'cohort': cohort,
                    'method': method,
                    'baseline': nat_col,
                    'n_native_pred_g3': int((nat_pred == 3).sum()),
                    'n_method_pred_g3': int((pred == 3).sum()),
                    'n_kept_pred_g3': int(kept_g3.sum()),
                    'n_newly_pred_g3': int(newly_g3.sum()),
                    'n_dropped_pred_g3': int(lost_g3.sum()),
                    'true_g3_gained_correct': int(((y == 3) & newly_g3).sum()),
                    'true_g3_lost_correct': int(((y == 3) & lost_g3).sum()),
                    'true_g3_net_gain': int(((y == 3) & newly_g3).sum() - ((y == 3) & lost_g3).sum()),
                }
                for g in range(5):
                    trow[f'new_g3_true_g{g}'] = int((newly_g3 & (y == g)).sum())
                    if g != 3:
                        trow[f'new_g3_prev_pred_g{g}'] = int((newly_g3 & (nat_pred == g)).sum())
                trans_rows.append(trow)

    df_fold = pd.DataFrame(fold_rows)
    order_map = {m[0]: i for i, m in enumerate(methods)}
    num_cols = [c for c in df_fold.columns if c not in ('fold_id', 'cohort', 'cohort_label', 'method', 'variant')]
    df_sum = df_fold.groupby(['method', 'variant'], as_index=False)[num_cols].mean()
    df_sum['aggregation'] = 'equal-cohort mean (1/6); ratio averaged across fold ratios'
    df_sum['_ord'] = df_sum['method'].map(order_map)
    df_sum = df_sum.sort_values('_ord').drop(columns=['_ord']).reset_index(drop=True)

    df_trans = pd.DataFrame(trans_rows)
    t_cols = [c for c in df_trans.columns if c not in ('fold_id', 'cohort', 'method', 'baseline')]
    df_trans_sum = df_trans.groupby(['method', 'baseline'], as_index=False)[t_cols].sum()
    df_trans_sum['aggregation'] = 'sum of paired image counts across 6 target cohorts (54,883 images)'
    df_trans_sum['_ord'] = df_trans_sum['method'].map(order_map)
    df_trans_sum = df_trans_sum.sort_values('_ord').drop(columns=['_ord']).reset_index(drop=True)

    df_fold.to_csv(ctx.output_dir / 'analysis_b_five_grade_per_fold.csv', index=False)
    df_sum.to_csv(ctx.output_dir / 'analysis_b_five_grade_summary.csv', index=False)
    df_trans.to_csv(ctx.output_dir / 'analysis_b_g3_transitions_per_fold.csv', index=False)
    df_trans_sum.to_csv(ctx.output_dir / 'analysis_b_g3_transitions_summary.csv', index=False)
    status = dict(
        analysis_id='B',
        title='Five-grade recall, G3 prediction ratio, and paired G3 transitions',
        status='COMPLETE',
        reason='Computed 5-grade recalls, per-fold ratios, equal-cohort means, and image_key-matched G3 transitions.',
        output_files='analysis_b_five_grade_per_fold.csv; analysis_b_five_grade_summary.csv; analysis_b_g3_transitions_per_fold.csv; analysis_b_g3_transitions_summary.csv',
        source_paths=f'{ctx.p0_dir}; {ctx.e0_dir}; {ctx.e1_dir}',
    )
    return df_fold, df_sum, df_trans, df_trans_sum, status


# %% [markdown]
# ## Analysis C — P0 audit by image counts and per-image prediction agreement

# %%
def _extract_archive_ids(z: Any, split: str) -> Tuple[Optional[np.ndarray], str, str]:
    """Return (id_array, replay_column_name, missing_reason) for an archived NPZ split."""
    z_keys = list(z.files)
    if split == 'target':
        candidates = [('ood_image_keys', 'image_key'), ('ood_image_ids', 'image_id'), ('image_ids', 'image_id')]
    else:
        candidates = [('val_image_keys', 'image_key'), ('val_image_ids', 'image_id')]
    for npz_key, p0_col in candidates:
        if npz_key in z_keys:
            return z[npz_key], p0_col, ''
    expected = 'ood_image_keys/ood_image_ids/image_ids' if split == 'target' else 'val_image_keys/val_image_ids'
    return None, ('image_id'), f'lacks per-image ID array ({expected}; keys={z_keys})'


def _normalize_id_series(values: Any, side_label: str) -> pd.Series:
    """Validate non-null, non-empty, unique IDs for strict one-to-one linkage."""
    s = pd.Series(list(values), dtype=object)
    if s.isna().any():
        raise ValueError(f'Null image IDs found in {side_label}')
    str_s = s.astype(str)
    if str_s.str.strip().eq('').any():
        raise ValueError(f'Empty image IDs found in {side_label}')
    if str_s.duplicated().any():
        dup = str_s[str_s.duplicated()].iloc[0]
        raise ValueError(f'Duplicate image ID "{dup}" found in {side_label}')
    return str_s


def _match_split_predictions(
    y_arch: np.ndarray,
    p_arch: np.ndarray,
    arch_ids: Optional[np.ndarray],
    p0_df: pd.DataFrame,
    id_col: str,
    missing_reason: str,
    npz_name: str,
    label: str,
) -> Dict[str, Any]:
    """Perform strict ID-based linkage or return NaN linkage fields when IDs are unavailable."""
    n_arch = int(len(y_arch))
    n_rep = int(len(p0_df))
    if arch_ids is None:
        return {
            'archived_has_image_ids': False,
            'replay_has_image_ids': bool(id_col in p0_df and not p0_df[id_col].isna().any()),
            'n_matched_by_id': np.nan,
            'n_archived_only': np.nan,
            'n_replay_only': np.nan,
            'id_coverage_archived': np.nan,
            'id_coverage_replay': np.nan,
            'n_same_prediction': np.nan,
            'n_changed_prediction': np.nan,
            'prediction_agreement_rate': np.nan,
            'g3_wrong_to_right': np.nan,
            'g3_right_to_wrong': np.nan,
            'label_mismatches_on_matched': np.nan,
            'agreement_status': 'NOT_ESTIMABLE',
            'missing_artifact': f'{npz_name} {missing_reason}; row-order pairing prohibited',
        }
    if id_col not in p0_df or p0_df[id_col].isna().any() or p0_df[id_col].astype(str).str.strip().eq('').any():
        return {
            'archived_has_image_ids': True,
            'replay_has_image_ids': False,
            'n_matched_by_id': np.nan,
            'n_archived_only': np.nan,
            'n_replay_only': np.nan,
            'id_coverage_archived': np.nan,
            'id_coverage_replay': np.nan,
            'n_same_prediction': np.nan,
            'n_changed_prediction': np.nan,
            'prediction_agreement_rate': np.nan,
            'g3_wrong_to_right': np.nan,
            'g3_right_to_wrong': np.nan,
            'label_mismatches_on_matched': np.nan,
            'agreement_status': 'NOT_ESTIMABLE',
            'missing_artifact': f'Replay predictions missing valid {id_col} column ({label}); row-order pairing prohibited',
        }
    if len(arch_ids) != n_arch:
        raise ValueError(f'Archived ID length ({len(arch_ids)}) != prediction length ({n_arch}) in {label}')

    arch_key = _normalize_id_series(arch_ids, f'{label} archived')
    rep_key = _normalize_id_series(p0_df[id_col].to_numpy(), f'{label} replay')

    df_arch = pd.DataFrame({'_id': arch_key, 'y_arch': np.asarray(y_arch, int), 'p_arch': np.asarray(p_arch, int)})
    df_rep = pd.DataFrame({'_id': rep_key, 'y_rep': p0_df['y_true'].to_numpy(int), 'p_rep': p0_df['pred_native'].to_numpy(int)})
    matched = df_arch.merge(df_rep, on='_id', how='inner')

    n_matched = int(len(matched))
    n_arch_only = n_arch - n_matched
    n_rep_only = n_rep - n_matched
    cov_arch = float(n_matched / n_arch) if n_arch > 0 else np.nan
    cov_rep = float(n_matched / n_rep) if n_rep > 0 else np.nan

    if n_matched == 0:
        return {
            'archived_has_image_ids': True,
            'replay_has_image_ids': True,
            'n_matched_by_id': 0,
            'n_archived_only': n_arch_only,
            'n_replay_only': n_rep_only,
            'id_coverage_archived': cov_arch,
            'id_coverage_replay': cov_rep,
            'n_same_prediction': np.nan,
            'n_changed_prediction': np.nan,
            'prediction_agreement_rate': np.nan,
            'g3_wrong_to_right': np.nan,
            'g3_right_to_wrong': np.nan,
            'label_mismatches_on_matched': 0,
            'agreement_status': 'DISJOINT_ZERO_OVERLAP',
            'missing_artifact': f'Valid ID join in {label} found 0 common IDs; agreement undefined (0 denominator).',
        }

    mismatches = int((matched['y_arch'] != matched['y_rep']).sum())
    if mismatches > 0:
        raise ValueError(f'Ground-truth label mismatch on {mismatches} matched IDs in {label}')

    ya = matched['y_arch'].to_numpy(int)
    pa = matched['p_arch'].to_numpy(int)
    pr = matched['p_rep'].to_numpy(int)
    n_same = int((pa == pr).sum())
    n_changed = int((pa != pr).sum())
    rate = float(n_same / n_matched)
    g3_mask = (ya == 3)
    w2r = int((g3_mask & (pa != 3) & (pr == 3)).sum())
    r2w = int((g3_mask & (pa == 3) & (pr != 3)).sum())
    ag_status = 'ESTIMABLE_FULL' if (n_arch_only == 0 and n_rep_only == 0) else 'ESTIMABLE_PARTIAL_SUBSET'
    note = '' if ag_status == 'ESTIMABLE_FULL' else (
        f'Matched subset only ({n_matched} matched; {n_arch_only} archived-only; {n_rep_only} replay-only).'
    )
    return {
        'archived_has_image_ids': True,
        'replay_has_image_ids': True,
        'n_matched_by_id': n_matched,
        'n_archived_only': n_arch_only,
        'n_replay_only': n_rep_only,
        'id_coverage_archived': cov_arch,
        'id_coverage_replay': cov_rep,
        'n_same_prediction': n_same,
        'n_changed_prediction': n_changed,
        'prediction_agreement_rate': rate,
        'g3_wrong_to_right': w2r,
        'g3_right_to_wrong': r2w,
        'label_mismatches_on_matched': 0,
        'agreement_status': ag_status,
        'missing_artifact': note,
    }


def _aggregate_c_status(statuses: pd.Series) -> str:
    vals = set(statuses.astype(str))
    n_tot = len(statuses)
    n_full = int((statuses == 'ESTIMABLE_FULL').sum())
    n_est = int(statuses.str.startswith('ESTIMABLE').sum())
    n_disjoint = int((statuses == 'DISJOINT_ZERO_OVERLAP').sum())
    n_not = int((statuses == 'NOT_ESTIMABLE').sum())
    if n_full == n_tot:
        return 'ESTIMABLE_FULL'
    if n_est > 0:
        return 'PARTIAL_ESTIMABLE'
    if n_disjoint == n_tot:
        return 'DISJOINT_ZERO_OVERLAP'
    if n_not == n_tot:
        return 'NOT_ESTIMABLE'
    return f'MIXED_UNLINKED ({",".join(sorted(vals))})'


def run_analysis_c(ctx: AnalysisContext, folds: Sequence[int] = tuple(range(6))) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Audit archived vs. replay image counts and perform strict ID-matched prediction agreement linkage."""
    if ctx.archived_preds_dir is None or ctx.reference_json is None:
        status = dict(
            analysis_id='C',
            title='P0 image-count discrepancies and ID-matched prediction agreement',
            status='NOT_ESTIMABLE',
            reason='Archived raw_preds NPZs or reference JSON not found.',
            output_files='',
            source_paths=str(ctx.p0_dir),
        )
        return pd.DataFrame(), pd.DataFrame(), status

    ref = json.loads(ctx.reference_json.read_text(encoding='utf-8'))
    g0_df = pd.read_csv(ctx.p0_dir / 'g0_comparison.csv')

    rows: List[Dict[str, Any]] = []
    for variant, v_long in VARIANT_NAMES.items():
        for fold in folds:
            cohort = COHORTS[fold]
            npz_path = ctx.archived_preds_dir / f'{v_long}__{cohort}.npz'
            with np.load(npz_path, allow_pickle=False) as z:
                val_ids, val_col, val_miss = _extract_archive_ids(z, 'source_val')
                ood_ids, ood_col, ood_miss = _extract_archive_ids(z, 'target')
                splits_info = [
                    ('source_val', z['val_targets'].astype(int), z['val_preds'].astype(int), val_ids, val_col, val_miss),
                    ('target', z['ood_targets'].astype(int), z['ood_preds'].astype(int), ood_ids, ood_col, ood_miss),
                ]
                for split, y_arch, p_arch, arch_ids, p0_col, miss_reason in splits_info:
                    p0_df = _read_p0_predictions(ctx, variant, fold, split)
                    y_rep = p0_df['y_true'].to_numpy(int)
                    p_rep = p0_df['pred_native'].to_numpy(int)
                    m_arch = compute_metrics(y_arch, p_arch)
                    m_rep = compute_metrics(y_rep, p_rep)

                    ref_rec = ref[v_long][cohort]
                    sup_key = 'val_per_grade_support' if split == 'source_val' else 'ood_per_grade_support'
                    ref_sup = [int(ref_rec[sup_key][f'grade_{g}']) for g in range(5)]
                    arch_sup = [m_arch[f'support_g{g}'] for g in range(5)]
                    rep_sup = [m_rep[f'support_g{g}'] for g in range(5)]
                    supports_match = (ref_sup == arch_sup == rep_sup)

                    g0_sub = g0_df[(g0_df['variant'] == variant) & (g0_df['fold_id'] == fold) & (g0_df['split'] == split)]
                    n_pass = int((g0_sub['status'] == 'PASS').sum())
                    n_disc = int((g0_sub['status'] == 'DISCREPANCY').sum())

                    linkage = _match_split_predictions(
                        y_arch, p_arch, arch_ids, p0_df, p0_col, miss_reason, npz_path.name,
                        f'{variant}/{cohort}/{split}',
                    )

                    n_g3 = m_rep['true_g3']
                    rec_row: Dict[str, Any] = {
                        'variant': variant,
                        'fold_id': fold,
                        'cohort': cohort,
                        'cohort_label': COHORT_LABELS[cohort],
                        'split': split,
                        'n_archived': m_arch['n'],
                        'n_replay': m_rep['n'],
                        'supports_identical': bool(supports_match),
                        **linkage,
                        'n_g3_true': n_g3,
                        'g3_pp_per_image': 100.0 / n_g3 if n_g3 > 0 else np.nan,
                        'n_g3_correct_archived': m_arch['correct_g3'],
                        'n_g3_correct_replay': m_rep['correct_g3'],
                        'delta_g3_correct_net': m_rep['correct_g3'] - m_arch['correct_g3'],
                        'recall_g3_archived_pct': m_arch['recall_g3'] * 100.0,
                        'recall_g3_replay_pct': m_rep['recall_g3'] * 100.0,
                        'delta_recall_g3_pp': (m_rep['recall_g3'] - m_arch['recall_g3']) * 100.0,
                        'within1_archived_pct': m_arch['within1'] * 100.0,
                        'within1_replay_pct': m_rep['within1'] * 100.0,
                        'delta_within1_pp': (m_rep['within1'] - m_arch['within1']) * 100.0,
                        'qwk_archived': m_arch['qwk'],
                        'qwk_replay': m_rep['qwk'],
                        'delta_qwk': m_rep['qwk'] - m_arch['qwk'],
                        'g0_checks_total': len(g0_sub),
                        'g0_checks_pass': n_pass,
                        'g0_checks_discrepancy': n_disc,
                    }
                    for g in range(5):
                        rec_row[f'delta_correct_g{g}_net'] = m_rep[f'correct_g{g}'] - m_arch[f'correct_g{g}']
                    rows.append(rec_row)

    df_c = pd.DataFrame(rows)
    sum_rows: List[Dict[str, Any]] = []
    for (variant, split), grp in df_c.groupby(['variant', 'split'], sort=False):
        grp_status = _aggregate_c_status(grp['agreement_status'])
        est_mask = grp['agreement_status'].str.startswith('ESTIMABLE')
        n_est = int(est_mask.sum())
        n_not = int((grp['agreement_status'] == 'NOT_ESTIMABLE').sum())
        n_dis = int((grp['agreement_status'] == 'DISJOINT_ZERO_OVERLAP').sum())
        has_linkage_counts = bool(grp['n_matched_by_id'].notna().any())
        matched_sum = float(grp['n_matched_by_id'].sum(min_count=1)) if has_linkage_counts else np.nan
        arch_only_sum = float(grp['n_archived_only'].sum(min_count=1)) if has_linkage_counts else np.nan
        rep_only_sum = float(grp['n_replay_only'].sum(min_count=1)) if has_linkage_counts else np.nan
        same_sum = float(grp.loc[est_mask, 'n_same_prediction'].sum()) if n_est > 0 else np.nan
        chg_sum = float(grp.loc[est_mask, 'n_changed_prediction'].sum()) if n_est > 0 else np.nan
        est_matched_denom = float(grp.loc[est_mask, 'n_matched_by_id'].sum()) if n_est > 0 else 0.0
        ag_rate = float(same_sum / est_matched_denom) if (n_est > 0 and est_matched_denom > 0) else np.nan
        w2r_sum = float(grp.loc[est_mask, 'g3_wrong_to_right'].sum()) if n_est > 0 else np.nan
        r2w_sum = float(grp.loc[est_mask, 'g3_right_to_wrong'].sum()) if n_est > 0 else np.nan

        if n_est == len(grp) and (grp['agreement_status'] == 'ESTIMABLE_FULL').all():
            ag_scope = 'full_all_cohorts'
        elif n_est > 0:
            ag_scope = f'matched_subset_in_{n_est}_of_{len(grp)}_cohorts'
        elif n_dis == len(grp):
            ag_scope = 'disjoint_zero_overlap_undefined_rate'
        else:
            ag_scope = 'unavailable_missing_image_ids'

        sum_rows.append({
            'variant': variant,
            'split': split,
            'cohorts': len(grp),
            'n_archived_total': int(grp['n_archived'].sum()),
            'n_replay_total': int(grp['n_replay'].sum()),
            'all_supports_identical': bool(grp['supports_identical'].all()),
            'agreement_status': grp_status,
            'agreement_scope': ag_scope,
            'n_splits_estimable': n_est,
            'n_splits_not_estimable': n_not,
            'n_splits_disjoint': n_dis,
            'n_matched_by_id_total': matched_sum,
            'n_archived_only_total': arch_only_sum,
            'n_replay_only_total': rep_only_sum,
            'n_same_prediction_total': same_sum,
            'n_changed_prediction_total': chg_sum,
            'prediction_agreement_rate_matched': ag_rate,
            'g3_wrong_to_right_total': w2r_sum,
            'g3_right_to_wrong_total': r2w_sum,
            'n_g3_true_total': int(grp['n_g3_true'].sum()),
            'n_g3_correct_archived_total': int(grp['n_g3_correct_archived'].sum()),
            'n_g3_correct_replay_total': int(grp['n_g3_correct_replay'].sum()),
            'delta_g3_correct_net_total': int(grp['delta_g3_correct_net'].sum()),
            'equal_cohort_recall_g3_archived_pct': float(grp['recall_g3_archived_pct'].mean()),
            'equal_cohort_recall_g3_replay_pct': float(grp['recall_g3_replay_pct'].mean()),
            'equal_cohort_delta_recall_g3_pp': float(grp['delta_recall_g3_pp'].mean()),
            'equal_cohort_within1_archived_pct': float(grp['within1_archived_pct'].mean()),
            'equal_cohort_within1_replay_pct': float(grp['within1_replay_pct'].mean()),
            'equal_cohort_delta_within1_pp': float(grp['delta_within1_pp'].mean()),
            'g0_checks_pass_total': int(grp['g0_checks_pass'].sum()),
            'g0_checks_discrepancy_total': int(grp['g0_checks_discrepancy'].sum()),
            'delta_correct_g0_net_total': int(grp['delta_correct_g0_net'].sum()),
            'delta_correct_g1_net_total': int(grp['delta_correct_g1_net'].sum()),
            'delta_correct_g2_net_total': int(grp['delta_correct_g2_net'].sum()),
            'delta_correct_g4_net_total': int(grp['delta_correct_g4_net'].sum()),
        })
    df_sum = pd.DataFrame(sum_rows)
    df_c.to_csv(ctx.output_dir / 'analysis_c_p0_image_counts.csv', index=False)
    df_sum.to_csv(ctx.output_dir / 'analysis_c_p0_summary.csv', index=False)

    all_full = bool((df_c['agreement_status'] == 'ESTIMABLE_FULL').all())
    n_pass_all = int(df_c['g0_checks_pass'].sum())
    n_disc_all = int(df_c['g0_checks_discrepancy'].sum())
    n_chk_all = int(df_c['g0_checks_total'].sum())
    n_est_rows = int(df_c['agreement_status'].str.startswith('ESTIMABLE').sum())
    n_not_rows = int((df_c['agreement_status'] == 'NOT_ESTIMABLE').sum())
    n_dis_rows = int((df_c['agreement_status'] == 'DISJOINT_ZERO_OVERLAP').sum())
    overall_status = 'COMPLETE' if all_full else 'PARTIAL'
    status = dict(
        analysis_id='C',
        title='P0 image-count discrepancies and ID-matched prediction agreement',
        status=overall_status,
        reason=(
            f'Verifiable image-count and support checks completed across {len(df_c)} variant-fold-split rows '
            f'({n_pass_all} PASS / {n_disc_all} DISCREPANCY of {n_chk_all} metric checks). '
            f'ID-matched agreement rows: {n_est_rows} estimable, {n_dis_rows} disjoint zero-overlap, {n_not_rows} NOT_ESTIMABLE.'
        ),
        output_files='analysis_c_p0_image_counts.csv; analysis_c_p0_summary.csv',
        source_paths=f'{ctx.p0_dir}; {ctx.archived_preds_dir}; {ctx.reference_json}',
    )
    return df_c, df_sum, status


# %% [markdown]
# ## Analysis D — Max $p_3$ and Grade 3 decision-region overlap across all 6 $V_3$ folds

# %%
def _clm_probs_from_grid(s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    u = np.clip(theta[None, :] - s[:, None], -20.0, 20.0)
    survival = np.exp(-np.exp(u))
    p = np.column_stack([-np.expm1(-np.exp(u[:, 0])), survival[:, :-1] - survival[:, 1:], survival[:, -1]])
    p /= p.sum(1, keepdims=True)
    return p


def _interval_summary(s: np.ndarray, mask: np.ndarray, grid_lo: float, grid_hi: float) -> Tuple[List[Tuple[float, float]], float, bool]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return [], 0.0, False
    step = float(s[1] - s[0])
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    groups = np.split(idx, splits)
    intervals = [(float(s[g[0]]), float(s[g[-1]])) for g in groups]
    # Measure continuous span covered by grid points (right - left)
    total_len = sum(r - l for l, r in intervals)
    touches = bool(idx[0] == 0 or idx[-1] == len(s) - 1)
    _ = step, grid_lo, grid_hi
    return intervals, float(total_len), touches


def run_analysis_d(ctx: AnalysisContext) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Analyze max p3 and native vs. cutpoint Grade 3 decision intervals across all 6 V3 folds."""
    rows: List[Dict[str, Any]] = []
    for fold, cohort in enumerate(COHORTS):
        p0_t = _read_p0_predictions(ctx, 'V3', fold, 'target')
        theta = p0_t[[f'theta{k}' for k in range(4)]].iloc[0].to_numpy(float)
        fr = ctx.e0_dir / 'lodo' / 'units' / f'V3_fold_{fold}_{cohort}' / 'fold_report'
        grid_df = pd.read_csv(fr / 'decision_grid_float64_diagnostic.csv')
        iv_df = pd.read_csv(fr / 'decision_intervals.csv')

        s_grid = grid_df['s'].to_numpy(float)
        step = float(s_grid[1] - s_grid[0])
        p_grid = _clm_probs_from_grid(s_grid, theta)
        # Verify forward probability reconstruction reproduces grid rounded_expectation / native_replay
        rebuilt_native = np.rint(p_grid @ np.arange(5)).clip(0, 4).astype(int)
        if not np.array_equal(rebuilt_native, grid_df['native_replay'].to_numpy(int)):
            raise ValueError(f'Grid CLM probability reconstruction mismatch on {cohort}')

        p3_grid = p_grid[:, 3]
        idx_max = int(np.argmax(p3_grid))
        idx_tgt = int(p0_t['p3'].idxmax())

        nat_mask = grid_df['native_replay'].to_numpy(int) == 3
        cut_mask = grid_df['cutpoint_latent'].to_numpy(int) == 3
        cibd_mask = grid_df['cibd_selected'].to_numpy(int) == 3
        inter_mask = nat_mask & cut_mask

        nat_ivs, nat_len, nat_touch = _interval_summary(s_grid, nat_mask, float(s_grid[0]), float(s_grid[-1]))
        cut_ivs, cut_len, cut_touch = _interval_summary(s_grid, cut_mask, float(s_grid[0]), float(s_grid[-1]))
        cibd_ivs, cibd_len, cibd_touch = _interval_summary(s_grid, cibd_mask, float(s_grid[0]), float(s_grid[-1]))
        _, overlap_len, _ = _interval_summary(s_grid, inter_mask, float(s_grid[0]), float(s_grid[-1]))

        # Cross-check against saved decision_intervals.csv for single-interval cases
        saved_nat = iv_df[(iv_df['rule'] == 'native_replay') & (iv_df['grade'] == 3)]
        saved_cut = iv_df[(iv_df['rule'] == 'cutpoint_latent') & (iv_df['grade'] == 3)]
        nat_left = float(saved_nat['left'].min()) if not saved_nat.empty else np.nan
        nat_right = float(saved_nat['right'].max()) if not saved_nat.empty else np.nan
        cut_left = float(saved_cut['left'].min()) if not saved_cut.empty else np.nan
        cut_right = float(saved_cut['right'].max()) if not saved_cut.empty else np.nan

        # Exact analytical overlap on [nat_left, nat_right] and [theta2, theta3] when single interval
        analytic_cut_len = float(theta[3] - theta[2])
        analytic_overlap = max(0.0, min(nat_right, theta[3]) - max(nat_left, theta[2])) if (len(nat_ivs) == 1 and len(cut_ivs) == 1) else overlap_len

        rows.append({
            'fold_id': fold,
            'cohort': cohort,
            'cohort_label': COHORT_LABELS[cohort],
            'grid_points': len(s_grid),
            'grid_min_s': float(s_grid[0]),
            'grid_max_s': float(s_grid[-1]),
            'grid_step': step,
            'max_p3_grid': float(p3_grid[idx_max]),
            's_at_max_p3_grid': float(s_grid[idx_max]),
            'max_p3_target_images': float(p0_t.loc[idx_tgt, 'p3']),
            's_at_max_p3_target_images': float(p0_t.loc[idx_tgt, 's']),
            'theta0': float(theta[0]),
            'theta1': float(theta[1]),
            'theta2': float(theta[2]),
            'theta3': float(theta[3]),
            'native_g3_n_intervals': len(nat_ivs),
            'native_g3_left': nat_left,
            'native_g3_right': nat_right,
            'native_g3_length': nat_len,
            'native_touches_grid_edge': nat_touch,
            'cutpoint_g3_n_intervals': len(cut_ivs),
            'cutpoint_g3_left': cut_left,
            'cutpoint_g3_right': cut_right,
            'cutpoint_g3_length_grid': cut_len,
            'cutpoint_g3_length_analytic': analytic_cut_len,
            'cutpoint_touches_grid_edge': cut_touch,
            'cibd_g3_n_intervals': len(cibd_ivs),
            'cibd_g3_length': cibd_len,
            'cibd_touches_grid_edge': cibd_touch,
            'overlap_length_grid': overlap_len,
            'overlap_ratio_grid': float(overlap_len / cut_len) if cut_len > 0 else np.nan,
            'overlap_length_analytic': analytic_overlap,
            'overlap_ratio_analytic': float(analytic_overlap / analytic_cut_len) if analytic_cut_len > 0 else np.nan,
            'shift_left_native_minus_theta2': nat_left - float(theta[2]) if np.isfinite(nat_left) else np.nan,
            'shift_right_native_minus_theta3': nat_right - float(theta[3]) if np.isfinite(nat_right) else np.nan,
            'upward_shift_both_bounds': bool(np.isfinite(nat_left) and nat_left > theta[2] and nat_right > theta[3]),
        })

    df_d = pd.DataFrame(rows)
    df_sum = pd.DataFrame([{
        'n_folds': len(df_d),
        'max_p3_grid_min': float(df_d['max_p3_grid'].min()),
        'max_p3_grid_max': float(df_d['max_p3_grid'].max()),
        'max_p3_grid_mean': float(df_d['max_p3_grid'].mean()),
        'max_p3_target_images_min': float(df_d['max_p3_target_images'].min()),
        'max_p3_target_images_max': float(df_d['max_p3_target_images'].max()),
        'overlap_ratio_grid_min': float(df_d['overlap_ratio_grid'].min()),
        'overlap_ratio_grid_max': float(df_d['overlap_ratio_grid'].max()),
        'overlap_ratio_grid_mean': float(df_d['overlap_ratio_grid'].mean()),
        'n_folds_upward_shift': int(df_d['upward_shift_both_bounds'].sum()),
        'n_folds_touching_grid_edge': int((df_d['native_touches_grid_edge'] | df_d['cutpoint_touches_grid_edge']).sum()),
        'idrid_overlap_ratio_grid': float(df_d.loc[df_d['cohort'] == 'idrid', 'overlap_ratio_grid'].iloc[0]),
        'idrid_max_p3_grid': float(df_d.loc[df_d['cohort'] == 'idrid', 'max_p3_grid'].iloc[0]),
    }])
    df_d.to_csv(ctx.output_dir / 'analysis_d_v3_latent_regions_per_fold.csv', index=False)
    df_sum.to_csv(ctx.output_dir / 'analysis_d_v3_latent_regions_summary.csv', index=False)
    status = dict(
        analysis_id='D',
        title='Max p3 and Grade 3 decision-region overlap across 6 V3 folds',
        status='COMPLETE',
        reason='Reconstructed float64 cloglog grids and interval overlaps across all 6 V3 folds.',
        output_files='analysis_d_v3_latent_regions_per_fold.csv; analysis_d_v3_latent_regions_summary.csv',
        source_paths=f'{ctx.p0_dir}; {ctx.e0_dir}',
    )
    return df_d, df_sum, status


# %% [markdown]
# ## Analysis E — Paired recorded-grouping-key bootstrap sensitivity analysis

# %%
def _group_cm_vectors(y: np.ndarray, pred: np.ndarray, group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Return shape (n_groups, 25) flattened 5x5 confusion counts per grouping key."""
    bins = group_codes * 25 + y * 5 + pred
    return np.bincount(bins, minlength=n_groups * 25).reshape(n_groups, 25).astype(np.float64)


def _metrics_from_cm_batch(cm_batch: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorized metrics over shape (R, 5, 5) confusion matrices."""
    total = cm_batch.sum(axis=(1, 2))
    row_sum = cm_batch.sum(axis=2)
    col_sum = cm_batch.sum(axis=1)
    n3 = row_sum[:, 3]
    pred3 = col_sum[:, 3]

    recall_g3 = np.divide(cm_batch[:, 3, 3], n3, out=np.full(len(cm_batch), np.nan), where=n3 > 0)
    g3_precision = np.divide(cm_batch[:, 3, 3], pred3, out=np.full(len(cm_batch), np.nan), where=pred3 > 0)

    idx = np.arange(5)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist2 = dist ** 2
    within1 = np.divide((cm_batch * (dist <= 1)).sum(axis=(1, 2)), total, out=np.full(len(cm_batch), np.nan), where=total > 0)
    mae = np.divide((cm_batch * dist).sum(axis=(1, 2)), total, out=np.full(len(cm_batch), np.nan), where=total > 0)

    expected = np.divide(row_sum[:, :, None] * col_sum[:, None, :], total[:, None, None],
                         out=np.zeros_like(cm_batch), where=total[:, None, None] > 0)
    denom = (expected * dist2).sum(axis=(1, 2))
    numer = (cm_batch * dist2).sum(axis=(1, 2))
    qwk = np.divide(denom - numer, denom, out=np.full(len(cm_batch), np.nan), where=denom > 0)

    return {
        'recall_g3_pp': recall_g3 * 100.0,
        'within1_pp': within1 * 100.0,
        'g3_precision_pp': g3_precision * 100.0,
        'qwk': qwk,
        'mae': mae,
    }


def run_analysis_e(ctx: AnalysisContext) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run paired recorded-grouping-key bootstrap sensitivity analysis (2,000 reps by default)."""
    reps = ctx.bootstrap_reps
    seed = ctx.seed
    contrasts = [
        ('cibd_v3_minus_native_v3', 'V3 CIBD minus V3 native_replay'),
        ('probe_s3_minus_native_v3', 'V3 probe s3 minus V3 native_replay'),
    ]
    metric_keys = ['recall_g3_pp', 'within1_pp', 'g3_precision_pp', 'qwk', 'mae']

    # Pre-load matched target frames and group statistics for all 6 folds
    fold_bundles: List[Dict[str, Any]] = []
    for fold, cohort in enumerate(COHORTS):
        p0_v3 = _read_p0_predictions(ctx, 'V3', fold, 'target')
        e0_v3 = pd.read_csv(ctx.e0_dir / 'lodo' / 'units' / f'V3_fold_{fold}_{cohort}' / 'fold_report' / 'target_decisions.csv',
                            dtype={'image_key': str, 'patient_key': str})
        s3_pr = pd.read_csv(ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / 's_v3' / 'predict_target' / 'predictions.csv',
                            dtype={'image_key': str, 'patient_key': str})
        df = _merge_on_image_key(p0_v3[['image_key', 'y_true', 'patient_key', 'patient_identity_status', 'pred_native']],
                                 e0_v3, ['cibd_selected'], f'E_e0_{cohort}')
        df = _merge_on_image_key(df, s3_pr, ['prediction'], f'E_s3_{cohort}').rename(columns={'prediction': 'probe_s3'})

        codes, keys = pd.factorize(df['patient_key'], sort=True)
        n_groups = len(keys)
        y = df['y_true'].to_numpy(int)
        cm_nat = _group_cm_vectors(y, df['pred_native'].to_numpy(int), codes, n_groups)
        cm_cibd = _group_cm_vectors(y, df['cibd_selected'].to_numpy(int), codes, n_groups)
        cm_s3 = _group_cm_vectors(y, df['probe_s3'].to_numpy(int), codes, n_groups)

        # Draw R group-weight matrices of shape (reps, n_groups) using fold-specific child seed
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
        weights = np.empty((reps, n_groups), dtype=np.float64)
        for r in range(reps):
            weights[r] = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)

        fold_bundles.append(dict(
            fold=fold, cohort=cohort, n_images=len(df), n_groups=n_groups,
            identity_statuses=sorted(set(df['patient_identity_status'].astype(str))),
            y=y, nat=df['pred_native'].to_numpy(int), cibd=df['cibd_selected'].to_numpy(int), s3=df['probe_s3'].to_numpy(int),
            cm_nat=cm_nat, cm_cibd=cm_cibd, cm_s3=cm_s3, weights=weights,
        ))

    out_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []

    for c_id, c_label in contrasts:
        macro_diffs = {mk: np.zeros(reps, dtype=np.float64) for mk in metric_keys}
        macro_valid = np.ones(reps, dtype=bool)
        per_fold_point = {mk: [] for mk in metric_keys}

        for b in fold_bundles:
            w = b['weights']
            cm_a_boot = (w @ (b['cm_cibd'] if c_id == 'cibd_v3_minus_native_v3' else b['cm_s3'])).reshape(reps, 5, 5)
            cm_b_boot = (w @ b['cm_nat']).reshape(reps, 5, 5)
            ma_boot = _metrics_from_cm_batch(cm_a_boot)
            mb_boot = _metrics_from_cm_batch(cm_b_boot)

            ma_pt = compute_metrics(b['y'], b['cibd'] if c_id == 'cibd_v3_minus_native_v3' else b['s3'])
            mb_pt = compute_metrics(b['y'], b['nat'])
            pt_map = {
                'recall_g3_pp': (ma_pt['recall_g3'] - mb_pt['recall_g3']) * 100.0,
                'within1_pp': (ma_pt['within1'] - mb_pt['within1']) * 100.0,
                'g3_precision_pp': (ma_pt['g3_precision'] - mb_pt['g3_precision']) * 100.0,
                'qwk': ma_pt['qwk'] - mb_pt['qwk'],
                'mae': ma_pt['mae'] - mb_pt['mae'],
            }
            for mk in metric_keys:
                diff_vec = ma_boot[mk] - mb_boot[mk]
                finite = np.isfinite(diff_vec)
                n_val = int(finite.sum())
                n_undef = int((~finite).sum())
                lo = float(np.percentile(diff_vec[finite], 2.5)) if n_val > 0 else np.nan
                hi = float(np.percentile(diff_vec[finite], 97.5)) if n_val > 0 else np.nan
                per_fold_point[mk].append(pt_map[mk])
                macro_diffs[mk] = macro_diffs[mk] + diff_vec / 6.0
                if mk == 'recall_g3_pp':
                    macro_valid &= finite

                out_rows.append({
                    'contrast': c_id,
                    'contrast_label': c_label,
                    'scope': f"fold_{b['fold']}_{b['cohort']}",
                    'fold_id': b['fold'],
                    'cohort': b['cohort'],
                    'metric': mk,
                    'point_estimate': pt_map[mk],
                    'ci_low_95': lo,
                    'ci_high_95': hi,
                    'n_images': b['n_images'],
                    'n_groups': b['n_groups'],
                    'reps_total': reps,
                    'valid_reps': n_val,
                    'undefined_reps': n_undef,
                    'resampling_unit': 'recorded_grouping_key_sensitivity',
                    'primary_patient_cluster_status': 'NOT_ESTIMABLE',
                    'seed': seed,
                })
                diag_rows.append({
                    'contrast': c_id,
                    'scope': b['cohort'],
                    'metric': mk,
                    'n_images': b['n_images'],
                    'n_groups': b['n_groups'],
                    'identity_status': ','.join(b['identity_statuses']),
                    'valid_reps': n_val,
                    'undefined_reps': n_undef,
                    'undefined_rule': 'g3_precision=NaN if predicted_g3==0; recall_g3=NaN if true_g3==0; qwk=NaN if expected_denom==0; percentiles over finite replicates',
                })

        # Equal-cohort mean across 6 cohorts (cohort-stratified grouping-key bootstrap)
        for mk in metric_keys:
            mvec = macro_diffs[mk]
            finite = np.isfinite(mvec)
            n_val = int(finite.sum())
            n_undef = int((~finite).sum())
            out_rows.append({
                'contrast': c_id,
                'contrast_label': c_label,
                'scope': 'equal_cohort_mean_6',
                'fold_id': -2,
                'cohort': 'equal_cohort_mean_6',
                'metric': mk,
                'point_estimate': float(np.mean(per_fold_point[mk])),
                'ci_low_95': float(np.percentile(mvec[finite], 2.5)) if n_val > 0 else np.nan,
                'ci_high_95': float(np.percentile(mvec[finite], 97.5)) if n_val > 0 else np.nan,
                'n_images': int(sum(b['n_images'] for b in fold_bundles)),
                'n_groups': int(sum(b['n_groups'] for b in fold_bundles)),
                'reps_total': reps,
                'valid_reps': n_val,
                'undefined_reps': n_undef,
                'resampling_unit': 'recorded_grouping_key_sensitivity_macro6',
                'primary_patient_cluster_status': 'NOT_ESTIMABLE',
                'seed': seed,
            })

    df_e = pd.DataFrame(out_rows)
    df_diag = pd.DataFrame(diag_rows)
    df_e.to_csv(ctx.output_dir / 'analysis_e_grouping_key_bootstrap.csv', index=False)
    df_diag.to_csv(ctx.output_dir / 'analysis_e_bootstrap_diagnostics.csv', index=False)
    status = dict(
        analysis_id='E',
        title='Paired recorded-grouping-key bootstrap sensitivity analysis',
        status='COMPLETE',
        reason=(
            f'Completed {reps}-replicate recorded-grouping-key bootstrap sensitivity analysis (seed={seed}). '
            'Primary patient-cluster CIs remain NOT_ESTIMABLE pending clinical identity verification.'
        ),
        output_files='analysis_e_grouping_key_bootstrap.csv; analysis_e_bootstrap_diagnostics.csv',
        source_paths=f'{ctx.p0_dir}; {ctx.e0_dir}; {ctx.e1_dir}',
    )
    return df_e, df_diag, status


# %% [markdown]
# ## Analysis F — Probe regularization $C$ selection by source-validation QWK

# %%
def _load_linear_state(arrays_path: Path) -> Dict[str, np.ndarray]:
    with np.load(arrays_path, allow_pickle=False) as arr:
        return {k: np.array(arr[k], copy=True) for k in ('scaler_mean', 'scaler_scale', 'coef', 'intercept', 'classes')}


def _predict_from_state(x: np.ndarray, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xs = (x - state['scaler_mean']) / state['scaler_scale']
    logits = xs @ state['coef'].T + state['intercept']
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = state['classes'][probs.argmax(axis=1)]
    return pred, probs


def _predict_saved_linear(x: np.ndarray, arrays_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    return _predict_from_state(x, _load_linear_state(arrays_path))


def run_analysis_f(
    ctx: AnalysisContext,
    folds: Sequence[int] = tuple(range(6)),
    reps: Sequence[Tuple[str, str]] = (('z_v2', 'V2'), ('z_v3', 'V3'), ('s_v3', 'V3')),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Select probe C by source-validation QWK and compare against E1 macro-recall selection."""
    tie_rule = 'converged==True; sort by (-val_qwk, -val_macro_recall, -val_g3, val_nll, C)'
    cand_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []

    for fold in folds:
        cohort = COHORTS[fold]
        for rep, variant in reps:
            df_val = _read_p0_predictions(ctx, variant, fold, 'source_val')
            df_tgt = _read_p0_predictions(ctx, variant, fold, 'target')
            y_val = df_val['y_true'].to_numpy(int)
            y_tgt = df_tgt['y_true'].to_numpy(int)
            if rep == 's_v3':
                x_val = df_val['s'].to_numpy(np.float64).reshape(-1, 1)
                x_tgt = df_tgt['s'].to_numpy(np.float64).reshape(-1, 1)
            else:
                z_val = np.load(ctx.p0_dir / 'lodo' / f'{variant}_fold_{fold}_{cohort}' / 'source_val' / 'z.npy', mmap_mode='r', allow_pickle=False)
                z_tgt = np.load(ctx.p0_dir / 'lodo' / f'{variant}_fold_{fold}_{cohort}' / 'target' / 'z.npy', mmap_mode='r', allow_pickle=False)
                x_val = np.asarray(z_val, dtype=np.float64)
                x_tgt = np.asarray(z_tgt, dtype=np.float64)
            x_tr: Optional[np.ndarray] = None
            y_tr: Optional[np.ndarray] = None

            sel_path = ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / 'fold_report' / f'{rep}_selection.json'
            c_macro_saved = float(json.loads(sel_path.read_text())['selected']['C']) if sel_path.is_file() else None

            fold_cands: List[Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, np.ndarray]]] = []
            macro_cands: List[Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, np.ndarray]]] = []
            for c in C_GRID:
                c_dir = ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / rep / f'C_{c:g}'
                res_json = c_dir / 'result.json'
                arr_npz = c_dir / 'arrays.npz'
                if res_json.is_file() and arr_npz.is_file():
                    meta = json.loads(res_json.read_text())
                    converged = bool(meta['converged'])
                    state = _load_linear_state(arr_npz)
                    pred_val, probs_val = _predict_from_state(x_val, state)
                    source_mode = 'saved_e1_candidate_state'
                else:
                    # Fallback CPU refit strictly on source_train; retain fitted state in memory (never write into ctx.e1_dir)
                    if x_tr is None:
                        df_tr = _read_p0_predictions(ctx, variant, fold, 'source_train')
                        y_tr = df_tr['y_true'].to_numpy(int)
                        if rep == 's_v3':
                            x_tr = df_tr['s'].to_numpy(np.float64).reshape(-1, 1)
                        else:
                            x_tr = np.asarray(np.load(ctx.p0_dir / 'lodo' / f'{variant}_fold_{fold}_{cohort}' / 'source_train' / 'z.npy', mmap_mode='r'), dtype=np.float64)
                    scaler = StandardScaler().fit(x_tr)
                    model = LogisticRegression(C=c, solver='lbfgs', max_iter=5000, tol=1e-6, class_weight=None, random_state=42)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always', ConvergenceWarning)
                        model.fit(scaler.transform(x_tr), y_tr)
                    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    if not np.array_equal(model.classes_, np.arange(5)):
                        raise ValueError('Probe classes do not map to grades 0..4')
                    state = {
                        'scaler_mean': np.array(scaler.mean_, dtype=np.float64, copy=True),
                        'scaler_scale': np.array(scaler.scale_, dtype=np.float64, copy=True),
                        'coef': np.array(model.coef_, dtype=np.float64, copy=True),
                        'intercept': np.array(model.intercept_, dtype=np.float64, copy=True),
                        'classes': np.array(model.classes_, dtype=int, copy=True),
                    }
                    probs_val = model.predict_proba(scaler.transform(x_val))
                    pred_val = model.classes_[probs_val.argmax(1)]
                    source_mode = 'cpu_refit'

                mv = compute_metrics(y_val, pred_val)
                val_nll = float(-np.log(np.maximum(probs_val[np.arange(len(y_val)), y_val], 1e-12)).mean())
                cand_rec = {
                    'fold_id': fold,
                    'cohort': cohort,
                    'representation': rep,
                    'C': float(c),
                    'converged': converged,
                    'val_qwk': mv['qwk'],
                    'val_within1': mv['within1'],
                    'val_macro_recall': mv['macro_recall'],
                    'val_g3_recall': mv['recall_g3'],
                    'val_nll': val_nll,
                    'source_mode': source_mode,
                    'tie_breaker': tie_rule,
                }
                cand_rows.append(cand_rec)
                if converged:
                    qwk_key = (mv['qwk'], mv['macro_recall'], mv['recall_g3'], -val_nll, -float(c))
                    mac_key = (mv['macro_recall'], mv['recall_g3'], -val_nll, -float(c))
                    fold_cands.append((qwk_key, cand_rec, state))
                    macro_cands.append((mac_key, cand_rec, state))

            if not fold_cands:
                raise RuntimeError(f'No converged candidates for {cohort}/{rep}')
            _, best_cand, best_state = max(fold_cands, key=lambda item: item[0])
            c_qwk = float(best_cand['C'])
            _, mac_cand, mac_state = max(macro_cands, key=lambda item: item[0])
            c_macro = c_macro_saved if c_macro_saved is not None else float(mac_cand['C'])

            e1_tgt_path = ctx.e1_dir / 'units' / f'fold_{fold}_{cohort}' / rep / 'predict_target' / 'predictions.csv'
            if e1_tgt_path.is_file():
                e1_tgt_df = pd.read_csv(e1_tgt_path, dtype={'image_key': str})
                m_df_macro = _merge_on_image_key(df_tgt[['image_key', 'y_true']], e1_tgt_df, ['prediction'], f'F_macro_{cohort}_{rep}')
                mt_macro = compute_metrics(m_df_macro['y_true'], m_df_macro['prediction'])
            else:
                pred_tgt_mac, _ = _predict_from_state(x_tgt, mac_state)
                mt_macro = compute_metrics(y_tgt, pred_tgt_mac)

            # Evaluate held-out target strictly AFTER source-only selection using the winning candidate's actual state
            pred_tgt_qwk, _ = _predict_from_state(x_tgt, best_state)
            mt_qwk = compute_metrics(y_tgt, pred_tgt_qwk)
            if e1_tgt_path.is_file() and c_qwk == c_macro and best_cand['source_mode'] == 'saved_e1_candidate_state':
                if not np.array_equal(pred_tgt_qwk, m_df_macro['prediction'].to_numpy(int)):
                    raise ValueError(f'Target prediction mismatch when C_qwk == C_macro on {cohort}/{rep}')

            comp_rows.append({
                'fold_id': fold,
                'cohort': cohort,
                'cohort_label': COHORT_LABELS[cohort],
                'representation': rep,
                'C_macro_recall': c_macro,
                'C_qwk': c_qwk,
                'c_changed': bool(c_qwk != c_macro),
                'winning_state_origin': best_cand['source_mode'],
                'macro_recall_g3_pct': mt_macro['recall_g3'] * 100.0,
                'qwk_sel_recall_g3_pct': mt_qwk['recall_g3'] * 100.0,
                'delta_recall_g3_pp': (mt_qwk['recall_g3'] - mt_macro['recall_g3']) * 100.0,
                'macro_g3_precision_pct': mt_macro['g3_precision'] * 100.0,
                'qwk_sel_g3_precision_pct': mt_qwk['g3_precision'] * 100.0,
                'delta_g3_precision_pp': (mt_qwk['g3_precision'] - mt_macro['g3_precision']) * 100.0,
                'macro_within1_pct': mt_macro['within1'] * 100.0,
                'qwk_sel_within1_pct': mt_qwk['within1'] * 100.0,
                'delta_within1_pp': (mt_qwk['within1'] - mt_macro['within1']) * 100.0,
                'macro_qwk': mt_macro['qwk'],
                'qwk_sel_qwk': mt_qwk['qwk'],
                'delta_qwk': mt_qwk['qwk'] - mt_macro['qwk'],
                'macro_mae': mt_macro['mae'],
                'qwk_sel_mae': mt_qwk['mae'],
                'delta_mae': mt_qwk['mae'] - mt_macro['mae'],
                'tie_breaker': tie_rule,
            })

    df_cand = pd.DataFrame(cand_rows)
    df_comp = pd.DataFrame(comp_rows)
    num_cols = [c for c in df_comp.columns if c not in ('fold_id', 'cohort', 'cohort_label', 'representation', 'C_macro_recall', 'C_qwk', 'c_changed', 'winning_state_origin', 'tie_breaker')]
    df_sum = df_comp.groupby('representation', as_index=False)[num_cols].mean()
    n_chg = df_comp.groupby('representation', as_index=False)['c_changed'].sum().rename(columns={'c_changed': 'n_folds_c_changed'})
    df_comp['_is_refit'] = (df_comp['winning_state_origin'] == 'cpu_refit').astype(int)
    df_comp['_is_cached'] = (df_comp['winning_state_origin'] == 'saved_e1_candidate_state').astype(int)
    n_orig = df_comp.groupby('representation', as_index=False)[['_is_cached', '_is_refit']].sum().rename(
        columns={'_is_cached': 'n_folds_cached_winner', '_is_refit': 'n_folds_cpu_refit_winner'}
    )
    df_comp = df_comp.drop(columns=['_is_refit', '_is_cached'])
    df_sum = df_sum.merge(n_chg, on='representation').merge(n_orig, on='representation')

    rep_set = set(df_sum['representation'])
    if {'s_v3', 'z_v3'} <= rep_set:
        s3_qwk_g3 = float(df_sum.loc[df_sum['representation'] == 's_v3', 'qwk_sel_recall_g3_pct'].iloc[0])
        z3_qwk_g3 = float(df_sum.loc[df_sum['representation'] == 'z_v3', 'qwk_sel_recall_g3_pct'].iloc[0])
        s3_mac_g3 = float(df_sum.loc[df_sum['representation'] == 's_v3', 'macro_recall_g3_pct'].iloc[0])
        z3_mac_g3 = float(df_sum.loc[df_sum['representation'] == 'z_v3', 'macro_recall_g3_pct'].iloc[0])
        s3_gt_mac = bool(s3_mac_g3 > z3_mac_g3)
        s3_gt_qwk = bool(s3_qwk_g3 > z3_qwk_g3)
        preserved = bool(s3_gt_mac == s3_gt_qwk)
        reason_txt = (
            f'Evaluated all converged C candidates; s3 > z3 on mean target G3 recall '
            f'{"preserved" if preserved else "changed"} ({s3_qwk_g3:.2f}% vs {z3_qwk_g3:.2f}%, ranking_preserved={preserved}).'
        )
    else:
        s3_gt_mac, s3_gt_qwk, preserved = np.nan, np.nan, np.nan
        reason_txt = f'Evaluated converged C candidates for {sorted(rep_set)} across {len(folds)} fold(s).'

    df_sum['s3_gt_z3_under_macro_recall'] = s3_gt_mac
    df_sum['s3_gt_z3_under_qwk'] = s3_gt_qwk
    df_sum['ranking_preserved'] = preserved
    df_sum['aggregation'] = f'equal-cohort mean (1/{len(folds)})'

    df_cand.to_csv(ctx.output_dir / 'analysis_f_probe_qwk_candidates.csv', index=False)
    df_comp.to_csv(ctx.output_dir / 'analysis_f_probe_comparison_per_fold.csv', index=False)
    df_sum.to_csv(ctx.output_dir / 'analysis_f_probe_comparison_summary.csv', index=False)
    status = dict(
        analysis_id='F',
        title='Probe regularization C selected by source-validation QWK',
        status='COMPLETE',
        reason=reason_txt,
        output_files='analysis_f_probe_qwk_candidates.csv; analysis_f_probe_comparison_per_fold.csv; analysis_f_probe_comparison_summary.csv',
        source_paths=f'{ctx.p0_dir}; {ctx.e1_dir}',
    )
    return df_cand, df_comp, df_sum, status


# %% [markdown]
# ## Analysis G — CIBD with 0.5 pp (`0.005`) source-validation Within-1 loss budget

# %%
def _select_from_grid(source_grid: pd.DataFrame, native_val_within1: float, budget: float) -> Tuple[Dict[str, Any], pd.Series]:
    feasible = (source_grid['within1'] >= native_val_within1 - budget) & source_grid['recall_g3'].notna()
    eligible = source_grid[feasible]
    if eligible.empty:
        return dict(alpha=np.nan, tau=np.nan, q=np.nan, fallback=True), pd.Series(dtype=object)
    best = eligible.sort_values(
        ['recall_g3', 'macro_recall', 'within1', 'qwk', 'tau', 'q', 'alpha'],
        ascending=[False, False, False, False, True, True, True],
    ).iloc[0]
    return dict(alpha=float(best['alpha']), tau=float(best['tau']), q=int(best['q']), fallback=False), best


def _lookup_target_row(target_grid: pd.DataFrame, sel: Dict[str, Any], native_target_row: pd.Series) -> pd.Series:
    if sel['fallback']:
        return native_target_row
    hit = target_grid[
        np.isclose(target_grid['alpha'], sel['alpha'])
        & np.isclose(target_grid['tau'], sel['tau'])
        & (target_grid['q'].astype(int) == int(sel['q']))
    ]
    if len(hit) != 1:
        raise ValueError(f'Expected 1 matching target grid row for {sel}, got {len(hit)}')
    return hit.iloc[0]


def run_analysis_g(ctx: AnalysisContext) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Reproduce 1.0 pp E0 CIBD selection and evaluate a tighter 0.5 pp source Within-1 budget."""
    dec_df = pd.read_csv(ctx.e0_dir / 'lodo' / 'decision_metrics.csv')
    fold_rows: List[Dict[str, Any]] = []

    for variant in ('V2', 'V3'):
        for fold, cohort in enumerate(COHORTS):
            fr = ctx.e0_dir / 'lodo' / 'units' / f'{variant}_fold_{fold}_{cohort}' / 'fold_report'
            sg = pd.read_csv(fr / 'source_grid.csv')
            tg = pd.read_csv(fr / 'target_grid_EXPLORATORY.csv')
            saved_sel = json.loads((fr / 'selected_params.json').read_text())['selected']

            nat_val = dec_df[(dec_df['variant'] == variant) & (dec_df['fold_id'] == fold) & (dec_df['split'] == 'source_val') & (dec_df['decoder'] == 'native_replay')].iloc[0]
            nat_tgt = dec_df[(dec_df['variant'] == variant) & (dec_df['fold_id'] == fold) & (dec_df['split'] == 'target') & (dec_df['decoder'] == 'native_replay')].iloc[0]

            sel_10, row_val_10 = _select_from_grid(sg, float(nat_val['within1']), 0.010)
            if (sel_10['fallback'] != bool(saved_sel.get('fallback', False))) or (
                not sel_10['fallback'] and (
                    not np.isclose(sel_10['alpha'], saved_sel['alpha'])
                    or not np.isclose(sel_10['tau'], saved_sel['tau'])
                    or int(sel_10['q']) != int(saved_sel['q'])
                )
            ):
                raise ValueError(f'1.0 pp baseline reproduction failed on {variant}/{cohort}: {sel_10} vs {saved_sel}')

            sel_05, row_val_05 = _select_from_grid(sg, float(nat_val['within1']), 0.005)
            row_tgt_10 = _lookup_target_row(tg, sel_10, nat_tgt)
            row_tgt_05 = _lookup_target_row(tg, sel_05, nat_tgt)

            params_changed = (
                sel_05['fallback'] != sel_10['fallback']
                or (not sel_05['fallback'] and (
                    not np.isclose(sel_05['alpha'], sel_10['alpha'])
                    or not np.isclose(sel_05['tau'], sel_10['tau'])
                    or int(sel_05['q']) != int(sel_10['q'])
                ))
            )

            margin_10_pp = (float(row_val_10['within1']) - (float(nat_val['within1']) - 0.010)) * 100.0 if not sel_10['fallback'] else np.nan
            margin_05_pp = (float(row_val_05['within1']) - (float(nat_val['within1']) - 0.005)) * 100.0 if not sel_05['fallback'] else np.nan

            for label, budget_pp, sel, r_tgt, margin_pp in [
                ('cibd_1.0pp_budget', 1.0, sel_10, row_tgt_10, margin_10_pp),
                ('cibd_0.5pp_budget', 0.5, sel_05, row_tgt_05, margin_05_pp),
            ]:
                d_g3_pp = (float(r_tgt['recall_g3']) - float(nat_tgt['recall_g3'])) * 100.0
                d_w1_pp = (float(r_tgt['within1']) - float(nat_tgt['within1'])) * 100.0
                fold_rows.append({
                    'variant': variant,
                    'fold_id': fold,
                    'cohort': cohort,
                    'cohort_label': COHORT_LABELS[cohort],
                    'rule': label,
                    'source_budget_pp': budget_pp,
                    'fallback': bool(sel['fallback']),
                    'alpha': sel['alpha'],
                    'tau': sel['tau'],
                    'q': sel['q'],
                    'params_changed_vs_1pp': bool(params_changed) if label == 'cibd_0.5pp_budget' else False,
                    'source_within1_margin_above_threshold_pp': margin_pp,
                    'native_target_g3_recall_pct': float(nat_tgt['recall_g3']) * 100.0,
                    'target_g3_recall_pct': float(r_tgt['recall_g3']) * 100.0,
                    'delta_g3_recall_pp': d_g3_pp,
                    'target_g3_precision_pct': float(r_tgt['g3_precision']) * 100.0,
                    'native_target_within1_pct': float(nat_tgt['within1']) * 100.0,
                    'target_within1_pct': float(r_tgt['within1']) * 100.0,
                    'delta_within1_pp': d_w1_pp,
                    'target_qwk': float(r_tgt['qwk']),
                    'delta_qwk': float(r_tgt['qwk']) - float(nat_tgt['qwk']),
                    'target_mae': float(r_tgt['mae']),
                    'delta_mae': float(r_tgt['mae']) - float(nat_tgt['mae']),
                    'meets_target_g3_margin_3pp': bool(d_g3_pp >= 3.0),
                    'meets_target_within1_margin_1pp': bool(d_w1_pp >= -1.0),
                    'meets_target_joint_hd': bool(d_g3_pp >= 3.0 and d_w1_pp >= -1.0),
                })

    df_g = pd.DataFrame(fold_rows)
    num_cols = [
        'source_within1_margin_above_threshold_pp', 'native_target_g3_recall_pct', 'target_g3_recall_pct',
        'delta_g3_recall_pp', 'target_g3_precision_pct', 'native_target_within1_pct', 'target_within1_pct',
        'delta_within1_pp', 'target_qwk', 'delta_qwk', 'target_mae', 'delta_mae',
    ]
    df_sum = df_g.groupby(['variant', 'rule', 'source_budget_pp'], as_index=False)[num_cols].mean()
    cnt_cols = ['params_changed_vs_1pp', 'fallback', 'meets_target_g3_margin_3pp', 'meets_target_within1_margin_1pp', 'meets_target_joint_hd']
    df_cnt = df_g.groupby(['variant', 'rule', 'source_budget_pp'], as_index=False)[cnt_cols].sum()
    df_sum = df_sum.merge(df_cnt, on=['variant', 'rule', 'source_budget_pp'])
    df_sum['aggregation'] = 'equal-cohort mean (1/6); target H_D uses original >=+3 pp G3 and >=-1 pp Within-1 margins'

    df_g.to_csv(ctx.output_dir / 'analysis_g_cibd_05pp_per_fold.csv', index=False)
    df_sum.to_csv(ctx.output_dir / 'analysis_g_cibd_05pp_summary.csv', index=False)
    status = dict(
        analysis_id='G',
        title='CIBD with 0.5 pp source-validation Within-1 loss budget',
        status='COMPLETE',
        reason='Verified 12/12 exact reproduction of 1.0 pp baseline, then evaluated locked 0.5 pp source selections on target.',
        output_files='analysis_g_cibd_05pp_per_fold.csv; analysis_g_cibd_05pp_summary.csv',
        source_paths=f'{ctx.e0_dir}',
    )
    return df_g, df_sum, status


# %% [markdown]
# ## Report generation and CLI / Notebook runners

# %%
def _format_name_list(names: Sequence[str]) -> str:
    """Format a sequence of names cleanly for English prose."""
    items = [str(x) for x in names]
    if not items:
        return 'none'
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def _fold_count_phrase(count: int, total: int) -> str:
    if total <= 0:
        return f'{count}/{total} folds'
    if count == total:
        return f'all {total} folds'
    return f'{count} of {total} folds'


def write_supplementary_report(ctx: AnalysisContext, statuses: List[Dict[str, Any]], results: Dict[str, Any]) -> Path:
    """Write status CSV, provenance JSON, and Vietnamese Markdown report derived strictly from current results."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(ctx.output_dir / 'analysis_status.csv', index=False)

    versions = {
        name: package_metadata.version(name)
        for name in ('numpy', 'pandas', 'scikit-learn', 'scipy')
    }
    start_prov = dict(ctx.code_provenance or {})
    code_path_str = start_prov.get('code_path')
    code_path_obj = Path(code_path_str) if code_path_str else (Path(__file__).resolve() if '__file__' in globals() else None)
    code_sha_start = start_prov.get('code_sha256_start')
    if code_sha_start is None:
        code_sha_start = hashlib.sha256(code_path_obj.read_bytes()).hexdigest() if (code_path_obj and code_path_obj.is_file()) else NOTEBOOK_CODE_SHA256
    code_sha_end = hashlib.sha256(code_path_obj.read_bytes()).hexdigest() if (code_path_obj and code_path_obj.is_file()) else NOTEBOOK_CODE_SHA256
    script_path_obj = Path(__file__).resolve() if '__file__' in globals() else None
    script_sha_end = hashlib.sha256(script_path_obj.read_bytes()).hexdigest() if (script_path_obj and script_path_obj.is_file()) else None
    code_modified_during_run = bool(code_sha_start != code_sha_end)
    selected_analyses = [str(k) for k in results.keys() if k != 'baseline']
    if not selected_analyses and statuses:
        selected_analyses = [str(s['analysis_id']) for s in statuses if 'analysis_id' in s]

    prov = {
        'repo_root': str(ctx.repo_root) if ctx.repo_root else 'standalone_colab_or_external',
        'in_colab': IN_COLAB,
        'execution_mode': start_prov.get('execution_mode', ctx.execution_mode),
        'selected_analyses': selected_analyses,
        'p0_dir': str(ctx.p0_dir),
        'e0_dir': str(ctx.e0_dir),
        'e1_dir': str(ctx.e1_dir),
        'reference_json': str(ctx.reference_json) if ctx.reference_json else None,
        'archived_preds_dir': str(ctx.archived_preds_dir) if ctx.archived_preds_dir else None,
        'folds_dir': str(ctx.folds_dir) if ctx.folds_dir else None,
        'output_dir': str(ctx.output_dir),
        'seed': ctx.seed,
        'bootstrap_reps': ctx.bootstrap_reps,
        'code_path': str(code_path_obj) if code_path_obj else None,
        'code_sha256': code_sha_end,
        'code_sha256_start': code_sha_start,
        'code_sha256_end': code_sha_end,
        'code_modified_during_run': code_modified_during_run,
        'script_sha256_start': start_prov.get('script_sha256_start'),
        'script_sha256_end': script_sha_end,
        'notebook_embedded_code_sha256': start_prov.get('notebook_embedded_code_sha256', NOTEBOOK_CODE_SHA256),
        'git_commit': start_prov.get('git_commit'),
        'git_dirty_relevant_files': start_prov.get('git_dirty_relevant_files'),
        'git_dirty_paths': start_prov.get('git_dirty_paths', []),
        'versions': versions,
    }
    (ctx.output_dir / 'provenance.json').write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# Báo cáo Phân tích Bổ sung (Post Hoc, CPU-Only)',
        '',
        '## 1. Truy vết nguồn và kiểm chứng nền tảng',
        f"- **Kho mã nguồn**: `{prov['repo_root']}` (`mode={prov['execution_mode']}`, `git_commit={prov['git_commit']}`, `dirty_relevant={prov['git_dirty_relevant_files']}`)",
        f"- **Mã nguồn SHA-256 (start -> end)**: `{code_sha_start[:12]}...` -> `{code_sha_end[:12]}...` (`modified_during_run={code_modified_during_run}`)",
        f"- **Phân tích được chọn trong lượt chạy**: `{', '.join(selected_analyses) if selected_analyses else 'NONE'}`",
        f"- **P0 cache (chỉ đọc)**: `{ctx.p0_dir}`",
        f"- **E0 cache (chỉ đọc)**: `{ctx.e0_dir}`",
        f"- **E1 cache (chỉ đọc)**: `{ctx.e1_dir}`",
        f"- **Archived reference / NPZ (chỉ đọc)**: `{ctx.reference_json}` / `{ctx.archived_preds_dir}`",
        f"- **Thư mục đầu ra mới**: `{ctx.output_dir}` (seed={ctx.seed}, bootstrap_reps={ctx.bootstrap_reps}, phương pháp tổng hợp: equal-cohort mean `1/6` trừ khi ghi rõ `pooled`)",
        '',
        '### Kiểm chứng baseline từ dữ liệu theo fold',
    ]
    if code_modified_during_run:
        lines.insert(6, f"- **CẢNH BÁO**: Tệp mã nguồn đã thay đổi trong lúc chạy (`{code_sha_start}` != `{code_sha_end}`).")
    if 'baseline' in results and isinstance(results['baseline'], pd.DataFrame) and not results['baseline'].empty:
        for _, r in results['baseline'].iterrows():
            lines.append(f"- {r['check']}: observed `{r['observed']}` vs expected `{r['expected']}` (`matched={r['matched']}`)")
    else:
        lines.append('- Không có bảng kiểm chứng baseline trong phiên chạy này.')
    lines.extend(['', '## 2. Trạng thái thực thi (analysis_status.csv)', ''])
    for s in statuses:
        lines.append(f"- **{s['analysis_id']} ({s['title']})**: `{s['status']}` — {s['reason']}")

    # Summarize verified findings strictly from current invocation `results`
    lines.extend(['', '## 3. Tóm tắt phát hiện mới so với kết quả chính'])
    if 'A' in results:
        a_fold, a_sum = results['A'][0], results['A'][1]
        n_a_cohorts = int(a_fold['cohort'].nunique()) if ('cohort' in a_fold.columns and not a_fold.empty) else 6
        lines.append(f'### A. So sánh V2 và V3 dưới cùng decoder (equal-cohort mean trên {n_a_cohorts} cohort, target)')
        for _, r in a_sum.iterrows():
            v2_g3 = f"{r['v2_recall_g3_pct']:.2f}%" if pd.notna(r['v2_recall_g3_pct']) else 'N/A'
            v3_g3 = f"{r['v3_recall_g3_pct']:.2f}%" if pd.notna(r['v3_recall_g3_pct']) else 'N/A'
            dg3 = f"{r['diff_v3_minus_v2_recall_g3_pp']:+.2f} pp" if pd.notna(r['diff_v3_minus_v2_recall_g3_pp']) else 'N/A'
            v2_w1 = f"{r['v2_within1_pct']:.2f}%" if pd.notna(r['v2_within1_pct']) else 'N/A'
            v3_w1 = f"{r['v3_within1_pct']:.2f}%" if pd.notna(r['v3_within1_pct']) else 'N/A'
            lines.append(f"- `{r['decoder']}`: G3 V2={v2_g3}, V3={v3_g3} (V3-V2={dg3}); Within-1 V2={v2_w1}, V3={v3_w1}.")

    if 'B' in results:
        b_fold, b_sum, b_trans, b_trans_sum = results['B'][0], results['B'][1], results['B'][2], results['B'][3]
        n_b_cohorts = int(b_trans['cohort'].nunique()) if ('cohort' in b_trans.columns and not b_trans.empty) else (
            int(b_fold['cohort'].nunique()) if ('cohort' in b_fold.columns and not b_fold.empty) else 0
        )
        lines.append('### B. Recall 5 grade, tỷ lệ dự đoán G3 và dịch chuyển nhãn ghép cặp')
        for _, r in b_sum.iterrows():
            g3p = f"{r['g3_precision_pct']:.2f}%" if pd.notna(r['g3_precision_pct']) else 'N/A'
            lines.append(
                f"- `{r['method']}`: G0={r['recall_g0_pct']:.2f}%, G1={r['recall_g1_pct']:.2f}%, "
                f"G2={r['recall_g2_pct']:.2f}%, G3={r['recall_g3_pct']:.2f}%, G4={r['recall_g4_pct']:.2f}%, "
                f"G3 prec={g3p}, pred/true G3 ratio={r['predicted_true_g3_ratio']:.2f}."
            )
        for _, r in b_trans_sum.iterrows():
            lines.append(
                f"- Chuyển sang G3 (`{r['method']}` vs `{r['baseline']}`, tổng {n_b_cohorts} cohort): +{int(r['n_newly_pred_g3'])} ảnh mới được gán G3 "
                f"(true G0={int(r['new_g3_true_g0'])}, G1={int(r['new_g3_true_g1'])}, G2={int(r['new_g3_true_g2'])}, "
                f"G3={int(r['new_g3_true_g3'])}, G4={int(r['new_g3_true_g4'])}; "
                f"từ dự đoán cũ G2={int(r['new_g3_prev_pred_g2'])}, G4={int(r['new_g3_prev_pred_g4'])}); G3 đúng ròng {int(r['true_g3_net_gain']):+d}."
            )

    if 'C' in results:
        c_df, c_sum = results['C'][0], results['C'][1]
        lines.append('### C. Kiểm tra P0 theo số ảnh và truy vết liên kết ID từng ảnh')
        if c_sum.empty or c_df.empty:
            c_stat = next((s for s in statuses if s.get('analysis_id') == 'C'), {})
            lines.append(f"- Không có bảng số liệu liên kết: `{c_stat.get('status', 'NOT_ESTIMABLE')}` — {c_stat.get('reason', '')}")
        else:
            for _, r in c_sum.iterrows():
                ag_st = str(r['agreement_status'])
                ag_scope = str(r.get('agreement_scope', ''))
                if pd.notna(r.get('prediction_agreement_rate_matched')):
                    link_txt = (
                        f"status=`{ag_st}` ({ag_scope}), matched={int(r['n_matched_by_id_total'])}, "
                        f"agreement={r['prediction_agreement_rate_matched']*100:.2f}% "
                        f"(same={int(r['n_same_prediction_total'])}, changed={int(r['n_changed_prediction_total'])}, "
                        f"G3 wrong->right={int(r['g3_wrong_to_right_total'])}, right->wrong={int(r['g3_right_to_wrong_total'])})"
                    )
                elif pd.notna(r.get('n_matched_by_id_total')):
                    link_txt = (
                        f"status=`{ag_st}` ({ag_scope}), matched={int(r['n_matched_by_id_total'])}, "
                        f"agreement=NaN (không có cặp trùng ID)"
                    )
                else:
                    link_txt = f"status=`{ag_st}` ({ag_scope}), ID linkage/agreement=`NOT_ESTIMABLE` (`NaN`)"
                lines.append(
                    f"- `{r['variant']}` `{r['split']}`: n_arch={int(r['n_archived_total'])}, n_replay={int(r['n_replay_total'])} "
                    f"(khớp support 100%: {bool(r['all_supports_identical'])}); {link_txt}; "
                    f"G3 đúng archived={int(r['n_g3_correct_archived_total'])} vs replay={int(r['n_g3_correct_replay_total'])} "
                    f"(chênh ròng {int(r['delta_g3_correct_net_total']):+d}); equal-cohort G3={r['equal_cohort_recall_g3_archived_pct']:.2f}% -> {r['equal_cohort_recall_g3_replay_pct']:.2f}%; "
                    f"G0 net={int(r['delta_correct_g0_net_total']):+d}, G1 net={int(r['delta_correct_g1_net_total']):+d}."
                )
            tgt_c = c_df[c_df['split'] == 'target']
            if not tgt_c.empty:
                lines.append(f"- Biên độ chênh lệch G3 target theo từng fold: từ {int(tgt_c['delta_g3_correct_net'].min()):+d} đến {int(tgt_c['delta_g3_correct_net'].max()):+d} ảnh.")

    if 'D' in results:
        d_df, d_sum = results['D'][0], results['D'][1]
        ds = d_sum.iloc[0]
        n_d_folds = int(ds['n_folds'])
        n_d_up = int(ds['n_folds_upward_shift'])
        lines.append(f'### D. Max p3 và độ chồng vùng G3 trên {n_d_folds} fold V3')
        lines.append(
            f"- Trên lưới latent float64: `max p3` nằm trong [{ds['max_p3_grid_min']:.4f}, {ds['max_p3_grid_max']:.4f}] "
            f"(trên ảnh target thực: [{ds['max_p3_target_images_min']:.4f}, {ds['max_p3_target_images_max']:.4f}])."
        )
        idrid_txt = (
            f" (IDRiD: {ds['idrid_overlap_ratio_grid']*100:.2f}%)"
            if ('idrid_overlap_ratio_grid' in ds and pd.notna(ds['idrid_overlap_ratio_grid'])) else ''
        )
        lines.append(
            f"- Dịch biên lên ở cả hai đầu xảy ra ở **{n_d_up}/{n_d_folds}** fold; "
            f"tỷ lệ chồng `overlap / cutpoint_length` trên lưới từ {ds['overlap_ratio_grid_min']*100:.1f}% đến {ds['overlap_ratio_grid_max']*100:.1f}%{idrid_txt}."
        )

    if 'E' in results:
        e_df = results['E'][0]
        lines.append(f'### E. Recorded-grouping-key bootstrap sensitivity analysis ({ctx.bootstrap_reps:,} lượt, seed={ctx.seed})')
        for _, r in e_df[e_df['scope'] == 'equal_cohort_mean_6'].iterrows():
            if r['metric'] in ('recall_g3_pp', 'within1_pp', 'g3_precision_pp', 'qwk', 'mae'):
                lines.append(
                    f"- `{r['contrast']}` `{r['metric']}`: point={r['point_estimate']:+.4f}, "
                    f"95% CI=[{r['ci_low_95']:+.4f}, {r['ci_high_95']:+.4f}] (valid_reps={int(r['valid_reps'])}/{int(r['reps_total'])})."
                )
        lines.append('- Ghi chú: Khoảng tin cậy `patient_cluster` trong kết quả chính vẫn giữ `NOT_ESTIMABLE` do khóa nhóm là `legacy_unverified`.')

    if 'F' in results:
        f_comp, f_sum = results['F'][1], results['F'][2]
        n_f_folds = int(f_comp['fold_id'].nunique()) if ('fold_id' in f_comp.columns and not f_comp.empty) else 6
        lines.append('### F. Chọn C của probe theo source-validation QWK')
        for _, r in f_sum.iterrows():
            lines.append(
                f"- `{r['representation']}`: đổi C ở {int(r['n_folds_c_changed'])}/{n_f_folds} fold "
                f"(winner origin: cached={int(r.get('n_folds_cached_winner', 0))}, cpu_refit={int(r.get('n_folds_cpu_refit_winner', 0))}); "
                f"mean target G3 recall {r['macro_recall_g3_pct']:.2f}% (macro-recall) -> {r['qwk_sel_recall_g3_pct']:.2f}% (QWK, "
                f"delta={r['delta_recall_g3_pp']:+.2f} pp), Within-1 delta={r['delta_within1_pp']:+.2f} pp, QWK delta={r['delta_qwk']:+.4f}."
            )
        rep_set_f = set(f_sum['representation'])
        if {'s_v3', 'z_v3'} <= rep_set_f:
            s3_qwk = float(f_sum.loc[f_sum['representation'] == 's_v3', 'qwk_sel_recall_g3_pct'].iloc[0])
            z3_qwk = float(f_sum.loc[f_sum['representation'] == 'z_v3', 'qwk_sel_recall_g3_pct'].iloc[0])
            s3_mac = float(f_sum.loc[f_sum['representation'] == 's_v3', 'macro_recall_g3_pct'].iloc[0])
            z3_mac = float(f_sum.loc[f_sum['representation'] == 'z_v3', 'macro_recall_g3_pct'].iloc[0])
            preserved = bool((s3_mac > z3_mac) == (s3_qwk > z3_qwk))
            order_desc = '`s_v3 > z_v3`' if s3_qwk > z3_qwk else ('`z_v3 > s_v3`' if z3_qwk > s3_qwk else '`s_v3 == z_v3`')
            lines.append(
                f"- So sánh `s_v3` vs `z_v3` theo equal-cohort target G3 recall: {order_desc} "
                f"({s3_qwk:.2f}% vs {z3_qwk:.2f}%; ranking preserved vs macro-recall: `{preserved}`)."
            )

    if 'G' in results:
        g_fold, g_sum = results['G'][0], results['G'][1]
        lines.append('### G. CIBD với ngân sách mất Within-1 trên nguồn 0,5 pp (post hoc) so với 1,0 pp (chính)')
        for _, r in g_sum.iterrows():
            var, rule = r['variant'], r['rule']
            sub_g = g_fold[(g_fold['variant'] == var) & (g_fold['rule'] == rule)]
            n_g_folds = len(sub_g) if not sub_g.empty else 6
            n_joint = int(sub_g['meets_target_joint_hd'].sum()) if not sub_g.empty else int(r['meets_target_joint_hd'])
            missed = [str(x) for x in sub_g.loc[~sub_g['meets_target_joint_hd'].astype(bool), 'cohort_label'].tolist()] if not sub_g.empty else []
            mean_g3_d = float(r['delta_g3_recall_pp'])
            mean_w1_d = float(r['delta_within1_pp'])
            mean_both = bool(mean_g3_d >= 3.0 and mean_w1_d >= -1.0)
            missed_str = f"; cohort chưa đạt biên: {', '.join(missed)}" if missed else '; tất cả cohort đạt biên'
            lines.append(
                f"- `{var}` `{rule}`: đổi tham số ở {int(r['params_changed_vs_1pp'])}/{n_g_folds} fold, fallback={int(r['fallback'])}/{n_g_folds}; "
                f"equal-cohort target G3={r['target_g3_recall_pct']:.2f}% (delta={mean_g3_d:+.2f} pp), "
                f"Within-1={r['target_within1_pct']:.2f}% (delta={mean_w1_d:+.2f} pp), QWK={r['target_qwk']:.4f}; "
                f"mean đạt đồng thời hai biên target H_D={mean_both}; cấp fold đạt biên: {n_joint}/{n_g_folds}{missed_str}."
            )

    # Copy-ready English LaTeX snippets derived strictly from executed analyses in this run
    lines.extend([
        '',
        '## 4. Đoạn LaTeX tiếng Anh sẵn sàng dán vào bản thảo',
        '',
        '### Chương 4 (Phương pháp — Sensitivity & Post Hoc Checks)',
        '```latex',
        '\\paragraph{Post hoc sensitivity analyses.}',
    ])
    ch4_parts: List[str] = [
        f"To test the robustness of the main diagnostic findings without altering prespecified hypothesis decisions or model weights, we conducted post hoc CPU-only analyses ({', '.join(selected_analyses) if selected_analyses else 'baseline check'}) on the cached checkpoints and outputs."
    ]
    if 'A' in results or 'B' in results:
        ch4_parts.append(
            'For comparative decoder diagnostics, we evaluated $V_2$ and $V_3$ under identical fixed-output decoding rules and tabulated five-grade recall, predicted-to-reference Grade~3 ratios, and paired image-level transitions.'
        )
    if 'C' in results:
        ch4_parts.append(
            'For archive reconciliation, we audited per-grade image counts between archived DMIP records and P0 replay while requiring strict non-null unique per-image identifier linkage for sample-wise agreement (reporting \\texttt{NOT\\_ESTIMABLE} when identifiers are unavailable).'
        )
    if 'D' in results:
        ch4_parts.append(
            'For latent calibration geometry, we reconstructed float64 cumulative-link probabilities across $V_3$ latent grids to measure peak Grade~3 probability ($\\max p_3$) and decision-interval overlap.'
        )
    if 'E' in results:
        ch4_parts.append(
            f"For uncertainty sensitivity, we computed a {ctx.bootstrap_reps}-replicate recorded-grouping-key bootstrap analysis (seed~{ctx.seed}, equal cohort weight $1/6$) for the target differences $V_3$~CIBD minus native $V_3$ and $s_3$ probe minus native $V_3$, leaving primary patient-cluster intervals as \\texttt{{NOT\\_ESTIMABLE}}."
        )
    if 'F' in results:
        ch4_parts.append(
            'For E1 hyperparameter selection sensitivity, we repeated source-validation regularization selection using QWK (tie-broken by macro recall, Grade~3 recall, lower NLL, and smaller $C$) before evaluating held-out target cohorts.'
        )
    if 'G' in results:
        ch4_parts.append(
            'For E0 budget sensitivity, distinct from the prespecified primary 1.0~pp source-validation Within-1 loss budget, we evaluated a stricter post hoc 0.5~pp source-validation budget while keeping the prespecified target $H_D$ evaluation thresholds (+3.0~pp Grade~3 recall and $-1.0$~pp Within-1 accuracy) unchanged.'
        )
    lines.extend(ch4_parts)
    lines.extend([
        '```',
        '',
        '### Chương 5 (Kết quả — Post Hoc Verification)',
        '```latex',
        '\\paragraph{Post hoc verification and sensitivity findings.}',
    ])
    ch5_parts: List[str] = []
    if 'A' in results:
        a_sum = results['A'][1]
        a_bits = []
        for _, r in a_sum.iterrows():
            if pd.notna(r['diff_v3_minus_v2_recall_g3_pp']):
                dec_tex = str(r['decoder']).replace('_', r'\_')
                a_bits.append(f"\\texttt{{{dec_tex}}}: {r['v3_recall_g3_pct']:.2f}\\% vs.\\ {r['v2_recall_g3_pct']:.2f}\\% ({r['diff_v3_minus_v2_recall_g3_pp']:+.2f}~pp)")
        if a_bits:
            ch5_parts.append('Under matched fixed-output decoders, target equal-cohort Grade~3 recall differences ($V_3 - V_2$) were ' + '; '.join(a_bits) + '.')
    if 'B' in results:
        b_trans_sum = results['B'][3]
        if not b_trans_sum.empty:
            b_bits = [f"\\texttt{{{str(r['method']).replace('_', r'\_')}}} ({int(r['true_g3_net_gain']):+d} net true Grade~3)" for _, r in b_trans_sum.iterrows()]
            ch5_parts.append('Paired image-level transitions over native predictions yielded ' + ', '.join(b_bits) + '.')
    if 'C' in results:
        c_df, c_sum = results['C'][0], results['C'][1]
        if c_sum.empty:
            c_stat = next((s for s in statuses if s.get('analysis_id') == 'C'), {})
            st_tex = str(c_stat.get('status', 'NOT_ESTIMABLE')).replace('_', r'\_')
            ch5_parts.append(f"Archive-to-replay sample linkage status was \\texttt{{{st_tex}}}.")
        else:
            tgt_sum = c_sum[c_sum['split'] == 'target']
            c_bits = []
            for _, r in tgt_sum.iterrows():
                ag_st = str(r['agreement_status']).replace('_', r'\_')
                if pd.notna(r.get('prediction_agreement_rate_matched')):
                    c_bits.append(
                        f"{r['variant']} ({int(r['delta_g3_correct_net_total']):+d} net Grade~3, ID-matched agreement {r['prediction_agreement_rate_matched']*100:.2f}\\% across {int(r['n_matched_by_id_total'])} images, status \\texttt{{{ag_st}}})"
                    )
                else:
                    c_bits.append(
                        f"{r['variant']} ({int(r['delta_g3_correct_net_total']):+d} net Grade~3, linkage \\texttt{{{ag_st}}})"
                    )
            ch5_parts.append('Archive reconciliation across target cohorts found ' + '; '.join(c_bits) + '.')
    if 'D' in results:
        ds = results['D'][1].iloc[0]
        n_d_folds = int(ds['n_folds'])
        n_d_up = int(ds['n_folds_upward_shift'])
        ch5_parts.append(
            f"Across {n_d_folds} $V_3$ folds, peak Grade~3 probability on the diagnostic latent grid remained bounded between {ds['max_p3_grid_min']:.3f} and {ds['max_p3_grid_max']:.3f}, and native rounded expectation shifted the Grade~3 interval upward relative to the learned cutpoints in {_fold_count_phrase(n_d_up, n_d_folds)} (grid overlap {ds['overlap_ratio_grid_min']*100:.1f}\\%--{ds['overlap_ratio_grid_max']*100:.1f}\\%)."
        )
    if 'E' in results:
        e_df = results['E'][0]
        e_macro = e_df[e_df['scope'] == 'equal_cohort_mean_6']
        cibd_g3_sub = e_macro[(e_macro['contrast'] == 'cibd_v3_minus_native_v3') & (e_macro['metric'] == 'recall_g3_pp')]
        cibd_w1_sub = e_macro[(e_macro['contrast'] == 'cibd_v3_minus_native_v3') & (e_macro['metric'] == 'within1_pp')]
        s3_g3_sub = e_macro[(e_macro['contrast'] == 'probe_s3_minus_native_v3') & (e_macro['metric'] == 'recall_g3_pp')]
        if not cibd_g3_sub.empty and not cibd_w1_sub.empty and not s3_g3_sub.empty:
            cibd_g3_ci, cibd_w1_ci, s3_g3_ci = cibd_g3_sub.iloc[0], cibd_w1_sub.iloc[0], s3_g3_sub.iloc[0]
            ch5_parts.append(
                f"In the post hoc recorded-grouping-key bootstrap sensitivity analysis ({ctx.bootstrap_reps} replicates, seed~{ctx.seed}), $V_3$ CIBD changed equal-cohort Grade~3 recall by {cibd_g3_ci['point_estimate']:+.2f}~pp (95\\% CI [{cibd_g3_ci['ci_low_95']:+.2f}, {cibd_g3_ci['ci_high_95']:+.2f}]) and Within-1 accuracy by {cibd_w1_ci['point_estimate']:+.2f}~pp ([{cibd_w1_ci['ci_low_95']:+.2f}, {cibd_w1_ci['ci_high_95']:+.2f}]), while the $s_3$ probe shifted Grade~3 recall by {s3_g3_ci['point_estimate']:+.2f}~pp ([{s3_g3_ci['ci_low_95']:+.2f}, {s3_g3_ci['ci_high_95']:+.2f}])."
            )
    if 'F' in results:
        f_comp, f_sum = results['F'][1], results['F'][2]
        n_f_folds = int(f_comp['fold_id'].nunique()) if ('fold_id' in f_comp.columns and not f_comp.empty) else 6
        rep_set_f = set(f_sum['representation'])
        if {'s_v3', 'z_v3'} <= rep_set_f:
            s3_f = f_sum[f_sum['representation'] == 's_v3'].iloc[0]
            z3_f = f_sum[f_sum['representation'] == 'z_v3'].iloc[0]
            s3_val = float(s3_f['qwk_sel_recall_g3_pct'])
            z3_val = float(z3_f['qwk_sel_recall_g3_pct'])
            s3_mac = float(s3_f['macro_recall_g3_pct'])
            z3_mac = float(z3_f['macro_recall_g3_pct'])
            preserved = bool((s3_mac > z3_mac) == (s3_val > z3_val))
            if s3_val > z3_val:
                ord_phrase = 'preserving' if preserved else 'yielding'
                ord_rel = 'scalar-over-vector'
            elif z3_val > s3_val:
                ord_phrase = 'reversing the prior ordering in favor of' if not preserved else 'preserving'
                ord_rel = 'vector-over-scalar'
            else:
                ord_phrase = 'tying'
                ord_rel = 'scalar-vs-vector'
            ch5_parts.append(
                f"Selecting E1 regularization by source-validation QWK instead of macro recall changed $C$ in {int(s3_f['n_folds_c_changed'])}/{n_f_folds} folds for $s_3$ and {int(z3_f['n_folds_c_changed'])}/{n_f_folds} folds for $\\mathbf{{z}}_3$, yielding equal-cohort target Grade~3 recalls of {s3_val:.2f}\\% ($s_3$) vs.\\ {z3_val:.2f}\\% ($\\mathbf{{z}}_3$) and {ord_phrase} the {ord_rel} ordering."
            )
        else:
            f_bits = [f"{r['representation']} ({int(r['n_folds_c_changed'])}/{n_f_folds} C changes, {r['qwk_sel_recall_g3_pct']:.2f}\\%)" for _, r in f_sum.iterrows()]
            ch5_parts.append('Selecting E1 regularization by source-validation QWK yielded ' + ', '.join(f_bits) + '.')
    if 'G' in results:
        g_fold, g_sum = results['G'][0], results['G'][1]
        v3_05_sub = g_sum[(g_sum['variant'] == 'V3') & (g_sum['rule'] == 'cibd_0.5pp_budget')]
        v3_10_sub = g_sum[(g_sum['variant'] == 'V3') & (g_sum['rule'] == 'cibd_1.0pp_budget')]
        if not v3_05_sub.empty:
            r05 = v3_05_sub.iloc[0]
            sub_g05 = g_fold[(g_fold['variant'] == 'V3') & (g_fold['rule'] == 'cibd_0.5pp_budget')]
            n_g_folds = len(sub_g05) if not sub_g05.empty else 6
            n_joint = int(sub_g05['meets_target_joint_hd'].sum()) if not sub_g05.empty else int(r05['meets_target_joint_hd'])
            mean_g3_05 = float(r05['delta_g3_recall_pp'])
            mean_w1_05 = float(r05['delta_within1_pp'])
            ch5_parts.append(
                f"Tightening the source-validation CIBD Within-1 loss budget from 1.0~pp (primary) to 0.5~pp (post hoc) changed $V_3$ selections in {int(r05['params_changed_vs_1pp'])}/{n_g_folds} folds and yielded a post hoc equal-cohort target Grade~3 recall change of {mean_g3_05:+.2f}~pp alongside a Within-1 change of {mean_w1_05:+.2f}~pp ({n_joint}/{n_g_folds} folds meeting both prespecified target $H_D$ margins)."
            )
            if not v3_10_sub.empty:
                r10 = v3_10_sub.iloc[0]
                ch5_parts.append(
                    f"Under the 1.0~pp primary rule reproduced from cached grids, $V_3$ equal-cohort target changes were {float(r10['delta_g3_recall_pp']):+.2f}~pp Grade~3 recall and {float(r10['delta_within1_pp']):+.2f}~pp Within-1 accuracy."
                )
    if not ch5_parts:
        ch5_parts.append('No supplementary analyses (A--G) were executed in this invocation.')
    lines.extend(ch5_parts)
    lines.extend([
        '```',
        '',
        '### Chương 6 (Ảnh hưởng tới RQ2 và RQ3)',
        '```latex',
    ])
    ch6_parts: List[str] = []
    rq2_supported_under_1pp: Optional[bool] = None
    if 'G' in results:
        g_fold, g_sum = results['G'][0], results['G'][1]
        v3_10_sub = g_sum[(g_sum['variant'] == 'V3') & (g_sum['rule'] == 'cibd_1.0pp_budget')]
        v3_05_sub = g_sum[(g_sum['variant'] == 'V3') & (g_sum['rule'] == 'cibd_0.5pp_budget')]
        if not v3_10_sub.empty:
            r10 = v3_10_sub.iloc[0]
            m10_g3 = float(r10['delta_g3_recall_pp'])
            m10_w1 = float(r10['delta_within1_pp'])
            rq2_supported_under_1pp = bool(m10_g3 >= 3.0 and m10_w1 >= -1.0)
            ci_clause = ''
            if 'E' in results:
                e_df = results['E'][0]
                e_w1 = e_df[(e_df['contrast'] == 'cibd_v3_minus_native_v3') & (e_df['scope'] == 'equal_cohort_mean_6') & (e_df['metric'] == 'within1_pp')]
                if not e_w1.empty:
                    ew = e_w1.iloc[0]
                    ci_clause = f"; 95\\% recorded-grouping-key bootstrap CI $[{ew['ci_low_95']:+.2f}, {ew['ci_high_95']:+.2f}]$~pp"
            hd_status_txt = 'satisfies $H_D$' if rq2_supported_under_1pp else 'leaves $H_D$ unsupported'
            ch6_parts.append(
                f"For RQ2, the prespecified 1.0~pp primary source budget {hd_status_txt} ({m10_g3:+.2f}~pp equal-cohort Grade~3 recall change, {m10_w1:+.2f}~pp equal-cohort Within-1 change{ci_clause})."
            )
        elif 'E' in results:
            e_df = results['E'][0]
            e_w1 = e_df[(e_df['contrast'] == 'cibd_v3_minus_native_v3') & (e_df['scope'] == 'equal_cohort_mean_6') & (e_df['metric'] == 'within1_pp')]
            if not e_w1.empty:
                ew = e_w1.iloc[0]
                rq2_supported_under_1pp = bool(float(ew['point_estimate']) >= -1.0)
                hd_status_txt = 'meets the Within-1 margin' if rq2_supported_under_1pp else 'leaves $H_D$ unsupported'
                ch6_parts.append(
                    f"For RQ2, the prespecified 1.0~pp primary source budget {hd_status_txt} ({ew['point_estimate']:+.2f}~pp equal-cohort Within-1 change; 95\\% recorded-grouping-key bootstrap CI $[{ew['ci_low_95']:+.2f}, {ew['ci_high_95']:+.2f}]$~pp)."
                )
        if not v3_05_sub.empty:
            r05 = v3_05_sub.iloc[0]
            sub_g05 = g_fold[(g_fold['variant'] == 'V3') & (g_fold['rule'] == 'cibd_0.5pp_budget')]
            n_g_folds = len(sub_g05) if not sub_g05.empty else 6
            n_joint = int(sub_g05['meets_target_joint_hd'].sum()) if not sub_g05.empty else int(r05['meets_target_joint_hd'])
            missed = [str(x) for x in sub_g05.loc[~sub_g05['meets_target_joint_hd'].astype(bool), 'cohort_label'].tolist()] if not sub_g05.empty else []
            m05_g3 = float(r05['delta_g3_recall_pp'])
            m05_w1 = float(r05['delta_within1_pp'])
            mean_both = bool(m05_g3 >= 3.0 and m05_w1 >= -1.0)
            if mean_both:
                mean_phrase = 'the equal-cohort mean target trade-off met both target margins'
            else:
                mean_phrase = 'the equal-cohort mean target trade-off did not meet both target margins'
            if missed:
                cohort_phrase = f"with {n_joint}/{n_g_folds} cohorts satisfying both criteria while {_format_name_list(missed)} missed a prespecified target margin"
            else:
                cohort_phrase = f"with all {n_g_folds}/{n_g_folds} cohorts satisfying both criteria"
            ch6_parts.append(
                f"Under the post hoc 0.5~pp source budget, {mean_phrase} ({m05_g3:+.2f}~pp Grade~3 recall, {m05_w1:+.2f}~pp Within-1 accuracy), {cohort_phrase}; this informs future source-selection safety margins without altering the prespecified primary 1.0~pp hypothesis decision."
            )
    elif 'E' in results:
        e_df = results['E'][0]
        e_w1 = e_df[(e_df['contrast'] == 'cibd_v3_minus_native_v3') & (e_df['scope'] == 'equal_cohort_mean_6') & (e_df['metric'] == 'within1_pp')]
        if not e_w1.empty:
            ew = e_w1.iloc[0]
            ch6_parts.append(
                f"For RQ2 uncertainty sensitivity under the primary 1.0~pp budget, the equal-cohort Within-1 change was {ew['point_estimate']:+.2f}~pp (95\\% recorded-grouping-key bootstrap CI $[{ew['ci_low_95']:+.2f}, {ew['ci_high_95']:+.2f}]$~pp)."
            )

    if 'D' in results:
        ds = results['D'][1].iloc[0]
        n_d_folds = int(ds['n_folds'])
        n_d_up = int(ds['n_folds_upward_shift'])
        ch6_parts.append(
            f"Latent-grid inspection across {n_d_folds} folds confirmed that bounded peak Grade~3 probability (max $p_3$ {ds['max_p3_grid_min']:.3f}--{ds['max_p3_grid_max']:.3f}) and upward boundary displacement occurred in {_fold_count_phrase(n_d_up, n_d_folds)}."
        )

    if 'F' in results:
        f_sum = results['F'][2]
        rep_set_f = set(f_sum['representation'])
        if {'s_v3', 'z_v3'} <= rep_set_f:
            s3_f = f_sum[f_sum['representation'] == 's_v3'].iloc[0]
            z3_f = f_sum[f_sum['representation'] == 'z_v3'].iloc[0]
            s3_val = float(s3_f['qwk_sel_recall_g3_pct'])
            z3_val = float(z3_f['qwk_sel_recall_g3_pct'])
            s3_mac = float(s3_f['macro_recall_g3_pct'])
            z3_mac = float(z3_f['macro_recall_g3_pct'])
            preserved = bool((s3_mac > z3_mac) == (s3_val > z3_val))
            if s3_val > z3_val:
                rq3_txt = (
                    f"For RQ3, selecting probe regularization by source-validation QWK yielded higher equal-cohort target Grade~3 recall for $s_3$ ({s3_val:.2f}\\%) than for $\\mathbf{{z}}_3$ ({z3_val:.2f}\\%), "
                    f"{'preserving the scalar-over-vector ordering and leaving $H_R$ unsupported under both selection criteria' if preserved else 'showing scalar superiority under QWK selection'}."
                )
            elif z3_val > s3_val:
                rq3_txt = (
                    f"For RQ3, selecting probe regularization by source-validation QWK reversed the primary ordering, yielding higher equal-cohort target Grade~3 recall for $\\mathbf{{z}}_3$ ({z3_val:.2f}\\%) than for $s_3$ ({s3_val:.2f}\\%), which warrants re-examining $H_R$ under QWK selection."
                )
            else:
                rq3_txt = (
                    f"For RQ3, selecting probe regularization by source-validation QWK produced equal target Grade~3 recall for $s_3$ and $\\mathbf{{z}}_3$ ({s3_val:.2f}\\%)."
                )
            ch6_parts.append(rq3_txt)
        else:
            ch6_parts.append('For RQ3, only a partial subset of E1 representations was executed in this run; full $s_3$ vs.\\ $\\mathbf{z}_3$ comparison requires both representations.')

    if not ch6_parts:
        ch6_parts.append('No RQ2 or RQ3 sensitivity checks (D, E, F, or G) were executed in this invocation.')
    lines.extend(ch6_parts)
    lines.append('```')

    report_path = ctx.output_dir / 'SUPPLEMENTARY_REPORT.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def run_selected_analyses(ctx: AnalysisContext, analyses: Sequence[str] = ALL_ANALYSES) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute requested analyses in deterministic order and write report + status table."""
    selected = tuple(a.upper() for a in analyses)
    invalid = [a for a in selected if a not in ALL_ANALYSES]
    if invalid:
        raise ValueError(f'Unknown analysis IDs: {invalid}; expected subset of {ALL_ANALYSES}')

    results: Dict[str, Any] = {'baseline': verify_baselines(ctx)}
    statuses: List[Dict[str, Any]] = []
    runners = {
        'A': run_analysis_a,
        'B': run_analysis_b,
        'C': run_analysis_c,
        'D': run_analysis_d,
        'E': run_analysis_e,
        'F': run_analysis_f,
        'G': run_analysis_g,
    }
    for aid in selected:
        out = runners[aid](ctx)
        results[aid] = out[:-1]
        statuses.append(out[-1])

    write_supplementary_report(ctx, statuses, results)
    return pd.DataFrame(statuses), results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run CPU-only thesis supplementary analyses (A-G).')
    parser.add_argument('--p0-dir', type=Path, default=None, help='Completed P0 run directory (default: result/p0_fp32_main_lodo_v1 or external fallback)')
    parser.add_argument('--e0-dir', type=Path, default=None, help='Completed E0 run directory')
    parser.add_argument('--e1-dir', type=Path, default=None, help='Completed E1 run directory')
    parser.add_argument('--reference-json', type=Path, default=None, help='Archived lodo_summary_with_calibrated_screening.json')
    parser.add_argument('--archived-preds-dir', type=Path, default=None, help='Archived raw_preds NPZ directory')
    parser.add_argument('--folds-dir', type=Path, default=None, help='Archived lodo_folds CSV directory')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory (default: result/thesis_supplementary_v1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrap sensitivity analysis (default: 42)')
    parser.add_argument('--bootstrap-reps', type=int, default=2000, help='Bootstrap replicates for Analysis E (default: 2000)')
    parser.add_argument('--analyses', nargs='+', default=list(ALL_ANALYSES), choices=list(ALL_ANALYSES), help='Subset of analyses to run (default: A B C D E F G)')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = build_context(
        p0_dir=args.p0_dir,
        e0_dir=args.e0_dir,
        e1_dir=args.e1_dir,
        output_dir=args.output_dir,
        reference_json=args.reference_json,
        archived_preds_dir=args.archived_preds_dir,
        folds_dir=args.folds_dir,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        execution_mode='cli_script',
    )
    status_df, results = run_selected_analyses(ctx, args.analyses)
    print(f'Output directory: {ctx.output_dir}')
    print(results['baseline'].to_string(index=False))
    print(status_df[['analysis_id', 'status', 'output_files']].to_string(index=False))
    return 0


# %% [markdown]
# ## Interactive Notebook Execution (Questions A–G)
#
# Resolves paths from the configuration cell above (`DR_*` environment variables, Google Drive, or local repository `result/`),
# validates baselines, and executes each question in its own cell.

# %%
if __name__ == '__main__' and _running_in_notebook():
    from IPython.display import display
    REPO_ROOT = find_repo_root()
    if REPO_ROOT is not None:
        os.chdir(REPO_ROOT)
    CTX = build_context(
        repo_root=REPO_ROOT,
        p0_dir=P0_RUN,
        e0_dir=E0_RUN,
        e1_dir=E1_RUN,
        output_dir=OUTPUT_DIR,
        reference_json=REFERENCE_PATH,
        archived_preds_dir=RAW_PREDS,
        folds_dir=FOLD_SOURCE,
        seed=SEED,
        bootstrap_reps=BOOTSTRAP_REPS,
        execution_mode='notebook',
    )
    RESULTS = {'baseline': verify_baselines(CTX)}
    STATUSES = []
    print('In Colab:', IN_COLAB, '| Repo root:', REPO_ROOT or 'standalone')
    print('P0_RUN:', CTX.p0_dir)
    print('E0_RUN:', CTX.e0_dir)
    print('E1_RUN:', CTX.e1_dir)
    print('RAW_PREDS:', CTX.archived_preds_dir)
    print('REFERENCE:', CTX.reference_json)
    print('Output dir:', CTX.output_dir)
    display(RESULTS['baseline'])

# %% [markdown]
# ### Question A: Compare $V_2$ and $V_3$ under identical decoders

# %%
if __name__ == '__main__' and _running_in_notebook() and 'A' in ANALYSES_TO_RUN:
    df_a_fold, df_a_sum, st_a = run_analysis_a(CTX)
    RESULTS['A'] = (df_a_fold, df_a_sum)
    STATUSES.append(st_a)
    display(df_a_sum)

# %% [markdown]
# ### Question B: Five-grade recall, Grade 3 prediction ratio, and paired transitions

# %%
if __name__ == '__main__' and _running_in_notebook() and 'B' in ANALYSES_TO_RUN:
    df_b_fold, df_b_sum, df_b_trans, df_b_trans_sum, st_b = run_analysis_b(CTX)
    RESULTS['B'] = (df_b_fold, df_b_sum, df_b_trans, df_b_trans_sum)
    STATUSES.append(st_b)
    display(df_b_sum)
    display(df_b_trans_sum)

# %% [markdown]
# ### Question C: P0 image counts and ID-matched prediction agreement audit

# %%
if __name__ == '__main__' and _running_in_notebook() and 'C' in ANALYSES_TO_RUN:
    df_c, df_c_sum, st_c = run_analysis_c(CTX)
    RESULTS['C'] = (df_c, df_c_sum)
    STATUSES.append(st_c)
    if not df_c_sum.empty:
        display(df_c_sum)
    else:
        print('Analysis C status:', st_c['status'], '-', st_c['reason'])

# %% [markdown]
# ### Question D: Six-fold $V_3$ peak $p_3$ and Grade 3 decision-region overlap

# %%
if __name__ == '__main__' and _running_in_notebook() and 'D' in ANALYSES_TO_RUN:
    df_d, df_d_sum, st_d = run_analysis_d(CTX)
    RESULTS['D'] = (df_d, df_d_sum)
    STATUSES.append(st_d)
    display(df_d[['fold_id', 'cohort', 'max_p3_grid', 'max_p3_target_images', 'native_g3_left', 'native_g3_right', 'theta2', 'theta3', 'overlap_ratio_grid', 'upward_shift_both_bounds']])

# %% [markdown]
# ### Question E: Recorded-grouping-key paired bootstrap sensitivity analysis

# %%
if __name__ == '__main__' and _running_in_notebook() and 'E' in ANALYSES_TO_RUN:
    df_e, df_e_diag, st_e = run_analysis_e(CTX)
    RESULTS['E'] = (df_e, df_e_diag)
    STATUSES.append(st_e)
    display(df_e[df_e['scope'] == 'equal_cohort_mean_6'])

# %% [markdown]
# ### Question F: Probe regularization $C$ selected by source-validation QWK

# %%
if __name__ == '__main__' and _running_in_notebook() and 'F' in ANALYSES_TO_RUN:
    df_f_cand, df_f_comp, df_f_sum, st_f = run_analysis_f(CTX)
    RESULTS['F'] = (df_f_cand, df_f_comp, df_f_sum)
    STATUSES.append(st_f)
    display(df_f_sum)

# %% [markdown]
# ### Question G: CIBD with a 0.5 pp source-validation Within-1 loss budget

# %%
if __name__ == '__main__' and _running_in_notebook() and 'G' in ANALYSES_TO_RUN:
    df_g, df_g_sum, st_g = run_analysis_g(CTX)
    RESULTS['G'] = (df_g, df_g_sum)
    STATUSES.append(st_g)
    display(df_g_sum)

# %% [markdown]
# ### Final status table and Vietnamese report

# %%
if __name__ == '__main__' and _running_in_notebook():
    report_path = write_supplementary_report(CTX, STATUSES, RESULTS)
    status_df = pd.read_csv(CTX.output_dir / 'analysis_status.csv')
    display(status_df[['analysis_id', 'status', 'reason']])
    print('Report written to:', report_path)

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
