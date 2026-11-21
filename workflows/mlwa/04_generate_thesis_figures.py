# %% [markdown]
# # Thesis Quantitative Figure Generator (Revised Overleaf Bundle — v2)
#
# Generates publication-grade vector PDFs, 300-DPI PNG previews, exact plotted CSVs,
# `figure_manifest.json`, and `FIGURE_INTEGRATION.md` from archived DMIP references,
# P0 replay, E0/E1 primary caches, and post-hoc supplementary CPU outputs.
#
# Incorporates T1-E and T1-F visual conventions: V2 blue, V3 red, consistent Grade 3 highlight,
# >= 8 pt typography at ~16 cm print width, proper Unicode minus (−) and arrows (→),
# 1-decimal tradeoff labels, explicit zoomed-view tail annotations, and short/full LaTeX captions.

# %%
NOTEBOOK_CODE_SHA256 = '71542bbb2166002d44bb80e3b1b90da5e871c0445cf3183634c27186b2c208af'

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
        _deps = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'matplotlib': 'matplotlib'}
        _missing = [pkg for mod, pkg in _deps.items() if importlib.util.find_spec(mod) is None]
        if _missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *_missing])

    THESIS_ROOT = Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()
    P0_RUN = None
    E0_RUN = None
    E1_RUN = None
    SUPP_RUN = None
    OUTPUT_DIR = None

# %%
import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

COHORTS: Tuple[str, ...] = ('aptos', 'idrid', 'messidor2', 'ddr', 'eyepacs', 'deepdrid')
COHORT_LABELS: Dict[str, str] = {
    'aptos': 'APTOS 2019',
    'idrid': 'IDRiD',
    'messidor2': 'Messidor-2',
    'ddr': 'DDR',
    'eyepacs': 'EyePACS',
    'deepdrid': 'DeepDRiD',
}
COHORT_SHORT: Dict[str, str] = {
    'aptos': 'APTOS',
    'idrid': 'IDRiD',
    'messidor2': 'Messidor-2',
    'ddr': 'DDR',
    'eyepacs': 'EyePACS',
    'deepdrid': 'DeepDRiD',
}
COHORT_MARKERS: Dict[str, str] = {
    'aptos': 'o',
    'idrid': 's',
    'messidor2': '^',
    'ddr': 'D',
    'eyepacs': 'v',
    'deepdrid': 'P',
}
COHORT_COLORS: Dict[str, str] = {
    'aptos': '#1B9E77',
    'idrid': '#D95F02',
    'messidor2': '#7570B3',
    'ddr': '#E7298A',
    'eyepacs': '#66A61E',
    'deepdrid': '#A6761D',
}

# Unified color and marker system per T1-F §3: V2 blue, V3 red, consistent Grade 3 highlight
COLOR_V2 = '#0072B2'        # Blue for nominal MIL+GC (V2)
COLOR_V3 = '#C0392B'        # Red for ordinal CLM (V3)
COLOR_G3 = '#B02A24'        # Consistent Grade 3 highlight crimson
COLOR_W1 = '#1F618D'        # Within-1 blue-slate
COLOR_GRAY = '#566573'
MINUS = '\u2212'
ARROW = '\u2192'

GRADE_COLORS: Dict[int, str] = {
    0: '#5D6D7E',
    1: '#2E86C1',
    2: '#D68910',
    3: COLOR_G3,
    4: '#6C3483',
}
GRADE_STYLES: Dict[int, str] = {0: '--', 1: '-.', 2: ':', 3: '-', 4: (0, (3, 1, 1, 1))}

# Typography sized for ~16 cm (6.3 in) thesis print width so minimum rendered text >= 8 pt
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.0,
    'axes.titlesize': 10.0,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9.2,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.2,
    'axes.unicode_minus': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


def _fmt_signed(val: float, decimals: int = 1) -> str:
    """Format signed float with proper Unicode minus (−) and explicit + for positive values."""
    s = f'{val:+.{decimals}f}'
    if s.startswith('-'):
        return MINUS + s[1:]
    return s


def _fmt_num(val: float, decimals: int = 2) -> str:
    s = f'{val:.{decimals}f}'
    return s.replace('-', MINUS)


def find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    cands = []
    if start is not None:
        cands.append(Path(start).resolve())
    cands.append(Path.cwd().resolve())
    if '__file__' in globals():
        cands.append(Path(__file__).resolve().parent)
    for base in cands:
        for cur in (base, *base.parents):
            if (cur / 'pyproject.toml').is_file() and (cur / 'workflows' / 'mlwa').is_dir():
                return cur
    return None


def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def _first_existing(cands: Sequence[Optional[Path]]) -> Optional[Path]:
    for c in cands:
        if c is not None and Path(c).expanduser().exists():
            return Path(c).expanduser().resolve()
    return None


@dataclass
class FigureContext:
    repo_root: Optional[Path]
    p0_dir: Path
    e0_dir: Path
    e1_dir: Path
    supp_dir: Path
    output_dir: Path
    localization_asset: Optional[Path]
    execution_mode: str = 'module_api'


def build_context(
    repo_root: Optional[Path] = None,
    p0_dir: Optional[Path] = None,
    e0_dir: Optional[Path] = None,
    e1_dir: Optional[Path] = None,
    supp_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    execution_mode: str = 'module_api',
) -> FigureContext:
    """Resolve read-only experiment/supplementary inputs and isolated v2 figure bundle."""
    root = find_repo_root(repo_root)
    ext_mlwa = Path(r'D:\Github\deep-learning-for-computer-vision\notebooks\mlwa-journals\outputs')

    res_p0 = _first_existing([
        p0_dir, _env_path('DR_P0_RUN'),
        (root / 'result' / 'p0_fp32_main_lodo_v1') if root else None,
        ext_mlwa / 'p0_fp32_main_lodo_v1',
    ])
    res_e0 = _first_existing([
        e0_dir, _env_path('DR_E0_RUN'),
        (root / 'result' / 'p0_fp32_main_lodo_v1_e0_resume_v2') if root else None,
        ext_mlwa / 'p0_fp32_main_lodo_v1_e0_resume_v2',
    ])
    res_e1 = _first_existing([
        e1_dir, _env_path('DR_E1_RUN'),
        (root / 'result' / 'p0_fp32_main_lodo_v1_e1_resume_v2') if root else None,
        ext_mlwa / 'p0_fp32_main_lodo_v1_e1_resume_v2',
    ])
    res_supp = _first_existing([
        supp_dir, _env_path('DR_SUPP_RUN'),
        (root / 'result' / 'thesis_supplementary_v1') if root else None,
    ])
    for name, p in [('P0', res_p0), ('E0', res_e0), ('E1', res_e1), ('SUPP', res_supp)]:
        if p is None:
            raise FileNotFoundError(f'Unable to resolve required read-only {name} directory.')

    loc_asset = _first_existing([
        (root / 'result' / 'figure_8_xai_qualitative_grid.pdf') if root else None,
        Path(r'D:\Github\DR-evaluation-study2026\documents\figure_8_xai_qualitative_grid.pdf'),
    ])

    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
    elif _env_path('DR_FIG_OUTPUT') is not None:
        out = _env_path('DR_FIG_OUTPUT').expanduser().resolve()  # type: ignore[union-attr]
    elif root is not None:
        out = (root / 'result' / 'thesis_figures_v2').resolve()
    else:
        out = (Path.cwd() / 'thesis_figures_v2').resolve()

    (out / 'figures' / 'chapter05').mkdir(parents=True, exist_ok=True)
    (out / 'figures' / 'appendix').mkdir(parents=True, exist_ok=True)
    return FigureContext(
        repo_root=root,
        p0_dir=res_p0,
        e0_dir=res_e0,
        e1_dir=res_e1,
        supp_dir=res_supp,
        output_dir=out,
        localization_asset=loc_asset,
        execution_mode=execution_mode,
    )


def _save_chart(fig: plt.Figure, out_dir: Path, rel_stem: str, df_data: pd.DataFrame) -> Dict[str, Any]:
    """Save vector PDF, 300-DPI PNG preview, and exact plotted CSV rows."""
    pdf_path = out_dir / f'{rel_stem}.pdf'
    png_path = out_dir / f'{rel_stem}.png'
    csv_path = out_dir / f'{rel_stem}_data.csv'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    df_data.to_csv(csv_path, index=False)
    return {
        'pdf': str(pdf_path.relative_to(out_dir)).replace('\\', '/'),
        'png': str(png_path.relative_to(out_dir)).replace('\\', '/'),
        'csv': str(csv_path.relative_to(out_dir)).replace('\\', '/'),
        'pdf_sha256': hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        'n_rows': int(len(df_data)),
    }


def _clm_probs(s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    u = np.clip(theta[None, :] - s[:, None], -20.0, 20.0)
    surv = np.exp(-np.exp(u))
    probs = np.column_stack([-np.expm1(-np.exp(u[:, 0])), surv[:, :-1] - surv[:, 1:], surv[:, -1]])
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


# %% [markdown]
# ## Chart 1: Archived Architectural Trade-off (`figures/chapter05/tradeoff.pdf`)

# %%
def build_chart_1_tradeoff(ctx: FigureContext) -> Dict[str, Any]:
    g0 = pd.read_csv(ctx.p0_dir / 'g0_comparison.csv')
    sub = g0[(g0['split'] == 'target') & (g0['metric'].isin(['recall_g3', 'within1']))]
    piv = sub.pivot(index=['fold_id', 'cohort', 'metric'], columns='variant', values='reference').reset_index()
    piv['delta_pp'] = (piv['V3'] - piv['V2']) * 100.0
    piv['cohort_label'] = piv['cohort'].map(COHORT_LABELS)
    piv['split'] = 'target'
    piv['source_kind'] = 'archived_reference'
    piv['units'] = 'percentage_points'
    piv['aggregation'] = 'per_cohort'
    piv['model_contrast'] = 'V3_minus_V2'

    mean_rows = []
    for metric in ['recall_g3', 'within1']:
        m_sub = piv[piv['metric'] == metric]
        mean_rows.append({
            'fold_id': -1,
            'cohort': 'equal_cohort_mean',
            'cohort_label': 'Equal-cohort mean (1/6)',
            'metric': metric,
            'V2': float(m_sub['V2'].mean()),
            'V3': float(m_sub['V3'].mean()),
            'delta_pp': float(m_sub['delta_pp'].mean()),
            'split': 'target',
            'source_kind': 'archived_reference',
            'units': 'percentage_points',
            'aggregation': 'equal_cohort_mean_6',
            'model_contrast': 'V3_minus_V2',
        })
    df_out = pd.concat([piv.sort_values(['fold_id', 'metric']), pd.DataFrame(mean_rows)], ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    ax.axvline(0.0, color='#333333', linestyle='--', linewidth=1.0, zorder=1)

    y_positions = [7, 6, 5, 4, 3, 2, 0.5]
    order_cohorts = list(COHORTS) + ['equal_cohort_mean']
    y_map = dict(zip(order_cohorts, y_positions))

    for idx_c, c in enumerate(order_cohorts):
        y = y_map[c]
        r_g3 = float(df_out[(df_out['cohort'] == c) & (df_out['metric'] == 'recall_g3')]['delta_pp'].iloc[0])
        r_w1 = float(df_out[(df_out['cohort'] == c) & (df_out['metric'] == 'within1')]['delta_pp'].iloc[0])
        is_mean = (c == 'equal_cohort_mean')
        lw = 2.0 if is_mean else 1.2
        ms = 8.6 if is_mean else 6.8
        ax.plot([r_g3, r_w1], [y, y], color='#BDC3C7' if not is_mean else '#566573', linewidth=lw, zorder=2)

        ax.scatter([r_g3], [y], color=COLOR_G3, marker='D' if is_mean else 'o', s=ms**2,
                   edgecolor='white', linewidth=0.7, zorder=4,
                   label='Grade 3 recall ($V_3 - V_2$)' if idx_c == 0 else None)
        ax.scatter([r_w1], [y], color=COLOR_W1, marker='P' if is_mean else 's', s=ms**2,
                   edgecolor='white', linewidth=0.7, zorder=4,
                   label='Within-1 accuracy ($V_3 - V_2$)' if idx_c == 0 else None)

        # 1 decimal place + proper Unicode minus (−) per T1-E §5
        left_val, right_val = (r_g3, r_w1) if r_g3 <= r_w1 else (r_w1, r_g3)
        ax.text(left_val - 0.70, y, _fmt_signed(left_val, 1), va='center', ha='right', fontsize=8.3,
                fontweight='bold' if is_mean else 'normal', color=COLOR_G3 if left_val == r_g3 else COLOR_W1)
        right_bbox = dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85) if (-3.5 < right_val < 0.0) else None
        ax.text(right_val + 0.70, y, _fmt_signed(right_val, 1), va='center', ha='left', fontsize=8.3,
                fontweight='bold' if is_mean else 'normal', color=COLOR_W1 if right_val == r_w1 else COLOR_G3,
                bbox=right_bbox, zorder=6)

    ax.axhline(1.25, color='#AAB7B8', linestyle=':', linewidth=1.0)
    ax.axhspan(-0.15, 1.25, color='#F4F6F7', zorder=0)
    ax.set_yticks(y_positions)
    ylabels = [COHORT_LABELS[c] for c in COHORTS] + ['Equal-cohort mean (1/6)']
    ax.set_yticklabels(ylabels)
    ax.get_yticklabels()[-1].set_fontweight('bold')

    ax.set_xlim(-32.0, 8.0)
    ax.set_ylim(-0.15, 7.65)
    ax.set_xlabel(r'Archived target change, $100 \times (V_3 - V_2)$ (percentage points)')
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    ax.legend(loc='lower center', bbox_to_anchor=(0.48, 1.02), ncol=2, frameon=True,
              facecolor='white', edgecolor='#D5D8DC', framealpha=0.95, fontsize=8.2)
    info = _save_chart(fig, ctx.output_dir, 'figures/chapter05/tradeoff', df_out)
    info['means_pp'] = {r['metric']: r['delta_pp'] for r in mean_rows}
    return info


# %% [markdown]
# ## Chart 2: IDRiD $V_3$ Latent Severity & Decision Regions + 6-Fold Appendix E Companion

# %%
def build_chart_2_latent_regions(ctx: FigureContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fold_dir = ctx.e0_dir / 'lodo' / 'units' / 'V3_fold_1_idrid' / 'fold_report'
    grid_df = pd.read_csv(fold_dir / 'decision_grid_float64_diagnostic.csv')
    iv_df = pd.read_csv(fold_dir / 'decision_intervals.csv')
    p0_df = pd.read_csv(ctx.p0_dir / 'lodo' / 'V3_fold_1_idrid' / 'target' / 'predictions.csv')
    d_folds = pd.read_csv(ctx.supp_dir / 'analysis_d_v3_latent_regions_per_fold.csv')
    idrid_d = d_folds[d_folds['cohort'] == 'idrid'].iloc[0]

    theta = np.array([float(p0_df[f'theta{k}'].iloc[0]) for k in range(4)], dtype=np.float64)
    if np.max(np.abs(theta - np.array([idrid_d[f'theta{k}'] for k in range(4)]))) > 1e-6:
        raise ValueError('Theta mismatch between IDRiD P0 predictions and Analysis D summary.')

    s_grid = grid_df['s'].to_numpy(dtype=np.float64)
    probs_grid = _clm_probs(s_grid, theta)
    idx_peak = int(np.argmax(probs_grid[:, 3]))
    peak_s_grid = float(s_grid[idx_peak])
    peak_p3_grid = float(probs_grid[idx_peak, 3])
    obs_max_p3 = float(p0_df['p3'].max())

    nat_row = iv_df[(iv_df['rule'] == 'native_replay') & (iv_df['grade'] == 3)].iloc[0]
    cut_row = iv_df[(iv_df['rule'] == 'cutpoint_latent') & (iv_df['grade'] == 3)].iloc[0]
    cibd_row = iv_df[(iv_df['rule'] == 'cibd_selected') & (iv_df['grade'] == 3)].iloc[0]

    ov_left = max(float(nat_row['left']), float(cut_row['left']))
    ov_right = min(float(nat_row['right']), float(cut_row['right']))
    ov_len = max(0.0, ov_right - ov_left)
    cut_len = float(cut_row['right']) - float(cut_row['left'])
    ov_frac_pct = 100.0 * ov_len / cut_len

    xlim = (-3.3, 4.35)
    # Explicitly count & document any target points outside the zoomed view
    tail_hi = p0_df[p0_df['s'] > xlim[1]]
    n_tail_hi = int(len(tail_hi))
    max_s_obs = float(p0_df['s'].max())
    tail_by_g = {int(g): int((tail_hi['y_true'] == g).sum()) for g in range(5) if (tail_hi['y_true'] == g).any()}

    df_scores = p0_df[['image_key', 'y_true', 's', 'p3', 'pred_native']].copy()
    df_scores['cohort'] = 'idrid'
    df_scores['fold_id'] = 1
    df_scores['variant'] = 'V3'
    df_scores['split'] = 'target'
    df_scores['peak_p3_grid'] = peak_p3_grid
    df_scores['peak_s_grid'] = peak_s_grid
    df_scores['obs_max_p3'] = obs_max_p3
    df_scores['overlap_pct_of_cutpoint'] = ov_frac_pct
    df_scores['outside_zoom_gt_4_35'] = df_scores['s'] > xlim[1]

    s_dense = np.linspace(xlim[0], xlim[1], 600)
    fig, (ax_dist, ax_prob, ax_band) = plt.subplots(
        3, 1, sharex=True, figsize=(6.35, 6.3),
        gridspec_kw={'height_ratios': [1.22, 1.35, 0.90], 'hspace': 0.18},
    )

    # Panel A: Within-grade normalized empirical score densities
    for g in range(5):
        sg = p0_df.loc[p0_df['y_true'] == g, 's'].to_numpy(dtype=np.float64)
        kde = gaussian_kde(sg, bw_method=0.35)
        dens = kde(s_dense)
        is_g3 = (g == 3)
        ax_dist.plot(
            s_dense, dens, color=GRADE_COLORS[g], linestyle=GRADE_STYLES[g],
            linewidth=2.3 if is_g3 else 1.4, zorder=4 if is_g3 else 2,
            label=f'Grade {g} (n={len(sg)})',
        )
        if is_g3:
            ax_dist.fill_between(s_dense, 0, dens, color=COLOR_G3, alpha=0.14, zorder=3)
    for t in theta:
        ax_dist.axvline(t, color='#AEB6BF', linestyle=':', linewidth=0.9, zorder=1)
    ax_dist.set_ylabel('Within-grade\ndensity')
    ax_dist.set_ylim(0, 1.04)
    ax_dist.legend(loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=3, frameon=True,
                   facecolor='white', edgecolor='#D5D8DC', fontsize=8.0)
    ax_dist.text(
        0.985, 0.93,
        f'IDRiD target (n = {len(p0_df)}, per-grade density)\n'
        f'{n_tail_hi} points at $s > 4.35$ (max $s = {max_s_obs:.2f}$)',
        transform=ax_dist.transAxes, ha='right', va='top', fontsize=8.0,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#D5D8DC', alpha=0.92),
    )

    # Panel B: CLM class probabilities and learned cutpoints
    mask_x = (s_grid >= xlim[0]) & (s_grid <= xlim[1])
    sg_clip = s_grid[mask_x]
    pg_clip = probs_grid[mask_x]
    for g in range(5):
        is_g3 = (g == 3)
        ax_prob.plot(
            sg_clip, pg_clip[:, g], color=GRADE_COLORS[g], linestyle=GRADE_STYLES[g],
            linewidth=2.5 if is_g3 else 1.3, zorder=4 if is_g3 else 2,
            label=f'$p_{g}(s)$',
        )
    for idx_t, t in enumerate(theta):
        ax_prob.axvline(t, color='#7F8C8D', linestyle='--', linewidth=0.95, zorder=1)
        ax_prob.text(t, 1.02, rf'$\theta_{idx_t}$', ha='center', va='bottom', fontsize=8.5, color='#2C3E50',
                     transform=ax_prob.get_xaxis_transform())
    ax_prob.scatter([peak_s_grid], [peak_p3_grid], color=COLOR_G3, s=44, zorder=5)
    ax_prob.annotate(
        f'Grid $\\max p_3 = {peak_p3_grid:.3f}$\nat $s = {peak_s_grid:.2f}$',
        xy=(peak_s_grid, peak_p3_grid), xytext=(peak_s_grid + 0.62, 0.50),
        arrowprops=dict(arrowstyle='->', color=COLOR_G3, lw=1.0),
        fontsize=8.2, color=COLOR_G3, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.86),
        zorder=6,
    )
    ax_prob.set_ylabel('Class\nprobability')
    ax_prob.set_ylim(-0.02, 1.03)
    ax_prob.grid(axis='y', linestyle=':', alpha=0.35)
    ax_prob.legend(loc='center left', bbox_to_anchor=(0.01, 0.44), frameon=True,
                   facecolor='white', edgecolor='#D5D8DC', fontsize=8.0)

    # Panel C: Horizontal Grade 3 decision bands
    bands = [
        ('CIBD (source-locked)', float(cibd_row['left']), float(cibd_row['right']), 0.6, COLOR_V2),
        ('Raw latent cutpoints', float(cut_row['left']), float(cut_row['right']), 1.6, '#27AE60'),
        ('Native (rounded exp.)', float(nat_row['left']), float(nat_row['right']), 2.6, COLOR_V3),
    ]
    for label_b, lo, hi, y0, col in bands:
        ax_band.barh(y0, hi - lo, left=lo, height=0.52, color=col, alpha=0.82, edgecolor='#2C3E50', linewidth=0.6)
        ax_band.text(lo - 0.08, y0, f'{label_b}\n[{lo:.2f}, {hi:.2f}]', ha='right', va='center', fontsize=8.0, color='#1C2833')

    ax_band.axvspan(ov_left, ov_right, ymin=0.45, ymax=0.98, color='#F1C40F', alpha=0.22, zorder=1)
    ax_band.axvline(ov_left, ymin=0.45, ymax=0.98, color='#7D6608', linestyle=':', linewidth=1.0)
    ax_band.axvline(ov_right, ymin=0.45, ymax=0.98, color='#7D6608', linestyle=':', linewidth=1.0)
    ax_band.annotate(
        f'Native\u2013cutpoint overlap:\n{ov_frac_pct:.1f}% of raw width',
        xy=(0.5 * (ov_left + ov_right), 2.1), xytext=(2.62, 1.48),
        arrowprops=dict(arrowstyle='->', color='#7D6608', lw=1.0),
        fontsize=8.0, color='#5D4037', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.22', fc='#FEF9E7', ec='#D4AC0D'),
    )
    ax_band.set_yticks([])
    ax_band.set_ylim(0.05, 3.28)
    ax_band.set_xlim(*xlim)
    ax_band.set_xlabel('Latent severity score $s$ (IDRiD-held-out $V_3$ checkpoint)')
    fig.align_ylabels([ax_dist, ax_prob, ax_band])
    info_main = _save_chart(fig, ctx.output_dir, 'figures/chapter05/latent-regions', df_scores)
    info_main.update({
        'peak_p3_grid': peak_p3_grid,
        'obs_max_p3': obs_max_p3,
        'overlap_pct_of_cutpoint': ov_frac_pct,
        'n_images': int(len(p0_df)),
        'n_outside_zoom_gt_4_35': n_tail_hi,
        'outside_zoom_by_grade': tail_by_g,
        'max_s_obs': max_s_obs,
    })

    # Appendix E Companion: 6-fold V3 latent region representativeness (post hoc)
    df_d_out = d_folds[['fold_id', 'cohort', 'cohort_label', 'overlap_ratio_grid', 'max_p3_grid', 'max_p3_target_images', 'upward_shift_both_bounds']].copy()
    df_d_out['overlap_pct'] = df_d_out['overlap_ratio_grid'] * 100.0
    df_d_out['variant'] = 'V3'
    df_d_out['split'] = 'target_plus_grid'

    fig_app, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.35, 3.25), gridspec_kw={'wspace': 0.14})
    y_idx = np.arange(len(COHORTS))[::-1]
    for pos, (_, row) in zip(y_idx, df_d_out.sort_values('fold_id').iterrows()):
        is_idrid = (row['cohort'] == 'idrid')
        col = '#7B1E1A' if is_idrid else COLOR_V3
        ms = 8.2 if is_idrid else 6.4
        ax1.scatter([row['overlap_pct']], [pos], color=col, s=ms**2, marker='D' if is_idrid else 'o',
                    edgecolor='white', linewidth=0.7, zorder=4)
        ax1.text(row['overlap_pct'] + 1.5, pos, f"{row['overlap_pct']:.1f}%", va='center', ha='left',
                 fontsize=8.2, fontweight='bold' if is_idrid else 'normal', color=col)

        ax2.scatter([row['max_p3_grid']], [pos], color=col, s=ms**2, marker='D' if is_idrid else 'o',
                    edgecolor='white', linewidth=0.7, zorder=4)
        ax2.scatter([row['max_p3_target_images']], [pos], color='#2C3E50', s=30, marker='x', zorder=5)
        ax2.text(row['max_p3_grid'] + 0.009, pos, f"{row['max_p3_grid']:.3f}", va='center', ha='left',
                 fontsize=8.2, fontweight='bold' if is_idrid else 'normal', color=col)

    ax1.set_yticks(y_idx)
    ax1.set_yticklabels([COHORT_LABELS[c] for c in COHORTS])
    ax1.get_yticklabels()[1].set_fontweight('bold')
    ax1.set_xlim(15, 78)
    ax1.set_xlabel('Native\u2013cutpoint overlap (% raw width)')
    ax1.set_title('Decision-interval overlap', fontsize=9.2, pad=20)
    ax1.grid(axis='x', linestyle=':', alpha=0.35)

    ax2.set_xlim(0.20, 0.57)
    ax2.set_xlabel('Peak Grade 3 probability ($\\max p_3$)')
    ax2.set_title('Peak $p_3$ (grid vs. observed)', fontsize=9.2, pad=20)
    ax2.grid(axis='x', linestyle=':', alpha=0.35)
    app_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V3, markersize=6.8, label='Diagnostic grid $\\max p_3$'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#7B1E1A', markersize=6.8, label='IDRiD worked example'),
        Line2D([0], [0], marker='x', color='#2C3E50', linestyle='None', markersize=6.5, label='Observed image $\\max p_3$'),
    ]
    fig_app.legend(handles=app_handles, loc='lower center', bbox_to_anchor=(0.52, 0.99),
                   ncol=3, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#D5D8DC')
    info_app = _save_chart(fig_app, ctx.output_dir, 'figures/appendix/latent-regions-folds', df_d_out)
    return info_main, info_app


# %% [markdown]
# ## Chart 3: Native Grade 3 Recall on Source Validation & Target (Chapter 5 + Appendix D)

# %%
def build_chart_3_source_target(ctx: FigureContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dm = pd.read_csv(ctx.e0_dir / 'lodo' / 'decision_metrics.csv')
    nat = dm[(dm['decoder'] == 'native_replay') & (dm['variant'].isin(['V2', 'V3'])) & (dm['split'].isin(['source_val', 'target']))].copy()
    nat['recall_g3_pct'] = nat['recall_g3'] * 100.0
    nat['cohort_label'] = nat['cohort'].map(COHORT_LABELS)
    nat['units'] = 'percent'

    means = nat.groupby(['variant', 'split'], as_index=False)['recall_g3_pct'].mean()
    means['aggregation'] = 'equal_cohort_mean_6'

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y_pos = {'V2': 1.0, 'V3': 0.0}
    offsets = {'source_val': 0.14, 'target': -0.14}
    for var, color in [('V2', COLOR_V2), ('V3', COLOR_V3)]:
        v_src = float(means[(means['variant'] == var) & (means['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
        v_tgt = float(means[(means['variant'] == var) & (means['split'] == 'target')]['recall_g3_pct'].iloc[0])
        yc = y_pos[var]
        ax.plot([v_src, v_tgt], [yc + offsets['source_val'], yc + offsets['target']],
                color='#7F8C8D', linewidth=1.5, zorder=2)
        ax.scatter([v_src], [yc + offsets['source_val']], color=color, marker='o', s=68,
                   edgecolor='white', linewidth=0.8, zorder=4)
        ax.scatter([v_tgt], [yc + offsets['target']], color=color, marker='D', s=64,
                   facecolor='white', edgecolor=color, linewidth=1.8, zorder=4)
        ax.text(v_src + 1.2, yc + offsets['source_val'], f'Source: {v_src:.1f}%', va='center', ha='left', fontsize=8.4, color=color, fontweight='bold')
        if var == 'V2':
            ax.text(v_tgt - 1.4, yc + offsets['target'], f'Target: {v_tgt:.1f}%', va='center', ha='right', fontsize=8.4, color=color)
        else:
            ax.text(v_tgt + 1.2, yc + offsets['target'], f'Target: {v_tgt:.1f}%', va='center', ha='left', fontsize=8.4, color=color)

    v2_s = float(means[(means['variant'] == 'V2') & (means['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
    v3_s = float(means[(means['variant'] == 'V3') & (means['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
    ax.annotate(f'Source gap: {_fmt_signed(v3_s - v2_s, 1)} pp', xy=(0.5 * (v2_s + v3_s), 0.55),
                xytext=(48.5, 0.52), ha='center', va='center', fontsize=8.2,
                bbox=dict(boxstyle='round,pad=0.24', fc='#F8F9F9', ec='#BDC3C7'))

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(['$V_2$ (Nominal, native)', '$V_3$ (Ordinal, native)'])
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlim(22, 77)
    ax.set_xlabel('Mean Grade 3 recall across six folds (%)')
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#34495E', markersize=7.5, label='Source validation (5-cohort fold mixture)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#34495E', markeredgewidth=1.6, markersize=7.0, label='Held-out target cohort'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.48, 1.03), ncol=2,
              frameon=True, facecolor='white', edgecolor='#D5D8DC', fontsize=8.0)
    info_main = _save_chart(fig, ctx.output_dir, 'figures/chapter05/source-target-g3', means)
    info_main['means'] = means.to_dict(orient='records')

    # Appendix D companion: fold-level source-mixture vs target per fold
    fig_app, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.35, 3.5), gridspec_kw={'wspace': 0.10})
    y_idx = np.arange(len(COHORTS))[::-1]
    for ax, var, col, ttl in [(axes[0], 'V2', COLOR_V2, '$V_2$ (Nominal, native replay)'),
                              (axes[1], 'V3', COLOR_V3, '$V_3$ (Ordinal, native replay)')]:
        sub_v = nat[nat['variant'] == var]
        for pos, (fold, c) in zip(y_idx, enumerate(COHORTS)):
            vs = float(sub_v[(sub_v['fold_id'] == fold) & (sub_v['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
            vt = float(sub_v[(sub_v['fold_id'] == fold) & (sub_v['split'] == 'target')]['recall_g3_pct'].iloc[0])
            ax.plot([vs, vt], [pos, pos], color='#95A5A6', linewidth=1.2, zorder=2)
            ax.scatter([vs], [pos], color=col, marker='o', s=48, edgecolor='white', linewidth=0.6, zorder=4)
            ax.scatter([vt], [pos], color=col, marker='D', s=44, facecolor='white', edgecolor=col, linewidth=1.5, zorder=4)
        ax.set_title(ttl, fontsize=9.2, pad=20)
        ax.set_xlabel('Grade 3 recall (%)')
        ax.grid(axis='x', linestyle=':', alpha=0.35)
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels([f'{COHORT_LABELS[c]} fold' for c in COHORTS])
    axes[0].set_xlim(2, 78)
    st_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#34495E', markersize=7.0, label='Source-val mixture (5 source cohorts)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#34495E', markeredgewidth=1.5, markersize=6.8, label='Held-out target cohort'),
    ]
    fig_app.legend(handles=st_handles, loc='lower center', bbox_to_anchor=(0.52, 0.99), ncol=2,
                   fontsize=8.0, frameon=True, facecolor='white', edgecolor='#D5D8DC')
    df_fold_out = nat[['fold_id', 'cohort', 'cohort_label', 'variant', 'split', 'decoder', 'recall_g3_pct', 'units']].sort_values(['variant', 'fold_id', 'split'])
    df_fold_out['aggregation'] = 'per_fold_source_mixture_vs_heldout_target'
    info_app = _save_chart(fig_app, ctx.output_dir, 'figures/appendix/source-target-folds', df_fold_out)
    return info_main, info_app


# %% [markdown]
# ## Chart 4: Cohort-Level Effects of CIBD (`figures/chapter05/decoding.pdf`) + Post-Hoc Appendix E Panel

# %%
def build_chart_4_decoding(ctx: FigureContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dm = pd.read_csv(ctx.e0_dir / 'lodo' / 'decision_metrics.csv')
    tgt = dm[(dm['split'] == 'target') & (dm['decoder'].isin(['native_replay', 'cibd_selected']))].copy()
    rows = []
    for var in ('V2', 'V3'):
        for fold, cohort in enumerate(COHORTS):
            nat = tgt[(tgt['variant'] == var) & (tgt['fold_id'] == fold) & (tgt['decoder'] == 'native_replay')].iloc[0]
            cib = tgt[(tgt['variant'] == var) & (tgt['fold_id'] == fold) & (tgt['decoder'] == 'cibd_selected')].iloc[0]
            dx = (float(cib['recall_g3']) - float(nat['recall_g3'])) * 100.0
            dy = (float(cib['within1']) - float(nat['within1'])) * 100.0
            rows.append({
                'variant': var, 'fold_id': fold, 'cohort': cohort, 'cohort_label': COHORT_LABELS[cohort],
                'split': 'target', 'rule_pair': 'cibd_selected_minus_native_replay',
                'delta_g3_recall_pp': dx, 'delta_within1_pp': dy,
                'improves_both_endpoints': bool(dx > 0.0 and dy > 0.0),
                'meets_margin_g3_3pp': bool(dx >= 3.0),
                'meets_margin_w1_minus1pp': bool(dy >= -1.0),
                'meets_both_margins': bool(dx >= 3.0 and dy >= -1.0),
                'units': 'percentage_points', 'aggregation': 'per_cohort',
            })
    df_c = pd.DataFrame(rows)
    mean_rows = []
    for var in ('V2', 'V3'):
        sub = df_c[df_c['variant'] == var]
        dx = float(sub['delta_g3_recall_pp'].mean())
        dy = float(sub['delta_within1_pp'].mean())
        mean_rows.append({
            'variant': var, 'fold_id': -1, 'cohort': 'equal_cohort_mean', 'cohort_label': f'{var} equal-cohort mean',
            'split': 'target', 'rule_pair': 'cibd_selected_minus_native_replay',
            'delta_g3_recall_pp': dx, 'delta_within1_pp': dy,
            'improves_both_endpoints': bool(dx > 0.0 and dy > 0.0),
            'meets_margin_g3_3pp': bool(dx >= 3.0),
            'meets_margin_w1_minus1pp': bool(dy >= -1.0),
            'meets_both_margins': bool(dx >= 3.0 and dy >= -1.0),
            'units': 'percentage_points', 'aggregation': 'equal_cohort_mean_6',
        })
    df_all = pd.concat([df_c, pd.DataFrame(mean_rows)], ignore_index=True)

    # Main 12-point chart (V2 + V3 primary E0, compact height suited for [t] placement)
    fig, ax = plt.subplots(figsize=(6.4, 4.15))
    xlim = (-9.5, 32.8)
    ylim = (-3.05, 2.75)
    # Lightly shade ONLY the practical-margin region x >= +3 pp, y >= -1 pp
    ax.add_patch(mpatches.Rectangle((3.0, -1.0), xlim[1] - 3.0, ylim[1] - (-1.0),
                                    facecolor='#D5F5E3', alpha=0.45, zorder=0))
    ax.axhline(0.0, color='#85929E', linewidth=0.95, zorder=1)
    ax.axvline(0.0, color='#85929E', linewidth=0.95, zorder=1)
    ax.axvline(3.0, color='#1E8449', linestyle='--', linewidth=1.15, zorder=2)
    ax.axhline(-1.0, color='#1E8449', linestyle='--', linewidth=1.15, zorder=2)

    lbl_offsets: Dict[Tuple[str, str], Tuple[float, float, str, str]] = {
        ('V2', 'aptos'): (-0.65, -0.10, 'right', 'top'),
        ('V2', 'idrid'): (-0.65, 0.08, 'right', 'bottom'),
        ('V2', 'messidor2'): (-0.60, 0.04, 'right', 'center'),
        ('V2', 'ddr'): (-0.65, 0.02, 'right', 'center'),
        ('V2', 'eyepacs'): (-0.60, 0.14, 'right', 'bottom'),
        ('V2', 'deepdrid'): (0.65, -0.10, 'left', 'top'),
        ('V3', 'aptos'): (-0.65, -0.05, 'right', 'center'),
        ('V3', 'idrid'): (-0.60, 0.12, 'right', 'bottom'),
        ('V3', 'messidor2'): (0.60, 0.10, 'left', 'bottom'),
        ('V3', 'ddr'): (0.60, 0.06, 'left', 'center'),
        ('V3', 'eyepacs'): (0.65, -0.08, 'left', 'top'),
        ('V3', 'deepdrid'): (0.60, 0.10, 'left', 'bottom'),
    }

    for var, col, mkr in [('V2', COLOR_V2, 's'), ('V3', COLOR_V3, 'o')]:
        sub = df_c[df_c['variant'] == var]
        for _, r in sub.iterrows():
            x, y = float(r['delta_g3_recall_pp']), float(r['delta_within1_pp'])
            ax.scatter([x], [y], color=col, marker=mkr, s=50, edgecolor='white', linewidth=0.7, zorder=4)
            dx_off, dy_off, ha, va = lbl_offsets[(var, str(r['cohort']))]
            ax.text(x + dx_off, y + dy_off, f"{COHORT_SHORT[str(r['cohort'])]}", fontsize=8.0,
                    ha=ha, va=va, color=col,
                    bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.78) if (var == 'V2' and str(r['cohort']) == 'ddr') else None)
        m_row = df_all[(df_all['variant'] == var) & (df_all['cohort'] == 'equal_cohort_mean')].iloc[0]
        mx, my = float(m_row['delta_g3_recall_pp']), float(m_row['delta_within1_pp'])
        ax.scatter([mx], [my], color=col, marker='*', s=180, edgecolor='#1C2833', linewidth=0.8, zorder=5)
        tx, ty = (13.0, 1.90) if var == 'V2' else (17.2, -2.25)
        var_tex = '$V_2$' if var == 'V2' else '$V_3$'
        ax.annotate(
            f'{var_tex} mean ({_fmt_signed(mx, 2)}, {_fmt_signed(my, 2)})',
            xy=(mx, my), xytext=(tx, ty), ha='left', va='center',
            fontsize=8.1, fontweight='bold', color=col,
            arrowprops=dict(arrowstyle='->', color=col, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=col, alpha=0.92),
            zorder=6,
        )

    # Explicitly distinguish quadrant (>0, >0) from shaded practical-margin box
    ax.text(32.1, 2.60, f'Meets both margins\n($\\Delta\\mathrm{{G3}}\\geq+3$ pp, $\\Delta\\mathrm{{W1}}\\geq{MINUS}1$ pp)',
            ha='right', va='top', fontsize=8.0, color='#145A32')
    ax.text(-8.9, 2.58, 'Upper-right ($>0,>0$)\ndoes not imply\n$\\Delta\\mathrm{G3}\\geq+3$ pp',
            ha='left', va='top', fontsize=8.0, color='#4D5656',
            bbox=dict(boxstyle='round,pad=0.20', fc='#F8F9F9', ec='#D5D8DC', alpha=0.9))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'Target Grade 3 recall change, $100 \times (\mathrm{G3}_{\mathrm{CIBD}} - \mathrm{G3}_{\mathrm{native}})$ (pp)')
    ax.set_ylabel(r'Target Within-1 change, $100 \times (\mathrm{W1}_{\mathrm{CIBD}} - \mathrm{W1}_{\mathrm{native}})$ (pp)')
    ax.grid(linestyle=':', alpha=0.3)

    v2_imp = int(df_c[df_c['variant'] == 'V2']['improves_both_endpoints'].sum())
    v2_met = int(df_c[df_c['variant'] == 'V2']['meets_both_margins'].sum())
    v3_met = int(df_c[df_c['variant'] == 'V3']['meets_both_margins'].sum())
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_V2, markersize=6.8,
               label=f'$V_2$ ({v2_imp}/6 improve both; {v2_met}/6 meet margins)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V3, markersize=7.0,
               label=f'$V_3$ ({v3_met}/6 meet margins)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#566573', markeredgecolor='#1C2833',
               markersize=10.0, label='Equal-cohort mean (1/6)'),
    ]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=2,
              frameon=True, facecolor='white', edgecolor='#D5D8DC', fontsize=8.0)
    info_main = _save_chart(fig, ctx.output_dir, 'figures/chapter05/decoding', df_all)
    info_main['v2_met'] = v2_met
    info_main['v3_met'] = v3_met

    # Appendix E companion: V3 primary 1.0 pp vs post hoc 0.5 pp budget (proper Unicode arrow → and minus −)
    dg = pd.read_csv(ctx.supp_dir / 'analysis_g_cibd_05pp_per_fold.csv')
    dg3 = dg[dg['variant'] == 'V3'].copy()
    fig_app, ax_a = plt.subplots(figsize=(6.35, 4.15))
    xlim_a, ylim_a = (-8.5, 25.5), (-3.0, 1.35)
    ax_a.add_patch(mpatches.Rectangle((3.0, -1.0), xlim_a[1] - 3.0, ylim_a[1] - (-1.0),
                                      facecolor='#D5F5E3', alpha=0.45, zorder=0))
    ax_a.axhline(0.0, color='#85929E', linewidth=0.95, zorder=1)
    ax_a.axvline(0.0, color='#85929E', linewidth=0.95, zorder=1)
    ax_a.axvline(3.0, color='#1E8449', linestyle='--', linewidth=1.15, zorder=2)
    ax_a.axhline(-1.0, color='#1E8449', linestyle='--', linewidth=1.15, zorder=2)

    app_lbl_pos = {
        'aptos': (0.0, 0.36, 'center', 'bottom'),
        'idrid': (0.0, 0.14, 'center', 'bottom'),
        'messidor2': (0.55, 0.08, 'left', 'bottom'),
        'ddr': (0.55, 0.08, 'left', 'bottom'),
        'eyepacs': (-0.55, -0.05, 'right', 'top'),
        'deepdrid': (-0.45, 0.06, 'right', 'bottom'),
    }
    for fold, c in enumerate(COHORTS):
        r10 = dg3[(dg3['fold_id'] == fold) & (dg3['rule'] == 'cibd_1.0pp_budget')].iloc[0]
        r05 = dg3[(dg3['fold_id'] == fold) & (dg3['rule'] == 'cibd_0.5pp_budget')].iloc[0]
        x1, y1 = float(r10['delta_g3_recall_pp']), float(r10['delta_within1_pp'])
        x2, y2 = float(r05['delta_g3_recall_pp']), float(r05['delta_within1_pp'])
        ax_a.annotate('', xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2, shrinkA=4, shrinkB=4), zorder=3)
        ax_a.scatter([x1], [y1], color='white', edgecolor=COLOR_V3, marker='o', s=54, linewidth=1.7, zorder=4)
        ax_a.scatter([x2], [y2], color=COLOR_V3, edgecolor='white', marker='s', s=54, linewidth=0.7, zorder=5)
        dx, dy, ha, va = app_lbl_pos[c]
        ax_a.text(
            x2 + dx, y2 + dy, COHORT_SHORT[c], fontsize=8.1, color='#1C2833', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.78) if c in ('deepdrid', 'eyepacs') else None,
            zorder=6,
        )

    m10 = dg3[dg3['rule'] == 'cibd_1.0pp_budget'][['delta_g3_recall_pp', 'delta_within1_pp']].mean()
    m05 = dg3[dg3['rule'] == 'cibd_0.5pp_budget'][['delta_g3_recall_pp', 'delta_within1_pp']].mean()
    ax_a.scatter([m10['delta_g3_recall_pp']], [m10['delta_within1_pp']], color='white', edgecolor='#1C2833', marker='*', s=150, linewidth=1.3, zorder=6)
    ax_a.scatter([m05['delta_g3_recall_pp']], [m05['delta_within1_pp']], color='#1C2833', edgecolor='white', marker='*', s=165, linewidth=0.7, zorder=6)
    ax_a.annotate('', xy=(m05['delta_g3_recall_pp'], m05['delta_within1_pp']), xytext=(m10['delta_g3_recall_pp'], m10['delta_within1_pp']),
                  arrowprops=dict(arrowstyle='->', color='#1C2833', lw=1.4, ls='--'), zorder=5)

    mean_lbl = (
        f"Mean: ({_fmt_signed(float(m10['delta_g3_recall_pp']), 2)}, {_fmt_signed(float(m10['delta_within1_pp']), 2)}) "
        f"{ARROW} ({_fmt_signed(float(m05['delta_g3_recall_pp']), 2)}, {_fmt_signed(float(m05['delta_within1_pp']), 2)})"
    )
    handles_a = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=COLOR_V3, markeredgewidth=1.7, markersize=6.8,
               label='Primary $V_3$ (1.0 pp; 3/6 meet margins)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_V3, markersize=6.8,
               label='Post hoc $V_3$ (0.5 pp; 4/6 meet margins)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#1C2833', markersize=9.5, label=mean_lbl),
    ]
    ax_a.legend(handles=handles_a, loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=1,
                frameon=True, facecolor='white', edgecolor='#D5D8DC', fontsize=8.0)
    ax_a.set_xlim(*xlim_a)
    ax_a.set_ylim(*ylim_a)
    ax_a.set_xlabel('Target Grade 3 recall change vs. native $V_3$ (pp)')
    ax_a.set_ylabel('Target Within-1 change vs. native $V_3$ (pp)')
    ax_a.grid(linestyle=':', alpha=0.3)
    info_app = _save_chart(fig_app, ctx.output_dir, 'figures/appendix/decoding-05pp', dg3)
    return info_main, info_app


# %% [markdown]
# ## Chart 5: Probe Recall & Precision (`figures/chapter05/probing.pdf`)

# %%
def build_chart_5_probing(ctx: FigureContext) -> Dict[str, Any]:
    e0_dm = pd.read_csv(ctx.e0_dir / 'lodo' / 'decision_metrics.csv')
    e1_pm = pd.read_csv(ctx.e1_dir / 'probe_metrics.csv')

    nat_t = e0_dm[(e0_dm['split'] == 'target') & (e0_dm['decoder'] == 'native_replay') & (e0_dm['variant'].isin(['V2', 'V3']))].copy()
    nat_t['row_key'] = nat_t['variant'].map({'V2': 'native_v2', 'V3': 'native_v3'})
    nat_t['rule_source'] = 'P0/E0_replayed_native'

    pr_t = e1_pm[(e1_pm['split'] == 'target') & (e1_pm['control_seed'].isna()) & (e1_pm['representation'].isin(['z_v2', 'z_v3', 's_v3']))].copy()
    pr_t['row_key'] = pr_t['representation'].map({'z_v2': 'probe_z2', 'z_v3': 'probe_z3', 's_v3': 'probe_s3'})
    pr_t['rule_source'] = 'E1_source_macro_recall_selected'

    keep_cols = ['row_key', 'fold_id', 'cohort', 'recall_g3', 'g3_precision', 'within1', 'qwk', 'rule_source']
    df_p = pd.concat([nat_t[keep_cols], pr_t[keep_cols]], ignore_index=True)
    df_p['cohort_label'] = df_p['cohort'].map(COHORT_LABELS)
    df_p['recall_g3_pct'] = df_p['recall_g3'] * 100.0
    df_p['g3_precision_pct'] = df_p['g3_precision'] * 100.0
    df_p['split'] = 'target'
    df_p['units'] = 'percent'
    df_p['aggregation'] = 'per_cohort_with_equal_cohort_mean'

    row_order = [
        ('native_v2', 'Native $V_2$', COLOR_V2, True),
        ('probe_z2', 'Probe $\\mathbf{z}_2$', COLOR_V2, False),
        ('native_v3', 'Native $V_3$', COLOR_V3, True),
        ('probe_z3', 'Probe $\\mathbf{z}_3$', COLOR_V3, False),
        ('probe_s3', 'Probe $s_3$', COLOR_V3, False),
    ]
    fig, (ax_r, ax_p) = plt.subplots(1, 2, sharey=True, figsize=(6.5, 3.8), gridspec_kw={'wspace': 0.10})
    y_map = {rk: 4 - i for i, (rk, _, _, _) in enumerate(row_order)}

    jitter = np.linspace(-0.16, 0.16, len(COHORTS))
    jit_map = dict(zip(COHORTS, jitter))

    for rk, _, col, is_native in row_order:
        y0 = y_map[rk]
        sub = df_p[df_p['row_key'] == rk].sort_values('fold_id')
        for _, r in sub.iterrows():
            c = str(r['cohort'])
            yj = y0 + jit_map[c]
            ax_r.scatter([r['recall_g3_pct']], [yj], color=COHORT_COLORS[c], marker=COHORT_MARKERS[c],
                         s=30, alpha=0.84, edgecolor='white', linewidth=0.4, zorder=3)
            ax_p.scatter([r['g3_precision_pct']], [yj], color=COHORT_COLORS[c], marker=COHORT_MARKERS[c],
                         s=30, alpha=0.84, edgecolor='white', linewidth=0.4, zorder=3)

        mr = float(sub['recall_g3_pct'].mean())
        mp = float(sub['g3_precision_pct'].mean())
        ax_r.scatter([mr], [y0], color=col if not is_native else 'white', edgecolor=col,
                     marker='D', s=76, linewidth=1.8, zorder=5)
        ax_p.scatter([mp], [y0], color=col if not is_native else 'white', edgecolor=col,
                     marker='D', s=76, linewidth=1.8, zorder=5)
        ax_r.text(mr, y0 + 0.25, f'{mr:.2f}%', ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=col)
        ax_p.text(mp, y0 + 0.25, f'{mp:.2f}%', ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=col)

    ax_r.axhline(2.5, color='#CBD5E1', linestyle=':', linewidth=1.0)
    ax_p.axhline(2.5, color='#CBD5E1', linestyle=':', linewidth=1.0)
    ax_r.set_yticks([y_map[r[0]] for r in row_order])
    ax_r.set_yticklabels([r[1] for r in row_order])
    ax_r.set_ylim(-0.55, 4.70)
    ax_r.set_xlim(3, 82)
    ax_p.set_xlim(3, 66)
    ax_r.set_xlabel('Grade 3 recall (%)')
    ax_p.set_xlabel('Grade 3 precision (%)')
    ax_r.set_title('Target Grade 3 recall', fontsize=9.4)
    ax_p.set_title('Target Grade 3 precision', fontsize=9.4)
    ax_r.grid(axis='x', linestyle=':', alpha=0.35)
    ax_p.grid(axis='x', linestyle=':', alpha=0.35)

    cohort_handles = [
        Line2D([0], [0], marker=COHORT_MARKERS[c], color='w', markerfacecolor=COHORT_COLORS[c],
               markersize=6.0, label=COHORT_SHORT[c])
        for c in COHORTS
    ] + [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#34495E', markersize=7.0, label='Equal-cohort mean')
    ]
    fig.legend(handles=cohort_handles, loc='upper center', ncol=4, bbox_to_anchor=(0.52, -0.02),
               frameon=True, facecolor='white', edgecolor='#D5D8DC', fontsize=8.0)
    info = _save_chart(fig, ctx.output_dir, 'figures/chapter05/probing', df_p)
    info['probe_means'] = df_p.groupby('row_key')[['recall_g3_pct', 'g3_precision_pct']].mean().to_dict()
    return info


# %% [markdown]
# ## Chart 6: Integrated Operating-Point Map (`figures/chapter05/operating-points.pdf`)

# %%
def build_chart_6_operating_points(ctx: FigureContext) -> Dict[str, Any]:
    a_sum = pd.read_csv(ctx.supp_dir / 'analysis_a_summary.csv')
    e1_eq = pd.read_csv(ctx.e1_dir / 'equal_cohort_means.csv')
    e1_pm = pd.read_csv(ctx.e1_dir / 'probe_metrics.csv')
    p_sub = e1_pm[(e1_pm['split'] == 'target') & (e1_pm['control_seed'].isna())].groupby('representation')[['within1', 'recall_g3', 'g3_precision']].mean()

    pts = []
    for _, r in a_sum.iterrows():
        dec = str(r['decoder'])
        if pd.notna(r['v2_within1_pct']) and pd.notna(r['v2_recall_g3_pct']):
            lbl = '$V_2$ Native (= argmax)' if dec == 'argmax' else (
                '$V_2$ Posterior median' if dec == 'posterior_median' else (
                    '$V_2$ Rounded expectation' if dec == 'rounded_expectation' else '$V_2$ CIBD (source-selected)'
                )
            )
            pts.append({
                'model': 'V2', 'method_family': 'CIBD' if dec == 'cibd_selected' else ('Native' if dec == 'argmax' else 'Conventional_E0'),
                'rule': dec, 'label': lbl, 'source': 'analysis_a_summary',
                'within1_pct': float(r['v2_within1_pct']),
                'recall_g3_pct': float(r['v2_recall_g3_pct']),
                'g3_precision_pct': float(r['v2_g3_precision_pct']),
                'highlight': (dec == 'rounded_expectation'),
            })
        if pd.notna(r['v3_within1_pct']) and pd.notna(r['v3_recall_g3_pct']):
            lbl = '$V_3$ Argmax' if dec == 'argmax' else (
                '$V_3$ Posterior median' if dec == 'posterior_median' else (
                    '$V_3$ Native (= rounded exp.)' if dec == 'rounded_expectation' else (
                        '$V_3$ CIBD (source-selected)' if dec == 'cibd_selected' else '$V_3$ Latent cutpoints'
                    )
                )
            )
            pts.append({
                'model': 'V3', 'method_family': 'CIBD' if dec == 'cibd_selected' else ('Native' if dec == 'rounded_expectation' else 'Conventional_E0'),
                'rule': dec, 'label': lbl, 'source': 'analysis_a_summary',
                'within1_pct': float(r['v3_within1_pct']),
                'recall_g3_pct': float(r['v3_recall_g3_pct']),
                'g3_precision_pct': float(r['v3_g3_precision_pct']),
                'highlight': (dec == 'cibd_selected'),
            })

    for rep, model, lbl, hi in [
        ('z_v2', 'V2', 'Probe $\\mathbf{z}_2$ (E1)', False),
        ('z_v3', 'V3', 'Probe $\\mathbf{z}_3$ (E1)', False),
        ('s_v3', 'V3', 'Probe $s_3$ (E1 scalar)', True),
    ]:
        eq_r = e1_eq[e1_eq['method'] == rep].iloc[0]
        w1 = float(p_sub.loc[rep, 'within1']) * 100.0
        g3 = float(p_sub.loc[rep, 'recall_g3']) * 100.0
        if abs(w1 - float(eq_r['within1']) * 100.0) > 1e-9 or abs(g3 - float(eq_r['recall_g3']) * 100.0) > 1e-9:
            raise ValueError(f'E1 mean cross-check mismatch on {rep}')
        pts.append({
            'model': model, 'method_family': 'E1_Probe_Scalar' if rep == 's_v3' else 'E1_Probe_Vector',
            'rule': f'probe_{rep}', 'label': lbl, 'source': 'e1_equal_cohort_means',
            'within1_pct': w1, 'recall_g3_pct': g3, 'g3_precision_pct': float(eq_r['g3_precision']) * 100.0,
            'highlight': hi,
        })

    df_op = pd.DataFrame(pts)
    df_op['split'] = 'target'
    df_op['units'] = 'percent'
    df_op['aggregation'] = 'equal_cohort_mean_6'

    fig, ax = plt.subplots(figsize=(6.8, 4.95))
    v3_nat = df_op[(df_op['model'] == 'V3') & (df_op['rule'] == 'rounded_expectation')].iloc[0]
    w1_bound = float(v3_nat['within1_pct']) - 1.0
    g3_bound = float(v3_nat['recall_g3_pct']) + 3.0
    ax.axvline(w1_bound, color='#85929E', linestyle=':', linewidth=1.0, zorder=1)
    ax.axhline(g3_bound, color='#85929E', linestyle=':', linewidth=1.0, zorder=1)
    ax.text(w1_bound + 0.12, 27.2, f'$V_3$ Native {MINUS} 1 pp', rotation=90, va='bottom', ha='left', fontsize=8.0, color='#5D6D7E')
    ax.text(84.0, g3_bound + 0.45, '$V_3$ Native + 3 pp', va='bottom', ha='left', fontsize=8.0, color='#5D6D7E')

    marker_map = {
        'Native': 's',
        'Conventional_E0': 'o',
        'CIBD': '*',
        'E1_Probe_Vector': 'D',
        'E1_Probe_Scalar': '^',
    }
    text_offsets: Dict[Tuple[str, str], Tuple[float, float, str, str]] = {
        ('V2', 'argmax'): (0.14, -0.45, 'left', 'top'),
        ('V2', 'posterior_median'): (-0.20, 0.25, 'right', 'bottom'),
        ('V2', 'rounded_expectation'): (0.24, -0.20, 'left', 'center'),
        ('V2', 'cibd_selected'): (-0.42, 0.05, 'right', 'center'),
        ('V2', 'probe_z_v2'): (-0.20, -0.15, 'right', 'top'),
        ('V3', 'argmax'): (0.20, 0.45, 'left', 'bottom'),
        ('V3', 'posterior_median'): (-0.20, 0.35, 'right', 'bottom'),
        ('V3', 'rounded_expectation'): (0.05, -0.95, 'left', 'top'),
        ('V3', 'cutpoint_latent'): (-0.20, 0.40, 'right', 'bottom'),
        ('V3', 'cibd_selected'): (-0.26, -0.25, 'right', 'top'),
        ('V3', 'probe_z_v3'): (-0.22, -0.10, 'right', 'center'),
        ('V3', 'probe_s_v3'): (0.12, 1.25, 'center', 'bottom'),
    }

    for _, r in df_op.iterrows():
        col = COLOR_V2 if r['model'] == 'V2' else COLOR_V3
        fam = str(r['method_family'])
        mkr = marker_map[fam]
        hi = bool(r['highlight'])
        sz = 175 if mkr == '*' else (95 if hi else 62)
        ec = '#1C2833' if hi else 'white'
        lw = 1.2 if hi else 0.7
        ax.scatter([r['within1_pct']], [r['recall_g3_pct']], color=col, marker=mkr, s=sz,
                   edgecolor=ec, linewidth=lw, zorder=5 if hi else 4)
        dx, dy, ha, va = text_offsets[(str(r['model']), str(r['rule']))]
        ax.text(r['within1_pct'] + dx, r['recall_g3_pct'] + dy, str(r['label']),
                fontsize=8.0, fontweight='bold' if hi else 'normal',
                color=col if hi else '#1C2833', ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec=col, alpha=0.92) if hi else None)

    ax.set_xlim(83.6, 95.8)
    ax.set_ylim(26.6, 58.6)
    ax.set_xlabel('Equal-cohort target Within-1 accuracy (%)')
    ax.set_ylabel('Equal-cohort target Grade 3 recall (%)')
    ax.grid(linestyle=':', alpha=0.3)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V2, markersize=7.5, label='$V_2$ nominal family'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V3, markersize=7.5, label='$V_3$ ordinal CLM family'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#566573', markersize=6.8, label='Native rule'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#566573', markersize=6.8, label='Conventional E0 rule'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#566573', markeredgecolor='#1C2833', markersize=10.0, label='Source-selected CIBD (E0)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#566573', markersize=6.5, label='Frozen vector probe $\\mathbf{z}$ (E1)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#566573', markeredgecolor='#1C2833', markersize=7.5, label='Frozen scalar probe $s_3$ (E1)'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=3,
              frameon=True, facecolor='white', edgecolor='#D5D8DC', fontsize=8.0, columnspacing=1.0)
    return _save_chart(fig, ctx.output_dir, 'figures/chapter05/operating-points', df_op)


# %% [markdown]
# ## Documentation & Manifest Generation (`FIGURE_INTEGRATION.md`, `figure_manifest.json`)

# %%
def write_integration_guide(ctx: FigureContext, meta: Dict[str, Any]) -> Path:
    """Write Overleaf integration guide with short/full captions, anchors, and localization reuse note."""
    c1 = meta['charts']['tradeoff']
    c2 = meta['charts']['latent_regions']
    c3 = meta['charts']['source_target_g3']
    c4 = meta['charts']['decoding']
    c5 = meta['charts']['probing']

    loc_note = str(ctx.localization_asset) if ctx.localization_asset else 'MISSING: verify result/figure_8_xai_qualitative_grid.pdf'

    lines = [
        '# Figure Integration Guide (`result/thesis_figures_v2/`)',
        '',
        'Copy the `figures/` tree from this bundle directly into the Overleaf thesis project root. Manual draw.io diagrams (`figures/chapter03/architecture.pdf`, `figures/chapter04/evaluation.pdf`) remain untouched and are specified separately in `DRAWIO_SPECIFICATIONS.md`.',
        '',
        '## Visual & Editorial Conventions Applied (T1-E / T1-F)',
        '- Unified palette: $V_2$ blue (`#0072B2`), $V_3$ red (`#C0392B`), consistent Grade 3 highlight (`#B02A24`).',
        '- Single readable font family (`DejaVu Sans`), minimum rendered text $\\ge 8$~pt at $\\approx 16$~cm insertion width, Unicode mathematical minus (`−`) and arrow (`→`), and true subscripts.',
        '- Every chart uses short and full LaTeX captions (`\\caption[short]{full}`) so the List of Figures remains concise.',
        '- `decoding.pdf` uses `[t]` placement to prevent single-figure float pages; stale `Cohorts: 1 aptos, 2 idrid...` text is removed.',
        '',
        '---',
        '',
        '## Chapter 5 Quantitative Figures',
        '',
        '### 1. Archived Architectural Trade-off (`figures/chapter05/tradeoff.pdf`)',
        f"- **Source CSV**: `{ctx.p0_dir.name}/g0_comparison.csv` (`split=target`, `metric in {{recall_g3, within1}}`, `reference` column) -> `{c1['csv']}`",
        '- **Section / Anchor**: `Ordinal Modeling and the Severity Trade-off` (`\\label{subsec:5-3-3}`), existing `\\IfFileExists{figures/chapter05/tradeoff.pdf}` block',
        '- **LaTeX Label**: `\\label{fig:tradeoff}` (placement `[htbp]`)',
        '- **Introductory Sentence**: `Figure~\\ref{fig:tradeoff} compares cohort-level changes in exact Grade~3 recall and Within-1 accuracy from the archived nominal configuration ($V_2$) to the ordinal cumulative-link configuration ($V_3$).`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Archived Grade~3 recall and Within-1 changes ($V_3 - V_2$)]{%',
        r'  Archived target changes from $V_2$ (nominal) to $V_3$ (ordinal CLM) across six held-out cohorts and their equal-cohort mean ($1/6$), reported to one decimal place in percentage points ($100 \times (V_3 - V_2)$).',
        r'  Exact Grade~3 recall declines in every cohort ($-12.0$~pp mean), whereas Within-1 accuracy improves in five of six cohorts ($+1.5$~pp mean, declining only on DeepDRiD).}',
        '```',
        '',
        '### 2. Latent Severity and Grade 3 Decision Regions (`figures/chapter05/latent-regions.pdf`)',
        f"- **Source Files**: `{ctx.e0_dir.name}/lodo/units/V3_fold_1_idrid/fold_report/decision_grid_float64_diagnostic.csv`, `decision_intervals.csv`, `selected_params.json`, `{ctx.p0_dir.name}/lodo/V3_fold_1_idrid/target/predictions.csv`, verified against `analysis_d_v3_latent_regions_per_fold.csv` -> `{c2['csv']}`",
        '- **Section / Anchor**: `Decision Regions and Grade 3 Error Redistribution` (`\\label{subsec:5-7-2}`), first paragraph',
        '- **LaTeX Label**: `\\label{fig:latent-regions}` (placement `[htbp]`)',
        '- **Introductory Sentence**: `Figure~\\ref{fig:latent-regions} aligns the empirical IDRiD target score distributions, cumulative-link class probabilities, and Grade~3 decision bands on a shared latent severity axis.`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[IDRiD $V_3$ latent severity distributions and Grade~3 decision regions]{%',
        f"  Latent severity mechanism for the IDRiD-held-out $V_3$ checkpoint ($n={c2['n_images']}$ target images; horizontal axis zoomed to $[-3.30, 4.35]$ with {c2['n_outside_zoom_gt_4_35']} images above $4.35$: 27 Grade~4, 3 Grade~3, 2 Grade~2, and 1 Grade~0, maximum $s={c2['max_s_obs']:.2f}$).",
        f"  Top: within-grade normalized empirical target score densities. Middle: cumulative-link probabilities with cutpoints $\\theta_0=-1.504$, $\\theta_1=-0.468$, $\\theta_2=0.605$, $\\theta_3=1.552$ and peak diagnostic-grid $p_3={c2['peak_p3_grid']:.3f}$ (observed image maximum ${c2['obs_max_p3']:.3f}$).",
        f"  Bottom: Grade~3 prediction bands under source-locked CIBD ($\\alpha=0.5, \\tau=1.0, q=2$), raw latent cutpoints, and native rounded expectation; the native rule overlaps {c2['overlap_pct_of_cutpoint']:.1f}\\% of the raw-cutpoint width.}}",
        '```',
        '',
        '### 3. Native Grade 3 Recall on Source and Target (`figures/chapter05/source-target-g3.pdf`)',
        f"- **Source CSV**: `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`decoder=native_replay`, `V2` & `V3`, `source_val` vs. `target`) -> `{c3['csv']}`",
        '- **Section / Anchor**: `Source--Target Comparison and Latent Severity Distributions` (`\\label{subsec:5-6-2}`) or `Joint Diagnostic Evidence` (`\\label{subsec:5-10-1}`); add reference to Appendix D Figure~\\ref{fig:app-source-target-folds}',
        '- **LaTeX Label**: `\\label{fig:source-target-g3}` (placement `[htbp]`)',
        '- **Introductory Sentence**: `Figure~\\ref{fig:source-target-g3} compares the equal-cohort mean native Grade~3 recall of replayed $V_2$ and $V_3$ checkpoints on source validation and held-out targets.`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Mean native Grade~3 recall on source validation and targets]{%',
        r'  Equal-cohort six-fold mean Grade~3 recall (\%) for replayed native $V_2$ (argmax) and $V_3$ (rounded expectation) checkpoints on source validation ($61.9\%$ versus $30.9\%$) and held-out targets ($43.6\%$ versus $32.2\%$).',
        r'  The $-31.0$~pp ordinal deficit already present on source validation narrows on targets mainly because $V_2$ recall declines under transfer.}',
        '```',
        '',
        '### 4. Cohort-Level Effects of Primary CIBD (`figures/chapter05/decoding.pdf`)',
        f"- **Source CSV**: `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`split=target`, `decoder in {{native_replay, cibd_selected}}`, `V2` and `V3`) -> `{c4['csv']}`",
        '- **Section / Anchor**: `Cost-interpolated Bayes decoding within E0` — `Cohort Robustness and Limits of Decision Intervention` (`\\label{sec:5-8-4}`), existing conditional block',
        '- **LaTeX Label**: `\\label{fig:decoding}` (**use `[t]` placement**; remove obsolete `Cohorts: 1 aptos, 2 idrid...` text)',
        '- **Introductory Sentence**: `Figure~\\ref{fig:decoding} plots paired target changes in Grade~3 recall and Within-1 accuracy from replayed native decoding to primary source-selected CIBD across all 12 model--cohort combinations.`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\begin{figure}[t]',
        r'\centering\includegraphics[width=.90\linewidth,height=.40\textheight,keepaspectratio]{figures/chapter05/decoding.pdf}',
        r'\caption[Target changes from native decoding to primary source-selected CIBD]{%',
        r'  Paired target changes ($100 \times (\mathrm{CIBD} - \mathrm{native})$) under primary source-selected CIBD (1.0~pp source allowance) across six held-out cohorts for $V_2$ (blue squares; equal-cohort mean $+10.10$~pp Grade~3 recall, $+0.99$~pp Within-1) and $V_3$ (red circles; $+8.85$~pp, $-1.16$~pp).',
        r'  Dashed lines mark the practical margins ($x = +3$~pp recall, $y = -1$~pp Within-1) with the jointly satisfied region shaded; all six $V_2$ cohorts improve on both endpoints, whereas 4 of 6 $V_2$ cohorts (Messidor-2 and DDR gain $<3$~pp recall) and 3 of 6 $V_3$ cohorts satisfy both margins.}',
        r'\label{fig:decoding}',
        r'\end{figure}',
        '```',
        '',
        '### 5. Frozen Probe Recall and Precision (`figures/chapter05/probing.pdf`)',
        f"- **Source CSVs**: `{ctx.e1_dir.name}/probe_metrics.csv` (`split=target`, primary probes) + `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`native_replay`) -> `{c5['csv']}`",
        '- **Section / Anchor**: `Linear Accessibility in Frozen Features and Scalar Scores` (`\\label{subsec:5-9-1}`), existing conditional block',
        '- **LaTeX Label**: `\\label{fig:probing}` (placement `[htbp]`)',
        '- **Introductory Sentence**: `Figure~\\ref{fig:probing} compares held-out Grade~3 recall and precision across the six cohorts for replayed native classifiers and primary source-trained E1 probes.`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Target Grade~3 recall and precision under frozen-input probing]{%',
        r'  Held-out Grade~3 recall (left) and precision (right) across six cohorts (small markers) and equal-cohort means (diamonds) for replayed native classifiers and primary E1 linear probes selected by source-validation macro recall.',
        r'  The scalar $s_3$ probe raises recall in every cohort ($54.09\%$ mean versus $32.22\%$ native $V_3$, $51.09\%$ $\mathbf{z}_3$, and $44.64\%$ $\mathbf{z}_2$), while six-cohort mean precision remains comparable ($31.41\%$ versus $31.71\%$ native) amid wide cohort variation ($9.4\%$--$59.4\%$).}',
        '```',
        '',
        '### 6. Integrated Operating-Point Map (`figures/chapter05/operating-points.pdf`)',
        f"- **Source CSVs**: `thesis_supplementary_v1/analysis_a_summary.csv` and `{ctx.e1_dir.name}/equal_cohort_means.csv` -> `{meta['charts']['operating_points']['csv']}`",
        '- **Section / Anchor**: `Joint Diagnostic Evidence` (`\\label{subsec:5-10-1}`), existing block',
        '- **LaTeX Label**: `\\label{fig:operating-points}` (placement `[htbp]`)',
        '- **Introductory Sentence**: `Figure~\\ref{fig:operating-points} synthesizes absolute equal-cohort target Within-1 accuracy and Grade~3 recall across conventional E0 decoders, primary source-selected CIBD, and primary E1 probes.`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Equal-cohort target operating points of E0 decoders and E1 probes]{%',
        r'  Absolute equal-cohort six-fold target means for replayed $V_2$ (blue) and $V_3$ (red) E0 decision rules and primary E1 linear probes.',
        r'  Coincident rules are deduplicated ($V_2$ native coincides with argmax; $V_3$ native coincides with rounded expectation), and latent-cutpoint decoding applies only to $V_3$ CLM.',
        r'  Points represent distinct mappings rather than a continuous training trajectory; the scalar $s_3$ probe was fitted on source labels in E1 outside the prespecified E0 decoder test.}',
        '```',
        '',
        '---',
        '',
        '## Appendix Companions & Localization Reuse Note',
        '',
        '### A1. Six-Fold Latent Region Representativeness (`figures/appendix/latent-regions-folds.pdf`)',
        '- **Source CSV**: `thesis_supplementary_v1/analysis_d_v3_latent_regions_per_fold.csv`',
        '- **Section / Anchor**: Appendix E (`\\label{app:decoder_results}`), existing block `\\label{fig:latent-regions-folds}`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Grade~3 interval overlap and peak probability by fold (post hoc)]{%',
        r'  Post hoc six-fold $V_3$ decision-interval overlap (left, percentage of raw latent-cutpoint width) and peak Grade~3 probability (right, diagnostic-grid circles versus observed target-image crosses).',
        r'  IDRiD (diamonds; $46.8\%$ overlap, grid $\max p_3=0.336$) lies near the center of the cross-fold range ($22.2\%$--$63.0\%$ overlap; $\max p_3$ $0.243$--$0.482$).}',
        '```',
        '',
        '### A2. Fold-Level Source vs. Target Native Grade 3 Recall (`figures/appendix/source-target-folds.pdf`)',
        f"- **Source CSV**: `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`decoder=native_replay`, `V2` & `V3`, `source_val` vs. `target`)",
        '- **Section / Anchor**: Appendix D (`\\label{app:reproduction}`), after `Table~\\ref{tab:ch5-p0}`',
        '- **LaTeX Label**: `\\label{fig:app-source-target-folds}`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Fold-level native Grade~3 recall on source mixtures and targets]{%',
        r"  Per-fold Grade~3 recall (\%) for replayed native $V_2$ (left) and $V_3$ (right), comparing each fold's five-cohort source-validation mixture (filled circles) with the held-out target cohort (open diamonds).",
        r'  Connecting segments denote within-fold comparisons and do not follow the same patients or images across domains.}',
        '```',
        '',
        '### A3. Post Hoc 0.5 pp vs. Primary 1.0 pp CIBD Sensitivity (`figures/appendix/decoding-05pp.pdf`)',
        '- **Source CSV**: `thesis_supplementary_v1/analysis_g_cibd_05pp_per_fold.csv` (`variant=V3`)',
        '- **Section / Anchor**: Appendix E (`\\label{appsec:cibd-budget-sensitivity}`), existing block `\\label{fig:cibd-budget-sensitivity}`',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[CIBD source-allowance sensitivity (post hoc)]{%',
        r'  Post hoc sensitivity comparison of $V_3$ CIBD target changes under the primary 1.0~pp source-validation allowance (open circles) versus a stricter post hoc 0.5~pp allowance (filled squares), connected within each cohort.',
        r'  Under the 0.5~pp allowance, 4 of 6 cohorts (APTOS~2019, IDRiD, Messidor-2, and DeepDRiD) satisfy both target margins ($+6.56$~pp mean Grade~3 recall, $-0.71$~pp Within-1).}',
        '```',
        '',
        '### R1. Localization Grid Reuse Note (`figures/chapter05/localization.pdf`)',
        f"- **Verified Original Asset**: `{loc_note}` (underlying images and masks preserved unchanged; copy/rename to `figures/chapter05/localization.pdf` in Overleaf).",
        '- **Section / Anchor**: `Localization Evidence and Failure Cases` (`\\label{subsec:5-5-2}`), immediately after `Table~\\ref{tab:ch5-localization}` (`\\label{fig:localization-grid}`).',
        '- **Short & Full Caption**:',
        '```latex',
        r'\caption[Selected IDRiD localization pointing-game misses]{%',
        r'  Qualitative overlays from the archived 81-image IDRiD lesion-localization evaluation showing local gated attention and global Grad-CAM maps for five selected pointing-game misses \citep{nguyen2026clinically}.',
        r'  These illustrative failure cases highlight misalignment between peak saliency and annotated lesion masks rather than representing the full 81-image distribution.}',
        '```',
        '',
    ]
    guide_path = ctx.output_dir / 'FIGURE_INTEGRATION.md'
    guide_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return guide_path


def run_all(ctx: FigureContext) -> Dict[str, Any]:
    """Generate all six main charts, three appendix charts, integration guide, and provenance manifest."""
    c1 = build_chart_1_tradeoff(ctx)
    c2_main, c2_app = build_chart_2_latent_regions(ctx)
    c3_main, c3_app = build_chart_3_source_target(ctx)
    c4_main, c4_app = build_chart_4_decoding(ctx)
    c5 = build_chart_5_probing(ctx)
    c6 = build_chart_6_operating_points(ctx)

    git_commit: Optional[str] = None
    if ctx.repo_root is not None and (ctx.repo_root / '.git').exists():
        try:
            git_commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], cwd=str(ctx.repo_root),
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        except Exception:
            pass

    script_path = Path(__file__).resolve() if '__file__' in globals() else None
    manifest: Dict[str, Any] = {
        'execution_mode': ctx.execution_mode,
        'git_commit': git_commit,
        'script_sha256': hashlib.sha256(script_path.read_bytes()).hexdigest() if (script_path and script_path.is_file()) else NOTEBOOK_CODE_SHA256,
        'inputs': {
            'p0_dir': str(ctx.p0_dir),
            'e0_dir': str(ctx.e0_dir),
            'e1_dir': str(ctx.e1_dir),
            'supp_dir': str(ctx.supp_dir),
            'localization_asset': str(ctx.localization_asset) if ctx.localization_asset else None,
        },
        'output_dir': str(ctx.output_dir),
        'charts': {
            'tradeoff': c1,
            'latent_regions': c2_main,
            'latent_regions_folds': c2_app,
            'source_target_g3': c3_main,
            'source_target_folds': c3_app,
            'decoding': c4_main,
            'decoding_05pp': c4_app,
            'probing': c5,
            'operating_points': c6,
        },
    }
    guide_path = write_integration_guide(ctx, manifest)
    manifest['integration_guide'] = str(guide_path.relative_to(ctx.output_dir)).replace('\\', '/')
    (ctx.output_dir / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Generate revised quantitative thesis figures and Overleaf bundle (CPU-only).')
    p.add_argument('--p0-dir', type=Path, default=None)
    p.add_argument('--e0-dir', type=Path, default=None)
    p.add_argument('--e1-dir', type=Path, default=None)
    p.add_argument('--supp-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = build_context(
        p0_dir=args.p0_dir, e0_dir=args.e0_dir, e1_dir=args.e1_dir,
        supp_dir=args.supp_dir, output_dir=args.output_dir,
        execution_mode='cli_script',
    )
    manifest = run_all(ctx)
    print(f'Figure bundle generated at: {ctx.output_dir}')
    for name, item in manifest['charts'].items():
        print(f"  {name:22s} -> {item['pdf']} ({item['n_rows']} rows)")
    print(f"Integration guide: {ctx.output_dir / manifest['integration_guide']}")
    return 0


# %%
if __name__ == '__main__' and _running_in_notebook():
    from IPython.display import display, Image
    CTX = build_context(
        repo_root=find_repo_root(),
        p0_dir=P0_RUN, e0_dir=E0_RUN, e1_dir=E1_RUN, supp_dir=SUPP_RUN, output_dir=OUTPUT_DIR,
        execution_mode='notebook',
    )
    MANIFEST = run_all(CTX)
    print('Figure bundle:', CTX.output_dir)
    for k, info in MANIFEST['charts'].items():
        print(k, '->', info['pdf'])
        display(Image(filename=str(CTX.output_dir / info['png'])))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
