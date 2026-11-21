# %% [markdown]
# # Thesis Quantitative Figure Generator (Overleaf Bundle)
#
# Generates publication-grade vector PDFs, 300-DPI PNG previews, exact plotted CSVs,
# `figure_manifest.json`, and `FIGURE_INTEGRATION.md` from archived DMIP references,
# P0 replay, E0/E1 primary caches, and post-hoc supplementary CPU outputs.
#
# All source directories (`result/p0_*`, `result/thesis_supplementary_v1/`, and archived study paths)
# are treated strictly as read-only inputs.

# %%
NOTEBOOK_CODE_SHA256 = '8ef046820a8c9511f46e8afa916923575a7b6450c49315b5fc3e92c1c1b979e2'

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

# Unified color and marker system across every chart
COLOR_V2 = '#205780'        # Deep blue for nominal MIL+GC (V2)
COLOR_V3 = '#BF3D31'        # Vermilion for ordinal CLM (V3)
COLOR_G3 = '#C0392B'        # Focal Grade 3 crimson
COLOR_W1 = '#1F618D'        # Within-1 slate-blue
COLOR_PASS = '#27AE60'      # Shaded practical-margin region
COLOR_GRAY = '#566573'

GRADE_COLORS: Dict[int, str] = {
    0: '#999999',
    1: '#75AEDA',
    2: '#EAA32D',
    3: COLOR_G3,
    4: '#6C3483',
}
GRADE_STYLES: Dict[int, str] = {0: '--', 1: '-.', 2: ':', 3: '-', 4: (0, (3, 1, 1, 1))}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.8,
    'axes.titlesize': 9.8,
    'axes.titleweight': 'normal',
    'axes.labelsize': 9.0,
    'xtick.labelsize': 8.3,
    'ytick.labelsize': 8.3,
    'legend.fontsize': 8.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})


def find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from start/cwd/script to find repo root containing pyproject.toml."""
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
    """Resolve read-only experiment/supplementary inputs and isolated figure output bundle."""
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
        out = (root / 'result' / 'thesis_figures_v1').resolve()
    else:
        out = (Path.cwd() / 'thesis_figures_v1').resolve()

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
    for axis in fig.axes:
        axis.tick_params(width=.65, length=3)
        for spine in axis.spines.values():
            spine.set_linewidth(.65)
        for label in axis.texts:
            label.set_fontsize(max(8, label.get_fontsize()))
            if label.get_bbox_patch() is not None:
                label.get_bbox_patch().set_visible(False)
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
    """Compute float64 cumulative-link cloglog probabilities from latent grid s and cutpoints theta."""
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

    # Compute equal-cohort mean separately and mark clearly
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

    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    ax.axvline(0.0, color='#333333', linestyle='--', linewidth=1.0, zorder=1)

    y_positions = [7, 6, 5, 4, 3, 2, 0.5]  # Gap + divider before equal-cohort mean
    order_cohorts = list(COHORTS) + ['equal_cohort_mean']
    y_map = dict(zip(order_cohorts, y_positions))

    for idx_c, c in enumerate(order_cohorts):
        y = y_map[c]
        r_g3 = float(df_out[(df_out['cohort'] == c) & (df_out['metric'] == 'recall_g3')]['delta_pp'].iloc[0])
        r_w1 = float(df_out[(df_out['cohort'] == c) & (df_out['metric'] == 'within1')]['delta_pp'].iloc[0])
        is_mean = (c == 'equal_cohort_mean')
        lw = 2.0 if is_mean else 1.2
        ms = 8.5 if is_mean else 6.8
        ax.plot([r_g3, r_w1], [y, y], color='#BDC3C7' if not is_mean else '#566573', linewidth=lw, zorder=2)

        ax.scatter([r_g3], [y], color=COLOR_G3, marker='D' if is_mean else 'o', s=ms**2,
                   edgecolor='white', linewidth=0.7, zorder=4,
                   label='Grade 3 recall ($V_3 - V_2$)' if idx_c == 0 else None)
        ax.scatter([r_w1], [y], color=COLOR_W1, marker='P' if is_mean else 's', s=ms**2,
                   edgecolor='white', linewidth=0.7, zorder=4,
                   label='Within-1 accuracy ($V_3 - V_2$)' if idx_c == 0 else None)

        # Direct non-overlapping labels on left/right sides of dots
        left_val, right_val = (r_g3, r_w1) if r_g3 <= r_w1 else (r_w1, r_g3)
        ax.text(left_val - 0.65, y, f'{left_val:+.2f}', va='center', ha='right', fontsize=8.0,
                fontweight='bold' if is_mean else 'normal', color=COLOR_G3 if left_val == r_g3 else COLOR_W1)
        ax.text(right_val + 0.65, y, f'{right_val:+.2f}', va='center', ha='left', fontsize=8.0,
                fontweight='bold' if is_mean else 'normal', color=COLOR_W1 if right_val == r_w1 else COLOR_G3)

    ax.axhline(1.25, color='#AAB7B8', linestyle=':', linewidth=1.0)
    ax.axhspan(-0.15, 1.25, color='#F4F6F7', zorder=0)
    ax.set_yticks(y_positions)
    ylabels = [COHORT_LABELS[c] for c in COHORTS] + ['Equal-cohort mean (1/6)']
    ax.set_yticklabels(ylabels)
    ax.get_yticklabels()[-1].set_fontweight('bold')

    ax.set_xlim(-32.2, 8.2)
    ax.set_ylim(-0.15, 7.65)
    ax.set_xlabel(r'Archived target change, $100 \times (V_3 - V_2)$ (percentage points)')
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    ax.legend(loc='lower center', bbox_to_anchor=(0.48, 1.02), ncol=2, frameon=False,
              facecolor='white', edgecolor='#D5D8DC', framealpha=0.95)
    info = _save_chart(fig, ctx.output_dir, 'figures/chapter05/tradeoff', df_out)
    info['means_pp'] = {r['metric']: r['delta_pp'] for r in mean_rows}
    return info


# %% [markdown]
# ## Chart 2: IDRiD $V_3$ Latent Severity & Decision Regions + 6-Fold Appendix Companion

# %%
def build_chart_2_latent_regions(ctx: FigureContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fold_dir = ctx.e0_dir / 'lodo' / 'units' / 'V3_fold_1_idrid' / 'fold_report'
    grid_df = pd.read_csv(fold_dir / 'decision_grid_float64_diagnostic.csv')
    iv_df = pd.read_csv(fold_dir / 'decision_intervals.csv')
    sel = json.loads((fold_dir / 'selected_params.json').read_text(encoding='utf-8'))
    p0_df = pd.read_csv(ctx.p0_dir / 'lodo' / 'V3_fold_1_idrid' / 'target' / 'predictions.csv')
    d_folds = pd.read_csv(ctx.supp_dir / 'analysis_d_v3_latent_regions_per_fold.csv')
    idrid_d = d_folds[d_folds['cohort'] == 'idrid'].iloc[0]

    # Checkpoint consistency verification
    theta = np.array([float(p0_df[f'theta{k}'].iloc[0]) for k in range(4)], dtype=np.float64)
    if np.max(np.abs(theta - np.array([idrid_d[f'theta{k}'] for k in range(4)]))) > 1e-6:
        raise ValueError('Theta mismatch between IDRiD P0 predictions and Analysis D summary.')

    s_grid = grid_df['s'].to_numpy(dtype=np.float64)
    probs_grid = _clm_probs(s_grid, theta)
    idx_peak = int(np.argmax(probs_grid[:, 3]))
    peak_s_grid = float(s_grid[idx_peak])
    peak_p3_grid = float(probs_grid[idx_peak, 3])
    obs_max_p3 = float(p0_df['p3'].max())
    obs_max_s = float(p0_df.loc[p0_df['p3'].idxmax(), 's'])

    nat_row = iv_df[(iv_df['rule'] == 'native_replay') & (iv_df['grade'] == 3)].iloc[0]
    cut_row = iv_df[(iv_df['rule'] == 'cutpoint_latent') & (iv_df['grade'] == 3)].iloc[0]
    cibd_row = iv_df[(iv_df['rule'] == 'cibd_selected') & (iv_df['grade'] == 3)].iloc[0]

    ov_left = max(float(nat_row['left']), float(cut_row['left']))
    ov_right = min(float(nat_row['right']), float(cut_row['right']))
    ov_len = max(0.0, ov_right - ov_left)
    cut_len = float(cut_row['right']) - float(cut_row['left'])
    ov_frac_pct = 100.0 * ov_len / cut_len

    # Save exact data used in main figure (empirical scores + interval metadata summary)
    df_scores = p0_df[['image_key', 'y_true', 's', 'p3', 'pred_native']].copy()
    df_scores['cohort'] = 'idrid'
    df_scores['fold_id'] = 1
    df_scores['variant'] = 'V3'
    df_scores['split'] = 'target'
    df_scores['peak_p3_grid'] = peak_p3_grid
    df_scores['peak_s_grid'] = peak_s_grid
    df_scores['obs_max_p3'] = obs_max_p3
    df_scores['overlap_pct_of_cutpoint'] = ov_frac_pct

    xlim = (-4, 5)
    fig, (density, prob, bands) = plt.subplots(
        3, 1, sharex=True, figsize=(6.25, 5.1),
        gridspec_kw={'height_ratios': [0.85, 1.45, 1], 'hspace': .20})
    visible = (s_grid >= xlim[0]) & (s_grid <= xlim[1])
    for grade, color in GRADE_COLORS.items():
        scores = p0_df.loc[p0_df.y_true == grade, 's'].to_numpy()
        density.plot(s_grid[visible], gaussian_kde(scores, bw_method=.35)(s_grid[visible]),
                     color=color, lw=1.6, label=f'Grade {grade}')
        prob.plot(s_grid[visible], probs_grid[visible, grade], color=color, lw=1.7)
    density.legend(ncol=5, loc='lower center', bbox_to_anchor=(.5, 1.03), frameon=False)
    density.set_ylabel('Density')
    prob.set_ylabel('Class probability')
    prob.set_ylim(0, 1.04)
    prob.annotate(fr'max $p_3$ = {peak_p3_grid:.2f}', (peak_s_grid, peak_p3_grid),
                  xytext=(20, 9), textcoords='offset points', color=COLOR_G3)
    rules = [('Native', 'native_replay'), ('Median', 'posterior_median'),
             ('Argmax', 'argmax'), ('Cutpoint', 'cutpoint_latent'), ('CIBD', 'cibd_selected')]
    for y, (name, rule) in enumerate(rules):
        for row in iv_df[iv_df.rule == rule].itertuples():
            grade = int(row.grade)
            bands.barh(y, row.right-row.left, left=row.left, height=.69,
                       color=GRADE_COLORS[grade], alpha=1 if grade == 3 else .30)
        g3 = iv_df[(iv_df.rule == rule) & (iv_df.grade == 3)]
        intervals = ', '.join(f'[{v.left:.2f}, {v.right:.2f}]' for v in g3.itertuples())
        bands.text(1.01, y, intervals, transform=bands.get_yaxis_transform(), va='center', fontsize=8)
    bands.text(1.01, 1.03, 'Grade 3 interval', transform=bands.transAxes, fontsize=8)
    bands.set_yticks(range(len(rules)), [name for name, rule in rules])
    bands.invert_yaxis()
    for i,t in enumerate(theta):
        for ax in (prob,bands):
            ax.axvline(t, lw=.7, ls=':', color='#555555')
        prob.text(t, 1.01, rf'$\theta_{i}$', transform=prob.get_xaxis_transform(), ha='center')
    bands.set_xlim(*xlim)
    bands.set_xlabel('Latent severity score')
    fig.subplots_adjust(right=.80, left=.15, bottom=.10, top=.89)
    info_main = _save_chart(fig, ctx.output_dir, 'figures/chapter05/latent-regions', df_scores)
    info_main.update({
        'peak_p3_grid': peak_p3_grid,
        'obs_max_p3': obs_max_p3,
        'overlap_pct_of_cutpoint': ov_frac_pct,
        'n_images': int(len(p0_df)),
    })

    # Appendix Companion: 6-fold V3 latent region representativeness
    df_d_out = d_folds[['fold_id', 'cohort', 'cohort_label', 'overlap_ratio_grid', 'max_p3_grid', 'max_p3_target_images', 'upward_shift_both_bounds']].copy()
    df_d_out['overlap_pct'] = df_d_out['overlap_ratio_grid'] * 100.0
    df_d_out['variant'] = 'V3'
    df_d_out['split'] = 'target_plus_grid'

    fig_app, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.4, 3.15), gridspec_kw={'wspace': 0.14})
    y_idx = np.arange(len(COHORTS))[::-1]
    for pos, (_, row) in zip(y_idx, df_d_out.sort_values('fold_id').iterrows()):
        is_idrid = (row['cohort'] == 'idrid')
        col = COLOR_G3 if is_idrid else COLOR_V3
        ms = 8.0 if is_idrid else 6.2
        ax1.scatter([row['overlap_pct']], [pos], color=col, s=ms**2, marker='D' if is_idrid else 'o',
                    edgecolor='white', linewidth=0.7, zorder=4)
        ax1.text(row['overlap_pct'] + 1.4, pos, f"{row['overlap_pct']:.1f}%", va='center', ha='left',
                 fontsize=8.0, fontweight='bold' if is_idrid else 'normal', color=col)

        ax2.scatter([row['max_p3_grid']], [pos], color=col, s=ms**2, marker='D' if is_idrid else 'o',
                    edgecolor='white', linewidth=0.7, zorder=4)
        ax2.scatter([row['max_p3_target_images']], [pos], color='#2C3E50', s=28, marker='x', zorder=5)
        ax2.text(row['max_p3_grid'] + 0.009, pos, f"{row['max_p3_grid']:.3f}", va='center', ha='left',
                 fontsize=8.0, fontweight='bold' if is_idrid else 'normal', color=col)

    ax1.set_yticks(y_idx)
    ax1.set_yticklabels([COHORT_LABELS[c] for c in COHORTS])
    ax1.get_yticklabels()[1].set_fontweight('bold')
    ax1.set_xlim(15, 76)
    ax1.set_xlabel('Native–cutpoint overlap (% raw width)')
    ax1.set_title('Decision-interval overlap', fontsize=9.0, pad=18)
    ax1.grid(axis='x', linestyle=':', alpha=0.35)

    ax2.set_xlim(0.20, 0.56)
    ax2.set_xlabel('Peak Grade 3 probability ($\\max p_3$)')
    ax2.set_title('Peak $p_3$ (grid vs. observed)', fontsize=9.0, pad=18)
    ax2.grid(axis='x', linestyle=':', alpha=0.35)
    app_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V3, markersize=6.8, label='Diagnostic grid $\\max p_3$'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_G3, markersize=6.8, label='IDRiD worked example'),
        Line2D([0], [0], marker='x', color='#2C3E50', linestyle='None', markersize=6.5, label='Observed target image $\\max p_3$'),
    ]
    fig_app.legend(handles=app_handles, loc='lower center', bbox_to_anchor=(0.52, 0.99),
                   ncol=3, fontsize=7.4, frameon=False, facecolor='white', edgecolor='#D5D8DC')
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

    # Main Chapter 5 compact 4-value paired-dot chart
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    y_pos = {'V2': 1.0, 'V3': 0.0}
    offsets = {'source_val': 0.14, 'target': -0.14}
    for var, color in [('V2', COLOR_V2), ('V3', COLOR_V3)]:
        v_src = float(means[(means['variant'] == var) & (means['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
        v_tgt = float(means[(means['variant'] == var) & (means['split'] == 'target')]['recall_g3_pct'].iloc[0])
        yc = y_pos[var]
        ax.plot([v_src, v_tgt], [yc + offsets['source_val'], yc + offsets['target']],
                color='#7F8C8D', linewidth=1.5, zorder=2)
        ax.scatter([v_src], [yc + offsets['source_val']], color=color, marker='o', s=68,
                   edgecolor='white', linewidth=0.8, zorder=4,
                   label='Source validation (fold mixture)' if var == 'V2' else None)
        ax.scatter([v_tgt], [yc + offsets['target']], color=color, marker='D', s=64,
                   facecolor='white', edgecolor=color, linewidth=1.8, zorder=4,
                   label='Held-out target' if var == 'V2' else None)
        ax.text(v_src + 1.1, yc + offsets['source_val'], f'Source: {v_src:.1f}%', va='center', ha='left', fontsize=8.3, color=color, fontweight='bold')
        ax.text(v_tgt + 1.1, yc + offsets['target'], f'Target: {v_tgt:.1f}%', va='center', ha='left', fontsize=8.3, color=color)

    # Gap bracket annotation
    v2_s = float(means[(means['variant'] == 'V2') & (means['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
    v3_s = float(means[(means['variant'] == 'V3') & (means['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
    ax.annotate(f'Source gap: {v3_s - v2_s:+.1f} pp', xy=(0.5 * (v2_s + v3_s), 0.55),
                xytext=(48.0, 0.52), ha='center', va='center', fontsize=8.0,
                bbox=dict(boxstyle='round,pad=0.22', fc='#F8F9F9', ec='#BDC3C7'))

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(['$V_2$ (Nominal, native)', '$V_3$ (Ordinal, native)'])
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlim(22, 76)
    ax.set_xlabel('Mean Grade 3 recall across six folds (%)')
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#34495E', markersize=7.5, label='Source validation (5-cohort fold mixture)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#34495E', markeredgewidth=1.6, markersize=7.0, label='Held-out target cohort'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.48, 1.03), ncol=2,
              frameon=False, facecolor='white', edgecolor='#D5D8DC', fontsize=7.8)
    info_main = _save_chart(fig, ctx.output_dir, 'figures/chapter05/source-target-g3', means)
    info_main['means'] = means.to_dict(orient='records')

    # Appendix companion: fold-level source-mixture vs target per fold
    fig_app, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.5, 3.45), gridspec_kw={'wspace': 0.10})
    y_idx = np.arange(len(COHORTS))[::-1]
    for ax, var, col, ttl in [(axes[0], 'V2', COLOR_V2, '$V_2$ (Nominal, native replay)'),
                              (axes[1], 'V3', COLOR_V3, '$V_3$ (Ordinal, native replay)')]:
        sub_v = nat[nat['variant'] == var]
        for pos, (fold, c) in zip(y_idx, enumerate(COHORTS)):
            vs = float(sub_v[(sub_v['fold_id'] == fold) & (sub_v['split'] == 'source_val')]['recall_g3_pct'].iloc[0])
            vt = float(sub_v[(sub_v['fold_id'] == fold) & (sub_v['split'] == 'target')]['recall_g3_pct'].iloc[0])
            # Within-fold comparison segment (explicitly NOT patient tracking)
            ax.plot([vs, vt], [pos, pos], color='#95A5A6', linewidth=1.2, zorder=2)
            ax.scatter([vs], [pos], color=col, marker='o', s=46, edgecolor='white', linewidth=0.6, zorder=4)
            ax.scatter([vt], [pos], color=col, marker='D', s=42, facecolor='white', edgecolor=col, linewidth=1.5, zorder=4)
        ax.set_title(ttl, fontsize=9.0, pad=18)
        ax.set_xlabel('Grade 3 recall (%)')
        ax.grid(axis='x', linestyle=':', alpha=0.35)
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels([f'{COHORT_LABELS[c]} fold' for c in COHORTS])
    axes[0].set_xlim(2, 78)
    st_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#34495E', markersize=6.8, label='Source-val mixture (5 source cohorts)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='#34495E', markeredgewidth=1.5, markersize=6.5, label='Held-out target cohort'),
    ]
    fig_app.legend(handles=st_handles, loc='lower center', bbox_to_anchor=(0.52, 0.99), ncol=2,
                   fontsize=7.6, frameon=False, facecolor='white', edgecolor='#D5D8DC')
    df_fold_out = nat[['fold_id', 'cohort', 'cohort_label', 'variant', 'split', 'decoder', 'recall_g3_pct', 'units']].sort_values(['variant', 'fold_id', 'split'])
    df_fold_out['aggregation'] = 'per_fold_source_mixture_vs_heldout_target'
    info_app = _save_chart(fig_app, ctx.output_dir, 'figures/appendix/source-target-folds', df_fold_out)
    return info_main, info_app


# %% [markdown]
# ## Chart 4: Cohort-Level Effects of CIBD (`figures/chapter05/decoding.pdf`) + Post-Hoc Appendix Panel

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
            'meets_margin_g3_3pp': bool(dx >= 3.0),
            'meets_margin_w1_minus1pp': bool(dy >= -1.0),
            'meets_both_margins': bool(dx >= 3.0 and dy >= -1.0),
            'units': 'percentage_points', 'aggregation': 'equal_cohort_mean_6',
        })
    df_all = pd.concat([df_c, pd.DataFrame(mean_rows)], ignore_index=True)

    # Main 12-point chart (V2 + V3 primary E0)
    fig, ax = plt.subplots(figsize=(6.6, 4.55))
    xlim = (-9.5, 32.8)
    ylim = (-3.05, 2.75)
    # Lightly shade ONLY the practical-margin region x >= +3 pp, y >= -1 pp
    ax.add_patch(mpatches.Rectangle((3.0, -1.0), xlim[1] - 3.0, ylim[1] - (-1.0),
                                    facecolor='#D5F5E3', alpha=0.45, zorder=0))
    ax.axhline(0.0, color='#99A3A4', linewidth=0.9, zorder=1)
    ax.axvline(0.0, color='#99A3A4', linewidth=0.9, zorder=1)
    ax.axvline(3.0, color='#1E8449', linestyle='--', linewidth=1.1, zorder=2)
    ax.axhline(-1.0, color='#1E8449', linestyle='--', linewidth=1.1, zorder=2)

    # Custom non-overlapping offsets tailored per (variant, cohort)
    lbl_offsets: Dict[Tuple[str, str], Tuple[float, float, str, str]] = {
        ('V2', 'aptos'): (-0.65, -0.10, 'right', 'top'),
        ('V2', 'idrid'): (-0.65, 0.08, 'right', 'bottom'),
        ('V2', 'messidor2'): (-0.60, 0.04, 'right', 'center'),
        ('V2', 'ddr'): (0.55, 0.10, 'left', 'bottom'),
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
            ax.scatter([x], [y], color=col, marker=mkr, s=48, edgecolor='white', linewidth=0.7, zorder=4)
            dx_off, dy_off, ha, va = lbl_offsets[(var, str(r['cohort']))]
            ax.text(x + dx_off, y + dy_off, f"{COHORT_SHORT[str(r['cohort'])]}", fontsize=7.6,
                    ha=ha, va=va, color=col)
        m_row = df_all[(df_all['variant'] == var) & (df_all['cohort'] == 'equal_cohort_mean')].iloc[0]
        mx, my = float(m_row['delta_g3_recall_pp']), float(m_row['delta_within1_pp'])
        ax.scatter([mx], [my], color=col, marker='*', s=175, edgecolor='#1C2833', linewidth=0.8, zorder=5)
        tx, ty = (14.2, 2.05) if var == 'V2' else (17.5, -2.25)
        ax.annotate(
            f'{var} mean ({mx:+.2f}, {my:+.2f})',
            xy=(mx, my), xytext=(tx, ty), ha='left', va='center',
            fontsize=7.8, fontweight='bold', color=col,
            arrowprops=dict(arrowstyle='->', color=col, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=col, alpha=0.92),
            zorder=6,
        )

    ax.text(32.1, 2.58, 'Satisfies both margins\n($\\Delta\\mathrm{G3}\\geq+3$ pp, $\\Delta\\mathrm{W1}\\geq-1$ pp)',
            ha='right', va='top', fontsize=7.6, color='#145A32')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'Target Grade 3 recall change, $100 \times (\mathrm{G3}_{\mathrm{CIBD}} - \mathrm{G3}_{\mathrm{native}})$ (pp)')
    ax.set_ylabel(r'Target Within-1 change, $100 \times (\mathrm{W1}_{\mathrm{CIBD}} - \mathrm{W1}_{\mathrm{native}})$ (pp)')
    ax.grid(linestyle=':', alpha=0.3)

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_V2, markersize=6.8, label='$V_2$ cohort (4/6 meet both; 6/6 > 0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V3, markersize=7.0, label='$V_3$ cohort (3/6 meet both)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#566573', markeredgecolor='#1C2833', markersize=10.0, label='Equal-cohort mean (1/6)'),
    ]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=3,
              frameon=False, facecolor='white', edgecolor='#D5D8DC', fontsize=7.4)
    info_main = _save_chart(fig, ctx.output_dir, 'figures/chapter05/decoding', df_all)
    info_main['v2_met'] = int(df_c[df_c['variant'] == 'V2']['meets_both_margins'].sum())
    info_main['v3_met'] = int(df_c[df_c['variant'] == 'V3']['meets_both_margins'].sum())

    # Appendix companion: V3 primary 1.0 pp vs post hoc 0.5 pp budget
    dg = pd.read_csv(ctx.supp_dir / 'analysis_g_cibd_05pp_per_fold.csv')
    dg3 = dg[dg['variant'] == 'V3'].copy()
    fig_app, ax_a = plt.subplots(figsize=(6.4, 4.25))
    xlim_a, ylim_a = (-8.5, 25.5), (-3.0, 1.35)
    ax_a.add_patch(mpatches.Rectangle((3.0, -1.0), xlim_a[1] - 3.0, ylim_a[1] - (-1.0),
                                      facecolor='#D5F5E3', alpha=0.45, zorder=0))
    ax_a.axhline(0.0, color='#99A3A4', linewidth=0.9, zorder=1)
    ax_a.axvline(0.0, color='#99A3A4', linewidth=0.9, zorder=1)
    ax_a.axvline(3.0, color='#1E8449', linestyle='--', linewidth=1.1, zorder=2)
    ax_a.axhline(-1.0, color='#1E8449', linestyle='--', linewidth=1.1, zorder=2)

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
        ax_a.scatter([x1], [y1], color='white', edgecolor=COLOR_V3, marker='o', s=52, linewidth=1.6, zorder=4)
        ax_a.scatter([x2], [y2], color=COLOR_V3, edgecolor='white', marker='o', s=58, linewidth=0.7, zorder=5)
        dx, dy, ha, va = app_lbl_pos[c]
        ax_a.text(x2 + dx, y2 + dy, COHORT_SHORT[c], fontsize=7.8, color='#1C2833', ha=ha, va=va)

    # Equal-cohort means for 1.0 pp vs post hoc 0.5 pp
    m10 = dg3[dg3['rule'] == 'cibd_1.0pp_budget'][['delta_g3_recall_pp', 'delta_within1_pp']].mean()
    m05 = dg3[dg3['rule'] == 'cibd_0.5pp_budget'][['delta_g3_recall_pp', 'delta_within1_pp']].mean()
    ax_a.scatter([m10['delta_g3_recall_pp']], [m10['delta_within1_pp']], color='white', edgecolor='#1C2833', marker='*', s=145, linewidth=1.3, zorder=6)
    ax_a.scatter([m05['delta_g3_recall_pp']], [m05['delta_within1_pp']], color='#1C2833', edgecolor='white', marker='*', s=160, linewidth=0.7, zorder=6)
    ax_a.annotate('', xy=(m05['delta_g3_recall_pp'], m05['delta_within1_pp']), xytext=(m10['delta_g3_recall_pp'], m10['delta_within1_pp']),
                  arrowprops=dict(arrowstyle='->', color='#1C2833', lw=1.4, ls='--'), zorder=5)

    handles_a = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=COLOR_V3, markeredgewidth=1.6, markersize=6.8, label='Primary $V_3$ (1.0 pp; 3/6 meet margins)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_V3, markersize=7.0, label='Post hoc $V_3$ (0.5 pp; 4/6 meet margins)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#1C2833', markersize=9.5, label=f"Mean: ({m10['delta_g3_recall_pp']:+.2f}, {m10['delta_within1_pp']:+.2f}) -> ({m05['delta_g3_recall_pp']:+.2f}, {m05['delta_within1_pp']:+.2f})"),
    ]
    ax_a.legend(handles=handles_a, loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=2,
                frameon=False, facecolor='white', edgecolor='#D5D8DC', fontsize=7.5)
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
    fig, (ax_r, ax_p) = plt.subplots(1, 2, sharey=True, figsize=(6.6, 3.65), gridspec_kw={'wspace': 0.10})
    y_map = {rk: 4 - i for i, (rk, _, _, _) in enumerate(row_order)}

    # Consistent slight vertical jitter per cohort across BOTH panels so cohorts stay aligned
    jitter = np.linspace(-0.16, 0.16, len(COHORTS))
    jit_map = dict(zip(COHORTS, jitter))

    for rk, _, col, is_native in row_order:
        y0 = y_map[rk]
        sub = df_p[df_p['row_key'] == rk].sort_values('fold_id')
        for _, r in sub.iterrows():
            c = str(r['cohort'])
            yj = y0 + jit_map[c]
            ax_r.scatter([r['recall_g3_pct']], [yj], color=COHORT_COLORS[c], marker=COHORT_MARKERS[c],
                         s=28, alpha=0.82, edgecolor='white', linewidth=0.4, zorder=3)
            ax_p.scatter([r['g3_precision_pct']], [yj], color=COHORT_COLORS[c], marker=COHORT_MARKERS[c],
                         s=28, alpha=0.82, edgecolor='white', linewidth=0.4, zorder=3)

        mr = float(sub['recall_g3_pct'].mean())
        mp = float(sub['g3_precision_pct'].mean())
        ax_r.scatter([mr], [y0], color=col if not is_native else 'white', edgecolor=col,
                     marker='D', s=75, linewidth=1.8, zorder=5)
        ax_p.scatter([mp], [y0], color=col if not is_native else 'white', edgecolor=col,
                     marker='D', s=75, linewidth=1.8, zorder=5)
        # Offset mean labels above each row's diamond to prevent collision with cohort dots
        ax_r.text(mr, y0 + 0.25, f'{mr:.2f}%', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=col)
        ax_p.text(mp, y0 + 0.25, f'{mp:.2f}%', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=col)

    ax_r.axhline(2.5, color='#CBD5E1', linestyle=':', linewidth=1.0)
    ax_p.axhline(2.5, color='#CBD5E1', linestyle=':', linewidth=1.0)
    ax_r.set_yticks([y_map[r[0]] for r in row_order])
    ax_r.set_yticklabels([r[1] for r in row_order])
    ax_r.set_ylim(-0.55, 4.65)
    ax_r.set_xlim(3, 82)
    ax_p.set_xlim(3, 66)
    ax_r.set_xlabel('Grade 3 recall (%)')
    ax_p.set_xlabel('Grade 3 precision (%)')
    ax_r.set_title('Target Grade 3 recall', fontsize=9.2)
    ax_p.set_title('Target Grade 3 precision', fontsize=9.2)
    ax_r.grid(axis='x', linestyle=':', alpha=0.35)
    ax_p.grid(axis='x', linestyle=':', alpha=0.35)

    # Cohort legend + mean legend at bottom
    cohort_handles = [
        Line2D([0], [0], marker=COHORT_MARKERS[c], color='w', markerfacecolor=COHORT_COLORS[c],
               markersize=5.8, label=COHORT_SHORT[c])
        for c in COHORTS
    ] + [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#34495E', markersize=6.8, label='Equal-cohort mean')
    ]
    fig.legend(handles=cohort_handles, loc='upper center', ncol=4, bbox_to_anchor=(0.52, 0.00),
               frameon=False, facecolor='white', edgecolor='#D5D8DC', fontsize=7.6)
    return _save_chart(fig, ctx.output_dir, 'figures/chapter05/probing', df_p)
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
    # Deduplicated common-decoder E0 operating points from Analysis A
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

    # Primary E1 probes cross-checked against e1 equal_cohort_means.csv
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

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    native = df_op.query("model == 'V3' and rule == 'rounded_expectation'").iloc[0]
    xb, yb = native.within1_pct-1, native.recall_g3_pct+3
    ax.set(xlim=(88.55,93.15), ylim=(28,57.2), xlabel='Within-1 accuracy (%)',
           ylabel='Grade 3 recall (%)')
    ax.fill_between([xb,93.15], yb, 57.2, color='#EDF4EE', zorder=0)
    ax.axvline(xb, ls='--', lw=.85, color='#588260')
    ax.axhline(yb, ls='--', lw=.85, color='#588260')
    ax.text(93.06, 56.7, 'Reference margins\nrelative to native V₃', ha='right', va='top', color='#446E4D', fontsize=8)
    names = {'argmax':'Argmax','posterior_median':'Median','rounded_expectation':'Rounded mean',
             'cutpoint_latent':'Latent cutpoint','cibd_selected':'CIBD',
             'probe_z_v2':'Probe $z_2$','probe_z_v3':'Probe $z_3$','probe_s_v3':'Probe $s_3$'}
    for row in df_op.itertuples():
        color = COLOR_V2 if row.model=='V2' else COLOR_V3
        if row.within1_pct < ax.get_xlim()[0]:
            ax.text(.02,.035,f'{row.model} argmax outside view ({row.within1_pct:.2f}, {row.recall_g3_pct:.2f})',
                    transform=ax.transAxes, color=color, fontsize=8)
            continue
        probe = row.rule.startswith('probe_')
        ax.scatter(row.within1_pct,row.recall_g3_pct,s=32,marker='s' if probe else 'o',color=color,zorder=3)
        name=names[row.rule]
        if row.method_family=='Native': name=f'Native (= {name.lower()})'
        left = row.rule in ('probe_s_v3','probe_z_v3') or (row.model=='V2' and row.rule in ('argmax','posterior_median'))
        ax.annotate(name,(row.within1_pct,row.recall_g3_pct),xytext=(-6 if left else 5,-9 if row.method_family=='Native' else 3),
                    textcoords='offset points',ha='right' if left else 'left',va='center',color=color,fontsize=8)
    ax.legend(handles=[Line2D([],[],ls='',marker=m,color=c,label=l) for m,c,l in
              [('o',COLOR_V2,'V₂ decoders'),('o',COLOR_V3,'V₃ decoders'),('s','#666666','Linear probes')]],
              loc='upper left',frameon=False)
    ax.grid(color='#dddddd',lw=.5,alpha=.65)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save_chart(fig, ctx.output_dir, 'figures/chapter05/operating-points', df_op)


# %% [markdown]
# ## Documentation & Manifest Generation (`FIGURE_INTEGRATION.md`, `figure_manifest.json`)

# %%
def write_integration_guide(ctx: FigureContext, meta: Dict[str, Any]) -> Path:
    """Write Overleaf integration guide covering every generated chart plus localization reuse note."""
    c1 = meta['charts']['tradeoff']
    c2 = meta['charts']['latent_regions']
    c3 = meta['charts']['source_target_g3']
    c4 = meta['charts']['decoding']
    c5 = meta['charts']['probing']

    loc_note = str(ctx.localization_asset) if ctx.localization_asset else 'MISSING: verify result/figure_8_xai_qualitative_grid.pdf'

    lines = [
        '# Figure Integration Guide (`result/thesis_figures_v1/`)',
        '',
        'Copy the `figures/` tree from this bundle directly into the Overleaf thesis project root. Draw.io placeholders (`figures/chapter03/architecture.pdf`, `figures/chapter04/evaluation.pdf`) were intentionally left untouched.',
        '',
        '## Summary & Discrepancy Notes vs. `UEL_Thesis_official_v1.zip`',
        '',
        '1. **Chart 3 (`source-target-g3.pdf`)**: `chapters/05_results_discussion.tex` currently states the four native Grade 3 recall means (`61.9%`, `30.9%`, `43.6%`, `32.2%`) only in prose (§5.6.2 line 295 and §5.10.1 line 467), with no compact summary table. Therefore **both** the compact Chapter 5 paired-dot chart (`figures/chapter05/source-target-g3.pdf`) and the fold-level Appendix D companion (`figures/appendix/source-target-folds.pdf`) were generated.',
        '2. **Chart 4 (`decoding.pdf`) vs. existing `chapters/05_results_discussion.tex` / `FIGURES_vi.md`**: The existing ZIP block (`subsec:5-8-4`, `\\label{fig:decoding}`) describes only $V_3$ (`6` cohorts) and `FIGURES_vi.md` swaps x/y (`X=delta Within-1, Y=delta G3 recall`). Per supervisor feedback §7 and specification, the generated `decoding.pdf` plots **all 12 $V_2$ and $V_3$ cohort points** with **$x = \\Delta$ Grade 3 recall (pp)** and **$y = \\Delta$ Within-1 accuracy (pp)**, practical margins at $x=+3$~pp and $y=-1$~pp, and equal-cohort means. Update the LaTeX caption accordingly using the copy-ready caption below.',
        f"3. **Localization reuse asset**: Original figure verified at `{loc_note}`; copy/rename it to `figures/chapter05/localization.pdf` in Overleaf.",
        '',
        '---',
        '',
        '## Chapter 5 Quantitative Figures',
        '',
        '### 1. Archived Architectural Trade-off (`figures/chapter05/tradeoff.pdf`)',
        f"- **Source CSV**: `{ctx.p0_dir.name}/g0_comparison.csv` (`split=target`, `metric in {{recall_g3, within1}}`, `reference` column) -> `{c1['csv']}`",
        '- **Section Title**: `5.3.3 Ordinal Modeling and the Severity Trade-off` (`\\label{subsec:5-3-3}`)',
        '- **Manuscript Block Status**: Existing `\\IfFileExists{figures/chapter05/tradeoff.pdf}` block in `chapters/05_results_discussion.tex` (lines 114–120)',
        '- **Insertion Anchor**: Immediately after `\\label{subsec:5-3-3}` heading',
        '- **LaTeX Label**: `\\label{fig:tradeoff}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:tradeoff} compares the cohort-level changes in Grade~3 recall and Within-1 accuracy from the archived nominal configuration ($V_2$) to the ordinal cumulative-link configuration ($V_3$).`',
        '- **Self-Contained Caption**: `Archived target changes from $V_2$ (nominal) to $V_3$ (ordinal CLM) across six held-out cohorts and their equal-cohort mean ($1/6$), in percentage points ($100 \\times (V_3 - V_2)$). Exact Grade~3 recall drops in every cohort ($-11.98$~pp mean) while Within-1 accuracy improves in five of six cohorts ($+1.46$~pp mean, declining only on DeepDRiD).`',
        '',
        '### 2. Latent Severity and Grade 3 Decision Regions (`figures/chapter05/latent-regions.pdf`)',
        f"- **Source Files**: `{ctx.e0_dir.name}/lodo/units/V3_fold_1_idrid/fold_report/decision_grid_float64_diagnostic.csv`, `decision_intervals.csv`, `selected_params.json`, `{ctx.p0_dir.name}/lodo/V3_fold_1_idrid/target/predictions.csv`, verified against `analysis_d_v3_latent_regions_per_fold.csv` -> `{c2['csv']}`",
        '- **Section Title**: `5.7.2 Decision Regions and Grade 3 Error Redistribution` (`\\label{subsec:5-7-2}`)',
        '- **Manuscript Block Status**: **New** figure block (wrap with `\\IfFileExists{figures/chapter05/latent-regions.pdf}{...}{}`)',
        '- **Insertion Anchor**: First paragraph of `\\subsection{Decision Regions and Grade 3 Error Redistribution}` (`\\label{subsec:5-7-2}`)',
        '- **LaTeX Label**: `\\label{fig:latent-regions}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:latent-regions} aligns the empirical IDRiD score distributions, cumulative-link class probabilities, and resulting Grade~3 decision bands on a shared latent severity axis.`',
        f"- **Self-Contained Caption**: `Latent severity mechanism for the IDRiD-held-out $V_3$ checkpoint ($n={c2['n_images']}$ target images; horizontal view shown over $[-4, 5]$ with Grade~4 extending to $7.06$). Top: within-grade normalized empirical target score densities. Middle: cumulative-link class probabilities showing cutpoints $\\theta_0=-1.504$, $\\theta_1=-0.468$, $\\theta_2=0.605$, $\\theta_3=1.552$ and peak diagnostic-grid $p_3={c2['peak_p3_grid']:.3f}$ (observed image maximum ${c2['obs_max_p3']:.3f}$). Bottom: Grade~3 prediction intervals under source-locked CIBD ($\\alpha=0.5, \\tau=1.0, q=2$), raw latent cutpoints, median, argmax and native rounded expectation; the native rule overlaps {c2['overlap_pct_of_cutpoint']:.1f}\\% of the raw-cutpoint interval width.`",
        '',
        '### 3. Native Grade 3 Recall on Source and Target (`figures/chapter05/source-target-g3.pdf`)',
        f"- **Source CSV**: `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`decoder=native_replay`, `variant in {{V2, V3}}`, `split in {{source_val, target}}`) -> `{c3['csv']}`",
        '- **Section Title**: `5.6.2 Source--Target Comparison and Latent Severity Distributions` (`\\label{subsec:5-6-2}`)',
        '- **Manuscript Block Status**: **New** compact figure block (placed at §5.6.2 since Chapter 5 currently states these four means only in prose)',
        '- **Insertion Anchor**: End of paragraph 1 of `\\subsection{Source--Target Comparison and Latent Severity Distributions}` (`\\label{subsec:5-6-2}`)',
        '- **LaTeX Label**: `\\label{fig:source-target-g3}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:source-target-g3} summarizes the equal-cohort mean Grade~3 recall of replayed $V_2$ and $V_3$ checkpoints on source validation and held-out targets.`',
        '- **Self-Contained Caption**: `Equal-cohort six-fold mean Grade~3 recall (%) for replayed native $V_2$ (argmax) and $V_3$ (rounded expectation) checkpoints on source validation ($61.9\\%$ vs.\\ $30.9\\%$) and held-out target cohorts ($43.6\\%$ vs.\\ $32.2\\%$). The $-31.0$~pp ordinal deficit already present on source validation indicates the trade-off does not arise solely from cross-domain transfer.`',
        '',
        '### 4. Cohort-Level Effects of CIBD (`figures/chapter05/decoding.pdf`)',
        f"- **Source CSV**: `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`split=target`, `decoder in {{native_replay, cibd_selected}}`, `V2` and `V3`) -> `{c4['csv']}`",
        '- **Section Title**: `5.7.6 Cost-interpolated Bayes decoding within E0` — `Cohort Robustness and Limits of Decision Intervention` (`\\label{sec:5-8-4}`)',
        '- **Manuscript Block Status**: **Existing conditional block** (`\\IfFileExists{figures/chapter05/decoding.pdf}`) in `chapters/05_results_discussion.tex` (lines 409–415) — **caption update needed** to reflect both $V_2$ and $V_3$ points and updated x/y axes',
        '- **Insertion Anchor**: Beginning of `\\subsubsection{Cohort Robustness and Limits of Decision Intervention}` (`chapters/05_results_discussion.tex`, line 409)',
        '- **LaTeX Label**: `\\label{fig:decoding}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:decoding} plots the paired target changes in Grade~3 recall and Within-1 accuracy under primary source-selected CIBD relative to replayed native decoding across all 12 model--cohort combinations.`',
        '- **Self-Contained Caption**: `Target changes from replayed native decoding to primary source-selected CIBD (1.0~pp source budget) across all six held-out cohorts for $V_2$ (squares; equal-cohort mean $+10.10$~pp Grade~3 recall, $+0.99$~pp Within-1) and $V_3$ (circles; $+8.85$~pp, $-1.16$~pp). Dashed lines mark the prespecified practical margins ($x = +3$~pp Grade~3 recall, $y = -1$~pp Within-1 accuracy), with the jointly satisfied region lightly shaded; 4 of 6 $V_2$ cohorts (Messidor-2 and DDR fall below $+3$~pp recall despite positive changes in both metrics) and 3 of 6 $V_3$ cohorts satisfy both margins.`',
        '',
        '### 5. Frozen Probe Recall and Precision (`figures/chapter05/probing.pdf`)',
        f"- **Source CSVs**: `{ctx.e1_dir.name}/probe_metrics.csv` (`split=target`, `control_seed` empty, primary probes) paired with `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`native_replay`) -> `{c5['csv']}`",
        '- **Section Title**: `5.8.1 Linear Accessibility in Frozen Features and Scalar Scores` (`\\label{subsec:5-9-1}`)',
        '- **Manuscript Block Status**: **Existing conditional block** (`\\IfFileExists{figures/chapter05/probing.pdf}`) in `chapters/05_results_discussion.tex` (lines 424–430)',
        '- **Insertion Anchor**: Immediately after `\\label{subsec:5-9-1}` (`chapters/05_results_discussion.tex`, line 424)',
        '- **LaTeX Label**: `\\label{fig:probing}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:probing} compares target Grade~3 recall and precision across all six held-out cohorts for replayed native classifiers and source-trained E1 linear probes.`',
        '- **Self-Contained Caption**: `Target Grade~3 recall (left) and Grade~3 precision (right) across six held-out cohorts (small markers) and their equal-cohort means (diamonds) for replayed native classifiers and primary E1 probes selected by source-validation macro recall. The $s_3$ scalar probe raises recall in every cohort ($54.09\\%$ mean vs.\\ $32.22\\%$ native $V_3$, $51.09\\%$ $\\mathbf{z}_3$, and $44.64\\%$ $\\mathbf{z}_2$), while six-cohort mean precision remains comparable across $V_3$ rules ($31.41\\%$ vs.\\ $31.71\\%$ native) amid wide cohort variation ($9.4\\%$--$59.4\\%$).`',
        '',
        '### 6. Integrated Operating-Point Map (`figures/chapter05/operating-points.pdf`)',
        f"- **Source CSVs**: `thesis_supplementary_v1/analysis_a_summary.csv` and `{ctx.e1_dir.name}/equal_cohort_means.csv` -> `{meta['charts']['operating_points']['csv']}`",
        '- **Section Title**: `5.9.1 Joint Diagnostic Evidence` (`\\label{subsec:5-10-1}`)',
        '- **Manuscript Block Status**: **Existing commented block** (`% \\begin{figure}...`) in `chapters/05_results_discussion.tex` (lines 487–491) — uncomment and wrap with `\\IfFileExists` or insert directly',
        '- **Insertion Anchor**: End of `\\subsection{Joint Diagnostic Evidence}` (`chapters/05_results_discussion.tex`, line 487)',
        '- **LaTeX Label**: `\\label{fig:operating-points}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:operating-points} synthesizes the equal-cohort target performance of conventional E0 rules, primary source-selected CIBD, and primary E1 probes on shared Within-1 and Grade~3 recall axes.`',
        '- **Self-Contained Caption**: `Equal-cohort six-fold target means for replayed $V_2$ (blue) and $V_3$ (orange-red) decision rules and source-trained E1 probes. Coincident rules are deduplicated ($V_2$ native coincides with argmax; $V_3$ native coincides with rounded expectation), and latent-cutpoint decoding applies only to $V_3$ CLM. Markers denote rule families rather than a continuous training trajectory; the scalar $s_3$ probe was fitted on source labels in E1 outside the prespecified E0 decoder test.`',
        '',
        '---',
        '',
        '## Appendix Companions & Localization Reuse Note',
        '',
        '### A1. Six-Fold Latent Region Representativeness (`figures/appendix/latent-regions-folds.pdf`)',
        '- **Source CSV**: `thesis_supplementary_v1/analysis_d_v3_latent_regions_per_fold.csv`',
        '- **Section / Anchor**: Appendix G (`\\label{app:derivations}`), after `\\section{Cumulative probabilities and latent boundaries}` (or Appendix E end)',
        '- **LaTeX Label**: `\\label{fig:app-latent-regions-folds}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:app-latent-regions-folds} compares the native--cutpoint Grade~3 interval overlap and peak Grade~3 probability across all six $V_3$ folds to contextualize the IDRiD example.`',
        '- **Self-Contained Caption**: `Six-fold $V_3$ decision-interval overlap fraction (left, percent of raw latent-cutpoint width) and peak Grade~3 probability (right, diagnostic grid circles vs.\\ observed target-image maxima crosses). IDRiD (diamonds; $46.8\\%$ overlap, grid $\\max p_3=0.336$) lies near the center of the cross-fold range ($22.2\\%$--$63.0\\%$ overlap; $\\max p_3$ $0.243$--$0.482$).`',
        '',
        '### A2. Fold-Level Source vs. Target Native Grade 3 Recall (`figures/appendix/source-target-folds.pdf`)',
        f"- **Source CSV**: `{ctx.e0_dir.name}/lodo/decision_metrics.csv` (`decoder=native_replay`, `V2` & `V3`, `source_val` vs. `target`)",
        '- **Section / Anchor**: Appendix D (`\\label{app:reproduction}`), end of `\\section{Scope of the replay evidence}` after `Table~\\ref{tab:ch5-p0}`',
        '- **LaTeX Label**: `\\label{fig:app-source-target-folds}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:app-source-target-folds} details the per-fold native Grade~3 recall on each fold\'s five-cohort source-validation mixture alongside its held-out target cohort.`',
        '- **Self-Contained Caption**: `Per-fold Grade~3 recall (%) for replayed native $V_2$ (left) and $V_3$ (right) comparing the five-cohort source-validation mixture of each fold (filled circles) with the corresponding held-out target cohort (open diamonds). Connecting segments denote within-fold comparisons only and do not follow the same patients or images across domains.`',
        '',
        '### A3. Post Hoc 0.5 pp vs. Primary 1.0 pp CIBD Sensitivity (`figures/appendix/decoding-05pp.pdf`)',
        '- **Source CSV**: `thesis_supplementary_v1/analysis_g_cibd_05pp_per_fold.csv` (`variant=V3`)',
        '- **Section / Anchor**: Appendix E (`\\label{app:decoder_results}`), end of `\\section{Parameter selection and transfer}`',
        '- **LaTeX Label**: `\\label{fig:app-decoding-05pp}`',
        '- **Introductory Sentence**: `Figure~\\ref{fig:app-decoding-05pp} contrasts the primary 1.0~pp source-budget $V_3$ CIBD operating points with the post hoc 0.5~pp sensitivity selections across all six held-out cohorts.`',
        '- **Self-Contained Caption**: `Post hoc sensitivity comparison of $V_3$ CIBD target changes under the primary 1.0~pp source-validation Within-1 budget (open circles) versus a stricter post hoc 0.5~pp budget (filled circles), connected within each cohort. Under the post hoc 0.5~pp budget, 4 of 6 cohorts (APTOS~2019, IDRiD, Messidor-2, and DeepDRiD) meet both practical target margins ($+6.56$~pp mean Grade~3 recall, $-0.71$~pp Within-1).`',
        '',
        '### R1. Localization Grid Reuse Note (`figures/chapter05/localization.pdf`)',
        f"- **Verified Original Asset**: `{loc_note}`",
        '- **Section / Anchor**: `5.5.2 Localization Evidence and Failure Cases` (`\\label{subsec:5-5-2}`), immediately after `Table~\\ref{tab:ch5-localization}` (`chapters/05_results_discussion.tex`, lines 269–273)',
        '- **Manuscript Block Status**: Existing commented block (`% \\begin{figure}...\\label{fig:localization-grid}`) — copy verified asset to `figures/chapter05/localization.pdf` in Overleaf, uncomment, and replace caption.',
        '- **Introductory Sentence**: `Figure~\\ref{fig:localization-grid} illustrates five representative pointing-game failure cases drawn from the 81-image IDRiD lesion-localization evaluation.`',
        '- **Self-Contained Caption**: `Qualitative saliency examples from the archived IDRiD lesion-localization evaluation (81 matched crops) showing local gated attention and global Grad-CAM overlays for five selected pointing-game misses \\citep{nguyen2026clinically}. These illustrative failure cases highlight misalignment between top saliency responses and annotated lesion masks rather than the full 81-image distribution.`',
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
    p = argparse.ArgumentParser(description='Generate quantitative thesis figures and Overleaf bundle (CPU-only).')
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
