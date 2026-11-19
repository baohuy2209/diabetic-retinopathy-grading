"""Publication-ready scientific figure generation suite for diabetic retinopathy evaluation study.

Adheres to strict academic visual standards (IEEE / Springer / Nature):
- Publication typography (clean sans-serif).
- Standard journal figure widths (single-column: 3.5 in, double-column: 7.0 in).
- Clean palettes (navy, crimson, amber, emerald, slate).
- High resolution raster (300 DPI) and vector (PDF) exports.
- Includes Figure 2 through Figure 8 publication suite matching 05_component_analysis_lodo.ipynb.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from paths import FIGURES_DIR, LODO_RESULTS_DIR, SUMMARY_JSON_PATH



# ==============================================================================
# PUBLICATION COLOR PALETTES
# ==============================================================================
C_NAVY     = "#1f4e79"
C_GREEN    = "#2e7d32"
C_ORANGE   = "#e65100"
C_RED      = "#c53030"
C_CRIMSON  = "#c53030"
C_SLATE    = "#4a5568"
C_GRAY     = "#94a3b8"
C_DARKGRAY = "#475569"
C_TEAL     = "#0d9488"
C_CHAMP    = "#008080"
C_EMERALD  = "#16a34a"
C_AMBER    = "#d97706"
C_PURPLE   = "#6b21a8"

PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "accent": "#d62728",
    "slate": "#7f7f7f",
    "dark": "#1b2838",
    "light": "#f4f6f8",
    "heads": {
        "Softmax_CE": "#7f7f7f",
        "Softmax_QWK": "#1f77b4",
        "CORAL": "#2ca02c",
        "CORN": "#ff7f0e",
        "CLM_QWK": "#d62728",
    }
}

def configure_publication_style():
    """
    Sets up cohesive typography, axis weights, subtle grids, and high-resolution
    export parameters suitable for IEEE/Springer journal publication figures.
    """
    plt.rcParams.update({
        # --- Typography & Font Hierarchy ---
        # Cross-platform sans-serif fallbacks with proportional font sizes
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.family': 'sans-serif',
        'font.size': 10.5,
        'axes.labelsize': 11.5,
        'axes.titlesize': 12.5,
        'xtick.labelsize': 10.0,
        'ytick.labelsize': 10.0,
        'legend.fontsize': 10.0,
        'figure.titlesize': 14.0,

        # --- Axis Spines & Grid Styling ---
        # Dark-slate borders with non-distracting dashed gridlines
        'axes.linewidth': 1.1,
        'axes.edgecolor': '#2d3748',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.45,
        'grid.linestyle': '--',
        'grid.color': '#a0aec0',

        # --- Export & Rendering Quality ---
        # 300 DPI high-resolution output with automatic padding clipping
        'mathtext.fontset': 'dejavusans',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08
    })

    # Base clean background theme
    sns.set_theme(style="whitegrid")

# Apply publication style globally
configure_publication_style()

# Academic Color Palette Definitions (IEEE / Springer Guidelines)
C_NAVY     = "#1f4e79"  # In-Domain Validation / Baseline Primary
C_ORANGE   = "#e65100"  # Grade 1 (H1 Microaneurysm) / Accent
C_CHAMP    = "#008080"  # Proposed Champion Model (Teal)
C_TEAL     = "#0d9488"  # Alternative Teal Accent
C_GREEN    = "#2e7d32"  # Success / Accuracy / High Recall
C_RED      = "#c53030"  # Error / Violation / Severe Drop
C_CRIMSON  = "#c53030"  # Grade 3 (H2 Severe NPDR)
C_AMBER    = "#d97706"  # Warning / Intermediate Stage
C_PURPLE   = "#6b21a8"  # Monotonic cutpoints theta of CLM (H3)
C_SLATE    = "#4a5568"  # Neutral / Secondary
C_GRAY     = "#94a3b8"  # Muted baseline / Classes G0, G2, G4
C_DARKGRAY = "#475569"
C_EMERALD  = "#16a34a"


# ======================================================================


def save_fig_dual(fig: plt.Figure, basename: str, out_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
    """Helper to save figure as both 300-DPI PNG and vector PDF."""
    target_dir = Path(out_dir) if out_dir is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    png_path = target_dir / f"{basename}.png"
    pdf_path = target_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def save_figure(fig: plt.Figure, out_path: Union[str, Path], dpi: int = 300) -> None:
    """Save figure to both PDF and high-res PNG."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

# CELL 31
# ======================================================================
def plot_fig2_dataset_distribution(dataset_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Renders Fig. 2: Dataset Composition & Natural Label Shift across 6 Cohorts from real master_df."""
    configure_publication_style()
    df = pd.crosstab(dataset_df["dataset_key"], dataset_df["diagnosis"])
    df.columns = [f"Grade {c}" for c in df.columns]
    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.2, 1.0]})

    # (a) Total Sample Volume per Source
    total_counts = df.sum(axis=1)
    bars = ax1.barh(df.index, total_counts, color=C_NAVY, edgecolor="#0f172a", lw=0.9, alpha=0.88)
    ax1.set_xlabel("Total Number of Images (Log Scale)", fontweight='bold')
    ax1.set_xscale("log")
    ax1.set_title(f"(a) Multi-Center Cohort Sizes ({len(dataset_df):,} Total Images)", fontsize=11.5, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax1.text(w * 1.15, bar.get_y() + bar.get_height()/2, f"{int(w):,}",
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#1e293b")

    # (b) Stacked Proportions (ICDRS 0-4)
    grade_colors = ["#3b82f6", C_ORANGE, "#eab308", C_CRIMSON, "#7e22ce"]
    bottom = np.zeros(len(df_pct))
    for i, col in enumerate(df_pct.columns):
        vals = df_pct[col].values
        ax2.barh(df_pct.index, vals, left=bottom, label=col, color=grade_colors[i], edgecolor="#1e293b", lw=0.6, alpha=0.9)
        bottom += vals

    ax2.set_xlabel("Class Distribution (%)", fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_title("(b) Clinical Label Shift Across Cohorts (ICDRS 0-4)", fontsize=11.5, fontweight='bold')
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", title="Clinical Severity")

    plt.suptitle("Figure 2: Multi-Center Dataset Composition and Prior Label Shift ($P(Y)$)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ======================================================================
# CELL 32
# ======================================================================
# ==============================================================================
# 7.2 FIG 3 (PER-GRADE SLOPE CHART - CENTRAL THESIS) & FIG 4 (WATERFALL DECOMPOSITION)
# ==============================================================================
def plot_fig3_slope_chart_per_grade_recall(
    in_recalls: Optional[Dict[Any, float]] = None,
    ood_recalls: Optional[Dict[Any, float]] = None,
    lodo_results: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders Fig. 3: Slope chart per-grade recall (In-Domain -> Out-of-Domain).
    Central Thesis:
      - Grade 1 and Grade 3 drop steeply under domain shift.
      - Grades 0, 2, 4 remain relatively flat.
      - Emphasis styling: ONLY G1 (Orange) and G3 (Crimson) are colored; G0, G2, G4 are muted gray.
    Supports either explicit per-grade recall dictionaries or extracts 6-fold LODO means from lodo_results.
    """
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # If in_recalls/ood_recalls not provided, attempt extraction from lodo_results
    if (in_recalls is None or ood_recalls is None) and lodo_results is not None:
        target_dict = lodo_results
        if any(k in ["V3_PlusOrdinalHead_H3_Champion", "Champion"] for k in lodo_results):
            target_dict = lodo_results.get("V3_PlusOrdinalHead_H3_Champion", lodo_results.get("Champion"))
        elif any(isinstance(v, dict) and any(dk in DATASET_KEYS for dk in v.keys()) for v in lodo_results.values()):
            target_dict = next(v for v in lodo_results.values() if isinstance(v, dict) and any(dk in DATASET_KEYS for dk in v.keys()))

        extracted_in = {g: [] for g in range(5)}
        extracted_ood = {g: [] for g in range(5)}
        for fold_key, fold_entry in target_dict.items():
            if isinstance(fold_entry, dict):
                val_rec = fold_entry.get("val_per_grade_recall", {})
                ood_rec = fold_entry.get("ood_per_grade_recall", {})
                for g in range(5):
                    extracted_in[g].append(float(val_rec[f"grade_{g}"]))
                    extracted_ood[g].append(float(ood_rec[f"grade_{g}"]))

        if any(extracted_in.values()):
            in_recalls = {
                g: float(np.nanmean(extracted_in[g]))
                if len(extracted_in[g]) > 0 else np.nan
                for g in range(5)
            }

            ood_recalls = {
                g: float(np.nanmean(extracted_ood[g]))
                if len(extracted_ood[g]) > 0 else np.nan
                for g in range(5)
            }

    # Normalize recall dictionaries to integer keys 0..4
    def _read_grade_recall(recalls, grade):
        if not recalls:
            return np.nan

        candidate_keys = [
            grade,
            str(grade),
            f"grade_{grade}",
            f"Grade_{grade}",
        ]

        for key in candidate_keys:
            if key in recalls:
                value = recalls[key]
                if value is None:
                    return np.nan
                return float(value)

        return np.nan


    clean_in = {}
    clean_ood = {}

    for g in range(5):
        clean_in[g] = _read_grade_recall(in_recalls, g)
        clean_ood[g] = _read_grade_recall(ood_recalls, g)

    x_in, x_ood = 0.0, 1.0
    grade_meta = {
        0: ("Grade 0 (No DR)", C_GRAY, 1.8, "--"),
        1: ("Grade 1 (Mild NPDR - Microaneurysms)", C_ORANGE, 3.4, "-"),
        2: ("Grade 2 (Moderate NPDR)", C_GRAY, 1.8, "--"),
        3: ("Grade 3 (Severe NPDR - 4-2-1 Rule)", C_CRIMSON, 3.4, "-"),
        4: ("Grade 4 (Proliferative DR)", C_GRAY, 1.8, "--"),
    }

    ax.axvline(x_in, color="#cbd5e1", lw=1.5, zorder=1)
    ax.axvline(x_ood, color="#cbd5e1", lw=1.5, zorder=1)

    for g in range(5):
        y1 = clean_in[g]
        y2 = clean_ood[g]
        delta = y2 - y1
        name, col, lw, ls = grade_meta[g]
        is_emphasis = g in [1, 3]

        ax.plot([x_in, x_ood], [y1, y2], color=col, lw=lw, linestyle=ls, zorder=4 if is_emphasis else 2)
        ax.scatter([x_in, x_ood], [y1, y2], color=col, s=80 if is_emphasis else 45,
                   edgecolor="#1e293b", lw=1.0, zorder=5 if is_emphasis else 3)

        ax.text(x_in - 0.03, y1, f"G{g}: {y1*100:.1f}%", ha="right", va="center",
                fontsize=10 if is_emphasis else 9, fontweight="bold" if is_emphasis else "normal",
                color="#0f172a" if is_emphasis else C_DARKGRAY)

        delta_str = f"({delta*100:+.1f}%)"
        ax.text(x_ood + 0.03, y2, f"G{g}: {y2*100:.1f}% {delta_str}", ha="left", va="center",
                fontsize=10 if is_emphasis else 9, fontweight="bold" if is_emphasis else "normal",
                color=col if is_emphasis else C_DARKGRAY)

    ax.set_xlim(-0.35, 1.45)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([x_in, x_ood])
    ax.set_xticklabels(["In-Domain (Internal Val)", "Out-of-Domain (External Center)"],
                       fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Per-Grade Sensitivity / Recall", fontsize=11.5, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.set_title(
        "Per-Grade Recall Across LODO Folds\n"
        "Internal Validation vs Held-Out OOD Cohorts",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def extract_ablation_ood_qwk_summary(
    lodo_results: Dict[str, Any],
    summary_json_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts the empirical mean and per-fold Out-of-Domain QWK (QWK ngoại miền)
    for each architecture variant in the ablation waterfall (V0 -> V1 -> V2 -> V3).
    Strictly reads experimental metrics without dummy or synthetic numbers.
    """
    if not lodo_results and summary_json_path and os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, "r") as f:
                lodo_results = json.load(f)
        except Exception as e:
            print(f"[WARN] Unable to load summary_json for ablation extraction: {e}")

    summary = {}
    if not lodo_results:
        return summary

    variant_slots = [
        ("V0_Baseline_Softmax_GlobalOnly", ["V0", "Baseline", "GlobalOnly"]),
        ("V1_PlusLocalMIL_H1", ["V1", "LocalMIL", "Local_MIL", "H1"]),
        ("V2_PlusGlobalContext_H2", ["V2", "GlobalContext", "NonLocal", "Quadrant", "H2"]),
        ("V3_PlusOrdinalHead_H3_Champion", ["V3", "Champion", "CLM", "Ordinal", "H3"]),
    ]

    for canonical_name, aliases in variant_slots:
        matched_dict = None
        matched_key = None

        if canonical_name in lodo_results:
            matched_dict = lodo_results[canonical_name]
            matched_key = canonical_name
        else:
            for k in lodo_results.keys():
                if any(alias.lower() in k.lower() for alias in aliases):
                    matched_dict = lodo_results[k]
                    matched_key = k
                    break

        if matched_dict is None and "V3" in canonical_name:
            if any(k in DATASET_KEYS for k in lodo_results.keys()):
                matched_dict = lodo_results
                matched_key = "V3_PlusOrdinalHead_H3_Champion"

        if matched_dict is not None and isinstance(matched_dict, dict):
            fold_qwks = []
            for fold_key, fold_entry in matched_dict.items():
                if isinstance(fold_entry, dict):
                    for q_field in ["ood_test_qwk", "ood_qwk", "test_qwk", "QWK", "qwk"]:
                        if q_field in fold_entry and fold_entry[q_field] is not None:
                            try:
                                fold_qwks.append(float(fold_entry[q_field]))
                                break
                            except (ValueError, TypeError):
                                pass

            if fold_qwks:
                summary[canonical_name] = {
                    "matched_key": matched_key,
                    "mean_ood_qwk": float(np.mean(fold_qwks)),
                    "std_ood_qwk": float(np.std(fold_qwks)),
                    "num_folds": len(fold_qwks),
                    "fold_qwks": fold_qwks
                }

    return summary


def plot_fig4_waterfall_qwk_decomposition(
    lodo_results: Optional[Dict[str, Any]] = None,
    baseline_qwk: Optional[float] = None,
    components: Optional[List[Tuple[str, float]]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders Fig. 4: Waterfall Chart Decomposing Out-of-Domain QWK Gain by Component.
    Sequence: Baseline (V0) -> +Native MIL (H1) -> +Global Attention (H2) -> +Ordinal CLM (H3) -> Champion (V3).
    Strictly derives all values from real empirical Out-of-Domain QWK (QWK ngoại miền) metrics across 6 LODO folds.
    """
    configure_publication_style()

    summary_path = globals().get("SUMMARY_JSON_PATH", None)
    emp_summary = extract_ablation_ood_qwk_summary(lodo_results or {}, summary_json_path=summary_path)

    v0_meta = emp_summary.get("V0_Baseline_Softmax_GlobalOnly")
    v1_meta = emp_summary.get("V1_PlusLocalMIL_H1")
    v2_meta = emp_summary.get("V2_PlusGlobalContext_H2")
    v3_meta = emp_summary.get("V3_PlusOrdinalHead_H3_Champion")

    if v0_meta and v1_meta and v2_meta and v3_meta:
        # Full empirical 4-variant waterfall derived directly from experimental results
        q0 = v0_meta["mean_ood_qwk"]
        q1 = v1_meta["mean_ood_qwk"]
        q2 = v2_meta["mean_ood_qwk"]
        q3 = v3_meta["mean_ood_qwk"]

        baseline_qwk = q0
        components = [
            ("+Native MIL (H1)", q1 - q0),
            ("+Global Attention (H2)", q2 - q1),
            ("+Ordinal CLM (H3)", q3 - q2)
        ]
        print(f"[Fig. 4] Computed Waterfall from Empirical LODO Results (6 Folds Mean OOD QWK):")
        print(f"    - V0 Baseline Global Only : {q0:.4f} (+/- {v0_meta['std_ood_qwk']:.4f})")
        print(f"    - V1 +Native MIL (H1)     : {q1:.4f} (Gain: {q1 - q0:+.4f})")
        print(f"    - V2 +Global Context (H2) : {q2:.4f} (Gain: {q2 - q1:+.4f})")
        print(f"    - V3 +Ordinal CLM (H3)    : {q3:.4f} (Gain: {q3 - q2:+.4f})")
        print(f"    - Total Champion OOD QWK  : {q3:.4f}")
    elif len(emp_summary) >= 2:
        # Partial empirical waterfall with available variants
        sorted_keys = sorted(emp_summary.keys())
        first_k = sorted_keys[0]
        baseline_qwk = emp_summary[first_k]["mean_ood_qwk"]
        components = []
        prev_q = baseline_qwk
        for k in sorted_keys[1:]:
            cur_q = emp_summary[k]["mean_ood_qwk"]
            label = f"+{k.replace('V1_', '').replace('V2_', '').replace('V3_', '')}"
            components.append((label, cur_q - prev_q))
            prev_q = cur_q
        print(f"[Fig. 4] Computed Partial Waterfall across {len(emp_summary)} available variants from empirical LODO results.")
    elif baseline_qwk is not None and components is not None:
        # Use explicitly passed arguments
        pass

    labels = ["Vanilla Baseline (V0)"] + [c[0] for c in components] + ["Proposed Champion (V3)"]
    running_total = baseline_qwk
    bottoms = [0.0]
    heights = [baseline_qwk]
    colors = [C_NAVY]

    for label, delta in components:
        if delta >= 0:
            bottoms.append(running_total)
            heights.append(delta)
            colors.append(C_EMERALD)
        else:
            bottoms.append(running_total + delta)
            heights.append(abs(delta))
            colors.append(C_CRIMSON)
        running_total += delta

    bottoms.append(0.0)
    heights.append(running_total)
    colors.append(C_TEAL)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(labels))
    width = 0.55

    bars = ax.bar(x, heights, width, bottom=bottoms, color=colors, edgecolor="#1e293b", lw=1.0, alpha=0.92)

    for i in range(len(components)):
        step_y = bottoms[i+1] + heights[i+1] if components[i][1] >= 0 else bottoms[i+1]
        ax.plot([x[i] + width/2, x[i+1] - width/2], [bottoms[i+1], bottoms[i+1]],
                color="#64748b", linestyle=":", lw=1.4)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        b = bar.get_y()
        val_str = f"{h:.3f}" if (i == 0 or i == len(bars) - 1) else f"{components[i-1][1]:+.3f}"
        y_pos = b + h/2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, val_str,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Out-of-Domain Quadratic Weighted Kappa (QWK)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.set_title("Figure 4: Waterfall Decomposition of Out-of-Domain QWK Gain by Component\n"
                 "Empirical Resolution of Hypotheses $H_1$, $H_2$, and $H_3$ across 6-Fold LODO Benchmark",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ======================================================================
# CELL 33
# ======================================================================
# ==============================================================================
# 7.3 FIG 5 (LODO HEATMAP), FIG 6 (LATENT AXIS), FIG 7 (ROC), & FIG 8 (XAI GRID)
# ==============================================================================
def _lodo_number(value):
    return np.nan if value is None else float(value)


def get_lodo_calibration_row(fold_entry, target_specificity=0.95):
    for row in fold_entry.get("ood_calibrated_screening", []):
        target = row.get("Target_Specificity")
        if isinstance(target, str):
            target = float(target.rstrip("%")) / 100.0 if target.endswith("%") else float(target)
        if target is not None and np.isclose(float(target), target_specificity):
            return row
    return {}


def plot_fig5_lodo_heatmap(results_dict: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    configure_publication_style()
    target_dict = results_dict
    if any(k in ["V3_PlusOrdinalHead_H3_Champion", "Champion"] for k in results_dict):
        target_dict = results_dict.get("V3_PlusOrdinalHead_H3_Champion", results_dict.get("Champion"))
    elif any(isinstance(v, dict) and any(dk in DATASET_KEYS for dk in v) for v in results_dict.values()):
        target_dict = next(v for v in results_dict.values()
                           if isinstance(v, dict) and any(dk in DATASET_KEYS for dk in v))
    keys = [k for k in target_dict if isinstance(target_dict[k], dict)]
    if not keys:
        raise ValueError("No fold results for Fig.5")
    names = [target_dict[k].get("held_out_dataset", DATASET_CONFIG.get(k, {}).get("name", k)) for k in keys]
    calibration = [get_lodo_calibration_row(target_dict[k]) for k in keys]
    data = {
        "In-Domain Val QWK": [_lodo_number(target_dict[k].get("in_domain_val_qwk")) for k in keys],
        "OOD Test QWK": [_lodo_number(target_dict[k].get("ood_test_qwk")) for k in keys],
        "Delta QWK Drop": [_lodo_number(target_dict[k].get("delta_lodo")) for k in keys],
        "OOD Within-1 Acc (%)": [100 * _lodo_number(target_dict[k].get("ood_within_1_grade_acc")) for k in keys],
        "OOD Sensitivity at Val tau (%)": [100 * _lodo_number(r.get("Test_Sensitivity")) for r in calibration],
        "OOD Specificity at same tau (%)": [100 * _lodo_number(r.get("Test_Specificity")) for r in calibration],
    }
    df = pd.DataFrame(data, index=names).T
    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    sns.heatmap(df, annot=True, fmt=".3f", mask=df.isna(), cmap="YlGnBu",
                cbar_kws={"label": "Metric Value"}, linewidths=1.2, linecolor="#1e293b", ax=ax)
    ax.set_title("LODO Generalization Matrix\n"
                 "Screening tau chosen on validation at target specificity 95%; OOD specificity may differ",
                 fontsize=11, fontweight="bold", pad=15)
    ax.set_ylabel("Evaluated Metric")
    ax.set_xlabel("Held-Out External Test Cohort")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fig6_latent_severity_axis(
    in_scores: Dict[int, np.ndarray],
    ood_scores: Dict[int, np.ndarray],
    cutpoints: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders Fig. 6: 1D Continuous Latent Severity Score Axis s(x) of CLM + Learned Cutpoints theta.
    Key Features:
      - Demonstrates continuous disease severity spectrum.
      - Overlays Out-of-Domain score distribution with dashed lines to visibly expose Domain Shift.
    """
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grade_colors = ["#3b82f6", C_ORANGE, "#eab308", C_CRIMSON, "#7e22ce"]

    for g in range(5):
        if len(in_scores.get(g, [])) > 1:
            sns.kdeplot(in_scores[g], ax=ax, color=grade_colors[g], fill=True, alpha=0.35, lw=2.0, label=f"Grade {g} (In-Domain)")
        if len(ood_scores.get(g, [])) > 1:
            sns.kdeplot(ood_scores[g], ax=ax, color=grade_colors[g], linestyle="--", lw=1.8, alpha=0.85, label=f"Grade {g} (Out-of-Domain)" if g == 0 else "")

    for i, theta in enumerate(cutpoints):
        ax.axvline(theta, color=C_PURPLE, linestyle=":", lw=2.2, zorder=5)
        ax.text(theta, ax.get_ylim()[1] * 0.90, f"$\theta_{i+1} = {theta:.2f}$",
                ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=C_PURPLE,
                bbox=dict(boxstyle="round,pad=0.2", fc="#faf5ff", ec=C_PURPLE, lw=1.0))

    ax.set_xlabel(r"1D Latent Severity Score: $s(\mathbf{x}) = \mathbf{w}^T \mathbf{z}$", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Sample Density", fontsize=11.5, fontweight="bold")
    ax.set_title("Figure 6: 1D Latent Severity Spectrum of Cumulative Link Model (CLM) & Learned Cutpoints $\theta$\n"
                 "Overlaid Dashed Curves Visibly Capture Out-of-Domain Distributional Shift",
                 fontsize=12, fontweight="bold", pad=15)

    handles = [
        Patch(facecolor="#3b82f6", alpha=0.5, edgecolor="#1e293b", label="In-Domain Solid Distributions"),
        plt.Line2D([0], [0], color="#3b82f6", linestyle="--", lw=2, label="Out-of-Domain Dashed (Domain Shift)"),
        plt.Line2D([0], [0], color=C_PURPLE, linestyle=":", lw=2, label="Monotonic Cutpoints $\theta_1 < \theta_2 < \theta_3 < \theta_4$")
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fig7_referable_dr_roc(
    in_fpr: np.ndarray, in_tpr: np.ndarray, in_auc: float,
    ood_fpr: np.ndarray, ood_tpr: np.ndarray, ood_auc: float,
    save_path: Optional[str] = None,
    calibration_row: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    if np.isfinite(in_auc):
        ax.plot(in_fpr, in_tpr, color=C_NAVY, lw=2.5,
                label=f"In-Domain Val (AUC = {in_auc:.3f})")
    if np.isfinite(ood_auc):
        ax.plot(ood_fpr, ood_tpr, color=C_ORANGE, lw=2.5, linestyle="--",
                label=f"OOD (AUC = {ood_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.2)
    if calibration_row:
        for prefix, label, color in (("Val", "Val", C_NAVY), ("Test", "OOD", C_ORANGE)):
            sensitivity = _lodo_number(calibration_row.get(f"{prefix}_Sensitivity"))
            specificity = _lodo_number(calibration_row.get(f"{prefix}_Specificity"))
            if np.isfinite(sensitivity) and np.isfinite(specificity):
                ax.scatter([1.0 - specificity], [sensitivity], color=color, s=100,
                    marker="X", zorder=5,
                    label=f"{label} at same Val tau: Sens {sensitivity:.1%}, Spec {specificity:.1%}")
    else:
        ax.text(0.03, 0.97, "No calibrated operating point available", transform=ax.transAxes, va="top")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Referable DR ROC\nOperating points use one threshold selected on validation")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fig8_real_xai(
    eval_model: nn.Module,
    test_loader: DataLoader,
    xai_samples: Optional[List[Dict[str, Any]]] = None,
    idrid_xai_dataset: Optional[Dataset] = None,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders Fig. 8: Qualitative XAI Grid grounded against IDRiD Expert Lesion Masks.
    Displays 5-Column High-Resolution Benchmark:
      Col 0: Raw Retinal Fundus Image (ROI Cropped)
      Col 1: Expert Ophthalmologist Ground-Truth Lesion Mask (IDRiD)
      Col 2: Architecture-Native Local Attention-MIL Saliency Map (~128px resolution, H1)
      Col 3: Global Grad-CAM Saliency Map (ResNet-50 Layer 4, H2)
      Col 4: Pathological Saliency Overlay & Pointing Game Peak
    """
    configure_publication_style()
    eval_model.eval()

    if xai_samples and len(xai_samples) > 0:
        n_rows = min(5, len(xai_samples))
        fig, axes = plt.subplots(n_rows, 5, figsize=(16.0, 3.2 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for row in range(n_rows):
            s = xai_samples[row]
            g_t = s["global_img"]
            if isinstance(g_t, torch.Tensor):
                if g_t.dtype == torch.uint8:
                    orig = g_t.permute(1, 2, 0).cpu().numpy()
                else:
                    orig = g_t.permute(1, 2, 0).cpu().numpy()
                    orig = np.clip((orig * std + mean) * 255.0, 0, 255).astype(np.uint8)
            else:
                orig = np.asarray(g_t, dtype=np.uint8)

            mask_gt = s["mask_gt"]
            cam_g = np.nan_to_num(s["cam_global"], nan=0.0, posinf=0.0, neginf=0.0)
            cam_m = np.nan_to_num(s["saliency_mil"], nan=0.0, posinf=0.0, neginf=0.0)

            axes[row, 0].imshow(orig)
            axes[row, 0].set_title(f"Image {s['image_id']}: Preprocessed Fundus", fontweight="bold", fontsize=10)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_gt, cmap="gray")
            axes[row, 1].set_title("IDRiD Expert Lesion Mask", fontweight="bold", fontsize=10, color="#b91c1c")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(cam_m, cmap="inferno")
            m_hit_str = ("Hit" if s["m_hit"] else "Miss") if s.get("m_hit") is not None else "N/A: " + s.get("m_status", "unavailable")
            axes[row, 2].set_title(f"Local Attention-MIL ({m_hit_str})", fontweight="bold", fontsize=10, color="#047857")
            axes[row, 2].axis("off")

            axes[row, 3].imshow(cam_g, cmap="jet")
            g_hit_str = ("Hit" if s["g_hit"] else "Miss") if s.get("g_hit") is not None else "N/A: " + s.get("g_status", "unavailable")
            axes[row, 3].set_title(f"Global Grad-CAM ({g_hit_str})", fontweight="bold", fontsize=10, color="#1d4ed8")
            axes[row, 3].axis("off")

            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_m), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
            axes[row, 4].imshow(overlay if s.get("m_status") == "ok" else orig)
            axes[row, 4].set_title("Saliency Overlay", fontweight="bold", fontsize=10)
            axes[row, 4].axis("off")

        plt.suptitle("Figure 8: Quantitative & Qualitative XAI Verification Grid against IDRiD Ground-Truth Lesion Masks\n"
                     "Local attention and global Grad-CAM; N/A maps provide no localization evidence",
                     fontsize=12.5, fontweight="bold", y=1.01)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # Fallback to test loader samples if no mask samples available
    cam_extractor = FundusGradCAM(eval_model)
    grade_samples = {}
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for batch in test_loader:
        labels = batch["label"].cpu().numpy()
        for i, lbl in enumerate(labels):
            if lbl not in grade_samples and lbl > 0:
                g_tensor = batch["global_img"][i:i+1].to(device)
                l_tensor = batch["local_patches"][i:i+1].to(device)
                g_tensor, l_tensor = normalize_model_inputs(
                    g_tensor,
                    l_tensor,
                    device
                )
                p_mask = batch.get("patch_mask", None)
                p_mask_t = p_mask[i:i+1].to(device) if p_mask is not None else None

                cam, _ = cam_extractor.generate_cam(g_tensor, l_tensor, patch_mask=p_mask_t)
                orig_img = g_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if g_tensor.dtype != torch.uint8:
                    orig_img = np.clip((orig_img * std + mean) * 255.0, 0, 255).astype(np.uint8)

                cam_res = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
                grade_samples[lbl] = (orig_img, cam_res, cam_extractor.last_status)
        if len(grade_samples) >= 4:
            break

    if not grade_samples:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "XAI Demonstration Samples", ha="center", va="center")
        return fig

    sorted_grades = sorted(grade_samples.keys())
    fig, axes = plt.subplots(len(sorted_grades), 3, figsize=(10.5, 3.2 * len(sorted_grades)))
    if len(sorted_grades) == 1:
        axes = np.expand_dims(axes, 0)

    for row, g in enumerate(sorted_grades):
        orig, cam, status = grade_samples[g]
        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"Grade {g}: Preprocessed Fundus Image", fontweight="bold", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title(f"Grade {g}: Grad-CAM ({status})", fontweight="bold", fontsize=10)
        axes[row, 1].axis("off")

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)

        axes[row, 2].imshow(overlay if status == "ok" else orig)
        axes[row, 2].set_title(f"Grade {g}: Saliency Overlay", fontweight="bold", fontsize=10)
        axes[row, 2].axis("off")

    plt.suptitle("Figure 8: Qualitative XAI Grid - Empirical Grad-CAM from Proposed Model (H5)\n"
                 "Qualitative visualization only; lesion alignment is not verified without masks",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ======================================================================
