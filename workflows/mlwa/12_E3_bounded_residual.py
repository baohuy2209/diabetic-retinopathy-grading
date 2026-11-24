# %% [markdown]
# # 12 · E3 Gated Bounded Residual Correction (H3 Gate & Controls)
# 
# > **Standalone Single-File Notebook**: All required modules are embedded directly below. Upload only this `.ipynb` file to Google Colab and run all cells (`Runtime -> Run all`); no repository clone or zip bundle is required.

# %%
NOTEBOOK_CODE_SHA256 = 'a2d343b621e873691b956bc014ec09a588d3a57bad906eed661c2e91e72f734a'

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
        deps = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'pyarrow': 'pyarrow', 'torch': 'torch'}
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')
    return Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _setup_colab_environment()
    MODE = 'gate_check'
    OUTPUT_DIR = None
    SMOKE = True

# %% [markdown]
# ## Embedded Standalone Implementation (8 self-contained modules)

# %% [markdown]
# ### Module `experiments.mlwa.constants`

# %%
"""Canonical constants, status codes, and portable environment resolution for MLWA journal experiments."""


import os
from pathlib import Path
from typing import Dict, Optional, Tuple

COHORT_ORDER: Tuple[str, ...] = ("aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid")
CANONICAL_FOLD_TO_COHORT: Dict[int, str] = {i: c for i, c in enumerate(COHORT_ORDER)}
COHORT_TO_FOLD: Dict[str, int] = {c: i for i, c in enumerate(COHORT_ORDER)}

COHORT_LABELS: Dict[str, str] = {
    "aptos": "APTOS 2019",
    "idrid": "IDRiD",
    "messidor2": "Messidor-2",
    "ddr": "DDR",
    "eyepacs": "EyePACS",
    "deepdrid": "DeepDRiD",
}

PROB_COLS: Tuple[str, ...] = ("p0", "p1", "p2", "p3", "p4")
THETA_COLS: Tuple[str, ...] = ("theta0", "theta1", "theta2", "theta3")
TIE_TOL: float = 1e-12
SCHEMA_VERSION: str = "mlwa_journal_v1"

STATUS_PLANNED = "PLANNED"
STATUS_PREFLIGHT_OK = "PREFLIGHT_OK"
STATUS_PENDING_ASSETS = "PENDING_ASSETS"
STATUS_RUNNING = "RUNNING"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_FAILED = "FAILED"
STATUS_COMPLETE = "COMPLETE"
STATUS_DIAGNOSTIC_REPLAY = "DIAGNOSTIC_REPLAY"
STATUS_SKIPPED_GATE_NOT_MET = "SKIPPED_GATE_NOT_MET"
STATUS_TEST_ONLY = "TEST_ONLY"

STATUS_CODES: Tuple[str, ...] = (
    STATUS_PLANNED,
    STATUS_PREFLIGHT_OK,
    STATUS_PENDING_ASSETS,
    STATUS_RUNNING,
    STATUS_INTERRUPTED,
    STATUS_FAILED,
    STATUS_COMPLETE,
    STATUS_DIAGNOSTIC_REPLAY,
    STATUS_SKIPPED_GATE_NOT_MET,
    STATUS_TEST_ONLY,
)

REQUIRED_PREDICTION_COLUMNS: Tuple[str, ...] = (
    "study_id",
    "protocol_id",
    "run_id",
    "experiment",
    "config_id",
    "architecture_id",
    "head_type",
    "objective_id",
    "clm_link",
    "fold_id",
    "cohort",
    "seed_train",
    "split_role",
    "epoch",
    "checkpoint_sha256",
    "image_key",
    "image_id",
    "patient_key",
    "patient_identity_status",
    "y_true",
    *PROB_COLS,
    "pred_native",
    "s",
    *THETA_COLS,
    "feature_file",
    "feature_row",
    "manifest_sha256",
    "preprocessing_id",
    "schema_version",
    "code_sha256",
    "plan_sha256",
)


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate repository root cleanly without hardcoded external Windows paths."""
    cands = []
    if start is not None:
        cands.append(Path(start).resolve())
    cands.append(Path.cwd().resolve())
    if "__file__" in globals():
        cands.append(Path(__file__).resolve().parents[2])
    for base in cands:
        for cur in (base, *base.parents):
            if (cur / "pyproject.toml").is_file() and (cur / "experiments" / "mlwa").is_dir():
                return cur
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    return Path.cwd().resolve()


def resolve_runtime_paths(repo_root: Optional[Path] = None) -> Dict[str, Path]:
    """Resolve portable DR_* paths for local and Google Colab sessions."""
    root = find_repo_root(repo_root)
    dr_root = Path(os.environ.get("DR_ROOT", str(root))).expanduser().resolve()
    inputs = Path(os.environ.get("DR_INPUTS", str(dr_root / "inputs"))).expanduser().resolve()
    outputs_base = Path(os.environ.get("DR_JOURNAL_OUTPUTS", os.environ.get("DR_OUTPUTS", str(dr_root / "outputs")))).expanduser().resolve()
    work = Path(os.environ.get("DR_WORK", str(dr_root / "work"))).expanduser().resolve()

    images = Path(os.environ.get("DR_IMAGES", str(inputs / "native_cropped"))).expanduser().resolve()
    folds = Path(os.environ.get("DR_FOLDS", str(inputs / "lodo_folds"))).expanduser().resolve()
    checkpoints = Path(os.environ.get("DR_CHECKPOINTS", str(inputs / "checkpoints"))).expanduser().resolve()
    pooled_ckpts = Path(os.environ.get("DR_POOLED_CHECKPOINTS", str(inputs / "checkpoints_head_ablation"))).expanduser().resolve()
    pooled_meta = Path(os.environ.get("DR_POOLED_METADATA", str(inputs / "hybrid"))).expanduser().resolve()
    p0_run = Path(os.environ.get("DR_P0_RUN", str(root / "result" / "p0_fp32_main_lodo_v1"))).expanduser().resolve()

    return {
        "repo_root": root,
        "dr_root": dr_root,
        "inputs": inputs,
        "outputs": outputs_base,
        "work": work,
        "images": images,
        "folds": folds,
        "checkpoints": checkpoints,
        "pooled_checkpoints": pooled_ckpts,
        "pooled_metadata": pooled_meta,
        "p0_run": p0_run,
    }

# %% [markdown]
# ### Module `experiments.mlwa.evaluator`

# %%
"""Authoritative MLWA evaluation engine: metrics, RDR screening, hierarchical aggregation, contrasts, and CIs."""


import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve


def compute_decision_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute exact 5-class decision metrics with explicit undefined handling (never silent zeros)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("y_true and y_pred must have matching non-empty 1D shape.")

    cm = confusion_matrix(y, p, labels=np.arange(5))
    support = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    recall = np.divide(cm.diagonal(), support, out=np.full(5, np.nan), where=support > 0)
    precision = np.divide(cm.diagonal(), pred_counts, out=np.full(5, np.nan), where=pred_counts > 0)

    full_support = bool((support > 0).all())
    macro_recall = float(np.mean(recall)) if full_support else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qwk = float(cohen_kappa_score(y, p, labels=np.arange(5), weights="quadratic"))

    n_total = len(y)
    n3 = int(support[3])
    pred3 = int(pred_counts[3])
    out: Dict[str, Any] = {
        "n": int(n_total),
        "qwk": qwk,
        "accuracy": float(np.mean(y == p)),
        "within1": float(np.mean(np.abs(y - p) <= 1)),
        "mae": float(np.mean(np.abs(y - p))),
        "macro_recall": macro_recall,
        "macro_recall_defined": full_support,
        "g3_precision": float(precision[3]),
        "g3_precision_defined": bool(pred3 > 0),
        "true_g3": n3,
        "predicted_g3": pred3,
        "true_g3_prevalence": float(n3 / n_total),
        "predicted_g3_prevalence": float(pred3 / n_total),
        "predicted_true_g3_ratio": float(pred3 / n3) if n3 > 0 else np.nan,
        "g3_to_2_count": int(cm[3, 2]),
        "g3_to_01_count": int(cm[3, 0] + cm[3, 1]),
        "g3_to_4_count": int(cm[3, 4]),
        "g3_to_2_fraction": float(cm[3, 2] / n3) if n3 > 0 else np.nan,
        "g3_to_01_fraction": float((cm[3, 0] + cm[3, 1]) / n3) if n3 > 0 else np.nan,
        "confusion_matrix": cm.tolist(),
    }
    for k in range(5):
        out[f"recall_g{k}"] = float(recall[k])
        out[f"precision_g{k}"] = float(precision[k])
        out[f"support_g{k}"] = int(support[k])
        out[f"precision_g{k}_defined"] = bool(pred_counts[k] > 0)
        out[f"recall_g{k}_defined"] = bool(support[k] > 0)
    return out


def compute_posterior_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Compute decoder-independent posterior quality metrics (NLL, RPS mean-4 / sum-4, 15-bin ECE, RDR AUROC)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape != (len(y), 5) or len(y) == 0:
        raise ValueError(f"Expected probs shape ({len(y)}, 5), got {p.shape}.")
    if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Invalid probabilities in compute_posterior_metrics.")

    p_y = p[np.arange(len(y)), y]
    nll_mean = float(-np.log(np.maximum(p_y, 1e-12)).mean())

    # Ranked Probability Score over 4 cumulative thresholds: explicit mean across 4 thresholds & sum
    cdf = p.cumsum(axis=1)[:, :4]
    target_cdf = (y[:, None] <= np.arange(4)[None, :]).astype(np.float64)
    sq_err = (cdf - target_cdf) ** 2
    rps_mean_4 = float(sq_err.mean())
    rps_sum_4 = float(sq_err.sum(axis=1).mean())

    # 15-bin ECE using top-probability confidence and argmax correctness
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    bins = np.minimum((confidence * 15.0).astype(int), 14)
    ece_15 = 0.0
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece_15 += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))

    y_rdr = (y >= 2).astype(int)
    risk_rdr = p[:, 2:].sum(axis=1)
    rdr_defined = bool(len(np.unique(y_rdr)) == 2)
    rdr_auroc = float(roc_auc_score(y_rdr, risk_rdr)) if rdr_defined else np.nan

    return {
        "nll_mean": nll_mean,
        "rps_mean_4": rps_mean_4,
        "rps_sum_4": rps_sum_4,
        "ece_15": float(ece_15),
        "rdr_auroc": rdr_auroc,
        "rdr_auroc_defined": rdr_defined,
    }


def calibrate_rdr_threshold(
    y_source_val: np.ndarray,
    p_source_val: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Select RDR (>=2) threshold strictly on source_val ROC at target specificity."""
    y_bin = (np.asarray(y_source_val, dtype=int) >= 2).astype(int)
    if len(np.unique(y_bin)) != 2:
        raise ValueError("Source validation must contain both RDR-negative and RDR-positive samples.")
    risk = np.asarray(p_source_val, dtype=np.float64)[:, 2:].sum(axis=1)
    fpr, tpr, thresholds = roc_curve(y_bin, risk, drop_intermediate=False)
    elig = np.flatnonzero(fpr <= 1.0 - float(target_specificity) + 1e-12)
    if len(elig) == 0:
        raise ValueError(f"No threshold achieved specificity >= {target_specificity} on source_val.")
    idx = max(elig, key=lambda j: (tpr[j], -fpr[j], thresholds[j]))
    return float(thresholds[idx])


def evaluate_rdr_screening(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_bin = np.asarray(y_true, dtype=int) >= 2
    pred_pos = np.asarray(probs, dtype=np.float64)[:, 2:].sum(axis=1) >= float(threshold)
    sens = float(pred_pos[y_bin].mean()) if y_bin.any() else np.nan
    spec = float((~pred_pos[~y_bin]).mean()) if (~y_bin).any() else np.nan
    return {
        "rdr_threshold": float(threshold),
        "rdr_sensitivity": sens,
        "rdr_specificity": spec,
        "rdr_sensitivity_defined": bool(y_bin.any()),
        "rdr_specificity_defined": bool((~y_bin).any()),
    }


def aggregate_cohort_seed_metrics(
    per_cohort_seed_df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = ("config_id", "decoder"),
) -> Dict[str, pd.DataFrame]:
    """Compute hierarchical summaries:
    (a) per_cohort: mean and sample SD (ddof=1) across seeds per cohort;
    (b) equal_cohort: per-seed 6-cohort mean, then mean and sample SD (ddof=1) across seed means.
    """
    required = list(group_cols) + ["fold_id", "cohort", "seed_train"] + list(metric_cols)
    missing = [c for c in required if c not in per_cohort_seed_df.columns]
    if missing:
        raise ValueError(f"Input table missing columns: {missing}")

    # (a) Per-cohort across training seeds
    per_cohort_rows: List[Dict[str, Any]] = []
    for keys, grp in per_cohort_seed_df.groupby(list(group_cols) + ["fold_id", "cohort"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(list(group_cols) + ["fold_id", "cohort"], keys))
        base["n_seeds"] = int(grp["seed_train"].nunique())
        base["aggregation"] = "per_cohort_across_seeds"
        for m in metric_cols:
            vals = grp[m].dropna().to_numpy(dtype=np.float64)
            base[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            base[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        per_cohort_rows.append(base)
    per_cohort_df = pd.DataFrame(per_cohort_rows)

    # (b) Per-seed equal-6-cohort mean, then mean & sample SD across seed means
    seed_mean_rows: List[Dict[str, Any]] = []
    for keys, grp in per_cohort_seed_df.groupby(list(group_cols) + ["seed_train"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(list(group_cols) + ["seed_train"], keys))
        base["n_cohorts"] = int(grp["cohort"].nunique())
        for m in metric_cols:
            vals = grp[m].to_numpy(dtype=np.float64)
            base[m] = float(np.mean(vals)) if (len(vals) == len(COHORT_ORDER) and np.isfinite(vals).all()) else np.nan
        seed_mean_rows.append(base)
    seed_means_df = pd.DataFrame(seed_mean_rows)

    eq_rows: List[Dict[str, Any]] = []
    for keys, grp in seed_means_df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(list(group_cols), keys))
        base["n_seeds"] = int(grp["seed_train"].nunique())
        base["n_cohorts_expected"] = len(COHORT_ORDER)
        base["aggregation"] = "equal_cohort_mean_of_seed_means"
        for m in metric_cols:
            vals = grp[m].dropna().to_numpy(dtype=np.float64)
            base[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            base[f"{m}_sd_across_seeds"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        eq_rows.append(base)
    equal_cohort_df = pd.DataFrame(eq_rows)

    return {
        "per_cohort": per_cohort_df,
        "per_seed_equal_cohort": seed_means_df,
        "equal_cohort": equal_cohort_df,
    }


def compute_factorial_contrasts(
    per_cell_df: pd.DataFrame,
    metric_cols: Sequence[str],
    cell_col: str = "cell",
    cell_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired within-(fold_id, seed_train) 2x2 factorial contrasts before any rounding."""
    cmap = cell_map or {"A": "A", "B": "B", "C": "C", "D": "D"}
    pivoted: List[Dict[str, Any]] = []
    for (fold_id, cohort, seed_train), grp in per_cell_df.groupby(["fold_id", "cohort", "seed_train"], sort=True):
        by_c = {str(r[cell_col]): r for _, r in grp.iterrows()}
        if not all(cmap[k] in by_c for k in ("A", "B", "C", "D")):
            continue
        A, B, C, D = by_c[cmap["A"]], by_c[cmap["B"]], by_c[cmap["C"]], by_c[cmap["D"]]
        for m in metric_cols:
            va, vb, vc, vd = float(A[m]), float(B[m]), float(C[m]), float(D[m])
            head_main = 0.5 * ((vc - va) + (vd - vb))
            loss_main = 0.5 * ((vb - va) + (vd - vc))
            interaction = (vd - vc) - (vb - va)
            contrasts = {
                "head_main": ("0.5 * ((C - A) + (D - B))", head_main),
                "loss_main": ("0.5 * ((B - A) + (D - C))", loss_main),
                "interaction": ("(D - C) - (B - A)", interaction),
                "head_simple_ce": ("C - A", vc - va),
                "head_simple_qwk": ("D - B", vd - vb),
                "loss_simple_softmax": ("B - A", vb - va),
                "loss_simple_clm": ("D - C", vd - vc),
            }
            is_pct = m in ("recall_g3", "macro_recall", "within1", "accuracy", "g3_precision", "ece_15")
            for cname, (formula, val) in contrasts.items():
                pivoted.append(
                    {
                        "fold_id": int(fold_id),
                        "cohort": str(cohort),
                        "seed_train": int(seed_train),
                        "metric": m,
                        "contrast": cname,
                        "formula": formula,
                        "value_fraction": val,
                        "value_pp": val * 100.0 if is_pct else np.nan,
                        "units": "fraction_and_pp" if is_pct else "native_metric_units",
                    }
                )
    detail_df = pd.DataFrame(pivoted)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    # Summarize per-seed equal-cohort contrast and seed consistency across cohorts
    seed_level = (
        detail_df.groupby(["metric", "contrast", "formula", "units", "seed_train"], as_index=False)[["value_fraction", "value_pp"]]
        .mean()
    )
    summary_rows: List[Dict[str, Any]] = []
    for (metric, contrast, formula, units), grp in seed_level.groupby(["metric", "contrast", "formula", "units"], sort=False):
        vals = grp["value_fraction"].dropna().to_numpy(dtype=np.float64)
        pps = grp["value_pp"].dropna().to_numpy(dtype=np.float64)
        cohort_means = detail_df[(detail_df["metric"] == metric) & (detail_df["contrast"] == contrast)].groupby("cohort")["value_fraction"].mean()
        summary_rows.append(
            {
                "metric": metric,
                "contrast": contrast,
                "formula": formula,
                "units": units,
                "n_seeds": int(len(vals)),
                "mean_fraction": float(np.mean(vals)) if len(vals) else np.nan,
                "sd_across_seed_means_fraction": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "mean_pp": float(np.mean(pps)) if len(pps) else np.nan,
                "sd_across_seed_means_pp": float(np.std(pps, ddof=1)) if len(pps) > 1 else np.nan,
                "n_cohorts_positive": int((cohort_means > 0.0).sum()),
                "n_cohorts_negative": int((cohort_means < 0.0).sum()),
                "aggregation": "equal_cohort_mean_of_seed_means",
            }
        )
    return detail_df, pd.DataFrame(summary_rows)


def paired_bootstrap_ci(
    df: pd.DataFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    unit: str = "patient_cluster",
    metric: str = "recall_g3",
    reps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap preserving multiplicity & prediction pairing; patient_cluster requires verified keys."""
    if unit not in ("patient_cluster", "grouping_key", "image"):
        raise ValueError(f"Unsupported bootstrap unit '{unit}'.")
    y = df["y_true"].to_numpy(dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    pt = float(compute_decision_metrics(y, a)[metric] - compute_decision_metrics(y, b)[metric])

    statuses = set(df.get("patient_identity_status", pd.Series(["legacy_unverified"])).astype(str))
    if unit == "patient_cluster" and statuses != {"verified"}:
        return {
            "metric": metric,
            "unit": unit,
            "point_estimate": pt,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "valid_reps": 0,
            "reps": int(reps),
            "status": "NOT_ESTIMABLE",
            "reason": "patient_identity_status != verified",
        }

    if unit == "image":
        groups = [np.array([i], dtype=int) for i in range(len(df))]
    else:
        groups = [
            np.asarray(idx, dtype=int)
            for idx in df.reset_index(drop=True).groupby("patient_key", sort=True).indices.values()
        ]

    rng = np.random.default_rng(seed)
    deltas = np.full(reps, np.nan, dtype=np.float64)
    for r in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[i] for i in chosen])
        ma = compute_decision_metrics(y[ix], a[ix])[metric]
        mb = compute_decision_metrics(y[ix], b[ix])[metric]
        deltas[r] = ma - mb

    fin = np.isfinite(deltas)
    n_valid = int(fin.sum())
    if n_valid / max(1, reps) < 0.95:
        return {
            "metric": metric,
            "unit": unit,
            "point_estimate": pt,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "valid_reps": n_valid,
            "reps": int(reps),
            "status": "INSUFFICIENT_VALID_REPS",
        }

    return {
        "metric": metric,
        "unit": unit,
        "point_estimate": pt,
        "ci_low_95": float(np.quantile(deltas[fin], 0.025)),
        "ci_high_95": float(np.quantile(deltas[fin], 0.975)),
        "valid_reps": n_valid,
        "reps": int(reps),
        "status": "ESTIMABLE",
    }


def wilcoxon_six_cohort_summary(cohort_deltas: Sequence[float]) -> Dict[str, Any]:
    """Supporting paired Wilcoxon signed-rank test on n=6 cohort-level summaries (min p = 0.03125)."""
    vals = np.asarray(cohort_deltas, dtype=np.float64)
    if len(vals) != 6:
        raise ValueError(f"Expected n=6 cohort-level paired summaries, got {len(vals)}.")
    if np.allclose(vals, 0.0):
        return {"n_cohorts": 6, "statistic": 0.0, "p_value": 1.0, "note": "supporting_only_all_zero_deltas"}
    stat, pval = wilcoxon(vals, zero_method="wilcox", alternative="two-sided", method="exact")
    return {
        "n_cohorts": 6,
        "statistic": float(stat),
        "p_value": float(pval),
        "note": "supporting_only_not_primary_pseudoreplicate",
    }

# %% [markdown]
# ### Module `experiments.mlwa.decoders`

# %%
"""Deterministic Bayes decoders, versioned CIBD selectors, stability bootstrap, and latent region math."""


import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SELECTOR_IDS = (
    "legacy_within1_1pp_v1",
    "journal_dual_guardrail_v1",
    "journal_within1_05pp_v1",
)


def validate_probabilities(p: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Strictly validate N x 5 probability matrix without silent clipping or renormalization."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5 or len(arr) == 0:
        raise ValueError(f"Expected non-empty N x 5 probabilities, got shape {arr.shape}.")
    if not np.isfinite(arr).all() or (arr < 0.0).any() or (arr > 1.0 + tol).any():
        raise ValueError("Invalid probability entries (non-finite, negative, or > 1); no silent repair.")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=tol, rtol=0.0):
        raise ValueError("Probability rows do not sum to 1 within tolerance.")
    return arr


def prior_from_labels(y: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Estimate positive source_train class prior with explicit smoothing."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=5).astype(np.float64) + float(smoothing)
    return counts / counts.sum()


def decode_cibd(
    p: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    tau: float,
    q: int,
) -> np.ndarray:
    """Cost-interpolated Bayes decoding with exact raw cost C(k, j) and lower-grade tie convention."""
    probs = validate_probabilities(p)
    pi = np.asarray(prior, dtype=np.float64)
    if pi.shape != (5,) or (pi <= 0.0).any() or not np.isfinite(pi).all():
        raise ValueError("CIBD requires strictly positive finite 5-class source prior.")
    if q not in (1, 2):
        raise ValueError(f"Unsupported CIBD distance power q={q}; expected 1 or 2.")

    r = probs / (pi[None, :] ** float(tau))
    r /= r.sum(axis=1, keepdims=True)

    d = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    cost = float(alpha) * (d ** q) + (1.0 - float(alpha)) * (d != 0)
    risk = r @ cost.T
    return (risk <= risk.min(axis=1, keepdims=True) + TIE_TOL).argmax(axis=1).astype(int)


def decode_argmax(p: np.ndarray) -> np.ndarray:
    probs = validate_probabilities(p)
    return decode_cibd(probs, np.full(5, 0.2), alpha=0.0, tau=0.0, q=1)


def decode_posterior_median(p: np.ndarray) -> np.ndarray:
    probs = validate_probabilities(p)
    return decode_cibd(probs, np.full(5, 0.2), alpha=1.0, tau=0.0, q=1)


def decode_rounded_expectation_bayes(p: np.ndarray) -> np.ndarray:
    """Exact risk-minimizing decoder for squared error |k - j|^2 (breaks half-integer ties downward)."""
    probs = validate_probabilities(p)
    return decode_cibd(probs, np.full(5, 0.2), alpha=1.0, tau=0.0, q=2)


def decode_native_legacy_round(p: np.ndarray) -> np.ndarray:
    """Legacy PyTorch round-to-even expected grade (preserved solely for historical P0 compatibility)."""
    probs = validate_probabilities(p)
    expected = probs @ np.arange(5, dtype=np.float64)
    return np.clip(np.rint(expected), 0, 4).astype(int)


def decode_cutpoint_latent(s: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Raw CLM latent-cutpoint rule (valid ONLY when scalar s and 4 ordered cutpoints exist)."""
    s_arr = np.asarray(s, dtype=np.float64)
    th = np.asarray(thetas, dtype=np.float64)
    if th.ndim == 1:
        th = np.broadcast_to(th[None, :], (len(s_arr), 4))
    if s_arr.ndim != 1 or th.shape != (len(s_arr), 4):
        raise ValueError("Cutpoint decoding requires 1D scalar s and (N, 4) cutpoints.")
    if not np.isfinite(s_arr).all() or not np.isfinite(th).all():
        raise ValueError("Cutpoint decoding is undefined for non-CLM heads with missing s or theta.")
    if (np.diff(th, axis=1) <= 0.0).any():
        raise ValueError("Cutpoints must be strictly monotonic.")
    return (s_arr[:, None] > th).sum(axis=1).astype(int)


def decode_by_name(
    decoder_id: str,
    probs: np.ndarray,
    prior: Optional[np.ndarray] = None,
    cibd_params: Optional[Dict[str, Any]] = None,
    s: Optional[np.ndarray] = None,
    thetas: Optional[np.ndarray] = None,
    native_preds: Optional[np.ndarray] = None,
) -> np.ndarray:
    if decoder_id == "argmax":
        return decode_argmax(probs)
    if decoder_id == "posterior_median":
        return decode_posterior_median(probs)
    if decoder_id in ("rounded_expectation", "rounded_expectation_bayes"):
        return decode_rounded_expectation_bayes(probs)
    if decoder_id == "native_legacy_round":
        return decode_native_legacy_round(probs)
    if decoder_id == "native":
        if native_preds is not None:
            return np.asarray(native_preds, dtype=int)
        return decode_argmax(probs)
    if decoder_id == "cutpoint_latent":
        if s is None or thetas is None:
            raise ValueError("cutpoint_latent decoder is only defined for CLM heads exposing s and theta0..theta3.")
        return decode_cutpoint_latent(s, thetas)
    if decoder_id == "cibd_selected":
        if cibd_params is None or prior is None:
            raise ValueError("cibd_selected requires prior and selected cibd_params.")
        if cibd_params.get("fallback"):
            if native_preds is not None:
                return np.asarray(native_preds, dtype=int)
            return decode_argmax(probs)
        return decode_cibd(
            probs,
            prior,
            alpha=float(cibd_params["alpha"]),
            tau=float(cibd_params["tau"]),
            q=int(cibd_params["q"]),
        )
    raise ValueError(f"Unknown decoder_id '{decoder_id}'.")


def select_cibd(
    source_val_df: pd.DataFrame,
    prior: np.ndarray,
    selector_id: str = "legacy_within1_1pp_v1",
    baseline_col: str = "pred_native",
    alphas: Optional[Sequence[float]] = None,
    taus: Optional[Sequence[float]] = None,
    qs: Sequence[int] = (1, 2),
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Select CIBD (alpha, tau, q) strictly on source validation under an explicit versioned policy."""
    if selector_id not in SELECTOR_IDS:
        raise ValueError(f"Unknown selector_id '{selector_id}'. Expected one of {SELECTOR_IDS}.")
    if "target" in set(source_val_df.get("split_role", pd.Series(dtype=str)).astype(str)):
        raise ValueError("Refusing source CIBD selection on target split rows.")

    y = source_val_df["y_true"].to_numpy(dtype=int)
    p = validate_probabilities(source_val_df[list(PROB_COLS)].to_numpy(dtype=np.float64))
    if baseline_col in source_val_df.columns:
        base_pred = source_val_df[baseline_col].to_numpy(dtype=int)
    else:
        base_pred = decode_argmax(p)
    base_metrics = compute_decision_metrics(y, base_pred)

    alpha_grid = list(alphas) if alphas is not None else [round(x / 10.0, 1) for x in range(11)]
    tau_grid = list(taus) if taus is not None else [0.0, 0.25, 0.5, 0.75, 1.0]

    w1_budget = 0.005 if selector_id == "journal_within1_05pp_v1" else 0.01
    enforce_qwk = selector_id == "journal_dual_guardrail_v1"

    rows: List[Dict[str, Any]] = []
    for alpha in alpha_grid:
        for tau in tau_grid:
            for q in qs:
                pred = decode_cibd(p, prior, alpha=float(alpha), tau=float(tau), q=int(q))
                m = compute_decision_metrics(y, pred)
                w1_ok = bool(m["within1"] >= base_metrics["within1"] - w1_budget)
                qwk_ok = bool(m["qwk"] >= base_metrics["qwk"] - 0.01)
                eligible = (w1_ok and qwk_ok) if enforce_qwk else w1_ok
                rows.append(
                    {
                        "selector_id": selector_id,
                        "alpha": float(alpha),
                        "tau": float(tau),
                        "q": int(q),
                        "recall_g3": m["recall_g3"],
                        "macro_recall": m["macro_recall"],
                        "within1": m["within1"],
                        "qwk": m["qwk"],
                        "mae": m["mae"],
                        "g3_precision": m["g3_precision"],
                        "feasible": w1_ok,
                        "qwk_guardrail_met": qwk_ok,
                        "eligible": eligible,
                    }
                )
    grid = pd.DataFrame(rows)
    elig = grid[grid["eligible"] & grid["recall_g3"].notna()]
    if elig.empty:
        return {
            "selector_id": selector_id,
            "fallback": True,
            "reason": "NO_FEASIBLE_CANDIDATE",
            "within1_budget": w1_budget,
            "enforce_qwk_guardrail": enforce_qwk,
        }, grid

    best = elig.sort_values(
        ["recall_g3", "macro_recall", "within1", "qwk", "tau", "q", "alpha"],
        ascending=[False, False, False, False, True, True, True],
    ).iloc[0]
    return {
        "selector_id": selector_id,
        "alpha": float(best["alpha"]),
        "tau": float(best["tau"]),
        "q": int(best["q"]),
        "fallback": False,
        "within1_budget": w1_budget,
        "enforce_qwk_guardrail": enforce_qwk,
    }, grid


def bootstrap_cibd_selector_stability(
    source_val_df: pd.DataFrame,
    fixed_prior: np.ndarray,
    selector_id: str = "legacy_within1_1pp_v1",
    reps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Source-validation grouping-key selector stability resampling with fixed source_train prior."""
    if "patient_key" not in source_val_df.columns:
        raise ValueError("source_val_df must include 'patient_key' grouping column.")
    statuses = set(source_val_df.get("patient_identity_status", pd.Series(["legacy_unverified"])).astype(str))
    unit = "patient_cluster" if statuses == {"verified"} else "grouping_key"

    groups = [
        np.asarray(idx, dtype=int)
        for idx in source_val_df.reset_index(drop=True).groupby("patient_key", sort=True).indices.values()
    ]
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    fallback_count = 0

    for _ in range(reps):
        sample_idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        sub_df = source_val_df.iloc[sample_idx].reset_index(drop=True)
        sel, _ = select_cibd(sub_df, fixed_prior, selector_id=selector_id)
        if sel.get("fallback"):
            fallback_count += 1
            key = "FALLBACK"
        else:
            key = f"a={sel['alpha']:.1f}|t={sel['tau']:.2f}|q={sel['q']}"
        counts[key] = counts.get(key, 0) + 1

    top_key, top_cnt = max(counts.items(), key=lambda kv: kv[1])
    return {
        "selector_id": selector_id,
        "resampling_unit": unit,
        "prior_kept_fixed": True,
        "reps": int(reps),
        "n_groups": int(len(groups)),
        "fallback_count": int(fallback_count),
        "top_config": top_key,
        "top_frequency": float(top_cnt / max(1, reps)),
        "config_counts": counts,
    }


def compute_clm_decision_regions(
    thetas: Sequence[float],
    link: str = "cloglog",
    cibd_params: Optional[Dict[str, Any]] = None,
    prior: Optional[np.ndarray] = None,
    grid_min: float = -6.0,
    grid_max: float = 6.0,
    n_grid: int = 24001,
) -> Dict[str, Any]:
    """Compute exact decision intervals over latent score s for fitted cutpoints and link."""
    th = np.asarray(thetas, dtype=np.float64)
    if th.shape != (4,) or (np.diff(th) <= 0.0).any():
        raise ValueError(f"Expected 4 strictly increasing cutpoints, got {th}.")

    s_grid = np.linspace(grid_min, grid_max, n_grid)
    u = th[None, :] - s_grid[:, None]
    if link == "cloglog":
        u_clip = np.clip(u, -30.0, 20.0)
        cdf = -np.expm1(-np.exp(u_clip))
        analytic_median = th - math.log(math.log(2.0))
    elif link == "logit":
        cdf = 1.0 / (1.0 + np.exp(-np.clip(u, -60.0, 60.0)))
        analytic_median = th.copy()
    elif link == "probit":
        from scipy.special import ndtr
        cdf = ndtr(u)
        analytic_median = th.copy()
    else:
        raise ValueError(f"Unsupported link '{link}'.")

    cdf = np.clip(cdf, 0.0, 1.0)
    probs = np.column_stack([cdf[:, 0], np.diff(cdf, axis=1).clip(0.0), (1.0 - cdf[:, -1]).clip(0.0)])
    probs /= probs.sum(axis=1, keepdims=True)

    rules: Dict[str, np.ndarray] = {
        "cutpoint_latent": decode_cutpoint_latent(s_grid, th),
        "argmax": decode_argmax(probs),
        "posterior_median": decode_posterior_median(probs),
        "rounded_expectation_bayes": decode_rounded_expectation_bayes(probs),
    }
    if cibd_params is not None and prior is not None and not cibd_params.get("fallback"):
        rules["cibd_selected"] = decode_cibd(
            probs,
            prior,
            alpha=float(cibd_params["alpha"]),
            tau=float(cibd_params["tau"]),
            q=int(cibd_params["q"]),
        )

    intervals: List[Dict[str, Any]] = []
    for r_name, pred in rules.items():
        for g in range(5):
            mask = pred == g
            if mask.any():
                idx = np.flatnonzero(mask)
                intervals.append(
                    {
                        "rule": r_name,
                        "grade": g,
                        "left_s": float(s_grid[idx[0]]),
                        "right_s": float(s_grid[idx[-1]]),
                        "width": float(s_grid[idx[-1]] - s_grid[idx[0]]),
                    }
                )
            else:
                intervals.append({"rule": r_name, "grade": g, "left_s": np.nan, "right_s": np.nan, "width": 0.0})

    return {
        "link": link,
        "thetas": [float(x) for x in th.tolist()],
        "analytic_median_boundaries": [float(x) for x in analytic_median.tolist()],
        "intervals": intervals,
    }

# %% [markdown]
# ### Module `experiments.mlwa.config`

# %%
"""Validated configuration dataclasses and deterministic hashing for MLWA journal runs."""


import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


VALID_HEAD_TYPES = ("softmax", "clm", "coral", "corn")
VALID_CLM_LINKS = ("none", "cloglog", "logit", "probit")
VALID_ARCH_IDS = ("dual_branch_mil_v1", "global_resnet50_v1")
VALID_OBJECTIVES = (
    "softmax_ce",
    "softmax_ce_qwk",
    "clm_nll",
    "clm_nll_qwk",
    "cui_cb_ce",
    "menon_logit_adj_ce",
    "clm_cui_weighted_nll",
    "clm_logprob_adjusted_nll",
    "galdran_cost_sensitive_ce",
    "coral_bce",
    "corn_bce",
)
VALID_SAMPLING_POLICIES = ("natural_source_train", "quota_balanced")
VALID_DECODERS = ("argmax", "posterior_median", "rounded_expectation_bayes", "native_legacy_round")


@dataclass(frozen=True)
class SeedConfig:
    """Explicit separation of split, training, resampling, and probe-control seeds."""

    seed_train: int = 0
    seed_split: int = 42
    seed_resampling: int = 42
    seed_probe_control: Tuple[int, ...] = (0, 1, 2)

    def validate(self) -> None:
        if self.seed_train < 0 or self.seed_split < 0 or self.seed_resampling < 0:
            raise ValueError("All seeds must be non-negative integers.")
        if not self.seed_probe_control or any(s < 0 for s in self.seed_probe_control):
            raise ValueError("seed_probe_control must contain non-negative integer seeds.")


@dataclass(frozen=True)
class ModelSpec:
    """Architecture, head, link, and preprocessing contract (distinct from objective and decoder)."""

    architecture_id: str = "dual_branch_mil_v1"
    backbone_name: str = "resnet50"
    head_type: str = "softmax"
    clm_link: str = "none"
    pretrained: bool = True
    use_local_branch: bool = True
    use_nonlocal: bool = True
    use_quadrant_tokens: bool = True
    use_cbam: bool = True
    fusion_dim: int = 512
    dropout: float = 0.3
    eval_patch_chunk_size: Optional[int] = 32
    preprocessing_id: str = "ben_graham_green_512_fov15_p128_s96"

    def validate(self) -> None:
        if self.architecture_id not in VALID_ARCH_IDS:
            raise ValueError(f"Unsupported architecture_id '{self.architecture_id}'.")
        if self.head_type not in VALID_HEAD_TYPES:
            raise ValueError(f"Unsupported head_type '{self.head_type}'.")
        if self.head_type == "clm" and self.clm_link not in ("cloglog", "logit", "probit"):
            raise ValueError(f"CLM head requires clm_link in cloglog|logit|probit, got '{self.clm_link}'.")
        if self.head_type != "clm" and self.clm_link != "none":
            raise ValueError(f"Non-CLM head '{self.head_type}' must have clm_link='none', got '{self.clm_link}'.")
        if self.architecture_id == "global_resnet50_v1" and self.use_local_branch:
            raise ValueError("global_resnet50_v1 cannot enable use_local_branch=True.")
        if self.fusion_dim <= 0 or (self.eval_patch_chunk_size is not None and self.eval_patch_chunk_size <= 0):
            raise ValueError("fusion_dim and eval_patch_chunk_size must be positive.")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Explicit training objective specification preventing silent QWK or double-rebalancing bugs."""

    objective_id: str = "softmax_ce"
    use_qwk: bool = False
    target_qwk_weight: float = 0.4
    qwk_warmup_epochs: int = 15
    qwk_schedule_mode: str = "linear"
    cui_beta: float = 0.9999
    logit_adj_tau_train: float = 1.0
    galdran_lambda: float = 1.0
    galdran_power: int = 2
    allow_combined_rebalancing: bool = False

    def validate(self, model: ModelSpec, sampling_policy: str) -> None:
        if self.objective_id == "exp_mse":
            raise NotImplementedError(
                "NEEDS_METHOD_SPEC: verify exact Stelter et al. (NLDL 2026) exp_MSE formula "
                "from primary PMLR publication before enabling."
            )
        if self.objective_id not in VALID_OBJECTIVES:
            raise ValueError(f"Unsupported objective_id '{self.objective_id}'.")

        is_softmax_obj = self.objective_id in (
            "softmax_ce",
            "softmax_ce_qwk",
            "cui_cb_ce",
            "menon_logit_adj_ce",
            "galdran_cost_sensitive_ce",
        )
        is_clm_obj = self.objective_id in (
            "clm_nll",
            "clm_nll_qwk",
            "clm_cui_weighted_nll",
            "clm_logprob_adjusted_nll",
        )
        if is_softmax_obj and model.head_type != "softmax":
            raise ValueError(
                f"Objective '{self.objective_id}' requires head_type='softmax' (got '{model.head_type}'). "
                "Do not silently apply Softmax logit adjustment/CE to scalar CLM outputs."
            )
        if is_clm_obj and model.head_type != "clm":
            raise ValueError(f"Objective '{self.objective_id}' requires head_type='clm', got '{model.head_type}'.")
        if self.objective_id == "coral_bce" and model.head_type != "coral":
            raise ValueError("coral_bce requires head_type='coral'.")
        if self.objective_id == "corn_bce" and model.head_type != "corn":
            raise ValueError("corn_bce requires head_type='corn'.")

        if self.objective_id in ("softmax_ce_qwk", "clm_nll_qwk"):
            if not self.use_qwk or self.target_qwk_weight <= 0.0:
                raise ValueError(f"QWK-enabled objective '{self.objective_id}' requires use_qwk=True and target_qwk_weight > 0.")
        elif self.use_qwk:
            raise ValueError(f"Pure objective '{self.objective_id}' must set use_qwk=False (no silent QWK injection).")

        if not (0.0 <= self.cui_beta < 1.0):
            raise ValueError(f"cui_beta must lie in [0, 1), got {self.cui_beta}.")
        if self.qwk_schedule_mode not in ("linear", "fixed"):
            raise ValueError(f"Unsupported qwk_schedule_mode '{self.qwk_schedule_mode}'.")

        is_loss_rebalanced = self.objective_id in (
            "cui_cb_ce",
            "menon_logit_adj_ce",
            "clm_cui_weighted_nll",
            "clm_logprob_adjusted_nll",
        )
        if is_loss_rebalanced and sampling_policy == "quota_balanced" and not self.allow_combined_rebalancing:
            raise ValueError(
                f"Refusing unintended double rebalancing: objective '{self.objective_id}' with "
                f"sampling_policy='quota_balanced' requires allow_combined_rebalancing=True."
            )


@dataclass(frozen=True)
class TrainingSpec:
    """Shared optimization, batching, warmup, and checkpoint-selection settings."""

    epochs: int = 30
    batch_size: int = 12
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    cosine_t_max: int = 30
    min_lr: float = 1e-6
    patience: int = 15
    grad_clip: float = 1.0
    amp_mode: str = "auto"
    checkpoint_selection_decoder: str = "argmax"
    sampling_policy: str = "natural_source_train"
    quota_target_per_class: int = 2000
    allow_grad_accum_qwk_mismatch: bool = False

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size, grad_accum_steps, and patience must be positive.")
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("lr must be > 0 and weight_decay >= 0.")
        if self.checkpoint_selection_decoder not in VALID_DECODERS:
            raise ValueError(f"Unsupported checkpoint_selection_decoder '{self.checkpoint_selection_decoder}'.")
        if self.sampling_policy not in VALID_SAMPLING_POLICIES:
            raise ValueError(f"Unsupported sampling_policy '{self.sampling_policy}'.")
        if self.grad_accum_steps > 1 and not self.allow_grad_accum_qwk_mismatch:
            raise ValueError(
                "grad_accum_steps > 1 alters BatchNorm and batch-coupled QWK observation batch size relative to "
                "physical batch_size; set allow_grad_accum_qwk_mismatch=True explicitly if intended."
            )


@dataclass(frozen=True)
class JournalExperimentConfig:
    """Top-level immutable configuration unit for one MLWA journal training/eval job."""

    study_id: str = "mlwa_dr_ordinal_extension"
    protocol_id: str = "mlwa_journal_v1"
    run_id: str = "journal_run_001"
    experiment: str = "E2"
    config_id: str = "E2_A_softmax_ce"
    fold_id: int = 0
    cohort: str = "aptos"
    seeds: SeedConfig = field(default_factory=SeedConfig)
    model: ModelSpec = field(default_factory=ModelSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    is_test_only: bool = False

    def validate(self) -> None:
        if self.fold_id not in CANONICAL_FOLD_TO_COHORT:
            raise ValueError(f"Invalid fold_id={self.fold_id}; expected 0..5.")
        expected_cohort = CANONICAL_FOLD_TO_COHORT[self.fold_id]
        if self.cohort != expected_cohort:
            raise ValueError(
                f"Mismatched fold_id={self.fold_id} and cohort='{self.cohort}'; "
                f"canonical LODO cohort is '{expected_cohort}'."
            )
        self.seeds.validate()
        self.model.validate()
        self.training.validate()
        self.objective.validate(self.model, self.training.sampling_policy)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save_json(self, path: Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JournalExperimentConfig":
        cfg = cls(
            study_id=raw["study_id"],
            protocol_id=raw["protocol_id"],
            run_id=raw["run_id"],
            experiment=raw["experiment"],
            config_id=raw["config_id"],
            fold_id=int(raw["fold_id"]),
            cohort=str(raw["cohort"]),
            seeds=SeedConfig(
                seed_train=int(raw["seeds"]["seed_train"]),
                seed_split=int(raw["seeds"]["seed_split"]),
                seed_resampling=int(raw["seeds"]["seed_resampling"]),
                seed_probe_control=tuple(int(x) for x in raw["seeds"]["seed_probe_control"]),
            ),
            model=ModelSpec(**raw["model"]),
            objective=ObjectiveSpec(**raw["objective"]),
            training=TrainingSpec(**raw["training"]),
            is_test_only=bool(raw.get("is_test_only", False)),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load_json(cls, path: Path) -> "JournalExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


def build_factorial_cell_spec(cell: str, pretrained: bool = True) -> Tuple[str, ModelSpec, ObjectiveSpec]:
    """Return canonical ModelSpec and ObjectiveSpec for E2 cells A, B, C, or D."""
    c = cell.strip().upper()
    if c == "A":
        return (
            "E2_A_softmax_ce",
            ModelSpec(head_type="softmax", clm_link="none", pretrained=pretrained),
            ObjectiveSpec(objective_id="softmax_ce", use_qwk=False),
        )
    if c == "B":
        return (
            "E2_B_softmax_ce_qwk",
            ModelSpec(head_type="softmax", clm_link="none", pretrained=pretrained),
            ObjectiveSpec(objective_id="softmax_ce_qwk", use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15),
        )
    if c == "C":
        return (
            "E2_C_clm_cloglog_nll",
            ModelSpec(head_type="clm", clm_link="cloglog", pretrained=pretrained),
            ObjectiveSpec(objective_id="clm_nll", use_qwk=False),
        )
    if c == "D":
        return (
            "E2_D_clm_cloglog_nll_qwk",
            ModelSpec(head_type="clm", clm_link="cloglog", pretrained=pretrained),
            ObjectiveSpec(objective_id="clm_nll_qwk", use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15),
        )
    raise ValueError(f"Unknown E2 cell '{cell}'. Expected A, B, C, or D.")

# %% [markdown]
# ### Module `losses.qwk`

# %%
"""Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss.

Computes continuous soft confusion matrices to directly optimize clinical
ordinal agreement:
    QWK = 1 - (sum(W * O) / sum(W * E))
    Loss = sum(W * O) / (sum(W * E) + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousQWKLoss(nn.Module):
    """Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss.

    Minimizing (sum(W * O) / sum(W * E)) directly maximizes Cohen's quadratic
    weighted kappa agreement between soft predicted distributions and targets.
    """

    def __init__(self, num_classes: int = 5, eps: float = 1e-7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        # Quadratic penalty matrix: W_ij = (i - j)^2 / (C - 1)^2
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

        self.register_buffer("W", w)

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation of differentiable QWK loss.

        Args:
            pred_probs: Continuous class probabilities (B, num_classes).
            targets: Integer class indices (B,) or one-hot distributions (B, num_classes).

        Returns:
            Scalar differentiable loss tensor.
        """
        device = pred_probs.device
        n = pred_probs.size(0)
        if n == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()

        # Normalize predicted probabilities to simplex
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)

        # Soft observed confusion matrix: O = (P^T * Y) / N
        observed = torch.matmul(pred_probs.t(), y_one_hot) / (n + self.eps)

        # Expected confusion matrix under chance: E = hist_pred^T * hist_true
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (n + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (n + self.eps)
        expected = torch.matmul(hist_pred.t(), hist_true)

        penalty = self.W.to(device)
        num = torch.sum(penalty * observed)
        den = torch.sum(penalty * expected) + self.eps

        return num / den

# %% [markdown]
# ### Module `experiments.mlwa.objectives`

# %%
"""Training objectives, symmetric QWK warmup, Cui CB-CE, Menon logit adjustment, and Galdran CSR."""


from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectiveOutput:
    total_loss: torch.Tensor
    base_loss: float
    qwk_loss: float
    aux_loss: float
    active_qwk_weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_qwk_weight(epoch: int, spec: ObjectiveSpec) -> float:
    """Symmetric QWK lambda schedule shared identically by Softmax+QWK and CLM+QWK."""
    if not spec.use_qwk:
        return 0.0
    if spec.qwk_schedule_mode == "fixed":
        return float(spec.target_qwk_weight)
    warmup = max(1, int(spec.qwk_warmup_epochs))
    progress = min(1.0, max(1, int(epoch)) / float(warmup))
    return float(progress * spec.target_qwk_weight)


def compute_cui_effective_weights(
    class_counts: np.ndarray,
    beta: float = 0.9999,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cui et al. (CVPR 2019) class-balanced weights inversely proportional to (1 - beta^n) / (1 - beta)."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or len(counts) != 5:
        raise ValueError(f"Expected 5-class natural source_train counts, got shape {counts.shape}.")
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must lie in [0, 1), got {beta}.")

    observed = counts > 0
    if not observed.any():
        raise ValueError("Cannot compute Cui effective-number weights with zero total samples.")

    weights = np.zeros_like(counts, dtype=np.float64)
    if beta == 0.0:
        weights[observed] = 1.0
        effective_nums = counts.copy()
    else:
        log_beta = np.log(beta)
        effective_nums = np.zeros_like(counts, dtype=np.float64)
        # Use -expm1(n * log(beta)) / (1 - beta) for high numerical accuracy near beta -> 1.
        effective_nums[observed] = -np.expm1(counts[observed] * log_beta) / (1.0 - beta)
        weights[observed] = 1.0 / effective_nums[observed]
        mean_obs = float(weights[observed].mean())
        weights[observed] = weights[observed] / mean_obs

    absent_classes = [int(i) for i in range(5) if not observed[i]]
    meta = {
        "beta": float(beta),
        "natural_counts": [int(c) for c in counts.tolist()],
        "absent_classes": absent_classes,
        "weights": [float(w) for w in weights.tolist()],
    }
    return weights.astype(np.float32), meta


def compute_menon_log_prior(class_counts: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """Compute natural source_train class priors for Menon et al. (ICLR 2021) logit adjustment."""
    counts = np.asarray(class_counts, dtype=np.float64) + float(smoothing)
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Invalid class counts for Menon prior computation.")
    probs = counts / counts.sum()
    return np.log(np.maximum(probs, 1e-12)).astype(np.float32)


class JournalObjective(nn.Module):
    """Typed objective evaluator keeping likelihood, QWK, and rebalancing mechanisms explicit."""

    def __init__(
        self,
        spec: ObjectiveSpec,
        natural_train_counts: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.qwk_loss_fn = ContinuousQWKLoss(num_classes=5)
        self.meta: Dict[str, Any] = {"objective_id": spec.objective_id}

        self.register_buffer("cui_weights", torch.ones(5, dtype=torch.float32))
        self.register_buffer("log_prior", torch.zeros(5, dtype=torch.float32))

        if spec.objective_id in ("cui_cb_ce", "clm_cui_weighted_nll"):
            if natural_train_counts is None:
                raise ValueError(f"Objective '{spec.objective_id}' requires natural source_train class counts.")
            w, w_meta = compute_cui_effective_weights(natural_train_counts, beta=spec.cui_beta)
            self.cui_weights.copy_(torch.from_numpy(w))
            self.meta["cui"] = w_meta

        if spec.objective_id in ("menon_logit_adj_ce", "clm_logprob_adjusted_nll"):
            if natural_train_counts is None:
                raise ValueError(f"Objective '{spec.objective_id}' requires natural source_train class counts.")
            # Smooth minimally only if an observed fold somehow lacks a class; natural counts > 0 here.
            smooth = 1.0 if (np.asarray(natural_train_counts) == 0).any() else 0.0
            lp = compute_menon_log_prior(natural_train_counts, smoothing=smooth)
            self.log_prior.copy_(torch.from_numpy(lp))
            self.meta["menon"] = {
                "tau_train": float(spec.logit_adj_tau_train),
                "log_prior": [float(x) for x in lp.tolist()],
                "smoothing_used": smooth,
            }

        # Precompute Galdran cost matrix M_{y, c} = |y - c|^p
        grid = torch.arange(5, dtype=torch.float32)
        cost_mat = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0)) ** float(spec.galdran_power)
        self.register_buffer("galdran_cost", cost_mat)

    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        epoch: int = 1,
    ) -> ObjectiveOutput:
        oid = self.spec.objective_id
        targets = targets.long()
        active_qwk = compute_qwk_weight(epoch, self.spec)

        if oid == "softmax_ce":
            base = F.cross_entropy(outputs["logits"].float(), targets)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "softmax_ce_qwk":
            base = F.cross_entropy(outputs["logits"].float(), targets)
            qwk = self.qwk_loss_fn(outputs["probs"].float(), targets)
            total = base + active_qwk * qwk
            return ObjectiveOutput(total, float(base.detach().item()), float(qwk.detach().item()), 0.0, active_qwk, self.meta)

        if oid == "clm_nll":
            probs = outputs["probs"].float()
            p_y = probs[torch.arange(targets.size(0), device=targets.device), targets]
            base = -torch.log(torch.clamp(p_y, min=1e-12)).mean()
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "clm_nll_qwk":
            probs = outputs["probs"].float()
            p_y = probs[torch.arange(targets.size(0), device=targets.device), targets]
            base = -torch.log(torch.clamp(p_y, min=1e-12)).mean()
            qwk = self.qwk_loss_fn(probs, targets)
            total = base + active_qwk * qwk
            return ObjectiveOutput(total, float(base.detach().item()), float(qwk.detach().item()), 0.0, active_qwk, self.meta)

        if oid == "cui_cb_ce":
            w = self.cui_weights.to(targets.device)
            per_sample = F.cross_entropy(outputs["logits"].float(), targets, reduction="none")
            sample_w = w[targets]
            denom = sample_w.sum().clamp(min=1e-12)
            base = (per_sample * sample_w).sum() / denom
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "menon_logit_adj_ce":
            # Training-time logit adjustment adds +tau * log(pi) before CE (Menon et al., ICLR 2021).
            adj_logits = outputs["logits"].float() + float(self.spec.logit_adj_tau_train) * self.log_prior.to(targets.device).unsqueeze(0)
            base = F.cross_entropy(adj_logits, targets)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "clm_cui_weighted_nll":
            probs = outputs["probs"].float()
            p_y = probs[torch.arange(targets.size(0), device=targets.device), targets]
            per_sample = -torch.log(torch.clamp(p_y, min=1e-12))
            sample_w = self.cui_weights.to(targets.device)[targets]
            base = (per_sample * sample_w).sum() / sample_w.sum().clamp(min=1e-12)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "clm_logprob_adjusted_nll":
            log_probs = torch.log(torch.clamp(outputs["probs"].float(), min=1e-12))
            adj = log_probs + float(self.spec.logit_adj_tau_train) * self.log_prior.to(targets.device).unsqueeze(0)
            base = F.cross_entropy(adj, targets)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "galdran_cost_sensitive_ce":
            logits = outputs["logits"].float()
            probs = outputs["probs"].float()
            base = F.cross_entropy(logits, targets)
            cost_rows = self.galdran_cost.to(targets.device)[targets]
            reg = torch.sum(cost_rows * probs, dim=1).mean()
            total = base + float(self.spec.galdran_lambda) * reg
            return ObjectiveOutput(total, float(base.detach().item()), 0.0, float(reg.detach().item()), 0.0, self.meta)

        if oid == "coral_bce":
            rank_logits = outputs["rank_logits"].float()
            binary_targets = torch.stack([(targets > k).float() for k in range(4)], dim=1)
            base = F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "corn_bce":
            cond_logits = outputs["cond_logits"].float()
            losses = []
            for k in range(4):
                mask = targets >= k
                if mask.any():
                    losses.append(F.binary_cross_entropy_with_logits(cond_logits[mask, k], (targets[mask] > k).float()))
            base = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=targets.device, requires_grad=True)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        raise ValueError(f"Unhandled objective_id '{oid}'.")

# %% [markdown]
# ### Module `experiments.mlwa.brc`

# %%
"""Gated E3 Bounded Residual Correction (BRC) and mandatory capacity/retraining controls."""


import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate_h3_gate(
    probe_df: pd.DataFrame,
    evidence_type: str = "retrospective_legacy",
    min_positive_cohorts: int = 5,
    min_mean_delta_g3_pp: float = 0.0,
) -> Dict[str, Any]:
    """Paired vector-vs-scalar H3 gate (never uses simplistic single-CI overlap)."""
    required_reps = {"z_v3", "s_v3"}
    reps_present = set(probe_df.get("representation", pd.Series(dtype=str)).astype(str))
    if not required_reps.issubset(reps_present):
        return {
            "gate_open": False,
            "status": STATUS_SKIPPED_GATE_NOT_MET,
            "reason": "Missing z_v3 or s_v3 probe rows.",
            "evidence_type": evidence_type,
        }

    zv = probe_df[probe_df["representation"] == "z_v3"].sort_values(["fold_id"])
    sv = probe_df[probe_df["representation"] == "s_v3"].sort_values(["fold_id"])
    key_cols = [c for c in ("fold_id", "seed_train") if c in zv.columns and c in sv.columns]
    merged = zv.merge(sv, on=key_cols, suffixes=("_z", "_s"))
    if merged.empty:
        return {
            "gate_open": False,
            "status": STATUS_SKIPPED_GATE_NOT_MET,
            "reason": "No paired z_v3 vs s_v3 rows.",
            "evidence_type": evidence_type,
        }

    merged["delta_g3_pp"] = (merged["recall_g3_z"] - merged["recall_g3_s"]) * 100.0
    merged["delta_macro_pp"] = (merged["macro_recall_z"] - merged["macro_recall_s"]) * 100.0
    cohort_means = merged.groupby("fold_id")[["delta_g3_pp", "delta_macro_pp"]].mean()

    mean_g3 = float(cohort_means["delta_g3_pp"].mean())
    mean_macro = float(cohort_means["delta_macro_pp"].mean())
    n_pos_g3 = int((cohort_means["delta_g3_pp"] > 0.0).sum())
    n_pos_macro = int((cohort_means["delta_macro_pp"] > 0.0).sum())

    # Require multi-seed journal evidence AND consistent paired superiority on both endpoints
    gate_open = bool(
        evidence_type == "journal_multiseed"
        and mean_g3 > min_mean_delta_g3_pp
        and mean_macro > 0.0
        and n_pos_g3 >= min_positive_cohorts
        and n_pos_macro >= min_positive_cohorts
    )
    return {
        "gate_open": gate_open,
        "status": "H3_GATE_OPEN" if gate_open else STATUS_SKIPPED_GATE_NOT_MET,
        "evidence_type": evidence_type,
        "mean_paired_delta_g3_pp": mean_g3,
        "mean_paired_delta_macro_pp": mean_macro,
        "n_cohorts_z_gt_s_g3": n_pos_g3,
        "n_cohorts_z_gt_s_macro": n_pos_macro,
    }


class BoundedResidualCorrectionHead(nn.Module):
    """Residual-corrected CLM: log p'_j(x) = log(p_clm_j(x)) + beta * tanh(W z + b), softmax-normalized.

    Initialized with W=0, b=0 so initial prediction equals baseline p_clm within tolerance.
    Output name is residual_corrected_clm (not pure CLM).
    """

    model_label = "residual_corrected_clm"

    def __init__(self, in_features: int = 512, num_classes: int = 5, beta: float = 0.5, eps: float = 1e-12) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.beta = float(beta)
        self.eps = float(eps)
        self.linear = nn.Linear(in_features, num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, p_clm: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        p_safe = torch.clamp(p_clm.float(), min=self.eps)
        log_base = torch.log(p_safe)
        residual = self.beta * torch.tanh(self.linear(features.float()))
        return F.softmax(log_base + residual, dim=1)


class UnboundedResidualHead(nn.Module):
    """Unbounded residual control on frozen features (2,565 params for 512->5)."""

    def __init__(self, in_features: int = 512, num_classes: int = 5, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)
        self.linear = nn.Linear(in_features, num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, p_clm: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        log_base = torch.log(torch.clamp(p_clm.float(), min=self.eps))
        return F.softmax(log_base + self.linear(features.float()), dim=1)


def fit_temperature_scaling_source_val(
    p_val: np.ndarray,
    y_val: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """Fit scalar temperature T > 0 strictly on source_val NLL (Guo et al., 2017)."""
    log_p = torch.tensor(np.log(np.clip(p_val, eps, 1.0)), dtype=torch.float64)
    y_t = torch.tensor(y_val, dtype=torch.long)
    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(log_p / temp, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    temp_val = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).item())
    return temp_val, F.softmax(log_p / temp_val, dim=1).detach().numpy()


def apply_temperature_scaling(p: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    log_p = np.log(np.clip(p, eps, 1.0)) / float(temperature)
    log_p -= log_p.max(axis=1, keepdims=True)
    ex = np.exp(log_p)
    return ex / ex.sum(axis=1, keepdims=True)


def _train_linear_control(
    forward_fn,
    params: Sequence[nn.Parameter],
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    p_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    p_va: torch.Tensor,
    sample_weights_tr: Optional[torch.Tensor] = None,
    epochs: int = 80,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], int]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(list(params), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        probs = forward_fn(p_tr, x_tr)
        p_y = probs[torch.arange(len(y_tr)), y_tr].clamp(min=1e-12)
        nll = -torch.log(p_y)
        if sample_weights_tr is not None:
            loss = (nll * sample_weights_tr).sum() / sample_weights_tr.sum().clamp(min=1e-12)
        else:
            loss = nll.mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            vp = forward_fn(p_va, x_va)
            v_loss = float((-torch.log(vp[torch.arange(len(y_va)), y_va].clamp(min=1e-12))).mean().item())
        if v_loss < best_loss:
            best_loss = v_loss
            best_epoch = ep
            best_state = {f"p{i}": copy.deepcopy(par.detach()) for i, par in enumerate(params)}

    assert best_state is not None
    return best_state, best_epoch


def build_dfr_inspired_reweight_split(
    source_train_df: pd.DataFrame,
    reweight_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a grouped reweight subset strictly from source_train (never touching source_val)."""
    rng = np.random.default_rng(seed)
    groups = source_train_df["patient_key"].astype(str).unique()
    perm = rng.permutation(groups)
    n_rw = max(1, int(round(len(perm) * reweight_frac)))
    rw_groups = set(perm[:n_rw].tolist())
    mask_rw = source_train_df["patient_key"].astype(str).isin(rw_groups).to_numpy()

    # Compute inverse group-by-cohort-grade balancing weights on the carved reweight subset ONLY
    sub = source_train_df.loc[mask_rw].copy()
    sub["group_cell"] = sub["cohort"].astype(str) + "::" + sub["y_true"].astype(str)
    cell_counts = sub["group_cell"].value_counts().to_dict()
    weights = np.array([1.0 / max(1, cell_counts[c]) for c in sub["group_cell"]], dtype=np.float32)
    weights /= float(weights.mean())
    return mask_rw, weights


def run_e3_with_controls(
    z_train: np.ndarray,
    s_train: np.ndarray,
    p_train: np.ndarray,
    source_train_df: pd.DataFrame,
    z_val: np.ndarray,
    s_val: np.ndarray,
    p_val: np.ndarray,
    y_val: np.ndarray,
    z_target: np.ndarray,
    s_target: np.ndarray,
    p_target: np.ndarray,
    y_target: np.ndarray,
    beta_grid: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    seed: int = 42,
) -> pd.DataFrame:
    """Execute BRC and all 7 required controls (strictly source-fitted/selected, target evaluated once)."""
    y_tr = source_train_df["y_true"].to_numpy(dtype=int)
    zt = torch.tensor(z_train, dtype=torch.float32)
    st = torch.tensor(s_train[:, None], dtype=torch.float32)
    pt = torch.tensor(p_train, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)

    zv = torch.tensor(z_val, dtype=torch.float32)
    sv = torch.tensor(s_val[:, None], dtype=torch.float32)
    pv = torch.tensor(p_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    zg = torch.tensor(z_target, dtype=torch.float32)
    sg = torch.tensor(s_target[:, None], dtype=torch.float32)
    pg = torch.tensor(p_target, dtype=torch.float32)

    records: List[Dict[str, Any]] = []

    def _record(name: str, output_family: str, param_count: int, p_out: np.ndarray, extra: str = "") -> None:
        pred = np.argmax(p_out, axis=1).astype(int)
        dm = compute_decision_metrics(y_target, pred)
        pm = compute_posterior_metrics(y_target, p_out)
        records.append(
            {
                "method": name,
                "output_family": output_family,
                "trainable_params": int(param_count),
                "note": extra,
                "recall_g3": dm["recall_g3"],
                "macro_recall": dm["macro_recall"],
                "within1": dm["within1"],
                "qwk": dm["qwk"],
                "mae": dm["mae"],
                "nll_mean": pm["nll_mean"],
                "rps_mean_4": pm["rps_mean_4"],
                "ece_15": pm["ece_15"],
            }
        )

    # 1. Native CLM
    _record("native_clm", "pure_clm", 0, p_target)

    # 2. Source-only temperature scaling
    temp_val, _ = fit_temperature_scaling_source_val(p_val, y_val)
    _record("temperature_scaling_source_only", "calibrated_clm", 1, apply_temperature_scaling(p_target, temp_val), f"T={temp_val:.4f}")

    # 3. Bounded residual on z (beta selected on source_val NLL)
    best_brc_head: Optional[BoundedResidualCorrectionHead] = None
    best_brc_val_nll = float("inf")
    best_beta = beta_grid[0]
    for b in beta_grid:
        head = BoundedResidualCorrectionHead(in_features=z_train.shape[1], beta=float(b))
        st_dict, _ = _train_linear_control(head.forward, list(head.parameters()), zt, yt, pt, zv, yv, pv, seed=seed)
        for i, par in enumerate(head.parameters()):
            par.data.copy_(st_dict[f"p{i}"])
        with torch.no_grad():
            vp = head(pv, zv).numpy()
        vnll = compute_posterior_metrics(y_val, vp)["nll_mean"]
        if vnll < best_brc_val_nll:
            best_brc_val_nll = vnll
            best_brc_head = head
            best_beta = float(b)
    assert best_brc_head is not None
    with torch.no_grad():
        p_brc = best_brc_head(pg, zg).numpy()
    _record("brc_vector_z", "residual_corrected_clm", best_brc_head.parameter_count(), p_brc, f"beta={best_beta}")

    # 4. Bounded residual on scalar s
    s_head = BoundedResidualCorrectionHead(in_features=1, beta=best_beta)
    st_s, _ = _train_linear_control(s_head.forward, list(s_head.parameters()), st, yt, pt, sv, yv, pv, seed=seed)
    for i, par in enumerate(s_head.parameters()):
        par.data.copy_(st_s[f"p{i}"])
    with torch.no_grad():
        p_sbrc = s_head(pg, sg).numpy()
    _record("brc_scalar_s", "residual_corrected_clm", s_head.parameter_count(), p_sbrc, f"beta={best_beta}")

    # 5. Unbounded residual on z
    ub_head = UnboundedResidualHead(in_features=z_train.shape[1])
    st_u, _ = _train_linear_control(ub_head.forward, list(ub_head.parameters()), zt, yt, pt, zv, yv, pv, seed=seed)
    for i, par in enumerate(ub_head.parameters()):
        par.data.copy_(st_u[f"p{i}"])
    with torch.no_grad():
        p_ub = ub_head(pg, zg).numpy()
    _record("unbounded_residual_z", "residual_corrected_clm", ub_head.parameter_count(), p_ub)

    # 6. Full linear retrain on frozen z + 7. cRT class-balanced retrain on frozen z
    for m_name, use_cb in [("full_linear_retrain_z", False), ("crt_class_balanced_retrain", True)]:
        lin = nn.Linear(z_train.shape[1], 5)
        w_tr = None
        if use_cb:
            counts = np.bincount(y_tr, minlength=5).astype(np.float64)
            w_np, _ = compute_cui_effective_weights(counts, beta=0.9999)
            w_tr = torch.tensor(w_np[y_tr], dtype=torch.float32)
        st_l, _ = _train_linear_control(lambda _p, feat: F.softmax(lin(feat), dim=1), list(lin.parameters()), zt, yt, pt, zv, yv, pv, sample_weights_tr=w_tr, seed=seed)
        for i, par in enumerate(lin.parameters()):
            par.data.copy_(st_l[f"p{i}"])
        with torch.no_grad():
            p_l = F.softmax(lin(zg), dim=1).numpy()
        _record(m_name, "retrained_linear_head", sum(p.numel() for p in lin.parameters()), p_l)

    # 8. DFR-inspired source_train reweight split control (leaves source_val untouched)
    mask_rw_np, rw_weights = build_dfr_inspired_reweight_split(source_train_df, seed=seed)
    mask_rw = torch.as_tensor(np.array(mask_rw_np, copy=True), dtype=torch.bool)
    dfr_lin = nn.Linear(z_train.shape[1], 5)
    st_d, _ = _train_linear_control(
        lambda _p, feat: F.softmax(dfr_lin(feat), dim=1),
        list(dfr_lin.parameters()),
        zt[mask_rw], yt[mask_rw], pt[mask_rw],
        zv, yv, pv,
        sample_weights_tr=torch.tensor(np.array(rw_weights, copy=True), dtype=torch.float32),
        seed=seed,
    )
    for i, par in enumerate(dfr_lin.parameters()):
        par.data.copy_(st_d[f"p{i}"])
    with torch.no_grad():
        p_dfr = F.softmax(dfr_lin(zg), dim=1).numpy()
    _record("dfr_inspired_reweight_split", "retrained_linear_head", sum(p.numel() for p in dfr_lin.parameters()), p_dfr, "DFR_INSPIRED_COHORT_GRADE_PROXY")

    return pd.DataFrame(records)

# %% [markdown]
# ### Module `experiments.mlwa.data_contract`

# %%
"""Data, manifest, image-identity, P0 cache, and prediction schema contracts."""


import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def relative_image_path(value: str) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    lower = [p.lower() for p in parts]
    if "native_cropped" in lower:
        parts = parts[lower.index("native_cropped") + 1 :]
    if not parts or ".." in parts or any(":" in p for p in parts) or parts[0] == "/":
        raise ValueError(f"Unsafe or unresolved relative image path: {value}")
    return PurePosixPath(*parts).as_posix()


def stage_archive_safe(source: Path, workspace: Path) -> Path:
    """Safely extract an immutable ZIP to a keyed local folder or return an existing directory."""
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        return source
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        seen = set()
        for info in entries:
            name = info.filename.replace("\\", "/")
            rel = PurePosixPath(name)
            is_symlink = ((info.external_attr >> 16) & 0o170000) == 0o120000
            if rel.is_absolute() or ".." in rel.parts or ":" in name or is_symlink or not rel.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if name.casefold() in seen:
                raise ValueError(f"Duplicate ZIP member path: {info.filename}")
            seen.add(name.casefold())

        index = [(m.filename, m.file_size, m.CRC) for m in entries]
        key = object_hash(index)
        target = Path(workspace) / f"{source.stem}_{key[:16]}"
        ready = target / ".archive-complete"
        if ready.is_file() and ready.read_text(encoding="utf-8").strip() == key:
            return target

        temp = target.with_name(f"{target.name}.{uuid4().hex}.partial")
        temp.mkdir(parents=True, exist_ok=True)
        try:
            for m in entries:
                parts = PurePosixPath(m.filename.replace("\\", "/")).parts
                dest = temp.joinpath(*parts)
                if not dest.resolve().is_relative_to(temp.resolve()):
                    raise ValueError(f"Path traversal blocked: {m.filename}")
                if m.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(m) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 2**20)
            (temp / ".archive-complete").write_text(key, encoding="utf-8")
            try:
                temp.rename(target)
            except OSError:
                if not ready.is_file() or ready.read_text(encoding="utf-8").strip() != key:
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return target


def normalize_manifest_df(
    df: pd.DataFrame,
    patient_map: Optional[Path] = None,
    images_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Attach canonical image_key, grouping/patient_key, and identity status cleanly."""
    out = df.copy()
    id_col = "image_id"
    if id_col not in out.columns:
        raise ValueError("Manifest missing required column 'image_id'.")
    y_col = "diagnosis" if "diagnosis" in out.columns else ("y_true" if "y_true" in out.columns else None)
    if y_col is None:
        raise ValueError("Manifest missing label column ('diagnosis' or 'y_true').")
    cohort_col = "dataset_key" if "dataset_key" in out.columns else ("cohort" if "cohort" in out.columns else None)
    if cohort_col is None:
        raise ValueError("Manifest missing cohort column ('dataset_key' or 'cohort').")

    out["image_id"] = out[id_col].astype(str)
    out["cohort"] = (
        out[cohort_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"messidor-2": "messidor2", "messidor_2": "messidor2", "deep-drid": "deepdrid", "deep_drid": "deepdrid"})
    )
    out["y_true"] = pd.to_numeric(out[y_col], errors="raise").astype(int)
    if not out["y_true"].isin(range(5)).all():
        raise ValueError("Manifest contains invalid grade outside {0..4}.")

    if "image_key" not in out.columns:
        out["image_key"] = out["cohort"] + "::" + out["image_id"]
    else:
        out["image_key"] = out["image_key"].astype(str)

    if out["image_key"].duplicated().any():
        dups = out.loc[out["image_key"].duplicated(), "image_key"].head(5).tolist()
        raise ValueError(f"Duplicate image_key entries in manifest: {dups}")

    if "patient_key" in out.columns:
        out["patient_key"] = out["patient_key"].astype(str)
    elif "patient_id" in out.columns:
        out["patient_key"] = out["patient_id"].astype(str)
    else:
        out["patient_key"] = out["image_key"]

    if "patient_identity_status" not in out.columns:
        out["patient_identity_status"] = "legacy_unverified"

    if patient_map is not None:
        mapping = pd.read_csv(patient_map, dtype=str)
        if mapping["image_key"].duplicated().any() or mapping["patient_key"].isna().any():
            raise ValueError(f"Invalid verified patient mapping: {patient_map}")
        mp = mapping.set_index("image_key")["patient_key"]
        known = out["image_key"].isin(mp.index)
        out.loc[known, "patient_key"] = out.loc[known, "image_key"].map(mp)
        out.loc[known, "patient_identity_status"] = "verified"

    path_src_col = next((c for c in ("native_path", "raw_full_path", "original_relative_path") if c in out.columns), None)
    if path_src_col is not None:
        out["original_relative_path"] = out[path_src_col].astype(str).map(relative_image_path)
    if images_root is not None and "original_relative_path" in out.columns:
        root = Path(images_root)
        out["runtime_path"] = [str(root.joinpath(*PurePosixPath(p).parts)) for p in out["original_relative_path"]]
    return out


def verify_images_readable(
    df: pd.DataFrame,
    check_decode: bool = True,
    max_checks: Optional[int] = None,
) -> Dict[str, Any]:
    """Check file presence and OpenCV decode health; fail explicitly rather than skipping silently."""
    if "runtime_path" not in df.columns:
        raise ValueError("runtime_path column required to verify image files.")
    paths = df["runtime_path"].astype(str).tolist()
    if max_checks is not None:
        paths = paths[:max_checks]

    missing: List[str] = []
    unreadable: List[str] = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            missing.append(p)
            continue
        if check_decode:
            img = cv2.imread(str(fp))
            if img is None or img.size == 0:
                unreadable.append(p)

    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images; first 3: {missing[:3]}")
    if unreadable:
        raise OSError(f"{len(unreadable)} unreadable/corrupt images; first 3: {unreadable[:3]}")
    return {"checked": len(paths), "missing": 0, "unreadable": 0}


def build_natural_source_train(
    folds_dir: Path,
    fold_id: int,
    patient_map: Optional[Path] = None,
    images_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reconstruct natural-prevalence 85% source_train for fold_id from non-target test.csv - val.csv."""
    folds_dir = Path(folds_dir)
    if fold_id not in CANONICAL_FOLD_TO_COHORT:
        raise ValueError(f"Invalid fold_id={fold_id}")
    held_cohort = CANONICAL_FOLD_TO_COHORT[fold_id]

    va_raw = pd.read_csv(folds_dir / f"fold_{fold_id}_{held_cohort}_val.csv", dtype=str)
    va_df = normalize_manifest_df(va_raw, patient_map=patient_map, images_root=images_root)

    source_parts: List[pd.DataFrame] = []
    input_hashes: Dict[str, str] = {
        f"fold_{fold_id}_{held_cohort}_val.csv": sha256_file(folds_dir / f"fold_{fold_id}_{held_cohort}_val.csv")
    }
    for src_fold, src_cohort in CANONICAL_FOLD_TO_COHORT.items():
        if src_fold == fold_id:
            continue
        tp = folds_dir / f"fold_{src_fold}_{src_cohort}_test.csv"
        input_hashes[tp.name] = sha256_file(tp)
        part = normalize_manifest_df(pd.read_csv(tp, dtype=str), patient_map=patient_map, images_root=images_root)
        source_parts.append(part)

    pool_df = pd.concat(source_parts, ignore_index=True)
    if pool_df["image_key"].duplicated().any():
        raise ValueError(f"Duplicate image_key detected across source cohorts for fold {fold_id}.")

    va_keys = set(va_df["image_key"])
    nat_train = pool_df[~pool_df["image_key"].isin(va_keys)].copy().reset_index(drop=True)

    # Verify zero group overlap between reconstructed natural source_train and source_val
    nat_groups = set(nat_train["patient_key"])
    va_groups = set(va_df["patient_key"])
    overlap = nat_groups & va_groups
    if overlap:
        raise ValueError(f"Fold {fold_id} grouping-key overlap between natural source_train and val: {len(overlap)}")

    nat_counts = {int(k): int((nat_train["y_true"] == k).sum()) for k in range(5)}
    meta = {
        "fold_id": fold_id,
        "held_out_cohort": held_cohort,
        "pool_images": int(len(pool_df)),
        "natural_source_train_images": int(len(nat_train)),
        "source_val_images": int(len(va_df)),
        "realized_train_ratio": float(len(nat_train) / len(pool_df)),
        "natural_grade_counts": nat_counts,
        "input_manifest_sha256": input_hashes,
        "reconstructed_manifest_sha256": object_hash(
            {"image_keys": nat_train["image_key"].tolist(), "y_true": nat_train["y_true"].tolist()}
        ),
    }
    return nat_train, meta


def audit_lodo_manifests(
    folds_dir: Path,
    images_root: Optional[Path] = None,
    check_decode: bool = False,
) -> Dict[str, Any]:
    """Comprehensive audit of LODO CSV manifests, group leakage, quota vs natural counts, and optional images."""
    folds_dir = Path(folds_dir)
    fold_reports: List[Dict[str, Any]] = []
    inventory_per_cohort: Dict[str, int] = {}

    for fold_id, cohort in CANONICAL_FOLD_TO_COHORT.items():
        p_tr = folds_dir / f"fold_{fold_id}_{cohort}_train.csv"
        p_va = folds_dir / f"fold_{fold_id}_{cohort}_val.csv"
        p_te = folds_dir / f"fold_{fold_id}_{cohort}_test.csv"
        for p in (p_tr, p_va, p_te):
            if not p.is_file():
                raise FileNotFoundError(f"Missing LODO manifest: {p}")

        tr = normalize_manifest_df(pd.read_csv(p_tr, dtype=str), images_root=images_root)
        va = normalize_manifest_df(pd.read_csv(p_va, dtype=str), images_root=images_root)
        te = normalize_manifest_df(pd.read_csv(p_te, dtype=str), images_root=images_root)

        if set(te["cohort"].unique()) != {cohort}:
            raise ValueError(f"Fold {fold_id} test manifest cohort mismatch: {te['cohort'].unique().tolist()}")
        if cohort in set(tr["cohort"]) or cohort in set(va["cohort"]):
            raise ValueError(f"Held-out cohort '{cohort}' leaked into source train/val on fold {fold_id}.")

        for name_a, da, name_b, db in [("train", tr, "val", va), ("train", tr, "test", te), ("val", va, "test", te)]:
            img_ov = set(da["image_key"]) & set(db["image_key"])
            grp_ov = set(da["patient_key"]) & set(db["patient_key"])
            if img_ov or grp_ov:
                raise ValueError(
                    f"Fold {fold_id} overlap ({name_a} vs {name_b}): "
                    f"images={len(img_ov)}, groups={len(grp_ov)}"
                )

        nat_tr, nat_meta = build_natural_source_train(folds_dir, fold_id, images_root=images_root)
        tr_in_nat = set(tr["image_key"]).issubset(set(nat_tr["image_key"]))
        if not tr_in_nat:
            raise ValueError(f"Fold {fold_id} quota train is not a subset of natural source_train.")

        # Label consistency check across quota train and natural source_train
        merged_lbl = tr[["image_key", "y_true"]].merge(
            nat_tr[["image_key", "y_true"]], on="image_key", suffixes=("_quota", "_nat")
        )
        if not (merged_lbl["y_true_quota"] == merged_lbl["y_true_nat"]).all():
            raise ValueError(f"Fold {fold_id} label discrepancy between train.csv and test/pool manifests.")

        if images_root is not None:
            verify_images_readable(te, check_decode=check_decode, max_checks=50 if check_decode else None)

        inventory_per_cohort[cohort] = int(len(te))
        fold_reports.append(
            {
                "fold_id": fold_id,
                "cohort": cohort,
                "quota_train_images": int(len(tr)),
                "quota_grade_counts": {int(k): int((tr["y_true"] == k).sum()) for k in range(5)},
                "natural_source_train_images": nat_meta["natural_source_train_images"],
                "natural_grade_counts": nat_meta["natural_grade_counts"],
                "source_val_images": int(len(va)),
                "target_images": int(len(te)),
                "target_g3_support": int((te["y_true"] == 3).sum()),
                "realized_natural_train_ratio": nat_meta["realized_train_ratio"],
                "quota_subset_of_natural": True,
                "sha256": {
                    "train": sha256_file(p_tr),
                    "val": sha256_file(p_va),
                    "test": sha256_file(p_te),
                },
            }
        )

    return {
        "total_target_evaluated_images": int(sum(inventory_per_cohort.values())),
        "target_counts_by_cohort": inventory_per_cohort,
        "folds": fold_reports,
    }


def validate_p0_cache(
    p0_dir: Path,
    folds: Sequence[int] = tuple(range(6)),
    variants: Sequence[str] = ("V2", "V3"),
    splits: Sequence[str] = ("source_train", "source_val", "target"),
    require_features: bool = False,
) -> Dict[str, Any]:
    """Verify P0 COMPLETE.json digests, finite probabilities, alignment, and optional z.npy arrays."""
    p0_dir = Path(p0_dir)
    lodo_dir = p0_dir / "lodo"
    if not lodo_dir.is_dir():
        raise FileNotFoundError(f"Missing P0 lodo/ directory at: {p0_dir}")

    checked_shards = 0
    ckpt_hashes: Dict[str, str] = {}
    for variant in variants:
        for fold in folds:
            cohort = CANONICAL_FOLD_TO_COHORT[fold]
            unit_dir = lodo_dir / f"{variant}_fold_{fold}_{cohort}"
            for split in splits:
                shard_dir = unit_dir / split
                comp_path = shard_dir / "COMPLETE.json"
                if not comp_path.is_file():
                    raise ValueError(f"Incomplete P0 shard (missing COMPLETE.json): {shard_dir}")
                meta = json.loads(comp_path.read_text(encoding="utf-8"))
                ident = meta.get("identity", {})
                ckpt_sha = ident.get("checkpoint_sha256")
                unit_key = f"{variant}_fold_{fold}"
                if ckpt_sha is not None:
                    if unit_key in ckpt_hashes and ckpt_hashes[unit_key] != ckpt_sha:
                        raise ValueError(f"Mixed checkpoint_sha256 in {unit_key}/{split}")
                    ckpt_hashes[unit_key] = ckpt_sha

                files_meta = meta.get("files", {})
                pred_path = shard_dir / "predictions.csv"
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Missing predictions.csv in {shard_dir}")
                if "predictions.csv" in files_meta and sha256_file(pred_path) != files_meta["predictions.csv"]:
                    raise ValueError(f"Corrupted/tampered predictions.csv digest in {shard_dir}")

                df = pd.read_csv(pred_path)
                if df.empty or df["image_key"].duplicated().any():
                    raise ValueError(f"Empty or duplicate image_key in {pred_path}")
                p = df[list(PROB_COLS)].to_numpy(dtype=np.float64)
                if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
                    raise ValueError(f"Invalid probabilities in {pred_path}")

                z_path = shard_dir / "z.npy"
                if require_features or z_path.is_file():
                    if not z_path.is_file():
                        raise FileNotFoundError(f"Missing z.npy in {shard_dir}")
                    if "z.npy" in files_meta and sha256_file(z_path) != files_meta["z.npy"]:
                        raise ValueError(f"Corrupted/tampered z.npy digest in {shard_dir}")
                    z = np.load(z_path, mmap_mode="r")
                    if z.ndim != 2 or z.shape[0] != len(df):
                        raise ValueError(f"Row mismatch z.npy {z.shape} vs predictions {len(df)} in {shard_dir}")
                    if not np.isfinite(np.asarray(z[: min(8, len(df))])).all():
                        raise ValueError(f"Non-finite values in {z_path}")

                checked_shards += 1

    return {"p0_dir": str(p0_dir), "checked_shards": checked_shards, "checkpoint_hashes": ckpt_hashes}


def save_predictions_table(df: pd.DataFrame, out_dir: Path, prefer_parquet: bool = True) -> Path:
    """Save canonical per-image predictions to compressed Parquet (with CSV fallback)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions DataFrame missing required columns: {missing}")

    p = df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("Refusing to write corrupted non-simplex probabilities.")

    if prefer_parquet:
        try:
            target = out_dir / "predictions.parquet"
            tmp = out_dir / f"predictions.{uuid4().hex[:8]}.parquet.tmp"
            df.to_parquet(tmp, index=False, compression="snappy")
            os.replace(tmp, target)
            return target
        except Exception:
            pass

    target = out_dir / "predictions.csv"
    tmp = out_dir / f"predictions.{uuid4().hex[:8]}.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, target)
    return target


def load_predictions_table(directory_or_file: Path) -> pd.DataFrame:
    p = Path(directory_or_file)
    if p.is_dir():
        pq = p / "predictions.parquet"
        cs = p / "predictions.csv"
        p = pq if pq.is_file() else cs
    if not p.is_file():
        raise FileNotFoundError(f"Predictions table not found: {directory_or_file}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_feature_bundle(
    z: np.ndarray,
    image_keys: Sequence[str],
    out_dir: Path,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Write mmap-friendly z.npy + keyed z_manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(z, dtype=dtype)
    if arr.ndim != 2 or arr.shape[0] != len(image_keys):
        raise ValueError(f"Feature shape {arr.shape} does not match {len(image_keys)} image keys.")
    if not np.isfinite(arr).all():
        raise ValueError("Refusing to cache non-finite latent feature array.")

    z_path = out_dir / "z.npy"
    tmp = out_dir / f"z.{uuid4().hex[:8]}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, z_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_file": "z.npy",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "sha256": sha256_file(z_path),
        "image_keys_sha256": object_hash(list(image_keys)),
    }
    atomic_write_json(out_dir / "z_manifest.json", manifest)
    return manifest

# %% [markdown]
# ## Experiment Workflow & Entry Points

# %%
import argparse
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


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
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_e3_brc_nb')
    print(run_e3_smoke(_out) if globals().get('SMOKE', True) else {'status': STATUS_SKIPPED_GATE_NOT_MET})

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
