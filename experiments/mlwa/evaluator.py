"""Authoritative MLWA evaluation engine: metrics, RDR screening, hierarchical aggregation, contrasts, and CIs."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve

from experiments.mlwa.constants import COHORT_ORDER


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
