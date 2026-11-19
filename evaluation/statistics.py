"""Statistical significance testing for matched LODO cross-cohort benchmarks."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


def paired_wilcoxon_test(
    scores_a: Union[List[float], np.ndarray],
    scores_b: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Computes paired Wilcoxon signed-rank test on matched cohort performance metrics.

    Args:
        scores_a: Performance metric vector across folds for Method A (e.g. V3).
        scores_b: Matched performance metric vector across folds for Method B (e.g. V0).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dictionary with test statistic, p-value, median difference, and effect size.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite one-dimensional paired arrays")
    if len(a) != len(b):
        raise ValueError(f"Matched vectors must have equal length: {len(a)} != {len(b)}")

    diff = a - b
    n = len(diff)

    if n < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median_difference": float(np.median(diff)) if n > 0 else 0.0,
            "mean_difference": float(np.mean(diff)) if n > 0 else 0.0,
            "sample_size": n,
            "interpretation": "Insufficient sample size (N < 3) for Wilcoxon signed-rank test.",
        }

    # If all differences are exactly 0
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "median_difference": 0.0,
            "mean_difference": 0.0,
            "sample_size": n,
            "interpretation": "Identical paired scores across all evaluation folds.",
        }

    nonzero = diff[diff != 0]
    # Exact small-sample distribution where the usual no-ties condition holds.
    exact = len(nonzero) == n and len(np.unique(abs(diff))) == n
    method = "exact" if exact else "approx"
    res = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox", method=method)
    stat, p_val = float(res.statistic), float(res.pvalue)
    ranks = stats.rankdata(abs(nonzero))
    effect_size = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())

    return {
        "statistic": stat,
        "p_value": p_val,
        "median_difference": float(np.median(diff)),
        "mean_difference": float(np.mean(diff)),
        "rank_biserial_effect_size": effect_size,
        "sample_size": n,
        "method": method,
        "scope": "Matched-cohort exploratory comparison; not training-seed replication; p-value unadjusted",
        "significant_at_05": bool(p_val < 0.05) if not np.isnan(p_val) else False,
    }


def compute_metric_summary(scores: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Computes mean, standard deviation, median, IQR, min, max across folds."""
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return {}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
