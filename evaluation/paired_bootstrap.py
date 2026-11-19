"""Paired Grade-3 bootstrap used by the camera-ready analysis.

Intervals describe held-out image sampling, not training-seed variability.
Values are percentages; differences are percentage points. Cohorts are fixed.
"""

from typing import Any, Dict, Sequence

import numpy as np


def grade_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Reject missing, fractional, or out-of-range grades before any cast."""
    a = np.asarray(values)
    if a.ndim != 1 or not len(a):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    try:
        valid = np.isfinite(a).all() and np.equal(a, np.floor(a)).all()
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric grades") from exc
    if not valid or (a < 0).any() or (a > 4).any():
        raise ValueError(f"{name} must contain integer ICDR grades 0..4")
    return a.astype(np.int64)


def validate_pairs(y_true: np.ndarray, pred_v2: np.ndarray, pred_v3: np.ndarray):
    arrays = [grade_vector(x, n) for x, n in
              zip((y_true, pred_v2, pred_v3), ("y_true", "pred_v2", "pred_v3"))]
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Targets and paired predictions must have equal lengths")
    return arrays


def _summary(a, b, boot2, boot3, reps, seed, unit):
    c2, c3, delta = (np.percentile(x, [2.5, 97.5]) for x in
                     (boot2, boot3, np.asarray(boot3) - np.asarray(boot2)))
    # Preserve the archived float32 estimator, including percentile interpolation.
    return dict(
        n_grade3=len(a), sens_v2=round(float(a.mean()) * 100, 2),
        sens_v3=round(float(b.mean()) * 100, 2),
        diff_v3_minus_v2_pp=round((float(b.mean()) - float(a.mean())) * 100, 2),
        v2_ci_low=round(float(c2[0]), 2), v2_ci_high=round(float(c2[1]), 2),
        v3_ci_low=round(float(c3[0]), 2), v3_ci_high=round(float(c3[1]), 2),
        diff_ci_low_pp=round(float(delta[0]), 2), diff_ci_high_pp=round(float(delta[1]), 2),
        resampling_unit=unit, bootstrap_reps=reps, seed=seed,
    )


def paired_bootstrap_grade3_sensitivity(
    y_true: np.ndarray, pred_v2: np.ndarray, pred_v3: np.ndarray,
    reps: int = 2000, seed: int = 42,
) -> Dict[str, Any]:
    """Percentile CIs using identical Grade-3 image indices for both variants."""
    if reps < 1:
        raise ValueError("bootstrap replicates must be positive")
    y, p2, p3 = validate_pairs(y_true, pred_v2, pred_v3)
    mask = y == 3
    if not mask.any():
        raise ValueError("Grade-3 sensitivity is undefined: no Grade-3 cases")
    a, b = (p2[mask] == 3).astype(np.float32), (p3[mask] == 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    boot2, boot3 = [], []
    for _ in range(reps):
        idx = rng.integers(0, len(a), size=len(a))
        boot2.append(a[idx].mean() * 100.0)
        boot3.append(b[idx].mean() * 100.0)
    return _summary(a, b, boot2, boot3, reps, seed, "image_level")


def macro_bootstrap_grade3_sensitivity(
    cohort_data: Sequence[Dict[str, Any]], reps: int = 2000, seed: int = 42,
) -> Dict[str, Any]:
    """Resample within each fixed cohort, then average cohort sensitivities equally."""
    if not cohort_data or reps < 1:
        raise ValueError("At least one cohort and positive bootstrap replicates are required")
    groups = []
    for c in cohort_data:
        y, p2, p3 = validate_pairs(c["y_true"], c["pred_v2"], c["pred_v3"])
        if not np.any(y == 3):
            raise ValueError(f"No Grade-3 cases in {c['cohort']}; macro is undefined")
        groups.append(((p2[y == 3] == 3).astype(np.float32),
                       (p3[y == 3] == 3).astype(np.float32)))
    rng = np.random.default_rng(seed)
    boot2, boot3 = [], []
    for _ in range(reps):
        v2, v3 = [], []
        for a, b in groups:
            idx = rng.integers(0, len(a), size=len(a))
            v2.append(a[idx].mean())
            v3.append(b[idx].mean())
        boot2.append(float(np.mean(v2)) * 100)
        boot3.append(float(np.mean(v3)) * 100)
    a = np.asarray([a.mean() for a, _ in groups], dtype=np.float32)
    b = np.asarray([b.mean() for _, b in groups], dtype=np.float32)
    result = _summary(a, b, boot2, boot3, reps, seed, "stratified_by_cohort_macro")
    result.update(n_grade3=sum(len(a) for a, _ in groups),
                  cohort=f"Macro Unweighted ({len(groups)} Cohorts)", fold=-2)
    return result


def grade3_confusion(y_true: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
    """Count all G3 destinations; categorical referral is not threshold screening."""
    y, pred, _ = validate_pairs(y_true, prediction, prediction)
    counts = np.bincount(pred[y == 3], minlength=5)
    n, correct = int(counts.sum()), int(counts[3])
    errors, adjacent = n - correct, int(counts[2] + counts[4])
    return dict(n_actual_g3=n, **{f"g3_to_{i}": int(x) for i, x in enumerate(counts)},
                g3_to_01=int(counts[0] + counts[1]), g3_to_24=adjacent,
                g3_correct=correct, n_g3_misclassified=errors,
                pct_errors_assigned_to_grade24=adjacent / errors * 100 if errors else None)
