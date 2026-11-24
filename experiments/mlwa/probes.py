"""Frozen-encoder E1 probe fitting, source-only C selection, permutation controls, and convergence logging."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.mlwa.evaluator import compute_decision_metrics, compute_posterior_metrics


def within_source_cohort_permutation(
    cohorts: Sequence[str],
    seed: int,
) -> np.ndarray:
    """Permute indices strictly within each source cohort (negative control preserves cohort sizes)."""
    c_arr = np.asarray(cohorts, dtype=str)
    rng = np.random.default_rng(seed)
    perm = np.arange(len(c_arr), dtype=int)
    for c in np.unique(c_arr):
        idx = np.flatnonzero(c_arr == c)
        perm[idx] = rng.permutation(idx)
    return perm


def fit_frozen_logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit StandardScaler and L2 multinomial logistic regression strictly on source_train."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=int)
    if x_tr.ndim != 2 or len(x_tr) != len(y_tr) or len(y_tr) == 0:
        raise ValueError("Invalid source_train feature/label dimensions.")

    scaler = StandardScaler().fit(x_tr)
    x_tr_s = scaler.transform(x_tr)
    classes = np.unique(y_tr)
    if len(classes) != 5:
        return {
            "converged": False,
            "reason": f"EXPECTED_5_CLASSES_GOT_{classes.tolist()}",
            "C": float(c_value),
            "scaler": scaler,
            "model": None,
        }

    clf = LogisticRegression(
        C=float(c_value),
        max_iter=int(max_iter),
        solver="lbfgs",
        random_state=int(random_state),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_tr_s, y_tr)
        had_conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    n_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else 0
    converged = (not had_conv_warn) and (n_iter < int(max_iter))
    return {
        "converged": bool(converged),
        "n_iter": n_iter,
        "reason": "OK" if converged else "CONVERGENCE_WARNING_OR_MAX_ITER",
        "C": float(c_value),
        "scaler": scaler,
        "model": clf,
    }


def predict_frozen_probe(fit_obj: Dict[str, Any], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate fitted scaler+classifier without refitting on val or target."""
    if not fit_obj.get("converged") or fit_obj.get("model") is None:
        raise RuntimeError(f"Cannot predict with non-converged/invalid probe: {fit_obj.get('reason')}")
    x_s = fit_obj["scaler"].transform(np.asarray(x, dtype=np.float64))
    probs = fit_obj["model"].predict_proba(x_s)
    preds = fit_obj["model"].classes_[np.argmax(probs, axis=1)].astype(int)
    return probs, preds


def run_source_selected_probe(
    x_source_train: np.ndarray,
    y_source_train: np.ndarray,
    x_source_val: np.ndarray,
    y_source_val: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    representation_id: str,
    fold_id: int,
    cohort: str,
    c_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0),
    selection_metric: str = "macro_recall",
    max_iter: int = 1000,
    control_permutation_seed: Optional[int] = None,
    source_train_cohorts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Full source-only E1 pipeline: scaler+weights on source_train, C selected on source_val."""
    if selection_metric not in ("macro_recall", "qwk"):
        raise ValueError(f"Unsupported probe selection_metric '{selection_metric}'.")

    y_fit = np.asarray(y_source_train, dtype=int).copy()
    if control_permutation_seed is not None:
        if source_train_cohorts is None:
            raise ValueError("source_train_cohorts required for within-cohort permutation control.")
        perm = within_source_cohort_permutation(source_train_cohorts, control_permutation_seed)
        y_fit = y_fit[perm]

    candidates: List[Dict[str, Any]] = []
    valid_fits: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for c_val in c_grid:
        fit_obj = fit_frozen_logistic_probe(x_source_train, y_fit, c_val, max_iter=max_iter)
        if not fit_obj["converged"]:
            candidates.append(
                {
                    "C": float(c_val),
                    "converged": False,
                    "reason": fit_obj["reason"],
                    "source_val_macro_recall": np.nan,
                    "source_val_qwk": np.nan,
                }
            )
            continue
        p_val, pred_val = predict_frozen_probe(fit_obj, x_source_val)
        dm_val = compute_decision_metrics(y_source_val, pred_val)
        candidates.append(
            {
                "C": float(c_val),
                "converged": True,
                "reason": "OK",
                "source_val_macro_recall": float(dm_val["macro_recall"]),
                "source_val_qwk": float(dm_val["qwk"]),
                "source_val_recall_g3": float(dm_val["recall_g3"]),
                "source_val_within1": float(dm_val["within1"]),
            }
        )
        valid_fits.append((float(c_val), fit_obj, dm_val))

    if not valid_fits:
        return {
            "status": "NO_CONVERGED_CANDIDATE",
            "representation": representation_id,
            "fold_id": fold_id,
            "cohort": cohort,
            "selection_metric": selection_metric,
            "control_seed": control_permutation_seed,
            "candidates": candidates,
        }

    # Select C on source_val ONLY; tie-break by smaller C (stronger regularization).
    best_c, best_fit, best_val_dm = max(
        valid_fits,
        key=lambda item: (float(item[2][selection_metric]), -float(item[0])),
    )
    p_tgt, pred_tgt = predict_frozen_probe(best_fit, x_target)
    tgt_dm = compute_decision_metrics(y_target, pred_tgt)
    tgt_pm = compute_posterior_metrics(y_target, p_tgt)

    return {
        "status": "COMPLETE",
        "representation": representation_id,
        "fold_id": fold_id,
        "cohort": cohort,
        "selection_metric": selection_metric,
        "control_seed": control_permutation_seed,
        "selected_C": best_c,
        "candidates": candidates,
        "source_val_metrics": best_val_dm,
        "target_decision_metrics": tgt_dm,
        "target_posterior_metrics": tgt_pm,
        "target_probs": p_tgt,
        "target_preds": pred_tgt,
    }
