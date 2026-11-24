"""Deterministic Bayes decoders, versioned CIBD selectors, stability bootstrap, and latent region math."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.mlwa.constants import PROB_COLS, THETA_COLS, TIE_TOL
from experiments.mlwa.evaluator import compute_decision_metrics

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
