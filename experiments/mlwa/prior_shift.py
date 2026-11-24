"""E5 Unlabelled Target Prior Adaptation (Saerens EM & Lipton BBSE) with strict label isolation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from experiments.mlwa.decoders import validate_probabilities


def _project_to_simplex_with_floor(vec: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = len(v)
    if floor * n >= 1.0:
        raise ValueError("Simplex floor exceeds 1/C.")
    rem = 1.0 - floor * n
    res = minimize(
        lambda x: 0.5 * float(np.sum((x - v) ** 2)),
        x0=np.full(n, 1.0 / n),
        bounds=[(floor, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
        method="SLSQP",
    )
    if res.success:
        out = np.clip(res.x, floor, 1.0)
        return out / out.sum()
    shifted = np.maximum(v - floor, 0.0)
    if shifted.sum() <= 0:
        return np.full(n, 1.0 / n)
    return floor + rem * (shifted / shifted.sum())


def estimate_prior_saerens_em(
    p_target: np.ndarray,
    pi_source: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-7,
    min_prior: float = 1e-6,
) -> Dict[str, Any]:
    """Saerens et al. (Neural Computation, 2002) EM prior estimation using UNLABELLED target posteriors only."""
    p = validate_probabilities(p_target)
    pi_s = np.asarray(pi_source, dtype=np.float64)
    if pi_s.shape != (5,) or (pi_s <= 0.0).any() or not np.allclose(pi_s.sum(), 1.0, atol=1e-6):
        raise ValueError("pi_source must be a strictly positive 5-class probability vector.")

    pi_t = pi_s.copy()
    converged = False
    n_iter = 0
    max_delta = float("inf")

    for it in range(1, int(max_iter) + 1):
        ratio = pi_t / pi_s
        unnorm = p * ratio[None, :]
        denom = unnorm.sum(axis=1, keepdims=True)
        if (denom <= 0.0).any() or not np.isfinite(denom).all():
            return {
                "method": "saerens_em",
                "status": "FAILED_NUMERICAL_DEGENERACY",
                "converged": False,
                "pi_target": np.full(5, np.nan),
                "iterations": it,
            }
        post = unnorm / denom
        pi_next = post.mean(axis=0)
        pi_next = np.maximum(pi_next, min_prior)
        pi_next /= pi_next.sum()
        max_delta = float(np.max(np.abs(pi_next - pi_t)))
        pi_t = pi_next
        n_iter = it
        if max_delta < float(tol):
            converged = True
            break

    return {
        "method": "saerens_em",
        "status": "CONVERGED" if converged else "MAX_ITER_REACHED",
        "converged": bool(converged),
        "iterations": int(n_iter),
        "max_delta": float(max_delta),
        "pi_target": pi_t,
        "kl_target_to_source": float(np.sum(pi_t * np.log(np.maximum(pi_t, 1e-12) / pi_s))),
    }


def estimate_prior_bbse(
    y_source_val: np.ndarray,
    p_source_val: np.ndarray,
    p_target: np.ndarray,
    mode: str = "soft",
    cond_threshold: float = 1e4,
    min_class_support: int = 5,
) -> Dict[str, Any]:
    """Lipton et al. (ICML 2018) Black Box Shift Estimation with explicit conditioning & rank audits.

    Target labels are strictly NOT accepted by this API.
    """
    ys = np.asarray(y_source_val, dtype=int)
    ps = validate_probabilities(p_source_val)
    pt = validate_probabilities(p_target)

    if mode not in ("hard", "soft"):
        raise ValueError(f"Unsupported BBSE mode '{mode}'; expected 'hard' or 'soft'.")

    support = np.bincount(ys, minlength=5)
    if (support < int(min_class_support)).any():
        return {
            "method": f"bbse_{mode}",
            "status": "INSUFFICIENT_SOURCE_CLASS_SUPPORT",
            "identifiable": False,
            "class_support": support.tolist(),
            "pi_target": np.full(5, np.nan),
        }

    pi_source = support.astype(np.float64) / float(len(ys))
    # Build 5 x 5 joint confusion matrix C_{i, j} = P(yhat = i, y = j) on source_val
    c_mat = np.zeros((5, 5), dtype=np.float64)
    if mode == "hard":
        yhat_s = ps.argmax(axis=1)
        for i in range(5):
            for j in range(5):
                c_mat[i, j] = float(np.mean((yhat_s == i) & (ys == j)))
        mu_target = np.bincount(pt.argmax(axis=1), minlength=5).astype(np.float64) / float(len(pt))
    else:
        for j in range(5):
            mask = ys == j
            c_mat[:, j] = ps[mask].mean(axis=0) * pi_source[j]
        mu_target = pt.mean(axis=0)

    rank = int(np.linalg.matrix_rank(c_mat))
    cond_num = float(np.linalg.cond(c_mat)) if rank == 5 else float("inf")
    if rank < 5:
        return {
            "method": f"bbse_{mode}",
            "status": "RANK_DEFICIENT_NOT_IDENTIFIABLE",
            "identifiable": False,
            "rank": rank,
            "condition_number": cond_num,
            "class_support": support.tolist(),
            "pi_target": np.full(5, np.nan),
        }
    if cond_num > float(cond_threshold):
        return {
            "method": f"bbse_{mode}",
            "status": "UNSTABLE_CONDITIONING",
            "identifiable": False,
            "rank": rank,
            "condition_number": cond_num,
            "class_support": support.tolist(),
            "pi_target": np.full(5, np.nan),
        }

    # Solve C * w = mu_target directly, then pi_target = w * pi_source constrained to simplex
    w_raw = np.linalg.solve(c_mat, mu_target)
    pi_raw = w_raw * pi_source
    was_clipped_to_simplex = bool((pi_raw <= 1e-6).any() or not np.allclose(pi_raw.sum(), 1.0, atol=1e-3))

    # Constrained simplex quadratic optimization: min_{pi in Delta} ||(C / pi_source) pi - mu_target||^2
    c_cond = c_mat / pi_source[None, :]
    res = minimize(
        lambda pi: 0.5 * float(np.sum((c_cond @ pi - mu_target) ** 2)),
        x0=_project_to_simplex_with_floor(pi_raw, floor=1e-6),
        bounds=[(1e-6, 1.0)] * 5,
        constraints={"type": "eq", "fun": lambda pi: float(np.sum(pi) - 1.0)},
        method="SLSQP",
    )
    pi_simplex = res.x / res.x.sum() if res.success else _project_to_simplex_with_floor(pi_raw, floor=1e-6)
    return {
        "method": f"bbse_{mode}",
        "status": "COMPLETE",
        "identifiable": True,
        "rank": rank,
        "condition_number": cond_num,
        "class_support": support.tolist(),
        "was_clipped_to_simplex": was_clipped_to_simplex,
        "pi_raw": pi_raw,
        "pi_target": pi_simplex,
        "pi_source_val": pi_source,
    }


def apply_prior_shift_ratio(
    p_target: np.ndarray,
    pi_source: np.ndarray,
    pi_target_hat: np.ndarray,
    cibd_tau_active: float = 0.0,
) -> np.ndarray:
    """Reweight posteriors by (pi_target_hat / pi_source), blocking double-correction with CIBD tau."""
    if abs(float(cibd_tau_active)) > 1e-12:
        raise ValueError(
            "Refusing double prior correction: E5 (pi_target/pi_source) ratio cannot be multiplied "
            "simultaneously with CIBD pi**(-tau) when tau != 0."
        )
    p = validate_probabilities(p_target)
    ps = np.asarray(pi_source, dtype=np.float64)
    pt = np.asarray(pi_target_hat, dtype=np.float64)
    if (ps <= 0.0).any() or (pt <= 0.0).any() or not np.isfinite(pt).all():
        raise ValueError("Source and estimated target priors must be finite and strictly positive.")
    adj = p * (pt / ps)[None, :]
    return adj / adj.sum(axis=1, keepdims=True)
