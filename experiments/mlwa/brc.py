"""Gated E3 Bounded Residual Correction (BRC) and mandatory capacity/retraining controls."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.mlwa.constants import STATUS_SKIPPED_GATE_NOT_MET
from experiments.mlwa.decoders import prior_from_labels
from experiments.mlwa.evaluator import compute_decision_metrics, compute_posterior_metrics
from experiments.mlwa.objectives import compute_cui_effective_weights


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
