"""Training objectives, symmetric QWK warmup, Cui CB-CE, Menon logit adjustment, and Galdran CSR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.mlwa.config import ObjectiveSpec
from losses.qwk import ContinuousQWKLoss


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
