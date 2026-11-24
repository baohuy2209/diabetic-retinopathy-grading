"""Decoupled model, head, link, and checkpoint adapters for MLWA journal experiments."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from experiments.mlwa.config import ModelSpec
from models.fusion import DualBranchFusion
from models.heads.clm import CumulativeLinkModelQWKHead as LegacyCLMHead
from models.heads.coral import CORALHead as LegacyCORALHead
from models.heads.corn import CORNHead as LegacyCORNHead
from models.heads.softmax import SoftmaxCEHead as LegacySoftmaxCEHead
from models.heads.softmax import SoftmaxQWKHead as LegacySoftmaxQWKHead
from models.streams.global_context import GlobalBranch
from models.streams.local_mil import LocalMILBranch


class JournalSoftmaxHead(nn.Module):
    """Pure Softmax categorical output head (probabilities only; decoder externalized)."""

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            logits = self.fc(x.float())
            probs = F.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "preds": torch.argmax(probs, dim=1),
        }


def _safe_clm_cdf(u: torch.Tensor, link: str) -> torch.Tensor:
    """Evaluate strictly valid FP32 CDF F(u) = P(Y <= k | u = theta_k - s) with finite gradients."""
    if link == "logit":
        return torch.sigmoid(u)
    if link == "cloglog":
        # Stable 1 - exp(-exp(u)): -expm1(-exp(u)) avoids cancellation near u -> -inf,
        # while smooth clamping keeps exp(u) finite cleanly without NaN gradients.
        u_safe = torch.clamp(u, min=-30.0, max=4.0)
        return -torch.expm1(-torch.exp(u_safe))
    if link == "probit":
        return 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
    raise ValueError(f"Unsupported CLM link '{link}'. Expected 'cloglog', 'logit', or 'probit'.")


class JournalCLMHead(nn.Module):
    """Cumulative Link Model head with explicit link separation (cloglog | logit | probit) in FP32."""

    def __init__(self, in_features: int = 512, num_classes: int = 5, link: str = "cloglog") -> None:
        super().__init__()
        if link not in ("cloglog", "logit", "probit"):
            raise ValueError(f"Unsupported CLM link '{link}'.")
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.link = link

        self.score_fc = nn.Linear(in_features, 1, bias=False)
        nn.init.zeros_(self.score_fc.weight)

        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)

    def get_monotonic_cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments.float()) + 1e-4
        pts = [self.first_cutpoint.float()]
        for inc in increments:
            pts.append(pts[-1] + inc)
        return torch.stack(pts)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            x_f32 = x.float()
            s = self.score_fc(x_f32).squeeze(1)
            cutpoints = self.get_monotonic_cutpoints()
            u = cutpoints.unsqueeze(0) - s.unsqueeze(1)
            cum_probs = _safe_clm_cdf(u, self.link)
            cum_probs = torch.clamp(cum_probs, min=0.0, max=1.0)
            rank_probs = 1.0 - cum_probs

            # Compute exact non-negative interval differences without altering ordering
            cols = [cum_probs[:, 0:1]]
            for k in range(1, self.num_cutpoints):
                cols.append((cum_probs[:, k : k + 1] - cum_probs[:, k - 1 : k]).clamp(min=0.0))
            cols.append((1.0 - cum_probs[:, -1:]).clamp(min=0.0))
            probs = torch.cat(cols, dim=1)
            probs_sum = probs.sum(dim=1, keepdim=True)
            probs = torch.where(probs_sum > 0.0, probs / probs_sum, torch.full_like(probs, 1.0 / self.num_classes))

            expected_grade = (probs * torch.arange(self.num_classes, device=x.device, dtype=torch.float32)).sum(dim=1)
            preds_round = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()

        return {
            "severity_score": s,
            "latent_score": s,
            "cutpoints": cutpoints,
            "thetas": cutpoints,
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds_round,
        }


class JournalCORALHead(nn.Module):
    """CORAL head that explicitly reports threshold crossings instead of silently masking them."""

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            rank_logits = self.linear(x.float()) + self.biases.float()
            raw_rank_probs = torch.sigmoid(rank_logits)
            preds = torch.sum(raw_rank_probs > 0.5, dim=1).long()

            # Compute raw consecutive differences BEFORE any repair
            cols = [1.0 - raw_rank_probs[:, 0:1]]
            for k in range(1, self.num_thresholds):
                cols.append(raw_rank_probs[:, k - 1 : k] - raw_rank_probs[:, k : k + 1])
            cols.append(raw_rank_probs[:, -1:])
            raw_interval_diffs = torch.cat(cols, dim=1)

            crossing_amount = (-torch.diff(raw_rank_probs, dim=1)).clamp(min=0.0)
            # Note: valid non-increasing sequence has P(y > k) <= P(y > k-1) => diff <= 0
            positive_jump = torch.diff(raw_rank_probs, dim=1) > 1e-6
            has_crossing = positive_jump.any(dim=1)
            probs = raw_interval_diffs.clamp(min=0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return {
            "rank_logits": rank_logits,
            "rank_probs": raw_rank_probs,
            "raw_interval_diffs": raw_interval_diffs,
            "crossing_amount": crossing_amount,
            "has_threshold_crossing": has_crossing,
            "monotonicity_violation_rate": has_crossing.float().mean(),
            "probs": probs,
            "preds": preds,
        }


class JournalCORNHead(nn.Module):
    """CORN conditional ordinal head preserving strict rank consistency."""

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            cond_logits = self.fc(x.float())
            cond_probs = torch.sigmoid(cond_logits)
            cum_probs = torch.cumprod(cond_probs, dim=1)
            preds = torch.sum(cum_probs > 0.5, dim=1).long()

            cols = [1.0 - cum_probs[:, 0:1]]
            for k in range(1, self.num_thresholds):
                cols.append(cum_probs[:, k - 1 : k] - cum_probs[:, k : k + 1])
            cols.append(cum_probs[:, -1:])
            raw_interval_diffs = torch.cat(cols, dim=1)

            positive_jump = torch.diff(cum_probs, dim=1) > 1e-6
            has_crossing = positive_jump.any(dim=1)
            probs = raw_interval_diffs.clamp(min=0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return {
            "cond_logits": cond_logits,
            "cond_probs": cond_probs,
            "cum_probs": cum_probs,
            "rank_probs": cum_probs,
            "raw_interval_diffs": raw_interval_diffs,
            "has_threshold_crossing": has_crossing,
            "monotonicity_violation_rate": has_crossing.float().mean(),
            "probs": probs,
            "preds": preds,
        }


class JournalDualBranchEncoder(nn.Module):
    """4096-D global + 2560-D local -> 6656-D -> 512-D encoder with whole-bag patch chunking."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.architecture_id = spec.architecture_id
        self.use_local_branch = spec.use_local_branch
        self.latent_dim = spec.fusion_dim
        self.eval_patch_chunk_size = spec.eval_patch_chunk_size

        self.global_branch = GlobalBranch(
            pretrained=spec.pretrained,
            use_cbam=spec.use_cbam,
            use_nonlocal=spec.use_nonlocal,
            use_quadrant_tokens=spec.use_quadrant_tokens,
        )
        if spec.use_local_branch:
            self.local_mil_branch = LocalMILBranch(pretrained=spec.pretrained, use_dual_pooling=True)
            local_dim = self.local_mil_branch.out_dim
        else:
            self.local_mil_branch = None
            local_dim = 0

        self.fusion = DualBranchFusion(
            global_dim=self.global_branch.out_dim,
            local_dim=local_dim,
            fusion_dim=spec.fusion_dim,
            use_local_branch=spec.use_local_branch,
            dropout_rate=spec.dropout,
        )

    def _encode_local_whole_bag(
        self,
        local_patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode patches in chunks while keeping whole-bag max and gated-attention pooling intact."""
        mil = self.local_mil_branch
        assert mil is not None
        bsz, k_patches, c, h, w = local_patches.shape
        flat_patches = local_patches.view(bsz * k_patches, c, h, w)
        chunk = flat_patches.shape[0] if self.eval_patch_chunk_size is None else max(1, int(self.eval_patch_chunk_size))

        if not self.training and flat_patches.shape[0] > chunk:
            chunks = []
            for start in range(0, flat_patches.shape[0], chunk):
                feats = mil.features(flat_patches[start : start + chunk])
                chunks.append(torch.flatten(mil.gap(feats), 1))
            patch_feats = torch.cat(chunks, dim=0)
        else:
            patch_feats = torch.flatten(mil.gap(mil.features(flat_patches)), 1)

        patch_feats = patch_feats.view(bsz, k_patches, mil.patch_dim)
        if patch_mask is not None:
            masked_feats = patch_feats.masked_fill(~patch_mask.unsqueeze(-1), -1e4)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)

        if mil.use_dual_pooling:
            att_pooled = mil.gated_pool(patch_feats, patch_mask=patch_mask)
            return torch.cat([max_pooled, att_pooled], dim=1)
        return max_pooled

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f_global = self.global_branch(global_img)
        if self.use_local_branch and self.local_mil_branch is not None:
            if local_patches is None:
                raise ValueError("local_patches is required when use_local_branch=True.")
            f_local = self._encode_local_whole_bag(local_patches, patch_mask=patch_mask)
            return self.fusion(f_global, f_local)
        return self.fusion(f_global, None)


class JournalGlobalResNet50Encoder(nn.Module):
    """Single-branch ResNet-50 + 512-D projection (for explicit fallback tier only)."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.architecture_id = spec.architecture_id
        self.latent_dim = spec.fusion_dim
        weights = models.ResNet50_Weights.DEFAULT if spec.pretrained else None
        base = models.resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(2048, spec.fusion_dim),
            nn.LayerNorm(spec.fusion_dim),
            nn.Mish(),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.stem(global_img)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.proj(self.gap(out).flatten(1))


class JournalGradingModel(nn.Module):
    """Complete decoupled journal model pairing an encoder adapter and output head."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.architecture_id = spec.architecture_id
        self.preprocessing_id = spec.preprocessing_id
        self.forward_id = f"{spec.architecture_id}__{spec.head_type}_{spec.clm_link}"

        if spec.architecture_id == "dual_branch_mil_v1":
            self.encoder: nn.Module = JournalDualBranchEncoder(spec)
        elif spec.architecture_id == "global_resnet50_v1":
            self.encoder = JournalGlobalResNet50Encoder(spec)
        else:
            raise ValueError(f"Unsupported architecture_id '{spec.architecture_id}'.")

        if spec.head_type == "softmax":
            self.head: nn.Module = JournalSoftmaxHead(in_features=spec.fusion_dim, num_classes=5)
        elif spec.head_type == "clm":
            self.head = JournalCLMHead(in_features=spec.fusion_dim, num_classes=5, link=spec.clm_link)
        elif spec.head_type == "coral":
            self.head = JournalCORALHead(in_features=spec.fusion_dim, num_classes=5)
        elif spec.head_type == "corn":
            self.head = JournalCORNHead(in_features=spec.fusion_dim, num_classes=5)
        else:
            raise ValueError(f"Unsupported head_type '{spec.head_type}'.")

    def shared_encoder_digest(self) -> str:
        """SHA-256 of encoder state_dict tensors so shared initialization across cells is verifiable."""
        h = hashlib.sha256()
        for name, tensor in sorted(self.encoder.state_dict().items()):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent = self.encoder(global_img, local_patches=local_patches, patch_mask=patch_mask)
        out = self.head(latent)
        out["latent"] = latent
        return out


class PooledResNet50StemProjAdapter(nn.Module):
    """Strict adapter for historical pooled head ablation checkpoints using stem/layer1..4/proj keys."""

    architecture_id = "pooled_resnet50_stem_proj_v1"
    preprocessing_id = "ben_graham_green_512"

    def __init__(self, head_type: str) -> None:
        super().__init__()
        self.head_type = head_type
        base = models.resnet50(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.2),
        )
        mapping = {
            "Softmax_CE": lambda: LegacySoftmaxCEHead(512, 5),
            "Softmax_QWK": lambda: LegacySoftmaxQWKHead(512, 5, qwk_weight=0.4),
            "CORAL": lambda: LegacyCORALHead(512, 5),
            "CORN": lambda: LegacyCORNHead(512, 5),
            "CLM_QWK": lambda: LegacyCLMHead(512, 5, qwk_weight=0.4),
        }
        if head_type not in mapping:
            raise ValueError(f"Unknown pooled head_type '{head_type}'.")
        self.head = mapping[head_type]()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        z = self.proj(self.gap(out).flatten(1))
        res = self.head(z)
        res["latent"] = z
        return res
