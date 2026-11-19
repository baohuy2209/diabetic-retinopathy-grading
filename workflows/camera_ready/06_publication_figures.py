# %% [markdown]
# # Export DMIP and camera-ready paper figures

# %%
from pathlib import Path
import os
from IPython.display import display

# Edit these defaults in Colab, or set the DR_* environment variables before running Python.
IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ

if IN_COLAB:
    import importlib.util
    import subprocess
    import sys
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'cv2': 'opencv-python-headless', 'albumentations': 'albumentations', 'seaborn': 'seaborn'}
    missing = [package for module, package in dependencies.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

if IN_COLAB and not Path('/content/drive/MyDrive').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
THESIS_ROOT = Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()
INPUTS = Path(os.environ.get('DR_INPUTS', THESIS_ROOT / 'inputs')).resolve()
OUTPUTS = Path(os.environ.get('DR_OUTPUTS', THESIS_ROOT / 'outputs')).resolve()
WORK = Path(os.environ.get('DR_WORK', '/content/dr_thesis_work' if IN_COLAB else THESIS_ROOT / 'work')).resolve()

# %%
FOLD_SOURCE = Path(os.environ.get('DR_FOLDS', INPUTS / 'lodo_folds'))
REFERENCE = Path(os.environ.get('DR_REFERENCE', INPUTS / 'reference' / 'lodo_summary_with_calibrated_screening.json'))
FIGURES = ['slope', 'heatmap', 'waterfall']  # Also: dataset, latent, xai
DEFAULT_SUMMARY_JSON = REFERENCE
DEFAULT_CHECKPOINT = Path(os.environ.get('DR_FIGURE_CHECKPOINT', INPUTS / 'checkpoints' / 'lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth'))
LATENT_SCORES = Path(os.environ.get('DR_LATENT_SCORES', INPUTS / 'latent_scores_champion_fold_1.npz'))
DEFAULT_IDRID_DIR = Path(os.environ.get('DR_IDRID', INPUTS / 'idrid_segmentation'))
DEFAULT_CAMERA_READY_DIR = DEFAULT_OUTPUT_DIR = OUTPUTS / 'camera_ready' / 'figures'

# %%
import hashlib
import json
import shutil
import zipfile
from pathlib import PurePosixPath
from uuid import uuid4

def stage_archive(source, workspace):
    """Use an existing directory or extract an immutable ZIP to a versioned SSD folder."""
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        return source
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        index = [(m.filename, m.file_size, m.CRC) for m in members]
        key = hashlib.sha256(json.dumps(index).encode()).hexdigest()
        target = Path(workspace) / (source.stem + '_' + key[:16])
        ready = target / '.archive-complete'
        if ready.is_file() and ready.read_text() == key:
            return target
        temp = target.with_name(target.name + '.' + uuid4().hex + '.partial')
        temp.mkdir(parents=True)
        try:
            for m in members:
                parts = PurePosixPath(m.filename.replace(chr(92), '/')).parts
                if not parts or any(p in ('/', '..') or ':' in p for p in parts):
                    raise ValueError(f'Invalid ZIP member: {m.filename}')
                dest = temp.joinpath(*parts)
                if m.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(m) as src, dest.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 2**20)
            (temp / '.archive-complete').write_text(key)
            try:
                temp.rename(target)
            except OSError:
                if not ready.is_file() or ready.read_text() != key:
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return target

def containing(root, filename):
    if (root / filename).is_file():
        return root
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f'Expected one {filename} below {root}, found {len(matches)}')
    return matches[0].parent

# %%
"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - DUAL-BRANCH MIL MODEL ARCHITECTURE
================================================================================
Core Research Focus: Dual-Branch Multi-Scale Representation (Global Context + Local MIL).
Architecture Overview:
  1. Global Branch: ResNet-50 + NonLocalBlock2D + CBAM Attention + QuadrantTokenAggregator -> 
     F_Global (4096-d: 2048-d GAP + 2048-d Quadrant Tokens).
     (Designed for 512x512 full fundus to capture 4-quadrant hemorrhages and 4-2-1 rule for Grade 3 / H2).
  2. Local MIL Branch: EfficientNet-B0 applied per sliding-window native patches -> GAP -> 
     Dual Pooling (1D Max-Pooling + Masked Gated Attention) -> F_Local (2560-d).
     (Designed for native-resolution patches to preserve focal microaneurysms for Grade 1 / H1).
  3. Multi-Scale Fusion: Concatenation [F_Global, F_Local] (6656-d) -> 
     Projection MLP (with LayerNorm + Mish) -> Shared Latent (512-d).
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple


# ==============================================================================
# 1. CBAM ATTENTION MODULE (CHANNEL & SPATIAL ATTENTION)
# ==============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using AvgPool & MaxPool, feeding through a shared MLP.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_dim = max(8, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Aggregates channel information via AvgPool & MaxPool along the channel axis,
    followed by a 7x7 spatial convolution.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) sequentially applying
    Channel Attention and Spatial Attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channel = x * self.channel_att(x)
        x_out = x_channel * self.spatial_att(x_channel)
        return x_out


# ==============================================================================
# 2. NON-LOCAL BLOCK & QUADRANT TOKEN AGGREGATOR (ARCHITECTURAL MECHANISMS FOR H2)
# ==============================================================================

class NonLocalBlock2D(nn.Module):
    """
    Embedded Gaussian Non-local block (Wang et al., 2018) for long-range spatial relational reasoning (H2).
    Computes full self-attention across the H*W feature map before CBAM local refinement.
    """
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.inter_channels = max(1, in_channels // reduction)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g     = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # Zero-init output projection to preserve pretrained backbone stability at start of training
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # (B, HW, C')
        phi   = self.phi(x).view(B, self.inter_channels, -1)                     # (B, C', HW)
        g     = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)      # (B, HW, C')

        attn = torch.softmax(torch.bmm(theta, phi) / (self.inter_channels ** 0.5), dim=-1)  # (B, HW, HW)
        y = torch.bmm(attn, g)                       # (B, HW, C')
        y = y.permute(0, 2, 1).view(B, self.inter_channels, H, W)
        y = self.out_conv(y)
        return x + y  # residual connection


class QuadrantTokenAggregator(nn.Module):
    """
    Splits feature map into 4 clinical retinal quadrants matching the ICDR 4-2-1 rule for Grade 3 (H2).
    Pools each into a quadrant token and applies a 1-layer Transformer encoder with quadrant positional embeddings.
    """
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.quadrant_pos_embed = nn.Parameter(torch.zeros(1, 4, in_channels))
        nn.init.trunc_normal_(self.quadrant_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        quadrants = [
            x[:, :, :h_mid, :w_mid],   # top-left
            x[:, :, :h_mid, w_mid:],   # top-right
            x[:, :, h_mid:, :w_mid],   # bottom-left
            x[:, :, h_mid:, w_mid:],   # bottom-right
        ]
        tokens = torch.stack([self.pool(q).flatten(1) for q in quadrants], dim=1)  # (B, 4, C)
        tokens = tokens + self.quadrant_pos_embed
        tokens = self.encoder(tokens)
        quadrant_summary = tokens.mean(dim=1)  # (B, C)
        return quadrant_summary


# ==============================================================================
# 3. GLOBAL CONTEXT BRANCH (RESNET-50 + NON-LOCAL + CBAM + QUADRANT TOKENS)
# ==============================================================================

class GlobalBranch(nn.Module):
    """
    Global Context Branch:
      Input: (B, 3, 512, 512) or (B, 3, 300, 300)
      Backbone: ResNet-50 (Pretrained on ImageNet)
      Attention: NonLocalBlock2D + CBAM applied to Layer 4 (2048 channels)
      Aggregation: GAP (2048-d) concatenated with Quadrant Token Aggregator (2048-d)
      Output: F_Global vector of dimension 4096 (2048 + 2048)
    """
    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_nonlocal: bool = True,
        use_quadrant_tokens: bool = True,
    ):
        super(GlobalBranch, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)
        
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4  # Output: (B, 2048, H/32, W/32)
        
        self.use_nonlocal = use_nonlocal
        if use_nonlocal:
            self.non_local = NonLocalBlock2D(in_channels=2048)

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(in_channels=2048, reduction_ratio=16)
            
        self.use_quadrant_tokens = use_quadrant_tokens
        if use_quadrant_tokens:
            self.quadrant_agg = QuadrantTokenAggregator(in_channels=2048)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048 + (2048 if use_quadrant_tokens else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, H', W')
        
        if self.use_nonlocal:
            x = self.non_local(x)
        if self.use_cbam:
            x = self.cbam(x)
            
        feat_gap = torch.flatten(self.gap(x), 1)  # (B, 2048)
        if self.use_quadrant_tokens:
            feat_quadrant = self.quadrant_agg(x)  # (B, 2048)
            feat = torch.cat([feat_gap, feat_quadrant], dim=1)  # (B, 4096)
        else:
            feat = feat_gap
        return feat


# ==============================================================================
# 4. GATED ATTENTION POOLING & LOCAL MIL BRANCH (DUAL POOLING)
# ==============================================================================

class GatedAttentionPool(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning (Ilse et al., 2018).
    Learns instance-level weights a_k = softmax(w^T (tanh(V h_k) * sigm(U h_k))).
    Enables lesion counting and spatial density aggregation for Severe DR (Grade 3 - H2).
    """
    def __init__(self, in_features: int = 1280, hidden_dim: int = 128):
        super(GatedAttentionPool, self).__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        a = self.attention_weights(a_v * a_u)  # (B, K, 1)
        if patch_mask is not None:
            mask_3d = patch_mask.unsqueeze(-1)
            a = a.masked_fill(~mask_3d, -1e4)
        a = torch.softmax(a, dim=1)
        weighted_out = torch.sum(x * a, dim=1)  # (B, in_features)
        return weighted_out


class LocalMILBranch(nn.Module):
    """
    Local Multi-Instance Learning (MIL) Branch with Dual Pooling:
      Input: (B, K, 3, 128, 128)
      Backbone: EfficientNet-B0 applied per patch
      Pooling: GAP per patch -> (B, K, 1280)
      Dual Aggregation:
        - 1D Max-Pooling (Captures extreme focal microaneurysms for Grade 1 - H1)
        - Gated Attention Pooling (Aggregates multi-lesion density for Grade 3 - H2)
      Output: F_Local feature vector of dimension 2560 (1280 + 1280)
    """
    def __init__(self, pretrained: bool = True, use_dual_pooling: bool = True):
        super(LocalMILBranch, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_effnet = models.efficientnet_b0(weights=weights)
        
        self.features = base_effnet.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_dim = 1280
        self.use_dual_pooling = use_dual_pooling
        
        if use_dual_pooling:
            self.gated_pool = GatedAttentionPool(in_features=self.patch_dim, hidden_dim=128)
            self.out_dim = self.patch_dim * 2  # 2560
        else:
            self.out_dim = self.patch_dim      # 1280

    def forward(self, patches: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, C, H, W = patches.shape
        flat_patches = patches.view(B * K, C, H, W)
        
        # Colab eval: keep every patch, bound CNN activation memory.
        chunk_size = getattr(self, "eval_patch_chunk_size", 32)
        if not self.training and flat_patches.shape[0] > chunk_size:
            chunks = []
            for start in range(0, flat_patches.shape[0], chunk_size):
                features = self.features(flat_patches[start:start + chunk_size])
                chunks.append(torch.flatten(self.gap(features), 1))
            patch_feats = torch.cat(chunks, dim=0)
        else:
            patch_feats = self.features(flat_patches)
            patch_feats = self.gap(patch_feats)
            patch_feats = torch.flatten(patch_feats, 1)
        patch_feats = patch_feats.view(B, K, self.patch_dim)
        
        # 1. 1D Max-Pooling (Focal micro-lesions for H1)
        if patch_mask is not None:
            masked_feats = patch_feats.masked_fill(~patch_mask.unsqueeze(-1), -1e4)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)
        
        if self.use_dual_pooling:
            # 2. Gated Attention Pooling (Lesion counting for H2)
            att_pooled = self.gated_pool(patch_feats, patch_mask=patch_mask)
            f_local = torch.cat([max_pooled, att_pooled], dim=1)
        else:
            f_local = max_pooled
            
        return f_local


# ==============================================================================
# 5. MULTI-SCALE DUAL-BRANCH FUSED ARCHITECTURE
# ==============================================================================

class DualBranchMILModel(nn.Module):
    """
    Unified Dual-Branch Multi-Scale Deep Learning Network:
      Combines Global Context (ResNet-50 + NonLocal + CBAM + Quadrant Tokens: 4096-d) and Local MIL (2560-d).
      Combined dimension: 4096 + 2560 = 6656-d -> LayerNorm MLP -> 512-d Latent Embedding.
    """
    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_dual_pooling: bool = True,
        fusion_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        super(DualBranchMILModel, self).__init__()
        
        self.global_branch = GlobalBranch(pretrained=pretrained, use_cbam=use_cbam)
        self.local_mil_branch = LocalMILBranch(pretrained=pretrained, use_dual_pooling=use_dual_pooling)
        
        combined_dim = self.global_branch.out_dim + self.local_mil_branch.out_dim  # 4096 + 2560 = 6656
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.latent_dim = fusion_dim

    def forward(
        self, 
        global_img: torch.Tensor, 
        local_patches: torch.Tensor, 
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f_global = self.global_branch(global_img)
        f_local = self.local_mil_branch(local_patches, patch_mask=patch_mask)
        f_fused = torch.cat([f_global, f_local], dim=1)
        latent = self.fusion_layer(f_fused)
        return latent

# %%
"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - LOSSES & ORDINAL OUTPUT HEADS
================================================================================
Core Research Focus: Ordinal Regression Output Head Ablation & Differentiable QWK Loss.
Supported Heads & Loss Formulations:
  1. Softmax_CE: Linear(5) + Standard/Weighted Cross-Entropy Loss.
  2. Softmax_QWK: Linear(5) + Cross-Entropy + Continuous Differentiable QWK Loss.
  3. CORAL: Consistent Rank Logits (Shared weight, independent threshold biases) + Ordinal BCE.
  4. CORN: Conditional Ordinal Regression Neural Network + Conditional Cross-Entropy.
  5. CLM_QWK (Champion): Cumulative Link Model (Complementary log-log link with monotonic
     cutpoints) + Interval NLL Loss + Continuous QWK Loss with linear warmup schedule.
  6. Rank Monotonicity Violations Metric Function.
  7. End-to-End Pipeline Wrapper: DRFullModel integrating DualBranchMILModel with any Head.
================================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union


# ==============================================================================
# 0. CLINICAL RANK MONOTONICITY VIOLATION METRIC
# ==============================================================================

def rank_monotonicity_violations(rank_probs: np.ndarray) -> float:
    """
    Computes the proportion of patient samples that violate the strict clinical
    monotonicity constraint on cumulative rank probabilities P(y > k).
    
    Constraint: P(y > 0) >= P(y > 1) >= P(y > 2) >= P(y > 3)  (Strictly non-increasing)
    
    Args:
        rank_probs: Array of shape (N, C-1) containing P(y > k) for k = 0..C-2.
        
    Returns:
        float: Fraction of samples (in [0.0, 1.0]) with at least one threshold violation.
    """
    if rank_probs is None or len(rank_probs) == 0:
        return 0.0
    rank_probs = np.asarray(rank_probs, dtype=np.float64)
    if rank_probs.ndim == 1:
        rank_probs = rank_probs.reshape(1, -1)
        
    # diffs = P(y > k+1) - P(y > k). For non-increasing sequence, diffs must be <= 0.
    # A violation occurs if diffs > +1e-4.
    diffs = np.diff(rank_probs, axis=1)
    violations = np.any(diffs > 1e-4, axis=1)
    return float(np.mean(violations))


# ==============================================================================
# 1. CONTINUOUS DIFFERENTIABLE QUADRATIC WEIGHTED KAPPA (QWK) LOSS
# ==============================================================================

class ContinuousQWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss for PyTorch.
    Computes soft confusion matrix and quadratic penalty weighting:
      Loss = sum(W * O) / sum(W * E)
    """
    def __init__(self, num_classes: int = 5, eps: float = 1e-7):
        super(ContinuousQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        
        # Quadratic weight matrix W_ij = (i - j)^2 / (C - 1)^2
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("W", w)

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_probs: Softmax or interval probability distribution (B, num_classes).
            targets: Ground truth integer class labels (B,) or one-hot targets (B, num_classes).
        """
        device = pred_probs.device
        N = pred_probs.size(0)
        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()
            
        # Ensure probabilities sum to 1
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)
        
        # Soft observed confusion matrix: O = P^T * Y
        O = torch.matmul(pred_probs.t(), y_one_hot)  # (C, C)
        O = O / (N + self.eps)
        
        # Expected confusion matrix under independence: E = hist_pred * hist_true^T
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (N + self.eps)  # (1, C)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (N + self.eps)   # (1, C)
        E = torch.matmul(hist_pred.t(), hist_true)                        # (C, C)
        
        # Quadratic weighted numerator and denominator
        W = self.W.to(device)
        num = torch.sum(W * O)
        den = torch.sum(W * E) + self.eps
        
        # QWK Loss = num / den (Minimizing this maximizes Cohen's Kappa)
        qwk_loss = num / den
        return qwk_loss


# ==============================================================================
# 2. OUTPUT HEAD 1: SOFTMAX CLASSIFICATION (CE)
# ==============================================================================

class SoftmaxCEHead(nn.Module):
    """
    Standard Categorical Classification Head:
      Linear(latent_dim, 5) -> Cross-Entropy Loss.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(SoftmaxCEHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)  # (B, 5)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        # Compute empirical rank probabilities P(y > k) = sum_{j > k} p_j
        rank_probs = torch.zeros(x.size(0), self.num_classes - 1, device=x.device)
        for k in range(self.num_classes - 1):
            rank_probs[:, k] = probs[:, k+1:].sum(dim=1)
        return {"logits": logits, "probs": probs, "preds": preds, "rank_probs": rank_probs}

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = outputs["logits"]
        return F.cross_entropy(logits, targets, weight=class_weights)


# ==============================================================================
# 3. OUTPUT HEAD 2: SOFTMAX + CONTINUOUS QWK LOSS
# ==============================================================================

class SoftmaxQWKHead(nn.Module):
    """
    Softmax Classification Head trained with Hybrid Multitask Loss:
      Loss = Cross-Entropy + lambda * Continuous_QWK_Loss
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 1.0
    ):
        super(SoftmaxQWKHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.qwk_weight = qwk_weight
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float):
        self.qwk_weight = weight

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        # Expected continuous value for ordinal predictions
        continuous_pred = torch.sum(probs * torch.arange(self.num_classes, device=x.device), dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()
        rank_probs = torch.zeros(x.size(0), self.num_classes - 1, device=x.device)
        for k in range(self.num_classes - 1):
            rank_probs[:, k] = probs[:, k+1:].sum(dim=1)
        return {"logits": logits, "probs": probs, "preds": preds, "continuous_preds": continuous_pred, "rank_probs": rank_probs}

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = outputs["logits"]
        probs = outputs["probs"]
        
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights)
        qwk = self.qwk_loss(probs, targets)
        
        total_loss = ce_loss + self.qwk_weight * qwk
        return total_loss


# ==============================================================================
# 4. OUTPUT HEAD 3: CORAL (CONSISTENT RANK LOGITS)
# ==============================================================================

class CORALHead(nn.Module):
    """
    Consistent Rank Logits (CORAL) for Ordinal Regression (Cheng et al., 2020).
    Features:
      - Shared weight vector W across all rank thresholds.
      - Independent rank threshold biases b_0, b_1, ..., b_{K-2}.
      - Probability of exceeding rank k: P(y > k) = sigmoid(W^T x + b_k).
      - Note: Monotonicity violations can occur if biases learn disordered values.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORALHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # 4 thresholds for 5 classes
        
        # Shared linear weight (without bias)
        self.linear = nn.Linear(in_features, 1, bias=False)
        # Independent threshold biases
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        with torch.no_grad():
            self.biases.copy_(torch.linspace(1.5, -1.5, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_logit = self.linear(x)  # (B, 1)
        # Binary rank logits for each threshold: z_k = W^T x + b_k
        rank_logits = base_logit + self.biases  # (B, 4)
        raw_rank_probs = torch.sigmoid(rank_logits)  # P(y > k), shape (B, 4)
        
        # Predict class by counting number of satisfied thresholds (P(y > k) > 0.5)
        preds = torch.sum(raw_rank_probs > 0.5, dim=1).long()  # [0..4]
        
        # Reconstruct class probabilities: P(y = k) = P(y > k-1) - P(y > k)
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - raw_rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(raw_rank_probs[:, k - 1] - raw_rank_probs[:, k])
        probs[:, -1] = raw_rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        
        # Return raw_rank_probs (BEFORE ReLU) for genuine monotonicity violation auditing
        return {
            "rank_logits": rank_logits,
            "rank_probs": raw_rank_probs,
            "probs": probs,
            "preds": preds
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        rank_logits = outputs["rank_logits"]  # (B, 4)
        device = targets.device
        B = targets.size(0)
        
        # Encode targets into binary rank indicators: binary_targets[i, k] = 1 if target > k else 0
        binary_targets = torch.zeros(B, self.num_thresholds, device=device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
            
        bce_loss = F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")
        return bce_loss


# ==============================================================================
# 5. OUTPUT HEAD 4: CORN (CONDITIONAL ORDINAL REGRESSION NN)
# ==============================================================================

class CORNHead(nn.Module):
    """
    Conditional Ordinal Regression Neural Network (CORN - Shi et al., 2021).
    Models conditional probabilities: P(y > k | y >= k) = sigmoid(g(x)_k).
    Guarantees mathematically consistent cumulative probability chains via cumulative product.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORNHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # 4 conditional tasks
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cond_logits = self.fc(x)  # (B, 4)
        cond_probs = torch.sigmoid(cond_logits)  # P(y > k | y >= k)
        
        # Compute cumulative probabilities: P(y > k) = prod_{j=0}^k P(y > j | y >= j)
        cum_probs = torch.cumprod(cond_probs, dim=1)  # (B, 4)
        
        # Predict class via thresholding cumulative probabilities
        preds = torch.sum(cum_probs > 0.5, dim=1).long()  # [0..4]
        
        # Reconstruct interval class probabilities
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(cum_probs[:, k - 1] - cum_probs[:, k])
        probs[:, -1] = cum_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        
        return {
            "cond_logits": cond_logits,
            "cond_probs": cond_probs,
            "rank_probs": cum_probs,
            "probs": probs,
            "preds": preds
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cond_logits = outputs["cond_logits"]  # (B, 4)
        loss = 0.0
        valid_tasks = 0
        
        for k in range(self.num_thresholds):
            mask = (targets >= k)
            if mask.sum() > 0:
                subset_logits = cond_logits[mask, k]
                subset_targets = (targets[mask] > k).float()
                task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
                loss += task_loss
                valid_tasks += 1
                
        return loss / max(1, valid_tasks)


# ==============================================================================
# 6. OUTPUT HEAD 5: CLM_QWK (CHAMPION - CUMULATIVE LINK MODEL + QWK)
# ==============================================================================

class CumulativeLinkModelQWKHead(nn.Module):
    """
    Champion Output Architecture: Cumulative Link Model (CLM) with Complementary Log-Log (clog-log)
    Link Function & Hybrid Continuous QWK Loss with Linear Warmup.
    
    Mathematical Formulation:
      - Latent Severity Score: s(x) = W^T x (scalar continuous representation)
      - Strictly Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4 parameterized
        via softplus increments: theta_k = theta_{k-1} + softplus(alpha_k) + 1e-4
      - Complementary Log-Log (clog-log) link: F(z) = 1 - exp(-exp(z))
      - Cumulative Probability: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
      - Rank Probability: P(Y > k) = 1 - P(Y <= k) = exp(-exp(theta_k - s(x)))
      - Interval Probability: P(Y = k) = P(Y <= k) - P(Y <= k-1)
      - Loss: Negative Log-Likelihood (NLL) + lambda * Continuous_QWK_Loss
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 1.2
    ):
        super(CumulativeLinkModelQWKHead, self).__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1  # 4 cutpoints for 5 grades
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0  # Warmup starts at 0.0
        
        # Linear projection to single latent severity score
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        
        # Parameterize monotonic cutpoints using positive softplus increments
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)
        
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float):
        """Sets active QWK weight (for linear warmup schedule)."""
        self.current_qwk_weight = weight

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        """Computes strictly ordered cutpoints theta_1 < theta_2 < ... < theta_4."""
        increments = F.softplus(self.cutpoint_increments) + 1e-4  # Strictly positive
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)  # (4,)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Complementary log-log link function: F(z) = 1 - exp(-exp(z))."""
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        severity_score = self.score_fc(x).squeeze(1)  # (B,)
        cutpoints = self._get_monotonic_cutpoints()    # (4,)
        
        # Compute latent differences: (theta_k - s(x)) of shape (B, 4)
        z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
        
        # Cumulative probabilities P(Y <= k) for k = 0, 1, 2, 3
        cum_probs = self._cloglog_cdf(z)  # (B, 4)
        
        # Rank probabilities P(Y > k) = 1.0 - P(Y <= k)
        rank_probs = 1.0 - cum_probs      # (B, 4)
        
        # Compute discrete interval probabilities P(Y = k)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.num_cutpoints):
            probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
        probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        
        # Expected continuous grade prediction
        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * class_indices, dim=1)  # (B,)
        preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()
        
        return {
            "severity_score": severity_score,
            "cutpoints": cutpoints,
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        probs = outputs["probs"]  # (B, 5)
        B = targets.size(0)
        
        # 1. Negative Log-Likelihood of observed ordinal grade intervals
        eps = 1e-7
        target_probs = probs[torch.arange(B, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))
        
        if class_weights is not None:
            weights_per_sample = class_weights[targets]
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()
            
        # 2. Continuous Quadratic Weighted Kappa Loss
        qwk = self.qwk_loss(probs, targets)
        
        total_loss = nll_loss + self.current_qwk_weight * qwk
        return total_loss


# ==============================================================================
# 7. INTEGRATED PIPELINE WRAPPER MODEL FACTORY
# ==============================================================================

class DRFullModel(nn.Module):
    """
    End-to-End Deep Learning Architecture integrating DualBranchMILModel backbone
    with any of the 5 selectable Ordinal Regression Heads.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head_type: str = "CLM_QWK",
        num_classes: int = 5,
        qwk_weight: float = 1.2
    ):
        super(DRFullModel, self).__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.num_classes = num_classes
        latent_dim = getattr(backbone, "latent_dim", 512)
        
        # Instantiate requested head
        head_key = head_type.upper()
        if head_key in ["SOFTMAX_CE", "CE"]:
            self.head = SoftmaxCEHead(in_features=latent_dim, num_classes=num_classes)
        elif head_key in ["SOFTMAX_QWK", "QWK"]:
            self.head = SoftmaxQWKHead(in_features=latent_dim, num_classes=num_classes, qwk_weight=qwk_weight)
        elif head_key in ["CORAL"]:
            self.head = CORALHead(in_features=latent_dim, num_classes=num_classes)
        elif head_key in ["CORN"]:
            self.head = CORNHead(in_features=latent_dim, num_classes=num_classes)
        elif head_key in ["CLM_QWK", "CLM", "CHAMPION"]:
            self.head = CumulativeLinkModelQWKHead(in_features=latent_dim, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f"Unknown head_type: '{head_type}'. Choose from Softmax_CE, Softmax_QWK, CORAL, CORN, CLM_QWK.")

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        latent = self.backbone(global_img, local_patches, patch_mask=patch_mask)
        outputs = self.head(latent)
        outputs["latent"] = latent
        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.head.loss_fn(outputs, targets, class_weights=class_weights)

    def set_qwk_warmup(self, epoch: int, total_warmup_epochs: int = 5):
        """Linear warmup of QWK loss weight during early epochs."""
        if hasattr(self.head, "set_qwk_weight") and hasattr(self.head, "target_qwk_weight"):
            if epoch <= total_warmup_epochs:
                current_w = self.head.target_qwk_weight * (epoch / float(total_warmup_epochs))
            else:
                current_w = self.head.target_qwk_weight
            self.head.set_qwk_weight(current_w)


def build_model(
    head_type: str = "CLM_QWK",
    pretrained: bool = True,
    use_cbam: bool = True,
    use_dual_pooling: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 1.2
) -> DRFullModel:
    """
    Factory function to construct complete DR Multi-Scale Pipeline model with designated output head.
    """

        
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=use_cbam,
        use_dual_pooling=use_dual_pooling,
        fusion_dim=fusion_dim
    )
    
    model = DRFullModel(
        backbone=backbone,
        head_type=head_type,
        num_classes=num_classes,
        qwk_weight=qwk_weight
    )
    return model

# %%
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json

import albumentations as A
import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch

# ==============================================================================

# ==============================================================================
def configure_publication_style():
    """Configure publication style."""
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Helvetica",
                "Liberation Sans",
            ],
            "font.family": "sans-serif",
            "font.size": 11.0,
            "axes.labelsize": 12.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "legend.title_fontsize": 10.5,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#2d3748",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "grid.color": "#cbd5e1",
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    sns.set_theme(
        style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.45}
    )



C_NAVY = "#1a365d"
C_ORANGE = "#e65100"
C_CRIMSON = "#c53030"
C_PURPLE = "#6b21a8"
C_GRAY = "#94a3b8"
C_DARKGRAY = "#475569"
C_TEAL = "#0d9488"
C_EMERALD = "#16a34a"


def safe_save_figure(
    fig: plt.Figure, pdf_path: Path, png_path: Path, dpi: int = 300
) -> Tuple[Path, Path]:
    """Save a vector PDF and a PNG preview, using an alternate filename when the destination is locked."""
    actual_pdf = pdf_path
    try:
        fig.savefig(actual_pdf, format="pdf", bbox_inches="tight", dpi=dpi)
    except PermissionError:
        actual_pdf = pdf_path.with_name(
            f"{pdf_path.stem}_updated{pdf_path.suffix}"
        )
        print(
            f"[File locked] '{pdf_path.name}' is open in another program."
        )
        print(f"  -> Saved to alternate path: '{actual_pdf.name}'")
        fig.savefig(actual_pdf, format="pdf", bbox_inches="tight", dpi=dpi)

    actual_png = png_path
    try:
        fig.savefig(actual_png, format="png", bbox_inches="tight", dpi=dpi)
    except PermissionError:
        actual_png = png_path.with_name(
            f"{png_path.stem}_updated{png_path.suffix}"
        )
        print(
            f"[File locked] '{png_path.name}' is locked. Saving to '{actual_png.name}'..."
        )
        fig.savefig(actual_png, format="png", bbox_inches="tight", dpi=dpi)

    return actual_pdf, actual_png


# ==============================================================================


# ==============================================================================

# ==============================================================================
def plot_fig2_dataset_distribution_lodo(
    lodo_folds_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Path]:
    """Plot fig2 dataset distribution lodo."""
    configure_publication_style()
    if lodo_folds_dir is None:
        lodo_folds_dir = containing(stage_archive(FOLD_SOURCE, WORK / 'archives'), 'fold_0_aptos_test.csv')
    lodo_folds_dir = Path(lodo_folds_dir)

    dataset_keys = ["aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid"]
    test_cohort_dfs = [
        pd.read_csv(lodo_folds_dir / f"fold_{i}_{k}_test.csv")
        for i, k in enumerate(dataset_keys)
        if (lodo_folds_dir / f"fold_{i}_{k}_test.csv").exists()
    ]
    if not test_cohort_dfs:
        print("Missing split CSV files in", lodo_folds_dir)
        return None, None

    dataset_df = pd.concat(test_cohort_dfs, ignore_index=True)
    df = pd.crosstab(dataset_df["dataset_key"], dataset_df["diagnosis"])
    df.columns = [f"Grade {c}" for c in df.columns]
    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )


    total_counts = df.sum(axis=1)
    bars = ax1.barh(
        df.index,
        total_counts,
        color=C_NAVY,
        edgecolor="#0f172a",
        lw=0.9,
        alpha=0.88,
    )
    ax1.set_xlabel("Total Number of Images (Log Scale)", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_title(
        f"(a) Multi-Center Cohort Sizes ({len(dataset_df):,} Total Images)",
        fontsize=11.5,
        fontweight="bold",
    )
    for bar in bars:
        w = bar.get_width()
        ax1.text(
            w * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1e293b",
        )


    grade_colors = ["#3b82f6", C_ORANGE, "#ca8a04", C_CRIMSON, "#7e22ce"]
    bottom = np.zeros(len(df_pct))
    for i, col in enumerate(df_pct.columns):
        vals = df_pct[col].values
        ax2.barh(
            df_pct.index,
            vals,
            left=bottom,
            label=col,
            color=grade_colors[i],
            edgecolor="#1e293b",
            lw=0.6,
            alpha=0.9,
        )
        bottom += vals

    ax2.set_xlabel("Class Distribution (%)", fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_title(
        "(b) Clinical Label Shift Across Cohorts (ICDRS 0-4)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        title="Clinical Severity",
    )

    plt.tight_layout()
    if output_dir is None:
        output_dir = DEFAULT_CAMERA_READY_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "figure_2_dataset_distribution.pdf"
    png_path = out_path / "figure_2_dataset_distribution.png"
    actual_pdf, actual_png = safe_save_figure(fig, pdf_path, png_path, dpi=300)
    plt.close(fig)
    print(f"Exported figure 2 LODO Dataset:")
    print(f"  -> Vector PDF: {actual_pdf}")
    print(f"  -> Preview PNG: {actual_png}")
    return fig, actual_pdf


# ==============================================================================

# ==============================================================================
def plot_fig4_waterfall_qwk_decomposition(
    summary_json_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Path]:
    """Plot fig4 waterfall qwk decomposition."""
    configure_publication_style()
    if summary_json_path is None:
        summary_json_path = DEFAULT_SUMMARY_JSON
    summary_json_path = Path(summary_json_path)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        lodo_results = json.load(f)

    variant_slots = [
        ("V0_Baseline_Softmax_GlobalOnly", ["V0", "Baseline", "GlobalOnly"]),
        ("V1_PlusLocalMIL_H1", ["V1", "LocalMIL", "Local_MIL", "H1"]),
        (
            "V2_PlusGlobalContext_H2",
            ["V2", "GlobalContext", "NonLocal", "Quadrant", "H2"],
        ),
        (
            "V3_PlusOrdinalHead_H3_Champion",
            ["V3", "Champion", "CLM", "Ordinal", "H3"],
        ),
    ]

    emp_summary = {}
    for canonical_name, aliases in variant_slots:
        if canonical_name in lodo_results:
            v_data = lodo_results[canonical_name]
            fold_qwks = [
                float(f["ood_test_qwk"])
                for f in v_data.values()
                if "ood_test_qwk" in f
            ]
            if fold_qwks:
                emp_summary[canonical_name] = {
                    "mean": float(np.mean(fold_qwks)),
                    "std": float(np.std(fold_qwks)),
                }

    q0 = emp_summary["V0_Baseline_Softmax_GlobalOnly"]["mean"]
    q1 = emp_summary["V1_PlusLocalMIL_H1"]["mean"]
    q2 = emp_summary["V2_PlusGlobalContext_H2"]["mean"]
    q3 = emp_summary["V3_PlusOrdinalHead_H3_Champion"]["mean"]

    baseline_qwk = q0
    components = [
        ("+Native MIL (H1)", q1 - q0),
        ("+Global Attention (H2)", q2 - q1),
        ("+Ordinal CLM (H3)", q3 - q2),
    ]

    labels = (
        ["Vanilla Baseline (V0)"]
        + [c[0] for c in components]
        + ["Proposed Champion (V3)"]
    )
    running_total = baseline_qwk
    bottoms = [0.0]
    heights = [baseline_qwk]
    colors = [C_NAVY]

    for label, delta in components:
        if delta >= 0:
            bottoms.append(running_total)
            heights.append(delta)
            colors.append(C_EMERALD)
        else:
            bottoms.append(running_total + delta)
            heights.append(abs(delta))
            colors.append(C_CRIMSON)
        running_total += delta

    bottoms.append(0.0)
    heights.append(running_total)
    colors.append(C_TEAL)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(labels))
    width = 0.55

    bars = ax.bar(
        x,
        heights,
        width,
        bottom=bottoms,
        color=colors,
        edgecolor="#1e293b",
        lw=1.0,
        alpha=0.92,
    )

    for i in range(len(components)):
        ax.plot(
            [x[i] + width / 2, x[i + 1] - width / 2],
            [bottoms[i + 1], bottoms[i + 1]],
            color="#64748b",
            linestyle=":",
            lw=1.4,
        )

    for i, bar in enumerate(bars):
        h = bar.get_height()
        b = bar.get_y()
        val_str = (
            f"{h:.3f}"
            if (i == 0 or i == len(bars) - 1)
            else f"{components[i-1][1]:+.3f}"
        )
        y_pos = b + h / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            val_str,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=15, ha="right", fontsize=10, fontweight="bold"
    )
    ax.set_ylabel(
        "Out-of-Domain Quadratic Weighted Kappa (QWK)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if output_dir is None:
        output_dir = DEFAULT_CAMERA_READY_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "fig4_waterfall_qwk_decomposition.pdf"
    png_path = out_path / "fig4_waterfall_qwk_decomposition.png"
    actual_pdf, actual_png = safe_save_figure(fig, pdf_path, png_path, dpi=300)
    plt.close(fig)
    print(f"Exported figure 4 Waterfall QWK Decomposition:")
    print(f"  -> Vector PDF: {actual_pdf}")
    print(f"  -> Preview PNG: {actual_png}")
    return fig, actual_pdf


# ==============================================================================

# ==============================================================================
def plot_fig5_lodo_heatmap(
    summary_json_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Path]:
    """Plot fig5 lodo heatmap."""
    configure_publication_style()
    if summary_json_path is None:
        summary_json_path = DEFAULT_SUMMARY_JSON
    summary_json_path = Path(summary_json_path)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        lodo_results = json.load(f)

    target_dict = lodo_results["V3_PlusOrdinalHead_H3_Champion"]
    keys = list(target_dict.keys())
    names = [target_dict[k].get("held_out_dataset", k) for k in keys]

    def _get_calib(f_entry):
        for r in f_entry.get("ood_calibrated_screening", []):
            t = r.get("Target_Specificity")
            if t == 0.95 or t == "95%" or t == "95.0%":
                return r
        return {}

    calibs = [_get_calib(target_dict[k]) for k in keys]
    data = {
        "In-Domain Val QWK": [
            target_dict[k].get("in_domain_val_qwk") for k in keys
        ],
        "OOD Test QWK": [target_dict[k].get("ood_test_qwk") for k in keys],
        "Delta QWK Drop": [target_dict[k].get("delta_lodo") for k in keys],
        "OOD Within-1 Acc (%)": [
            100 * target_dict[k].get("ood_within_1_grade_acc") for k in keys
        ],
        "OOD Sens @ Val tau (%)": [
            100 * float(r.get("Test_Sensitivity", 0)) for r in calibs
        ],
        "OOD Specific @ same tau (%)": [
            100 * float(r.get("Test_Specificity", 0)) for r in calibs
        ],
    }
    df = pd.DataFrame(data, index=names).T

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        mask=df.isna(),
        cmap="YlGnBu",
        cbar_kws={"label": "Metric Value"},
        linewidths=1.2,
        linecolor="#1e293b",
        ax=ax,
    )
    ax.set_ylabel("Evaluated Metric", fontweight="bold", fontsize=11)
    ax.set_xlabel(
        "Held-Out External Test Cohort", fontweight="bold", fontsize=11
    )

    plt.tight_layout()
    if output_dir is None:
        output_dir = DEFAULT_CAMERA_READY_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "fig5_lodo_heatmap.pdf"
    png_path = out_path / "fig5_lodo_heatmap.png"
    actual_pdf, actual_png = safe_save_figure(fig, pdf_path, png_path, dpi=300)
    plt.close(fig)
    print(f"Exported figure 5 LODO Generalization Matrix Heatmap:")
    print(f"  -> Vector PDF: {actual_pdf}")
    print(f"  -> Preview PNG: {actual_png}")
    return fig, actual_pdf



# ==============================================================================
def plot_fig3_slope_chart(
    summary_json_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    variant_name: str = "V3_PlusOrdinalHead_H3_Champion",
) -> Tuple[plt.Figure, Path]:
    """Compare source and held-out grade sensitivity across the six folds."""
    configure_publication_style()

    if summary_json_path is None:
        summary_json_path = DEFAULT_SUMMARY_JSON
    summary_json_path = Path(summary_json_path)

    if not summary_json_path.exists():
        raise FileNotFoundError(summary_json_path)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if variant_name not in data:
        variant_name = list(data.keys())[0]

    v_data = data[variant_name]
    val_recalls_all = {g: [] for g in range(5)}
    ood_recalls_all = {g: [] for g in range(5)}

    for fold_metrics in v_data.values():
        vr = fold_metrics.get("val_per_grade_recall", {})
        or_ = fold_metrics.get("ood_per_grade_recall", {})
        for g in range(5):
            val_recalls_all[g].append(vr.get(f"grade_{g}", 0.0))
            ood_recalls_all[g].append(or_.get(f"grade_{g}", 0.0))


    val_macro = {g: float(np.mean(val_recalls_all[g])) for g in range(5)}
    ood_macro = {g: float(np.mean(ood_recalls_all[g])) for g in range(5)}

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    x_val, x_ood = 0.0, 1.0

    meta = {
        0: ("Grade 0", C_GRAY, 1.8, "--"),
        1: ("Grade 1 (Mild NPDR - H1)", C_ORANGE, 3.2, "-"),
        2: ("Grade 2", C_GRAY, 1.8, "--"),
        3: ("Grade 3 (Severe NPDR - H2)", C_CRIMSON, 3.2, "-"),
        4: ("Grade 4", C_GRAY, 1.8, "--"),
    }


    ax.axvline(x_val, color="#cbd5e1", lw=1.5, zorder=1)
    ax.axvline(x_ood, color="#cbd5e1", lw=1.5, zorder=1)

    for g in range(5):
        y1 = val_macro[g]
        y2 = ood_macro[g]
        delta = y2 - y1
        _, col, lw, ls = meta[g]
        is_emphasis = g in [1, 3]


        ax.plot(
            [x_val, x_ood],
            [y1, y2],
            color=col,
            lw=lw,
            linestyle=ls,
            zorder=4 if is_emphasis else 2,
        )
        ax.scatter(
            [x_val, x_ood],
            [y1, y2],
            color=col,
            s=80 if is_emphasis else 45,
            edgecolor="#1e293b",
            lw=1.0,
            zorder=5 if is_emphasis else 3,
        )


        ax.text(
            x_val - 0.03,
            y1,
            f"G{g}: {y1*100:.1f}%",
            ha="right",
            va="center",
            fontsize=15.0 if is_emphasis else 13.5,
            fontweight="bold" if is_emphasis else "normal",
            color=col if is_emphasis else C_DARKGRAY,
        )


        diff_val = round(delta * 100, 1)
        if diff_val < 0:
            delta_str = f"(−{abs(diff_val):.1f} pp)"
        elif diff_val > 0:
            delta_str = f"(+{diff_val:.1f} pp)"
        else:
            delta_str = "(0.0 pp)"
        ax.text(
            x_ood + 0.03,
            y2,
            f"G{g}: {y2*100:.1f}% {delta_str}",
            ha="left",
            va="center",
            fontsize=15.0 if is_emphasis else 13.5,
            fontweight="bold" if is_emphasis else "normal",
            color=col if is_emphasis else C_DARKGRAY,
        )

    ax.set_xlim(-0.42, 1.66)
    ax.set_ylim(0.20, 0.85)

    y_ticks = np.arange(0.20, 0.86, 0.10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y*100)}%" for y in y_ticks], fontsize=13.5)
    ax.set_xticks([x_val, x_ood])
    ax.set_xticklabels(
        ["Source validation", "Out-of-Domain (Held-out cohort)"],
        fontweight="bold",
        fontsize=15.0,
    )
    ax.set_ylabel(
        "Grade-Specific Sensitivity (%)", fontweight="bold", fontsize=15.0
    )
    ax.tick_params(axis="both", which="major", labelsize=13.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=C_ORANGE,
            lw=3.2,
            label="Grade 1 (Mild NPDR / Focal Lesions)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=C_CRIMSON,
            lw=3.2,
            label="Grade 3 (Severe NPDR / 4-Quadrant)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=C_GRAY,
            lw=1.8,
            linestyle="--",
            label="Grades 0, 2, 4 (Other grades)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=13.5,
    )


    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "fig2_camera_ready.pdf"
    png_path = out_path / "fig2_camera_ready.png"

    plt.tight_layout()
    actual_pdf, actual_png = safe_save_figure(fig, pdf_path, png_path, dpi=300)

    safe_save_figure(
        fig,
        out_path / "fig3_slope_chart.pdf",
        out_path / "fig3_slope_chart.png",
        dpi=300,
    )
    plt.close(fig)

    print(f"Exported figure 3:")
    print(f"  -> Vector PDF: {actual_pdf}")
    print(f"  -> Preview PNG: {actual_png}")
    return fig, actual_pdf


# ==============================================================================

# ==============================================================================
def extract_thetas_from_checkpoint(ckpt_path: Union[str, Path]) -> List[float]:
    """Extract thetas from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    c1 = sd["head.first_cutpoint"].item()
    increments = F.softplus(sd["head.cutpoint_increments"]).cpu().numpy()
    thetas = [float(c1)]
    for inc in increments:
        thetas.append(float(thetas[-1] + inc))
    return thetas


def plot_fig6_latent_spectrum(
    ckpt_path: Optional[Union[str, Path]] = None,
    in_scores: Optional[Dict[int, np.ndarray]] = None,
    ood_scores: Optional[Dict[int, np.ndarray]] = None,
    cutpoints: Optional[List[float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Path]:
    """Plot measured validation and target severity distributions with checkpoint cutpoints."""
    configure_publication_style()

    if ckpt_path is None:
        ckpt_path = DEFAULT_CHECKPOINT
    ckpt_path = Path(ckpt_path)


    if cutpoints is None:
        if ckpt_path.exists():
            cutpoints = extract_thetas_from_checkpoint(ckpt_path)
            print(
                f"[INFO] Loaded cutpoints from '{ckpt_path.name}': {cutpoints}"
            )
        else:
            raise FileNotFoundError(ckpt_path)
    th1, th2, th3, th4 = cutpoints

    default_scores_cache = LATENT_SCORES
    if in_scores is None or ood_scores is None:
        if default_scores_cache.exists():
            npz_data = np.load(default_scores_cache)
            if in_scores is None:
                in_scores = {g: npz_data[f"in_grade_{g}"] for g in range(5)}
            if ood_scores is None:
                ood_scores = {g: npz_data[f"ood_grade_{g}"] for g in range(5)}
            print(
                f"[INFO] Loaded measured severity scores: '{default_scores_cache.name}'"
            )
        else:
            raise FileNotFoundError(default_scores_cache)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    grade_colors = ["#2563eb", C_ORANGE, "#ca8a04", C_CRIMSON, "#7e22ce"]


    for g in range(5):
        if len(in_scores.get(g, [])) > 1:
            sns.kdeplot(
                in_scores[g],
                ax=ax,
                color=grade_colors[g],
                fill=True,
                alpha=0.32,
                lw=2.0,
                label=f"Grade {g} (In-Domain)",
            )


    for g in range(5):
        if len(ood_scores.get(g, [])) > 1:
            sns.kdeplot(
                ood_scores[g],
                ax=ax,
                color=grade_colors[g],
                linestyle="--",
                lw=1.8,
                alpha=0.85,
                label=f"Grade {g} (Out-of-Domain)" if g == 0 else "",
            )

    current_y_max = ax.get_ylim()[1]
    extended_y_max = current_y_max * 1.35
    ax.set_ylim(0, extended_y_max)


    for i, theta in enumerate(cutpoints):
        ax.axvline(theta, color=C_PURPLE, linestyle=":", lw=2.2, zorder=5)

        y_multipliers = [1.17, 0.95, 1.06, 0.85]
        y_pos = current_y_max * y_multipliers[i % len(y_multipliers)]

        label_text = rf"$\theta_{{{i}}} = {theta:.2f}$"
        ax.text(
            theta,
            y_pos,
            label_text,
            ha="center",
            va="bottom",
            fontsize=14.0,
            fontweight="bold",
            color=C_PURPLE,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.20",
                fc="#faf5ff",
                ec=C_PURPLE,
                lw=1.2,
                alpha=1.0,
            ),
        )

    ax.set_xlabel(
        r"1D Latent Severity Score: $s(\mathbf{x}) = \mathbf{w}^T \mathbf{z}$",
        fontsize=15.0,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Sample Distribution Density", fontsize=15.0, fontweight="bold"
    )
    ax.tick_params(axis="both", which="major", labelsize=13.5)

    handles = [
        Patch(
            facecolor="#2563eb",
            alpha=0.45,
            edgecolor="#1e293b",
            label="Source validation (solid)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#2563eb",
            linestyle="--",
            lw=2,
            label="Held-out target (dashed)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=C_PURPLE,
            linestyle=":",
            lw=2.2,
            label=r"Monotonic Cutpoints $\theta_0 < \theta_1 < \theta_2 < \theta_3$",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=13.5,
    )

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "fig3_camera_ready.pdf"
    png_path = out_path / "fig3_camera_ready.png"

    plt.tight_layout()
    actual_pdf, actual_png = safe_save_figure(fig, pdf_path, png_path, dpi=300)

    safe_save_figure(
        fig,
        out_path / "figure_6_latent_spectrum.pdf",
        out_path / "figure_6_latent_spectrum.png",
        dpi=300,
    )
    plt.close(fig)

    print(f"Exported figure 6:")
    print(f"  -> Vector PDF: {actual_pdf}")
    print(f"  -> Preview PNG: {actual_png}")
    return fig, actual_pdf


# ==============================================================================


# ==============================================================================


def apply_green_channel_ablation(
    img_rgb: np.ndarray, ablation_mode: str = "ben_graham_green"
) -> np.ndarray:
    """Apply green channel ablation."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)

    if ablation_mode == "ben_graham_green":
        blurred = cv2.GaussianBlur(g, (0, 0), 30)
        adjusted = cv2.addWeighted(g, 4, blurred, -4, 128)
        processed = cv2.merge([adjusted, adjusted, adjusted])
    elif ablation_mode == "clahe_green":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g)
        processed = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    else:
        processed = cv2.merge([g, g, g])

    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


def compute_fov_mask(img_np: np.ndarray, erode_px: int = 15) -> np.ndarray:
    """Compute a fundus field-of-view mask from the original RGB image."""
    if img_np.ndim == 3:
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np


    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        fov_mask = cleaned


    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        fov_mask = cv2.erode(fov_mask, kernel_erode)

    return (fov_mask > 0).astype(np.uint8)


def extract_mil_patches_sliding(
    img_np: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    mode: str = "eval",
    max_patches_eval: int = 256,
    return_anchors: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
    """Extract local patches and their locations for attention-map reconstruction."""
    h, w, _ = img_np.shape
    p = patch_size

    if fov_mask is None:
        fov_mask = compute_fov_mask(img_np, erode_px=15)

    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = 0, 0, h, w

    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    s = max(1, stride)
    y_starts = (
        list(range(y_min, y_min + content_h - p + 1, s))
        if content_h > p
        else [y_min]
    )
    x_starts = (
        list(range(x_min, x_min + content_w - p + 1, s))
        if content_w > p
        else [x_min]
    )

    if content_h > p and (y_min + content_h - p) not in y_starts:
        y_starts.append(y_min + content_h - p)
    if content_w > p and (x_min + content_w - p) not in x_starts:
        x_starts.append(x_min + content_w - p)

    cands, cand_anchors = [], []
    for y0 in y_starts:
        for x0 in x_starts:
            y1c, x1c = min(y0 + p, h), min(x0 + p, w)
            patch = img_np[y0:y1c, x0:x1c]
            mask_patch = fov_mask[y0:y1c, x0:x1c]
            fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
            if fov_ratio < fov_thresh:
                continue

            if patch.shape[0] < p or patch.shape[1] < p:
                pad_h = max(0, p - patch.shape[0])
                pad_w = max(0, p - patch.shape[1])
                patch = np.pad(
                    patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                )
            cands.append(patch)
            cand_anchors.append([y0, x0])

    if len(cands) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        patch = img_np[cy : cy + p, cx : cx + p]
        if patch.shape[0] < p or patch.shape[1] < p:
            patch = np.pad(
                patch,
                (
                    (0, max(0, p - patch.shape[0])),
                    (0, max(0, p - patch.shape[1])),
                    (0, 0),
                ),
                mode="reflect",
            )
        cands = [patch]
        cand_anchors = [[cy, cx]]

    if len(cands) > max_patches_eval:
        idx = np.linspace(0, len(cands) - 1, max_patches_eval).astype(int)
        selected = [cands[i] for i in idx]
        selected_anchors = [cand_anchors[i] for i in idx]
    else:
        selected = cands
        selected_anchors = cand_anchors

    patches_np = np.stack(selected, axis=0)
    patches_u8 = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()

    if return_anchors:
        return patches_u8, np.asarray(selected_anchors, dtype=np.int32)
    return patches_u8


def normalize_model_inputs(
    global_img: torch.Tensor, local_patches: torch.Tensor, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize model inputs."""
    mean_g = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_g = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    mean_l = mean_g.view(1, 1, 3, 1, 1)
    std_l = std_g.view(1, 1, 3, 1, 1)

    if global_img.dtype == torch.uint8:
        global_img = global_img.float().div(255.0).sub(mean_g).div(std_g)
    if local_patches.dtype == torch.uint8:
        local_patches = local_patches.float().div(255.0).sub(mean_l).div(std_l)

    return global_img, local_patches


class IDRiDXAIEvalDataset(Dataset):
    """Idridxaievaldataset."""

    def __init__(
        self,
        image_dirs: List[Union[str, Path]],
        mask_dirs: List[Union[str, Path]],
        img_size: int = 512,
        ablation_mode: str = "ben_graham_green",
    ):
        self.img_size = img_size
        self.ablation_mode = ablation_mode
        self.global_transform = A.Compose(
            [
                A.Resize(
                    height=img_size,
                    width=img_size,
                    interpolation=cv2.INTER_LINEAR,
                ),
            ]
        )

        self.samples = []
        for img_dir, mask_dir in zip(image_dirs, mask_dirs):
            img_dir = Path(img_dir)
            mask_dir = Path(mask_dir)
            if not img_dir.exists() or not mask_dir.exists():
                continue

            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]:
                    base_id = f.stem

                    for ext in [".png", ".tif", ".jpg", ".jpeg"]:
                        mask_candidate = mask_dir / f"{base_id}{ext}"
                        if mask_candidate.exists():
                            self.samples.append(
                                (base_id, str(f), str(mask_candidate))
                            )
                            break

        print(
            f"[IDRiD Dataset] Found {len(self.samples)} aligned image/mask pairs."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id, img_path, mask_path = self.samples[idx]

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Unreadable image: {img_path}")
        img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_np.shape

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Unreadable mask: {mask_path}")
        lesion_mask = (m > 0).astype(np.uint8)


        fov_mask_raw = compute_fov_mask(img_np, erode_px=0)
        ys, xs = np.where(fov_mask_raw)
        if len(ys) > 0 and len(xs) > 0:
            y0, y1, x0, x1 = (
                int(ys.min()),
                int(ys.max()) + 1,
                int(xs.min()),
                int(xs.max()) + 1,
            )
            img_np = img_np[y0:y1, x0:x1]
            lesion_mask = lesion_mask[y0:y1, x0:x1]

        fov_mask = compute_fov_mask(img_np, erode_px=15)
        processed_np = apply_green_channel_ablation(
            img_np, ablation_mode=self.ablation_mode
        )

        global_augmented = self.global_transform(image=processed_np)
        global_tensor = (
            torch.from_numpy(global_augmented["image"])
            .permute(2, 0, 1)
            .contiguous()
        )


        raw_fundus_resized = cv2.resize(
            img_np,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )


        local_patches, patch_anchors = extract_mil_patches_sliding(
            processed_np,
            fov_mask=fov_mask,
            patch_size=128,
            stride=96,
            fov_thresh=0.5,
            mode="eval",
            max_patches_eval=256,
            return_anchors=True,
        )

        mask_gt_resized = cv2.resize(
            lesion_mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return {
            "image_id": img_id,
            "global_img": global_tensor,
            "raw_fundus": raw_fundus_resized,
            "local_patches": local_patches,
            "native_shape": tuple(processed_np.shape[:2]),
            "anchors": torch.from_numpy(
                np.asarray(patch_anchors, dtype=np.int32)
            ),
            "mask_gt": torch.tensor(mask_gt_resized, dtype=torch.uint8),
        }


def load_champion_model_from_checkpoint(
    ckpt_path: Union[str, Path], device: str = "cpu"
) -> nn.Module:
    """Load champion model from checkpoint."""
    

    

    backbone = DualBranchMILModel(
        pretrained=False, use_cbam=True, use_dual_pooling=True, fusion_dim=512
    )
    model = DRFullModel(
        backbone=backbone, head_type="CLM_QWK", num_classes=5, qwk_weight=0.4
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)


    remap_sd = {}
    for k, v in sd.items():
        new_k = k.replace(
            "backbone.fusion.projection.", "backbone.fusion_layer."
        )
        remap_sd[new_k] = v

    model.load_state_dict(remap_sd, strict=True)
    model.to(device)
    model.eval()


    pool = model.backbone.local_mil_branch.gated_pool

    def hooked_pool_forward(
        x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        a_v = pool.attention_v(x)
        a_u = pool.attention_u(x)
        a = pool.attention_weights(a_v * a_u)
        if patch_mask is not None:
            mask_3d = patch_mask.unsqueeze(-1)
            a = a.masked_fill(~mask_3d, -1e4)
        a = torch.softmax(a, dim=1)
        pool.last_attention_weights = a.detach()
        return torch.sum(x * a, dim=1)

    pool.forward = hooked_pool_forward
    return model


class FundusGradCAM:
    """Fundusgradcam."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = model.backbone.global_branch.layer4[-1]
        self.gradient = None
        self.activation = None
        self.hook_handles = []

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activation = out

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    @torch.enable_grad()
    def generate_cam(
        self, global_img: torch.Tensor, local_patches: torch.Tensor
    ) -> np.ndarray:
        self.remove_hooks()
        self._register_hooks()
        self.model.eval()

        g_in = global_img.clone().detach().requires_grad_(True)
        outputs = self.model(g_in, local_patches)

        if isinstance(outputs, dict) and "severity_score" in outputs:
            score = outputs["severity_score"].flatten()[0]
        elif isinstance(outputs, dict) and "logits" in outputs:
            score = outputs["logits"].max()
        else:
            score = outputs.flatten()[0]

        self.model.zero_grad()
        score.backward(retain_graph=False)

        weights = self.gradient.float().mean(dim=(2, 3), keepdim=True)
        cam = (
            F.relu((weights * self.activation.float()).sum(dim=1))[0]
            .detach()
            .cpu()
            .numpy()
        )
        ptp = float(np.ptp(cam))
        if ptp > 1e-12:
            cam = (cam - cam.min()) / ptp
        else:
            cam = np.zeros_like(cam)

        self.remove_hooks()
        return cam.astype(np.float32)


def generate_attention_mil_saliency_map(
    model: nn.Module,
    global_img: torch.Tensor,
    local_patches: torch.Tensor,
    anchors: np.ndarray,
    canvas_size: Tuple[int, int] = (512, 512),
    patch_size: int = 128,
    native_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Generate attention mil saliency map."""
    if native_shape is None:
        native_shape = canvas_size

    model.eval()
    pool = model.backbone.local_mil_branch.gated_pool
    pool.last_attention_weights = None

    with torch.no_grad():
        model(global_img, local_patches)

    weights = getattr(pool, "last_attention_weights", None)
    if weights is None:
        return np.zeros(canvas_size, dtype=np.float32)

    weights = weights[0].detach().float().cpu().numpy().reshape(-1)
    anchors = np.asarray(anchors).reshape(-1, 2)

    h, w = map(int, native_shape)
    canvas = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros_like(canvas)

    for k, (y, x) in enumerate(anchors):
        if k >= len(weights):
            break
        y, x = int(y), int(x)
        canvas[
            y : min(y + patch_size, h), x : min(x + patch_size, w)
        ] += weights[k]
        counts[y : min(y + patch_size, h), x : min(x + patch_size, w)] += 1

    result = np.divide(
        canvas, counts, out=np.zeros_like(canvas), where=counts > 0
    )
    result = cv2.resize(
        result, (canvas_size[1], canvas_size[0]), interpolation=cv2.INTER_LINEAR
    )

    ptp = float(np.ptp(result))
    if ptp > 1e-12:
        result = (result - result.min()) / ptp
    else:
        result = np.zeros_like(result)

    return result.astype(np.float32)


def compute_pointing_game_hit(heatmap: np.ndarray, mask_gt: np.ndarray) -> bool:
    """Compute pointing game hit."""
    mask_bool = mask_gt > 0
    if not mask_bool.any():
        return False
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(mask_bool[peak_y, peak_x])


def plot_fig8_qualitative_xai(
    ckpt_path: Optional[Union[str, Path]] = None,
    idrid_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    num_samples: int = 5,
    device: str = "cpu",
) -> Tuple[plt.Figure, Path]:
    """Plot fig8 qualitative xai."""
    configure_publication_style()

    if ckpt_path is None:
        ckpt_path = DEFAULT_CHECKPOINT
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    if idrid_dir is None:
        idrid_dir = DEFAULT_IDRID_DIR
    idrid_dir = Path(idrid_dir)


    image_dirs = [
        idrid_dir / "DR_Testing_Set" / "Fundus Images",
        idrid_dir / "DR_Training_Set" / "Fundus Images",
        idrid_dir / "idrid_segmentation_total_images",
    ]
    mask_dirs = [
        idrid_dir / "DR_Testing_Set" / "Combined Masks",
        idrid_dir / "DR_Training_Set" / "Combined Masks",
        idrid_dir / "idrid_segmentation_total_masks",
    ]

    print(f"[XAI PIPELINE] Loading checkpoint: '{ckpt_path.name}'")
    model = load_champion_model_from_checkpoint(ckpt_path, device=device)
    cam_extractor = FundusGradCAM(model)

    print(
        f"[XAI PIPELINE] Loading IDRiD segmentation data from '{idrid_dir}'"
    )
    dataset = IDRiDXAIEvalDataset(
        image_dirs=image_dirs, mask_dirs=mask_dirs, img_size=512
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"[ERROR] No image/mask pairs in '{idrid_dir}'!"
        )

    n_samples = min(num_samples, len(dataset))
    print(
        f"[XAI PIPELINE] Computing saliency for {n_samples} observations."
    )

    records = []
    t_start = time.time()

    for idx in range(n_samples):
        item = dataset[idx]
        img_id = item["image_id"]
        raw_fundus = item["raw_fundus"]
        mask_gt = item["mask_gt"].cpu().numpy()

        g_norm, l_norm = normalize_model_inputs(
            item["global_img"].unsqueeze(0).to(device),
            item["local_patches"].unsqueeze(0).to(device),
            device=device,
        )


        cam_g = cam_extractor.generate_cam(g_norm, l_norm)
        cam_g_resized = cv2.resize(
            cam_g, (512, 512), interpolation=cv2.INTER_LINEAR
        )


        cam_m = generate_attention_mil_saliency_map(
            model=model,
            global_img=g_norm,
            local_patches=l_norm,
            anchors=item["anchors"].cpu().numpy(),
            canvas_size=(512, 512),
            patch_size=128,
            native_shape=item["native_shape"],
        )


        g_hit = compute_pointing_game_hit(cam_g_resized, mask_gt)
        m_hit = compute_pointing_game_hit(cam_m, mask_gt)


        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_m), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(raw_fundus, 0.62, heatmap_rgb, 0.38, 0)

        records.append(
            {
                "image_id": img_id,
                "raw_fundus": raw_fundus,
                "mask_gt": mask_gt,
                "cam_mil": cam_m,
                "cam_grad": cam_g_resized,
                "overlay": overlay,
                "g_hit": g_hit,
                "m_hit": m_hit,
            }
        )
        print(
            f"  [{idx+1}/{n_samples}] {img_id}: Grad-CAM Hit={g_hit}, Attention-MIL Hit={m_hit}"
        )

    cam_extractor.remove_hooks()
    print(
        f"[XAI PIPELINE] Extraction completed in {time.time() - t_start:.1f}s."
    )

    # ==============================================================================

    # ==============================================================================
    fig, axes = plt.subplots(n_samples, 5, figsize=(15.0, 3.1 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row in range(n_samples):
        rec = records[row]
        img_id = rec["image_id"]


        axes[row, 0].imshow(rec["raw_fundus"])
        axes[row, 0].set_ylabel(
            img_id, fontsize=11.5, fontweight="bold", labelpad=8
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])


        axes[row, 1].imshow(rec["mask_gt"], cmap="gray")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])


        axes[row, 2].imshow(rec["cam_mil"], cmap="inferno")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        # Badge Pointing Game Hit / Miss
        m_color = "#15803d" if rec["m_hit"] else "#b91c1c"
        m_text = "Pointing: HIT" if rec["m_hit"] else "Pointing: MISS"
        axes[row, 2].text(
            0.04,
            0.90,
            m_text,
            transform=axes[row, 2].transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25", fc=m_color, ec="none", alpha=0.85
            ),
        )


        axes[row, 3].imshow(rec["cam_grad"], cmap="jet")
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])
        # Badge Pointing Game Hit / Miss
        g_color = "#15803d" if rec["g_hit"] else "#b91c1c"
        g_text = "Pointing: HIT" if rec["g_hit"] else "Pointing: MISS"
        axes[row, 3].text(
            0.04,
            0.90,
            g_text,
            transform=axes[row, 3].transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25", fc=g_color, ec="none", alpha=0.85
            ),
        )


        axes[row, 4].imshow(rec["overlay"])
        axes[row, 4].set_xticks([])
        axes[row, 4].set_yticks([])


    col_headers = [
        "Fundus Photograph",
        "Expert Lesion Mask",
        "Local Attention-MIL",
        "Global Grad-CAM",
        "Pathological Overlay",
    ]
    for c, header in enumerate(col_headers):
        axes[n_samples - 1, c].set_xlabel(
            header, fontsize=11.5, fontweight="bold", labelpad=8
        )


    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "figure_8_xai_qualitative_grid.pdf"
    png_path = out_path / "figure_8_xai_qualitative_grid.png"

    plt.tight_layout()
    actual_pdf, actual_png = safe_save_figure(fig, pdf_path, png_path, dpi=300)
    plt.close(fig)

    print(f"Exported figure 8:")
    print(f"  -> Vector PDF: {actual_pdf}")
    print(f"  -> Preview PNG: {actual_png}")
    return fig, actual_pdf

# %%
for figure in FIGURES:
    if figure == 'dataset':
        plot_fig2_dataset_distribution_lodo(output_dir=DEFAULT_OUTPUT_DIR)
    elif figure == 'slope':
        plot_fig3_slope_chart(output_dir=DEFAULT_OUTPUT_DIR)
    elif figure == 'heatmap':
        plot_fig5_lodo_heatmap(output_dir=DEFAULT_OUTPUT_DIR)
    elif figure == 'waterfall':
        plot_fig4_waterfall_qwk_decomposition(output_dir=DEFAULT_OUTPUT_DIR)
    elif figure == 'latent':
        plot_fig6_latent_spectrum(output_dir=DEFAULT_OUTPUT_DIR)
    elif figure == 'xai':
        plot_fig8_qualitative_xai(output_dir=DEFAULT_OUTPUT_DIR)
    else:
        raise ValueError(f'Unknown figure: {figure}')
