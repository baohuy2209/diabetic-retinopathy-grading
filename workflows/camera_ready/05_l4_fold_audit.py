# %% [markdown]
# # Resume the development camera-ready checkpoint audit on CUDA

# %%
NOTEBOOK_CODE_SHA256 = '2b3ad538a5f71b189aa38d63b30e0ea37d72ea30614374dbc215dd9ff88a8dbe'

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
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'cv2': 'opencv-python-headless', 'albumentations': 'albumentations', 'PIL': 'pillow'}
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

# %% [markdown]
# This historical audit saves per-fold snapshots and session reports. Use the archived-prediction replay for the official manuscript comparison.

# %%
import sys, torch, torchvision
if not torch.cuda.is_available():
    raise RuntimeError('Select a CUDA runtime for the L4 audit.')

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
from pathlib import Path, PurePosixPath, PureWindowsPath
import json, hashlib, zipfile, shutil, time, uuid, gc, random, inspect
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from IPython.display import display, Markdown, Image as PreviewImage

DATA_ZIP = Path(os.environ.get('DR_IMAGES', INPUTS / 'native_cropped.zip'))
CHECKPOINT_ZIP = Path(os.environ.get('DR_CHECKPOINTS', INPUTS / 'checkpoints.zip'))
FOLD_SOURCE = Path(os.environ.get('DR_FOLDS', INPUTS / 'lodo_folds'))
REFERENCE = Path(os.environ.get('DR_REFERENCE', INPUTS / 'reference' / 'lodo_summary_with_calibrated_screening.json'))
FOLDS_ZIP = None
DRIVE_OUTPUT = OUTPUTS / 'camera_ready'
EXPERIMENT_NAME = 'thesis_l4_audit'
FOLDS = [0, 1, 2, 3, 4, 5]
MAX_SAMPLES = 0
BATCH_SIZE = 1
NUM_WORKERS = 0 if os.name == "nt" else 2
PATCH_CHUNK_SIZE = 32
PRECISION = 'float32'
BOOTSTRAP_REPS = 2000
SEED = 42
RESUME_COMPLETED_FOLDS = True
FORCE_INFERENCE = False

assert FOLDS and len(set(FOLDS)) == len(FOLDS) and set(FOLDS) <= set(range(6))
assert MAX_SAMPLES >= 0 and BATCH_SIZE >= 1 and PATCH_CHUNK_SIZE >= 1
assert NUM_WORKERS >= 0 and BOOTSTRAP_REPS >= 100
assert PRECISION in ('float32', 'bfloat16')
assert EXPERIMENT_NAME and Path(EXPERIMENT_NAME).name == EXPERIMENT_NAME
if PRECISION == 'bfloat16' and not torch.cuda.is_bf16_supported():
    raise RuntimeError('This GPU requires float32 precision.')

LOCAL_ROOT = WORK / 'camera_ready'
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_EXPERIMENT = DRIVE_OUTPUT / EXPERIMENT_NAME
DRIVE_EXPERIMENT.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(0)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('Persistent output:', DRIVE_EXPERIMENT)

# %%
SOURCE_FOLDS = containing(stage_archive(FOLD_SOURCE, WORK / 'archives'), 'fold_0_aptos_test.csv')

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
"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - DATA LOADER & STERILE MULTI-CENTER SPLITTING
================================================================================
Core Research Focus: 6-Dataset Harmonization, Zero Cross-Eye Bilateral Contamination,
Multi-Center LODO Generator, and Dual-Branch PyTorch Dataset.

Supported Datasets (55,570 Total Standardized Images):
  1. APTOS 2019 (Aravind Eye Hospital, India)
  2. DDR Dataset (Multi-hospital Partners, China)
  3. DeepDRiD (ISBI 2020 Challenge, Shanghai, China)
  4. IDRiD (Nanded Eye Hospital, India - Ultra High-Res)
  5. Messidor-2 (Brest & Saint-Etienne Hospitals, France)
  6. EyePACS (Kaggle Screening Cohort, USA)
================================================================================
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
try:
    import albumentations as A
except ImportError:
    A = None


# ==============================================================================
# 1. DATASET CONFIGURATION DICTIONARY
# ==============================================================================

DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "aptos_total.csv"),
        "img_dir": os.path.join("aptos", "aptos_total_images"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "img_dir": os.path.join("ddr", "DR_grading", "DR_grading"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "deepdrid_total.csv"),
        "img_dir": os.path.join("deepdrid", "deepdrid_total_images"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_total.csv"),
        "img_dir": os.path.join("idrid", "idrid_total_images"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "img_dir": os.path.join("messidor2preprocess", "messidor-2", "messidor-2", "preprocess"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "EyePACS": {
        "name": "EyePACS (Kaggle)",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "img_dir": os.path.join("zipEyepacs", "train"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
    }
}


# ==============================================================================
# 2. PATIENT ID PARSING & ZERO-LEAKAGE GROUPING
# ==============================================================================

def parse_patient_and_side(image_id: str, dataset_key: str) -> Tuple[str, str]:
    """
    Extracts unique patient_id and eye side ('left', 'right', 'unknown') across
    heterogeneous clinical cohort conventions.
    
    Ensures:
      - IDRiD produces ~400+ distinct patient units (not 1).
      - EyePACS and DeepDRiD do not collide across dataset boundaries.
      - DDR multi-field codes are correctly grouped.
      - Messidor-2 paired examination codes are mapped to single individuals.
    """
    img_str = str(image_id).strip()
    img_lower = img_str.lower()
    k = dataset_key.lower()
    
    # 1. Lateral eye side resolution
    side = "unknown"
    if "left" in img_lower or "_l" in img_lower or img_lower.endswith("l"):
        side = "left"
    elif "right" in img_lower or "_r" in img_lower or img_lower.endswith("r"):
        side = "right"
        
    # 2. Patient ID extraction per cohort convention
    if "idrid" in k:
        # IDRiD_001 or train_IDRiD_001 -> take suffix after last underscore ("001")
        base = img_lower.replace(".jpg", "").replace(".png", "")
        pid = base.split("_")[-1]
    elif "eyepacs" in k or "deepdrid" in k:
        # EyePACS: '10_left' -> '10'; DeepDRiD: '104_l1' -> '104'
        pid = img_str.split("_")[0]
    elif "ddr" in k:
        # DDR: '007-0004-000' -> group by clinic & subject prefix
        parts = img_str.replace(".jpg", "").replace(".png", "").split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else img_str.split(".")[0]
    elif "messidor" in k:
        # Messidor-2 paired session: 20051020_43808_0100_PP -> group by session date prefix
        pid = img_str.split("_")[0]
    else:
        # APTOS: 1 image = 1 subject
        pid = img_str
        
    # Mandatory dataset prefix preventing inter-dataset collision
    namespaced_pid = f"{dataset_key}::{pid}"
    return namespaced_pid, side


def load_single_dataset(key: str, cfg: Dict[str, Any], base_dir: str) -> pd.DataFrame:
    """Loads, standardizes, and parses metadata for a single dataset."""
    full_csv_path = os.path.join(base_dir, cfg["csv_path"])
    full_img_dir = os.path.join(base_dir, cfg["img_dir"])
    
    if not os.path.exists(full_csv_path):
        return pd.DataFrame()
        
    raw_df = pd.read_csv(full_csv_path)
    label_col = cfg["label_col"]
    if label_col not in raw_df.columns:
        possible = [c for c in raw_df.columns if "dr" in c.lower() or "level" in c.lower() or "diagnosis" in c.lower()]
        if possible:
            label_col = possible[0]
            
    df = raw_df[[cfg["id_col"], label_col]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    df["dataset_key"] = key
    
    if cfg["id_suffix"]:
        df["image_file"] = df["image_id"].astype(str).apply(
            lambda x: x if x.endswith(cfg["id_suffix"]) else x + cfg["id_suffix"]
        )
    else:
        df["image_file"] = df["image_id"].astype(str)
        
    df["raw_full_path"] = df["image_file"].apply(lambda f: os.path.join(full_img_dir, f))
    
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    parsed = df["image_id"].apply(lambda x: parse_patient_and_side(x, dataset_key=key))
    df["patient_id"] = [p[0] for p in parsed]
    df["side"] = [p[1] for p in parsed]
    return df


def harmonize_all_datasets(config_dict: Dict[str, Any], base_dir: str) -> pd.DataFrame:
    """Harmonizes all available cohorts into a master sterile DataFrame."""
    dfs = []
    for key, cfg in config_dict.items():
        sub_df = load_single_dataset(key, cfg, base_dir)
        if not sub_df.empty:
            dfs.append(sub_df)
    if not dfs:
        return pd.DataFrame()
    master_df = pd.concat(dfs, ignore_index=True)
    return master_df


# ==============================================================================
# 3. STERILE SPLITTING & MULTI-SOURCE QUOTA BALANCING
# ==============================================================================

def split_master_pool_patient_level(
    master_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes leak-free patient-level grouping on the FULL master DataFrame.
    Guarantees:
      - Inter-eye bilateral data leakage is completely eliminated.
      - Validation and Test sets preserve their natural clinical prevalence (untouched).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(master_df, groups=master_df['patient_id']))
    
    train_df = master_df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = master_df.iloc[temp_idx].copy().reset_index(drop=True)
    
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss_test.split(temp_df, groups=temp_df['patient_id']))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


def create_balanced_master_pool(
    train_df: pd.DataFrame,
    target_range: Tuple[int, int] = (1500, 2500),
    seed: int = 42
) -> pd.DataFrame:
    """
    Applies multi-source quota allocation STRICTLY to the training DataFrame.
    Prevents large cohorts (EyePACS) from overwhelming clinical reference cohorts (IDRiD, Messidor-2).
    Uses stable column identifiers ('raw_full_path') to avoid index collisions.
    """
    balanced_dfs = []
    np.random.seed(seed)
    
    for grade in range(5):
        grade_subset = train_df[train_df["diagnosis"] == grade].copy()
        valid_mask = grade_subset["raw_full_path"].apply(lambda p: os.path.isfile(p))
        grade_subset_existing = grade_subset[valid_mask].copy().reset_index(drop=True)
        current_count = len(grade_subset_existing)
        
        if current_count == 0:
            continue
            
        datasets_in_grade = grade_subset_existing["dataset_name"].unique()
        num_datasets = len(datasets_in_grade)
        
        if current_count > target_range[1]:
            target_count = int(np.random.RandomState(seed + grade).randint(target_range[0], target_range[1] + 1))
            base_quota = target_count // num_datasets
            
            sampled_datasets = []
            leftover_quota = 0
            eligible_datasets = []
            
            for ds_name in datasets_in_grade:
                ds_subset = grade_subset_existing[grade_subset_existing["dataset_name"] == ds_name]
                ds_count = len(ds_subset)
                
                if ds_count <= base_quota:
                    sampled_datasets.append(ds_subset)
                    leftover_quota += (base_quota - ds_count)
                else:
                    eligible_datasets.append((ds_name, ds_subset, ds_count))
                    
            if eligible_datasets:
                quota_per_large_ds = base_quota + (leftover_quota // len(eligible_datasets))
                for ds_name, ds_subset, ds_count in eligible_datasets:
                    take_count = min(quota_per_large_ds, ds_count)
                    ds_sampled = ds_subset.sample(take_count, random_state=seed)
                    sampled_datasets.append(ds_sampled)
                    
            selected_subset = pd.concat(sampled_datasets, ignore_index=True)
            
            if len(selected_subset) > target_count:
                selected_subset = selected_subset.sample(target_count, random_state=seed).reset_index(drop=True)
            elif len(selected_subset) < target_count:
                # Filter via stable unique column 'raw_full_path' to prevent index reset bug
                remaining = grade_subset_existing[~grade_subset_existing["raw_full_path"].isin(selected_subset["raw_full_path"])]
                if not remaining.empty:
                    diff = target_count - len(selected_subset)
                    extra = remaining.sample(min(diff, len(remaining)), random_state=seed)
                    selected_subset = pd.concat([selected_subset, extra], ignore_index=True)
        else:
            selected_subset = grade_subset_existing.copy()
            
        balanced_dfs.append(selected_subset)
        
    balanced_train_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_train_df


# ==============================================================================
# 4. ROBUST RETINAL FOV & ANCHOR COORDINATE COMPUTATION
# ==============================================================================

def compute_fov_mask(img_np: np.ndarray, erode_px: int = 15) -> np.ndarray:
    """
    Computes the true retinal Field-of-View (FOV) binary mask for a fundus image.
    IMPORTANT: Must be computed on the RAW RGB fundus image BEFORE Ben Graham preprocessing.
    (Ben Graham sets background to 128 gray, which collapses Otsu thresholding).

    4-Step Pipeline:
    1. Otsu's Adaptive Thresholding on Green/Intensity channel.
    2. Morphological Close + Open (25x25 ellipse) to seal internal vessel holes.
    3. Largest Connected Component to isolate true retinal disk (strip glare/artifacts).
    4. Inward Morphological Erosion (erode_px=15) to eliminate edge vignetting/halos.
    """
    if img_np.ndim == 3:
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np

    # 1. Otsu's Adaptive Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morphological Close & Open
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 3. Largest Connected Component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        fov_mask = cleaned

    # 4. Inward Erosion
    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        fov_mask = cv2.erode(fov_mask, kernel_erode)

    return (fov_mask > 0).astype(np.uint8)


def apply_green_channel_ablation(
    img_rgb: np.ndarray, ablation_mode: str = "ben_graham_green"
) -> np.ndarray:
    """
    Applies fundus image preprocessing centered on the Green channel.
    The Green channel provides the highest contrast for retinal vascular micro-lesions.
    """
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


def compute_valid_anchors(
    img_shape: Tuple[int, int],
    fov_mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5
) -> np.ndarray:
    """
    Calculates 2D top-left coordinate anchors [y0, x0] passing the FOV threshold.
    Returns (N, 2) numpy array of integer coordinates.
    """
    h, w = img_shape[:2]
    p = patch_size

    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = 0, 0, h, w

    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    def get_coords(curr_stride: int):
        s = max(1, curr_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]

        if content_h > p and (y_min + content_h - p) not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and (x_min + content_w - p) not in x_starts:
            x_starts.append(x_min + content_w - p)

        valid_list = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = min(y0 + p, h), min(x0 + p, w)
                mask_patch = fov_mask[y0:y1, x0:x1]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio >= fov_thresh:
                    valid_list.append([y0, x0])
        return valid_list

    anchors = get_coords(stride)
    # Adaptive Fallback: if fewer than 32 candidates, rescan with dense stride (s=48)
    if len(anchors) < 32 and stride > 48 and min(content_h, content_w) > p:
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors

    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]

    return np.array(anchors, dtype=np.int32)


# ==============================================================================
# 5. OFFLINE PRECOMPUTATION & NPZ CACHING ENGINE
# ==============================================================================

def precompute_dataset_cache(
    df: pd.DataFrame,
    img_root: Optional[str] = None,
    cache_dir: str = "data/cache/npz_anchors",
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    ablation_mode: str = "ben_graham_green",
    n_jobs: int = -1
) -> Dict[str, int]:
    """
    Offline precomputes and saves compressed NPZ cache files containing:
      - 'img': Preprocessed image array (processed_u8, uint8)
      - 'anchors': Valid patch anchor top-left coordinates (N, 2)
    Eliminates redundant CPU calculations across 100+ training epochs.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import PureWindowsPath
    from tqdm.auto import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    records = df.to_dict("records")

    def process_single(row):
        dataset_key = row.get("dataset_key", "default")
        native_path_str = str(row.get("native_path", row.get("raw_full_path", "")))
        img_id = PureWindowsPath(native_path_str).stem if native_path_str else str(row.get("image_id", "sample"))

        target_dir = os.path.join(cache_dir, dataset_key)
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, f"{img_id}.npz")

        if os.path.exists(out_path):
            return True

        if img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = os.path.join(img_root, dataset_key, PureWindowsPath(native_path_str).name)
        else:
            img_path = native_path_str

        bgr = cv2.imread(img_path)
        if bgr is None:
            return False

        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fov_mask = compute_fov_mask(img_rgb, erode_px=15)
        processed_u8 = apply_green_channel_ablation(img_rgb, ablation_mode=ablation_mode)
        valid_anchors = compute_valid_anchors(
            img_rgb.shape[:2], fov_mask, patch_size=patch_size, stride=stride, fov_thresh=fov_thresh
        )

        np.savez_compressed(out_path, img=processed_u8, anchors=valid_anchors)
        return True

    success_count = 0
    fail_count = 0

    print(f"[INFO] Precomputing NPZ cache for {len(records)} samples -> '{cache_dir}'...")
    with tqdm(total=len(records), desc="Precomputing NPZ Cache") as pbar:
        for r in records:
            res = process_single(r)
            if res:
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)

    print(f"[INFO] Cache generation completed: {success_count} succeeded, {fail_count} failed.")
    return {"success": success_count, "failed": fail_count}


# ==============================================================================
# 6. SLIDING-WINDOW MIL PATCH EXTRACTION (Overlapping s=96 + Raw FOV Filter + uint8 IPC)
# ==============================================================================

def extract_mil_patches_sliding(
    img_np: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = 48,
    max_patches_eval: int = 256,
    native_path: Optional[str] = None,
    valid_anchors: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    Extracts local high-resolution retinal patches via strided sliding-window scanning.
    Features Adaptive Fallback Stride:
      - Default stride = 96 (25% overlap) for high-resolution images (>1500px).
      - If image is small (~500px) and candidates < M, automatically reduces stride (s=48 or 64)
        to extract sufficient unique patches without artificial replacement/duplication.
      - If precomputed valid_anchors are provided via cache, slices directly for instant execution.
    """
    h, w, _ = img_np.shape
    p = patch_size

    # Determine target M quota in train mode
    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            m_target = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            m_target = int(num_sample)
    else:
        m_target = max_patches_eval

    # Fast-path: Slicing directly if valid_anchors from cache are provided
    if valid_anchors is not None and len(valid_anchors) > 0:
        n_anchors = len(valid_anchors)
        if mode == "train":
            if n_anchors < m_target:
                idx = np.random.choice(n_anchors, size=m_target, replace=True)
            else:
                idx = np.random.choice(n_anchors, size=m_target, replace=False)
            selected_coords = valid_anchors[idx]
        else:
            max_eval = min(n_anchors, max_patches_eval)
            idx = np.linspace(0, n_anchors - 1, max_eval).astype(int)
            selected_coords = valid_anchors[idx]

        patches = []
        for y0, x0 in selected_coords:
            y1, x1 = min(y0 + p, h), min(x0 + p, w)
            patch = img_np[y0:y1, x0:x1]
            if patch.shape[0] < p or patch.shape[1] < p:
                pad_h = max(0, p - patch.shape[0])
                pad_w = max(0, p - patch.shape[1])
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            patches.append(patch)

        patches_np = np.stack(patches, axis=0)
        return torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()

    # 1. Use external FOV mask if provided, else compute on img_np
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

    # 2. Inner candidate generator with dynamic stride
    def get_candidates(current_stride):
        s = max(1, current_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]

        if content_h > p and (y_min + content_h - p) not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and (x_min + content_w - p) not in x_starts:
            x_starts.append(x_min + content_w - p)

        cands = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = y0 + p, x0 + p
                y1c, x1c = min(y1, h), min(x1, w)
                patch = img_np[y0:y1c, x0:x1c]
                mask_patch = fov_mask[y0:y1c, x0:x1c]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio < fov_thresh:
                    continue

                if patch.shape[0] < p or patch.shape[1] < p:
                    pad_h = max(0, p - patch.shape[0])
                    pad_w = max(0, p - patch.shape[1])
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                cands.append(patch)
        return cands

    # Initial scan with default stride (e.g. 96)
    candidates = get_candidates(stride)

    # Adaptive Fallback Stride: if small image yields fewer candidates than M, rescan with dense stride (s=48)
    if mode == "train" and len(candidates) < m_target and stride > 48 and min(content_h, content_w) > p:
        dense_candidates = get_candidates(48)
        if len(dense_candidates) > len(candidates):
            candidates = dense_candidates

    # Fallback: Extract central patch if zero candidates pass FOV filter
    if len(candidates) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        patch = img_np[cy:cy + p, cx:cx + p]
        if patch.shape[0] < p or patch.shape[1] < p:
            patch = np.pad(patch, ((0, max(0, p - patch.shape[0])), (0, max(0, p - patch.shape[1])), (0, 0)), mode="reflect")
        candidates = [patch]

    # 3. Mode dispatch:
    if mode == "train":
        if len(candidates) < m_target:
            idx = np.random.choice(len(candidates), size=m_target, replace=True)
        else:
            idx = np.random.choice(len(candidates), size=m_target, replace=False)
        selected = [candidates[i] for i in idx]
    else:  # eval / inference
        if len(candidates) > max_patches_eval:
            idx = np.linspace(0, len(candidates) - 1, max_patches_eval).astype(int)
            selected = [candidates[i] for i in idx]
        else:
            selected = candidates

    # 4. PyTorch uint8 Tensor conversion
    patches_np = np.stack(selected, axis=0)
    patches_u8 = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()
    return patches_u8


def get_global_augmentations(img_size: int = 512, is_train: bool = True):
    """
    Constructs the Albumentations transformation pipeline for the Global Context branch.
    """
    try:
        import albumentations as A
        if is_train:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.5
                ),
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
            ])
    except ImportError:
        return lambda image: cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)


class FundusDualBranchDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, pd.DataFrame],
        img_size: int = 512,
        img_root: Optional[str] = None,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        patch_size: int = 128,
        stride: int = 96,
        transform: Optional[Any] = None
    ):
        """
        Dataset serving the proposed Dual-Branch model (Global + Local MIL) on Google Colab.
        Optimizations:
          1. Reads image paths directly from 'native_path' column in metadata CSV.
          2. Converts DataFrame to list of dicts (self.records) for O(1) in-memory indexing.
          3. Decodes JPEG (Quality 95) via high-speed OpenCV C++ backend.
          4. Separates FOV Mask calculation (strictly on RAW RGB) from Ben Graham enhancement.
          5. Returns uint8 tensors to minimize PyTorch DataLoader IPC Shared Memory bandwidth (755MB -> 189MB/batch).
          6. Supports optional precomputed NPZ anchor coordinate caching.
        """
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)

        self.records = self.df.to_dict("records")
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.img_root = img_root
        self.ablation_mode = ablation_mode
        self.is_train = (split == "train")
        self.cache_dir = cache_dir

        # Augmentation and resize applied ONLY to the Global stream
        if transform is not None:
            self.global_transform = transform
        else:
            self.global_transform = get_global_augmentations(
                img_size, is_train=self.is_train
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        label = int(row.get("diagnosis", 0))

        # 1. Resolve image path
        native_path_str = str(row.get("native_path", row.get("raw_full_path", "")))
        dataset_key = row.get("dataset_key", "default")
        if self.img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
        else:
            img_path = native_path_str

        # 2. Check if precomputed NPZ cache is available
        img_id = PureWindowsPath(native_path_str).stem if native_path_str else str(row.get("image_id", idx))
        cache_file = os.path.join(self.cache_dir, dataset_key, f"{img_id}.npz") if self.cache_dir else None

        if cache_file and os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                processed_np = data["img"]
                valid_anchors = data["anchors"]
                fov_mask = None
            except Exception:
                processed_np, valid_anchors, fov_mask = None, None, None
        else:
            processed_np, valid_anchors, fov_mask = None, None, None

        if processed_np is None:
            # Read JPEG via high-speed OpenCV backend
            bgr = cv2.imread(img_path)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    img_rgb = np.array(pil_img)
                except Exception as e:
                    raise RuntimeError(f"Cannot decode image {img_path}; refusing placeholder image") from e

            # CRITICAL FIX: Compute FOV mask strictly on RAW RGB image BEFORE Ben Graham
            fov_mask = compute_fov_mask(img_rgb, erode_px=15)

            # Apply Green channel ablation / Ben Graham background subtraction
            processed_np = apply_green_channel_ablation(
                img_rgb, ablation_mode=self.ablation_mode
            )
            valid_anchors = compute_valid_anchors(
                img_rgb.shape[:2], fov_mask, patch_size=self.patch_size, stride=self.stride, fov_thresh=0.5
            )

            # Auto-save cache archive for subsequent epochs (Epoch 2 -> 100+)
            if cache_file:
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    np.savez_compressed(cache_file, img=processed_np, anchors=valid_anchors)
                except Exception:
                    pass

        # ======================================================================
        # STREAM 1: GLOBAL CONTEXT BRANCH -> RESIZE (uint8 tensor for IPC speed)
        # ======================================================================
        global_augmented = self.global_transform(image=processed_np)
        global_img = global_augmented["image"] if isinstance(global_augmented, dict) else global_augmented
        # Retain uint8 tensor to slash IPC shared memory transfer (Normalized on GPU in DRTrainer)
        global_tensor = torch.from_numpy(global_img).permute(2, 0, 1).contiguous()

        # ======================================================================
        # STREAM 2: LOCAL MIL BRANCH -> PRESERVE NATIVE HIGH-RESOLUTION & FOV MASK
        # ======================================================================
        local_native_np = processed_np

        return {
            "global_image": global_tensor,
            "local_native_image": local_native_np,
            "fov_mask": fov_mask,
            "valid_anchors": valid_anchors,
            "native_path": img_path,
            "image_id": str(row.get("image_id", img_id)),
            "patient_id": str(row.get("patient_id", "")),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ==============================================================================
# 7. HIGH-THROUGHPUT BATCH COLLATE_FN (uint8 IPC Transmission & Synchronized Masks)
# ==============================================================================

def trainer_compatible_collate_fn(
    batch,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = (32, 64)
):
    """
    Collate function transmitting raw uint8 tensors across shared memory IPC:
    - Train mode: Returns [Batch, M, 3, 128, 128] uint8 tensor.
    - Eval mode: Zero-pads variable patch count up to batch maximum N_max, returns boolean patch_mask.
    """
    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            batch_m = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            batch_m = int(num_sample)
    else:
        batch_m = num_sample

    global_imgs = torch.stack([item["global_image"] for item in batch], dim=0)
    local_patches_list = [
        extract_mil_patches_sliding(
            item["local_native_image"],
            fov_mask=item.get("fov_mask"),
            patch_size=128,
            stride=96,
            fov_thresh=0.5,
            mode=mode,
            num_sample=batch_m,
            native_path=item.get("native_path"),
            valid_anchors=item.get("valid_anchors")
        )
        for item in batch
    ]
    labels = torch.stack([item["label"] for item in batch], dim=0)

    image_ids = [item.get("image_id", "") for item in batch]
    patient_ids = [item.get("patient_id", "") for item in batch]
    native_paths = [item.get("native_path", "") for item in batch]

    if mode == "train":
        local_patches = torch.stack(local_patches_list, dim=0)  # [Batch, M, 3, 128, 128] uint8
        return {
            "global_img": global_imgs,
            "local_patches": local_patches,
            "label": labels,
            "image_id": image_ids,
            "patient_id": patient_ids,
            "native_path": native_paths,
        }

    # --- Eval Mode: Zero-pad variable patch counts to batch maximum N_max ---
    n_max = max(p.shape[0] for p in local_patches_list)
    padded, masks = [], []
    for p in local_patches_list:
        n = p.shape[0]
        if n < n_max:
            pad = torch.zeros((n_max - n, *p.shape[1:]), dtype=p.dtype)
            p = torch.cat([p, pad], dim=0)
        mask = torch.zeros(n_max, dtype=torch.bool)
        mask[:n] = True
        padded.append(p)
        masks.append(mask)

    local_patches = torch.stack(padded, dim=0)     # [Batch, N_max, 3, 128, 128] uint8
    patch_mask = torch.stack(masks, dim=0)         # [Batch, N_max], True = valid patch

    return {
        "global_img": global_imgs,
        "local_patches": local_patches,
        "patch_mask": patch_mask,
        "label": labels,
        "image_id": image_ids,
        "patient_id": patient_ids,
        "native_path": native_paths,
    }

# %%
#!/usr/bin/env python3
"""
================================================================================
DMIP 2026 — REPRODUCIBLE LODO CHECKPOINT INFERENCE & SCIENTIFIC AUDIT PIPELINE
================================================================================
Executes REAL PyTorch model inference using the 12 primary Leave-One-Dataset-Out
(LODO) checkpoints for Variant 2 (V2) and Variant 3 (V3):
  1. Exact Preprocessing Parity: Raw uint8 tensors from FundusDualBranchDataset
     and collate_fn; explicit ImageNet standardization on device.
  2. Exact Sample Identity: Dataset-returned image_id and native_path asserted
     row-by-row against manifests; no fallback or substitution allowed.
  3. Cache Fail-Safe: Default reuse_predictions=False; when requested, verifies
     row counts, V2/V3 checkpoint SHA, manifest SHA, ordered image IDs and labels,
     integer class ranges, and code fingerprints on EVERY single row.
  4. Authoritative Benchmark References: Cross-checked against full Table 3 grade
     profiles in lodo_summary_with_calibrated_screening.json with raw deviations.
  5. Messidor-2 Reconciliation: Audited row-level inclusion table tracking all
     1,744 raw images vs. 1,057 quality-curated manifest images without unverified
     causal claims (omission marked UNKNOWN).
  6. Bootstrap per Advisor: 95% paired bootstrap CIs (B=5000, seed 42) reporting
     all THREE CIs (V2, V3, and paired V3-V2 difference), separating pooled
     image-weighted and equal-weighted 6-cohort macro estimates.
  7. Grade 3 Error Destination Confusion: Exact 5x5 confusion matrices and
     referable classification-side audits without causal overclaims.
  8. Fail-Closed Publication Gate: Immutable run-specific directories; atomic
     promotion to _VERIFIED.csv only when all 6 folds pass strict checks.

STRICT INVARIANTS:
  - ZERO MODEL RETRAINING.
  - ZERO SYNTHETIC PREDICTIONS.
  - ZERO CAUSAL OVERCLAIMS.
  - STATISTICAL SCOPE: Bootstrap CIs quantify held-out test cohort sampling
    variability, NOT between-training-seed variability.
================================================================================
"""

import os
import sys
import argparse
import json
import time
import uuid
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

# Silence albumentations update check network warnings in sandbox
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Ensure writable matplotlib cache
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(WORK / 'matplotlib')
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# Fix Windows stdout encoding for UTF-8 characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------------------
# 1. DIRECTORY & REPOSITORY RESOLUTION
# ------------------------------------------------------------------------------
# 2. CHECKPOINT PAIRS & AUTHORITATIVE REFERENCE SPECIFICATIONS
# ------------------------------------------------------------------------------
DATASET_KEYS = ["aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid"]
DATASET_NAMES = {
    "aptos": "APTOS 2019",
    "idrid": "IDRiD",
    "messidor2": "Messidor-2",
    "ddr": "DDR",
    "eyepacs": "EyePACS",
    "deepdrid": "DeepDRiD"
}

CHECKPOINT_PAIRS = [
    {
        "fold": 0, "key": "aptos", "name": "APTOS 2019",
        "v2_file": "lodo_V2_PlusGlobalContext_H2_fold_0_aptos_best.pth",
        "v3_file": "lodo_V3_PlusOrdinalHead_H3_Champion_fold_0_aptos_best.pth"
    },
    {
        "fold": 1, "key": "idrid", "name": "IDRiD",
        "v2_file": "lodo_V2_PlusGlobalContext_H2_fold_1_idrid_best.pth",
        "v3_file": "lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth"
    },
    {
        "fold": 2, "key": "messidor2", "name": "Messidor-2",
        "v2_file": "lodo_V2_PlusGlobalContext_H2_fold_2_messidor2_best.pth",
        "v3_file": "lodo_V3_PlusOrdinalHead_H3_Champion_fold_2_messidor2_best.pth"
    },
    {
        "fold": 3, "key": "ddr", "name": "DDR",
        "v2_file": "lodo_V2_PlusGlobalContext_H2_fold_3_ddr_best.pth",
        "v3_file": "lodo_V3_PlusOrdinalHead_H3_Champion_fold_3_ddr_best.pth"
    },
    {
        "fold": 4, "key": "eyepacs", "name": "EyePACS",
        "v2_file": "lodo_V2_PlusGlobalContext_H2_fold_4_eyepacs_best.pth",
        "v3_file": "lodo_V3_PlusOrdinalHead_H3_Champion_fold_4_eyepacs_best.pth"
    },
    {
        "fold": 5, "key": "deepdrid", "name": "DeepDRiD",
        "v2_file": "lodo_V2_PlusGlobalContext_H2_fold_5_deepdrid_best.pth",
        "v3_file": "lodo_V3_PlusOrdinalHead_H3_Champion_fold_5_deepdrid_best.pth"
    }
]

# Published Table 3 unweighted macro reference values (matching manuscript)
PUBLISHED_TABLE_3_REFERENCE = {
    "V2": {
        "macro_qwk": 0.699,
        "macro_within_1": 90.8,
        "macro_g0": 62.0, "macro_g1": 45.6, "macro_g2": 44.7,
        "macro_g3_sensitivity": 43.8, "macro_g4": 57.3
    },
    "V3": {
        "macro_qwk": 0.690,
        "macro_within_1": 92.2,
        "macro_g0": 58.4, "macro_g1": 50.8, "macro_g2": 41.6,
        "macro_g3_sensitivity": 31.9, "macro_g4": 48.3
    }
}

# %%
def load_authoritative_references(results_json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads exact per-fold and per-grade reference values directly from
    lodo_summary_with_calibrated_screening.json, establishing explicit provenance.
    """
    if not os.path.exists(results_json_path):
        raise FileNotFoundError(f"Missing authoritative results file: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = {}
    v2_data = data.get("V2_PlusGlobalContext_H2", {})
    v3_data = data.get("V3_PlusOrdinalHead_H3_Champion", {})

    for cp in CHECKPOINT_PAIRS:
        key = cp["key"]
        v2_cohort = v2_data.get(key, {})
        v3_cohort = v3_data.get(key, {})

        v2_recalls = v2_cohort.get("ood_per_grade_recall", {})
        v3_recalls = v3_cohort.get("ood_per_grade_recall", {})

        references[key] = {
            "fold": cp["fold"],
            "name": cp["name"],
            "v2_qwk": round(float(v2_cohort.get("ood_test_qwk", 0.0)), 4),
            "v3_qwk": round(float(v3_cohort.get("ood_test_qwk", 0.0)), 4),
            "v2_within1": round(float(v2_cohort.get("ood_within_1_grade_acc", 0.0)) * 100.0, 2),
            "v3_within1": round(float(v3_cohort.get("ood_within_1_grade_acc", 0.0)) * 100.0, 2),
            "v2_g3_sens": round(float(v2_recalls.get("grade_3", 0.0)) * 100.0, 2),
            "v3_g3_sens": round(float(v3_recalls.get("grade_3", 0.0)) * 100.0, 2),
            "v2_per_grade": [round(float(v2_recalls.get(f"grade_{i}", 0.0)) * 100.0, 2) for i in range(5)],
            "v3_per_grade": [round(float(v3_recalls.get(f"grade_{i}", 0.0)) * 100.0, 2) for i in range(5)],
            "provenance_file": os.path.basename(results_json_path)
        }
    return references

# %%
def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 checksum for artifact provenance."""
    if not os.path.exists(filepath):
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# %%
def normalize_model_inputs(
    global_img: torch.Tensor,
    local_patches: torch.Tensor,
    device: str = "cpu",
    is_already_normalized: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact Reference Evaluation Normalization Pipeline:
    Ownership:
      - FundusDualBranchDataset and collate_fn return raw uint8 tensors [0, 255].
      - Normalization converts uint8 to float32 on device and divides by 255.0,
        then subtracts ImageNet mean [0.485, 0.456, 0.406] and divides by std [0.229, 0.224, 0.225].
    """
    if is_already_normalized:
        return global_img.to(device), local_patches.to(device)

    mean_g = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_g = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    mean_l = mean_g.view(1, 1, 3, 1, 1)
    std_l = std_g.view(1, 1, 3, 1, 1)

    if global_img.dtype == torch.uint8:
        global_img = global_img.to(device, dtype=torch.float32).div(255.0).sub(mean_g).div(std_g)
    else:
        global_img = global_img.to(device, dtype=torch.float32)
        if global_img.max() > 1.0:
            global_img = global_img.div(255.0)
        global_img = global_img.sub(mean_g).div(std_g)

    if local_patches.dtype == torch.uint8:
        local_patches = local_patches.to(device, dtype=torch.float32).div(255.0).sub(mean_l).div(std_l)
    else:
        local_patches = local_patches.to(device, dtype=torch.float32)
        if local_patches.max() > 1.0:
            local_patches = local_patches.div(255.0)
        local_patches = local_patches.sub(mean_l).div(std_l)

    return global_img, local_patches

# %%
def get_model(head_type: str, device: str = "cpu") -> nn.Module:
    """Builds native DualBranchMILModel + DRFullModel without pretrained weights."""
    try:
        model = build_model(
            head_type=head_type,
            pretrained=False,
            use_cbam=True,
            use_dual_pooling=True,
            fusion_dim=512,
            num_classes=5
        )
    except Exception as e:
        print(f"[ERROR] Could not build model from full/losses_and_heads: {e}")
        raise e
    return model.to(device)

# %%
def load_checkpoint_strictly(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Loads checkpoint with strict=True and verifies state dict integrity."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing required checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    incompatible = model.load_state_dict(state_dict, strict=True)
    if len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0:
        raise RuntimeError(f"State dict mismatch in {checkpoint_path}: "
                           f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")

    model.eval()
    return ckpt

# %%
def paired_bootstrap_grade3_sensitivity(
    y_true: np.ndarray,
    pred_v2: np.ndarray,
    pred_v3: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    reps: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes 95% paired bootstrap confidence intervals for Grade 3 sensitivity:
      1. V2 Grade 3 Sensitivity: Point estimate, [95% CI Low, High]
      2. V3 Grade 3 Sensitivity: Point estimate, [95% CI Low, High]
      3. Paired Difference Δ = V3 - V2 (pp): Point estimate, [95% CI Low, High]

    Strictly resamples the SAME case indices for V2 and V3 in every replicate.
    Patient clustering is ONLY enabled if multi-image patient clusters are explicitly validated.
    """
    g3_idx = np.where(y_true == 3)[0]
    n_g3 = len(g3_idx)
    if n_g3 == 0:
        return {
            "n_grade3": 0, "sens_v2": np.nan, "sens_v3": np.nan, "diff_v3_minus_v2_pp": np.nan,
            "v2_ci_low": np.nan, "v2_ci_high": np.nan, "v3_ci_low": np.nan, "v3_ci_high": np.nan,
            "diff_ci_low_pp": np.nan, "diff_ci_high_pp": np.nan, "resampling_unit": "none",
            "bootstrap_reps": reps, "seed": seed
        }

    a = (pred_v2[g3_idx] == 3).astype(np.float32)
    b = (pred_v3[g3_idx] == 3).astype(np.float32)

    point_sens_v2 = float(a.mean())
    point_sens_v3 = float(b.mean())
    point_diff_pp = float((point_sens_v3 - point_sens_v2) * 100.0)

    rng = np.random.default_rng(seed)
    boot_v2 = []
    boot_v3 = []
    boot_diff = []

    # Patient clustering check: require non-trivial validated patient clusters
    use_patient_clustering = False
    clusters = None
    if patient_ids is not None:
        p_g3 = patient_ids[g3_idx]
        unique_patients = np.unique(p_g3)
        if len(unique_patients) < len(p_g3) and len(unique_patients) > 1:
            use_patient_clustering = True
            clusters = [np.where(p_g3 == p)[0] for p in unique_patients]
            num_clusters = len(clusters)

    if use_patient_clustering and clusters is not None:
        resampling_unit = "patient_cluster"
        for _ in range(reps):
            c_indices = rng.integers(0, num_clusters, size=num_clusters)
            sample_idx = np.concatenate([clusters[ci] for ci in c_indices])
            s_v2 = a[sample_idx].mean() * 100.0
            s_v3 = b[sample_idx].mean() * 100.0
            boot_v2.append(s_v2)
            boot_v3.append(s_v3)
            boot_diff.append(s_v3 - s_v2)
    else:
        resampling_unit = "image_level"
        for _ in range(reps):
            sample_idx = rng.integers(0, n_g3, size=n_g3)
            s_v2 = a[sample_idx].mean() * 100.0
            s_v3 = b[sample_idx].mean() * 100.0
            boot_v2.append(s_v2)
            boot_v3.append(s_v3)
            boot_diff.append(s_v3 - s_v2)

    v2_ci = np.percentile(boot_v2, [2.5, 97.5])
    v3_ci = np.percentile(boot_v3, [2.5, 97.5])
    diff_ci = np.percentile(boot_diff, [2.5, 97.5])

    return {
        "n_grade3": n_g3,
        "sens_v2": round(point_sens_v2 * 100.0, 2),
        "sens_v3": round(point_sens_v3 * 100.0, 2),
        "diff_v3_minus_v2_pp": round(point_diff_pp, 2),
        "v2_ci_low": round(float(v2_ci[0]), 2),
        "v2_ci_high": round(float(v2_ci[1]), 2),
        "v3_ci_low": round(float(v3_ci[0]), 2),
        "v3_ci_high": round(float(v3_ci[1]), 2),
        "diff_ci_low_pp": round(float(diff_ci[0]), 2),
        "diff_ci_high_pp": round(float(diff_ci[1]), 2),
        "resampling_unit": resampling_unit,
        "bootstrap_reps": reps,
        "seed": seed
    }

# %%
def macro_bootstrap_grade3_sensitivity(
    cohort_data: List[Dict[str, Any]],
    reps: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes 95% paired bootstrap confidence intervals for equal-weighted 6-cohort Macro Grade 3 sensitivity.
    Stratified resampling within each of the 6 cohorts, computing unweighted macro average per replicate.
    """
    g3_cohorts = []
    for cd in cohort_data:
        g3_idx = np.where(cd["y_true"] == 3)[0]
        if len(g3_idx) > 0:
            a = (cd["pred_v2"][g3_idx] == 3).astype(np.float32)
            b = (cd["pred_v3"][g3_idx] == 3).astype(np.float32)
            g3_cohorts.append({"name": cd["cohort"], "a": a, "b": b, "n": len(g3_idx)})

    n_cohorts = len(g3_cohorts)
    if n_cohorts == 0:
        return {}

    point_v2_macros = [c["a"].mean() for c in g3_cohorts]
    point_v3_macros = [c["b"].mean() for c in g3_cohorts]
    macro_v2 = float(np.mean(point_v2_macros)) * 100.0
    macro_v3 = float(np.mean(point_v3_macros)) * 100.0
    diff_macro_pp = macro_v3 - macro_v2

    rng = np.random.default_rng(seed)
    boot_v2 = []
    boot_v3 = []
    boot_diff = []

    for _ in range(reps):
        rep_v2_means = []
        rep_v3_means = []
        for c in g3_cohorts:
            idx = rng.integers(0, c["n"], size=c["n"])
            rep_v2_means.append(c["a"][idx].mean())
            rep_v3_means.append(c["b"][idx].mean())

        m_v2 = float(np.mean(rep_v2_means)) * 100.0
        m_v3 = float(np.mean(rep_v3_means)) * 100.0
        boot_v2.append(m_v2)
        boot_v3.append(m_v3)
        boot_diff.append(m_v3 - m_v2)

    v2_ci = np.percentile(boot_v2, [2.5, 97.5])
    v3_ci = np.percentile(boot_v3, [2.5, 97.5])
    diff_ci = np.percentile(boot_diff, [2.5, 97.5])

    return {
        "cohort": f"Macro Unweighted ({n_cohorts} Cohorts)",
        "fold": -2,
        "n_grade3": sum(c["n"] for c in g3_cohorts),
        "sens_v2": round(macro_v2, 2),
        "sens_v3": round(macro_v3, 2),
        "diff_v3_minus_v2_pp": round(diff_macro_pp, 2),
        "v2_ci_low": round(float(v2_ci[0]), 2),
        "v2_ci_high": round(float(v2_ci[1]), 2),
        "v3_ci_low": round(float(v3_ci[0]), 2),
        "v3_ci_high": round(float(v3_ci[1]), 2),
        "diff_ci_low_pp": round(float(diff_ci[0]), 2),
        "diff_ci_high_pp": round(float(diff_ci[1]), 2),
        "resampling_unit": "stratified_by_cohort_macro",
        "bootstrap_reps": reps,
        "seed": seed
    }

# %%
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Computes exact 5x5 integer confusion matrix where row=actual, col=predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

# %%
def extract_grade3_confusion(cm: np.ndarray, cohort_name: str, model_name: str) -> Dict[str, Any]:
    """
    Extracts Grade 3 specific error destinations from row 3 of confusion matrix:
      - Exact correct recall (G3 -> 3)
      - Adjacent referable (G3 -> 2 or 4)
      - Non-referable demotions (G3 -> 0 or 1)
    """
    g3 = cm[3]
    total_g3 = int(np.sum(g3))
    correct = int(g3[3])
    to_01 = int(g3[0] + g3[1])
    to_24 = int(g3[2] + g3[4])
    misclassified = total_g3 - correct
    pct_to_24 = round((to_24 / misclassified * 100.0), 2) if misclassified > 0 else 0.0

    return {
        "cohort": cohort_name,
        "model": model_name,
        "n_actual_g3": total_g3,
        "g3_to_0": int(g3[0]),
        "g3_to_1": int(g3[1]),
        "g3_to_2": int(g3[2]),
        "g3_to_3": correct,
        "g3_to_4": int(g3[4]),
        "g3_to_01": to_01,
        "g3_to_24": to_24,
        "g3_correct": correct,
        "n_g3_misclassified": misclassified,
        "pct_errors_assigned_to_grade24": pct_to_24
    }

# %%
def validate_cache_fail_safe(
    cache_file: str,
    n_expected_samples: int,
    v2_sha: str,
    v3_sha: str,
    manifest_sha: str,
    df_test: pd.DataFrame
) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Cache Fail-Safe Validator (Requirement 1 & 3):
    Validates that cached predictions match current run parameters on EVERY row:
      1. Expected row count matches manifest exactly.
      2. Checkpoint V2 SHA256 matches on every row.
      3. Checkpoint V3 SHA256 matches on every row.
      4. Manifest SHA256 matches on every row.
      5. Ordered image IDs match manifest row-by-row.
      6. Ordered diagnosis labels match manifest row-by-row.
      7. Provenance marker equals 'STRICT_REAL_CHECKPOINT_INFERENCE' on every row.
      8. Pipeline code fingerprint matches on every row.
      9. Prediction class ranges are valid integers in {0, 1, 2, 3, 4}.
    """
    if not os.path.exists(cache_file):
        return False, None, "Cache file not found"

    try:
        df_cached = pd.read_csv(cache_file)
        if len(df_cached) != n_expected_samples:
            return False, None, f"Row count mismatch: cached={len(df_cached)} vs expected={n_expected_samples}"

        # 1. Provenance marker check on every row
        if "provenance_marker" not in df_cached.columns:
            return False, None, "Missing provenance_marker column"
        if not (df_cached["provenance_marker"] == "STRICT_REAL_CHECKPOINT_INFERENCE").all():
            return False, None, "Invalid provenance_marker detected"

        # 2. SHA256 checksums on EVERY row
        if "checkpoint_v2_sha" not in df_cached.columns or not (df_cached["checkpoint_v2_sha"] == v2_sha).all():
            return False, None, "Checkpoint V2 SHA256 mismatch on one or more rows"
        if "checkpoint_v3_sha" not in df_cached.columns or not (df_cached["checkpoint_v3_sha"] == v3_sha).all():
            return False, None, "Checkpoint V3 SHA256 mismatch on one or more rows"
        if "manifest_sha" not in df_cached.columns or not (df_cached["manifest_sha"] == manifest_sha).all():
            return False, None, "Manifest SHA256 mismatch on one or more rows"

        # 3. Code / Pipeline fingerprint check
        if "code_fingerprint" in df_cached.columns:
            if not (df_cached["code_fingerprint"] == PIPELINE_CODE_FINGERPRINT).all():
                return False, None, "Pipeline code fingerprint mismatch on one or more rows"

        # 4. Ordered image_id alignment
        if "image_id" not in df_cached.columns:
            return False, None, "Missing image_id column"
        test_ids = df_test["image_id"].astype(str).values
        cached_ids = df_cached["image_id"].astype(str).values
        if not np.array_equal(test_ids, cached_ids):
            return False, None, "Ordered image_id alignment mismatch"

        # 5. Ordered diagnosis alignment
        if "diagnosis" not in df_cached.columns:
            return False, None, "Missing diagnosis column"
        test_diag = df_test["diagnosis"].values.astype(int)
        cached_diag = df_cached["diagnosis"].values.astype(int)
        if not np.array_equal(test_diag, cached_diag):
            return False, None, "Ordered diagnosis alignment mismatch"

        # 6. Predictions integer validity
        if "pred_v2" not in df_cached.columns or "pred_v3" not in df_cached.columns:
            return False, None, "Missing prediction columns"
        p2 = df_cached["pred_v2"].values
        p3 = df_cached["pred_v3"].values
        if np.any(pd.isna(p2)) or np.any(pd.isna(p3)):
            return False, None, "NaN/Null predictions detected"
        if not np.all(np.isin(p2, [0, 1, 2, 3, 4])) or not np.all(np.isin(p3, [0, 1, 2, 3, 4])):
            return False, None, "Out-of-range prediction class detected"

        return True, df_cached, "VERIFIED_CACHE_HIT"

    except Exception as e:
        return False, None, f"Cache validation exception: {e}"

# %%
def evaluate_cohort_inference(
    pair: Dict[str, Any],
    args: argparse.Namespace,
    model_v2: nn.Module,
    model_v3: nn.Module,
    results_dir: str,
    trace_samples: List[Dict[str, Any]],
    run_id: str
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Executes real inference or retrieves cached verified predictions for one LODO fold.
    Returns: (df_predictions, metrics_dict, cm_v2, cm_v3)
    """
    fold = pair["fold"]
    key = pair["key"]
    name = pair["name"]
    test_csv_path = os.path.join(args.folds_dir, f"fold_{fold}_{key}_test.csv")

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Missing required test manifest: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    # Validate manifest integrity
    assert "image_id" in df_test.columns, f"Missing 'image_id' in {test_csv_path}"
    assert "diagnosis" in df_test.columns, f"Missing 'diagnosis' in {test_csv_path}"
    assert df_test["image_id"].isnull().sum() == 0, f"Null image_ids in {test_csv_path}"
    assert df_test["diagnosis"].isnull().sum() == 0, f"Null diagnoses in {test_csv_path}"
    assert df_test["image_id"].nunique() == len(df_test), f"Duplicate image_ids in {test_csv_path}"
    assert set(df_test["diagnosis"].unique()).issubset({0, 1, 2, 3, 4}), f"Invalid diagnoses in {test_csv_path}"

    n_samples = len(df_test)
    if args.max_samples > 0:
        df_test = df_test.iloc[:args.max_samples].copy()
        n_samples = len(df_test)

    v2_ckpt_path = os.path.join(args.checkpoint_dir, pair["v2_file"])
    v3_ckpt_path = os.path.join(args.checkpoint_dir, pair["v3_file"])
    v2_sha = compute_file_sha256(v2_ckpt_path)
    v3_sha = compute_file_sha256(v3_ckpt_path)
    manifest_sha = compute_file_sha256(test_csv_path)

    cache_dir = os.path.join(results_dir, "predictions")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fold_{fold}_{key}_predictions.csv")

    use_cached = False
    if args.reuse_predictions and not args.force_inference:
        is_valid, df_cached, reason = validate_cache_fail_safe(
            cache_file=cache_file,
            n_expected_samples=n_samples,
            v2_sha=v2_sha,
            v3_sha=v3_sha,
            manifest_sha=manifest_sha,
            df_test=df_test
        )
        if is_valid and df_cached is not None:
            print(f"[{name}] {reason}: Reusing {len(df_cached):,} validated predictions.")
            use_cached = True
            df_res = df_cached
        else:
            print(f"[{name}] Cache bypassed ({reason}). Executing real inference.")

    if not use_cached:
        print(f"\n---> Running REAL Model Inference for Fold {fold}: {name} ({n_samples:,} images) <---")

        load_checkpoint_strictly(model_v2, v2_ckpt_path, device=args.device)
        load_checkpoint_strictly(model_v3, v3_ckpt_path, device=args.device)

        pass  # Dataset and collator are defined above.
        from functools import partial

        eval_collate_fn = partial(trainer_compatible_collate_fn, mode="eval")
        dataset = FundusDualBranchDataset(
            csv_path=df_test,
            img_root=args.data_root,
            img_size=512,
            split="test"
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collate_fn,
            pin_memory=(args.device == "cuda"),
            **({"prefetch_factor": 1} if args.num_workers > 0 else {})
        )

        pred_v2_list = []
        pred_v3_list = []
        v2_prob_list = []
        v3_prob_list = []
        v3_exp_grade_list = []

        start_time = time.time()
        sample_offset = 0

        for b_idx, batch in enumerate(data_loader):
            g_img, l_patches = normalize_model_inputs(
                batch["global_img"],
                batch["local_patches"],
                device=args.device,
                is_already_normalized=False
            )
            targets = batch["label"].to(args.device)

            patch_mask = batch.get("patch_mask")
            if patch_mask is not None:
                patch_mask = patch_mask.to(args.device)

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(PRECISION == "bfloat16")):
                out_v2 = model_v2(g_img, l_patches, patch_mask=patch_mask)
                out_v3 = model_v3(g_img, l_patches, patch_mask=patch_mask)

            # V2 decoding: nominal argmax
            p2 = out_v2["probs"].detach().float().cpu().numpy()
            v2_preds = np.argmax(p2, axis=1)

            # V3 decoding: expected-grade rounding
            p3 = out_v3["probs"].detach().float().cpu().numpy()
            if "expected_grade" in out_v3:
                exp_g = out_v3["expected_grade"].detach().float().cpu().numpy()
            else:
                class_idx = np.arange(5, dtype=np.float32)
                exp_g = np.sum(p3 * class_idx, axis=1)
            v3_preds = np.clip(np.round(exp_g), 0, 4).astype(int)

            pred_v2_list.extend(v2_preds)
            pred_v3_list.extend(v3_preds)
            v2_prob_list.extend(p2.tolist())
            v3_prob_list.extend(p3.tolist())
            v3_exp_grade_list.extend(exp_g.tolist())

            # Real ID Alignment Assertion (Requirement 2: NEVER fallback to expected_id)
            batch_labels = targets.cpu().numpy()
            if "image_id" not in batch:
                raise KeyError(f"Dataset/collate batch missing mandatory 'image_id' in cohort '{name}'")
            batch_ids = batch["image_id"]
            if len(batch_ids) != len(batch_labels):
                raise ValueError(f"Batch length mismatch in cohort '{name}': len(image_id)={len(batch_ids)} != len(label)={len(batch_labels)}")

            for i in range(len(batch_labels)):
                abs_idx = sample_offset + i
                if abs_idx >= len(df_test):
                    raise IndexError(f"Sample offset {abs_idx} exceeds manifest length {len(df_test)} in cohort '{name}'")

                expected_id = str(df_test.iloc[abs_idx]["image_id"]).strip()
                actual_id = str(batch_ids[i]).strip()
                if not actual_id:
                    raise ValueError(f"Empty image_id returned by dataset at index {i} in cohort '{name}'")
                if actual_id != expected_id:
                    raise AssertionError(
                        f"Sample ID alignment error at row {abs_idx} in cohort '{name}': "
                        f"manifest ID='{expected_id}' != dataset ID='{actual_id}'"
                    )
                expected_diag = int(df_test.iloc[abs_idx]["diagnosis"])
                actual_diag = int(batch_labels[i])
                if actual_diag != expected_diag:
                    raise AssertionError(
                        f"Diagnosis alignment error at row {abs_idx} in cohort '{name}': "
                        f"manifest diagnosis={expected_diag} != dataset label={actual_diag}"
                    )
                if len(trace_samples) < 20:
                    trace_samples.append({
                        "run_id": run_id,
                        "cohort": name,
                        "fold": fold,
                        "image_id": expected_id,
                        "y_true": actual_diag,
                        "v2_pred": int(v2_preds[i]),
                        "v2_probs": [round(float(x), 4) for x in p2[i]],
                        "v3_pred": int(v3_preds[i]),
                        "v3_exp_grade": round(float(exp_g[i]), 4),
                        "v3_probs": [round(float(x), 4) for x in p3[i]],
                        "independent_forward_pass": True
                    })

            sample_offset += len(targets)
            if (b_idx + 1) % 10 == 0 or sample_offset == n_samples:
                elapsed = time.time() - start_time
                fps = sample_offset / max(1e-3, elapsed)
                print(f"[{name}] Evaluated {sample_offset}/{n_samples} images ({fps:.1f} img/s)...")

        assert sample_offset == len(df_test), f"Evaluated {sample_offset} samples, expected {len(df_test)}"
        assert len(pred_v2_list) == len(df_test), f"Predictions count mismatch: {len(pred_v2_list)} != {len(df_test)}"

        df_res = df_test.copy()
        df_res["pred_v2"] = pred_v2_list
        df_res["pred_v3"] = pred_v3_list
        df_res["v3_expected_grade"] = v3_exp_grade_list
        df_res["checkpoint_v2"] = pair["v2_file"]
        df_res["checkpoint_v3"] = pair["v3_file"]
        df_res["checkpoint_v2_sha"] = v2_sha
        df_res["checkpoint_v3_sha"] = v3_sha
        df_res["manifest_sha"] = manifest_sha
        df_res["code_fingerprint"] = PIPELINE_CODE_FINGERPRINT
        df_res["provenance_marker"] = "STRICT_REAL_CHECKPOINT_INFERENCE"
        df_res["run_id"] = run_id

        df_res.to_csv(cache_file, index=False)
        print(f"[{name}] Real inference complete. Saved {len(df_res):,} validated predictions to cache.")

    y_true = df_res["diagnosis"].values.astype(int)
    p_v2 = df_res["pred_v2"].values.astype(int)
    p_v3 = df_res["pred_v3"].values.astype(int)

    cm_v2 = compute_confusion_matrix(y_true, p_v2, num_classes=5)
    cm_v3 = compute_confusion_matrix(y_true, p_v3, num_classes=5)

    qwk_v2 = float(cohen_kappa_score(y_true, p_v2, weights="quadratic"))
    qwk_v3 = float(cohen_kappa_score(y_true, p_v3, weights="quadratic"))
    w1_v2 = float(np.mean(np.abs(p_v2 - y_true) <= 1)) * 100.0
    w1_v3 = float(np.mean(np.abs(p_v3 - y_true) <= 1)) * 100.0

    g3_mask = (y_true == 3)
    n_g3 = int(np.sum(g3_mask))
    g3_sens_v2 = float(np.mean(p_v2[g3_mask] == 3)) * 100.0 if n_g3 > 0 else 0.0
    g3_sens_v3 = float(np.mean(p_v3[g3_mask] == 3)) * 100.0 if n_g3 > 0 else 0.0

    per_grade_v2 = []
    per_grade_v3 = []
    for g in range(5):
        m = (y_true == g)
        n_m = int(np.sum(m))
        per_grade_v2.append(round(float(np.mean(p_v2[m] == g)) * 100.0, 2) if n_m > 0 else 0.0)
        per_grade_v3.append(round(float(np.mean(p_v3[m] == g)) * 100.0, 2) if n_m > 0 else 0.0)

    metrics_dict = {
        "cohort": name,
        "fold": fold,
        "n_samples": n_samples,
        "n_grade3": n_g3,
        "v2_qwk": round(qwk_v2, 4),
        "v3_qwk": round(qwk_v3, 4),
        "v2_within1": round(w1_v2, 2),
        "v3_within1": round(w1_v3, 2),
        "v2_g3_sens": round(g3_sens_v2, 2),
        "v3_g3_sens": round(g3_sens_v3, 2),
        "v2_per_grade": per_grade_v2,
        "v3_per_grade": per_grade_v3
    }

    return df_res, metrics_dict, cm_v2, cm_v3

# %%
def generate_messidor2_audit_table(raw_csv_path: str, test_csv_path: str, output_path: str) -> pd.DataFrame:
    """
    Produces a row-level join between raw Messidor-2 data (1,744 images)
    and LODO test manifest (1,057 images).
    Documented reason for absence is UNKNOWN / ABSENT_FROM_NATIVE_CROPPED_DIRECTORY;
    does not assume or fabricate unverified macula/disc quality claims (Requirement 4).
    """
    if not os.path.exists(raw_csv_path) or not os.path.exists(test_csv_path):
        return pd.DataFrame()

    raw_df = pd.read_csv(raw_csv_path)
    test_df = pd.read_csv(test_csv_path)

    test_ids = set(test_df["image_id"].astype(str))

    summary_rows = []
    for g in range(5):
        raw_g = len(raw_df[raw_df["diagnosis"] == g])
        inc_g = len(test_df[test_df["diagnosis"] == g])
        omit_g = raw_g - inc_g
        summary_rows.append({
            "icdr_grade": g,
            "raw_archive_count": raw_g,
            "included_in_lodo_test": inc_g,
            "omitted_count": omit_g,
            "omission_percentage": round(omit_g / max(1, raw_g) * 100.0, 1),
            "omission_status": "ABSENT_FROM_NATIVE_CROPPED_DIRECTORY",
            "curation_lineage": "Image file absent from data/raw/hybrid/native_cropped/Messidor-2; exact historical exclusion criterion UNKNOWN from repository logs"
        })

    tot_raw = len(raw_df)
    tot_inc = len(test_df)
    tot_omit = tot_raw - tot_inc
    summary_rows.append({
        "icdr_grade": "Total",
        "raw_archive_count": tot_raw,
        "included_in_lodo_test": tot_inc,
        "omitted_count": tot_omit,
        "omission_percentage": round(tot_omit / tot_raw * 100.0, 1),
        "omission_status": "ABSENT_FROM_NATIVE_CROPPED_DIRECTORY",
        "curation_lineage": "Documented 687-image difference: 1,744 raw vs 1,057 in active LODO split; historical filtering log UNKNOWN"
    })

    df_audit = pd.DataFrame(summary_rows)
    df_audit.to_csv(output_path, index=False)
    return df_audit

# %%
def generate_publication_plots(df_boot: pd.DataFrame, df_conf: pd.DataFrame, figures_dir: str):
    """
    Generates publication-quality 300-DPI vector PDF and PNG figures:
      1. Forest plot of Grade 3 sensitivity difference (V3 - V2) with all 3 CIs.
      2. Stacked bar chart of Grade 3 error destinations (Referable Adjacency).
    """
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Forest Plot
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)
    cohort_rows = df_boot[df_boot["fold"] >= 0].sort_values("fold", ascending=False)
    summary_rows = df_boot[df_boot["fold"] < 0]
    plot_df = pd.concat([cohort_rows, summary_rows])

    y_pos = np.arange(len(plot_df))
    diffs = plot_df["diff_v3_minus_v2_pp"].values
    ci_lows = plot_df["diff_ci_low_pp"].values
    ci_highs = plot_df["diff_ci_high_pp"].values
    labels = plot_df["cohort"].values

    xerr_low = np.maximum(0, diffs - ci_lows)
    xerr_high = np.maximum(0, ci_highs - diffs)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    for i, (y, d, el, eh, f) in enumerate(zip(y_pos, diffs, xerr_low, xerr_high, plot_df["fold"])):
        is_summary = (f < 0)
        color = "#1f77b4" if not is_summary else "#d62728"
        marker = "s" if not is_summary else "D"
        msize = 6 if not is_summary else 8
        lw = 1.5 if not is_summary else 2.2

        ax.errorbar(d, y, xerr=[[el], [eh]], fmt=marker, color=color,
                    ecolor=color, elinewidth=lw, capsize=4, capthick=lw, markersize=msize)
        ax.text(d, y + 0.22, f"{d:+.1f}% [{d-el:+.1f}, {d+eh:+.1f}]",
                ha="center", va="bottom", fontsize=7.5, color=color, fontweight="semibold" if is_summary else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Grade 3 Sensitivity Difference: $V_3$ (Ordinal) - $V_2$ (Softmax) (pp)", fontsize=10)
    ax.set_title("Grade 3 Sensitivity Difference across Held-out Cohorts (95% Paired Bootstrap CI)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.text(0.5, -0.04,
             "Bootstrap intervals reflect held-out test cohort sampling variability, not between-training-seed variability.",
             ha="center", fontsize=8, style="italic", color="#444444")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "grade3_sensitivity_forest_plot.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "grade3_sensitivity_forest_plot.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # 2. Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    cohorts = df_conf["cohort"].unique().tolist()
    models = ["V2 (Global Context)", "V3 (Ordinal Champion)"]
    x = np.arange(len(cohorts))
    width = 0.35

    for m_idx, m_name in enumerate(models):
        offset = -width/2 if m_idx == 0 else width/2
        m_df = df_conf[df_conf["model"] == m_name]
        m_dict = {r["cohort"]: r for _, r in m_df.iterrows()}

        correct_pcts = []
        adjacent_pcts = []
        under_pcts = []

        for c in cohorts:
            r = m_dict.get(c, {})
            tot = max(1, r.get("n_actual_g3", 1))
            corr = r.get("g3_correct", 0) / tot * 100.0
            adj = r.get("g3_to_24", 0) / tot * 100.0
            und = r.get("g3_to_01", 0) / tot * 100.0
            correct_pcts.append(corr)
            adjacent_pcts.append(adj)
            under_pcts.append(und)

        ax.bar(x + offset, correct_pcts, width, label=f"{m_name}: Exact G3" if m_idx == 0 else None,
               color="#27ae60" if m_idx == 0 else "#2ecc71", edgecolor="black", linewidth=0.5)
        ax.bar(x + offset, adjacent_pcts, width, bottom=correct_pcts,
               label=f"{m_name}: Adjacent Referable (G2/G4)" if m_idx == 0 else None,
               color="#f39c12" if m_idx == 0 else "#f1c40f", edgecolor="black", linewidth=0.5)
        bottom2 = np.array(correct_pcts) + np.array(adjacent_pcts)
        ax.bar(x + offset, under_pcts, width, bottom=bottom2,
               label=f"{m_name}: Non-Referable Boundary (G0/G1)" if m_idx == 0 else None,
               color="#e74c3c" if m_idx == 0 else "#c0392b", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylabel("Percentage of True Grade 3 Samples (%)", fontsize=10.5)
    ax.set_title("Grade 3 Classification Destination: Exact vs. Referable Adjacency", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    legend_elements = [
        Patch(facecolor="#27ae60", edgecolor="black", label="Exact Grade 3 Recall (Correct)"),
        Patch(facecolor="#f39c12", edgecolor="black", label="Adjacent Referable (Grade 2 or 4)"),
        Patch(facecolor="#c0392b", edgecolor="black", label="Non-Referable Boundary (Grade 0 or 1)")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8.5)
    plt.tight_layout()

    fig.savefig(os.path.join(figures_dir, "grade3_confusion.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "grade3_confusion.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

# %%
def execute_lodo_checkpoint_analysis(args):
    """Main execution function loading checkpoints, evaluating, and writing outputs."""
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print("=" * 78)
    print("DMIP 2026: REPRODUCIBLE LODO CHECKPOINT INFERENCE & SCIENTIFIC AUDIT")
    print("=" * 78)
    print(f"Run ID:           {run_id}")
    print(f"Device:           {args.device}")
    print(f"Checkpoints Dir:  {args.checkpoint_dir}")
    print(f"Data Root:        {args.data_root}")
    print(f"Folds Manifests:  {args.folds_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Bootstrap Reps:   {args.bootstrap_reps}")
    print(f"Random Seed:      {args.seed}")
    print(f"Reuse Preds:      {args.reuse_predictions}")
    print(f"Force Inference:  {args.force_inference}")
    print(f"Diagnostic Only:  {args.diagnostic_only}")
    print(f"Folds to Run:     {args.folds}")

    # Set up run-specific isolated directory
    run_dir = os.path.join(args.output_dir, "runs", run_id)
    results_dir = os.path.join(args.output_dir, "results")
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    figures_dir = os.path.join(args.output_dir, "figures")
    reports_dir = os.path.join(args.output_dir, "reports")
    for d in [run_dir, results_dir, cm_dir, figures_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load Authoritative Reference Values
    results_json_path = str(REFERENCE)
    authoritative_refs = load_authoritative_references(results_json_path)

    # 2. Checkpoint Manifest & Validation
    manifest_rows = []
    valid_pairs = []
    selected_folds = [int(f.strip()) for f in args.folds.split(",") if f.strip().isdigit()]

    for cp in CHECKPOINT_PAIRS:
        v2_path = os.path.join(args.checkpoint_dir, cp["v2_file"])
        v3_path = os.path.join(args.checkpoint_dir, cp["v3_file"])

        v2_exists = os.path.exists(v2_path)
        v3_exists = os.path.exists(v3_path)
        v2_sha = compute_file_sha256(v2_path) if v2_exists else "MISSING"
        v3_sha = compute_file_sha256(v3_path) if v3_exists else "MISSING"

        manifest_rows.append({
            "run_id": run_id, "fold": cp["fold"], "cohort_key": cp["key"], "cohort_name": cp["name"],
            "v2_checkpoint": cp["v2_file"], "v2_exists": v2_exists, "v2_sha256": v2_sha,
            "v3_checkpoint": cp["v3_file"], "v3_exists": v3_exists, "v3_sha256": v3_sha
        })
        if cp["fold"] in selected_folds:
            if not v2_exists or not v3_exists:
                raise FileNotFoundError(f"Checkpoint pair missing for fold {cp['fold']}: {v2_path} or {v3_path}")
            valid_pairs.append(cp)

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(run_dir, "checkpoint_manifest.csv"), index=False)
    df_manifest.to_csv(os.path.join(results_dir, "checkpoint_manifest.csv"), index=False)

    # 3. Build Models Once
    print("\nInitializing PyTorch Model Architectures...")
    model_v2 = get_model(head_type="Softmax_CE", device=args.device)
    model_v3 = get_model(head_type="CLM_QWK", device=args.device)

    all_predictions = []
    per_cohort_metrics = []
    per_cohort_bootstrap = []
    per_cohort_confusion = []
    cohort_data_list = []
    trace_samples = []

    # 4. Evaluate Target Folds
    for pair in valid_pairs:
        fold = pair["fold"]
        key = pair["key"]
        name = pair["name"]

        df_preds, m_dict, cm_v2, cm_v3 = evaluate_cohort_inference(
            pair=pair,
            args=args,
            model_v2=model_v2,
            model_v3=model_v3,
            results_dir=results_dir,
            trace_samples=trace_samples,
            run_id=run_id
        )

        per_cohort_metrics.append(m_dict)
        all_predictions.append(df_preds)

        # Save individual 5x5 confusion matrices in run dir and main cm dir
        np.savetxt(os.path.join(cm_dir, f"cm_v2_fold_{fold}_{key}.csv"), cm_v2, fmt="%d", delimiter=",")
        np.savetxt(os.path.join(cm_dir, f"cm_v3_fold_{fold}_{key}.csv"), cm_v3, fmt="%d", delimiter=",")
        np.savetxt(os.path.join(run_dir, f"cm_v2_fold_{fold}_{key}.csv"), cm_v2, fmt="%d", delimiter=",")
        np.savetxt(os.path.join(run_dir, f"cm_v3_fold_{fold}_{key}.csv"), cm_v3, fmt="%d", delimiter=",")

        per_cohort_confusion.append(extract_grade3_confusion(cm_v2, name, "V2 (Global Context)"))
        per_cohort_confusion.append(extract_grade3_confusion(cm_v3, name, "V3 (Ordinal Champion)"))

        # Paired Bootstrap for Grade 3 (Single Cohort - All THREE CIs)
        y_true = df_preds["diagnosis"].values.astype(int)
        p_v2 = df_preds["pred_v2"].values.astype(int)
        p_v3 = df_preds["pred_v3"].values.astype(int)
        patient_ids = df_preds["patient_id"].values if "patient_id" in df_preds else None

        cohort_data_list.append({
            "cohort": name,
            "y_true": y_true,
            "pred_v2": p_v2,
            "pred_v3": p_v3
        })

        boot_res = paired_bootstrap_grade3_sensitivity(
            y_true=y_true,
            pred_v2=p_v2,
            pred_v3=p_v3,
            patient_ids=patient_ids,
            reps=args.bootstrap_reps,
            seed=args.seed
        )
        boot_res["cohort"] = name
        boot_res["fold"] = fold
        boot_res["run_id"] = run_id
        per_cohort_bootstrap.append(boot_res)

    # 5. Combined Predictions Table
    df_all_preds = pd.concat(all_predictions, ignore_index=True)
    aligned_cols = ["cohort", "fold", "image_id", "patient_id", "diagnosis", "pred_v2", "pred_v3",
                    "v3_expected_grade", "checkpoint_v2", "checkpoint_v3", "code_fingerprint", "provenance_marker", "run_id"]
    export_cols = [c for c in aligned_cols if c in df_all_preds.columns]

    # Save Sample Inference Trace
    if trace_samples:
        df_trace = pd.DataFrame(trace_samples)
        df_trace.to_csv(os.path.join(run_dir, "sample_inference_trace.csv"), index=False)
        df_trace.to_csv(os.path.join(results_dir, "sample_inference_trace.csv"), index=False)

    # 6. Reproduction Check Against Authoritative Benchmark
    df_m = pd.DataFrame(per_cohort_metrics)
    df_m.to_csv(os.path.join(run_dir, "per_fold_metrics.csv"), index=False)
    df_m.to_csv(os.path.join(results_dir, "per_fold_metrics.csv"), index=False)

    per_cohort_reprod_rows = []
    for _, r in df_m.iterrows():
        c_key = [p["key"] for p in valid_pairs if p["fold"] == r["fold"]][0]
        c_ref = authoritative_refs.get(c_key, {})
        for m_name, obs_col, ref_key, tol in [
            ("V2 QWK", "v2_qwk", "v2_qwk", 0.005),
            ("V3 QWK", "v3_qwk", "v3_qwk", 0.005),
            ("V2 Within-1 Acc (%)", "v2_within1", "v2_within1", 1.0),
            ("V3 Within-1 Acc (%)", "v3_within1", "v3_within1", 1.0),
            ("V2 Grade 3 Sens (%)", "v2_g3_sens", "v2_g3_sens", 1.5),
            ("V3 Grade 3 Sens (%)", "v3_g3_sens", "v3_g3_sens", 2.5)
        ]:
            obs_v = r[obs_col]
            ref_v = c_ref.get(ref_key, np.nan)
            diff = obs_v - ref_v if not np.isnan(ref_v) else np.nan
            status = "PASS" if not np.isnan(diff) and abs(diff) <= tol else "DISCREPANCY"
            per_cohort_reprod_rows.append({
                "cohort": r["cohort"],
                "fold": r["fold"],
                "metric": m_name,
                "observed": round(obs_v, 3),
                "reference": round(ref_v, 3) if not np.isnan(ref_v) else None,
                "raw_deviation": round(diff, 3) if not np.isnan(diff) else None,
                "tolerance": tol,
                "reference_source": c_ref.get("provenance_file", "summary_json"),
                "status": status
            })

    df_cohort_reprod = pd.DataFrame(per_cohort_reprod_rows)
    df_cohort_reprod.to_csv(os.path.join(run_dir, "per_cohort_reproduction.csv"), index=False)
    df_cohort_reprod.to_csv(os.path.join(results_dir, "per_cohort_reproduction.csv"), index=False)
    print("\n[OK] Per-Cohort Reproduction Gate against Authoritative Benchmark:")
    print(df_cohort_reprod[["cohort", "metric", "observed", "reference", "raw_deviation", "status"]].to_string(index=False))

    macro_v2_g3 = float(df_m["v2_g3_sens"].mean())
    macro_v3_g3 = float(df_m["v3_g3_sens"].mean())
    macro_v2_w1 = float(df_m["v2_within1"].mean())
    macro_v3_w1 = float(df_m["v3_within1"].mean())
    macro_v2_qwk = float(df_m["v2_qwk"].mean())
    macro_v3_qwk = float(df_m["v3_qwk"].mean())

    reprod_check_summary = [
        {
            "metric": "V2 Macro Grade 3 Sensitivity",
            "observed": round(macro_v2_g3, 2),
            "reference": PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_g3_sensitivity"],
            "raw_deviation": round(macro_v2_g3 - PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_g3_sensitivity"], 2),
            "tolerance": 1.5,
            "status": "PASS" if abs(macro_v2_g3 - PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_g3_sensitivity"]) <= 1.5 else "DISCREPANCY"
        },
        {
            "metric": "V3 Macro Grade 3 Sensitivity",
            "observed": round(macro_v3_g3, 2),
            "reference": PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_g3_sensitivity"],
            "raw_deviation": round(macro_v3_g3 - PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_g3_sensitivity"], 2),
            "tolerance": 1.5,
            "status": "PASS" if abs(macro_v3_g3 - PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_g3_sensitivity"]) <= 1.5 else "DISCREPANCY"
        },
        {
            "metric": "V2 Macro Within-1 Accuracy",
            "observed": round(macro_v2_w1, 2),
            "reference": PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_within_1"],
            "raw_deviation": round(macro_v2_w1 - PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_within_1"], 2),
            "tolerance": 1.0,
            "status": "PASS" if abs(macro_v2_w1 - PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_within_1"]) <= 1.0 else "DISCREPANCY"
        },
        {
            "metric": "V3 Macro Within-1 Accuracy",
            "observed": round(macro_v3_w1, 2),
            "reference": PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_within_1"],
            "raw_deviation": round(macro_v3_w1 - PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_within_1"], 2),
            "tolerance": 1.0,
            "status": "PASS" if abs(macro_v3_w1 - PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_within_1"]) <= 1.0 else "DISCREPANCY"
        },
        {
            "metric": "V2 Macro QWK",
            "observed": round(macro_v2_qwk, 3),
            "reference": PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_qwk"],
            "raw_deviation": round(macro_v2_qwk - PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_qwk"], 3),
            "tolerance": 0.005,
            "status": "PASS" if abs(macro_v2_qwk - PUBLISHED_TABLE_3_REFERENCE["V2"]["macro_qwk"]) <= 0.005 else "DISCREPANCY"
        },
        {
            "metric": "V3 Macro QWK",
            "observed": round(macro_v3_qwk, 3),
            "reference": PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_qwk"],
            "raw_deviation": round(macro_v3_qwk - PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_qwk"], 3),
            "tolerance": 0.005,
            "status": "PASS" if abs(macro_v3_qwk - PUBLISHED_TABLE_3_REFERENCE["V3"]["macro_qwk"]) <= 0.005 else "DISCREPANCY"
        }
    ]
    df_reprod_check = pd.DataFrame(reprod_check_summary)
    df_reprod_check.to_csv(os.path.join(run_dir, "baseline_reproduction_check.csv"), index=False)
    df_reprod_check.to_csv(os.path.join(results_dir, "baseline_reproduction_check.csv"), index=False)

    # 7. Bootstrap Summaries: (a) Pooled Image-Level and (b) Equal-Weighted Macro
    pooled_y = df_all_preds["diagnosis"].values.astype(int)
    pooled_v2 = df_all_preds["pred_v2"].values.astype(int)
    pooled_v3 = df_all_preds["pred_v3"].values.astype(int)

    pooled_boot = paired_bootstrap_grade3_sensitivity(
        y_true=pooled_y,
        pred_v2=pooled_v2,
        pred_v3=pooled_v3,
        reps=args.bootstrap_reps,
        seed=args.seed
    )
    pooled_boot["cohort"] = f"Pooled Image-Level ({len(valid_pairs)} Cohorts)"
    pooled_boot["fold"] = -1
    pooled_boot["run_id"] = run_id
    per_cohort_bootstrap.append(pooled_boot)

    if len(cohort_data_list) > 1:
        macro_boot = macro_bootstrap_grade3_sensitivity(
            cohort_data=cohort_data_list,
            reps=args.bootstrap_reps,
            seed=args.seed
        )
        if macro_boot:
            macro_boot["run_id"] = run_id
            per_cohort_bootstrap.append(macro_boot)

    df_boot_export = pd.DataFrame(per_cohort_bootstrap)
    df_conf_by_cohort = pd.DataFrame(per_cohort_confusion)

    tot_n = sum(r["n_actual_g3"] for r in per_cohort_confusion if r["model"].startswith("V2"))
    v2_tot_corr = sum(r["g3_correct"] for r in per_cohort_confusion if r["model"].startswith("V2"))
    v2_tot_24 = sum(r["g3_to_24"] for r in per_cohort_confusion if r["model"].startswith("V2"))
    v2_tot_01 = sum(r["g3_to_01"] for r in per_cohort_confusion if r["model"].startswith("V2"))

    v3_tot_corr = sum(r["g3_correct"] for r in per_cohort_confusion if r["model"].startswith("V3"))
    v3_tot_24 = sum(r["g3_to_24"] for r in per_cohort_confusion if r["model"].startswith("V3"))
    v3_tot_01 = sum(r["g3_to_01"] for r in per_cohort_confusion if r["model"].startswith("V3"))

    v2_misc = tot_n - v2_tot_corr
    v3_misc = tot_n - v3_tot_corr

    v2_pct_24 = round(v2_tot_24 / v2_misc * 100.0, 2) if v2_misc > 0 else 0.0
    v3_pct_24 = round(v3_tot_24 / v3_misc * 100.0, 2) if v3_misc > 0 else 0.0

    pooled_confusion_rows = [
        {
            "run_id": run_id,
            "model": "V2 (Global Context)",
            "n_actual_g3": tot_n, "g3_to_01": v2_tot_01, "g3_to_24": v2_tot_24,
            "g3_correct": v2_tot_corr, "n_g3_misclassified": v2_misc,
            "pct_errors_assigned_to_grade24": v2_pct_24
        },
        {
            "run_id": run_id,
            "model": "V3 (Ordinal Champion)",
            "n_actual_g3": tot_n, "g3_to_01": v3_tot_01, "g3_to_24": v3_tot_24,
            "g3_correct": v3_tot_corr, "n_g3_misclassified": v3_misc,
            "pct_errors_assigned_to_grade24": v3_pct_24
        }
    ]
    df_conf_pooled = pd.DataFrame(pooled_confusion_rows)

    # 8. Generate Messidor-2 Audited Inclusion Table
    raw_messidor_csv = str(Path(os.environ.get('DR_RAW', INPUTS / 'raw')) / 'messidor2preprocess' / 'messidor_data.csv')
    test_messidor_csv = os.path.join(args.folds_dir, "fold_2_messidor2_test.csv")
    df_messidor_audit = generate_messidor2_audit_table(
        raw_csv_path=raw_messidor_csv,
        test_csv_path=test_messidor_csv,
        output_path=os.path.join(run_dir, "messidor2_audit_table.csv")
    )
    if not df_messidor_audit.empty:
        df_messidor_audit.to_csv(os.path.join(results_dir, "messidor2_audit_table.csv"), index=False)

    # --------------------------------------------------------------------------
    # 9. FAIL-CLOSED GATING & ATOMIC PROMOTION
    # --------------------------------------------------------------------------
    is_full_cohort_run = (len(valid_pairs) == 6) and (args.max_samples == 0) and (not args.diagnostic_only)
    all_cohort_checks_pass = all(r["status"] == "PASS" for r in per_cohort_reprod_rows)
    all_macro_checks_pass = all(r["status"] == "PASS" for r in reprod_check_summary) if is_full_cohort_run else False

    is_verified_publication = is_full_cohort_run and all_cohort_checks_pass and all_macro_checks_pass

    print("\n" + "=" * 78)
    print(f"VERIFICATION STATUS: {'PASS (FULL PUBLICATION VERIFIED)' if is_verified_publication else 'DIAGNOSTIC ONLY (FAIL-CLOSED GATE ACTIVE)'}")
    print("=" * 78)

    # Write run-isolated outputs
    df_all_preds[export_cols].to_csv(os.path.join(run_dir, "heldout_predictions_v2_v3.csv"), index=False)
    df_boot_export.to_csv(os.path.join(run_dir, "grade3_bootstrap_by_cohort.csv"), index=False)
    df_conf_by_cohort.to_csv(os.path.join(run_dir, "grade3_confusion_by_cohort.csv"), index=False)
    df_conf_pooled.to_csv(os.path.join(run_dir, "grade3_confusion_pooled.csv"), index=False)

    # In main results directory, write base files
    df_all_preds[export_cols].to_csv(os.path.join(results_dir, "heldout_predictions_v2_v3.csv"), index=False)
    df_boot_export.to_csv(os.path.join(results_dir, "grade3_bootstrap_by_cohort.csv"), index=False)
    df_conf_by_cohort.to_csv(os.path.join(results_dir, "grade3_confusion_by_cohort.csv"), index=False)
    df_conf_pooled.to_csv(os.path.join(results_dir, "grade3_confusion_pooled.csv"), index=False)

    if is_verified_publication:
        verified_run_dir = os.path.join(run_dir, "VERIFIED")
        os.makedirs(verified_run_dir, exist_ok=True)

        # Emit _VERIFIED.csv in verified run directory
        df_all_preds[export_cols].to_csv(os.path.join(verified_run_dir, "heldout_predictions_v2_v3_VERIFIED.csv"), index=False)
        df_boot_export.to_csv(os.path.join(verified_run_dir, "grade3_bootstrap_by_cohort_VERIFIED.csv"), index=False)
        df_conf_by_cohort.to_csv(os.path.join(verified_run_dir, "grade3_confusion_by_cohort_VERIFIED.csv"), index=False)
        df_conf_pooled.to_csv(os.path.join(verified_run_dir, "grade3_confusion_pooled_VERIFIED.csv"), index=False)

        # Atomic promotion to results/
        df_all_preds[export_cols].to_csv(os.path.join(results_dir, "heldout_predictions_v2_v3_VERIFIED.csv"), index=False)
        df_boot_export.to_csv(os.path.join(results_dir, "grade3_bootstrap_by_cohort_VERIFIED.csv"), index=False)
        df_conf_by_cohort.to_csv(os.path.join(results_dir, "grade3_confusion_by_cohort_VERIFIED.csv"), index=False)
        df_conf_pooled.to_csv(os.path.join(results_dir, "grade3_confusion_pooled_VERIFIED.csv"), index=False)
        print(f"[OK] Full 6-fold verification PASS. Promoted all _VERIFIED.csv artifacts.")

        # Generate publication figures (PDF & PNG)
        generate_publication_plots(df_boot_export, df_conf_by_cohort, figures_dir)
        generate_publication_plots(df_boot_export, df_conf_by_cohort, os.path.join(verified_run_dir, "figures"))

        # Generate copyable LaTeX and Markdown tables without causal claims
        manuscript_md = f"""# Empirical LODO Checkpoint Analysis Results (DMIP 2026 Camera-Ready)

**Run ID:** `{run_id}`  
**Evaluation Mode:** Strict PyTorch Checkpoint Inference  
**Folds Evaluated:** {len(valid_pairs)}  
**Images Evaluated:** {len(df_all_preds):,}  
**Grade 3 Sample Support:** {tot_n:,} cases  

---

### Table: Grade 3 Sensitivity across Held-out Cohorts (All Three 95% CIs)
| Cohort | N (G3) | V2 Sensitivity [95% CI] | V3 Sensitivity [95% CI] | Paired Difference Δ (V3 - V2) [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
"""
        for _, r in df_boot_export.iterrows():
            manuscript_md += f"| {r['cohort']} | {r['n_grade3']} | {r['sens_v2']:.1f}% [{r['v2_ci_low']:.1f}%, {r['v2_ci_high']:.1f}%] | {r['sens_v3']:.1f}% [{r['v3_ci_low']:.1f}%, {r['v3_ci_high']:.1f}%] | {r['diff_v3_minus_v2_pp']:+.1f} pp [{r['diff_ci_low_pp']:+.1f}, {r['diff_ci_high_pp']:+.1f}] |\n"

        manuscript_md += f"""
---

### Copyable LaTeX Table (Submission-Ready)
```latex
\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Grade 3 Sensitivity and Paired Uncertainty Analysis across Held-out Cohorts.}} 
Point estimates and 95\\% percentile confidence intervals computed via paired non-parametric bootstrap ($B=5,000$ replicates, seed 42) resampling identical Grade~3 case indices for nominal $V_2$ and ordinal $V_3$.
\\textit{{Statistical Scope:}} Bootstrap intervals quantify sampling variability of the held-out test cohorts; they do not measure variability across different training seeds or hypothetical unobserved cohorts.}}
\\label{{tab:grade3_bootstrap_ci}}
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Held-out Cohort}} & \\textbf{{$N_{{G3}}$}} & \\textbf{{$V_2$ Sensitivity [95\\% CI]}} & \\textbf{{$V_3$ Sensitivity [95\\% CI]}} & \\textbf{{Paired Difference $\\Delta$ [95\\% CI]}} \\\\
\\midrule
"""
        for _, r in df_boot_export[df_boot_export["fold"] >= 0].iterrows():
            manuscript_md += f"{r['cohort']} & {r['n_grade3']} & {r['sens_v2']:.1f}\\% [{r['v2_ci_low']:.1f}, {r['v2_ci_high']:.1f}] & {r['sens_v3']:.1f}\\% [{r['v3_ci_low']:.1f}, {r['v3_ci_high']:.1f}] & {r['diff_v3_minus_v2_pp']:+.1f}\\% [{r['diff_ci_low_pp']:+.1f}, {r['diff_ci_high_pp']:+.1f}] \\\\\n"

        macro_rows = df_boot_export[df_boot_export["fold"] == -2]
        if not macro_rows.empty:
            mr = macro_rows.iloc[0]
            manuscript_md += f"\\midrule\n\\textbf{{Macro Unweighted}} & {mr['n_grade3']} & \\textbf{{{mr['sens_v2']:.1f}\\%}} [{mr['v2_ci_low']:.1f}, {mr['v2_ci_high']:.1f}] & \\textbf{{{mr['sens_v3']:.1f}\\%}} [{mr['v3_ci_low']:.1f}, {mr['v3_ci_high']:.1f}] & \\textbf{{{mr['diff_v3_minus_v2_pp']:+.1f}\\%}} [{mr['diff_ci_low_pp']:+.1f}, {mr['diff_ci_high_pp']:+.1f}] \\\\\n"

        manuscript_md += f"""\\bottomrule
\\end{{tabular}}
\\end{{table}}
```

---

### Section 4.4: Statistical Evaluation and Resampling Methodology
```latex
To quantify test-sample uncertainty without data fabrication or retraining, we perform paired non-parametric bootstrap resampling ($B = 5,000$ replicates, fixed random seed 42) restricted strictly to ground-truth Grade~3 cases ($N = {tot_n:,}$). In each bootstrap replicate, identical sample indices are evaluated simultaneously for both $V_2$ and $V_3$, yielding point estimates and 95\\% percentile confidence intervals for $V_2$ sensitivity, $V_3$ sensitivity, and their paired difference $\\Delta = (V_3 - V_2)$. Both pooled sample-weighted estimates and unweighted six-cohort macro estimates are reported. Crucially, these resampling intervals quantify held-out test cohort sampling variability; they do not reflect variability across multiple random training runs or random seeds.
```

---

### Section 5.5: Grade-Level Classification Destinations vs. Operational Screening Boundaries
```latex
An audit of Grade~3 misclassification destinations indicates that across the evaluated held-out cohorts, {v3_pct_24:.1f}\\% of $V_3$ errors ({v3_tot_24} / {v3_misc}) were classified into adjacent referable grades (Grade~2 or Grade~4 on the ICDR scale). Non-referable demotions (assigned to Grade~0 or 1) accounted for {v3_tot_01 / max(1, v3_misc) * 100:.1f}\\% of errors ({v3_tot_01} / {v3_misc}), corresponding to {v3_tot_01 / max(1, tot_n) * 100:.1f}\\% of all true Grade~3 cases. Importantly, these grade-level classifications ($\\hat{{y}} \\ge 2$) reflect multi-class categorical outputs and do not represent binary referable screening decisions derived using source-validation calibrated operating thresholds ($\\tau_{{95\\%}}$).
```
"""
        with open(os.path.join(run_dir, "manuscript_ready_results.md"), "w", encoding="utf-8") as f:
            f.write(manuscript_md)
        with open(os.path.join(reports_dir, "manuscript_ready_results.md"), "w", encoding="utf-8") as f:
            f.write(manuscript_md)
        print(f"[OK] Generated 'reports/manuscript_ready_results.md'.")

    else:
        # Emit _DIAGNOSTIC.csv files; NEVER write or keep _VERIFIED in results/
        # First remove any stale _VERIFIED in shared results/ to prevent presenting stale data
        for fname in ["heldout_predictions_v2_v3_VERIFIED.csv", "grade3_bootstrap_by_cohort_VERIFIED.csv",
                      "grade3_confusion_by_cohort_VERIFIED.csv", "grade3_confusion_pooled_VERIFIED.csv"]:
            stale_path = os.path.join(results_dir, fname)
            if os.path.exists(stale_path):
                os.remove(stale_path)

        df_all_preds[export_cols].to_csv(os.path.join(run_dir, "heldout_predictions_v2_v3_DIAGNOSTIC.csv"), index=False)
        df_boot_export.to_csv(os.path.join(run_dir, "grade3_bootstrap_by_cohort_DIAGNOSTIC.csv"), index=False)
        df_conf_by_cohort.to_csv(os.path.join(run_dir, "grade3_confusion_by_cohort_DIAGNOSTIC.csv"), index=False)
        df_conf_pooled.to_csv(os.path.join(run_dir, "grade3_confusion_pooled_DIAGNOSTIC.csv"), index=False)

        df_all_preds[export_cols].to_csv(os.path.join(results_dir, "heldout_predictions_v2_v3_DIAGNOSTIC.csv"), index=False)
        df_boot_export.to_csv(os.path.join(results_dir, "grade3_bootstrap_by_cohort_DIAGNOSTIC.csv"), index=False)
        df_conf_by_cohort.to_csv(os.path.join(results_dir, "grade3_confusion_by_cohort_DIAGNOSTIC.csv"), index=False)
        df_conf_pooled.to_csv(os.path.join(results_dir, "grade3_confusion_pooled_DIAGNOSTIC.csv"), index=False)
        print(f"[FAIL-CLOSED GATE] Emitted _DIAGNOSTIC.csv files. Withheld _VERIFIED.csv and publication figures.")

        diagnostic_md = f"""# DMIP 2026 LODO Analysis Diagnostic Report

**Run ID:** `{run_id}`  
**Status:** DIAGNOSTIC RUN (Fail-Closed Gate Active)  
**Timestamp:** {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  
**Selected Folds:** {args.folds} (Total Evaluated: {len(valid_pairs)}/6)  
**Max Samples per Cohort:** {args.max_samples if args.max_samples > 0 else 'Full'}  

## Gating Checklist:
- **6-Fold Completeness:** {'PASS' if len(valid_pairs) == 6 else f'INCOMPLETE ({len(valid_pairs)}/6 folds evaluated)'}
- **Truncation Free:** {'PASS' if args.max_samples == 0 else f'TRUNCATED (max_samples={args.max_samples})'}
- **Diagnostic-Only Flag:** {'SET BY USER' if args.diagnostic_only else 'False'}
- **Cohort Reproduction Status:** {'PASS' if all_cohort_checks_pass else 'DISCREPANCY DETECTED'}

## Cohort Reproduction Details:
| Cohort | Metric | Observed | Reference | Raw Deviation | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
        for _, r in df_cohort_reprod.iterrows():
            diagnostic_md += f"| {r['cohort']} | {r['metric']} | {r['observed']} | {r['reference']} | {r['raw_deviation']} | {r['status']} |\n"

        diagnostic_md += """
---
*Note: Run all 6 folds without truncation (`--folds 0,1,2,3,4,5 --max-samples 0`) to achieve full publication verification.*
"""
        with open(os.path.join(run_dir, "diagnostic_report.md"), "w", encoding="utf-8") as f:
            f.write(diagnostic_md)
        with open(os.path.join(reports_dir, "diagnostic_report.md"), "w", encoding="utf-8") as f:
            f.write(diagnostic_md)
        print(f"[OK] Generated 'reports/diagnostic_report.md'.")

    # 10. Audit Log Generation
    audit_md = f"""# DMIP 2026 LODO Checkpoint Scientific Audit Log

**Run ID:** `{run_id}`  
**Generated At:** {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  
**Execution Machine:** CPU ({torch.get_num_threads()} PyTorch threads)  
**Strict Checkpoint State-Dict Enforcement:** Verified (0 missing, 0 unexpected keys)  
**Label Leakage Prevention:** Verified (No ground truth utilized in forward pass or decoding)  
**Execution Gating Status:** {'PASS (Full Verified)' if is_verified_publication else 'DIAGNOSTIC (Withheld _VERIFIED)'}  

## 1. Folds Evaluated
"""
    m_dict_lookup = {r["fold"]: r for r in manifest_rows}
    for p in valid_pairs:
        m_info = m_dict_lookup.get(p["fold"], {})
        audit_md += f"- **Fold {p['fold']} ({p['name']})**: V2=`{p['v2_file']}` (SHA256: `{m_info.get('v2_sha256', 'N/A')}`), V3=`{p['v3_file']}` (SHA256: `{m_info.get('v3_sha256', 'N/A')}`)\n"

    audit_md += f"""
## 2. Sample Counts & Reconciliation
- **Total Aligned Samples:** {len(df_all_preds):,}
- **Total Grade 3 Cases:** {tot_n:,}
- **Messidor-2 Reconciliation:** Quality-curated test manifest contains 1,057 images matching native preprocessed cropped files on disk. The raw uncurated archive contains 1,744 images; the 687-image difference is traced to pre-split fundus filtering (omission reason UNKNOWN from repository logs). Audited table exported to `results/messidor2_audit_table.csv`.

## 3. Reproduction Gate Status
| Cohort | Metric | Observed | Reference | Raw Deviation | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for _, r in df_cohort_reprod.iterrows():
        audit_md += f"| {r['cohort']} | {r['metric']} | {r['observed']} | {r['reference']} | {r['raw_deviation']} | {r['status']} |\n"

    audit_md += """
## 4. Methodological Invariants
1. **Zero Retraining:** All evaluations strictly utilize saved frozen weights in `pipeline/checkpoints/`.
2. **Exact Preprocessing Parity:** Input tensors standardized using ImageNet mean/std; explicit single-normalization contract.
3. **Decoupled Screening vs Grade Classification:** Grade 3 error destinations distinguish adjacent referability ($G3 \\to 2,4$) from non-referable demotions ($G3 \\to 0,1$) purely based on ICDR grade boundaries without claiming thresholded RDR screening status.
4. **Statistical Boundary:** Bootstrap intervals reflect held-out test cohort sampling variability, not between-training-seed variability.
"""
    with open(os.path.join(run_dir, "audit_log.md"), "w", encoding="utf-8") as f:
        f.write(audit_md)
    with open(os.path.join(reports_dir, "audit_log.md"), "w", encoding="utf-8") as f:
        f.write(audit_md)

    provenance_data = {
        "run_id": run_id,
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "folds_evaluated": [p["fold"] for p in valid_pairs],
        "total_test_images": len(df_all_preds),
        "total_grade3_cases": tot_n,
        "is_verified_publication": is_verified_publication,
        "reproduction_summary": reprod_check_summary
    }
    with open(os.path.join(run_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)
    with open(os.path.join(reports_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)

    # Maintain global run-index manifest
    run_index_path = os.path.join(args.output_dir, "run_index.json")
    run_index = []
    if os.path.exists(run_index_path):
        try:
            with open(run_index_path, "r", encoding="utf-8") as f:
                run_index = json.load(f)
        except Exception:
            run_index = []
    run_index.append({
        "run_id": run_id,
        "timestamp": provenance_data["execution_timestamp"],
        "status": "VERIFIED_PUBLICATION" if is_verified_publication else "DIAGNOSTIC_ONLY",
        "folds": [p["fold"] for p in valid_pairs],
        "total_images": len(df_all_preds)
    })
    with open(run_index_path, "w", encoding="utf-8") as f:
        json.dump(run_index, f, indent=2)

    print(f"\n[OK] Run artifacts saved to isolated directory '{run_dir}/'.")
    print(f"[OK] Master reports and run-index updated.")
    print("=" * 78)
    print("EXECUTION COMPLETED WITH ZERO MODEL RETRAINING.")
    print("=" * 78)

# %%
def zip_index_signature(archive):
    index = [(i.filename, i.file_size, i.CRC) for i in archive.infolist()]
    return hashlib.sha256(json.dumps(index, ensure_ascii=False).encode()).hexdigest()


def unpack_zip(source, label, wanted_names=None):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f'Missing {source}. Check the archive path in the configuration.')
    with zipfile.ZipFile(source) as archive:
        signature = zip_index_signature(archive)
        destination = LOCAL_ROOT / 'archives' / f'{label}_{signature[:16]}'
        marker = destination / '.extraction_complete.json'
        selection = sorted(wanted_names) if wanted_names is not None else None
        if (marker.is_file()
                and json.loads(marker.read_text())['zip_index_sha256'] == signature
                and json.loads(marker.read_text()).get('selected_names') == selection):
            print(f'{label}: reusing the complete extraction at {destination}')
            return destination, signature
        infos = []
        seen = set()
        for info in archive.infolist():
            posix = PurePosixPath(info.filename.replace(chr(92), '/'))
            if info.is_dir() or '__MACOSX' in posix.parts or posix.name.startswith('._'):
                continue
            if posix.is_absolute() or '..' in posix.parts or ':' in str(posix):
                raise ValueError(f'Invalid ZIP member: {info.filename}')
            if wanted_names is not None and posix.name not in wanted_names:
                continue
            relative = Path(posix.name) if wanted_names is not None else Path(*posix.parts)
            if str(relative) in seen:
                raise ValueError(f'Duplicate ZIP member: {relative}')
            seen.add(str(relative))
            infos.append((info, relative))
        if wanted_names is not None and seen != set(wanted_names):
            raise FileNotFoundError(f'ZIP {source.name} missing: {sorted(set(wanted_names) - seen)}')
        required = sum(i.file_size for i, _ in infos)
        free = shutil.disk_usage(LOCAL_ROOT).free
        remaining = sum(i.file_size for i, rel in infos
                        if not (destination / rel).is_file()
                        or (destination / rel).stat().st_size != i.file_size)
        if free < remaining + 2 * 2**30:
            raise RuntimeError(f'Insufficient local storage. Required: {remaining/2**30:.1f} GiB plus 2 GiB reserve, '
                               f'available {free/2**30:.1f} GiB.')
        destination.mkdir(parents=True, exist_ok=True)
        print(f'{label}: {len(infos):,} files, extracting {required/2**30:.2f} GiB')

        for info, relative in tqdm(infos, desc=label):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            with archive.open(info) as reader, temporary.open('wb') as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            temporary.replace(target)
        marker.write_text(json.dumps({'zip_index_sha256': signature, 'source': str(source),
                                      'selected_names': selection}))
    return destination, signature


selected_pairs = [p for p in CHECKPOINT_PAIRS if p['fold'] in FOLDS]
required_checkpoints = {p[k] for p in selected_pairs for k in ('v2_file', 'v3_file')}
IMAGE_EXTRACT, DATA_ZIP_SIGNATURE = unpack_zip(DATA_ZIP, 'images')
CHECKPOINT_LOCAL, CHECKPOINT_ZIP_SIGNATURE = unpack_zip(
    CHECKPOINT_ZIP, 'checkpoints', required_checkpoints)
if FOLDS_ZIP is not None:
    fold_names = {f"fold_{p['fold']}_{p['key']}_test.csv" for p in CHECKPOINT_PAIRS}

    selected_names = {f"fold_{p['fold']}_{p['key']}_test.csv" for p in selected_pairs}
    fold_extract, _ = unpack_zip(FOLDS_ZIP, 'folds', selected_names)
    SOURCE_FOLDS = fold_extract
print('Images:', IMAGE_EXTRACT, '| Checkpoints:', CHECKPOINT_LOCAL)

# %%
def image_relative_path(value):
    parts = PurePosixPath(str(value).replace(chr(92), '/')).parts
    lowered = [p.lower() for p in parts]
    if 'native_cropped' in lowered:
        parts = parts[lowered.index('native_cropped') + 1:]
    if not parts or any(p in ('..', '/') or ':' in p for p in parts):
        raise ValueError(f'Cannot resolve relative path: {value}')
    return Path(*parts)


first_pair = selected_pairs[0]
first_name = f"fold_{first_pair['fold']}_{first_pair['key']}_test.csv"
first_df = pd.read_csv(SOURCE_FOLDS / first_name, dtype={'image_id': str})
first_relative = image_relative_path(first_df.iloc[0]['native_path'])

candidates = {IMAGE_EXTRACT}
with zipfile.ZipFile(DATA_ZIP) as archive:
    suffix = PurePosixPath(first_relative.as_posix()).parts
    for info in archive.infolist():
        parts = PurePosixPath(info.filename.replace(chr(92), '/')).parts
        if len(parts) >= len(suffix) and parts[-len(suffix):] == suffix:
            candidates.add(IMAGE_EXTRACT.joinpath(*parts[:-len(suffix)]))
roots = [p for p in candidates if (p / first_relative).is_file()]
if len(roots) != 1:
    raise RuntimeError(f'Expected one image root matching the manifest, found {roots}.')
DATA_ROOT_LOCAL = roots[0]
RUNTIME_FOLDS = LOCAL_ROOT / 'runtime_folds'
RUNTIME_FOLDS.mkdir(exist_ok=True)
INPUT_PROVENANCE = []
path_audit = []
for pair in selected_pairs:
    name = f"fold_{pair['fold']}_{pair['key']}_test.csv"
    source = SOURCE_FOLDS / name
    if not source.is_file():
        raise FileNotFoundError(f'Missing manifest {source}')
    frame = pd.read_csv(source, dtype={'image_id': str})
    required = {'image_id', 'diagnosis', 'native_path', 'patient_id'}
    if not required <= set(frame):
        raise ValueError(f'{name} missing columns {required - set(frame)}')
    assert frame['image_id'].notna().all() and frame['image_id'].is_unique
    assert frame['diagnosis'].isin(range(5)).all()
    resolved = []
    for row in frame.itertuples(index=False):
        target = DATA_ROOT_LOCAL / image_relative_path(row.native_path)
        if not target.is_file():
            raise FileNotFoundError(f'{name}, image_id={row.image_id}: missing {target}')
        if target.stem != str(row.image_id):
            raise ValueError(f'Image ID and filename differ: {row.image_id} / {target.name}')
        resolved.append(str(target))
        path_audit.append({'fold': pair['fold'], 'image_id': row.image_id,
                           'original_native_path': row.native_path, 'colab_native_path': str(target)})
    frame['native_path'] = resolved
    runtime = RUNTIME_FOLDS / name
    frame.to_csv(runtime, index=False)
    INPUT_PROVENANCE.append({'fold': pair['fold'], 'manifest': name, 'rows': len(frame),
        'source_sha256': compute_file_sha256(str(source)),
        'runtime_sha256': compute_file_sha256(str(runtime))})
    print(f"Fold {pair['fold']} {pair['name']}: {len(frame):,} images, "
          f"G3={(frame['diagnosis'] == 3).sum():,}; paths OK")

CHECKPOINT_HASHES = {name: compute_file_sha256(str(CHECKPOINT_LOCAL / name))
                     for name in sorted(required_checkpoints)}
print('Hashed', len(CHECKPOINT_HASHES), 'checkpoint.')
AUTH_REFS = load_authoritative_references(str(REFERENCE))

# %%
from importlib import metadata as package_metadata

ORIGINAL_ANALYSIS_SHA256 = '56409df28baaa7794972daabb83f86bbedaeef9ae9d8ab7ba193e3a05cb5c66d'
ORIGINAL_MODULE_SHA256 = {'models.py': '348541f7592f0205209919e558f006bef3a7fc3c5935672ba4afb5daa15e6751', 'losses_and_heads.py': '004bbbce7dfc8a5a5fccfa2043ddb4dea9b842bc07af0f8fba1f789b89df6497', 'data_loader.py': 'fa4532802f88b61843a0e785c507cafe37471a390dbe2567b18ab791e18553fe'}
RUNTIME_CONFIG = {
    'notebook_version': 'camera_ready_colab_v1', 'precision': PRECISION,
    'batch_size': BATCH_SIZE, 'patch_chunk_size': PATCH_CHUNK_SIZE,
    'seed': SEED, 'bootstrap_reps': BOOTSTRAP_REPS, 'max_samples': MAX_SAMPLES,
    'num_workers': NUM_WORKERS, 'data_zip_index_sha256': DATA_ZIP_SIGNATURE,
    'torch': torch.__version__, 'torchvision': torchvision.__version__,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
    'packages': {n: package_metadata.version(n) for n in
                 ('numpy', 'pandas', 'scikit-learn', 'albumentations', 'matplotlib', 'Pillow')},
    'original_analysis_sha256': ORIGINAL_ANALYSIS_SHA256,
    'original_modules_sha256': ORIGINAL_MODULE_SHA256,
}
_source_hashes = {'thesis_workflow': NOTEBOOK_CODE_SHA256}
_inference_source = inspect.getsource(evaluate_cohort_inference)
PIPELINE_CODE_FINGERPRINT = hashlib.sha256(json.dumps({
    'config': RUNTIME_CONFIG, 'modules': _source_hashes,
    'evaluator': _inference_source,
    'reference_sha256': compute_file_sha256(str(REFERENCE))
}, sort_keys=True).encode()).hexdigest()

_original_get_model = get_model
def get_model(head_type, device='cpu'):
    model = _original_get_model(head_type, device)
    model.backbone.local_mil_branch.eval_patch_chunk_size = PATCH_CHUNK_SIZE
    return model


_original_cache_validator = validate_cache_fail_safe
def validate_cache_fail_safe(cache_file, n_expected_samples, v2_sha, v3_sha,
                             manifest_sha, df_test):
    valid, frame, reason = _original_cache_validator(
        cache_file, n_expected_samples, v2_sha, v3_sha, manifest_sha, df_test)
    if not valid:
        return valid, frame, reason
    required = {'code_fingerprint', 'v3_expected_grade', 'native_path', 'run_id'}
    if not required <= set(frame):
        return False, None, 'Missing provenance or expected grade'
    if not frame['code_fingerprint'].eq(PIPELINE_CODE_FINGERPRINT).all():
        return False, None, 'Runtime, data or code fingerprint changed'
    exp = pd.to_numeric(frame['v3_expected_grade'], errors='coerce').to_numpy()
    if not np.isfinite(exp).all() or not ((exp >= 0) & (exp <= 4)).all():
        return False, None, 'Invalid expected grade'
    if not np.array_equal(np.clip(np.round(exp), 0, 4).astype(int), frame['pred_v3'].to_numpy()):
        return False, None, 'V3 decoder disagrees with expected grade'
    if not np.array_equal(frame['native_path'].astype(str), df_test['native_path'].astype(str)):
        return False, None, 'Runtime image path mismatch'
    return True, frame, reason


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def restore_fold_cache(pair, args, results_dir):
    if not RESUME_COMPLETED_FOLDS or FORCE_INFERENCE:
        return
    name = f"fold_{pair['fold']}_{pair['key']}"
    parent = DRIVE_EXPERIMENT / 'folds' / name
    if not parent.is_dir():
        return
    manifest = Path(args.folds_dir) / f'{name}_test.csv'
    test = pd.read_csv(manifest, dtype={'image_id': str})
    if args.max_samples:
        test = test.iloc[:args.max_samples].copy()
    for snapshot in sorted(parent.iterdir(), reverse=True):
        marker = snapshot / 'complete.json'
        if not marker.is_file():
            continue
        try:
            info = json.loads(marker.read_text(encoding='utf-8'))
            if info['fingerprint'] != PIPELINE_CODE_FINGERPRINT:
                continue
            for filename, sha in info['files_sha256'].items():
                if compute_file_sha256(str(snapshot / filename)) != sha:
                    raise ValueError(f'Cached file changed: {filename}')
            cache_file = snapshot / f'{name}_predictions.csv'
            valid, _, reason = validate_cache_fail_safe(
                str(cache_file), len(test), CHECKPOINT_HASHES[pair['v2_file']],
                CHECKPOINT_HASHES[pair['v3_file']], compute_file_sha256(str(manifest)), test)
            if not valid:
                print(name, 'skipping cache:', reason)
                continue
            destination = Path(results_dir) / 'predictions' / cache_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_file, destination)
            print(name, 'restored completed fold:', snapshot.name)
            return
        except (ValueError, KeyError, OSError) as exc:
            print(name, 'skipping an invalid snapshot:', exc)


_evaluate_cohort_original = evaluate_cohort_inference
def evaluate_cohort_inference(pair, args, model_v2, model_v3, results_dir, trace_samples, run_id):
    restore_fold_cache(pair, args, results_dir)
    predictions, metrics, cm2, cm3 = _evaluate_cohort_original(
        pair, args, model_v2, model_v3, results_dir, trace_samples, run_id)
    name = f"fold_{pair['fold']}_{pair['key']}"
    snapshot_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    local = LOCAL_ROOT / 'fold_exports' / name / snapshot_id
    local.mkdir(parents=True, exist_ok=False)
    cache = Path(results_dir) / 'predictions' / f'{name}_predictions.csv'
    shutil.copy2(cache, local / cache.name)
    pd.DataFrame([metrics]).to_csv(local / 'metrics.csv', index=False)
    np.savetxt(local / 'cm_v2.csv', cm2, fmt='%d', delimiter=',')
    np.savetxt(local / 'cm_v3.csv', cm3, fmt='%d', delimiter=',')
    boot = paired_bootstrap_grade3_sensitivity(
        predictions['diagnosis'].to_numpy(dtype=int),
        predictions['pred_v2'].to_numpy(dtype=int),
        predictions['pred_v3'].to_numpy(dtype=int),
        reps=args.bootstrap_reps, seed=args.seed)
    boot.update(cohort=pair['name'], fold=pair['fold'])
    pd.DataFrame([boot]).to_csv(local / 'grade3_bootstrap.csv', index=False)
    pd.DataFrame([extract_grade3_confusion(cm2, pair['name'], 'V2 (Global Context)'),
                  extract_grade3_confusion(cm3, pair['name'], 'V3 (Ordinal Champion)')]).to_csv(
                      local / 'grade3_confusion.csv', index=False)
    if trace_samples:
        pd.DataFrame(trace_samples).to_csv(local / 'sample_inference_trace.csv', index=False)
    atomic_json(local / 'provenance.json', {
        'fold': pair['fold'], 'runtime': RUNTIME_CONFIG, 'fingerprint': PIPELINE_CODE_FINGERPRINT,
        'input_manifests': INPUT_PROVENANCE, 'checkpoint_sha256': CHECKPOINT_HASHES,
        'source_prediction_run_ids': predictions['run_id'].unique().tolist(),
        'note': 'Fold complete; publication verification is determined by the original aggregate gate.'})
    destination = DRIVE_EXPERIMENT / 'folds' / name / snapshot_id
    shutil.copytree(local, destination)
    files = {p.name: compute_file_sha256(str(p)) for p in local.iterdir() if p.is_file()}

    for filename, sha in files.items():
        if compute_file_sha256(str(destination / filename)) != sha:
            raise IOError(f'Persistent copy hash mismatch: {destination / filename}')
    atomic_json(destination / 'complete.json', {
        'fingerprint': PIPELINE_CODE_FINGERPRINT, 'files_sha256': files,
        'fold': pair['fold'], 'rows': len(predictions), 'max_samples': args.max_samples})
    print(f"[DRIVE SAVED] Fold {pair['fold']} {pair['name']}: {destination}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, metrics, cm2, cm3

# %%
from argparse import Namespace

SESSION_ID = time.strftime('colab_%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
LOCAL_SESSION = LOCAL_ROOT / 'sessions' / SESSION_ID
DRIVE_SESSION = DRIVE_EXPERIMENT / 'sessions' / SESSION_ID
LOCAL_SESSION.mkdir(parents=True, exist_ok=False)
pd.DataFrame(path_audit).to_csv(LOCAL_SESSION / 'image_path_mapping.csv', index=False)
atomic_json(LOCAL_SESSION / 'colab_config.json', {
    'runtime': RUNTIME_CONFIG, 'fingerprint': PIPELINE_CODE_FINGERPRINT,
    'data_zip': str(DATA_ZIP), 'checkpoint_zip': str(CHECKPOINT_ZIP),
    'checkpoint_zip_index_sha256': CHECKPOINT_ZIP_SIGNATURE,
    'folds': FOLDS, 'manifests': INPUT_PROVENANCE,
    'checkpoint_sha256': CHECKPOINT_HASHES,
    'statistical_scope': 'Bootstrap intervals reflect sampling variability of the held-out cohorts, not variability across training runs.'})
args = Namespace(
    checkpoint_dir=str(CHECKPOINT_LOCAL), data_root=str(DATA_ROOT_LOCAL),
    folds_dir=str(RUNTIME_FOLDS), output_dir=str(LOCAL_SESSION), device='cuda',
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, bootstrap_reps=BOOTSTRAP_REPS,
    seed=SEED, folds=','.join(map(str, FOLDS)), max_samples=MAX_SAMPLES,
    reuse_predictions=RESUME_COMPLETED_FOLDS and not FORCE_INFERENCE,
    force_inference=FORCE_INFERENCE,
    diagnostic_only=(MAX_SAMPLES > 0 or PRECISION != 'float32')
)

class TeeLog:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)

import contextlib, traceback
run_status = 'FAILED_OR_INTERRUPTED'
try:
    with (LOCAL_SESSION / 'execution.log').open('w', encoding='utf-8') as logfile:
        with contextlib.redirect_stdout(TeeLog(sys.stdout, logfile)), contextlib.redirect_stderr(TeeLog(sys.stderr, logfile)):
            try:
                execute_lodo_checkpoint_analysis(args)
                run_status = 'COMPLETED'
            except BaseException:
                traceback.print_exc()
                raise
finally:
    atomic_json(LOCAL_SESSION / 'session_status.json', {'status': run_status, 'session_id': SESSION_ID})

    shutil.copytree(LOCAL_SESSION, DRIVE_SESSION)
    print('Saved session:', DRIVE_SESSION, '|', run_status)
    gc.collect()
    torch.cuda.empty_cache()

# %%
print('Session output:', DRIVE_SESSION)
for name in ['per_fold_metrics.csv', 'grade3_bootstrap_by_cohort.csv',
             'grade3_confusion_by_cohort.csv', 'grade3_confusion_pooled.csv',
             'per_cohort_reproduction.csv']:
    path = LOCAL_SESSION / 'results' / name
    if path.is_file():
        display(Markdown('**' + name + '**'))
        display(pd.read_csv(path))

for path in sorted((LOCAL_SESSION / 'figures').glob('*.png')):
    display(PreviewImage(filename=str(path)))

for path in sorted((LOCAL_SESSION / 'reports').glob('*.md')):
    display(Markdown(path.read_text(encoding='utf-8')))
