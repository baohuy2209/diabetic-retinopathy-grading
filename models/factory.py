"""Unified model construction factory for diabetic retinopathy models and ablation variants."""

from typing import Union

import torch.nn as nn

from models.full_model import DRFullModel, DualBranchMILModel, ResNet50AblationModel
from models.variants import (
    build_v0_model,
    build_v1_model,
    build_v2_model,
    build_v3_model,
)


def build_model(
    head_type: str = "CLM_QWK",
    pretrained: bool = True,
    use_cbam: bool = True,
    use_dual_pooling: bool = True,
    use_local_branch: bool = True,
    use_nonlocal: bool = True,
    use_quadrant_tokens: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 0.4,
    dropout_rate: float = 0.3,
) -> DRFullModel:
    """Builds a full multi-scale diabetic retinopathy model with custom branch/head configuration."""
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=use_cbam,
        use_dual_pooling=use_dual_pooling,
        use_local_branch=use_local_branch,
        use_nonlocal=use_nonlocal,
        use_quadrant_tokens=use_quadrant_tokens,
        fusion_dim=fusion_dim,
        dropout_rate=dropout_rate,
    )
    return DRFullModel(
        backbone=backbone,
        head_type=head_type,
        num_classes=num_classes,
        qwk_weight=qwk_weight,
    )


def build_variant(
    variant: str,
    pretrained: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 0.4,
    dropout_rate: float = 0.3,
) -> DRFullModel:
    """Builds a standard paper ablation variant by name (v0, v1, v2, v3)."""
    variant_norm = variant.strip().lower()
    if variant_norm in ("v0", "baseline") or variant_norm.startswith("v0") or "v0_" in variant_norm:
        return build_v0_model(
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    if variant_norm in ("v1", "local_mil") or variant_norm.startswith("v1") or "v1_" in variant_norm:
        return build_v1_model(
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    if variant_norm in ("v2", "global_context") or variant_norm.startswith("v2") or "v2_" in variant_norm:
        return build_v2_model(
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    if variant_norm in ("v3", "clm", "ordinal_clm", "proposed", "champion") or variant_norm.startswith("v3") or "v3_" in variant_norm:
        return build_v3_model(
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            qwk_weight=qwk_weight,
            dropout_rate=dropout_rate,
        )
    raise ValueError(f"Unknown variant '{variant}'. Expected one of: v0, v1, v2, v3 or full variant name.")


def build_ablation_model(
    head_type: str = "CLM_QWK",
    pretrained: bool = True,
    num_classes: int = 5,
    qwk_weight: float = 0.4,
) -> ResNet50AblationModel:
    """Builds a standardized ResNet-50 single-branch model for head ablation."""
    return ResNet50AblationModel(
        head_type=head_type,
        pretrained=pretrained,
        num_classes=num_classes,
        qwk_weight=qwk_weight,
    )
