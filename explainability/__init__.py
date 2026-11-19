"""Explainable AI (XAI) modules: Grad-CAM, MIL Attention Saliency, IDRiD evaluation, and metrics."""

from explainability.gradcam import FundusGradCAM, GradCAM
from explainability.idrid_dataset import IDRiDLesionDataset, IDRiDXAIEvalDataset
from explainability.metrics import (
    area_matched_saliency_recall,
    compute_quantitative_xai,
    evaluate_xai_sample,
    pixel_auroc,
    pointing_game,
)
from explainability.mil_saliency import (
    MILAttentionSaliency,
    generate_attention_mil_saliency_map,
)

__all__ = [
    "GradCAM",
    "FundusGradCAM",
    "MILAttentionSaliency",
    "generate_attention_mil_saliency_map",
    "IDRiDLesionDataset",
    "IDRiDXAIEvalDataset",
    "pointing_game",
    "area_matched_saliency_recall",
    "pixel_auroc",
    "evaluate_xai_sample",
    "compute_quantitative_xai",
]
