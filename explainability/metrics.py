"""Quantitative explainability evaluation metrics: Pointing Game, Area-Matched Saliency Recall, and pixel AUROC."""

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import auc, roc_curve


def pointing_game(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
    tolerance_pixels: int = 0,
) -> float:
    """Evaluates whether the peak saliency point falls within or near a ground-truth lesion.

    Args:
        saliency_map: Continuous 2D heatmap of shape (H, W).
        lesion_mask: Binary boolean or {0, 1} mask of shape (H, W).
        tolerance_pixels: Euclidean distance tolerance around ground truth lesions.

    Returns:
        1.0 for a hit (maximum within lesion + tolerance), 0.0 for a miss.
    """
    s_map = np.asarray(saliency_map, dtype=float)
    l_mask = np.asarray(lesion_mask, dtype=bool)

    if not np.any(l_mask):
        return float("nan")

    # Locate the peak saliency point
    y_peak, x_peak = np.unravel_index(np.argmax(s_map), s_map.shape)

    if tolerance_pixels <= 0:
        return 1.0 if l_mask[y_peak, x_peak] else 0.0

    # Expand lesion mask using Euclidean distance transform or dilation
    dist = distance_transform_edt(~l_mask)
    return 1.0 if dist[y_peak, x_peak] <= tolerance_pixels else 0.0


def area_matched_saliency_recall(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
) -> float:
    """Computes the fraction of lesion area covered by the top-k saliency pixels.

    The top fraction k is set exactly equal to the ground-truth lesion area proportion,
    eliminating bias from arbitrary binarization thresholds.

    Args:
        saliency_map: 2D heatmap of shape (H, W).
        lesion_mask: 2D binary ground-truth mask.

    Returns:
        Recall fraction in [0.0, 1.0].
    """
    s_map = np.asarray(saliency_map, dtype=float)
    l_mask = np.asarray(lesion_mask, dtype=bool)

    lesion_area = int(np.sum(l_mask))
    total_pixels = l_mask.size

    if lesion_area == 0 or total_pixels == 0:
        return float("nan")

    # Select exactly K pixels, including when scores tie at the cutoff.
    result = _xai_measure(s_map, l_mask, seed=42)
    return float(result["recall"]) if result["recall"] is not None else float("nan")


def pixel_auroc(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
) -> float:
    """Computes pixel-wise Area Under the Receiver Operating Characteristic curve.

    Args:
        saliency_map: Continuous 2D heatmap of shape (H, W).
        lesion_mask: Binary boolean mask of shape (H, W).

    Returns:
        Pixel AUROC score in [0.0, 1.0].
    """
    s_map = np.asarray(saliency_map, dtype=float)
    l_mask = np.asarray(lesion_mask, dtype=int)

    if len(np.unique(l_mask)) < 2:
        return float("nan")

    # Subsample if resolution is very large to avoid memory bottlenecks
    if s_map.size > 262144:  # > 512x512
        step = int(np.ceil(np.sqrt(s_map.size / 262144)))
        s_map = s_map[::step, ::step]
        l_mask = l_mask[::step, ::step]

    fpr, tpr, _ = roc_curve(l_mask.ravel(), s_map.ravel())
    score = float(auc(fpr, tpr))
    return score if not np.isnan(score) else 0.5


def evaluate_xai_sample(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
    tolerance_pixels: int = 0,
) -> Dict[str, float]:
    """Computes all three quantitative XAI metrics for a single image-saliency pair."""
    return {
        "pointing_game_accuracy": pointing_game(saliency_map, lesion_mask, tolerance_pixels=tolerance_pixels),
        "area_matched_saliency_recall": area_matched_saliency_recall(saliency_map, lesion_mask),
        "pixel_auroc": pixel_auroc(saliency_map, lesion_mask),
    }


def _xai_map_status(heatmap: np.ndarray) -> str:
    heatmap = np.asarray(heatmap)
    if not np.isfinite(heatmap).all():
        return "nonfinite_map"
    if float(np.ptp(heatmap)) <= 1e-12:
        return "constant_map"
    return "ok"


def _xai_measure(heatmap: np.ndarray, mask: np.ndarray, seed: int = 42, threshold: float = 0.5) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    mask = np.asarray(mask, dtype=bool)
    heatmap = np.asarray(heatmap)
    if heatmap.shape != mask.shape:
        raise ValueError("Heatmap and mask must share coordinates and shape")
    status = _xai_map_status(heatmap)
    if not mask.any():
        status = "no_positive_mask"
    result = dict(status=status, hit=None, recall=None, auc=None, iou=None)
    if status != "ok":
        return result
    flat = heatmap.ravel()
    permutation = np.random.default_rng(seed).permutation(flat.size)
    order = permutation[np.argsort(-flat[permutation], kind="stable")]
    k = int(mask.sum())
    selected = np.zeros(flat.size, dtype=bool)
    selected[order[:k]] = True
    result["hit"] = bool(mask.ravel()[order[0]])
    result["recall"] = float(mask.ravel()[selected].sum() / k) if k > 0 else 0.0
    result["selected_pixels"] = k
    if not mask.all() and mask.any():
        result["auc"] = float(roc_auc_score(mask.ravel(), flat))
    binary = heatmap >= threshold
    union = (binary | mask).sum()
    result["iou"] = float((binary & mask).sum() / union) if union > 0 else 0.0
    return result


def compute_quantitative_xai(
    model: Any,
    idrid_xai_dataset: Any,
    device: str = "cuda",
    cam_threshold: float = 0.5,
    return_samples: bool = False,
) -> Dict[str, Any]:
    """Computes rigorous quantitative XAI fidelity benchmarks matching Cell 30."""
    import torch
    from explainability.gradcam import FundusGradCAM
    from explainability.mil_saliency import generate_attention_mil_saliency_map

    if not len(idrid_xai_dataset):
        raise ValueError("No image/mask pairs found for XAI")
    model.to(device).eval()
    grad_cam = FundusGradCAM(model)
    records, samples = [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    mean_l = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std_l = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    for idx in range(len(idrid_xai_dataset)):
        sample = idrid_xai_dataset[idx]
        g = sample["global_img"].unsqueeze(0).to(device)
        l = sample["local_patches"].unsqueeze(0).to(device)
        if g.dtype == torch.uint8:
            g = g.float().div_(255.0).sub_(mean).div_(std)
        if l.dtype == torch.uint8:
            l = l.float().div_(255.0).sub_(mean_l).div_(std_l)

        mask = sample["mask_gt"].cpu().numpy() > 0
        cam, _ = grad_cam.generate_cam(g, l)
        cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]))
        mil = generate_attention_mil_saliency_map(
            model, g, l,
            sample["anchors"].cpu().numpy(),
            canvas_size=mask.shape,
            native_shape=sample["native_shape"],
        )
        gm = _xai_measure(cam, mask, seed=42 + idx, threshold=cam_threshold)
        mm = _xai_measure(mil, mask, seed=42 + idx, threshold=cam_threshold)
        if getattr(grad_cam, "last_status", "ok") != "ok":
            gm.update(status=grad_cam.last_status, hit=None, recall=None, auc=None, iou=None)
        q = float(mask.mean())
        records.append(dict(image_id=sample["image_id"], random_baseline=q, GradCAM=gm, MIL=mm))
        if return_samples and len(samples) < 5:
            samples.append(dict(
                image_id=sample["image_id"], global_img=sample["global_img"],
                mask_gt=mask, cam_global=cam, saliency_mil=mil, g_hit=gm["hit"], m_hit=mm["hit"],
                g_status=gm["status"], m_status=mm["status"], random_baseline=q,
            ))

    def average(values: List[Any], scale: float = 1.0) -> Optional[float]:
        valid_vals = [v for v in values if v is not None]
        return float(np.mean(valid_vals) * scale) if valid_vals else None

    results: Dict[str, Any] = {
        "Random_Baseline": average([r["random_baseline"] for r in records], 100.0),
        "n_images": len(records),
        "per_image": records,
        "protocol": "Exact K=mask-positive pixels; deterministic randomized ties; undefined metrics excluded with counts",
        "GradCAM_target": "CLM severity_score when available; otherwise selected logit",
        "pixel_domain": "entire resized crop (including pixels outside patch coverage)",
    }
    for method in ("GradCAM", "MIL"):
        valid = [r for r in records if r[method]["status"] == "ok"]
        results[method + "_n_valid"] = len(valid)
        results[method + "_n_invalid"] = len(records) - len(valid)
        results[method + "_Random_Baseline"] = average([r["random_baseline"] for r in valid], 100.0)
        for name, key, scale in (("Pointing_Game", "hit", 100.0), ("Area_Matched_Recall", "recall", 1.0), ("Pixel_AUROC", "auc", 1.0)):
            values = [r[method][key] for r in valid]
            results[method + "_" + name] = average(values, scale)
            results[method + "_" + name + "_n"] = sum(v is not None for v in values)

    paired = [r for r in records if r["GradCAM"]["status"] == r["MIL"]["status"] == "ok"]
    results["paired_n"] = len(paired)
    results["paired_comparison"] = {
        m: {k: average([r[m][k] for r in paired]) for k in ("hit", "recall", "auc")}
        for m in ("GradCAM", "MIL")
    }
    results["Mean_IoU_Informational"] = average([r["GradCAM"]["iou"] for r in records])
    if return_samples:
        results["samples"] = samples
    return results

