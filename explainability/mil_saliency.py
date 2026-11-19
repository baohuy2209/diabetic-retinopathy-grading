"""Local Multi-Instance Learning (MIL) Attention Saliency projection."""

from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class MILAttentionSaliency:
    """Projects patch-level attention weights into a spatial fundus saliency heatmap."""

    def __init__(self, patch_size: int = 128, sigma_smooth: float = 0.0) -> None:
        self.patch_size = patch_size
        self.sigma_smooth = sigma_smooth

    def generate_saliency_map(
        self,
        attention_weights: Union[np.ndarray, torch.Tensor],
        patch_anchors: Union[np.ndarray, torch.Tensor],
        img_shape: Tuple[int, int],
        normalize: bool = True,
    ) -> np.ndarray:
        """Projects scalar attention weights into 2D full-image heatmap.

        Args:
            attention_weights: (K,) or (1, K) array of non-negative attention weights summing to 1.
            patch_anchors: (K, 2) array of integer coordinates [y0, x0].
            img_shape: (H, W) destination spatial resolution.
            normalize: If True, normalizes final heatmap to [0.0, 1.0].

        Returns:
            2D numpy array of shape (H, W) in float32.
        """
        if isinstance(attention_weights, torch.Tensor):
            weights = attention_weights.detach().cpu().squeeze().numpy()
        else:
            weights = np.asarray(attention_weights, dtype=np.float32).squeeze()

        if isinstance(patch_anchors, torch.Tensor):
            anchors = patch_anchors.detach().cpu().numpy()
        else:
            anchors = np.asarray(patch_anchors, dtype=np.int32)

        h, w = img_shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        p = self.patch_size

        if weights.ndim == 0:
            weights = np.array([float(weights)])

        for idx, (y0, x0) in enumerate(anchors):
            if idx >= len(weights):
                break
            y1 = min(y0 + p, h)
            x1 = min(x0 + p, w)
            if y1 > y0 and x1 > x0:
                w_val = float(weights[idx])
                heatmap[y0:y1, x0:x1] += w_val
                count_map[y0:y1, x0:x1] += 1.0

        # Average over overlapping patch coverage
        valid_mask = count_map > 0
        heatmap[valid_mask] /= count_map[valid_mask]

        if self.sigma_smooth > 0:
            ksize = int(self.sigma_smooth * 4) + 1
            if ksize % 2 == 0:
                ksize += 1
            heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), self.sigma_smooth)

        if normalize:
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)


def generate_attention_mil_saliency_map(
    model: Any,
    global_img: torch.Tensor,
    local_patches: torch.Tensor,
    anchors: Any,
    patch_mask: Optional[torch.Tensor] = None,
    canvas_size: Tuple[int, int] = (512, 512),
    patch_size: int = 128,
    native_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Extracts architecture-native attention weights and projects them onto spatial canvas."""
    if native_shape is None:
        native_shape = canvas_size
    model.eval()
    branch = getattr(getattr(model, "backbone", model), "local_mil_branch", None)
    pool = getattr(branch, "gated_pool", None)
    if pool is None:
        return np.full(canvas_size, np.nan, np.float32)
    captured = []
    layer = getattr(pool, "attention_weights", None)
    if not isinstance(layer, torch.nn.Module):
        return np.full(canvas_size, np.nan, np.float32)
    # Observe the existing layer without changing checkpoint parameters or forward values.
    handle = layer.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    try:
        with torch.no_grad():
            model(global_img, local_patches, patch_mask=patch_mask)
    except Exception:
        return np.full(canvas_size, np.nan, np.float32)
    finally:
        handle.remove()
    weights = captured[-1] if captured else None
    if weights is None:
        return np.full(canvas_size, np.nan, np.float32)
    if patch_mask is not None:
        weights = weights.masked_fill(~patch_mask.unsqueeze(-1), -1e4)
    weights = torch.softmax(weights, dim=1)
    weights = weights[0].detach().float().cpu().numpy().reshape(-1)
    anchors = np.asarray(anchors).reshape(-1, 2)
    if len(weights) != len(anchors):
        return np.full(canvas_size, np.nan, np.float32)
    valid = np.ones(len(weights), dtype=bool)
    if patch_mask is not None:
        valid = patch_mask[0].detach().cpu().numpy().astype(bool).reshape(-1)
        if len(valid) != len(weights):
            return np.full(canvas_size, np.nan, np.float32)
    h, w = map(int, native_shape)
    canvas = np.zeros((h, w), np.float32)
    counts = np.zeros_like(canvas)
    for k, (y, x) in enumerate(anchors):
        if not valid[k]:
            continue
        y, x = int(y), int(x)
        if not (0 <= y < h and 0 <= x < w):
            continue
        canvas[y : min(y + patch_size, h), x : min(x + patch_size, w)] += weights[k]
        counts[y : min(y + patch_size, h), x : min(x + patch_size, w)] += 1
    result = np.divide(canvas, counts, out=np.zeros_like(canvas), where=counts > 0)
    result = cv2.resize(result, (canvas_size[1], canvas_size[0]), interpolation=cv2.INTER_LINEAR)
    ptp = float(np.ptp(result))
    if np.isfinite(result).all() and ptp > 1e-12:
        result = (result - result.min()) / ptp
    return result

