"""Grad-CAM implementation for retinal fundus models with categorical and ordinal heads."""

from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _xai_map_status(heatmap: np.ndarray) -> str:
    heatmap = np.asarray(heatmap)
    if not np.isfinite(heatmap).all():
        return "nonfinite_map"
    if float(np.ptp(heatmap)) <= 1e-12:
        return "constant_map"
    return "ok"


class FundusGradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) with memory-safe hook cleanup.

    Extracts visual attention saliency heatmaps for both categorical and ordinal CLM models.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        if target_layer is None:
            if hasattr(model, "backbone") and hasattr(model.backbone, "global_branch") and hasattr(model.backbone.global_branch, "layer4"):
                target_layer = model.backbone.global_branch.layer4[-1]
            elif hasattr(model, "layer4"):
                target_layer = model.layer4[-1]
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.hook_handles: List[Any] = []
        self.last_status = "unavailable"
        self.remove_hooks()

    def _register_hooks(self) -> None:
        if self.target_layer is None:
            raise ValueError("Cannot find Grad-CAM target layer")

        def forward_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            self.activation = output

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))

    def remove_hooks(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    @torch.enable_grad()
    def generate_cam(
        self,
        global_img: torch.Tensor,
        local_patches: torch.Tensor,
        target_class: Optional[int] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, int]:
        self.remove_hooks()
        self.activation = self.gradient = None
        self.last_status = "unavailable"
        self._register_hooks()
        try:
            self.model.eval()
            global_img = global_img.detach().requires_grad_(True)

            if patch_mask is not None:
                try:
                    outputs = self.model(global_img, local_patches, patch_mask=patch_mask)
                except TypeError:
                    outputs = self.model(global_img, local_patches)
            else:
                outputs = self.model(global_img, local_patches)

            pred_class = None
            if isinstance(outputs, dict) and "preds" in outputs:
                pred_class = int(outputs["preds"][0].item())

            if isinstance(outputs, dict) and "severity_score" in outputs:
                score = outputs["severity_score"].flatten()[0]
                if target_class is None:
                    target_class = pred_class if pred_class is not None else 0
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
                if target_class is None:
                    target_class = int(torch.argmax(logits, dim=-1)[0].item())
                score = logits[0, target_class]
            elif isinstance(outputs, torch.Tensor):
                if outputs.dim() > 1 and outputs.size(-1) == 5:
                    if target_class is None:
                        target_class = int(torch.argmax(outputs, dim=-1)[0].item())
                    score = outputs[0, target_class]
                else:
                    score = outputs.flatten()[0]
                    if target_class is None:
                        target_class = pred_class if pred_class is not None else 0
            else:
                score = outputs[0]
                if target_class is None:
                    target_class = pred_class if pred_class is not None else 0

            if self.activation is None or not score.requires_grad:
                self.last_status = "no_gradient"
                return np.zeros(global_img.shape[-2:], np.float32), (target_class or 0)

            gradient = torch.autograd.grad(score, self.activation, allow_unused=True)[0]
            if gradient is None:
                self.last_status = "no_gradient"
                return np.zeros(global_img.shape[-2:], np.float32), (target_class or 0)

            weights = gradient.float().mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.activation.float()).sum(dim=1))[0].detach().cpu().numpy()
            self.last_status = _xai_map_status(cam)
            if self.last_status == "ok":
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)

            return cam.astype(np.float32), (target_class or 0)

        finally:
            self.remove_hooks()
            self.activation = self.gradient = None

    def generate(
        self,
        global_img: torch.Tensor,
        local_patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Convenience alias returning (300, 300) interpolated heatmap."""
        cam, _ = self.generate_cam(global_img, local_patches, patch_mask=patch_mask)
        cam_resized = cv2.resize(cam, (300, 300), interpolation=cv2.INTER_LINEAR)
        return cam_resized


# Backward compatibility alias
GradCAM = FundusGradCAM
