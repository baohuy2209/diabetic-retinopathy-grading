"""IDRiD pixel-level lesion segmentation dataset loader and composite mask builder."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import RAW_DATA_DIR


class IDRiDLesionDataset(Dataset):
    """Dataset for IDRiD pixel-level diabetic retinopathy lesion annotations.

    Ground-truth lesion types:
    - Microaneurysms (MA)
    - Hemorrhages (HE)
    - Hard Exudates (EX)
    - Soft Exudates (SE)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = RAW_DATA_DIR / "idrid_segmentation",
        target_size: Tuple[int, int] = (512, 512),
        lesion_types: Tuple[str, ...] = ("MA", "HE", "EX", "SE"),
        use_synthetic_if_missing: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.lesion_types = lesion_types
        self.use_synthetic_if_missing = use_synthetic_if_missing

        self.samples = self._discover_samples()

    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        if self.data_dir.exists():
            img_candidates = list(self.data_dir.glob("**/*Original_Images*/*.jpg")) + list(
                self.data_dir.glob("**/*.jpg")
            )
            for img_p in img_candidates:
                if "mask" in img_p.name.lower():
                    continue
                base_id = img_p.stem
                # Find matching lesion masks
                mask_paths = {}
                for l_type in self.lesion_types:
                    matches = list(self.data_dir.glob(f"**/*{l_type}*/*{base_id}*")) + list(
                        self.data_dir.glob(f"**/*{base_id}*{l_type}*")
                    )
                    if matches:
                        mask_paths[l_type] = matches[0]

                if mask_paths:
                    samples.append({
                        "image_id": base_id,
                        "image_path": img_p,
                        "mask_paths": mask_paths,
                    })

        if not samples and self.use_synthetic_if_missing:
            # Generate 10 synthetic samples for verification and dry runs
            for idx in range(10):
                samples.append({
                    "image_id": f"idrid_syn_{idx:03d}",
                    "image_path": None,
                    "mask_paths": {},
                    "synthetic": True,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.samples[idx]

        if info.get("synthetic", False) or info.get("image_path") is None:
            # Synthetic sample generator
            h, w = self.target_size
            img = np.zeros((h, w, 3), dtype=np.uint8)
            y, x = np.ogrid[:h, :w]
            disk = ((x - w // 2) ** 2 + (y - h // 2) ** 2) <= (min(h, w) // 2 - 20) ** 2
            img[disk] = [70, 150, 200]

            # Synthetic lesion mask
            mask = np.zeros((h, w), dtype=np.uint8)
            # Add a couple of synthetic lesion circles
            l1 = ((x - w // 2 - 40) ** 2 + (y - h // 2 - 30) ** 2) <= 15 ** 2
            l2 = ((x - w // 2 + 50) ** 2 + (y - h // 2 + 40) ** 2) <= 20 ** 2
            mask[l1 | l2] = 255

            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return {
                "image_id": info["image_id"],
                "image": tensor_img,
                "lesion_mask": mask > 0,
            }

        # Load real image
        bgr = cv2.imread(str(info["image_path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]

        # Build union lesion mask
        composite_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for l_type, m_path in info["mask_paths"].items():
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                composite_mask = np.maximum(composite_mask, (m > 0).astype(np.uint8) * 255)

        # Resize to target size
        resized_img = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))
        resized_mask = cv2.resize(composite_mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        tensor_img = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        return {
            "image_id": info["image_id"],
            "image": tensor_img,
            "lesion_mask": resized_mask > 0,
        }


class IDRiDXAIEvalDataset(Dataset):
    """Dedicated dataset for quantitative XAI fidelity evaluation with Dual-Crop Alignment."""

    def __init__(
        self,
        image_dirs: List[Union[str, Path]],
        mask_dirs: List[Union[str, Path]],
        img_size: int = 512,
        ablation_mode: str = "ben_graham_green",
    ) -> None:
        import os
        import albumentations as A
        from data.fov import compute_fov_mask
        from data.patches import extract_mil_patches_sliding
        from data.preprocessing import apply_green_channel_ablation

        self.img_size = img_size
        self.ablation_mode = ablation_mode
        self.compute_fov_mask = compute_fov_mask
        self.extract_mil_patches = extract_mil_patches_sliding
        self.apply_green_channel_ablation = apply_green_channel_ablation
        self.global_transform = A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        ])

        self.image_records = []
        for img_dir in image_dirs:
            p_dir = Path(img_dir)
            if not p_dir.exists():
                continue
            for fname in sorted(os.listdir(p_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                    img_id = os.path.splitext(fname)[0]
                    img_path = str(p_dir / fname)
                    self.image_records.append((img_id, img_path))

        self.samples = []
        for img_id, img_path in self.image_records:
            mask_path = None
            for mask_dir in mask_dirs:
                p_mask = Path(mask_dir)
                if not p_mask.exists():
                    continue
                candidates = [f"{img_id}.tif", f"{img_id}.png", f"{img_id}.jpg", f"{img_id}.jpeg"]
                for c in candidates:
                    p = str(p_mask / c)
                    if os.path.exists(p):
                        mask_path = p
                        break
                if mask_path:
                    break
            if mask_path is not None:
                self.samples.append((img_id, img_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id, img_path, mask_path = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_np.shape

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        if m.shape != (h, w):
            raise ValueError(
                f"Image/mask dimensions differ: {img_path}: {(h, w)} vs {m.shape}; verify alignment before XAI"
            )
        lesion_mask = (m > 0).astype(np.uint8)

        fov_mask_raw = self.compute_fov_mask(img_np, erode_px=0)
        ys, xs = np.where(fov_mask_raw)
        if len(ys) > 0 and len(xs) > 0:
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            img_np = img_np[y0:y1, x0:x1]
            lesion_mask = lesion_mask[y0:y1, x0:x1]

        fov_mask = self.compute_fov_mask(img_np, erode_px=15)
        processed_np = self.apply_green_channel_ablation(img_np, ablation_mode=self.ablation_mode)
        global_augmented = self.global_transform(image=processed_np)
        global_img = global_augmented["image"]
        global_tensor = torch.from_numpy(global_img).permute(2, 0, 1).contiguous()

        local_patches, patch_anchors = self.extract_mil_patches(
            processed_np,
            fov_mask=fov_mask,
            patch_size=128,
            stride=96,
            fov_thresh=0.5,
            mode="eval",
            max_patches_eval=256,
            return_anchors=True,
        )

        mask_gt_resized = cv2.resize(lesion_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return {
            "image_id": img_id,
            "global_img": global_tensor,
            "local_patches": local_patches,
            "native_shape": tuple(processed_np.shape[:2]),
            "anchors": torch.from_numpy(np.asarray(patch_anchors, dtype=np.int32)),
            "mask_gt": torch.tensor(mask_gt_resized, dtype=torch.uint8),
        }

