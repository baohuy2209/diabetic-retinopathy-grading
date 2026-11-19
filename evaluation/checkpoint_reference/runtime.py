"""Strict reference checkpoint loading and preprocessing for existing V2/V3 weights."""

from pathlib import Path, PurePosixPath

import numpy as np
import torch

from evaluation.camera_ready import sha256


def build_checkpoint_model(variant: str, checkpoint: Path, device: str = "cpu", patch_chunk_size: int = 48):
    from .losses_and_heads import build_model

    if variant not in ("v2", "v3"):
        raise ValueError("Reference checkpoint inference supports v2 and v3")
    model = build_model(head_type="Softmax_CE" if variant == "v2" else "CLM_QWK",
                        pretrained=False, use_cbam=True, use_dual_pooling=True, fusion_dim=512, num_classes=5)
    # Only load checkpoint files from a trusted source. Legacy dictionaries include metadata.
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    weights = state.get("model_state_dict", state)
    model.load_state_dict(weights, strict=True)
    model.backbone.local_mil_branch.eval_patch_chunk_size = patch_chunk_size
    return model.to(device).eval()


def normalize_inputs(global_img, local_patches, device):
    if global_img.dtype != torch.uint8 or local_patches.dtype != torch.uint8:
        raise ValueError("Reference inputs must be raw uint8 tensors; normalization belongs to this function")
    mean = torch.tensor([.485, .456, .406], device=device)
    std = torch.tensor([.229, .224, .225], device=device)
    g = global_img.to(device).float().div(255)
    l = local_patches.to(device).float().div(255)
    return ((g - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1),
            (l - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1))


def resolve_manifest_images(frame, data_root: Path):
    """Rebase Windows/Colab paths explicitly; fail before inference on missing files."""
    result = frame.copy()
    if "native_path" not in result:
        raise ValueError("Manifest must include native_path")
    paths = []
    for row in result.itertuples(index=False):
        raw = str(row.native_path).replace("\\", "/")
        parts = list(PurePosixPath(raw).parts)
        lower = [p.lower() for p in parts]
        if "native_cropped" in lower:
            parts = parts[lower.index("native_cropped") + 1:]
        elif Path(raw).is_file():
            if Path(raw).stem != str(row.image_id):
                raise ValueError(f"Image filename and manifest ID differ: {raw}")
            paths.append(str(Path(raw).resolve()))
            continue
        if not parts or any(p in ("..", "/") or ":" in p for p in parts):
            raise ValueError(f"Cannot safely rebase path: {raw}")
        path = Path(data_root).joinpath(*parts).resolve()
        if not path.is_relative_to(Path(data_root).resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing image {row.image_id}: {path}")
        if path.stem != str(row.image_id):
            raise ValueError(f"Image filename and manifest ID differ: {path}")
        paths.append(str(path))
    result["native_path"] = paths
    return result


def code_fingerprint(runtime):
    import json
    import hashlib

    root = Path(__file__).parent
    files = {p.name: sha256(p) for p in sorted(root.glob("*.py"))}
    repo = root.parent.parent
    for relative in ("experiments/checkpoint_inference.py", "evaluation/camera_ready.py", "evaluation/paired_bootstrap.py"):
        files[relative] = sha256(repo / relative)
    return hashlib.sha256(json.dumps(dict(modules=files, runtime=runtime), sort_keys=True).encode()).hexdigest()


def validate_cache(frame, manifest, metadata):
    """Validate every cache row; hash presence is distinct from benchmark reproduction."""
    if len(frame) != len(manifest):
        raise ValueError("Cache sample count mismatch")
    for col in ("image_id", "diagnosis", "native_path"):
        if col not in frame or not np.array_equal(frame[col].astype(str), manifest[col].astype(str)):
            raise ValueError(f"Cache ordered {col} mismatch")
    for key, value in metadata.items():
        if key not in frame or not frame[key].eq(value).all():
            raise ValueError(f"Cache row provenance mismatch: {key}")
    from evaluation.paired_bootstrap import validate_pairs
    validate_pairs(frame.diagnosis.to_numpy(), frame.pred_v2.to_numpy(), frame.pred_v3.to_numpy())
    exp = frame.v3_expected_grade.to_numpy()
    if not np.isfinite(exp).all() or (exp < 0).any() or (exp > 4).any() or not np.array_equal(np.clip(np.round(exp), 0, 4), frame.pred_v3):
        raise ValueError("Cache expected-grade decoder mismatch")
