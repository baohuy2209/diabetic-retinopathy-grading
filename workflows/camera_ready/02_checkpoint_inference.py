# %% [markdown]
# # Rerun the six V2/V3 checkpoint pairs

# %%
NOTEBOOK_CODE_SHA256 = 'ea34464e4faaa1f3fc1e6e58cf412173c8ddbcae89113ffaa77774654301b30f'

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

# %%
FOLD_SOURCE = Path(os.environ.get('DR_FOLDS', INPUTS / 'lodo_folds'))
REFERENCE = Path(os.environ.get('DR_REFERENCE', INPUTS / 'reference' / 'lodo_summary_with_calibrated_screening.json'))
IMAGE_INPUT = Path(os.environ.get('DR_IMAGES', INPUTS / 'native_cropped.zip'))
CHECKPOINT_SOURCE = Path(os.environ.get('DR_CHECKPOINTS', INPUTS / 'checkpoints.zip'))
REUSE_FROM = None  # Optional path to a previous run's predictions directory.

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

# %% [markdown]
# ## evaluation/paired_bootstrap.py

# %%
"""Paired Grade-3 bootstrap used by the camera-ready analysis.

Intervals describe held-out image sampling, not training-seed variability.
Values are percentages; differences are percentage points. Cohorts are fixed.
"""

from typing import Any, Dict, Sequence

import numpy as np


def grade_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Reject missing, fractional, or out-of-range grades before any cast."""
    a = np.asarray(values)
    if a.ndim != 1 or not len(a):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    try:
        valid = np.isfinite(a).all() and np.equal(a, np.floor(a)).all()
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric grades") from exc
    if not valid or (a < 0).any() or (a > 4).any():
        raise ValueError(f"{name} must contain integer ICDR grades 0..4")
    return a.astype(np.int64)


def validate_pairs(y_true: np.ndarray, pred_v2: np.ndarray, pred_v3: np.ndarray):
    arrays = [grade_vector(x, n) for x, n in
              zip((y_true, pred_v2, pred_v3), ("y_true", "pred_v2", "pred_v3"))]
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Targets and paired predictions must have equal lengths")
    return arrays


def _summary(a, b, boot2, boot3, reps, seed, unit):
    c2, c3, delta = (np.percentile(x, [2.5, 97.5]) for x in
                     (boot2, boot3, np.asarray(boot3) - np.asarray(boot2)))
    # Preserve the archived float32 estimator, including percentile interpolation.
    return dict(
        n_grade3=len(a), sens_v2=round(float(a.mean()) * 100, 2),
        sens_v3=round(float(b.mean()) * 100, 2),
        diff_v3_minus_v2_pp=round((float(b.mean()) - float(a.mean())) * 100, 2),
        v2_ci_low=round(float(c2[0]), 2), v2_ci_high=round(float(c2[1]), 2),
        v3_ci_low=round(float(c3[0]), 2), v3_ci_high=round(float(c3[1]), 2),
        diff_ci_low_pp=round(float(delta[0]), 2), diff_ci_high_pp=round(float(delta[1]), 2),
        resampling_unit=unit, bootstrap_reps=reps, seed=seed,
    )


def paired_bootstrap_grade3_sensitivity(
    y_true: np.ndarray, pred_v2: np.ndarray, pred_v3: np.ndarray,
    reps: int = 2000, seed: int = 42,
) -> Dict[str, Any]:
    """Percentile CIs using identical Grade-3 image indices for both variants."""
    if reps < 1:
        raise ValueError("bootstrap replicates must be positive")
    y, p2, p3 = validate_pairs(y_true, pred_v2, pred_v3)
    mask = y == 3
    if not mask.any():
        raise ValueError("Grade-3 sensitivity is undefined: no Grade-3 cases")
    a, b = (p2[mask] == 3).astype(np.float32), (p3[mask] == 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    boot2, boot3 = [], []
    for _ in range(reps):
        idx = rng.integers(0, len(a), size=len(a))
        boot2.append(a[idx].mean() * 100.0)
        boot3.append(b[idx].mean() * 100.0)
    return _summary(a, b, boot2, boot3, reps, seed, "image_level")


def macro_bootstrap_grade3_sensitivity(
    cohort_data: Sequence[Dict[str, Any]], reps: int = 2000, seed: int = 42,
) -> Dict[str, Any]:
    """Resample within each fixed cohort, then average cohort sensitivities equally."""
    if not cohort_data or reps < 1:
        raise ValueError("At least one cohort and positive bootstrap replicates are required")
    groups = []
    for c in cohort_data:
        y, p2, p3 = validate_pairs(c["y_true"], c["pred_v2"], c["pred_v3"])
        if not np.any(y == 3):
            raise ValueError(f"No Grade-3 cases in {c['cohort']}; macro is undefined")
        groups.append(((p2[y == 3] == 3).astype(np.float32),
                       (p3[y == 3] == 3).astype(np.float32)))
    rng = np.random.default_rng(seed)
    boot2, boot3 = [], []
    for _ in range(reps):
        v2, v3 = [], []
        for a, b in groups:
            idx = rng.integers(0, len(a), size=len(a))
            v2.append(a[idx].mean())
            v3.append(b[idx].mean())
        boot2.append(float(np.mean(v2)) * 100)
        boot3.append(float(np.mean(v3)) * 100)
    a = np.asarray([a.mean() for a, _ in groups], dtype=np.float32)
    b = np.asarray([b.mean() for _, b in groups], dtype=np.float32)
    result = _summary(a, b, boot2, boot3, reps, seed, "stratified_by_cohort_macro")
    result.update(n_grade3=sum(len(a) for a, _ in groups),
                  cohort=f"Macro Unweighted ({len(groups)} Cohorts)", fold=-2)
    return result


def grade3_confusion(y_true: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
    """Count all G3 destinations; categorical referral is not threshold screening."""
    y, pred, _ = validate_pairs(y_true, prediction, prediction)
    counts = np.bincount(pred[y == 3], minlength=5)
    n, correct = int(counts.sum()), int(counts[3])
    errors, adjacent = n - correct, int(counts[2] + counts[4])
    return dict(n_actual_g3=n, **{f"g3_to_{i}": int(x) for i, x in enumerate(counts)},
                g3_to_01=int(counts[0] + counts[1]), g3_to_24=adjacent,
                g3_correct=correct, n_g3_misclassified=errors,
                pct_errors_assigned_to_grade24=adjacent / errors * 100 if errors else None)

# %% [markdown]
# ## evaluation/camera_ready.py

# %%
"""Portable validation and immutable exports for V2/V3 post-hoc analysis."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score






COHORTS = ("aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid")
VARIANTS = {"v2": "V2_PlusGlobalContext_H2", "v3": "V3_PlusOrdinalHead_H3_Champion"}
NAMES = dict(zip(COHORTS, ("APTOS 2019", "IDRiD", "Messidor-2", "DDR", "EyePACS", "DeepDRiD")))
STATISTICAL_SCOPE = ("Bootstrap intervals reflect sampling variability of the held-out cohorts, "
                     "not variability across training runs.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run(output_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(output_dir) / (stamp + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    return run


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    if not {"image_id", "diagnosis"} <= set(frame):
        raise ValueError(f"Missing image_id/diagnosis: {path}")
    if frame.image_id.isna().any() or frame.image_id.str.strip().eq("").any() or not frame.image_id.is_unique:
        raise ValueError(f"Missing or duplicate image IDs: {path}")
    validate_pairs(frame.diagnosis.to_numpy(), frame.diagnosis.to_numpy(), frame.diagnosis.to_numpy())
    return frame


def load_archive_pair(raw_dir: Path, folds_dir: Path, cohort: str, allow_manifest_order: bool = False):
    """Load original NPZs without pickle. Never synthesize predictions from a summary."""
    fold = COHORTS.index(cohort)
    manifest_path = Path(folds_dir) / f"fold_{fold}_{cohort}_test.csv"
    manifest = read_manifest(manifest_path)
    arrays, inputs = {}, [dict(path=str(manifest_path.resolve()), sha256=sha256(manifest_path))]
    for short, variant in VARIANTS.items():
        path = Path(raw_dir) / f"{variant}__{cohort}.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays[short] = {key: archive[key] for key in archive.files}
        z = arrays[short]
        if not {"ood_targets", "ood_preds", "ood_probs"} <= set(z):
            raise ValueError(f"Incomplete archived predictions: {path}")
        validate_pairs(z["ood_targets"], z["ood_preds"], z["ood_preds"])
        if not np.array_equal(z["ood_targets"], manifest.diagnosis):
            raise ValueError(f"Ordered targets do not match the manifest: {path}")
        ids = z.get("ood_image_ids", z.get("image_ids"))
        if ids is None and not allow_manifest_order:
            raise ValueError("Legacy NPZ lacks image IDs. Confirm documented unshuffled order with --allow-manifest-order")
        if ids is not None and not np.array_equal(ids.astype(str), manifest.image_id.to_numpy()):
            raise ValueError(f"Archived image ID mismatch: {path}")
        probs = z["ood_probs"]
        if probs.shape != (len(manifest), 5) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError(f"Invalid class probabilities: {path}")
        decoder = probs.argmax(1) if short == "v2" else np.clip(np.round((probs * np.arange(5)).sum(1)), 0, 4)
        if not np.array_equal(decoder, z["ood_preds"]):
            raise ValueError(f"Stored prediction/decoder mismatch: {path}")
        inputs.append(dict(path=str(path.resolve()), sha256=sha256(path),
                           identity="archived_image_ids" if ids is not None else "documented_manifest_order"))
    frame = manifest.copy()
    frame["pred_v2"] = arrays["v2"]["ood_preds"].astype(int)
    frame["pred_v3"] = arrays["v3"]["ood_preds"].astype(int)
    return frame, inputs


def load_csv_pair(path: Path, manifest_path: Path):
    frame = pd.read_csv(path, dtype={"image_id": str})
    manifest = read_manifest(manifest_path)
    for col in ("image_id", "diagnosis"):
        if col not in frame or not np.array_equal(frame[col], manifest[col]):
            raise ValueError(f"Ordered {col} mismatch in {path}")
    validate_pairs(frame.diagnosis.to_numpy(), frame.pred_v2.to_numpy(), frame.pred_v3.to_numpy())
    return frame, [dict(path=str(p.resolve()), sha256=sha256(p)) for p in (path, manifest_path)]


def analyze_frames(
    frames: Dict[str, pd.DataFrame], reference_path: Optional[Path], output_dir: Path,
    inputs: List[Dict[str, Any]], reps: int = 2000, seed: int = 42,
    source_mode: str = "archived_predictions", exact_reference: bool = True,
    diagnostic_only: bool = False,
) -> Dict[str, Any]:
    """A reference match never certifies training or historical per-image provenance."""
    run = new_run(output_dir)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8")) if reference_path else None
    if reference_path:
        inputs = inputs + [dict(path=str(Path(reference_path).resolve()), sha256=sha256(reference_path))]
    metrics, checks, boots, confusion, data = [], [], [], [], []
    try:
        for cohort in COHORTS:
            if cohort not in frames:
                continue
            frame = frames[cohort]
            y, p2, p3 = validate_pairs(frame.diagnosis.to_numpy(), frame.pred_v2.to_numpy(), frame.pred_v3.to_numpy())
            fold = COHORTS.index(cohort)
            row = dict(cohort=NAMES[cohort], fold=fold, n_samples=len(y), n_grade3=int((y == 3).sum()))
            for variant, pred in (("v2", p2), ("v3", p3)):
                values = dict(qwk=float(cohen_kappa_score(y, pred, labels=range(5), weights="quadratic")),
                              within1=float(np.mean(abs(y - pred) <= 1) * 100),
                              g3_sens=float(np.mean(pred[y == 3] == 3) * 100))
                row.update({f"{variant}_{key}": value for key, value in values.items()})
                conf = grade3_confusion(y, pred)
                confusion.append(dict(cohort=NAMES[cohort], model=variant.upper(), **conf))
                if reference is not None:
                    ref = reference[VARIANTS[variant]][cohort]
                    expected = dict(qwk=float(ref["ood_test_qwk"]),
                                    within1=float(ref["ood_within_1_grade_acc"]) * 100,
                                    g3_sens=float(ref["ood_per_grade_recall"]["grade_3"]) * 100)
                    if [ref["ood_per_grade_support"][f"grade_{i}"] for i in range(5)] != np.bincount(y, minlength=5).tolist():
                        raise ValueError(f"Reference grade support mismatch: {cohort}/{variant}")
                    for key, value in values.items():
                        tolerance = 1e-12 if exact_reference else (.005 if key == "qwk" else 1.0 if key == "within1" else 1.5 if variant == "v2" else 2.5)
                        delta = value - expected[key]
                        checks.append(dict(cohort=cohort, model=variant, metric=key, observed=value,
                                           reference=expected[key], difference=delta, tolerance=tolerance,
                                           status="PASS" if abs(delta) <= tolerance else "DISCREPANCY"))
            metrics.append(row)
            boot = paired_bootstrap_grade3_sensitivity(y, p2, p3, reps=reps, seed=seed)
            boots.append(dict(cohort=NAMES[cohort], fold=fold, **boot))
            data.append(dict(cohort=cohort, y_true=y, pred_v2=p2, pred_v3=p3))
        if not data:
            raise ValueError("No cohorts supplied")
        boots.append(macro_bootstrap_grade3_sensitivity(data, reps=reps, seed=seed))
        boots.append(dict(cohort="Pooled images", fold=-1, **paired_bootstrap_grade3_sensitivity(
            np.concatenate([d["y_true"] for d in data]), np.concatenate([d["pred_v2"] for d in data]),
            np.concatenate([d["pred_v3"] for d in data]), reps=reps, seed=seed)))
        complete = set(frames) == set(COHORTS)
        matched = complete and reference is not None and len(checks) == 36 and all(c["status"] == "PASS" for c in checks)
        status = "BENCHMARK_REPRODUCED" if matched and not diagnostic_only else "DIAGNOSTIC"
        suffix = "_" + status
        pd.DataFrame(metrics).to_csv(run / ("per_fold_metrics" + suffix + ".csv"), index=False)
        pd.DataFrame(checks).to_csv(run / "reference_checks.csv", index=False)
        pd.DataFrame(boots).to_csv(run / ("grade3_bootstrap" + suffix + ".csv"), index=False)
        pd.DataFrame(confusion).to_csv(run / ("grade3_confusion_by_cohort" + suffix + ".csv"), index=False)
        pooled = []
        for variant in ("v2", "v3"):
            pooled.append(dict(model=variant.upper(), **grade3_confusion(
                np.concatenate([d["y_true"] for d in data]), np.concatenate([d["pred_" + variant] for d in data]))))
        pd.DataFrame(pooled).to_csv(run / ("grade3_confusion_pooled" + suffix + ".csv"), index=False)
        report = dict(status=status, full_six_cohort_analysis=complete, benchmark_reference_matched=matched,
                      source_mode=source_mode, total_test_images=sum(len(d["y_true"]) for d in data),
                      total_grade3_cases=sum(int((d["y_true"] == 3).sum()) for d in data),
                      bootstrap_reps=reps, seed=seed, statistical_scope=STATISTICAL_SCOPE,
                      resampling_unit="image_level", inputs=inputs, run_directory=str(run.resolve()),
                      provenance_note="Benchmark reproduction is distinct from checkpoint identity verification. Legacy manifest-order pairing does not independently prove per-image identity.")
        (run / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return report
    except Exception as exc:
        (run / "FAILED.json").write_text(json.dumps(dict(status="FAILED", error=str(exc), inputs=inputs), indent=2), encoding="utf-8")
        raise

# %% [markdown]
# ## evaluation/checkpoint_reference/models.py

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

# %% [markdown]
# ## evaluation/checkpoint_reference/losses_and_heads.py

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

# %% [markdown]
# ## evaluation/checkpoint_reference/data_loader.py

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

# %% [markdown]
# ## evaluation/checkpoint_reference/runtime.py

# %%
"""Strict reference checkpoint loading and preprocessing for existing V2/V3 weights."""

from pathlib import Path, PurePosixPath

import numpy as np
import torch




def build_checkpoint_model(variant: str, checkpoint: Path, device: str = "cpu", patch_chunk_size: int = 48):


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

    return hashlib.sha256(json.dumps(dict(workflow=NOTEBOOK_CODE_SHA256, runtime=runtime), sort_keys=True).encode()).hexdigest()


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

    validate_pairs(frame.diagnosis.to_numpy(), frame.pred_v2.to_numpy(), frame.pred_v3.to_numpy())
    exp = frame.v3_expected_grade.to_numpy()
    if not np.isfinite(exp).all() or (exp < 0).any() or (exp > 4).any() or not np.array_equal(np.clip(np.round(exp), 0, 4), frame.pred_v3):
        raise ValueError("Cache expected-grade decoder mismatch")

# %% [markdown]
# ## experiments/checkpoint_inference.py

# %%
"""True V2/V3 checkpoint inference with strict cache validation and reference gating.

This command never trains. Its output remains DIAGNOSTIC if any reference check fails.
"""

import argparse
from functools import partial
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader








def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("result/checkpoint_inference"))
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-chunk-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Diagnostic subset only; 0 = full cohort")
    parser.add_argument("--reuse-from", type=Path, help="Explicit prior run/predictions directory; cache hashes must match")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.patch_chunk_size < 0 or args.max_samples < 0 or len(set(args.cohorts)) != len(args.cohorts):
        parser.error("Invalid batch/workers/chunk/subset/cohort arguments")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    runtime = dict(seed=args.seed, precision="float32", batch_size=args.batch_size,
                   patch_chunk_size=args.patch_chunk_size, torch=torch.__version__,
                   device=args.device, python=platform.python_version(), reference_sha256=sha256(args.reference))
    fingerprint = code_fingerprint(runtime)
    run = new_run(args.output_dir)
    predictions = run / "predictions"
    predictions.mkdir()
    frames, inputs, trace, events = {}, [], [], []
    try:
        for cohort in args.cohorts:
            fold = COHORTS.index(cohort)
            stem = f"fold_{fold}_{cohort}"
            source = args.folds_dir / (stem + "_test.csv")
            manifest = read_manifest(source)
            if args.max_samples:
                manifest = manifest.iloc[:args.max_samples].copy()
            manifest = resolve_manifest_images(manifest, args.data_root)
            checkpoints = {v: args.checkpoint_dir / f"lodo_{name}_fold_{fold}_{cohort}_best.pth" for v, name in VARIANTS.items()}
            hashes = {v: sha256(path) for v, path in checkpoints.items()}
            meta = dict(checkpoint_v2_sha=hashes["v2"], checkpoint_v3_sha=hashes["v3"], manifest_sha=sha256(source),
                        code_fingerprint=fingerprint, provenance_marker="STRICT_REAL_CHECKPOINT_INFERENCE")
            inputs += [dict(path=str(p.resolve()), sha256=sha256(p)) for p in [source, *checkpoints.values()]]
            cache = args.reuse_from / (stem + "_predictions.csv") if args.reuse_from else None
            if cache and cache.is_file():
                frame = pd.read_csv(cache, dtype={"image_id": str})
                validate_cache(frame, manifest, meta)
                events.append(dict(cohort=cohort, mode="VALIDATED_CACHE", source=str(cache.resolve()), sha256=sha256(cache)))
            else:
                model2 = build_checkpoint_model("v2", checkpoints["v2"], args.device, args.patch_chunk_size)
                model3 = build_checkpoint_model("v3", checkpoints["v3"], args.device, args.patch_chunk_size)
                dataset = FundusDualBranchDataset(csv_path=manifest, img_root=str(args.data_root), img_size=512, split="test")
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    collate_fn=partial(trainer_compatible_collate_fn, mode="eval"))
                p2, p3, expected = [], [], []
                offset = 0
                for batch in loader:
                    ids = [str(i) for i in batch["image_id"]]
                    n = len(ids)
                    if ids != manifest.image_id.iloc[offset:offset+n].tolist() or not np.array_equal(batch["label"].numpy(), manifest.diagnosis.iloc[offset:offset+n]):
                        raise ValueError(f"Actual loader ID/target misalignment: {cohort}/{offset}")
                    g, l = normalize_inputs(batch["global_img"], batch["local_patches"], args.device)
                    mask = batch["patch_mask"].to(args.device)
                    with torch.inference_mode():
                        a, b = model2(g, l, patch_mask=mask), model3(g, l, patch_mask=mask)
                    aa, bb = a["probs"].cpu().numpy(), b["probs"].cpu().numpy()
                    exp = b["expected_grade"].cpu().numpy()
                    p2.extend(aa.argmax(1).tolist())
                    p3.extend(np.clip(np.round(exp), 0, 4).astype(int).tolist())
                    expected.extend(exp.tolist())
                    for i in range(min(n, max(0, 3 - sum(t["cohort"] == cohort for t in trace)))):
                        trace.append(dict(cohort=cohort, image_id=ids[i], y_true=int(batch["label"][i]),
                                          v2_probs=aa[i].tolist(), v3_probs=bb[i].tolist()))
                    offset += n
                    print(f"{cohort}: {offset}/{len(manifest)}", flush=True)
                if offset != len(manifest):
                    raise ValueError("Incomplete inference")
                frame = manifest.assign(pred_v2=p2, pred_v3=p3, v3_expected_grade=expected, **meta)
                events.append(dict(cohort=cohort, mode="REAL_FORWARD_INFERENCE"))
                del model2, model3
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            frame.to_csv(predictions / (stem + "_predictions.csv"), index=False)
            frames[cohort] = frame
        (run / "inference_provenance.json").write_text(json.dumps(dict(runtime=runtime, fingerprint=fingerprint,
            events=events, inputs=inputs, sample_trace=trace), indent=2), encoding="utf-8")
        # Partial diagnostic subsets must not be compared with full-cohort supports.
        report = analyze_frames(frames, None if args.max_samples else args.reference, run / "analysis", inputs,
                                reps=args.bootstrap_reps, seed=args.seed, source_mode="checkpoint_inference",
                                exact_reference=False, diagnostic_only=bool(args.max_samples))
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "BENCHMARK_REPRODUCED" else 2
    except Exception as exc:
        (run / "FAILED.json").write_text(json.dumps(dict(status="FAILED", error=str(exc), completed_cohorts=list(frames)), indent=2), encoding="utf-8")
        raise

# %%
folds = containing(stage_archive(FOLD_SOURCE, WORK / 'archives'), 'fold_0_aptos_test.csv')
bundle = stage_archive(IMAGE_INPUT, WORK / 'archives')
roots = [bundle] if bundle.name == 'native_cropped' else [p for p in bundle.rglob('native_cropped') if p.is_dir()]
if len(roots) > 1:
    raise ValueError('Multiple native_cropped directories. Select one image tree.')
images = roots[0] if roots else bundle
checkpoints = containing(stage_archive(CHECKPOINT_SOURCE, WORK / 'archives'), 'lodo_V2_PlusGlobalContext_H2_fold_0_aptos_best.pth')
args = ['--data-root', str(images), '--checkpoint-dir', str(checkpoints), '--folds-dir', str(folds),
        '--reference', str(REFERENCE), '--output-dir', str(OUTPUTS / 'camera_ready' / 'inference'),
        '--device', 'cuda' if torch.cuda.is_available() else 'cpu']
if REUSE_FROM is not None:
    args.extend(['--reuse-from', str(REUSE_FROM)])
status = main(args)
print('Exit status:', status)
