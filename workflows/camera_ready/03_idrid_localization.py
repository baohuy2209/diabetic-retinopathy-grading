# %% [markdown]
# # Evaluate IDRiD lesion localization

# %%
NOTEBOOK_CODE_SHA256 = '10baa7ef1b79bac5cfe5cda5d6ea510b8348e9a25828f5332b56095f1fd0c55f'

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
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'cv2': 'opencv-python-headless', 'albumentations': 'albumentations'}
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
IDRID = Path(os.environ.get('DR_IDRID', INPUTS / 'idrid_segmentation'))
CHECKPOINT_SOURCE = Path(os.environ.get('DR_CHECKPOINTS', INPUTS / 'checkpoints.zip'))
IMAGE_DIRS = [IDRID / part / 'Fundus Images' for part in ('DR_Testing_Set', 'DR_Training_Set')]
MASK_DIRS = [IDRID / part / 'Combined Masks' for part in ('DR_Testing_Set', 'DR_Training_Set')]

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
RAW_DATA_DIR = INPUTS / 'raw'

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
# ## data/fov.py

# %%
"""Morphological Field-of-View (FOV) mask computation for fundus photographs."""

from typing import Any, Optional, Tuple
import cv2
import numpy as np


def compute_fov_mask(
    img_np: np.ndarray,
    erode_px: int = 15,
    threshold: Optional[int] = None,
    kernel_size: int = 25,
    **kwargs: Any,
) -> np.ndarray:
    """Computes the true retinal Field-of-View (FOV) binary mask for a single fundus image.

    4-Step Pipeline:
        1. Green Channel + Otsu's or fixed Thresholding: Dynamically calculates optimal
           bimodal threshold or uses fixed threshold.
        2. Morphological Close & Open (ellipse): Seals internal vessel holes and removes noise.
        3. Largest Connected Component: Strictly retains the main retinal disk (removes artifacts/watermarks).
        4. Inward Morphological Erosion (erode_px): Strips away peripheral edge vignetting/halos.

    Args:
        img_np: Input fundus image array (H, W, 3) in RGB/BGR or (H, W) grayscale.
        erode_px: Inward erosion radius in pixels to remove peripheral halo artifacts.
        threshold: Optional fixed threshold. If None, Otsu's adaptive threshold is used.
        kernel_size: Kernel dimension for morphological closing and opening.

    Returns:
        Boolean 2D array of shape (H, W) where True indicates valid retinal aperture.
    """
    if img_np is None or img_np.size == 0:
        return np.zeros((1, 1), dtype=bool)

    if img_np.ndim == 3:
        # Green channel offers highest absorption contrast for retinal boundaries
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np

    # 1. Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is not None and threshold > 0:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morphological Close & Open
    k_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
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

    return fov_mask > 0


compute_morphological_fov_mask = compute_fov_mask


def apply_fov_mask(
    img: np.ndarray,
    mask: np.ndarray,
    fill_value: int = 0,
) -> np.ndarray:
    """Sets all pixels outside the FOV mask to fill_value."""
    out = img.copy()
    out[~mask] = fill_value
    return out


def compute_soft_fov_mask(
    img_np: np.ndarray,
    blur_sigma: float = 5.0,
    **kwargs: Any,
) -> np.ndarray:
    """Computes a soft-edged continuous [0, 1] FOV mask."""
    hard_mask = compute_fov_mask(img_np, **kwargs).astype(np.float32)
    ksize = int(blur_sigma * 4) + 1
    if ksize % 2 == 0:
        ksize += 1
    soft_mask = cv2.GaussianBlur(hard_mask, (ksize, ksize), blur_sigma)
    return np.clip(soft_mask, 0.0, 1.0)


def extract_circular_fov_region(
    img_np: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops the image and mask to the bounding box of the circular FOV."""
    if mask is None:
        mask = compute_fov_mask(img_np)
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return img_np, mask
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    return img_np[y_min : y_max + 1, x_min : x_max + 1], mask[y_min : y_max + 1, x_min : x_max + 1]


def filter_features_by_fov(
    features: np.ndarray,
    fov_mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Filters spatial features according to FOV coverage threshold."""
    return features

# %% [markdown]
# ## data/patches.py

# %%
"""Sliding-window patch extraction and foreground anchor caching for local MIL."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch




def compute_valid_anchors(
    img_shape: Tuple[int, int],
    fov_mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
) -> np.ndarray:
    """Calculates 2D top-left coordinate anchors [y0, x0] meeting the FOV threshold.

    Args:
        img_shape: (H, W) or (H, W, C) of source image.
        fov_mask: Boolean/binary mask of shape (H, W).
        patch_size: Square patch side length (p).
        stride: Step size between consecutive sliding window steps.
        fov_thresh: Minimum fraction of pixels within the FOV mask (default: 0.5).

    Returns:
        (N, 2) numpy array of integer top-left anchor coordinates [y0, x0].
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

    def get_coords(curr_stride: int) -> List[List[int]]:
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
                fov_ratio = float(mask_patch.mean()) if mask_patch.size > 0 else 0.0
                if fov_ratio >= fov_thresh:
                    valid_list.append([y0, x0])
        return valid_list

    anchors = get_coords(stride)
    if len(anchors) < 32 and stride > 48 and min(content_h, content_w) > p:
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors

    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]

    return np.array(anchors, dtype=np.int32)


def extract_mil_patches_sliding(
    img_np: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = 48,
    max_patches_eval: int = 256,
    valid_anchors: Optional[np.ndarray] = None,
    return_anchors: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
    """Extracts sliding-window patches from native-resolution fundus images.

    Args:
        img_np: Input image array of shape (H, W, 3).
        fov_mask: Optional binary mask of shape (H, W).
        patch_size: Square patch size (default 128).
        stride: Sliding stride (default 96).
        fov_thresh: Minimum foreground ratio.
        mode: 'train' (random subsample) or 'eval' (deterministic uniform sampling).
        num_sample: Fixed int or (min, max) tuple of patches to sample in training.
        max_patches_eval: Upper limit on evaluated patches in evaluation mode.
        valid_anchors: Pre-cached (N, 2) array of coordinates.
        return_anchors: If True, also returns the coordinates of selected patches.

    Returns:
        tensor_patches: Tensor of shape (K, 3, patch_size, patch_size) in uint8 or float32.
        (optional) selected_coords: (K, 2) numpy array of anchor coordinates.
    """
    h, w, _ = img_np.shape
    p = patch_size

    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            m_target = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            m_target = int(num_sample)
    else:
        m_target = max_patches_eval

    # Fast path if pre-computed valid_anchors are provided
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
        tensor_patches = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()
        if return_anchors:
            return tensor_patches, np.asarray(selected_coords, dtype=np.int32)
        return tensor_patches

    # On-the-fly calculation if no anchors are provided
    if fov_mask is None:
        fov_mask = compute_fov_mask(img_np)

    anchors = compute_valid_anchors(
        img_shape=(h, w),
        fov_mask=fov_mask,
        patch_size=patch_size,
        stride=stride,
        fov_thresh=fov_thresh,
    )

    return extract_mil_patches_sliding(
        img_np=img_np,
        fov_mask=fov_mask,
        patch_size=patch_size,
        stride=stride,
        fov_thresh=fov_thresh,
        mode=mode,
        num_sample=num_sample,
        max_patches_eval=max_patches_eval,
        valid_anchors=anchors,
        return_anchors=return_anchors,
    )


def compute_foreground_ratio(patch_mask: np.ndarray) -> float:
    """Computes the foreground fraction of a patch mask."""
    if patch_mask.size == 0:
        return 0.0
    return float(np.mean(patch_mask > 0))


def filter_foreground_patches(
    patches: np.ndarray,
    fov_mask: np.ndarray,
    anchors: np.ndarray,
    patch_size: int = 128,
    min_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters patches according to their foreground ratio within the FOV mask."""
    valid_indices = []
    for i, (y0, x0) in enumerate(anchors):
        patch_m = fov_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
        if compute_foreground_ratio(patch_m) >= min_ratio:
            valid_indices.append(i)
    if not valid_indices:
        return patches, anchors
    idx = np.array(valid_indices)
    return patches[idx], anchors[idx]


class ForegroundPatchCache:
    """In-memory or on-disk cache for computed patch anchor coordinates."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}

    def get(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieves cached anchors for an image ID."""
        if image_id in self._memory_cache:
            return self._memory_cache[image_id]
        if self.cache_dir:
            p = self.cache_dir / f"{image_id}_anchors.npy"
            if p.exists():
                arr = np.load(p)
                self._memory_cache[image_id] = arr
                return arr
        return None

    def set(self, image_id: str, anchors: np.ndarray) -> None:
        """Stores anchors for an image ID."""
        self._memory_cache[image_id] = anchors
        if self.cache_dir:
            p = self.cache_dir / f"{image_id}_anchors.npy"
            np.save(p, anchors)

# %% [markdown]
# ## data/preprocessing.py

# %%
"""Fundus image preprocessing: circular ROI cropping and green channel enhancement."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def crop_fundus_roi(
    img: np.ndarray,
    tolerance: int = 7,
) -> np.ndarray:
    """Automatically removes non-diagnostic black borders surrounding the retinal aperture.

    Args:
        img: Input image array in BGR or RGB format (H, W, 3) or grayscale (H, W).
        tolerance: Pixel intensity threshold below which pixels are treated as black background.

    Returns:
        Cropped image array bounded tightly around the retinal area.
    """
    if img is None or img.size == 0:
        return img

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mask = gray > tolerance
    if not np.any(mask):
        return img

    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    ymin, ymax = np.where(row_mask)[0][[0, -1]]
    xmin, xmax = np.where(col_mask)[0][[0, -1]]

    cropped = img[ymin : ymax + 1, xmin : xmax + 1]
    return cropped if cropped.size > 0 else img


def extract_green_channel(
    img: np.ndarray,
    as_three_channel: bool = False,
) -> np.ndarray:
    """Extracts the green channel (index 1 in RGB) offering highest vascular/lesion contrast.

    Args:
        img: Image array of shape (H, W, 3).
        as_three_channel: If True, replicates green channel across all 3 channels.

    Returns:
        Green channel array (H, W) or (H, W, 3).
    """
    if img is None or img.ndim != 3:
        return img

    green = img[:, :, 1]
    if as_three_channel:
        return np.stack([green, green, green], axis=-1)
    return green


def apply_green_channel_ablation(
    img_rgb: np.ndarray,
    ablation_mode: str = "ben_graham_green",
) -> np.ndarray:
    """Applies fundus image preprocessing ablations centered on the green channel.

    Modes:
        - 'ben_graham_green': Gaussian background subtraction on bilateral-denoised green channel.
        - 'clahe_green': Contrast Limited Adaptive Histogram Equalization.
        - 'raw_green' (or fallback): Replicates raw green channel across 3 channels.

    Args:
        img_rgb: Input RGB image array of shape (H, W, 3).
        ablation_mode: 'ben_graham_green', 'clahe_green', or 'raw_green'.

    Returns:
        Processed 3-channel RGB image array of shape (H, W, 3).
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


def apply_ben_graham_enhancement(img_rgb: np.ndarray, sigma: int = 30) -> np.ndarray:
    """Applies classic Ben Graham Gaussian subtraction on RGB or green channel."""
    return apply_green_channel_ablation(img_rgb, ablation_mode="ben_graham_green")


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    return apply_green_channel_ablation(img, ablation_mode="clahe_green")

# %% [markdown]
# ## explainability/idrid_dataset.py

# %%
"""IDRiD pixel-level lesion segmentation dataset loader and composite mask builder."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset




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

# %% [markdown]
# ## explainability/gradcam.py

# %%
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

# %% [markdown]
# ## explainability/mil_saliency.py

# %%
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

# %% [markdown]
# ## explainability/metrics.py

# %%
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

# %% [markdown]
# ## experiments/xai.py

# %%
"""Evaluate real IDRiD image/mask pairs using the held-out V3 checkpoint.

No generated patches, random weights, or synthetic masks in real evaluation.
The data owner must independently verify supplied binary-mask annotation categories.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch







class QuantitativeXAIRunner:
    """Frozen-checkpoint localization evaluation; never trains or selects models."""

    def __init__(self, checkpoint_path=None, variant="v3", dataset_dir=None,
                 results_dir=Path("result/xai"), device="cpu", max_samples=0,
                 dry_run=False, image_dirs: Optional[Sequence[Path]] = None,
                 mask_dirs: Optional[Sequence[Path]] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.variant, self.results_dir = variant, Path(results_dir)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.max_samples, self.dry_run = max_samples, dry_run
        self.image_dirs = list(image_dirs or ([Path(dataset_dir) / "images"] if dataset_dir else []))
        self.mask_dirs = list(mask_dirs or ([Path(dataset_dir) / "masks"] if dataset_dir else []))

    def load_model(self):
        if self.variant != "v3":
            raise ValueError("The camera-ready XAI protocol evaluates the IDRiD-held-out V3 model")
        if not self.checkpoint_path or not self.checkpoint_path.is_file():
            raise FileNotFoundError("A real IDRiD-held-out V3 checkpoint is required")
        required = "lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth"
        if self.checkpoint_path.name != required:
            raise ValueError(f"Expected the published held-out checkpoint: {required}")
        return build_checkpoint_model("v3", self.checkpoint_path, self.device)

    def run(self):
        if self.dry_run:
            return dict(status="SMOKE_TEST_ONLY", note="No empirical localization metrics computed")
        if self.max_samples < 0 or not self.image_dirs or not self.mask_dirs:
            raise ValueError("Supply image_dirs and mask_dirs; max_samples must be nonnegative")
        for p in [*self.image_dirs, *self.mask_dirs]:
            if not Path(p).is_dir():
                raise FileNotFoundError(p)
        model = self.load_model()
        dataset = IDRiDXAIEvalDataset(image_dirs=self.image_dirs, mask_dirs=self.mask_dirs)
        if not len(dataset):
            raise ValueError("No real image/mask pairs found")
        if self.max_samples:
            dataset.samples = dataset.samples[:self.max_samples]
        summary = compute_quantitative_xai(model, dataset, device=self.device, return_samples=False)
        run = new_run(self.results_dir)
        pairs = [dict(image_id=i, image_path=str(Path(p).resolve()), image_sha256=sha256(Path(p)),
                      mask_path=str(Path(m).resolve()), mask_sha256=sha256(Path(m))) for i, p, m in dataset.samples]
        summary.update(checkpoint=str(self.checkpoint_path.resolve()), checkpoint_sha256=sha256(self.checkpoint_path),
                       status="DIAGNOSTIC_SUBSET" if self.max_samples else "EVALUATED_REAL_PAIRS",
                       cohort="IDRiD", pairs=pairs, mask_categories="Supplied binary masks; annotation categories require independent verification")
        (run / "metrics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("result/xai"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args(argv)
    runner = QuantitativeXAIRunner(checkpoint_path=args.checkpoint_path, image_dirs=args.image_dirs,
        mask_dirs=args.mask_dirs, results_dir=args.results_dir, device=args.device, max_samples=args.max_samples)
    print(json.dumps(runner.run(), indent=2))

# %%
filename = 'lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth'
checkpoint = containing(stage_archive(CHECKPOINT_SOURCE, WORK / 'archives'), filename) / filename
args = ['--checkpoint-path', str(checkpoint), '--image-dirs', *map(str, IMAGE_DIRS),
        '--mask-dirs', *map(str, MASK_DIRS), '--results-dir', str(OUTPUTS / 'camera_ready' / 'xai'),
        '--device', 'cuda' if torch.cuda.is_available() else 'cpu']
status = main(args)
print('Exit status:', status)
