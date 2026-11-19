# %% [markdown]
# # P0 · Frozen historical LODO replay

# %%
NOTEBOOK_CODE_SHA256 = '68b299dc31bc0dc082ecfcba9b77ec2b8886bb3613f608632b56311d855b4442'

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
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'cv2': 'opencv-python-headless'}
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
import subprocess, sys
IMAGE_SOURCE = Path(os.environ.get('DR_IMAGES', INPUTS / 'native_cropped.zip'))
CHECKPOINT_SOURCE = Path(os.environ.get('DR_CHECKPOINTS', INPUTS / 'checkpoints.zip'))
FOLD_SOURCE = Path(os.environ.get('DR_FOLDS', INPUTS / 'lodo_folds'))
REFERENCE_PATH = Path(os.environ.get('DR_REFERENCE', INPUTS / 'reference' / 'lodo_summary_with_calibrated_screening.json'))
LOCAL_ROOT = WORK / 'mlwa'
RUN_ID = os.environ.get('DR_P0_ID', 'p0_thesis_fp32')
DRIVE_OUTPUT = OUTPUTS / 'mlwa'
FOLDS = list(range(6))
VARIANTS = ['V2', 'V3']
SPLITS = ['source_train', 'source_val', 'target']
MAX_SAMPLES = 0
PATCH_CHUNK_SIZE = int(os.environ.get('DR_PATCH_CHUNK', 32))
SHARD_SIZE = int(os.environ.get('DR_SHARD_SIZE', 64))
PREPROCESS_WORKERS = int(os.environ.get('DR_PREPROCESS_WORKERS', 2))
if min(PATCH_CHUNK_SIZE, SHARD_SIZE, PREPROCESS_WORKERS) < 1:
    raise ValueError('Chunk, shard and worker counts must be positive.')
RUN_POOLED = False
POOLED_HEAD_CHECKPOINT_ROOT = INPUTS / 'checkpoints_head_ablation'
POOLED_HEAD_CHECKPOINT_ZIP = None
POOLED_HEAD_MAP = None
POOLED_MANIFESTS = None
if MAX_SAMPLES and 'diagnostic' not in RUN_ID.lower():
    raise ValueError('Use a diagnostic run ID for a sample limit.')
config = dict(run_id=RUN_ID, folds=FOLDS, variants=VARIANTS, splits=SPLITS, device='auto',
              patch_chunk_size=PATCH_CHUNK_SIZE, shard_size=SHARD_SIZE, max_samples=MAX_SAMPLES,
              local_run=str(LOCAL_ROOT / 'runs' / RUN_ID), drive_run=str(DRIVE_OUTPUT / RUN_ID),
              images_root_override=None, checkpoint_root_override=None,
              verified_patient_map=os.environ.get('DR_PATIENT_MAP'))

# %% [markdown]
# Stage the original split manifests, reference summary, image ZIP and checkpoints. The historical asset checks bind the replay to the supplied DMIP splits. Each completed shard is saved under the run ID. The optional pooled benchmark uses its own checkpoint and manifest map.

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
"""Shared definitions copied as readable cells into every standalone notebook."""
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from tqdm.auto import tqdm

COHORTS = ['aptos', 'idrid', 'messidor2', 'ddr', 'eyepacs', 'deepdrid']
VARIANT_NAMES = {'V2': 'V2_PlusGlobalContext_H2', 'V3': 'V3_PlusOrdinalHead_H3_Champion'}
PROB_COLS = [f'p{k}' for k in range(5)]
TIE_TOL = 1e-12


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def unpack_zip(archive_path, destination):
    """Preserve full paths; reject traversal, symlinks and duplicate member paths.

    Index signature identifies filename/size/CRC, NOT cryptographic byte identity.
    Incomplete extractions never get an extraction marker.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        entries = z.infolist()
        index = [(i.filename, i.file_size, i.CRC) for i in entries]
        signature = object_hash(index)
        marker = destination / '_EXTRACTED.json'
        seen = set()
        for info in entries:
            name = info.filename.replace('\\', '/')
            rel = PurePosixPath(name)
            if (rel.is_absolute() or '..' in rel.parts or ':' in name
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError(f'Unsafe ZIP member: {name}')
            target = destination.joinpath(*rel.parts).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(name)
            if name.casefold() in seen:
                raise ValueError(f'Duplicate ZIP member path: {name}')
            seen.add(name.casefold())
        if marker.exists():
            old = json.loads(marker.read_text())
            if old['zip_index_signature'] != signature:
                raise ValueError(f'Archive changed; choose a NEW extraction directory: {destination}')
            if all(i.is_dir() or (destination / i.filename).is_file()
                   and (destination / i.filename).stat().st_size == i.file_size for i in entries):
                return destination
        needed = sum(i.file_size for i in entries if not i.is_dir())
        if shutil.disk_usage(destination).free < needed * 1.05:
            raise OSError(f'Need about {needed / 2**30:.1f} GiB free to extract {archive_path}')
        for info in tqdm(entries, desc=f'Unzip {Path(archive_path).name}'):
            if info.is_dir():
                continue
            target = destination.joinpath(*PurePosixPath(info.filename.replace('\\', '/')).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        save_json(marker, dict(zip_index_signature=signature, archive=str(archive_path),
                               uncompressed_bytes=needed))
    return destination


def validate_probs(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or not len(p):
        raise ValueError('Need nonempty N x 5 posterior')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-6).any():
        raise ValueError('Invalid posterior values; no silent repair')
    if not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Posterior rows do not sum to one')
    return p


def decision_metrics(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    cm = confusion_matrix(y, pred, labels=np.arange(5))
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.full(5, np.nan), where=support > 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        qwk = cohen_kappa_score(y, pred, labels=np.arange(5), weights='quadratic')
    n3, predicted3 = int(support[3]), int(cm[:, 3].sum())
    result = dict(n=len(y), qwk=float(qwk), accuracy=float(np.mean(y == pred)),
                  within1=float(np.mean(abs(y - pred) <= 1)), mae=float(np.mean(abs(y - pred))),
                  macro_recall=float(np.mean(recall)), g3_precision=float(cm[3, 3] / predicted3)
                  if predicted3 else np.nan, g3_precision_defined=bool(predicted3),
                  true_g3=n3, predicted_g3=predicted3,
                  true_g3_prevalence=n3 / len(y), predicted_g3_prevalence=predicted3 / len(y),
                  predicted_true_g3_ratio=predicted3 / n3 if n3 else np.nan)
    for k in range(5):
        result[f'recall_g{k}'] = float(recall[k])
        result[f'support_g{k}'] = int(support[k])
        result[f'g3_to_{k}_count'] = int(cm[3, k])
        result[f'g3_to_{k}_fraction_all'] = float(cm[3, k] / n3) if n3 else np.nan
    errors = n3 - cm[3, 3]
    result['g3_adjacent_fraction_errors'] = float((cm[3, 2] + cm[3, 4]) / errors) if errors else np.nan
    result['g3_nonreferable_fraction_all'] = float(cm[3, :2].sum() / n3) if n3 else np.nan
    return result


def posterior_metrics(y, p):
    p = validate_probs(p)
    y = np.asarray(y, int)
    cdf = p.cumsum(1)[:, :4]
    rps = np.mean((cdf - (y[:, None] <= np.arange(4))) ** 2)
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
              for b in range(15) if (bins == b).any())
    risk = p[:, 2:].sum(1)
    return dict(nll_mean=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
                rps_loss=float(rps), ece_15=float(ece),
                rdr_auroc=float(roc_auc_score(y >= 2, risk)) if len(np.unique(y >= 2)) == 2 else np.nan)


def screening_threshold(y_source_val, p_source_val, specificity=0.95):
    truth = np.asarray(y_source_val) >= 2
    if len(np.unique(truth)) != 2:
        raise ValueError('Source validation needs referable and non-referable images')
    fpr, tpr, thresholds = roc_curve(truth, validate_probs(p_source_val)[:, 2:].sum(1), drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= 1 - specificity + 1e-12)
    # Highest sensitivity; then highest specificity; then highest threshold.
    i = max(eligible, key=lambda j: (tpr[j], -fpr[j], thresholds[j]))
    return float(thresholds[i])


def screening_metrics(y, p, threshold):
    truth = np.asarray(y) >= 2
    pred = np.asarray(p)[:, 2:].sum(1) >= threshold
    return dict(threshold=threshold, sensitivity=float(pred[truth].mean()) if truth.any() else np.nan,
                specificity=float((~pred[~truth]).mean()) if (~truth).any() else np.nan)


def prior_from_labels(y, smoothing=1.0):
    counts = np.bincount(np.asarray(y, int), minlength=5).astype(float) + smoothing
    return counts / counts.sum()


def cibd(p, prior, alpha, tau, q):
    p = validate_probs(p)
    prior = np.asarray(prior, float)
    if (prior <= 0).any():
        raise ValueError('Use the prespecified smoothed SOURCE TRAIN prior')
    r = p / prior[None, :] ** tau  # exact zeros stay zero
    r /= r.sum(1, keepdims=True)
    d = abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    cost = alpha * d ** q + (1 - alpha) * (d != 0)
    risk = r @ cost.T  # row=prediction, column=true
    return (risk <= risk.min(1, keepdims=True) + TIE_TOL).argmax(1)


def decode_all(df, prior, selected):
    p = validate_probs(df[PROB_COLS].to_numpy())
    result = {'native_replay': df.pred_native.to_numpy(int), 'argmax': p.argmax(1),
              'posterior_median': (p.cumsum(1) >= 0.5).argmax(1),
              'rounded_expectation': cibd(p, prior, 1, 0, 2)}
    if df.s.notna().all() and df[[f'theta{k}' for k in range(4)]].notna().all().all():
        result['cutpoint_latent'] = (df.s.to_numpy()[:, None] > df[[f'theta{k}' for k in range(4)]].to_numpy()).sum(1)
    result['cibd_selected'] = (df.pred_native.to_numpy(int) if selected.get('fallback') else
                               cibd(p, prior, selected['alpha'], selected['tau'], selected['q']))
    return result


def select_cibd(source_val, prior):
    y = source_val.y_true.to_numpy(int)
    native = decision_metrics(y, source_val.pred_native.to_numpy(int))
    rows = []
    for alpha in np.arange(11) / 10:
        for tau in [0, .25, .5, .75, 1]:
            for q in [1, 2]:
                m = decision_metrics(y, cibd(source_val[PROB_COLS].to_numpy(), prior, alpha, tau, q))
                rows.append(dict(alpha=float(alpha), tau=tau, q=q, **m,
                                 feasible=m['within1'] >= native['within1'] - .01,
                                 qwk_guardrail_met=m['qwk'] >= native['qwk'] - .01))
    grid = pd.DataFrame(rows)
    eligible = grid[grid.feasible & grid.recall_g3.notna()]
    if eligible.empty:
        return dict(fallback=True, reason='NO_FEASIBLE_CANDIDATE'), grid
    best = eligible.sort_values(['recall_g3', 'macro_recall', 'within1', 'qwk', 'tau', 'q', 'alpha'],
                                ascending=[False, False, False, False, True, True, True]).iloc[0]
    return dict(alpha=float(best.alpha), tau=float(best.tau), q=int(best.q), fallback=False), grid


def paired_cluster_ci(df, pred_a, pred_b, metric='recall_g3', reps=2000, seed=42):
    """All images and multiplicity retained; unverified patient keys => NA, never singleton CI."""
    point = decision_metrics(df.y_true, pred_a)[metric] - decision_metrics(df.y_true, pred_b)[metric]
    statuses = set(df.patient_identity_status.astype(str))
    result = dict(difference=point, lower=np.nan, upper=np.nan, valid=0, invalid=reps,
                  reps=reps, unit='patient_cluster', status='NOT_ESTIMABLE')
    if statuses != {'verified'}:
        return result
    groups = [np.asarray(v, int) for v in df.reset_index(drop=True).groupby('patient_key', sort=True).indices.values()]
    rng = np.random.default_rng(seed)
    values = []
    y, a, b = df.y_true.to_numpy(int), np.asarray(pred_a), np.asarray(pred_b)
    for _ in range(reps):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        delta = decision_metrics(y[ix], a[ix])[metric] - decision_metrics(y[ix], b[ix])[metric]
        values.append(delta)
    v = np.asarray(values)
    good = np.isfinite(v)
    result.update(valid=int(good.sum()), invalid=int((~good).sum()))
    if good.mean() >= .95:
        result.update(lower=float(np.quantile(v[good], .025)), upper=float(np.quantile(v[good], .975)), status='ESTIMABLE')
    return result


def verify_artifact(directory):
    directory = Path(directory)
    marker = directory / 'COMPLETE.json'
    if not marker.exists():
        raise ValueError(f'Incomplete P0 shard: {directory}')
    meta = json.loads(marker.read_text())
    for name, digest in meta['files'].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f'Artifact hash mismatch: {directory / name}')
    return meta


def load_shard(p0_run, variant, fold, split, features=False):
    directory = Path(p0_run) / 'lodo' / f'{variant}_fold_{fold}_{COHORTS[fold]}' / split
    meta = verify_artifact(directory)
    df = pd.read_csv(directory / 'predictions.csv', dtype={'image_key': str, 'image_id': str, 'patient_key': str})
    if df.image_key.duplicated().any() or df.image_key.tolist() != meta['image_keys']:
        raise ValueError('Image identity/order corruption')
    if set(df.variant) != {variant} or set(df.fold_id) != {fold} or set(df.split_role) != {split}:
        raise ValueError('Wrong variant/fold/split artifact')
    validate_probs(df[PROB_COLS].to_numpy())
    if not features:
        return df, meta
    z = np.load(directory / 'z.npy', mmap_mode='r', allow_pickle=False)
    if z.shape != (len(df), 512) or not np.isfinite(z).all():
        raise ValueError('Invalid feature matrix')
    return df, z, meta


def load_pooled_shard(p0_run, head, split):
    directory = Path(p0_run) / 'pooled' / head / split
    meta = verify_artifact(directory)
    df = pd.read_csv(directory / 'predictions.csv', dtype={'image_key': str, 'image_id': str, 'patient_key': str})
    if df.image_key.duplicated().any() or df.image_key.tolist() != meta['image_keys']:
        raise ValueError('Pooled image identity/order mismatch')
    if set(df.protocol_id) != {'pooled_head_benchmark'} or set(df.variant) != {head} or set(df.split_role) != {split}:
        raise ValueError('Wrong pooled protocol/head/split')
    validate_probs(df[PROB_COLS].to_numpy())
    return df, meta


def complete_directory(directory, identity, image_keys=None):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'COMPLETE.json'}
    save_json(directory / 'COMPLETE.json', dict(identity=identity, image_keys=image_keys, files=files,
                                               completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def copy_completed(source, destination):
    """Snapshot analysis fold; completion is written last after all copied hashes match."""
    source, destination = Path(source), Path(destination)
    meta = verify_artifact(source)
    destination.mkdir(parents=True, exist_ok=True)
    flag = destination / 'COMPLETE.json'
    if flag.exists():
        flag.unlink()
    for name, digest in meta['files'].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        if sha256_file(target) != digest:
            raise IOError(f'Analysis snapshot checksum mismatch: {name}')
    shutil.copy2(source / 'COMPLETE.json', flag)


def aggregate_tables(table, out, group_cols):
    numeric = [c for c in table.select_dtypes(include='number').columns if c not in ['fold_id', 'n']]
    macro = table.groupby(group_cols, dropna=False)[numeric].mean().reset_index()
    coverage = table.groupby(group_cols, dropna=False).fold_id.nunique().reset_index(name='cohorts_available')
    macro = macro.merge(coverage, on=group_cols)
    macro['aggregation'] = macro.cohorts_available.map(lambda k: f'equal-cohort macro-{k} available')
    macro.to_csv(Path(out) / 'equal_cohort_means.csv', index=False)
    return macro

# %%
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# %%
class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM (Convolutional Block Attention Module).
    Exploits inter-channel relationships of features by combining spatial Average Pooling
    and Max Pooling descriptors through a shared bottleneck MLP.
    """

    def __init__(self, in_channels: int, reduction_ratio: int=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_dim = max(8, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM (Convolutional Block Attention Module).
    Captures inter-spatial relationships of features by computing channel-wise Average
    and Max statistics, followed by a large receptive field 7x7 convolution.
    """

    def __init__(self, kernel_size: int=7):
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
    Convolutional Block Attention Module (CBAM).
    Sequentially applies 1D Channel Attention followed by 2D Spatial Attention:
      1. Channel Refinement: F' = Mc(F) ⊗ F
      2. Spatial Refinement: F'' = Ms(F') ⊗ F'
    """

    def __init__(self, in_channels: int, reduction_ratio: int=16, kernel_size: int=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channel = x * self.channel_att(x)
        x_out = x_channel * self.spatial_att(x_channel)
        return x_out

class NonLocalBlock2D(nn.Module):
    """
    Embedded Gaussian Non-local block (Wang et al., 2018).

    Why this is needed for H2: CBAM's spatial attention is a single 7x7 conv
    applied to the ALREADY-pooled feature map - its receptive field only spans
    a fixed local neighborhood, so it can re-weight "this region matters more"
    but cannot let one spatial location directly relate to another far-away
    location. Grade 3's 4-2-1 rule needs exactly that: to know whether lesions
    appear in >= 2 or all 4 retinal quadrants, the network must compare
    evidence across quadrants that CBAM's local conv structurally cannot reach.

    Non-local computes attention between EVERY pair of spatial positions
    (full self-attention over the H*W feature map), so a lesion signal in the
    top-left quadrant can directly inform the representation at the
    bottom-right quadrant - the counting/relational operation CBAM lacks.
    """

    def __init__(self, in_channels: int, reduction: int=2):
        super().__init__()
        self.inter_channels = max(1, in_channels // reduction)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        phi = self.phi(x).view(B, self.inter_channels, -1)
        g = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(theta, phi) / self.inter_channels ** 0.5, dim=-1)
        y = torch.bmm(attn, g)
        y = y.permute(0, 2, 1).view(B, self.inter_channels, H, W)
        y = self.out_conv(y)
        return x + y

class QuadrantTokenAggregator(nn.Module):
    """
    Splits the final feature map into the 4 spatial quadrants that match the
    clinical retinal quadrant convention used by the ICDR 4-2-1 rule for
    Grade 3 (hemorrhages in 4 quadrants / venous beading in >=2 / IRMA in
    >=1). Each quadrant is pooled into one token, then a small self-attention
    encoder (1-layer Transformer) lets the 4 quadrant tokens directly compare
    against each other before being aggregated - the explicit "how many
    quadrants show signal" reasoning that neither CBAM nor plain GAP performs
    (GAP just averages everything into one vector and destroys which quadrant
    contributed what).
    """

    def __init__(self, in_channels: int, num_heads: int=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=in_channels * 2, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.quadrant_pos_embed = nn.Parameter(torch.zeros(1, 4, in_channels))
        nn.init.trunc_normal_(self.quadrant_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_mid, w_mid = (H // 2, W // 2)
        quadrants = [x[:, :, :h_mid, :w_mid], x[:, :, :h_mid, w_mid:], x[:, :, h_mid:, :w_mid], x[:, :, h_mid:, w_mid:]]
        tokens = torch.stack([self.pool(q).flatten(1) for q in quadrants], dim=1)
        tokens = tokens + self.quadrant_pos_embed
        tokens = self.encoder(tokens)
        quadrant_summary = tokens.mean(dim=1)
        return quadrant_summary

class GlobalBranch(nn.Module):
    """
    Global Context Branch:
      - Input: Whole-fundus image tensor (Batch, 3, 512, 512)
      - Backbone: Pre-trained ResNet-50 (extracts deep macro-vascular and structural features)
      - Attention: Non-local block (long-range spatial relations) -> CBAM (channel+spatial
        refinement), applied on layer4 feature maps (2048 channels)
      - Pooling: GAP branch (2048-d, overall context) CONCATENATED with Quadrant Token
        branch (2048-d, explicit 4-quadrant relational aggregation
        review to give H2 / the Grade-3 4-2-1 rule an actual architectural mechanism,
        since CBAM's 7x7 conv on the pooled map alone cannot relate distant quadrants)
      - Output: Global context feature vector F_Global, dimension 4096 (2048+2048) when
        Quadrant Tokens are enabled, else 2048
    """

    def __init__(self, pretrained: bool=True, use_cbam: bool=True, use_nonlocal: bool=True, use_quadrant_tokens: bool=True):
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
        self.layer4 = base_resnet.layer4
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
        x = self.layer4(x)
        if self.use_nonlocal:
            x = self.non_local(x)
        if self.use_cbam:
            x = self.cbam(x)
        feat_gap = self.gap(x)
        feat_gap = torch.flatten(feat_gap, 1)
        if self.use_quadrant_tokens:
            feat_quadrant = self.quadrant_agg(x)
            feat = torch.cat([feat_gap, feat_quadrant], dim=1)
        else:
            feat = feat_gap
        return feat

class GatedAttentionPool(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning (Ilse et al., 2018).
    Learns instance-level attention weights a_k = softmax(w^T (tanh(V h_k) * sigm(U h_k))).
    Enables lesion counting and spatial density aggregation for Severe DR (Grade 3 - H2).
    """

    def __init__(self, in_features: int=1280, hidden_dim: int=128):
        super(GatedAttentionPool, self).__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        x: [B, K, in_features]
        patch_mask: Optional boolean tensor of shape [B, K] (True = real patch, False = padding patch)
        """
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        a = self.attention_weights(a_v * a_u)
        if patch_mask is not None:
            mask_3d = patch_mask.unsqueeze(-1)
            a = a.masked_fill(~mask_3d, -10000.0)
        a = torch.softmax(a, dim=1)
        self.last_attention_weights = a.detach()
        weighted_out = torch.sum(x * a, dim=1)
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
      Output: F_Local feature vector of dimension 2560 (1280 + 1280) if use_dual_pooling else 1280
    """

    def __init__(self, pretrained: bool=True, use_dual_pooling: bool=True):
        super(LocalMILBranch, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_effnet = models.efficientnet_b0(weights=weights)
        self.features = base_effnet.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_dim = 1280
        self.use_dual_pooling = use_dual_pooling
        if use_dual_pooling:
            self.gated_pool = GatedAttentionPool(in_features=self.patch_dim, hidden_dim=128)
            self.out_dim = self.patch_dim * 2
        else:
            self.out_dim = self.patch_dim

    def forward(self, patches: torch.Tensor, patch_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        B, K, C, H, W = patches.shape
        flat_patches = patches.view(B * K, C, H, W)
        patch_feats = self.features(flat_patches)
        patch_feats = self.gap(patch_feats)
        patch_feats = torch.flatten(patch_feats, 1)
        patch_feats = patch_feats.view(B, K, self.patch_dim)
        if patch_mask is not None:
            masked_feats = patch_feats.masked_fill(~patch_mask.unsqueeze(-1), -10000.0)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)
        if self.use_dual_pooling:
            att_pooled = self.gated_pool(patch_feats, patch_mask=patch_mask)
            f_local = torch.cat([max_pooled, att_pooled], dim=1)
        else:
            f_local = max_pooled
        return f_local

class DualBranchMILModel(nn.Module):
    """
    Unified Dual-Branch Multi-Scale Deep Learning Network:
      Combines Global Context (ResNet-50 + Non-Local + CBAM + Quadrant Tokens, 4096-d)
      and Local Micro-Lesions (MIL EfficientNet-B0 Dual Pooling, 2560-d).
      Supports parametric flags (use_local_branch, use_nonlocal, use_quadrant_tokens)
      to execute the full ablation waterfall (V0 -> V1 -> V2 -> V3).
    """

    def __init__(self, pretrained: bool=True, use_cbam: bool=True, use_dual_pooling: bool=True, use_local_branch: bool=True, use_nonlocal: bool=True, use_quadrant_tokens: bool=True, fusion_dim: int=512, dropout_rate: float=0.3):
        super(DualBranchMILModel, self).__init__()
        self.use_local_branch = use_local_branch
        self.global_branch = GlobalBranch(pretrained=pretrained, use_cbam=use_cbam, use_nonlocal=use_nonlocal, use_quadrant_tokens=use_quadrant_tokens)
        if use_local_branch:
            self.local_mil_branch = LocalMILBranch(pretrained=pretrained, use_dual_pooling=use_dual_pooling)
            combined_dim = self.global_branch.out_dim + self.local_mil_branch.out_dim
        else:
            self.local_mil_branch = None
            combined_dim = self.global_branch.out_dim
        self.fusion_layer = nn.Sequential(nn.Linear(combined_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.Mish(inplace=True), nn.Dropout(p=dropout_rate), nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.latent_dim = fusion_dim

    def forward(self, global_img: torch.Tensor, local_patches: Optional[torch.Tensor]=None, patch_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        f_global = self.global_branch(global_img)
        if self.use_local_branch and self.local_mil_branch is not None and (local_patches is not None):
            f_local = self.local_mil_branch(local_patches, patch_mask=patch_mask)
            f_fused = torch.cat([f_global, f_local], dim=1)
        else:
            f_fused = f_global
        latent = self.fusion_layer(f_fused)
        return latent

class ContinuousQWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss for PyTorch.
    Computes continuous soft confusion matrices to directly optimize clinical ordinal agreement:
      QWK = 1 - (sum(W * O) / sum(W * E))
      Loss = sum(W * O) / (sum(W * E) + eps)
    """

    def __init__(self, num_classes: int=5, eps: float=1e-07):
        super(ContinuousQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = (i - j) ** 2 / (num_classes - 1) ** 2
        self.register_buffer('W', w)

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of differentiable QWK loss.

        Args:
            pred_probs: Continuous class probabilities (Batch, num_classes) summing to 1.
            targets: Ground truth integer class indices (Batch,) or one-hot targets (Batch, num_classes).

        Returns:
            torch.Tensor: Scalar differentiable QWK loss value.
        """
        device = pred_probs.device
        N = pred_probs.size(0)
        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)
        O = torch.matmul(pred_probs.t(), y_one_hot)
        O = O / (N + self.eps)
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (N + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (N + self.eps)
        E = torch.matmul(hist_pred.t(), hist_true)
        W = self.W.to(device)
        num = torch.sum(W * O)
        den = torch.sum(W * E) + self.eps
        qwk_loss = num / den
        return qwk_loss

class SoftmaxCEHead(nn.Module):
    """
    Output Head 1: Standard Categorical Classification Baseline.
      - Architecture: Linear(latent_dim, num_classes) -> Softmax.
      - Loss: Categorical Cross-Entropy (treats disease stages as mutually independent nominal classes).
    """

    def __init__(self, in_features: int=512, num_classes: int=5):
        super(SoftmaxCEHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes unconstrained categorical logits, probabilities, and class predictions.
        """
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {'logits': logits, 'probs': probs, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Computes Standard Categorical Cross-Entropy Loss with optional class weighting.
        """
        logits = outputs['logits']
        return F.cross_entropy(logits, targets, weight=class_weights)

class SoftmaxQWKHead(nn.Module):
    """
    Output Head 2: Softmax Classification Head with Hybrid Multitask Loss.
      - Architecture: Linear(latent_dim, num_classes) -> Softmax.
      - Loss: Total_Loss = Cross_Entropy + lambda * Continuous_QWK_Loss.
      - Inference: Computes expected continuous severity score via probability expectation:
                   E[y] = sum_{k=0}^{C-1} (k * P(y=k)), rounded to nearest integer grade.
    """

    def __init__(self, in_features: int=512, num_classes: int=5, qwk_weight: float=1.0):
        super(SoftmaxQWKHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.qwk_weight = qwk_weight
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes categorical probabilities and expected continuous severity values.
        """
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        continuous_pred = torch.sum(probs * torch.arange(self.num_classes, device=x.device), dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()
        return {'logits': logits, 'probs': probs, 'preds': preds, 'continuous_preds': continuous_pred}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Computes composite multitask loss combining point-wise CE and global QWK optimization.
        """
        logits = outputs['logits']
        probs = outputs['probs']
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights)
        qwk = self.qwk_loss(probs, targets)
        total_loss = ce_loss + self.qwk_weight * qwk
        return total_loss

class CORALHead(nn.Module):
    """
    Output Head 3: Consistent Rank Logits (CORAL) for Ordinal Regression.
    Features:
      - Shared weight vector W across all rank threshold sub-tasks (ensures parallel hyperplanes).
      - Independent learnable threshold biases b_1, b_2, ..., b_{K-1}.
      - Cumulative probability formulation: P(y > k) = sigmoid(W^T x + b_k).
    """

    def __init__(self, in_features: int=512, num_classes: int=5):
        super(CORALHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        with torch.no_grad():
            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes cumulative rank logits, cumulative threshold probabilities, and discrete class predictions.
        """
        base_logit = self.linear(x)
        rank_logits = base_logit + self.biases
        rank_probs = torch.sigmoid(rank_logits)
        preds = torch.sum(rank_probs > 0.5, dim=1).long()
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(rank_probs[:, k - 1] - rank_probs[:, k])
        probs[:, -1] = rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-07)
        return {'rank_logits': rank_logits, 'rank_probs': rank_probs, 'probs': probs, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Computes multi-task Binary Cross-Entropy loss across all binary rank threshold indicators.
        """
        rank_logits = outputs['rank_logits']
        device = targets.device
        B = targets.size(0)
        binary_targets = torch.zeros(B, self.num_thresholds, device=device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
        bce_loss = F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction='mean')
        return bce_loss

class CORNHead(nn.Module):
    """
    Output Head 4: Conditional Ordinal Regression Neural Network (CORN).
    Features:
      - Models conditional forward transition probabilities: P(y > k | y >= k) = sigmoid(g(x)_k).
      - Applies chain rule multiplication: P(y > k) = prod_{j=0}^k P(y > j | y >= j).
      - Mathematically guarantees strictly non-increasing cumulative probabilities without threshold crossing.
    """

    def __init__(self, in_features: int=512, num_classes: int=5):
        super(CORNHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes conditional transition probabilities, cumulative probabilities via cumulative product,
        and discrete ordinal class predictions.
        """
        cond_logits = self.fc(x)
        cond_probs = torch.sigmoid(cond_logits)
        cum_probs = torch.cumprod(cond_probs, dim=1)
        preds = torch.sum(cum_probs > 0.5, dim=1).long()
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(cum_probs[:, k - 1] - cum_probs[:, k])
        probs[:, -1] = cum_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-07)
        return {'cond_logits': cond_logits, 'cond_probs': cond_probs, 'cum_probs': cum_probs, 'probs': probs, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Computes masked Binary Cross-Entropy loss conditioned on each sub-task's eligible sample subset.
        """
        cond_logits = outputs['cond_logits']
        B = targets.size(0)
        device = targets.device
        loss = 0.0
        valid_tasks = 0
        for k in range(self.num_thresholds):
            mask = targets >= k
            if mask.sum() > 0:
                subset_logits = cond_logits[mask, k]
                subset_targets = (targets[mask] > k).float()
                task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
                loss += task_loss
                valid_tasks += 1
        return loss / max(1, valid_tasks)

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
      - Rank Probability: P(Y > k) = 1 - P(Y <= k)
      - Interval Probability: P(Y = k) = P(Y <= k) - P(Y <= k-1)
      - Loss: Negative Log-Likelihood (NLL) + lambda * Continuous_QWK_Loss
    """

    def __init__(self, in_features: int=512, num_classes: int=5, qwk_weight: float=1.2):
        super(CumulativeLinkModelQWKHead, self).__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        nn.init.zeros_(self.score_fc.weight)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float):
        """Dynamically updates the lambda weight for continuous QWK loss during warmup."""
        self.current_qwk_weight = weight

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments) + 0.0001
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = x.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            severity_score = self.score_fc(x_fp32).squeeze(1)
            cutpoints = self._get_monotonic_cutpoints().float()
            z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
            z_safe = torch.clamp(z, min=-20.0, max=20.0)
            log_survival = -torch.exp(z_safe)
            survival = torch.exp(log_survival)
            probs = torch.empty(x.size(0), self.num_classes, device=x.device, dtype=torch.float32)
            probs[:, 0] = -torch.expm1(log_survival[:, 0])
            for k in range(1, self.num_cutpoints):
                probs[:, k] = survival[:, k - 1] - survival[:, k]
            probs[:, -1] = survival[:, -1]
            probs = probs.clamp_min(0.0)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
            cum_probs = 1.0 - survival
            rank_probs = survival
            class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
            expected_grade = torch.sum(probs * class_indices, dim=1)
            preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()
        return {'severity_score': severity_score, 'latent_score': severity_score, 'cutpoints': cutpoints, 'thetas': cutpoints, 'cum_probs': cum_probs, 'rank_probs': rank_probs, 'probs': probs, 'expected_grade': expected_grade, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        probs = outputs['probs']
        B = targets.size(0)
        eps = 1e-07
        target_probs = probs[torch.arange(B, device=targets.device), targets]
        nll_loss = -torch.log(target_probs + 1e-12)
        if class_weights is not None:
            weights_per_sample = class_weights[targets]
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()
        qwk = self.qwk_loss(probs, targets)
        total_loss = nll_loss + self.current_qwk_weight * qwk
        return total_loss

class DRFullModel(nn.Module):

    def __init__(self, backbone: DualBranchMILModel, head_type: str='CLM_QWK', num_classes: int=5, qwk_weight: float=0.4):
        super(DRFullModel, self).__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.num_classes = num_classes
        in_features = backbone.latent_dim
        if head_type == 'Softmax_CE':
            self.head = SoftmaxCEHead(in_features=in_features, num_classes=num_classes)
        elif head_type == 'Softmax_QWK':
            self.head = SoftmaxQWKHead(in_features=in_features, num_classes=num_classes, qwk_weight=qwk_weight)
        elif head_type == 'CORAL':
            self.head = CORALHead(in_features=in_features, num_classes=num_classes)
        elif head_type == 'CORN':
            self.head = CORNHead(in_features=in_features, num_classes=num_classes)
        elif head_type == 'CLM_QWK':
            self.head = CumulativeLinkModelQWKHead(in_features=in_features, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f"Unknown head_type: '{head_type}'.")

    def forward(self, global_img: torch.Tensor, local_patches: Optional[torch.Tensor]=None, patch_mask: Optional[torch.Tensor]=None, targets: Optional[torch.Tensor]=None, class_weights: Optional[torch.Tensor]=None, return_loss: bool=False) -> Dict[str, Any]:
        latent = self.backbone(global_img, local_patches=local_patches, patch_mask=patch_mask)
        head_outputs = self.head(latent)
        head_outputs['latent'] = latent
        if return_loss and targets is not None:
            head_outputs['loss'] = self.head.loss_fn(head_outputs, targets, class_weights=class_weights)
        return head_outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.head.loss_fn(outputs, targets, class_weights=class_weights)

def build_model(head_type: str='CLM_QWK', pretrained: bool=True, use_cbam: bool=True, use_local_branch: bool=True, use_nonlocal: bool=True, use_quadrant_tokens: bool=True, fusion_dim: int=512, num_classes: int=5, qwk_weight: float=0.4) -> DRFullModel:
    backbone = DualBranchMILModel(pretrained=pretrained, use_cbam=use_cbam, use_local_branch=use_local_branch, use_nonlocal=use_nonlocal, use_quadrant_tokens=use_quadrant_tokens, fusion_dim=fusion_dim)
    model = DRFullModel(backbone=backbone, head_type=head_type, num_classes=num_classes, qwk_weight=qwk_weight)
    return model

# %%
def apply_green_channel_ablation(img_rgb: np.ndarray, ablation_mode: str='ben_graham_green') -> np.ndarray:
    """
    Applies fundus image preprocessing ablations centered on the Green channel.
    The Green channel provides the highest contrast for retinal vascular micro-lesions
    (microaneurysms, hemorrhages, and exudates).

    Args:
        img_rgb: Input fundus image array in RGB format (H, W, 3).
        ablation_mode: Preprocessing strategy:
            - 'ben_graham_green': Gaussian background subtraction on Green channel.
            - 'clahe_green': Contrast Limited Adaptive Histogram Equalization.
            - 'raw_green' (default/fallback): Unmodified Green channel.

    Returns:
        Processed 3-channel pseudo-RGB image array (H, W, 3).
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)
    if ablation_mode == 'ben_graham_green':
        blurred = cv2.GaussianBlur(g, (0, 0), 30)
        adjusted = cv2.addWeighted(g, 4, blurred, -4, 128)
        processed = cv2.merge([adjusted, adjusted, adjusted])
    elif ablation_mode == 'clahe_green':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g)
        processed = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    else:
        processed = cv2.merge([g, g, g])
    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

def compute_fov_mask(img_np: np.ndarray, erode_px: int=15) -> np.ndarray:
    """
    Computes the true retinal Field-of-View (FOV) binary mask for a SINGLE fundus image.
    IMPORTANT: Must be computed on the RAW RGB fundus image BEFORE Ben Graham preprocessing.
    (Ben Graham sets background to 128 gray, which collapses Otsu thresholding).

    4-Step Pipeline:
    1. Otsu's Adaptive Thresholding: Dynamically calculates the optimal bimodal threshold on raw image.
    2. Morphological Close + Open (25x25 ellipse): Seals internal vessel holes and removes noise.
    3. Largest Connected Component: Strictly retains the main retinal disk (removes artifacts/watermarks).
    4. Inward Morphological Erosion (erode_px=15): Strips away peripheral edge vignetting/halos.
    """
    if img_np.ndim == 3:
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        fov_mask = cleaned
    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        fov_mask = cv2.erode(fov_mask, kernel_erode)
    return (fov_mask > 0).astype(np.uint8)

def compute_valid_anchors(img_shape: Tuple[int, int], fov_mask: np.ndarray, patch_size: int=128, stride: int=96, fov_thresh: float=0.5) -> np.ndarray:
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
        y_min, x_min, y_max, x_max = (0, 0, h, w)
    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    def get_coords(curr_stride: int):
        s = max(1, curr_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]
        if content_h > p and y_min + content_h - p not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and x_min + content_w - p not in x_starts:
            x_starts.append(x_min + content_w - p)
        valid_list = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = (min(y0 + p, h), min(x0 + p, w))
                mask_patch = fov_mask[y0:y1, x0:x1]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio >= fov_thresh:
                    valid_list.append([y0, x0])
        return valid_list
    anchors = get_coords(stride)
    if len(anchors) < 32 and stride > 48 and (min(content_h, content_w) > p):
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors
    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]
    return np.array(anchors, dtype=np.int32)

def extract_mil_patches_sliding(img_np: np.ndarray, fov_mask: Optional[np.ndarray]=None, patch_size: int=128, stride: int=96, fov_thresh: float=0.5, mode: str='train', num_sample: Union[int, Tuple[int, int]]=48, max_patches_eval: int=256, native_path: Optional[str]=None, valid_anchors: Optional[np.ndarray]=None, return_anchors: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
    h, w, _ = img_np.shape
    p = patch_size
    if mode == 'train':
        if isinstance(num_sample, (tuple, list)):
            m_target = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            m_target = int(num_sample)
    else:
        m_target = max_patches_eval
    if valid_anchors is not None and len(valid_anchors) > 0:
        n_anchors = len(valid_anchors)
        if mode == 'train':
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
            y1, x1 = (min(y0 + p, h), min(x0 + p, w))
            patch = img_np[y0:y1, x0:x1]
            if patch.shape[0] < p or patch.shape[1] < p:
                pad_h = max(0, p - patch.shape[0])
                pad_w = max(0, p - patch.shape[1])
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            patches.append(patch)
        patches_np = np.stack(patches, axis=0)
        tensor_patches = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()
        if return_anchors:
            return (tensor_patches, np.asarray(selected_coords, dtype=np.int32))
        return tensor_patches
    if fov_mask is None:
        fov_mask = compute_fov_mask(img_np, erode_px=15)
    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = (0, 0, h, w)
    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    def get_candidates(current_stride):
        s = max(1, current_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]
        if content_h > p and y_min + content_h - p not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and x_min + content_w - p not in x_starts:
            x_starts.append(x_min + content_w - p)
        cands, cand_anchors = ([], [])
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = (y0 + p, x0 + p)
                y1c, x1c = (min(y1, h), min(x1, w))
                patch = img_np[y0:y1c, x0:x1c]
                mask_patch = fov_mask[y0:y1c, x0:x1c]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio < fov_thresh:
                    continue
                if patch.shape[0] < p or patch.shape[1] < p:
                    pad_h = max(0, p - patch.shape[0])
                    pad_w = max(0, p - patch.shape[1])
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                cands.append(patch)
                cand_anchors.append([y0, x0])
        return (cands, cand_anchors)
    candidates, cand_anchors = get_candidates(stride)
    if mode == 'train' and len(candidates) < m_target and (stride > 48) and (min(content_h, content_w) > p):
        dense_candidates, dense_anchors = get_candidates(48)
        if len(dense_candidates) > len(candidates):
            candidates, cand_anchors = (dense_candidates, dense_anchors)
    if len(candidates) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        patch = img_np[cy:cy + p, cx:cx + p]
        if patch.shape[0] < p or patch.shape[1] < p:
            patch = np.pad(patch, ((0, max(0, p - patch.shape[0])), (0, max(0, p - patch.shape[1])), (0, 0)), mode='reflect')
        candidates = [patch]
        cand_anchors = [[cy, cx]]
    if mode == 'train':
        if len(candidates) < m_target:
            idx = np.random.choice(len(candidates), size=m_target, replace=True)
        else:
            idx = np.random.choice(len(candidates), size=m_target, replace=False)
        selected = [candidates[i] for i in idx]
        selected_anchors = [cand_anchors[i] for i in idx]
    elif len(candidates) > max_patches_eval:
        idx = np.linspace(0, len(candidates) - 1, max_patches_eval).astype(int)
        selected = [candidates[i] for i in idx]
        selected_anchors = [cand_anchors[i] for i in idx]
    else:
        selected = candidates
        selected_anchors = cand_anchors
    patches_np = np.stack(selected, axis=0)
    patches_u8 = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()
    if return_anchors:
        return (patches_u8, np.asarray(selected_anchors, dtype=np.int32))
    return patches_u8

# %%
TARGET_QWK_WEIGHT = .4
SOFTMAX_QWK_TARGET = .4
class PooledContinuousQWKLoss(nn.Module):
    """
    Differentiable approximation of Cohen's Quadratic Weighted Kappa (QWK) Loss.
    Computes soft continuous confusion matrices directly from predicted probability distributions:
      QWK = 1 - (sum(W * O) / sum(W * E))
      Loss = sum(W * O) / (sum(W * E) + eps)
    """

    def __init__(self, num_classes: int=5, eps: float=1e-07):
        super(PooledContinuousQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = (i - j) ** 2 / (num_classes - 1) ** 2
        self.register_buffer('W', w)

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = pred_probs.device
        N = pred_probs.size(0)
        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)
        O = torch.matmul(pred_probs.t(), y_one_hot) / (N + self.eps)
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (N + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (N + self.eps)
        E = torch.matmul(hist_pred.t(), hist_true)
        W = self.W.to(device)
        num = torch.sum(W * O)
        den = torch.sum(W * E) + self.eps
        qwk_loss = num / den
        return qwk_loss

class PooledSoftmaxCEHead(nn.Module):
    """
    Output Head 1: Standard Categorical Classification Baseline.
      - Architecture: Linear(latent_dim, num_classes) -> Softmax.
      - Loss: Categorical Cross-Entropy (treats disease stages as mutually independent nominal classes).
    """

    def __init__(self, in_features: int=512, num_classes: int=5):
        super(PooledSoftmaxCEHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {'logits': logits, 'probs': probs, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        logits = outputs['logits']
        if class_weights is not None:
            class_weights = class_weights.to(targets.device)
        return F.cross_entropy(logits, targets, weight=class_weights)

class PooledSoftmaxQWKHead(nn.Module):
    """
    Output Head 2: Softmax Classification Head with Hybrid Multitask Loss.
      - Architecture: Linear(latent_dim, num_classes) -> Softmax.
      - Loss: Total_Loss = Cross_Entropy + lambda * Continuous_QWK_Loss.
      - Inference: Computes expected continuous severity score via probability expectation:
                   E[y] = sum_{k=0}^{C-1} (k * P(y=k)), rounded to nearest integer grade.
    """

    def __init__(self, in_features: int=512, num_classes: int=5, qwk_weight: float=SOFTMAX_QWK_TARGET):
        super(PooledSoftmaxQWKHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0
        self.qwk_loss = PooledContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float):
        """Dynamically updates the lambda weight during training warmup."""
        self.current_qwk_weight = float(weight)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        continuous_pred = torch.sum(probs * torch.arange(self.num_classes, device=x.device, dtype=torch.float32), dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()
        return {'logits': logits, 'probs': probs, 'preds': preds, 'continuous_preds': continuous_pred}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        logits = outputs['logits']
        probs = outputs['probs']
        if class_weights is not None:
            class_weights = class_weights.to(targets.device)
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights)
        qwk = self.qwk_loss(probs, targets)
        total_loss = ce_loss + self.current_qwk_weight * qwk
        return total_loss

class PooledCORALHead(nn.Module):
    """
    Output Head 3: Consistent Rank Logits (CORAL) for Ordinal Regression.
    Features:
      - Shared weight vector W across all rank threshold sub-tasks (ensures parallel hyperplanes).
      - Independent learnable threshold biases b_1, b_2, ..., b_{K-1}.
      - Cumulative probability formulation: P(y > k) = sigmoid(W^T x + b_k).
    """

    def __init__(self, in_features: int=512, num_classes: int=5):
        super(PooledCORALHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        with torch.no_grad():
            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_logit = self.linear(x)
        rank_logits = base_logit + self.biases
        rank_probs = torch.sigmoid(rank_logits)
        preds = torch.sum(rank_probs > 0.5, dim=1).long()
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(rank_probs[:, k - 1] - rank_probs[:, k])
        probs[:, -1] = rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-07)
        return {'rank_logits': rank_logits, 'rank_probs': rank_probs, 'probs': probs, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        rank_logits = outputs['rank_logits']
        device = targets.device
        B = targets.size(0)
        binary_targets = torch.zeros(B, self.num_thresholds, device=device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
        bce_loss = F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction='none')
        if class_weights is not None:
            class_weights = class_weights.to(device)
            sample_weights = class_weights[targets].unsqueeze(1)
            return (bce_loss * sample_weights).mean()
        return bce_loss.mean()

class PooledCORNHead(nn.Module):
    """
    Output Head 4: Conditional Ordinal Regression Neural Network (CORN).
    Features:
      - Models conditional forward transition probabilities: P(y > k | y >= k) = sigmoid(g(x)_k).
      - Applies chain rule multiplication: P(y > k) = prod_{j=0}^k P(y > j | y >= j).
      - Mathematically guarantees strictly non-increasing cumulative probabilities without threshold crossing.
    """

    def __init__(self, in_features: int=512, num_classes: int=5):
        super(PooledCORNHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cond_logits = self.fc(x)
        cond_probs = torch.sigmoid(cond_logits)
        cum_probs = torch.cumprod(cond_probs, dim=1)
        preds = torch.sum(cum_probs > 0.5, dim=1).long()
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(cum_probs[:, k - 1] - cum_probs[:, k])
        probs[:, -1] = cum_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-07)
        return {'cond_logits': cond_logits, 'cond_probs': cond_probs, 'cum_probs': cum_probs, 'rank_probs': cum_probs, 'probs': probs, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        cond_logits = outputs['cond_logits']
        device = targets.device
        if class_weights is not None:
            class_weights = class_weights.to(device)
        losses = []
        for k in range(self.num_thresholds):
            mask = targets >= k
            if mask.sum() > 0:
                subset_logits = cond_logits[mask, k]
                subset_targets = (targets[mask] > k).float()
                task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets, reduction='none')
                if class_weights is not None:
                    task_loss = task_loss * class_weights[targets[mask]]
                losses.append(task_loss.mean())
            else:
                losses.append(0.0 * cond_logits[:, k].sum())
        return torch.stack(losses).mean()

class PooledCumulativeLinkModelQWKHead(nn.Module):
    """
    Champion Output Architecture: Cumulative Link Model (CLM) with Complementary Log-Log (clog-log)
    Link Function & Hybrid Continuous QWK Loss with Step-Level Linear Warmup.

    Mathematical Formulation:
      - Latent Severity Score: s(x) = W^T x (scalar continuous representation)
      - Strictly Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4 parameterized
        via softplus increments: theta_k = theta_{k-1} + softplus(alpha_k) + 1e-4
      - Complementary Log-Log (clog-log) link: F(z) = 1 - exp(-exp(z))
      - Cumulative Probability: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
      - Rank Probability: P(Y > k) = 1 - P(Y <= k)
      - Interval Probability: P(Y = k) = P(Y <= k) - P(Y <= k-1)
      - Loss: Negative Log-Likelihood (NLL) + lambda * Continuous_QWK_Loss
    """

    def __init__(self, in_features: int=512, num_classes: int=5, qwk_weight: float=TARGET_QWK_WEIGHT):
        super(PooledCumulativeLinkModelQWKHead, self).__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)
        self.qwk_loss = PooledContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float):
        """Dynamically updates the lambda weight for continuous QWK loss during warmup."""
        self.current_qwk_weight = float(weight)

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments) + 0.0001
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        severity_score = self.score_fc(x).squeeze(1)
        cutpoints = self._get_monotonic_cutpoints()
        z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
        cum_probs = self._cloglog_cdf(z)
        rank_probs = 1.0 - cum_probs
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.num_cutpoints):
            probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
        probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-07)
        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * class_indices, dim=1)
        preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()
        return {'severity_score': severity_score, 'latent_score': severity_score, 'cutpoints': cutpoints, 'thetas': cutpoints, 'cum_probs': cum_probs, 'rank_probs': rank_probs, 'probs': probs, 'expected_grade': expected_grade, 'preds': preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        probs = outputs['probs']
        B = targets.size(0)
        eps = 1e-07
        target_probs = probs[torch.arange(B, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))
        if class_weights is not None:
            class_weights = class_weights.to(targets.device)
            weights_per_sample = class_weights[targets]
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()
        qwk = self.qwk_loss(probs, targets)
        total_loss = nll_loss + self.current_qwk_weight * qwk
        return total_loss

class PooledResNet50AblationModel(nn.Module):
    """
    Standardized ResNet-50 Feature Extractor with 512-d Latent Projection
    paired modularly with any of the 5 candidate output heads.
    """

    def __init__(self, head_type: str='CLM_QWK', pretrained: bool=True, num_classes: int=5, qwk_weight: float=TARGET_QWK_WEIGHT):
        super().__init__()
        self.head_type = head_type
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)
        self.stem = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Linear(2048, 512), nn.LayerNorm(512), nn.Mish(), nn.Dropout(0.2))
        if head_type == 'Softmax_CE':
            self.head = PooledSoftmaxCEHead(in_features=512, num_classes=num_classes)
        elif head_type == 'Softmax_QWK':
            self.head = PooledSoftmaxQWKHead(in_features=512, num_classes=num_classes, qwk_weight=SOFTMAX_QWK_TARGET)
        elif head_type == 'CORAL':
            self.head = PooledCORALHead(in_features=512, num_classes=num_classes)
        elif head_type == 'CORN':
            self.head = PooledCORNHead(in_features=512, num_classes=num_classes)
        elif head_type == 'CLM_QWK':
            self.head = PooledCumulativeLinkModelQWKHead(in_features=512, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f'Unknown head type: {head_type}')

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out).flatten(1)
        z = self.proj(out)
        return z

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor]=None, class_weights: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        z = self.extract_features(x)
        outputs = self.head(z)
        if targets is not None:
            outputs['loss'] = self.head.loss_fn(outputs, targets, class_weights=class_weights)
        return outputs

# %%
"""Notebook-native P0 inference. All architecture definitions precede this cell."""
import gc
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def relative_image_path(value):
    parts = PurePosixPath(str(value).replace('\\', '/')).parts
    lower = [p.lower() for p in parts]
    if 'native_cropped' in lower:
        parts = parts[lower.index('native_cropped') + 1:]
    if not parts or '..' in parts or any(':' in p for p in parts) or parts[0] == '/':
        raise ValueError(f'Unsafe/unresolved image path: {value}')
    return PurePosixPath(*parts).as_posix()


def read_manifest(path, images_root, patient_map=None):
    df = pd.read_csv(path, dtype={'image_id': str, 'patient_id': str})
    required = ['image_id', 'diagnosis', 'dataset_key', 'native_path', 'patient_id']
    if any(c not in df for c in required) or df[required].isna().any().any():
        raise ValueError(f'Incomplete manifest: {path}')
    if not df.diagnosis.isin(range(5)).all():
        raise ValueError('Invalid grade')
    df['original_relative_path'] = df.native_path.map(relative_image_path)
    df['image_key'] = df.dataset_key.astype(str) + '::' + df.image_id.astype(str)
    if df.image_key.duplicated().any():
        raise ValueError('Duplicate image identity')
    df['patient_key'] = df.patient_id.astype(str)
    df['patient_identity_status'] = 'legacy_unverified'
    # Never claim a legacy date/string key is a verified patient. An explicit map is required.
    if patient_map is not None:
        mapping = pd.read_csv(patient_map, dtype=str)
        if mapping.image_key.duplicated().any() or mapping.patient_key.isna().any():
            raise ValueError('Invalid verified patient mapping')
        mapping = mapping.set_index('image_key').patient_key
        known = df.image_key.isin(mapping.index)
        df.loc[known, 'patient_key'] = df.loc[known, 'image_key'].map(mapping)
        df.loc[known, 'patient_identity_status'] = 'verified'
    df['runtime_path'] = [str(Path(images_root).joinpath(*PurePosixPath(p).parts))
                          for p in df.original_relative_path]
    missing = [p for p in df.runtime_path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing images; first: {missing[:3]}')
    return df


def find_unique_root(extracted, required_relpaths):
    extracted = Path(extracted)
    probe = PurePosixPath(required_relpaths[0])
    candidates = []
    for p in extracted.rglob(probe.name):
        root = p
        for _ in probe.parts:
            root = root.parent
        if all(root.joinpath(*PurePosixPath(rel).parts).is_file() for rel in required_relpaths):
            candidates.append(root)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError(f'Expected one archive root, got {candidates}; set explicit root override')
    return candidates[0]


def resolve_assets(manifest_root, data_extract, ckpt_extract, config):
    frames = {}
    for fold in config['folds']:
        cohort = COHORTS[fold]
        for split, suffix in [('source_train', 'train'), ('source_val', 'val'), ('target', 'test')]:
            path = Path(manifest_root) / f'fold_{fold}_{cohort}_{suffix}.csv'
            frame = pd.read_csv(path, dtype={'image_id': str, 'patient_id': str})
            frames[(fold, split)] = (path, frame)
        tr, va, te = [frames[(fold, s)][1] for s in ['source_train', 'source_val', 'target']]
        for a, b in [(tr, va), (tr, te), (va, te)]:
            ka = set(a.dataset_key.astype(str) + '::' + a.image_id.astype(str))
            kb = set(b.dataset_key.astype(str) + '::' + b.image_id.astype(str))
            if ka & kb or set(a.patient_id) & set(b.patient_id):
                raise ValueError(f'Image/patient overlap in fold {fold}')
        if set(te.fold_key) != {cohort} or cohort in set(tr.fold_key) or cohort in set(va.fold_key):
            raise ValueError('Wrong held-out cohort mapping')
    required = sorted(set(relative_image_path(p) for _, df in frames.values() for p in df.native_path))
    image_root = Path(config['images_root_override']) if config.get('images_root_override') else find_unique_root(data_extract, required)
    names = [f'lodo_{VARIANT_NAMES[v]}_fold_{f}_{COHORTS[f]}_best.pth'
             for f in config['folds'] for v in config['variants']]
    checkpoint_root = (Path(config['checkpoint_root_override']) if config.get('checkpoint_root_override')
                       else find_unique_root(ckpt_extract, names))
    resolved = {}
    for (fold, split), (path, _) in frames.items():
        frame = read_manifest(path, image_root, config.get('verified_patient_map'))
        if config['max_samples']:
            frame = frame.iloc[:config['max_samples']].copy()
        resolved[(fold, split)] = (path, frame)
    return image_root, checkpoint_root, resolved


def prepare_image(row, pooled=False):
    bgr = cv2.imread(str(row.runtime_path))
    if bgr is None:
        raise ValueError(f'Cannot decode {row.runtime_path}')
    raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    processed = apply_green_channel_ablation(raw)
    glob = torch.from_numpy(cv2.resize(processed, (512, 512), interpolation=cv2.INTER_LINEAR)).permute(2, 0, 1).contiguous()
    if pooled:
        return glob, None
    mask = compute_fov_mask(raw, erode_px=15)
    anchors = compute_valid_anchors(raw.shape[:2], mask, patch_size=128, stride=96, fov_thresh=.5)
    patches = extract_mil_patches_sliding(processed, valid_anchors=anchors, mode='eval', max_patches_eval=256)
    return glob, patches


def normalize_u8(t, device):
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    return (t.to(device=device, dtype=torch.float32) / 255 - mean) / std


def pooled_bag_from_embeddings(branch, embeddings, mask=None):
    masked = embeddings if mask is None else embeddings.masked_fill(~mask.unsqueeze(-1), -1e4)
    maximum = masked.max(1).values
    return torch.cat([maximum, branch.gated_pool(embeddings, patch_mask=mask)], 1) if branch.use_dual_pooling else maximum


@torch.inference_mode()
def forward_image(model, global_u8, local_u8, device, chunk_size=32, pooled=False):
    glob = normalize_u8(global_u8.unsqueeze(0), device)
    if pooled:
        z = model.extract_features(glob)
    else:
        backbone = model.backbone
        fg = backbone.global_branch(glob)
        branch = backbone.local_mil_branch
        encoded = []
        for chunk in local_u8.split(chunk_size):
            # Only a bounded patch chunk is normalized/encoded on GPU.
            encoded.append(branch.gap(branch.features(normalize_u8(chunk, device))).flatten(1))
        embeddings = torch.cat(encoded).unsqueeze(0)
        fl = pooled_bag_from_embeddings(branch, embeddings)
        z = backbone.fusion_layer(torch.cat([fg, fl], 1))
    out = model.head(z)
    out['latent'] = z
    if 'severity_score' in out:
        torch.testing.assert_close(out['severity_score'].reshape(-1), model.head.score_fc(z).reshape(-1))
    return out


def frozen_model(checkpoint_path, variant, device, pooled=False):
    model = (PooledResNet50AblationModel(head_type=variant, pretrained=False) if pooled else
             build_model(head_type='CLM_QWK' if variant == 'V3' else 'Softmax_CE', pretrained=False,
                         use_cbam=True, use_local_branch=True, use_nonlocal=True,
                         use_quadrant_tokens=True, fusion_dim=512))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    model.eval().requires_grad_(False).to(device)
    epoch = checkpoint.get('epoch')
    del checkpoint
    return model, epoch


def prepared_rows(piece, pooled, workers):
    """Prepare a bounded number of images on CPU, yielding results in manifest order."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    if workers == 1:
        for index, row in piece.iterrows():
            yield index, row, prepare_image(row, pooled=pooled)
        return
    rows = iter(piece.iterrows())
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def enqueue():
            item = next(rows, None)
            if item is not None:
                index, row = item
                pending.append((index, row, executor.submit(prepare_image, row, pooled=pooled)))
        for _ in range(workers):
            enqueue()
        while pending:
            index, row, future = pending.popleft()
            arrays = future.result()
            enqueue()
            yield index, row, arrays


def infer_split(model, frame, destination, identity, device, config, pooled=False):
    destination = Path(destination)
    if (destination / 'COMPLETE.json').exists():
        meta = verify_artifact(destination)
        if meta['identity'] != identity:
            raise ValueError(f'Resume provenance mismatch: {destination}; use a new RUN_ID')
        if meta['image_keys'] != frame.image_key.tolist():
            raise ValueError('Resume manifest alignment mismatch')
        print('Resume verified:', destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    # Small completed parts survive a disconnect during a long source_train split.
    for start in tqdm(range(0, len(frame), config['shard_size']), desc=destination.parent.name + '/' + destination.name):
        piece = frame.iloc[start:start + config['shard_size']]
        part = destination / 'parts' / f'{start:07d}'
        part_identity = dict(identity, row_start=start, count=len(piece))
        if (part / 'COMPLETE.json').exists():
            old = verify_artifact(part)
            if old['identity'] != part_identity or old['image_keys'] != piece.image_key.tolist():
                raise ValueError('Part cache mismatch')
            continue
        part.mkdir(parents=True, exist_ok=True)
        records, features = [], []
        for row_index, row, (global_u8, local_u8) in prepared_rows(piece, pooled, PREPROCESS_WORKERS):
            out = forward_image(model, global_u8, local_u8, device, config['patch_chunk_size'], pooled=pooled)
            p = out['probs'].cpu().numpy()[0]
            validate_probs(p[None, :])
            record = dict(protocol_id='pooled_head_benchmark' if pooled else 'lodo',
                          run_id=config['run_id'], fold_id=identity['fold'], held_out_key=identity['cohort'],
                          cohort=row.fold_key if 'fold_key' in row else row.dataset_key,
                          split_role=identity['split'], image_id=str(row.image_id), image_key=row.image_key,
                          patient_key=row.patient_key, patient_identity_status=row.patient_identity_status,
                          y_true=int(row.diagnosis), manifest_row_index=int(row_index),
                          original_relative_path=row.original_relative_path, variant=identity['variant'],
                          training_seed='unknown_checkpoint_metadata', inference_seed=42,
                          checkpoint_path=identity['checkpoint_path'], ckpt_sha256=identity['checkpoint_sha256'],
                          source_manifest_sha256=identity['manifest_sha256'], code_sha256=identity['code_sha256'],
                          forward_implementation_id=identity['forward_id'], precision_id='fp32_tf32_off',
                          preprocessing_id='green_bilateral_bilinear512' if pooled else 'main_lodo_raw_fov_green_bilinear_eval256_v1',
                          expected_grade=float(out.get('expected_grade', (out['probs'] * torch.arange(5, device=device)).sum(1))[0]),
                          pred_native=int(out['preds'][0]), epoch=identity['epoch'],
                          s=float(out['severity_score'][0]) if 'severity_score' in out else np.nan,
                          feature_file='z.npy', feature_row=int(row_index), feature_dim=512,
                          feature_tap='head_input_fusion512' if not pooled else 'head_input_projection512')
            record.update({f'p{k}': float(p[k]) for k in range(5)})
            theta = out.get('cutpoints')
            record.update({f'theta{k}': float(theta[k]) if theta is not None else np.nan for k in range(4)})
            # Preserve raw ordinal logits/ranks before any legacy clamp/normalization.
            for key in ['logits', 'rank_logits', 'rank_probs', 'cum_probs', 'cond_logits', 'cond_probs']:
                if key in out:
                    record[key + '_raw'] = json.dumps(out[key].detach().cpu().numpy()[0].tolist())
            if pooled and identity['variant'] in ['CORAL', 'CORN']:
                rank = out['rank_probs'].cpu().numpy()[0]
                raw_pmf = np.r_[1-rank[0], rank[:-1]-rank[1:], rank[-1]]
                record['raw_rank_monotone'] = bool(np.all(np.diff(rank) <= 0))
                record['raw_pmf'] = json.dumps(raw_pmf.tolist())
                record['posterior_postprocessing'] = 'historical_relu_adjacent_difference_normalize1e7'
            records.append(record)
            features.append(out['latent'].cpu().numpy()[0])
        pd.DataFrame(records).to_csv(part / 'predictions.csv', index=False)
        np.save(part / 'z.npy', np.asarray(features, np.float32))
        complete_directory(part, part_identity, piece.image_key.tolist())
        snapshot_directory(part, Path(config['drive_run']) / destination.relative_to(config['local_run']) / 'parts' / part.name)
    parts = sorted((destination / 'parts').iterdir())
    z = np.lib.format.open_memmap(destination / 'z.npy', mode='w+', dtype=np.float32, shape=(len(frame), 512))
    offset = 0
    dfs = []
    for part in parts:
        verify_artifact(part)
        f = np.load(part / 'z.npy', allow_pickle=False)
        z[offset:offset + len(f)] = f
        offset += len(f)
        dfs.append(pd.read_csv(part / 'predictions.csv', dtype={'image_id': str, 'image_key': str, 'patient_key': str}))
    z.flush()
    del z
    df = pd.concat(dfs, ignore_index=True)
    if offset != len(frame) or df.image_key.tolist() != frame.image_key.tolist() or not np.array_equal(df.y_true, frame.diagnosis):
        raise ValueError('Forward-to-manifest identity/label alignment failure')
    df.to_csv(destination / 'predictions.csv', index=False)
    complete_directory(destination, identity, frame.image_key.tolist())
    snapshot_directory(destination, Path(config['drive_run']) / destination.relative_to(config['local_run']))


def snapshot_directory(source, destination):
    source, destination = Path(source), Path(destination)
    meta = verify_artifact(source)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / 'COMPLETE.json'
    if final.exists():
        final.unlink()  # Only this derived snapshot's completion flag, never inputs.
    for name, digest in meta['files'].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256_file(target) != digest:
            temporary = target.with_name(target.name + '.copying')
            shutil.copy2(source / name, temporary)
            if sha256_file(temporary) != digest:
                raise IOError(f'Drive copy failed: {name}')
            os.replace(temporary, target)
    shutil.copy2(source / 'COMPLETE.json', final)


def restore_p0(drive_run, local_run):
    source = Path(drive_run)
    if not source.exists():
        return
    for branch in ['lodo', 'pooled']:
        # Include completed parts even when the enclosing split was interrupted.
        for marker in sorted((source / branch).rglob('COMPLETE.json')):
            directory = marker.parent
            snapshot_directory(directory, Path(local_run) / directory.relative_to(source))


def g0_table(p0_run, reference_path, config):
    reference = json.loads(Path(reference_path).read_text('utf-8'))
    rows = []
    mapping = {'qwk': ('in_domain_val_qwk', 'ood_test_qwk'),
               'within1': ('val_within_1_grade_acc', 'ood_within_1_grade_acc'),
               'rdr_auroc': ('val_referable_auc', 'ood_referable_auc')}
    for fold in config['folds']:
        for variant in config['variants']:
            ref = reference[VARIANT_NAMES[variant]][COHORTS[fold]]
            for split in ['source_val', 'target']:
                df, meta = load_shard(p0_run, variant, fold, split)
                metrics = {**decision_metrics(df.y_true, df.pred_native), **posterior_metrics(df.y_true, df[PROB_COLS])}
                index = int(split == 'target')
                for metric, keys in mapping.items():
                    expected = ref.get(keys[index])
                    actual = metrics[metric]
                    passed = expected is not None and np.isfinite(actual) and abs(actual - expected) <= .005
                    rows.append(dict(fold_id=fold, cohort=COHORTS[fold], variant=variant, split=split,
                                     metric=metric, actual=actual, reference=expected,
                                     tolerance=.005, status='DIAGNOSTIC' if config['max_samples'] else
                                     ('PASS' if passed else 'NOT_COMPARABLE' if expected is None else 'DISCREPANCY')))
                for grade in range(5):
                    expected = ref.get('ood_per_grade_recall' if index else 'val_per_grade_recall', {}).get(f'grade_{grade}')
                    support = ref.get('ood_per_grade_support' if index else 'val_per_grade_support', {}).get(f'grade_{grade}')
                    actual = metrics[f'recall_g{grade}']
                    passed = expected is not None and np.isfinite(actual) and abs(actual - expected) <= .005 and (support is None or support == metrics[f'support_g{grade}'])
                    rows.append(dict(fold_id=fold, cohort=COHORTS[fold], variant=variant, split=split,
                                     metric=f'recall_g{grade}', actual=actual, reference=expected, tolerance=.005,
                                     support=metrics[f'support_g{grade}'], reference_support=support,
                                     status='DIAGNOSTIC' if config['max_samples'] else 'PASS' if passed else 'DISCREPANCY'))
    table = pd.DataFrame(rows)
    table.to_csv(Path(p0_run) / 'g0_comparison.csv', index=False)
    status = ('BASELINE_REPRODUCED' if len(config['folds']) == 6 and set(config['variants']) == {'V2', 'V3'}
              and set(table.status) == {'PASS'} else 'DIAGNOSTIC')
    save_json(Path(p0_run) / 'G0_STATUS.json', dict(status=status, folds=config['folds'],
              reference_sha256=sha256_file(reference_path), note='No target tuning; missing references cannot pass'))
    return table


def run_p0_notebook(config, manifest_root, data_extract, ckpt_extract, reference_path):
    device = torch.device(config['device'] if config['device'] != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)
    run = Path(config['local_run'])
    run.mkdir(parents=True, exist_ok=True)
    restore_p0(config['drive_run'], run)
    image_root, checkpoint_root, frames = resolve_assets(manifest_root, data_extract, ckpt_extract, config)
    save_json(run / 'resolved_assets.json', dict(images_root=str(image_root), checkpoint_root=str(checkpoint_root),
                                               manifest_counts={str(k): len(v[1]) for k, v in frames.items()}))
    started = time.perf_counter()
    for variant in config['variants']:
        for fold in config['folds']:
            cohort = COHORTS[fold]
            path = checkpoint_root / f'lodo_{VARIANT_NAMES[variant]}_fold_{fold}_{cohort}_best.pth'
            digest = sha256_file(path)
            model, epoch = frozen_model(path, variant, device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            try:
                for split in config['splits']:
                    manifest_path, frame = frames[(fold, split)]
                    identity = dict(checkpoint_path=str(path), checkpoint_sha256=digest, epoch=epoch,
                                    manifest_sha256=sha256_file(manifest_path), runtime_manifest_sha256=object_hash(frame.to_dict('records')),
                                    code_sha256=NOTEBOOK_CODE_SHA256, variant=variant, fold=fold, cohort=cohort,
                                    split=split, forward_id='main_lodo_survival_fp32_clamp20_expm1_norm1e12',
                                    patch_cap=256, patch_chunk=config['patch_chunk_size'], precision='fp32_tf32_off',
                                    diagnostic_limit=config['max_samples'],
                                    data_archive_sha256=config['data_archive_sha256'],
                                    image_inputs=object_hash([(row.original_relative_path, Path(row.runtime_path).stat().st_size)
                                                             for row in frame.itertuples()]))
                    destination = run / 'lodo' / f'{variant}_fold_{fold}_{cohort}' / split
                    infer_split(model, frame, destination, identity, device, config)
            finally:
                del model
                gc.collect()
                if device.type == 'cuda':
                    save_json(run / f'compute_{variant}_{fold}.json', dict(peak_allocated=torch.cuda.max_memory_allocated(),
                              peak_reserved=torch.cuda.max_memory_reserved(), elapsed_seconds=time.perf_counter() - started))
                    torch.cuda.empty_cache()
    if {'source_val', 'target'} <= set(config['splits']):
        table = g0_table(run, reference_path, config)
        display(table)
        for name in ['g0_comparison.csv', 'G0_STATUS.json', 'resolved_assets.json']:
            shutil.copy2(run / name, Path(config['drive_run']) / name)
        if set(config['variants']) == {'V2', 'V3'}:
            historical_g3_replay(run, config)
            for name in ['dmip_g3_paired_image_bootstrap.csv', 'dmip_g3_bootstrap_macro.json']:
                shutil.copy2(run / name, Path(config['drive_run']) / name)
    return run


def historical_g3_replay(run, config, reps=2000, seed=42):
    """Historical image-level replay, separate from MLWA patient-cluster inference."""
    rng = np.random.default_rng(seed)
    rows, replicates = [], []
    for fold in config['folds']:
        a, _ = load_shard(run, 'V2', fold, 'target')
        b, _ = load_shard(run, 'V3', fold, 'target')
        if not a[['image_key', 'y_true']].equals(b[['image_key', 'y_true']]):
            raise ValueError('Historical paired-image alignment failure')
        ix = np.flatnonzero(a.y_true.to_numpy() == 3)
        if not len(ix):
            rows.append(dict(fold_id=fold, cohort=COHORTS[fold], status='NO_TRUE_G3'))
            continue
        correct_a = (a.pred_native.to_numpy()[ix] == 3).astype(float)
        correct_b = (b.pred_native.to_numpy()[ix] == 3).astype(float)
        draws = np.empty((reps, 2))
        for replicate in range(reps):
            sample = rng.integers(len(ix), size=len(ix))
            draws[replicate] = correct_a[sample].mean(), correct_b[sample].mean()
        replicates.append(draws)
        delta = draws[:, 1] - draws[:, 0]
        rows.append(dict(fold_id=fold, cohort=COHORTS[fold], status='HISTORICAL_IMAGE_LEVEL', true_g3=len(ix),
                         v2_recall=correct_a.mean(), v3_recall=correct_b.mean(),
                         v3_minus_v2=float(correct_b.mean()-correct_a.mean()),
                         v3_minus_v2_lower=float(np.quantile(delta, .025)),
                         v3_minus_v2_upper=float(np.quantile(delta, .975)), reps=reps, seed=seed))
    pd.DataFrame(rows).to_csv(Path(run) / 'dmip_g3_paired_image_bootstrap.csv', index=False)
    macro = np.mean(replicates, axis=0) if replicates else None
    save_json(Path(run) / 'dmip_g3_bootstrap_macro.json', dict(
        aggregation=f'equal-cohort macro-{len(replicates)} available', coverage=len(replicates),
        v3_minus_v2_lower=float(np.quantile(macro[:, 1]-macro[:, 0], .025)) if macro is not None else None,
        v3_minus_v2_upper=float(np.quantile(macro[:, 1]-macro[:, 0], .975)) if macro is not None else None,
        unit='paired true-G3 images within cohort', reps=reps, seed=seed,
        limitation='Historical image bootstrap; not patient CI or training-seed variability'))


def run_pooled_branch(config, image_root, checkpoint_root=None, mapping=None, manifests=None):
    """Independent pooled benchmark. Exact checkpoint/split hashes must be supplied, never guessed."""
    run = Path(config['local_run'])
    expected = ['Softmax_CE', 'Softmax_QWK', 'CORAL', 'CORN', 'CLM_QWK']
    candidates = []
    if checkpoint_root and Path(checkpoint_root).exists():
        for path in sorted(Path(checkpoint_root).rglob('*.pth')):
            candidates.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path)))
    status = dict(status='PENDING_ASSETS', candidates=candidates,
                  required_heads=expected, required_constructor='main_head_ablation_stem_proj_v1',
                  reason='Configure exact head->path+sha256 map AND historical pooled train/val/test manifests with hashes')
    status_path = run / 'pooled_asset_status.json'
    save_json(status_path, status)
    if not mapping or set(mapping) != set(expected) or not manifests or set(manifests) != {'source_train', 'source_val', 'target'}:
        print('Pooled branch PENDING_ASSETS. Inventory:', status_path)
        return status
    frames = {}
    for split, entry in manifests.items():
        if sha256_file(entry['path']) != entry['sha256']:
            raise ValueError('Pooled split manifest hash mismatch')
        frames[split] = read_manifest(entry['path'], image_root, config.get('verified_patient_map'))
    for a, b in [('source_train', 'source_val'), ('source_train', 'target'), ('source_val', 'target')]:
        if set(frames[a].image_key) & set(frames[b].image_key) or set(frames[a].patient_key) & set(frames[b].patient_key):
            raise ValueError('Pooled historical splits overlap')
    device = torch.device(config['device'] if config['device'] != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    for head, entry in mapping.items():
        if entry['model_implementation'] != 'main_head_ablation_stem_proj_v1' or entry['preprocessing'] != 'green_bilateral_bilinear512':
            raise ValueError('Pooled implementation/preprocessing not audited')
        path = Path(entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Pooled checkpoint hash mismatch')
        copied = Path(config['local_run']) / 'pooled_checkpoint_copies' / head / path.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
        if sha256_file(copied) != entry['sha256']:
            raise IOError('Pooled checkpoint copy hash mismatch')
        model, epoch = frozen_model(copied, head, device, pooled=True)
        try:
            for split, frame in frames.items():
                if config['max_samples']:
                    frame = frame.iloc[:config['max_samples']]
                identity = dict(checkpoint_path=str(path), checkpoint_sha256=entry['sha256'], epoch=epoch,
                                manifest_sha256=manifests[split]['sha256'], code_sha256=NOTEBOOK_CODE_SHA256,
                                variant=head, fold=-1, cohort='pooled', split=split,
                                forward_id='main_head_ablation_' + head, diagnostic_limit=config['max_samples'])
                infer_split(model, frame, run / 'pooled' / head / split, identity, device, config, pooled=True)
        finally:
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    status.update(status='INFERENCE_COMPLETE', reason='Strict-loaded explicitly mapped pooled checkpoints')
    save_json(status_path, status)
    return status

# %%
ASSET_HASHES = {'manifests/fold_0_aptos_test.csv': '8b850484cc598d705f9dea5c194fcad16037408cb495cce3d240058a8ffe8f17', 'manifests/fold_0_aptos_train.csv': '537bbbd73a85bafd3fd132c4911ff5ab42bf8cf3d1c5b5d2caed6dd8641c1d34', 'manifests/fold_0_aptos_val.csv': 'fb2b58806341923cff95b276453932ef3344dfa60ff290b983504064168ab589', 'manifests/fold_1_idrid_test.csv': '6bb97f569a7ed0f310c7eb29d5a3d5f66972433fe403cbbd8bcc54fd667942c2', 'manifests/fold_1_idrid_train.csv': 'a005e6a31a54b5462fb24c4ad5536cf876c2149fd1c9c2245852ef3ec2d8d300', 'manifests/fold_1_idrid_val.csv': '6a83c6478296bd0f65d1fb5d1c3477c80507ca067656366eee9e4c4da417faed', 'manifests/fold_2_messidor2_test.csv': 'd91ad2701f1bf52a6f1e778a0639285a62357622a5edd4bcb9541d1659a3753b', 'manifests/fold_2_messidor2_train.csv': '03eb9321d1b11fa9aa2325f03bf14aabb9a026316a5d4f11c2d1d0903b5d125a', 'manifests/fold_2_messidor2_val.csv': '75faa046252dbf1b0809151b26c5f184e576309cd662fe9c5369df2a0b3bb75d', 'manifests/fold_3_ddr_test.csv': '3fd43a2153554a2bdf0b1c4ebea916c1212cc3ced69e4bc5e0f79e9a72e99316', 'manifests/fold_3_ddr_train.csv': 'bca0b649dd5905e4833de45df01fe60f328ffaf53728bad28204ace17dedd6b6', 'manifests/fold_3_ddr_val.csv': 'e511e32947b49fafcd2ac07e3f28b0db2653a68dcff1ff249bfe929f5891e512', 'manifests/fold_4_eyepacs_test.csv': '1f76f3289b5dba1d2eea8b89493c66a5ed35a640bc2a7c88bf3aa87817d268ee', 'manifests/fold_4_eyepacs_train.csv': '5753e5546978d1ec8632f85ecf52a7db76e88e8316a3ae1a114395de1e0532ca', 'manifests/fold_4_eyepacs_val.csv': 'a82baf4f966dba87d41946236d79984d2ede4fc93398b74da991f28bb992f9ee', 'manifests/fold_5_deepdrid_test.csv': 'db0d244e12e258ad9c627ac9a329ca149e1f71dd427518dee26707c6af50f804', 'manifests/fold_5_deepdrid_train.csv': '8ea2980012b88e01433f25c0a4c9500129e65ef3c1eeca04e43eab3da917db8b', 'manifests/fold_5_deepdrid_val.csv': '9def9032d68eec45a212c416df781c6e2cf764ab1c269a7c4638102e0ecfbdc9', 'reference_summary.json': '92c0c7ac8b809bfb2fc8ea572687acd62d5b8cc3e91f9894b2c48b42399616b7'}
SOURCE_HASHES = {'notebooks\\dmip-conference\\diabetic_retinopathy\\pipeline\\model\\main_model_lodo.ipynb': 'e88f73279488e5c6f407c7757dc7643f3ae67f33bfc74b879467c2e6cf4a284d', 'notebooks\\dmip-conference\\diabetic_retinopathy\\pipeline\\model\\main_model_head_ablation.ipynb': '898e842061ccf41a6732867e619b9c67819e3a140efe8a2bbc78c851e0cd72a1', 'camera_ready_analysis\\camera_ready_analysis_colab_L4.ipynb': '63f8ddbacdd9104edc04bd76813b1e81768f46ab78a73bb7e2ce2fd093a8e284', 'documents\\G-12.10.26.pdf': 'baadd9834e1b3e005d1a3305fa8c5ea25aaf6d4a3fbf50b3672901d0f02b960a', 'paper-reference\\UEL___BaoHuy___DMIP_2026.pdf': '94a2d18a497596407d524642a5838ee7772f40838a0f6d87c6fb3ab27470c083'}
ANALYSIS_PLAN_SHA256 = 'b78c337ac3310748532665040d80350e7bf34408b45bb1da8e14bb9e44bac09b'

# %%
MANIFEST_ROOT = containing(stage_archive(FOLD_SOURCE, WORK / 'archives'), 'fold_0_aptos_train.csv')
for relpath, expected in ASSET_HASHES.items():
    path = MANIFEST_ROOT / Path(relpath).name if relpath.startswith('manifests/') else REFERENCE_PATH
    if sha256_file(path) != expected:
        raise ValueError(f'Historical asset digest mismatch: {path}')
DATA_EXTRACT = stage_archive(IMAGE_SOURCE, WORK / 'archives')
CHECKPOINT_EXTRACT = stage_archive(CHECKPOINT_SOURCE, WORK / 'archives')
if IMAGE_SOURCE.is_dir():
    # A direct image tree has no ZIP digest. Hash relative names and file contents in order.
    digest = hashlib.sha256()
    files = sorted(p for p in IMAGE_SOURCE.rglob('*') if p.is_file())
    for path in tqdm(files, desc='Fingerprinting image tree'):
        relative = path.relative_to(IMAGE_SOURCE).as_posix()
        digest.update(json.dumps([relative, sha256_file(path)], ensure_ascii=True).encode() + b'\n')
    config['data_archive_sha256'] = 'tree-sha256:' + digest.hexdigest()
else:
    config['data_archive_sha256'] = sha256_file(IMAGE_SOURCE)

# %%
from IPython.display import display
Path(config['drive_run']).mkdir(parents=True, exist_ok=True)
save_json(Path(config['drive_run']) / 'runtime_config.json', config)
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(Path(config['drive_run']) / 'environment_freeze.txt').write_text(freeze)
save_json(Path(config['drive_run']) / 'notebook_provenance.json', dict(code_sha256=NOTEBOOK_CODE_SHA256,
          plan_sha256=ANALYSIS_PLAN_SHA256, assets=ASSET_HASHES, source_notebooks=SOURCE_HASHES))
try:
    P0_RUN = run_p0_notebook(config, MANIFEST_ROOT, DATA_EXTRACT, CHECKPOINT_EXTRACT, REFERENCE_PATH)
    save_json(Path(config['drive_run']) / 'execution_status.json', dict(status='INFERENCE_COMPLETE', run=str(P0_RUN)))
except BaseException as error:
    save_json(Path(config['drive_run']) / 'execution_status.json', dict(status='INTERRUPTED_OR_FAILED', error=repr(error)))
    raise
print('Use this P0_RUN for E0 and E1:', config['drive_run'])

# %%
pooled_root = POOLED_HEAD_CHECKPOINT_ROOT
if RUN_POOLED:
    if POOLED_HEAD_CHECKPOINT_ZIP is not None:
        pooled_root = unpack_zip(POOLED_HEAD_CHECKPOINT_ZIP, LOCAL_ROOT / 'pooled_checkpoints_ssd')
    images_root = Path(json.loads((Path(config['local_run']) / 'resolved_assets.json').read_text())['images_root'])
    pooled_status = run_pooled_branch(config, images_root, pooled_root, POOLED_HEAD_MAP, POOLED_MANIFESTS)
else:
    pooled_status = dict(status='DISABLED_OPTIONAL_BRANCH',
        reason='Only the LODO protocol was requested.',
        lodo_affected=False)
save_json(Path(config['drive_run']) / 'pooled_asset_status.json', pooled_status)
display(pooled_status)
