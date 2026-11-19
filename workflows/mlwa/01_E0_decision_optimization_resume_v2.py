# %% [markdown]
# # E0 · Source-selected decision optimization

# %%
NOTEBOOK_CODE_SHA256 = '80940bc51a49dc5da4e5922bbde42cf4cca78b16e3605406dc3c0475bdd112dc'

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
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'threadpoolctl': 'threadpoolctl'}
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

# %% [markdown]
# Set P0_RUN to the completed P0 folder on Drive. Use a CPU runtime. Keep the same output ID, inputs, settings and library versions to resume. Completed units are checked before reuse.

# %%
# Replace this path with the completed P0 run from your Colab session.
P0_RUN = Path(os.environ.get('DR_P0_RUN', OUTPUTS / 'mlwa' / 'p0_thesis_fp32')).resolve()
if not (P0_RUN / 'lodo').is_dir():
    raise FileNotFoundError(f'Set P0_RUN to the existing run folder containing lodo/: {P0_RUN}')
FOLDS = list(range(6))
VARIANTS = ['V2', 'V3']
OUTPUT_ID = os.environ.get('DR_ANALYSIS_ID', 'e0_thesis_resume_v2')
LOCAL_ROOT = WORK / 'mlwa'
LOCAL_INPUT = LOCAL_ROOT / 'p0_input_cache' / OUTPUT_ID
LOCAL_OUTPUT = LOCAL_ROOT / OUTPUT_ID
DRIVE_OUTPUT = OUTPUTS / 'mlwa' / (P0_RUN.name + '_' + OUTPUT_ID)
CPU_THREADS = max(1, min(4, os.cpu_count() or 2))
BOOTSTRAP_REPS = 2000
BOOTSTRAP_BATCH = 50
STABILITY_REPS = 1000
RUN_LODO = True
RUN_POOLED_E0 = False

# %%
ANALYSIS_PLAN_SHA256 = '0fd996e6eb1c69851492b739abd2543237ff3b5beae1a8c5fb6ea8957ffea967'

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
"""Transactional CPU analysis cache. Embedded as readable notebook source."""
import sys
import platform
import importlib.metadata as package_metadata
from threadpoolctl import threadpool_limits


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value), encoding='utf-8')
    os.replace(temp, path)


def separate_paths(p0_run, *outputs):
    source = Path(p0_run).resolve()
    for output in outputs:
        target = Path(output).resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError('P0 input and E0/E1 output/cache must be separate directories')


def runtime_versions():
    return {name: package_metadata.version(name) for name in
            ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib']}


def array_hash(array):
    array=np.ascontiguousarray(array)
    h=hashlib.sha256()
    h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


def linear_state_hash(state):
    return object_hash({name:array_hash(array) for name,array in state.items()})


def checked_member(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Unsafe artifact member path')
    return path


def read_complete(root):
    root = Path(root)
    marker = root / 'COMPLETE.json'
    if not marker.is_file():
        raise ValueError(f'Incomplete artifact: {root}')
    meta = json.loads(marker.read_text('utf-8'))
    for name, digest in meta['files'].items():
        path = checked_member(root, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Corrupt completed artifact: {path}')
    return meta


def seal_unit(root, identity):
    root = Path(root)
    files = {p.name: sha256_file(p) for p in root.iterdir()
             if p.is_file() and p.name != 'COMPLETE.json' and not p.name.endswith(('.tmp', '.copying'))}
    write_json(root / 'COMPLETE.json', dict(identity=identity, files=files))


def transfer_complete(source, destination):
    """Copy only a completed unit; publish marker last. Never touch P0 inputs."""
    source, destination = Path(source), Path(destination)
    meta = read_complete(source)
    destination.mkdir(parents=True, exist_ok=True)
    flag = destination / 'COMPLETE.json'
    if flag.exists():
        old = read_complete(destination)
        if old['identity'] != meta['identity']:
            raise ValueError('Existing output identity differs; use a new OUTPUT_ID')
        if old['files'] == meta['files']:
            return
        flag.unlink()
    for name, digest in meta['files'].items():
        target = checked_member(destination, name)
        temporary = target.with_name(target.name + '.copying')
        shutil.copy2(checked_member(source, name), temporary)
        if sha256_file(temporary) != digest:
            raise IOError(f'Copy checksum mismatch: {name}')
        os.replace(temporary, target)
    write_json(flag, meta)


class AnalysisCache:
    """One writer per OUTPUT_ID. Resume individual fits, controls and bootstrap batches."""

    def __init__(self, local, drive, code, cpu_threads=2):
        self.local, self.drive = Path(local), Path(drive)
        self.code = code
        self.versions = runtime_versions()
        self.cpu_threads = int(cpu_threads)
        self.hits, self.computed = 0, 0
        self.local.mkdir(parents=True, exist_ok=True)
        self.drive.mkdir(parents=True, exist_ok=True)

    def unit(self, key, identity, producer):
        identity = dict(identity, code=self.code, versions=self.versions)
        local = checked_member(self.local / 'units', key)
        remote = checked_member(self.drive / 'units', key)
        if (remote / 'COMPLETE.json').exists() and not (local / 'COMPLETE.json').exists():
            transfer_complete(remote, local)
        if (local / 'COMPLETE.json').exists():
            marker = read_complete(local)
            if marker['identity'] != identity:
                raise ValueError(f'Cache inputs/config/version changed at {key}; use a new OUTPUT_ID')
            self.hits += 1
        else:
            local.mkdir(parents=True, exist_ok=True)
            # No completion marker until producer and both payload writes succeed.
            metadata, arrays = producer(local)
            write_json(local / 'result.json', metadata)
            temporary = local / 'arrays.npz.tmp'
            with open(temporary, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, local / 'arrays.npz')
            seal_unit(local, identity)
            self.computed += 1
        transfer_complete(local, remote)
        metadata = json.loads((local / 'result.json').read_text('utf-8'))
        with np.load(local / 'arrays.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return metadata, arrays, local

    def export_summary(self, key, identity, producer):
        meta, arrays, folder = self.unit(key, identity, producer)
        marker = read_complete(folder)
        # Only summary files, not all fitted models/features, are copied again.
        for name in marker['files']:
            if name in ['result.json', 'arrays.npz']:
                continue
            shutil.copy2(folder / name, self.local / name)
            shutil.copy2(folder / name, self.drive / name)
        result = dict(status='ANALYSIS_COMPLETE', summary_unit=key, summary_identity=identity,
                      cache_hits=self.hits, computed_units=self.computed,
                      complete_scope='requested folds only; consult coverage.csv', versions=self.versions)
        write_json(self.local / 'execution_status.json', result)
        write_json(self.drive / 'execution_status.json', result)
        write_json(self.drive / 'COMPLETE.json', dict(summary_unit=key,
                   summary_marker_sha256=sha256_file(folder / 'COMPLETE.json'), identity=identity))
        return meta


def stage_p0_inputs(source, destination, folds, variants, features, protocol='lodo'):
    """Copy only predictions/z to SSD; never rehash/copy P0's redundant part files.

    The source COMPLETE manifest supplies cryptographic digests for these files.
    A marker is read before and after copying to detect concurrent changes.
    """
    source, destination = Path(source), Path(destination)
    separate_paths(source, destination)
    required = ['predictions.csv'] + (['z.npy'] if features else [])
    evidence, model_ids = {}, {}
    for fold in folds:
        for variant in variants:
            identities = []
            for split in ['source_train', 'source_val', 'target']:
                relative = (Path('lodo') / f'{variant}_fold_{fold}_{COHORTS[fold]}' / split
                            if protocol=='lodo' else Path('pooled') / variant / split)
                src, dst = source / relative, destination / relative
                marker_path = src / 'COMPLETE.json'
                if not marker_path.exists():
                    raise ValueError(f'P0 split is not complete: {src}. Do not substitute old four-image smoke caches.')
                original = marker_path.read_bytes()
                marker_digest = hashlib.sha256(original).hexdigest()
                meta = json.loads(original)
                if any(name not in meta.get('files', {}) for name in required):
                    raise ValueError(f'P0 artifact lacks required predictions/features: {src}')
                identity = meta['identity']
                if identity.get('variant') != variant or identity.get('fold') != fold or identity.get('split') != split:
                    raise ValueError('Wrong P0 marker fold/variant/split')
                identities.append(identity)
                cached = dst / 'COMPLETE.json'
                ready=False
                if cached.exists():
                    old = read_complete(dst)
                    if old.get('source_marker_sha256') != marker_digest:
                        raise ValueError(f'P0 input changed: {src}; choose new input cache and OUTPUT_ID')
                    ready=all(name in old['files'] for name in required)
                if not ready:
                    dst.mkdir(parents=True, exist_ok=True)
                    for name in required:
                        target = dst / name
                        temporary = target.with_name(name + '.copying')
                        shutil.copy2(checked_member(src, name), temporary)
                        if sha256_file(temporary) != meta['files'][name]:
                            raise ValueError(f'P0 core file hash mismatch: {src / name}')
                        os.replace(temporary, target)
                    if marker_path.read_bytes() != original:
                        raise ValueError('P0 marker changed while copying; retry after the run is stable')
                    subset = dict(meta, files={name: meta['files'][name] for name in required},
                                  source_marker_sha256=marker_digest, source_path=str(src))
                    write_json(cached, subset)
                frame=pd.read_csv(dst/'predictions.csv',dtype={'image_key':str,'image_id':str,'patient_key':str})
                columns=['image_key','image_id','patient_key','patient_identity_status','y_true','variant','fold_id','split_role',*PROB_COLS]
                if any(c not in frame for c in columns) or frame[columns].isna().any().any():
                    raise ValueError(f'Missing P0 identities/labels/posteriors: {src}')
                if frame.image_key.duplicated().any() or frame.image_key.tolist()!=meta.get('image_keys'):
                    raise ValueError('P0 image identity/order mismatch')
                if not frame.y_true.isin(range(5)).all() or set(frame.variant)!={variant} or set(frame.fold_id)!={fold} or set(frame.split_role)!={split}:
                    raise ValueError('P0 invalid labels or wrong fold/variant/split rows')
                validate_probs(frame[PROB_COLS])
                if 'feature_row' in frame and not np.array_equal(frame.feature_row.to_numpy(),np.arange(len(frame))):
                    raise ValueError('P0 feature row references are misaligned')
                if features:
                    z=np.load(dst/'z.npy',mmap_mode='r',allow_pickle=False)
                    if z.shape!=(len(frame),512) or not np.isfinite(z).all():
                        raise ValueError('Invalid P0 fusion-z feature matrix')
                    del z
                    if variant=='V3' and ('s' not in frame or not np.isfinite(frame.s.to_numpy(float)).all()):
                        raise ValueError('E1 requires true V3 scalar scores')
                evidence[f'{variant}/{fold}/{split}'] = dict(source_marker_sha256=marker_digest,
                        identity=identity, files={name: meta['files'][name] for name in required})
            fields=['checkpoint_sha256','code_sha256','forward_id','diagnostic_limit']
            if protocol=='lodo':
                fields+=['precision','patch_cap','patch_chunk','data_archive_sha256']
            for field in fields:
                if any(field not in identity for identity in identities):
                    raise ValueError(f'Missing P0 provenance field: {field}')
                if len({object_hash(identity[field]) for identity in identities}) != 1:
                    raise ValueError(f'Mixed checkpoint/preprocess provenance across splits: {field}')
            model_ids[f'{variant}/{fold}'] = object_hash(identities[0])
    write_json(destination / 'input_inventory.json', evidence)
    return evidence


def cached_pair_ci(cache, key, df, a, b, metrics, reps, batch_size, identity, seed=42):
    """Same sampled patient multiplicity for a/b and every metric in a batch."""
    points = {metric: decision_metrics(df.y_true, a)[metric] - decision_metrics(df.y_true, b)[metric]
              for metric in metrics}
    if set(df.patient_identity_status.astype(str)) != {'verified'}:
        return [dict(metric=metric, difference=points[metric], lower=np.nan, upper=np.nan,
                     valid=0, invalid=reps, reps=reps, unit='patient_cluster', status='NOT_ESTIMABLE') for metric in metrics]
    groups = list(df.reset_index(drop=True).groupby('patient_key', sort=True).indices.values())
    y, a, b = df.y_true.to_numpy(int), np.asarray(a), np.asarray(b)
    blocks = []
    for start in range(0, reps, batch_size):
        count = min(batch_size, reps-start)
        def produce(folder, start=start, count=count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, start]))
            values = np.empty((count, len(metrics)))
            for j in range(count):
                ix = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
                ma, mb = decision_metrics(y[ix], a[ix]), decision_metrics(y[ix], b[ix])
                values[j] = [ma[metric] - mb[metric] for metric in metrics]
            return dict(start=start, count=count, seed=seed), dict(differences=values)
        _, arrays, _ = cache.unit(f'{key}/bootstrap_{start:05d}',
            dict(identity, start=start, count=count, metrics=metrics, seed=seed, unit='verified_patient_cluster'), produce)
        blocks.append(arrays['differences'])
    values = np.vstack(blocks)
    result = []
    for k, metric in enumerate(metrics):
        good = np.isfinite(values[:, k])
        estimable = good.mean() >= .95
        result.append(dict(metric=metric, difference=points[metric],
                      lower=float(np.quantile(values[good, k], .025)) if estimable else np.nan,
                      upper=float(np.quantile(values[good, k], .975)) if estimable else np.nan,
                      valid=int(good.sum()), invalid=int((~good).sum()), reps=reps, unit='patient_cluster',
                      status='ESTIMABLE' if estimable else 'INSUFFICIENT_VALID_REPLICATES'))
    return result

# %%
"""Source-selected decoding on immutable P0 posteriors; no torch/encoder dependency."""

import matplotlib.pyplot as plt

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

def export_figure(fig, stem):
    stem = Path(stem)
    fig.tight_layout()
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def save_confusion(y, pred, stem):
    cm = confusion_matrix(y, pred, labels=range(5))
    pd.DataFrame(cm, index=range(5), columns=range(5)).to_csv(str(stem) + '_counts.csv')
    normal = np.divide(cm, cm.sum(1, keepdims=True), out=np.full((5, 5), np.nan), where=cm.sum(1, keepdims=True) > 0)
    pd.DataFrame(normal, index=range(5), columns=range(5)).to_csv(str(stem) + '_row_normalized.csv')

def latent_figures(val, target, prior, selected, directory):
    if target.s.isna().any():
        return
    theta = target[[f'theta{k}' for k in range(4)]].iloc[0].to_numpy(float)
    # Actual learned theta per fold; broad enough to contain data and all tails.
    lo = min(val.s.min(), target.s.min(), theta[0] - 5)
    hi = max(val.s.max(), target.s.max(), theta[-1] + 5)
    s = np.linspace(lo, hi, 20001)
    u = np.clip(theta[None, :] - s[:, None], -20, 20)
    survival = np.exp(-np.exp(u))
    p = np.column_stack([-np.expm1(-np.exp(u[:, 0])), survival[:, :-1] - survival[:, 1:], survival[:, -1]])
    p /= p.sum(1, keepdims=True)
    frame = pd.DataFrame(p, columns=PROB_COLS)
    frame['s'] = s
    for k in range(4):
        frame[f'theta{k}'] = theta[k]
    frame['pred_native'] = np.rint(p @ np.arange(5)).clip(0, 4).astype(int)
    decoded = decode_all(frame, prior, selected)
    regions = pd.DataFrame(dict(s=s, **decoded))
    regions.to_csv(directory / 'decision_grid_float64_diagnostic.csv', index=False)
    # Transitions retain skipped grades/multiple intervals instead of assuming one interval each.
    transitions = []
    for rule, pred in decoded.items():
        cuts = np.r_[0, np.flatnonzero(np.diff(pred)) + 1, len(s)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            transitions.append(dict(rule=rule, grade=int(pred[a]), left=float(s[a]), right=float(s[b - 1]),
                                    resolution=float(s[1] - s[0]), analytic_forward='float64_clamp20_diagnostic'))
    pd.DataFrame(transitions).to_csv(directory / 'decision_intervals.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    image = np.stack(list(decoded.values()))
    ax.imshow(image, aspect='auto', extent=[lo, hi, len(decoded) - .5, -.5], interpolation='nearest', vmin=0, vmax=4, cmap='viridis')
    ax.set_yticks(range(len(decoded)), list(decoded))
    for boundary in theta:
        ax.axvline(boundary, color='white', linestyle='--', linewidth=.7)
    ax.set_xlabel('Learned latent score s; dashed = actual cutpoints')
    export_figure(fig, directory / 'decision_regions')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    bins = np.linspace(lo, hi, 65)
    counts = []
    for name, df in [('source_val', val), ('target', target)]:
        axes[0].hist(df.s, bins=bins, density=True, histtype='step', label=name)
        order = np.sort(df.s.to_numpy())
        axes[1].plot(order, np.arange(1, len(order) + 1) / len(order), label=name)
        for grade in range(5):
            hist, edges = np.histogram(df.loc[df.y_true == grade, 's'], bins=bins)
            counts.extend(dict(split=name, grade=grade, left=float(edges[k]), right=float(edges[k + 1]), count=int(v)) for k, v in enumerate(hist))
    for ax in axes:
        ax.set_xlabel('s'); ax.legend()
    axes[0].set_ylabel('Density'); axes[1].set_ylabel('ECDF')
    export_figure(fig, directory / 'latent_hist_ecdf')
    pd.DataFrame(counts).to_csv(directory / 'latent_histogram_counts.csv', index=False)

def selection_stability(val, prior, reps=1000, seed=42):
    """Source-only refits; legacy grouping is explicitly distinguished from verified patients."""
    rng = np.random.default_rng(seed)
    codes, keys = pd.factorize(val.patient_key, sort=True)
    parameters = [(float(alpha), tau, q) for alpha in np.arange(11) / 10 for tau in [0, .25, .5, .75, 1] for q in [1, 2]]
    predictions = np.stack([cibd(val[PROB_COLS], prior, a, t, q) for a, t, q in parameters] + [val.pred_native.to_numpy(int)])
    y = val.y_true.to_numpy(int)
    m = len(predictions)
    bin_ids = (25 * np.arange(m)[:, None] + 5 * y[None, :] + predictions).ravel()
    distances = abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    rows = []
    for replicate in tqdm(range(reps), desc='Source-only selector bootstrap'):
        cluster_weights = np.bincount(rng.integers(len(keys), size=len(keys)), minlength=len(keys))
        weights = cluster_weights[codes]
        cm = np.bincount(bin_ids, weights=np.tile(weights, m), minlength=25*m).reshape(m, 5, 5)
        support = cm.sum(2)
        if support[0, 3] == 0:
            rows.append(dict(replicate=replicate, status='NO_G3'))
            continue
        recall = np.divide(np.diagonal(cm, axis1=1, axis2=2), support, out=np.full((m, 5), np.nan), where=support > 0)
        total = cm.sum((1, 2))
        within = (cm * (distances <= 1)).sum((1, 2)) / total
        expected = support[:, :, None] * cm.sum(1)[:, None, :] / total[:, None, None]
        denominator = (expected * distances**2).sum((1, 2))
        qwk = 1 - np.divide((cm * distances**2).sum((1, 2)), denominator,
                            out=np.full(m, np.nan), where=denominator > 0)
        eligible = [i for i in range(len(parameters)) if within[i] >= within[-1] - .01]
        if eligible:
            i = max(eligible, key=lambda i: (recall[i, 3], np.mean(recall[i]), within[i], qwk[i],
                                              -parameters[i][1], -parameters[i][2], -parameters[i][0]))
            alpha, tau, q = parameters[i]
            chosen = dict(alpha=alpha, tau=tau, q=q, fallback=False)
        else:
            chosen = dict(fallback=True, reason='NO_FEASIBLE_CANDIDATE')
        rows.append(dict(replicate=replicate, status='OK', **chosen))
    return pd.DataFrame(rows), ('patient_cluster' if set(val.patient_identity_status) == {'verified'} else 'legacy_grouping_key')
"""E0 with per-fold, source-selection and bootstrap-batch resume."""


def run_e0_resumable(p0_run, local_input, local_output, drive_output, folds, variants,
                     bootstrap_reps=2000, stability_reps=1000, bootstrap_batch=50, cpu_threads=2,protocol='lodo'):
    separate_paths(p0_run, local_input, local_output, drive_output)
    if not folds or not variants:
        raise ValueError('Choose at least one fold and variant')
    evidence = stage_p0_inputs(p0_run, local_input, folds, variants, features=False,protocol=protocol)
    cache = AnalysisCache(local_output, drive_output, NOTEBOOK_CODE_SHA256, cpu_threads)
    all_rows, all_prob, all_ci, all_predictions = [], [], [], []
    coverage = []
    for fold in folds:
        for variant in variants:
            cohort=COHORTS[fold] if protocol=='lodo' else 'pooled'
            tag = f'{variant}_fold_{fold}_{cohort}'
            read_cache=(lambda split:load_shard(local_input,variant,fold,split)) if protocol=='lodo' else (lambda split:load_pooled_shard(local_input,variant,split))
            source_evidence = {s: evidence[f'{variant}/{fold}/{s}'] for s in ['source_train', 'source_val']}
            fold_identity = dict(stage='E0', inputs={s: evidence[f'{variant}/{fold}/{s}'] for s in
                                 ['source_train', 'source_val', 'target']}, bootstrap_reps=bootstrap_reps,
                                 stability_reps=stability_reps, bootstrap_batch=bootstrap_batch,
                                 plan=ANALYSIS_PLAN_SHA256)
            def produce_fold(folder):
                train, _ = read_cache('source_train')
                val, _ = read_cache('source_val')
                prior = prior_from_labels(train.y_true)
                def select_source(path):
                    chosen, grid = select_cibd(val, prior)
                    threshold = screening_threshold(val.y_true, val[PROB_COLS])
                    grid.to_csv(path / 'source_grid.csv', index=False)
                    return dict(selected=chosen, threshold=threshold, prior=prior.tolist(),
                                objective='G3; Within1>=native-.01; ties macro,within1,qwk,lower tau,q,alpha'), {}
                locked, _, selection_dir = cache.unit(tag + '/source_selection',
                           dict(source=source_evidence, smoothing=1, grid='alpha0:.1:1,tau0:.25:1,q1/2'), select_source)
                chosen, threshold = locked['selected'], locked['threshold']
                write_json(folder / 'selected_params.json', locked)
                shutil.copy2(selection_dir / 'source_grid.csv', folder / 'source_grid.csv')
                # Held-out labels are first used after the source-only config is frozen.
                target, _ = read_cache('target')
                rows, prob, uncertainty, saved_target = [], [], [], []
                for split, df in [('source_val', val), ('target', target)]:
                    predictions = decode_all(df, prior, chosen)
                    prob.append(dict(fold_id=fold, cohort=cohort, variant=variant, split=split,
                                **posterior_metrics(df.y_true, df[PROB_COLS]),
                                **{'screening_'+k: v for k,v in screening_metrics(df.y_true, df[PROB_COLS], threshold).items()}))
                    saved = df[['image_key','image_id','y_true','patient_key','patient_identity_status']].copy()
                    for rule, pred in predictions.items():
                        saved[rule] = pred
                        rows.append(dict(fold_id=fold, cohort=cohort, variant=variant,
                                    split=split, decoder=rule, **decision_metrics(df.y_true, pred)))
                        save_confusion(df.y_true, pred, folder / f'{split}_{rule}')
                    saved.to_csv(folder / f'{split}_decisions.csv', index=False)
                    if split == 'target':
                        saved_target = saved
                        ci = cached_pair_ci(cache, tag + '/cibd_minus_native', df,
                              predictions['cibd_selected'], predictions['native_replay'],
                              ['recall_g3','within1','qwk'], bootstrap_reps, bootstrap_batch,
                              dict(target=evidence[f'{variant}/{fold}/target'], selected=locked))
                        uncertainty.extend(dict(fold_id=fold, cohort=cohort, variant=variant, **r) for r in ci)
                grid = pd.read_csv(selection_dir / 'source_grid.csv')
                exploratory = []
                for candidate in grid.itertuples():
                    exploratory.append(dict(alpha=candidate.alpha, tau=candidate.tau, q=candidate.q,
                        **decision_metrics(target.y_true, cibd(target[PROB_COLS], prior, candidate.alpha, candidate.tau, candidate.q))))
                tg = pd.DataFrame(exploratory)
                tg.to_csv(folder / 'target_grid_EXPLORATORY.csv', index=False)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(grid.within1, grid.recall_g3, alpha=.4, label='source validation grid')
                ax.scatter(tg.within1, tg.recall_g3, alpha=.3, label='target grid (exploratory)')
                for split in ['source_val','target']:
                    for rule, marker in [('native_replay','X'),('cibd_selected','*')]:
                        row = next(r for r in rows if r['split']==split and r['decoder']==rule)
                        ax.scatter(row['within1'], row['recall_g3'], marker=marker, s=100, label=f'{split}/{rule}')
                ax.set(xlabel='Within-1 accuracy', ylabel='G3 recall'); ax.legend(fontsize=6)
                export_figure(fig, folder / 'pareto')
                latent_figures(val, target, prior, chosen, folder)
                stability_blocks = []
                for start in tqdm(range(0, stability_reps, bootstrap_batch), desc=tag+' source bootstrap'):
                    count = min(bootstrap_batch, stability_reps-start)
                    def stability_unit(path, start=start, count=count):
                        seed = int(np.random.SeedSequence([42,start]).generate_state(1)[0])
                        result, unit = selection_stability(val, prior, count, seed)
                        result['replicate'] += start
                        return dict(rows=result.to_dict('records'), unit=unit, seed=seed), {}
                    st, _, _ = cache.unit(tag+f'/selection_bootstrap_{start:05d}',
                        dict(source=source_evidence, start=start, count=count, prior=prior.tolist(), seed=42), stability_unit)
                    stability_blocks.extend(st['rows'])
                if stability_blocks:
                    st = pd.DataFrame(stability_blocks)
                    st.to_csv(folder / 'selection_bootstrap.csv', index=False)
                    columns = [c for c in ['alpha','tau','q','fallback','status'] if c in st]
                    freq = st.groupby(columns, dropna=False).size().reset_index(name='count')
                    freq['frequency'] = freq['count']/stability_reps
                    freq.to_csv(folder / 'selection_frequencies.csv', index=False)
                    write_json(folder / 'selection_bootstrap_metadata.json', dict(reps=stability_reps,
                        unit='patient_cluster' if set(val.patient_identity_status)=={'verified'} else 'legacy_grouping_key',
                        seed_policy='SeedSequence([42,batch_start]); fixed batch size',
                        entropy=float(-(freq.frequency*np.log(freq.frequency)).sum())))
                return dict(rows=rows, posterior=prob, uncertainty=uncertainty), {
                    'y_true': saved_target.y_true.to_numpy(int),
                    **{rule: saved_target[rule].to_numpy(int) for rule in decode_all(target, prior, chosen)}}
            result, predictions, _ = cache.unit(tag + '/fold_report', fold_identity, produce_fold)
            all_rows.extend(result['rows']); all_prob.extend(result['posterior']); all_ci.extend(result['uncertainty'])
            for decoder, pred in predictions.items():
                if decoder != 'y_true':
                    all_predictions.append(pd.DataFrame(dict(variant=variant, decoder=decoder,
                                                y_true=predictions['y_true'], pred=pred)))
            coverage.append(dict(fold_id=fold, cohort=cohort, variant=variant, status='ANALYSIS_COMPLETE'))
            print(f'E0 {tag}: complete; cache hits={cache.hits}, computed units={cache.computed}')
    summary_identity = dict(stage='E0', evidence=evidence, folds=folds, variants=variants,
                            bootstrap_reps=bootstrap_reps, stability_reps=stability_reps, bootstrap_batch=bootstrap_batch)
    def summary(folder):
        table = pd.DataFrame(all_rows)
        table.to_csv(folder/'decision_metrics.csv', index=False)
        pd.DataFrame(all_prob).to_csv(folder/'posterior_and_screening_metrics.csv', index=False)
        pd.DataFrame(all_ci).to_csv(folder/'paired_cibd_minus_native.csv', index=False)
        pd.DataFrame(coverage).to_csv(folder/'coverage.csv', index=False)
        if protocol=='lodo':
            aggregate_tables(table[table.split=='target'], folder, ['variant','decoder'])
        pooled = pd.concat(all_predictions, ignore_index=True)
        weighted = [dict(variant=v, decoder=d, aggregation='pooled image weighted', **decision_metrics(df.y_true,df.pred))
                    for (v,d),df in pooled.groupby(['variant','decoder'])]
        pd.DataFrame(weighted).to_csv(folder/'pooled_image_weighted_metrics.csv', index=False)
        fig,ax = plt.subplots(figsize=(9,4))
        table[table.split=='target'].groupby(['variant','decoder'])[[f'g3_to_{k}_count' for k in range(5)]].sum().plot.bar(stacked=True,ax=ax)
        ax.set_ylabel('True G3 images (pooled counts)');export_figure(fig,folder/'g3_destinations')
        return dict(rows=len(table), folds=folds, variants=variants), {}
    cache.export_summary('summary', summary_identity, summary)
    table=pd.DataFrame(all_rows)
    display(table[['fold_id','variant','split','decoder','qwk','within1','recall_g3']])
    return table, cache

# %% [markdown]
# Run the analysis. The G0 status records the historical replay comparison. Analysis completion alone does not establish benchmark reproduction.

# %%
from IPython.display import display
separate_paths(P0_RUN,LOCAL_INPUT,LOCAL_OUTPUT,DRIVE_OUTPUT)
assert len(FOLDS)==len(set(FOLDS)) and FOLDS and all(f in range(6) for f in FOLDS)
assert BOOTSTRAP_REPS>0 and BOOTSTRAP_BATCH>0 and CPU_THREADS>0
g0_path=P0_RUN/'G0_STATUS.json'
g0=json.loads(g0_path.read_text('utf-8')) if g0_path.exists() else {'status':'UNVERIFIED'}
print('P0 G0:',g0)
print('Interpret this run with the P0 G0 status printed above.')
session=dict(input=str(P0_RUN),folds=FOLDS,p0_g0=g0,versions=runtime_versions(),
             code_sha256=NOTEBOOK_CODE_SHA256,plan_sha256=ANALYSIS_PLAN_SHA256)
LOCAL_OUTPUT.mkdir(parents=True,exist_ok=True);DRIVE_OUTPUT.mkdir(parents=True,exist_ok=True)
write_json(LOCAL_OUTPUT/'input_status.json',session);write_json(DRIVE_OUTPUT/'input_status.json',session)
write_json(DRIVE_OUTPUT/'RUN_STATUS.json',dict(status='RUNNING',**session))
try:
    assert RUN_LODO or RUN_POOLED_E0, 'Select at least one protocol'
    if RUN_LODO:
        table,cache=run_e0_resumable(P0_RUN,LOCAL_INPUT/'lodo_inputs',LOCAL_OUTPUT/'lodo',DRIVE_OUTPUT/'lodo',
            FOLDS,VARIANTS,BOOTSTRAP_REPS,STABILITY_REPS,BOOTSTRAP_BATCH,CPU_THREADS)
    if RUN_POOLED_E0:
        pooled_table,pooled_cache=run_e0_resumable(P0_RUN,LOCAL_INPUT/'pooled_inputs',LOCAL_OUTPUT/'pooled',DRIVE_OUTPUT/'pooled',
            [-1],['Softmax_CE','Softmax_QWK','CORAL','CORN','CLM_QWK'],BOOTSTRAP_REPS,STABILITY_REPS,BOOTSTRAP_BATCH,CPU_THREADS,protocol='pooled')
except BaseException as error:
    # Completed units are already on Drive. Do not recursively copy partial folders or stale markers.
    write_json(DRIVE_OUTPUT/'RUN_STATUS.json',dict(status='INTERRUPTED_OR_FAILED',error=repr(error),**session))
    raise
write_json(DRIVE_OUTPUT/'RUN_STATUS.json',dict(status='REQUESTED_SCOPE_ANALYSIS_COMPLETE',**session))
print('Results:',DRIVE_OUTPUT)
print('Resume with the same configuration and output ID. Keep the units directory.')
