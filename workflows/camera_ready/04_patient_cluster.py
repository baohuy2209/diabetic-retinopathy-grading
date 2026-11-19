# %% [markdown]
# # Paired patient-cluster analysis of Grade 3 sensitivity

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
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib'}
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
SOURCE_MODE = 'raw_npz'  # raw_npz, prediction_csv, or fold_cache
INPUT_ROOT = Path(os.environ.get('DR_RAW_PREDS', INPUTS / 'raw_preds'))
PATIENT_MAPPINGS = {}  # cohort -> verified mapping CSV
ALLOW_MANIFEST_ORDER = os.environ.get('DR_ALLOW_MANIFEST_ORDER') == '1'
FOLDS = list(range(6))
OFFICIAL_PAIRING = None  # Optional Messidor-2 CSV/XLSX
LEFT_COLUMN = RIGHT_COLUMN = None
AUDIT_MESSIDOR_SPLITS = False

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
"""Paired patient-cluster G3 bootstrap from saved predictions only.

No torch imports, model loading, image access, inference fallback or input writes.
The estimand remains image-weighted G3 sensitivity; patients are the resampling unit.
"""

import hashlib
import inspect
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

CLUSTER_METHOD_VERSION = "paired_patient_cluster_g3_v1"
CLUSTER_COHORTS = ("aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid")
CLUSTER_NAMES = ("APTOS 2019", "IDRiD", "Messidor-2", "DDR", "EyePACS", "DeepDRiD")
CLUSTER_VARIANTS = {"v2": "V2_PlusGlobalContext_H2", "v3": "V3_PlusOrdinalHead_H3_Champion"}
CLUSTER_SCOPE = ("Paired patients with at least one ground-truth Grade-3 image are sampled "
    "with replacement within each fixed cohort; all their Grade-3 images are retained. "
    "Sensitivity remains image-weighted. Intervals do not quantify training-seed variability "
    "or sampling of new cohorts, and depend on the validity of the supplied patient grouping.")


def cluster_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_grade_arrays(y, p2, p3):
    arrays = [np.asarray(x) for x in (y, p2, p3)]
    if any(a.ndim != 1 or not len(a) for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError("Targets/predictions must be nonempty, aligned one-dimensional arrays")
    for a in arrays:
        if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all() or (a < 0).any() or (a > 4).any():
            raise ValueError("Targets/predictions must be integer grades 0..4")
    return tuple(a.astype(np.int64) for a in arrays)


def cluster_normalize_image_key(value):
    name = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    while re.search(r"\.(?:jpe?g|png|tiff?)$", name, flags=re.I):
        name = name.rsplit(".", 1)[0]
    return name.lower()


def make_messidor_pair_mapping(pairing_file, left_column, right_column, manifest_file,
                               output_file, sheet_name=0):
    """Convert an AUTHORITATIVE explicit left/right pairing table, never filename proximity.

    One anonymous exam ID is derived from each supplied pair. This groups paired eyes;
    it does not establish linkage between separate visits without additional metadata.
    """
    pairing_file, output_file = Path(pairing_file), Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite mapping: {output_file}")
    pairs = (pd.read_csv(pairing_file, dtype=str) if pairing_file.suffix.lower() == ".csv"
             else pd.read_excel(pairing_file, sheet_name=sheet_name, dtype=str))
    if not {left_column, right_column} <= set(pairs):
        raise ValueError(f"Choose actual pairing columns from {list(pairs.columns)}")
    lookup = {}
    for left, right in pairs[[left_column, right_column]].itertuples(index=False, name=None):
        if pd.isna(left) or pd.isna(right) or not str(left).strip() or not str(right).strip():
            raise ValueError("Pairing file contains an incomplete pair; inspect it before conversion")
        keys = [cluster_normalize_image_key(left), cluster_normalize_image_key(right)]
        if keys[0] == keys[1] or any(key in lookup for key in keys):
            raise ValueError("Duplicate/conflicting image in the pairing table")
        exam = hashlib.sha256("|".join(sorted(keys)).encode()).hexdigest()[:16]
        for key in keys:
            lookup[key] = "Messidor-2::exam_" + exam
    manifest = pd.read_csv(manifest_file, dtype={"image_id": str})
    keys = manifest.image_id.map(cluster_normalize_image_key)
    if not keys.is_unique:
        raise ValueError("Image-name normalization creates a collision in the manifest")
    missing = manifest.loc[~keys.isin(lookup), "image_id"].tolist()
    if missing:
        raise ValueError(f"Official pairs do not cover {len(missing)} manifest images; examples: {missing[:5]}")
    result = pd.DataFrame({"cohort": "messidor2", "image_id": manifest.image_id,
        "patient_id": keys.map(lookup), "grouping_source": "official_left_right_exam_pairing",
        "pairing_file_sha256": cluster_file_sha256(pairing_file)})
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def audit_messidor_mapping_in_saved_splits(mapping_file, folds_dir, output_file=None):
    """Audit EXISTING splits with corrected grouping; do not regenerate any partition."""
    mapping = pd.read_csv(mapping_file, dtype=str)
    if not {"image_id", "patient_id"} <= set(mapping) or not mapping.image_id.is_unique or mapping.patient_id.isna().any():
        raise ValueError("A complete, unique image-to-patient mapping is required")
    lookup = mapping.set_index("image_id").patient_id
    records = []
    for fold, cohort in enumerate(CLUSTER_COHORTS):
        groups, missing = {}, {}
        for split in ("train", "val", "test"):
            path = Path(folds_dir) / f"fold_{fold}_{cohort}_{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"All 18 saved manifests are needed for split audit: {path}")
            frame = pd.read_csv(path, dtype={"image_id": str})
            if "dataset_key" not in frame:
                raise ValueError("dataset_key is needed to identify Messidor-2 records")
            selected = frame.loc[frame.dataset_key.astype(str).str.lower().str.replace("-", "", regex=False).eq("messidor2")]
            ids = selected.image_id.map(lookup)
            missing[split] = int(ids.isna().sum())
            groups[split] = set(ids.dropna())
        overlaps = {a + "_" + b: len(groups[a] & groups[b])
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        records.append(dict(fold=fold, held_out=cohort,
            **{f"unmapped_{key}": value for key, value in missing.items()}, **overlaps,
            status="INCOMPLETE_MAPPING" if any(missing.values()) else "GROUP_OVERLAP_DETECTED" if any(overlaps.values()) else "NO_OVERLAP_IN_SUPPLIED_GROUPING"))
    result = pd.DataFrame(records)
    if output_file is not None:
        output_file = Path(output_file)
        if output_file.exists():
            raise FileExistsError(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
    return result


def cluster_attach_patient_ids(frame, manifest, cohort, mapping_file=None):
    """Join grouping as a post-hoc sidecar; leave prediction cache/manifests unchanged."""
    for col in ("image_id", "diagnosis", "patient_id"):
        if col not in manifest:
            raise ValueError(f"Missing manifest column {col}")
    if manifest.image_id.isna().any() or not manifest.image_id.is_unique:
        raise ValueError("Manifest image IDs must be unique and nonmissing")
    if not np.array_equal(frame.image_id.astype(str), manifest.image_id.astype(str)):
        raise ValueError("Ordered image IDs do not match the test manifest")
    if not np.array_equal(frame.diagnosis, manifest.diagnosis):
        raise ValueError("Ordered target labels do not match the test manifest")
    ids = manifest.patient_id.copy()
    source = "manifest_patient_id; biological identity not independently certified"
    if mapping_file is not None:
        mapping = pd.read_csv(mapping_file, dtype=str)
        if "cohort" in mapping:
            mapping = mapping.loc[mapping.cohort.str.lower().eq(cohort)]
        if not {"image_id", "patient_id"} <= set(mapping) or mapping.image_id.isna().any() or not mapping.image_id.is_unique:
            raise ValueError("Mapping must contain unique nonmissing image_id and patient_id columns")
        ids = manifest.image_id.map(mapping.set_index("image_id").patient_id)
        source = "supplied_mapping:" + cluster_file_sha256(mapping_file)
    if ids.isna().any() or ids.astype(str).str.strip().isin(("", "nan", "None", "unknown")).any():
        raise ValueError(f"Missing/placeholder patient IDs for {cohort}")
    ids = ids.astype(str).str.strip().to_numpy()
    date_groups = pd.Series(ids).str.fullmatch(r"(?:.*::)?(?:19|20)\d{6}").all()
    status = "NEEDS_PATIENT_MAPPING" if cohort == "messidor2" and date_groups else "GROUPING_AVAILABLE"
    y = frame.diagnosis.to_numpy()
    audit = dict(cohort=cohort, n_images=len(frame), n_patient_groups=len(np.unique(ids)),
        n_grade3_images=int((y == 3).sum()), n_grade3_patient_groups=len(np.unique(ids[y == 3])),
        grouping_source=source, grouping_status=status,
        note="Date-prefix groups are not verified patient identities" if status != "GROUPING_AVAILABLE" else
             "Group identity remains conditional on source metadata; singleton IDs do not prove independent patients")
    return ids, audit


def _cluster_prepare(y, p2, p3, patient_ids):
    y, p2, p3 = cluster_grade_arrays(y, p2, p3)
    ids = np.asarray(patient_ids, dtype=object)
    if ids.shape != y.shape or pd.isna(ids).any() or any(not str(i).strip() for i in ids):
        raise ValueError("Aligned nonmissing patient IDs are required; no image-level fallback")
    g3 = y == 3
    if not g3.any():
        raise ValueError("No ground-truth Grade-3 cases; sensitivity is undefined")
    a, b = (p2[g3] == 3).astype(np.float32), (p3[g3] == 3).astype(np.float32)
    unique, codes = np.unique(ids[g3].astype(str), return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Fewer than two Grade-3 patient clusters; uncertainty cannot be estimated")
    singleton = len(unique) == len(a)
    # Preserve archived image order/RNG behavior when every G3 patient has one image.
    if singleton:
        codes = np.arange(len(a))
    n = len(unique)
    return dict(a=a, b=b, n_clusters=n, singleton=singleton,
        counts=np.bincount(codes, minlength=n), correct2=np.bincount(codes, weights=a, minlength=n),
        correct3=np.bincount(codes, weights=b, minlength=n), n_patients=len(np.unique(ids.astype(str))))


def _cluster_draw(group, rng):
    sample = rng.integers(0, group["n_clusters"], size=group["n_clusters"])
    if group["singleton"]:
        return group["a"][sample].mean(), group["b"][sample].mean()
    # Exactly equivalent to concatenating every selected patient's G3 images,
    # including repeated patients and variable cluster sizes (ratio of totals).
    denominator = group["counts"][sample].sum()
    return (np.float32(group["correct2"][sample].sum() / denominator),
            np.float32(group["correct3"][sample].sum() / denominator))


def _cluster_summary(point2, point3, samples, reps, seed):
    values = np.asarray(samples, dtype=np.float64) * 100.0
    delta = values[:, 1] - values[:, 0]
    lo2, hi2 = np.percentile(values[:, 0], [2.5, 97.5])
    lo3, hi3 = np.percentile(values[:, 1], [2.5, 97.5])
    lod, hid = np.percentile(delta, [2.5, 97.5])
    return dict(sens_v2=float(point2) * 100, sens_v3=float(point3) * 100,
        diff_v3_minus_v2_pp=(float(point3) - float(point2)) * 100,
        v2_ci_low=float(lo2), v2_ci_high=float(hi2), v3_ci_low=float(lo3), v3_ci_high=float(hi3),
        diff_ci_low_pp=float(lod), diff_ci_high_pp=float(hid),
        diff_interval_excludes_zero=bool(hid < 0 or lod > 0), bootstrap_reps=reps, seed=seed)


def paired_patient_cluster_g3(y, p2, p3, patient_ids, reps=2000, seed=42):
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("Positive integer bootstrap replicates required")
    group = _cluster_prepare(y, p2, p3, patient_ids)
    rng = np.random.default_rng(seed)
    result = _cluster_summary(group["a"].mean(), group["b"].mean(),
        [_cluster_draw(group, rng) for _ in range(reps)], reps, seed)
    result.update(n_grade3=len(group["a"]), n_grade3_patients=group["n_clusters"],
        n_patients=group["n_patients"], singleton_equivalent_image=group["singleton"],
        resampling_unit="patient_cluster", estimand="image_weighted_grade3_sensitivity")
    return result


def macro_patient_cluster_g3(cohort_data, reps=2000, seed=42):
    if not cohort_data or not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("Cohorts and positive integer bootstrap replicates required")
    groups = [_cluster_prepare(d["y"], d["p2"], d["p3"], d["patient_ids"]) for d in cohort_data]
    rng, samples = np.random.default_rng(seed), []
    for _ in range(reps):
        draws = [_cluster_draw(g, rng) for g in groups]
        samples.append((np.mean([v[0] for v in draws]), np.mean([v[1] for v in draws])))
    result = _cluster_summary(np.mean([g["a"].mean() for g in groups]),
        np.mean([g["b"].mean() for g in groups]), samples, reps, seed)
    result.update(cohort=f"Macro equal cohorts ({len(groups)})", fold=-2,
        n_grade3=sum(len(g["a"]) for g in groups), n_grade3_patients=sum(g["n_clusters"] for g in groups),
        resampling_unit="patient_cluster_within_fixed_cohort_macro", estimand="equal_weight_cohort_macro")
    return result


def cluster_read_npz(raw_dir, cohort, manifest, allow_manifest_order=False):
    frame, inputs = manifest[["image_id", "diagnosis"]].copy(), []
    for key, variant in CLUSTER_VARIANTS.items():
        path = Path(raw_dir) / f"{variant}__{cohort}.npz"
        with np.load(path, allow_pickle=False) as z:
            y, pred, _ = cluster_grade_arrays(z["ood_targets"], z["ood_preds"], z["ood_preds"])
            if not np.array_equal(y, manifest.diagnosis):
                raise ValueError(f"Ordered archived targets differ from manifest: {path}")
            ids = z["ood_image_ids"] if "ood_image_ids" in z.files else z["image_ids"] if "image_ids" in z.files else None
            if ids is None and not allow_manifest_order:
                raise ValueError("Legacy arrays lack image IDs; documented unshuffled order must be acknowledged explicitly")
            if ids is not None and not np.array_equal(ids.astype(str), manifest.image_id):
                raise ValueError(f"Archived image IDs do not match: {path}")
            probs = z["ood_probs"]
            if probs.shape != (len(y), 5) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
                raise ValueError(f"Invalid probabilities: {path}")
            decoded = probs.argmax(1) if key == "v2" else np.round((probs * np.arange(5)).sum(1)).clip(0, 4)
            if not np.array_equal(decoded, pred):
                raise ValueError(f"Decoder and stored predictions disagree: {path}")
            frame["pred_" + key] = pred
            inputs.append(dict(file=str(path), sha256=cluster_file_sha256(path),
                pairing="archived_image_ids" if ids is not None else "documented_manifest_order_assumption"))
    return frame, inputs


def cluster_read_saved_csv(path, manifest, fingerprint=None):
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    if not {"image_id", "diagnosis", "pred_v2", "pred_v3"} <= set(frame):
        raise ValueError(f"Missing prediction columns: {path}")
    cluster_grade_arrays(frame.diagnosis, frame.pred_v2, frame.pred_v3)
    if not np.array_equal(frame.image_id, manifest.image_id) or not np.array_equal(frame.diagnosis, manifest.diagnosis):
        raise ValueError(f"Saved prediction row/target order differs from manifest: {path}")
    if fingerprint is not None:
        required = {"provenance_marker", "code_fingerprint", "checkpoint_v2_sha", "checkpoint_v3_sha", "manifest_sha", "run_id", "v3_expected_grade"}
        if not required <= set(frame) or not frame.provenance_marker.eq("STRICT_REAL_CHECKPOINT_INFERENCE").all():
            raise ValueError(f"Incomplete strict snapshot provenance: {path}")
        if not frame.code_fingerprint.eq(fingerprint).all():
            raise ValueError("Snapshot fingerprint differs on one or more prediction rows")
        for col in ("checkpoint_v2_sha", "checkpoint_v3_sha", "manifest_sha", "run_id"):
            if frame[col].isna().any() or frame[col].nunique() != 1:
                raise ValueError(f"Inconsistent snapshot provenance: {col}")
        expected = pd.to_numeric(frame.v3_expected_grade, errors="coerce").to_numpy()
        if not np.isfinite(expected).all() or (expected < 0).any() or (expected > 4).any() or not np.array_equal(np.round(expected).clip(0, 4), frame.pred_v3):
            raise ValueError("Snapshot expected-grade decoder mismatch")
    return frame


def cluster_read_fold_cache(cache_root, fold, cohort, manifest):
    name = f"fold_{fold}_{cohort}"
    parent = Path(cache_root) / name
    rejected = []
    for snapshot in sorted(parent.iterdir(), reverse=True) if parent.is_dir() else []:
        marker = snapshot / "complete.json"
        if not marker.is_file():
            continue
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            prediction_name = name + "_predictions.csv"
            if info["fold"] != fold or info.get("max_samples", 0) != 0 or info["rows"] != len(manifest):
                raise ValueError("Partial or wrong-fold snapshot")
            if prediction_name not in info["files_sha256"]:
                raise ValueError("Prediction CSV is not covered by the completion marker")
            for filename, digest in info["files_sha256"].items():
                path = (snapshot / filename).resolve()
                if not path.is_relative_to(snapshot.resolve()) or cluster_file_sha256(path) != digest:
                    raise ValueError(f"Snapshot file integrity failure: {filename}")
            path = snapshot / prediction_name
            frame = cluster_read_saved_csv(path, manifest, info["fingerprint"])
            return frame, [dict(file=str(path), sha256=cluster_file_sha256(path),
                complete_marker_sha256=cluster_file_sha256(marker), cached_inference_fingerprint=info["fingerprint"],
                source_run_ids=frame.run_id.unique().tolist(), pairing="ordered_cached_image_ids")]
        except (ValueError, KeyError, OSError, TypeError) as exc:
            rejected.append(f"{snapshot.name}: {exc}")
    raise FileNotFoundError(f"No complete, valid prediction cache for {name}. No inference was run. " + "; ".join(rejected[-3:]))


def run_patient_cluster_posthoc(config):
    mode, folds = config["source_mode"], config.get("folds", list(range(6)))
    if mode not in ("fold_cache", "raw_npz", "prediction_csv") or not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(6)):
        raise ValueError("Choose one supported prediction source and unique valid folds")
    reference_path = Path(config["reference_json"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rows, audits, checks, inputs, data, blocked = [], [], [], [], [], []
    for fold in sorted(folds):
        cohort = CLUSTER_COHORTS[fold]
        path = Path(config["folds_dir"]) / f"fold_{fold}_{cohort}_test.csv"
        manifest = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
        inputs.append(dict(file=str(path), sha256=cluster_file_sha256(path)))
        if mode == "raw_npz":
            frame, provenance = cluster_read_npz(config["input_root"], cohort, manifest, config.get("allow_manifest_order", False))
        elif mode == "fold_cache":
            frame, provenance = cluster_read_fold_cache(config["input_root"], fold, cohort, manifest)
        else:
            csv_path = Path(config["input_root"]) / f"fold_{fold}_{cohort}_predictions.csv"
            frame = cluster_read_saved_csv(csv_path, manifest)
            provenance = [dict(file=str(csv_path), sha256=cluster_file_sha256(csv_path), pairing="ordered_saved_image_ids",
                provenance_note="Plain CSV import does not independently certify checkpoint execution")]
        inputs.extend(provenance)
        y, p2, p3 = cluster_grade_arrays(frame.diagnosis, frame.pred_v2, frame.pred_v3)
        g3 = y == 3
        if not g3.any():
            raise ValueError(f"No Grade-3 cases in {cohort}")
        point = dict(cohort=CLUSTER_NAMES[fold], fold=fold, n_grade3=int(g3.sum()),
            sens_v2=float((p2[g3] == 3).astype(np.float32).mean()) * 100,
            sens_v3=float((p3[g3] == 3).astype(np.float32).mean()) * 100)
        point["diff_v3_minus_v2_pp"] = point["sens_v3"] - point["sens_v2"]
        for key, pred in (("v2", p2), ("v3", p3)):
            ref = reference[CLUSTER_VARIANTS[key]][cohort]
            if [ref["ood_per_grade_support"][f"grade_{i}"] for i in range(5)] != np.bincount(y, minlength=5).tolist():
                raise ValueError(f"Benchmark grade support differs: {cohort}/{key}")
            values = dict(qwk=float(cohen_kappa_score(y, pred, labels=range(5), weights="quadratic")),
                within1=float(np.mean(abs(y - pred) <= 1)) * 100, g3_sens=point["sens_" + key])
            expected = dict(qwk=float(ref["ood_test_qwk"]), within1=float(ref["ood_within_1_grade_acc"]) * 100,
                g3_sens=float(ref["ood_per_grade_recall"]["grade_3"]) * 100)
            for metric, value in values.items():
                tolerance = 1e-5 if mode == "raw_npz" else .005 if metric == "qwk" else 1 if metric == "within1" else 1.5 if key == "v2" else 2.5
                checks.append(dict(cohort=cohort, model=key, metric=metric, observed=value,
                    reference=expected[metric], difference=value-expected[metric], tolerance=tolerance,
                    status="PASS" if abs(value-expected[metric]) <= tolerance else "DISCREPANCY"))
        mapping = config.get("patient_mappings", {}).get(cohort)
        if mapping:
            inputs.append(dict(file=str(mapping), sha256=cluster_file_sha256(mapping)))
        ids, audit = cluster_attach_patient_ids(frame, manifest, cohort, mapping)
        audits.append(audit)
        if audit["grouping_status"] != "GROUPING_AVAILABLE" or audit["n_grade3_patient_groups"] < 2:
            reason = audit["note"] if audit["grouping_status"] != "GROUPING_AVAILABLE" else "Fewer than two Grade-3 clusters"
            blocked.append(dict(cohort=cohort, reason=reason))
            point.update(analysis_status="NEEDS_PATIENT_MAPPING", resampling_unit="not_estimated")
            rows.append(point)
            continue
        result = paired_patient_cluster_g3(y, p2, p3, ids, config["reps"], config["seed"])
        result.update(cohort=CLUSTER_NAMES[fold], fold=fold, analysis_status="ESTIMATED_CONDITIONAL_ON_GROUPING")
        rows.append(result)
        data.append(dict(y=y, p2=p2, p3=p3, patient_ids=ids))
    if not blocked:
        rows.append(macro_patient_cluster_g3(data, config["reps"], config["seed"]))
    else:
        rows.append(dict(cohort=f"Macro equal cohorts ({len(folds)})", fold=-2,
            sens_v2=float(np.mean([r["sens_v2"] for r in rows])),
            sens_v3=float(np.mean([r["sens_v3"] for r in rows])),
            diff_v3_minus_v2_pp=float(np.mean([r["diff_v3_minus_v2_pp"] for r in rows])),
            n_grade3=sum(r["n_grade3"] for r in rows), analysis_status="INCOMPLETE_PATIENT_MAPPING_NO_MACRO_CI",
            resampling_unit="not_estimated"))
    complete = set(folds) == set(range(6))
    matched = complete and len(checks) == 36 and all(c["status"] == "PASS" for c in checks)
    status = "READY_FOR_REVIEW" if complete and matched and not blocked else "DIAGNOSTIC"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = Path(config["output_root"]) / ("patient_cluster_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / f"grade3_patient_cluster_{status}.csv", index=False)
    table.round(1).to_csv(output / f"grade3_patient_cluster_display_1dp_{status}.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "patient_grouping_audit.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "reference_checks.csv", index=False)
    report = dict(status=status, source_mode=mode, method=CLUSTER_METHOD_VERSION, reps=config["reps"], seed=config["seed"],
        full_six_cohorts=complete, benchmark_point_checks_match=matched, blocked_cohorts=blocked,
        statistical_scope=CLUSTER_SCOPE, inputs=inputs,
        reference_sha256=cluster_file_sha256(reference_path), inference_executed=False,
        note="Grouping availability is not independent certification of patient identity. Historical diagnostic predictions remain diagnostic.",
        output_directory=str(output), packages={"numpy": np.__version__, "pandas": pd.__version__})
    try:
        statistic_sources = "\n".join(inspect.getsource(fn) for fn in (
            _cluster_prepare, _cluster_draw, _cluster_summary, paired_patient_cluster_g3, macro_patient_cluster_g3))
        report["statistics_code_sha256"] = hashlib.sha256(statistic_sources.encode()).hexdigest()
    except (OSError, TypeError):
        report["statistics_code_sha256"] = None
        report["statistics_code_note"] = "Runtime source introspection unavailable; retain the executed notebook with this report"
    (output / "patient_cluster_provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Patient-cluster Grade-3 analysis", "", "Status: " + status, "", CLUSTER_SCOPE, "",
        "No training, inference, checkpoint loading or writes to input caches/manifests were performed.", "",
        "Point estimates retain image weighting; only the uncertainty calculation changes."]
    if blocked:
        lines.extend(["", "Macro CI is withheld until every requested cohort has usable patient grouping."])
        lines.extend(f"- {b['cohort']}: {b['reason']}" for b in blocked)
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table, report

# %%
folds = containing(stage_archive(FOLD_SOURCE, WORK / 'archives'), 'fold_0_aptos_test.csv')
output = OUTPUTS / 'camera_ready' / 'patient_cluster'
if OFFICIAL_PAIRING is not None:
    mapping = output / 'mappings' / (uuid.uuid4().hex + '.csv')
    make_messidor_pair_mapping(OFFICIAL_PAIRING, LEFT_COLUMN, RIGHT_COLUMN,
                               folds / 'fold_2_messidor2_test.csv', mapping)
    PATIENT_MAPPINGS['messidor2'] = mapping
if AUDIT_MESSIDOR_SPLITS:
    display(audit_messidor_mapping_in_saved_splits(PATIENT_MAPPINGS['messidor2'], folds))
table, report = run_patient_cluster_posthoc(dict(source_mode=SOURCE_MODE, input_root=INPUT_ROOT,
    output_root=output, folds=FOLDS, folds_dir=folds, reference_json=REFERENCE, reps=2000,
    seed=42, patient_mappings=PATIENT_MAPPINGS, allow_manifest_order=ALLOW_MANIFEST_ORDER))
display(table)
print(report['status'], report['output_directory'])
