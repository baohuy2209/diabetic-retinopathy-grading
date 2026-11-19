# %% [markdown]
# # Replay the archived V2/V3 predictions

# %%
NOTEBOOK_CODE_SHA256 = '4908b8851a1c88072f307d92497f804f2cd020badd6f940fe9fec1fec2a14096'

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
RAW_PREDS = Path(os.environ.get('DR_RAW_PREDS', INPUTS / 'raw_preds'))
ALLOW_MANIFEST_ORDER = os.environ.get('DR_ALLOW_MANIFEST_ORDER') == '1'  # Enable only when archived loader order is independently documented.

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
# ## experiments/camera_ready.py

# %%
"""Run paired bootstrap/confusion from real saved predictions; never trains models."""

import argparse
import json
from pathlib import Path




def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-preds", type=Path, help="Original posthoc_fixed/raw_preds NPZ directory")
    source.add_argument("--predictions-dir", type=Path, help="Directory with fold_N_cohort_predictions.csv")
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="lodo_summary_with_calibrated_screening.json")
    parser.add_argument("--output-dir", type=Path, default=Path("result/camera_ready"))
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--allow-manifest-order", action="store_true", help="Explicitly acknowledge legacy NPZs omit IDs; requires documented unshuffled order")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.cohorts)) != len(args.cohorts):
        parser.error("Duplicate cohorts are not allowed")
    frames, inputs = {}, []
    for cohort in args.cohorts:
        if args.raw_preds:
            frame, provenance = load_archive_pair(args.raw_preds, args.folds_dir, cohort, args.allow_manifest_order)
        else:
            stem = f"fold_{COHORTS.index(cohort)}_{cohort}"
            frame, provenance = load_csv_pair(args.predictions_dir / (stem + "_predictions.csv"),
                                              args.folds_dir / (stem + "_test.csv"))
        frames[cohort] = frame
        inputs.extend(provenance)
    report = analyze_frames(frames, args.reference, args.output_dir, inputs, reps=args.bootstrap_reps,
                            seed=args.seed, source_mode="archived_predictions" if args.raw_preds else "prediction_csv",
                            exact_reference=bool(args.raw_preds), diagnostic_only=args.diagnostic_only)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "BENCHMARK_REPRODUCED" else 2

# %%
folds = containing(stage_archive(FOLD_SOURCE, WORK / 'archives'), 'fold_0_aptos_test.csv')
args = ['--raw-preds', str(RAW_PREDS), '--folds-dir', str(folds), '--reference', str(REFERENCE),
        '--output-dir', str(OUTPUTS / 'camera_ready' / 'archive'), '--bootstrap-reps', '2000', '--seed', '42']
if ALLOW_MANIFEST_ORDER:
    args.append('--allow-manifest-order')
status = main(args)
print('Exit status:', status)
