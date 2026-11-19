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

from evaluation.paired_bootstrap import (
    grade3_confusion, macro_bootstrap_grade3_sensitivity,
    paired_bootstrap_grade3_sensitivity, validate_pairs,
)

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
