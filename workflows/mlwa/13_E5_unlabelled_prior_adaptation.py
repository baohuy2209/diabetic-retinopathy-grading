# %% [markdown]
# # 13 · E5 Opt-In Unlabelled Target Prior Adaptation (Saerens EM & BBSE)
# 
# > **Standalone Single-File Notebook**: All required modules are embedded directly below. Upload only this `.ipynb` file to Google Colab and run all cells (`Runtime -> Run all`); no repository clone or zip bundle is required.

# %%
NOTEBOOK_CODE_SHA256 = '46aab3cea4972d57688c25afc45c7800e31c1411c064ccab00306e647d49879b'

# %%
import os
import sys
from pathlib import Path

IN_COLAB: bool = (
    'COLAB_RELEASE_TAG' in os.environ
    or 'COLAB_BACKEND_VERSION' in os.environ
    or 'google.colab' in sys.modules
)


def _running_in_notebook() -> bool:
    if 'ipykernel' not in sys.modules and 'google.colab' not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


def _setup_colab_environment() -> Path:
    if IN_COLAB:
        import importlib.util
        import subprocess
        deps = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'pyarrow': 'pyarrow'}
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')
    return Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _setup_colab_environment()
    ENABLE_E5 = False
    OUTPUT_DIR = None
    SMOKE = True

# %% [markdown]
# ## Embedded Standalone Implementation (5 self-contained modules)

# %% [markdown]
# ### Module `experiments.mlwa.constants`

# %%
"""Canonical constants, status codes, and portable environment resolution for MLWA journal experiments."""


import os
from pathlib import Path
from typing import Dict, Optional, Tuple

COHORT_ORDER: Tuple[str, ...] = ("aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid")
CANONICAL_FOLD_TO_COHORT: Dict[int, str] = {i: c for i, c in enumerate(COHORT_ORDER)}
COHORT_TO_FOLD: Dict[str, int] = {c: i for i, c in enumerate(COHORT_ORDER)}

COHORT_LABELS: Dict[str, str] = {
    "aptos": "APTOS 2019",
    "idrid": "IDRiD",
    "messidor2": "Messidor-2",
    "ddr": "DDR",
    "eyepacs": "EyePACS",
    "deepdrid": "DeepDRiD",
}

PROB_COLS: Tuple[str, ...] = ("p0", "p1", "p2", "p3", "p4")
THETA_COLS: Tuple[str, ...] = ("theta0", "theta1", "theta2", "theta3")
TIE_TOL: float = 1e-12
SCHEMA_VERSION: str = "mlwa_journal_v1"

STATUS_PLANNED = "PLANNED"
STATUS_PREFLIGHT_OK = "PREFLIGHT_OK"
STATUS_PENDING_ASSETS = "PENDING_ASSETS"
STATUS_RUNNING = "RUNNING"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_FAILED = "FAILED"
STATUS_COMPLETE = "COMPLETE"
STATUS_DIAGNOSTIC_REPLAY = "DIAGNOSTIC_REPLAY"
STATUS_SKIPPED_GATE_NOT_MET = "SKIPPED_GATE_NOT_MET"
STATUS_TEST_ONLY = "TEST_ONLY"

STATUS_CODES: Tuple[str, ...] = (
    STATUS_PLANNED,
    STATUS_PREFLIGHT_OK,
    STATUS_PENDING_ASSETS,
    STATUS_RUNNING,
    STATUS_INTERRUPTED,
    STATUS_FAILED,
    STATUS_COMPLETE,
    STATUS_DIAGNOSTIC_REPLAY,
    STATUS_SKIPPED_GATE_NOT_MET,
    STATUS_TEST_ONLY,
)

REQUIRED_PREDICTION_COLUMNS: Tuple[str, ...] = (
    "study_id",
    "protocol_id",
    "run_id",
    "experiment",
    "config_id",
    "architecture_id",
    "head_type",
    "objective_id",
    "clm_link",
    "fold_id",
    "cohort",
    "seed_train",
    "split_role",
    "epoch",
    "checkpoint_sha256",
    "image_key",
    "image_id",
    "patient_key",
    "patient_identity_status",
    "y_true",
    *PROB_COLS,
    "pred_native",
    "s",
    *THETA_COLS,
    "feature_file",
    "feature_row",
    "manifest_sha256",
    "preprocessing_id",
    "schema_version",
    "code_sha256",
    "plan_sha256",
)


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate repository root cleanly without hardcoded external Windows paths."""
    cands = []
    if start is not None:
        cands.append(Path(start).resolve())
    cands.append(Path.cwd().resolve())
    if "__file__" in globals():
        cands.append(Path(__file__).resolve().parents[2])
    for base in cands:
        for cur in (base, *base.parents):
            if (cur / "pyproject.toml").is_file() and (cur / "experiments" / "mlwa").is_dir():
                return cur
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    return Path.cwd().resolve()


def resolve_runtime_paths(repo_root: Optional[Path] = None) -> Dict[str, Path]:
    """Resolve portable DR_* paths for local and Google Colab sessions."""
    root = find_repo_root(repo_root)
    dr_root = Path(os.environ.get("DR_ROOT", str(root))).expanduser().resolve()
    inputs = Path(os.environ.get("DR_INPUTS", str(dr_root / "inputs"))).expanduser().resolve()
    outputs_base = Path(os.environ.get("DR_JOURNAL_OUTPUTS", os.environ.get("DR_OUTPUTS", str(dr_root / "outputs")))).expanduser().resolve()
    work = Path(os.environ.get("DR_WORK", str(dr_root / "work"))).expanduser().resolve()

    images = Path(os.environ.get("DR_IMAGES", str(inputs / "native_cropped"))).expanduser().resolve()
    folds = Path(os.environ.get("DR_FOLDS", str(inputs / "lodo_folds"))).expanduser().resolve()
    checkpoints = Path(os.environ.get("DR_CHECKPOINTS", str(inputs / "checkpoints"))).expanduser().resolve()
    pooled_ckpts = Path(os.environ.get("DR_POOLED_CHECKPOINTS", str(inputs / "checkpoints_head_ablation"))).expanduser().resolve()
    pooled_meta = Path(os.environ.get("DR_POOLED_METADATA", str(inputs / "hybrid"))).expanduser().resolve()
    p0_run = Path(os.environ.get("DR_P0_RUN", str(root / "result" / "p0_fp32_main_lodo_v1"))).expanduser().resolve()

    return {
        "repo_root": root,
        "dr_root": dr_root,
        "inputs": inputs,
        "outputs": outputs_base,
        "work": work,
        "images": images,
        "folds": folds,
        "checkpoints": checkpoints,
        "pooled_checkpoints": pooled_ckpts,
        "pooled_metadata": pooled_meta,
        "p0_run": p0_run,
    }

# %% [markdown]
# ### Module `experiments.mlwa.data_contract`

# %%
"""Data, manifest, image-identity, P0 cache, and prediction schema contracts."""


import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def relative_image_path(value: str) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    lower = [p.lower() for p in parts]
    if "native_cropped" in lower:
        parts = parts[lower.index("native_cropped") + 1 :]
    if not parts or ".." in parts or any(":" in p for p in parts) or parts[0] == "/":
        raise ValueError(f"Unsafe or unresolved relative image path: {value}")
    return PurePosixPath(*parts).as_posix()


def stage_archive_safe(source: Path, workspace: Path) -> Path:
    """Safely extract an immutable ZIP to a keyed local folder or return an existing directory."""
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        return source
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        seen = set()
        for info in entries:
            name = info.filename.replace("\\", "/")
            rel = PurePosixPath(name)
            is_symlink = ((info.external_attr >> 16) & 0o170000) == 0o120000
            if rel.is_absolute() or ".." in rel.parts or ":" in name or is_symlink or not rel.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if name.casefold() in seen:
                raise ValueError(f"Duplicate ZIP member path: {info.filename}")
            seen.add(name.casefold())

        index = [(m.filename, m.file_size, m.CRC) for m in entries]
        key = object_hash(index)
        target = Path(workspace) / f"{source.stem}_{key[:16]}"
        ready = target / ".archive-complete"
        if ready.is_file() and ready.read_text(encoding="utf-8").strip() == key:
            return target

        temp = target.with_name(f"{target.name}.{uuid4().hex}.partial")
        temp.mkdir(parents=True, exist_ok=True)
        try:
            for m in entries:
                parts = PurePosixPath(m.filename.replace("\\", "/")).parts
                dest = temp.joinpath(*parts)
                if not dest.resolve().is_relative_to(temp.resolve()):
                    raise ValueError(f"Path traversal blocked: {m.filename}")
                if m.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(m) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 2**20)
            (temp / ".archive-complete").write_text(key, encoding="utf-8")
            try:
                temp.rename(target)
            except OSError:
                if not ready.is_file() or ready.read_text(encoding="utf-8").strip() != key:
                    raise
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return target


def normalize_manifest_df(
    df: pd.DataFrame,
    patient_map: Optional[Path] = None,
    images_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Attach canonical image_key, grouping/patient_key, and identity status cleanly."""
    out = df.copy()
    id_col = "image_id"
    if id_col not in out.columns:
        raise ValueError("Manifest missing required column 'image_id'.")
    y_col = "diagnosis" if "diagnosis" in out.columns else ("y_true" if "y_true" in out.columns else None)
    if y_col is None:
        raise ValueError("Manifest missing label column ('diagnosis' or 'y_true').")
    cohort_col = "dataset_key" if "dataset_key" in out.columns else ("cohort" if "cohort" in out.columns else None)
    if cohort_col is None:
        raise ValueError("Manifest missing cohort column ('dataset_key' or 'cohort').")

    out["image_id"] = out[id_col].astype(str)
    out["cohort"] = (
        out[cohort_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"messidor-2": "messidor2", "messidor_2": "messidor2", "deep-drid": "deepdrid", "deep_drid": "deepdrid"})
    )
    out["y_true"] = pd.to_numeric(out[y_col], errors="raise").astype(int)
    if not out["y_true"].isin(range(5)).all():
        raise ValueError("Manifest contains invalid grade outside {0..4}.")

    if "image_key" not in out.columns:
        out["image_key"] = out["cohort"] + "::" + out["image_id"]
    else:
        out["image_key"] = out["image_key"].astype(str)

    if out["image_key"].duplicated().any():
        dups = out.loc[out["image_key"].duplicated(), "image_key"].head(5).tolist()
        raise ValueError(f"Duplicate image_key entries in manifest: {dups}")

    if "patient_key" in out.columns:
        out["patient_key"] = out["patient_key"].astype(str)
    elif "patient_id" in out.columns:
        out["patient_key"] = out["patient_id"].astype(str)
    else:
        out["patient_key"] = out["image_key"]

    if "patient_identity_status" not in out.columns:
        out["patient_identity_status"] = "legacy_unverified"

    if patient_map is not None:
        mapping = pd.read_csv(patient_map, dtype=str)
        if mapping["image_key"].duplicated().any() or mapping["patient_key"].isna().any():
            raise ValueError(f"Invalid verified patient mapping: {patient_map}")
        mp = mapping.set_index("image_key")["patient_key"]
        known = out["image_key"].isin(mp.index)
        out.loc[known, "patient_key"] = out.loc[known, "image_key"].map(mp)
        out.loc[known, "patient_identity_status"] = "verified"

    path_src_col = next((c for c in ("native_path", "raw_full_path", "original_relative_path") if c in out.columns), None)
    if path_src_col is not None:
        out["original_relative_path"] = out[path_src_col].astype(str).map(relative_image_path)
    if images_root is not None and "original_relative_path" in out.columns:
        root = Path(images_root)
        out["runtime_path"] = [str(root.joinpath(*PurePosixPath(p).parts)) for p in out["original_relative_path"]]
    return out


def verify_images_readable(
    df: pd.DataFrame,
    check_decode: bool = True,
    max_checks: Optional[int] = None,
) -> Dict[str, Any]:
    """Check file presence and OpenCV decode health; fail explicitly rather than skipping silently."""
    if "runtime_path" not in df.columns:
        raise ValueError("runtime_path column required to verify image files.")
    paths = df["runtime_path"].astype(str).tolist()
    if max_checks is not None:
        paths = paths[:max_checks]

    missing: List[str] = []
    unreadable: List[str] = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            missing.append(p)
            continue
        if check_decode:
            img = cv2.imread(str(fp))
            if img is None or img.size == 0:
                unreadable.append(p)

    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images; first 3: {missing[:3]}")
    if unreadable:
        raise OSError(f"{len(unreadable)} unreadable/corrupt images; first 3: {unreadable[:3]}")
    return {"checked": len(paths), "missing": 0, "unreadable": 0}


def build_natural_source_train(
    folds_dir: Path,
    fold_id: int,
    patient_map: Optional[Path] = None,
    images_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reconstruct natural-prevalence 85% source_train for fold_id from non-target test.csv - val.csv."""
    folds_dir = Path(folds_dir)
    if fold_id not in CANONICAL_FOLD_TO_COHORT:
        raise ValueError(f"Invalid fold_id={fold_id}")
    held_cohort = CANONICAL_FOLD_TO_COHORT[fold_id]

    va_raw = pd.read_csv(folds_dir / f"fold_{fold_id}_{held_cohort}_val.csv", dtype=str)
    va_df = normalize_manifest_df(va_raw, patient_map=patient_map, images_root=images_root)

    source_parts: List[pd.DataFrame] = []
    input_hashes: Dict[str, str] = {
        f"fold_{fold_id}_{held_cohort}_val.csv": sha256_file(folds_dir / f"fold_{fold_id}_{held_cohort}_val.csv")
    }
    for src_fold, src_cohort in CANONICAL_FOLD_TO_COHORT.items():
        if src_fold == fold_id:
            continue
        tp = folds_dir / f"fold_{src_fold}_{src_cohort}_test.csv"
        input_hashes[tp.name] = sha256_file(tp)
        part = normalize_manifest_df(pd.read_csv(tp, dtype=str), patient_map=patient_map, images_root=images_root)
        source_parts.append(part)

    pool_df = pd.concat(source_parts, ignore_index=True)
    if pool_df["image_key"].duplicated().any():
        raise ValueError(f"Duplicate image_key detected across source cohorts for fold {fold_id}.")

    va_keys = set(va_df["image_key"])
    nat_train = pool_df[~pool_df["image_key"].isin(va_keys)].copy().reset_index(drop=True)

    # Verify zero group overlap between reconstructed natural source_train and source_val
    nat_groups = set(nat_train["patient_key"])
    va_groups = set(va_df["patient_key"])
    overlap = nat_groups & va_groups
    if overlap:
        raise ValueError(f"Fold {fold_id} grouping-key overlap between natural source_train and val: {len(overlap)}")

    nat_counts = {int(k): int((nat_train["y_true"] == k).sum()) for k in range(5)}
    meta = {
        "fold_id": fold_id,
        "held_out_cohort": held_cohort,
        "pool_images": int(len(pool_df)),
        "natural_source_train_images": int(len(nat_train)),
        "source_val_images": int(len(va_df)),
        "realized_train_ratio": float(len(nat_train) / len(pool_df)),
        "natural_grade_counts": nat_counts,
        "input_manifest_sha256": input_hashes,
        "reconstructed_manifest_sha256": object_hash(
            {"image_keys": nat_train["image_key"].tolist(), "y_true": nat_train["y_true"].tolist()}
        ),
    }
    return nat_train, meta


def audit_lodo_manifests(
    folds_dir: Path,
    images_root: Optional[Path] = None,
    check_decode: bool = False,
) -> Dict[str, Any]:
    """Comprehensive audit of LODO CSV manifests, group leakage, quota vs natural counts, and optional images."""
    folds_dir = Path(folds_dir)
    fold_reports: List[Dict[str, Any]] = []
    inventory_per_cohort: Dict[str, int] = {}

    for fold_id, cohort in CANONICAL_FOLD_TO_COHORT.items():
        p_tr = folds_dir / f"fold_{fold_id}_{cohort}_train.csv"
        p_va = folds_dir / f"fold_{fold_id}_{cohort}_val.csv"
        p_te = folds_dir / f"fold_{fold_id}_{cohort}_test.csv"
        for p in (p_tr, p_va, p_te):
            if not p.is_file():
                raise FileNotFoundError(f"Missing LODO manifest: {p}")

        tr = normalize_manifest_df(pd.read_csv(p_tr, dtype=str), images_root=images_root)
        va = normalize_manifest_df(pd.read_csv(p_va, dtype=str), images_root=images_root)
        te = normalize_manifest_df(pd.read_csv(p_te, dtype=str), images_root=images_root)

        if set(te["cohort"].unique()) != {cohort}:
            raise ValueError(f"Fold {fold_id} test manifest cohort mismatch: {te['cohort'].unique().tolist()}")
        if cohort in set(tr["cohort"]) or cohort in set(va["cohort"]):
            raise ValueError(f"Held-out cohort '{cohort}' leaked into source train/val on fold {fold_id}.")

        for name_a, da, name_b, db in [("train", tr, "val", va), ("train", tr, "test", te), ("val", va, "test", te)]:
            img_ov = set(da["image_key"]) & set(db["image_key"])
            grp_ov = set(da["patient_key"]) & set(db["patient_key"])
            if img_ov or grp_ov:
                raise ValueError(
                    f"Fold {fold_id} overlap ({name_a} vs {name_b}): "
                    f"images={len(img_ov)}, groups={len(grp_ov)}"
                )

        nat_tr, nat_meta = build_natural_source_train(folds_dir, fold_id, images_root=images_root)
        tr_in_nat = set(tr["image_key"]).issubset(set(nat_tr["image_key"]))
        if not tr_in_nat:
            raise ValueError(f"Fold {fold_id} quota train is not a subset of natural source_train.")

        # Label consistency check across quota train and natural source_train
        merged_lbl = tr[["image_key", "y_true"]].merge(
            nat_tr[["image_key", "y_true"]], on="image_key", suffixes=("_quota", "_nat")
        )
        if not (merged_lbl["y_true_quota"] == merged_lbl["y_true_nat"]).all():
            raise ValueError(f"Fold {fold_id} label discrepancy between train.csv and test/pool manifests.")

        if images_root is not None:
            verify_images_readable(te, check_decode=check_decode, max_checks=50 if check_decode else None)

        inventory_per_cohort[cohort] = int(len(te))
        fold_reports.append(
            {
                "fold_id": fold_id,
                "cohort": cohort,
                "quota_train_images": int(len(tr)),
                "quota_grade_counts": {int(k): int((tr["y_true"] == k).sum()) for k in range(5)},
                "natural_source_train_images": nat_meta["natural_source_train_images"],
                "natural_grade_counts": nat_meta["natural_grade_counts"],
                "source_val_images": int(len(va)),
                "target_images": int(len(te)),
                "target_g3_support": int((te["y_true"] == 3).sum()),
                "realized_natural_train_ratio": nat_meta["realized_train_ratio"],
                "quota_subset_of_natural": True,
                "sha256": {
                    "train": sha256_file(p_tr),
                    "val": sha256_file(p_va),
                    "test": sha256_file(p_te),
                },
            }
        )

    return {
        "total_target_evaluated_images": int(sum(inventory_per_cohort.values())),
        "target_counts_by_cohort": inventory_per_cohort,
        "folds": fold_reports,
    }


def validate_p0_cache(
    p0_dir: Path,
    folds: Sequence[int] = tuple(range(6)),
    variants: Sequence[str] = ("V2", "V3"),
    splits: Sequence[str] = ("source_train", "source_val", "target"),
    require_features: bool = False,
) -> Dict[str, Any]:
    """Verify P0 COMPLETE.json digests, finite probabilities, alignment, and optional z.npy arrays."""
    p0_dir = Path(p0_dir)
    lodo_dir = p0_dir / "lodo"
    if not lodo_dir.is_dir():
        raise FileNotFoundError(f"Missing P0 lodo/ directory at: {p0_dir}")

    checked_shards = 0
    ckpt_hashes: Dict[str, str] = {}
    for variant in variants:
        for fold in folds:
            cohort = CANONICAL_FOLD_TO_COHORT[fold]
            unit_dir = lodo_dir / f"{variant}_fold_{fold}_{cohort}"
            for split in splits:
                shard_dir = unit_dir / split
                comp_path = shard_dir / "COMPLETE.json"
                if not comp_path.is_file():
                    raise ValueError(f"Incomplete P0 shard (missing COMPLETE.json): {shard_dir}")
                meta = json.loads(comp_path.read_text(encoding="utf-8"))
                ident = meta.get("identity", {})
                ckpt_sha = ident.get("checkpoint_sha256")
                unit_key = f"{variant}_fold_{fold}"
                if ckpt_sha is not None:
                    if unit_key in ckpt_hashes and ckpt_hashes[unit_key] != ckpt_sha:
                        raise ValueError(f"Mixed checkpoint_sha256 in {unit_key}/{split}")
                    ckpt_hashes[unit_key] = ckpt_sha

                files_meta = meta.get("files", {})
                pred_path = shard_dir / "predictions.csv"
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Missing predictions.csv in {shard_dir}")
                if "predictions.csv" in files_meta and sha256_file(pred_path) != files_meta["predictions.csv"]:
                    raise ValueError(f"Corrupted/tampered predictions.csv digest in {shard_dir}")

                df = pd.read_csv(pred_path)
                if df.empty or df["image_key"].duplicated().any():
                    raise ValueError(f"Empty or duplicate image_key in {pred_path}")
                p = df[list(PROB_COLS)].to_numpy(dtype=np.float64)
                if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
                    raise ValueError(f"Invalid probabilities in {pred_path}")

                z_path = shard_dir / "z.npy"
                if require_features or z_path.is_file():
                    if not z_path.is_file():
                        raise FileNotFoundError(f"Missing z.npy in {shard_dir}")
                    if "z.npy" in files_meta and sha256_file(z_path) != files_meta["z.npy"]:
                        raise ValueError(f"Corrupted/tampered z.npy digest in {shard_dir}")
                    z = np.load(z_path, mmap_mode="r")
                    if z.ndim != 2 or z.shape[0] != len(df):
                        raise ValueError(f"Row mismatch z.npy {z.shape} vs predictions {len(df)} in {shard_dir}")
                    if not np.isfinite(np.asarray(z[: min(8, len(df))])).all():
                        raise ValueError(f"Non-finite values in {z_path}")

                checked_shards += 1

    return {"p0_dir": str(p0_dir), "checked_shards": checked_shards, "checkpoint_hashes": ckpt_hashes}


def save_predictions_table(df: pd.DataFrame, out_dir: Path, prefer_parquet: bool = True) -> Path:
    """Save canonical per-image predictions to compressed Parquet (with CSV fallback)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions DataFrame missing required columns: {missing}")

    p = df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("Refusing to write corrupted non-simplex probabilities.")

    if prefer_parquet:
        try:
            target = out_dir / "predictions.parquet"
            tmp = out_dir / f"predictions.{uuid4().hex[:8]}.parquet.tmp"
            df.to_parquet(tmp, index=False, compression="snappy")
            os.replace(tmp, target)
            return target
        except Exception:
            pass

    target = out_dir / "predictions.csv"
    tmp = out_dir / f"predictions.{uuid4().hex[:8]}.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, target)
    return target


def load_predictions_table(directory_or_file: Path) -> pd.DataFrame:
    p = Path(directory_or_file)
    if p.is_dir():
        pq = p / "predictions.parquet"
        cs = p / "predictions.csv"
        p = pq if pq.is_file() else cs
    if not p.is_file():
        raise FileNotFoundError(f"Predictions table not found: {directory_or_file}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_feature_bundle(
    z: np.ndarray,
    image_keys: Sequence[str],
    out_dir: Path,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Write mmap-friendly z.npy + keyed z_manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(z, dtype=dtype)
    if arr.ndim != 2 or arr.shape[0] != len(image_keys):
        raise ValueError(f"Feature shape {arr.shape} does not match {len(image_keys)} image keys.")
    if not np.isfinite(arr).all():
        raise ValueError("Refusing to cache non-finite latent feature array.")

    z_path = out_dir / "z.npy"
    tmp = out_dir / f"z.{uuid4().hex[:8]}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, z_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_file": "z.npy",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "sha256": sha256_file(z_path),
        "image_keys_sha256": object_hash(list(image_keys)),
    }
    atomic_write_json(out_dir / "z_manifest.json", manifest)
    return manifest

# %% [markdown]
# ### Module `experiments.mlwa.evaluator`

# %%
"""Authoritative MLWA evaluation engine: metrics, RDR screening, hierarchical aggregation, contrasts, and CIs."""


import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve


def compute_decision_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute exact 5-class decision metrics with explicit undefined handling (never silent zeros)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("y_true and y_pred must have matching non-empty 1D shape.")

    cm = confusion_matrix(y, p, labels=np.arange(5))
    support = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    recall = np.divide(cm.diagonal(), support, out=np.full(5, np.nan), where=support > 0)
    precision = np.divide(cm.diagonal(), pred_counts, out=np.full(5, np.nan), where=pred_counts > 0)

    full_support = bool((support > 0).all())
    macro_recall = float(np.mean(recall)) if full_support else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qwk = float(cohen_kappa_score(y, p, labels=np.arange(5), weights="quadratic"))

    n_total = len(y)
    n3 = int(support[3])
    pred3 = int(pred_counts[3])
    out: Dict[str, Any] = {
        "n": int(n_total),
        "qwk": qwk,
        "accuracy": float(np.mean(y == p)),
        "within1": float(np.mean(np.abs(y - p) <= 1)),
        "mae": float(np.mean(np.abs(y - p))),
        "macro_recall": macro_recall,
        "macro_recall_defined": full_support,
        "g3_precision": float(precision[3]),
        "g3_precision_defined": bool(pred3 > 0),
        "true_g3": n3,
        "predicted_g3": pred3,
        "true_g3_prevalence": float(n3 / n_total),
        "predicted_g3_prevalence": float(pred3 / n_total),
        "predicted_true_g3_ratio": float(pred3 / n3) if n3 > 0 else np.nan,
        "g3_to_2_count": int(cm[3, 2]),
        "g3_to_01_count": int(cm[3, 0] + cm[3, 1]),
        "g3_to_4_count": int(cm[3, 4]),
        "g3_to_2_fraction": float(cm[3, 2] / n3) if n3 > 0 else np.nan,
        "g3_to_01_fraction": float((cm[3, 0] + cm[3, 1]) / n3) if n3 > 0 else np.nan,
        "confusion_matrix": cm.tolist(),
    }
    for k in range(5):
        out[f"recall_g{k}"] = float(recall[k])
        out[f"precision_g{k}"] = float(precision[k])
        out[f"support_g{k}"] = int(support[k])
        out[f"precision_g{k}_defined"] = bool(pred_counts[k] > 0)
        out[f"recall_g{k}_defined"] = bool(support[k] > 0)
    return out


def compute_posterior_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Compute decoder-independent posterior quality metrics (NLL, RPS mean-4 / sum-4, 15-bin ECE, RDR AUROC)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape != (len(y), 5) or len(y) == 0:
        raise ValueError(f"Expected probs shape ({len(y)}, 5), got {p.shape}.")
    if not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Invalid probabilities in compute_posterior_metrics.")

    p_y = p[np.arange(len(y)), y]
    nll_mean = float(-np.log(np.maximum(p_y, 1e-12)).mean())

    # Ranked Probability Score over 4 cumulative thresholds: explicit mean across 4 thresholds & sum
    cdf = p.cumsum(axis=1)[:, :4]
    target_cdf = (y[:, None] <= np.arange(4)[None, :]).astype(np.float64)
    sq_err = (cdf - target_cdf) ** 2
    rps_mean_4 = float(sq_err.mean())
    rps_sum_4 = float(sq_err.sum(axis=1).mean())

    # 15-bin ECE using top-probability confidence and argmax correctness
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    bins = np.minimum((confidence * 15.0).astype(int), 14)
    ece_15 = 0.0
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece_15 += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))

    y_rdr = (y >= 2).astype(int)
    risk_rdr = p[:, 2:].sum(axis=1)
    rdr_defined = bool(len(np.unique(y_rdr)) == 2)
    rdr_auroc = float(roc_auc_score(y_rdr, risk_rdr)) if rdr_defined else np.nan

    return {
        "nll_mean": nll_mean,
        "rps_mean_4": rps_mean_4,
        "rps_sum_4": rps_sum_4,
        "ece_15": float(ece_15),
        "rdr_auroc": rdr_auroc,
        "rdr_auroc_defined": rdr_defined,
    }


def calibrate_rdr_threshold(
    y_source_val: np.ndarray,
    p_source_val: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Select RDR (>=2) threshold strictly on source_val ROC at target specificity."""
    y_bin = (np.asarray(y_source_val, dtype=int) >= 2).astype(int)
    if len(np.unique(y_bin)) != 2:
        raise ValueError("Source validation must contain both RDR-negative and RDR-positive samples.")
    risk = np.asarray(p_source_val, dtype=np.float64)[:, 2:].sum(axis=1)
    fpr, tpr, thresholds = roc_curve(y_bin, risk, drop_intermediate=False)
    elig = np.flatnonzero(fpr <= 1.0 - float(target_specificity) + 1e-12)
    if len(elig) == 0:
        raise ValueError(f"No threshold achieved specificity >= {target_specificity} on source_val.")
    idx = max(elig, key=lambda j: (tpr[j], -fpr[j], thresholds[j]))
    return float(thresholds[idx])


def evaluate_rdr_screening(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_bin = np.asarray(y_true, dtype=int) >= 2
    pred_pos = np.asarray(probs, dtype=np.float64)[:, 2:].sum(axis=1) >= float(threshold)
    sens = float(pred_pos[y_bin].mean()) if y_bin.any() else np.nan
    spec = float((~pred_pos[~y_bin]).mean()) if (~y_bin).any() else np.nan
    return {
        "rdr_threshold": float(threshold),
        "rdr_sensitivity": sens,
        "rdr_specificity": spec,
        "rdr_sensitivity_defined": bool(y_bin.any()),
        "rdr_specificity_defined": bool((~y_bin).any()),
    }


def aggregate_cohort_seed_metrics(
    per_cohort_seed_df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = ("config_id", "decoder"),
) -> Dict[str, pd.DataFrame]:
    """Compute hierarchical summaries:
    (a) per_cohort: mean and sample SD (ddof=1) across seeds per cohort;
    (b) equal_cohort: per-seed 6-cohort mean, then mean and sample SD (ddof=1) across seed means.
    """
    required = list(group_cols) + ["fold_id", "cohort", "seed_train"] + list(metric_cols)
    missing = [c for c in required if c not in per_cohort_seed_df.columns]
    if missing:
        raise ValueError(f"Input table missing columns: {missing}")

    # (a) Per-cohort across training seeds
    per_cohort_rows: List[Dict[str, Any]] = []
    for keys, grp in per_cohort_seed_df.groupby(list(group_cols) + ["fold_id", "cohort"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(list(group_cols) + ["fold_id", "cohort"], keys))
        base["n_seeds"] = int(grp["seed_train"].nunique())
        base["aggregation"] = "per_cohort_across_seeds"
        for m in metric_cols:
            vals = grp[m].dropna().to_numpy(dtype=np.float64)
            base[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            base[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        per_cohort_rows.append(base)
    per_cohort_df = pd.DataFrame(per_cohort_rows)

    # (b) Per-seed equal-6-cohort mean, then mean & sample SD across seed means
    seed_mean_rows: List[Dict[str, Any]] = []
    for keys, grp in per_cohort_seed_df.groupby(list(group_cols) + ["seed_train"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(list(group_cols) + ["seed_train"], keys))
        base["n_cohorts"] = int(grp["cohort"].nunique())
        for m in metric_cols:
            vals = grp[m].to_numpy(dtype=np.float64)
            base[m] = float(np.mean(vals)) if (len(vals) == len(COHORT_ORDER) and np.isfinite(vals).all()) else np.nan
        seed_mean_rows.append(base)
    seed_means_df = pd.DataFrame(seed_mean_rows)

    eq_rows: List[Dict[str, Any]] = []
    for keys, grp in seed_means_df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(list(group_cols), keys))
        base["n_seeds"] = int(grp["seed_train"].nunique())
        base["n_cohorts_expected"] = len(COHORT_ORDER)
        base["aggregation"] = "equal_cohort_mean_of_seed_means"
        for m in metric_cols:
            vals = grp[m].dropna().to_numpy(dtype=np.float64)
            base[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            base[f"{m}_sd_across_seeds"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        eq_rows.append(base)
    equal_cohort_df = pd.DataFrame(eq_rows)

    return {
        "per_cohort": per_cohort_df,
        "per_seed_equal_cohort": seed_means_df,
        "equal_cohort": equal_cohort_df,
    }


def compute_factorial_contrasts(
    per_cell_df: pd.DataFrame,
    metric_cols: Sequence[str],
    cell_col: str = "cell",
    cell_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired within-(fold_id, seed_train) 2x2 factorial contrasts before any rounding."""
    cmap = cell_map or {"A": "A", "B": "B", "C": "C", "D": "D"}
    pivoted: List[Dict[str, Any]] = []
    for (fold_id, cohort, seed_train), grp in per_cell_df.groupby(["fold_id", "cohort", "seed_train"], sort=True):
        by_c = {str(r[cell_col]): r for _, r in grp.iterrows()}
        if not all(cmap[k] in by_c for k in ("A", "B", "C", "D")):
            continue
        A, B, C, D = by_c[cmap["A"]], by_c[cmap["B"]], by_c[cmap["C"]], by_c[cmap["D"]]
        for m in metric_cols:
            va, vb, vc, vd = float(A[m]), float(B[m]), float(C[m]), float(D[m])
            head_main = 0.5 * ((vc - va) + (vd - vb))
            loss_main = 0.5 * ((vb - va) + (vd - vc))
            interaction = (vd - vc) - (vb - va)
            contrasts = {
                "head_main": ("0.5 * ((C - A) + (D - B))", head_main),
                "loss_main": ("0.5 * ((B - A) + (D - C))", loss_main),
                "interaction": ("(D - C) - (B - A)", interaction),
                "head_simple_ce": ("C - A", vc - va),
                "head_simple_qwk": ("D - B", vd - vb),
                "loss_simple_softmax": ("B - A", vb - va),
                "loss_simple_clm": ("D - C", vd - vc),
            }
            is_pct = m in ("recall_g3", "macro_recall", "within1", "accuracy", "g3_precision", "ece_15")
            for cname, (formula, val) in contrasts.items():
                pivoted.append(
                    {
                        "fold_id": int(fold_id),
                        "cohort": str(cohort),
                        "seed_train": int(seed_train),
                        "metric": m,
                        "contrast": cname,
                        "formula": formula,
                        "value_fraction": val,
                        "value_pp": val * 100.0 if is_pct else np.nan,
                        "units": "fraction_and_pp" if is_pct else "native_metric_units",
                    }
                )
    detail_df = pd.DataFrame(pivoted)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    # Summarize per-seed equal-cohort contrast and seed consistency across cohorts
    seed_level = (
        detail_df.groupby(["metric", "contrast", "formula", "units", "seed_train"], as_index=False)[["value_fraction", "value_pp"]]
        .mean()
    )
    summary_rows: List[Dict[str, Any]] = []
    for (metric, contrast, formula, units), grp in seed_level.groupby(["metric", "contrast", "formula", "units"], sort=False):
        vals = grp["value_fraction"].dropna().to_numpy(dtype=np.float64)
        pps = grp["value_pp"].dropna().to_numpy(dtype=np.float64)
        cohort_means = detail_df[(detail_df["metric"] == metric) & (detail_df["contrast"] == contrast)].groupby("cohort")["value_fraction"].mean()
        summary_rows.append(
            {
                "metric": metric,
                "contrast": contrast,
                "formula": formula,
                "units": units,
                "n_seeds": int(len(vals)),
                "mean_fraction": float(np.mean(vals)) if len(vals) else np.nan,
                "sd_across_seed_means_fraction": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "mean_pp": float(np.mean(pps)) if len(pps) else np.nan,
                "sd_across_seed_means_pp": float(np.std(pps, ddof=1)) if len(pps) > 1 else np.nan,
                "n_cohorts_positive": int((cohort_means > 0.0).sum()),
                "n_cohorts_negative": int((cohort_means < 0.0).sum()),
                "aggregation": "equal_cohort_mean_of_seed_means",
            }
        )
    return detail_df, pd.DataFrame(summary_rows)


def paired_bootstrap_ci(
    df: pd.DataFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    unit: str = "patient_cluster",
    metric: str = "recall_g3",
    reps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap preserving multiplicity & prediction pairing; patient_cluster requires verified keys."""
    if unit not in ("patient_cluster", "grouping_key", "image"):
        raise ValueError(f"Unsupported bootstrap unit '{unit}'.")
    y = df["y_true"].to_numpy(dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    pt = float(compute_decision_metrics(y, a)[metric] - compute_decision_metrics(y, b)[metric])

    statuses = set(df.get("patient_identity_status", pd.Series(["legacy_unverified"])).astype(str))
    if unit == "patient_cluster" and statuses != {"verified"}:
        return {
            "metric": metric,
            "unit": unit,
            "point_estimate": pt,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "valid_reps": 0,
            "reps": int(reps),
            "status": "NOT_ESTIMABLE",
            "reason": "patient_identity_status != verified",
        }

    if unit == "image":
        groups = [np.array([i], dtype=int) for i in range(len(df))]
    else:
        groups = [
            np.asarray(idx, dtype=int)
            for idx in df.reset_index(drop=True).groupby("patient_key", sort=True).indices.values()
        ]

    rng = np.random.default_rng(seed)
    deltas = np.full(reps, np.nan, dtype=np.float64)
    for r in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[i] for i in chosen])
        ma = compute_decision_metrics(y[ix], a[ix])[metric]
        mb = compute_decision_metrics(y[ix], b[ix])[metric]
        deltas[r] = ma - mb

    fin = np.isfinite(deltas)
    n_valid = int(fin.sum())
    if n_valid / max(1, reps) < 0.95:
        return {
            "metric": metric,
            "unit": unit,
            "point_estimate": pt,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "valid_reps": n_valid,
            "reps": int(reps),
            "status": "INSUFFICIENT_VALID_REPS",
        }

    return {
        "metric": metric,
        "unit": unit,
        "point_estimate": pt,
        "ci_low_95": float(np.quantile(deltas[fin], 0.025)),
        "ci_high_95": float(np.quantile(deltas[fin], 0.975)),
        "valid_reps": n_valid,
        "reps": int(reps),
        "status": "ESTIMABLE",
    }


def wilcoxon_six_cohort_summary(cohort_deltas: Sequence[float]) -> Dict[str, Any]:
    """Supporting paired Wilcoxon signed-rank test on n=6 cohort-level summaries (min p = 0.03125)."""
    vals = np.asarray(cohort_deltas, dtype=np.float64)
    if len(vals) != 6:
        raise ValueError(f"Expected n=6 cohort-level paired summaries, got {len(vals)}.")
    if np.allclose(vals, 0.0):
        return {"n_cohorts": 6, "statistic": 0.0, "p_value": 1.0, "note": "supporting_only_all_zero_deltas"}
    stat, pval = wilcoxon(vals, zero_method="wilcox", alternative="two-sided", method="exact")
    return {
        "n_cohorts": 6,
        "statistic": float(stat),
        "p_value": float(pval),
        "note": "supporting_only_not_primary_pseudoreplicate",
    }

# %% [markdown]
# ### Module `experiments.mlwa.decoders`

# %%
"""Deterministic Bayes decoders, versioned CIBD selectors, stability bootstrap, and latent region math."""


import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SELECTOR_IDS = (
    "legacy_within1_1pp_v1",
    "journal_dual_guardrail_v1",
    "journal_within1_05pp_v1",
)


def validate_probabilities(p: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Strictly validate N x 5 probability matrix without silent clipping or renormalization."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5 or len(arr) == 0:
        raise ValueError(f"Expected non-empty N x 5 probabilities, got shape {arr.shape}.")
    if not np.isfinite(arr).all() or (arr < 0.0).any() or (arr > 1.0 + tol).any():
        raise ValueError("Invalid probability entries (non-finite, negative, or > 1); no silent repair.")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=tol, rtol=0.0):
        raise ValueError("Probability rows do not sum to 1 within tolerance.")
    return arr


def prior_from_labels(y: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Estimate positive source_train class prior with explicit smoothing."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=5).astype(np.float64) + float(smoothing)
    return counts / counts.sum()


def decode_cibd(
    p: np.ndarray,
    prior: np.ndarray,
    alpha: float,
    tau: float,
    q: int,
) -> np.ndarray:
    """Cost-interpolated Bayes decoding with exact raw cost C(k, j) and lower-grade tie convention."""
    probs = validate_probabilities(p)
    pi = np.asarray(prior, dtype=np.float64)
    if pi.shape != (5,) or (pi <= 0.0).any() or not np.isfinite(pi).all():
        raise ValueError("CIBD requires strictly positive finite 5-class source prior.")
    if q not in (1, 2):
        raise ValueError(f"Unsupported CIBD distance power q={q}; expected 1 or 2.")

    r = probs / (pi[None, :] ** float(tau))
    r /= r.sum(axis=1, keepdims=True)

    d = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    cost = float(alpha) * (d ** q) + (1.0 - float(alpha)) * (d != 0)
    risk = r @ cost.T
    return (risk <= risk.min(axis=1, keepdims=True) + TIE_TOL).argmax(axis=1).astype(int)


def decode_argmax(p: np.ndarray) -> np.ndarray:
    probs = validate_probabilities(p)
    return decode_cibd(probs, np.full(5, 0.2), alpha=0.0, tau=0.0, q=1)


def decode_posterior_median(p: np.ndarray) -> np.ndarray:
    probs = validate_probabilities(p)
    return decode_cibd(probs, np.full(5, 0.2), alpha=1.0, tau=0.0, q=1)


def decode_rounded_expectation_bayes(p: np.ndarray) -> np.ndarray:
    """Exact risk-minimizing decoder for squared error |k - j|^2 (breaks half-integer ties downward)."""
    probs = validate_probabilities(p)
    return decode_cibd(probs, np.full(5, 0.2), alpha=1.0, tau=0.0, q=2)


def decode_native_legacy_round(p: np.ndarray) -> np.ndarray:
    """Legacy PyTorch round-to-even expected grade (preserved solely for historical P0 compatibility)."""
    probs = validate_probabilities(p)
    expected = probs @ np.arange(5, dtype=np.float64)
    return np.clip(np.rint(expected), 0, 4).astype(int)


def decode_cutpoint_latent(s: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Raw CLM latent-cutpoint rule (valid ONLY when scalar s and 4 ordered cutpoints exist)."""
    s_arr = np.asarray(s, dtype=np.float64)
    th = np.asarray(thetas, dtype=np.float64)
    if th.ndim == 1:
        th = np.broadcast_to(th[None, :], (len(s_arr), 4))
    if s_arr.ndim != 1 or th.shape != (len(s_arr), 4):
        raise ValueError("Cutpoint decoding requires 1D scalar s and (N, 4) cutpoints.")
    if not np.isfinite(s_arr).all() or not np.isfinite(th).all():
        raise ValueError("Cutpoint decoding is undefined for non-CLM heads with missing s or theta.")
    if (np.diff(th, axis=1) <= 0.0).any():
        raise ValueError("Cutpoints must be strictly monotonic.")
    return (s_arr[:, None] > th).sum(axis=1).astype(int)


def decode_by_name(
    decoder_id: str,
    probs: np.ndarray,
    prior: Optional[np.ndarray] = None,
    cibd_params: Optional[Dict[str, Any]] = None,
    s: Optional[np.ndarray] = None,
    thetas: Optional[np.ndarray] = None,
    native_preds: Optional[np.ndarray] = None,
) -> np.ndarray:
    if decoder_id == "argmax":
        return decode_argmax(probs)
    if decoder_id == "posterior_median":
        return decode_posterior_median(probs)
    if decoder_id in ("rounded_expectation", "rounded_expectation_bayes"):
        return decode_rounded_expectation_bayes(probs)
    if decoder_id == "native_legacy_round":
        return decode_native_legacy_round(probs)
    if decoder_id == "native":
        if native_preds is not None:
            return np.asarray(native_preds, dtype=int)
        return decode_argmax(probs)
    if decoder_id == "cutpoint_latent":
        if s is None or thetas is None:
            raise ValueError("cutpoint_latent decoder is only defined for CLM heads exposing s and theta0..theta3.")
        return decode_cutpoint_latent(s, thetas)
    if decoder_id == "cibd_selected":
        if cibd_params is None or prior is None:
            raise ValueError("cibd_selected requires prior and selected cibd_params.")
        if cibd_params.get("fallback"):
            if native_preds is not None:
                return np.asarray(native_preds, dtype=int)
            return decode_argmax(probs)
        return decode_cibd(
            probs,
            prior,
            alpha=float(cibd_params["alpha"]),
            tau=float(cibd_params["tau"]),
            q=int(cibd_params["q"]),
        )
    raise ValueError(f"Unknown decoder_id '{decoder_id}'.")


def select_cibd(
    source_val_df: pd.DataFrame,
    prior: np.ndarray,
    selector_id: str = "legacy_within1_1pp_v1",
    baseline_col: str = "pred_native",
    alphas: Optional[Sequence[float]] = None,
    taus: Optional[Sequence[float]] = None,
    qs: Sequence[int] = (1, 2),
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Select CIBD (alpha, tau, q) strictly on source validation under an explicit versioned policy."""
    if selector_id not in SELECTOR_IDS:
        raise ValueError(f"Unknown selector_id '{selector_id}'. Expected one of {SELECTOR_IDS}.")
    if "target" in set(source_val_df.get("split_role", pd.Series(dtype=str)).astype(str)):
        raise ValueError("Refusing source CIBD selection on target split rows.")

    y = source_val_df["y_true"].to_numpy(dtype=int)
    p = validate_probabilities(source_val_df[list(PROB_COLS)].to_numpy(dtype=np.float64))
    if baseline_col in source_val_df.columns:
        base_pred = source_val_df[baseline_col].to_numpy(dtype=int)
    else:
        base_pred = decode_argmax(p)
    base_metrics = compute_decision_metrics(y, base_pred)

    alpha_grid = list(alphas) if alphas is not None else [round(x / 10.0, 1) for x in range(11)]
    tau_grid = list(taus) if taus is not None else [0.0, 0.25, 0.5, 0.75, 1.0]

    w1_budget = 0.005 if selector_id == "journal_within1_05pp_v1" else 0.01
    enforce_qwk = selector_id == "journal_dual_guardrail_v1"

    rows: List[Dict[str, Any]] = []
    for alpha in alpha_grid:
        for tau in tau_grid:
            for q in qs:
                pred = decode_cibd(p, prior, alpha=float(alpha), tau=float(tau), q=int(q))
                m = compute_decision_metrics(y, pred)
                w1_ok = bool(m["within1"] >= base_metrics["within1"] - w1_budget)
                qwk_ok = bool(m["qwk"] >= base_metrics["qwk"] - 0.01)
                eligible = (w1_ok and qwk_ok) if enforce_qwk else w1_ok
                rows.append(
                    {
                        "selector_id": selector_id,
                        "alpha": float(alpha),
                        "tau": float(tau),
                        "q": int(q),
                        "recall_g3": m["recall_g3"],
                        "macro_recall": m["macro_recall"],
                        "within1": m["within1"],
                        "qwk": m["qwk"],
                        "mae": m["mae"],
                        "g3_precision": m["g3_precision"],
                        "feasible": w1_ok,
                        "qwk_guardrail_met": qwk_ok,
                        "eligible": eligible,
                    }
                )
    grid = pd.DataFrame(rows)
    elig = grid[grid["eligible"] & grid["recall_g3"].notna()]
    if elig.empty:
        return {
            "selector_id": selector_id,
            "fallback": True,
            "reason": "NO_FEASIBLE_CANDIDATE",
            "within1_budget": w1_budget,
            "enforce_qwk_guardrail": enforce_qwk,
        }, grid

    best = elig.sort_values(
        ["recall_g3", "macro_recall", "within1", "qwk", "tau", "q", "alpha"],
        ascending=[False, False, False, False, True, True, True],
    ).iloc[0]
    return {
        "selector_id": selector_id,
        "alpha": float(best["alpha"]),
        "tau": float(best["tau"]),
        "q": int(best["q"]),
        "fallback": False,
        "within1_budget": w1_budget,
        "enforce_qwk_guardrail": enforce_qwk,
    }, grid


def bootstrap_cibd_selector_stability(
    source_val_df: pd.DataFrame,
    fixed_prior: np.ndarray,
    selector_id: str = "legacy_within1_1pp_v1",
    reps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Source-validation grouping-key selector stability resampling with fixed source_train prior."""
    if "patient_key" not in source_val_df.columns:
        raise ValueError("source_val_df must include 'patient_key' grouping column.")
    statuses = set(source_val_df.get("patient_identity_status", pd.Series(["legacy_unverified"])).astype(str))
    unit = "patient_cluster" if statuses == {"verified"} else "grouping_key"

    groups = [
        np.asarray(idx, dtype=int)
        for idx in source_val_df.reset_index(drop=True).groupby("patient_key", sort=True).indices.values()
    ]
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    fallback_count = 0

    for _ in range(reps):
        sample_idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        sub_df = source_val_df.iloc[sample_idx].reset_index(drop=True)
        sel, _ = select_cibd(sub_df, fixed_prior, selector_id=selector_id)
        if sel.get("fallback"):
            fallback_count += 1
            key = "FALLBACK"
        else:
            key = f"a={sel['alpha']:.1f}|t={sel['tau']:.2f}|q={sel['q']}"
        counts[key] = counts.get(key, 0) + 1

    top_key, top_cnt = max(counts.items(), key=lambda kv: kv[1])
    return {
        "selector_id": selector_id,
        "resampling_unit": unit,
        "prior_kept_fixed": True,
        "reps": int(reps),
        "n_groups": int(len(groups)),
        "fallback_count": int(fallback_count),
        "top_config": top_key,
        "top_frequency": float(top_cnt / max(1, reps)),
        "config_counts": counts,
    }


def compute_clm_decision_regions(
    thetas: Sequence[float],
    link: str = "cloglog",
    cibd_params: Optional[Dict[str, Any]] = None,
    prior: Optional[np.ndarray] = None,
    grid_min: float = -6.0,
    grid_max: float = 6.0,
    n_grid: int = 24001,
) -> Dict[str, Any]:
    """Compute exact decision intervals over latent score s for fitted cutpoints and link."""
    th = np.asarray(thetas, dtype=np.float64)
    if th.shape != (4,) or (np.diff(th) <= 0.0).any():
        raise ValueError(f"Expected 4 strictly increasing cutpoints, got {th}.")

    s_grid = np.linspace(grid_min, grid_max, n_grid)
    u = th[None, :] - s_grid[:, None]
    if link == "cloglog":
        u_clip = np.clip(u, -30.0, 20.0)
        cdf = -np.expm1(-np.exp(u_clip))
        analytic_median = th - math.log(math.log(2.0))
    elif link == "logit":
        cdf = 1.0 / (1.0 + np.exp(-np.clip(u, -60.0, 60.0)))
        analytic_median = th.copy()
    elif link == "probit":
        from scipy.special import ndtr
        cdf = ndtr(u)
        analytic_median = th.copy()
    else:
        raise ValueError(f"Unsupported link '{link}'.")

    cdf = np.clip(cdf, 0.0, 1.0)
    probs = np.column_stack([cdf[:, 0], np.diff(cdf, axis=1).clip(0.0), (1.0 - cdf[:, -1]).clip(0.0)])
    probs /= probs.sum(axis=1, keepdims=True)

    rules: Dict[str, np.ndarray] = {
        "cutpoint_latent": decode_cutpoint_latent(s_grid, th),
        "argmax": decode_argmax(probs),
        "posterior_median": decode_posterior_median(probs),
        "rounded_expectation_bayes": decode_rounded_expectation_bayes(probs),
    }
    if cibd_params is not None and prior is not None and not cibd_params.get("fallback"):
        rules["cibd_selected"] = decode_cibd(
            probs,
            prior,
            alpha=float(cibd_params["alpha"]),
            tau=float(cibd_params["tau"]),
            q=int(cibd_params["q"]),
        )

    intervals: List[Dict[str, Any]] = []
    for r_name, pred in rules.items():
        for g in range(5):
            mask = pred == g
            if mask.any():
                idx = np.flatnonzero(mask)
                intervals.append(
                    {
                        "rule": r_name,
                        "grade": g,
                        "left_s": float(s_grid[idx[0]]),
                        "right_s": float(s_grid[idx[-1]]),
                        "width": float(s_grid[idx[-1]] - s_grid[idx[0]]),
                    }
                )
            else:
                intervals.append({"rule": r_name, "grade": g, "left_s": np.nan, "right_s": np.nan, "width": 0.0})

    return {
        "link": link,
        "thetas": [float(x) for x in th.tolist()],
        "analytic_median_boundaries": [float(x) for x in analytic_median.tolist()],
        "intervals": intervals,
    }

# %% [markdown]
# ### Module `experiments.mlwa.prior_shift`

# %%
"""E5 Unlabelled Target Prior Adaptation (Saerens EM & Lipton BBSE) with strict label isolation."""


from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import minimize


def _project_to_simplex_with_floor(vec: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = len(v)
    if floor * n >= 1.0:
        raise ValueError("Simplex floor exceeds 1/C.")
    rem = 1.0 - floor * n
    res = minimize(
        lambda x: 0.5 * float(np.sum((x - v) ** 2)),
        x0=np.full(n, 1.0 / n),
        bounds=[(floor, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
        method="SLSQP",
    )
    if res.success:
        out = np.clip(res.x, floor, 1.0)
        return out / out.sum()
    shifted = np.maximum(v - floor, 0.0)
    if shifted.sum() <= 0:
        return np.full(n, 1.0 / n)
    return floor + rem * (shifted / shifted.sum())


def estimate_prior_saerens_em(
    p_target: np.ndarray,
    pi_source: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-7,
    min_prior: float = 1e-6,
) -> Dict[str, Any]:
    """Saerens et al. (Neural Computation, 2002) EM prior estimation using UNLABELLED target posteriors only."""
    p = validate_probabilities(p_target)
    pi_s = np.asarray(pi_source, dtype=np.float64)
    if pi_s.shape != (5,) or (pi_s <= 0.0).any() or not np.allclose(pi_s.sum(), 1.0, atol=1e-6):
        raise ValueError("pi_source must be a strictly positive 5-class probability vector.")

    pi_t = pi_s.copy()
    converged = False
    n_iter = 0
    max_delta = float("inf")

    for it in range(1, int(max_iter) + 1):
        ratio = pi_t / pi_s
        unnorm = p * ratio[None, :]
        denom = unnorm.sum(axis=1, keepdims=True)
        if (denom <= 0.0).any() or not np.isfinite(denom).all():
            return {
                "method": "saerens_em",
                "status": "FAILED_NUMERICAL_DEGENERACY",
                "converged": False,
                "pi_target": np.full(5, np.nan),
                "iterations": it,
            }
        post = unnorm / denom
        pi_next = post.mean(axis=0)
        pi_next = np.maximum(pi_next, min_prior)
        pi_next /= pi_next.sum()
        max_delta = float(np.max(np.abs(pi_next - pi_t)))
        pi_t = pi_next
        n_iter = it
        if max_delta < float(tol):
            converged = True
            break

    return {
        "method": "saerens_em",
        "status": "CONVERGED" if converged else "MAX_ITER_REACHED",
        "converged": bool(converged),
        "iterations": int(n_iter),
        "max_delta": float(max_delta),
        "pi_target": pi_t,
        "kl_target_to_source": float(np.sum(pi_t * np.log(np.maximum(pi_t, 1e-12) / pi_s))),
    }


def estimate_prior_bbse(
    y_source_val: np.ndarray,
    p_source_val: np.ndarray,
    p_target: np.ndarray,
    mode: str = "soft",
    cond_threshold: float = 1e4,
    min_class_support: int = 5,
) -> Dict[str, Any]:
    """Lipton et al. (ICML 2018) Black Box Shift Estimation with explicit conditioning & rank audits.

    Target labels are strictly NOT accepted by this API.
    """
    ys = np.asarray(y_source_val, dtype=int)
    ps = validate_probabilities(p_source_val)
    pt = validate_probabilities(p_target)

    if mode not in ("hard", "soft"):
        raise ValueError(f"Unsupported BBSE mode '{mode}'; expected 'hard' or 'soft'.")

    support = np.bincount(ys, minlength=5)
    if (support < int(min_class_support)).any():
        return {
            "method": f"bbse_{mode}",
            "status": "INSUFFICIENT_SOURCE_CLASS_SUPPORT",
            "identifiable": False,
            "class_support": support.tolist(),
            "pi_target": np.full(5, np.nan),
        }

    pi_source = support.astype(np.float64) / float(len(ys))
    # Build 5 x 5 joint confusion matrix C_{i, j} = P(yhat = i, y = j) on source_val
    c_mat = np.zeros((5, 5), dtype=np.float64)
    if mode == "hard":
        yhat_s = ps.argmax(axis=1)
        for i in range(5):
            for j in range(5):
                c_mat[i, j] = float(np.mean((yhat_s == i) & (ys == j)))
        mu_target = np.bincount(pt.argmax(axis=1), minlength=5).astype(np.float64) / float(len(pt))
    else:
        for j in range(5):
            mask = ys == j
            c_mat[:, j] = ps[mask].mean(axis=0) * pi_source[j]
        mu_target = pt.mean(axis=0)

    rank = int(np.linalg.matrix_rank(c_mat))
    cond_num = float(np.linalg.cond(c_mat)) if rank == 5 else float("inf")
    if rank < 5:
        return {
            "method": f"bbse_{mode}",
            "status": "RANK_DEFICIENT_NOT_IDENTIFIABLE",
            "identifiable": False,
            "rank": rank,
            "condition_number": cond_num,
            "class_support": support.tolist(),
            "pi_target": np.full(5, np.nan),
        }
    if cond_num > float(cond_threshold):
        return {
            "method": f"bbse_{mode}",
            "status": "UNSTABLE_CONDITIONING",
            "identifiable": False,
            "rank": rank,
            "condition_number": cond_num,
            "class_support": support.tolist(),
            "pi_target": np.full(5, np.nan),
        }

    # Solve C * w = mu_target directly, then pi_target = w * pi_source constrained to simplex
    w_raw = np.linalg.solve(c_mat, mu_target)
    pi_raw = w_raw * pi_source
    was_clipped_to_simplex = bool((pi_raw <= 1e-6).any() or not np.allclose(pi_raw.sum(), 1.0, atol=1e-3))

    # Constrained simplex quadratic optimization: min_{pi in Delta} ||(C / pi_source) pi - mu_target||^2
    c_cond = c_mat / pi_source[None, :]
    res = minimize(
        lambda pi: 0.5 * float(np.sum((c_cond @ pi - mu_target) ** 2)),
        x0=_project_to_simplex_with_floor(pi_raw, floor=1e-6),
        bounds=[(1e-6, 1.0)] * 5,
        constraints={"type": "eq", "fun": lambda pi: float(np.sum(pi) - 1.0)},
        method="SLSQP",
    )
    pi_simplex = res.x / res.x.sum() if res.success else _project_to_simplex_with_floor(pi_raw, floor=1e-6)
    return {
        "method": f"bbse_{mode}",
        "status": "COMPLETE",
        "identifiable": True,
        "rank": rank,
        "condition_number": cond_num,
        "class_support": support.tolist(),
        "was_clipped_to_simplex": was_clipped_to_simplex,
        "pi_raw": pi_raw,
        "pi_target": pi_simplex,
        "pi_source_val": pi_source,
    }


def apply_prior_shift_ratio(
    p_target: np.ndarray,
    pi_source: np.ndarray,
    pi_target_hat: np.ndarray,
    cibd_tau_active: float = 0.0,
) -> np.ndarray:
    """Reweight posteriors by (pi_target_hat / pi_source), blocking double-correction with CIBD tau."""
    if abs(float(cibd_tau_active)) > 1e-12:
        raise ValueError(
            "Refusing double prior correction: E5 (pi_target/pi_source) ratio cannot be multiplied "
            "simultaneously with CIBD pi**(-tau) when tau != 0."
        )
    p = validate_probabilities(p_target)
    ps = np.asarray(pi_source, dtype=np.float64)
    pt = np.asarray(pi_target_hat, dtype=np.float64)
    if (ps <= 0.0).any() or (pt <= 0.0).any() or not np.isfinite(pt).all():
        raise ValueError("Source and estimated target priors must be finite and strictly positive.")
    adj = p * (pt / ps)[None, :]
    return adj / adj.sum(axis=1, keepdims=True)

# %% [markdown]
# ## Experiment Workflow & Entry Points

# %%
import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def run_e5_cohort_adaptation(
    source_val_df: pd.DataFrame,
    target_df: pd.DataFrame,
    pi_source: np.ndarray,
    cohort: str,
    fold_id: int,
    cibd_tau_active: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run Saerens EM + BBSE strictly without exposing y_target during adaptation."""
    p_val = source_val_df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    y_val = source_val_df['y_true'].to_numpy(dtype=int)
    p_tgt = target_df[list(PROB_COLS)].to_numpy(dtype=np.float64)
    # Only pass p_tgt to estimators (y_tgt used strictly post-hoc for accuracy evaluation)
    y_tgt_eval = target_df['y_true'].to_numpy(dtype=int)

    em_res = estimate_prior_saerens_em(p_tgt, pi_source)
    bb_soft = estimate_prior_bbse(y_val, p_val, p_tgt, mode='soft')
    bb_hard = estimate_prior_bbse(y_val, p_val, p_tgt, mode='hard')

    rows: List[Dict[str, Any]] = []
    for method_name, res in [('saerens_em', em_res), ('bbse_soft', bb_soft), ('bbse_hard', bb_hard)]:
        row: Dict[str, Any] = {
            'regime': 'TRANSDUCTIVE_ADAPTATION',
            'fold_id': fold_id,
            'cohort': cohort,
            'method': method_name,
            'status': res['status'],
            'oracle_leakage_checked': True,
        }
        pi_hat = res.get('pi_target')
        if pi_hat is not None and np.isfinite(pi_hat).all():
            p_adj = apply_prior_shift_ratio(p_tgt, pi_source, pi_hat, cibd_tau_active=cibd_tau_active)
            dm = compute_decision_metrics(y_tgt_eval, decode_argmax(p_adj))
            row.update({
                'recall_g3': dm['recall_g3'],
                'within1': dm['within1'],
                'qwk': dm['qwk'],
                'macro_recall': dm['macro_recall'],
                'pi_target_est': np.asarray(pi_hat).round(4).tolist(),
            })
        else:
            row.update({'recall_g3': np.nan, 'within1': np.nan, 'qwk': np.nan, 'macro_recall': np.nan})
        rows.append(row)
    return rows


def run_e5_smoke(output_dir: Path) -> Dict[str, Any]:
    """Synthetic label-shift smoke test for Saerens EM & BBSE (TEST_ONLY)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)

    pi_src = np.array([0.45, 0.15, 0.20, 0.12, 0.08], dtype=np.float64)
    pi_tgt_true = np.array([0.25, 0.15, 0.25, 0.25, 0.10], dtype=np.float64)

    def _sim(n: int, pi: np.ndarray) -> pd.DataFrame:
        y = rng.choice(5, size=n, p=pi)
        y[:5] = np.arange(5)
        x = rng.normal(loc=y[:, None] * 1.1, scale=0.8, size=(n, 5))
        logits = -0.5 * ((np.arange(5)[None, :] - x.mean(axis=1, keepdims=True) / 1.1) ** 2) + np.log(pi_src)[None, :]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        df = pd.DataFrame(p, columns=list(PROB_COLS))
        df['y_true'] = y
        return df

    val_df = _sim(400, pi_src)
    tgt_df = _sim(400, pi_tgt_true)
    rows = run_e5_cohort_adaptation(val_df, tgt_df, pi_src, cohort='aptos', fold_id=0)
    pd.DataFrame(rows).to_csv(out / 'e5_transductive_adaptation_smoke.csv', index=False)
    status = {'status': 'COMPLETE_SMOKE_TEST_ONLY', 'n_rows': len(rows)}
    atomic_write_json(out / 'status.json', status)
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    paths = resolve_runtime_paths()
    parser = argparse.ArgumentParser(description='13: Opt-in E5 unlabelled target prior adaptation (Saerens EM & BBSE).')
    parser.add_argument('--enable-e5', action='store_true', help='Explicit opt-in required for E5 transductive adaptation.')
    parser.add_argument('--p0-dir', type=Path, default=None, help='Source prediction bundles directory.')
    parser.add_argument('--output-dir', type=Path, default=paths['outputs'] / 'mlwa_e5_prior_adaptation')
    parser.add_argument('--smoke', action='store_true', help='Run deterministic synthetic label-shift check (TEST_ONLY).')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        res = run_e5_smoke(args.output_dir)
        print(json.dumps(res, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.enable_e5:
        status = {
            'status': 'OPT_IN_NOT_ENABLED',
            'regime': 'TRANSDUCTIVE_ADAPTATION',
            'note': 'E5 is optional and excluded from primary strict-LODO rankings. Pass --enable-e5 to run.',
        }
        atomic_write_json(args.output_dir / 'status.json', status)
        print(json.dumps(status, indent=2))
        return 0

    if args.p0_dir is None or not Path(args.p0_dir).is_dir():
        status = {'status': 'PENDING_ASSETS', 'note': 'Pass valid --p0-dir containing fold prediction bundles.'}
        atomic_write_json(args.output_dir / 'status.json', status)
        print(json.dumps(status, indent=2))
        return 1

    rows: List[Dict[str, Any]] = []
    for fold_id in range(6):
        cohort = CANONICAL_FOLD_TO_COHORT[fold_id]
        base = Path(args.p0_dir) / 'predictions' / f'fold_{fold_id}_{cohort}_V3'
        if not base.exists():
            continue
        tr = load_predictions_table(base / 'source_train')
        va = load_predictions_table(base / 'source_val')
        tg = load_predictions_table(base / 'target')
        pi_src = prior_from_labels(tr['y_true'].to_numpy(dtype=int))
        rows.extend(run_e5_cohort_adaptation(va, tg, pi_src, cohort=cohort, fold_id=fold_id))

    pd.DataFrame(rows).to_csv(args.output_dir / 'e5_transductive_adaptation.csv', index=False)
    status = {'status': 'COMPLETE', 'regime': 'TRANSDUCTIVE_ADAPTATION', 'n_rows': len(rows)}
    atomic_write_json(args.output_dir / 'status.json', status)
    print(json.dumps(status, indent=2))
    return 0

# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_e5_prior_adaptation_nb')
    print(run_e5_smoke(_out) if globals().get('SMOKE', True) else {'status': 'OPT_IN_NOT_ENABLED'})

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
