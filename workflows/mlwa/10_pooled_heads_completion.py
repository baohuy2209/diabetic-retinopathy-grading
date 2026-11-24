# %% [markdown]
# # 10 · Historical Pooled Five-Head Audit & Strict Reconstruction (`PENDING_ASSETS` Aware)
# 
# > **Standalone Single-File Notebook**: All required modules are embedded directly below. Upload only this `.ipynb` file to Google Colab and run all cells (`Runtime -> Run all`); no repository clone or zip bundle is required.

# %%
NOTEBOOK_CODE_SHA256 = '27fb1c072e5ab1869430c1c7e4c105c015e446b1bb3a25a09b360ecd42c4710c'

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
        deps = {
            'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn',
            'pyarrow': 'pyarrow', 'torch': 'torch', 'torchvision': 'torchvision',
        }
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')
    return Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _setup_colab_environment()
    MODE = 'audit'  # 'audit' | 'smoke'
    OUTPUT_DIR = None
    SMOKE = True

# %% [markdown]
# ## Embedded Standalone Implementation (16 self-contained modules)

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
# ### Module `experiments.mlwa.config`

# %%
"""Validated configuration dataclasses and deterministic hashing for MLWA journal runs."""


import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


VALID_HEAD_TYPES = ("softmax", "clm", "coral", "corn")
VALID_CLM_LINKS = ("none", "cloglog", "logit", "probit")
VALID_ARCH_IDS = ("dual_branch_mil_v1", "global_resnet50_v1")
VALID_OBJECTIVES = (
    "softmax_ce",
    "softmax_ce_qwk",
    "clm_nll",
    "clm_nll_qwk",
    "cui_cb_ce",
    "menon_logit_adj_ce",
    "clm_cui_weighted_nll",
    "clm_logprob_adjusted_nll",
    "galdran_cost_sensitive_ce",
    "coral_bce",
    "corn_bce",
)
VALID_SAMPLING_POLICIES = ("natural_source_train", "quota_balanced")
VALID_DECODERS = ("argmax", "posterior_median", "rounded_expectation_bayes", "native_legacy_round")


@dataclass(frozen=True)
class SeedConfig:
    """Explicit separation of split, training, resampling, and probe-control seeds."""

    seed_train: int = 0
    seed_split: int = 42
    seed_resampling: int = 42
    seed_probe_control: Tuple[int, ...] = (0, 1, 2)

    def validate(self) -> None:
        if self.seed_train < 0 or self.seed_split < 0 or self.seed_resampling < 0:
            raise ValueError("All seeds must be non-negative integers.")
        if not self.seed_probe_control or any(s < 0 for s in self.seed_probe_control):
            raise ValueError("seed_probe_control must contain non-negative integer seeds.")


@dataclass(frozen=True)
class ModelSpec:
    """Architecture, head, link, and preprocessing contract (distinct from objective and decoder)."""

    architecture_id: str = "dual_branch_mil_v1"
    backbone_name: str = "resnet50"
    head_type: str = "softmax"
    clm_link: str = "none"
    pretrained: bool = True
    use_local_branch: bool = True
    use_nonlocal: bool = True
    use_quadrant_tokens: bool = True
    use_cbam: bool = True
    fusion_dim: int = 512
    dropout: float = 0.3
    eval_patch_chunk_size: Optional[int] = 32
    preprocessing_id: str = "ben_graham_green_512_fov15_p128_s96"

    def validate(self) -> None:
        if self.architecture_id not in VALID_ARCH_IDS:
            raise ValueError(f"Unsupported architecture_id '{self.architecture_id}'.")
        if self.head_type not in VALID_HEAD_TYPES:
            raise ValueError(f"Unsupported head_type '{self.head_type}'.")
        if self.head_type == "clm" and self.clm_link not in ("cloglog", "logit", "probit"):
            raise ValueError(f"CLM head requires clm_link in cloglog|logit|probit, got '{self.clm_link}'.")
        if self.head_type != "clm" and self.clm_link != "none":
            raise ValueError(f"Non-CLM head '{self.head_type}' must have clm_link='none', got '{self.clm_link}'.")
        if self.architecture_id == "global_resnet50_v1" and self.use_local_branch:
            raise ValueError("global_resnet50_v1 cannot enable use_local_branch=True.")
        if self.fusion_dim <= 0 or (self.eval_patch_chunk_size is not None and self.eval_patch_chunk_size <= 0):
            raise ValueError("fusion_dim and eval_patch_chunk_size must be positive.")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Explicit training objective specification preventing silent QWK or double-rebalancing bugs."""

    objective_id: str = "softmax_ce"
    use_qwk: bool = False
    target_qwk_weight: float = 0.4
    qwk_warmup_epochs: int = 15
    qwk_schedule_mode: str = "linear"
    cui_beta: float = 0.9999
    logit_adj_tau_train: float = 1.0
    galdran_lambda: float = 1.0
    galdran_power: int = 2
    allow_combined_rebalancing: bool = False

    def validate(self, model: ModelSpec, sampling_policy: str) -> None:
        if self.objective_id == "exp_mse":
            raise NotImplementedError(
                "NEEDS_METHOD_SPEC: verify exact Stelter et al. (NLDL 2026) exp_MSE formula "
                "from primary PMLR publication before enabling."
            )
        if self.objective_id not in VALID_OBJECTIVES:
            raise ValueError(f"Unsupported objective_id '{self.objective_id}'.")

        is_softmax_obj = self.objective_id in (
            "softmax_ce",
            "softmax_ce_qwk",
            "cui_cb_ce",
            "menon_logit_adj_ce",
            "galdran_cost_sensitive_ce",
        )
        is_clm_obj = self.objective_id in (
            "clm_nll",
            "clm_nll_qwk",
            "clm_cui_weighted_nll",
            "clm_logprob_adjusted_nll",
        )
        if is_softmax_obj and model.head_type != "softmax":
            raise ValueError(
                f"Objective '{self.objective_id}' requires head_type='softmax' (got '{model.head_type}'). "
                "Do not silently apply Softmax logit adjustment/CE to scalar CLM outputs."
            )
        if is_clm_obj and model.head_type != "clm":
            raise ValueError(f"Objective '{self.objective_id}' requires head_type='clm', got '{model.head_type}'.")
        if self.objective_id == "coral_bce" and model.head_type != "coral":
            raise ValueError("coral_bce requires head_type='coral'.")
        if self.objective_id == "corn_bce" and model.head_type != "corn":
            raise ValueError("corn_bce requires head_type='corn'.")

        if self.objective_id in ("softmax_ce_qwk", "clm_nll_qwk"):
            if not self.use_qwk or self.target_qwk_weight <= 0.0:
                raise ValueError(f"QWK-enabled objective '{self.objective_id}' requires use_qwk=True and target_qwk_weight > 0.")
        elif self.use_qwk:
            raise ValueError(f"Pure objective '{self.objective_id}' must set use_qwk=False (no silent QWK injection).")

        if not (0.0 <= self.cui_beta < 1.0):
            raise ValueError(f"cui_beta must lie in [0, 1), got {self.cui_beta}.")
        if self.qwk_schedule_mode not in ("linear", "fixed"):
            raise ValueError(f"Unsupported qwk_schedule_mode '{self.qwk_schedule_mode}'.")

        is_loss_rebalanced = self.objective_id in (
            "cui_cb_ce",
            "menon_logit_adj_ce",
            "clm_cui_weighted_nll",
            "clm_logprob_adjusted_nll",
        )
        if is_loss_rebalanced and sampling_policy == "quota_balanced" and not self.allow_combined_rebalancing:
            raise ValueError(
                f"Refusing unintended double rebalancing: objective '{self.objective_id}' with "
                f"sampling_policy='quota_balanced' requires allow_combined_rebalancing=True."
            )


@dataclass(frozen=True)
class TrainingSpec:
    """Shared optimization, batching, warmup, and checkpoint-selection settings."""

    epochs: int = 30
    batch_size: int = 12
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    cosine_t_max: int = 30
    min_lr: float = 1e-6
    patience: int = 15
    grad_clip: float = 1.0
    amp_mode: str = "auto"
    checkpoint_selection_decoder: str = "argmax"
    sampling_policy: str = "natural_source_train"
    quota_target_per_class: int = 2000
    allow_grad_accum_qwk_mismatch: bool = False

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size, grad_accum_steps, and patience must be positive.")
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("lr must be > 0 and weight_decay >= 0.")
        if self.checkpoint_selection_decoder not in VALID_DECODERS:
            raise ValueError(f"Unsupported checkpoint_selection_decoder '{self.checkpoint_selection_decoder}'.")
        if self.sampling_policy not in VALID_SAMPLING_POLICIES:
            raise ValueError(f"Unsupported sampling_policy '{self.sampling_policy}'.")
        if self.grad_accum_steps > 1 and not self.allow_grad_accum_qwk_mismatch:
            raise ValueError(
                "grad_accum_steps > 1 alters BatchNorm and batch-coupled QWK observation batch size relative to "
                "physical batch_size; set allow_grad_accum_qwk_mismatch=True explicitly if intended."
            )


@dataclass(frozen=True)
class JournalExperimentConfig:
    """Top-level immutable configuration unit for one MLWA journal training/eval job."""

    study_id: str = "mlwa_dr_ordinal_extension"
    protocol_id: str = "mlwa_journal_v1"
    run_id: str = "journal_run_001"
    experiment: str = "E2"
    config_id: str = "E2_A_softmax_ce"
    fold_id: int = 0
    cohort: str = "aptos"
    seeds: SeedConfig = field(default_factory=SeedConfig)
    model: ModelSpec = field(default_factory=ModelSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    is_test_only: bool = False

    def validate(self) -> None:
        if self.fold_id not in CANONICAL_FOLD_TO_COHORT:
            raise ValueError(f"Invalid fold_id={self.fold_id}; expected 0..5.")
        expected_cohort = CANONICAL_FOLD_TO_COHORT[self.fold_id]
        if self.cohort != expected_cohort:
            raise ValueError(
                f"Mismatched fold_id={self.fold_id} and cohort='{self.cohort}'; "
                f"canonical LODO cohort is '{expected_cohort}'."
            )
        self.seeds.validate()
        self.model.validate()
        self.training.validate()
        self.objective.validate(self.model, self.training.sampling_policy)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save_json(self, path: Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JournalExperimentConfig":
        cfg = cls(
            study_id=raw["study_id"],
            protocol_id=raw["protocol_id"],
            run_id=raw["run_id"],
            experiment=raw["experiment"],
            config_id=raw["config_id"],
            fold_id=int(raw["fold_id"]),
            cohort=str(raw["cohort"]),
            seeds=SeedConfig(
                seed_train=int(raw["seeds"]["seed_train"]),
                seed_split=int(raw["seeds"]["seed_split"]),
                seed_resampling=int(raw["seeds"]["seed_resampling"]),
                seed_probe_control=tuple(int(x) for x in raw["seeds"]["seed_probe_control"]),
            ),
            model=ModelSpec(**raw["model"]),
            objective=ObjectiveSpec(**raw["objective"]),
            training=TrainingSpec(**raw["training"]),
            is_test_only=bool(raw.get("is_test_only", False)),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load_json(cls, path: Path) -> "JournalExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


def build_factorial_cell_spec(cell: str, pretrained: bool = True) -> Tuple[str, ModelSpec, ObjectiveSpec]:
    """Return canonical ModelSpec and ObjectiveSpec for E2 cells A, B, C, or D."""
    c = cell.strip().upper()
    if c == "A":
        return (
            "E2_A_softmax_ce",
            ModelSpec(head_type="softmax", clm_link="none", pretrained=pretrained),
            ObjectiveSpec(objective_id="softmax_ce", use_qwk=False),
        )
    if c == "B":
        return (
            "E2_B_softmax_ce_qwk",
            ModelSpec(head_type="softmax", clm_link="none", pretrained=pretrained),
            ObjectiveSpec(objective_id="softmax_ce_qwk", use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15),
        )
    if c == "C":
        return (
            "E2_C_clm_cloglog_nll",
            ModelSpec(head_type="clm", clm_link="cloglog", pretrained=pretrained),
            ObjectiveSpec(objective_id="clm_nll", use_qwk=False),
        )
    if c == "D":
        return (
            "E2_D_clm_cloglog_nll_qwk",
            ModelSpec(head_type="clm", clm_link="cloglog", pretrained=pretrained),
            ObjectiveSpec(objective_id="clm_nll_qwk", use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15),
        )
    raise ValueError(f"Unknown E2 cell '{cell}'. Expected A, B, C, or D.")

# %% [markdown]
# ### Module `models.fusion`

# %%
"""Multi-scale dual-branch feature fusion module.

Combines high-level global context representations (e.g., from ResNet-50)
with fine-grained local lesion features (e.g., from EfficientNet-B0 MIL)
using a robust projection MLP with LayerNorm to prevent batch-size-1 instability.
"""

from typing import Optional

import torch
import torch.nn as nn


class DualBranchFusion(nn.Module):
    """Fuses global context and local lesion bag representations into a shared latent space.

    Architecture:
        [f_global, f_local] -> Linear(in_dim, fusion_dim) -> LayerNorm -> Mish
        -> Dropout(p) -> Linear(fusion_dim, fusion_dim) -> LayerNorm

    LayerNorm is used instead of BatchNorm to ensure numerical stability even
    when the batch size is 1 (e.g. at the end of a dataset or during inference).
    """

    def __init__(
        self,
        global_dim: int = 4096,
        local_dim: int = 2560,
        fusion_dim: int = 512,
        use_local_branch: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.fusion_dim = fusion_dim
        self.use_local_branch = use_local_branch

        combined_dim = global_dim + local_dim if use_local_branch else global_dim

        self.projection = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        f_global: torch.Tensor,
        f_local: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Projects and fuses features from global and local branches.

        Args:
            f_global: Tensor of shape (B, global_dim).
            f_local: Optional tensor of shape (B, local_dim). Required if
                use_local_branch is True.

        Returns:
            Latent representation tensor of shape (B, fusion_dim).
        """
        if self.use_local_branch and f_local is not None:
            f_fused = torch.cat([f_global, f_local], dim=1)
        else:
            f_fused = f_global

        return self.projection(f_fused)

# %% [markdown]
# ### Module `losses.qwk`

# %%
"""Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss.

Computes continuous soft confusion matrices to directly optimize clinical
ordinal agreement:
    QWK = 1 - (sum(W * O) / sum(W * E))
    Loss = sum(W * O) / (sum(W * E) + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousQWKLoss(nn.Module):
    """Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss.

    Minimizing (sum(W * O) / sum(W * E)) directly maximizes Cohen's quadratic
    weighted kappa agreement between soft predicted distributions and targets.
    """

    def __init__(self, num_classes: int = 5, eps: float = 1e-7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        # Quadratic penalty matrix: W_ij = (i - j)^2 / (C - 1)^2
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

        self.register_buffer("W", w)

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation of differentiable QWK loss.

        Args:
            pred_probs: Continuous class probabilities (B, num_classes).
            targets: Integer class indices (B,) or one-hot distributions (B, num_classes).

        Returns:
            Scalar differentiable loss tensor.
        """
        device = pred_probs.device
        n = pred_probs.size(0)
        if n == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()

        # Normalize predicted probabilities to simplex
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)

        # Soft observed confusion matrix: O = (P^T * Y) / N
        observed = torch.matmul(pred_probs.t(), y_one_hot) / (n + self.eps)

        # Expected confusion matrix under chance: E = hist_pred^T * hist_true
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (n + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (n + self.eps)
        expected = torch.matmul(hist_pred.t(), hist_true)

        penalty = self.W.to(device)
        num = torch.sum(penalty * observed)
        den = torch.sum(penalty * expected) + self.eps

        return num / den

# %% [markdown]
# ### Module `models.heads.clm`

# %%
"""Cumulative Link Model (CLM) ordinal head with complementary log-log link.

Mathematical Formulation:
    - Latent Severity Score: s(x) = W^T x
    - Strictly Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4 parameterized
      via softplus increments: theta_k = theta_{k-1} + softplus(alpha_k) + 1e-4
    - Complementary Log-Log (clog-log) link: F(z) = 1 - exp(-exp(z))
    - Cumulative Probability: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
    - Rank Probability: P(Y > k) = 1 - P(Y <= k)
    - Interval Probability: P(Y = k) = P(Y <= k) - P(Y <= k-1)
    - Loss: Negative Log-Likelihood (NLL) + lambda * Continuous_QWK_Loss
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeLinkModelQWKHead(nn.Module):
    """Cumulative Link Model (CLM) head with softplus cutpoints and clog-log link.
    
    Guarantees strict numerical stability under bfloat16/float16 mixed precision by executing
    monotonic cutpoints, clog-log link calculations, and NLL losses in full FP32.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = qwk_weight

        self.score_fc = nn.Linear(in_features, 1, bias=False)
        nn.init.zeros_(self.score_fc.weight)

        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)

        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float) -> None:
        """Dynamically updates the lambda weight for continuous QWK loss during warmup."""
        self.current_qwk_weight = float(weight)

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        """Constructs strictly monotonic cutpoints using softplus increments."""
        increments = F.softplus(self.cutpoint_increments) + 1e-4
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass computing severity scores, cumulative probabilities, and predictions."""
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            x_f32 = x.float()
            b = x_f32.size(0)
            severity_score = self.score_fc(x_f32).squeeze(1)  # (B,)
            cutpoints = self._get_monotonic_cutpoints()       # (4,)

            z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)  # (B, 4)
            z_safe = torch.clamp(z, min=-15.0, max=15.0)

            # S(z) = exp(-exp(z)), P(Y <= k) = 1 - S(z)
            surv = torch.exp(-torch.exp(z_safe))
            cum_probs = 1.0 - surv
            rank_probs = surv

            probs = torch.zeros(b, self.num_classes, device=x.device, dtype=torch.float32)
            probs[:, 0] = cum_probs[:, 0]
            for k in range(1, self.num_cutpoints):
                probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
            probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])

            probs_sum = probs.sum(dim=1, keepdim=True)
            probs = torch.where(probs_sum > 0, probs / probs_sum, torch.full_like(probs, 1.0 / self.num_classes))

            class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
            expected_grade = torch.sum(probs * class_indices, dim=1)
            preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()

        return {
            "severity_score": severity_score,
            "latent_score": severity_score,
            "cutpoints": cutpoints,
            "thetas": cutpoints,
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes hybrid loss: NLL + lambda * Continuous_QWK."""
        probs = outputs["probs"].float()
        b = targets.size(0)
        eps = 1e-12

        target_probs = probs[torch.arange(b, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))

        if class_weights is not None:
            weights_per_sample = class_weights[targets].float()
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()

        qwk = self.qwk_loss(probs, targets)
        return nll_loss + self.current_qwk_weight * qwk

# %% [markdown]
# ### Module `losses.ordinal`

# %%
"""Ordinal classification and ranking loss functions.

Includes:
- CORAL loss: Multi-task binary cross-entropy across rank indicators.
- CORN loss: Conditional subset binary cross-entropy across forward transitions.
- Ordinal NLL loss: Negative log-likelihood over reconstructed categorical probabilities.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_loss(
    rank_logits: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 4,
) -> torch.Tensor:
    """Computes binary cross-entropy loss for Consistent Rank Logits (CORAL).

    Args:
        rank_logits: Tensor of shape (B, num_thresholds).
        targets: Integer target tensor of shape (B,).
        num_thresholds: Number of binary cutpoints (num_classes - 1).

    Returns:
        Scalar binary cross-entropy loss.
    """
    batch_size = targets.size(0)
    device = targets.device

    binary_targets = torch.zeros(batch_size, num_thresholds, device=device)
    for k in range(num_thresholds):
        binary_targets[:, k] = (targets > k).float()

    return F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")


def corn_loss(
    cond_logits: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 4,
) -> torch.Tensor:
    """Computes conditional binary cross-entropy loss for CORN.

    Each sub-task k only considers samples where target >= k.

    Args:
        cond_logits: Tensor of shape (B, num_thresholds).
        targets: Integer target tensor of shape (B,).
        num_thresholds: Number of conditional tasks (num_classes - 1).

    Returns:
        Scalar conditional binary cross-entropy loss.
    """
    loss = torch.tensor(0.0, device=targets.device)
    valid_tasks = 0

    for k in range(num_thresholds):
        mask = targets >= k
        if mask.sum() > 0:
            subset_logits = cond_logits[mask, k]
            subset_targets = (targets[mask] > k).float()
            task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
            loss = loss + task_loss
            valid_tasks += 1

    return loss / max(1, valid_tasks)


def ordinal_nll_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes negative log-likelihood on ordinal categorical probabilities.

    Args:
        probs: Categorical probabilities (B, num_classes).
        targets: Integer labels (B,).
        class_weights: Optional per-class weight tensor (num_classes,).
        eps: Minimum probability value for clamping to prevent log(0).

    Returns:
        Scalar NLL loss.
    """
    batch_size = targets.size(0)
    target_probs = probs[torch.arange(batch_size, device=targets.device), targets]
    nll = -torch.log(torch.clamp(target_probs, min=eps))

    if class_weights is not None:
        weights = class_weights[targets]
        return (nll * weights).mean()
    return nll.mean()


class CORALLoss(nn.Module):
    """Module wrapper for CORAL ordinal binary cross-entropy loss."""

    def __init__(self, num_thresholds: int = 4) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds

    def forward(self, rank_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return coral_loss(rank_logits, targets, num_thresholds=self.num_thresholds)


class CORNLoss(nn.Module):
    """Module wrapper for CORN conditional binary cross-entropy loss."""

    def __init__(self, num_thresholds: int = 4) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds

    def forward(self, cond_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return corn_loss(cond_logits, targets, num_thresholds=self.num_thresholds)


class OrdinalNLLLoss(nn.Module):
    """Module wrapper for ordinal negative log-likelihood loss."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, eps: float = 1e-7) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ordinal_nll_loss(probs, targets, class_weights=self.class_weights, eps=self.eps)

# %% [markdown]
# ### Module `models.heads.coral`

# %%
"""Consistent Rank Logits (CORAL) head for ordinal regression.

Reference:
    Cao, Mirjalili, & Raschka (2020). Rank consistent ordinal regression for neural networks.
    Pattern Recognition Letters, 140, 325-331.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALHead(nn.Module):
    """Consistent Rank Logits (CORAL) output head.

    Features:
        - Shared weight vector W across all binary threshold tasks (parallel hyperplanes).
        - Independent learnable threshold biases b_1, b_2, ..., b_{K-1}.
        - Cumulative probability formulation: P(y > k) = sigmoid(W^T x + b_k).
        - Interval probabilities reconstructed via consecutive differences with ReLU.
    """

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))

        # Guide monotonic training by initializing descending thresholds
        with torch.no_grad():
            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes rank logits, cumulative threshold probabilities, and class predictions."""
        base_logit = self.linear(x)  # (B, 1)
        rank_logits = base_logit + self.biases  # (B, num_thresholds)
        rank_probs = torch.sigmoid(rank_logits)  # P(y > k) in [0, 1]

        # Discrete prediction: count exceeded 0.5 thresholds
        preds = torch.sum(rank_probs > 0.5, dim=1).long()

        # Reconstruct interval probabilities: P(y = k) = P(y > k-1) - P(y > k)
        b = x.size(0)
        probs = torch.zeros(b, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(rank_probs[:, k - 1] - rank_probs[:, k])
        probs[:, -1] = rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)

        return {
            "rank_logits": rank_logits,
            "rank_probs": rank_probs,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes multi-task binary cross-entropy loss across threshold indicators."""
        return coral_loss(outputs["rank_logits"], targets, num_thresholds=self.num_thresholds)

# %% [markdown]
# ### Module `models.heads.corn`

# %%
"""Conditional Ordinal Regression Neural Network (CORN) head.

Reference:
    Shi, Cao, & Raschka (2021). Deep neural networks for rank-consistent ordinal
    regression based on conditional probabilities. Pattern Recognition Letters, 152, 1-8.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORNHead(nn.Module):
    """Conditional Ordinal Regression Neural Network (CORN) output head.

    Features:
        - Models forward conditional transition probabilities: P(y > k | y >= k) = sigmoid(g(x)_k).
        - Chain rule multiplication: P(y > k) = prod_{j=0}^k P(y > j | y >= j).
        - Non-increasing cumulative probabilities guaranteed by cumulative product.
    """

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes conditional logits, cumulative probabilities, and discrete predictions."""
        cond_logits = self.fc(x)  # (B, num_thresholds)
        cond_probs = torch.sigmoid(cond_logits)  # P(y > k | y >= k)
        cum_probs = torch.cumprod(cond_probs, dim=1)  # P(y > k)

        preds = torch.sum(cum_probs > 0.5, dim=1).long()

        b = x.size(0)
        probs = torch.zeros(b, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(cum_probs[:, k - 1] - cum_probs[:, k])
        probs[:, -1] = cum_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)

        return {
            "cond_logits": cond_logits,
            "cond_probs": cond_probs,
            "cum_probs": cum_probs,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes conditional binary cross-entropy loss."""
        return corn_loss(outputs["cond_logits"], targets, num_thresholds=self.num_thresholds)

# %% [markdown]
# ### Module `models.heads.softmax`

# %%
"""Softmax classification heads for diabetic retinopathy grading.

Includes:
- SoftmaxCEHead: Baseline unconstrained categorical head with standard Cross-Entropy.
- SoftmaxQWKHead: Multi-task categorical head with CE + continuous QWK loss and expected grade rounding.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxCEHead(nn.Module):
    """Standard Categorical Classification Baseline Head.

    Architecture:
        Linear(in_features, num_classes) -> Softmax
    Loss:
        Categorical Cross-Entropy (treats grades as independent nominal classes).
    """

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes categorical logits, probabilities, and class predictions.

        Args:
            x: Latent representation tensor of shape (B, in_features).

        Returns:
            Dict containing 'logits', 'probs', and 'preds'.
        """
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes categorical cross-entropy loss."""
        return F.cross_entropy(outputs["logits"], targets, weight=class_weights)


class SoftmaxQWKHead(nn.Module):
    """Categorical Head with Hybrid Cross-Entropy and Continuous QWK Loss.

    Inference computes the expected continuous severity:
        E[y] = sum_{k=0}^{C-1} k * P(y = k)
    and rounds to the nearest integer grade clamped to [0, C - 1].
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.qwk_weight = qwk_weight

        self.fc = nn.Linear(in_features, num_classes)
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes probabilities and expected continuous severity values."""
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)

        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        continuous_pred = torch.sum(probs * class_indices, dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "continuous_preds": continuous_pred,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes composite multitask loss: CE + lambda * Continuous_QWK."""
        ce_loss = F.cross_entropy(outputs["logits"], targets, weight=class_weights)
        qwk = self.qwk_loss(outputs["probs"], targets)
        return ce_loss + self.qwk_weight * qwk

# %% [markdown]
# ### Module `models.attention.cbam`

# %%
"""Convolutional Block Attention Module (CBAM) for feature refinement."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM (Woo et al., 2018).

    Exploits inter-channel relationships of features by combining spatial Average
    Pooling and Max Pooling descriptors through a shared bottleneck MLP.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_dim = max(8, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)  # [B, in_channels, 1, 1]


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM (Woo et al., 2018).

    Captures spatial relationships by pooling across channels (avg & max),
    followed by a large receptive field 7x7 convolution.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)  # [B, 1, H, W]


class CBAM(nn.Module):
    """Sequential Channel and Spatial Attention (CBAM)."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# %% [markdown]
# ### Module `models.attention.non_local`

# %%
"""Non-Local Neural Networks block for long-range spatial relational reasoning."""

import torch
import torch.nn as nn


class NonLocalBlock2D(nn.Module):
    """Embedded Gaussian Non-local block (Wang et al., 2018).

    Computes self-attention between all spatial positions in the feature map,
    enabling comparison of lesion signals across distant quadrants.
    The output projection is initialized to zeros so the block begins as an
    identity mapping (residual connection), preventing initial fine-tuning instability.
    """

    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // reduction)

        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # Zero-initialize the output projection
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW, C']
        phi = self.phi(x).view(B, self.inter_channels, -1)                     # [B, C', HW]
        g = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)          # [B, HW, C']

        attn = torch.softmax(torch.bmm(theta, phi) / (self.inter_channels ** 0.5), dim=-1)  # [B, HW, HW]
        y = torch.bmm(attn, g)                                                  # [B, HW, C']
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)
        y = self.out_conv(y)
        return x + y  # Residual connection

# %% [markdown]
# ### Module `models.attention.quadrant_tokens`

# %%
"""Quadrant Token Aggregator for clinical retinal quadrant relational reasoning."""

import torch
import torch.nn as nn


class QuadrantTokenAggregator(nn.Module):
    """Explicit 4-quadrant relational aggregation matching the ICDR 4-2-1 convention.

    Splits the final feature map into 4 spatial quadrants (top-left, top-right,
    bottom-left, bottom-right). Each quadrant is pooled into a token with learned
    quadrant positional embeddings, and a Transformer encoder layer allows cross-quadrant
    relational comparison before aggregation.
    """

    def __init__(self, in_channels: int = 2048, num_heads: int = 4):
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

        # Learned positional embedding for 4 quadrants
        self.quadrant_pos_embed = nn.Parameter(torch.zeros(1, 4, in_channels))
        nn.init.trunc_normal_(self.quadrant_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        quadrants = [
            x[:, :, :h_mid, :w_mid],   # Top-left
            x[:, :, :h_mid, w_mid:],   # Top-right
            x[:, :, h_mid:, :w_mid],   # Bottom-left
            x[:, :, h_mid:, w_mid:],   # Bottom-right
        ]
        tokens = torch.stack([self.pool(q).flatten(1) for q in quadrants], dim=1)  # [B, 4, C]
        tokens = tokens + self.quadrant_pos_embed
        tokens = self.encoder(tokens)
        quadrant_summary = tokens.mean(dim=1)  # [B, C]
        return quadrant_summary

# %% [markdown]
# ### Module `models.streams.global_context`

# %%
"""Global Context stream for macroscopic retinal fundus feature representation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class GlobalBranch(nn.Module):
    """Global Context Branch for whole-fundus feature extraction.

    Architecture:
      - Input: Whole-fundus image tensor [Batch, 3, 512, 512]
      - Backbone: Pre-trained ResNet-50 (conv1 -> layer4)
      - Attention: NonLocalBlock2D -> CBAM on layer4 feature maps (2048 channels)
      - Pooling: Global Average Pooling (2048-d) concatenated with Quadrant Token Aggregator (2048-d)
      - Output: Global feature vector F_Global of dimension 4096 (if use_quadrant_tokens) else 2048
    """

    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_nonlocal: bool = True,
        use_quadrant_tokens: bool = True,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)

        # ResNet stem and stages
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

        # Cache for Grad-CAM activations
        self.last_feature_map: Optional[torch.Tensor] = None

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map through layer4 with optional attention."""
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

        self.last_feature_map = x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating F_Global vector [B, out_dim]."""
        feat_map = self.extract_feature_map(x)
        gap_feat = torch.flatten(self.gap(feat_map), 1)

        if self.use_quadrant_tokens:
            quad_feat = self.quadrant_agg(feat_map)
            out = torch.cat([gap_feat, quad_feat], dim=1)
        else:
            out = gap_feat

        return out


# Public canonical alias
GlobalContextStream = GlobalBranch

# %% [markdown]
# ### Module `models.streams.local_mil`

# %%
"""Local Multi-Instance Learning (MIL) stream with Dual Pooling."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class GatedAttentionPool(nn.Module):
    """Gated Attention Mechanism for Multiple Instance Learning (Ilse et al., 2018).

    Learns instance-level attention weights:
        a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k)))
    Enables lesion counting and spatial density aggregation.
    """

    def __init__(self, in_features: int = 1280, hidden_dim: int = 128):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Patch features [B, K, in_features]
            patch_mask: Optional boolean tensor [B, K] (True = valid foreground patch)
        Returns:
            weighted_out: Bag-level representation [B, in_features]
        """
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        a = self.attention_weights(a_v * a_u)  # [B, K, 1]

        if patch_mask is not None:
            mask_3d = patch_mask.bool().unsqueeze(-1)  # [B, K, 1]
            a = a.masked_fill(~mask_3d, -1e4)

        a = torch.softmax(a, dim=1)  # [B, K, 1]
        self.last_attention_weights = a.detach()

        weighted_out = torch.sum(x * a, dim=1)  # [B, in_features]
        return weighted_out


class LocalMILBranch(nn.Module):
    """Local Multi-Instance Learning Branch with Dual Pooling.

    Processes K native-resolution foreground patches:
      - Patch Backbone: Pre-trained EfficientNet-B0
      - Pooling per patch: AdaptiveAvgPool2d -> [B, K, 1280]
      - Dual Aggregation:
          1. 1D Max-Pooling (extreme focal lesion detector) -> [B, 1280]
          2. Gated Attention Pooling (lesion density aggregator) -> [B, 1280]
      - Output: F_Local vector of dimension 2560 (1280 + 1280) if use_dual_pooling else 1280
    """

    def __init__(self, pretrained: bool = True, use_dual_pooling: bool = True):
        super().__init__()
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
            self.gated_pool = None
            self.out_dim = self.patch_dim      # 1280

    def forward(self, patches: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            patches: Tensor of shape [B, K, 3, H_p, W_p]
            patch_mask: Optional boolean tensor [B, K]
        Returns:
            f_local: Aggregated local feature vector [B, out_dim]
        """
        B, K, C, H, W = patches.shape
        flat_patches = patches.view(B * K, C, H, W)
        patch_feats = self.features(flat_patches)
        patch_feats = self.gap(patch_feats)
        patch_feats = torch.flatten(patch_feats, 1)
        patch_feats = patch_feats.view(B, K, self.patch_dim)

        # 1. 1D Max-Pooling (focal lesions)
        if patch_mask is not None:
            bool_mask = patch_mask.bool()
            masked_feats = patch_feats.masked_fill(~bool_mask.unsqueeze(-1), -1e4)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)

        if self.use_dual_pooling and self.gated_pool is not None:
            # 2. Gated Attention Pooling (lesion density)
            att_pooled = self.gated_pool(patch_feats, patch_mask=patch_mask)
            f_local = torch.cat([max_pooled, att_pooled], dim=1)
        else:
            f_local = max_pooled

        return f_local

# %% [markdown]
# ### Module `experiments.mlwa.models_adapters`

# %%
"""Decoupled model, head, link, and checkpoint adapters for MLWA journal experiments."""


import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

LegacyCLMHead = CumulativeLinkModelQWKHead
LegacyCORALHead = CORALHead
LegacyCORNHead = CORNHead
LegacySoftmaxCEHead = SoftmaxCEHead
LegacySoftmaxQWKHead = SoftmaxQWKHead


class JournalSoftmaxHead(nn.Module):
    """Pure Softmax categorical output head (probabilities only; decoder externalized)."""

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            logits = self.fc(x.float())
            probs = F.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "preds": torch.argmax(probs, dim=1),
        }


def _safe_clm_cdf(u: torch.Tensor, link: str) -> torch.Tensor:
    """Evaluate strictly valid FP32 CDF F(u) = P(Y <= k | u = theta_k - s) with finite gradients."""
    if link == "logit":
        return torch.sigmoid(u)
    if link == "cloglog":
        # Stable 1 - exp(-exp(u)): -expm1(-exp(u)) avoids cancellation near u -> -inf,
        # while smooth clamping keeps exp(u) finite cleanly without NaN gradients.
        u_safe = torch.clamp(u, min=-30.0, max=4.0)
        return -torch.expm1(-torch.exp(u_safe))
    if link == "probit":
        return 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
    raise ValueError(f"Unsupported CLM link '{link}'. Expected 'cloglog', 'logit', or 'probit'.")


class JournalCLMHead(nn.Module):
    """Cumulative Link Model head with explicit link separation (cloglog | logit | probit) in FP32."""

    def __init__(self, in_features: int = 512, num_classes: int = 5, link: str = "cloglog") -> None:
        super().__init__()
        if link not in ("cloglog", "logit", "probit"):
            raise ValueError(f"Unsupported CLM link '{link}'.")
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.link = link

        self.score_fc = nn.Linear(in_features, 1, bias=False)
        nn.init.zeros_(self.score_fc.weight)

        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)

    def get_monotonic_cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments.float()) + 1e-4
        pts = [self.first_cutpoint.float()]
        for inc in increments:
            pts.append(pts[-1] + inc)
        return torch.stack(pts)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            x_f32 = x.float()
            s = self.score_fc(x_f32).squeeze(1)
            cutpoints = self.get_monotonic_cutpoints()
            u = cutpoints.unsqueeze(0) - s.unsqueeze(1)
            cum_probs = _safe_clm_cdf(u, self.link)
            cum_probs = torch.clamp(cum_probs, min=0.0, max=1.0)
            rank_probs = 1.0 - cum_probs

            # Compute exact non-negative interval differences without altering ordering
            cols = [cum_probs[:, 0:1]]
            for k in range(1, self.num_cutpoints):
                cols.append((cum_probs[:, k : k + 1] - cum_probs[:, k - 1 : k]).clamp(min=0.0))
            cols.append((1.0 - cum_probs[:, -1:]).clamp(min=0.0))
            probs = torch.cat(cols, dim=1)
            probs_sum = probs.sum(dim=1, keepdim=True)
            probs = torch.where(probs_sum > 0.0, probs / probs_sum, torch.full_like(probs, 1.0 / self.num_classes))

            expected_grade = (probs * torch.arange(self.num_classes, device=x.device, dtype=torch.float32)).sum(dim=1)
            preds_round = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()

        return {
            "severity_score": s,
            "latent_score": s,
            "cutpoints": cutpoints,
            "thetas": cutpoints,
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds_round,
        }


class JournalCORALHead(nn.Module):
    """CORAL head that explicitly reports threshold crossings instead of silently masking them."""

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            rank_logits = self.linear(x.float()) + self.biases.float()
            raw_rank_probs = torch.sigmoid(rank_logits)
            preds = torch.sum(raw_rank_probs > 0.5, dim=1).long()

            # Compute raw consecutive differences BEFORE any repair
            cols = [1.0 - raw_rank_probs[:, 0:1]]
            for k in range(1, self.num_thresholds):
                cols.append(raw_rank_probs[:, k - 1 : k] - raw_rank_probs[:, k : k + 1])
            cols.append(raw_rank_probs[:, -1:])
            raw_interval_diffs = torch.cat(cols, dim=1)

            crossing_amount = (-torch.diff(raw_rank_probs, dim=1)).clamp(min=0.0)
            # Note: valid non-increasing sequence has P(y > k) <= P(y > k-1) => diff <= 0
            positive_jump = torch.diff(raw_rank_probs, dim=1) > 1e-6
            has_crossing = positive_jump.any(dim=1)
            probs = raw_interval_diffs.clamp(min=0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return {
            "rank_logits": rank_logits,
            "rank_probs": raw_rank_probs,
            "raw_interval_diffs": raw_interval_diffs,
            "crossing_amount": crossing_amount,
            "has_threshold_crossing": has_crossing,
            "monotonicity_violation_rate": has_crossing.float().mean(),
            "probs": probs,
            "preds": preds,
        }


class JournalCORNHead(nn.Module):
    """CORN conditional ordinal head preserving strict rank consistency."""

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            cond_logits = self.fc(x.float())
            cond_probs = torch.sigmoid(cond_logits)
            cum_probs = torch.cumprod(cond_probs, dim=1)
            preds = torch.sum(cum_probs > 0.5, dim=1).long()

            cols = [1.0 - cum_probs[:, 0:1]]
            for k in range(1, self.num_thresholds):
                cols.append(cum_probs[:, k - 1 : k] - cum_probs[:, k : k + 1])
            cols.append(cum_probs[:, -1:])
            raw_interval_diffs = torch.cat(cols, dim=1)

            positive_jump = torch.diff(cum_probs, dim=1) > 1e-6
            has_crossing = positive_jump.any(dim=1)
            probs = raw_interval_diffs.clamp(min=0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return {
            "cond_logits": cond_logits,
            "cond_probs": cond_probs,
            "cum_probs": cum_probs,
            "rank_probs": cum_probs,
            "raw_interval_diffs": raw_interval_diffs,
            "has_threshold_crossing": has_crossing,
            "monotonicity_violation_rate": has_crossing.float().mean(),
            "probs": probs,
            "preds": preds,
        }


class JournalDualBranchEncoder(nn.Module):
    """4096-D global + 2560-D local -> 6656-D -> 512-D encoder with whole-bag patch chunking."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.architecture_id = spec.architecture_id
        self.use_local_branch = spec.use_local_branch
        self.latent_dim = spec.fusion_dim
        self.eval_patch_chunk_size = spec.eval_patch_chunk_size

        self.global_branch = GlobalBranch(
            pretrained=spec.pretrained,
            use_cbam=spec.use_cbam,
            use_nonlocal=spec.use_nonlocal,
            use_quadrant_tokens=spec.use_quadrant_tokens,
        )
        if spec.use_local_branch:
            self.local_mil_branch = LocalMILBranch(pretrained=spec.pretrained, use_dual_pooling=True)
            local_dim = self.local_mil_branch.out_dim
        else:
            self.local_mil_branch = None
            local_dim = 0

        self.fusion = DualBranchFusion(
            global_dim=self.global_branch.out_dim,
            local_dim=local_dim,
            fusion_dim=spec.fusion_dim,
            use_local_branch=spec.use_local_branch,
            dropout_rate=spec.dropout,
        )

    def _encode_local_whole_bag(
        self,
        local_patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode patches in chunks while keeping whole-bag max and gated-attention pooling intact."""
        mil = self.local_mil_branch
        assert mil is not None
        bsz, k_patches, c, h, w = local_patches.shape
        flat_patches = local_patches.view(bsz * k_patches, c, h, w)
        chunk = flat_patches.shape[0] if self.eval_patch_chunk_size is None else max(1, int(self.eval_patch_chunk_size))

        if not self.training and flat_patches.shape[0] > chunk:
            chunks = []
            for start in range(0, flat_patches.shape[0], chunk):
                feats = mil.features(flat_patches[start : start + chunk])
                chunks.append(torch.flatten(mil.gap(feats), 1))
            patch_feats = torch.cat(chunks, dim=0)
        else:
            patch_feats = torch.flatten(mil.gap(mil.features(flat_patches)), 1)

        patch_feats = patch_feats.view(bsz, k_patches, mil.patch_dim)
        if patch_mask is not None:
            masked_feats = patch_feats.masked_fill(~patch_mask.unsqueeze(-1), -1e4)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)

        if mil.use_dual_pooling:
            att_pooled = mil.gated_pool(patch_feats, patch_mask=patch_mask)
            return torch.cat([max_pooled, att_pooled], dim=1)
        return max_pooled

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f_global = self.global_branch(global_img)
        if self.use_local_branch and self.local_mil_branch is not None:
            if local_patches is None:
                raise ValueError("local_patches is required when use_local_branch=True.")
            f_local = self._encode_local_whole_bag(local_patches, patch_mask=patch_mask)
            return self.fusion(f_global, f_local)
        return self.fusion(f_global, None)


class JournalGlobalResNet50Encoder(nn.Module):
    """Single-branch ResNet-50 + 512-D projection (for explicit fallback tier only)."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.architecture_id = spec.architecture_id
        self.latent_dim = spec.fusion_dim
        weights = models.ResNet50_Weights.DEFAULT if spec.pretrained else None
        base = models.resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(2048, spec.fusion_dim),
            nn.LayerNorm(spec.fusion_dim),
            nn.Mish(),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.stem(global_img)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.proj(self.gap(out).flatten(1))


class JournalGradingModel(nn.Module):
    """Complete decoupled journal model pairing an encoder adapter and output head."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.architecture_id = spec.architecture_id
        self.preprocessing_id = spec.preprocessing_id
        self.forward_id = f"{spec.architecture_id}__{spec.head_type}_{spec.clm_link}"

        if spec.architecture_id == "dual_branch_mil_v1":
            self.encoder: nn.Module = JournalDualBranchEncoder(spec)
        elif spec.architecture_id == "global_resnet50_v1":
            self.encoder = JournalGlobalResNet50Encoder(spec)
        else:
            raise ValueError(f"Unsupported architecture_id '{spec.architecture_id}'.")

        if spec.head_type == "softmax":
            self.head: nn.Module = JournalSoftmaxHead(in_features=spec.fusion_dim, num_classes=5)
        elif spec.head_type == "clm":
            self.head = JournalCLMHead(in_features=spec.fusion_dim, num_classes=5, link=spec.clm_link)
        elif spec.head_type == "coral":
            self.head = JournalCORALHead(in_features=spec.fusion_dim, num_classes=5)
        elif spec.head_type == "corn":
            self.head = JournalCORNHead(in_features=spec.fusion_dim, num_classes=5)
        else:
            raise ValueError(f"Unsupported head_type '{spec.head_type}'.")

    def shared_encoder_digest(self) -> str:
        """SHA-256 of encoder state_dict tensors so shared initialization across cells is verifiable."""
        h = hashlib.sha256()
        for name, tensor in sorted(self.encoder.state_dict().items()):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent = self.encoder(global_img, local_patches=local_patches, patch_mask=patch_mask)
        out = self.head(latent)
        out["latent"] = latent
        return out


class PooledResNet50StemProjAdapter(nn.Module):
    """Strict adapter for historical pooled head ablation checkpoints using stem/layer1..4/proj keys."""

    architecture_id = "pooled_resnet50_stem_proj_v1"
    preprocessing_id = "ben_graham_green_512"

    def __init__(self, head_type: str) -> None:
        super().__init__()
        self.head_type = head_type
        base = models.resnet50(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.2),
        )
        mapping = {
            "Softmax_CE": lambda: LegacySoftmaxCEHead(512, 5),
            "Softmax_QWK": lambda: LegacySoftmaxQWKHead(512, 5, qwk_weight=0.4),
            "CORAL": lambda: LegacyCORALHead(512, 5),
            "CORN": lambda: LegacyCORNHead(512, 5),
            "CLM_QWK": lambda: LegacyCLMHead(512, 5, qwk_weight=0.4),
        }
        if head_type not in mapping:
            raise ValueError(f"Unknown pooled head_type '{head_type}'.")
        self.head = mapping[head_type]()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        z = self.proj(self.gap(out).flatten(1))
        res = self.head(z)
        res["latent"] = z
        return res

# %% [markdown]
# ## Experiment Workflow & Entry Points

# %%
import argparse
import json
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch


POOLED_HEADS = ('Softmax_CE', 'Softmax_QWK', 'CORAL', 'CORN', 'CLM_QWK')
FILENAME_MAP = {
    'Softmax_CE': ['resnet50_softmax_ce_best.pth', 'softmax_ce_best.pth'],
    'Softmax_QWK': ['resnet50_softmax_qwk_best.pth', 'softmax_qwk_best.pth'],
    'CORAL': ['resnet50_coral_best.pth', 'coral_best.pth'],
    'CORN': ['resnet50_corn_best.pth', 'corn_best.pth'],
    'CLM_QWK': ['resnet50_clm_qwk_best.pth', 'clm_qwk_best.pth'],
}


def audit_pooled_assets(
    pooled_checkpoints_dir: Optional[Path],
    pooled_metadata_dir: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Strictly audit pooled 5-head checkpoints and manifests without using strict=False."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_status: Dict[str, Any] = {}
    if pooled_metadata_dir is not None and Path(pooled_metadata_dir).is_dir():
        for fn in ('train_balanced_metadata.csv', 'val_metadata.csv', 'test_metadata.csv'):
            fp = Path(pooled_metadata_dir) / fn
            manifest_status[fn] = {'exists': fp.is_file(), 'sha256': sha256_file(fp) if fp.is_file() else None}

    head_reports: Dict[str, Any] = {}
    missing_items = []
    for h_name in POOLED_HEADS:
        found = None
        if pooled_checkpoints_dir is not None and Path(pooled_checkpoints_dir).is_dir():
            for cand in FILENAME_MAP[h_name]:
                cp = Path(pooled_checkpoints_dir) / cand
                if cp.is_file():
                    found = cp
                    break
        if found is None:
            missing_items.append(h_name)
            head_reports[h_name] = {'status': STATUS_PENDING_ASSETS, 'reason': 'checkpoint_missing'}
            continue

        ck = torch.load(found, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        ep = ck.get('epoch') if isinstance(ck, dict) else None
        vq = ck.get('val_qwk') if isinstance(ck, dict) else None

        adapter = PooledResNet50StemProjAdapter(h_name)
        try:
            adapter.load_state_dict(sd, strict=True)
            strict_ok = True
            strict_err = None
        except Exception as exc:
            strict_ok = False
            strict_err = str(exc)[:300]

        usable = strict_ok and (ep is not None and ep > 1)
        if not usable:
            missing_items.append(f'{h_name} (epoch={ep}, strict_ok={strict_ok})')
        head_reports[h_name] = {
            'status': 'VERIFIED' if usable else STATUS_PENDING_ASSETS,
            'checkpoint': str(found),
            'sha256': sha256_file(found),
            'epoch': ep,
            'val_qwk': vq,
            'strict_load_stem_proj_v1': strict_ok,
            'strict_error': strict_err,
        }

    overall = 'READY_FOR_POOLED_INFERENCE' if not missing_items else STATUS_PENDING_ASSETS
    report = {
        'protocol_id': 'historical_pooled_heads_v1',
        'status': overall,
        'missing_or_incomplete': missing_items,
        'manifests': manifest_status,
        'heads': head_reports,
        'note': (
            'Applying CIBD to 5 pooled heads evaluates the separate pooled split (8,075 test images), '
            'NOT the 6-fold LODO protocol. Training a fresh multi-head study uses a new protocol_id.'
        ),
    }
    atomic_write_json(out / 'POOLED_HEADS_STATUS.json', report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='10: Audit/reconstruct historical five-head pooled benchmark.')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--list-jobs', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--pooled-checkpoints-dir', type=Path, default=None)
    p.add_argument('--pooled-metadata-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_pooled_heads'))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_jobs:
        print(f'Historical pooled heads ({len(POOLED_HEADS)}): {POOLED_HEADS} on separate protocol historical_pooled_heads_v1')
        return 0
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_OK', 'heads': list(POOLED_HEADS), 'output_dir': str(args.output_dir)}))
        return 0
    if args.smoke:
        # Smoke test strict adapter forward pass on 2 dummy images with zero downloads
        rows = []
        x = torch.randn(2, 3, 64, 64)
        for h_name in POOLED_HEADS:
            m = PooledResNet50StemProjAdapter(h_name).eval()
            out = m(x)
            rows.append({'head': h_name, 'probs_shape': list(out['probs'].shape), 'row_sums': out['probs'].sum(1).tolist()})
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / 'POOLED_SMOKE_TEST_ONLY.json', {'status': 'TEST_ONLY', 'heads': rows})
        print(json.dumps({'status': 'TEST_ONLY', 'heads': len(rows)}))
        return 0
    paths = resolve_runtime_paths()
    rep = audit_pooled_assets(
        args.pooled_checkpoints_dir or paths['pooled_checkpoints'],
        args.pooled_metadata_dir or paths['pooled_metadata'],
        args.output_dir,
    )
    print(json.dumps(rep, indent=2))
    return 0

# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_pooled_heads')
    if SMOKE or MODE == 'smoke':
        _rows = []
        _x = torch.randn(2, 3, 64, 64)
        for _h in POOLED_HEADS:
            _m = PooledResNet50StemProjAdapter(_h).eval()
            _o = _m(_x)
            _rows.append({'head': _h, 'probs_shape': list(_o['probs'].shape), 'row_sums': _o['probs'].sum(1).tolist()})
        _out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_out / 'POOLED_SMOKE_TEST_ONLY.json', {'status': 'TEST_ONLY', 'heads': _rows})
        print(json.dumps({'status': 'TEST_ONLY', 'heads': len(_rows)}))
    else:
        print(audit_pooled_assets(_paths['pooled_checkpoints'], _paths['pooled_metadata'], _out))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
