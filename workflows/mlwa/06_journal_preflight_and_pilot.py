# %% [markdown]
# # 06 · MLWA Journal Preflight, Asset Audit & Pilot Profiler
# 
# > **Standalone Single-File Notebook**: All required modules are embedded directly below. Upload only this `.ipynb` file to Google Colab and run all cells (`Runtime -> Run all`); no repository clone or zip bundle is required.

# %%
NOTEBOOK_CODE_SHA256 = '3c031720d2fb1cd027ceba442f5cbcca755d9491913dcd90514449d360f85025'

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
            'cv2': 'opencv-python-headless', 'albumentations': 'albumentations',
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
    MODE = 'smoke'  # 'smoke' | 'preflight' | 'pilot'
    OUTPUT_DIR = None
    SMOKE = True

# %% [markdown]
# ## Embedded Standalone Implementation (31 self-contained modules)

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
# ### Module `experiments.mlwa.objectives`

# %%
"""Training objectives, symmetric QWK warmup, Cui CB-CE, Menon logit adjustment, and Galdran CSR."""


from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectiveOutput:
    total_loss: torch.Tensor
    base_loss: float
    qwk_loss: float
    aux_loss: float
    active_qwk_weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_qwk_weight(epoch: int, spec: ObjectiveSpec) -> float:
    """Symmetric QWK lambda schedule shared identically by Softmax+QWK and CLM+QWK."""
    if not spec.use_qwk:
        return 0.0
    if spec.qwk_schedule_mode == "fixed":
        return float(spec.target_qwk_weight)
    warmup = max(1, int(spec.qwk_warmup_epochs))
    progress = min(1.0, max(1, int(epoch)) / float(warmup))
    return float(progress * spec.target_qwk_weight)


def compute_cui_effective_weights(
    class_counts: np.ndarray,
    beta: float = 0.9999,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cui et al. (CVPR 2019) class-balanced weights inversely proportional to (1 - beta^n) / (1 - beta)."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or len(counts) != 5:
        raise ValueError(f"Expected 5-class natural source_train counts, got shape {counts.shape}.")
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must lie in [0, 1), got {beta}.")

    observed = counts > 0
    if not observed.any():
        raise ValueError("Cannot compute Cui effective-number weights with zero total samples.")

    weights = np.zeros_like(counts, dtype=np.float64)
    if beta == 0.0:
        weights[observed] = 1.0
        effective_nums = counts.copy()
    else:
        log_beta = np.log(beta)
        effective_nums = np.zeros_like(counts, dtype=np.float64)
        # Use -expm1(n * log(beta)) / (1 - beta) for high numerical accuracy near beta -> 1.
        effective_nums[observed] = -np.expm1(counts[observed] * log_beta) / (1.0 - beta)
        weights[observed] = 1.0 / effective_nums[observed]
        mean_obs = float(weights[observed].mean())
        weights[observed] = weights[observed] / mean_obs

    absent_classes = [int(i) for i in range(5) if not observed[i]]
    meta = {
        "beta": float(beta),
        "natural_counts": [int(c) for c in counts.tolist()],
        "absent_classes": absent_classes,
        "weights": [float(w) for w in weights.tolist()],
    }
    return weights.astype(np.float32), meta


def compute_menon_log_prior(class_counts: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """Compute natural source_train class priors for Menon et al. (ICLR 2021) logit adjustment."""
    counts = np.asarray(class_counts, dtype=np.float64) + float(smoothing)
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Invalid class counts for Menon prior computation.")
    probs = counts / counts.sum()
    return np.log(np.maximum(probs, 1e-12)).astype(np.float32)


class JournalObjective(nn.Module):
    """Typed objective evaluator keeping likelihood, QWK, and rebalancing mechanisms explicit."""

    def __init__(
        self,
        spec: ObjectiveSpec,
        natural_train_counts: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.qwk_loss_fn = ContinuousQWKLoss(num_classes=5)
        self.meta: Dict[str, Any] = {"objective_id": spec.objective_id}

        self.register_buffer("cui_weights", torch.ones(5, dtype=torch.float32))
        self.register_buffer("log_prior", torch.zeros(5, dtype=torch.float32))

        if spec.objective_id in ("cui_cb_ce", "clm_cui_weighted_nll"):
            if natural_train_counts is None:
                raise ValueError(f"Objective '{spec.objective_id}' requires natural source_train class counts.")
            w, w_meta = compute_cui_effective_weights(natural_train_counts, beta=spec.cui_beta)
            self.cui_weights.copy_(torch.from_numpy(w))
            self.meta["cui"] = w_meta

        if spec.objective_id in ("menon_logit_adj_ce", "clm_logprob_adjusted_nll"):
            if natural_train_counts is None:
                raise ValueError(f"Objective '{spec.objective_id}' requires natural source_train class counts.")
            # Smooth minimally only if an observed fold somehow lacks a class; natural counts > 0 here.
            smooth = 1.0 if (np.asarray(natural_train_counts) == 0).any() else 0.0
            lp = compute_menon_log_prior(natural_train_counts, smoothing=smooth)
            self.log_prior.copy_(torch.from_numpy(lp))
            self.meta["menon"] = {
                "tau_train": float(spec.logit_adj_tau_train),
                "log_prior": [float(x) for x in lp.tolist()],
                "smoothing_used": smooth,
            }

        # Precompute Galdran cost matrix M_{y, c} = |y - c|^p
        grid = torch.arange(5, dtype=torch.float32)
        cost_mat = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0)) ** float(spec.galdran_power)
        self.register_buffer("galdran_cost", cost_mat)

    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        epoch: int = 1,
    ) -> ObjectiveOutput:
        oid = self.spec.objective_id
        targets = targets.long()
        active_qwk = compute_qwk_weight(epoch, self.spec)

        if oid == "softmax_ce":
            base = F.cross_entropy(outputs["logits"].float(), targets)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "softmax_ce_qwk":
            base = F.cross_entropy(outputs["logits"].float(), targets)
            qwk = self.qwk_loss_fn(outputs["probs"].float(), targets)
            total = base + active_qwk * qwk
            return ObjectiveOutput(total, float(base.detach().item()), float(qwk.detach().item()), 0.0, active_qwk, self.meta)

        if oid == "clm_nll":
            probs = outputs["probs"].float()
            p_y = probs[torch.arange(targets.size(0), device=targets.device), targets]
            base = -torch.log(torch.clamp(p_y, min=1e-12)).mean()
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "clm_nll_qwk":
            probs = outputs["probs"].float()
            p_y = probs[torch.arange(targets.size(0), device=targets.device), targets]
            base = -torch.log(torch.clamp(p_y, min=1e-12)).mean()
            qwk = self.qwk_loss_fn(probs, targets)
            total = base + active_qwk * qwk
            return ObjectiveOutput(total, float(base.detach().item()), float(qwk.detach().item()), 0.0, active_qwk, self.meta)

        if oid == "cui_cb_ce":
            w = self.cui_weights.to(targets.device)
            per_sample = F.cross_entropy(outputs["logits"].float(), targets, reduction="none")
            sample_w = w[targets]
            denom = sample_w.sum().clamp(min=1e-12)
            base = (per_sample * sample_w).sum() / denom
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "menon_logit_adj_ce":
            # Training-time logit adjustment adds +tau * log(pi) before CE (Menon et al., ICLR 2021).
            adj_logits = outputs["logits"].float() + float(self.spec.logit_adj_tau_train) * self.log_prior.to(targets.device).unsqueeze(0)
            base = F.cross_entropy(adj_logits, targets)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "clm_cui_weighted_nll":
            probs = outputs["probs"].float()
            p_y = probs[torch.arange(targets.size(0), device=targets.device), targets]
            per_sample = -torch.log(torch.clamp(p_y, min=1e-12))
            sample_w = self.cui_weights.to(targets.device)[targets]
            base = (per_sample * sample_w).sum() / sample_w.sum().clamp(min=1e-12)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "clm_logprob_adjusted_nll":
            log_probs = torch.log(torch.clamp(outputs["probs"].float(), min=1e-12))
            adj = log_probs + float(self.spec.logit_adj_tau_train) * self.log_prior.to(targets.device).unsqueeze(0)
            base = F.cross_entropy(adj, targets)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "galdran_cost_sensitive_ce":
            logits = outputs["logits"].float()
            probs = outputs["probs"].float()
            base = F.cross_entropy(logits, targets)
            cost_rows = self.galdran_cost.to(targets.device)[targets]
            reg = torch.sum(cost_rows * probs, dim=1).mean()
            total = base + float(self.spec.galdran_lambda) * reg
            return ObjectiveOutput(total, float(base.detach().item()), 0.0, float(reg.detach().item()), 0.0, self.meta)

        if oid == "coral_bce":
            rank_logits = outputs["rank_logits"].float()
            binary_targets = torch.stack([(targets > k).float() for k in range(4)], dim=1)
            base = F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        if oid == "corn_bce":
            cond_logits = outputs["cond_logits"].float()
            losses = []
            for k in range(4):
                mask = targets >= k
                if mask.any():
                    losses.append(F.binary_cross_entropy_with_logits(cond_logits[mask, k], (targets[mask] > k).float()))
            base = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=targets.device, requires_grad=True)
            return ObjectiveOutput(base, float(base.detach().item()), 0.0, 0.0, 0.0, self.meta)

        raise ValueError(f"Unhandled objective_id '{oid}'.")

# %% [markdown]
# ### Module `experiments.mlwa.plan`

# %%
"""Draft/Frozen verification and reproducible locking for docs/mlwa/analysis_plan.md."""


import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _parse_plan_metadata(text: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in text.splitlines()[:25]:
        m = re.match(r"^\*\*([A-Za-z0-9_]+)\*\*:\s*(.+?)\s*$", line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip("` ")
    return meta


def compute_plan_sha256(plan_path: Path) -> str:
    data = Path(plan_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _git_head(repo_root: Path) -> Optional[str]:
    if not (repo_root / ".git").exists():
        return None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def freeze_analysis_plan(
    plan_path: Path,
    lock_path: Path,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote status to FROZEN cleanly and write an immutable lock JSON."""
    plan_path = Path(plan_path).resolve()
    lock_path = Path(lock_path).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Analysis plan not found: {plan_path}")

    text = plan_path.read_text(encoding="utf-8")
    meta = _parse_plan_metadata(text)
    if meta.get("status", "DRAFT").upper() == "DRAFT":
        text = re.sub(
            r"^\*\*status\*\*:\s*DRAFT\b",
            "**status**: FROZEN",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        plan_path.write_text(text, encoding="utf-8", newline="\n")
        meta = _parse_plan_metadata(text)

    if meta.get("status", "").upper() != "FROZEN":
        raise ValueError(f"Cannot lock plan with status='{meta.get('status')}'; expected DRAFT or FROZEN.")

    plan_sha = compute_plan_sha256(plan_path)
    root = Path(repo_root).resolve() if repo_root else plan_path.parents[2]
    lock_payload: Dict[str, Any] = {
        "schema": "mlwa_analysis_plan_lock_v1",
        "status": "FROZEN",
        "protocol_id": meta.get("protocol_id", "mlwa_journal_v1"),
        "study_id": meta.get("study_id", "mlwa_dr_ordinal_extension"),
        "created_date": meta.get("created_date"),
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_head(root),
        "plan_relpath": str(plan_path),
        "plan_sha256": plan_sha,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock_payload


def verify_frozen_plan(
    plan_path: Optional[Path],
    lock_path: Optional[Path],
    allow_test_only: bool = False,
) -> Dict[str, Any]:
    """Refuse production runs unless plan status is FROZEN and lock hash matches."""
    if allow_test_only:
        if plan_path is not None and Path(plan_path).is_file():
            return {
                "status": "TEST_ONLY",
                "plan_sha256": compute_plan_sha256(Path(plan_path)),
                "protocol_id": "TEST_ONLY_SYNTHETIC_PLAN",
            }
        return {"status": "TEST_ONLY", "plan_sha256": "TEST_ONLY", "protocol_id": "TEST_ONLY_SYNTHETIC_PLAN"}

    if plan_path is None or lock_path is None:
        raise ValueError("Production execution requires both --plan-path and --plan-lock-path.")
    plan_p = Path(plan_path)
    lock_p = Path(lock_path)
    if not plan_p.is_file():
        raise FileNotFoundError(f"Analysis plan not found: {plan_p}")
    text = plan_p.read_text(encoding="utf-8")
    meta = _parse_plan_metadata(text)
    if meta.get("status", "").upper() != "FROZEN":
        raise RuntimeError(
            f"Refusing production run against unfrozen plan ({plan_p}, status={meta.get('status')}). "
            "Run `python -m experiments.mlwa.plan freeze docs/mlwa/analysis_plan.md --out docs/mlwa/analysis_plan.lock.json` first."
        )
    if not lock_p.is_file():
        raise RuntimeError(f"Refusing production run without plan lock file: {lock_p}")
    lock = json.loads(lock_p.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN":
        raise RuntimeError(f"Plan lock status is not FROZEN: {lock.get('status')}")
    actual_sha = compute_plan_sha256(plan_p)
    if lock.get("plan_sha256") != actual_sha:
        raise RuntimeError(
            f"Analysis plan modified after freeze! lock={lock.get('plan_sha256')}, actual={actual_sha}"
        )
    return lock

# %% [markdown]
# ### Module `experiments.mlwa.registry`

# %%
"""Machine-readable experiment registry and expanded job matrix builder (unique jobs & reuse map)."""


import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


def build_expanded_job_matrix(
    include_fallback: bool = False,
    seeds: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """Enumerate every unique training/evaluation job with explicit reuse links and priority tier."""
    rows: List[Dict[str, Any]] = []

    def _add_lodo_jobs(
        experiment: str,
        config_id: str,
        priority: str,
        arch_tier: str,
        head_type: str,
        objective_id: str,
        clm_link: str,
        reuses_config_id: str = "",
        status: str = STATUS_PLANNED,
    ) -> None:
        for fold_id, cohort in enumerate(COHORT_ORDER):
            for seed in seeds:
                rows.append(
                    {
                        "job_id": f"{config_id}__f{fold_id}_{cohort}__s{seed}",
                        "experiment": experiment,
                        "config_id": config_id,
                        "priority": priority,
                        "architecture_tier": arch_tier,
                        "head_type": head_type,
                        "objective_id": objective_id,
                        "clm_link": clm_link,
                        "fold_id": fold_id,
                        "cohort": cohort,
                        "seed_train": int(seed),
                        "reuses_config_id": reuses_config_id,
                        "is_unique_training_job": (reuses_config_id == ""),
                        "compute_type": "gpu_train",
                        "status": status,
                    }
                )

    # Pilots (smallest natural source_train fold 4 eyepacs, largest fold 1 idrid)
    for fid, c in [(4, "eyepacs"), (1, "idrid")]:
        rows.append(
            {
                "job_id": f"PILOT__{c}__s0",
                "experiment": "PILOT",
                "config_id": "E2_D_clm_cloglog_nll_qwk",
                "priority": "mandatory_preflight",
                "architecture_tier": "primary_dual_branch",
                "head_type": "clm",
                "objective_id": "clm_nll_qwk",
                "clm_link": "cloglog",
                "fold_id": fid,
                "cohort": c,
                "seed_train": 0,
                "reuses_config_id": "",
                "is_unique_training_job": False,
                "compute_type": "gpu_profile",
                "status": STATUS_PLANNED,
            }
        )

    # E2 Mandatory 2x2 factorial (4 x 6 x 3 = 72 unique runs)
    _add_lodo_jobs("E2", "E2_A_softmax_ce", "mandatory", "primary_dual_branch", "softmax", "softmax_ce", "none")
    _add_lodo_jobs("E2", "E2_B_softmax_ce_qwk", "mandatory", "primary_dual_branch", "softmax", "softmax_ce_qwk", "none")
    _add_lodo_jobs("E2", "E2_C_clm_cloglog_nll", "mandatory", "primary_dual_branch", "clm", "clm_nll", "cloglog")
    _add_lodo_jobs("E2", "E2_D_clm_cloglog_nll_qwk", "mandatory", "primary_dual_branch", "clm", "clm_nll_qwk", "cloglog")

    # E2b Recommended Link Ablation (+18 unique runs; D reused)
    _add_lodo_jobs("E2b", "E2b_clm_logit_nll_qwk", "recommended", "primary_dual_branch", "clm", "clm_nll_qwk", "logit")

    # E4 Mandatory Imbalance Baselines (+36 unique runs; A reused)
    _add_lodo_jobs("E4", "E4_cui_cb_ce", "mandatory", "primary_dual_branch", "softmax", "cui_cb_ce", "none")
    _add_lodo_jobs("E4", "E4_menon_logit_adj_ce", "mandatory", "primary_dual_branch", "softmax", "menon_logit_adj_ce", "none")

    # E4 Optional Supplementary comparators
    _add_lodo_jobs("E4_supp", "E4_galdran_csr_ce", "recommended_optional", "primary_dual_branch", "softmax", "galdran_cost_sensitive_ce", "none")

    if include_fallback:
        _add_lodo_jobs("E2_fallback", "E2_FB_A_softmax_ce", "fallback", "global_resnet50_v1", "softmax", "softmax_ce", "none")
        _add_lodo_jobs("E2_fallback", "E2_FB_B_softmax_ce_qwk", "fallback", "global_resnet50_v1", "softmax", "softmax_ce_qwk", "none")
        _add_lodo_jobs("E2_fallback", "E2_FB_C_clm_cloglog_nll", "fallback", "global_resnet50_v1", "clm", "clm_nll", "cloglog")
        _add_lodo_jobs("E2_fallback", "E2_FB_D_clm_cloglog_nll_qwk", "fallback", "global_resnet50_v1", "clm", "clm_nll_qwk", "cloglog")
        _add_lodo_jobs("E2_fallback_confirm", "E2_FB_CONFIRM_V2", "fallback", "primary_dual_branch", "softmax", "softmax_ce", "none")
        _add_lodo_jobs("E2_fallback_confirm", "E2_FB_CONFIRM_V3", "fallback", "primary_dual_branch", "clm", "clm_nll_qwk", "cloglog")

    return pd.DataFrame(rows)


def build_experiment_registry(include_fallback: bool = False) -> Dict[str, Any]:
    df = build_expanded_job_matrix(include_fallback=include_fallback)
    summary_counts = {
        "e2_mandatory_unique_jobs": int(len(df[df["experiment"] == "E2"])),
        "e2b_recommended_unique_jobs": int(len(df[df["experiment"] == "E2b"])),
        "e4_mandatory_unique_jobs": int(len(df[df["experiment"] == "E4"])),
        "e4_supp_optional_jobs": int(len(df[df["experiment"] == "E4_supp"])),
        "pilot_jobs": int(len(df[df["experiment"] == "PILOT"])),
        "total_mandatory_primary_gpu_jobs": int(len(df[(df["priority"] == "mandatory") & df["is_unique_training_job"]])),
    }
    return {
        "schema": "mlwa_experiment_registry_v1",
        "counts": summary_counts,
        "legacy_completed": {
            "P0_LODO": {"status": STATUS_DIAGNOSTIC_REPLAY, "run_dir": "result/p0_fp32_main_lodo_v1", "seed": 42},
            "P0_POOLED_5HEADS": {"status": STATUS_PENDING_ASSETS, "reason": "Missing trained 25-epoch checkpoints (local files are epoch=1 smoke)"},
            "E0_LEGACY": {"status": STATUS_COMPLETE, "run_dir": "result/p0_fp32_main_lodo_v1_e0_resume_v2"},
            "E1_LEGACY": {"status": STATUS_COMPLETE, "run_dir": "result/p0_fp32_main_lodo_v1_e1_resume_v2"},
        },
        "gated_or_opt_in_post_gpu": {
            "E3_BRC": {"status": STATUS_SKIPPED_GATE_NOT_MET, "reason": "Legacy E1 s_v3 (0.5409) > z_v3 (0.5109); awaits journal E1 multi-seed check."},
            "E5_PRIOR_ADAPTATION": {"status": "OPT_IN_CPU", "flag": "--enable-e5"},
            "E6_GENERALITY": {"status": STATUS_SKIPPED_GATE_NOT_MET, "flag": "--enable-e6"},
        },
    }


def write_registry_artifacts(docs_dir: Path) -> Tuple[Path, Path]:
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    reg = build_experiment_registry(include_fallback=False)
    reg_path = docs_dir / "experiment_registry.json"
    reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jm = build_expanded_job_matrix(include_fallback=True)
    jm_path = docs_dir / "job_matrix.csv"
    jm.to_csv(jm_path, index=False)
    return reg_path, jm_path

# %% [markdown]
# ### Module `data.fov`

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
# ### Module `data.registry`

# %%
"""Cohort registry defining metadata, paths, and clinical protocol specifications."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CohortSpec:
    """Clinical and organizational specifications for a fundus image cohort."""

    dataset_key: str
    display_name: str
    country: str
    camera_device: str
    field_of_view: str
    grading_protocol: str
    id_col: str
    label_col: str
    id_suffix: str = ""
    csv_relative_path: str = ""
    raw_img_dir: str = ""


COHORT_REGISTRY: Dict[str, CohortSpec] = {
    "aptos": CohortSpec(
        dataset_key="aptos",
        display_name="APTOS 2019",
        country="India",
        camera_device="Zeiss, Topcon",
        field_of_view="45°",
        grading_protocol="Single-grader ICDRS",
        id_col="id_code",
        label_col="diagnosis",
        id_suffix=".png",
        csv_relative_path="data/raw/aptos/train.csv",
        raw_img_dir="data/raw/aptos/train_images",
    ),
    "ddr": CohortSpec(
        dataset_key="ddr",
        display_name="DDR",
        country="China",
        camera_device="Canon CR-2, Topcon",
        field_of_view="45°",
        grading_protocol="Multi-expert consensus",
        id_col="image_id",
        label_col="dr_level",
        id_suffix=".jpg",
        csv_relative_path="data/raw/ddr/train.csv",
        raw_img_dir="data/raw/ddr/train",
    ),
    "deepdrid": CohortSpec(
        dataset_key="deepdrid",
        display_name="DeepDRiD",
        country="China",
        camera_device="Topcon TRC-NW400",
        field_of_view="45°",
        grading_protocol="2-field paired stereoscopic",
        id_col="image_id",
        label_col="patient_DR_Level",
        id_suffix=".jpg",
        csv_relative_path="data/raw/deepdrid/metadata.csv",
        raw_img_dir="data/raw/deepdrid/images",
    ),
    "idrid": CohortSpec(
        dataset_key="idrid",
        display_name="IDRiD",
        country="India",
        camera_device="Kowa VX-10alpha",
        field_of_view="50°",
        grading_protocol="Dual-retina specialist panel",
        id_col="Image name",
        label_col="Retinopathy grade",
        id_suffix=".jpg",
        csv_relative_path="data/raw/idrid/train.csv",
        raw_img_dir="data/raw/idrid/images",
    ),
    "messidor2": CohortSpec(
        dataset_key="messidor2",
        display_name="Messidor-2",
        country="France",
        camera_device="Topcon TRC NW6",
        field_of_view="45°",
        grading_protocol="Adjudicated multi-grader panel",
        id_col="image_id",
        label_col="adjudicated_dr_grade",
        id_suffix=".jpg",
        csv_relative_path="data/raw/messidor2/messidor_data.csv",
        raw_img_dir="data/raw/messidor2/images",
    ),
    "eyepacs": CohortSpec(
        dataset_key="eyepacs",
        display_name="EyePACS",
        country="USA",
        camera_device="Centervue, Canon, Topcon",
        field_of_view="45°",
        grading_protocol="Tele-ophthalmology reading network",
        id_col="image",
        label_col="level",
        id_suffix=".jpeg",
        csv_relative_path="data/raw/eyepacs/trainLabels.csv",
        raw_img_dir="data/raw/eyepacs/train",
    ),
}


def get_cohort_spec(dataset_key: str) -> CohortSpec:
    """Retrieves CohortSpec for a valid dataset key."""
    key = dataset_key.strip().lower()
    if key not in COHORT_REGISTRY:
        raise KeyError(
            f"Unknown dataset_key '{dataset_key}'. Registered keys: {list(COHORT_REGISTRY.keys())}"
        )
    return COHORT_REGISTRY[key]


def list_registered_cohorts() -> List[str]:
    """Returns list of registered dataset keys."""
    return list(COHORT_REGISTRY.keys())

# %% [markdown]
# ### Module `data.metadata`

# %%
"""Canonical metadata record schema and validation for harmonized fundus datasets."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class MasterFundusRecord:
    """Standardized record for an individual fundus photograph across all six cohorts."""

    image_id: str
    diagnosis: int
    dataset_name: str
    dataset_key: str
    raw_full_path: str
    native_path: str
    patient_id: str
    side: str = "unknown"
    original_image_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates all field constraints according to data model specifications."""
        if not self.image_id or not isinstance(self.image_id, str):
            raise ValueError(f"image_id must be a non-empty string, got: {self.image_id}")

        if not isinstance(self.diagnosis, int) or self.diagnosis not in (0, 1, 2, 3, 4):
            raise ValueError(f"diagnosis must be integer in {{0, 1, 2, 3, 4}}, got: {self.diagnosis}")

        key = self.dataset_key.strip().lower()
        if key not in COHORT_REGISTRY:
            raise ValueError(f"dataset_key '{self.dataset_key}' not in registered cohorts: {list(COHORT_REGISTRY.keys())}")
        self.dataset_key = key

        if "::" not in self.patient_id:
            raise ValueError(f"patient_id must be namespaced with '::' (e.g. 'aptos::001'), got: {self.patient_id}")

        valid_sides = {"left", "right", "unknown"}
        self.side = self.side.strip().lower()
        if self.side not in valid_sides:
            raise ValueError(f"side must be one of {valid_sides}, got: {self.side}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MasterFundusRecord":
        """Constructs record from dictionary."""
        return cls(
            image_id=str(data["image_id"]),
            diagnosis=int(data["diagnosis"]),
            dataset_name=str(data["dataset_name"]),
            dataset_key=str(data["dataset_key"]),
            raw_full_path=str(data["raw_full_path"]),
            native_path=str(data["native_path"]),
            patient_id=str(data["patient_id"]),
            side=str(data.get("side", "unknown")),
            original_image_id=str(data["original_image_id"]) if data.get("original_image_id") is not None else None,
        )


def records_to_dataframe(records: List[MasterFundusRecord]) -> pd.DataFrame:
    """Converts a list of MasterFundusRecord instances to a structured pandas DataFrame."""
    return pd.DataFrame([r.to_dict() for r in records])


def dataframe_to_records(df: pd.DataFrame) -> List[MasterFundusRecord]:
    """Converts a pandas DataFrame back into a list of validated MasterFundusRecord objects."""
    records = []
    for _, row in df.iterrows():
        records.append(MasterFundusRecord.from_dict(row.to_dict()))
    return records


def validate_records(records: Union[List[MasterFundusRecord], pd.DataFrame]) -> bool:
    """Validates that all records comply with the MasterFundusRecord schema.

    Returns:
        True if all records are valid.

    Raises:
        ValueError if any record is invalid.
    """
    if isinstance(records, pd.DataFrame):
        recs = dataframe_to_records(records)
    else:
        recs = records

    for rec in recs:
        if not isinstance(rec, MasterFundusRecord):
            raise ValueError(f"Expected MasterFundusRecord, got {type(rec)}")
    return True

# %% [markdown]
# ### Module `data.patches`

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
# ### Module `data.preprocessing`

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
# ### Module `data.transforms`

# %%
"""Augmentation and transformation pipelines for global and local patch streams."""

from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np


def get_global_augmentations(img_size: int = 512, is_train: bool = True) -> A.Compose:
    """Constructs the Albumentations transformation pipeline for the Global Context branch.

    Args:
        img_size: Target spatial resolution (width and height) in pixels.
        is_train: If True, applies randomized spatial and photometric augmentations;
            otherwise, applies deterministic bilinear resize only.

    Returns:
        Albumentations Compose transform pipeline.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-30, 30),
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.5,
                ),
            ]
        )

    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        ]
    )


def get_patch_augmentations(patch_size: int = 128, is_train: bool = True) -> Optional[A.Compose]:
    """Constructs patch-level augmentations preserving micro-lesion morphological details."""
    if not is_train:
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )


def get_eval_transforms(img_size: int = 512) -> A.Compose:
    """Returns deterministic evaluation transforms for global image."""
    return get_global_augmentations(img_size=img_size, is_train=False)


class Denormalize:
    """Inverts ImageNet normalization for visualization."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, tensor_or_img: np.ndarray) -> np.ndarray:
        """Denormalizes an image array from [-mean/std] back to [0, 255]."""
        img = tensor_or_img * self.std + self.mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img


class DualBranchTransform:
    """Applies combined global and patch branch transformations."""

    def __init__(self, global_size: int = 512, patch_size: int = 128, is_train: bool = True) -> None:
        self.global_transform = get_global_augmentations(img_size=global_size, is_train=is_train)
        self.patch_transform = get_patch_augmentations(patch_size=patch_size, is_train=is_train)

    def transform_global(self, img: np.ndarray) -> np.ndarray:
        return self.global_transform(image=img)["image"]

    def transform_patch(self, patch: np.ndarray) -> np.ndarray:
        if self.patch_transform is not None:
            return self.patch_transform(image=patch)["image"]
        return patch

# %% [markdown]
# ### Module `utils.logging`

# %%
"""Structured logging utilities for research experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "dr_eval", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger printing to stdout and optionally to a log file.

    Args:
        name: Name of the logger.
        log_file: Optional path to write log output.
        level: Logging level (e.g., logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# %% [markdown]
# ### Module `data.datasets`

# %%
"""PyTorch Dataset implementations for dual-branch multi-scale diabetic retinopathy models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


logger = get_logger(__name__)


class FundusDualBranchDataset(Dataset):
    """PyTorch Dataset serving the multi-scale dual-branch architecture.

    Features:
        - Stream 1: Global Context whole-image resized to img_size x img_size with spatial/photometric augmentations.
        - Stream 2: Native-resolution sliding-window patches extracted around valid retinal FOV anchors.
        - Fast in-memory list-of-dict records for O(1) indexing.
        - High-throughput uint8 tensor return to minimize IPC Shared Memory bandwidth.
    """

    def __init__(
        self,
        records: Union[List[MasterFundusRecord], pd.DataFrame, str, Path],
        img_size: int = 512,
        patch_size: int = 128,
        stride: int = 96,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        num_patches_train: int = 64,
        max_patches_eval: int = 256,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.is_train = split.lower() == "train"
        self.ablation_mode = ablation_mode
        self.cache_dir = cache_dir
        self.num_patches_train = num_patches_train
        self.max_patches_eval = max_patches_eval

        # Normalize records into a list of dictionaries
        if isinstance(records, (str, Path)):
            p = Path(records)
            if not p.exists():
                raise FileNotFoundError(f"Metadata file not found: {p}")
            df = pd.read_csv(p)
            self.records = df.to_dict("records")
        elif isinstance(records, pd.DataFrame):
            self.records = records.to_dict("records")
        elif isinstance(records, list):
            self.records = [r.to_dict() if hasattr(r, "to_dict") else r for r in records]
        else:
            raise TypeError(f"Unsupported records type: {type(records)}")

        self.global_transform = get_global_augmentations(img_size=img_size, is_train=self.is_train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        label = int(row.get("diagnosis", 0))
        image_id = str(row.get("image_id", idx))
        patient_id = str(row.get("patient_id", f"unknown::{image_id}"))
        dataset_key = str(row.get("dataset_key", "default"))

        # Resolve image filepath
        img_path = str(row.get("native_path", row.get("raw_full_path", "")))

        # Load image via OpenCV
        img_rgb = None
        if img_path and Path(img_path).exists():
            bgr = cv2.imread(img_path)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if img_rgb is None:
            raise FileNotFoundError(f"Cannot decode fundus image {image_id}: {img_path}")

        # 1. Morphological FOV mask on RAW image
        fov_mask = compute_fov_mask(img_rgb)

        # 2. Green channel enhancement / Ben Graham normalization
        processed_rgb = apply_green_channel_ablation(img_rgb, ablation_mode=self.ablation_mode)

        # 3. Stream 1: Global Context (Resize + Augmentations)
        augmented = self.global_transform(image=processed_rgb)
        global_u8 = augmented["image"] if isinstance(augmented, dict) else augmented
        global_tensor = torch.from_numpy(global_u8).permute(2, 0, 1).contiguous()

        # 4. Stream 2: Local MIL Patches
        patches_tensor, selected_anchors = extract_mil_patches_sliding(
            img_np=processed_rgb,
            fov_mask=fov_mask,
            patch_size=self.patch_size,
            stride=self.stride,
            mode=self.split,
            num_sample=self.num_patches_train,
            max_patches_eval=self.max_patches_eval,
            return_anchors=True,
        )

        k_patches = patches_tensor.shape[0]
        patch_mask = torch.ones(k_patches, dtype=torch.bool)

        return {
            "global_img": global_tensor,
            "local_patches": patches_tensor,
            "patch_mask": patch_mask,
            "patch_coords": torch.from_numpy(selected_anchors),
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": image_id,
            "patient_id": patient_id,
            "dataset_key": dataset_key,
        }

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
# ### Module `experiments.mlwa.trainer`

# %%
"""Epoch-boundary resumable trainer, RNG stream manager, shared encoder init, and pilot profiler."""


import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


@dataclass
class RNGStreams:
    """Separate RNG streams so head construction never perturbs data order/augmentation."""

    seed_train: int
    model_init_seed: int
    data_order_seed: int
    aug_seed: int

    @classmethod
    def from_seed_train(cls, seed_train: int) -> "RNGStreams":
        return cls(
            seed_train=seed_train,
            model_init_seed=10000 + seed_train * 97,
            data_order_seed=20000 + seed_train * 193,
            aug_seed=30000 + seed_train * 389,
        )


def capture_rng_states(loader_generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
    }
    return state


def restore_rng_states(state: Dict[str, Any], loader_generator: Optional[torch.Generator] = None) -> None:
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state and state["torch_cpu"] is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader_generator is not None and state.get("loader_generator") is not None:
        loader_generator.set_state(state["loader_generator"])


def initialize_shared_encoder_state(model: JournalGradingModel, seed_train: int) -> str:
    """Re-initialize non-pretrained encoder+fusion state deterministically per seed_train across A/B/C/D."""
    streams = RNGStreams.from_seed_train(seed_train)
    cpu_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(streams.model_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(streams.model_init_seed)
        np.random.seed(streams.model_init_seed)
        random.seed(streams.model_init_seed)

        if not model.spec.pretrained:
            fresh = type(model.encoder)(model.spec)
            model.encoder.load_state_dict(fresh.state_dict(), strict=True)
        else:
            # Preserve ImageNet-pretrained backbones; re-seed randomly initialized fusion & attention modules.

            custom_classes = (CBAM, NonLocalBlock2D, QuadrantTokenAggregator, DualBranchFusion, GatedAttentionPool)
            enc = model.encoder
            if hasattr(enc, "global_branch"):
                gb = enc.global_branch
                if getattr(gb, "use_nonlocal", False):
                    gb.non_local.load_state_dict(NonLocalBlock2D(in_channels=2048).state_dict(), strict=True)
                if getattr(gb, "use_cbam", False):
                    gb.cbam.load_state_dict(CBAM(in_channels=2048, reduction_ratio=16).state_dict(), strict=True)
                if getattr(gb, "use_quadrant_tokens", False):
                    gb.quadrant_agg.load_state_dict(QuadrantTokenAggregator(in_channels=2048).state_dict(), strict=True)
            if hasattr(enc, "local_mil_branch") and enc.local_mil_branch is not None and getattr(enc.local_mil_branch, "use_dual_pooling", False):
                enc.local_mil_branch.gated_pool.load_state_dict(
                    GatedAttentionPool(in_features=enc.local_mil_branch.patch_dim, hidden_dim=128).state_dict(),
                    strict=True,
                )
            if hasattr(enc, "fusion"):
                enc.fusion.load_state_dict(
                    DualBranchFusion(
                        global_dim=enc.global_branch.out_dim,
                        local_dim=enc.local_mil_branch.out_dim if enc.local_mil_branch is not None else 0,
                        fusion_dim=model.spec.fusion_dim,
                        use_local_branch=model.spec.use_local_branch,
                        dropout_rate=model.spec.dropout,
                    ).state_dict(),
                    strict=True,
                )
            if hasattr(enc, "proj"):
                for mod in enc.proj.modules():
                    if hasattr(mod, "reset_parameters") and not isinstance(mod, custom_classes):
                        mod.reset_parameters()
    finally:
        torch.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
    return model.shared_encoder_digest()


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: JournalExperimentConfig,
) -> Tuple[torch.optim.Optimizer, CosineAnnealingLR]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = AdamW(
        [
            {"params": decay, "weight_decay": cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.lr,
    )
    sched = CosineAnnealingLR(opt, T_max=cfg.training.cosine_t_max, eta_min=cfg.training.min_lr)
    return opt, sched


def atomic_torch_save(payload: Dict[str, Any], target: Path) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid4().hex[:8]}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_trusted_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    weights_only_if_eval: bool = True,
) -> Dict[str, Any]:
    """Load local trusted checkpoint with strict state loading and explicit schema validation."""
    cp = Path(checkpoint_path)
    if not cp.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")

    eval_only = optimizer is None and scheduler is None and scaler is None and weights_only_if_eval
    if eval_only:
        ckpt = torch.load(cp, map_location=device, weights_only=True)
    else:
        ckpt = torch.load(cp, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint schema in {cp}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def _normalize_batch_u8(x: torch.Tensor, device: str) -> torch.Tensor:
    if x.dtype == torch.uint8:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
        if x.ndim == 4:
            mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
        elif x.ndim == 5:
            mean, std = mean.view(1, 1, 3, 1, 1), std.view(1, 1, 3, 1, 1)
        return x.to(device).float().div_(255.0).sub_(mean).div_(std)
    return x.to(device).float()


@torch.no_grad()
def predict_split(
    model: JournalGradingModel,
    loader: DataLoader,
    cfg: JournalExperimentConfig,
    split_role: str,
    epoch: int,
    checkpoint_sha256: str,
    manifest_sha256: str,
    code_sha256: str,
    plan_sha256: str,
    out_dir: Optional[Path] = None,
    cache_features: bool = False,
    device: str = "cpu",
    save_full_table: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Run deterministic frozen-eval forward pass (with BN & dropout frozen) and format canonical records."""
    model.eval()
    rows: List[Dict[str, Any]] = []
    latents: List[np.ndarray] = []

    for batch in loader:
        g_img = _normalize_batch_u8(batch["global_img"], device)
        l_patches = batch.get("local_patches")
        if l_patches is not None:
            l_patches = _normalize_batch_u8(l_patches, device)
        p_mask = batch.get("patch_mask")
        if p_mask is not None:
            p_mask = p_mask.to(device)

        out = model(g_img, local_patches=l_patches, patch_mask=p_mask)
        probs = out["probs"].detach().cpu().numpy().astype(np.float64)
        preds = out["preds"].detach().cpu().numpy().astype(int)
        z = out["latent"].detach().cpu().numpy().astype(np.float32)
        s_vals = out["severity_score"].detach().cpu().numpy().astype(np.float64) if "severity_score" in out else None
        th = out["thetas"].detach().cpu().numpy().astype(np.float64) if "thetas" in out else None

        y_true = batch["label"].detach().cpu().numpy().astype(int)
        bsz = len(y_true)
        img_keys = batch.get("image_key", [f"{cfg.cohort}::{i}" for i in range(bsz)])
        img_ids = batch.get("image_id", [str(k).split("::")[-1] for k in img_keys])
        cohorts = batch.get("cohort", [cfg.cohort] * bsz)
        pat_keys = batch.get("patient_key", img_keys)
        pat_stats = batch.get("patient_identity_status", ["legacy_unverified"] * bsz)

        if cache_features:
            latents.append(z)

        feat_offset = sum(len(x) for x in latents[:-1]) if cache_features else -1
        for i in range(bsz):
            rec: Dict[str, Any] = {
                "study_id": cfg.study_id,
                "protocol_id": cfg.protocol_id,
                "run_id": cfg.run_id,
                "experiment": cfg.experiment,
                "config_id": cfg.config_id,
                "architecture_id": cfg.model.architecture_id,
                "head_type": cfg.model.head_type,
                "objective_id": cfg.objective.objective_id,
                "clm_link": cfg.model.clm_link,
                "fold_id": cfg.fold_id,
                "cohort": str(cohorts[i]),
                "seed_train": cfg.seeds.seed_train,
                "split_role": split_role,
                "epoch": int(epoch),
                "checkpoint_sha256": checkpoint_sha256,
                "image_key": str(img_keys[i]),
                "image_id": str(img_ids[i]),
                "patient_key": str(pat_keys[i]),
                "patient_identity_status": str(pat_stats[i]),
                "y_true": int(y_true[i]),
                "pred_native": int(preds[i]),
                "s": float(s_vals[i]) if s_vals is not None else np.nan,
                "feature_file": "z.npy" if cache_features else "",
                "feature_row": int(feat_offset + i) if cache_features else -1,
                "manifest_sha256": manifest_sha256,
                "preprocessing_id": cfg.model.preprocessing_id,
                "schema_version": SCHEMA_VERSION,
                "code_sha256": code_sha256,
                "plan_sha256": plan_sha256,
            }
            for k, col in enumerate(PROB_COLS):
                rec[col] = float(probs[i, k])
            for k, col in enumerate(THETA_COLS):
                rec[col] = float(th[k]) if th is not None else np.nan
            rows.append(rec)

    df = pd.DataFrame(rows)
    z_all = np.concatenate(latents, axis=0) if (cache_features and latents) else None

    if out_dir is not None and save_full_table:
        out_dir = Path(out_dir)
        save_predictions_table(df, out_dir)
        if z_all is not None:
            save_feature_bundle(z_all, df["image_key"].tolist(), out_dir)

    return df, z_all


def train_single_unit(
    cfg: JournalExperimentConfig,
    model: JournalGradingModel,
    objective: JournalObjective,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    output_dir: Path,
    source_train_manifest_sha256: str,
    source_val_manifest_sha256: str,
    code_sha256: str,
    plan_sha256: str,
    device: str = "cpu",
    loader_generator: Optional[torch.Generator] = None,
    max_steps_per_epoch: Optional[int] = None,
    simulate_interrupt_after_epoch: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train one fold/seed/cell strictly on source_train + source_val (NO target access).

    Supports deterministic exact epoch-boundary resume. Mid-epoch interrupts restart cleanly
    from the pre-epoch boundary checkpoint (_last.pth).
    """
    cfg.validate()
    unit_dir = Path(output_dir) / cfg.run_id / cfg.config_id / f"fold_{cfg.fold_id}_{cfg.cohort}" / f"seed_{cfg.seeds.seed_train}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = unit_dir / "COMPLETE.json"

    cfg_hash = cfg.config_hash()
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": cfg_hash,
        "plan_sha256": plan_sha256,
        "code_sha256": code_sha256,
        "source_train_manifest_sha256": source_train_manifest_sha256,
        "source_val_manifest_sha256": source_val_manifest_sha256,
    }

    if resume and complete_marker.is_file():
        old_meta = json.loads(complete_marker.read_text(encoding="utf-8"))
        if old_meta.get("identity") != expected_identity:
            raise ValueError(
                f"Stale/mismatched COMPLETE.json at {unit_dir}: "
                "refusing silent reuse on altered config, manifest, plan, or code."
            )
        return old_meta

    model = model.to(device)
    init_encoder_sha = model.shared_encoder_digest()
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    use_amp = (cfg.training.amp_mode != "none") and ("cuda" in str(device))
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda" if "cuda" in str(device) else "cpu", enabled=use_scaler)

    last_ckpt_path = unit_dir / "checkpoint_last.pth"
    best_ckpt_path = unit_dir / "checkpoint_best.pth"
    val_epochs_dir = unit_dir / "source_val_epochs"
    val_epochs_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    best_qwk = -float("inf")
    best_base_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    resumed_from_boundary = False
    history: List[Dict[str, Any]] = []

    if resume and last_ckpt_path.is_file():
        ckpt = load_trusted_checkpoint(
            last_ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler if use_scaler else None, device=device, weights_only_if_eval=False,
        )
        if ckpt.get("config_hash") != cfg_hash or ckpt.get("plan_sha256") != plan_sha256:
            raise ValueError(f"Resume checkpoint hash mismatch in {last_ckpt_path}.")
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_qwk = float(ckpt["best_qwk"])
        best_base_loss = float(ckpt["best_base_loss"])
        best_epoch = int(ckpt["best_epoch"])
        epochs_no_improve = int(ckpt["epochs_no_improve"])
        history = list(ckpt["history"])
        restore_rng_states(ckpt["rng_states"], loader_generator=loader_generator)
        resumed_from_boundary = True
    else:
        streams = RNGStreams.from_seed_train(cfg.seeds.seed_train)
        random.seed(streams.aug_seed)
        np.random.seed(streams.aug_seed)
        torch.manual_seed(streams.aug_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(streams.aug_seed)

    atomic_write_json(unit_dir / "RUN_STATUS.json", {"status": STATUS_RUNNING, "identity": expected_identity})

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_total = 0.0
        train_base = 0.0
        train_qwk = 0.0
        n_train = 0
        active_lam = compute_qwk_weight(epoch, cfg.objective)

        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(source_train_loader, start=1):
            if max_steps_per_epoch is not None and step_idx > max_steps_per_epoch:
                break
            g_img = _normalize_batch_u8(batch["global_img"], device)
            l_patches = _normalize_batch_u8(batch["local_patches"], device) if batch.get("local_patches") is not None else None
            p_mask = batch.get("patch_mask")
            if p_mask is not None:
                p_mask = p_mask.to(device)
            targets = batch["label"].to(device).long()

            with torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype, enabled=use_amp):
                out = model(g_img, local_patches=l_patches, patch_mask=p_mask)
                obj_out = objective.compute(out, targets, epoch=epoch)
                loss = obj_out.total_loss / float(cfg.training.grad_accum_steps)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step_idx % cfg.training.grad_accum_steps == 0) or (step_idx == len(source_train_loader)):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    if cfg.training.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg.training.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            bs = targets.size(0)
            train_total += float(obj_out.total_loss.detach().item()) * bs
            train_base += obj_out.base_loss * bs
            train_qwk += obj_out.qwk_loss * bs
            n_train += bs

        scheduler.step()

        # Evaluate source_val after every epoch with the plan's shared decoder
        model.eval()
        val_probs_list, val_y_list, val_keys, val_s = [], [], [], []
        val_base_sum = 0.0
        val_qwk_sum = 0.0
        n_val = 0
        thetas_val: Optional[np.ndarray] = None
        with torch.no_grad():
            for batch in source_val_loader:
                g_img = _normalize_batch_u8(batch["global_img"], device)
                l_patches = _normalize_batch_u8(batch["local_patches"], device) if batch.get("local_patches") is not None else None
                p_mask = batch.get("patch_mask")
                if p_mask is not None:
                    p_mask = p_mask.to(device)
                targets = batch["label"].to(device).long()
                out = model(g_img, local_patches=l_patches, patch_mask=p_mask)
                obj_out = objective.compute(out, targets, epoch=epoch)
                bs = targets.size(0)
                val_base_sum += obj_out.base_loss * bs
                val_qwk_sum += obj_out.qwk_loss * bs
                n_val += bs
                val_probs_list.append(out["probs"].detach().cpu().numpy().astype(np.float64))
                val_y_list.append(targets.detach().cpu().numpy().astype(int))
                val_keys.extend(batch.get("image_key", [f"val_{n_val + i}" for i in range(bs)]))
                if "severity_score" in out:
                    val_s.append(out["severity_score"].detach().cpu().numpy().astype(np.float64))
                if "thetas" in out and thetas_val is None:
                    thetas_val = out["thetas"].detach().cpu().numpy().astype(np.float64)

        val_probs = np.concatenate(val_probs_list, axis=0)
        val_y = np.concatenate(val_y_list, axis=0)
        val_s_arr = np.concatenate(val_s, axis=0) if val_s else None

        # Save per-epoch source_val probabilities cleanly
        ep_df = pd.DataFrame({"image_key": val_keys, "y_true": val_y, "epoch": epoch})
        for k, c in enumerate(PROB_COLS):
            ep_df[c] = val_probs[:, k]
        if val_s_arr is not None:
            ep_df["s"] = val_s_arr
        if thetas_val is not None:
            for k, c in enumerate(THETA_COLS):
                ep_df[c] = float(thetas_val[k])
        ep_df.to_csv(val_epochs_dir / f"source_val_epoch_{epoch:03d}.csv", index=False)

        sel_pred = decode_by_name(
            cfg.training.checkpoint_selection_decoder,
            val_probs,
            s=val_s_arr,
            thetas=thetas_val,
        )
        val_dm = compute_decision_metrics(val_y, sel_pred)
        val_pm = compute_posterior_metrics(val_y, val_probs)
        val_base_mean = val_base_sum / max(1, n_val)
        val_qwk_loss_mean = val_qwk_sum / max(1, n_val)

        # Deterministic tie-breaking: higher QWK, lower val base_loss, earlier epoch
        cur_qwk = float(val_dm["qwk"])
        is_best = (cur_qwk > best_qwk + 1e-12) or (
            abs(cur_qwk - best_qwk) <= 1e-12 and val_base_mean < best_base_loss - 1e-12
        )
        if is_best:
            best_qwk = cur_qwk
            best_base_loss = val_base_mean
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history.append(
            {
                "epoch": epoch,
                "active_qwk_weight": active_lam,
                "train_total_loss": train_total / max(1, n_train),
                "train_base_loss": train_base / max(1, n_train),
                "train_qwk_loss": train_qwk / max(1, n_train),
                "val_base_loss": val_base_mean,
                "val_qwk_loss": val_qwk_loss_mean,
                "val_qwk": cur_qwk,
                "val_within1": float(val_dm["within1"]),
                "val_recall_g3": float(val_dm["recall_g3"]),
                "val_nll": float(val_pm["nll_mean"]),
                "val_rps_mean_4": float(val_pm["rps_mean_4"]),
                "wall_seconds": time.perf_counter() - t0,
            }
        )

        # Save weights-only best state + full resumable epoch-boundary last state
        ckpt_state = {
            "schema_version": SCHEMA_VERSION,
            "resume_granularity": "epoch_boundary",
            "epoch": epoch,
            "global_step": global_step,
            "active_qwk_weight": active_lam,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_scaler else None,
            "best_qwk": best_qwk,
            "best_base_loss": best_base_loss,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
            "rng_states": capture_rng_states(loader_generator=loader_generator),
            "config_hash": cfg_hash,
            "plan_sha256": plan_sha256,
            "code_sha256": code_sha256,
            "initial_shared_encoder_sha256": init_encoder_sha,
        }
        atomic_torch_save(ckpt_state, last_ckpt_path)
        if is_best:
            best_weights_only = {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch,
                "best_qwk": best_qwk,
                "model_state_dict": ckpt_state["model_state_dict"],
                "config_hash": cfg_hash,
                "plan_sha256": plan_sha256,
            }
            atomic_torch_save(best_weights_only, best_ckpt_path)

        if simulate_interrupt_after_epoch is not None and epoch == simulate_interrupt_after_epoch:
            raise KeyboardInterrupt(f"Simulated interrupt cleanly after epoch {epoch}")

        if epochs_no_improve >= cfg.training.patience:
            break

    # Export authoritative checkpoint_params.json + restore best weights for cached source splits
    load_trusted_checkpoint(best_ckpt_path, model=model, device=device, weights_only_if_eval=True)
    best_ckpt_sha = sha256_file(best_ckpt_path)

    thetas_best: Optional[List[float]] = None
    if cfg.model.head_type == "clm":
        thetas_best = [float(x) for x in model.head.get_monotonic_cutpoints().detach().cpu().tolist()]

    checkpoint_params = {
        "schema_version": SCHEMA_VERSION,
        "config_id": cfg.config_id,
        "fold_id": cfg.fold_id,
        "cohort": cfg.cohort,
        "seed_train": cfg.seeds.seed_train,
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk,
        "checkpoint_sha256": best_ckpt_sha,
        "initial_shared_encoder_sha256": init_encoder_sha,
        "thetas": thetas_best,
        "objective_metadata": objective.meta,
    }
    atomic_write_json(unit_dir / "checkpoint_params.json", checkpoint_params)
    pd.DataFrame(history).to_csv(unit_dir / "training_history.csv", index=False)

    # Cache source_train & source_val predictions and 512-D frozen z
    predict_split(
        model, source_train_loader, cfg, "source_train", best_epoch, best_ckpt_sha,
        source_train_manifest_sha256, code_sha256, plan_sha256,
        out_dir=unit_dir / "source_train", cache_features=True, device=device,
    )
    predict_split(
        model, source_val_loader, cfg, "source_val", best_epoch, best_ckpt_sha,
        source_val_manifest_sha256, code_sha256, plan_sha256,
        out_dir=unit_dir / "source_val", cache_features=True, device=device,
    )

    completion = {
        "status": STATUS_TEST_ONLY if cfg.is_test_only else STATUS_COMPLETE,
        "resume_granularity": "epoch_boundary",
        "resumed_from_boundary": resumed_from_boundary,
        "identity": expected_identity,
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk,
        "best_checkpoint_sha256": best_ckpt_sha,
        "initial_shared_encoder_sha256": init_encoder_sha,
        "files": {
            "checkpoint_best.pth": best_ckpt_sha,
            "checkpoint_params.json": sha256_file(unit_dir / "checkpoint_params.json"),
            "training_history.csv": sha256_file(unit_dir / "training_history.csv"),
        },
    }
    atomic_write_json(complete_marker, completion)
    atomic_write_json(unit_dir / "RUN_STATUS.json", completion)
    return completion


def run_pilot_profile(
    cfg: JournalExperimentConfig,
    model: JournalGradingModel,
    objective: JournalObjective,
    loader: DataLoader,
    n_train_images_full: int,
    output_path: Path,
    device: str = "cpu",
    n_profile_steps: int = 4,
) -> Dict[str, Any]:
    """Measure step wall time, peak VRAM, and project full campaign budget transparently."""
    model = model.to(device)
    model.train()
    opt, _ = build_optimizer_and_scheduler(model, cfg)
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    step_times: List[float] = []
    data_times: List[float] = []
    patch_counts: List[int] = []
    t_data_start = time.perf_counter()
    for step_idx, batch in enumerate(loader, start=1):
        data_times.append(time.perf_counter() - t_data_start)
        t_step = time.perf_counter()
        g_img = _normalize_batch_u8(batch["global_img"], device)
        l_patches = _normalize_batch_u8(batch["local_patches"], device) if batch.get("local_patches") is not None else None
        if l_patches is not None:
            patch_counts.append(int(l_patches.shape[1]))
        targets = batch["label"].to(device).long()
        opt.zero_grad(set_to_none=True)
        out = model(g_img, local_patches=l_patches)
        obj = objective.compute(out, targets, epoch=1)
        obj.total_loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - t_step)
        if step_idx >= n_profile_steps:
            break
        t_data_start = time.perf_counter()

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if (torch.cuda.is_available() and "cuda" in str(device))
        else 0.0
    )
    mean_step_s = float(np.mean(step_times)) if step_times else 0.0
    steps_per_epoch = max(1, int(np.ceil(n_train_images_full / float(cfg.training.batch_size))))
    est_epoch_s = mean_step_s * steps_per_epoch
    est_run_hours = (est_epoch_s * cfg.training.epochs) / 3600.0

    report = {
        "measured": {
            "device": gpu_name,
            "fold_id": cfg.fold_id,
            "cohort": cfg.cohort,
            "n_profile_steps": len(step_times),
            "mean_step_seconds": mean_step_s,
            "mean_data_seconds": float(np.mean(data_times)) if data_times else 0.0,
            "mean_patches_per_image": float(np.mean(patch_counts)) if patch_counts else 0.0,
            "peak_vram_mb": peak_vram_mb,
        },
        "projected": {
            "is_projection": True,
            "n_train_images_full": int(n_train_images_full),
            "steps_per_epoch": steps_per_epoch,
            "projected_epoch_seconds": est_epoch_s,
            "projected_single_run_hours_at_max_epochs": est_run_hours,
            "projected_72_unit_e2_hours": est_run_hours * 72.0,
            "projected_36_unit_e4_hours": est_run_hours * 36.0,
        },
    }
    atomic_write_json(output_path, report)
    return report

# %% [markdown]
# ### Module `experiments.mlwa.runner_utils`

# %%
"""Shared Colab/local dataset, smoke-fixture, and workflow runner helpers for 06-15 scripts."""


import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDRDataset(Dataset):
    """Deterministic tiny in-memory synthetic dataset used strictly in TEST_ONLY smoke/unit runs."""

    def __init__(
        self,
        n_samples: int = 20,
        cohort: str = "aptos",
        split_role: str = "source_train",
        seed: int = 42,
        include_patches: bool = True,
        img_size: int = 64,
        n_patches: int = 4,
        patch_size: int = 32,
    ) -> None:
        self.n_samples = n_samples
        self.cohort = cohort
        self.split_role = split_role
        self.include_patches = include_patches
        rng = np.random.default_rng(seed)
        self.labels = np.tile(np.arange(5), (n_samples + 4) // 5)[:n_samples].astype(int)
        # Slightly shift class-dependent intensity patterns cleanly
        self.imgs = np.clip(
            rng.normal(110.0 + self.labels[:, None, None, None] * 10.0, 20.0, (n_samples, 3, img_size, img_size)),
            0,
            255,
        ).astype(np.uint8)
        self.patches = (
            np.clip(
                rng.normal(110.0 + self.labels[:, None, None, None, None] * 10.0, 20.0, (n_samples, n_patches, 3, patch_size, patch_size)),
                0,
                255,
            ).astype(np.uint8)
            if include_patches
            else None
        )
        self.image_keys = [f"{cohort}::syn_{split_role}_{i:04d}" for i in range(n_samples)]
        self.patient_keys = [f"{cohort}::syn_pat_{i // 2:04d}" for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "global_img": torch.from_numpy(self.imgs[idx].copy()),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "image_key": self.image_keys[idx],
            "image_id": self.image_keys[idx].split("::")[-1],
            "cohort": self.cohort,
            "patient_key": self.patient_keys[idx],
            "patient_identity_status": "legacy_unverified",
        }
        if self.patches is not None:
            item["local_patches"] = torch.from_numpy(self.patches[idx].copy())
            item["patch_mask"] = torch.ones(self.patches.shape[1], dtype=torch.bool)
        return item


class ManifestDualBranchAdapter(Dataset):
    """Wraps FundusDualBranchDataset to expose canonical image_key/patient_key/cohort contract fields."""

    def __init__(self, df: pd.DataFrame, split: str = "train", max_patches_eval: int = 256) -> None:
        self.df = df.reset_index(drop=True)
        self.inner = FundusDualBranchDataset(
            records=self.df.assign(
                dataset_key=self.df["cohort"],
                diagnosis=self.df["y_true"],
                native_path=self.df["runtime_path"],
            ),
            split=split,
            max_patches_eval=max_patches_eval,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = self.inner[idx]
        row = self.df.iloc[idx]
        out["image_key"] = str(row["image_key"])
        out["image_id"] = str(row["image_id"])
        out["cohort"] = str(row["cohort"])
        out["patient_key"] = str(row["patient_key"])
        out["patient_identity_status"] = str(row["patient_identity_status"])
        return out


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-K local patch bags deterministically."""
    out: Dict[str, Any] = {
        "global_img": torch.stack([b["global_img"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "image_key": [b["image_key"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "cohort": [b["cohort"] for b in batch],
        "patient_key": [b["patient_key"] for b in batch],
        "patient_identity_status": [b["patient_identity_status"] for b in batch],
    }
    if "local_patches" in batch[0]:
        max_k = max(b["local_patches"].shape[0] for b in batch)
        c, h, w = batch[0]["local_patches"].shape[1:]
        padded = torch.zeros((len(batch), max_k, c, h, w), dtype=batch[0]["local_patches"].dtype)
        masks = torch.zeros((len(batch), max_k), dtype=torch.bool)
        for i, b in enumerate(batch):
            k = b["local_patches"].shape[0]
            padded[i, :k] = b["local_patches"]
            masks[i, :k] = b["patch_mask"]
        out["local_patches"] = padded
        out["patch_mask"] = masks
    return out


def get_code_identity() -> str:
    """Compute SHA-256 fingerprint across core experiments/mlwa modules or standalone notebook code."""
    nb_sha = globals().get("NOTEBOOK_CODE_SHA256")
    if isinstance(nb_sha, str) and len(nb_sha) == 64 and nb_sha != "0" * 64:
        return nb_sha
    if "__file__" in globals():
        root = Path(__file__).resolve().parent
        if (root / "constants.py").is_file():
            h = hashlib.sha256()
            for p in sorted(root.glob("*.py")):
                h.update(p.name.encode("utf-8"))
                h.update(p.read_bytes())
            return h.hexdigest()
    return str(nb_sha or "standalone_notebook")


def run_smoke_training_job(
    experiment: str,
    config_id: str,
    model_spec: ModelSpec,
    objective_spec: ObjectiveSpec,
    output_dir: Path,
    fold_id: int = 0,
    seed_train: int = 0,
    run_id: str = "smoke_test_only",
) -> Dict[str, Any]:
    """Execute a bounded CPU TEST_ONLY smoke training unit using tiny synthetic fixtures."""
    cohort = CANONICAL_FOLD_TO_COHORT[fold_id]
    tiny_model_spec = ModelSpec(
        architecture_id="global_resnet50_v1" if model_spec.architecture_id == "global_resnet50_v1" else "dual_branch_mil_v1",
        head_type=model_spec.head_type,
        clm_link=model_spec.clm_link,
        pretrained=False,
        use_local_branch=False if model_spec.architecture_id == "global_resnet50_v1" else model_spec.use_local_branch,
        use_nonlocal=False if model_spec.architecture_id == "global_resnet50_v1" else False,
        use_quadrant_tokens=False if model_spec.architecture_id == "global_resnet50_v1" else False,
        use_cbam=False if model_spec.architecture_id == "global_resnet50_v1" else False,
        fusion_dim=model_spec.fusion_dim,
        dropout=0.1,
        eval_patch_chunk_size=2,
        preprocessing_id="TEST_ONLY_SYNTHETIC",
    )
    train_spec = TrainingSpec(
        epochs=2,
        batch_size=5,
        cosine_t_max=2,
        patience=2,
        amp_mode="none",
        sampling_policy="natural_source_train",
    )
    cfg = JournalExperimentConfig(
        study_id="TEST_ONLY_SMOKE",
        protocol_id="TEST_ONLY_SMOKE",
        run_id=run_id,
        experiment=experiment,
        config_id=config_id,
        fold_id=fold_id,
        cohort=cohort,
        seeds=SeedConfig(seed_train=seed_train),
        model=tiny_model_spec,
        objective=objective_spec,
        training=train_spec,
        is_test_only=True,
    )
    cfg.validate()

    include_patches = tiny_model_spec.architecture_id == "dual_branch_mil_v1"
    ds_tr = SyntheticDRDataset(n_samples=10, cohort="idrid", split_role="source_train", seed=101 + seed_train, include_patches=include_patches)
    ds_va = SyntheticDRDataset(n_samples=10, cohort="ddr", split_role="source_val", seed=202 + fold_id, include_patches=include_patches)
    streams = RNGStreams.from_seed_train(seed_train)
    gen = torch.Generator().manual_seed(streams.data_order_seed)
    tr_loader = DataLoader(ds_tr, batch_size=5, shuffle=True, generator=gen, collate_fn=pad_collate_fn)
    va_loader = DataLoader(ds_va, batch_size=5, shuffle=False, collate_fn=pad_collate_fn)

    model = JournalGradingModel(tiny_model_spec)
    initialize_shared_encoder_state(model, seed_train=seed_train)
    nat_counts = np.bincount(ds_tr.labels, minlength=5)
    objective = JournalObjective(objective_spec, natural_train_counts=nat_counts)

    return train_single_unit(
        cfg=cfg,
        model=model,
        objective=objective,
        source_train_loader=tr_loader,
        source_val_loader=va_loader,
        output_dir=output_dir,
        source_train_manifest_sha256=object_hash(ds_tr.image_keys),
        source_val_manifest_sha256=object_hash(ds_va.image_keys),
        code_sha256=get_code_identity(),
        plan_sha256="TEST_ONLY",
        device="cpu",
        loader_generator=gen,
        max_steps_per_epoch=2,
        resume=False,
    )


def prepare_real_fold_loaders(
    folds_dir: Path,
    images_root: Path,
    cfg: JournalExperimentConfig,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, Dict[str, str], torch.Generator]:
    """Prepare source_train, source_val, and held-out target DataLoaders from verified manifests."""
    fold_id = cfg.fold_id
    cohort = cfg.cohort
    nat_tr_df, _ = build_natural_source_train(folds_dir, fold_id, images_root=images_root)
    va_df = normalize_manifest_df(pd.read_csv(folds_dir / f"fold_{fold_id}_{cohort}_val.csv", dtype=str), images_root=images_root)
    te_df = normalize_manifest_df(pd.read_csv(folds_dir / f"fold_{fold_id}_{cohort}_test.csv", dtype=str), images_root=images_root)

    natural_counts = np.array([(nat_tr_df["y_true"] == k).sum() for k in range(5)], dtype=np.int64)
    if cfg.training.sampling_policy == "quota_balanced":
        tr_df = normalize_manifest_df(pd.read_csv(folds_dir / f"fold_{fold_id}_{cohort}_train.csv", dtype=str), images_root=images_root)
    else:
        tr_df = nat_tr_df

    bs = batch_size or cfg.training.batch_size
    streams = RNGStreams.from_seed_train(cfg.seeds.seed_train)
    gen = torch.Generator().manual_seed(streams.data_order_seed)

    ds_tr = ManifestDualBranchAdapter(tr_df, split="train")
    ds_va = ManifestDualBranchAdapter(va_df, split="eval")
    ds_te = ManifestDualBranchAdapter(te_df, split="eval")

    tr_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, generator=gen, num_workers=num_workers, collate_fn=pad_collate_fn)
    va_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=pad_collate_fn)
    te_loader = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=pad_collate_fn)

    hashes = {
        "source_train": object_hash(tr_df["image_key"].tolist()),
        "source_val": object_hash(va_df["image_key"].tolist()),
        "target": object_hash(te_df["image_key"].tolist()),
    }
    return tr_loader, va_loader, te_loader, natural_counts, hashes, gen

# %% [markdown]
# ## Experiment Workflow & Entry Points

# %%
import argparse
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


def run_preflight_audit(
    folds_dir: Optional[Path],
    p0_dir: Optional[Path],
    pooled_checkpoints_dir: Optional[Path],
    pooled_metadata_dir: Optional[Path],
    output_dir: Path,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    check_images_root: Optional[Path] = None,
    allow_draft_plan_audit: bool = True,
) -> Dict[str, Any]:
    """Non-destructive preflight audit across LODO manifests, P0 cache, pooled assets, and job matrix."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    plan_status: Dict[str, Any]
    if allow_draft_plan_audit and plan_path is not None and (plan_lock_path is None or not Path(plan_lock_path).is_file()):
        plan_status = {'status': 'DRAFT_AUDITED_NOT_YET_FROZEN', 'plan_path': str(plan_path)}
    else:
        plan_status = verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=allow_draft_plan_audit)

    manifest_audit = None
    if folds_dir is not None and Path(folds_dir).is_dir():
        manifest_audit = audit_lodo_manifests(Path(folds_dir), images_root=check_images_root, check_decode=False)
        atomic_write_json(out / 'lodo_manifest_audit.json', manifest_audit)

    p0_audit = None
    if p0_dir is not None and (Path(p0_dir) / 'lodo').is_dir():
        p0_audit = validate_p0_cache(Path(p0_dir), require_features=True)
        atomic_write_json(out / 'p0_cache_audit.json', p0_audit)

    pooled_audit: Dict[str, Any] = {'status': STATUS_PENDING_ASSETS, 'checkpoints': []}
    if pooled_checkpoints_dir is not None and Path(pooled_checkpoints_dir).is_dir():
        ckpts = sorted(Path(pooled_checkpoints_dir).glob('*.pth'))
        for cp in ckpts:
            meta = torch.load(cp, map_location='cpu', weights_only=False)
            ep = meta.get('epoch') if isinstance(meta, dict) else None
            vq = meta.get('val_qwk') if isinstance(meta, dict) else None
            pooled_audit['checkpoints'].append({'name': cp.name, 'epoch': ep, 'val_qwk': vq})
        all_trained = bool(ckpts) and all((c.get('epoch') or 0) > 1 for c in pooled_audit['checkpoints'])
        pooled_audit['status'] = STATUS_PREFLIGHT_OK if all_trained else STATUS_PENDING_ASSETS
        pooled_audit['reason'] = (
            'All checkpoints have epoch > 1'
            if all_trained
            else 'Only epoch=1 smoke/incomplete checkpoints found; trained 25-epoch checkpoints not present.'
        )
    atomic_write_json(out / 'pooled_asset_audit.json', pooled_audit)

    jobs_df = build_expanded_job_matrix(include_fallback=True)
    jobs_df.to_csv(out / 'expanded_job_matrix.csv', index=False)

    summary = {
        'status': STATUS_PREFLIGHT_OK,
        'plan_status': plan_status,
        'manifest_audit_present': manifest_audit is not None,
        'p0_cache_audit_present': p0_audit is not None,
        'pooled_status': pooled_audit['status'],
        'total_matrix_rows_with_fallback': int(len(jobs_df)),
    }
    atomic_write_json(out / 'PREFLIGHT_SUMMARY.json', summary)
    return summary


def run_pilot(
    output_dir: Path,
    folds_dir: Optional[Path] = None,
    images_root: Optional[Path] = None,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    smoke: bool = False,
    device: str = 'auto',
) -> Dict[str, Any]:
    """Profile smallest (fold 4 EyePACS) and largest (fold 1 IDRiD) source_train folds."""
    verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=smoke)
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = []

    for fid in (4, 1):
        cohort = CANONICAL_FOLD_TO_COHORT[fid]
        mspec = ModelSpec(
            head_type='clm',
            clm_link='cloglog',
            pretrained=not smoke,
            use_local_branch=True,
            use_nonlocal=not smoke,
            use_quadrant_tokens=not smoke,
            use_cbam=not smoke,
            eval_patch_chunk_size=4 if smoke else 32,
        )
        ospec = ObjectiveSpec(objective_id='clm_nll_qwk', use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15)
        tspec = TrainingSpec(epochs=2 if smoke else 30, batch_size=5 if smoke else 12, amp_mode='none' if dev == 'cpu' else 'auto')
        cfg = JournalExperimentConfig(
            study_id='TEST_ONLY_PILOT' if smoke else 'mlwa_dr_ordinal_extension',
            protocol_id='TEST_ONLY_PILOT' if smoke else 'mlwa_journal_v1',
            run_id='pilot_smoke' if smoke else 'pilot_profile_v1',
            experiment='PILOT',
            config_id='E2_D_clm_cloglog_nll_qwk',
            fold_id=fid,
            cohort=cohort,
            seeds=SeedConfig(seed_train=0),
            model=mspec,
            objective=ospec,
            training=tspec,
            is_test_only=smoke,
        )
        model = JournalGradingModel(mspec)
        if smoke:
            ds = SyntheticDRDataset(n_samples=10, cohort=cohort, seed=42 + fid, include_patches=True)
            loader = DataLoader(ds, batch_size=5, shuffle=False, collate_fn=pad_collate_fn)
            nat_counts = np.bincount(ds.labels, minlength=5)
            n_full = 16791 if fid == 4 else 46010
        else:
            if folds_dir is None or images_root is None:
                raise ValueError('Real pilot requires --folds-dir and --images-root.')
            loader, _, _, nat_counts, _, _ = prepare_real_fold_loaders(Path(folds_dir), Path(images_root), cfg)
            n_full = int(nat_counts.sum())
        obj = JournalObjective(ospec, natural_train_counts=nat_counts)
        rep = run_pilot_profile(
            cfg, model, obj, loader, n_train_images_full=n_full,
            output_path=out / f'pilot_fold_{fid}_{cohort}.json', device=dev,
        )
        reports.append(rep)
    return {'status': 'TEST_ONLY' if smoke else 'PILOT_COMPLETE', 'pilots': reports}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='06: MLWA journal preflight audit, job matrix, and pilot profiler.')
    p.add_argument('--preflight', action='store_true', help='Run non-destructive asset/manifest/P0 audit.')
    p.add_argument('--list-jobs', action='store_true', help='Print expanded job matrix summary and exit.')
    p.add_argument('--dry-run', action='store_true', help='Validate arguments/paths cleanly without training.')
    p.add_argument('--pilot', action='store_true', help='Run bounded pilot step/VRAM profiler on folds 4 & 1.')
    p.add_argument('--smoke', action='store_true', help='Run TEST_ONLY CPU smoke pilot on synthetic fixtures.')
    p.add_argument('--folds-dir', type=Path, default=None)
    p.add_argument('--images-root', type=Path, default=None)
    p.add_argument('--p0-dir', type=Path, default=None)
    p.add_argument('--pooled-checkpoints-dir', type=Path, default=None)
    p.add_argument('--pooled-metadata-dir', type=Path, default=None)
    p.add_argument('--plan-path', type=Path, default=Path('docs/mlwa/analysis_plan.md'))
    p.add_argument('--plan-lock-path', type=Path, default=Path('docs/mlwa/analysis_plan.lock.json'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_journal_preflight'))
    p.add_argument('--device', type=str, default='auto')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_runtime_paths()

    if args.list_jobs:
        df = build_expanded_job_matrix(include_fallback=True)
        print(df.groupby(['experiment', 'priority', 'is_unique_training_job']).size().to_string())
        return 0

    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_OK', 'output_dir': str(args.output_dir), 'p0_dir': str(args.p0_dir or paths['p0_run'])}, indent=2))
        return 0

    if args.smoke or args.pilot:
        res = run_pilot(
            output_dir=args.output_dir,
            folds_dir=args.folds_dir,
            images_root=args.images_root,
            plan_path=args.plan_path,
            plan_lock_path=args.plan_lock_path,
            smoke=args.smoke,
            device=args.device,
        )
        print(json.dumps(res, indent=2))
        return 0

    res = run_preflight_audit(
        folds_dir=args.folds_dir,
        p0_dir=args.p0_dir or paths['p0_run'],
        pooled_checkpoints_dir=args.pooled_checkpoints_dir,
        pooled_metadata_dir=args.pooled_metadata_dir,
        output_dir=args.output_dir,
        plan_path=args.plan_path,
        plan_lock_path=args.plan_lock_path,
        check_images_root=args.images_root,
        allow_draft_plan_audit=True,
    )
    print(json.dumps(res, indent=2))
    return 0

# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR else (_paths['outputs'] / 'mlwa_journal_preflight')
    if SMOKE or MODE == 'smoke':
        print(run_pilot(_out, smoke=True, device='cpu'))
    elif MODE == 'pilot':
        print(run_pilot(_out, folds_dir=_paths['folds'], images_root=_paths['images'], smoke=False))
    else:
        print(run_preflight_audit(_paths['folds'], _paths['p0_run'], _paths['pooled_checkpoints'], _paths['pooled_metadata'], _out))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
