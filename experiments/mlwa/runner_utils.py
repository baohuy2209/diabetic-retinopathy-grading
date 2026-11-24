"""Shared Colab/local dataset, smoke-fixture, and workflow runner helpers for 06-15 scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import FundusDualBranchDataset
from experiments.mlwa.config import (
    JournalExperimentConfig,
    ModelSpec,
    ObjectiveSpec,
    SeedConfig,
    TrainingSpec,
)
from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT, COHORT_ORDER
from experiments.mlwa.data_contract import (
    atomic_write_json,
    build_natural_source_train,
    normalize_manifest_df,
    object_hash,
)
from experiments.mlwa.models_adapters import JournalGradingModel
from experiments.mlwa.objectives import JournalObjective
from experiments.mlwa.trainer import RNGStreams, initialize_shared_encoder_state, train_single_unit


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
    """Compute SHA-256 fingerprint across core experiments/mlwa modules."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


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
