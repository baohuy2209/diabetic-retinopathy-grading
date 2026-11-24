"""Epoch-boundary resumable trainer, RNG stream manager, shared encoder init, and pilot profiler."""

from __future__ import annotations

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

from experiments.mlwa.config import JournalExperimentConfig
from experiments.mlwa.constants import (
    PROB_COLS,
    SCHEMA_VERSION,
    STATUS_COMPLETE,
    STATUS_RUNNING,
    STATUS_TEST_ONLY,
    THETA_COLS,
)
from experiments.mlwa.data_contract import (
    atomic_write_json,
    object_hash,
    save_feature_bundle,
    save_predictions_table,
    sha256_file,
)
from experiments.mlwa.decoders import decode_by_name
from experiments.mlwa.evaluator import compute_decision_metrics, compute_posterior_metrics
from experiments.mlwa.models_adapters import JournalGradingModel
from experiments.mlwa.objectives import JournalObjective, compute_qwk_weight


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
            from models.attention.cbam import CBAM
            from models.attention.non_local import NonLocalBlock2D
            from models.attention.quadrant_tokens import QuadrantTokenAggregator
            from models.fusion import DualBranchFusion
            from models.streams.local_mil import GatedAttentionPool

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
