"""Validated configuration dataclasses and deterministic hashing for MLWA journal runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT

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
