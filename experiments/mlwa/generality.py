"""E6 Generality (G2 Allocation Gate, Frozen Encoder & External Ordinal Dataset Interfaces)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.mlwa.constants import STATUS_SKIPPED_GATE_NOT_MET


@dataclass(frozen=True)
class PretrainingExposureAudit:
    """Explicit domain-exposure audit required before running any alternative frozen encoder in E6."""

    encoder_id: str
    pretrained_sources: Tuple[str, ...]
    overlaps_lodo_cohorts: Tuple[str, ...]
    exposure_risk: str
    approved_for_clean_generalization: bool
    notes: str = ""

    def validate(self) -> None:
        if self.overlaps_lodo_cohorts and self.approved_for_clean_generalization:
            raise ValueError(
                f"Encoder '{self.encoder_id}' overlaps LODO cohorts {self.overlaps_lodo_cohorts}; "
                "cannot claim clean cross-domain generalization."
            )


KNOWN_ENCODER_AUDITS: Dict[str, PretrainingExposureAudit] = {
    "convnext_tiny_imagenet1k": PretrainingExposureAudit(
        encoder_id="convnext_tiny_imagenet1k",
        pretrained_sources=("ImageNet-1K",),
        overlaps_lodo_cohorts=(),
        exposure_risk="LOW_NO_RETINAL_OVERLAP",
        approved_for_clean_generalization=True,
        notes="General-purpose vision backbone with zero retinal cohort exposure.",
    ),
    "retfound_cfp_v1": PretrainingExposureAudit(
        encoder_id="retfound_cfp_v1",
        pretrained_sources=("MEH-MIDAS", "EyePACS"),
        overlaps_lodo_cohorts=("eyepacs",),
        exposure_risk="HIGH_EYEPACS_PRETRAINING_EXPOSURE",
        approved_for_clean_generalization=False,
        notes="Pretrained on EyePACS; excluded from primary MLWA study per supervisor plan.",
    ),
    "retfound_green_v1": PretrainingExposureAudit(
        encoder_id="retfound_green_v1",
        pretrained_sources=("DDR", "FundusMix"),
        overlaps_lodo_cohorts=("ddr",),
        exposure_risk="HIGH_DDR_PRETRAINING_EXPOSURE",
        approved_for_clean_generalization=False,
        notes="Pretrained with DDR exposure; excluded from primary MLWA study.",
    ),
}


def evaluate_g2_gate(
    intervention_per_cohort: pd.DataFrame,
    baseline_per_cohort: pd.DataFrame,
    e4_best_g3_mean: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate supervisor G2 resource-allocation gate (non-tuning, post-lock gate only)."""
    req = {"fold_id", "cohort", "seed_train", "recall_g3", "within1", "qwk"}
    if not req.issubset(intervention_per_cohort.columns) or not req.issubset(baseline_per_cohort.columns):
        return {"gate_open": False, "status": STATUS_SKIPPED_GATE_NOT_MET, "reason": "MISSING_REQUIRED_COLUMNS"}

    m = intervention_per_cohort.merge(
        baseline_per_cohort, on=["fold_id", "cohort", "seed_train"], suffixes=("_int", "_base")
    )
    if m.empty:
        return {"gate_open": False, "status": STATUS_SKIPPED_GATE_NOT_MET, "reason": "NO_PAIRED_ROWS"}

    m["d_g3_pp"] = (m["recall_g3_int"] - m["recall_g3_base"]) * 100.0
    m["d_w1_pp"] = (m["within1_int"] - m["within1_base"]) * 100.0
    m["d_qwk"] = m["qwk_int"] - m["qwk_base"]

    per_cohort = m.groupby("cohort")[["d_g3_pp", "d_w1_pp", "d_qwk"]].mean()
    n_improved = int((per_cohort["d_g3_pp"] > 0.0).sum())
    seed_consistent = bool((m.groupby("cohort")["d_g3_pp"].apply(lambda s: (s > 0).all())).sum() >= 5)

    seed_eq_g3 = float(m.groupby("seed_train")["d_g3_pp"].mean().mean())
    seed_eq_w1 = float(m.groupby("seed_train")["d_w1_pp"].mean().mean())
    seed_eq_qwk = float(m.groupby("seed_train")["d_qwk"].mean().mean())
    int_g3_abs = float(m.groupby("seed_train")["recall_g3_int"].mean().mean())

    beats_e4 = True if e4_best_g3_mean is None else bool(int_g3_abs > float(e4_best_g3_mean))
    c_recall = seed_eq_g3 >= 5.0
    c_cohorts = n_improved >= 5
    c_w1 = seed_eq_w1 >= -1.0
    c_qwk = seed_eq_qwk >= -0.01

    gate_open = bool(c_recall and c_cohorts and seed_consistent and c_w1 and c_qwk and beats_e4)
    return {
        "gate_open": gate_open,
        "status": "G2_GATE_OPEN" if gate_open else STATUS_SKIPPED_GATE_NOT_MET,
        "mean_delta_g3_pp": seed_eq_g3,
        "mean_delta_within1_pp": seed_eq_w1,
        "mean_delta_qwk": seed_eq_qwk,
        "n_cohorts_improved": n_improved,
        "seed_consistent_5of6": seed_consistent,
        "beats_e4_baselines": beats_e4,
        "criteria": {
            "g3_ge_5pp": c_recall,
            "cohorts_ge_5_of_6": c_cohorts,
            "seed_consistent": seed_consistent,
            "within1_ge_minus1pp": c_w1,
            "qwk_ge_minus001": c_qwk,
            "beats_e4": beats_e4,
        },
    }


class FrozenEncoderAdapter(ABC):
    """Plugin interface for alternative frozen vision encoders in optional E6."""

    encoder_id: str
    feature_dim: int
    exposure_audit: PretrainingExposureAudit

    @abstractmethod
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Return deterministic (N, D) float32 embeddings without finetuning or weight download."""


@dataclass(frozen=True)
class ExternalDatasetContract:
    """Explicit schema for any secondary non-DR ordinal dataset in optional E6."""

    dataset_id: str
    num_classes: int
    grade_names: Tuple[str, ...]
    target_class_index: int
    primary_endpoint_name: str
    population_description: str

    def validate(self) -> None:
        if self.num_classes < 3 or len(self.grade_names) != self.num_classes:
            raise ValueError("grade_names length must equal num_classes >= 3.")
        if not (0 <= self.target_class_index < self.num_classes):
            raise ValueError("target_class_index out of bounds.")


class ExternalOrdinalDatasetAdapter(ABC):
    """Plugin interface for a second ordinal dataset (separate from DR Grade 3 contract)."""

    contract: ExternalDatasetContract

    @abstractmethod
    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (source_train_df, source_val_df, target_df) with y_true, image_key, patient_key."""
