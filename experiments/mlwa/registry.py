"""Machine-readable experiment registry and expanded job matrix builder (unique jobs & reuse map)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from experiments.mlwa.constants import (
    COHORT_ORDER,
    STATUS_COMPLETE,
    STATUS_DIAGNOSTIC_REPLAY,
    STATUS_PENDING_ASSETS,
    STATUS_PLANNED,
    STATUS_SKIPPED_GATE_NOT_MET,
)


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
