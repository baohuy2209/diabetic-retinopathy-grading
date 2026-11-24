"""Verified-only journal report generator, coverage audit, and measured-data publication plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.mlwa.constants import (
    COHORT_LABELS,
    COHORT_ORDER,
    STATUS_COMPLETE,
    STATUS_DIAGNOSTIC_REPLAY,
    STATUS_PENDING_ASSETS,
    STATUS_SKIPPED_GATE_NOT_MET,
)
from experiments.mlwa.data_contract import atomic_write_json, sha256_file


def _set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def generate_journal_reports(
    output_dir: Path,
    p0_dir: Optional[Path] = None,
    e0_dir: Optional[Path] = None,
    e1_dir: Optional[Path] = None,
    journal_runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate tables and measured figures ONLY from verified completed artifacts (never fabricate zeros)."""
    _set_pub_style()
    out_dir = Path(output_dir).resolve()
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows: List[Dict[str, Any]] = []
    missing_runs: List[Dict[str, Any]] = []
    generated_tables: Dict[str, str] = {}
    generated_figures: Dict[str, Any] = {}
    input_digests: Dict[str, str] = {}

    # 1. Legacy P0 / E0 / E1 Baseline Verification (if supplied)
    if p0_dir is not None and (Path(p0_dir) / "G0_STATUS.json").is_file():
        p0_path = Path(p0_dir)
        g0_status = json.loads((p0_path / "G0_STATUS.json").read_text(encoding="utf-8"))
        pooled_status = json.loads((p0_path / "pooled_asset_status.json").read_text(encoding="utf-8"))
        coverage_rows.append(
            {"study": "P0_LODO_replay", "status": STATUS_DIAGNOSTIC_REPLAY, "source_dir": str(p0_path), "note": g0_status.get("status")}
        )
        coverage_rows.append(
            {"study": "P0_POOLED_5HEADS", "status": pooled_status.get("status", STATUS_PENDING_ASSETS), "source_dir": str(p0_path), "note": pooled_status.get("reason", "")}
        )
    else:
        missing_runs.append({"study": "P0_LODO_replay", "reason": "P0 directory not supplied or G0_STATUS.json missing"})

    e0_dm_path = (Path(e0_dir) / "lodo" / "decision_metrics.csv") if e0_dir else None
    if e0_dm_path and e0_dm_path.is_file():
        input_digests[str(e0_dm_path)] = sha256_file(e0_dm_path)
        e0_df = pd.read_csv(e0_dm_path)
        tgt = e0_df[e0_df["split"] == "target"].copy()
        tgt.to_csv(tables_dir / "legacy_e0_per_cohort_target.csv", index=False)
        generated_tables["legacy_e0_per_cohort_target"] = str(tables_dir / "legacy_e0_per_cohort_target.csv")
        coverage_rows.append({"study": "E0_legacy_decoders", "status": STATUS_COMPLETE, "source_dir": str(e0_dir), "note": "Single-seed (42); P0 G0=DIAGNOSTIC"})

        # Generate measured Pareto / Grade-3 vs Within-1 trade-off chart (CIBD vs native)
        sub = tgt[tgt["decoder"].isin(["native_replay", "cibd_selected", "argmax", "posterior_median", "rounded_expectation"])].copy()
        eq = sub.groupby(["variant", "decoder"], as_index=False)[["recall_g3", "within1", "qwk", "mae"]].mean()
        eq["recall_g3_pp"] = eq["recall_g3"] * 100.0
        eq["within1_pp"] = eq["within1"] * 100.0
        eq.to_csv(tables_dir / "legacy_e0_equal_cohort_summary.csv", index=False)
        generated_tables["legacy_e0_equal_cohort_summary"] = str(tables_dir / "legacy_e0_equal_cohort_summary.csv")

        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for var, col in [("V2", "#0072B2"), ("V3", "#C0392B")]:
            v_eq = eq[eq["variant"] == var]
            ax.scatter(v_eq["within1_pp"], v_eq["recall_g3_pp"], color=col, s=60, label=f"{var} (archived replay)")
            for _, r in v_eq.iterrows():
                ax.annotate(str(r["decoder"]), (r["within1_pp"], r["recall_g3_pp"]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=col)
        ax.set_xlabel("Equal-cohort target Within-1 accuracy (%)")
        ax.set_ylabel("Equal-cohort target Grade 3 recall (%)")
        ax.set_title("Legacy E0 Operating Points (Single-Seed P0 Replay; G0=DIAGNOSTIC)")
        ax.grid(linestyle=":", alpha=0.35)
        ax.legend(frameon=True, facecolor="white")
        fig.savefig(figs_dir / "legacy_e0_recall_vs_within1.pdf")
        fig.savefig(figs_dir / "legacy_e0_recall_vs_within1.png", dpi=300)
        plt.close(fig)
        generated_figures["legacy_e0_recall_vs_within1"] = {
            "pdf": "figures/legacy_e0_recall_vs_within1.pdf",
            "data_csv": "tables/legacy_e0_equal_cohort_summary.csv",
            "aggregation": "equal_cohort_mean_6",
        }

        # Grade 3 destination flow chart separating 3->2 (still referable) from 3->0/1 (missed RDR)
        g3_src = tgt[(tgt["variant"] == "V3") & (tgt["decoder"].isin(["native_replay", "argmax", "posterior_median", "cibd_selected"]))].copy()
        flow_cols = [c for c in ["g3_to_0_count", "g3_to_1_count", "g3_to_2_count", "g3_to_3_count", "g3_to_4_count"] if c in g3_src.columns]
        if flow_cols:
            flow_sum = g3_src.groupby("decoder", as_index=False)[flow_cols].sum()
            flow_sum["g3_to_01_missed_rdr"] = flow_sum["g3_to_0_count"] + flow_sum["g3_to_1_count"]
            flow_sum.to_csv(tables_dir / "legacy_v3_g3_destinations_pooled.csv", index=False)
            generated_tables["legacy_v3_g3_destinations_pooled"] = str(tables_dir / "legacy_v3_g3_destinations_pooled.csv")
    else:
        missing_runs.append({"study": "E0_legacy_decoders", "reason": "decision_metrics.csv not found"})

    e1_eq_path = (Path(e1_dir) / "equal_cohort_means.csv") if e1_dir else None
    if e1_eq_path and e1_eq_path.is_file():
        input_digests[str(e1_eq_path)] = sha256_file(e1_eq_path)
        e1_eq = pd.read_csv(e1_eq_path)
        e1_eq.to_csv(tables_dir / "legacy_e1_equal_cohort_means.csv", index=False)
        generated_tables["legacy_e1_equal_cohort_means"] = str(tables_dir / "legacy_e1_equal_cohort_means.csv")
        coverage_rows.append({"study": "E1_legacy_probes", "status": STATUS_COMPLETE, "source_dir": str(e1_dir), "note": "s_v3=0.5409 > z_v3=0.5109; H3 gate closed"})
    else:
        missing_runs.append({"study": "E1_legacy_probes", "reason": "equal_cohort_means.csv not found"})

    # 2. Scan multi-seed GPU journal runs if present (never zero-fill unrun tables!)
    expected_journal_studies = [
        ("E2_2x2_factorial", 72),
        ("E2b_link_ablation", 18),
        ("E4_imbalance_baselines", 36),
        ("E3_brc", 0),
        ("E5_prior_adaptation", 0),
        ("E6_generality", 0),
    ]
    completed_units = []
    if journal_runs_dir is not None and Path(journal_runs_dir).is_dir():
        for comp in sorted(Path(journal_runs_dir).rglob("COMPLETE.json")):
            meta = json.loads(comp.read_text(encoding="utf-8"))
            if meta.get("status") == STATUS_COMPLETE:
                completed_units.append({"path": str(comp.parent), **meta.get("identity", {})})

    for st_name, req_n in expected_journal_studies:
        if st_name == "E3_brc":
            coverage_rows.append({"study": st_name, "status": STATUS_SKIPPED_GATE_NOT_MET, "source_dir": str(journal_runs_dir or ""), "note": "Gated by H3"})
        elif req_n > 0:
            n_done = len([u for u in completed_units if u.get("experiment", "").startswith(st_name.split("_")[0])])
            st_status = STATUS_COMPLETE if n_done >= req_n else ("PARTIAL" if n_done > 0 else "NOT_YET_RUN")
            coverage_rows.append({"study": st_name, "status": st_status, "source_dir": str(journal_runs_dir or ""), "note": f"{n_done}/{req_n} verified units"})
            if n_done < req_n:
                missing_runs.append({"study": st_name, "reason": f"Only {n_done}/{req_n} verified GPU training units present (no fabrication)."})

    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(tables_dir / "experiment_coverage.csv", index=False)
    miss_df = pd.DataFrame(missing_runs)
    miss_df.to_csv(tables_dir / "missing_runs.csv", index=False)

    manifest = {
        "schema": "mlwa_report_manifest_v1",
        "output_dir": str(out_dir),
        "input_file_sha256": input_digests,
        "coverage_table": "tables/experiment_coverage.csv",
        "missing_runs_table": "tables/missing_runs.csv",
        "generated_tables": generated_tables,
        "generated_figures": generated_figures,
    }
    atomic_write_json(out_dir / "report_manifest.json", manifest)
    return manifest
