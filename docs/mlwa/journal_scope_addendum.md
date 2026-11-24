# MLWA Journal Extension Scope Addendum

**Date**: 2026-11-24  
**Base Commit**: `c3e94a616e29dfe0df2ecc7e85339a3761351766`  
**Applies To**: *Machine Learning with Applications (MLWA)* journal extension (author's extension of the DMIP 2026 study and the six-chapter graduation thesis)

---

## 1. Relationship to the Graduation Thesis Constitution

The repository constitution at [`.specify/memory/constitution.md`](../../.specify/memory/constitution.md) (`v0.1.0-draft`, pending formal author ratification) governs the completed six-chapter graduation thesis (*Investigating Ordinal Decision-Making and Severity Trade-offs in Cross-Domain Diabetic Retinopathy Grading*). Under Principle I of that draft constitution, factorial training (`E2`), link ablations (`E2b`), bounded residual correction (`E3`), imbalance baselines (`E4`), unlabelled prior adaptation (`E5`), and generality extensions (`E6`) were deferred as future work beyond the graduation thesis.

This addendum formally records the separation between those two deliverables:

1. **Completed Graduation Thesis Scope (Frozen / Immutable)**:
   - Chapters 1–6, thesis `RQ1`–`RQ3` (`O1`–`O6`), `P0` historical replay (`workflows/mlwa/00_P0_historical_reproduction.py`), primary `E0` (`01_E0_decision_optimization_resume_v2.py`), primary `E1` (`02_E1_representation_probes_resume_v2.py`), and supplementary analyses `03`–`05` under `result/`.
   - All archived DMIP outputs and thesis `result/` directories MUST remain untouched and strictly read-only.

2. **MLWA Journal Extension Scope (Opened by This Extension)**:
   - Author-authorized implementation, auditing, protocol locking, tests, and Colab/local run pipelines for:
     - **P0 & Pooled Five-Head Completion** (`workflows/mlwa/10_pooled_heads_completion.py`),
     - **E2 Controlled 2×2 Head × Objective Factorial** (`07_E2_factorial_training.py`),
     - **E2b Link Ablation: logit vs. cloglog** (`08_E2b_link_ablation.py`),
     - **E4 Mandatory Imbalance Baselines: Cui effective-number CB-CE & Menon logit adjustment** (`09_E4_imbalance_baselines.py`),
     - **Refreshed E0/CIBD & E1 Representation Probes** (`11_journal_decoders_and_probes.py`),
     - **E3 Gated Bounded Residual Correction** (`12_E3_bounded_residual.py`, gated by paired H3 criterion),
     - **E5 Opt-In Unlabelled Target Prior Adaptation (Saerens EM & BBSE)** (`13_E5_unlabelled_prior_adaptation.py`),
     - **E6 Opt-In Generality & G2 Allocation Gate** (`14_E6_generality.py`),
     - **Verified-Only Journal Reporting & Figure Generation** (`15_journal_reports.py`).
   - All journal outputs MUST write to distinct `protocol_id` / `run_id` folders under an explicit journal output root (`outputs/` or `DR_JOURNAL_OUTPUTS`), never overwriting `result/`.

> **Epistemic Integrity Note**: Opening `E2`–`E6` for the MLWA extension does **not** alter the historical thesis scope, does **not** imply retrospective preregistration of completed thesis or DMIP runs, and does **not** record author ratification of `v0.1.0-draft`. Any analyses executed on the six existing LODO cohorts after thesis inspection are documented as retrospective / development-cohort analyses anchored by `docs/mlwa/analysis_plan.md` prior to launching journal GPU campaigns.

---

## 2. Separate MLWA Journal Research Questions (Do Not Renumber Thesis RQs)

The graduation thesis retains its original RQ1–RQ3 mapping unchanged. The MLWA journal extension uses the following four-question research map:

- **Journal RQ1 (Decision Layer)**: *With posteriors held fixed, how much of the change in Grade 3 recall and Within-1 accuracy is attributable to the decision rule?* (`E0`, CIBD vs. argmax / median / rounded expectation / cutpoint decoding).
- **Journal RQ2 (Posterior Layer)**: *With the decision rule held fixed, how do the output head and the training objective change posterior quality and grade-specific behaviour?* (`E2` 2×2 factorial across `Softmax`/`CLM` × `CE`/`NLL` with/without symmetric QWK warmup, plus `E2b` link ablation `logit` vs. `cloglog` and pooled five-head completion).
- **Journal RQ3 (Representation Layer)**: *Does the penultimate representation $\mathbf{z}$ retain Grade 3-discriminative information that the one-dimensional projection $s$ discards?* (`E1` frozen-input linear probes on $\mathbf{z}_{\text{nominal}}$, $\mathbf{z}_{\text{ordinal}}$, $s_{\text{ordinal}}$ with within-cohort permutation controls).
- **Journal RQ4 (Intervention & Baselines)**: *Does a source-selected decision layer, and only if warranted a bounded residual correction, improve Grade 3 recall within stated severity-proximity constraints across cohorts and seeds relative to simple rebalancing baselines?* (Refreshed `E0`/CIBD vs. `E4` Cui CB-CE and Menon logit adjustment, plus gated `E3` BRC if H3 opens, plus optional `E5`/`E6`).

Null, negative, heterogeneous, or cohort-dependent outcomes are valid publishable findings. Neither selection logic nor reporting assumes CIBD, CLM, vector probing, or BRC wins.
