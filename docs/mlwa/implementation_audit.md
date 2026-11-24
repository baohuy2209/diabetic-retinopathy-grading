# MLWA Journal Extension — Implementation & Asset Audit

**Audit Date**: 2026-11-24  
**Repository**: `D:\Github\diabetic-retinopathy-grading`  
**Git Commit**: `c3e94a616e29dfe0df2ecc7e85339a3761351766` (checked after restoring `nbformat==5.11.1` and syncing `05_within1_transition_decomposition.ipynb`; 49/49 baseline pytest checks passing)

---

## 1. Experiment & Implementation Inventory

| Study / Code | Status | Code Path(s) | Available Verified Artifacts | Gaps / Missing Assets / Notes |
| --- | --- | --- | --- | --- |
| **Historical DMIP LODO (`V0`–`V3`)** | **Executed (Archived)** | `workflows/dmip/05_lodo_component_analysis.py`, `experiments/camera_ready.py` | External `D:\Github\DR-evaluation-study2026\result\reference\lodo_summary_with_calibrated_screening.json`, 24 NPZs (`raw_preds/`), 48 LODO `.pth` (`best` + `last` for `V0`–`V3` × 6 folds) | Single training seed (`seed=42`); NPZs lack `image_id` (manifest order required). |
| **P0 Historical LODO Replay (`V2`, `V3`)** | **Executed (`DIAGNOSTIC`)** | `workflows/mlwa/00_P0_historical_reproduction.py` | `result/p0_fp32_main_lodo_v1/` (12 units × 3 splits = 36/36 `predictions.csv`, 36/36 `z.npy`, 36/36 `COMPLETE.json`) | `execution_status=INFERENCE_COMPLETE`, **`G0_STATUS=DIAGNOSTIC`** (`g0_comparison.csv`: 192 checks = `118 PASS`, `74 DISCREPANCY`). Does NOT pass G0 benchmark tolerance; downstream E0/E1 anchor on `pred_native` replay. |
| **P0 Five-Head Pooled Replay** | **`PENDING_ASSETS`** | `workflows/mlwa/00_P0_historical_reproduction.py`, `workflows/mlwa/10_pooled_heads_completion.py` | `result/p0_fp32_main_lodo_v1/pooled_asset_status.json` (`RUN_POOLED=False`); pooled CSVs (`9,079 / 8,238 / 8,075`) on local `D:` | Local `checkpoints_head_ablation/*.pth` (5 heads) and `checkpoints/head_ablation/resnet50_*.pth` (2 heads) are all **epoch=1 smoke checkpoints** (`val_qwk=0.0`, `Accuracy=0.20–0.25`), NOT the trained 25-epoch checkpoints. Exact constructor of `backbone.projection.1` in `checkpoints_head_ablation/` is unverified. |
| **Primary E0 (CIBD & Decoders)** | **Executed (Single-Seed)** | `workflows/mlwa/01_E0_decision_optimization_resume_v2.py` | `result/p0_fp32_main_lodo_v1_e0_resume_v2/` (`RUN_STATUS=REQUESTED_SCOPE_ANALYSIS_COMPLETE`, `p0_g0=DIAGNOSTIC`) | Uses `seed=42` replayed checkpoints; `select_cibd` records `qwk_guardrail_met` without filtering on it; patient-cluster CIs are `NOT_ESTIMABLE`. |
| **Primary E1 (Linear Probes)** | **Executed (Single-Seed)** | `workflows/mlwa/02_E1_representation_probes_resume_v2.py` | `result/p0_fp32_main_lodo_v1_e1_resume_v2/` (`RUN_STATUS=REQUESTED_SCOPE_ANALYSIS_COMPLETE`) | Equal-cohort target Grade 3 recall means: `z_v2=0.446396`, `z_v3=0.510868`, **`s_v3=0.540930`** (`s_v3 > z_v3`). |
| **Thesis Supplementary (`03`–`05`)** | **Executed (Post hoc / CPU)** | `workflows/mlwa/03_*`, `04_*`, `05_*` | `result/thesis_supplementary_v1/`, `thesis_within1_decomposition_v1/`, `thesis_figures_v2/` (plus `v2_verify`, `v2_nb_verify`, `v2_partial_g`) | `05` notebook had stale `code_sha256` at checkout; now re-synced cleanly. Never conflate partial `v2_partial_g` with full `v1`/`v2_verify`. |
| **E2 Controlled 2×2 Factorial** | **Implemented (Not Yet GPU Run)** | `experiments/mlwa/`, `workflows/mlwa/07_E2_factorial_training.py` | Smoke-tested only (`TEST_ONLY`) | Requires 72 GPU training runs (`4 cells × 6 folds × 3 seeds {0,1,2}`) on full dual-branch architecture (or 72 + 36 fallback). |
| **E2b Link Ablation (`logit` vs `cloglog`)** | **Implemented (Not Yet GPU Run)** | `experiments/mlwa/models_adapters.py`, `workflows/mlwa/08_E2b_link_ablation.py` | Smoke-tested only (`TEST_ONLY`) | Requires 18 new GPU runs (`logit` CLM-NLL+QWK × 6 folds × 3 seeds), reusing 18 `E2-D` (`cloglog`) runs. |
| **E4 Imbalance Baselines (Cui, Menon)** | **Implemented (Not Yet GPU Run)** | `experiments/mlwa/objectives.py`, `workflows/mlwa/09_E4_imbalance_baselines.py` | Smoke-tested only (`TEST_ONLY`) | Requires 36 mandatory GPU runs (`2 baselines × 6 folds × 3 seeds`), reusing 18 `E2-A` runs. `exp_MSE` marked `NEEDS_METHOD_SPEC`. |
| **Refreshed Journal E0/E1** | **Implemented** | `workflows/mlwa/11_journal_decoders_and_probes.py` | Ready for verified P0 or new E2/E2b/E4 outputs | Awaits E2/E4 training completion for multi-seed tables. |
| **E3 BRC (Gated by H3)** | **Implemented (Gate Closed)** | `experiments/mlwa/brc.py`, `workflows/mlwa/12_E3_bounded_residual.py` | Legacy E1 (`s_v3 > z_v3`) keeps gate **`SKIPPED_GATE_NOT_MET`** | Production disabled unless journal multi-seed E1 opens the paired H3 gate. |
| **E5 Prior Adaptation (Opt-in)** | **Implemented (Opt-in)** | `experiments/mlwa/prior_shift.py`, `workflows/mlwa/13_E5_unlabelled_prior_adaptation.py` | CPU-only Saerens EM & BBSE ready; opt-in via `--enable-e5` | Outputs segregated into `TRANSDUCTIVE_ADAPTATION` tables. |
| **E6 Generality & G2 Gate (Opt-in)** | **Implemented (Gate/Adapter)** | `experiments/mlwa/generality.py`, `workflows/mlwa/14_E6_generality.py` | Interfaces & G2 gate report ready (`SKIPPED_GATE_NOT_MET` until E2/E4 run) | No automatic weight downloads or dataset scraping. |

---

## 2. Inventory of Existing `result/` Directories (Strictly Read-Only)

- `result/p0_fp32_main_lodo_v1/`: `INFERENCE_COMPLETE`, **`G0_STATUS=DIAGNOSTIC`** (118 PASS / 74 DISCREPANCY), `pooled_asset_status=PENDING_ASSETS`, 36/36 shards complete. **Authoritative P0 reference cache.**
- `result/p0_fp32_main_lodo_v1_e0_resume_v2/`: `REQUESTED_SCOPE_ANALYSIS_COMPLETE`, anchored to `p0_fp32_main_lodo_v1`, `code_sha256=e71d402a...`, `plan_sha256=0fd996e6...`.
- `result/p0_fp32_main_lodo_v1_e1_resume_v2/`: `REQUESTED_SCOPE_ANALYSIS_COMPLETE`, anchored to `p0_fp32_main_lodo_v1`, `code_sha256=3405a3c9...`, `plan_sha256=0fd996e6...`.
- `result/thesis_supplementary_v1/`: Primary thesis supplementary report (all A–G executed; earlier provenance schema without `execution_mode`).
- `result/thesis_supplementary_v2_verify/` & `thesis_supplementary_v2_nb_verify/`: Post-audit CLI and clean-notebook parity verifications across analyses `A`–`G`.
- `result/thesis_supplementary_v2_partial_g/`: Partial CLI verification running **only analysis `G`** (`selected_analyses=['G']`) — **NEVER** treat as a complete A–G substitute.
- `result/thesis_within1_decomposition_v1/` & `result/thesis_figures_v2/`: Completed Within-1 paired transition decomposition, 0.5 pp grouping-key bootstrap, and revised thesis Overleaf bundle.

---

## 3. Data, Denominator & Natural vs. Quota Manifest Audit

Verified directly against `D:\Github\deep-learning-for-computer-vision\data\processed\lodo_folds` and `D:\Github\deep-learning-for-computer-vision\data\raw\hybrid`:

1. **Canonical Fold Ordering (Locked)**:
   `0=aptos`, `1=idrid`, `2=messidor2`, `3=ddr`, `4=eyepacs`, `5=deepdrid`.
   *(Warning: `config.py:DataConfig.registered_cohorts` lists `['aptos', 'ddr', 'deepdrid', 'idrid', 'messidor2', 'eyepacs']` — never use `registered_cohorts` index for fold IDs!)*
2. **Inventory vs. Evaluated LODO Target Denominators**:
   - Inventory across 6 cohorts: **55,570 images**, **35,474 dataset-specific grouping keys** (`legacy_unverified`, not verified patient IDs).
   - Evaluated LODO targets (`sum of fold_*_test.csv`): **54,883 images** (APTOS 3,662; IDRiD 516; Messidor-2 **1,057** vs 1,744 inventory; DDR 12,522; EyePACS 35,126; DeepDRiD 2,000).
3. **CRITICAL FINDING — Quota-Balanced `fold_*_train.csv` vs. Natural `source_train`**:
   All six archived `fold_{0..5}_{cohort}_train.csv` files are **unique-image quota-balanced subsets**, NOT natural 85% partitions:
   - Fold 0 (`aptos` out): quota `9,202`, `val=7,891` (`15.4%`), recovered natural `source_train=43,330` (`84.6%`, `G0..G4 = {28281, 3068, 9005, 1373, 1603}`).
   - Fold 1 (`idrid` out): quota `10,352`, `val=8,357` (`15.4%`), recovered natural `source_train=46,010` (`84.6%`, `G0..G4 = {29747, 3383, 9623, 1476, 1781}`).
   - Fold 2 (`messidor2` out): quota `9,786`, `val=8,143` (`15.1%`), recovered natural `source_train=45,683` (`84.9%`, `G0..G4 = {29444, 3258, 9643, 1503, 1835}`).
   - Fold 3 (`ddr` out): quota `8,224`, `val=6,560` (`15.5%`), recovered natural `source_train=35,801` (`84.5%`, `G0..G4 = {24525, 2866, 5981, 1360, 1069}`).
   - Fold 4 (`eyepacs` out): quota `7,159`, `val=2,966` (`15.0%`), recovered natural `source_train=16,791` (`85.0%`, `G0..G4 = {7982, 1345, 5351, 856, 1257}`).
   - Fold 5 (`deepdrid` out): quota `8,543`, `val=7,990` (`15.1%`), recovered natural `source_train=44,893` (`84.9%`, `G0..G4 = {29222, 3124, 9577, 1254, 1716}`).
   Across all 6 folds, `quota_train ⊆ natural_source_train` (`tr_subset_nat=True`) and `patient_id` overlap between `natural_source_train` and `val` is strictly **0**.
   - **Implication**: Journal `E2` and `E4` MUST use the reconstructed **natural-prevalence `source_train`** (preserving exact 85/15 grouped partitions) for unweighted E2 cells (`A`–`D`) and for natural `n_j` / `pi_j` in Cui CB-CE and Menon logit adjustment. Quota-balanced `fold_*_train.csv` priors are retained ONLY under legacy/quota-protocol adapters.
4. **Pooled Metadata**: `train_metadata.csv` (`9,079`, `6,729` keys), `train_balanced_metadata.csv` (`9,079`, `6,729` keys), `val_metadata.csv` (`8,238`, `5,218` keys), `test_metadata.csv` (`8,075`, `5,219` keys).

---

## 4. Architecture, Preprocessing & Objective Discrepancies Found in Existing Code

| Component | Modular (`models/`, `losses/`, `training/`) | Reference (`evaluation/checkpoint_reference/` & P0) | Resolution in `experiments/mlwa/` |
| --- | --- | --- | --- |
| **Fusion attribute & dim** | `DualBranchMILModel.fusion` (`4096 + 2560 = 6656 -> 512`) | `DualBranchMILModel.fusion_layer` (`4096 + 2560 = 6656 -> 512`) | Explicit adapter `JournalDualBranchEncoder` with recorded `architecture_id` & `forward_id`; never infer compatibility from `512-D` width alone. |
| **QWK warmup parity** | `SoftmaxQWKHead` lacks `set_qwk_weight`/`target_qwk_weight`; `update_qwk_warmup` silently sets only CLM! | Both pooled heads warm up (`0.0 -> 0.4`), whereas LODO `CLM` defaulted to `qwk_weight=1.2` then patched to `0.4`. | `experiments/mlwa/objectives.py` applies identical `compute_qwk_weight()` schedule to both Cell B (`Softmax+CE+QWK`) and Cell D (`CLM+NLL+QWK`). |
| **CLM clamp & pure NLL** | `models/heads/clm.py`: `clamp(-15, 15)`, `current_qwk_weight=0.4` by default | `checkpoint_reference`: `clamp(-10, 4)`, `current_qwk_weight=0.0` | `JournalCLMHead` decouples link (`cloglog`/`logit`/`probit`) from loss; genuine pure `clm_nll` (`qwk_weight=0`) and stable FP32 `-expm1(-exp(u))`. |
| **CORAL / CORN threshold crossing** | Silently applies `F.relu(rank_probs[:, k-1] - rank_probs[:, k])` in `forward` | Also uses `F.relu` and logs `rank_monotonicity_violations` separately | `JournalCORALHead` / `JournalCORNHead` compute raw differences first and explicitly report `monotonicity_violation_rate` + `has_threshold_crossing`. |
| **Tie breaking** | `torch.round` (round-to-even) in heads | `torch.round` in P0; `cibd(..., 1, 0, 2)` breaks ties to **lower grade** | Preserve `decode_native_legacy_round` separately from canonical risk-minimizing `decode_rounded_expectation_bayes`. |
| **Class weighting / balancing** | `data.compute_class_weights` = inverse frequency; `balance_training_records` = resampling | Both unused in primary MLWA runs | Separate `cui_cb_ce` (Cui 2019), `menon_logit_adj_ce` (Menon 2021), `galdran_cost_sensitive_ce` (Galdran 2020), and `exp_mse` (`NEEDS_METHOD_SPEC`). |
| **CIBD selection constraint** | `select_cibd` computes `qwk_guardrail_met` but filters only on `within1` `feasible` | Same in `01_E0_decision_optimization_resume_v2.py` | Keep `legacy_within1_1pp_v1` unchanged; version `journal_dual_guardrail_v1` and `journal_within1_05pp_v1` separately. |
| **Pooled checkpoint families** | `ResNet50AblationModel`: `stem.*`, `layer1..4`, `proj.0..1`, `head.*` | `checkpoints_head_ablation/*.pth`: `backbone.*`, `backbone.projection.0`, `backbone.projection.2` (epoch=1 smoke only) | Strict adapters refusing `strict=False`; `10_pooled_heads_completion.py` writes `PENDING_ASSETS` for unverified/missing 25-epoch checkpoints. |

---

## 5. PDF Plan (`G-12.10.26.pdf`) vs. Code/Data Discrepancies

1. **Historical single-seed P0 vs. multi-seed `{0,1,2}`**: PDF assumes fresh 3-seed E2/E4 runs; historical seed 42 checkpoints are NOT part of `{0,1,2}` and must never be counted toward the `24` or `72` runs.
2. **Quota-balanced vs. patient-level 85/15 source train**: Existing `fold_*_train.csv` were quota-balanced; natural 85/15 `source_train` is reconstructed deterministically from non-target `test.csv` minus `val.csv`.
3. **Appendix B `"V3 + class-balanced CE"`**: V3 uses a scalar CLM head trained with categorical NLL, not a nominal Softmax CE logit vector. Journal `E4` keeps nominal `cui_cb_ce` & `menon_logit_adj_ce` strictly paired against Softmax-CE baseline (`A`), with optional typed CLM variants (`clm_cui_weighted_nll`, `clm_logprob_adjusted_nll`) separated as extra arms.
4. **Literature table verification**: Primary papers verified directly: Cui et al. (CVPR 2019), Menon et al. (ICLR 2021), Kang et al. (ICLR 2020), Kirichenko et al. (ICLR 2023), Guo et al. (ICML 2017), Hewitt & Liang (EMNLP 2019), Saerens et al. (Neural Comp 2002), Lipton et al. (ICML 2018), Galdran et al. (MICCAI 2020 Cost-Sensitive Regularization: $\mathcal{L}_{\mathrm{CE}} + \lambda \sum_c M_{y,c} p_c$, $M_{y,c}=|y-c|^2$). Stelter et al. (NLDL PMLR 2026 `exp_MSE`) lacks a verified equation in the offline workspace and is therefore gated as **`NEEDS_METHOD_SPEC`**.
5. **What cannot be satisfied from saved summaries alone**: E2 2×2 factorial (`72` runs), E2b logit link (`18` runs), E4 baselines (`36` runs), per-epoch `source_val` probability trajectories, patient-cluster CIs (blocked by `legacy_unverified` identity keys), and 25-epoch historical 5-head pooled evaluation (blocked by `PENDING_ASSETS` of trained checkpoints).
