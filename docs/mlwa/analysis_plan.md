# MLWA Journal Extension — Analysis Plan

**status**: DRAFT  
**created_date**: 2026-11-24  
**protocol_id**: mlwa_journal_v1  
**study_id**: mlwa_dr_ordinal_extension  
**base_commit**: c3e94a616e29dfe0df2ecc7e85339a3761351766  
**freeze_command**: `python -m experiments.mlwa.plan freeze docs/mlwa/analysis_plan.md --out docs/mlwa/analysis_plan.lock.json`

> Production training, pooled evaluation, and journal inference commands MUST verify the frozen lock file (`analysis_plan.lock.json`) and refuse execution if `status != FROZEN` or the plan digest does not match (unless `--smoke` explicitly creates a `TEST_ONLY` run).

---

## 1. Research Questions (MLWA Journal Map)

1. **RQ1**: With posteriors held fixed, how much of the change in Grade 3 recall and Within-1 accuracy is attributable to the decision rule?
2. **RQ2**: With the decision rule held fixed, how do the output head and the training objective change posterior quality and grade-specific behaviour?
3. **RQ3**: Does the penultimate representation $\mathbf{z}$ retain Grade 3-discriminative information that the one-dimensional projection $s$ discards?
4. **RQ4**: Does a source-selected decision layer, and only if warranted a bounded residual correction, improve Grade 3 recall within stated severity-proximity constraints across cohorts and seeds relative to simple rebalancing baselines?

---

## 2. Cohorts, Denominators & Separate Seed Policy

- **Locked canonical LODO fold order**:
  `0=aptos`, `1=idrid`, `2=messidor2`, `3=ddr`, `4=eyepacs`, `5=deepdrid`.
- **Populations & identities**:
  - Six-cohort inventory: `55,570` images (`35,474` dataset-specific grouping keys).
  - Archived six-fold evaluated targets: `54,883` images (`Messidor-2`: `1,057` evaluated vs. `1,744` inventory).
  - Patient identity status: recorded keys without verified clinical linkage MUST remain `patient_identity_status = "legacy_unverified"`.
- **Natural vs. Quota Training Sampling Policy**:
  - Primary journal `E2` / `E2b` / `E4` protocol uses **`natural_source_train`** (reconstructed deterministically per fold from the 5 non-target `test` cohorts minus `source_val`, preserving exact 85/15 grouped separation and exact natural class priors $\boldsymbol{\pi}_{\text{source\_train}}$).
  - Historical quota-balanced `fold_*_train.csv` is retained strictly under separate `sampling_policy = "quota_balanced"` for compatibility audits.
- **Separated RNG Seeds (Never Re-Generate Splits on Training Seed Change)**:
  - `seed_split = 42` (locked manifests),
  - `seed_train ∈ {0, 1, 2}` (three independent training runs per cohort/configuration; historical `seed=42` checkpoints are never counted as seed 0, 1, or 2),
  - `seed_probe_control ∈ {0, 1, 2}` (within-source-cohort label permutations in E1),
  - `seed_resampling = 42` (paired bootstraps and selector stability resampling).

---

## 3. Experimental Designs, Shared Budget & Hyperparameter Policy

### 3.1 E2 Controlled 2×2 Head × Objective Factorial (`72` Primary Units)
- **Cells** (`4 cells × 6 folds × 3 seeds = 72` unique runs):
  - **Cell A**: `Softmax` head + categorical Cross-Entropy (`softmax_ce`, $\lambda=0$).
  - **Cell B**: `Softmax` head + categorical Cross-Entropy + $\lambda(t)\cdot\mathcal{L}_{\mathrm{QWK}}$ (`softmax_ce_qwk`).
  - **Cell C**: `CLM(cloglog)` head + pure categorical NLL (`clm_nll`, $\lambda=0$ — genuine pure NLL mode).
  - **Cell D**: `CLM(cloglog)` head + categorical NLL + $\lambda(t)\cdot\mathcal{L}_{\mathrm{QWK}}$ (`clm_nll_qwk`).
- **Architecture tiers (Explicit, Never Auto-Switched on OOM)**:
  - Primary preference: `primary_dual_branch` (`DualBranchMILModel` — 4,096-D global + 2,560-D local MIL -> 512-D).
  - Documented fallback: `fallback_global_resnet50` (`72` global ResNet-50 factorial runs + `36` matched `V2`/`V3` full dual-branch confirmation runs over 3 seeds if GPU budget requires fallback).
- **Shared Optimizer, Schedule, Batch & Checkpoint Policy**:
  - Shared initial encoder+fusion digest per `(fold_id, seed_train)` verified via SHA-256 across Cells A–D.
  - Optimizer: `AdamW`, `lr = 1e-4`, `weight_decay = 1e-4` (with bias/norm parameter group separation), `grad_clip = 1.0`.
  - LR schedule: `CosineAnnealingLR(T_max = 30, eta_min = 1e-6)`, `max_epochs = 30`, early stopping `patience = 15`.
  - Effective optimization batch & QWK observation batch: physical `batch_size = 12` (`grad_accum_steps = 1` default so BatchNorm and batch-coupled `ContinuousQWKLoss` operate on the exact same batch; any `grad_accum_steps > 1` is logged explicitly as differing in BN/QWK batch size).
  - Symmetric QWK warmup schedule (Cells B and D identically): linear warmup `0.0 -> 0.4` over `15` epochs (`epoch 1..15`), then `0.4`. Store `base_loss` and `qwk_loss` separately per epoch.
  - **Shared inference & checkpoint selection decoder**: **`argmax`** is the locked primary fixed decoder for source-validation checkpoint selection (maximum QWK, tie-break: lower val `base_loss`, earlier epoch) and primary fixed-decoder factorial tables. `posterior_median`, `rounded_expectation_bayes`, and each cell's native decoder are reported in separate labeled sensitivity tables.
  - Source-validation probabilities `p0..p4` (plus `s`, `theta0..3` for CLM) are saved after EVERY epoch so post-hoc checkpoint selection studies require zero retraining. Target evaluation occurs strictly after checkpoint and source-only selections are locked.
- **Paired Factorial Contrasts** (computed within `(fold_id, seed_train)` before rounding):
  - `head_main = 0.5 * ((C - A) + (D - B))`
  - `loss_main = 0.5 * ((B - A) + (D - C))`
  - `interaction = (D - C) - (B - A)`
  - Plus all 4 simple effects (`C - A`, `D - B`, `B - A`, `D - C`).

### 3.2 E2b Link Ablation (`18` New Units)
- Compares `CLM(logit)` vs. `CLM(cloglog)` under identical Cell D (`NLL + symmetric QWK`) settings (`6 folds × 3 seeds`), reusing the 18 `E2-D` runs.

### 3.3 E4 Mandatory Imbalance Baselines (`36` Mandatory Units)
- Reuses `E2-A` (`Softmax + CE`) as common baseline (`6 folds × 3 seeds`):
  1. **Cui et al. (CVPR 2019) Effective-Number Class-Balanced CE (`cui_cb_ce`)**: $w_j \propto (1 - \beta) / (1 - \beta^{n_j})$ using unique natural `source_train` counts $n_j$, default $\beta = 0.9999$ (observed-class mean normalized to 1.0; explicit zero-count handling).
  2. **Menon et al. (ICLR 2021) Logit Adjustment (`menon_logit_adj_ce`)**: training loss $\mathrm{CE}(\mathbf{f}(x) + \tau_{\mathrm{train}}\log(\boldsymbol{\pi}_{\mathrm{source\_train}}), y)$, default $\tau_{\mathrm{train}} = 1.0$, evaluating raw prior-balanced logits `softmax(f(x))` without double post-hoc subtraction.
- Supplementary typed arms (`clm_cui_weighted_nll`, `clm_logprob_adjusted_nll`, `galdran_cost_sensitive_ce`) are explicitly named and never silently substituted for Softmax baselines. `exp_MSE` remains `NEEDS_METHOD_SPEC`.

### 3.4 Decision Layer (E0/CIBD), Tie Conventions & Constraints
- **Deterministic Bayes Tie Convention**: canonical CIBD and Bayes special cases (`argmax`, `posterior_median`, lower-tie `rounded_expectation_bayes`) select the **lower grade** on ties within `TIE_TOL = 1e-12`. Legacy P0/torch round-to-even is preserved separately as `native_legacy_round`.
- **CIBD Cost & Prior Reweighting**: raw $C_\alpha(k, j) = \alpha |k - j|^q + (1 - \alpha)\mathbb{I}[k \neq j]$ over $\alpha \in \{0, 0.1, \dots, 1\}$, $\tau \in \{0, 0.25, 0.5, 0.75, 1\}$, $q \in \{1, 2\}$; $r_j \propto p_j \hat{\pi}_j^{-\tau}$ with Laplace-smoothed `source_train` prior $\hat{\pi}_j$ (`smoothing = 1.0`).
- **Versioned CIBD Selectors (Source-Validation Only)**:
  - `legacy_within1_1pp_v1`: exact legacy E0 policy (`within1 >= native - 0.01`; `qwk_guardrail_met` logged without filtering; fallback to native if empty).
  - `journal_dual_guardrail_v1`: enforces BOTH `within1 >= native - 0.01` AND `qwk >= native - 0.01`.
  - `journal_within1_05pp_v1`: stricter `within1 >= native - 0.005` sensitivity policy.
- **Practical Endpoint Margins (Distinct from Selector Filters & G2 Gate)**:
  - `H1` / primary decision endpoint: target Grade 3 recall gain $\ge +3\text{ pp}$ vs. baseline with target Within-1 loss $\le 1\text{ pp}$ (and endpoint QWK loss $\le 0.01$).
  - `G2` GPU allocation gate: source-selected intervention achieves $\Delta\mathrm{G3\ recall} \ge +5\text{ pp}$ on equal-cohort mean, positive recall change in $\ge 5/6$ cohorts with seed stability, $\Delta\mathrm{Within1} \ge -1\text{ pp}$, $\Delta\mathrm{QWK} \ge -0.01$, and outperforms `E4` baselines.
- **Target Pareto grids** (`target_grid_EXPLORATORY`): strictly descriptive; NEVER used to select any operating point or model.

### 3.5 E1, E3 (Gated), E5 & E6 Policies
- **E1**: Frozen encoder (`model.eval()`), exact paired `image_key` alignment across `z_nominal`, `z_ordinal`, `s_ordinal`; `StandardScaler` + multinomial `LogisticRegression` fitted on `source_train`; `C ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000}` selected on `source_val` (`macro_recall` primary, `qwk` secondary); within-source-cohort label permutations (`seed_probe_control ∈ {0,1,2}`).
- **E3 (BRC, Gated by H3)**: Remains **`SKIPPED_GATE_NOT_MET`** unless prespecified journal multi-seed E1 shows positive paired `z_ordinal - s_ordinal` superiority across cohorts (`>= 5/6` cohorts and positive seed-consistent mean paired delta on Grade 3 recall & macro recall without non-trivial uncertainty collapse — NEVER judged by naive single-metric CI overlap).
- **E5 (Opt-in Prior Adaptation)**: Saerens EM & BBSE strictly without target labels during estimation; outputs segregated in `TRANSDUCTIVE_ADAPTATION` tables; never multiplies $\hat{\pi}_{\text{target}} / \hat{\pi}_{\text{source}}$ together with CIBD $\pi^{-\tau}$.
- **E6 (Opt-in Generality)**: Requires G2 gate open OR explicit `--enable-e6`; requires pretraining exposure audit and separate protocol/endpoint contracts for external ordinal datasets.

---

## 4. Metrics, Aggregation, Uncertainty & Multiplicity

- **Posterior metrics (decoder-independent)**: `nll_mean`, `rps_mean_4` (mean over 4 cumulative thresholds; `rps_sum_4 = 4 * rps_mean_4`), `ece_15` (15 bins, top-probability argmax correctness), `rdr_auroc` from `sum(p2..p4)`.
- **Decision metrics**: `accuracy`, fixed-5-class `macro_recall`, per-grade `recall_g0..4` & `precision_g0..4` & `support_g0..4`, `qwk`, `within1`, `mae`, `predicted_true_g3_ratio`, `g3_to_2_count/frac` (still referable), `g3_to_01_count/frac` (missed RDR). Undefined precision/recall/AUROC stays `NaN` with explicit `*_defined` boolean flags.
- **RDR screening**: threshold chosen strictly on `source_val` ROC at target specificity `0.95` and applied frozen to target.
- **Hierarchical Cohort × Seed Aggregation**:
  1. Per-cohort across seeds: `mean` and sample SD (`ddof=1`).
  2. Primary equal-cohort summary: compute 6-cohort mean within each `seed_train ∈ {0,1,2}`, then report `mean` and sample SD (`ddof=1`) across those **3 seed-level equal-cohort means** (never average 6 per-cohort SDs!).
  3. Image-pooled summary: secondary table clearly labeled `image_pooled`.
  4. Never average hyperparameters (`alpha`, `tau`, `q`, `selected_C`) into a pseudo-deployed config.
- **Uncertainty & Multiplicity**:
  - `patient_cluster` bootstrap (`2,000` reps, `seed=42`) runs ONLY when `patient_identity_status == "verified"` for all images; otherwise `status = "NOT_ESTIMABLE"`. Unverified-key resampling is explicitly named `grouping_key_bootstrap`; conditional image resampling is named `image_bootstrap`.
  - Source-val selector stability bootstrap (`1,000` reps, `seed=42`) holds `source_train` priors fixed.
  - Exact two-sided / paired Wilcoxon across `n = 6` cohort summaries is supporting evidence only (`min two-sided p = 0.03125`); `18` `(cohort × seed)` cells are NEVER treated as independent domains.
  - Primary confirmatory endpoint per family is prespecified above; all additional pairwise contrasts and sensitivity tables are explicitly labeled exploratory.

---

## 5. Resume & Output Schema Contract

- Atomic epoch-boundary checkpoints (`_last.pth`, `_best.pth`, `checkpoint_meta.json`) save weights, optimizer, scheduler, scaler, `epoch`, `global_step`, `active_qwk_weight`, RNG states, `config_hash`, `plan_sha256`, `code_sha256`. Mid-epoch interrupts restart deterministically from the pre-epoch boundary (`RESUMED_FROM_EPOCH_BOUNDARY`).
- Predictions stored in compressed Parquet (`predictions.parquet`, fallback `.csv`) with full required identity/provenance columns, paired with authoritative per-checkpoint `checkpoint_params.json` and `z.npy` + `z_manifest.json` (`COMPLETE.json` with SHA-256 verification).
