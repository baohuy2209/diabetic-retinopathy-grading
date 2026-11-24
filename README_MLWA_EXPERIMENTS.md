# MLWA Journal Extension — Runbook & Reproducibility Guide

Operational instructions for executing, auditing, resuming, and reporting the **Machine Learning with Applications (MLWA)** journal extension in `D:\Github\diabetic-retinopathy-grading`.

> **Read-only rule**: Historical thesis outputs under [`result/`](result/) (`p0_fp32_main_lodo_v1/`, `p0_fp32_main_lodo_v1_e0_resume_v2/`, `p0_fp32_main_lodo_v1_e1_resume_v2/`) and the historical 6-chapter thesis ZIP are **read-only references**. All journal verification, preflight, smoke, and GPU training runs MUST write under `outputs/` (or `DR_JOURNAL_OUTPUTS`) using distinct `run_id`s.

---

## 1. Scope & Governance Documents

- [`docs/mlwa/journal_scope_addendum.md`](docs/mlwa/journal_scope_addendum.md) — MLWA journal scope (`RQ1`–`RQ4`, `E2`–`E6`), preserving [`constitution.md`](.specify/memory/constitution.md) (`v0.1.0-draft`).
- [`docs/mlwa/analysis_plan.md`](docs/mlwa/analysis_plan.md) — Prespecified experimental & statistical protocol (currently `DRAFT`; freeze before production training).
- [`docs/mlwa/implementation_audit.md`](docs/mlwa/implementation_audit.md) — Line-referenced audit of repository discrepancies, asset availability, and literature verification.
- [`docs/mlwa/experiment_registry.json`](docs/mlwa/experiment_registry.json) & [`docs/mlwa/job_matrix.csv`](docs/mlwa/job_matrix.csv) — Authoritative unique job schedule (126 primary GPU training units).
- [`docs/mlwa/drawio_journal_specs.md`](docs/mlwa/drawio_journal_specs.md) — Node/edge/caption specifications for draw.io journal diagrams.

---

## 2. Freezing the Analysis Plan (Required Before Production Training)

Production training workflows (`07`, `08`, `09`) verify `docs/mlwa/analysis_plan.lock.json` via `verify_frozen_plan(...)` and **refuse production execution** if `status != FROZEN` or the plan digest mismatches (unless `--smoke` or `--dry-run` is explicitly passed for `TEST_ONLY` verification).

```powershell
# 1. Inspect or finalize docs/mlwa/analysis_plan.md
#    (Update **status**: FROZEN in docs/mlwa/analysis_plan.md once final parameters are locked.)
python -m experiments.mlwa.plan freeze docs/mlwa/analysis_plan.md --out docs/mlwa/analysis_plan.lock.json

# 2. Verify lock integrity
python -m experiments.mlwa.plan verify docs/mlwa/analysis_plan.md --lock docs/mlwa/analysis_plan.lock.json
```

---

## 3. Environment Setup (Local Windows & Google Colab)

### Local Windows / Workstation

Configure dataset & asset paths via environment variables (no hardcoded external paths inside repo modules; see also [`configs/local.example.ps1`](configs/local.example.ps1)):

```powershell
$env:DR_ROOT = (Get-Location).Path
$env:DR_INPUTS = "D:\Github\deep-learning-for-computer-vision\data"
$env:DR_FOLDS = "D:\Github\deep-learning-for-computer-vision\data\processed\lodo_folds"
$env:DR_IMAGES = "D:\Github\deep-learning-for-computer-vision\data\raw\native_cropped"
$env:DR_POOLED_CHECKPOINTS = "D:\Github\deep-learning-for-computer-vision\checkpoints_head_ablation"
$env:DR_POOLED_METADATA = "D:\Github\deep-learning-for-computer-vision\data\raw\hybrid"
$env:DR_P0_RUN = "$env:DR_ROOT\result\p0_fp32_main_lodo_v1"
$env:DR_JOURNAL_OUTPUTS = "$env:DR_ROOT\outputs\mlwa_journal_runs"
```

### Google Colab (Drive + Standalone Bundle)

1. Build the deterministic source bundle locally (re-run whenever `experiments/` changes):

```powershell
python tools/build_mlwa_bundle.py
# Produces dist/mlwa_journal_bundle.zip and dist/mlwa_journal_bundle.sha256
```

2. Upload `dist/mlwa_journal_bundle.zip` to Google Drive at `/content/drive/MyDrive/dr_thesis/inputs/mlwa_journal_bundle.zip` (or set `DR_MLWA_BUNDLE`), along with extracted/zipped fundus images and `lodo_folds` manifests.
3. Open any companion notebook under `notebooks/mlwa/06_...ipynb`–`15_...ipynb`. Each notebook executes `_bootstrap_colab_repo()` to mount Drive, extract `mlwa_journal_bundle.zip` to local fast scratch (`$DR_WORK/mlwa_pkg`), add it to `sys.path`, and save checkpoints atomically under `DR_JOURNAL_OUTPUTS`.

---

## 4. Workflow Index & Job Schedule

All workflows live in `workflows/mlwa/` with byte-verified Colab mirrors in `notebooks/mlwa/`:

| Script | Role | Regime / Gate | GPU / CPU |
|---|---|---|---|
| `06_journal_preflight_and_pilot.py` | Preflight manifest/asset audit, cache validation & 1-epoch pilot profiler | Pre-run audit | CPU (preflight) / GPU (pilot) |
| `07_E2_factorial_training.py` | Controlled 2×2 Softmax/CLM × CE/NLL±QWK factorial (`72` units) | Frozen plan, `natural_source_train` | GPU (`T4`/`L4`/`A100`) |
| `08_E2b_link_ablation.py` | `CLM(logit)` vs `CLM(cloglog)` link ablation (`18` new units; reuses `E2-D`) | Frozen plan | GPU |
| `09_E4_imbalance_baselines.py` | Mandatory `Cui CB-CE` & `Menon Logit Adj` (`36` new units; reuses `E2-A`) | Frozen plan | GPU |
| `10_pooled_heads_completion.py` | Strict audit/reconstruction of historical pooled 5-head study (`PENDING_ASSETS` aware) | Audit / `PENDING_ASSETS` | CPU audit / GPU if trained |
| `11_journal_decoders_and_probes.py` | Refreshed E0 decoders (`select_cibd`, 1,000-rep stability, RDR) + E1 probes | Strict source-only selection | CPU |
| `12_E3_bounded_residual.py` | Gated E3 Bounded Residual Correction (`BRC`) + 7 controls | **Gated by H3** (`SKIPPED_GATE_NOT_MET`) | CPU / GPU if gate opens |
| `13_E5_unlabelled_prior_adaptation.py` | Opt-in E5 Saerens EM & Lipton BBSE (`--enable-e5`) | `TRANSDUCTIVE_ADAPTATION` (no target labels) | CPU |
| `14_E6_generality.py` | Opt-in E6 G2 gate, pretraining exposure audit & external interfaces | **Gated by G2** + exposure audit | CPU |
| `15_journal_reports.py` | Verified-only CSV tables, PNG/PDF plots, `experiment_coverage.csv`, `missing_runs.csv` | Read-only synthesis | CPU |

### Job Count Accounting (Zero Duplication)

- **Primary architecture (`primary_dual_branch`)**:
  - `E2`: `4 cells (A, B, C, D) × 6 folds × 3 seeds` = **72** unique training units
  - `E2b`: `1 new arm (CLM logit) × 6 folds × 3 seeds` = **18** unique training units (`cloglog` reuses `E2-D`)
  - `E4` mandatory: `2 baselines (Cui CB-CE, Menon Logit Adj) × 6 folds × 3 seeds` = **36** unique training units (`unweighted CE` reuses `E2-A`)
  - **Primary total**: **126 unique GPU training units** (162 virtual rows minus 36 reused `E2-A`/`E2-D` runs).
- **Documented fallback architecture (`fallback_global_resnet50`)**, only if pilot profiling shows `primary_dual_branch` exceeds compute budget (never auto-switched silently):
  - `72` ResNet-50 `E2` units + `18` `E2b` + `36` `E4` + `36` matched `V2`/`V3` dual-branch confirmation units = **162 unique GPU units**.

---

## 5. Step-by-Step Execution Commands

### Step 1: Preflight & Asset Audit (CPU)

```powershell
python workflows/mlwa/06_journal_preflight_and_pilot.py `
  --preflight `
  --folds-dir $env:DR_FOLDS `
  --p0-dir result/p0_fp32_main_lodo_v1 `
  --pooled-checkpoints-dir $env:DR_POOLED_CHECKPOINTS `
  --pooled-metadata-dir $env:DR_POOLED_METADATA `
  --output-dir outputs/mlwa_journal_preflight
```

### Step 2: Baseline Reports from Completed Legacy P0 / E0 / E1 (CPU)

```powershell
python workflows/mlwa/15_journal_reports.py `
  --p0-dir result/p0_fp32_main_lodo_v1 `
  --e0-dir result/p0_fp32_main_lodo_v1_e0_resume_v2 `
  --e1-dir result/p0_fp32_main_lodo_v1_e1_resume_v2 `
  --output-dir outputs/mlwa_journal_reports_baseline
```

### Step 3: Multi-Seed GPU Training (Per-Fold / Per-Seed, Epoch-Boundary Resume)

After locking `docs/mlwa/analysis_plan.md` (`FROZEN`) and freezing `analysis_plan.lock.json`, execute one fold/seed shard at a time (e.g. Fold 0, Seed 0):

```powershell
# Inspect exact jobs before running
python workflows/mlwa/07_E2_factorial_training.py --list-jobs

# Run single shard (supports automatic epoch-boundary resume on restart into the same --output-dir)
python workflows/mlwa/07_E2_factorial_training.py `
  --cells A B C D --folds 0 --seeds 0 `
  --folds-dir $env:DR_FOLDS --images-root $env:DR_IMAGES `
  --output-dir outputs/mlwa_journal_runs --run-id mlwa_e2_v1

# E2b link ablation (CLM logit)
python workflows/mlwa/08_E2b_link_ablation.py `
  --folds 0 --seeds 0 `
  --folds-dir $env:DR_FOLDS --images-root $env:DR_IMAGES `
  --output-dir outputs/mlwa_journal_runs --run-id mlwa_e2b_v1

# E4 imbalance baselines (Cui CB-CE & Menon Logit Adj)
python workflows/mlwa/09_E4_imbalance_baselines.py `
  --baselines cui_cb_ce menon_logit_adj_ce --folds 0 --seeds 0 `
  --folds-dir $env:DR_FOLDS --images-root $env:DR_IMAGES `
  --output-dir outputs/mlwa_journal_runs --run-id mlwa_e4_v1
```

#### Atomic Resume & Integrity Behavior
- Each epoch atomically writes `resume_latest.pt` (model, optimizer, LR scheduler, RNG states, best metrics, epoch history), `epoch_metrics.csv`, `epoch_history/epoch_XXX/source_val.{parquet|csv}`, and `last_checkpoint.sha256`.
- Resuming verifies exact match on `config_id`, `fold_id`, `seed_train`, `source_train_manifest_sha256`, `source_val_manifest_sha256`, `plan_sha256`, and `code_sha256` before continuing, and skips units that already possess a verified `COMPLETE.json`.

### Step 4: Downstream Decoders, Probes, Gated/Opt-In Studies & Final Reports (CPU)

```powershell
python workflows/mlwa/11_journal_decoders_and_probes.py --journal-runs-dir outputs/mlwa_journal_runs --output-dir outputs/mlwa_e0_e1_refreshed
python workflows/mlwa/12_E3_bounded_residual.py --probe-csv outputs/mlwa_e0_e1_refreshed/e1_probe_metrics.csv --evidence-type journal_multiseed --output-dir outputs/mlwa_e3_brc
# Optional E5 transductive prior adaptation (only if explicitly included as supplementary regime):
python workflows/mlwa/13_E5_unlabelled_prior_adaptation.py --enable-e5 --p0-dir result/p0_fp32_main_lodo_v1 --output-dir outputs/mlwa_e5_prior_adaptation
# Regenerate all verified tables, figures, coverage & missing-run manifests:
python workflows/mlwa/15_journal_reports.py --p0-dir result/p0_fp32_main_lodo_v1 --e0-dir result/p0_fp32_main_lodo_v1_e0_resume_v2 --e1-dir result/p0_fp32_main_lodo_v1_e1_resume_v2 --journal-runs-dir outputs/mlwa_journal_runs --output-dir outputs/mlwa_journal_reports
```

---

## 6. Bounded CPU Smoke Verification (`TEST_ONLY`)

All workflows accept `--smoke` to validate contracts, shapes, gradient flow, metrics, gates, and resume behavior on synthetic data without downloading pretrained weights or touching `result/`:

```powershell
python workflows/mlwa/06_journal_preflight_and_pilot.py --smoke --output-dir outputs/mlwa_smoke_test_only/06
python workflows/mlwa/07_E2_factorial_training.py --smoke --folds 0 --seeds 0 --output-dir outputs/mlwa_smoke_test_only/07
python workflows/mlwa/08_E2b_link_ablation.py --smoke --folds 0 --seeds 0 --output-dir outputs/mlwa_smoke_test_only/08
python workflows/mlwa/09_E4_imbalance_baselines.py --smoke --folds 0 --seeds 0 --output-dir outputs/mlwa_smoke_test_only/09
python workflows/mlwa/11_journal_decoders_and_probes.py --smoke --output-dir outputs/mlwa_smoke_test_only/11
python workflows/mlwa/12_E3_bounded_residual.py --force-smoke-test-only --output-dir outputs/mlwa_smoke_test_only/12
python workflows/mlwa/13_E5_unlabelled_prior_adaptation.py --smoke --output-dir outputs/mlwa_smoke_test_only/13
python workflows/mlwa/14_E6_generality.py --smoke --output-dir outputs/mlwa_smoke_test_only/14
python workflows/mlwa/15_journal_reports.py --smoke --output-dir outputs/mlwa_smoke_test_only/15
```

Notebook synchronization and full test suite:

```powershell
python tools/build_notebooks.py --check
python -m pytest -q tests/test_smoke.py tests/test_camera_ready.py
python -m pytest -q
```
