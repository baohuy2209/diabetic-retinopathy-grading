# Diabetic retinopathy grading

This repository contains the graduation thesis experiments on five-grade diabetic retinopathy grading. It combines the official DMIP implementation, its camera-ready analyses and the P0, E0 and E1 extension. Each workflow has an executable Python file and a standalone Google Colab notebook with the same code.

## If you have already finished P0

Open [E0 decision optimization](notebooks/mlwa/01_E0_decision_optimization_resume_v2.ipynb) or [E1 representation probes](notebooks/mlwa/02_E1_representation_probes_resume_v2.ipynb) in Colab. In its configuration cell set `P0_RUN` to the **run folder containing `lodo/`**, for example:

```python
P0_RUN = Path('/content/drive/MyDrive/mlwa_journals_p0_e0_e1/p0_fp32_main_lodo_v1')
```

Use the exact folder name from your completed run. A CPU runtime is sufficient for both analyses. You can run them independently in separate sessions with distinct output IDs. Each stages the required P0 files on the local SSD. E0 reads predictions; E1 also reads fusion features. New reports go under `outputs/mlwa/` in the thesis workspace on Drive. The run status includes the historical G0 result.

After a runtime reset, mount the same Drive and run the cells from the top with the same inputs, configuration, package versions and output ID. Each analysis validates completed units before skipping them. E1 resumes after the last completed regularization candidate or control fit. An interrupted optimizer fit restarts that one fit. The thesis version uses its own code fingerprint and output IDs. Reuse a completed P0 directly. To continue a partially completed analysis written by an earlier notebook, use that notebook and its original output ID.

## Experiment index

Each notebook below has a Python counterpart under `workflows/` with the same relative path and stem.

| Study | Notebook | Purpose and input |
| --- | --- | --- |
| DMIP | [Prepare the pooled data](notebooks/dmip/02_prepare_pooled_data.ipynb) | Read six raw cohort image/label sources, crop the images and create pooled patient partitions. |
| DMIP | [Generate LODO manifests](notebooks/dmip/01_generate_lodo_manifests.ipynb) | Use raw metadata and an existing native-cropped image tree to regenerate six train/validation/test partitions. For checkpoint replay, supply the archived splits. |
| DMIP | [Pooled main model](notebooks/dmip/03_pooled_main_model.ipynb) | Historical pooled dual-branch training, evaluation, source screening calibration and visual analyses. |
| DMIP | [Pooled head ablation](notebooks/dmip/04_pooled_head_ablation.ipynb) | Evaluate Softmax CE, Softmax QWK, CORAL, CORN and CLM QWK on the historical pooled protocol. |
| DMIP | [LODO component analysis](notebooks/dmip/05_lodo_component_analysis.ipynb) | Historical V0–V3 experiment across APTOS, IDRiD, Messidor-2, DDR, EyePACS and DeepDRiD, including per-grade evaluation, screening and plots. |
| Camera-ready | [Archived predictions](notebooks/camera_ready/01_archived_predictions.ipynb) | Recompute the published image-level paired bootstrap, confusion destinations and reference checks from V2/V3 NPZ files. |
| Camera-ready | [Checkpoint inference](notebooks/camera_ready/02_checkpoint_inference.ipynb) | Evaluate the original V2/V3 checkpoint pairs against the archived test manifests. |
| Camera-ready | [IDRiD localization](notebooks/camera_ready/03_idrid_localization.ipynb) | Evaluate real Grad-CAM and native MIL attention against aligned lesion masks with the fold-1 V3 checkpoint. |
| Camera-ready | [Patient-cluster analysis](notebooks/camera_ready/04_patient_cluster.ipynb) | Secondary paired bootstrap on saved predictions. Supply a verified mapping where the historical grouping is inadequate. |
| Camera-ready | [L4 fold audit](notebooks/camera_ready/05_l4_fold_audit.ipynb) | Development audit with persistent per-fold snapshots, manifest rebasing, discrepancy reports and optional Messidor-2 reconciliation. Requires CUDA and the image/checkpoint ZIPs. |
| Camera-ready | [Publication figures](notebooks/camera_ready/06_publication_figures.ipynb) | Export reference-summary charts, measured latent distributions and IDRiD localization panels. Select the figures for which you have inputs. |
| MLWA | [P0 historical replay](notebooks/mlwa/00_P0_historical_reproduction.ipynb) | Run frozen V2/V3 on source train, source validation and target. Save predictions, 512-dimensional features, shard identities and G0 comparisons. |
| MLWA | [E0 decision optimization](notebooks/mlwa/01_E0_decision_optimization_resume_v2.ipynb) | Estimate priors on source train, select decoders and screening thresholds on source validation, and evaluate held-out cohorts. |
| MLWA | [E1 representation probes](notebooks/mlwa/02_E1_representation_probes_resume_v2.ipynb) | Compare frozen V2 features, V3 features and the V3 severity axis with source-trained probes and permuted-label controls. |

The pooled studies and the six-fold LODO benchmark have different splits and weights. The official `models/` and `training/` packages include the modular DMIP scaffold. The archived checkpoint commands use `evaluation/checkpoint_reference/`. P0 retains the historical LODO forward implementation; the camera-ready replay has its own decoder and reference checks.

## Run from Python

From the root of this repository, use Python 3.10 or later with compatible PyTorch and torchvision builds.

```powershell
python -m pip install -e ".[dev]"
. .\configs\local.example.ps1
python workflows/mlwa/01_E0_decision_optimization_resume_v2.py
python workflows/mlwa/02_E1_representation_probes_resume_v2.py
```

The [local configuration](configs/local.example.ps1) points to the existing local data, reference summary, archived predictions, checkpoints and P0 folder on the D: drive. It sets process-level environment variables. Verify the paths if you move a source checkout. Training, P0 replay and E0/E1 analysis are separate entry points. The training workflows may download pretrained backbone weights on first use.

The official camera-ready CLI also accepts explicit paths. Legacy NPZs without image IDs require an explicit manifest-order acknowledgement backed by the historical unshuffled test loader.

```powershell
python -m experiments.camera_ready --raw-preds "$env:DR_RAW_PREDS" --folds-dir "$env:DR_FOLDS" --reference "$env:DR_REFERENCE" --output-dir outputs/camera_ready/archive --allow-manifest-order
```

See [the official reproduction protocol](docs/CAMERA_READY_REPRODUCTION.md) for the reference tolerances, individual commands and output schema.

## Run on Colab

Upload the required `.ipynb` alone, choose CPU for post-hoc analysis or GPU for image models, and edit the paths near the top. By default `THESIS_ROOT` is `/content/drive/MyDrive/dr_thesis`. Persistent results use `THESIS_ROOT/outputs`, and temporary SSD files use `/content/dr_thesis_work`. On a local system the defaults are the current working directory and its `work/` subdirectory. Missing Colab analysis dependencies are installed by the notebook. The code cells contain their model and analysis implementations directly.

| Input | Expected location under `inputs/` or override |
| --- | --- |
| Cohort images | `native_cropped.zip` or an extracted cohort directory via `DR_IMAGES`. The L4 audit takes a ZIP. |
| Six historical folds | `lodo_folds/` or the associated ZIP via `DR_FOLDS`. Expect `fold_0_aptos_train.csv` through `fold_5_deepdrid_test.csv`. |
| Original weights | `checkpoints.zip` or a directory via `DR_CHECKPOINTS`. The L4 audit takes a ZIP. |
| Benchmark reference | `reference/lodo_summary_with_calibrated_screening.json` or `DR_REFERENCE`. |
| Archived predictions | `raw_preds/` or `DR_RAW_PREDS`. One NPZ per variant and cohort. |
| Pooled inputs | The three historical CSVs under `hybrid/` or `DR_POOLED_METADATA`, plus the associated image tree. |
| IDRiD localization | `idrid_segmentation/` or `DR_IDRID`, with aligned image and mask directories. |
| Existing P0 | Set `P0_RUN` in E0/E1 or `DR_P0_RUN` in the environment. Each selected variant/fold/split must have a valid completion marker. |

For P0, set the optional pooled-head map and manifests only when running the independent five-head benchmark. By default P0 runs the two historical LODO variants. On a runtime reset it restores completed shards from persistent storage and prepares the local inputs again. The image ZIP digest or the digest of an extracted image tree binds results to the input bytes. CPU preprocessing workers preserve manifest order. `DR_PATCH_CHUNK` controls the number of patches encoded together; patch geometry and whole-bag pooling stay fixed.

## Maintenance and validation

Edit the corresponding file under `workflows/`, then run:

```powershell
python tools/build_notebooks.py
python tools/build_notebooks.py --check
python -m pytest -q
```

Regeneration updates cache fingerprints. Start a new output ID when changing executable code. [Source inventory](docs/source_inventory.csv) records source hashes and selection decisions, including historical drafts and teaching demonstrations. [Verification](docs/VERIFICATION.md) documents which checks used real artifacts and which used software fixtures. Historical outputs, raw images, weights and patient manifests are supplied separately.

