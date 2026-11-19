# Integration and verification

## Source selection

The [inventory](source_inventory.csv) indexes 316 source files across the official repository, the three development directories and the two supplied E0/E1 notebooks. It records file hashes and code-text hashes. Four development notebooks were empty or invalid JSON. They are identified in the inventory. The official five DMIP notebooks supply the historical preparation, pooled training, head ablation and LODO workflows. Preliminary per-dataset trials, classroom examples, backups and synthetic demonstrations are outside the official study.

The root Python packages preserve the official DMIP module layout. The checkpoint inference CLI uses its checkpoint-compatible reference implementation. The supplementary L4 audit retains the development analysis and its per-fold snapshots. The patient-cluster workflow comes from the separate CPU section of the L4 notebook. The figure workflow requires measured latent scores and learned cutpoints. The reviewed E0/E1 notebook engines take precedence over the older MLWA Python launchers.

The standalone notebook sources use explicit asset paths. Publication and training outputs are directed to the thesis workspace. Model weights, subject-level data and previous authors' result directories are external inputs. Uploaded notebooks can execute independently of this repository. The scripts and notebooks contain their respective end-to-end experiment logic.

## Software checks

The 43 regression checks cover official evaluation and checkpoint behavior, source-only selection, E0/E1 resume after interruptions, cache identity and corruption, paired bootstrap batches, P0 ordered preprocessing, ZIP staging and notebook/script synchronization. Software fixtures are confined to tests. Notebook validation checks the Jupyter schema, compiles each code cell, verifies cache fingerprints and checks that saved outputs are empty.

## Checks with original study artifacts

- The configuration cells of the three historical DMIP training notebooks resolved the real local assets. The head-ablation loader read 9,079 training, 8,238 validation and 8,075 test rows. The check stopped before training.
- The official Python CLI and its standalone archived-prediction workflow both reached `BENCHMARK_REPRODUCED` across the six held-out cohorts with 2,000 image-level bootstrap replicates and seed 42. Both used the 12 archived V2/V3 NPZ files, the six historical test CSVs and the same reference JSON. There were 54,883 test images and 1,858 Grade-3 cases. Legacy NPZs lack image IDs. These runs explicitly used the historical unshuffled test-manifest order. The analysis checks ordered labels and the expected decoding for each variant. This assumption limits independent attribution to individual images.
- The P0 notebook definitions loaded both original fold-0 checkpoints on CPU. One real APTOS image passed the probability and 512-dimensional feature-shape checks for each variant.
- The E1 staging code validated all 36 completed splits of the local `p0_fp32_main_lodo_v1` run, including core hashes, row alignment and feature arrays. The run's own G0 file reports `DIAGNOSTIC` even though its execution status is `INFERENCE_COMPLETE`. Keep that status attached to analyses of this particular P0 run.
- The publication workflow exported the slope, heatmap and waterfall figures to PDF and PNG from the supplied reference summary.

The validation used the local machine. Full retraining, a new six-fold P0 inference run, full E0/E1 estimation on the study data and a live Colab session are outside this validation. The local integration checks and synthetic test metrics do not provide new thesis effect estimates.

## Interpretation of the extension

E0 computes source-training priors and selects the operating rule on source validation. Its cost-sensitive decision is based on conditional expected costs. Its logit adjustment uses a source prior. Calibration and ordinal scoring use the original posterior. [Elkan](https://cseweb.ucsd.edu/~elkan/rescale.pdf) · [Menon et al.](https://arxiv.org/abs/2007.07314) · [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) · [Vargas et al.](https://arxiv.org/abs/1905.13392)

E1 fits a source-training scaler and regularized linear probes on frozen representations. Each fold selects regularization on its own source validation set. Within-cohort permuted training labels provide a negative control. Differences between probe results describe decodability under this protocol. [Hewitt and Liang](https://aclanthology.org/D19-1275/) · [Gulrajani and Lopez-Paz](https://arxiv.org/abs/2007.01434) · [Kang et al.](https://arxiv.org/abs/1910.09217) · [Kirichenko et al.](https://arxiv.org/abs/2204.02937)

