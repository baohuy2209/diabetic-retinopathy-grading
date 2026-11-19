# Camera-ready reproduction protocol

These commands are for an explicitly requested future run. They are not required to add existing logs; follow [the import guide](../result/README.md) for that task. No experiments run automatically.

## A. Archived predictions: no checkpoints or images needed

From the repository root, after supplying the files described in the import guide:

```powershell
python -m experiments.camera_ready --raw-preds result/raw_preds --folds-dir result/manifests/lodo_folds --reference result/reference/lodo_summary_with_calibrated_screening.json --allow-manifest-order --bootstrap-reps 2000 --seed 42 --output-dir result/recomputed
```

Omit `--allow-manifest-order` if the archives contain `ood_image_ids` or `image_ids`. For legacy archives without IDs, pass it only when documented unshuffled loader order justifies that assumption. Ordered target equality alone does not prove image identity. Files must follow `<variant>__<cohort>.npz`; variants are `V2_PlusGlobalContext_H2` and `V3_PlusOrdinalHead_H3_Champion`.

Canonical fold order: `0=aptos`, `1=idrid`, `2=messidor2`, `3=ddr`, `4=eyepacs`, `5=deepdrid`. NPZ keys are `ood_targets`, `ood_preds`, and `ood_probs` (N×5). Values, ordered manifest targets, optional IDs, probability sums, and variant-specific decoding are validated before analysis.

The camera-ready estimator uses 2,000 replicates, seed 42, paired Grade-3 image indices and float32 sensitivity means, as in the archived analysis. Both variant CIs and the CI of their paired difference are exported. Macro resampling occurs within each fixed cohort before equal-weight averaging; cohorts themselves are not resampled. Pooled images are reported separately. Cohorts without Grade-3 cases fail rather than silently contributing zero.

## B. Future checkpoint inference (external weights)

Use a native-cropped image root with cohort subdirectories. Place checkpoints outside the repository; the expected names are `lodo_<variant>_fold_<index>_<cohort>_best.pth`.

```powershell
python -m experiments.checkpoint_inference --checkpoint-dir E:/external/checkpoints --data-root E:/fundus/native_cropped --folds-dir result/manifests/lodo_folds --reference result/reference/lodo_summary_with_calibrated_screening.json --device cuda --batch-size 1 --patch-chunk-size 48 --output-dir result/new_inference
```

This is actual inference. Do not execute it when only importing logs. The checkpoint-compatible implementation is independent of the development repository; source module hashes are in `evaluation/checkpoint_reference/SOURCE_PROVENANCE.json`. It uses strict state loading, original preprocessing/collation, uint8-to-ImageNet normalization exactly once, evaluation patch masks, V2 argmax, and V3 rounded expected grade. Missing or unreadable images fail.

`--max-samples N` and partial `--cohorts` are diagnostic. Optional `--reuse-from <prior-run>/predictions` checks every cached row's ordered identity, labels, paths, checkpoint/manifest/code hashes and V3 expected-grade decoding. Cache validation does not certify unchanged image bytes; retain image dataset provenance independently. A valid cache does not automatically establish agreement with the benchmark. Small probability traces and runtime/code hashes are retained, with new output directories for every invocation.

## C. Real localization evaluation

```powershell
python -m experiments.xai --checkpoint-path E:/external/checkpoints/lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth --image-dirs E:/IDRiD/images --mask-dirs E:/IDRiD/binary_masks --device cuda --results-dir result/new_xai
```

This is actual inference/gradient evaluation. Use original aligned image/mask pairs and independently document which lesion categories the supplied binary masks contain. Shapes must agree before joint cropping/resizing. The command uses the held-out IDRiD V3 model, real patches and native learned MIL attention. Grad-CAM targets the CLM severity score. Pointing Game uses no dilation; area matching selects exactly K positive-mask pixels with seeded tie handling. Nonfinite, constant or unavailable maps are reported as undefined and excluded with valid counts. A subset is marked diagnostic; the number of real pairs is counted, not forced to 81.

## D. Reference and status policy

For archived predictions, reference comparisons use numerical equality (1e-12 tolerance). For new inference, the inherited diagnostic tolerances are QWK 0.005, Within-1 1.0 percentage point, Grade-3 sensitivity 1.5 pp for V2 and 2.5 pp for V3. These tolerances are comparison gates, not significance tests or proof of identical predictions. The full difference table is always retained; do not widen tolerances to conceal discrepancies.

All six cohorts and 36 passing comparisons are required for `BENCHMARK_REPRODUCED`. Otherwise the status is `DIAGNOSTIC`; explicit diagnostic overrides cannot be promoted. CLI exit status is 0 for reproduction, 2 for a completed diagnostic analysis, and nonzero for invalid input/runtime failure. Failed runs retain a failure record. No status claims seed-level robustness or independently verified historical identity.

Exports include per-fold metrics, reference checks, bootstrap CI tables, per-cohort and pooled five-way Grade-3 confusion, counts, input hashes and provenance. Percentages and percentage points are explicit in the column names/documentation; QWK remains a fraction. Screening uses P(Y≥2), calibrates on source validation only, and freezes the threshold for the target. Grade destinations alone cannot determine risk-threshold screening decisions.

The modular training API and historical notebooks are retained separately. They are not a guarantee of bitwise retraining equivalence to archived weights. No training was needed to port the camera-ready evaluation code.
