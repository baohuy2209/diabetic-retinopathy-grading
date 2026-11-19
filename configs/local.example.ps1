# From the thesis repository root, run ". .\configs\local.example.ps1".
# These locations match the two existing local checkouts. All new outputs use the thesis repository.
$devRepo = 'D:\Github\deep-learning-for-computer-vision'
$paperRepo = 'D:\Github\DR-evaluation-study2026'
$pipeline = Join-Path $devRepo 'notebooks/dmip-conference/diabetic_retinopathy/pipeline'
$env:DR_ROOT = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:DR_RAW = Join-Path $devRepo 'data/raw'
$env:DR_HYBRID = Join-Path $env:DR_ROOT 'outputs/prepared/hybrid'
$env:DR_POOLED_METADATA = Join-Path $devRepo 'data/raw/hybrid'
$env:DR_IMAGES = Join-Path $env:DR_POOLED_METADATA 'native_cropped'
$env:DR_FOLDS = Join-Path $devRepo 'data/processed/lodo_folds'
$env:DR_CHECKPOINTS = Join-Path $pipeline 'checkpoints'
$env:DR_IDRID = Join-Path $env:DR_RAW 'idrid_segmentation'
$env:DR_REFERENCE = Join-Path $paperRepo 'result/reference/lodo_summary_with_calibrated_screening.json'
$env:DR_RAW_PREDS = Join-Path $paperRepo 'result/raw_preds'
$env:DR_LATENT_SCORES = Join-Path $pipeline 'results/latent_scores_champion_fold_1.npz'
$env:DR_FIGURE_CHECKPOINT = Join-Path $env:DR_CHECKPOINTS 'lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth'
$env:DR_P0_RUN = Join-Path $devRepo 'notebooks/mlwa-journals/outputs/p0_fp32_main_lodo_v1'

