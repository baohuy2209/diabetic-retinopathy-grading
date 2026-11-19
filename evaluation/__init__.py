"""Evaluation metrics, grade-specific statistics, and LODO bipartite orchestration."""

from evaluation.grade_metrics import calculate_per_grade_metrics
from evaluation.lodo import evaluate_lodo_fold, evaluate_partition, run_inference_on_loader
from evaluation.metrics import (
    calculate_dr_metrics,
    calculate_referable_sensitivity_at_specificity,
    extract_rank_probs,
    print_per_grade_report,
    rank_monotonicity_violations,
)
from evaluation.ordinal_metrics import calculate_ordinal_metrics
from evaluation.rdr import (
    calibrate_and_evaluate_screening,
    calibrate_rdr_threshold,
    evaluate_rdr_with_frozen_threshold,
)
from evaluation.result_schema import LODOFoldResult, MetricBundle, validate_lodo_result_dict
from evaluation.statistics import compute_metric_summary, paired_wilcoxon_test

__all__ = [
    "calculate_dr_metrics",
    "calculate_referable_sensitivity_at_specificity",
    "extract_rank_probs",
    "rank_monotonicity_violations",
    "print_per_grade_report",
    "calculate_per_grade_metrics",
    "calculate_ordinal_metrics",
    "paired_wilcoxon_test",
    "compute_metric_summary",
    "MetricBundle",
    "LODOFoldResult",
    "validate_lodo_result_dict",
    "evaluate_lodo_fold",
    "evaluate_partition",
    "run_inference_on_loader",
    "calibrate_and_evaluate_screening",
    "calibrate_rdr_threshold",
    "evaluate_rdr_with_frozen_threshold",
]
