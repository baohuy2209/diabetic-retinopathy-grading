"""Referable Diabetic Retinopathy (RDR) screening calibration and evaluation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


@dataclass
class RDRCalibrationResult:
    """Stores calibrated threshold and source validation metrics."""

    target_specificity: float
    calibrated_threshold: float
    validation_sensitivity: float
    validation_specificity: float
    validation_auc: float
    total_val_samples: int


@dataclass
class RDREvaluationResult:
    """Stores evaluation metrics on a target/test set using a frozen threshold."""

    frozen_threshold: float
    test_sensitivity: float
    test_specificity: float
    test_auc: float
    total_test_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


def _screening_inputs(labels, probabilities, labels_are_binary):
    from evaluation.paired_bootstrap import grade_vector
    grades = grade_vector(np.asarray(labels), "screening labels")
    if labels_are_binary and np.any(grades > 1):
        raise ValueError("Binary labels must be 0 or 1")
    y = grades if labels_are_binary else (grades >= 2).astype(int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Screening probabilities must be finite, aligned, and in [0, 1]")
    return y, p


def calibrate_rdr_threshold(
    y_true_val: Union[List[int], np.ndarray],
    y_probs_val: Union[List[float], np.ndarray],
    target_specificity: float = 0.95,
    *, labels_are_binary: bool = False,
) -> RDRCalibrationResult:
    """Calibrates screening operating threshold tau* strictly on source validation data.

    Referable DR is defined as Grade >= 2.
    Threshold is chosen such that validation specificity >= target_specificity (default: 0.95).

    Args:
        y_true_val: Validation grades 0..4. For binary indicators explicitly set labels_are_binary=True.
        y_probs_val: Predicted referable probability (P(Y >= 2)) on validation set.
        target_specificity: Desired clinical specificity operating point (default: 0.95).

    Returns:
        RDRCalibrationResult holding frozen threshold and validation metrics.
    """
    if not 0 <= target_specificity <= 1:
        raise ValueError("target_specificity must lie in [0, 1]")
    y_true_bin, probs = _screening_inputs(y_true_val, y_probs_val, labels_are_binary)
    n_samples = len(y_true_bin)
    if len(np.unique(y_true_bin)) < 2:
        raise ValueError("Source validation requires both referable and non-referable cases")

    fpr, tpr, thresholds = roc_curve(y_true_bin, probs)
    val_auc = float(auc(fpr, tpr))
    specificity = 1.0 - fpr

    # Find operating points where specificity >= target_specificity
    valid_indices = np.where(specificity >= target_specificity)[0]

    if len(valid_indices) == 0:
        # Fallback to point of highest specificity
        best_idx = int(np.argmax(specificity))
    else:
        # Among points meeting target specificity, maximize sensitivity (TPR)
        best_idx = int(valid_indices[np.argmax(tpr[valid_indices])])

    threshold_star = float(thresholds[best_idx])
    # An all-negative operating point must remain above every observed score.
    # Clipping this threshold to 1 changes decisions for probabilities equal to 1.
    if not np.isfinite(threshold_star):
        threshold_star = float(np.nextafter(probs.max(), np.inf))

    # Verify actual sensitivity and specificity achieved at threshold_star
    preds_at_thresh = (probs >= threshold_star).astype(int)
    cm = confusion_matrix(y_true_bin, preds_at_thresh, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    actual_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    actual_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return RDRCalibrationResult(
        target_specificity=target_specificity,
        calibrated_threshold=threshold_star,
        validation_sensitivity=actual_sens,
        validation_specificity=actual_spec,
        validation_auc=val_auc,
        total_val_samples=n_samples,
    )


def evaluate_rdr_with_frozen_threshold(
    y_true_test: Union[List[int], np.ndarray],
    y_probs_test: Union[List[float], np.ndarray],
    frozen_threshold: float,
    *, labels_are_binary: bool = False,
) -> RDREvaluationResult:
    """Evaluates test data using a previously frozen operating threshold.

    Guarantees zero-leakage: the test data has no influence on the decision threshold.
    """
    if np.isnan(frozen_threshold):
        raise ValueError("Frozen threshold cannot be NaN")
    y_true_bin, probs = _screening_inputs(y_true_test, y_probs_test, labels_are_binary)
    n_samples = len(y_true_bin)

    preds = (probs >= frozen_threshold).astype(int)
    cm = confusion_matrix(y_true_bin, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    test_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true_bin)) >= 2:
        fpr, tpr, _ = roc_curve(y_true_bin, probs)
        test_auc = float(auc(fpr, tpr))
    else:
        test_auc = float("nan")

    return RDREvaluationResult(
        frozen_threshold=frozen_threshold,
        test_sensitivity=test_sens,
        test_specificity=test_spec,
        test_auc=test_auc,
        total_test_samples=n_samples,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )


def calibrate_and_evaluate_screening(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
    test_targets: np.ndarray,
    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
    *, labels_are_binary: bool = False,
) -> Any:
    """Calibrates threshold tau on in-domain validation, freezes tau and evaluates OOD test."""
    import pandas as pd

    def referable_probabilities(values):
        values = np.asarray(values)
        if values.ndim == 2 and values.shape[1] == 5:
            if (values < 0).any() or not np.allclose(values.sum(1), 1, atol=1e-5):
                raise ValueError("Class probabilities must be normalized and nonnegative")
            return values[:, 2:].sum(1)
        if values.ndim == 2 and values.shape[1] == 2:
            if not labels_are_binary:
                raise ValueError("Two-class probabilities require labels_are_binary=True")
            return values[:, 1]
        return values

    val_score = referable_probabilities(val_probs)
    test_score = referable_probabilities(test_probs)
    records = []
    for target in target_specificities:
        calibration = calibrate_rdr_threshold(val_targets, val_score, target,
                                              labels_are_binary=labels_are_binary)
        evaluation = evaluate_rdr_with_frozen_threshold(test_targets, test_score,
            calibration.calibrated_threshold, labels_are_binary=labels_are_binary)
        records.append(dict(Target_Specificity=f"{int(target * 100)}%",
            Calibrated_Threshold_tau=calibration.calibrated_threshold,
            Val_Sensitivity=calibration.validation_sensitivity,
            Val_Specificity=calibration.validation_specificity,
            Test_Sensitivity=evaluation.test_sensitivity,
            Test_Specificity=evaluation.test_specificity,
            Test_AUC=evaluation.test_auc))
    return pd.DataFrame(records)
