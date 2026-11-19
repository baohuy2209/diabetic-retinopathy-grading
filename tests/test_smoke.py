"""Smoke tests for CI and repository health verification."""

import numpy as np
import pytest
import torch


def test_package_imports() -> None:
    """Verifies that all 11 core repository packages import cleanly."""
    import config
    import data
    import evaluation
    import experiments
    import explainability
    import losses
    import models
    import paths
    import training
    import utils
    import visualization

    assert paths.REPO_ROOT.exists()
    assert config.IMG_SIZE == 512
    assert len(config.ARCHITECTURE_VARIANTS) == 4


def test_build_all_variants() -> None:
    """Verifies that all 4 architecture ablation variants can be built."""
    from models.factory import build_variant

    for v_code in ["v0", "v1", "v2", "v3"]:
        model = build_variant(v_code, pretrained=False)
        assert model is not None
        assert hasattr(model, "forward")


def test_clm_head_numerical_stability() -> None:
    """Verifies that the CLM ordinal head produces valid probabilities without NaNs."""
    from models.heads.clm import CumulativeLinkModelQWKHead

    head = CumulativeLinkModelQWKHead(in_features=512, num_classes=5)
    dummy_feat = torch.randn(4, 512, dtype=torch.float32)

    outputs = head(dummy_feat)
    probs = outputs["probs"]
    preds = outputs["preds"]

    assert not torch.isnan(probs).any(), "CLM output contains NaN"
    assert not torch.isinf(probs).any(), "CLM output contains Inf"
    assert (probs >= 0.0).all(), "CLM probabilities must be >= 0"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)
    assert preds.shape == (4,)


def test_rdr_calibration() -> None:
    """Verifies bipartite threshold calibration and evaluation."""
    from evaluation.rdr import calibrate_and_evaluate_screening

    val_probs = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.1, 0.1, 0.5, 0.2, 0.1],
        [0.05, 0.05, 0.1, 0.2, 0.6],
        [0.7, 0.15, 0.05, 0.05, 0.05],
    ])
    val_targets = np.array([0, 2, 4, 0])

    test_probs = np.array([
        [0.85, 0.05, 0.05, 0.03, 0.02],
        [0.05, 0.05, 0.6, 0.2, 0.1],
    ])
    test_targets = np.array([0, 3])

    df = calibrate_and_evaluate_screening(val_probs, val_targets, test_probs, test_targets)
    assert not df.empty
    assert "Calibrated_Threshold_tau" in df.columns
