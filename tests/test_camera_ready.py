"""Protocol regression tests with tiny artificial arrays; no empirical evaluation."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.camera_ready import COHORTS, VARIANTS, analyze_frames, load_archive_pair
from evaluation.checkpoint_reference.runtime import normalize_inputs, validate_cache
from evaluation.paired_bootstrap import (
    grade3_confusion, macro_bootstrap_grade3_sensitivity,
    paired_bootstrap_grade3_sensitivity, validate_pairs,
)


def test_paired_resampling_preserves_identical_predictions():
    y = np.array([0, 3, 3, 3, 4])
    p = np.array([0, 3, 2, 4, 4])
    r = paired_bootstrap_grade3_sensitivity(y, p, p, reps=100)
    assert r["diff_v3_minus_v2_pp"] == r["diff_ci_low_pp"] == r["diff_ci_high_pp"] == 0
    assert r["v2_ci_low"] == r["v3_ci_low"]
    assert r == paired_bootstrap_grade3_sensitivity(y, p, p, reps=100)


def test_macro_weights_cohorts_equally_instead_of_pooling_images():
    small = dict(cohort="a", y_true=np.full(2, 3), pred_v2=np.full(2, 3), pred_v3=np.full(2, 2))
    large = dict(cohort="b", y_true=np.full(20, 3), pred_v2=np.full(20, 2), pred_v3=np.full(20, 3))
    r = macro_bootstrap_grade3_sensitivity([small, large], reps=20)
    assert r["sens_v2"] == r["sens_v3"] == 50
    assert r["n_grade3"] == 22
    assert r["diff_ci_low_pp"] == r["diff_ci_high_pp"] == 0


@pytest.mark.parametrize("bad", [[3.1], [5], [-1], [np.nan], [], [[3]], ["3"]])
def test_reject_invalid_grades(bad):
    with pytest.raises(ValueError):
        validate_pairs(np.asarray(bad), np.asarray(bad), np.asarray(bad))


def test_no_grade3_cannot_be_reported_as_zero_sensitivity():
    with pytest.raises(ValueError, match="no Grade-3"):
        paired_bootstrap_grade3_sensitivity(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))


def test_confusion_preserves_all_five_destinations():
    r = grade3_confusion(np.array([3] * 5 + [0]), np.array([0, 1, 2, 3, 4, 4]))
    assert [r[f"g3_to_{i}"] for i in range(5)] == [1] * 5
    assert r["n_actual_g3"] == 5 and r["g3_to_01"] == r["g3_to_24"] == 2
    assert r["pct_errors_assigned_to_grade24"] == 50


def _archive_fixture(tmp_path):
    frame = pd.DataFrame(dict(image_id=["a", "b", "c"], diagnosis=[0, 3, 4]))
    frame.to_csv(tmp_path / "fold_0_aptos_test.csv", index=False)
    for name in VARIANTS.values():
        np.savez(tmp_path / (name + "__aptos.npz"), ood_targets=[0, 3, 4],
                 ood_preds=[0, 3, 4], ood_probs=np.eye(5)[[0, 3, 4]])


def test_legacy_pairing_requires_explicit_order_assumption(tmp_path):
    _archive_fixture(tmp_path)
    with pytest.raises(ValueError, match="lacks image IDs"):
        load_archive_pair(tmp_path, tmp_path, "aptos")
    frame, inputs = load_archive_pair(tmp_path, tmp_path, "aptos", True)
    assert len(frame) == 3 and inputs[1]["identity"] == "documented_manifest_order"
    # Matching counts alone cannot detect a reversed target ordering.
    path = tmp_path / (VARIANTS["v2"] + "__aptos.npz")
    np.savez(path, ood_targets=[4, 3, 0], ood_preds=[4, 3, 0], ood_probs=np.eye(5)[[4, 3, 0]])
    with pytest.raises(ValueError, match="Ordered targets"):
        load_archive_pair(tmp_path, tmp_path, "aptos", True)


def test_decoder_and_image_identity_checked_independently(tmp_path):
    _archive_fixture(tmp_path)
    path = tmp_path / (VARIANTS["v3"] + "__aptos.npz")
    np.savez(path, ood_targets=[0, 3, 4], ood_preds=[0, 3, 4],
             ood_probs=np.eye(5)[[0, 3, 4]], ood_image_ids=["c", "b", "a"])
    with pytest.raises(ValueError, match="image ID mismatch"):
        load_archive_pair(tmp_path, tmp_path, "aptos", True)
    np.savez(path, ood_targets=[0, 3, 4], ood_preds=[0, 2, 4], ood_probs=np.eye(5)[[0, 3, 4]])
    with pytest.raises(ValueError, match="decoder mismatch"):
        load_archive_pair(tmp_path, tmp_path, "aptos", True)


def test_complete_reference_gate_and_diagnostic_override(tmp_path):
    frame = pd.DataFrame(dict(image_id=["a", "b", "c"], diagnosis=[0, 3, 4], pred_v2=[0, 3, 4], pred_v3=[0, 3, 4]))
    frames = {cohort: frame.copy() for cohort in COHORTS}
    ref = {name: {c: dict(ood_test_qwk=1, ood_within_1_grade_acc=1,
            ood_per_grade_recall={"grade_3": 1},
            ood_per_grade_support={f"grade_{i}": int(i in (0, 3, 4)) for i in range(5)})
            for c in COHORTS} for name in VARIANTS.values()}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    report = analyze_frames(frames, path, tmp_path / "out", [], reps=10)
    assert report["status"] == "BENCHMARK_REPRODUCED" and report["total_test_images"] == 18
    partial = analyze_frames({"aptos": frame}, path, tmp_path / "out", [], reps=10)
    assert partial["status"] == "DIAGNOSTIC" and partial["run_directory"] != report["run_directory"]
    forced = analyze_frames(frames, path, tmp_path / "out", [], reps=10, diagnostic_only=True)
    assert forced["status"] == "DIAGNOSTIC"
    ref[VARIANTS["v3"]]["aptos"]["ood_test_qwk"] = .8
    path.write_text(json.dumps(ref), encoding="utf-8")
    mismatch = analyze_frames(frames, path, tmp_path / "out", [], reps=10)
    assert mismatch["status"] == "DIAGNOSTIC"


def test_cache_checks_every_row_not_just_first_row():
    manifest = pd.DataFrame(dict(image_id=["a", "b"], diagnosis=[3, 3], native_path=["a.jpg", "b.jpg"]))
    metadata = dict(checkpoint_v2_sha="abc", provenance_marker="real")
    frame = manifest.assign(pred_v2=[3, 2], pred_v3=[3, 2], v3_expected_grade=[3., 2.], **metadata)
    validate_cache(frame, manifest, metadata)
    frame.loc[1, "checkpoint_v2_sha"] = "wrong"
    with pytest.raises(ValueError, match="row provenance"):
        validate_cache(frame, manifest, metadata)
    frame.loc[1, "checkpoint_v2_sha"] = "abc"
    frame.loc[1, "v3_expected_grade"] = 3.1
    with pytest.raises(ValueError, match="decoder mismatch"):
        validate_cache(frame, manifest, metadata)


def test_normalization_occurs_exactly_once_and_keeps_masked_bags():
    g = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)
    l = torch.zeros((1, 2, 3, 2, 2), dtype=torch.uint8)
    gn, ln = normalize_inputs(g, l, "cpu")
    assert gn[0, 0, 0, 0].item() == pytest.approx((1 - .485) / .229)
    assert ln[0, 0, 0, 0, 0].item() == pytest.approx(-.485 / .229)
    with pytest.raises(ValueError, match="raw uint8"):
        normalize_inputs(gn, ln, "cpu")


def test_modular_evaluation_passes_local_branch_and_padding_mask():
    from evaluation.lodo import run_inference_on_loader

    class Spy(torch.nn.Module):
        def forward(self, global_img, patch_imgs, patch_mask):
            assert global_img.dtype == patch_imgs.dtype == torch.float32
            assert patch_imgs.shape == (1, 2, 3, 2, 2)
            assert patch_mask.tolist() == [[True, False]]
            return dict(probs=torch.tensor([[0., 0., 0., 1., 0.]]), preds=torch.tensor([3]))

    batch = dict(label=torch.tensor([3]), global_img=torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
        local_patches=torch.zeros(1, 2, 3, 2, 2, dtype=torch.uint8), patch_mask=torch.tensor([[True, False]]))
    y, pred, _ = run_inference_on_loader(Spy(), [batch], torch.device("cpu"))
    assert y.tolist() == pred.tolist() == [3]


def test_screening_preserves_all_negative_operating_point_and_frozen_threshold():
    from evaluation.rdr import calibrate_rdr_threshold, evaluate_rdr_with_frozen_threshold
    c = calibrate_rdr_threshold([0, 2], [1., 1.], target_specificity=.95)
    assert c.calibrated_threshold > 1 and c.validation_specificity == 1
    r = evaluate_rdr_with_frozen_threshold([0, 3], [1., 1.], c.calibrated_threshold)
    assert r.frozen_threshold == c.calibrated_threshold and r.false_positives == r.true_positives == 0
    with pytest.raises(ValueError, match="both referable"):
        calibrate_rdr_threshold([0, 1], [.2, .9])


def test_xai_exact_k_and_undefined_map_policy():
    from explainability.metrics import _xai_measure, pointing_game
    heat = np.array([[1., 1.], [1., 0.]])
    mask = np.array([[True, False], [False, False]])
    r = _xai_measure(heat, mask)
    assert r["selected_pixels"] == 1
    assert r == _xai_measure(heat, mask)
    assert _xai_measure(np.ones((2, 2)), mask)["status"] == "constant_map"
    assert _xai_measure(np.full((2, 2), np.nan), mask)["hit"] is None
    assert pointing_game(np.array([[0., 1.], [0., 0.]]), mask) == 0  # No dilation by default.


def test_signed_rank_effect_direction():
    from evaluation.statistics import paired_wilcoxon_test
    r = paired_wilcoxon_test([1., 2., 3., 4., 5., 6.], [0.] * 6)
    assert r["rank_biserial_effect_size"] == 1
    assert r["p_value"] == pytest.approx(.03125)


def test_reference_code_integrity():
    from pathlib import Path
    import hashlib
    root = Path(__file__).resolve().parents[1] / "evaluation/checkpoint_reference"
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    for entry in provenance["files"]:
        normalized = (root / entry["file"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(normalized).hexdigest() == entry["ported_sha256_lf"]


def test_mil_reads_learned_attention_without_changing_weights_or_leaving_hooks():
    from types import SimpleNamespace
    from explainability.mil_saliency import generate_attention_mil_saliency_map

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.Module()
            self.pool.attention_weights = torch.nn.Linear(1, 1, bias=False)
            self.pool.attention_weights.weight.data.fill_(1)
            self.backbone = SimpleNamespace(local_mil_branch=SimpleNamespace(gated_pool=self.pool))

        def forward(self, global_img, patches, patch_mask=None):
            return self.pool.attention_weights(torch.tensor([[[0.], [2.]]]))

    model = Toy()
    before = model.pool.attention_weights.weight.detach().clone()
    saliency = generate_attention_mil_saliency_map(model, None, None, [[0, 0], [0, 2]],
        canvas_size=(2, 4), native_shape=(2, 4), patch_size=2)
    assert np.isfinite(saliency).all() and saliency[:, 2:].mean() > saliency[:, :2].mean()
    assert torch.equal(before, model.pool.attention_weights.weight)
    assert not model.pool.attention_weights._forward_hooks


def test_inventory_is_read_only_and_excludes_weights(tmp_path):
    from experiments.audit_bundle import inspect_bundle
    path = tmp_path / "execution.log"
    path.write_text("artificial fixture", encoding="utf-8")
    report = inspect_bundle(tmp_path)
    assert report["status"] == "INVENTORIED_ONLY" and report["n_files"] == 1
    assert path.read_text(encoding="utf-8") == "artificial fixture"
    assert len(list(tmp_path.iterdir())) == 1
    (tmp_path / "weights.pth").write_bytes(b"test")
    with pytest.raises(ValueError, match="Keep checkpoints outside"):
        inspect_bundle(tmp_path)
