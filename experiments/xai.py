"""Evaluate real IDRiD image/mask pairs using the held-out V3 checkpoint.

No generated patches, random weights, or synthetic masks in real evaluation.
The data owner must independently verify supplied binary-mask annotation categories.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from evaluation.camera_ready import new_run, sha256
from evaluation.checkpoint_reference.runtime import build_checkpoint_model
from explainability.idrid_dataset import IDRiDXAIEvalDataset
from explainability.metrics import compute_quantitative_xai


class QuantitativeXAIRunner:
    """Frozen-checkpoint localization evaluation; never trains or selects models."""

    def __init__(self, checkpoint_path=None, variant="v3", dataset_dir=None,
                 results_dir=Path("result/xai"), device="cpu", max_samples=0,
                 dry_run=False, image_dirs: Optional[Sequence[Path]] = None,
                 mask_dirs: Optional[Sequence[Path]] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.variant, self.results_dir = variant, Path(results_dir)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.max_samples, self.dry_run = max_samples, dry_run
        self.image_dirs = list(image_dirs or ([Path(dataset_dir) / "images"] if dataset_dir else []))
        self.mask_dirs = list(mask_dirs or ([Path(dataset_dir) / "masks"] if dataset_dir else []))

    def load_model(self):
        if self.variant != "v3":
            raise ValueError("The camera-ready XAI protocol evaluates the IDRiD-held-out V3 model")
        if not self.checkpoint_path or not self.checkpoint_path.is_file():
            raise FileNotFoundError("A real IDRiD-held-out V3 checkpoint is required")
        required = "lodo_V3_PlusOrdinalHead_H3_Champion_fold_1_idrid_best.pth"
        if self.checkpoint_path.name != required:
            raise ValueError(f"Expected the published held-out checkpoint: {required}")
        return build_checkpoint_model("v3", self.checkpoint_path, self.device)

    def run(self):
        if self.dry_run:
            return dict(status="SMOKE_TEST_ONLY", note="No empirical localization metrics computed")
        if self.max_samples < 0 or not self.image_dirs or not self.mask_dirs:
            raise ValueError("Supply image_dirs and mask_dirs; max_samples must be nonnegative")
        for p in [*self.image_dirs, *self.mask_dirs]:
            if not Path(p).is_dir():
                raise FileNotFoundError(p)
        model = self.load_model()
        dataset = IDRiDXAIEvalDataset(image_dirs=self.image_dirs, mask_dirs=self.mask_dirs)
        if not len(dataset):
            raise ValueError("No real image/mask pairs found")
        if self.max_samples:
            dataset.samples = dataset.samples[:self.max_samples]
        summary = compute_quantitative_xai(model, dataset, device=self.device, return_samples=False)
        run = new_run(self.results_dir)
        pairs = [dict(image_id=i, image_path=str(Path(p).resolve()), image_sha256=sha256(Path(p)),
                      mask_path=str(Path(m).resolve()), mask_sha256=sha256(Path(m))) for i, p, m in dataset.samples]
        summary.update(checkpoint=str(self.checkpoint_path.resolve()), checkpoint_sha256=sha256(self.checkpoint_path),
                       status="DIAGNOSTIC_SUBSET" if self.max_samples else "EVALUATED_REAL_PAIRS",
                       cohort="IDRiD", pairs=pairs, mask_categories="Supplied binary masks; annotation categories require independent verification")
        (run / "metrics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--mask-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("result/xai"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args(argv)
    runner = QuantitativeXAIRunner(checkpoint_path=args.checkpoint_path, image_dirs=args.image_dirs,
        mask_dirs=args.mask_dirs, results_dir=args.results_dir, device=args.device, max_samples=args.max_samples)
    print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
