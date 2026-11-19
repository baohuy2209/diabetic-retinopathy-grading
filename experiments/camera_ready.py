"""Run paired bootstrap/confusion from real saved predictions; never trains models."""

import argparse
import json
from pathlib import Path

from evaluation.camera_ready import COHORTS, analyze_frames, load_archive_pair, load_csv_pair


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-preds", type=Path, help="Original posthoc_fixed/raw_preds NPZ directory")
    source.add_argument("--predictions-dir", type=Path, help="Directory with fold_N_cohort_predictions.csv")
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="lodo_summary_with_calibrated_screening.json")
    parser.add_argument("--output-dir", type=Path, default=Path("result/camera_ready"))
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--allow-manifest-order", action="store_true", help="Explicitly acknowledge legacy NPZs omit IDs; requires documented unshuffled order")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.cohorts)) != len(args.cohorts):
        parser.error("Duplicate cohorts are not allowed")
    frames, inputs = {}, []
    for cohort in args.cohorts:
        if args.raw_preds:
            frame, provenance = load_archive_pair(args.raw_preds, args.folds_dir, cohort, args.allow_manifest_order)
        else:
            stem = f"fold_{COHORTS.index(cohort)}_{cohort}"
            frame, provenance = load_csv_pair(args.predictions_dir / (stem + "_predictions.csv"),
                                              args.folds_dir / (stem + "_test.csv"))
        frames[cohort] = frame
        inputs.extend(provenance)
    report = analyze_frames(frames, args.reference, args.output_dir, inputs, reps=args.bootstrap_reps,
                            seed=args.seed, source_mode="archived_predictions" if args.raw_preds else "prediction_csv",
                            exact_reference=bool(args.raw_preds), diagnostic_only=args.diagnostic_only)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "BENCHMARK_REPRODUCED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
