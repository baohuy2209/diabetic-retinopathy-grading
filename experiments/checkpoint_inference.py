"""True V2/V3 checkpoint inference with strict cache validation and reference gating.

This command never trains. Its output remains DIAGNOSTIC if any reference check fails.
"""

import argparse
from functools import partial
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation.camera_ready import COHORTS, VARIANTS, analyze_frames, new_run, read_manifest, sha256
from evaluation.checkpoint_reference.data_loader import FundusDualBranchDataset, trainer_compatible_collate_fn
from evaluation.checkpoint_reference.runtime import (
    build_checkpoint_model, code_fingerprint, normalize_inputs, resolve_manifest_images, validate_cache,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("result/checkpoint_inference"))
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-chunk-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Diagnostic subset only; 0 = full cohort")
    parser.add_argument("--reuse-from", type=Path, help="Explicit prior run/predictions directory; cache hashes must match")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.patch_chunk_size < 0 or args.max_samples < 0 or len(set(args.cohorts)) != len(args.cohorts):
        parser.error("Invalid batch/workers/chunk/subset/cohort arguments")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    runtime = dict(seed=args.seed, precision="float32", batch_size=args.batch_size,
                   patch_chunk_size=args.patch_chunk_size, torch=torch.__version__,
                   device=args.device, python=platform.python_version(), reference_sha256=sha256(args.reference))
    fingerprint = code_fingerprint(runtime)
    run = new_run(args.output_dir)
    predictions = run / "predictions"
    predictions.mkdir()
    frames, inputs, trace, events = {}, [], [], []
    try:
        for cohort in args.cohorts:
            fold = COHORTS.index(cohort)
            stem = f"fold_{fold}_{cohort}"
            source = args.folds_dir / (stem + "_test.csv")
            manifest = read_manifest(source)
            if args.max_samples:
                manifest = manifest.iloc[:args.max_samples].copy()
            manifest = resolve_manifest_images(manifest, args.data_root)
            checkpoints = {v: args.checkpoint_dir / f"lodo_{name}_fold_{fold}_{cohort}_best.pth" for v, name in VARIANTS.items()}
            hashes = {v: sha256(path) for v, path in checkpoints.items()}
            meta = dict(checkpoint_v2_sha=hashes["v2"], checkpoint_v3_sha=hashes["v3"], manifest_sha=sha256(source),
                        code_fingerprint=fingerprint, provenance_marker="STRICT_REAL_CHECKPOINT_INFERENCE")
            inputs += [dict(path=str(p.resolve()), sha256=sha256(p)) for p in [source, *checkpoints.values()]]
            cache = args.reuse_from / (stem + "_predictions.csv") if args.reuse_from else None
            if cache and cache.is_file():
                frame = pd.read_csv(cache, dtype={"image_id": str})
                validate_cache(frame, manifest, meta)
                events.append(dict(cohort=cohort, mode="VALIDATED_CACHE", source=str(cache.resolve()), sha256=sha256(cache)))
            else:
                model2 = build_checkpoint_model("v2", checkpoints["v2"], args.device, args.patch_chunk_size)
                model3 = build_checkpoint_model("v3", checkpoints["v3"], args.device, args.patch_chunk_size)
                dataset = FundusDualBranchDataset(csv_path=manifest, img_root=str(args.data_root), img_size=512, split="test")
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    collate_fn=partial(trainer_compatible_collate_fn, mode="eval"))
                p2, p3, expected = [], [], []
                offset = 0
                for batch in loader:
                    ids = [str(i) for i in batch["image_id"]]
                    n = len(ids)
                    if ids != manifest.image_id.iloc[offset:offset+n].tolist() or not np.array_equal(batch["label"].numpy(), manifest.diagnosis.iloc[offset:offset+n]):
                        raise ValueError(f"Actual loader ID/target misalignment: {cohort}/{offset}")
                    g, l = normalize_inputs(batch["global_img"], batch["local_patches"], args.device)
                    mask = batch["patch_mask"].to(args.device)
                    with torch.inference_mode():
                        a, b = model2(g, l, patch_mask=mask), model3(g, l, patch_mask=mask)
                    aa, bb = a["probs"].cpu().numpy(), b["probs"].cpu().numpy()
                    exp = b["expected_grade"].cpu().numpy()
                    p2.extend(aa.argmax(1).tolist())
                    p3.extend(np.clip(np.round(exp), 0, 4).astype(int).tolist())
                    expected.extend(exp.tolist())
                    for i in range(min(n, max(0, 3 - sum(t["cohort"] == cohort for t in trace)))):
                        trace.append(dict(cohort=cohort, image_id=ids[i], y_true=int(batch["label"][i]),
                                          v2_probs=aa[i].tolist(), v3_probs=bb[i].tolist()))
                    offset += n
                    print(f"{cohort}: {offset}/{len(manifest)}", flush=True)
                if offset != len(manifest):
                    raise ValueError("Incomplete inference")
                frame = manifest.assign(pred_v2=p2, pred_v3=p3, v3_expected_grade=expected, **meta)
                events.append(dict(cohort=cohort, mode="REAL_FORWARD_INFERENCE"))
                del model2, model3
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            frame.to_csv(predictions / (stem + "_predictions.csv"), index=False)
            frames[cohort] = frame
        (run / "inference_provenance.json").write_text(json.dumps(dict(runtime=runtime, fingerprint=fingerprint,
            events=events, inputs=inputs, sample_trace=trace), indent=2), encoding="utf-8")
        # Partial diagnostic subsets must not be compared with full-cohort supports.
        report = analyze_frames(frames, None if args.max_samples else args.reference, run / "analysis", inputs,
                                reps=args.bootstrap_reps, seed=args.seed, source_mode="checkpoint_inference",
                                exact_reference=False, diagnostic_only=bool(args.max_samples))
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "BENCHMARK_REPRODUCED" else 2
    except Exception as exc:
        (run / "FAILED.json").write_text(json.dumps(dict(status="FAILED", error=str(exc), completed_cohorts=list(frames)), indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
