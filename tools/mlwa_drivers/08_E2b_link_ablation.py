# %% [markdown]
# # 08 · E2b CLM Link Ablation (logit vs. cloglog, reusing E2-D)

# %%
NOTEBOOK_CODE_SHA256 = 'd943083cbec772afdd03d597dc3667f06eaf831812ae12b99cc600010de0b7ca'

# %%
import os
import sys
from pathlib import Path

IN_COLAB: bool = (
    'COLAB_RELEASE_TAG' in os.environ
    or 'COLAB_BACKEND_VERSION' in os.environ
    or 'google.colab' in sys.modules
)


def _running_in_notebook() -> bool:
    if 'ipykernel' not in sys.modules and 'google.colab' not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


def _setup_colab_environment() -> Path:
    if IN_COLAB:
        import importlib.util
        import subprocess
        deps = {
            'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn',
            'pyarrow': 'pyarrow', 'torch': 'torch', 'torchvision': 'torchvision',
            'cv2': 'opencv-python-headless', 'albumentations': 'albumentations',
        }
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')
    return Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _setup_colab_environment()
    FOLDS = list(range(6))
    SEEDS = [0, 1, 2]
    OUTPUT_DIR = None
    SMOKE = True

# %%
import argparse
import json
from typing import Dict, List, Optional, Sequence

import torch

from experiments.mlwa.config import JournalExperimentConfig, ModelSpec, ObjectiveSpec, SeedConfig, TrainingSpec
from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT, resolve_runtime_paths
from experiments.mlwa.decoders import compute_clm_decision_regions
from experiments.mlwa.models_adapters import JournalGradingModel
from experiments.mlwa.objectives import JournalObjective
from experiments.mlwa.plan import verify_frozen_plan
from experiments.mlwa.runner_utils import get_code_identity, prepare_real_fold_loaders, run_smoke_training_job
from experiments.mlwa.trainer import initialize_shared_encoder_state, train_single_unit


def run_e2b_logit_jobs(
    output_dir: Path,
    folds: Sequence[int] = (0,),
    seeds: Sequence[int] = (0,),
    folds_dir: Optional[Path] = None,
    images_root: Optional[Path] = None,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    run_id: str = 'e2b_link_ablation_v1',
    smoke: bool = False,
    device: str = 'auto',
) -> List[Dict[str, object]]:
    """Train matched CLM(logit) + NLL + symmetric QWK jobs (18 unique units; cloglog reuses E2-D)."""
    lock = verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=smoke)
    results = []
    ospec = ObjectiveSpec(objective_id='clm_nll_qwk', use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15)

    for fid in folds:
        cohort = CANONICAL_FOLD_TO_COHORT[fid]
        for seed in seeds:
            mspec = ModelSpec(head_type='clm', clm_link='logit', pretrained=not smoke)
            if smoke:
                res = run_smoke_training_job(
                    experiment='E2b',
                    config_id='E2b_clm_logit_nll_qwk',
                    model_spec=mspec,
                    objective_spec=ospec,
                    output_dir=output_dir,
                    fold_id=fid,
                    seed_train=seed,
                    run_id=f'{run_id}_TEST_ONLY',
                )
                reg = compute_clm_decision_regions([-1.5, -0.5, 0.5, 1.5], link='logit')
                results.append({'fold_id': fid, 'seed_train': seed, 'result': res, 'analytic_median': reg['analytic_median_boundaries']})
                continue
            if folds_dir is None or images_root is None:
                raise ValueError('Production E2b requires --folds-dir and --images-root.')
            dev = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            cfg = JournalExperimentConfig(
                run_id=run_id,
                experiment='E2b',
                config_id='E2b_clm_logit_nll_qwk',
                fold_id=fid,
                cohort=cohort,
                seeds=SeedConfig(seed_train=seed),
                model=mspec,
                objective=ospec,
                training=TrainingSpec(),
            )
            tr_loader, va_loader, _, nat_counts, hashes, gen = prepare_real_fold_loaders(Path(folds_dir), Path(images_root), cfg)
            model = JournalGradingModel(mspec)
            initialize_shared_encoder_state(model, seed)
            obj = JournalObjective(ospec, natural_train_counts=nat_counts)
            res = train_single_unit(
                cfg=cfg, model=model, objective=obj, source_train_loader=tr_loader, source_val_loader=va_loader,
                output_dir=output_dir, source_train_manifest_sha256=hashes['source_train'],
                source_val_manifest_sha256=hashes['source_val'], code_sha256=get_code_identity(),
                plan_sha256=str(lock['plan_sha256']), device=dev, loader_generator=gen,
            )
            results.append({'fold_id': fid, 'seed_train': seed, 'result': res})
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='08: E2b CLM logit vs cloglog ablation (reuses E2-D).')
    p.add_argument('--folds', nargs='+', type=int, default=list(range(6)))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--list-jobs', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--folds-dir', type=Path, default=None)
    p.add_argument('--images-root', type=Path, default=None)
    p.add_argument('--plan-path', type=Path, default=Path('docs/mlwa/analysis_plan.md'))
    p.add_argument('--plan-lock-path', type=Path, default=Path('docs/mlwa/analysis_plan.lock.json'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_journal_e2b'))
    p.add_argument('--run-id', type=str, default='e2b_link_ablation_v1')
    p.add_argument('--device', type=str, default='auto')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_jobs:
        print(f'18 unique logit CLM-NLL+QWK jobs (6 folds x 3 seeds); cloglog baseline reuses E2-D.')
        return 0
    if args.preflight or args.dry_run:
        verify_frozen_plan(args.plan_path, args.plan_lock_path, allow_test_only=args.smoke or args.dry_run)
        print(json.dumps({'status': 'PREFLIGHT_OK', 'new_units': len(args.folds) * len(args.seeds), 'reused_baseline': 'E2_D_clm_cloglog_nll_qwk'}))
        return 0
    folds = [0] if (args.smoke and args.folds == list(range(6))) else args.folds
    seeds = [0] if (args.smoke and args.seeds == [0, 1, 2]) else args.seeds
    res = run_e2b_logit_jobs(
        output_dir=args.output_dir, folds=folds, seeds=seeds, folds_dir=args.folds_dir, images_root=args.images_root,
        plan_path=args.plan_path, plan_lock_path=args.plan_lock_path, run_id=args.run_id, smoke=args.smoke, device=args.device,
    )
    print(json.dumps({'completed_jobs': len(res)}, indent=2))
    return 0


# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_journal_e2b')
    print(run_e2b_logit_jobs(
        _out,
        folds=[0] if SMOKE else FOLDS,
        seeds=[0] if SMOKE else SEEDS,
        folds_dir=_paths['folds'],
        images_root=_paths['images'],
        smoke=SMOKE,
    ))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
