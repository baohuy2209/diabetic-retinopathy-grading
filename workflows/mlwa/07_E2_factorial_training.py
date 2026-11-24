# %% [markdown]
# # 07 · E2 Controlled 2x2 Head x Objective Factorial Training

# %%
NOTEBOOK_CODE_SHA256 = 'd049194bd18ea93d309b9600fb158fdca60f47f82f74afc3824039e1b1df1373'

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


def _bootstrap_colab_repo() -> Path:
    import importlib.util
    import subprocess
    import zipfile

    if IN_COLAB:
        deps = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'pyarrow': 'pyarrow'}
        missing = [pkg for mod, pkg in deps.items() if importlib.util.find_spec(mod) is None]
        if missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        if not Path('/content/drive/MyDrive').is_dir():
            from google.colab import drive
            drive.mount('/content/drive')

    dr_root = Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()
    bundle_zip = Path(os.environ.get('DR_MLWA_BUNDLE', dr_root / 'inputs' / 'mlwa_journal_bundle.zip'))
    local_pkg = Path(os.environ.get('DR_WORK', '/content/dr_thesis_work' if IN_COLAB else dr_root / 'work')).resolve() / 'mlwa_pkg'
    if (dr_root / 'experiments' / 'mlwa').is_dir():
        repo = dr_root
    elif bundle_zip.is_file():
        local_pkg.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip) as zf:
            zf.extractall(local_pkg)
        repo = local_pkg
    else:
        repo = Path.cwd().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _bootstrap_colab_repo()
    CELLS = ['A', 'B', 'C', 'D']
    FOLDS = list(range(6))
    SEEDS = [0, 1, 2]
    ARCH_TIER = 'primary_dual_branch'  # or 'fallback_global_resnet50'
    SMOKE = True

# %%
import argparse
import json
from typing import Dict, List, Optional, Sequence

import torch

from experiments.mlwa.config import (
    JournalExperimentConfig,
    ModelSpec,
    SeedConfig,
    TrainingSpec,
    build_factorial_cell_spec,
)
from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT, resolve_runtime_paths
from experiments.mlwa.models_adapters import JournalGradingModel
from experiments.mlwa.objectives import JournalObjective
from experiments.mlwa.plan import verify_frozen_plan
from experiments.mlwa.runner_utils import (
    get_code_identity,
    prepare_real_fold_loaders,
    run_smoke_training_job,
)
from experiments.mlwa.trainer import initialize_shared_encoder_state, train_single_unit


def list_e2_jobs(
    cells: Sequence[str] = ('A', 'B', 'C', 'D'),
    folds: Sequence[int] = tuple(range(6)),
    seeds: Sequence[int] = (0, 1, 2),
    arch_tier: str = 'primary_dual_branch',
) -> List[Dict[str, object]]:
    jobs = []
    for cell in cells:
        cfg_id, _, _ = build_factorial_cell_spec(cell)
        for fid in folds:
            cohort = CANONICAL_FOLD_TO_COHORT[fid]
            for seed in seeds:
                jobs.append({
                    'cell': cell,
                    'config_id': cfg_id,
                    'fold_id': fid,
                    'cohort': cohort,
                    'seed_train': seed,
                    'architecture_tier': arch_tier,
                })
    return jobs


def run_e2_jobs(
    output_dir: Path,
    cells: Sequence[str] = ('A', 'B', 'C', 'D'),
    folds: Sequence[int] = (0,),
    seeds: Sequence[int] = (0,),
    arch_tier: str = 'primary_dual_branch',
    folds_dir: Optional[Path] = None,
    images_root: Optional[Path] = None,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    run_id: str = 'e2_factorial_v1',
    smoke: bool = False,
    device: str = 'auto',
) -> List[Dict[str, object]]:
    lock = verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=smoke)
    results: List[Dict[str, object]] = []
    arch_id = 'global_resnet50_v1' if arch_tier == 'fallback_global_resnet50' else 'dual_branch_mil_v1'

    for job in list_e2_jobs(cells=cells, folds=folds, seeds=seeds, arch_tier=arch_tier):
        cfg_id, mspec_base, ospec = build_factorial_cell_spec(str(job['cell']), pretrained=not smoke)
        mspec = ModelSpec(
            architecture_id=arch_id,
            head_type=mspec_base.head_type,
            clm_link=mspec_base.clm_link,
            pretrained=not smoke,
            use_local_branch=(arch_id == 'dual_branch_mil_v1'),
            use_nonlocal=(arch_id == 'dual_branch_mil_v1'),
            use_quadrant_tokens=(arch_id == 'dual_branch_mil_v1'),
            use_cbam=(arch_id == 'dual_branch_mil_v1'),
        )
        if smoke:
            res = run_smoke_training_job(
                experiment='E2',
                config_id=cfg_id,
                model_spec=mspec,
                objective_spec=ospec,
                output_dir=output_dir,
                fold_id=int(job['fold_id']),
                seed_train=int(job['seed_train']),
                run_id=f'{run_id}_TEST_ONLY',
            )
            results.append({'job': job, 'result': res})
            continue

        if folds_dir is None or images_root is None:
            raise ValueError('Production E2 training requires --folds-dir and --images-root.')
        dev = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        cfg = JournalExperimentConfig(
            run_id=run_id,
            experiment='E2',
            config_id=cfg_id,
            fold_id=int(job['fold_id']),
            cohort=str(job['cohort']),
            seeds=SeedConfig(seed_train=int(job['seed_train'])),
            model=mspec,
            objective=ospec,
            training=TrainingSpec(),
        )
        tr_loader, va_loader, _, nat_counts, hashes, gen = prepare_real_fold_loaders(Path(folds_dir), Path(images_root), cfg)
        model = JournalGradingModel(mspec)
        initialize_shared_encoder_state(model, cfg.seeds.seed_train)
        obj = JournalObjective(ospec, natural_train_counts=nat_counts)
        res = train_single_unit(
            cfg=cfg,
            model=model,
            objective=obj,
            source_train_loader=tr_loader,
            source_val_loader=va_loader,
            output_dir=output_dir,
            source_train_manifest_sha256=hashes['source_train'],
            source_val_manifest_sha256=hashes['source_val'],
            code_sha256=get_code_identity(),
            plan_sha256=str(lock['plan_sha256']),
            device=dev,
            loader_generator=gen,
        )
        results.append({'job': job, 'result': res})
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='07: E2 2x2 Head x Objective Factorial Training (72 units full).')
    p.add_argument('--cells', nargs='+', default=['A', 'B', 'C', 'D'], choices=['A', 'B', 'C', 'D'])
    p.add_argument('--folds', nargs='+', type=int, default=list(range(6)))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--architecture-tier', default='primary_dual_branch', choices=['primary_dual_branch', 'fallback_global_resnet50'])
    p.add_argument('--list-jobs', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--folds-dir', type=Path, default=None)
    p.add_argument('--images-root', type=Path, default=None)
    p.add_argument('--plan-path', type=Path, default=Path('docs/mlwa/analysis_plan.md'))
    p.add_argument('--plan-lock-path', type=Path, default=Path('docs/mlwa/analysis_plan.lock.json'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_journal_e2'))
    p.add_argument('--run-id', type=str, default='e2_factorial_v1')
    p.add_argument('--device', type=str, default='auto')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_jobs:
        jobs = list_e2_jobs(args.cells, args.folds, args.seeds, args.architecture_tier)
        print(f'Total unique E2 jobs: {len(jobs)}')
        for j in jobs[:10]:
            print(' ', j)
        return 0
    if args.preflight or args.dry_run:
        lock = verify_frozen_plan(args.plan_path, args.plan_lock_path, allow_test_only=args.smoke or args.dry_run)
        print(json.dumps({'status': 'PREFLIGHT_OK', 'n_jobs': len(list_e2_jobs(args.cells, args.folds, args.seeds, args.architecture_tier)), 'plan': lock}, indent=2))
        return 0
    folds = [0] if (args.smoke and args.folds == list(range(6))) else args.folds
    seeds = [0] if (args.smoke and args.seeds == [0, 1, 2]) else args.seeds
    res = run_e2_jobs(
        output_dir=args.output_dir, cells=args.cells, folds=folds, seeds=seeds,
        arch_tier=args.architecture_tier, folds_dir=args.folds_dir, images_root=args.images_root,
        plan_path=args.plan_path, plan_lock_path=args.plan_lock_path, run_id=args.run_id,
        smoke=args.smoke, device=args.device,
    )
    print(json.dumps({'completed_jobs': len(res)}, indent=2))
    return 0


# %%
if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths()
    print(run_e2_jobs(
        output_dir=_paths['outputs'] / 'mlwa_journal_e2',
        cells=CELLS,
        folds=[0] if SMOKE else FOLDS,
        seeds=[0] if SMOKE else SEEDS,
        arch_tier=ARCH_TIER,
        folds_dir=_paths['folds'],
        images_root=_paths['images'],
        smoke=SMOKE,
    ))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
