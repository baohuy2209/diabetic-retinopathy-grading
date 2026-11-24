# %% [markdown]
# # 09 · E4 Mandatory Imbalance Baselines (Cui Effective-Number CB-CE & Menon Logit Adjustment)

# %%
NOTEBOOK_CODE_SHA256 = '49fab0eff3a2b70e792ffefbc628fe99ef36628d8595a034cf2ce5dd1418d54c'

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


if __name__ == '__main__' and _running_in_notebook():
    _REPO = _bootstrap_colab_repo()
    BASELINES = ['cui_cb_ce', 'menon_logit_adj_ce']
    FOLDS = list(range(6))
    SEEDS = [0, 1, 2]
    SMOKE = True

# %%
import argparse
import json
from typing import Dict, List, Optional, Sequence

import torch

from experiments.mlwa.config import JournalExperimentConfig, ModelSpec, ObjectiveSpec, SeedConfig, TrainingSpec
from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT, resolve_runtime_paths
from experiments.mlwa.models_adapters import JournalGradingModel
from experiments.mlwa.objectives import JournalObjective
from experiments.mlwa.plan import verify_frozen_plan
from experiments.mlwa.runner_utils import get_code_identity, prepare_real_fold_loaders, run_smoke_training_job
from experiments.mlwa.trainer import initialize_shared_encoder_state, train_single_unit

E4_CONFIG_MAP = {
    'cui_cb_ce': ('E4_cui_cb_ce', ModelSpec(head_type='softmax', clm_link='none'), ObjectiveSpec(objective_id='cui_cb_ce', cui_beta=0.9999)),
    'menon_logit_adj_ce': ('E4_menon_logit_adj_ce', ModelSpec(head_type='softmax', clm_link='none'), ObjectiveSpec(objective_id='menon_logit_adj_ce', logit_adj_tau_train=1.0)),
    'galdran_cost_sensitive_ce': ('E4_galdran_csr_ce', ModelSpec(head_type='softmax', clm_link='none'), ObjectiveSpec(objective_id='galdran_cost_sensitive_ce', galdran_lambda=1.0)),
    'clm_cui_weighted_nll': ('E4_clm_cui_nll', ModelSpec(head_type='clm', clm_link='cloglog'), ObjectiveSpec(objective_id='clm_cui_weighted_nll', cui_beta=0.9999)),
    'clm_logprob_adjusted_nll': ('E4_clm_logprob_adj_nll', ModelSpec(head_type='clm', clm_link='cloglog'), ObjectiveSpec(objective_id='clm_logprob_adjusted_nll', logit_adj_tau_train=1.0)),
}


def run_e4_jobs(
    output_dir: Path,
    baselines: Sequence[str] = ('cui_cb_ce', 'menon_logit_adj_ce'),
    folds: Sequence[int] = (0,),
    seeds: Sequence[int] = (0,),
    folds_dir: Optional[Path] = None,
    images_root: Optional[Path] = None,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    run_id: str = 'e4_imbalance_v1',
    smoke: bool = False,
    device: str = 'auto',
) -> List[Dict[str, object]]:
    lock = verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=smoke)
    results = []
    for b_key in baselines:
        cfg_id, mspec_base, ospec = E4_CONFIG_MAP[b_key]
        for fid in folds:
            cohort = CANONICAL_FOLD_TO_COHORT[fid]
            for seed in seeds:
                mspec = ModelSpec(**{**mspec_base.__dict__, 'pretrained': not smoke})
                if smoke:
                    res = run_smoke_training_job(
                        experiment='E4',
                        config_id=cfg_id,
                        model_spec=mspec,
                        objective_spec=ospec,
                        output_dir=output_dir,
                        fold_id=fid,
                        seed_train=seed,
                        run_id=f'{run_id}_TEST_ONLY',
                    )
                    results.append({'baseline': b_key, 'fold_id': fid, 'seed_train': seed, 'result': res})
                    continue
                if folds_dir is None or images_root is None:
                    raise ValueError('Production E4 training requires --folds-dir and --images-root.')
                dev = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
                cfg = JournalExperimentConfig(
                    run_id=run_id,
                    experiment='E4',
                    config_id=cfg_id,
                    fold_id=fid,
                    cohort=cohort,
                    seeds=SeedConfig(seed_train=seed),
                    model=mspec,
                    objective=ospec,
                    training=TrainingSpec(sampling_policy='natural_source_train'),
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
                results.append({'baseline': b_key, 'fold_id': fid, 'seed_train': seed, 'result': res})
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='09: E4 Mandatory Imbalance Baselines (Cui CB-CE & Menon Logit Adj).')
    p.add_argument('--baselines', nargs='+', default=['cui_cb_ce', 'menon_logit_adj_ce'], choices=list(E4_CONFIG_MAP.keys()))
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
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_journal_e4'))
    p.add_argument('--run-id', type=str, default='e4_imbalance_v1')
    p.add_argument('--device', type=str, default='auto')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_jobs:
        n = len(args.baselines) * len(args.folds) * len(args.seeds)
        print(f'Unique E4 jobs ({args.baselines}): {n} (reuses E2_A_softmax_ce as unweighted baseline)')
        return 0
    if args.preflight or args.dry_run:
        verify_frozen_plan(args.plan_path, args.plan_lock_path, allow_test_only=args.smoke or args.dry_run)
        print(json.dumps({'status': 'PREFLIGHT_OK', 'baselines': args.baselines, 'jobs': len(args.baselines) * len(args.folds) * len(args.seeds)}))
        return 0
    folds = [0] if (args.smoke and args.folds == list(range(6))) else args.folds
    seeds = [0] if (args.smoke and args.seeds == [0, 1, 2]) else args.seeds
    res = run_e4_jobs(
        output_dir=args.output_dir, baselines=args.baselines, folds=folds, seeds=seeds,
        folds_dir=args.folds_dir, images_root=args.images_root, plan_path=args.plan_path,
        plan_lock_path=args.plan_lock_path, run_id=args.run_id, smoke=args.smoke, device=args.device,
    )
    print(json.dumps({'completed_jobs': len(res)}, indent=2))
    return 0


# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths()
    print(run_e4_jobs(_paths['outputs'] / 'mlwa_journal_e4', baselines=BASELINES, folds=[0] if SMOKE else FOLDS, seeds=[0] if SMOKE else SEEDS, smoke=SMOKE))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
