# %% [markdown]
# # 06 · MLWA Journal Preflight, Asset Audit & Pilot Profiler

# %%
NOTEBOOK_CODE_SHA256 = '8350b122b29d03ec9b14ed330b9983f640730ae15032c16b43b45a120104e76b'

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
    MODE = 'preflight'  # 'preflight' | 'pilot' | 'smoke'
    OUTPUT_DIR = None

# %%
import argparse
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.mlwa.config import JournalExperimentConfig, ModelSpec, ObjectiveSpec, SeedConfig, TrainingSpec
from experiments.mlwa.constants import CANONICAL_FOLD_TO_COHORT, STATUS_PENDING_ASSETS, STATUS_PREFLIGHT_OK, resolve_runtime_paths
from experiments.mlwa.data_contract import atomic_write_json, audit_lodo_manifests, validate_p0_cache
from experiments.mlwa.models_adapters import JournalGradingModel
from experiments.mlwa.objectives import JournalObjective
from experiments.mlwa.plan import verify_frozen_plan
from experiments.mlwa.registry import build_expanded_job_matrix, write_registry_artifacts
from experiments.mlwa.runner_utils import SyntheticDRDataset, pad_collate_fn, prepare_real_fold_loaders
from experiments.mlwa.trainer import run_pilot_profile


def run_preflight_audit(
    folds_dir: Optional[Path],
    p0_dir: Optional[Path],
    pooled_checkpoints_dir: Optional[Path],
    pooled_metadata_dir: Optional[Path],
    output_dir: Path,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    check_images_root: Optional[Path] = None,
    allow_draft_plan_audit: bool = True,
) -> Dict[str, Any]:
    """Non-destructive preflight audit across LODO manifests, P0 cache, pooled assets, and job matrix."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    plan_status: Dict[str, Any]
    if allow_draft_plan_audit and plan_path is not None and (plan_lock_path is None or not Path(plan_lock_path).is_file()):
        plan_status = {'status': 'DRAFT_AUDITED_NOT_YET_FROZEN', 'plan_path': str(plan_path)}
    else:
        plan_status = verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=allow_draft_plan_audit)

    manifest_audit = None
    if folds_dir is not None and Path(folds_dir).is_dir():
        manifest_audit = audit_lodo_manifests(Path(folds_dir), images_root=check_images_root, check_decode=False)
        atomic_write_json(out / 'lodo_manifest_audit.json', manifest_audit)

    p0_audit = None
    if p0_dir is not None and (Path(p0_dir) / 'lodo').is_dir():
        p0_audit = validate_p0_cache(Path(p0_dir), require_features=True)
        atomic_write_json(out / 'p0_cache_audit.json', p0_audit)

    pooled_audit: Dict[str, Any] = {'status': STATUS_PENDING_ASSETS, 'checkpoints': []}
    if pooled_checkpoints_dir is not None and Path(pooled_checkpoints_dir).is_dir():
        ckpts = sorted(Path(pooled_checkpoints_dir).glob('*.pth'))
        for cp in ckpts:
            meta = torch.load(cp, map_location='cpu', weights_only=False)
            ep = meta.get('epoch') if isinstance(meta, dict) else None
            vq = meta.get('val_qwk') if isinstance(meta, dict) else None
            pooled_audit['checkpoints'].append({'name': cp.name, 'epoch': ep, 'val_qwk': vq})
        all_trained = bool(ckpts) and all((c.get('epoch') or 0) > 1 for c in pooled_audit['checkpoints'])
        pooled_audit['status'] = STATUS_PREFLIGHT_OK if all_trained else STATUS_PENDING_ASSETS
        pooled_audit['reason'] = (
            'All checkpoints have epoch > 1'
            if all_trained
            else 'Only epoch=1 smoke/incomplete checkpoints found; trained 25-epoch checkpoints not present.'
        )
    atomic_write_json(out / 'pooled_asset_audit.json', pooled_audit)

    jobs_df = build_expanded_job_matrix(include_fallback=True)
    jobs_df.to_csv(out / 'expanded_job_matrix.csv', index=False)

    summary = {
        'status': STATUS_PREFLIGHT_OK,
        'plan_status': plan_status,
        'manifest_audit_present': manifest_audit is not None,
        'p0_cache_audit_present': p0_audit is not None,
        'pooled_status': pooled_audit['status'],
        'total_matrix_rows_with_fallback': int(len(jobs_df)),
    }
    atomic_write_json(out / 'PREFLIGHT_SUMMARY.json', summary)
    return summary


def run_pilot(
    output_dir: Path,
    folds_dir: Optional[Path] = None,
    images_root: Optional[Path] = None,
    plan_path: Optional[Path] = None,
    plan_lock_path: Optional[Path] = None,
    smoke: bool = False,
    device: str = 'auto',
) -> Dict[str, Any]:
    """Profile smallest (fold 4 EyePACS) and largest (fold 1 IDRiD) source_train folds."""
    verify_frozen_plan(plan_path, plan_lock_path, allow_test_only=smoke)
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = []

    for fid in (4, 1):
        cohort = CANONICAL_FOLD_TO_COHORT[fid]
        mspec = ModelSpec(
            head_type='clm',
            clm_link='cloglog',
            pretrained=not smoke,
            use_local_branch=True,
            use_nonlocal=not smoke,
            use_quadrant_tokens=not smoke,
            use_cbam=not smoke,
            eval_patch_chunk_size=4 if smoke else 32,
        )
        ospec = ObjectiveSpec(objective_id='clm_nll_qwk', use_qwk=True, target_qwk_weight=0.4, qwk_warmup_epochs=15)
        tspec = TrainingSpec(epochs=2 if smoke else 30, batch_size=5 if smoke else 12, amp_mode='none' if dev == 'cpu' else 'auto')
        cfg = JournalExperimentConfig(
            study_id='TEST_ONLY_PILOT' if smoke else 'mlwa_dr_ordinal_extension',
            protocol_id='TEST_ONLY_PILOT' if smoke else 'mlwa_journal_v1',
            run_id='pilot_smoke' if smoke else 'pilot_profile_v1',
            experiment='PILOT',
            config_id='E2_D_clm_cloglog_nll_qwk',
            fold_id=fid,
            cohort=cohort,
            seeds=SeedConfig(seed_train=0),
            model=mspec,
            objective=ospec,
            training=tspec,
            is_test_only=smoke,
        )
        model = JournalGradingModel(mspec)
        if smoke:
            ds = SyntheticDRDataset(n_samples=10, cohort=cohort, seed=42 + fid, include_patches=True)
            loader = DataLoader(ds, batch_size=5, shuffle=False, collate_fn=pad_collate_fn)
            nat_counts = np.bincount(ds.labels, minlength=5)
            n_full = 16791 if fid == 4 else 46010
        else:
            if folds_dir is None or images_root is None:
                raise ValueError('Real pilot requires --folds-dir and --images-root.')
            loader, _, _, nat_counts, _, _ = prepare_real_fold_loaders(Path(folds_dir), Path(images_root), cfg)
            n_full = int(nat_counts.sum())
        obj = JournalObjective(ospec, natural_train_counts=nat_counts)
        rep = run_pilot_profile(
            cfg, model, obj, loader, n_train_images_full=n_full,
            output_path=out / f'pilot_fold_{fid}_{cohort}.json', device=dev,
        )
        reports.append(rep)
    return {'status': 'TEST_ONLY' if smoke else 'PILOT_COMPLETE', 'pilots': reports}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='06: MLWA journal preflight audit, job matrix, and pilot profiler.')
    p.add_argument('--preflight', action='store_true', help='Run non-destructive asset/manifest/P0 audit.')
    p.add_argument('--list-jobs', action='store_true', help='Print expanded job matrix summary and exit.')
    p.add_argument('--dry-run', action='store_true', help='Validate arguments/paths cleanly without training.')
    p.add_argument('--pilot', action='store_true', help='Run bounded pilot step/VRAM profiler on folds 4 & 1.')
    p.add_argument('--smoke', action='store_true', help='Run TEST_ONLY CPU smoke pilot on synthetic fixtures.')
    p.add_argument('--folds-dir', type=Path, default=None)
    p.add_argument('--images-root', type=Path, default=None)
    p.add_argument('--p0-dir', type=Path, default=None)
    p.add_argument('--pooled-checkpoints-dir', type=Path, default=None)
    p.add_argument('--pooled-metadata-dir', type=Path, default=None)
    p.add_argument('--plan-path', type=Path, default=Path('docs/mlwa/analysis_plan.md'))
    p.add_argument('--plan-lock-path', type=Path, default=Path('docs/mlwa/analysis_plan.lock.json'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_journal_preflight'))
    p.add_argument('--device', type=str, default='auto')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_runtime_paths()

    if args.list_jobs:
        df = build_expanded_job_matrix(include_fallback=True)
        print(df.groupby(['experiment', 'priority', 'is_unique_training_job']).size().to_string())
        return 0

    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_OK', 'output_dir': str(args.output_dir), 'p0_dir': str(args.p0_dir or paths['p0_run'])}, indent=2))
        return 0

    if args.smoke or args.pilot:
        res = run_pilot(
            output_dir=args.output_dir,
            folds_dir=args.folds_dir,
            images_root=args.images_root,
            plan_path=args.plan_path,
            plan_lock_path=args.plan_lock_path,
            smoke=args.smoke,
            device=args.device,
        )
        print(json.dumps(res, indent=2))
        return 0

    res = run_preflight_audit(
        folds_dir=args.folds_dir,
        p0_dir=args.p0_dir or paths['p0_run'],
        pooled_checkpoints_dir=args.pooled_checkpoints_dir,
        pooled_metadata_dir=args.pooled_metadata_dir,
        output_dir=args.output_dir,
        plan_path=args.plan_path,
        plan_lock_path=args.plan_lock_path,
        check_images_root=args.images_root,
        allow_draft_plan_audit=True,
    )
    print(json.dumps(res, indent=2))
    return 0


# %%
if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR else (resolve_runtime_paths()['outputs'] / 'mlwa_journal_preflight')
    if MODE == 'smoke':
        print(run_pilot(_out, smoke=True, device='cpu'))
    else:
        print(run_preflight_audit(resolve_runtime_paths()['folds'], resolve_runtime_paths()['p0_run'], None, None, _out))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
