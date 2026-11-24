# %% [markdown]
# # 10 · Historical Pooled Five-Head Audit & Strict Reconstruction (`PENDING_ASSETS` Aware)

# %%
NOTEBOOK_CODE_SHA256 = '2ee124a79fca3d32f488109cd8b25f4cc58a08d2de96c7cdd43da53374eb6039'

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
    MODE = 'audit'  # 'audit' | 'smoke'
    OUTPUT_DIR = None
    SMOKE = True

# %%
import argparse
import json
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch

from experiments.mlwa.constants import STATUS_PENDING_ASSETS, resolve_runtime_paths
from experiments.mlwa.data_contract import atomic_write_json, sha256_file
from experiments.mlwa.models_adapters import PooledResNet50StemProjAdapter

POOLED_HEADS = ('Softmax_CE', 'Softmax_QWK', 'CORAL', 'CORN', 'CLM_QWK')
FILENAME_MAP = {
    'Softmax_CE': ['resnet50_softmax_ce_best.pth', 'softmax_ce_best.pth'],
    'Softmax_QWK': ['resnet50_softmax_qwk_best.pth', 'softmax_qwk_best.pth'],
    'CORAL': ['resnet50_coral_best.pth', 'coral_best.pth'],
    'CORN': ['resnet50_corn_best.pth', 'corn_best.pth'],
    'CLM_QWK': ['resnet50_clm_qwk_best.pth', 'clm_qwk_best.pth'],
}


def audit_pooled_assets(
    pooled_checkpoints_dir: Optional[Path],
    pooled_metadata_dir: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Strictly audit pooled 5-head checkpoints and manifests without using strict=False."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_status: Dict[str, Any] = {}
    if pooled_metadata_dir is not None and Path(pooled_metadata_dir).is_dir():
        for fn in ('train_balanced_metadata.csv', 'val_metadata.csv', 'test_metadata.csv'):
            fp = Path(pooled_metadata_dir) / fn
            manifest_status[fn] = {'exists': fp.is_file(), 'sha256': sha256_file(fp) if fp.is_file() else None}

    head_reports: Dict[str, Any] = {}
    missing_items = []
    for h_name in POOLED_HEADS:
        found = None
        if pooled_checkpoints_dir is not None and Path(pooled_checkpoints_dir).is_dir():
            for cand in FILENAME_MAP[h_name]:
                cp = Path(pooled_checkpoints_dir) / cand
                if cp.is_file():
                    found = cp
                    break
        if found is None:
            missing_items.append(h_name)
            head_reports[h_name] = {'status': STATUS_PENDING_ASSETS, 'reason': 'checkpoint_missing'}
            continue

        ck = torch.load(found, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        ep = ck.get('epoch') if isinstance(ck, dict) else None
        vq = ck.get('val_qwk') if isinstance(ck, dict) else None

        adapter = PooledResNet50StemProjAdapter(h_name)
        try:
            adapter.load_state_dict(sd, strict=True)
            strict_ok = True
            strict_err = None
        except Exception as exc:
            strict_ok = False
            strict_err = str(exc)[:300]

        usable = strict_ok and (ep is not None and ep > 1)
        if not usable:
            missing_items.append(f'{h_name} (epoch={ep}, strict_ok={strict_ok})')
        head_reports[h_name] = {
            'status': 'VERIFIED' if usable else STATUS_PENDING_ASSETS,
            'checkpoint': str(found),
            'sha256': sha256_file(found),
            'epoch': ep,
            'val_qwk': vq,
            'strict_load_stem_proj_v1': strict_ok,
            'strict_error': strict_err,
        }

    overall = 'READY_FOR_POOLED_INFERENCE' if not missing_items else STATUS_PENDING_ASSETS
    report = {
        'protocol_id': 'historical_pooled_heads_v1',
        'status': overall,
        'missing_or_incomplete': missing_items,
        'manifests': manifest_status,
        'heads': head_reports,
        'note': (
            'Applying CIBD to 5 pooled heads evaluates the separate pooled split (8,075 test images), '
            'NOT the 6-fold LODO protocol. Training a fresh multi-head study uses a new protocol_id.'
        ),
    }
    atomic_write_json(out / 'POOLED_HEADS_STATUS.json', report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='10: Audit/reconstruct historical five-head pooled benchmark.')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--list-jobs', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--pooled-checkpoints-dir', type=Path, default=None)
    p.add_argument('--pooled-metadata-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/mlwa_pooled_heads'))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_jobs:
        print(f'Historical pooled heads ({len(POOLED_HEADS)}): {POOLED_HEADS} on separate protocol historical_pooled_heads_v1')
        return 0
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_OK', 'heads': list(POOLED_HEADS), 'output_dir': str(args.output_dir)}))
        return 0
    if args.smoke:
        # Smoke test strict adapter forward pass on 2 dummy images with zero downloads
        rows = []
        x = torch.randn(2, 3, 64, 64)
        for h_name in POOLED_HEADS:
            m = PooledResNet50StemProjAdapter(h_name).eval()
            out = m(x)
            rows.append({'head': h_name, 'probs_shape': list(out['probs'].shape), 'row_sums': out['probs'].sum(1).tolist()})
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / 'POOLED_SMOKE_TEST_ONLY.json', {'status': 'TEST_ONLY', 'heads': rows})
        print(json.dumps({'status': 'TEST_ONLY', 'heads': len(rows)}))
        return 0
    paths = resolve_runtime_paths()
    rep = audit_pooled_assets(
        args.pooled_checkpoints_dir or paths['pooled_checkpoints'],
        args.pooled_metadata_dir or paths['pooled_metadata'],
        args.output_dir,
    )
    print(json.dumps(rep, indent=2))
    return 0


# %%
if __name__ == '__main__' and _running_in_notebook():
    _paths = resolve_runtime_paths(_REPO)
    _out = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else (_paths['outputs'] / 'mlwa_pooled_heads')
    if SMOKE or MODE == 'smoke':
        _rows = []
        _x = torch.randn(2, 3, 64, 64)
        for _h in POOLED_HEADS:
            _m = PooledResNet50StemProjAdapter(_h).eval()
            _o = _m(_x)
            _rows.append({'head': _h, 'probs_shape': list(_o['probs'].shape), 'row_sums': _o['probs'].sum(1).tolist()})
        _out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_out / 'POOLED_SMOKE_TEST_ONLY.json', {'status': 'TEST_ONLY', 'heads': _rows})
        print(json.dumps({'status': 'TEST_ONLY', 'heads': len(_rows)}))
    else:
        print(audit_pooled_assets(_paths['pooled_checkpoints'], _paths['pooled_metadata'], _out))

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
