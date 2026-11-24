# %% [markdown]
# # 14 · E6 Opt-In Generality (G2 Gate, Pretraining Exposure Audit & External Interfaces)

# %%
NOTEBOOK_CODE_SHA256 = '2131c61aaf882535da785f0cde4e54bf17209c8c58f07bec08d1f292d1f823d5'

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
    ENABLE_E6 = False
    SMOKE = True

# %%
import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.mlwa.constants import COHORT_ORDER, STATUS_SKIPPED_GATE_NOT_MET, resolve_runtime_paths
from experiments.mlwa.data_contract import atomic_write_json
from experiments.mlwa.generality import (
    ExternalDatasetContract,
    ExternalOrdinalDatasetAdapter,
    FrozenEncoderAdapter,
    KNOWN_ENCODER_AUDITS,
    evaluate_g2_gate,
)


class _SmokeFrozenEncoder(FrozenEncoderAdapter):
    encoder_id = 'convnext_tiny_imagenet1k'
    feature_dim = 16
    exposure_audit = KNOWN_ENCODER_AUDITS['convnext_tiny_imagenet1k']

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.normal(size=(len(images), self.feature_dim)).astype(np.float32)


class _SmokeExternalDataset(ExternalOrdinalDatasetAdapter):
    contract = ExternalDatasetContract(
        dataset_id='synthetic_kl_knee_oa_smoke',
        num_classes=5,
        grade_names=('KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4'),
        target_class_index=3,
        primary_endpoint_name='KL_3_recall',
        population_description='Synthetic smoke fixture (TEST_ONLY)',
    )

    def load_splits(self):
        self.contract.validate()
        df = pd.DataFrame({'y_true': [0, 1, 2, 3, 4], 'image_key': [f'k{i}' for i in range(5)], 'patient_key': [f'p{i}' for i in range(5)]})
        return df, df.copy(), df.copy()


def run_e6_smoke(output_dir: Path) -> Dict[str, Any]:
    """Verify G2 gate, exposure audits, and E6 plugin contracts offline without weight downloads (TEST_ONLY)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for audit in KNOWN_ENCODER_AUDITS.values():
        audit.validate()

    rows_int, rows_base = [], []
    for f, c in enumerate(COHORT_ORDER):
        for s in (0, 1, 2):
            rows_base.append({'fold_id': f, 'cohort': c, 'seed_train': s, 'recall_g3': 0.30, 'within1': 0.90, 'qwk': 0.70})
            rows_int.append({'fold_id': f, 'cohort': c, 'seed_train': s, 'recall_g3': 0.36, 'within1': 0.895, 'qwk': 0.695})

    g2_open = evaluate_g2_gate(pd.DataFrame(rows_int), pd.DataFrame(rows_base), e4_best_g3_mean=0.34)
    enc = _SmokeFrozenEncoder()
    feats = enc.extract_features(np.zeros((4, 3, 8, 8), dtype=np.float32))
    ext = _SmokeExternalDataset()
    tr, va, tg = ext.load_splits()

    status = {
        'status': 'COMPLETE_SMOKE_TEST_ONLY',
        'g2_open_check': g2_open,
        'known_encoder_audits': {k: asdict(v) for k, v in KNOWN_ENCODER_AUDITS.items()},
        'smoke_feature_shape': list(feats.shape),
        'external_contract_id': ext.contract.dataset_id,
    }
    atomic_write_json(out / 'status.json', status)
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    paths = resolve_runtime_paths()
    parser = argparse.ArgumentParser(description='14: Opt-in E6 generality (G2 gate & exposure audits; offline safe).')
    parser.add_argument('--enable-e6', action='store_true', help='Explicit opt-in after verifying G2 gate is open.')
    parser.add_argument('--intervention-csv', type=Path, default=None)
    parser.add_argument('--baseline-csv', type=Path, default=None)
    parser.add_argument('--e4-best-g3', type=float, default=None)
    parser.add_argument('--output-dir', type=Path, default=paths['outputs'] / 'mlwa_e6_generality')
    parser.add_argument('--smoke', action='store_true', help='Run offline contract & G2 gate smoke verification (TEST_ONLY).')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        res = run_e6_smoke(args.output_dir)
        print(json.dumps({'status': res['status'], 'g2_open': res['g2_open_check']['gate_open']}, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audits = {k: asdict(v) for k, v in KNOWN_ENCODER_AUDITS.items()}
    if not args.enable_e6 or args.intervention_csv is None or args.baseline_csv is None:
        status = {
            'status': 'OPT_IN_OR_GATE_INPUTS_NOT_SUPPLIED',
            'note': 'E6 requires --enable-e6 and verified post-lock intervention/baseline CSVs to check G2.',
            'known_encoder_audits': audits,
        }
        atomic_write_json(args.output_dir / 'status.json', status)
        print(json.dumps({'status': status['status']}, indent=2))
        return 0

    gate = evaluate_g2_gate(
        pd.read_csv(args.intervention_csv),
        pd.read_csv(args.baseline_csv),
        e4_best_g3_mean=args.e4_best_g3,
    )
    gate['known_encoder_audits'] = audits
    atomic_write_json(args.output_dir / 'g2_gate_status.json', gate)
    print(json.dumps({'status': gate['status'], 'gate_open': gate['gate_open']}, indent=2))
    return 0


# %%
if not _running_in_notebook() and '__file__' in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


if __name__ == '__main__' and _running_in_notebook():
    _out = resolve_runtime_paths()['outputs'] / 'mlwa_e6_generality_nb'
    print(run_e6_smoke(_out) if globals().get('SMOKE', True) else {'status': 'OPT_IN_OR_GATE_INPUTS_NOT_SUPPLIED'})

# %%
if __name__ == '__main__' and not _running_in_notebook():
    raise SystemExit(main())
