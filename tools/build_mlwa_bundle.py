"""Build a pinned, deterministic source archive dist/mlwa_journal_bundle.zip for standalone Colab execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

INCLUDED_PREFIXES = (
    "config.py",
    "paths.py",
    "experiments/",
    "models/",
    "losses/",
    "data/",
    "evaluation/",
    "training/",
    "utils/",
    "docs/mlwa/analysis_plan.md",
    "docs/mlwa/analysis_plan.lock.json",
)


def collect_bundle_files(repo_root: Path = ROOT) -> List[Path]:
    files: List[Path] = []
    for rel in INCLUDED_PREFIXES:
        p = repo_root / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts and child.suffix in (".py", ".json", ".md"):
                    files.append(child)
    return sorted(set(files))


def build_bundle(repo_root: Path = ROOT, out_zip: Optional[Path] = None) -> Path:
    dest = Path(out_zip) if out_zip else (repo_root / "dist" / "mlwa_journal_bundle.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = collect_bundle_files(repo_root)
    manifest = []
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            rel = fp.relative_to(repo_root).as_posix()
            data = fp.read_bytes()
            info = zipfile.ZipInfo(rel, date_time=(2026, 11, 24, 0, 0, 0))
            zf.writestr(info, data)
            manifest.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        m_info = zipfile.ZipInfo("BUNDLE_MANIFEST.json", date_time=(2026, 11, 24, 0, 0, 0))
        zf.writestr(m_info, json.dumps({"files": manifest}, indent=2).encode("utf-8"))

    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(".sha256").write_text(f"{sha}  {dest.name}\n", encoding="utf-8")
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = build_bundle(ROOT, out_zip=args.out)
    print(f"Built bundle: {out} (sha256={out.with_suffix('.sha256').read_text().strip()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
