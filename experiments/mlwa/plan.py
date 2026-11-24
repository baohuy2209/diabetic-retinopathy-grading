"""Draft/Frozen verification and reproducible locking for docs/mlwa/analysis_plan.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _parse_plan_metadata(text: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in text.splitlines()[:25]:
        m = re.match(r"^\*\*([A-Za-z0-9_]+)\*\*:\s*(.+?)\s*$", line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip("` ")
    return meta


def compute_plan_sha256(plan_path: Path) -> str:
    data = Path(plan_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _git_head(repo_root: Path) -> Optional[str]:
    if not (repo_root / ".git").exists():
        return None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def freeze_analysis_plan(
    plan_path: Path,
    lock_path: Path,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote status to FROZEN cleanly and write an immutable lock JSON."""
    plan_path = Path(plan_path).resolve()
    lock_path = Path(lock_path).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Analysis plan not found: {plan_path}")

    text = plan_path.read_text(encoding="utf-8")
    meta = _parse_plan_metadata(text)
    if meta.get("status", "DRAFT").upper() == "DRAFT":
        text = re.sub(
            r"^\*\*status\*\*:\s*DRAFT\b",
            "**status**: FROZEN",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        plan_path.write_text(text, encoding="utf-8", newline="\n")
        meta = _parse_plan_metadata(text)

    if meta.get("status", "").upper() != "FROZEN":
        raise ValueError(f"Cannot lock plan with status='{meta.get('status')}'; expected DRAFT or FROZEN.")

    plan_sha = compute_plan_sha256(plan_path)
    root = Path(repo_root).resolve() if repo_root else plan_path.parents[2]
    lock_payload: Dict[str, Any] = {
        "schema": "mlwa_analysis_plan_lock_v1",
        "status": "FROZEN",
        "protocol_id": meta.get("protocol_id", "mlwa_journal_v1"),
        "study_id": meta.get("study_id", "mlwa_dr_ordinal_extension"),
        "created_date": meta.get("created_date"),
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_head(root),
        "plan_relpath": str(plan_path),
        "plan_sha256": plan_sha,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock_payload


def verify_frozen_plan(
    plan_path: Optional[Path],
    lock_path: Optional[Path],
    allow_test_only: bool = False,
) -> Dict[str, Any]:
    """Refuse production runs unless plan status is FROZEN and lock hash matches."""
    if allow_test_only:
        if plan_path is not None and Path(plan_path).is_file():
            return {
                "status": "TEST_ONLY",
                "plan_sha256": compute_plan_sha256(Path(plan_path)),
                "protocol_id": "TEST_ONLY_SYNTHETIC_PLAN",
            }
        return {"status": "TEST_ONLY", "plan_sha256": "TEST_ONLY", "protocol_id": "TEST_ONLY_SYNTHETIC_PLAN"}

    if plan_path is None or lock_path is None:
        raise ValueError("Production execution requires both --plan-path and --plan-lock-path.")
    plan_p = Path(plan_path)
    lock_p = Path(lock_path)
    if not plan_p.is_file():
        raise FileNotFoundError(f"Analysis plan not found: {plan_p}")
    text = plan_p.read_text(encoding="utf-8")
    meta = _parse_plan_metadata(text)
    if meta.get("status", "").upper() != "FROZEN":
        raise RuntimeError(
            f"Refusing production run against unfrozen plan ({plan_p}, status={meta.get('status')}). "
            "Run `python -m experiments.mlwa.plan freeze docs/mlwa/analysis_plan.md --out docs/mlwa/analysis_plan.lock.json` first."
        )
    if not lock_p.is_file():
        raise RuntimeError(f"Refusing production run without plan lock file: {lock_p}")
    lock = json.loads(lock_p.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN":
        raise RuntimeError(f"Plan lock status is not FROZEN: {lock.get('status')}")
    actual_sha = compute_plan_sha256(plan_p)
    if lock.get("plan_sha256") != actual_sha:
        raise RuntimeError(
            f"Analysis plan modified after freeze! lock={lock.get('plan_sha256')}, actual={actual_sha}"
        )
    return lock


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify MLWA journal analysis_plan.md")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze", help="Freeze draft plan and write lock JSON.")
    p_freeze.add_argument("plan", type=Path)
    p_freeze.add_argument("--out", type=Path, required=True)

    p_verify = sub.add_parser("verify", help="Verify frozen plan matches lock JSON.")
    p_verify.add_argument("plan", type=Path)
    p_verify.add_argument("--lock", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        lock = freeze_analysis_plan(args.plan, args.out)
        print(json.dumps(lock, indent=2))
        return 0
    lock = verify_frozen_plan(args.plan, args.lock, allow_test_only=False)
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
