"""Read-only inventory of supplied result logs. Never runs models or statistics.

This records file identity; it does not certify the scientific correctness of logs.
"""

import argparse
import json
from pathlib import Path

from evaluation.camera_ready import sha256


def inspect_bundle(root: Path):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"External file link in bundle: {path}")
        if path.suffix.lower() in (".pth", ".pt", ".ckpt", ".safetensors"):
            raise ValueError(f"Keep checkpoints outside the result bundle: {path.name}")
        files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    return dict(status="INVENTORIED_ONLY", root=str(root), n_files=len(files), files=files,
                note="Input identity only; published/diagnostic provenance and numerical consistency require review.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(inspect_bundle(args.input_dir), indent=2))


if __name__ == "__main__":
    main()
