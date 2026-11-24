"""Generate 100% standalone single-file workflows/mlwa/06..15.py from tools/mlwa_drivers + shared modules."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DRIVERS_DIR = ROOT / "tools" / "mlwa_drivers"
WORKFLOWS_DIR = ROOT / "workflows" / "mlwa"
LOCAL_ROOTS = {"experiments", "models", "losses", "data", "evaluation", "training", "utils", "config", "paths"}


def _module_to_path(module_name: str, repo_root: Path = ROOT) -> Optional[Path]:
    parts = module_name.split(".")
    py_path = repo_root.joinpath(*parts).with_suffix(".py")
    if py_path.is_file():
        return py_path
    pkg_init = repo_root.joinpath(*parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    return None


def _extract_local_imports(source: str) -> List[str]:
    tree = ast.parse(source)
    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in LOCAL_ROOTS:
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in LOCAL_ROOTS:
                    mods.add(alias.name)
    return sorted(mods)


def resolve_transitive_closure(driver_path: Path, repo_root: Path = ROOT) -> List[Tuple[str, Path]]:
    """Return minimal dependency modules in topological (dependencies-first) order."""
    visited: Set[str] = set()
    ordered: List[Tuple[str, Path]] = []

    def _dfs(mod_name: str) -> None:
        if mod_name in visited:
            return
        visited.add(mod_name)
        mod_path = _module_to_path(mod_name, repo_root=repo_root)
        if mod_path is None:
            raise FileNotFoundError(f"Cannot resolve local dependency '{mod_name}' from {driver_path}")
        src = mod_path.read_text(encoding="utf-8-sig")
        for dep in _extract_local_imports(src):
            _dfs(dep)
        ordered.append((mod_name, mod_path))

    driver_src = driver_path.read_text(encoding="utf-8-sig")
    for dep in _extract_local_imports(driver_src):
        _dfs(dep)
    return ordered


def _rewrite_source_standalone(source: str, drop_main_def: bool = False, drop_if_main: bool = True) -> str:
    """Strip __future__ annotations and local imports cleanly via AST spans."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    replacements: List[Tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root_pkg = node.module.split(".")[0]
            if node.module == "__future__" or root_pkg in LOCAL_ROOTS:
                assert node.end_lineno is not None
                indent = lines[node.lineno - 1][: node.col_offset]
                alias_lines = [
                    f"{indent}{a.asname} = {a.name}\n"
                    for a in node.names
                    if a.asname and a.name != "*"
                ]
                replacements.append((node.lineno, node.end_lineno, "".join(alias_lines)))
        elif isinstance(node, ast.Import):
            local_names = [a for a in node.names if a.name.split(".")[0] in LOCAL_ROOTS]
            if local_names:
                assert node.end_lineno is not None
                if len(local_names) != len(node.names):
                    raise ValueError(f"Mixed local & stdlib import on line {node.lineno}; split them cleanly.")
                replacements.append((node.lineno, node.end_lineno, ""))

    for node in tree.body:
        if drop_if_main and isinstance(node, ast.If):
            seg = ast.get_source_segment(source, node.test) or ""
            if "__name__" in seg and "__main__" in seg:
                assert node.end_lineno is not None
                replacements.append((node.lineno, node.end_lineno, ""))
        if drop_main_def and isinstance(node, ast.FunctionDef) and node.name == "main":
            assert node.end_lineno is not None
            replacements.append((node.lineno, node.end_lineno, ""))

    # Deduplicate / merge non-overlapping spans and apply from bottom to top
    replacements.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for start_ln, end_ln, repl_text in replacements:
        repl_lines = repl_text.splitlines(keepends=True) if repl_text else []
        lines[start_ln - 1 : end_ln] = repl_lines

    text = "".join(lines)
    # Collapse excessive blank lines (> 2) caused by stripped import blocks
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip("\n") + "\n"


def _split_driver_cells(driver_source: str) -> List[Tuple[bool, str]]:
    """Split percent-cell script into [(is_markdown, body), ...]."""
    chunks = re.split(r"^# %%(\s*\[markdown\])?[ \t]*\n", driver_source, flags=re.MULTILINE)
    out: List[Tuple[bool, str]] = []
    for marker, body in zip(chunks[1::2], chunks[2::2]):
        out.append((bool(marker), body.strip("\n")))
    return out


def render_standalone_script(driver_path: Path, repo_root: Path = ROOT) -> str:
    """Build complete standalone percent-cell Python script for a single MLWA workflow."""
    driver_src = driver_path.read_text(encoding="utf-8-sig")
    cells = _split_driver_cells(driver_src)
    if len(cells) < 4:
        raise ValueError(f"Expected >= 4 percent cells in driver {driver_path}, found {len(cells)}")

    deps = resolve_transitive_closure(driver_path, repo_root=repo_root)

    parts: List[str] = []
    # Cell 0: Title markdown + explicit standalone notice
    _, title_body = cells[0]
    parts.append(
        "# %% [markdown]\n"
        f"{title_body}\n"
        "# \n"
        "# > **Standalone Single-File Notebook**: All required modules are embedded directly below. "
        "Upload only this `.ipynb` file to Google Colab and run all cells (`Runtime -> Run all`); "
        "no repository clone or zip bundle is required.\n"
    )

    # Cell 1: Placeholder hash (build_notebooks.py updates this atomically)
    parts.append("# %%\nNOTEBOOK_CODE_SHA256 = '0000000000000000000000000000000000000000000000000000000000000000'\n")

    # Cell 2: Colab/local setup & editable runtime parameters (always pure stdlib)
    _, setup_body = cells[2]
    parts.append(f"# %%\n{setup_body}\n")

    # Embedded module cells in topological order
    parts.append(
        "# %% [markdown]\n"
        f"# ## Embedded Standalone Implementation ({len(deps)} self-contained modules)\n"
    )
    for mod_name, mod_path in deps:
        raw_mod = mod_path.read_text(encoding="utf-8-sig")
        clean_mod = _rewrite_source_standalone(
            raw_mod,
            drop_main_def=(mod_name == "experiments.mlwa.plan"),
            drop_if_main=True,
        )
        parts.append(f"# %% [markdown]\n# ### Module `{mod_name}`\n")
        parts.append(f"# %%\n{clean_mod}")

    # Driver workflow, CLI main(), and notebook/CLI dispatch cells
    parts.append("# %% [markdown]\n# ## Experiment Workflow & Entry Points\n")
    for is_md, body in cells[3:]:
        if is_md:
            parts.append(f"# %% [markdown]\n{body}\n")
        else:
            clean_body = _rewrite_source_standalone(body, drop_main_def=False, drop_if_main=False)
            parts.append(f"# %%\n{clean_body}")

    rendered = "\n".join(parts).rstrip() + "\n"
    # Normalize trailing newlines inside lines
    compile(rendered, str(driver_path.name), "exec")
    return rendered


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if any standalone 06-15 workflow is out of date.")
    args = parser.parse_args(argv)

    drivers = sorted(p for p in DRIVERS_DIR.glob("*.py") if p.name[:2] >= "06")
    if len(drivers) != 10:
        raise RuntimeError(f"Expected 10 MLWA journal drivers (06..15) in {DRIVERS_DIR}, found {len(drivers)}")

    for drv in drivers:
        dest = WORKFLOWS_DIR / drv.name
        new_src = render_standalone_script(drv, repo_root=ROOT)
        if args.check:
            if not dest.is_file():
                raise ValueError(f"Missing standalone workflow: {dest}")
            old_src = dest.read_text(encoding="utf-8-sig")
            # Ignore NOTEBOOK_CODE_SHA256 line because build_notebooks.py populates it
            norm_old = re.sub(r"^NOTEBOOK_CODE_SHA256 = '[0-9a-f]{64}'$", "NOTEBOOK_CODE_SHA256 = '<HASH>'", old_src, flags=re.M)
            norm_new = re.sub(r"^NOTEBOOK_CODE_SHA256 = '[0-9a-f]{64}'$", "NOTEBOOK_CODE_SHA256 = '<HASH>'", new_src, flags=re.M)
            if norm_old != norm_new:
                raise ValueError(f"Out-of-date standalone workflow: {dest}")
        else:
            dest.write_text(new_src, encoding="utf-8", newline="\n")
            print(f"Inlined standalone: {dest.name} ({len(new_src.splitlines())} lines)")

    print(f"Checked/generated {len(drivers)} standalone MLWA workflows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
