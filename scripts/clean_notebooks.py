from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean_notebook(path: Path, strip_outputs: bool = True, strip_exec_count: bool = True) -> tuple[int, int]:
    """Return (cleared_outputs_count, cleared_execution_count_cells)."""
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    outputs_cleared = 0
    exec_cleared = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if strip_outputs and cell.get("outputs"):
            outputs_cleared += 1
            cell["outputs"] = []

        if strip_exec_count and cell.get("execution_count") is not None:
            exec_cleared += 1
            cell["execution_count"] = None

    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    return outputs_cleared, exec_cleared


def iter_notebooks(root: Path, include_archive: bool = False):
    for nb_path in sorted(root.glob("*.ipynb")):
        yield nb_path

    if include_archive:
        archive = root / "_archive"
        if archive.exists():
            for nb_path in sorted(archive.glob("*.ipynb")):
                yield nb_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean notebook outputs and execution counts.")
    parser.add_argument("--root", default="notebooks", help="Notebook directory root")
    parser.add_argument("--include-archive", action="store_true", help="Also clean notebooks/_archive")
    parser.add_argument("--keep-outputs", action="store_true", help="Do not clear cell outputs")
    parser.add_argument("--keep-exec-count", action="store_true", help="Do not clear execution_count")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Notebook root not found: {root}")

    total_outputs = 0
    total_exec = 0
    total_files = 0

    for nb_path in iter_notebooks(root, include_archive=args.include_archive):
        outputs, execs = clean_notebook(
            nb_path,
            strip_outputs=not args.keep_outputs,
            strip_exec_count=not args.keep_exec_count,
        )
        total_outputs += outputs
        total_exec += execs
        total_files += 1
        print(f"[CLEAN] {nb_path} | outputs_cells={outputs} exec_cells={execs}")

    print("\nDone")
    print(f"Notebooks: {total_files}")
    print(f"Cells with outputs cleared: {total_outputs}")
    print(f"Cells with execution_count cleared: {total_exec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
