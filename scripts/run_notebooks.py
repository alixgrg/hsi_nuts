"""Execute selected project notebooks sequentially and atomically.

Example
-------
conda run -n hsi-nuts python scripts/run_notebooks.py 02 03 03B 03C
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from jupyter_client.kernelspec import KernelSpecManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DEFAULT_KERNEL = "hsi-nuts"

NOTEBOOK_BY_ALIAS = {
    "00": "00_building_database.ipynb",
    "01": "01_database_quality_check.ipynb",
    "01B": "01B_spatial_ground_truth.ipynb",
    "02": "02_matrices_preprocessing.ipynb",
    "03": "03_pca_exploration_selection.ipynb",
    "03B": "03B_internal_calibration.ipynb",
    "03C": "03C_projection_spatial_calibration.ipynb",
    "04A": "04A_simca_grid_search.ipynb",
    "04B": "04B_simca_optuna_search.ipynb",
    "04C": "04C_simca_concat_refit.ipynb",
    "05": "05_simca_validation_robustness.ipynb",
    "06A": "06A_simca_pure_test.ipynb",
    "06B": "06B_simca_final_selection.ipynb",
    "07": "07_simca_mixture_application.ipynb",
}


def _selector_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for alias, filename in NOTEBOOK_BY_ALIAS.items():
        stem = Path(filename).stem
        for token in (alias, alias.lstrip("0"), filename, stem):
            if token:
                lookup[token.casefold()] = alias
    return lookup


SELECTOR_LOOKUP = _selector_lookup()


def resolve_notebook_selectors(selectors: Sequence[str]) -> list[tuple[str, Path]]:
    """Resolve aliases or canonical filenames while preserving input order."""
    resolved: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for selector in selectors:
        token = str(selector).strip()
        alias = SELECTOR_LOOKUP.get(token.casefold())
        if alias is None:
            choices = ", ".join(NOTEBOOK_BY_ALIAS)
            raise ValueError(
                f"Unknown notebook selector {selector!r}. Available aliases: {choices}."
            )
        if alias in seen:
            raise ValueError(f"Notebook {alias} was requested more than once.")
        path = NOTEBOOK_DIR / NOTEBOOK_BY_ALIAS[alias]
        if not path.is_file():
            raise FileNotFoundError(f"Notebook registered as {alias} is missing: {path}")
        resolved.append((alias, path))
        seen.add(alias)
    return resolved


def available_kernels() -> set[str]:
    return set(KernelSpecManager().find_kernel_specs())


def _atomic_write_notebook(notebook, path: Path) -> None:
    """Replace a notebook only after a complete temporary write."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ipynb",
            prefix=f".{path.stem}.",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
        ) as stream:
            temporary_path = Path(stream.name)
        nbformat.write(notebook, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def execute_notebook(
    path: Path,
    *,
    kernel_name: str,
    timeout: int | None,
    save: bool,
) -> float:
    """Execute one notebook from the project root and optionally save it."""
    notebook = nbformat.read(path, as_version=4)
    code_cell_ordinals = {
        index: ordinal
        for ordinal, index in enumerate(
            (
                index
                for index, cell in enumerate(notebook.cells)
                if cell.cell_type == "code" and cell.source.strip()
            ),
            start=1,
        )
    }

    def report_cell_start(*, cell, cell_index: int) -> None:
        ordinal = code_cell_ordinals.get(cell_index)
        if ordinal is not None:
            print(
                f"  [CELL {ordinal}/{len(code_cell_ordinals)}] starting",
                flush=True,
            )

    client = NotebookClient(
        notebook,
        kernel_name=str(kernel_name),
        timeout=timeout,
        allow_errors=False,
        record_timing=True,
        on_cell_execute=report_cell_start,
    )
    started = perf_counter()
    executed = client.execute(cwd=str(PROJECT_ROOT))
    duration = perf_counter() - started
    if save:
        _atomic_write_notebook(executed, path)
    return float(duration)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute project notebooks sequentially in the supplied order. "
            "A failure stops dependent notebooks by default."
        ),
        epilog=(
            "Example: python scripts/run_notebooks.py 02 03 03B 03C"
        ),
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        metavar="NOTEBOOK",
        help="Alias such as 02, 03, 03B or 03C; canonical filename is also accepted.",
    )
    parser.add_argument(
        "--kernel",
        default=DEFAULT_KERNEL,
        help=f"Jupyter kernel name (default: {DEFAULT_KERNEL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Per-cell timeout; 0 disables the timeout (default: 0).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Execute for validation without replacing notebook files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later notebooks after an error; unsafe for dependencies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the execution plan without starting a kernel.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered aliases and exit.",
    )
    return parser


def _print_catalog() -> None:
    for alias, filename in NOTEBOOK_BY_ALIAS.items():
        print(f"{alias:>3}  {filename}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        _print_catalog()
        return 0
    if not args.notebooks:
        parser.error("provide at least one notebook alias or use --list")
    if args.timeout < 0:
        parser.error("--timeout must be greater than or equal to zero")
    try:
        plan = resolve_notebook_selectors(args.notebooks)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Execution plan:")
    for index, (alias, path) in enumerate(plan, start=1):
        print(f"  {index}. {alias} -> {path.relative_to(PROJECT_ROOT)}")
    print(f"Kernel: {args.kernel}")
    print("Save: " + ("no" if args.no_save else "yes, atomically in place"))
    if args.dry_run:
        return 0

    kernels = available_kernels()
    if args.kernel not in kernels:
        print(
            f"ERROR: Jupyter kernel {args.kernel!r} is unavailable. "
            f"Available kernels: {sorted(kernels)}",
            file=sys.stderr,
        )
        return 2

    timeout = None if args.timeout == 0 else int(args.timeout)
    failures: list[tuple[str, str]] = []
    run_started = perf_counter()
    for index, (alias, path) in enumerate(plan, start=1):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(
            f"\n[START {index}/{len(plan)}] {alias} | {timestamp}",
            flush=True,
        )
        try:
            duration = execute_notebook(
                path,
                kernel_name=args.kernel,
                timeout=timeout,
                save=not args.no_save,
            )
        except CellExecutionError as exc:
            message = str(exc)
            failures.append((alias, message))
            print(f"[FAILED] {alias}\n{message}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
        except Exception as exc:  # kernel, I/O or serialization failure
            message = f"{type(exc).__name__}: {exc}"
            failures.append((alias, message))
            print(f"[FAILED] {alias} | {message}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
        else:
            print(f"[DONE] {alias} | {duration:.1f} s", flush=True)

    total_duration = perf_counter() - run_started
    if failures:
        print("\nRun failed:", file=sys.stderr)
        for alias, message in failures:
            first_line = message.splitlines()[0] if message else "unknown error"
            print(f"  - {alias}: {first_line}", file=sys.stderr)
        print(f"Elapsed: {total_duration:.1f} s", file=sys.stderr)
        return 1
    print(f"\nAll notebooks completed in {total_duration:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
