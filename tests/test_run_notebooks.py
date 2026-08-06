from pathlib import Path

import pytest

from scripts.run_notebooks import (
    NOTEBOOK_BY_ALIAS,
    build_parser,
    main,
    resolve_notebook_selectors,
)


def test_expected_aliases_are_registered():
    assert [alias for alias in ("02", "03", "03B", "03C") if alias not in NOTEBOOK_BY_ALIAS] == []
    assert NOTEBOOK_BY_ALIAS["03C"] == "03C_projection_spatial_calibration.ipynb"


def test_selectors_preserve_requested_order_and_accept_filenames():
    resolved = resolve_notebook_selectors(
        ["02", "03_pca_exploration_selection.ipynb", "3b", "03c"]
    )
    assert [alias for alias, _ in resolved] == ["02", "03", "03B", "03C"]
    assert all(isinstance(path, Path) and path.is_file() for _, path in resolved)


def test_duplicate_and_unknown_selectors_are_rejected():
    with pytest.raises(ValueError, match="more than once"):
        resolve_notebook_selectors(["03B", "3b"])
    with pytest.raises(ValueError, match="Unknown notebook selector"):
        resolve_notebook_selectors(["99"])


def test_cli_dry_run_does_not_start_a_kernel(capsys):
    assert main(["--dry-run", "02", "03", "03B", "03C"]) == 0
    output = capsys.readouterr().out
    assert "02 -> notebooks" in output
    assert "03C -> notebooks" in output
    assert "Kernel: hsi-nuts" in output


def test_parser_defaults_to_unlimited_cell_timeout():
    args = build_parser().parse_args(["02"])
    assert args.timeout == 0
    assert args.kernel == "hsi-nuts"
    assert not args.continue_on_error
