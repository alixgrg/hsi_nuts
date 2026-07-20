from __future__ import annotations

from pathlib import Path


def test_visualization_modules_do_not_import_scientific_selection_logic():
    visualization_dir = Path("src/visualization")
    forbidden_tokens = [
        "src.workflows.pca_selection",
        "add_pca_selection_score",
        "add_pca_selection_scores",
        "select_pca_preprocessing_shortlist",
    ]

    offenders = {}
    for path in visualization_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        found = [token for token in forbidden_tokens if token in text]
        if found:
            offenders[str(path)] = found

    assert offenders == {}
