"""Cross-notebook scientific-governance checks."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src import experiment_config as expcfg


def assert_no_forbidden_score_columns(
    tables: Mapping[str, pd.DataFrame],
    *,
    forbidden_columns: tuple[str, ...] = (
        expcfg.ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS
    ),
) -> pd.DataFrame:
    """Fail if an active protocol table exposes an arbitrary score column."""
    rows = []
    forbidden = {str(column).lower() for column in forbidden_columns}
    for table_name, table in tables.items():
        columns = [] if table is None else list(table.columns)
        matches = sorted(
            column
            for column in columns
            if str(column).lower() in forbidden
        )
        rows.append(
            {
                "table": str(table_name),
                "n_columns": len(columns),
                "forbidden_score_columns": ",".join(map(str, matches)),
                "score_free": not matches,
            }
        )
    audit = pd.DataFrame(rows)
    failures = audit.loc[~audit["score_free"]]
    if not failures.empty:
        raise RuntimeError(
            "Arbitrary weighted-score columns are forbidden in the active "
            f"protocol: {failures.to_dict('records')}"
        )
    return audit
