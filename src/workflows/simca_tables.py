from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

import src.experiment_config as expcfg
from src.utils import save_parquet


def get_table_contract(table_kind: str) -> Mapping[str, object]:
    """Return one centrally declared table contract.

    The contract itself lives in ``experiment_config``. This module never
    defines a schema, identifier set, filename registry or alias constant.
    """
    key = str(table_kind)
    try:
        return expcfg.PIPELINE_TABLE_CONTRACTS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown table_kind {key!r}. Add it to "
            "experiment_config.PIPELINE_TABLE_CONTRACTS."
        ) from exc


def resolve_table_kind_from_name(name: str) -> str | None:
    """Resolve one persisted filename through the central config registry."""
    filename = str(name)
    exact = expcfg.PIPELINE_TABLE_KIND_BY_FILE_NAME.get(filename)
    if exact is not None:
        return str(exact)

    for suffix, table_kind in expcfg.PIPELINE_TABLE_KIND_BY_FILE_SUFFIX.items():
        if filename.endswith(str(suffix)):
            return str(table_kind)
    return None


def ordered_existing_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    include_remaining: bool = False,
) -> list[str]:
    """Return existing columns in stable contract order."""
    selected: list[str] = []
    seen: set[str] = set()

    for column in map(str, columns):
        if column in df.columns and column not in seen:
            selected.append(column)
            seen.add(column)

    if include_remaining:
        for column in map(str, df.columns):
            if column not in seen:
                selected.append(column)
                seen.add(column)

    return selected


def _coalesce_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.combine_first(right)


def deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate labels by taking the first non-null value.

    This is a serialization safeguard only. It does not create aliases or
    reinterpret identifiers.
    """
    if df is None or len(df.columns) == 0:
        return pd.DataFrame() if df is None else df.copy()

    if not pd.Index(df.columns).duplicated().any():
        return df.copy()

    parts: list[pd.Series] = []
    for column in pd.Index(df.columns).drop_duplicates():
        current = df.loc[:, df.columns == column]
        merged = current.iloc[:, 0]
        for position in range(1, current.shape[1]):
            merged = _coalesce_series(merged, current.iloc[:, position])
        parts.append(merged.rename(column))

    return pd.concat(parts, axis=1)


def resolve_merge_suffix_columns(
    df: pd.DataFrame,
    *,
    prefer: str = "y",
    drop_suffix_columns: bool = True,
) -> pd.DataFrame:
    """Coalesce pandas ``_x`` / ``_y`` merge suffixes.

    The operation only recombines the same base column. It never maps legacy
    identifiers to canonical ones.
    """
    if prefer not in {"x", "y", "base"}:
        raise ValueError("prefer must be one of {'x', 'y', 'base'}.")

    out = deduplicate_column_names(df)
    bases = sorted(
        {
            str(column)[:-2]
            for column in out.columns
            if str(column).endswith("_x") or str(column).endswith("_y")
        }
    )

    for base in bases:
        x_column = f"{base}_x"
        y_column = f"{base}_y"

        if prefer == "y":
            candidates = (y_column, x_column, base)
        elif prefer == "x":
            candidates = (x_column, y_column, base)
        else:
            candidates = (base, y_column, x_column)

        available = [column for column in candidates if column in out.columns]
        if not available:
            continue

        merged = out[available[0]]
        for column in available[1:]:
            merged = _coalesce_series(merged, out[column])
        out[base] = merged

        if drop_suffix_columns:
            out = out.drop(
                columns=[x_column, y_column],
                errors="ignore",
            )

    return out


def canonicalize_simca_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply serialization-only column cleanup.

    Current notebooks 00-04C already emit canonical identifiers. Therefore
    this function deliberately does *not* create ``selected_config_id``,
    ``candidate_id`` aliases, ``m_effective`` aliases, or infer a new
    ``matrix_family``. Scientific naming remains owned by the upstream
    notebook contracts in ``experiment_config``.
    """
    if df is None or len(df.columns) == 0:
        return pd.DataFrame() if df is None else df.copy()
    return resolve_merge_suffix_columns(df)


def validate_table_contract(
    df: pd.DataFrame,
    table_kind: str,
    *,
    require_all_columns: bool = False,
    require_unique_key: bool = True,
) -> pd.DataFrame:
    """Validate one dataframe against its central schema and natural key."""
    contract = get_table_contract(table_kind)
    columns = tuple(map(str, contract.get("columns", ())))
    key_columns = tuple(map(str, contract.get("key_columns", ())))
    unique_key = bool(contract.get("unique_key", True))

    out = df.copy()

    if require_all_columns and columns:
        missing_columns = [column for column in columns if column not in out.columns]
        if missing_columns:
            raise KeyError(
                f"{table_kind} is missing configured columns: {missing_columns}."
            )

    if key_columns and len(out):
        missing_keys = [column for column in key_columns if column not in out.columns]
        if missing_keys:
            raise KeyError(
                f"{table_kind} is missing natural-key columns: {missing_keys}."
            )

        if require_unique_key and unique_key and out.duplicated(list(key_columns)).any():
            examples = (
                out.loc[
                    out.duplicated(list(key_columns), keep=False),
                    list(key_columns),
                ]
                .drop_duplicates()
                .head(10)
                .to_dict("records")
            )
            raise RuntimeError(
                f"{table_kind} duplicates its configured natural key "
                f"{key_columns}: {examples}."
            )

    return out


def drop_all_na_columns(
    df: pd.DataFrame,
    *,
    protected_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Drop all-NA columns while retaining natural-key columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    protected = set(map(str, protected_columns))
    keep = [
        column
        for column in df.columns
        if str(column) in protected or not df[column].isna().all()
    ]
    return df.loc[:, keep].copy()


def compact_simca_table(
    df: pd.DataFrame | None,
    table_kind: str | None = None,
    *,
    include_remaining: bool = False,
    drop_all_na: bool = True,
    validate_keys: bool = True,
    require_all_columns: bool = False,
) -> pd.DataFrame:
    """Compact a table using only the central experiment_config contract.

    If a configured table has ``columns=()`` the upstream schema is treated as
    intentionally dynamic: no columns are dropped, but its declared natural
    key is still protected and validated.
    """
    contract: Mapping[str, object] | None = None
    configured_columns: tuple[str, ...] = ()
    key_columns: tuple[str, ...] = ()

    if table_kind is not None:
        contract = get_table_contract(str(table_kind))
        configured_columns = tuple(map(str, contract.get("columns", ())))
        key_columns = tuple(map(str, contract.get("key_columns", ())))

    if df is None:
        return pd.DataFrame(columns=list(configured_columns))

    if len(df) == 0 and len(df.columns) == 0:
        return pd.DataFrame(columns=list(configured_columns))

    out = canonicalize_simca_columns(df)

    if table_kind is not None and validate_keys:
        out = validate_table_contract(
            out,
            str(table_kind),
            require_all_columns=require_all_columns,
            require_unique_key=True,
        )

    if configured_columns:
        ordered = ordered_existing_columns(
            out,
            configured_columns,
            include_remaining=include_remaining,
        )
        out = out.loc[:, ordered]

    if drop_all_na:
        out = drop_all_na_columns(
            out,
            protected_columns=key_columns,
        )

    return out.reset_index(drop=True)


def compact_simca_table_for_path(
    df: pd.DataFrame | None,
    path_or_name: Any,
    *,
    include_remaining: bool = False,
    drop_all_na: bool = True,
    validate_keys: bool = True,
    require_all_columns: bool = False,
) -> pd.DataFrame:
    """Compact a table using the filename-to-contract registry in config."""
    name = (
        getattr(path_or_name, "name", None)
        or str(path_or_name).replace("\\", "/").split("/")[-1]
    )
    table_kind = resolve_table_kind_from_name(str(name))

    return compact_simca_table(
        df,
        table_kind=table_kind,
        include_remaining=include_remaining,
        drop_all_na=drop_all_na,
        validate_keys=validate_keys,
        require_all_columns=require_all_columns,
    )


def read_simca_table(
    path_or_name: Any,
    *,
    required: bool = False,
    include_remaining: bool = False,
    drop_all_na: bool = True,
    validate_keys: bool = True,
    require_all_columns: bool = False,
) -> pd.DataFrame:
    """Read a parquet table and apply its configured persistence contract."""
    path = Path(path_or_name)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return compact_simca_table_for_path(
            pd.DataFrame(),
            path,
            include_remaining=include_remaining,
            drop_all_na=drop_all_na,
            validate_keys=validate_keys,
            require_all_columns=require_all_columns,
        )

    return compact_simca_table_for_path(
        pd.read_parquet(path),
        path,
        include_remaining=include_remaining,
        drop_all_na=drop_all_na,
        validate_keys=validate_keys,
        require_all_columns=require_all_columns,
    )


def write_simca_table(
    df: pd.DataFrame | None,
    path_or_name: Any,
    *,
    include_remaining: bool = False,
    drop_all_na: bool = True,
    validate_keys: bool = True,
    require_all_columns: bool = False,
) -> Path:
    """Compact and persist a dataframe under its central filename contract."""
    path = Path(path_or_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = compact_simca_table_for_path(
        pd.DataFrame() if df is None else df,
        path,
        include_remaining=include_remaining,
        drop_all_na=drop_all_na,
        validate_keys=validate_keys,
        require_all_columns=require_all_columns,
    )
    return save_parquet(out, path)


def concat_nonempty_tables(parts: Sequence[pd.DataFrame | None]) -> pd.DataFrame:
    """Concatenate non-empty tables with empty-safe behavior."""
    valid_parts = [part for part in parts if part is not None and len(part) > 0]
    return (
        pd.concat(valid_parts, ignore_index=True, sort=False)
        if valid_parts
        else pd.DataFrame()
    )


def iter_dataframe_batches(
    df: pd.DataFrame,
    batch_size: int,
    *,
    batch_prefix: str = "batch",
):
    """Yield stable ``(batch_id, row_start, row_stop, batch_df)`` chunks."""
    if batch_size is None or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")

    size = int(batch_size)
    for batch_index, start in enumerate(range(0, len(df), size), start=1):
        stop = min(start + size, len(df))
        yield (
            f"{batch_prefix}_{batch_index:04d}",
            start,
            stop,
            df.iloc[start:stop].copy(),
        )


def schema_diagnostics(df: pd.DataFrame | None) -> dict[str, Any]:
    """Return lightweight dataframe-schema diagnostics."""
    if df is None:
        return {
            "n_rows": 0,
            "n_columns": 0,
            "n_all_na_columns": 0,
            "n_suffix_columns": 0,
            "all_na_columns": "",
            "suffix_columns": "",
        }

    all_na_columns = [
        str(column)
        for column in df.columns
        if len(df) > 0 and df[column].isna().all()
    ]
    suffix_columns = [
        str(column)
        for column in df.columns
        if str(column).endswith("_x") or str(column).endswith("_y")
    ]

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_all_na_columns": int(len(all_na_columns)),
        "n_suffix_columns": int(len(suffix_columns)),
        "all_na_columns": ",".join(all_na_columns),
        "suffix_columns": ",".join(suffix_columns),
    }


def build_schema_manifest(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact schema manifest for a mapping of named tables."""
    rows: list[dict[str, Any]] = []
    for name, df in tables.items():
        row: dict[str, Any] = {"table_name": str(name)}
        row.update(schema_diagnostics(df))
        rows.append(row)
    return pd.DataFrame(rows)
