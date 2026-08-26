from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

import src.experiment_config as expcfg
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.utils import parse_preprocessing_steps
from src.matrices.matrix_registry import matrix_family_from_method

_CANDIDATE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "selected_rule_name": ("rule_variant", "rule_for_refit", "rule"),
    "rule_variant": ("selected_rule_name", "rule_for_refit", "rule"),
    "rule": ("rule_base", "rule_variant", "selected_rule_name"),
    "object_threshold": (
        "best_object_threshold",
        "object_threshold_median",
        "object_threshold_selected",
    ),
    "n_components": ("best_n_components", "n_components_median", "A"),
    "alpha": ("alpha_median", "alpha_selected"),
    "m": ("m_median", "m_effective"),
    "balanced_pixel_strategy": ("balanced_pixel_strategy_effective",),
}


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (Mapping, list, tuple, set, np.ndarray, pd.Index, pd.Series)):
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _first_non_missing(row: Mapping[str, Any], names: Sequence[str], default: Any = pd.NA) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if not _is_missing_scalar(value):
            return value
    return default


def _fill_from_aliases(df: pd.DataFrame, canonical: str, aliases: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if canonical not in out.columns:
        out[canonical] = pd.NA

    for alias in aliases:
        if alias not in out.columns:
            continue
        missing = out[canonical].map(_is_missing_scalar)
        out.loc[missing, canonical] = out.loc[missing, alias]
    return out


def _normalise_decision_mode(value: Any) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"2", "2way", "2_way", "two_way", "binary"}:
        return "2way"
    if token in {"3", "3way", "3_way", "three_way", "ternary"}:
        return "3way"
    return str(value)


# def infer_matrix_family(matrix_method: Any) -> str:
#     """Infer the canonical matrix family for a matrix method or matrix id."""
#     token = str(matrix_method)
#     if token in expcfg.SIMCA_MATRIX_METHOD_FAMILY:
#         return expcfg.SIMCA_MATRIX_METHOD_FAMILY[token]
#     if token.startswith("balanced_pixel_"):
#         return "pixel_matrix"
#     if token.startswith("object_"):
#         return "object_matrix"
#     return "unknown_matrix_family"


# def selection_track_from_parts(matrix_family: Any, decision_mode: Any) -> str:
#     """Return the canonical SIMCA selection track name."""
#     matrix_family = str(matrix_family)
#     decision_mode = _normalise_decision_mode(decision_mode)
#     track = f"{matrix_family}_{decision_mode}"
#     if track not in expcfg.SIMCA_SELECTION_TRACKS:
#         valid = ", ".join(expcfg.SIMCA_SELECTION_TRACKS)
#         raise ValueError(f"Unknown SIMCA selection track {track!r}. Valid tracks: {valid}.")
#     return track


def validate_simca_table_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    table_name: str = "SIMCA table",
) -> pd.DataFrame:
    """Raise a clear error when a SIMCA result table misses required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required column(s): {missing}")
    return df


# def validate_simca_selection_tracks(df: pd.DataFrame) -> pd.DataFrame:
#     """Validate that selection_track matches matrix_family and decision_mode."""
#     validate_simca_table_columns(
#         df,
#         ("selection_track", "matrix_family", "decision_mode"),
#         table_name="SIMCA selection/evaluation table",
#     )
#     invalid_rows = []
#     for idx, row in df.iterrows():
#         try:
#             expected = selection_track_from_parts(row["matrix_family"], row["decision_mode"])
#         except ValueError:
#             invalid_rows.append(idx)
#             continue
#         if str(row["selection_track"]) != expected:
#             invalid_rows.append(idx)

#     if invalid_rows:
#         preview = list(invalid_rows[:10])
#         raise ValueError(
#             "selection_track must match matrix_family + decision_mode. "
#             f"Invalid row index preview: {preview}"
#         )
#     return df


def validate_simca_candidate_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical SIMCA candidate-config table contract."""
    return validate_simca_table_columns(
        df,
        expcfg.SIMCA_CANDIDATE_CONFIG_REQUIRED_COLUMNS,
        table_name="SIMCA candidate config table",
    )


# def validate_simca_evaluation_contract(df: pd.DataFrame) -> pd.DataFrame:
#     """Validate the canonical SIMCA model-evaluation table contract."""
#     validate_simca_table_columns(
#         df,
#         expcfg.SIMCA_CANDIDATE_EVALUATION_REQUIRED_COLUMNS,
#         table_name="SIMCA candidate evaluation table",
#     )
#     return validate_simca_selection_tracks(df)


def normalize_simca_candidate_columns(
    df: pd.DataFrame,
    candidate_source: str | None = None,
    source_col: str = "candidate_source",
) -> pd.DataFrame:
    """Normalize common SIMCA candidate metadata columns before merging tables."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    for canonical, aliases in _CANDIDATE_COLUMN_ALIASES.items():
        out = _fill_from_aliases(out, canonical=canonical, aliases=aliases)

    if "matrix_family" not in out.columns:
        if "matrix_method" in out.columns:
            out["matrix_family"] = out["matrix_method"].map(matrix_family_from_method)
        elif "training_matrix_id" in out.columns:
            out["matrix_family"] = out["training_matrix_id"].map(matrix_family_from_method)

    if "training_matrix_id" not in out.columns and "matrix_method" in out.columns:
        out["training_matrix_id"] = out["matrix_method"].astype(str)

    if "preprocessing_steps" not in out.columns and "preprocessing" in out.columns:
        out["preprocessing_steps"] = out["preprocessing"]
    if "preprocessing_steps" in out.columns:
        out["preprocessing_steps"] = out["preprocessing_steps"].map(
            lambda value: "+".join(parse_preprocessing_steps(value))
        )

    if "decision_mode" in out.columns:
        out["decision_mode"] = out["decision_mode"].map(_normalise_decision_mode)

    if source_col not in out.columns:
        out[source_col] = candidate_source if candidate_source is not None else "unknown"
    elif candidate_source is not None:
        out[source_col] = out[source_col].where(
            ~out[source_col].map(_is_missing_scalar),
            candidate_source,
        )

    return out


# def add_selection_track(
#     df: pd.DataFrame,
#     matrix_family_col: str = "matrix_family",
#     decision_mode_col: str = "decision_mode",
#     track_col: str = "selection_track",
#     overwrite: bool = True,
# ) -> pd.DataFrame:
#     """Add the canonical 4-track SIMCA selection label to a table."""
#     out = normalize_simca_candidate_columns(df)
#     validate_simca_table_columns(
#         out,
#         (matrix_family_col, decision_mode_col),
#         table_name="SIMCA selection/evaluation table",
#     )

#     if track_col in out.columns and not overwrite:
#         return validate_simca_selection_tracks(out)

#     out[track_col] = [
#         selection_track_from_parts(row[matrix_family_col], row[decision_mode_col])
#         for _, row in out.iterrows()
#     ]
#     return out


def candidate_identity_payload(
    row: Mapping[str, Any] | pd.Series,
    id_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the canonical JSON payload used to hash one SIMCA candidate."""
    if id_columns is None:
        id_columns = expcfg.SIMCA_CANDIDATE_ID_COLUMNS

    row_dict = dict(row)
    payload: dict[str, Any] = {}
    for col in id_columns:
        if col not in row_dict:
            continue
        value = row_dict[col]
        if _is_missing_scalar(value):
            continue
        payload[col] = _json_ready_value(value, column=col)
    return payload


def _json_ready_value(value: Any, column: str | None = None) -> Any:
    if _is_missing_scalar(value):
        return None

    if column == "preprocessing_steps":
        return parse_preprocessing_steps(value)

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, Mapping):
        return {
            str(key): _json_ready_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }

    if isinstance(value, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        return [_json_ready_value(item) for item in list(value)]

    if isinstance(value, set):
        return sorted(_json_ready_value(item) for item in value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (int, bool, str)):
        return value.strip() if isinstance(value, str) else value

    return str(value)


def simca_candidate_key(
    row: Mapping[str, Any] | pd.Series,
    id_columns: Sequence[str] | None = None,
    prefix: str = "simca",
) -> str:
    """Build a stable candidate id from model-defining columns only."""
    payload = candidate_identity_payload(row, id_columns=id_columns)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def add_simca_candidate_ids(
    df: pd.DataFrame,
    id_columns: Sequence[str] | None = None,
    id_col: str = "candidate_id",
    prefix: str = "simca",
) -> pd.DataFrame:
    """Add stable candidate ids to a SIMCA candidate table."""
    out = normalize_simca_candidate_columns(df)
    out[id_col] = [
        simca_candidate_key(row, id_columns=id_columns, prefix=prefix)
        for _, row in out.iterrows()
    ]
    return out


def _split_sources(value: Any) -> list[str]:
    if _is_missing_scalar(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index, pd.Series)):
        raw = list(value)
    else:
        raw = str(value).replace(";", ",").split(",")
    return [str(item).strip() for item in raw if str(item).strip()]


def deduplicate_simca_candidates(
    df: pd.DataFrame,
    id_columns: Sequence[str] | None = None,
    id_col: str = "candidate_id",
    source_col: str = "candidate_source",
    sources_col: str = "candidate_sources",
) -> pd.DataFrame:
    """Deduplicate candidate configs while preserving all contributing sources."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = add_simca_candidate_ids(df, id_columns=id_columns, id_col=id_col)
    if source_col not in out.columns:
        out[source_col] = "unknown"
    if sources_col not in out.columns:
        out[sources_col] = pd.NA

    summary_rows = []
    for candidate_id, group in out.groupby(id_col, sort=False, dropna=False):
        sources: list[str] = []
        for _, row in group.iterrows():
            sources.extend(_split_sources(row.get(source_col, pd.NA)))
            sources.extend(_split_sources(row.get(sources_col, pd.NA)))
        unique_sources = sorted(set(sources)) or ["unknown"]
        summary_rows.append(
            {
                id_col: candidate_id,
                sources_col: ",".join(unique_sources),
                "n_candidate_sources": int(len(unique_sources)),
                "n_duplicate_rows": int(len(group)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    first_rows = out.drop_duplicates(id_col, keep="first").drop(
        columns=[
            sources_col,
            "n_candidate_sources",
            "n_duplicate_rows",
        ],
        errors="ignore",
    )
    return first_rows.merge(summary_df, on=id_col, how="left").reset_index(drop=True)


def deduplicate_simca_refit_configs(
    df: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
    id_col: str = "candidate_id",
    source_col: str = "candidate_source",
    sources_col: str = "candidate_sources",
    refit_id_col: str = "refit_config_id",
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collapse candidate rows that would trigger the same 04C refit configuration.

    This is intentionally separate from canonical candidate-id deduplication:
    grid-search and Optuna rows can have different candidate ids while still
    representing the same refit/projection configuration. The first row in the
    input order is kept, so callers should sort by their preferred score before
    using this function.
    """
    if df is None or len(df) == 0:
        empty = pd.DataFrame() if df is None else df.copy()
        return empty, empty.copy(), pd.DataFrame()

    out = normalize_simca_candidate_columns(df)
    if key_columns is None:
        key_columns = expcfg.SIMCA_REFIT_CONFIG_DEDUP_COLUMNS

    key_columns = tuple(key_columns)
    missing = [col for col in key_columns if col not in out.columns]
    if missing and strict:
        raise ValueError(f"Missing refit deduplication key column(s): {missing}")

    key_columns = tuple(col for col in key_columns if col in out.columns)
    if not key_columns:
        raise ValueError("No refit deduplication key columns are available.")

    out[refit_id_col] = [
        simca_candidate_key(row, id_columns=key_columns, prefix="refitcfg")
        for _, row in out.iterrows()
    ]
    out["refit_config_duplicate_rank"] = (
        out.groupby(refit_id_col, sort=False, dropna=False).cumcount() + 1
    )

    summary_rows = []
    for refit_config_id, group in out.groupby(refit_id_col, sort=False, dropna=False):
        sources: list[str] = []
        if source_col in group.columns:
            for value in group[source_col]:
                sources.extend(_split_sources(value))
        if sources_col in group.columns:
            for value in group[sources_col]:
                sources.extend(_split_sources(value))

        ids = (
            [str(value) for value in group[id_col].tolist()]
            if id_col in group.columns
            else []
        )

        row = {col: group.iloc[0][col] for col in key_columns}
        row[refit_id_col] = refit_config_id
        row["n_refit_config_candidates"] = int(len(group))
        row["refit_config_candidate_ids"] = ",".join(ids)
        row["refit_config_candidate_sources"] = ",".join(sorted(set(sources))) or "unknown"
        row["kept_candidate_id"] = ids[0] if ids else pd.NA
        row["dropped_candidate_ids"] = ",".join(ids[1:])
        row["n_candidates"] = int(len(group))
        row["candidate_sources"] = row["refit_config_candidate_sources"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    merge_cols = [
        refit_id_col,
        "n_refit_config_candidates",
        "refit_config_candidate_ids",
        "refit_config_candidate_sources",
    ]
    out = out.merge(
        summary_df[merge_cols],
        on=refit_id_col,
        how="left",
        validate="many_to_one",
    )
    if sources_col in out.columns:
        out[sources_col] = out["refit_config_candidate_sources"]

    kept = (
        out[out["refit_config_duplicate_rank"].eq(1)]
        .reset_index(drop=True)
    )
    dropped = (
        out[out["refit_config_duplicate_rank"].gt(1)]
        .reset_index(drop=True)
    )

    return kept, dropped, summary_df.reset_index(drop=True)


def deduplicate_metric_equivalent_simca_candidates(
    df: pd.DataFrame,
    metric_columns: Sequence[str] | None = None,
    parameter_groups: Mapping[str, Sequence[str]] | None = None,
    protected_columns: Sequence[str] | None = None,
    signatures_df: pd.DataFrame | None = None,
    signature_columns: Sequence[str] | None = None,
    allowed_varied_columns: Sequence[str] | None = None,
    max_varied_columns: int = 1,
    id_col: str = "candidate_id",
    group_id_col: str = "metric_equivalence_group_id",
    metric_round_decimals: int | None = 12,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collapse candidates with identical metrics and, when provided, signatures.

    The first row in the input order is kept. Callers should therefore sort the
    table by their preferred tie-break policy before using this function.
    """
    if df is None or len(df) == 0:
        empty = pd.DataFrame() if df is None else df.copy()
        return empty, empty.copy(), pd.DataFrame()

    out = normalize_simca_candidate_columns(df).reset_index(drop=True)
    out["metric_equivalence_original_order"] = np.arange(len(out), dtype=int)

    signature_columns = tuple(signature_columns or ())
    if signatures_df is not None:
        if id_col not in signatures_df.columns:
            raise ValueError(f"Missing {id_col!r} in signatures_df.")
        missing_signatures = [
            col for col in signature_columns
            if col not in signatures_df.columns
        ]
        if missing_signatures:
            raise ValueError(
                f"Missing output-signature column(s): {missing_signatures}"
            )
        out = out.merge(
            signatures_df[[id_col, *signature_columns]].drop_duplicates(id_col),
            on=id_col,
            how="left",
            validate="one_to_one",
        )

    metric_columns = tuple(metric_columns or expcfg.SIMCA_METRIC_EQUIVALENCE_METRIC_COLUMNS)
    protected_columns = tuple(
        protected_columns or expcfg.SIMCA_METRIC_EQUIVALENCE_PROTECTED_COLUMNS
    )
    parameter_groups = (
        {}
        if allowed_varied_columns is not None and parameter_groups is None
        else dict(
            parameter_groups or expcfg.SIMCA_METRIC_EQUIVALENCE_PARAMETER_GROUPS
        )
    )

    required = set(metric_columns) | set(protected_columns)
    if allowed_varied_columns is not None:
        required.update(allowed_varied_columns)
        required.update(signature_columns)
    else:
        for cols in parameter_groups.values():
            required.update(cols)
    missing = [col for col in required if col not in out.columns]
    if missing and strict:
        raise ValueError(f"Missing metric-equivalence column(s): {missing}")

    metric_columns = tuple(col for col in metric_columns if col in out.columns)
    protected_columns = tuple(col for col in protected_columns if col in out.columns)
    parameter_groups = {
        str(name): tuple(col for col in cols if col in out.columns)
        for name, cols in parameter_groups.items()
    }
    parameter_groups = {
        name: cols
        for name, cols in parameter_groups.items()
        if cols
    }
    if not metric_columns:
        raise ValueError("No metric-equivalence metric columns are available.")
    if allowed_varied_columns is None and not parameter_groups:
        raise ValueError("No metric-equivalence parameter groups are available.")

    metric_key_cols = []
    for col in metric_columns:
        key_col = f"__metric_equivalence_metric__{col}"
        if pd.api.types.is_numeric_dtype(out[col]):
            values = pd.to_numeric(out[col], errors="coerce")
            if metric_round_decimals is not None:
                values = values.round(int(metric_round_decimals))
            out[key_col] = values
        else:
            out[key_col] = out[col].astype("object")
        metric_key_cols.append(key_col)

    if allowed_varied_columns is not None:
        allowed_varied_columns = tuple(
            col for col in allowed_varied_columns
            if col in out.columns
        )
        if not allowed_varied_columns:
            raise ValueError("No allowed post-refit variant columns are available.")
        if max_varied_columns < 1:
            raise ValueError("max_varied_columns must be at least 1.")

        base_cols = metric_key_cols + list(protected_columns) + list(signature_columns)
        active = pd.Series(True, index=out.index)
        summary_rows: list[dict[str, Any]] = []
        group_counter = 0

        for _, equal_output_group in out.groupby(base_cols, sort=False, dropna=False):
            remaining = list(equal_output_group.index)
            while len(remaining) > 1:
                keep_idx = remaining[0]
                drop_indices: list[int] = []
                varied_by_idx: dict[int, list[str]] = {}
                for idx in remaining[1:]:
                    varied = [
                        col
                        for col in allowed_varied_columns
                        if not _values_equal(out.at[keep_idx, col], out.at[idx, col])
                    ]
                    if 0 < len(varied) <= int(max_varied_columns):
                        drop_indices.append(idx)
                        varied_by_idx[idx] = varied

                if not drop_indices:
                    remaining = remaining[1:]
                    continue

                group_counter += 1
                group_id = f"post_refit_eq_{group_counter:06d}"
                members = [keep_idx, *drop_indices]
                varied_columns = sorted(
                    {
                        col
                        for idx in drop_indices
                        for col in varied_by_idx[idx]
                    }
                )
                out.loc[members, group_id_col] = group_id
                out.loc[drop_indices, "metric_equivalence_kept_candidate_id"] = (
                    out.at[keep_idx, id_col]
                )
                out.loc[
                    drop_indices,
                    "metric_equivalence_varied_parameter_group",
                ] = "post_refit_theoretical_variant"
                out.loc[drop_indices, "metric_equivalence_drop_reason"] = (
                    "Identical refit metrics, predictions and decisions; "
                    f"only {','.join(varied_columns)} differs."
                )
                active.loc[drop_indices] = False

                summary_rows.append(
                    {
                        group_id_col: group_id,
                        "varied_parameter_group": "post_refit_theoretical_variant",
                        "varied_columns": ",".join(varied_columns),
                        "n_metric_equivalent_candidates": len(members),
                        "kept_candidate_id": out.at[keep_idx, id_col],
                        "dropped_candidate_ids": ",".join(
                            out.loc[drop_indices, id_col].astype(str)
                        ),
                        "prediction_signature_equal": True,
                        "decision_signature_equal": True,
                        "metrics_equal": True,
                        "technical_status": "duplicate_post_refit",
                        "reason": (
                            "Equivalent outputs after refit; deterministic "
                            "simplicity tie-break retained the first row."
                        ),
                    }
                )
                remaining = [
                    idx for idx in remaining
                    if idx not in drop_indices and idx != keep_idx
                ]

        kept = out.loc[active].copy().reset_index(drop=True)
        dropped = out.loc[~active].copy().reset_index(drop=True)
        summary_df = pd.DataFrame(summary_rows)
        temp_cols = [
            col for col in out.columns
            if col.startswith("__metric_equivalence_metric__")
        ]
        kept = kept.drop(columns=temp_cols, errors="ignore")
        dropped = dropped.drop(columns=temp_cols, errors="ignore")
        return kept, dropped, summary_df.reset_index(drop=True)

    protected_key_cols = list(protected_columns)
    group_columns_by_name = {
        name: list(cols)
        for name, cols in parameter_groups.items()
    }
    all_parameter_cols = []
    for cols in group_columns_by_name.values():
        all_parameter_cols.extend(cols)
    all_parameter_cols = list(dict.fromkeys(all_parameter_cols))

    active = pd.Series(True, index=out.index)
    summary_rows = []
    group_counter = 0

    for varied_group, varied_cols in group_columns_by_name.items():
        other_parameter_cols = [
            col
            for col in all_parameter_cols
            if col not in set(varied_cols)
        ]
        base_cols = metric_key_cols + protected_key_cols + other_parameter_cols
        if not base_cols:
            continue

        active_df = out.loc[active].copy()
        for _, group in active_df.groupby(base_cols, sort=False, dropna=False):
            if len(group) <= 1:
                continue
            n_varied_values = len(group.drop_duplicates(varied_cols))
            if n_varied_values <= 1:
                continue

            keep_idx = group.index[0]
            drop_indices = list(group.index[1:])
            if not drop_indices:
                continue

            group_counter += 1
            group_id = f"metric_eq_{group_counter:06d}"
            out.loc[[keep_idx] + drop_indices, group_id_col] = group_id
            out.loc[drop_indices, "metric_equivalence_kept_candidate_id"] = (
                out.loc[keep_idx, id_col] if id_col in out.columns else keep_idx
            )
            out.loc[drop_indices, "metric_equivalence_varied_parameter_group"] = varied_group
            out.loc[drop_indices, "metric_equivalence_drop_reason"] = (
                f"Identical metrics and only {varied_group!r} differs."
            )

            varied_records = (
                group[list(varied_cols)]
                .drop_duplicates()
                .to_dict("records")
            )
            metric_record = {
                col: out.loc[keep_idx, col]
                for col in metric_columns
            }
            summary_row = {
                group_id_col: group_id,
                "varied_parameter_group": varied_group,
                "varied_columns": ",".join(varied_cols),
                "n_metric_equivalent_candidates": int(len(group)),
                "kept_candidate_id": (
                    out.loc[keep_idx, id_col] if id_col in out.columns else str(keep_idx)
                ),
                "dropped_candidate_ids": (
                    ",".join(out.loc[drop_indices, id_col].astype(str))
                    if id_col in out.columns
                    else ",".join(map(str, drop_indices))
                ),
                "varied_values_json": json.dumps(
                    [
                        {
                            key: _json_ready_value(value)
                            for key, value in record.items()
                        }
                        for record in varied_records
                    ],
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "metric_columns": ",".join(metric_columns),
            }
            summary_row.update(metric_record)
            summary_rows.append(summary_row)
            active.loc[drop_indices] = False

    kept = out.loc[active].copy().reset_index(drop=True)
    dropped = out.loc[~active].copy().reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows)

    temp_cols = [col for col in out.columns if col.startswith("__metric_equivalence_metric__")]
    kept = kept.drop(columns=[col for col in temp_cols if col in kept.columns])
    dropped = dropped.drop(columns=[col for col in temp_cols if col in dropped.columns])

    return kept, dropped, summary_df.reset_index(drop=True)


def _values_equal(left: Any, right: Any) -> bool:
    """Scalar equality that treats two missing values as equal."""
    if _is_missing_scalar(left) and _is_missing_scalar(right):
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


def stable_frame_signature(
    df: pd.DataFrame,
    columns: Sequence[str],
    sort_columns: Sequence[str],
    round_decimals: int | None,
) -> str:
    """Hash selected, deterministically ordered columns of one result table."""
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return hashlib.sha256(b"").hexdigest()
    ordered = df.sort_values(
        [col for col in sort_columns if col in df.columns],
        kind="mergesort",
    )[cols].reset_index(drop=True)
    for col in cols:
        if pd.api.types.is_numeric_dtype(ordered[col]):
            values = pd.to_numeric(ordered[col], errors="coerce")
            if round_decimals is not None:
                values = values.round(int(round_decimals))
            ordered[col] = values
    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def build_simca_output_signatures(
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    id_col: str = "candidate_id",
    round_decimals: int | None = 12,
) -> pd.DataFrame:
    """
    Build compact exact-output signatures without saving the large predictions.

    Numeric outputs are rounded only for the post-refit equivalence test; the
    configured precision is deliberately much finer than reporting precision.
    """
    object_predictions = (
        pd.DataFrame() if object_predictions is None else object_predictions.copy()
    )
    pixel_predictions = (
        pd.DataFrame() if pixel_predictions is None else pixel_predictions.copy()
    )
    ids: list[str] = []
    for frame in (object_predictions, pixel_predictions):
        if len(frame) and id_col in frame.columns:
            ids.extend(frame[id_col].dropna().astype(str).tolist())
    candidate_ids = list(dict.fromkeys(ids))

    rows = []
    for candidate_id in candidate_ids:
        obj = (
            object_predictions[
                object_predictions[id_col].astype(str).eq(candidate_id)
            ]
            if len(object_predictions) and id_col in object_predictions.columns
            else pd.DataFrame()
        )
        pix = (
            pixel_predictions[
                pixel_predictions[id_col].astype(str).eq(candidate_id)
            ]
            if len(pixel_predictions) and id_col in pixel_predictions.columns
            else pd.DataFrame()
        )
        obj_entity = ("source_image", "object_id", "batch")
        pix_entity = ("source_image", "object_id", "batch", "row", "col")
        prediction_obj_cols = [
            *obj_entity,
            *[
                col for col in obj.columns
                if (
                    "target_pixel_ratio" in col
                    or col.startswith("predicted_")
                    or col.startswith("true_")
                )
                and col != "decision_3way"
            ],
        ]
        prediction_pix_cols = [
            *pix_entity,
            *[
                col for col in pix.columns
                if (
                    col.startswith("predicted_")
                    or col.startswith("pred_")
                    or col in {"rule_statistic", "rule_limit"}
                )
            ],
        ]
        decision_cols = [
            *obj_entity,
            *[
                col for col in obj.columns
                if col == "decision_3way" or col.startswith("predicted_")
            ],
        ]
        pred_payload = "|".join(
            (
                stable_frame_signature(
                    obj,
                    prediction_obj_cols,
                    obj_entity,
                    round_decimals,
                ),
                stable_frame_signature(
                    pix,
                    prediction_pix_cols,
                    pix_entity,
                    round_decimals,
                ),
            )
        )
        rows.append(
            {
                id_col: candidate_id,
                "prediction_signature": hashlib.sha256(
                    pred_payload.encode("ascii")
                ).hexdigest(),
                "decision_signature": stable_frame_signature(
                    obj,
                    decision_cols,
                    obj_entity,
                    round_decimals,
                ),
            }
        )
    return pd.DataFrame(rows, columns=expcfg.SIMCA_OUTPUT_SIGNATURE_COLUMNS)


# Private compatibility alias for older imports/checkpoints.
_stable_frame_signature = stable_frame_signature


def audit_simca_candidate_technical_status(
    candidates: pd.DataFrame,
    *,
    stage: str = "pre_refit",
    error_tables: Sequence[pd.DataFrame] = (),
    metric_tables: Sequence[pd.DataFrame] = (),
    object_predictions: pd.DataFrame | None = None,
    pixel_predictions: pd.DataFrame | None = None,
    calibrated_hyperparameters: pd.DataFrame | None = None,
    pca_selected_preprocessings: pd.DataFrame | None = None,
    id_col: str = "candidate_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only technical validity checks and return audit plus valid rows."""
    if candidates is None or len(candidates) == 0:
        audit = pd.DataFrame(columns=expcfg.SIMCA_TECHNICAL_AUDIT_COLUMNS)
        return audit, pd.DataFrame() if candidates is None else candidates.copy()

    out = normalize_simca_candidate_columns(candidates).reset_index(drop=True)
    if id_col not in out.columns:
        out = add_simca_candidate_ids(out, id_col=id_col)
    candidate_ids = out[id_col].astype(str)
    failures: list[pd.DataFrame] = []

    def add_failure(mask: pd.Series | np.ndarray, failure_type: str, message: str) -> None:
        mask_series = pd.Series(mask, index=out.index).fillna(False).astype(bool)
        if mask_series.any():
            failures.append(
                pd.DataFrame(
                    {
                        id_col: candidate_ids[mask_series],
                        "technical_failure_type": failure_type,
                        "technical_failure_message": message,
                    }
                )
            )

    required = (
        "matrix_family",
        "matrix_method",
        "preprocessing",
        "rule_variant",
        "n_components",
        "alpha",
        "decision_mode",
        "position_dilation_radius",
    )
    for col in required:
        if col not in out.columns:
            add_failure(np.ones(len(out), dtype=bool), "missing_configuration", f"Missing {col}.")
        else:
            add_failure(out[col].map(_is_missing_scalar), "missing_configuration", f"Missing {col}.")

    for col in (
        "n_components",
        "alpha",
        "position_dilation_radius",
    ):
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            add_failure(~np.isfinite(values), "nonfinite_configuration", f"{col} is not finite.")
    if "n_components" in out.columns:
        add_failure(
            pd.to_numeric(out["n_components"], errors="coerce").le(0),
            "impossible_components",
            "n_components must be strictly positive.",
        )
    if "decision_mode" in out.columns:
        is_two_way = out["decision_mode"].astype(str).eq("2way")
    else:
        is_two_way = pd.Series(False, index=out.index)
    if "object_threshold" not in out.columns:
        add_failure(
            is_two_way,
            "missing_configuration",
            "Missing object_threshold for a 2-way candidate.",
        )
    else:
        threshold = pd.to_numeric(out["object_threshold"], errors="coerce")
        add_failure(
            is_two_way & (~np.isfinite(threshold) | ~threshold.between(0.0, 1.0)),
            "impossible_threshold",
            "2-way object_threshold must be finite and in [0, 1].",
        )
    if "decision_mode" in out.columns:
        is_three_way = out["decision_mode"].astype(str).eq("3way")
        lower = pd.to_numeric(
            out.get("three_way_lower_threshold", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        upper = pd.to_numeric(
            out.get("three_way_upper_threshold", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        add_failure(
            is_three_way & (~np.isfinite(lower) | ~np.isfinite(upper) | lower.ge(upper)),
            "impossible_threshold",
            "3-way thresholds must be finite with lower < upper.",
        )

    balanced = out.get("matrix_method", pd.Series("", index=out.index)).astype(str).eq(
        "balanced_pixels"
    )
    m_values = pd.to_numeric(
        out.get("m_effective", out.get("m", pd.Series(np.nan, index=out.index))),
        errors="coerce",
    )
    add_failure(
        balanced & (~np.isfinite(m_values) | m_values.le(0)),
        "insufficient_pixels",
        "Balanced-pixel candidates require a finite positive m.",
    )

    if pca_selected_preprocessings is not None and len(pca_selected_preprocessings):
        selected_keys = set(
            zip(
                pca_selected_preprocessings["matrix_family"].astype(str),
                pca_selected_preprocessings["preprocessing"].astype(str),
            )
        )
        current_keys = list(
            zip(out["matrix_family"].astype(str), out["preprocessing"].astype(str))
        )
        add_failure(
            pd.Series([key not in selected_keys for key in current_keys], index=out.index),
            "preprocessing_not_selected",
            "Preprocessing is outside the PCA shortlist for its matrix family.",
        )

    if calibrated_hyperparameters is not None and len(calibrated_hyperparameters):
        calibration_ids = set(
            calibrated_hyperparameters.get(
                "calibration_id", pd.Series(dtype="object")
            ).dropna().astype(str)
        )
        if "calibration_id" in out.columns and calibration_ids:
            add_failure(
                ~out["calibration_id"].astype(str).isin(calibration_ids),
                "outside_calibrated_domain",
                "Candidate is outside the calibrated 03B domain.",
            )
            calibrated = calibrated_hyperparameters.drop_duplicates(
                "calibration_id"
            ).copy()
            compare_columns = (
                "matrix_method",
                "m",
                "balanced_pixel_strategy",
                "preprocessing",
                "preprocessing_steps",
                "rule_variant",
                "limit_source",
                "n_components",
                "alpha",
                "sg_window_length",
                "sg_polyorder",
                "position_dilation_radius",
                "object_threshold",
                "three_way_lower_threshold",
                "three_way_upper_threshold",
            )
            available = [
                col for col in compare_columns
                if col in out.columns and col in calibrated.columns
            ]
            calibration_view = calibrated[
                ["calibration_id", *available, *(
                    ["calibration_status"]
                    if "calibration_status" in calibrated.columns
                    else []
                )]
            ].rename(
                columns={
                    col: f"__calibrated__{col}"
                    for col in available
                }
            )
            compared = out[["calibration_id", *available]].merge(
                calibration_view,
                on="calibration_id",
                how="left",
                validate="many_to_one",
            )
            mismatch = pd.Series(False, index=out.index)
            for col in available:
                left = compared[col]
                right = compared[f"__calibrated__{col}"]
                both_present = ~left.map(_is_missing_scalar) & ~right.map(
                    _is_missing_scalar
                )
                if (
                    pd.api.types.is_numeric_dtype(left)
                    or pd.api.types.is_numeric_dtype(right)
                ):
                    equal = np.isclose(
                        pd.to_numeric(left, errors="coerce"),
                        pd.to_numeric(right, errors="coerce"),
                        equal_nan=True,
                    )
                else:
                    equal = left.astype(str).eq(right.astype(str))
                mismatch |= both_present & ~pd.Series(equal, index=out.index)
            add_failure(
                mismatch,
                "calibration_mismatch",
                "Candidate parameters or thresholds do not match 03B calibration.",
            )
            if "calibration_status" in compared.columns:
                add_failure(
                    ~compared["calibration_status"]
                    .astype(str)
                    .eq("calibrated_for_downstream_search"),
                    "invalid_calibration_status",
                    "03B status is not calibrated_for_downstream_search.",
                )

    known_ids = set(candidate_ids)
    for error_table in error_tables:
        if error_table is None or len(error_table) == 0:
            continue
        error_table = error_table.copy()
        join_col = next(
            (
                col for col in (id_col, "domain_config_id", "calibration_id")
                if col in error_table.columns and col in out.columns
            ),
            None,
        )
        if join_col is None:
            continue
        error_col = next(
            (
                col for col in (
                    "technical_failure_message",
                    "error_message",
                    "error",
                    "technical_errors",
                )
                if col in error_table.columns
            ),
            None,
        )
        if error_col is None:
            continue
        mapped = out[[id_col, join_col]].merge(
            error_table[[join_col, error_col]].dropna(subset=[error_col]),
            on=join_col,
            how="inner",
        )
        if len(mapped):
            mapped = mapped[mapped[id_col].astype(str).isin(known_ids)]
            failures.append(
                pd.DataFrame(
                    {
                        id_col: mapped[id_col].astype(str),
                        "technical_failure_type": "upstream_error",
                        "technical_failure_message": mapped[error_col].astype(str),
                    }
                )
            )

    if metric_tables:
        metric_ids: set[str] = set()
        for table in metric_tables:
            if table is not None and len(table) and id_col in table.columns:
                metric_ids.update(table[id_col].dropna().astype(str))
        if stage == "post_refit":
            add_failure(
                ~candidate_ids.isin(metric_ids),
                "missing_metrics",
                "No post-refit metrics were produced.",
            )

    for prediction_table, label in (
        (object_predictions, "object"),
        (pixel_predictions, "pixel"),
    ):
        if prediction_table is None:
            continue
        if id_col not in prediction_table.columns:
            add_failure(
                np.ones(len(out), dtype=bool),
                "missing_predictions",
                f"{label} predictions have no {id_col}.",
            )
            continue
        present = set(prediction_table[id_col].dropna().astype(str))
        add_failure(
            ~candidate_ids.isin(present),
            "missing_predictions",
            f"No {label} predictions were produced.",
        )
        numeric_output_cols = [
            col for col in prediction_table.columns
            if col in {"rule_statistic", "rule_limit"}
            or "target_pixel_ratio" in col
        ]
        if numeric_output_cols:
            bad_ids = set()
            for col in numeric_output_cols:
                numeric = pd.to_numeric(prediction_table[col], errors="coerce")
                bad_ids.update(
                    prediction_table.loc[~np.isfinite(numeric), id_col]
                    .dropna()
                    .astype(str)
                )
            add_failure(
                candidate_ids.isin(bad_ids),
                "nonfinite_output",
                f"{label} predictions contain NaN or Inf.",
            )

    if failures:
        failure_df = pd.concat(failures, ignore_index=True)
        grouped = (
            failure_df.groupby(id_col, sort=False, dropna=False)
            .agg(
                technical_failure_type=(
                    "technical_failure_type",
                    lambda values: ",".join(dict.fromkeys(map(str, values))),
                ),
                technical_failure_message=(
                    "technical_failure_message",
                    lambda values: " | ".join(dict.fromkeys(map(str, values))),
                ),
            )
            .reset_index()
        )
    else:
        grouped = pd.DataFrame(
            columns=(id_col, "technical_failure_type", "technical_failure_message")
        )

    audit = pd.DataFrame({id_col: candidate_ids}).drop_duplicates(id_col)
    audit = audit.merge(grouped, on=id_col, how="left", validate="one_to_one")
    audit["technical_status"] = np.where(
        audit["technical_failure_type"].isna(),
        "valid",
        "invalid",
    )
    audit["technical_failure_type"] = audit[
        "technical_failure_type"
    ].fillna("")
    audit["technical_failure_message"] = audit[
        "technical_failure_message"
    ].fillna("")
    audit = audit[list(expcfg.SIMCA_TECHNICAL_AUDIT_COLUMNS)]
    valid_ids = set(audit.loc[audit["technical_status"].eq("valid"), id_col])
    valid = out[out[id_col].astype(str).isin(valid_ids)].reset_index(drop=True)
    return audit.reset_index(drop=True), valid


def build_pca_preprocessing_configs_by_matrix_family(
    pca_selected_preprocessings_df: pd.DataFrame,
    expected_families: Sequence[str] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Build family-specific preprocessing configs from notebook 03 shortlist."""
    expected_families = tuple(expected_families or expcfg.SIMCA_MATRIX_FAMILIES)
    validate_simca_table_columns(
        pca_selected_preprocessings_df,
        expcfg.SIMCA_PCA_SHORTLIST_REQUIRED_COLUMNS,
        table_name="PCA preprocessing shortlist",
    )

    d = pca_selected_preprocessings_df.copy()
    d["matrix_family"] = d["matrix_family"].astype(str)
    d["preprocessing"] = d["preprocessing"].astype(str)
    d["preprocessing_steps"] = d["preprocessing_steps"].map(
        lambda value: "+".join(parse_preprocessing_steps(value))
    )

    unknown_families = sorted(set(d["matrix_family"]) - set(expected_families))
    if unknown_families:
        raise ValueError(f"Unknown matrix_family value(s) in PCA shortlist: {unknown_families}")

    conflicts = (
        d.drop_duplicates(["matrix_family", "preprocessing", "preprocessing_steps"])
        .groupby(["matrix_family", "preprocessing"], dropna=False)
        .size()
        .reset_index(name="n_step_variants")
    )
    conflicts = conflicts.loc[conflicts["n_step_variants"] > 1]
    if len(conflicts) > 0:
        preview = conflicts[["matrix_family", "preprocessing"]].to_dict("records")
        raise ValueError(
            "A preprocessing name has several step definitions within the same matrix family: "
            f"{preview}"
        )

    configs_by_family: dict[str, dict[str, tuple[str, ...]]] = {
        family: {}
        for family in expected_families
    }
    for _, row in d.drop_duplicates(["matrix_family", "preprocessing"]).iterrows():
        family = str(row["matrix_family"])
        name = str(row["preprocessing"])
        steps = tuple(parse_preprocessing_steps(row["preprocessing_steps"]))
        configs_by_family[family][name] = steps

    return {
        family: normalize_preprocessing_configs(configs)
        for family, configs in configs_by_family.items()
    }


def allowed_pca_preprocessing_pairs(
    pca_selected_preprocessings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return allowed (matrix_family, preprocessing) pairs from notebook 03."""
    configs_by_family = build_pca_preprocessing_configs_by_matrix_family(
        pca_selected_preprocessings_df
    )
    rows = []
    for family, configs in configs_by_family.items():
        for name, steps in configs.items():
            rows.append(
                {
                    "matrix_family": family,
                    "preprocessing": name,
                    "preprocessing_steps": "+".join(steps),
                }
            )
    return pd.DataFrame(rows)


def filter_simca_candidates_by_pca_preprocessing(
    candidates_df: pd.DataFrame,
    pca_selected_preprocessings_df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """Keep only candidate preprocessings allowed for their matrix family."""
    validate_simca_table_columns(
        candidates_df,
        ("matrix_family", "preprocessing"),
        table_name="SIMCA candidate table",
    )

    candidates = normalize_simca_candidate_columns(candidates_df)
    allowed = allowed_pca_preprocessing_pairs(pca_selected_preprocessings_df)
    allowed = allowed[["matrix_family", "preprocessing"]].drop_duplicates()
    checked = candidates.merge(
        allowed.assign(_pca_preprocessing_allowed=True),
        on=["matrix_family", "preprocessing"],
        how="left",
    )

    invalid_mask = checked["_pca_preprocessing_allowed"].isna()
    if strict and invalid_mask.any():
        invalid = (
            checked.loc[invalid_mask, ["matrix_family", "preprocessing"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            "Some SIMCA candidates use preprocessings not selected for their matrix family: "
            f"{invalid}"
        )

    return (
        checked.loc[~invalid_mask]
        .drop(columns=["_pca_preprocessing_allowed"])
        .reset_index(drop=True)
    )


def validate_simca_candidates_match_pca_preprocessing(
    candidates_df: pd.DataFrame,
    pca_selected_preprocessings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Blocking check for PCA-shortlisted preprocessing scope."""
    filter_simca_candidates_by_pca_preprocessing(
        candidates_df,
        pca_selected_preprocessings_df,
        strict=True,
    )
    return candidates_df


def hash_locked_validation_plan() -> str:
    """Hash only the prespecified 04C choices that may affect batch 3."""
    payload = {
        name: _json_ready_value(getattr(expcfg, name))
        for name in expcfg.SIMCA_CONCAT_REFIT_VALIDATION_PLAN_KEYS
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def hash_locked_validation_evaluation_rule() -> str:
    """Hash the versioned 04C metric and guardrail definitions."""
    payload = {
        name: _json_ready_value(getattr(expcfg, name))
        for name in expcfg.SIMCA_CONCAT_REFIT_EVALUATION_RULE_KEYS
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _locked_values_equal(left: Any, right: Any) -> bool:
    """Compare frozen scalar values while treating aligned missing values equally."""
    if _is_missing_scalar(left) and _is_missing_scalar(right):
        return True
    if _is_missing_scalar(left) or _is_missing_scalar(right):
        return False
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        try:
            return bool(float(left) == float(right))
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


# def build_locked_validation_candidate_pool(
#     calibrated_hyperparameters: pd.DataFrame,
#     calibration_domain: pd.DataFrame,
#     pareto_reference: pd.DataFrame,
#     projection_eligibility: pd.DataFrame,
#     *,
#     optuna_trials: pd.DataFrame | None = None,
#     optuna_pareto_candidates: pd.DataFrame | None = None,
# ) -> pd.DataFrame:
#     """Build the score-free task-31 pool from frozen 03B/03C/04A outputs.

#     Supported tracks use the protocol Pareto front. Unsupported domain-shift
#     tracks use the diagnostic Pareto front and remain explicitly diagnostic.
#     Optuna is attached only as provenance and never adds or removes a row.
#     """
#     required_domain = set(expcfg.SIMCA_CONCAT_REFIT_CANDIDATE_COLUMNS) - {
#         "validation_candidate_id",
#         "data_config_id",
#         "eligibility_status",
#         "candidate_front",
#         "visited_by_optuna",
#         "optuna_pareto",
#     }
#     required_pareto = {
#         "row_type",
#         "calibration_id",
#         "evaluation_track",
#         "eligibility_status",
#         "technical_status",
#         "protocol_pareto_front",
#         "diagnostic_pareto_front",
#     }
#     required_eligibility = {"evaluation_track", "eligibility_status"}
#     for frame, required, name in (
#         (calibration_domain, required_domain, "calibration_domain"),
#         (pareto_reference, required_pareto, "grid_pareto_reference"),
#         (projection_eligibility, required_eligibility, "projection_eligibility"),
#     ):
#         missing = sorted(required - set(frame.columns))
#         if missing:
#             raise KeyError(f"{name} is missing frozen columns: {missing}")

#     if expcfg.SIMCA_CONCAT_REFIT_MAX_CANDIDATES is not None:
#         raise RuntimeError(
#             "A validation candidate cap would select on row order; keep "
#             "SIMCA_CONCAT_REFIT_MAX_CANDIDATES=None."
#         )
#     eligibility = projection_eligibility[
#         ["evaluation_track", "eligibility_status"]
#     ].drop_duplicates()
#     if eligibility["evaluation_track"].astype(str).duplicated().any():
#         raise RuntimeError("03C eligibility must contain one row per track.")

#     configurations = pareto_reference.loc[
#         pareto_reference["row_type"].astype(str).eq("configuration")
#         & pareto_reference["technical_status"].astype(str).eq("calculable")
#     ].copy()
#     supported = configurations["eligibility_status"].isin(
#         expcfg.SIMCA_CONCAT_REFIT_SUPPORTED_ELIGIBILITY_STATUSES
#     )
#     unsupported = configurations["eligibility_status"].isin(
#         expcfg.SIMCA_CONCAT_REFIT_UNSUPPORTED_ELIGIBILITY_STATUSES
#     )
#     keep = (
#         supported & configurations["protocol_pareto_front"].astype(bool)
#     ) | (
#         unsupported & configurations["diagnostic_pareto_front"].astype(bool)
#     )
#     selected = configurations.loc[
#         keep, ["calibration_id", "evaluation_track", "eligibility_status"]
#     ].copy()
#     selected["candidate_front"] = np.where(
#         selected["eligibility_status"].isin(
#             expcfg.SIMCA_CONCAT_REFIT_SUPPORTED_ELIGIBILITY_STATUSES
#         ),
#         "protocol_pareto",
#         "diagnostic_pareto_unsupported_domain_shift",
#     )
#     if selected["calibration_id"].astype(str).duplicated().any():
#         raise RuntimeError("04A Pareto reference selected a calibration twice.")

#     domain = calibration_domain.copy()
#     if domain["domain_config_id"].astype(str).duplicated().any():
#         raise RuntimeError("03B domain_config_id must be unique.")
#     pool = selected.drop(columns="evaluation_track").merge(
#         domain,
#         on="calibration_id",
#         how="left",
#         validate="one_to_many",
#     )
#     if pool["domain_config_id"].isna().any():
#         raise RuntimeError("A selected 04A row is absent from the 03B domain.")
#     pool = pool.merge(
#         eligibility.rename(columns={"eligibility_status": "eligibility_status_03c"}),
#         on="evaluation_track",
#         how="left",
#         validate="many_to_one",
#     )
#     mismatch = ~pool.apply(
#         lambda row: _locked_values_equal(
#             row["eligibility_status"], row["eligibility_status_03c"]
#         ),
#         axis=1,
#     )
#     if mismatch.any():
#         raise RuntimeError("04A and 03C eligibility statuses disagree.")
#     pool = pool.drop(columns="eligibility_status_03c")

#     seed_specific_columns = {
#         "fit_config_id",
#         "projection_config_id",
#         "random_state",
#     }
#     locked_columns = tuple(
#         column
#         for column in expcfg.SIMCA_EXACT_CONFIG_COLUMNS
#         if column not in seed_specific_columns
#         and column in calibrated_hyperparameters.columns
#         and column in pool.columns
#     )
#     member_column = "member_evaluation_config_ids_json"
#     calibration_columns = ["calibration_id", *locked_columns]
#     if member_column in calibrated_hyperparameters.columns:
#         calibration_columns.append(member_column)
#     elif "evaluation_config_id" in calibrated_hyperparameters.columns:
#         calibration_columns.append("evaluation_config_id")
#     else:
#         raise KeyError(
#             "calibrated_hyperparameters has no member evaluation provenance."
#         )
#     calibration = calibrated_hyperparameters[
#         calibration_columns
#     ].drop_duplicates("calibration_id")
#     calibration["__calibrated_present"] = True
#     checked = pool.merge(
#         calibration,
#         on="calibration_id",
#         how="left",
#         suffixes=("", "__03b"),
#         validate="many_to_one",
#     )
#     if checked["__calibrated_present"].isna().any():
#         raise RuntimeError("A validation candidate has no calibrated 03B row.")
#     checked = checked.drop(columns="__calibrated_present")
#     if member_column in checked.columns:
#         member_ok = []
#         for evaluation_id, raw_members in zip(
#             checked["evaluation_config_id"], checked[member_column]
#         ):
#             try:
#                 members = set(map(str, json.loads(str(raw_members))))
#             except (TypeError, ValueError, json.JSONDecodeError) as exc:
#                 raise RuntimeError(
#                     "Invalid 03B member_evaluation_config_ids_json."
#                 ) from exc
#             member_ok.append(str(evaluation_id) in members)
#         if not all(member_ok):
#             raise RuntimeError(
#                 "A seed-specific domain row is absent from its frozen 03B "
#                 "calibration membership."
#             )
#         checked = checked.drop(columns=member_column)
#     elif "evaluation_config_id__03b" in checked.columns:
#         equal_members = checked["evaluation_config_id"].astype(str).eq(
#             checked["evaluation_config_id__03b"].astype(str)
#         )
#         if not equal_members.all():
#             raise RuntimeError("A validation row is not the frozen 03B member.")
#         checked = checked.drop(columns="evaluation_config_id__03b")
#     differences = []
#     for column in locked_columns:
#         other = f"{column}__03b"
#         equal = np.fromiter(
#             (
#                 _locked_values_equal(left, right)
#                 for left, right in zip(checked[column], checked[other])
#             ),
#             dtype=bool,
#             count=len(checked),
#         )
#         if not equal.all():
#             differences.append(column)
#         checked = checked.drop(columns=other)
#     if differences:
#         raise RuntimeError(
#             "04C candidates differ from frozen 03B hyperparameters: "
#             f"{differences}"
#         )
#     pool = checked
#     data_identity_columns = (
#         "matrix_method",
#         "m",
#         "balanced_pixel_strategy",
#         "preprocessing",
#         "preprocessing_steps",
#         "sg_window_length",
#         "sg_polyorder",
#         "random_state",
#     )
#     pool["data_config_id"] = [
#         simca_candidate_key(
#             row,
#             id_columns=data_identity_columns,
#             prefix="validation_data",
#         )
#         for _, row in pool.iterrows()
#     ]
#     pool["validation_candidate_id"] = [
#         simca_candidate_key(
#             row,
#             id_columns=(
#                 "calibration_id",
#                 "domain_config_id",
#                 "random_state",
#             ),
#             prefix="validation_candidate",
#         )
#         for _, row in pool.iterrows()
#     ]
#     if pool["validation_candidate_id"].duplicated().any():
#         raise RuntimeError("04C validation_candidate_id must be unique.")

#     visited_ids: set[str] = set()
#     if optuna_trials is not None and len(optuna_trials):
#         if "calibration_id" not in optuna_trials:
#             raise KeyError("optuna_trials has no calibration_id provenance.")
#         visited_ids = set(optuna_trials["calibration_id"].dropna().astype(str))
#     optuna_front_ids: set[str] = set()
#     if optuna_pareto_candidates is not None and len(optuna_pareto_candidates):
#         if "calibration_id" not in optuna_pareto_candidates:
#             raise KeyError("optuna_pareto_candidates has no calibration_id.")
#         optuna_front_ids = set(
#             optuna_pareto_candidates["calibration_id"].dropna().astype(str)
#         )
#     pool["visited_by_optuna"] = pool["calibration_id"].astype(str).isin(visited_ids)
#     pool["optuna_pareto"] = pool["calibration_id"].astype(str).isin(optuna_front_ids)
#     pool["target_class"] = expcfg.TARGET_CLASS
#     pool["non_target_label"] = expcfg.NON_TARGET_LABEL

#     unknown_tracks = set(pool["evaluation_track"].astype(str)) - set(
#         expcfg.SIMCA_EVALUATION_TRACKS
#     )
#     if unknown_tracks:
#         raise RuntimeError(f"Unknown 04C tracks: {sorted(unknown_tracks)}")
#     return pool.reindex(columns=expcfg.SIMCA_CONCAT_REFIT_CANDIDATE_COLUMNS)
