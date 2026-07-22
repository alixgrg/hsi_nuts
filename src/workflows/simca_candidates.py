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


def infer_matrix_family(matrix_method: Any) -> str:
    """Infer the canonical matrix family for a matrix method or matrix id."""
    token = str(matrix_method)
    if token in expcfg.SIMCA_MATRIX_METHOD_FAMILY:
        return expcfg.SIMCA_MATRIX_METHOD_FAMILY[token]
    if token.startswith("balanced_pixel_"):
        return "pixel_matrix"
    if token.startswith("object_"):
        return "object_matrix"
    return "unknown_matrix_family"


def selection_track_from_parts(matrix_family: Any, decision_mode: Any) -> str:
    """Return the canonical SIMCA selection track name."""
    matrix_family = str(matrix_family)
    decision_mode = _normalise_decision_mode(decision_mode)
    track = f"{matrix_family}_{decision_mode}"
    if track not in expcfg.SIMCA_SELECTION_TRACKS:
        valid = ", ".join(expcfg.SIMCA_SELECTION_TRACKS)
        raise ValueError(f"Unknown SIMCA selection track {track!r}. Valid tracks: {valid}.")
    return track


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


def validate_simca_selection_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that selection_track matches matrix_family and decision_mode."""
    validate_simca_table_columns(
        df,
        ("selection_track", "matrix_family", "decision_mode"),
        table_name="SIMCA selection/evaluation table",
    )
    invalid_rows = []
    for idx, row in df.iterrows():
        try:
            expected = selection_track_from_parts(row["matrix_family"], row["decision_mode"])
        except ValueError:
            invalid_rows.append(idx)
            continue
        if str(row["selection_track"]) != expected:
            invalid_rows.append(idx)

    if invalid_rows:
        preview = list(invalid_rows[:10])
        raise ValueError(
            "selection_track must match matrix_family + decision_mode. "
            f"Invalid row index preview: {preview}"
        )
    return df


def validate_simca_candidate_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical SIMCA candidate-config table contract."""
    return validate_simca_table_columns(
        df,
        expcfg.SIMCA_CANDIDATE_CONFIG_REQUIRED_COLUMNS,
        table_name="SIMCA candidate config table",
    )


def validate_simca_evaluation_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical SIMCA model-evaluation table contract."""
    validate_simca_table_columns(
        df,
        expcfg.SIMCA_CANDIDATE_EVALUATION_REQUIRED_COLUMNS,
        table_name="SIMCA candidate evaluation table",
    )
    return validate_simca_selection_tracks(df)


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
            out["matrix_family"] = out["matrix_method"].map(infer_matrix_family)
        elif "training_matrix_id" in out.columns:
            out["matrix_family"] = out["training_matrix_id"].map(infer_matrix_family)

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


def add_selection_track(
    df: pd.DataFrame,
    matrix_family_col: str = "matrix_family",
    decision_mode_col: str = "decision_mode",
    track_col: str = "selection_track",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Add the canonical 4-track SIMCA selection label to a table."""
    out = normalize_simca_candidate_columns(df)
    validate_simca_table_columns(
        out,
        (matrix_family_col, decision_mode_col),
        table_name="SIMCA selection/evaluation table",
    )

    if track_col in out.columns and not overwrite:
        return validate_simca_selection_tracks(out)

    out[track_col] = [
        selection_track_from_parts(row[matrix_family_col], row[decision_mode_col])
        for _, row in out.iterrows()
    ]
    return out


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
    id_col: str = "candidate_id",
    group_id_col: str = "metric_equivalence_group_id",
    metric_round_decimals: int | None = 12,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collapse candidates with identical metrics when one logical parameter differs.

    The first row in the input order is kept. Callers should therefore sort the
    table by their preferred tie-break policy before using this function.
    """
    if df is None or len(df) == 0:
        empty = pd.DataFrame() if df is None else df.copy()
        return empty, empty.copy(), pd.DataFrame()

    out = normalize_simca_candidate_columns(df).reset_index(drop=True)
    out["metric_equivalence_original_order"] = np.arange(len(out), dtype=int)

    metric_columns = tuple(metric_columns or expcfg.SIMCA_METRIC_EQUIVALENCE_METRIC_COLUMNS)
    protected_columns = tuple(
        protected_columns or expcfg.SIMCA_METRIC_EQUIVALENCE_PROTECTED_COLUMNS
    )
    parameter_groups = dict(
        parameter_groups or expcfg.SIMCA_METRIC_EQUIVALENCE_PARAMETER_GROUPS
    )

    required = set(metric_columns) | set(protected_columns)
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
    if not parameter_groups:
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
