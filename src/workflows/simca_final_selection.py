from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.workflows.simca_candidates import (
    normalize_simca_candidate_columns,
    validate_simca_table_columns,
)
from src.workflows.simca_robustness import select_track_primary_or_available_metrics
from src.workflows.simca_selection_utils import pareto_front_by_group
from src.workflows.simca_tables import compact_simca_table, write_simca_table


def split_flag_string(value: Any) -> list[str]:
    """Split semicolon-separated flags while ignoring empty/NA values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(flag) for flag in value if str(flag) and str(flag).lower() != "nan"})
    if pd.isna(value):
        return []
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return sorted({flag for flag in text.split(";") if flag and flag.lower() != "nan"})


def join_flags(flags: Sequence[str]) -> str:
    """Return a stable semicolon-separated flag string."""
    return ";".join(sorted({str(flag) for flag in flags if str(flag)}))


def _numeric(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _mode_for_group(df: pd.DataFrame) -> str:
    if "decision_mode" in df.columns and df["decision_mode"].notna().any():
        return str(df["decision_mode"].dropna().astype(str).iloc[0]).lower()
    if "selection_track" in df.columns and df["selection_track"].notna().any():
        track = str(df["selection_track"].dropna().astype(str).iloc[0]).lower()
        return "3way" if "3way" in track else "2way"
    return "2way"


def pareto_objective_columns(
    df: pd.DataFrame,
    *,
    decision_mode: str | None = None,
    two_way_minimize_cols: Sequence[str] = expcfg.SIMCA_FINAL_2WAY_PARETO_MINIMIZE_COLUMNS,
    three_way_minimize_cols: Sequence[str] = expcfg.SIMCA_FINAL_3WAY_PARETO_MINIMIZE_COLUMNS,
) -> list[str]:
    """Return Pareto rate columns for a final-selection track."""
    mode = str(decision_mode or _mode_for_group(df)).lower()
    configured = three_way_minimize_cols if mode == "3way" else two_way_minimize_cols
    cols = [col for col in configured if col in df.columns]
    if not cols and mode == "3way":
        cols = [col for col in ("fn_rate", "fp_rate") if col in df.columns]
    if not cols:
        raise ValueError(f"No Pareto objective columns found for decision_mode={mode!r}.")
    return cols


def pareto_front_mask(df: pd.DataFrame, minimize_cols: Sequence[str]) -> pd.Series:
    """Return True for non-dominated rows, minimizing the given objective columns."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=bool)
    cols = [col for col in minimize_cols if col in df.columns]
    if not cols:
        raise ValueError("No Pareto objective columns are present in the dataframe.")

    work = df.copy().reset_index(drop=True)
    work[cols] = work[cols].apply(pd.to_numeric, errors="coerce").fillna(np.inf)
    work["_pareto_row_id"] = np.arange(len(work), dtype=int)
    front = pareto_front_by_group(
        work,
        group_cols=(),
        minimize_cols=cols,
        maximize_cols=(),
        epsilon=0.0,
    )
    keep = set(front["_pareto_row_id"].astype(int).tolist())
    return pd.Series(
        [row_id in keep for row_id in work["_pareto_row_id"]],
        index=df.index,
        dtype=bool,
    )


def assign_pareto_tiers(df: pd.DataFrame, minimize_cols: Sequence[str]) -> pd.DataFrame:
    """Assign Pareto tiers by repeatedly removing the current non-dominated front."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    out["pareto_tier"] = pd.NA
    remaining = out.index.to_list()
    tier = 1

    while remaining:
        current = out.loc[remaining]
        front_mask = pareto_front_mask(current, minimize_cols=minimize_cols)
        front_index = current.loc[front_mask].index.to_list()
        if not front_index:
            front_index = remaining[:1]
        out.loc[front_index, "pareto_tier"] = int(tier)
        remaining = [idx for idx in remaining if idx not in set(front_index)]
        tier += 1

    out["pareto_tier"] = out["pareto_tier"].astype("Int64")
    out["is_pareto_front"] = out["pareto_tier"].eq(1)
    return out


def _tie_break_columns(df: pd.DataFrame, decision_mode: str) -> tuple[list[str], list[bool]]:
    configured = (
        expcfg.SIMCA_FINAL_TIEBREAK_3WAY_COLUMNS
        if str(decision_mode).lower() == "3way"
        else expcfg.SIMCA_FINAL_TIEBREAK_2WAY_COLUMNS
    )
    cols = ["pareto_tier"] + [col for col in configured if col in df.columns]
    ascending: list[bool] = []
    for col in cols:
        ascending.append(col not in {"balanced_accuracy", "coverage_rate", "decided_balanced_accuracy"})
    return cols, ascending


def sort_by_pareto(df: pd.DataFrame, *, decision_mode: str | None = None) -> pd.DataFrame:
    """Sort candidates by Pareto tier and deterministic rate-based tie-breakers."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()
    mode = decision_mode or _mode_for_group(df)
    cols, ascending = _tie_break_columns(df, str(mode).lower())
    return df.sort_values(cols, ascending=ascending).reset_index(drop=True)


def build_final_selection_guardrails(
    *,
    track_scoring_flags_df: pd.DataFrame,
    pure_test_metrics_df: pd.DataFrame,
    pure_test_guardrails_df: pd.DataFrame,
    candidate_panel_df: pd.DataFrame | None = None,
    expected_tracks: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
) -> pd.DataFrame:
    """Build notebook-06B input guardrails before Pareto selection."""
    rows: list[dict[str, Any]] = []

    def add(
        check_name: str,
        passed: bool,
        details: Any = "",
        severity: str = "error",
        n_records: int | None = None,
    ) -> None:
        rows.append(
            {
                "check_name": check_name,
                "passed": bool(passed),
                "status": "passed" if passed else "failed",
                "severity": severity,
                "details": str(details),
                "n_records": np.nan if n_records is None else int(n_records),
            }
        )

    expected = set(map(str, expected_tracks))
    add(
        "pure_test_guardrails_available",
        pure_test_guardrails_df is not None and len(pure_test_guardrails_df) > 0,
        n_records=0 if pure_test_guardrails_df is None else len(pure_test_guardrails_df),
    )
    if pure_test_guardrails_df is not None and len(pure_test_guardrails_df) > 0:
        passed = bool(pure_test_guardrails_df["passed"].astype(bool).all())
        failed = pure_test_guardrails_df.loc[
            ~pure_test_guardrails_df["passed"].astype(bool),
            "check_name",
        ].astype(str).tolist()
        add("pure_test_guardrails_all_passed", passed, failed)

    add(
        "validation_review_table_available",
        track_scoring_flags_df is not None and len(track_scoring_flags_df) > 0,
        n_records=0 if track_scoring_flags_df is None else len(track_scoring_flags_df),
    )
    if track_scoring_flags_df is not None and len(track_scoring_flags_df) > 0 and "selection_track" in track_scoring_flags_df.columns:
        observed = set(track_scoring_flags_df["selection_track"].dropna().astype(str))
        add("validation_review_tracks_complete", expected.issubset(observed), sorted(expected - observed))

    add(
        "pure_test_metrics_available",
        pure_test_metrics_df is not None and len(pure_test_metrics_df) > 0,
        n_records=0 if pure_test_metrics_df is None else len(pure_test_metrics_df),
    )
    if pure_test_metrics_df is not None and len(pure_test_metrics_df) > 0 and "selection_track" in pure_test_metrics_df.columns:
        observed = set(pure_test_metrics_df["selection_track"].dropna().astype(str))
        add("pure_test_metric_tracks_complete", expected.issubset(observed), sorted(expected - observed))

    add(
        "candidate_panel_available",
        candidate_panel_df is not None and len(candidate_panel_df) > 0,
        severity="warning",
        n_records=0 if candidate_panel_df is None else len(candidate_panel_df),
    )

    return compact_simca_table(pd.DataFrame(rows), table_kind="final_selection_guardrails")


def validate_final_selection_guardrails(guardrails_df: pd.DataFrame) -> pd.DataFrame:
    """Raise if an error-level notebook-06B input guardrail failed."""
    if guardrails_df is None or len(guardrails_df) == 0:
        raise RuntimeError("Final-selection guardrail table is empty.")
    failed = guardrails_df.loc[
        ~guardrails_df["passed"].astype(bool)
        & guardrails_df.get("severity", "error").astype(str).eq("error")
    ]
    if len(failed) > 0:
        raise RuntimeError(
            "Final-selection input guardrail failure(s): "
            + ", ".join(failed["check_name"].astype(str).tolist())
        )
    return guardrails_df


def _prepare_validation_review(track_scoring_flags_df: pd.DataFrame) -> pd.DataFrame:
    review = select_track_primary_or_available_metrics(track_scoring_flags_df).copy()
    columns = [
        "selected_config_id",
        "selection_track",
        "validation_metric_level",
        "previous_flags",
        "previous_flag_count",
    ]
    if "selected_config_id" not in review.columns or "selection_track" not in review.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    flag_columns = tuple(expcfg.SIMCA_FINAL_PREVIOUS_FLAG_COLUMNS)
    for _, row in review.drop_duplicates(["selected_config_id", "selection_track"]).iterrows():
        flags: list[str] = []
        for column in flag_columns:
            flags.extend(split_flag_string(row.get(column)))
        rows.append(
            {
                "selected_config_id": str(row.get("selected_config_id")),
                "selection_track": str(row.get("selection_track")),
                "validation_metric_level": row.get("metric_level", pd.NA),
                "previous_flags": join_flags(flags),
                "previous_flag_count": len(set(flags)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _merge_candidate_panel(pool: pd.DataFrame, candidate_panel_df: pd.DataFrame | None) -> pd.DataFrame:
    if candidate_panel_df is None or len(candidate_panel_df) == 0:
        return pool.copy()
    candidates = normalize_simca_candidate_columns(candidate_panel_df).copy()
    extra_cols = [col for col in candidates.columns if col == "selected_config_id" or col not in pool.columns]
    if "selected_config_id" not in extra_cols:
        extra_cols.insert(0, "selected_config_id")
    return pool.merge(
        candidates[extra_cols].drop_duplicates("selected_config_id"),
        on="selected_config_id",
        how="left",
        validate="many_to_one",
    )


def _attach_pure_test_errors(pool: pd.DataFrame, pure_test_errors_df: pd.DataFrame | None) -> pd.DataFrame:
    out = pool.copy()
    if pure_test_errors_df is None or len(pure_test_errors_df) == 0 or "selected_config_id" not in pure_test_errors_df.columns:
        out["has_pure_test_error"] = False
        out["pure_test_error"] = pd.NA
        return out
    errors = (
        pure_test_errors_df.dropna(subset=["selected_config_id"])
        .assign(selected_config_id=lambda df: df["selected_config_id"].astype(str))
        .groupby("selected_config_id", dropna=False)
        .agg(pure_test_error=("error", lambda values: "; ".join(sorted({str(v) for v in values if pd.notna(v)}))))
        .reset_index()
    )
    out["selected_config_id"] = out["selected_config_id"].astype(str)
    out = out.merge(errors, on="selected_config_id", how="left", validate="many_to_one")
    out["has_pure_test_error"] = out["pure_test_error"].notna()
    return out


def _flag_filter_mask(flags: pd.Series, flags_to_filter: Sequence[str]) -> pd.Series:
    blocked = set(map(str, flags_to_filter))
    if not blocked:
        return pd.Series(False, index=flags.index)
    return flags.fillna("").astype(str).apply(
        lambda value: bool(set(split_flag_string(value)).intersection(blocked))
    )


def apply_rate_threshold_filter_by_track(
    pool_df: pd.DataFrame,
    *,
    fn_rate_max: float | None = expcfg.SIMCA_FINAL_FN_RATE_MAX,
    fp_rate_max: float | None = expcfg.SIMCA_FINAL_FP_RATE_MAX,
    uncertain_rate_max: float | None = expcfg.SIMCA_FINAL_UNCERTAIN_RATE_MAX,
    track_order: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
) -> pd.DataFrame:
    """Apply strict rate thresholds before any Pareto comparison.

    A configured maximum is an open upper bound: a value is accepted only
    when ``rate < maximum``. ``uncertain_rate_max`` is only applicable to
    3way tracks; it is ignored for 2way tracks where uncertainty is absent.
    Missing values fail a configured, applicable threshold.
    """
    if pool_df is None or len(pool_df) == 0:
        return pd.DataFrame() if pool_df is None else pool_df.copy()

    out = pool_df.copy()
    candidates = out.get(
        "preselection_status",
        pd.Series("candidate", index=out.index, dtype="object"),
    ).astype(str).eq("candidate")
    any_threshold = any(value is not None for value in (fn_rate_max, fp_rate_max, uncertain_rate_max))
    out["rate_threshold_passed"] = True
    out["rate_threshold_status"] = "not_applied" if not any_threshold else "passed"
    out["rate_threshold_reason"] = ""

    for track in track_order:
        track_mask = out["selection_track"].astype(str).eq(str(track)) & candidates
        indices = out.index[track_mask].tolist()
        if not indices:
            continue

        group = out.loc[indices]
        reasons = pd.Series("", index=indices, dtype="object")
        passed = pd.Series(True, index=indices, dtype=bool)

        checks: list[tuple[str, float | None, bool]] = [
            ("fn_rate", fn_rate_max, True),
            ("fp_rate", fp_rate_max, True),
            (
                "uncertain_rate",
                uncertain_rate_max,
                "3way" in str(track).lower(),
            ),
        ]
        for column, maximum, applicable in checks:
            if maximum is None or not applicable:
                continue
            values = pd.to_numeric(group.get(column, pd.Series(np.nan, index=indices)), errors="coerce")
            failed = values.isna() | ~values.lt(float(maximum))
            passed &= ~failed
            reasons.loc[failed] = reasons.loc[failed].where(
                reasons.loc[failed].eq(""), reasons.loc[failed] + ";"
            ) + f"{column}>={float(maximum):g}"

        out.loc[indices, "rate_threshold_passed"] = passed.to_numpy()
        out.loc[indices, "rate_threshold_status"] = np.where(passed, "passed", "filtered")
        out.loc[indices, "rate_threshold_reason"] = reasons.to_numpy()

    return out


def build_final_selection_pool(
    *,
    pure_test_metrics_df: pd.DataFrame,
    track_scoring_flags_df: pd.DataFrame,
    candidate_panel_df: pd.DataFrame | None = None,
    pure_test_errors_df: pd.DataFrame | None = None,
    apply_previous_flag_filter: bool = expcfg.SIMCA_FINAL_APPLY_PREVIOUS_FLAG_FILTER,
    previous_flags_to_filter: Sequence[str] = expcfg.SIMCA_FINAL_PREVIOUS_FLAGS_TO_FILTER,
    exclude_pure_test_errors: bool = expcfg.SIMCA_FINAL_EXCLUDE_PURE_TEST_ERRORS,
) -> pd.DataFrame:
    """Build a compact final-selection pool without scores or threshold-based acceptance rules."""
    if pure_test_metrics_df is None or len(pure_test_metrics_df) == 0:
        raise ValueError("pure_test_metrics_df is empty.")
    if track_scoring_flags_df is None or len(track_scoring_flags_df) == 0:
        raise ValueError("track_scoring_flags_df is empty.")

    pool = select_track_primary_or_available_metrics(pure_test_metrics_df).reset_index(drop=True)
    validation = _prepare_validation_review(track_scoring_flags_df)
    pool["selected_config_id"] = pool["selected_config_id"].astype(str)
    pool = pool.merge(
        validation,
        on=["selected_config_id", "selection_track"],
        how="left",
        validate="many_to_one",
    )
    pool["previous_flags"] = pool["previous_flags"].fillna("missing_validation_review")
    pool["previous_flag_count"] = pool["previous_flags"].apply(lambda value: len(split_flag_string(value)))
    pool = _merge_candidate_panel(pool, candidate_panel_df)
    pool = _attach_pure_test_errors(pool, pure_test_errors_df)

    pool["previous_flag_filter_applied"] = bool(apply_previous_flag_filter)
    pool["filtered_by_previous_flags"] = (
        _flag_filter_mask(pool["previous_flags"], previous_flags_to_filter)
        if apply_previous_flag_filter
        else False
    )

    pool["preselection_status"] = "candidate"
    pool["filter_reason"] = ""
    if exclude_pure_test_errors:
        mask_error = pool["has_pure_test_error"].astype(bool)
        pool.loc[mask_error, "preselection_status"] = "filtered"
        pool.loc[mask_error, "filter_reason"] = "pure_test_refit_error"
    if apply_previous_flag_filter:
        mask_flags = pool["filtered_by_previous_flags"].astype(bool)
        pool.loc[mask_flags, "preselection_status"] = "filtered"
        pool.loc[mask_flags, "filter_reason"] = np.where(
            pool.loc[mask_flags, "filter_reason"].astype(str).eq(""),
            "previous_flag_filter",
            pool.loc[mask_flags, "filter_reason"].astype(str) + ";previous_flag_filter",
        )

    pool["selection_status"] = np.where(pool["preselection_status"].eq("candidate"), "not_selected", pool["preselection_status"])
    pool["is_final_selected"] = False
    return compact_simca_table(pool.reset_index(drop=True), table_kind="final_selection_pool")


def _is_applicable_diversity_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value) not in {"", "nan", "None", "not_applicable", "NA", "<NA>"}


def select_top_with_diversity(
    df: pd.DataFrame,
    *,
    top_n: int | None,
    diversity_columns: Sequence[str] = expcfg.SIMCA_FINAL_DIVERSITY_COLUMNS,
    cross_track_dedup_col: str = expcfg.SIMCA_FINAL_CROSS_TRACK_DEDUP_COL,
    excluded_dedup_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Select from an already Pareto-sorted table while encouraging configuration diversity."""
    if df is None or len(df) == 0 or (top_n is not None and int(top_n) <= 0):
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()

    ranked = df.reset_index(drop=True).copy()
    excluded_dedup_ids = excluded_dedup_ids or set()
    if cross_track_dedup_col in ranked.columns and excluded_dedup_ids:
        ranked = ranked.loc[~ranked[cross_track_dedup_col].astype(str).isin(excluded_dedup_ids)].reset_index(drop=True)
    if len(ranked) == 0:
        return ranked

    if top_n is None:
        return ranked.reset_index(drop=True)

    selected_indices: list[int] = []

    def add_index(idx: int) -> None:
        if idx not in selected_indices and len(selected_indices) < int(top_n):
            selected_indices.append(idx)

    add_index(0)

    for col in diversity_columns:
        if col not in ranked.columns or len(selected_indices) >= int(top_n):
            continue
        values = [str(value) for value in ranked[col] if _is_applicable_diversity_value(value)]
        if len(set(values)) <= 1:
            continue
        selected_values = {
            str(ranked.loc[idx, col])
            for idx in selected_indices
            if _is_applicable_diversity_value(ranked.loc[idx, col])
        }
        for idx, value in ranked[col].items():
            if idx in selected_indices:
                continue
            if _is_applicable_diversity_value(value) and str(value) not in selected_values:
                add_index(int(idx))
                break

    for idx in ranked.index:
        if len(selected_indices) >= int(top_n):
            break
        add_index(int(idx))

    return ranked.loc[selected_indices].copy().reset_index(drop=True)


def _annotate_track_pareto(group: pd.DataFrame) -> pd.DataFrame:
    if group is None or len(group) == 0:
        return pd.DataFrame() if group is None else group.copy()
    objectives = pareto_objective_columns(group)
    tiered = assign_pareto_tiers(group, minimize_cols=objectives)
    tiered = sort_by_pareto(tiered)
    tiered["pareto_rank_in_track"] = np.arange(1, len(tiered) + 1)
    return tiered


def _set_pareto_stage_columns(
    out: pd.DataFrame,
    *,
    stage: str,
    track_order: Sequence[str],
    eligible_mask: pd.Series,
    objective_builder,
) -> pd.DataFrame:
    """Annotate one pairwise Pareto-filter stage without ranking by a score."""
    tier_col = f"{stage}_pareto_tier"
    front_col = f"{stage}_pareto_front"
    status_col = f"{stage}_pareto_status"
    objectives_col = f"{stage}_pareto_objectives"

    out = out.copy()
    out[tier_col] = pd.NA
    out[front_col] = pd.NA
    out[status_col] = "not_applicable"
    out[objectives_col] = ""

    for track in track_order:
        track_mask = out["selection_track"].astype(str).eq(str(track)) & eligible_mask
        indices = out.index[track_mask].tolist()
        if not indices:
            continue
        group = out.loc[indices].copy()
        objectives = list(objective_builder(group))
        tiers = assign_pareto_tiers(group, minimize_cols=objectives)
        front = pareto_front_mask(group, minimize_cols=objectives)
        out.loc[indices, tier_col] = tiers["pareto_tier"].to_numpy()
        out.loc[indices, front_col] = front.to_numpy()
        out.loc[indices, status_col] = np.where(front.to_numpy(), "kept", "pareto_dominated")
        out.loc[indices, objectives_col] = ";".join(objectives)

    out[tier_col] = out[tier_col].astype("Int64")
    out[front_col] = out[front_col].astype("boolean")
    return out


def apply_metric_pareto_filter_by_track(
    pool_df: pd.DataFrame,
    *,
    track_order: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
) -> pd.DataFrame:
    """First filter: remove candidates dominated on the track error metrics."""
    if pool_df is None or len(pool_df) == 0:
        return pd.DataFrame() if pool_df is None else pool_df.copy()
    if "selection_track" not in pool_df.columns:
        raise ValueError("pool_df must contain selection_track.")

    eligible = pool_df["preselection_status"].astype(str).eq("candidate")
    if "rate_threshold_passed" in pool_df.columns:
        eligible &= pool_df["rate_threshold_passed"].fillna(False).astype(bool)
    return _set_pareto_stage_columns(
        pool_df,
        stage="metric",
        track_order=track_order,
        eligible_mask=eligible,
        objective_builder=lambda group: pareto_objective_columns(group),
    )


def _flag_objective_name(flag: str, used: set[str]) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(flag)).strip("_") or "unnamed"
    candidate = f"flag_pareto__{safe}"
    suffix = 2
    while candidate in used:
        candidate = f"flag_pareto__{safe}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def apply_flag_pareto_filter_by_track(
    pool_df: pd.DataFrame,
    *,
    track_order: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
) -> pd.DataFrame:
    """Second filter: remove models dominated on previous-notebook flags.

    Each distinct flag is a separate binary objective to minimize. A model is
    dominated only when another model has no flag that is worse and has at
    least one strictly better flag, within the same track and metric Pareto
    front. No flag count, score, or rating is used.
    """
    if pool_df is None or len(pool_df) == 0:
        return pd.DataFrame() if pool_df is None else pool_df.copy()

    out = pool_df.copy()
    if "previous_flags" not in out.columns:
        out["previous_flags"] = ""

    all_flags = sorted(
        {
            flag
            for value in out["previous_flags"]
            for flag in split_flag_string(value)
        }
    )
    used: set[str] = set(out.columns)
    flag_columns = [_flag_objective_name(flag, used) for flag in all_flags]
    flag_to_column = dict(zip(all_flags, flag_columns))
    for flag, column in flag_to_column.items():
        out[column] = out["previous_flags"].apply(
            lambda value, expected=flag: int(expected in set(split_flag_string(value)))
        )

    eligible = (
        out["preselection_status"].astype(str).eq("candidate")
        & out.get("rate_threshold_passed", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        & out.get("metric_pareto_front", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )

    if flag_columns:
        stage = _set_pareto_stage_columns(
            out,
            stage="flag",
            track_order=track_order,
            eligible_mask=eligible,
            objective_builder=lambda group: flag_columns,
        )
    else:
        stage = out.copy()
        stage["flag_pareto_tier"] = pd.Series(1, index=stage.index, dtype="Int64").where(eligible)
        stage["flag_pareto_front"] = eligible.astype("boolean")
        stage["flag_pareto_status"] = np.where(eligible, "kept", "not_applicable")
        stage["flag_pareto_objectives"] = ""

    stage["pareto_tier"] = stage["flag_pareto_tier"]
    stage["is_pareto_front"] = stage["flag_pareto_front"].fillna(False).astype(bool)
    return stage


def format_final_selection_reason(row: pd.Series) -> str:
    """Build a compact, non-score-based reason for final selection tables."""
    parts = []
    if pd.notna(row.get("pareto_tier", pd.NA)):
        parts.append(f"pareto_tier={int(row.get('pareto_tier'))}")
    if pd.notna(row.get("pareto_rank_in_track", pd.NA)):
        parts.append(f"pareto_rank={int(row.get('pareto_rank_in_track'))}")
    for metric in (
        "fn_rate",
        "fp_rate",
        "target_miss_rate",
        "non_target_false_accept_rate",
        "uncertain_rate",
        "balanced_accuracy",
        "coverage_rate",
        "decided_balanced_accuracy",
    ):
        if metric in row.index and pd.notna(row.get(metric)):
            parts.append(f"{metric}={float(row.get(metric)):.4f}")
    if pd.notna(row.get("previous_flags")) and str(row.get("previous_flags")):
        parts.append(f"previous_flags={row.get('previous_flags')}")
    if pd.notna(row.get("filter_reason")) and str(row.get("filter_reason")):
        parts.append(f"filter_reason={row.get('filter_reason')}")
    return "; ".join(parts)


def select_final_models_by_track(
    pool_df: pd.DataFrame,
    *,
    top_n_per_track: int | None = expcfg.SIMCA_FINAL_TOP_N_PER_TRACK,
    fn_rate_max: float | None = expcfg.SIMCA_FINAL_FN_RATE_MAX,
    fp_rate_max: float | None = expcfg.SIMCA_FINAL_FP_RATE_MAX,
    uncertain_rate_max: float | None = expcfg.SIMCA_FINAL_UNCERTAIN_RATE_MAX,
    apply_diversity: bool = expcfg.SIMCA_FINAL_APPLY_DIVERSITY,
    diversity_columns: Sequence[str] = expcfg.SIMCA_FINAL_DIVERSITY_COLUMNS,
    deduplicate_across_tracks: bool = expcfg.SIMCA_FINAL_DEDUPLICATE_ACROSS_TRACKS,
    cross_track_dedup_col: str = expcfg.SIMCA_FINAL_CROSS_TRACK_DEDUP_COL,
    track_order: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
    require_all_tracks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply both Pareto filters and return the surviving models by track.

    ``top_n_per_track=None`` keeps the complete second-stage Pareto front.
    An integer is an optional display/export cap applied only after both
    pairwise filters have completed.
    """
    if pool_df is None or len(pool_df) == 0:
        raise ValueError("pool_df is empty.")

    staged = apply_rate_threshold_filter_by_track(
        pool_df,
        fn_rate_max=fn_rate_max,
        fp_rate_max=fp_rate_max,
        uncertain_rate_max=uncertain_rate_max,
        track_order=track_order,
    )
    staged = apply_metric_pareto_filter_by_track(staged, track_order=track_order)
    staged = apply_flag_pareto_filter_by_track(staged, track_order=track_order)
    staged["selection_status"] = "not_selected"
    staged["is_final_selected"] = False
    staged["top_n_limit_applied"] = False
    staged["deduplicate_across_tracks"] = bool(deduplicate_across_tracks)
    staged["cross_track_dedup_col"] = cross_track_dedup_col
    staged["cross_track_deduplication_status"] = "not_applied"

    annotated_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    used_dedup_ids: set[str] = set()
    diversity_text = ",".join(map(str, diversity_columns))

    for track in track_order:
        track_mask = staged["selection_track"].astype(str).eq(str(track))
        track_df = staged.loc[track_mask].copy()
        if len(track_df) == 0:
            continue

        track_df["pareto_rank_in_track"] = pd.NA
        eligible = (
            track_df["preselection_status"].astype(str).eq("candidate")
            & track_df["metric_pareto_front"].fillna(False).astype(bool)
            & track_df["flag_pareto_front"].fillna(False).astype(bool)
        )
        track_df.loc[eligible, "pareto_rank_in_track"] = np.arange(1, int(eligible.sum()) + 1)
        track_df.loc[track_df["preselection_status"].astype(str).ne("candidate"), "selection_status"] = (
            track_df.loc[track_df["preselection_status"].astype(str).ne("candidate"), "preselection_status"].astype(str)
        )
        track_df.loc[
            track_df["preselection_status"].astype(str).eq("candidate")
            & ~track_df["rate_threshold_passed"].fillna(False).astype(bool),
            "selection_status",
        ] = "rate_threshold_filtered"
        track_df.loc[
            track_df["selection_status"].eq("rate_threshold_filtered"),
            "filter_reason",
        ] = track_df.loc[
            track_df["selection_status"].eq("rate_threshold_filtered"),
            "rate_threshold_reason",
        ]
        track_df.loc[
            track_df["preselection_status"].astype(str).eq("candidate")
            & track_df["rate_threshold_passed"].fillna(False).astype(bool)
            & ~track_df["metric_pareto_front"].fillna(False).astype(bool),
            "selection_status",
        ] = "metric_pareto_dominated"
        track_df.loc[track_df["selection_status"].eq("metric_pareto_dominated"), "filter_reason"] = "metric_pareto_dominated"
        track_df.loc[
            track_df["metric_pareto_front"].fillna(False).astype(bool)
            & ~track_df["flag_pareto_front"].fillna(False).astype(bool),
            "selection_status",
        ] = "flag_pareto_dominated"
        track_df.loc[track_df["selection_status"].eq("flag_pareto_dominated"), "filter_reason"] = "flag_pareto_dominated"

        candidates = track_df.loc[eligible].copy().sort_values("pareto_rank_in_track")
        candidates["deduplication_id"] = (
            candidates[cross_track_dedup_col].astype(str)
            if cross_track_dedup_col in candidates.columns
            else candidates["selected_config_id"].astype(str)
        )
        track_df["deduplication_id"] = pd.NA
        track_df.loc[candidates.index, "deduplication_id"] = candidates["deduplication_id"]
        available = candidates.copy()
        if deduplicate_across_tracks:
            duplicate_mask = available["deduplication_id"].isin(used_dedup_ids)
            track_df.loc[available.index[duplicate_mask], "selection_status"] = "duplicate_across_tracks"
            track_df.loc[available.index[duplicate_mask], "cross_track_deduplication_status"] = "already_selected_in_previous_track"
            available = available.loc[~duplicate_mask].copy()

        if apply_diversity:
            selected = select_top_with_diversity(
                available,
                top_n=top_n_per_track,
                diversity_columns=diversity_columns,
                cross_track_dedup_col=cross_track_dedup_col,
            )
        else:
            selected = available.copy() if top_n_per_track is None else available.head(int(top_n_per_track)).copy()
        selected = selected.reset_index(drop=False)
        selected_indices = selected["index"].tolist() if len(selected) else []

        if top_n_per_track is not None and len(available) > len(selected):
            not_selected = available.index.difference(selected_indices)
            track_df.loc[not_selected, "selection_status"] = "top_n_limit"
            track_df.loc[not_selected, "top_n_limit_applied"] = True

        if selected_indices:
            track_df.loc[selected_indices, "selection_status"] = "selected"
            track_df.loc[selected_indices, "is_final_selected"] = True
            track_df.loc[selected_indices, "final_rank_in_track"] = np.arange(1, len(selected_indices) + 1)
            track_df.loc[selected_indices, "diversity_rule_applied"] = bool(apply_diversity)
            track_df.loc[selected_indices, "diversity_columns"] = diversity_text
            track_df.loc[selected_indices, "diversity_reason"] = (
                "greedy_diversity_after_pareto_order" if apply_diversity else "pareto_order_only"
            )
            track_df.loc[selected_indices, "assigned_selection_track"] = str(track)
            track_df.loc[selected_indices, "cross_track_deduplication_status"] = (
                "kept_unique_assignment" if deduplicate_across_tracks else "not_applied"
            )
            selected_rows = track_df.loc[selected_indices].copy().sort_values("pareto_rank_in_track")
            selected_rows["final_rank_in_track"] = np.arange(1, len(selected_rows) + 1)
            selected_rows["selection_reason"] = [
                format_final_selection_reason(row) for _, row in selected_rows.iterrows()
            ]
            selected_parts.append(selected_rows)
            if deduplicate_across_tracks:
                used_dedup_ids.update(selected_rows["deduplication_id"].dropna().astype(str).tolist())

        track_df["selection_reason"] = [format_final_selection_reason(row) for _, row in track_df.iterrows()]
        annotated_parts.append(track_df)

    pool_annotated = (
        pd.concat(annotated_parts, ignore_index=True, sort=False)
        if annotated_parts
        else pool_df.iloc[0:0].copy()
    )
    selected_df = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pool_df.iloc[0:0].copy()
    )

    if require_all_tracks:
        observed_tracks = set(selected_df.get("selection_track", pd.Series(dtype=str)).dropna().astype(str))
        missing_tracks = sorted(set(map(str, track_order)) - observed_tracks)
        if missing_tracks:
            raise RuntimeError(f"No final model selected for track(s): {missing_tracks}")

    selected_df = compact_simca_table(selected_df.reset_index(drop=True), table_kind="final_selected_models")
    pool_annotated = compact_simca_table(pool_annotated.reset_index(drop=True), table_kind="final_selection_pool")
    summary_df = summarize_final_selection(pool_annotated)
    validate_final_selection_contract(selected_df, require_all_tracks=require_all_tracks, expected_tracks=track_order)
    return selected_df, pool_annotated, summary_df


def summarize_final_selection(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final-selection statuses without duplicating the full pool."""
    if pool_df is None or len(pool_df) == 0:
        return pd.DataFrame()
    out = (
        pool_df.groupby(
            ["selection_track", "selection_status", "preselection_status", "pareto_tier"],
            dropna=False,
        )
        .agg(
            n_rows=("selected_config_id", "size"),
            n_selected=("is_final_selected", "sum"),
        )
        .reset_index()
        .sort_values(["selection_track", "selection_status", "pareto_tier"], na_position="last")
        .reset_index(drop=True)
    )
    return compact_simca_table(out, table_kind="final_selection_summary")


def validate_final_selection_contract(
    selected_df: pd.DataFrame,
    *,
    require_all_tracks: bool = True,
    expected_tracks: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
) -> pd.DataFrame:
    """Validate the final model-selection table contract."""
    validate_simca_table_columns(
        selected_df,
        expcfg.SIMCA_FINAL_MODEL_SELECTION_REQUIRED_COLUMNS,
        table_name="SIMCA final model selection table",
    )
    if require_all_tracks:
        observed = set(selected_df["selection_track"].dropna().astype(str))
        missing = sorted(set(map(str, expected_tracks)) - observed)
        if missing:
            raise ValueError(f"Missing final selected model track(s): {missing}")
    return selected_df


def build_final_selection_protocol(settings: Mapping[str, Any], outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the notebook-06B protocol table."""
    row = dict(settings)
    for key, value in list(row.items()):
        if isinstance(value, Path):
            row[key] = str(value)
        elif isinstance(value, (dict, list, tuple, set)):
            row[key] = json.dumps(value, default=str)
    pool = outputs.get("pool", pd.DataFrame())
    selected = outputs.get("selected", pd.DataFrame())
    row.update(
        {
            "n_pool_rows": int(len(pool)),
            "n_candidate_rows": int(pool.get("preselection_status", pd.Series(dtype=str)).astype(str).eq("candidate").sum()),
            "n_selected_rows": int(len(selected)),
            "n_selected_tracks": int(selected.get("selection_track", pd.Series(dtype=str)).nunique()),
        }
    )
    return compact_simca_table(pd.DataFrame([row]), table_kind="final_selection_protocol")


def save_final_selection_outputs(
    *,
    pool_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    guardrails_df: pd.DataFrame,
    protocol_df: pd.DataFrame,
    paths: Mapping[str, str | Path],
) -> list[Path]:
    """Save the standard notebook-06B Pareto-selection output set."""
    saved: list[Path] = []
    for key, df in [
        ("pool", pool_df),
        ("selected", selected_df),
        ("summary", summary_df),
        ("guardrails", guardrails_df),
        ("protocol", protocol_df),
    ]:
        if key in paths:
            saved.append(write_simca_table(df, paths[key]))
    return saved
