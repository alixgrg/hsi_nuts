from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.workflows.simca_candidates import (
    validate_simca_evaluation_contract,
    validate_simca_selection_tracks,
)
from src.workflows.simca_selection_utils import (
    pareto_front_by_group,
    summarize_ablation_effects,
    summarize_metric_stability,
)
from src.workflows.simca_tables import compact_simca_table


DEFAULT_ROBUSTNESS_METRICS = (
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
    "target_sensitivity",
    "non_target_specificity",
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "screening_sensitivity",
    "decided_balanced_accuracy",
    "robustness_score",
)


def _series(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _numeric(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    return pd.to_numeric(_series(df, col, default), errors="coerce")


def _mode_series(df: pd.DataFrame) -> pd.Series:
    mode = _series(df, "decision_mode", "").astype("string").str.lower()
    if "selection_track" in df.columns:
        track = df["selection_track"].astype("string").str.lower()
        mode = mode.mask(mode.eq("") | mode.isna(), np.where(track.str.contains("3way", na=False), "3way", "2way"))
    return mode.fillna("")


def _flag_table(flags_by_row: list[list[str]], index) -> pd.DataFrame:
    flags = [";".join(flags) for flags in flags_by_row]
    return pd.DataFrame(
        {
            "robustness_flags": flags,
            "robustness_flag_count": [len(flags) for flags in flags_by_row],
        },
        index=index,
    )


def validate_no_pure_test_inputs(
    df: pd.DataFrame,
    stage_cols: Sequence[str] = ("evaluation_stage", "evaluation_split"),
) -> pd.DataFrame:
    """Reject pure-test or held-out-test tables from the robustness notebook."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    forbidden = ("pure_test", "heldout_test", "held_out_test", "external_test")
    for col in stage_cols:
        if col not in out.columns:
            continue
        values = out[col].astype("string").str.lower().fillna("")
        bad = values.apply(lambda value: any(token in value for token in forbidden))
        if bool(bad.any()):
            examples = sorted(values.loc[bad].dropna().unique().tolist())[:5]
            raise ValueError(
                "Notebook 05 must not consume pure-test tables. "
                f"Forbidden values found in {col}: {examples}"
            )
    return out


def validate_simca_robustness_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the common evaluation-table contract used by notebook 05."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()
    out = validate_no_pure_test_inputs(df)
    out = validate_simca_evaluation_contract(out)
    out = validate_simca_selection_tracks(out)
    return out


def select_track_primary_or_available_metrics(
    df: pd.DataFrame,
    track_specs: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Keep one metric level per track.

    The configured primary level is used when present. If it is absent, the
    secondary level is used. If neither is present, all available rows for the
    track are retained. This fallback matters for 3-way metrics, which are
    currently object-level even for pixel-matrix models.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()
    if "selection_track" not in df.columns or "metric_level" not in df.columns:
        return df.copy()

    specs = track_specs or expcfg.SIMCA_SELECTION_TRACK_SPECS
    parts: list[pd.DataFrame] = []
    for track, group in df.groupby("selection_track", dropna=False):
        track_name = str(track)
        spec = specs.get(track_name, {})
        metric_levels = group["metric_level"].astype("string")
        primary = spec.get("primary_metric_level")
        secondary = spec.get("secondary_metric_level")

        if primary and metric_levels.eq(primary).any():
            parts.append(group.loc[metric_levels.eq(primary)].copy())
        elif secondary and metric_levels.eq(secondary).any():
            parts.append(group.loc[metric_levels.eq(secondary)].copy())
        else:
            parts.append(group.copy())

    return pd.concat(parts, ignore_index=True, sort=False) if parts else df.iloc[0:0].copy()


def add_simca_robustness_scores(
    df: pd.DataFrame,
    warning_thresholds: Mapping[str, float] | None = None,
    two_way_weights: Mapping[str, float] | None = None,
    three_way_weights: Mapping[str, float] | None = None,
    score_col: str = "robustness_score",
) -> pd.DataFrame:
    """Add notebook-05 robustness score, within-track ranks, and warning flags."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    out = out.drop(
        columns=[
            score_col,
            "robustness_flags",
            "robustness_flag_count",
            "robustness_rank_in_track",
        ],
        errors="ignore",
    )
    thresholds = dict(expcfg.SIMCA_ROBUSTNESS_WARNING_THRESHOLDS)
    if warning_thresholds:
        thresholds.update(warning_thresholds)
    weights_2way = dict(expcfg.SIMCA_ROBUSTNESS_2WAY_SCORE_WEIGHTS)
    if two_way_weights:
        weights_2way.update(two_way_weights)
    weights_3way = dict(expcfg.SIMCA_ROBUSTNESS_3WAY_SCORE_WEIGHTS)
    if three_way_weights:
        weights_3way.update(three_way_weights)

    fn_rate = _numeric(out, "fn_rate", np.nan)
    fp_rate = _numeric(out, "fp_rate", np.nan)
    balanced_accuracy = _numeric(out, "balanced_accuracy", np.nan)
    target_miss = _numeric(out, "target_miss_rate", fn_rate)
    false_accept = _numeric(out, "non_target_false_accept_rate", fp_rate)
    uncertain = _numeric(out, "uncertain_rate", 0.0).fillna(0.0)
    coverage = _numeric(out, "coverage_rate", 1.0 - uncertain)
    screening_sensitivity = _numeric(out, "screening_sensitivity", 1.0 - target_miss)
    decided_balanced_accuracy = _numeric(out, "decided_balanced_accuracy", balanced_accuracy)

    mode = _mode_series(out)
    is_3way = mode.eq("3way")

    two_way_score = (
        weights_2way.get("fn_rate", 0.0) * fn_rate.fillna(1.0)
        + weights_2way.get("fp_rate", 0.0) * fp_rate.fillna(1.0)
        + weights_2way.get("balanced_accuracy", 0.0) * balanced_accuracy.fillna(0.0)
    )
    three_way_score = (
        weights_3way.get("target_miss_rate", 0.0) * target_miss.fillna(1.0)
        + weights_3way.get("non_target_false_accept_rate", 0.0) * false_accept.fillna(1.0)
        + weights_3way.get("uncertain_rate", 0.0) * uncertain.fillna(1.0)
        + weights_3way.get("coverage_rate", 0.0) * coverage.fillna(0.0)
        + weights_3way.get("screening_sensitivity", 0.0) * screening_sensitivity.fillna(0.0)
        + weights_3way.get("decided_balanced_accuracy", 0.0) * decided_balanced_accuracy.fillna(0.0)
    )

    out[score_col] = np.where(is_3way, three_way_score, two_way_score)

    flags_by_row: list[list[str]] = []
    for idx in out.index:
        flags: list[str] = []
        row_is_3way = bool(is_3way.loc[idx])
        if row_is_3way:
            if target_miss.loc[idx] > thresholds["target_miss_rate"]:
                flags.append("high_target_miss_rate")
            if false_accept.loc[idx] > thresholds["non_target_false_accept_rate"]:
                flags.append("high_non_target_false_accept_rate")
            if uncertain.loc[idx] > thresholds["uncertain_rate"]:
                flags.append("high_uncertain_rate")
            if coverage.loc[idx] < thresholds["coverage_rate"]:
                flags.append("low_coverage_rate")
            if decided_balanced_accuracy.loc[idx] < thresholds["decided_balanced_accuracy"]:
                flags.append("low_decided_balanced_accuracy")
        else:
            if fn_rate.loc[idx] > thresholds["fn_rate"]:
                flags.append("high_fn_rate")
            if fp_rate.loc[idx] > thresholds["fp_rate"]:
                flags.append("high_fp_rate")
            if balanced_accuracy.loc[idx] < thresholds["balanced_accuracy"]:
                flags.append("low_balanced_accuracy")
        if not np.isfinite(float(out.loc[idx, score_col])):
            flags.append("missing_robustness_score")
        flags_by_row.append(flags)

    out = pd.concat([out, _flag_table(flags_by_row, out.index)], axis=1)

    if "selection_track" in out.columns:
        out["robustness_rank_in_track"] = (
            out.groupby("selection_track", dropna=False)[score_col]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )
    else:
        out["robustness_rank_in_track"] = out[score_col].rank(method="dense", ascending=False).astype("Int64")

    return out


def _objective_columns_for_mode(
    df: pd.DataFrame,
    decision_mode: str,
    minimize_cols: Sequence[str] | None,
    maximize_cols: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    if minimize_cols is not None or maximize_cols is not None:
        mins = list(minimize_cols or ())
        maxs = list(maximize_cols or ())
    elif str(decision_mode).lower() == "3way":
        mins = list(expcfg.SIMCA_ROBUSTNESS_3WAY_PARETO_MINIMIZE_COLUMNS)
        maxs = list(expcfg.SIMCA_ROBUSTNESS_3WAY_PARETO_MAXIMIZE_COLUMNS)
    else:
        mins = list(expcfg.SIMCA_ROBUSTNESS_2WAY_PARETO_MINIMIZE_COLUMNS)
        maxs = list(expcfg.SIMCA_ROBUSTNESS_2WAY_PARETO_MAXIMIZE_COLUMNS)

    mins = [col for col in mins if col in df.columns]
    maxs = [col for col in maxs if col in df.columns]

    if not mins and not maxs:
        raise ValueError("No Pareto objective columns are present in the dataframe.")
    return mins, maxs


def build_pareto_diagnostics(
    df: pd.DataFrame,
    decision_mode: str,
    group_cols: Sequence[str] = ("selection_track", "matrix_family"),
    minimize_cols: Sequence[str] | None = None,
    maximize_cols: Sequence[str] | None = None,
    epsilon: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Pareto-front diagnostics for 2-way or 3-way validation metrics."""
    if df is None or len(df) == 0:
        empty = pd.DataFrame() if df is None else df.copy()
        return empty, empty, pd.DataFrame()

    d = add_simca_robustness_scores(df)
    mode = _mode_series(d)
    d = d.loc[mode.eq(str(decision_mode).lower())].copy()
    if d.empty:
        return d, d, pd.DataFrame()

    d = d.reset_index(drop=True)
    d["_pareto_row_id"] = np.arange(len(d))
    mins, maxs = _objective_columns_for_mode(d, decision_mode, minimize_cols, maximize_cols)
    groups = [col for col in group_cols if col in d.columns]
    eps = expcfg.SIMCA_ROBUSTNESS_PARETO_EPSILON if epsilon is None else float(epsilon)

    front = pareto_front_by_group(
        d,
        group_cols=groups,
        minimize_cols=mins,
        maximize_cols=maxs,
        epsilon=eps,
    ).copy()
    keep_ids = set(front["_pareto_row_id"].tolist())
    mode_token = str(decision_mode).lower()
    d[f"is_pareto_{mode_token}"] = d["_pareto_row_id"].isin(keep_ids)
    d[f"pareto_{mode_token}_reason"] = np.where(
        d[f"is_pareto_{mode_token}"],
        "non_dominated",
        "dominated_within_track_family",
    )

    audit_rows = []
    grouped = d.groupby(groups, dropna=False) if groups else [((), d)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(groups, key)}
        row.update(
            {
                "decision_mode": mode_token,
                "pareto_minimize_columns": ",".join(mins),
                "pareto_maximize_columns": ",".join(maxs),
                "n_before": int(len(group)),
                "n_pareto": int(group[f"is_pareto_{mode_token}"].sum()),
                "n_dominated": int((~group[f"is_pareto_{mode_token}"]).sum()),
            }
        )
        audit_rows.append(row)

    annotated = d.drop(columns=["_pareto_row_id"])
    front = front.drop(columns=["_pareto_row_id"])
    table_kind = "pareto_3way" if str(decision_mode).lower() == "3way" else "pareto_2way"
    front = compact_simca_table(front, table_kind=table_kind)
    annotated = compact_simca_table(annotated, table_kind=table_kind)
    audit = compact_simca_table(pd.DataFrame(audit_rows), table_kind="pareto_audit")
    return front.reset_index(drop=True), annotated.reset_index(drop=True), audit


def build_ablation_diagnostics(
    df: pd.DataFrame,
    factor_cols: Sequence[str] | None = None,
    group_cols: Sequence[str] = ("selection_track", "matrix_family", "decision_mode", "metric_level"),
    metric_cols: Sequence[str] = DEFAULT_ROBUSTNESS_METRICS,
) -> pd.DataFrame:
    """Summarize one-factor-at-a-time hyperparameter effects from 04C metrics."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    scored = add_simca_robustness_scores(df)
    factors = tuple(factor_cols or expcfg.SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS)
    out = summarize_ablation_effects(
        scored,
        factor_cols=factors,
        group_cols=group_cols,
        metric_cols=metric_cols,
    )
    return compact_simca_table(out, table_kind="ablation_diagnostics")


def build_random_state_stability_panel(
    candidate_panel_df: pd.DataFrame,
    scored_metrics_df: pd.DataFrame,
    max_per_track: int | None = None,
    prefer_balanced_pixels: bool | None = None,
    id_col: str = "selected_config_id",
) -> pd.DataFrame:
    """Prepare the compact candidate panel to refit over several random states."""
    if scored_metrics_df is None or len(scored_metrics_df) == 0:
        return pd.DataFrame()

    max_n = (
        expcfg.SIMCA_ROBUSTNESS_MAX_STABILITY_CANDIDATES_PER_TRACK
        if max_per_track is None
        else int(max_per_track)
    )
    prefer_balanced = (
        expcfg.SIMCA_ROBUSTNESS_PREFER_BALANCED_PIXELS_FOR_STABILITY
        if prefer_balanced_pixels is None
        else bool(prefer_balanced_pixels)
    )

    metrics = select_track_primary_or_available_metrics(
        add_simca_robustness_scores(scored_metrics_df)
    ).copy()
    if id_col not in metrics.columns:
        raise KeyError(f"Missing {id_col!r} in scored_metrics_df.")

    metrics["_balanced_pixel_priority"] = _series(metrics, "matrix_method", "").astype("string").eq("balanced_pixels")
    sort_cols = [
        "selection_track",
        "_balanced_pixel_priority",
        "robustness_score",
        "fn_rate",
        "fp_rate",
        "balanced_accuracy",
    ]
    sort_cols = [col for col in sort_cols if col in metrics.columns]
    ascending = []
    for col in sort_cols:
        if col == "_balanced_pixel_priority":
            ascending.append(not prefer_balanced)
        elif col in {"robustness_score", "balanced_accuracy"}:
            ascending.append(False)
        else:
            ascending.append(True)

    ranked = metrics.sort_values(sort_cols, ascending=ascending).copy()
    if "selection_track" in ranked.columns and max_n > 0:
        ranked = ranked.groupby("selection_track", group_keys=False, dropna=False).head(max_n)
    ranked = ranked.drop(columns=["_balanced_pixel_priority"], errors="ignore")

    panel_cols = [
        id_col,
        "selection_track",
        "decision_mode",
        "metric_level",
        "robustness_score",
        "robustness_rank_in_track",
        "robustness_flags",
    ]
    panel_cols = [col for col in panel_cols if col in ranked.columns]
    panel = ranked[panel_cols].drop_duplicates().copy()

    if candidate_panel_df is not None and len(candidate_panel_df) > 0 and id_col in candidate_panel_df.columns:
        candidate_cols = [col for col in candidate_panel_df.columns if col not in panel.columns or col == id_col]
        panel = candidate_panel_df[candidate_cols].merge(panel, on=id_col, how="inner", validate="one_to_many")

    if "selection_track" in panel.columns:
        track_summary = (
            panel.groupby(id_col, dropna=False)["selection_track"]
            .agg(
                stability_selection_tracks=lambda values: ",".join(sorted(values.astype(str).dropna().unique())),
                n_stability_selection_tracks="nunique",
            )
            .reset_index()
        )
        panel = panel.merge(track_summary, on=id_col, how="left")

    panel = (
        panel.sort_values(
            [col for col in ["selection_track", "robustness_rank_in_track", "robustness_score", id_col] if col in panel.columns],
            ascending=[True, True, False, True][: len([col for col in ["selection_track", "robustness_rank_in_track", "robustness_score", id_col] if col in panel.columns])],
        )
        .drop_duplicates(id_col)
        .reset_index(drop=True)
    )
    panel["stability_panel_reason"] = "top_robustness_candidates_per_track"
    return compact_simca_table(panel, table_kind="random_state_stability_panel")


def summarize_random_state_stability_metrics(
    stability_metrics_df: pd.DataFrame,
    config_cols: Sequence[str] | None = None,
    warning_thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Summarize random-state variability and add stability warning flags."""
    if stability_metrics_df is None or len(stability_metrics_df) == 0:
        return pd.DataFrame()

    default_config_cols = (
        "selected_config_id",
        "candidate_id",
        "matrix_family",
        "matrix_method",
        "training_matrix_id",
        "balanced_pixel_strategy_effective",
        "preprocessing",
        "rule_variant",
        "n_components",
        "object_threshold",
    )
    cfg_cols = tuple(config_cols or default_config_cols)
    summary = summarize_metric_stability(
        add_simca_robustness_scores(stability_metrics_df),
        config_cols=cfg_cols,
        metric_cols=(
            "fn_rate",
            "fp_rate",
            "balanced_accuracy",
            "target_sensitivity",
            "non_target_specificity",
            "robustness_score",
        ),
        seed_col="random_state",
    )

    thresholds = dict(expcfg.SIMCA_ROBUSTNESS_WARNING_THRESHOLDS)
    if warning_thresholds:
        thresholds.update(warning_thresholds)

    flags_by_row: list[list[str]] = []
    for _, row in summary.iterrows():
        flags: list[str] = []
        if pd.to_numeric(pd.Series([row.get("std_fn_rate")]), errors="coerce").iloc[0] > thresholds["std_fn_rate"]:
            flags.append("unstable_fn_rate")
        if pd.to_numeric(pd.Series([row.get("std_fp_rate")]), errors="coerce").iloc[0] > thresholds["std_fp_rate"]:
            flags.append("unstable_fp_rate")
        if pd.to_numeric(pd.Series([row.get("std_balanced_accuracy")]), errors="coerce").iloc[0] > thresholds["std_balanced_accuracy"]:
            flags.append("unstable_balanced_accuracy")
        flags_by_row.append(flags)

    out = pd.concat(
        [
            summary.reset_index(drop=True),
            _flag_table(flags_by_row, summary.index).reset_index(drop=True).rename(
                columns={
                    "robustness_flags": "stability_flags",
                    "robustness_flag_count": "stability_flag_count",
                }
            ),
        ],
        axis=1,
    )
    return compact_simca_table(out, table_kind="random_state_stability_summary")


def build_border_core_skip_table(reason: str, pixel_batch_dir: str | Path | None = None) -> pd.DataFrame:
    """Return a one-row diagnostic table when border/core cannot be computed."""
    out = pd.DataFrame(
        [
            {
                "border_core_status": "skipped",
                "skip_reason": str(reason),
                "pixel_batch_dir": "" if pixel_batch_dir is None else str(pixel_batch_dir),
                "required_04c_setting": "SAVE_BATCH_PIXEL_TABLES=True",
            }
        ]
    )
    return compact_simca_table(out, table_kind="border_core_status")


def build_border_core_diagnostics(
    pixel_df: pd.DataFrame,
    object_db: dict[str, dict],
    border_widths: Sequence[int] | None = None,
    object_thresholds: Sequence[float] | None = None,
    max_configs: int | None = None,
    min_core_pixels: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate object metrics after excluding border pixels for each config."""
    if pixel_df is None or len(pixel_df) == 0:
        return pd.DataFrame(), build_border_core_skip_table("empty_pixel_table")
    if "selected_config_id" not in pixel_df.columns:
        return pd.DataFrame(), build_border_core_skip_table("missing_selected_config_id")

    from src.decision.border import border_width_object_threshold_grid

    widths = tuple(border_widths or expcfg.SIMCA_ROBUSTNESS_BORDER_WIDTHS)
    min_core = expcfg.SIMCA_ROBUSTNESS_MIN_CORE_PIXELS if min_core_pixels is None else int(min_core_pixels)
    parts: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []

    grouped = pixel_df.groupby("selected_config_id", dropna=False)
    for idx, (config_id, group) in enumerate(grouped):
        if max_configs is not None and idx >= int(max_configs):
            break
        target_class = str(group["target_class"].dropna().iloc[0]) if "target_class" in group.columns and group["target_class"].notna().any() else expcfg.TARGET_CLASS
        non_target = str(group["non_target_label"].dropna().iloc[0]) if "non_target_label" in group.columns and group["non_target_label"].notna().any() else expcfg.NON_TARGET_LABEL
        thresholds = object_thresholds
        if thresholds is None:
            if "object_threshold" in group.columns and group["object_threshold"].notna().any():
                thresholds = sorted(pd.to_numeric(group["object_threshold"], errors="coerce").dropna().unique().tolist())
            else:
                thresholds = expcfg.SIMCA_OBJECT_THRESHOLDS
        try:
            summary, _tables = border_width_object_threshold_grid(
                pixel_df=group,
                object_db=object_db,
                target_class=target_class,
                non_target_label=non_target,
                border_widths=widths,
                object_thresholds=thresholds,
                min_core_pixels=min_core,
            )
            if len(summary) > 0:
                summary = summary.copy()
                summary["selected_config_id"] = config_id
                summary["zone"] = np.where(summary["border_width"].eq(0), "all_pixels", "core_without_border")
                for col in [
                    "candidate_id",
                    "selection_track",
                    "matrix_family",
                    "matrix_method",
                    "preprocessing",
                    "rule_variant",
                    "n_components",
                ]:
                    if col in group.columns:
                        summary[col] = group[col].dropna().iloc[0] if group[col].notna().any() else pd.NA
                parts.append(summary)
        except Exception as exc:
            errors.append({"selected_config_id": config_id, "border_core_status": "error", "error": repr(exc)})

    diagnostics = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    status = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["selected_config_id", "border_core_status", "error"])
    return (
        compact_simca_table(diagnostics, table_kind="border_core_diagnostics"),
        compact_simca_table(status, table_kind="border_core_status"),
    )


def build_duplicated_candidate_review(
    groups_df: pd.DataFrame,
    dropped_df: pd.DataFrame | None = None,
    refit_comparison_df: pd.DataFrame | None = None,
    group_col: str = "metric_equivalence_group_id",
) -> pd.DataFrame:
    """Summarize 04C metric-equivalent duplicate groups and optional refit checks."""
    if groups_df is None or len(groups_df) == 0:
        return pd.DataFrame()

    review = groups_df.copy()
    if dropped_df is not None and len(dropped_df) > 0 and group_col in dropped_df.columns:
        dropped_counts = dropped_df.groupby(group_col, dropna=False).size().rename("n_dropped_candidates").reset_index()
        review = review.merge(dropped_counts, on=group_col, how="left")
    else:
        review["n_dropped_candidates"] = np.nan
    review["n_dropped_candidates"] = review["n_dropped_candidates"].fillna(0).astype("Int64")

    if refit_comparison_df is not None and len(refit_comparison_df) > 0 and group_col in refit_comparison_df.columns:
        cols = [
            group_col,
            "n_refit_candidates",
            "all_post_refit_metrics_equal",
            "all_post_refit_metrics_match_pre_refit",
        ]
        cols = [col for col in cols if col in refit_comparison_df.columns]
        review = review.merge(refit_comparison_df[cols].drop_duplicates(group_col), on=group_col, how="left")
        equal = review["all_post_refit_metrics_equal"].fillna(False).astype(bool)
        matches_pre = review["all_post_refit_metrics_match_pre_refit"].fillna(False).astype(bool)
        has_refit = review["n_refit_candidates"].notna() if "n_refit_candidates" in review.columns else pd.Series(False, index=review.index)
        review["duplicated_refit_status"] = np.select(
            [
                has_refit & equal & matches_pre,
                has_refit & equal & ~matches_pre,
                has_refit & ~equal,
            ],
            [
                "verified_equal",
                "equal_after_refit_changed_from_pre",
                "diverged_after_refit",
            ],
            default="not_refit",
        )
    else:
        review["n_refit_candidates"] = np.nan
        review["all_post_refit_metrics_equal"] = pd.NA
        review["all_post_refit_metrics_match_pre_refit"] = pd.NA
        review["duplicated_refit_status"] = "not_run"

    review["needs_duplicate_manual_review"] = ~review["duplicated_refit_status"].eq("verified_equal")
    return compact_simca_table(review.reset_index(drop=True), table_kind="duplicated_candidate_review")


def summarize_duplicated_candidate_review(review_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate-review status by varied parameter family."""
    if review_df is None or len(review_df) == 0:
        return pd.DataFrame()
    group_cols = [col for col in ["varied_parameter_group", "duplicated_refit_status"] if col in review_df.columns]
    if not group_cols:
        return pd.DataFrame()
    out = (
        review_df.groupby(group_cols, dropna=False)
        .agg(
            n_groups=("metric_equivalence_group_id", "nunique"),
            n_dropped_candidates=("n_dropped_candidates", "sum"),
            n_manual_review=("needs_duplicate_manual_review", "sum"),
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return compact_simca_table(out, table_kind="duplicated_candidate_summary")


def build_track_scoring_table(
    metrics_df: pd.DataFrame,
    pareto_2way_df: pd.DataFrame | None = None,
    pareto_3way_df: pd.DataFrame | None = None,
    stability_summary_df: pd.DataFrame | None = None,
    duplicated_review_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the notebook-05 per-track review table without final selection."""
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()

    out = select_track_primary_or_available_metrics(add_simca_robustness_scores(metrics_df)).copy()

    merge_keys = [col for col in ["selected_config_id", "selection_track", "metric_level"] if col in out.columns]
    for mode, pareto_df in [("2way", pareto_2way_df), ("3way", pareto_3way_df)]:
        flag_col = f"is_pareto_{mode}"
        reason_col = f"pareto_{mode}_reason"
        if pareto_df is not None and len(pareto_df) > 0 and merge_keys:
            pcols = [col for col in merge_keys + [flag_col, reason_col] if col in pareto_df.columns]
            out = out.merge(pareto_df[pcols].drop_duplicates(merge_keys), on=merge_keys, how="left")
        elif flag_col not in out.columns:
            out[flag_col] = pd.NA
            out[reason_col] = pd.NA

    if stability_summary_df is not None and len(stability_summary_df) > 0 and "selected_config_id" in out.columns:
        scols = [
            col for col in [
                "selected_config_id",
                "n_random_states",
                "stability_score",
                "stability_flags",
                "stability_flag_count",
                "std_fn_rate",
                "std_fp_rate",
                "std_balanced_accuracy",
            ]
            if col in stability_summary_df.columns
        ]
        out = out.merge(stability_summary_df[scols].drop_duplicates("selected_config_id"), on="selected_config_id", how="left")

    if (
        duplicated_review_df is not None
        and len(duplicated_review_df) > 0
        and "metric_equivalence_group_id" in out.columns
        and "metric_equivalence_group_id" in duplicated_review_df.columns
    ):
        dcols = [
            col for col in [
                "metric_equivalence_group_id",
                "varied_parameter_group",
                "n_dropped_candidates",
                "duplicated_refit_status",
                "needs_duplicate_manual_review",
            ]
            if col in duplicated_review_df.columns
        ]
        out = out.merge(duplicated_review_df[dcols].drop_duplicates("metric_equivalence_group_id"), on="metric_equivalence_group_id", how="left")

    review_flags = []
    for _, row in out.iterrows():
        flags = []
        base = row.get("robustness_flags", "")
        if isinstance(base, str) and base:
            flags.extend([flag for flag in base.split(";") if flag])
        mode = str(row.get("decision_mode", "")).lower()
        pareto_col = f"is_pareto_{mode}"
        if pareto_col in out.columns:
            pareto_value = row.get(pareto_col)
            if pd.notna(pareto_value) and not bool(pareto_value):
                flags.append("dominated_in_pareto")
        if str(row.get("stability_flags", "")):
            flags.extend([flag for flag in str(row.get("stability_flags")).split(";") if flag and flag != "nan"])
        if bool(row.get("needs_duplicate_manual_review", False)):
            flags.append("duplicate_group_unverified")
        review_flags.append(sorted(set(flags)))

    out["review_flags"] = [";".join(flags) for flags in review_flags]
    out["review_flag_count"] = [len(flags) for flags in review_flags]
    if "selection_track" in out.columns:
        out["review_rank_in_track"] = (
            out.groupby("selection_track", dropna=False)["robustness_score"]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )
    out = out.sort_values(
        [col for col in ["selection_track", "review_rank_in_track", "review_flag_count"] if col in out.columns],
        ascending=[True, True, True][: len([col for col in ["selection_track", "review_rank_in_track", "review_flag_count"] if col in out.columns])],
    ).reset_index(drop=True)
    return compact_simca_table(out, table_kind="track_scoring_flags")
