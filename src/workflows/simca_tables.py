from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SIMCA_ID_COLUMNS = (
    "selected_config_id",
    "source_selected_config_id",
    "candidate_id",
    "model_candidate_id",
    "refit_config_id",
    "metric_equivalence_group_id",
)

SIMCA_TRACK_COLUMNS = (
    "selection_track",
    "matrix_family",
    "decision_mode",
    "metric_level",
    "evaluation_stage",
    "evaluation_split",
)

SIMCA_TARGET_COLUMNS = (
    "target_class",
    "non_target_label",
)

SIMCA_CONFIG_COLUMNS = (
    "model_family",
    "matrix_family",
    "matrix_method",
    "training_matrix_id",
    "m_effective",
    "balanced_pixel_strategy_effective",
    "preprocessing",
    "preprocessing_steps",
    "rule",
    "rule_variant",
    "selected_rule_name",
    "rule_for_refit",
    "limit_source",
    "n_components",
    "alpha",
    "object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "cv_n_splits",
    "n_cv_observations",
    "n_cv_groups",
)

SIMCA_PROVENANCE_COLUMNS = (
    "search_method",
    "candidate_source",
    "candidate_sources",
    "n_candidate_sources",
    "n_duplicate_rows",
    "refit_config_duplicate_rank",
    "n_refit_config_candidates",
    "refit_config_candidate_ids",
    "refit_config_candidate_sources",
)

SIMCA_2WAY_METRIC_COLUMNS = (
    "n",
    "tp",
    "fn",
    "fp",
    "tn",
    "target_sensitivity",
    "non_target_specificity",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "f1_score",
    "fn_rate",
    "fp_rate",
    "selection_score",
)

SIMCA_PIXEL_METRIC_COLUMNS = (
    "n_truth_pixels",
    "pixel_accuracy",
    "pixel_balanced_accuracy",
    "pixel_fn_rate",
    "pixel_fp_rate",
)

SIMCA_3WAY_METRIC_COLUMNS = (
    "n",
    "n_target",
    "n_non_target",
    "n_uncertain",
    "uncertain_rate",
    "coverage_rate",
    "target_miss_rate",
    "screening_sensitivity",
    "target_auto_accept_rate",
    "target_uncertain_rate",
    "non_target_false_accept_rate",
    "non_target_auto_reject_rate",
    "non_target_uncertain_rate",
    "decided_tp",
    "decided_fn",
    "decided_fp",
    "decided_tn",
    "decided_accuracy",
    "decided_balanced_accuracy",
    "three_way_score",
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
    "target_sensitivity",
    "non_target_specificity",
    "selection_score",
)

SIMCA_OPTUNA_COLUMNS = (
    "selection_split",
    "selection_strategy",
    "optuna_trial_number",
    "optuna_value",
    "value_0",
    "value_1",
    "value_2",
    "objective_fn_rate_max",
    "objective_fp_rate_mean",
    "objective_balanced_accuracy_mean",
)

SIMCA_DUPLICATE_COLUMNS = (
    "metric_equivalence_original_order",
    "metric_equivalence_group_id",
    "metric_equivalence_kept_candidate_id",
    "metric_equivalence_varied_parameter_group",
    "metric_equivalence_drop_reason",
    "varied_parameter_group",
    "varied_columns",
    "n_metric_equivalent_candidates",
    "kept_candidate_id",
    "dropped_candidate_ids",
    "varied_values_json",
    "n_dropped_candidates",
    "n_refit_candidates",
    "all_post_refit_metrics_equal",
    "all_post_refit_metrics_match_pre_refit",
    "duplicated_refit_status",
    "needs_duplicate_manual_review",
)

SIMCA_ROBUSTNESS_COLUMNS = (
    "robustness_score",
    "robustness_flags",
    "robustness_flag_count",
    "robustness_rank_in_track",
    "review_flags",
    "review_flag_count",
    "review_rank_in_track",
)

SIMCA_PARETO_COLUMNS = (
    "is_pareto_2way",
    "pareto_2way_reason",
    "is_pareto_3way",
    "pareto_3way_reason",
)

SIMCA_STABILITY_COLUMNS = (
    "random_state",
    "stability_selection_tracks",
    "n_stability_selection_tracks",
    "stability_panel_reason",
    "n_runs",
    "n_random_states",
    "stability_score",
    "stability_flags",
    "stability_flag_count",
    "mean_fn_rate",
    "std_fn_rate",
    "min_fn_rate",
    "max_fn_rate",
    "mean_fp_rate",
    "std_fp_rate",
    "min_fp_rate",
    "max_fp_rate",
    "mean_balanced_accuracy",
    "std_balanced_accuracy",
    "min_balanced_accuracy",
    "max_balanced_accuracy",
    "mean_target_sensitivity",
    "std_target_sensitivity",
    "mean_non_target_specificity",
    "std_non_target_specificity",
    "mean_robustness_score",
    "std_robustness_score",
)

SIMCA_BORDER_CORE_COLUMNS = (
    "selected_config_id",
    "candidate_id",
    "selection_track",
    "matrix_family",
    "matrix_method",
    "preprocessing",
    "rule_variant",
    "n_components",
    "border_width",
    "zone",
    "object_threshold",
    "min_core_pixels",
    "mean_core_fraction",
    "fallback_object_rate",
    "n",
    "tp",
    "fn",
    "fp",
    "tn",
    "target_sensitivity",
    "non_target_specificity",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "f1_score",
    "fn_rate",
    "fp_rate",
)

SIMCA_ERROR_COLUMNS = (
    SIMCA_ID_COLUMNS
    + SIMCA_TRACK_COLUMNS
    + SIMCA_TARGET_COLUMNS
    + SIMCA_CONFIG_COLUMNS
    + (
        "random_state",
        "source_image",
        "error",
    )
)

SIMCA_PIXEL_ERROR_SUMMARY_COLUMNS = (
    SIMCA_ID_COLUMNS
    + SIMCA_TRACK_COLUMNS
    + SIMCA_TARGET_COLUMNS
    + SIMCA_CONFIG_COLUMNS
    + (
        "source_image",
    )
    + SIMCA_2WAY_METRIC_COLUMNS
)

SIMCA_BATCH_MANIFEST_COLUMNS = (
    "batch_id",
    "row_start",
    "row_stop",
    "n_candidates",
    "n_refit_metric_rows",
    "n_2way_object_metric_rows",
    "n_2way_pixel_metric_rows",
    "n_3way_threshold_grid_rows",
    "n_3way_selected_threshold_rows",
    "n_3way_object_metric_rows",
    "n_object_rows",
    "n_pixel_rows",
    "n_3way_object_rows",
    "n_pixel_error_rows",
    "n_error_rows",
    "refit_metrics_path",
    "2way_object_metrics_path",
    "2way_pixel_metrics_path",
    "3way_threshold_grid_path",
    "3way_selected_thresholds_path",
    "3way_object_metrics_path",
    "objects_path",
    "pixels_path",
    "objects_3way_path",
)

SIMCA_PARETO_AUDIT_COLUMNS = (
    "selection_track",
    "matrix_family",
    "decision_mode",
    "pareto_minimize_columns",
    "pareto_maximize_columns",
    "n_before",
    "n_pareto",
    "n_dominated",
)

_SIMCA_ABLATION_BASE_METRICS = (
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

SIMCA_ABLATION_COLUMNS = (
    "selection_track",
    "matrix_family",
    "decision_mode",
    "metric_level",
    "factor",
    "factor_value",
    "factor_value_numeric",
    "n_configs",
) + tuple(
    f"{metric}_{stat}"
    for metric in _SIMCA_ABLATION_BASE_METRICS
    for stat in ("mean", "median", "std", "min", "max")
)

SIMCA_BORDER_CORE_STATUS_COLUMNS = (
    "selected_config_id",
    "border_core_status",
    "skip_reason",
    "pixel_batch_dir",
    "required_04c_setting",
    "error",
)

SIMCA_DUPLICATED_REFIT_COMPARISON_COLUMNS = (
    SIMCA_DUPLICATE_COLUMNS
    + (
        "candidate_ids",
        "selected_config_ids",
        "n_refit_candidates",
        "all_post_refit_metrics_equal",
        "all_post_refit_metrics_match_pre_refit",
    )
    + tuple(
        f"{metric}_{suffix}"
        for metric in ("n", "tp", "fn", "fp", "tn", "fn_rate", "fp_rate", "balanced_accuracy")
        for suffix in (
            "post_refit_nunique",
            "post_refit_min",
            "post_refit_max",
            "post_refit_all_equal",
            "matches_pre_refit_all",
            "max_abs_delta_vs_pre_refit",
        )
    )
)

SIMCA_DUPLICATED_CANDIDATE_SUMMARY_COLUMNS = (
    "varied_parameter_group",
    "duplicated_refit_status",
    "n_groups",
    "n_dropped_candidates",
    "n_manual_review",
)

SIMCA_THREE_WAY_THRESHOLD_COLUMNS = (
    SIMCA_ID_COLUMNS
    + SIMCA_TRACK_COLUMNS
    + SIMCA_TARGET_COLUMNS
    + SIMCA_CONFIG_COLUMNS
    + (
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    )
    + SIMCA_3WAY_METRIC_COLUMNS
)

SIMCA_PRE_REFIT_METRIC_COLUMNS = tuple(
    f"pre_refit_{col}"
    for col in (
        "n",
        "tp",
        "fn",
        "fp",
        "tn",
        "fn_rate",
        "fp_rate",
        "balanced_accuracy",
    )
)


SIMCA_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "candidate_panel": (
        SIMCA_ID_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_OPTUNA_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "candidate_metrics": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_OPTUNA_COLUMNS
    ),
    "refit_config_duplicates": (
        SIMCA_ID_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "metric_equivalence_groups": (
        SIMCA_DUPLICATE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + (
            "metric_columns",
        )
    ),
    "duplicated_refit_panel": (
        SIMCA_ID_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_PRE_REFIT_METRIC_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "simca_2way_metrics": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "simca_2way_pixel_metrics": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_PIXEL_METRIC_COLUMNS
    ),
    "simca_3way_metrics": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_3WAY_METRIC_COLUMNS
    ),
    "simca_metrics_review": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_PIXEL_METRIC_COLUMNS
        + SIMCA_3WAY_METRIC_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "pareto_2way": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_ROBUSTNESS_COLUMNS
        + SIMCA_PARETO_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "pareto_3way": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_3WAY_METRIC_COLUMNS
        + SIMCA_ROBUSTNESS_COLUMNS
        + SIMCA_PARETO_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "robustness_metrics": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_PIXEL_METRIC_COLUMNS
        + SIMCA_3WAY_METRIC_COLUMNS
        + SIMCA_ROBUSTNESS_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "track_scoring_flags": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_3WAY_METRIC_COLUMNS
        + SIMCA_ROBUSTNESS_COLUMNS
        + SIMCA_PARETO_COLUMNS
        + SIMCA_STABILITY_COLUMNS
        + SIMCA_DUPLICATE_COLUMNS
    ),
    "random_state_stability_panel": (
        SIMCA_ID_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_ROBUSTNESS_COLUMNS
        + SIMCA_STABILITY_COLUMNS
    ),
    "random_state_stability_metrics": (
        SIMCA_ID_COLUMNS
        + SIMCA_TRACK_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_PROVENANCE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
        + SIMCA_ROBUSTNESS_COLUMNS
        + SIMCA_STABILITY_COLUMNS
    ),
    "random_state_stability_summary": (
        SIMCA_ID_COLUMNS
        + SIMCA_TARGET_COLUMNS
        + SIMCA_CONFIG_COLUMNS
        + SIMCA_STABILITY_COLUMNS
    ),
    "duplicated_candidate_review": (
        SIMCA_DUPLICATE_COLUMNS
        + SIMCA_2WAY_METRIC_COLUMNS
    ),
    "duplicated_candidate_summary": SIMCA_DUPLICATED_CANDIDATE_SUMMARY_COLUMNS,
    "duplicated_refit_comparison": SIMCA_DUPLICATED_REFIT_COMPARISON_COLUMNS,
    "border_core_diagnostics": SIMCA_BORDER_CORE_COLUMNS,
    "border_core_status": SIMCA_BORDER_CORE_STATUS_COLUMNS,
    "simca_error_log": SIMCA_ERROR_COLUMNS,
    "simca_pixel_error_summary": SIMCA_PIXEL_ERROR_SUMMARY_COLUMNS,
    "simca_batch_manifest": SIMCA_BATCH_MANIFEST_COLUMNS,
    "pareto_audit": SIMCA_PARETO_AUDIT_COLUMNS,
    "ablation_diagnostics": SIMCA_ABLATION_COLUMNS,
    "three_way_threshold_grid": SIMCA_THREE_WAY_THRESHOLD_COLUMNS,
    "three_way_selected_thresholds": SIMCA_THREE_WAY_THRESHOLD_COLUMNS,
}


TABLE_KIND_BY_FILE_NAME: dict[str, str] = {
    "candidate_panel.parquet": "candidate_panel",
    "grid_model_candidates.parquet": "candidate_metrics",
    "grid_search_results.parquet": "candidate_metrics",
    "optuna_model_candidates.parquet": "candidate_metrics",
    "optuna_new_model_candidates.parquet": "candidate_metrics",
    "optuna_candidate_metrics.parquet": "candidate_metrics",
    "refit_config_dedup_summary.parquet": "refit_config_duplicates",
    "refit_config_duplicates.parquet": "refit_config_duplicates",
    "metric_equivalent_config_groups.parquet": "metric_equivalence_groups",
    "metric_equivalent_config_dropped.parquet": "candidate_panel",
    "duplicated_refit_panel.parquet": "duplicated_refit_panel",
    "duplicated_refit_2way_object_metrics.parquet": "simca_2way_metrics",
    "duplicated_refit_2way_pixel_metrics.parquet": "simca_2way_pixel_metrics",
    "duplicated_refit_metrics_long.parquet": "simca_metrics_review",
    "duplicated_refit_metric_comparison.parquet": "duplicated_refit_comparison",
    "duplicated_refit_errors.parquet": "simca_error_log",
    "duplicated_refit_pixel_errors_by_image.parquet": "simca_pixel_error_summary",
    "duplicated_refit_batch_manifest.parquet": "simca_batch_manifest",
    "validation_refit_2way_object_metrics.parquet": "simca_2way_metrics",
    "validation_refit_2way_pixel_metrics.parquet": "simca_2way_pixel_metrics",
    "validation_3way_threshold_grid.parquet": "three_way_threshold_grid",
    "validation_3way_selected_thresholds.parquet": "three_way_selected_thresholds",
    "validation_refit_3way_object_metrics.parquet": "simca_3way_metrics",
    "validation_refit_metrics_long.parquet": "simca_metrics_review",
    "validation_refit_errors.parquet": "simca_error_log",
    "validation_refit_pixel_errors_by_image.parquet": "simca_pixel_error_summary",
    "validation_refit_batch_manifest.parquet": "simca_batch_manifest",
    "robustness_scored_metrics.parquet": "robustness_metrics",
    "robustness_primary_metrics.parquet": "robustness_metrics",
    "pareto_2way_front.parquet": "pareto_2way",
    "pareto_2way_annotated.parquet": "pareto_2way",
    "pareto_2way_audit.parquet": "pareto_audit",
    "pareto_3way_front.parquet": "pareto_3way",
    "pareto_3way_annotated.parquet": "pareto_3way",
    "pareto_3way_audit.parquet": "pareto_audit",
    "ablation_diagnostics.parquet": "ablation_diagnostics",
    "random_state_stability_panel.parquet": "random_state_stability_panel",
    "random_state_stability_metrics.parquet": "random_state_stability_metrics",
    "random_state_stability_summary.parquet": "random_state_stability_summary",
    "random_state_stability_errors.parquet": "simca_error_log",
    "track_scoring_flags.parquet": "track_scoring_flags",
    "duplicated_candidate_review.parquet": "duplicated_candidate_review",
    "duplicated_candidate_summary.parquet": "duplicated_candidate_summary",
    "border_core_diagnostics.parquet": "border_core_diagnostics",
    "border_core_status.parquet": "border_core_status",
}


TABLE_KIND_BY_FILE_SUFFIX: dict[str, str] = {
    "_refit_metrics.parquet": "simca_metrics_review",
    "_2way_object_metrics.parquet": "simca_2way_metrics",
    "_2way_pixel_metrics.parquet": "simca_2way_pixel_metrics",
    "_3way_threshold_grid.parquet": "three_way_threshold_grid",
    "_3way_selected_thresholds.parquet": "three_way_selected_thresholds",
    "_3way_object_metrics.parquet": "simca_3way_metrics",
}


def _resolve_table_kind_from_name(name: str) -> str | None:
    if name in TABLE_KIND_BY_FILE_NAME:
        return TABLE_KIND_BY_FILE_NAME[name]
    for suffix, table_kind in TABLE_KIND_BY_FILE_SUFFIX.items():
        if name.endswith(suffix):
            return table_kind
    return None


def ordered_existing_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    include_remaining: bool = False,
) -> list[str]:
    """Return existing columns in a stable requested order."""
    selected = []
    seen = set()
    for col in columns:
        if col in df.columns and col not in seen:
            selected.append(col)
            seen.add(col)
    if include_remaining:
        for col in df.columns:
            if col not in seen:
                selected.append(col)
                seen.add(col)
    return selected


def _coalesce_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.combine_first(right)


def _infer_matrix_family(value: Any) -> str:
    token = str(value)
    if token.startswith("object_matrix"):
        return "object_matrix"
    if token.startswith("pixel_matrix"):
        return "pixel_matrix"
    if token.startswith("balanced_pixel_") or token in {"balanced_pixels", "all_pixels", "pixel"}:
        return "pixel_matrix"
    if token.startswith("object_") or token in {"object_mean", "object_median"}:
        return "object_matrix"
    return pd.NA


def deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse exact duplicate column labels by taking the first non-null value.

    Pandas can carry duplicate labels after notebook manipulations, but Parquet
    cannot. This function makes the table serializable before any schema policy
    is applied.
    """
    if df is None or len(df.columns) == 0:
        return pd.DataFrame() if df is None else df.copy()
    if not pd.Index(df.columns).duplicated().any():
        return df.copy()

    parts = []
    for col in pd.Index(df.columns).drop_duplicates():
        current = df.loc[:, df.columns == col]
        if current.shape[1] == 1:
            series = current.iloc[:, 0]
        else:
            series = current.iloc[:, 0]
            for idx in range(1, current.shape[1]):
                series = _coalesce_series(series, current.iloc[:, idx])
        parts.append(series.rename(col))
    return pd.concat(parts, axis=1)


def resolve_merge_suffix_columns(
    df: pd.DataFrame,
    *,
    prefer: str = "y",
    drop_suffix_columns: bool = True,
) -> pd.DataFrame:
    """Coalesce columns created by pandas merges with `_x` / `_y` suffixes."""
    out = deduplicate_column_names(df)
    columns = list(out.columns)
    bases = sorted(
        {
            str(col)[:-2]
            for col in columns
            if str(col).endswith("_x") or str(col).endswith("_y")
        }
    )
    for base in bases:
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        candidates = []
        if prefer == "y":
            candidates = [y_col, x_col, base]
        elif prefer == "x":
            candidates = [x_col, y_col, base]
        else:
            candidates = [base, y_col, x_col]
        candidates = [col for col in candidates if col in out.columns]
        if not candidates:
            continue
        merged = out[candidates[0]]
        for col in candidates[1:]:
            merged = _coalesce_series(merged, out[col])
        out[base] = merged
        if drop_suffix_columns:
            out = out.drop(columns=[x_col, y_col], errors="ignore")
    return out


def canonicalize_simca_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common SIMCA aliases before table-specific compacting."""
    if df is None or len(df.columns) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = resolve_merge_suffix_columns(df)

    if "non_target_label" not in out.columns and "non_target_class" in out.columns:
        out["non_target_label"] = out["non_target_class"]
    elif "non_target_label" in out.columns and "non_target_class" in out.columns:
        out["non_target_label"] = out["non_target_label"].combine_first(out["non_target_class"])

    if "m_effective" not in out.columns and "m" in out.columns:
        out["m_effective"] = out["m"]
    elif "m_effective" in out.columns and "m" in out.columns:
        out["m_effective"] = out["m_effective"].combine_first(out["m"])

    if "balanced_pixel_strategy_effective" not in out.columns and "balanced_pixel_strategy" in out.columns:
        out["balanced_pixel_strategy_effective"] = out["balanced_pixel_strategy"]
    elif "balanced_pixel_strategy_effective" in out.columns and "balanced_pixel_strategy" in out.columns:
        out["balanced_pixel_strategy_effective"] = out["balanced_pixel_strategy_effective"].combine_first(
            out["balanced_pixel_strategy"]
        )

    if "selected_rule_name" not in out.columns and "rule_variant" in out.columns:
        out["selected_rule_name"] = out["rule_variant"]
    elif "selected_rule_name" in out.columns and "rule_variant" in out.columns:
        out["selected_rule_name"] = out["selected_rule_name"].combine_first(out["rule_variant"])

    if "matrix_family" not in out.columns:
        if "matrix_method" in out.columns:
            out["matrix_family"] = out["matrix_method"].map(_infer_matrix_family)
        elif "training_matrix_id" in out.columns:
            out["matrix_family"] = out["training_matrix_id"].map(_infer_matrix_family)
        elif "selection_track" in out.columns:
            out["matrix_family"] = out["selection_track"].map(_infer_matrix_family)
    elif "matrix_method" in out.columns:
        inferred = out["matrix_method"].map(_infer_matrix_family)
        out["matrix_family"] = out["matrix_family"].combine_first(inferred)
    elif "selection_track" in out.columns:
        inferred = out["selection_track"].map(_infer_matrix_family)
        out["matrix_family"] = out["matrix_family"].combine_first(inferred)

    return out


def drop_all_na_columns(
    df: pd.DataFrame,
    *,
    protected_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Drop columns that are entirely non-applicable while preserving keys."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()
    protected = set(protected_columns)
    keep = [
        col
        for col in df.columns
        if col in protected or not df[col].isna().all()
    ]
    return df.loc[:, keep].copy()


def compact_simca_table(
    df: pd.DataFrame,
    table_kind: str | None = None,
    *,
    include_remaining: bool = False,
    drop_all_na: bool = True,
    protected_columns: Sequence[str] = SIMCA_ID_COLUMNS + SIMCA_TRACK_COLUMNS,
) -> pd.DataFrame:
    """
    Return a stable, compact SIMCA output table.

    The function removes merge suffix columns, normalizes common aliases, keeps
    table-specific columns in a predictable order, and drops columns that are
    entirely non-applicable for the current output.
    """
    if df is None:
        columns = SIMCA_TABLE_COLUMNS.get(str(table_kind), ()) if table_kind is not None else ()
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))
    if len(df) == 0 and len(df.columns) == 0:
        columns = SIMCA_TABLE_COLUMNS.get(str(table_kind), ()) if table_kind is not None else ()
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))

    out = canonicalize_simca_columns(df)
    if table_kind is not None:
        columns = SIMCA_TABLE_COLUMNS.get(str(table_kind), ())
        if columns:
            out = out.loc[:, ordered_existing_columns(out, columns, include_remaining=include_remaining)]
    if drop_all_na:
        out = drop_all_na_columns(out, protected_columns=protected_columns)
    return out.reset_index(drop=True)


def compact_simca_table_for_path(
    df: pd.DataFrame,
    path_or_name: Any,
    *,
    include_remaining: bool = False,
    drop_all_na: bool = True,
) -> pd.DataFrame:
    """Compact a SIMCA table using the output file name as schema selector."""
    name = getattr(path_or_name, "name", None) or str(path_or_name).replace("\\", "/").split("/")[-1]
    table_kind = _resolve_table_kind_from_name(str(name))
    return compact_simca_table(
        df,
        table_kind=table_kind,
        include_remaining=include_remaining,
        drop_all_na=drop_all_na,
    )


def schema_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight diagnostics for a dataframe schema."""
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
        str(col)
        for col in df.columns
        if len(df) > 0 and df[col].isna().all()
    ]
    suffix_columns = [
        str(col)
        for col in df.columns
        if str(col).endswith("_x") or str(col).endswith("_y")
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
    """Build a compact schema manifest for notebook outputs."""
    rows = []
    for name, df in tables.items():
        row = {"table_name": str(name)}
        row.update(schema_diagnostics(df))
        rows.append(row)
    return pd.DataFrame(rows)
