from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.decision.labels import DEFAULT_NON_TARGET_LABEL, DEFAULT_TARGET_CLASS
from src.utils import first_available_value, is_missing_value, to_numeric_metrics


SIMCA_RULE_METADATA: dict[str, dict[str, str]] = {
    "simple": {"rule_base": "simple", "rule_variant": "simple_chi2", "limit_source": "chi2"},
    "simple_chi2": {"rule_base": "simple", "rule_variant": "simple_chi2", "limit_source": "chi2"},
    "simple_emp_cv": {"rule_base": "simple", "rule_variant": "simple_emp_cv", "limit_source": "empirical_cv"},
    "alternative": {"rule_base": "alternative", "rule_variant": "alternative_chi2_fixed2", "limit_source": "chi2"},
    "alternative_chi2_fixed2": {"rule_base": "alternative", "rule_variant": "alternative_chi2_fixed2", "limit_source": "chi2"},
    "alternative_chi2_emp_cv": {"rule_base": "alternative", "rule_variant": "alternative_chi2_emp_cv", "limit_source": "empirical_cv"},
    "alternative_empHQ_fixed2": {"rule_base": "alternative", "rule_variant": "alternative_empHQ_fixed2", "limit_source": "chi2"},
    "alternative_empHQ_emp_cv": {"rule_base": "alternative", "rule_variant": "alternative_empHQ_emp_cv", "limit_source": "empirical_cv"},
    "data_driven": {"rule_base": "data_driven", "rule_variant": "data_driven_chi2", "limit_source": "chi2"},
    "data_driven_chi2": {"rule_base": "data_driven", "rule_variant": "data_driven_chi2", "limit_source": "chi2"},
    "data_driven_emp_cv": {"rule_base": "data_driven", "rule_variant": "data_driven_emp_cv", "limit_source": "empirical_cv"},
    "combined_index": {"rule_base": "combined_index", "rule_variant": "combined_index_chi2", "limit_source": "scaled_chi2"},
    "combined_index_chi2": {"rule_base": "combined_index", "rule_variant": "combined_index_chi2", "limit_source": "scaled_chi2"},
    "combined_index_emp_cv": {"rule_base": "combined_index", "rule_variant": "combined_index_emp_cv", "limit_source": "empirical_cv"},
}


# -----------------------------------------------------------------------------
# Flexible metric aliasing
# -----------------------------------------------------------------------------

DEFAULT_SELECTION_METRIC_ALIASES = {
    # Binary FN/FPR metrics.
    # Priority is conservative for FN: max before mean.
    "fn_rate": [
        "fn_rate",
        "validation_fn_rate",
        "pure_test_fn_rate",
        "max_fn_rate",
        "fn_rate_max",
        "objective_fn_rate_max",
        "value_0",
        "mean_fn_rate",
        "fn_rate_mean",
    ],
    "fp_rate": [
        "fp_rate",
        "validation_fp_rate",
        "pure_test_fp_rate",
        "mean_fp_rate",
        "fp_rate_mean",
        "objective_fp_rate_mean",
        "value_1",
        "max_fp_rate",
        "fp_rate_max",
    ],
    "balanced_accuracy": [
        "balanced_accuracy",
        "validation_balanced_accuracy",
        "pure_test_balanced_accuracy",
        "mean_balanced_accuracy",
        "balanced_accuracy_mean",
        "objective_balanced_accuracy_mean",
        "value_2",
    ],
    "target_sensitivity": [
        "target_sensitivity",
        "validation_target_sensitivity",
        "pure_test_target_sensitivity",
        "mean_target_sensitivity",
        "target_sensitivity_mean",
    ],
    "non_target_specificity": [
        "non_target_specificity",
        "validation_non_target_specificity",
        "pure_test_non_target_specificity",
        "mean_non_target_specificity",
        "non_target_specificity_mean",
    ],
    "f1_score": [
        "f1_score",
        "validation_f1_score",
        "pure_test_f1_score",
        "mean_f1_score",
        "f1_score_mean",
    ],
    "accuracy": [
        "accuracy",
        "validation_accuracy",
        "pure_test_accuracy",
        "mean_accuracy",
        "accuracy_mean",
    ],
    "precision": [
        "precision",
        "validation_precision",
        "pure_test_precision",
        "mean_precision",
        "precision_mean",
    ],
}


def _coalesce_numeric_columns(
    df: pd.DataFrame,
    candidates: Sequence[str],
    default: float = np.nan,
) -> pd.Series:
    """
    Return the first non-missing numeric value among candidate columns.

    Example:
    - if fn_rate is missing but fn_rate_max exists, use fn_rate_max.
    - if both fn_rate_max and fn_rate_mean exist, priority is given by order.
    """
    out = pd.Series(default, index=df.index, dtype="float64")

    for col in candidates:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), values)

    return out


def materialize_selection_metrics(
    df: pd.DataFrame,
    metric_aliases: dict[str, Sequence[str]] | None = None,
    overwrite: bool = False,
    keep_source_columns: bool = True,
) -> pd.DataFrame:
    """
    Create standard metric columns used by selection utilities.

    This makes selection functions compatible with:
    - classical grid summaries: fn_rate, fp_rate, balanced_accuracy
    - robust 04B summaries: mean_fn_rate, max_fn_rate, mean_fp_rate
    - Optuna 04B2 summaries: fn_rate_max, fp_rate_mean, balanced_accuracy_mean
    - pure test 04C summaries: pure_test_fn_rate, pure_test_fp_rate, etc.

    Parameters
    ----------
    overwrite:
        If False, existing canonical columns are preserved.
        If True, canonical columns are recomputed from aliases.

    keep_source_columns:
        If True, adds columns such as fn_rate_source to document
        which input column was used.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    aliases = {
        key: list(value)
        for key, value in DEFAULT_SELECTION_METRIC_ALIASES.items()
    }

    if metric_aliases:
        for key, value in metric_aliases.items():
            aliases[key] = list(value)

    for canonical, candidates in aliases.items():
        if canonical in out.columns and not overwrite:
            out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
            continue

        out[canonical] = _coalesce_numeric_columns(
            out,
            candidates=candidates,
            default=np.nan,
        )

        if keep_source_columns:
            source_col = f"{canonical}_source"
            source_values = pd.Series(pd.NA, index=out.index, dtype="object")

            for candidate in candidates:
                if candidate not in out.columns:
                    continue

                values = pd.to_numeric(out[candidate], errors="coerce")
                mask = source_values.isna() & values.notna()
                source_values.loc[mask] = candidate

            out[source_col] = source_values

    # Derive missing sensitivity/specificity from rates.
    if "target_sensitivity" in out.columns and "fn_rate" in out.columns:
        missing_sens = out["target_sensitivity"].isna() & out["fn_rate"].notna()
        out.loc[missing_sens, "target_sensitivity"] = 1.0 - out.loc[missing_sens, "fn_rate"]

        missing_fn = out["fn_rate"].isna() & out["target_sensitivity"].notna()
        out.loc[missing_fn, "fn_rate"] = 1.0 - out.loc[missing_fn, "target_sensitivity"]

    if "non_target_specificity" in out.columns and "fp_rate" in out.columns:
        missing_spec = out["non_target_specificity"].isna() & out["fp_rate"].notna()
        out.loc[missing_spec, "non_target_specificity"] = 1.0 - out.loc[missing_spec, "fp_rate"]

        missing_fp = out["fp_rate"].isna() & out["non_target_specificity"].notna()
        out.loc[missing_fp, "fp_rate"] = 1.0 - out.loc[missing_fp, "non_target_specificity"]

    return out


DEFAULT_DETECTION_SORT_PRIORITY: tuple[tuple[str, bool], ...] = (
    ("fn_rate", True),
    ("fp_rate", True),
    ("balanced_accuracy", False),
    ("f1_score", False),
    ("accuracy", False),
    ("precision", False),
)


def sort_detection_selection(
    df: pd.DataFrame,
    priority: Sequence[tuple[str, bool]] | None = None,
    *,
    materialize_metrics: bool = True,
) -> pd.DataFrame:
    """Sort binary-detection results without constructing a scalar score.

    The default order is deliberately lexicographic: false negatives first,
    false positives second, then descriptive performance metrics only as
    deterministic tie-breakers. ``True`` means ascending for a priority item.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = (
        materialize_selection_metrics(
            df,
            keep_source_columns=False,
        )
        if materialize_metrics
        else df.copy()
    )
    active_priority = tuple(priority or DEFAULT_DETECTION_SORT_PRIORITY)
    sort_columns = [
        column
        for column, _ in active_priority
        if column in out.columns
    ]
    if not sort_columns:
        return out.reset_index(drop=True)

    ascending_by_column = dict(active_priority)
    return out.sort_values(
        sort_columns,
        ascending=[ascending_by_column[column] for column in sort_columns],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def infer_model_family_from_rule_token(rule_token: str) -> str:
    """Infer whether a SIMCA rule token belongs to the standard or empirical-CV workflow."""
    token = str(rule_token)
    if token in {"simple", "alternative", "data_driven", "combined_index"}:
        return "standard_rule"
    if token.endswith("_emp_cv") or token in {
        "simple_chi2",
        "alternative_chi2_fixed2",
        "alternative_chi2_emp_cv",
        "alternative_empHQ_fixed2",
        "alternative_empHQ_emp_cv",
        "data_driven_chi2",
        "data_driven_emp_cv",
        "combined_index_chi2",
        "combined_index_emp_cv",
    }:
        return "empirical_cv_rule"
    return "unknown"


def normalize_simca_rule_columns(df: pd.DataFrame, model_family: str | None = None) -> pd.DataFrame:
    """Normalize SIMCA rule columns used by standard and empirical-CV summaries."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()
    out["rule_original"] = out["rule"].astype("object") if "rule" in out.columns else np.nan
    out["rule_variant_original"] = out["rule_variant"].astype("object") if "rule_variant" in out.columns else np.nan

    rows = []
    for _, row in out.iterrows():
        token = first_available_value(row, ["rule_variant", "rule", "selected_rule_name"], default="unknown")
        token = "unknown" if is_missing_value(token) else str(token)
        meta = SIMCA_RULE_METADATA.get(
            token,
            {
                "rule_base": str(row.get("rule", token)),
                "rule_variant": token,
                "limit_source": "unknown",
            },
        )

        family = model_family
        if family is None:
            family = first_available_value(row, ["model_family"], default=None)
        if family is None or is_missing_value(family):
            family = infer_model_family_from_rule_token(token)
        family = str(family)

        rows.append(
            {
                "rule_token": token,
                "rule": meta["rule_base"],
                "rule_variant": meta["rule_variant"],
                "selected_rule_name": meta["rule_variant"],
                "model_family": family,
                "rule_for_refit": meta["rule_base"] if family == "standard_rule" else meta["rule_variant"],
                "limit_source": meta["limit_source"],
            }
        )

    meta_df = pd.DataFrame(rows, index=out.index)
    for col in meta_df.columns:
        out[col] = meta_df[col]
    return out


def fill_selected_config_defaults(
    selected_configs_df: pd.DataFrame,
    default_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fill missing columns needed for refitting selected SIMCA configurations."""
    df = selected_configs_df.copy()
    defaults: dict[str, object] = {
        "sg_window_length": 11,
        "sg_polyorder": 2,
        "position_dilation_radius": 3,
        "m": 40,
        "balanced_pixel_strategy": "random",
        "matrix_method": "balanced_pixels",
        "alpha": 0.05,
        "object_threshold": 0.75,
        "target_class": DEFAULT_TARGET_CLASS,
        "non_target_label": DEFAULT_NON_TARGET_LABEL,
    }
    if default_values:
        defaults.update(default_values)

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    if "rule_variant" not in df.columns:
        df["rule_variant"] = np.nan
    if "selected_rule_name" not in df.columns:
        df["selected_rule_name"] = np.nan
    df["rule_variant"] = df["rule_variant"].fillna(df["selected_rule_name"])

    for col in ["n_components", "sg_window_length", "sg_polyorder", "position_dilation_radius", "m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["alpha", "object_threshold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ensure_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure candidate selection tables have common metadata columns."""
    out = df.copy()

    if "matrix_family" not in out.columns and "matrix_method" in out.columns:
        out["matrix_family"] = np.where(
            out["matrix_method"].astype(str).isin(["object_mean", "object_median"]),
            "object_matrix",
            "pixel_matrix",
        )
    if "training_matrix_id" not in out.columns and "matrix_method" in out.columns:
        out["training_matrix_id"] = out["matrix_method"].astype(str)
    if "selected_rule_name" not in out.columns:
        if "rule_variant" in out.columns:
            out["selected_rule_name"] = out["rule_variant"].fillna(out.get("rule", "unknown"))
        else:
            out["selected_rule_name"] = out.get("rule", "unknown")
    if "rule_for_refit" not in out.columns:
        if "model_family" in out.columns and "rule_variant" in out.columns and "rule" in out.columns:
            out["rule_for_refit"] = np.where(
                out["model_family"].astype(str).eq("standard_rule"),
                out["rule"].astype(str),
                out["rule_variant"].astype(str),
            )
        else:
            out["rule_for_refit"] = out["selected_rule_name"].astype(str)
    if "selected_config_id" not in out.columns:
        out["selected_config_id"] = [f"cand_{i:04d}" for i in range(len(out))]

    return out


def select_top_models(
    df: pd.DataFrame,
    n_per_training_matrix: int = 5,
    n_per_matrix_family: int = 15,
    n_overall: int = 30,
) -> pd.DataFrame:
    """Select a compact set of top candidate configurations."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    ranked = sort_detection_selection(ensure_candidate_columns(df))
    parts = []

    if "training_matrix_id" in ranked.columns and n_per_training_matrix:
        parts.append(ranked.groupby("training_matrix_id", group_keys=False, dropna=False).head(n_per_training_matrix))
    if "matrix_family" in ranked.columns and n_per_matrix_family:
        parts.append(ranked.groupby("matrix_family", group_keys=False, dropna=False).head(n_per_matrix_family))
    if n_overall:
        parts.append(ranked.head(n_overall))

    selected = pd.concat(parts, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
    return sort_detection_selection(selected)


def pareto_front(
    df: pd.DataFrame,
    minimize_cols: Sequence[str] = ("fn_rate", "fp_rate"),
    maximize_cols: Sequence[str] = ("balanced_accuracy",),
) -> pd.DataFrame:
    """Return non-dominated rows according to minimization and maximization objectives."""
    d = df.copy().reset_index(drop=True)
    values_min = d[list(minimize_cols)].to_numpy(dtype=float)
    values_max = d[list(maximize_cols)].to_numpy(dtype=float)
    keep = np.ones(len(d), dtype=bool)

    for i in range(len(d)):
        if not keep[i]:
            continue
        better_or_equal_min = values_min <= values_min[i]
        better_or_equal_max = values_max >= values_max[i]
        dominates_i = better_or_equal_min.all(axis=1) & better_or_equal_max.all(axis=1)
        strictly_better = (values_min < values_min[i]).any(axis=1) | (values_max > values_max[i]).any(axis=1)
        dominated_by_other = dominates_i & strictly_better
        dominated_by_other[i] = False
        if dominated_by_other.any():
            keep[i] = False

    return d[keep].copy()


def pareto_front_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    minimize_cols: Sequence[str],
    maximize_cols: Sequence[str] = (),
    epsilon: float = 0.0,
    keep_group_cols: bool = True,
) -> pd.DataFrame:
    """Return non-dominated rows independently inside each group.

    Objectives to maximize are multiplied by -1 so every objective is treated
    as a minimization objective. Exact duplicate objective vectors are reduced
    before the skyline search and restored afterwards.

    The implementation stores only the current non-dominated front. Its memory
    use is therefore O(n_objectives * frontier_size), rather than O(n²).
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    if float(epsilon) < 0.0:
        raise ValueError("epsilon must be non-negative.")

    groups = [column for column in group_cols if column in df.columns]
    minimize = [
        column for column in minimize_cols if column in df.columns
    ]
    maximize = [
        column for column in maximize_cols if column in df.columns
    ]
    objectives = [*minimize, *maximize]
    if not objectives:
        raise ValueError("At least one Pareto objective is required.")

    work = to_numeric_metrics(df, objectives).copy()
    finite = np.isfinite(work[objectives].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        bad = work.loc[~finite, objectives]
        raise ValueError(
            "Pareto objectives must be finite. "
            f"Invalid row count: {len(bad)}."
        )

    work["_pareto_input_order"] = np.arange(len(work), dtype=np.int64)
    grouped = (work.groupby(groups, dropna=False, sort=False)
        if groups
        else [((), work)]
    )

    kept_parts: list[pd.DataFrame] = []
    for _, group in grouped:
        group = group.copy()
        minimize_values = (
            group[minimize].to_numpy(dtype=float)
            if minimize
            else np.empty((len(group), 0), dtype=float)
        )
        maximize_values = (
            -group[maximize].to_numpy(dtype=float)
            if maximize
            else np.empty((len(group), 0), dtype=float)
        )
        values = np.column_stack([minimize_values, maximize_values])
        unique_values, inverse = np.unique(
            values,
            axis=0,
            return_inverse=True,
        )
        # Sorting on the first minimization objective usually keeps the active
        # skyline small. Later candidates can still remove earlier candidates
        # when the first objective is equal within epsilon.
        order = np.argsort(unique_values[:, 0], kind="mergesort")
        frontier_indices: list[int] = []

        for candidate_index in order:
            candidate = unique_values[candidate_index]
            if not frontier_indices:
                frontier_indices.append(int(candidate_index))
                continue
            front_index_array = np.asarray(
                frontier_indices,
                dtype=np.int64,
            )
            front_values = unique_values[front_index_array]
            front_no_worse = np.all(
                front_values <= candidate + float(epsilon),
                axis=1,
            )
            front_strictly_better = np.any(
                front_values < candidate - float(epsilon),
                axis=1,
            )
            if np.any(front_no_worse & front_strictly_better):
                continue
            candidate_no_worse = np.all(
                candidate <= front_values + float(epsilon),
                axis=1,
            )
            candidate_strictly_better = np.any(
                candidate < front_values - float(epsilon),
                axis=1,
            )
            dominated_front = (
                candidate_no_worse & candidate_strictly_better
            )
            if dominated_front.any():
                frontier_indices = [
                    index
                    for index, dominated in zip(
                        frontier_indices,
                        dominated_front,
                    )
                    if not bool(dominated)
                ]
            frontier_indices.append(int(candidate_index))
        frontier_unique = np.zeros(len(unique_values), dtype=bool)
        frontier_unique[np.asarray(frontier_indices, dtype=int)] = True
        keep = frontier_unique[inverse]
        kept_parts.append(group.loc[keep])

    result = pd.concat(kept_parts, ignore_index=False, sort=False)
    result = result.sort_values(
        "_pareto_input_order",
        kind="mergesort",
    ).drop(columns="_pareto_input_order")
    if not keep_group_cols and groups:
        result = result.drop(columns=groups)

    return result.reset_index(drop=True)


def pareto_front_with_witness(
    df: pd.DataFrame,
    *,
    minimize_cols: Sequence[str],
    maximize_cols: Sequence[str] = (),
    epsilon: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Return Pareto membership and one final-front dominator per row."""
    objectives = [
        *minimize_cols,
        *maximize_cols,
    ]
    values = df[objectives].to_numpy(dtype=float, copy=True)

    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be finite.")

    if maximize_cols:
        start = len(minimize_cols)
        values[:, start:] *= -1.0

    unique_values, inverse = np.unique(
        values,
        axis=0,
        return_inverse=True,
    )
    order = np.argsort(unique_values[:, 0], kind="mergesort")
    frontier: list[int] = []

    for candidate_index in order:
        candidate = unique_values[candidate_index]
        if not frontier:
            frontier.append(int(candidate_index))
            continue

        front_indices = np.asarray(frontier, dtype=int)
        front_values = unique_values[front_indices]

        front_dominates = (
            np.all(
                front_values <= candidate + epsilon,
                axis=1,
            )
            & np.any(
                front_values < candidate - epsilon,
                axis=1,
            )
        )
        if front_dominates.any():
            continue

        candidate_dominates = (
            np.all(
                candidate <= front_values + epsilon,
                axis=1,
            )
            & np.any(
                candidate < front_values - epsilon,
                axis=1,
            )
        )
        frontier = [
            index
            for index, dominated in zip(
                frontier,
                candidate_dominates,
            )
            if not dominated
        ]
        frontier.append(int(candidate_index))

    unique_front = np.zeros(len(unique_values), dtype=bool)
    unique_front[np.asarray(frontier, dtype=int)] = True
    front_mask = unique_front[inverse]

    witness = np.full(len(df), "", dtype=object)
    final_indices = np.flatnonzero(front_mask)
    final_values = values[final_indices]

    for row_index in np.flatnonzero(~front_mask):
        candidate = values[row_index]
        dominates = (
            np.all(
                final_values <= candidate + epsilon,
                axis=1,
            )
            & np.any(
                final_values < candidate - epsilon,
                axis=1,
            )
        )
        if dominates.any():
            witness[row_index] = str(
                df.iloc[
                    final_indices[np.flatnonzero(dominates)[0]]
                ]["model_id"]
            )

    return (
        pd.Series(front_mask, index=df.index),
        pd.Series(witness, index=df.index),
    )


def sequential_pareto_filter(
    df: pd.DataFrame,
    passes: Sequence[dict],
    id_col: str = "selected_config_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply several dominance/Pareto filtering passes.

    Each pass is a dict:
    {
        "name": "...",
        "group_cols": [...],
        "minimize_cols": [...],
        "maximize_cols": [...],
        "epsilon": 0.0,
    }

    Returns:
    - filtered dataframe
    - audit dataframe with n_before/n_after per pass
    """
    current = df.copy()
    audit_rows = []

    for p in passes:
        n_before = len(current)

        current = pareto_front_by_group(
            current,
            group_cols=p.get("group_cols", []),
            minimize_cols=p.get("minimize_cols", ["fn_rate", "fp_rate"]),
            maximize_cols=p.get("maximize_cols", ["balanced_accuracy"]),
            epsilon=float(p.get("epsilon", 0.0)),
        ).copy()

        n_after = len(current)

        current[f"kept_after_{p['name']}"] = True

        audit_rows.append({
            "pass_name": p["name"],
            "n_before": int(n_before),
            "n_after": int(n_after),
            "n_removed": int(n_before - n_after),
            "removed_rate": (n_before - n_after) / max(n_before, 1),
        })

    return current.reset_index(drop=True), pd.DataFrame(audit_rows)


def summarize_parameter_tendencies(
    df: pd.DataFrame,
    top_fraction: float = 0.15,
    min_top_n: int = 20,
) -> pd.DataFrame:
    """Summarize which hyperparameters appear most often among top-ranked models."""
    parameter_cols = [
        "matrix_family",
        "training_matrix_id",
        "matrix_method",
        "preprocessing",
        "selected_rule_name",
        "rule",
        "rule_variant",
        "n_components",
        "alpha",
        "object_threshold",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
        "m",
        "balanced_pixel_strategy",
    ]
    parameter_cols = [col for col in parameter_cols if col in df.columns]
    if "matrix_family" not in df.columns:
        return pd.DataFrame(columns=["parameter", "value", "count", "rate_in_top_models", "matrix_family", "n_top_models"])

    rows = []
    for family, group in df.groupby("matrix_family", dropna=False):
        ranked = sort_detection_selection(group).reset_index(drop=True)
        n_top = max(int(np.ceil(len(ranked) * float(top_fraction))), int(min_top_n))
        n_top = min(n_top, len(ranked))
        top = ranked.head(n_top).copy()

        for col in parameter_cols:
            if col == "matrix_family":
                continue
            vc = top[col].astype(str).value_counts(dropna=False).reset_index()
            vc.columns = ["value", "count"]
            vc["parameter"] = col
            vc["rate_in_top_models"] = vc["count"] / len(top)
            vc["matrix_family"] = family
            vc["n_top_models"] = len(top)
            rows.append(vc)

    if not rows:
        return pd.DataFrame(columns=["parameter", "value", "count", "rate_in_top_models", "matrix_family", "n_top_models"])

    out = pd.concat(rows, ignore_index=True, sort=False)
    return (
        out[["parameter", "value", "count", "rate_in_top_models", "matrix_family", "n_top_models"]]
        .sort_values(["matrix_family", "parameter", "count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def summarize_ablation_effects(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    group_cols: Sequence[str] | None = ("matrix_family",),
    metric_cols: Sequence[str] = (
        "balanced_accuracy",
        "target_sensitivity",
        "non_target_specificity",
        "fn_rate",
        "fp_rate",
    ),
) -> pd.DataFrame:
    """
    Summarize model performance by one hyperparameter/factor at a time.

    This is intended for ablation-style interpretation of grid-search results.
    It does not refit models.

    Important:
    factor_value is stored as string because it mixes categorical and numeric
    hyperparameters such as preprocessing, alpha, n_components, and rule_variant.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    factor_cols = [col for col in factor_cols if col in d.columns]
    metric_cols = [col for col in metric_cols if col in d.columns]

    if group_cols is None:
        group_cols = []
    group_cols = [col for col in list(group_cols) if col in d.columns]

    for col in metric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    def _label_value(value):
        if value is None:
            return pd.NA
        try:
            if pd.isna(value):
                return pd.NA
        except Exception:
            pass
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    def _numeric_value(value):
        try:
            value = float(value)
        except Exception:
            return np.nan
        return value if np.isfinite(value) else np.nan

    rows = []

    for factor in factor_cols:
        current_group_cols = group_cols + [factor]
        grouped = d.groupby(current_group_cols, dropna=False)

        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)

            row = {
                col: _label_value(value)
                for col, value in zip(current_group_cols, key)
            }

            factor_raw_value = key[-1]

            row["factor"] = str(factor)
            row["factor_value"] = _label_value(factor_raw_value)
            row["factor_value_numeric"] = _numeric_value(factor_raw_value)
            row["n_configs"] = int(len(group))

            for metric in metric_cols:
                values = pd.to_numeric(group[metric], errors="coerce")

                row[f"{metric}_mean"] = (
                    float(values.mean())
                    if values.notna().any()
                    else np.nan
                )
                row[f"{metric}_median"] = (
                    float(values.median())
                    if values.notna().any()
                    else np.nan
                )
                row[f"{metric}_std"] = (
                    float(values.std(ddof=0))
                    if values.notna().any()
                    else np.nan
                )
                row[f"{metric}_min"] = (
                    float(values.min())
                    if values.notna().any()
                    else np.nan
                )
                row[f"{metric}_max"] = (
                    float(values.max())
                    if values.notna().any()
                    else np.nan
                )

            rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    # Force label columns to string dtype for Parquet stability.
    label_cols = list(set(group_cols + factor_cols + ["factor", "factor_value"]))
    for col in label_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")

    sort_cols = [
        col for col in [
            "matrix_family",
            "factor",
            "fn_rate_mean",
            "fp_rate_mean",
            "balanced_accuracy_mean",
        ]
        if col in out.columns
    ]

    ascending = [
        False if col == "balanced_accuracy_mean" else True
        for col in sort_cols
    ]

    return (
        out
        .sort_values(sort_cols, ascending=ascending)
        .reset_index(drop=True)
    )


def summarize_metric_stability(
    df: pd.DataFrame,
    config_cols: Sequence[str],
    metric_cols: Sequence[str] = (
        "balanced_accuracy",
        "target_sensitivity",
        "non_target_specificity",
        "fn_rate",
        "fp_rate",
    ),
    seed_col: str = "random_state",
) -> pd.DataFrame:
    """
    Summarize metric variability across random seeds for each selected config.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    config_cols = [col for col in config_cols if col in d.columns]
    metric_cols = [col for col in metric_cols if col in d.columns]

    if not config_cols:
        raise ValueError("At least one config column must be available.")

    for col in metric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    rows = []

    for key, group in d.groupby(config_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            col: value
            for col, value in zip(config_cols, key)
        }

        row["n_runs"] = int(len(group))
        row["n_random_states"] = int(group[seed_col].nunique()) if seed_col in group.columns else np.nan

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
            row[f"min_{metric}"] = float(values.min()) if values.notna().any() else np.nan
            row[f"max_{metric}"] = float(values.max()) if values.notna().any() else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    sort_cols = [
        col for col in [
            "mean_fn_rate",
            "std_fn_rate",
            "mean_fp_rate",
            "std_fp_rate",
            "mean_balanced_accuracy",
        ]
        if col in out.columns
    ]

    ascending = [
        col != "mean_balanced_accuracy"
        for col in sort_cols
    ]

    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
