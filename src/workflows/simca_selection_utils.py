from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.decision.labels import DEFAULT_NON_TARGET_LABEL, DEFAULT_TARGET_CLASS
from src.utils import first_available_value, is_missing_value


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
}


def _float_metric(metrics: Mapping[str, Any], *names: str, default: float = np.nan) -> float:
    for name in names:
        value = metrics.get(name, np.nan)
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if np.isfinite(value):
            return value
    return float(default)


def detection_selection_score(
    metrics: Mapping[str, Any],
    objective_metric: str = "fn_fp_hierarchical",
    fn_weight: float = 10.0,
    fp_weight: float = 1.0,
    f1_weight: float = 0.05,
    accuracy_weight: float = 0.02,
    balanced_accuracy_weight: float = 0.0,
    min_target_sensitivity: float | None = None,
    min_non_target_specificity: float | None = None,
    constraint_penalty: float = 2.0,
) -> float:
    """Return a scalar score for binary target-vs-non-target model selection.

    Higher is better. The default objective prioritizes false negatives first,
    then false positives, then weak tie-breakers such as F1 and accuracy.
    """
    target_sens = _float_metric(metrics, "target_sensitivity")
    non_target_spec = _float_metric(metrics, "non_target_specificity")
    fn_rate = _float_metric(metrics, "fn_rate")
    fp_rate = _float_metric(metrics, "fp_rate")
    f1 = _float_metric(metrics, "f1_score", default=0.0)
    acc = _float_metric(metrics, "accuracy", default=0.0)
    ba = _float_metric(metrics, "balanced_accuracy", default=0.0)

    if not np.isfinite(fn_rate) and np.isfinite(target_sens):
        fn_rate = 1.0 - target_sens
    if not np.isfinite(fp_rate) and np.isfinite(non_target_spec):
        fp_rate = 1.0 - non_target_spec

    if objective_metric == "fn_fp_hierarchical":
        if not np.isfinite(fn_rate) or not np.isfinite(fp_rate):
            return -np.inf
        score = (
            -float(fn_weight) * fn_rate
            -float(fp_weight) * fp_rate
            +float(f1_weight) * (f1 if np.isfinite(f1) else 0.0)
            +float(accuracy_weight) * (acc if np.isfinite(acc) else 0.0)
            +float(balanced_accuracy_weight) * (ba if np.isfinite(ba) else 0.0)
        )
    elif objective_metric == "balanced_accuracy":
        score = ba
    elif objective_metric in metrics:
        score = _float_metric(metrics, objective_metric, default=-np.inf)
    else:
        raise ValueError(
            "objective_metric must be 'fn_fp_hierarchical', 'balanced_accuracy', "
            "or a metric column present in the input."
        )

    if not np.isfinite(score):
        return -np.inf

    if min_target_sensitivity is not None and np.isfinite(target_sens):
        score -= float(constraint_penalty) * max(0.0, float(min_target_sensitivity) - target_sens)
    if min_non_target_specificity is not None and np.isfinite(non_target_spec):
        score -= float(constraint_penalty) * max(0.0, float(min_non_target_specificity) - non_target_spec)

    return float(score)


def add_detection_selection_score(
    df: pd.DataFrame,
    score_col: str = "selection_score",
    **score_kwargs,
) -> pd.DataFrame:
    """Add a scalar selection score to a binary detection result table."""
    out = df.copy()

    if "fn_rate" not in out.columns and "target_sensitivity" in out.columns:
        out["fn_rate"] = 1.0 - pd.to_numeric(out["target_sensitivity"], errors="coerce")
    if "fp_rate" not in out.columns and "non_target_specificity" in out.columns:
        out["fp_rate"] = 1.0 - pd.to_numeric(out["non_target_specificity"], errors="coerce")

    out[score_col] = out.apply(
        lambda row: detection_selection_score(row.to_dict(), **score_kwargs),
        axis=1,
    )
    return out


def sort_detection_selection(
    df: pd.DataFrame,
    score_col: str = "selection_score",
    add_score: bool = True,
) -> pd.DataFrame:
    """Sort a detection result table with a FN-first hierarchy."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = add_detection_selection_score(df, score_col=score_col) if add_score else df.copy()
    sort_cols = [
        "fn_rate",
        "fp_rate",
        "f1_score",
        "accuracy",
        "balanced_accuracy",
        score_col,
    ]
    sort_cols = [col for col in sort_cols if col in out.columns]
    ascending = [col in {"fn_rate", "fp_rate"} for col in sort_cols]
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


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


def add_reference_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few alternative reference scores for sensitivity/specificity trade-offs."""
    out = df.copy()
    for col in ["fn_rate", "fp_rate", "balanced_accuracy", "target_sensitivity", "non_target_specificity", "f1_score", "accuracy"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "fn_rate" not in out.columns and "target_sensitivity" in out.columns:
        out["fn_rate"] = 1.0 - out["target_sensitivity"]
    if "fp_rate" not in out.columns and "non_target_specificity" in out.columns:
        out["fp_rate"] = 1.0 - out["non_target_specificity"]

    out["score_conservative_target"] = (
        -20.0 * out["fn_rate"].fillna(1.0)
        -2.0 * out["fp_rate"].fillna(1.0)
        +1.0 * out["balanced_accuracy"].fillna(0.0)
    )
    out["score_balanced_reference"] = (
        +3.0 * out["balanced_accuracy"].fillna(0.0)
        +1.0 * out.get("f1_score", 0.0)
        -3.0 * out["fn_rate"].fillna(1.0)
        -1.0 * out["fp_rate"].fillna(1.0)
    )
    out["score_specificity_control"] = (
        +3.0 * out["non_target_specificity"].fillna(0.0)
        -5.0 * out["fn_rate"].fillna(1.0)
        -3.0 * out["fp_rate"].fillna(1.0)
    )
    return out


def select_top_by_score(
    df: pd.DataFrame,
    score_col: str,
    n_per_family: int,
    strategy_name: str,
) -> pd.DataFrame:
    """Select top configurations per matrix family according to a score column."""
    out = (
        df.sort_values([score_col, "fn_rate", "fp_rate", "balanced_accuracy"], ascending=[False, True, True, False])
        .groupby("matrix_family", group_keys=False, dropna=False)
        .head(n_per_family)
        .copy()
    )
    out["selection_strategy"] = strategy_name
    return out


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
