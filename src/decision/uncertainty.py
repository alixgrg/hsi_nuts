from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    DEFAULT_NON_TARGET_LABEL,
    UNCERTAIN_LABEL,
    pixel_ratio_col,
    true_col as make_true_col,
)
from src.decision.metrics import coerce_binary_series

def add_three_way_object_decision(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    uncertain_label: str = UNCERTAIN_LABEL,
    ratio_col: str | None = None,
    lower_threshold: float = 0.40,
    upper_threshold: float = 0.75,
    output_col: str = "decision_3way",
) -> pd.DataFrame:
    """
    Add a three-way decision:
        non_target_class
        uncertain
        target_class

    Useful when false negatives are very costly and uncertain objects
    should be inspected manually.
    """
    if ratio_col is None:
        ratio_col = pixel_ratio_col(target_class)
    if ratio_col not in object_df.columns:
        raise KeyError(f"Missing ratio column: {ratio_col}")

    df = object_df.copy()
    ratio = df[ratio_col].astype(float)
    df[output_col] = np.where(
        ratio >= upper_threshold,
        target_class,
        np.where(
            ratio < lower_threshold,
            non_target_label,
            uncertain_label,
        ),
    )
    df["three_way_lower_threshold"] = float(lower_threshold)
    df["three_way_upper_threshold"] = float(upper_threshold)

    return df


def summarize_three_way_decision(
    object_df: pd.DataFrame,
    decision_col: str = "decision_3way",
) -> pd.DataFrame:
    """Summarize three-way decision counts and rates."""
    counts = object_df[decision_col].value_counts(dropna=False).rename("n").reset_index()
    counts = counts.rename(columns={"index": decision_col})
    counts["rate"] = counts["n"] / len(object_df) if len(object_df) > 0 else np.nan

    return counts


def evaluate_three_way_object_decision(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    uncertain_label: str = UNCERTAIN_LABEL,
    decision_col: str = "decision_3way",
    true_col: str | None = None,
    truth_available_ratio_col: str = "truth_available_ratio",
    min_truth_available_ratio: float = 0.50,
) -> dict:
    """
    Evaluate a three-way object decision.

    Three possible decisions:
    - target_class
    - non_target_class
    - uncertain

    Interpretation for screening:
    - target or uncertain = object should be inspected / kept
    - non_target = object rejected as non-target

    Main quantity of interest:
    - target_miss_rate: true target classified as non_target
    - uncertain_rate: proportion sent to manual/ambiguous category
    - non_target_false_accept_rate: true non-target classified as target
    """
    if true_col is None:
        true_col = make_true_col(target_class, 'object')
    if decision_col not in object_df.columns:
        raise KeyError(f"Missing decision column: {decision_col}")
    if true_col not in object_df.columns:
        raise KeyError(f"Missing true column: {true_col}")

    d = object_df.copy()
    valid = d[true_col].notna() & d[decision_col].notna()

    if truth_available_ratio_col in d.columns:
        truth_ratio = pd.to_numeric(
            d[truth_available_ratio_col],
            errors="coerce",
        )
        valid = valid & (truth_ratio >= float(min_truth_available_ratio))

    d = d[valid].copy()
    if len(d) == 0:
        return {
            "n": 0,
            "n_uncertain": 0,
            "uncertain_rate": np.nan,
            "coverage_rate": np.nan,
            "target_miss_rate": np.nan,
            "screening_sensitivity": np.nan,
            "non_target_false_accept_rate": np.nan,
            "non_target_auto_reject_rate": np.nan,
            "decided_accuracy": np.nan,
            "decided_balanced_accuracy": np.nan,
        }

    true_target_s = coerce_binary_series(
        d[true_col], target_class=target_class, non_target_class=non_target_label
    )
    d = d.loc[true_target_s.notna()].copy()
    true_target = true_target_s.loc[d.index].astype(bool).to_numpy()
    decision = d[decision_col].astype(str).to_numpy()

    target_label = str(target_class)
    is_target_decision = decision == target_label
    is_non_target_decision = decision == non_target_label
    is_uncertain = decision == uncertain_label
    is_decided = ~is_uncertain

    n = int(len(d))
    n_target = int(np.sum(true_target))
    n_non_target = int(np.sum(~true_target))

    n_uncertain = int(np.sum(is_uncertain))
    uncertain_rate = n_uncertain / n if n > 0 else np.nan
    coverage_rate = 1.0 - uncertain_rate if np.isfinite(uncertain_rate) else np.nan

    # Critical screening errors.
    target_missed = true_target & is_non_target_decision
    non_target_false_accept = (~true_target) & is_target_decision

    target_miss_rate = (
        float(np.sum(target_missed)) / n_target
        if n_target > 0
        else np.nan
    )

    screening_sensitivity = (
        1.0 - target_miss_rate
        if np.isfinite(target_miss_rate)
        else np.nan
    )

    non_target_false_accept_rate = (
        float(np.sum(non_target_false_accept)) / n_non_target
        if n_non_target > 0
        else np.nan
    )

    non_target_auto_reject_rate = (
        float(np.sum((~true_target) & is_non_target_decision)) / n_non_target
        if n_non_target > 0
        else np.nan
    )

    target_auto_accept_rate = (
        float(np.sum(true_target & is_target_decision)) / n_target
        if n_target > 0
        else np.nan
    )

    target_uncertain_rate = (
        float(np.sum(true_target & is_uncertain)) / n_target
        if n_target > 0
        else np.nan
    )

    non_target_uncertain_rate = (
        float(np.sum((~true_target) & is_uncertain)) / n_non_target
        if n_non_target > 0
        else np.nan
    )

    # Metrics only among decided objects.
    if np.sum(is_decided) > 0:
        y_true_decided = true_target[is_decided]
        y_pred_decided = is_target_decision[is_decided]

        tp = int(np.sum(y_true_decided & y_pred_decided))
        fn = int(np.sum(y_true_decided & (~y_pred_decided)))
        fp = int(np.sum((~y_true_decided) & y_pred_decided))
        tn = int(np.sum((~y_true_decided) & (~y_pred_decided)))

        sens = tp / (tp + fn) if tp + fn > 0 else np.nan
        spec = tn / (tn + fp) if tn + fp > 0 else np.nan

        decided_accuracy = (tp + tn) / len(y_true_decided)
        decided_balanced_accuracy = (
            0.5 * (sens + spec)
            if np.isfinite(sens) and np.isfinite(spec)
            else np.nan
        )
    else:
        tp = fn = fp = tn = 0
        decided_accuracy = np.nan
        decided_balanced_accuracy = np.nan

    return {
        "n": n,
        "n_target": n_target,
        "n_non_target": n_non_target,
        "n_uncertain": n_uncertain,
        "uncertain_rate": uncertain_rate,
        "coverage_rate": coverage_rate,
        "target_miss_rate": target_miss_rate,
        "screening_sensitivity": screening_sensitivity,
        "target_auto_accept_rate": target_auto_accept_rate,
        "target_uncertain_rate": target_uncertain_rate,
        "non_target_false_accept_rate": non_target_false_accept_rate,
        "non_target_auto_reject_rate": non_target_auto_reject_rate,
        "non_target_uncertain_rate": non_target_uncertain_rate,
        "decided_tp": tp,
        "decided_fn": fn,
        "decided_fp": fp,
        "decided_tn": tn,
        "decided_accuracy": decided_accuracy,
        "decided_balanced_accuracy": decided_balanced_accuracy,
    }


def three_way_object_threshold_grid(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    uncertain_label: str = UNCERTAIN_LABEL,
    lower_thresholds=(0.10, 0.20, 0.30, 0.40, 0.50),
    upper_thresholds=(0.60, 0.70, 0.75, 0.80, 0.90),
    ratio_col: str | None = None,
    true_col: str | None = None,
    decision_col: str = "decision_3way",
) -> pd.DataFrame:
    """
    Evaluate several lower/upper thresholds for the three-way object decision.
    """
    rows = []

    for lower in lower_thresholds:
        for upper in upper_thresholds:
            if float(lower) >= float(upper):
                continue

            tmp = add_three_way_object_decision(
                object_df=object_df,
                target_class=target_class,
                non_target_label=non_target_label,
                uncertain_label=uncertain_label,
                ratio_col=ratio_col,
                lower_threshold=float(lower),
                upper_threshold=float(upper),
                output_col=decision_col,
            )

            metrics = evaluate_three_way_object_decision(
                object_df=tmp,
                target_class=target_class,
                non_target_label=non_target_label,
                uncertain_label=uncertain_label,
                decision_col=decision_col,
                true_col=true_col,
            )

            metrics["three_way_lower_threshold"] = float(lower)
            metrics["three_way_upper_threshold"] = float(upper)

            rows.append(metrics)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
            ],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def three_way_object_threshold_grid_by_group(
    object_df: pd.DataFrame,
    group_cols,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    uncertain_label: str = UNCERTAIN_LABEL,
    lower_thresholds=(0.10, 0.20, 0.30, 0.40, 0.50),
    upper_thresholds=(0.60, 0.70, 0.75, 0.80, 0.90),
    ratio_col: str | None = None,
    true_col: str | None = None,
    decision_col: str = "decision_3way",
) -> pd.DataFrame:
    """
    Evaluate three-way thresholds independently for each model/config group.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    rows = []

    for key, group in object_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        summary = three_way_object_threshold_grid(
            object_df=group,
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
            lower_thresholds=lower_thresholds,
            upper_thresholds=upper_thresholds,
            ratio_col=ratio_col,
            true_col=true_col,
            decision_col=decision_col,
        )

        for col, value in zip(group_cols, key):
            summary[col] = value

        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows, ignore_index=True, sort=False)
        .sort_values(
            [
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
            ],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def select_three_way_threshold_one_config(
        group: pd.DataFrame,
        max_target_miss_rate: float,
        max_false_accept_rate: float,
        max_uncertain_rate: float) -> pd.Series:
    group = group.copy()

    eligible = group[
        (group["target_miss_rate"].fillna(1.0) <= max_target_miss_rate)
        & (group["non_target_false_accept_rate"].fillna(1.0) <= max_false_accept_rate)
        & (group["uncertain_rate"].fillna(1.0) <= max_uncertain_rate)
    ].copy()

    if eligible.empty:
        eligible = group.copy()

    return (
        eligible
        .sort_values(
            [
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
            ],
            ascending=[True, True, True],
        )
        .iloc[0]
    )


def select_three_way_threshold_pareto(
    threshold_grid_df: pd.DataFrame,
    max_target_miss_rate: float = 0.00,
    max_false_accept_rate: float | None = None,
    max_uncertain_rate: float | None = None,
    min_coverage: float | None = None,
    allow_infeasible_fallback: bool = True,
) -> pd.Series:
    """
    Select one lower/upper threshold pair using constraints and Pareto logic.

    Priority:
    1. target_miss_rate
    2. non_target_false_accept_rate
    3. uncertain_rate
    4. coverage_rate
    """
    # Local import avoids a decision/workflows package initialization cycle.
    from src.workflows.simca_selection_utils import pareto_front_by_group

    df = threshold_grid_df.copy()

    for col in [
        "target_miss_rate",
        "non_target_false_accept_rate",
        "uncertain_rate",
        "coverage_rate",
        "decided_balanced_accuracy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feasible = df.copy()

    feasible = feasible[
        feasible["target_miss_rate"].fillna(1.0) <= float(max_target_miss_rate)
    ].copy()

    if max_false_accept_rate is not None and len(feasible) > 0:
        feasible = feasible[
            feasible["non_target_false_accept_rate"].fillna(1.0)
            <= float(max_false_accept_rate)
        ].copy()

    if max_uncertain_rate is not None and len(feasible) > 0:
        feasible = feasible[
            feasible["uncertain_rate"].fillna(1.0)
            <= float(max_uncertain_rate)
        ].copy()

    if min_coverage is not None and len(feasible) > 0:
        feasible = feasible[
            feasible["coverage_rate"].fillna(0.0)
            >= float(min_coverage)
        ].copy()

    if feasible.empty:
        if allow_infeasible_fallback:
            feasible = df.copy()
        else:
            return pd.Series(
                {
                    "three_way_lower_threshold": np.nan,
                    "three_way_upper_threshold": np.nan,
                    "selection_status": (
                        "technically_calculable_but_not_acceptable"
                    ),
                    "feasible": False,
                    "n_feasible": 0,
                }
            )

    front = pareto_front_by_group(
        feasible,
        group_cols=[],
        minimize_cols=[
            "target_miss_rate",
            "non_target_false_accept_rate",
            "uncertain_rate",
        ],
        maximize_cols=["coverage_rate"],
    )

    selected = (
        front.sort_values(
            [
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
                "coverage_rate",
            ],
            ascending=[True, True, True, False],
        )
        .iloc[0]
    )
    selected = selected.copy()
    selected["selection_status"] = "acceptable"
    selected["feasible"] = True
    selected["n_feasible"] = int(len(feasible))
    return selected


def calibrate_three_way_thresholds_by_config(
    object_df: pd.DataFrame,
    config_cols: Sequence[str],
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    lower_thresholds=np.round(np.arange(0.05, 0.61, 0.05), 2),
    upper_thresholds=np.round(np.arange(0.40, 0.96, 0.05), 2),
    max_target_miss_rate: float = 0.00,
    max_false_accept_rate: float | None = None,
    max_uncertain_rate: float | None = None,
    min_coverage: float | None = None,
    allow_infeasible_fallback: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each selected model/config, evaluate all 3-way thresholds
    and select one lower/upper pair.
    """
    config_cols = [col for col in config_cols if col in object_df.columns]

    grid_parts = []
    selected_rows = []

    for key, group in object_df.groupby(config_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        grid = three_way_object_threshold_grid(
            object_df=group,
            target_class=target_class,
            non_target_label=non_target_label,
            lower_thresholds=lower_thresholds,
            upper_thresholds=upper_thresholds,
        )

        for col, value in zip(config_cols, key):
            grid[col] = value

        selected = select_three_way_threshold_pareto(
            grid,
            max_target_miss_rate=max_target_miss_rate,
            max_false_accept_rate=max_false_accept_rate,
            max_uncertain_rate=max_uncertain_rate,
            min_coverage=min_coverage,
            allow_infeasible_fallback=allow_infeasible_fallback,
        )

        grid_parts.append(grid)
        selected_rows.append(selected)

    grid_df = pd.concat(grid_parts, ignore_index=True, sort=False) if grid_parts else pd.DataFrame()
    selected_df = pd.DataFrame(selected_rows)

    return grid_df, selected_df


def apply_three_way_thresholds_by_config(
    object_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    config_id_col: str = "selected_config_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> pd.DataFrame:
    """
    Apply previously selected 3-way thresholds to an object-level table.
    No calibration is done here.
    """
    if object_df is None or len(object_df) == 0:
        return pd.DataFrame() if object_df is None else object_df.copy()

    if config_id_col not in object_df.columns:
        raise KeyError(f"Missing {config_id_col!r} in object_df.")
    if config_id_col not in thresholds_df.columns:
        raise KeyError(f"Missing {config_id_col!r} in thresholds_df.")

    threshold_cols = (
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    )
    missing = [col for col in threshold_cols if col not in thresholds_df.columns]
    if missing:
        raise KeyError(f"Missing 3-way threshold column(s): {missing}")

    lookup = thresholds_df[
        [config_id_col, *threshold_cols]
    ].drop_duplicates(config_id_col)
    out = object_df.drop(
        columns=list(threshold_cols),
        errors="ignore",
    ).merge(
        lookup,
        on=config_id_col,
        how="left",
        validate="many_to_one",
    )

    lower = pd.to_numeric(out["three_way_lower_threshold"], errors="coerce")
    upper = pd.to_numeric(out["three_way_upper_threshold"], errors="coerce")
    invalid = ~np.isfinite(lower) | ~np.isfinite(upper) | lower.ge(upper)
    if invalid.any():
        invalid_ids = (
            out.loc[invalid, config_id_col].astype(str).drop_duplicates().tolist()
        )
        raise ValueError(
            "Missing or invalid fixed 3-way thresholds for config(s): "
            f"{invalid_ids[:10]}"
        )

    ratio_col = pixel_ratio_col(target_class)
    if ratio_col not in out.columns:
        raise KeyError(f"Missing ratio column: {ratio_col}")
    ratio = pd.to_numeric(out[ratio_col], errors="coerce")
    if not np.isfinite(ratio).all():
        raise ValueError(f"{ratio_col!r} contains NaN or Inf.")

    out["decision_3way"] = np.select(
        (ratio.ge(upper), ratio.lt(lower)),
        (target_class, non_target_label),
        default=UNCERTAIN_LABEL,
    )
    return out


def evaluate_three_way_by_config(
    object_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    config_id_col: str = "selected_config_id",
    extra_group_cols: Sequence[str] = (),
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply fixed 3-way thresholds and evaluate metrics by configuration.
    """
    if object_df is None or len(object_df) == 0:
        return pd.DataFrame(), pd.DataFrame() if object_df is None else object_df.copy()

    objects_3way_df = apply_three_way_thresholds_by_config(
        object_df=object_df,
        thresholds_df=thresholds_df,
        config_id_col=config_id_col,
        target_class=target_class,
        non_target_label=non_target_label,
    )

    if len(objects_3way_df) == 0:
        return pd.DataFrame(), objects_3way_df

    group_cols = [config_id_col] + [
        col for col in extra_group_cols
        if col in objects_3way_df.columns
    ]

    rows = []

    for key, group in objects_3way_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        metrics = evaluate_three_way_object_decision(
            group,
            target_class=target_class,
            non_target_label=non_target_label,
        )

        for col, value in zip(group_cols, key):
            metrics[col] = value

        rows.append(metrics)

    return pd.DataFrame(rows), objects_3way_df


def add_three_way_confidence(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    uncertain_label: str = UNCERTAIN_LABEL,
    ratio_col: str | None = None,
    decision_col: str = "decision_3way",
    lower_col: str = "three_way_lower_threshold",
    upper_col: str = "three_way_upper_threshold",
    output_margin_col: str = "three_way_margin",
    output_confidence_col: str = "three_way_confidence",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Add a simple confidence score for a fixed 3-way object decision.

    Confidence is based on the distance to the closest decision boundary:
    - non_target: distance below lower threshold;
    - target: distance above upper threshold;
    - uncertain: distance inside the uncertainty interval.

    The score is normalized to [0, 1].
    """
    if object_df is None or len(object_df) == 0:
        return pd.DataFrame() if object_df is None else object_df.copy()

    if ratio_col is None:
        ratio_col = pixel_ratio_col(target_class)

    required = [ratio_col, decision_col, lower_col, upper_col]
    missing = [col for col in required if col not in object_df.columns]

    if missing:
        raise KeyError(f"Missing columns for 3-way confidence: {missing}")

    df = object_df.copy()

    ratio = pd.to_numeric(df[ratio_col], errors="coerce")
    lower = pd.to_numeric(df[lower_col], errors="coerce")
    upper = pd.to_numeric(df[upper_col], errors="coerce")
    decision = df[decision_col].astype(str)

    margin = pd.Series(np.nan, index=df.index, dtype="float64")
    confidence = pd.Series(np.nan, index=df.index, dtype="float64")

    mask_target = decision.eq(str(target_class))
    mask_non_target = decision.eq(str(non_target_label))
    mask_uncertain = decision.eq(str(uncertain_label))

    margin.loc[mask_target] = ratio.loc[mask_target] - upper.loc[mask_target]
    confidence.loc[mask_target] = margin.loc[mask_target] / (
        1.0 - upper.loc[mask_target]
    ).clip(lower=eps)

    margin.loc[mask_non_target] = lower.loc[mask_non_target] - ratio.loc[mask_non_target]
    confidence.loc[mask_non_target] = margin.loc[mask_non_target] / (
        lower.loc[mask_non_target]
    ).clip(lower=eps)

    width = (upper - lower).clip(lower=eps)
    midpoint = 0.5 * (lower + upper)

    margin.loc[mask_uncertain] = (
        0.5 * width.loc[mask_uncertain]
        - (ratio.loc[mask_uncertain] - midpoint.loc[mask_uncertain]).abs()
    )

    confidence.loc[mask_uncertain] = margin.loc[mask_uncertain] / (
        0.5 * width.loc[mask_uncertain]
    ).clip(lower=eps)

    df[output_margin_col] = margin
    df[output_confidence_col] = confidence.clip(lower=0.0, upper=1.0)

    df["three_way_confidence_bin"] = pd.cut(
        df[output_confidence_col],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["low", "medium", "high"],
    ).astype("object")

    return df


def three_way_confusion_table(
    df: pd.DataFrame,
    true_col: str,
    decision_col: str = "decision_3way",
    confidence_col: str | None = "three_way_confidence",
    group_cols: Sequence[str] = (),
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    uncertain_label: str = UNCERTAIN_LABEL,
) -> pd.DataFrame:
    """
    Build a long-format 3-way confusion table.

    Rows are true labels:
    - target_class
    - non_target_label

    Columns are 3-way decisions:
    - non_target_label
    - uncertain_label
    - target_class
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    group_cols = [col for col in group_cols if col in df.columns]

    required = [true_col, decision_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns for 3-way confusion table: {missing}")

    d = df.copy()
    d = d[d[true_col].notna() & d[decision_col].notna()].copy()

    if len(d) == 0:
        return pd.DataFrame()

    d["true_label_3way"] = np.where(
        coerce_binary_series(
            d[true_col], target_class=target_class, non_target_class=non_target_label
        ).fillna(False).astype(bool),
        target_class,
        non_target_label,
    )

    decisions = [
        non_target_label,
        uncertain_label,
        target_class,
    ]

    rows = []

    for key, group in d.groupby(group_cols, dropna=False) if group_cols else [((), d)]:
        if not isinstance(key, tuple):
            key = (key,)

        base = {
            col: value
            for col, value in zip(group_cols, key)
        }

        n_group = len(group)

        for true_label in [non_target_label, target_class]:
            true_group = group[group["true_label_3way"].astype(str).eq(str(true_label))]
            n_true = len(true_group)

            for decision in decisions:
                cell = true_group[
                    true_group[decision_col].astype(str).eq(str(decision))
                ]

                row = dict(base)
                row.update(
                    {
                        "true_label_3way": true_label,
                        "decision_3way": decision,
                        "n": int(len(cell)),
                        "n_true_label": int(n_true),
                        "n_group": int(n_group),
                        "row_rate": len(cell) / max(n_true, 1),
                        "global_rate": len(cell) / max(n_group, 1),
                    }
                )

                if confidence_col is not None and confidence_col in cell.columns:
                    conf = pd.to_numeric(cell[confidence_col], errors="coerce")
                    row["mean_confidence"] = float(conf.mean()) if conf.notna().any() else np.nan
                    row["median_confidence"] = float(conf.median()) if conf.notna().any() else np.nan
                else:
                    row["mean_confidence"] = np.nan
                    row["median_confidence"] = np.nan

                rows.append(row)

    return pd.DataFrame(rows)
