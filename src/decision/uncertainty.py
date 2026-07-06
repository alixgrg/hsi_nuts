from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    DEFAULT_NON_TARGET_LABEL,
    UNCERTAIN_LABEL,
    pixel_ratio_col,
    true_col as make_true_col,
)

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
            "three_way_score": np.nan,
        }

    true_target = d[true_col].astype(bool).to_numpy()
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

    # High score = few missed targets, few false accepts, not too many uncertain.
    three_way_score = (
        -20.0 * (target_miss_rate if np.isfinite(target_miss_rate) else 1.0)
        -3.0 * (
            non_target_false_accept_rate
            if np.isfinite(non_target_false_accept_rate)
            else 1.0
        )
        -0.5 * (uncertain_rate if np.isfinite(uncertain_rate) else 1.0)
        +1.0 * (
            screening_sensitivity
            if np.isfinite(screening_sensitivity)
            else 0.0
        )
        +0.25 * (
            non_target_auto_reject_rate
            if np.isfinite(non_target_auto_reject_rate)
            else 0.0
        )
    )

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
        "three_way_score": float(three_way_score),
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
                "three_way_score",
            ],
            ascending=[True, True, True, False],
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
                "three_way_score",
            ],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )