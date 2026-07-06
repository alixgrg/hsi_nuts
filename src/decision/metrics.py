from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    DEFAULT_NON_TARGET_LABEL,
    predicted_col as make_predicted_col,
    true_col as make_true_col,
)

def binary_detection_metrics(
    df: pd.DataFrame,
    true_col: str | None = None,
    pred_col: str | None = None,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> dict:
    """
    Compute binary detection metrics.

    Positive class = True = target class
    Negative class = False = non-target class
    """
    if true_col is None:
        true_col = make_true_col(target_class, "object")
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "object")

    d = df.dropna(subset=[true_col, pred_col]).copy()
    empty = {
        "target_class": target_class,
        "non_target_class": non_target_class,
        "n": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "target_sensitivity": np.nan,
        "non_target_specificity": np.nan,
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "f1_score": np.nan,
        "fn_rate": np.nan,
        "fp_rate": np.nan,
    }
    if len(d) == 0:
        return empty
    y_true = d[true_col].astype(bool).to_numpy()
    y_pred = d[pred_col].astype(bool).to_numpy()
    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    n = int(len(d))

    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    ba = 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan
    acc = (tp + tn) / n if n > 0 else np.nan
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = (
        2.0 * precision * sens / (precision + sens)
        if np.isfinite(precision)
        and np.isfinite(sens)
        and (precision + sens) > 0
        else np.nan
    )
    fn_rate = fn / (tp + fn) if tp + fn > 0 else np.nan
    fp_rate = fp / (fp + tn) if fp + tn > 0 else np.nan

    out = {
        "target_class": target_class,
        "non_target_class": non_target_class,
        "n": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "target_sensitivity": sens,
        "non_target_specificity": spec,
        "balanced_accuracy": ba,
        "accuracy": acc,
        "precision": precision,
        "f1_score": f1,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
    }
    return out


def metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    true_col: str | None = None,
    pred_col: str | None = None,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> pd.DataFrame:
    """Compute binary metrics by group, for example by batch or image."""
    rows = []

    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    for group_value, group_df in df.groupby(group_col, dropna=False):
        row = binary_detection_metrics(
            group_df,
            true_col=true_col,
            pred_col=pred_col,
            target_class=target_class,
            non_target_class=non_target_class,
        )
        row[group_col] = group_value
        rows.append(row)

    return pd.DataFrame(rows)


def add_detection_score(
    metrics_df: pd.DataFrame,
    sensitivity_weight: float = 10.0,
    specificity_weight: float = 1.0,
    score_col: str = "detection_score",
) -> pd.DataFrame:
    """
    Add a scalar score useful for model ranking.

    High score favors target sensitivity first, then non-target specificity.
    """
    df = metrics_df.copy()

    sens = df["target_sensitivity"].astype(float)
    spec = df["non_target_specificity"].astype(float)

    df[score_col] = sensitivity_weight * sens + specificity_weight * spec

    return df


def add_binary_confusion_case(
    df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    level: str = "object",
    true_col: str | None = None,
    pred_col: str | None = None,
    output_col: str | None = None,
    truth_available_col: str = "truth_available",
    truth_available_ratio_col: str = "truth_available_ratio",
    min_truth_available_ratio: float = 0.50,
) -> pd.DataFrame:
    """
    Add TP/TN/FP/FN labels to a pixel-level or object-level dataframe.

    level:
        "pixel" or "object"
    """
    if level not in {"pixel", "object"}:
        raise ValueError("level must be 'pixel' or 'object'.")
    if true_col is None:
        true_col = make_true_col(target_class, level)
    if pred_col is None:
        pred_col = make_predicted_col(target_class, level)
    if output_col is None:
        output_col = f"{level}_error_case"

    required = [true_col, pred_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    out = df.copy()

    valid = out[true_col].notna() & out[pred_col].notna()

    if level == "pixel" and truth_available_col in out.columns:
        valid = valid & out[truth_available_col].astype(bool)

    if level == "object" and truth_available_ratio_col in out.columns:
        truth_ratio = pd.to_numeric(
            out[truth_available_ratio_col],
            errors="coerce",
        )
        valid = valid & (
            truth_ratio >= float(min_truth_available_ratio)
        )

    truth = out[true_col].fillna(False).astype(bool)
    pred = out[pred_col].fillna(False).astype(bool)

    out[output_col] = "unavailable"

    out.loc[valid & truth & pred, output_col] = "TP"
    out.loc[valid & (~truth) & (~pred), output_col] = "TN"
    out.loc[valid & (~truth) & pred, output_col] = "FP"
    out.loc[valid & truth & (~pred), output_col] = "FN"

    out[f"{level}_is_error"] = out[output_col].isin(["FP", "FN"])
    out[f"{level}_is_fp"] = out[output_col].eq("FP")
    out[f"{level}_is_fn"] = out[output_col].eq("FN")

    return out


def _normalise_group_cols(group_cols):
    if group_cols is None:
        return ["source_image"]
    if isinstance(group_cols, str):
        return [group_cols]
    return list(group_cols)


def summarize_pixel_errors_by_image(
    pixel_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    group_cols=("source_image",),
    truth_available_col: str = "truth_available",
) -> pd.DataFrame:
    """
    Summarize pixel-level TP/TN/FP/FN by image, model, or any group columns.

    Examples
    --------
    group_cols=("source_image",)
    group_cols=("selected_config_id", "source_image")
    group_cols=("selected_config_id", "matrix_family", "source_image")
    """
    group_cols = _normalise_group_cols(group_cols)

    true_col = make_true_col(target_class, "pixel")
    pred_col = make_predicted_col(target_class, "pixel")

    required = group_cols + [true_col, pred_col]
    missing = [col for col in required if col not in pixel_df.columns]
    if missing:
        raise KeyError(f"Missing columns in pixel_df: {missing}")

    rows = []

    for key, group in pixel_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        g = group.copy()

        if truth_available_col in g.columns:
            g = g[g[truth_available_col].astype(bool)]
        if len(g) == 0:
            continue

        metrics = binary_detection_metrics(
            g,
            true_col=true_col,
            pred_col=pred_col,
            target_class=target_class,
            non_target_class=non_target_label,
        )

        row = {
            col: value
            for col, value in zip(group_cols, key)
        }
        row.update(metrics)
        row["n_truth_pixels"] = metrics["n"]
        row["pixel_accuracy"] = metrics["accuracy"]
        row["pixel_balanced_accuracy"] = metrics["balanced_accuracy"]
        row["pixel_fn_rate"] = metrics["fn_rate"]
        row["pixel_fp_rate"] = metrics["fp_rate"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["fn_rate", "fp_rate", "balanced_accuracy"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_object_errors_by_image(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    group_cols=("source_image",),
    truth_available_ratio_col: str = "truth_available_ratio",
    min_truth_available_ratio: float = 0.50,
) -> pd.DataFrame:
    """
    Summarize object-level TP/TN/FP/FN by image, model, or any group columns.
    """
    group_cols = _normalise_group_cols(group_cols)

    true_col = make_true_col(target_class, "object")
    pred_col = make_predicted_col(target_class, "object")

    required = group_cols + [true_col, pred_col]
    missing = [col for col in required if col not in object_df.columns]
    if missing:
        raise KeyError(f"Missing columns in object_df: {missing}")

    rows = []
    for key, group in object_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        g = group.copy()

        if truth_available_ratio_col in g.columns:
            truth_ratio = pd.to_numeric(
                g[truth_available_ratio_col],
                errors="coerce",
            )
            g = g[
                truth_ratio >= float(min_truth_available_ratio)
            ].copy()

        if len(g) == 0:
            continue

        metrics = binary_detection_metrics(
            g,
            true_col=true_col,
            pred_col=pred_col,
            target_class=target_class,
            non_target_class=non_target_label,
        )

        row = {
            col: value
            for col, value in zip(group_cols, key)
        }
        row.update(metrics)
        row["n_truth_objects"] = metrics["n"]
        row["object_accuracy"] = metrics["accuracy"]
        row["object_balanced_accuracy"] = metrics["balanced_accuracy"]
        row["object_fn_rate"] = metrics["fn_rate"]
        row["object_fp_rate"] = metrics["fp_rate"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["fn_rate", "fp_rate", "balanced_accuracy"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
