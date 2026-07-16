from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.visualization.common import (
    BINARY_CLASS_ORDER,
    THREE_WAY_CLASS_ORDER,
    class_color_map,
    show_or_return,
)
from src.visualization.plot_diagnostics import plot_metric_by_index, plot_xy_diagnostic
from src.visualization.plot_generic import plot_counts_by_group


def plot_simca_distance(
    simca_results,
    class_name,
    labels=None,
    object_ids=None,
    source_images=None,
    normalized: bool = True,
    title=None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    category_order: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
):
    """Plot SIMCA H/Q distances for one class model."""
    res = simca_results[class_name]
    if normalized:
        x = np.asarray(res["H_norm_limit"])
        y = np.asarray(res["Q_norm_limit"])
        x_title, y_title = "H / H limit", "Q / Q limit"
        vline, hline = 1.0, 1.0
    else:
        x = np.asarray(res["H"])
        y = np.asarray(res["Q"])
        x_title, y_title = "H", "Q"
        vline, hline = res["H_limit"], res["Q_limit"]

    accepted = np.asarray(res.get("accepted", [""] * len(x))).astype(str)
    rule_stat = np.asarray(res.get("rule_statistic", [""] * len(x))).astype(str)
    return plot_xy_diagnostic(
        x,
        y,
        labels=labels,
        object_ids=object_ids,
        source_images=source_images,
        accepted=accepted,
        rule_statistic=rule_stat,
        title=title or f"SIMCA distance — class={class_name}",
        x_title=x_title,
        y_title=y_title,
        vline=vline,
        hline=hline,
        width=width,
        height=height,
        category_order=category_order,
        color_map=color_map,
        show=show,
    )


def plot_simca_rule_metric(
    simca_results,
    class_name,
    labels=None,
    object_ids=None,
    source_images=None,
    title=None,
    show: bool = True,
):
    """Plot the SIMCA rule statistic by observation."""
    res = simca_results[class_name]
    return plot_metric_by_index(
        res["rule_statistic"],
        labels=labels,
        object_ids=object_ids,
        source_images=source_images,
        hline=res.get("rule_limit"),
        title=title or f"SIMCA rule statistic — class={class_name}",
        y_title="Rule statistic",
        show=show,
    )


def plot_decision_counts(
    results_df,
    true_label_col: str = "true_label",
    decision_col: str = "simca_case",
    title: str = "SIMCA prediction counts",
    show: bool = True,
):
    """Deprecated count plot; confusion heatmaps are generally more informative."""
    warnings.warn(
        "plot_decision_counts is kept for compatibility. Prefer a confusion heatmap.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_counts_by_group(
        results_df,
        group_col=true_label_col,
        category_col=decision_col,
        title=title,
        show=show,
    )


def _default_decision_style(label_col: str, labels: pd.Series | None):
    lower = str(label_col).lower()
    observed = set() if labels is None else set(labels.astype(str).str.lower())
    is_three_way = (
        "3way" in lower
        or "three_way" in lower
        or "uncertain" in observed
        or "ambiguous" in observed
    )
    if is_three_way:
        order = list(THREE_WAY_CLASS_ORDER)
    else:
        order = list(BINARY_CLASS_ORDER)
    return order, class_color_map(order)


def plot_simca_q_t2_dataframe(
    df: pd.DataFrame,
    level: str = "pixel",
    x_col: str | None = None,
    y_col: str | None = None,
    label_col: str = "decision_3way",
    confidence_col: str | None = None,
    object_id_col: str = "object_id",
    source_col: str = "source_image",
    title: str | None = None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    category_order: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    force_legend_groups: bool = True,
):
    """Plot SIMCA Q residuals versus H/T²-like distances from a dataframe.

    At object level, columns ending in ``_mean`` are means of pixel diagnostics;
    the default title and axis labels say so explicitly to avoid interpreting them
    as true object-level Hotelling T² values.
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe for SIMCA Q/T² plot.")
    d = df.copy()

    if x_col is None:
        if level == "object" and "H_norm_limit_mean" in d.columns:
            x_col = "H_norm_limit_mean"
        elif "H_norm_limit" in d.columns:
            x_col = "H_norm_limit"
        elif "H_mean" in d.columns:
            x_col = "H_mean"
        else:
            x_col = "H"
    if y_col is None:
        if level == "object" and "Q_norm_limit_mean" in d.columns:
            y_col = "Q_norm_limit_mean"
        elif "Q_norm_limit" in d.columns:
            y_col = "Q_norm_limit"
        elif "Q_mean" in d.columns:
            y_col = "Q_mean"
        else:
            y_col = "Q"

    missing = [column for column in (x_col, y_col) if column not in d.columns]
    if missing:
        raise KeyError(f"Missing SIMCA diagnostic columns: {missing}")

    labels = d[label_col].astype(str) if label_col in d.columns else None
    if category_order is None or color_map is None:
        default_order, default_map = _default_decision_style(label_col, labels)
        category_order = default_order if category_order is None else category_order
        color_map = default_map if color_map is None else color_map

    metadata = {}
    if confidence_col is not None and confidence_col in d.columns:
        metadata["confidence"] = pd.to_numeric(
            d[confidence_col], errors="coerce"
        ).round(3)
    if "object_error_case" in d.columns:
        metadata["object_error_case"] = d["object_error_case"].astype(str)
    if "pixel_error_case" in d.columns:
        metadata["pixel_error_case"] = d["pixel_error_case"].astype(str)

    is_mean_object = level == "object" and (
        str(x_col).endswith("_mean") or str(y_col).endswith("_mean")
    )
    if is_mean_object:
        default_title = "Object mean normalized pixel H versus mean normalized pixel Q"
        x_title = "Mean pixel H / H limit" if "norm" in x_col else "Mean pixel H"
        y_title = "Mean pixel Q / Q limit" if "norm" in y_col else "Mean pixel Q"
    else:
        default_title = f"{level.capitalize()} SIMCA: Q residuals vs H/T²"
        x_title = "H / limit" if "norm" in x_col else "H / T²-like distance"
        y_title = "Q residual / limit" if "norm" in y_col else "Q residual"

    fig = plot_xy_diagnostic(
        x=d[x_col].to_numpy(dtype=float),
        y=d[y_col].to_numpy(dtype=float),
        labels=labels,
        object_ids=d[object_id_col] if object_id_col in d.columns else None,
        source_images=d[source_col] if source_col in d.columns else None,
        title=title or default_title,
        x_title=x_title,
        y_title=y_title,
        vline=1.0 if "norm" in x_col and not is_mean_object else None,
        hline=1.0 if "norm" in y_col and not is_mean_object else None,
        width=width,
        height=height,
        show=False,
        category_order=category_order,
        color_map=color_map,
        force_legend_groups=force_legend_groups,
        **metadata,
    )
    return show_or_return(fig, show)
