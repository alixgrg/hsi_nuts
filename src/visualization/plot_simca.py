from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.visualization.common import show_or_return
from src.visualization.plot_diagnostics import (
    plot_metric_by_index,
    plot_xy_diagnostic,
)
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
):
    """Plot SIMCA H/Q distance plot for one class."""
    res = simca_results[class_name]

    if normalized:
        x = np.asarray(res["H_norm_limit"])
        y = np.asarray(res["Q_norm_limit"])
        x_title, y_title = "H / H_limit", "Q / Q_limit"
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
    """Plot SIMCA rule statistic by observation."""
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
    """Counts of SIMCA decisions grouped by true label."""
    return plot_counts_by_group(
        results_df,
        group_col=true_label_col,
        category_col=decision_col,
        title=title,
        show=show,
    )


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
    force_legend_groups: bool = False,
):
    """
    Plot SIMCA Q residuals vs Hotelling T² from a pixel/object dataframe.

    For pixels:
        default x = H_norm_limit
        default y = Q_norm_limit

    For objects:
        default x = H_norm_limit_mean
        default y = Q_norm_limit_mean
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

    required = [x_col, y_col]
    missing = [col for col in required if col not in d.columns]

    if missing:
        raise KeyError(f"Missing SIMCA diagnostic columns: {missing}")

    labels = d[label_col].astype(str) if label_col in d.columns else None

    metadata = {}

    if confidence_col is not None and confidence_col in d.columns:
        metadata["confidence"] = d[confidence_col].round(3).astype(str)

    if "object_error_case" in d.columns:
        metadata["object_error_case"] = d["object_error_case"].astype(str)

    if "pixel_error_case" in d.columns:
        metadata["pixel_error_case"] = d["pixel_error_case"].astype(str)

    fig = plot_xy_diagnostic(
        x=d[x_col].to_numpy(dtype=float),
        y=d[y_col].to_numpy(dtype=float),
        labels=labels,
        object_ids=d[object_id_col] if object_id_col in d.columns else None,
        source_images=d[source_col] if source_col in d.columns else None,
        title=title or f"{level.capitalize()} SIMCA: Q residuals vs Hotelling T²",
        x_title="Hotelling T² / limit" if "norm" in x_col else "Hotelling T²",
        y_title="Q residual / limit" if "norm" in y_col else "Q residual",
        vline=1.0 if "norm" in x_col else None,
        hline=1.0 if "norm" in y_col else None,
        width=width,
        height=height,
        show=False,
        category_order=category_order,
        color_map=color_map,
        force_legend_groups=force_legend_groups,
        **metadata,
    )

    return show_or_return(fig, show)