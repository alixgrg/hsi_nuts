from __future__ import annotations

import numpy as np

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