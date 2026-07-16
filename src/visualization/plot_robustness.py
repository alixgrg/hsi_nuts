from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.common import (
    apply_project_theme,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
    validate_columns,
)
from src.visualization.plot_diagnostics import plot_metric_heatmap


def plot_ablation_deltas(
    df: pd.DataFrame,
    factor_col: str = "ablation_factor",
    metric_cols: Sequence[str] = ("fn_rate", "fp_rate", "balanced_accuracy"),
    baseline_col: str | None = None,
    baseline_value=None,
    group_cols: Sequence[str] = (),
    title: str = "Ablation effects relative to the full model",
    width: int = 1050,
    height: int = 650,
    show: bool = True,
):
    """Plot mean metric changes with standard-deviation error bars.

    When precomputed ``delta_<metric>`` columns exist they are used directly.
    Otherwise deltas are computed against rows selected by ``baseline_col`` and
    ``baseline_value`` within each group.
    """
    validate_columns(df, [factor_col])
    d = df.copy()
    group_cols = [column for column in group_cols if column in d.columns]

    delta_frames = []
    for metric in metric_cols:
        if metric not in d.columns and f"delta_{metric}" not in d.columns:
            continue
        delta_col = f"delta_{metric}"
        tmp = d.copy()
        if delta_col not in tmp.columns:
            if baseline_col is None or baseline_col not in tmp.columns:
                raise ValueError(
                    f"No {delta_col!r} column and no valid baseline_col provided."
                )
            baseline_mask = tmp[baseline_col].eq(baseline_value)
            keys = group_cols
            if keys:
                baseline = (
                    tmp.loc[baseline_mask]
                    .groupby(keys, dropna=False)[metric]
                    .mean()
                    .rename("_baseline")
                    .reset_index()
                )
                tmp = tmp.merge(baseline, on=keys, how="left")
            else:
                tmp["_baseline"] = pd.to_numeric(
                    tmp.loc[baseline_mask, metric], errors="coerce"
                ).mean()
            tmp[delta_col] = pd.to_numeric(tmp[metric], errors="coerce") - pd.to_numeric(
                tmp["_baseline"], errors="coerce"
            )
        summary = (
            tmp.groupby(factor_col, dropna=False)[delta_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["metric"] = metric
        delta_frames.append(summary)

    if not delta_frames:
        raise ValueError("No requested metric or delta columns found.")
    summary = pd.concat(delta_frames, ignore_index=True)
    color_map = make_dynamic_color_map(summary["metric"], prefer_project_colors=False)

    fig = go.Figure()
    for metric in ordered_unique(summary["metric"]):
        sub = summary[summary["metric"].eq(metric)]
        fig.add_trace(
            go.Scatter(
                x=sub["mean"],
                y=sub[factor_col].astype(str),
                mode="markers",
                name=metric,
                marker=dict(size=10, color=color_map[metric]),
                error_x=dict(type="data", array=sub["std"].fillna(0), visible=True),
                hovertemplate=(
                    f"{factor_col}: %{{y}}<br>metric: {metric}<br>"
                    "mean delta: %{x:.4f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title="Change relative to baseline",
        yaxis_title=factor_col,
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_ablation_heatmap(
    df: pd.DataFrame,
    factor_col: str = "ablation_factor",
    metric_col: str = "metric",
    value_col: str = "delta",
    facet_col: str | None = None,
    title: str = "Ablation heatmap",
    show: bool = True,
    **kwargs,
):
    """Heatmap wrapper for long-format ablation summaries."""
    return plot_metric_heatmap(
        df,
        index_col=factor_col,
        columns_col=metric_col,
        value_col=value_col,
        facet_col=facet_col,
        aggfunc="mean",
        title=title,
        colorscale="RdBu",
        show=show,
        **kwargs,
    )


def plot_stability_intervals(
    df: pd.DataFrame,
    config_col: str = "selected_config_id",
    metric_col: str = "fn_rate",
    seed_col: str = "seed",
    family_col: str | None = "matrix_family",
    sort_by_mean: bool = True,
    title: str | None = None,
    width: int = 1000,
    height: int = 650,
    show: bool = True,
):
    """Show per-seed values plus mean ± standard deviation by configuration."""
    validate_columns(df, [config_col, metric_col])
    d = df.copy()
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    summary = (
        d.groupby(config_col, dropna=False)[metric_col]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .reset_index()
    )
    if family_col and family_col in d.columns:
        family_lookup = d.drop_duplicates(config_col).set_index(config_col)[family_col]
        summary[family_col] = summary[config_col].map(family_lookup)
    if sort_by_mean:
        summary = summary.sort_values("mean", ascending=True)
    order = summary[config_col].astype(str).tolist()

    groups = (
        summary[family_col].astype(str)
        if family_col and family_col in summary.columns
        else pd.Series(["all"] * len(summary), index=summary.index)
    )
    color_map = make_dynamic_color_map(groups, prefer_project_colors=False)

    fig = go.Figure()
    # Individual seed values first.
    for group in ordered_unique(groups):
        config_ids = summary.loc[groups.eq(group), config_col].astype(str)
        mask = d[config_col].astype(str).isin(config_ids)
        fig.add_trace(
            go.Scatter(
                x=d.loc[mask, metric_col],
                y=d.loc[mask, config_col].astype(str),
                mode="markers",
                name=f"{group} seeds",
                marker=dict(size=6, opacity=0.35, color=color_map[group]),
                customdata=(
                    d.loc[mask, [seed_col]].astype(str).to_numpy()
                    if seed_col in d.columns
                    else None
                ),
                hovertemplate=(
                    f"{metric_col}: %{{x:.4f}}<br>config: %{{y}}<br>"
                    + (f"{seed_col}: %{{customdata[0]}}<br>" if seed_col in d.columns else "")
                    + "<extra></extra>"
                ),
            )
        )
    for group in ordered_unique(groups):
        sub = summary.loc[groups.eq(group)]
        fig.add_trace(
            go.Scatter(
                x=sub["mean"],
                y=sub[config_col].astype(str),
                mode="markers",
                name=f"{group} mean ± sd",
                marker=dict(size=11, color=color_map[group], line=dict(width=1, color="black")),
                error_x=dict(type="data", array=sub["std"].fillna(0), visible=True),
                hovertemplate=(
                    "config: %{y}<br>mean: %{x:.4f}<br>"
                    "sd: %{error_x.array:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title or f"Stability across random seeds — {metric_col}",
        xaxis_title=metric_col,
        yaxis_title=config_col,
        yaxis=dict(categoryorder="array", categoryarray=order),
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_border_core_metrics(
    df: pd.DataFrame,
    border_width_col: str = "border_width",
    zone_col: str = "zone",
    metric_cols: Sequence[str] = ("fn_rate", "fp_rate"),
    config_col: str | None = "selected_config_id",
    title: str = "Border versus core error rates",
    width: int = 1100,
    height: int = 650,
    show: bool = True,
):
    """Trace border/core error rates as the excluded border width changes."""
    validate_columns(df, [border_width_col, zone_col])
    metrics = [column for column in metric_cols if column in df.columns]
    if not metrics:
        raise ValueError("No requested metric columns found.")
    d = df.copy()
    configs = (
        ordered_unique(d[config_col].astype(str))
        if config_col and config_col in d.columns
        else ["all"]
    )
    metric_colors = make_dynamic_color_map(metrics, prefer_project_colors=False)

    fig = go.Figure()
    for config in configs:
        config_mask = (
            d[config_col].astype(str).eq(config)
            if config_col and config_col in d.columns
            else pd.Series(True, index=d.index)
        )
        for zone in ordered_unique(d[zone_col].astype(str)):
            zone_mask = d[zone_col].astype(str).eq(zone)
            for metric in metrics:
                sub = d[config_mask & zone_mask].sort_values(border_width_col)
                if sub.empty:
                    continue
                dash = "solid" if zone.lower() == "core" else "dash"
                name_parts = [metric, zone]
                if len(configs) > 1:
                    name_parts.append(config)
                fig.add_trace(
                    go.Scatter(
                        x=sub[border_width_col],
                        y=sub[metric],
                        mode="lines+markers",
                        name=" | ".join(name_parts),
                        line=dict(color=metric_colors[metric], dash=dash),
                        marker=dict(symbol="circle" if zone.lower() == "core" else "diamond"),
                        hovertemplate=(
                            f"width: %{{x}}<br>{metric}: %{{y:.2%}}<br>"
                            f"zone: {zone}<extra></extra>"
                        ),
                    )
                )
    fig.update_layout(
        title=title,
        xaxis_title="Border width (pixels)",
        yaxis_title="Error rate",
        width=width,
        height=height,
    )
    fig.update_yaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_truth_dilation_sensitivity(
    df: pd.DataFrame,
    radius_col: str = "dilation_radius",
    metric_cols: Sequence[str] = ("fn_rate", "fp_rate", "balanced_accuracy"),
    config_col: str | None = "selected_config_id",
    title: str = "Sensitivity to mixture truth dilation",
    width: int = 1000,
    height: int = 600,
    show: bool = True,
):
    """Plot detection metrics against the truth-map dilation radius."""
    validate_columns(df, [radius_col])
    metrics = [column for column in metric_cols if column in df.columns]
    if not metrics:
        raise ValueError("No requested metric columns found.")
    d = df.copy()
    colors = make_dynamic_color_map(metrics, prefer_project_colors=False)
    configs = (
        ordered_unique(d[config_col].astype(str))
        if config_col and config_col in d.columns
        else ["all"]
    )
    fig = go.Figure()
    for config in configs:
        mask = (
            d[config_col].astype(str).eq(config)
            if config_col and config_col in d.columns
            else pd.Series(True, index=d.index)
        )
        for metric in metrics:
            sub = d[mask].sort_values(radius_col)
            name = metric if len(configs) == 1 else f"{metric} | {config}"
            fig.add_trace(
                go.Scatter(
                    x=sub[radius_col],
                    y=sub[metric],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=colors[metric]),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Dilation radius (pixels)",
        yaxis_title="Metric",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)
