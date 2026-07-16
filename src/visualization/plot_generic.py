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


def plot_bar_values(
    x,
    y,
    title: str = "Bar plot",
    x_title: str = "x",
    y_title: str = "y",
    width: int = 1000,
    height: int = 500,
    show: bool = True,
):
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_counts_by_group(
    df: pd.DataFrame,
    group_col: str,
    category_col: str,
    title: str = "Counts",
    width: int = 850,
    height: int = 500,
    show: bool = True,
    group_order: Sequence[str] | None = None,
    category_order: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    normalize: bool = False,
):
    """Grouped counts or within-category rates with stable class colours."""
    validate_columns(df, [group_col, category_col])
    counts = (
        df.groupby([group_col, category_col], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if normalize:
        totals = counts.groupby(category_col)["count"].transform("sum").clip(lower=1)
        counts["value"] = counts["count"] / totals
        y_col = "value"
    else:
        y_col = "count"

    groups = (
        [str(value) for value in group_order]
        if group_order is not None
        else ordered_unique(counts[group_col].astype(str))
    )
    categories = (
        [str(value) for value in category_order]
        if category_order is not None
        else ordered_unique(counts[category_col].astype(str))
    )
    if color_map is None:
        color_map = make_dynamic_color_map(groups)

    fig = go.Figure()
    for group in groups:
        sub = counts[counts[group_col].astype(str).eq(group)].copy()
        sub[category_col] = sub[category_col].astype(str)
        sub = sub.set_index(category_col).reindex(categories).reset_index()
        sub["count"] = sub["count"].fillna(0)
        if normalize:
            sub["value"] = sub["value"].fillna(0)
        fig.add_trace(
            go.Bar(
                x=sub[category_col],
                y=sub[y_col],
                name=group,
                marker_color=color_map.get(group),
                customdata=sub[["count"]].to_numpy(),
                hovertemplate=(
                    f"{group_col}: %{{fullData.name}}<br>"
                    f"{category_col}: %{{x}}<br>"
                    + ("rate: %{y:.1%}<br>" if normalize else "count: %{y}<br>")
                    + "n: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title="Rate" if normalize else "Count",
        barmode="group",
        width=width,
        height=height,
    )
    if normalize:
        fig.update_yaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_lines_from_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
    names: Sequence[str] | None = None,
    title: str = "Metrics",
    x_title: str | None = None,
    y_title: str = "Value",
    hlines: Sequence[tuple[float, str, str]] | None = None,
    percent_y: bool = False,
    width: int = 900,
    height: int = 550,
    show: bool = True,
):
    validate_columns(df, [x_col])
    names = list(y_cols) if names is None else list(names)

    fig = go.Figure()
    for column, name in zip(y_cols, names):
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=pd.to_numeric(df[column], errors="coerce"),
                mode="markers+lines",
                name=str(name),
            )
        )

    for y, dash, text in hlines or ():
        fig.add_hline(
            y=float(y),
            line_dash=dash,
            annotation_text=text,
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title or x_col,
        yaxis_title=y_title,
        width=width,
        height=height,
    )
    if percent_y:
        fig.update_yaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_distribution_with_curve(
    values,
    curve_x=None,
    curve_y=None,
    nbins: int = 30,
    title: str = "Distribution",
    x_title: str = "Value",
    curve_name: str = "theoretical",
    width: int = 850,
    height: int = 500,
    show: bool = True,
):
    fig = go.Figure(
        go.Histogram(
            x=np.asarray(values, dtype=float),
            histnorm="probability density",
            nbinsx=nbins,
            name="empirical",
            opacity=0.65,
        )
    )
    if curve_x is not None and curve_y is not None:
        fig.add_trace(
            go.Scatter(x=curve_x, y=curve_y, mode="lines", name=curve_name)
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Density",
        width=width,
        height=height,
        barmode="overlay",
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)
