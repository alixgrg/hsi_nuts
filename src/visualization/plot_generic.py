from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.common import show_or_return, validate_columns


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
    fig = go.Figure()
    fig.add_trace(
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
    return show_or_return(fig, show)


def plot_counts_by_group(
    df: pd.DataFrame,
    group_col: str,
    category_col: str,
    title: str = "Counts",
    width: int = 850,
    height: int = 500,
    show: bool = True,
):
    validate_columns(df, [group_col, category_col])

    counts = (
        df.groupby([group_col, category_col], dropna=False)
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()

    for group in counts[group_col].astype(str).unique():
        sub = counts[counts[group_col].astype(str) == group]
        fig.add_trace(
            go.Bar(
                x=sub[category_col],
                y=sub["count"],
                name=str(group),
                hovertemplate=(
                    f"{group_col}: %{{fullData.name}}<br>"
                    f"{category_col}: %{{x}}<br>"
                    "count: %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title="Count",
        barmode="group",
        width=width,
        height=height,
    )
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

    fig = go.Figure()

    if names is None:
        names = list(y_cols)

    for col, name in zip(y_cols, names):
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="markers+lines",
                name=str(name),
            )
        )

    if hlines:
        for y, dash, text in hlines:
            fig.add_hline(
                y=y,
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
    fig = go.Figure()

    fig.add_trace(
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
            go.Scatter(
                x=curve_x,
                y=curve_y,
                mode="lines",
                name=curve_name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Density",
        width=width,
        height=height,
    )

    return show_or_return(fig, show)