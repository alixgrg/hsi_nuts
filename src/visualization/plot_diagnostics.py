from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import as_1d_array
from src.visualization.common import (
    apply_project_theme,
    class_color_map,
    make_customdata,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
)


def plot_metric_by_index(
    values,
    labels=None,
    title: str = "Metric by observation",
    y_title: str = "Metric",
    hline=None,
    object_ids=None,
    source_images=None,
    width: int = 900,
    height: int = 500,
    show: bool = True,
    category_order: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    **metadata,
):
    values = np.asarray(values, dtype=float)
    n = len(values)
    labels = as_1d_array(labels, n, "all").astype(str)

    meta = dict(object_id=object_ids, source_image=source_images)
    meta.update(metadata)
    custom, hover_meta = make_customdata(n, **meta)
    x = np.arange(n)

    groups = (
        [str(value) for value in category_order]
        if category_order is not None
        else ordered_unique(labels)
    )
    if color_map is None:
        color_map = make_dynamic_color_map(groups)

    fig = go.Figure()
    for label in groups:
        mask = labels == label
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=values[mask],
                mode="markers",
                name=str(label),
                customdata=custom[mask],
                marker=dict(size=8, opacity=0.8, color=color_map.get(label)),
                hovertemplate=(
                    "index: %{x}<br>value: %{y:.4f}<br>"
                    + hover_meta
                    + "<extra></extra>"
                ),
            )
        )

    if hline is not None:
        fig.add_hline(y=float(hline), line_dash="dash")

    fig.update_layout(
        title=title,
        xaxis_title="Observation index",
        yaxis_title=y_title,
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_xy_diagnostic(
    x,
    y,
    labels=None,
    title: str = "Diagnostic plot",
    x_title: str = "x",
    y_title: str = "y",
    vline: float | None = None,
    hline: float | None = None,
    line_traces: Sequence[go.Scatter] | None = None,
    object_ids=None,
    source_images=None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    category_order: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    force_legend_groups: bool = False,
    legend_title: str = "class / decision",
    **metadata,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    labels = as_1d_array(labels, n, "all").astype(str)
    meta = dict(object_id=object_ids, source_image=source_images)
    meta.update(metadata)
    custom, hover_meta = make_customdata(n, **meta)

    label_groups = (
        [str(value) for value in category_order]
        if category_order is not None
        else ordered_unique(labels)
    )
    if color_map is None:
        color_map = make_dynamic_color_map(label_groups)

    fig = go.Figure()
    for label in label_groups:
        mask = labels == label
        if not mask.any() and not force_legend_groups:
            continue

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=str(label),
                customdata=custom[mask],
                showlegend=True,
                marker=dict(
                    size=9,
                    opacity=0.8,
                    color=color_map.get(str(label), "lightgray"),
                    line=dict(width=0.7, color="black"),
                ),
                hovertemplate=(
                    f"{x_title}: %{{x:.4f}}<br>"
                    f"{y_title}: %{{y:.4f}}<br>"
                    + hover_meta
                    + "<extra></extra>"
                ),
            )
        )

    for trace in line_traces or ():
        fig.add_trace(trace)
    if vline is not None:
        fig.add_vline(x=float(vline), line_dash="dash")
    if hline is not None:
        fig.add_hline(y=float(hline), line_dash="dash")

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=True,
        legend_title_text=legend_title,
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def _ordered_pivot(
    df: pd.DataFrame,
    *,
    index_col: str,
    columns_col: str,
    value_col: str,
    aggfunc: str,
    row_order: Sequence | None,
    column_order: Sequence | None,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=index_col,
        columns=columns_col,
        values=value_col,
        aggfunc=aggfunc,
    )
    if row_order is not None:
        pivot = pivot.reindex([value for value in row_order if value in pivot.index])
    if column_order is not None:
        pivot = pivot.reindex(columns=[value for value in column_order if value in pivot.columns])
    return pivot


def _heatmap_text(values: np.ndarray, annotation_format: str) -> np.ndarray:
    def format_one(value):
        if not np.isfinite(value):
            return ""
        return format(float(value), annotation_format)

    return np.vectorize(format_one, otypes=[object])(values)


def plot_metric_heatmap(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    value_col: str,
    facet_col: str | None = None,
    aggfunc: str = "max",
    title: str | None = None,
    colorbar_title: str | None = None,
    width: int = 950,
    height: int = 650,
    show: bool = True,
    colorscale: str = "Viridis",
    zmin=None,
    zmax=None,
    annotation_format: str = ".3f",
    row_order=None,
    column_order=None,
    facet_col_wrap: int = 3,
    shared_coloraxis: bool = True,
):
    """Plot one or several compact metric heatmaps.

    Facets are wrapped over several rows instead of multiplying the figure width
    by the number of groups.
    """
    required = [index_col, columns_col, value_col]
    if facet_col is not None:
        required.append(facet_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for heatmap: {missing}")

    d = df.copy()
    if facet_col is None:
        pivot = _ordered_pivot(
            d,
            index_col=index_col,
            columns_col=columns_col,
            value_col=value_col,
            aggfunc=aggfunc,
            row_order=row_order,
            column_order=column_order,
        )
        values = pivot.to_numpy(dtype=float)
        fig = go.Figure(
            go.Heatmap(
                z=values,
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=colorbar_title or value_col),
                text=_heatmap_text(values, annotation_format),
                texttemplate="%{text}",
                hovertemplate=(
                    f"{index_col}: %{{y}}<br>"
                    f"{columns_col}: %{{x}}<br>"
                    f"{value_col}: %{{z:.4f}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title or f"{value_col} heatmap",
            xaxis_title=columns_col,
            yaxis_title=index_col,
            width=width,
            height=height,
        )
        apply_project_theme(fig)
        return show_or_return(fig, show)

    facets = ordered_unique(d[facet_col].astype(str))
    n_cols = max(1, min(int(facet_col_wrap), len(facets)))
    n_rows = ceil(len(facets) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=facets,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for index, facet_value in enumerate(facets):
        row = index // n_cols + 1
        col = index % n_cols + 1
        sub = d[d[facet_col].astype(str).eq(facet_value)]
        pivot = _ordered_pivot(
            sub,
            index_col=index_col,
            columns_col=columns_col,
            value_col=value_col,
            aggfunc=aggfunc,
            row_order=row_order,
            column_order=column_order,
        )
        values = pivot.to_numpy(dtype=float)
        trace_kwargs = dict(
            z=values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            text=_heatmap_text(values, annotation_format),
            texttemplate="%{text}",
            hovertemplate=(
                f"{facet_col}: {facet_value}<br>"
                f"{index_col}: %{{y}}<br>"
                f"{columns_col}: %{{x}}<br>"
                f"{value_col}: %{{z:.4f}}<extra></extra>"
            ),
        )
        if shared_coloraxis:
            trace_kwargs["coloraxis"] = "coloraxis"
        else:
            trace_kwargs.update(
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=index == len(facets) - 1,
                colorbar=dict(title=colorbar_title or value_col),
            )
        fig.add_trace(go.Heatmap(**trace_kwargs), row=row, col=col)
        fig.update_xaxes(title_text=columns_col, row=row, col=col)
        fig.update_yaxes(title_text=index_col, row=row, col=col)

    layout = dict(
        title=title or f"{value_col} heatmap by {facet_col}",
        width=width,
        height=max(height, 360 * n_rows),
    )
    if shared_coloraxis:
        layout["coloraxis"] = dict(
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=colorbar_title or value_col),
        )
    fig.update_layout(**layout)
    apply_project_theme(fig)
    return show_or_return(fig, show)
