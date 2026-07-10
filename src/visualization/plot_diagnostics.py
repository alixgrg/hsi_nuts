from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import as_1d_array
from src.visualization.common import make_customdata, show_or_return


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
    **metadata,
):
    values = np.asarray(values, dtype=float)
    n = len(values)

    labels = as_1d_array(labels, n, "all").astype(str)

    meta = dict(
        object_id=object_ids,
        source_image=source_images,
    )
    meta.update(metadata)

    custom, hover_meta = make_customdata(n, **meta)
    x = np.arange(n)

    fig = go.Figure()

    for lab in np.unique(labels):
        mask = labels == lab

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=values[mask],
                mode="markers",
                name=str(lab),
                customdata=custom[mask],
                marker=dict(size=8, opacity=0.8),
                hovertemplate=(
                    "index: %{x}<br>"
                    "value: %{y:.4f}<br>"
                    + hover_meta
                    + "<extra></extra>"
                ),
            )
        )

    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash")

    fig.update_layout(
        title=title,
        xaxis_title="Observation index",
        yaxis_title=y_title,
        width=width,
        height=height,
    )

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
    **metadata,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    labels = as_1d_array(labels, n, "all").astype(str)

    meta = dict(
        object_id=object_ids,
        source_image=source_images,
    )
    meta.update(metadata)

    custom, hover_meta = make_customdata(n, **meta)

    fig = go.Figure()

    if category_order is None:
        label_groups = list(dict.fromkeys(labels.astype(str)))
    else:
        label_groups = [str(x) for x in category_order]

    for lab in label_groups:
        mask = labels.astype(str) == str(lab)

        if not mask.any() and not force_legend_groups:
            continue

        marker_kwargs = dict(size=9, opacity=0.8)

        if color_map is not None and str(lab) in color_map:
            marker_kwargs["color"] = color_map[str(lab)]

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=str(lab),
                customdata=custom[mask],
                showlegend=True,
                marker=marker_kwargs,
                hovertemplate=(
                    f"{x_title}: %{{x:.4f}}<br>"
                    f"{y_title}: %{{y:.4f}}<br>"
                    + hover_meta
                    + "<extra></extra>"
                ),
            )
        )

    if line_traces:
        for tr in line_traces:
            fig.add_trace(tr)

    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash")

    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash")

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=True,
        legend_title_text="class / decision",
        width=width,
        height=height,
    )

    return show_or_return(fig, show)


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
):
    """
    Plot a metric heatmap, for example:
    preprocessing x SIMCA rule using balanced_accuracy.
    """

    required = [index_col, columns_col, value_col]
    if facet_col is not None:
        required.append(facet_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for heatmap: {missing}")

    d = df.copy()

    if facet_col is None:
        pivot = d.pivot_table(
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc=aggfunc,
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.to_numpy(dtype=float),
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale="Viridis",
                colorbar=dict(title=colorbar_title or value_col),
                text=np.round(pivot.to_numpy(dtype=float), 3),
                texttemplate="%{text}",
                hovertemplate=(
                    f"{index_col}: %{{y}}<br>"
                    f"{columns_col}: %{{x}}<br>"
                    f"{value_col}: %{{z:.4f}}"
                    "<extra></extra>"
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

        return show_or_return(fig, show)

    facets = list(d[facet_col].astype(str).drop_duplicates())
    fig = make_subplots(
        rows=1,
        cols=len(facets),
        subplot_titles=facets,
        shared_yaxes=True,
    )

    for j, facet_value in enumerate(facets, start=1):
        sub = d[d[facet_col].astype(str).eq(facet_value)]

        pivot = sub.pivot_table(
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc=aggfunc,
        )

        fig.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(dtype=float),
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale="Viridis",
                colorbar=dict(title=colorbar_title or value_col),
                text=np.round(pivot.to_numpy(dtype=float), 3),
                texttemplate="%{text}",
                hovertemplate=(
                    f"{facet_col}: {facet_value}<br>"
                    f"{index_col}: %{{y}}<br>"
                    f"{columns_col}: %{{x}}<br>"
                    f"{value_col}: %{{z:.4f}}"
                    "<extra></extra>"
                ),
                showscale=(j == len(facets)),
            ),
            row=1,
            col=j,
        )

    fig.update_layout(
        title=title or f"{value_col} heatmap by {facet_col}",
        width=width * max(1, len(facets)),
        height=height,
    )

    return show_or_return(fig, show)