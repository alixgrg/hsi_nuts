from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.common import (
    apply_project_theme,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
    validate_columns,
)


def _pareto_mask(
    df: pd.DataFrame,
    minimize_cols: Sequence[str],
    maximize_cols: Sequence[str] = (),
) -> np.ndarray:
    values_min = df[list(minimize_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values_max = (
        df[list(maximize_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if maximize_cols
        else np.empty((len(df), 0))
    )
    valid = np.isfinite(values_min).all(axis=1)
    if values_max.shape[1]:
        valid &= np.isfinite(values_max).all(axis=1)
    out = np.zeros(len(df), dtype=bool)
    indices = np.flatnonzero(valid)
    for i in indices:
        no_worse_min = np.all(values_min[indices] <= values_min[i], axis=1)
        strictly_better_min = np.any(values_min[indices] < values_min[i], axis=1)
        if values_max.shape[1]:
            no_worse_max = np.all(values_max[indices] >= values_max[i], axis=1)
            strictly_better_max = np.any(values_max[indices] > values_max[i], axis=1)
        else:
            no_worse_max = np.ones(len(indices), dtype=bool)
            strictly_better_max = np.zeros(len(indices), dtype=bool)
        dominated = np.any(
            no_worse_min
            & no_worse_max
            & (strictly_better_min | strictly_better_max)
        )
        out[i] = not dominated
    return out


def plot_detection_pareto(
    df: pd.DataFrame,
    fn_col: str = "fn_rate",
    fp_col: str = "fp_rate",
    color_col: str = "balanced_accuracy",
    symbol_col: str | None = "matrix_family",
    id_col: str | None = "selected_config_id",
    selected_col: str | None = None,
    group_col: str | None = None,
    annotate_selected: bool = True,
    title: str = "SIMCA detection trade-off",
    width: int = 900,
    height: int = 650,
    show: bool = True,
):
    """Plot false-positive versus false-negative rates and the Pareto front."""
    validate_columns(df, [fn_col, fp_col])
    d = df.copy().reset_index(drop=True)
    x = pd.to_numeric(d[fp_col], errors="coerce")
    y = pd.to_numeric(d[fn_col], errors="coerce")
    color = pd.to_numeric(d[color_col], errors="coerce") if color_col in d.columns else None

    symbol_values = (
        d[symbol_col].astype(str)
        if symbol_col is not None and symbol_col in d.columns
        else pd.Series(["all"] * len(d))
    )
    symbols = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x"]
    symbol_map = {
        value: symbols[index % len(symbols)]
        for index, value in enumerate(ordered_unique(symbol_values))
    }

    fig = go.Figure()
    for symbol_value in ordered_unique(symbol_values):
        mask = symbol_values.eq(symbol_value)
        custom_cols = [column for column in (id_col, group_col, color_col) if column and column in d.columns]
        custom = d.loc[mask, custom_cols].astype(str).to_numpy() if custom_cols else None
        hover = "".join(
            f"{column}: %{{customdata[{index}]}}<br>"
            for index, column in enumerate(custom_cols)
        )
        marker = dict(
            size=11,
            symbol=symbol_map[symbol_value],
            opacity=0.85,
            line=dict(width=0.8, color="black"),
        )
        if color is not None:
            marker.update(
                color=color[mask],
                colorscale="Viridis",
                cmin=float(np.nanmin(color)),
                cmax=float(np.nanmax(color)),
                colorbar=dict(title=color_col),
                showscale=symbol_value == ordered_unique(symbol_values)[-1],
            )
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=symbol_value,
                marker=marker,
                customdata=custom,
                hovertemplate=(
                    f"{fp_col}: %{{x:.2%}}<br>{fn_col}: %{{y:.2%}}<br>"
                    + hover
                    + "<extra></extra>"
                ),
            )
        )

    if group_col is not None and group_col in d.columns:
        fronts = []
        for _, group in d.groupby(group_col, dropna=False):
            mask = _pareto_mask(group, minimize_cols=[fp_col, fn_col])
            fronts.append(group.loc[mask])
        front = pd.concat(fronts, ignore_index=True) if fronts else pd.DataFrame()
    else:
        front = d.loc[_pareto_mask(d, minimize_cols=[fp_col, fn_col])]
    if not front.empty:
        front = front.sort_values(fp_col)
        fig.add_trace(
            go.Scatter(
                x=front[fp_col],
                y=front[fn_col],
                mode="lines+markers",
                name="Pareto front",
                line=dict(color="black", dash="dash"),
                marker=dict(color="black", size=6),
                hoverinfo="skip",
            )
        )

    if selected_col is not None and selected_col in d.columns:
        selected = d[selected_col].fillna(False).astype(bool)
        if selected.any():
            fig.add_trace(
                go.Scatter(
                    x=x[selected],
                    y=y[selected],
                    mode="markers+text" if annotate_selected and id_col in d.columns else "markers",
                    text=d.loc[selected, id_col].astype(str) if annotate_selected and id_col in d.columns else None,
                    textposition="top center",
                    name="selected",
                    marker=dict(
                        symbol="circle-open",
                        size=20,
                        color="black",
                        line=dict(width=3, color="black"),
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="False-positive rate",
        yaxis_title="False-negative rate",
        width=width,
        height=height,
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_three_way_tradeoff(
    df: pd.DataFrame,
    uncertain_col: str = "uncertain_rate",
    miss_col: str = "target_miss_rate",
    false_accept_col: str = "non_target_false_accept_rate",
    coverage_col: str = "coverage_rate",
    group_col: str | None = "matrix_family",
    id_col: str | None = "selected_config_id",
    selected_col: str | None = None,
    title: str = "Three-way decision trade-off",
    width: int = 900,
    height: int = 650,
    show: bool = True,
):
    """Plot uncertainty against target misses, coloured by false accepts."""
    validate_columns(df, [uncertain_col, miss_col])
    d = df.copy()
    groups = (
        d[group_col].astype(str)
        if group_col is not None and group_col in d.columns
        else pd.Series(["all"] * len(d))
    )
    symbols = ["circle", "square", "diamond", "triangle-up", "triangle-down"]
    symbol_map = {
        value: symbols[index % len(symbols)]
        for index, value in enumerate(ordered_unique(groups))
    }
    color = pd.to_numeric(d[false_accept_col], errors="coerce") if false_accept_col in d.columns else None
    if coverage_col in d.columns:
        coverage = pd.to_numeric(d[coverage_col], errors="coerce").clip(0, 1)
        sizes = 8 + 20 * coverage.fillna(0.5)
    else:
        sizes = pd.Series([12] * len(d), index=d.index)

    fig = go.Figure()
    for group in ordered_unique(groups):
        mask = groups.eq(group)
        custom_cols = [column for column in (id_col, false_accept_col, coverage_col) if column and column in d.columns]
        custom = d.loc[mask, custom_cols].astype(str).to_numpy() if custom_cols else None
        hover = "".join(
            f"{column}: %{{customdata[{index}]}}<br>"
            for index, column in enumerate(custom_cols)
        )
        marker = dict(
            size=sizes[mask],
            symbol=symbol_map[group],
            opacity=0.82,
            line=dict(width=0.8, color="black"),
        )
        if color is not None:
            marker.update(
                color=color[mask],
                colorscale="Plasma",
                cmin=float(np.nanmin(color)),
                cmax=float(np.nanmax(color)),
                colorbar=dict(title=false_accept_col),
                showscale=group == ordered_unique(groups)[-1],
            )
        fig.add_trace(
            go.Scatter(
                x=d.loc[mask, uncertain_col],
                y=d.loc[mask, miss_col],
                mode="markers",
                name=group,
                marker=marker,
                customdata=custom,
                hovertemplate=(
                    f"{uncertain_col}: %{{x:.2%}}<br>{miss_col}: %{{y:.2%}}<br>"
                    + hover
                    + "<extra></extra>"
                ),
            )
        )

    if selected_col is not None and selected_col in d.columns:
        selected = d[selected_col].fillna(False).astype(bool)
        if selected.any():
            fig.add_trace(
                go.Scatter(
                    x=d.loc[selected, uncertain_col],
                    y=d.loc[selected, miss_col],
                    mode="markers",
                    name="selected",
                    marker=dict(symbol="circle-open", size=22, color="black", line=dict(width=3)),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Uncertain rate",
        yaxis_title="Target miss rate",
        width=width,
        height=height,
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_parameter_tendencies(
    df: pd.DataFrame,
    parameter_col: str = "parameter",
    value_col: str = "value",
    rate_col: str = "top_rate",
    family_col: str | None = "matrix_family",
    top_n: int | None = 30,
    title: str = "Parameter tendencies among top models",
    width: int = 1000,
    height: int = 700,
    show: bool = True,
):
    """Horizontal bar chart of parameter-value prevalence in selected models."""
    validate_columns(df, [parameter_col, value_col, rate_col])
    d = df.copy()
    d["parameter_value"] = d[parameter_col].astype(str) + " = " + d[value_col].astype(str)
    if top_n is not None:
        d = d.sort_values(rate_col, ascending=False).head(int(top_n))
    d = d.sort_values(rate_col, ascending=True)

    groups = (
        d[family_col].astype(str)
        if family_col is not None and family_col in d.columns
        else pd.Series(["all"] * len(d), index=d.index)
    )
    color_map = make_dynamic_color_map(groups, prefer_project_colors=False)
    fig = go.Figure()
    for group in ordered_unique(groups):
        mask = groups.eq(group)
        fig.add_trace(
            go.Bar(
                x=d.loc[mask, rate_col],
                y=d.loc[mask, "parameter_value"],
                orientation="h",
                name=group,
                marker_color=color_map[group],
                hovertemplate="%{y}<br>rate: %{x:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Presence among top models",
        yaxis_title="Parameter value",
        barmode="group",
        width=width,
        height=height,
    )
    fig.update_xaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_validation_test_shift(
    df: pd.DataFrame,
    validation_metric: str,
    test_metric: str,
    color_col: str | None = "matrix_family",
    id_col: str | None = "selected_config_id",
    annotate_top_n: int = 0,
    title: str | None = None,
    width: int = 800,
    height: int = 650,
    show: bool = True,
):
    """Compare validation and independent-test metrics against the identity line."""
    validate_columns(df, [validation_metric, test_metric])
    d = df.copy()
    groups = (
        d[color_col].astype(str)
        if color_col is not None and color_col in d.columns
        else pd.Series(["all"] * len(d), index=d.index)
    )
    color_map = make_dynamic_color_map(groups, prefer_project_colors=False)
    fig = go.Figure()
    for group in ordered_unique(groups):
        mask = groups.eq(group)
        custom = d.loc[mask, [id_col]].astype(str).to_numpy() if id_col and id_col in d.columns else None
        fig.add_trace(
            go.Scatter(
                x=d.loc[mask, validation_metric],
                y=d.loc[mask, test_metric],
                mode="markers",
                name=group,
                marker=dict(size=10, color=color_map[group], line=dict(width=0.8, color="black")),
                customdata=custom,
                hovertemplate=(
                    f"validation: %{{x:.4f}}<br>test: %{{y:.4f}}<br>"
                    + (f"{id_col}: %{{customdata[0]}}<br>" if custom is not None else "")
                    + "<extra></extra>"
                ),
            )
        )
    all_values = pd.concat(
        [pd.to_numeric(d[validation_metric], errors="coerce"), pd.to_numeric(d[test_metric], errors="coerce")]
    ).dropna()
    if not all_values.empty:
        lo, hi = float(all_values.min()), float(all_values.max())
        pad = 0.04 * max(hi - lo, 1e-9)
        fig.add_trace(
            go.Scatter(
                x=[lo - pad, hi + pad],
                y=[lo - pad, hi + pad],
                mode="lines",
                name="identity",
                line=dict(color="black", dash="dash"),
                hoverinfo="skip",
            )
        )
    if annotate_top_n > 0 and id_col in d.columns:
        shift = (
            pd.to_numeric(d[test_metric], errors="coerce")
            - pd.to_numeric(d[validation_metric], errors="coerce")
        ).abs()
        for index in shift.nlargest(int(annotate_top_n)).index:
            fig.add_annotation(
                x=d.loc[index, validation_metric],
                y=d.loc[index, test_metric],
                text=str(d.loc[index, id_col]),
                showarrow=True,
                ax=20,
                ay=-20,
            )
    fig.update_layout(
        title=title or f"Independent-test shift: {test_metric} vs {validation_metric}",
        xaxis_title=f"Validation — {validation_metric}",
        yaxis_title=f"Test — {test_metric}",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_model_metric_ranking(
    df: pd.DataFrame,
    metric_col: str,
    id_col: str = "selected_config_id",
    family_col: str | None = "matrix_family",
    ascending: bool = False,
    top_n: int = 20,
    title: str | None = None,
    width: int = 1000,
    height: int = 650,
    show: bool = True,
):
    validate_columns(df, [metric_col, id_col])
    d = df.sort_values(metric_col, ascending=ascending).head(int(top_n)).iloc[::-1]
    groups = (
        d[family_col].astype(str)
        if family_col and family_col in d.columns
        else pd.Series(["all"] * len(d), index=d.index)
    )
    color_map = make_dynamic_color_map(groups, prefer_project_colors=False)
    fig = go.Figure()
    for group in ordered_unique(groups):
        mask = groups.eq(group)
        fig.add_trace(
            go.Bar(
                x=d.loc[mask, metric_col],
                y=d.loc[mask, id_col].astype(str),
                orientation="h",
                name=group,
                marker_color=color_map[group],
                hovertemplate=f"%{{y}}<br>{metric_col}: %{{x:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title or f"Model ranking by {metric_col}",
        xaxis_title=metric_col,
        yaxis_title=id_col,
        barmode="group",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)
