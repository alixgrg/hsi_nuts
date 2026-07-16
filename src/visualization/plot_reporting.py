from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.decision.maps import make_object_error_map, make_pixel_error_map
from src.visualization.common import (
    ERROR_COLOR_MAP,
    apply_project_theme,
    background_image,
    crop_arrays_to_foreground,
    discrete_colorscale,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
    validate_columns,
)


def plot_per_image_performance(
    df: pd.DataFrame,
    image_col: str = "source_image",
    metric_cols: Sequence[str] = ("fn_rate", "fp_rate", "balanced_accuracy"),
    config_col: str | None = None,
    sort_metric: str = "fn_rate",
    worst_first: bool = True,
    top_n: int | None = None,
    title: str = "Performance by image",
    width: int = 1100,
    height: int = 650,
    show: bool = True,
):
    """Rank image-level metrics from easiest to hardest or vice versa."""
    validate_columns(df, [image_col])
    metrics = [column for column in metric_cols if column in df.columns]
    if not metrics:
        raise ValueError("No requested metric columns found.")
    d = df.copy()
    if sort_metric in d.columns:
        d = d.sort_values(sort_metric, ascending=not worst_first)
    if top_n is not None:
        d = d.head(int(top_n))
    d = d.iloc[::-1]

    colors = make_dynamic_color_map(metrics, prefer_project_colors=False)
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                x=d[metric],
                y=d[image_col].astype(str),
                orientation="h",
                name=metric,
                marker_color=colors[metric],
                customdata=(
                    d[[config_col]].astype(str).to_numpy()
                    if config_col and config_col in d.columns
                    else None
                ),
                hovertemplate=(
                    f"image: %{{y}}<br>{metric}: %{{x:.2%}}<br>"
                    + (f"{config_col}: %{{customdata[0]}}<br>" if config_col and config_col in d.columns else "")
                    + "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Metric value",
        yaxis_title=image_col,
        barmode="group",
        width=width,
        height=height,
    )
    fig.update_xaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_true_vs_predicted_object_counts(
    df: pd.DataFrame,
    true_count_col: str = "n_true_target_objects",
    predicted_count_col: str = "n_predicted_target_objects",
    image_col: str = "source_image",
    config_col: str | None = "selected_config_id",
    title: str = "True versus predicted target-object counts",
    width: int = 800,
    height: int = 650,
    show: bool = True,
):
    """Compare true and detected target counts per image against y=x."""
    validate_columns(df, [true_count_col, predicted_count_col])
    d = df.copy()
    groups = (
        d[config_col].astype(str)
        if config_col and config_col in d.columns
        else pd.Series(["all"] * len(d), index=d.index)
    )
    colors = make_dynamic_color_map(groups, prefer_project_colors=False)
    fig = go.Figure()
    for group in ordered_unique(groups):
        mask = groups.eq(group)
        custom = d.loc[mask, [image_col]].astype(str).to_numpy() if image_col in d.columns else None
        fig.add_trace(
            go.Scatter(
                x=d.loc[mask, true_count_col],
                y=d.loc[mask, predicted_count_col],
                mode="markers",
                name=group,
                marker=dict(size=11, color=colors[group], line=dict(width=0.8, color="black")),
                customdata=custom,
                hovertemplate=(
                    "true: %{x}<br>predicted: %{y}<br>"
                    + (f"{image_col}: %{{customdata[0]}}<br>" if custom is not None else "")
                    + "<extra></extra>"
                ),
            )
        )
    all_values = pd.concat(
        [pd.to_numeric(d[true_count_col], errors="coerce"), pd.to_numeric(d[predicted_count_col], errors="coerce")]
    ).dropna()
    if not all_values.empty:
        lo, hi = float(all_values.min()), float(all_values.max())
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="identity",
                line=dict(color="black", dash="dash"),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="True target-object count",
        yaxis_title="Predicted target-object count",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_stage_metric_comparison(
    df: pd.DataFrame,
    metric_col: str,
    stage_col: str = "stage",
    config_col: str = "selected_config_id",
    stages_order: Sequence[str] | None = None,
    title: str | None = None,
    width: int = 1000,
    height: int = 600,
    show: bool = True,
):
    """Trace one metric from validation to pure test and mixture application."""
    validate_columns(df, [metric_col, stage_col, config_col])
    d = df.copy()
    stage_order = (
        list(stages_order)
        if stages_order is not None
        else ordered_unique(d[stage_col].astype(str))
    )
    colors = make_dynamic_color_map(d[config_col].astype(str), prefer_project_colors=False)
    fig = go.Figure()
    for config in ordered_unique(d[config_col].astype(str)):
        sub = d[d[config_col].astype(str).eq(config)].copy()
        sub[stage_col] = pd.Categorical(sub[stage_col].astype(str), categories=stage_order, ordered=True)
        sub = sub.sort_values(stage_col)
        fig.add_trace(
            go.Scatter(
                x=sub[stage_col].astype(str),
                y=sub[metric_col],
                mode="lines+markers",
                name=config,
                line=dict(color=colors[config]),
            )
        )
    fig.update_layout(
        title=title or f"{metric_col} across evaluation stages",
        xaxis_title="Evaluation stage",
        yaxis_title=metric_col,
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_mixture_diagnostic_panel(
    image_key: str,
    image_db: dict,
    object_db: dict,
    object_df: pd.DataFrame,
    pixel_df: pd.DataFrame,
    target_class: str = "peanut",
    base: str = "image_ref",
    band: int | None = None,
    dilation_radius: int = 3,
    crop_to_objects: bool = True,
    padding: int = 5,
    title: str | None = None,
    width: int = 1200,
    height: int = 950,
    show: bool = True,
):
    """Four-panel mixture diagnostic: image, truth, object errors and pixel errors."""
    from src.decision.truth import target_truth_map_for_image

    background = background_image(image_db, image_key, base=base, band=band)
    truth, available = target_truth_map_for_image(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        target_class=target_class,
        dilation_radius=dilation_radius,
    )
    object_errors = make_object_error_map(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        object_df=object_df,
        target_class=target_class,
    )
    pixel_errors = make_pixel_error_map(
        image_key=image_key,
        image_db=image_db,
        pixel_df=pixel_df,
        target_class=target_class,
    )
    truth_plot = np.where(available, truth.astype(float), np.nan)
    object_plot = object_errors.astype(float)
    object_plot[object_plot == 0] = np.nan
    pixel_plot = pixel_errors.astype(float)
    pixel_plot[pixel_plot == 0] = np.nan

    if crop_to_objects:
        mask = np.asarray(image_db[image_key].get("labels", available)) > 0
        (background, truth_plot, object_plot, pixel_plot), _ = crop_arrays_to_foreground(
            [background, truth_plot, object_plot, pixel_plot],
            mask,
            padding=padding,
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Reference image",
            f"{target_class} ground truth",
            "Object-level errors",
            "Pixel-level errors",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )
    fig.add_trace(
        go.Heatmap(z=background, colorscale="Gray", showscale=False), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=truth_plot,
            zmin=0,
            zmax=1,
            colorscale=[[0, "royalblue"], [1, "limegreen"]],
            colorbar=dict(title="truth", tickvals=[0, 1], ticktext=["non-target", target_class], x=0.47),
        ),
        row=1,
        col=2,
    )
    error_scale = discrete_colorscale(
        [ERROR_COLOR_MAP[case] for case in ("TP", "TN", "FP", "FN")]
    )
    fig.add_trace(
        go.Heatmap(
            z=object_plot,
            zmin=1,
            zmax=4,
            colorscale=error_scale,
            colorbar=dict(
                title="object error",
                tickvals=[1, 2, 3, 4],
                ticktext=["TP", "TN", "FP", "FN"],
                x=0.47,
                y=0.20,
                len=0.35,
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=pixel_plot,
            zmin=1,
            zmax=4,
            colorscale=error_scale,
            colorbar=dict(
                title="pixel error",
                tickvals=[1, 2, 3, 4],
                ticktext=["TP", "TN", "FP", "FN"],
                x=1.02,
                y=0.20,
                len=0.35,
            ),
        ),
        row=2,
        col=2,
    )
    for row in (1, 2):
        for col in (1, 2):
            fig.update_yaxes(autorange="reversed", scaleanchor=f"x{(row - 1) * 2 + col}", row=row, col=col)
            fig.update_xaxes(title_text="column", row=row, col=col)
            fig.update_yaxes(title_text="row", row=row, col=col)
    fig.update_layout(
        title=title or f"Mixture diagnostic — {image_key}",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)
