from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils import filter_records, mask_value_to_nan, wavelength_axis
from src.visualization.common import (
    apply_project_theme,
    class_color,
    color_with_alpha,
    make_dynamic_color_map,
    show_or_return,
)


def plot_object_view(
    object_db_or_obj,
    object_id: str | None = None,
    spectrum_field: str = "mean_spectrum",
    show_spectrum: bool = True,
    show_std: bool = True,
    height: int = 500,
    width: int = 1000,
    show: bool = True,
):
    """Plot an object crop, its mask and optionally its mean spectrum."""
    obj = object_db_or_obj[object_id] if object_id is not None else object_db_or_obj
    object_label = object_id or "object"

    if "image_ref_crop" not in obj or "mask" not in obj:
        raise KeyError("Object must contain 'image_ref_crop' and 'mask'.")

    n_cols = 2 if show_spectrum else 1
    subplot_titles = (
        (f"{object_label} — crop", spectrum_field)
        if show_spectrum
        else (f"{object_label} — crop",)
    )
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)

    fig.add_trace(
        go.Heatmap(
            z=obj["image_ref_crop"],
            colorscale="Gray",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=mask_value_to_nan(obj["mask"], 0),
            colorscale=[[0, "rgba(255,0,0,0.35)"], [1, "rgba(255,0,0,0.35)"]],
            opacity=0.35,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)

    if show_spectrum:
        if spectrum_field not in obj:
            raise KeyError(f"Object has no spectrum field {spectrum_field!r}.")
        spectrum = np.asarray(obj[spectrum_field], dtype=float)
        x, x_title = wavelength_axis(spectrum.shape[0], obj.get("wavelengths"))
        color = class_color(obj.get("object_nut_type", "unknown"))

        if show_std and spectrum_field == "mean_spectrum" and "std_spectrum" in obj:
            std = np.asarray(obj["std_spectrum"], dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spectrum + std,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spectrum - std,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=color_with_alpha(color, 0.20),
                    name="±1 std",
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=spectrum,
                mode="lines",
                name=spectrum_field,
                line=dict(color=color),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text=x_title, row=1, col=2)
        fig.update_yaxes(title_text=obj.get("data_mode", "value"), row=1, col=2)

    fig.update_layout(
        title=(
            f"{object_label} | type={obj.get('object_nut_type')} | "
            f"source={obj.get('source_clean_key')} | area={obj.get('area_pixels')}"
        ),
        height=height,
        width=width,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_object_grid(
    objects_or_db,
    source_image: str | None = None,
    nut_type: str | None = None,
    title: str = "Object grid",
    subtitle_by_id: Mapping[str, str] | None = None,
    max_objects: int = 40,
    n_cols: int = 5,
    height_per_row: int = 220,
    width: int = 1100,
    sort_by: str | None = "area_pixels",
    descending: bool = True,
    show: bool = True,
):
    """Plot a grid of object crops with their segmentation masks."""
    if isinstance(objects_or_db, Mapping):
        objects = filter_records(
            objects_or_db,
            source_clean_key=source_image,
            object_nut_type=nut_type,
        )
    else:
        objects = list(objects_or_db)

    if sort_by is not None:
        objects = sorted(
            objects,
            key=lambda item: item[1].get(sort_by, -np.inf),
            reverse=descending,
        )
    selected = objects[: int(max_objects)]
    if not selected:
        raise ValueError("No object to plot.")

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            subtitle_by_id.get(str(object_id),f"{object_id}<br>area={obj.get('area_pixels')}")
            if subtitle_by_id is not None
            else f"{object_id}<br>area={obj.get('area_pixels')}"
        for object_id, obj in selected
        ],
    )

    for index, (_, obj) in enumerate(selected):
        row = index // n_cols + 1
        col = index % n_cols + 1
        fig.add_trace(
            go.Heatmap(z=obj["image_ref_crop"], colorscale="Gray", showscale=False),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Heatmap(
                z=mask_value_to_nan(obj["mask"], 0),
                colorscale=[[0, "red"], [1, "red"]],
                opacity=0.35,
                showscale=False,
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{index + 1}" if index else "x", row=row, col=col)

    fig.update_layout(title=title, height=height_per_row * n_rows, width=width)
    apply_project_theme(fig)
    return show_or_return(fig, show)


def _object_area_dataframe(
    object_db,
    source_image: str | None = None,
    nut_type: str | None = None,
) -> pd.DataFrame:
    objects = filter_records(
        object_db,
        source_clean_key=source_image,
        object_nut_type=nut_type,
    )
    rows = []
    for object_id, obj in objects:
        rows.append(
            {
                "object_id": object_id,
                "area_pixels": obj.get("area_pixels", np.nan),
                "object_nut_type": obj.get("object_nut_type", "unknown"),
                "batch": obj.get("batch", "unknown"),
                "source_image": obj.get("source_clean_key", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def plot_object_area_distribution(
    object_db_or_df,
    source_image: str | None = None,
    nut_type: str | None = None,
    area_col: str = "area_pixels",
    class_col: str = "object_nut_type",
    batch_col: str = "batch",
    kind: str = "box",
    facet_by_batch: bool = True,
    points: str | bool = "outliers",
    title: str = "Object area distribution",
    width: int = 1000,
    height: int = 600,
    show: bool = True,
):
    """Summarise object areas without drawing one unreadable bar per object."""
    if isinstance(object_db_or_df, pd.DataFrame):
        df = object_db_or_df.copy()
    else:
        df = _object_area_dataframe(object_db_or_df, source_image, nut_type)
    if df.empty:
        raise ValueError("No object found with these filters.")
    if area_col not in df.columns:
        raise KeyError(f"Missing area column: {area_col}")

    color_map = make_dynamic_color_map(df[class_col].astype(str)) if class_col in df.columns else None
    common = dict(
        data_frame=df,
        x=class_col if class_col in df.columns else None,
        y=area_col,
        color=class_col if class_col in df.columns else None,
        facet_col=batch_col if facet_by_batch and batch_col in df.columns else None,
        color_discrete_map=color_map,
        title=title,
    )
    if kind == "box":
        fig = px.box(**common, points=points)
    elif kind == "violin":
        fig = px.violin(**common, box=True, points=points)
    elif kind == "histogram":
        fig = px.histogram(
            df,
            x=area_col,
            color=class_col if class_col in df.columns else None,
            facet_col=batch_col if facet_by_batch and batch_col in df.columns else None,
            color_discrete_map=color_map,
            barmode="overlay",
            opacity=0.60,
            title=title,
        )
    else:
        raise ValueError("kind must be 'box', 'violin', or 'histogram'.")

    fig.update_layout(width=width, height=height)
    apply_project_theme(fig)
    return show_or_return(fig, show)

