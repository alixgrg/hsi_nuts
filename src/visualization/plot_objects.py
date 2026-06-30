from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import filter_records, mask_value_to_nan, wavelength_axis
from src.visualization.common import show_or_return
from src.visualization.plot_generic import plot_bar_values


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
    """Plot object crop, mask and optionally its spectrum."""
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

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Heatmap(
            z=obj["image_ref_crop"],
            colorscale="Gray",
            showscale=False,
            colorbar=dict(title="Image ref"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=mask_value_to_nan(obj["mask"], 0),
            colorscale="Reds",
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

        fig.add_trace(
            go.Scatter(
                x=x,
                y=spectrum,
                mode="lines",
                name=spectrum_field,
            ),
            row=1,
            col=2,
        )

        if show_std and spectrum_field == "mean_spectrum" and "std_spectrum" in obj:
            std = np.asarray(obj["std_spectrum"], dtype=float)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spectrum + std,
                    mode="lines",
                    name="+1 std",
                    line=dict(dash="dash"),
                    opacity=0.5,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spectrum - std,
                    mode="lines",
                    name="-1 std",
                    line=dict(dash="dash"),
                    opacity=0.5,
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(title_text=x_title, row=1, col=2)
        fig.update_yaxes(title_text=obj.get("data_mode", "value"), row=1, col=2)

    fig.update_layout(
        title=(
            f"{object_label} | type={obj.get('object_nut_type')} | "
            f"source={obj.get('source_clean_key')} | "
            f"area={obj.get('area_pixels')}"
        ),
        height=height,
        width=width,
    )

    return show_or_return(fig, show)


def plot_object_grid(
    objects_or_db,
    source_image: str | None = None,
    nut_type: str | None = None,
    title: str = "Object grid",
    max_objects: int = 40,
    n_cols: int = 5,
    height_per_row: int = 220,
    width: int = 1100,
    show: bool = True,
):
    """Plot a grid of object crops with masks."""
    if isinstance(objects_or_db, Mapping):
        objects = filter_records(
            objects_or_db,
            source_clean_key=source_image,
            object_nut_type=nut_type,
        )
    else:
        objects = list(objects_or_db)

    selected = objects[:max_objects]

    if not selected:
        raise ValueError("No object to plot.")

    n_rows = int(np.ceil(len(selected) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{oid}<br>area={obj.get('area_pixels')}"
            for oid, obj in selected
        ],
    )

    for idx, (_, obj) in enumerate(selected):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        fig.add_trace(
            go.Heatmap(
                z=obj["image_ref_crop"],
                colorscale="Gray",
                showscale=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Heatmap(
                z=mask_value_to_nan(obj["mask"], 0),
                colorscale="Reds",
                opacity=0.35,
                showscale=False,
            ),
            row=row,
            col=col,
        )

        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title=title,
        height=height_per_row * n_rows,
        width=width,
    )

    return show_or_return(fig, show)


def plot_object_areas(
    object_db,
    source_image=None,
    nut_type=None,
    show=True,
):
    """Bar plot of object areas."""
    objects = filter_records(
        object_db,
        source_clean_key=source_image,
        object_nut_type=nut_type,
    )

    if not objects:
        raise ValueError("No object found with these filters.")

    labels = [oid for oid, _ in objects]
    areas = [obj["area_pixels"] for _, obj in objects]

    suffix = ""
    if source_image is not None:
        suffix += f" — {source_image}"
    if nut_type is not None:
        suffix += f" — {nut_type}"

    return plot_bar_values(
        labels,
        areas,
        title=f"Object areas{suffix}",
        x_title="object_id",
        y_title="area_pixels",
        show=show,
    )