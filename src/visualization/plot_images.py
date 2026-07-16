from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go

from src.utils import mask_value_to_nan
from src.visualization.common import (
    apply_project_theme,
    background_image,
    crop_arrays_to_foreground,
    show_or_return,
)


def plot_hypercube_band_slider(
    cube: np.ndarray,
    wavelengths=None,
    title: str = "Hyperspectral image",
    value_name: str = "Value",
    colorscale: str = "Viridis",
    width: int = 700,
    height: int = 700,
    show: bool = True,
):
    """Interactive slider over spectral bands of a cube shaped ``(H, W, B)``."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError("cube must have shape (height, width, bands).")

    n_bands = cube.shape[2]
    wavelengths = None if wavelengths is None else np.asarray(wavelengths)
    if wavelengths is not None and len(wavelengths) != n_bands:
        raise ValueError(
            f"wavelengths length ({len(wavelengths)}) does not match "
            f"number of bands ({n_bands})."
        )

    fig = go.Figure()
    for index in range(n_bands):
        label = (
            f"band {index}"
            if wavelengths is None
            else f"{wavelengths[index]:.1f} nm"
        )
        fig.add_trace(
            go.Heatmap(
                z=cube[:, :, index],
                visible=index == 0,
                colorscale=colorscale,
                colorbar=dict(title=value_name),
                hovertemplate=(
                    "row: %{y}<br>col: %{x}<br>"
                    f"{label}<br>{value_name}: %{{z}}<extra></extra>"
                ),
            )
        )

    steps = []
    for index in range(n_bands):
        label = str(index) if wavelengths is None else f"{wavelengths[index]:.0f}"
        title_i = (
            f"{title} — band {index}"
            if wavelengths is None
            else f"{title} — band {index} — {wavelengths[index]:.1f} nm"
        )
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": [j == index for j in range(n_bands)]},
                    {"title": title_i},
                ],
                label=label,
            )
        )

    fig.update_layout(
        title=f"{title} — band 0",
        sliders=[dict(active=0, currentvalue={"prefix": "Band: "}, steps=steps)],
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_image2d(
    z: np.ndarray,
    title: str = "Image",
    colorscale: str = "Viridis",
    colorbar_title: str = "Value",
    width: int = 800,
    height: int = 700,
    reverse_y: bool = True,
    showscale: bool = True,
    show: bool = True,
):
    """Generic 2D heatmap."""
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array.")

    fig = go.Figure(
        go.Heatmap(
            z=z,
            colorscale=colorscale,
            showscale=showscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    if reverse_y:
        fig.update_yaxes(autorange="reversed", scaleanchor="x")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_image_overlay(
    background: np.ndarray,
    overlay: np.ndarray,
    title: str = "Image overlay",
    background_colorscale: str = "Gray",
    overlay_colorscale: Any = "Turbo",
    background_title: str = "Background",
    overlay_title: str = "Overlay",
    overlay_mask_value=0,
    alpha: float = 0.45,
    width: int = 850,
    height: int = 750,
    overlay_colorbar: dict | None = None,
    overlay_showscale: bool = True,
    zmin: float | None = None,
    zmax: float | None = None,
    crop_to_foreground: bool = False,
    foreground_mask: np.ndarray | None = None,
    padding: int = 5,
    show: bool = True,
):
    """Generic image plus semi-transparent overlay.

    ``crop_to_foreground=True`` crops both arrays using ``foreground_mask`` or,
    by default, the non-masked overlay pixels. This removes empty bands around
    the analysed objects while preserving pixel alignment.
    """
    background = np.asarray(background)
    overlay = np.asarray(overlay)
    if background.ndim != 2 or overlay.ndim != 2:
        raise ValueError("background and overlay must be 2D arrays.")
    if background.shape != overlay.shape:
        raise ValueError(
            f"background shape {background.shape} and overlay shape "
            f"{overlay.shape} must match."
        )

    overlay_plot = mask_value_to_nan(overlay, mask_value=overlay_mask_value)
    if crop_to_foreground:
        if foreground_mask is None:
            foreground_mask = np.isfinite(overlay_plot)
        (background, overlay_plot), _ = crop_arrays_to_foreground(
            [background, overlay_plot],
            np.asarray(foreground_mask, dtype=bool),
            padding=padding,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale=background_colorscale,
            showscale=False,
            colorbar=dict(title=background_title),
            hovertemplate=(
                "row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=overlay_plot,
            colorscale=overlay_colorscale,
            opacity=float(alpha),
            showscale=overlay_showscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=overlay_colorbar or dict(title=overlay_title, x=1.12),
            hovertemplate=(
                "row: %{y}<br>col: %{x}<br>overlay: %{z}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_label_overlay_from_image_db(
    image_db: Mapping[str, Mapping[str, Any]],
    image_key: str,
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    crop_to_objects: bool = True,
    padding: int = 5,
    show: bool = True,
):
    """Overlay segmentation labels over a background image."""
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")
    img = image_db[image_key]
    if "labels" not in img:
        raise KeyError(f"Image {image_key!r} has no 'labels' field.")

    labels = np.asarray(img["labels"])
    bg = background_image(image_db, image_key, base=base, band=band)
    return plot_image_overlay(
        bg,
        labels,
        title=title or f"Labels overlay — {image_key}",
        background_title=base,
        overlay_title="label",
        crop_to_foreground=crop_to_objects,
        foreground_mask=labels > 0,
        padding=padding,
        show=show,
    )
