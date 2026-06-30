from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.decision.maps import (
    make_pixel_error_map,
    make_pixel_prediction_map,
    make_object_error_map,
    make_object_fp_fn_map,
)
from src.visualization.common import background_image, show_or_return
from src.visualization.plot_images import plot_image_overlay


def plot_object_decision_map(
    image_db,
    object_db,
    results_df: pd.DataFrame,
    image_key: str,
    decision_col: str = "simca_case",
    object_id_col: str = "object_id",
    source_col: str = "source_image",
    decision_to_code: Mapping[str, int] | None = None,
    code_to_name: Mapping[int, str] | None = None,
    title: str | None = None,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """Overlay object-level decisions on an image."""
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    img = image_db[image_key]
    labels_img = img["labels"]

    if decision_to_code is None:
        decision_to_code = {
            "unknown": 1,
            "almond_only": 2,
            "peanut_only": 3,
            "ambiguous": 4,
            "non_peanut": 2,
            "peanut": 3,
            "uncertain": 4,
        }

    if code_to_name is None:
        code_to_name = {
            0: "background",
            1: "unknown",
            2: "non_peanut / almond_only",
            3: "peanut",
            4: "ambiguous / uncertain",
        }

    decision_map = np.zeros_like(labels_img, dtype=float)
    sub = results_df[results_df[source_col].astype(str) == str(image_key)]

    for _, row in sub.iterrows():
        obj_id = str(row[object_id_col])

        if obj_id not in object_db:
            continue

        label_id = object_db[obj_id]["label_id"]
        decision = str(row[decision_col])
        decision_map[labels_img == label_id] = decision_to_code.get(decision, 1)

    tickvals = sorted([c for c in code_to_name if c != 0])

    colorbar = dict(
        title="decision",
        tickvals=tickvals,
        ticktext=[code_to_name[c] for c in tickvals],
        x=1.12,
    )

    colorscale = [
        [0.00, "lightgray"],
        [0.25, "royalblue"],
        [0.50, "crimson"],
        [0.75, "orange"],
        [1.00, "purple"],
    ]

    return plot_image_overlay(
        img["image_ref"],
        decision_map,
        title=title or f"Object decisions — {image_key}",
        background_title="image_ref",
        overlay_title="decision",
        overlay_colorscale=colorscale,
        overlay_colorbar=colorbar,
        alpha=0.55,
        width=width,
        height=height,
        show=show,
    )


def plot_object_error_overlay(
    image_key: str,
    image_db,
    object_db,
    object_df: pd.DataFrame,
    target_class: str = "peanut",
    pred_col: str | None = None,
    true_col: str | None = None,
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.60,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """
    Overlay TP/TN/FP/FN object-level errors on an image.

    Codes
    -----
    TP : true target predicted target
    TN : true non-target predicted non-target
    FP : true non-target predicted target
    FN : true target predicted non-target
    """
    background = background_image(
        image_db,
        image_key,
        base=base,
        band=band,
    )

    err = make_object_error_map(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        object_df=object_df,
        target_class=target_class,
        pred_col=pred_col,
        true_col=true_col,
    )

    overlay = err.astype(float)
    overlay[overlay == 0] = np.nan

    colorscale = [
        [0.00, "limegreen"],   # 1 TP
        [0.33, "royalblue"],   # 2 TN
        [0.66, "orange"],      # 3 FP
        [1.00, "red"],         # 4 FN
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=False,
            colorbar=dict(title=base),
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=1,
            zmax=4,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(
                title="object error",
                tickvals=[1, 2, 3, 4],
                ticktext=["TP", "TN", "FP", "FN"],
                x=1.12,
            ),
        )
    )

    fig.update_layout(
        title=title or f"Object-level errors — {image_key}",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return show_or_return(fig, show)


def plot_object_fp_fn_overlay(
    image_key: str,
    image_db,
    object_db,
    object_df: pd.DataFrame,
    target_class: str = "peanut",
    pred_col: str | None = None,
    true_col: str | None = None,
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.75,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """
    Overlay only object-level false positives and false negatives.

    Codes
    -----
    1 : FP
    2 : FN
    """
    background = background_image(
        image_db,
        image_key,
        base=base,
        band=band,
    )

    fp_fn = make_object_fp_fn_map(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        object_df=object_df,
        target_class=target_class,
        pred_col=pred_col,
        true_col=true_col,
    )

    overlay = fp_fn.astype(float)
    overlay[overlay == 0] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=False,
            colorbar=dict(title=base),
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=1,
            zmax=2,
            colorscale=[
                [0.00, "orange"],  # FP
                [1.00, "red"],     # FN
            ],
            opacity=opacity,
            colorbar=dict(
                title="object error",
                tickvals=[1, 2],
                ticktext=["FP", "FN"],
                x=1.12,
            ),
        )
    )

    fig.update_layout(
        title=title or f"False positive / false negative objects — {image_key}",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return show_or_return(fig, show)


def plot_pixel_error_overlay(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    target_class: str = "peanut",
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.60,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """Overlay TP/TN/FP/FN pixel errors on an image."""
    background = background_image(
        image_db,
        image_key,
        base=base,
        band=band,
    )

    err = make_pixel_error_map(
        image_key=image_key,
        image_db=image_db,
        pixel_df=pixel_df,
        target_class=target_class,
    )

    overlay = err.astype(float)
    overlay[overlay == 0] = np.nan

    colorscale = [
        [0.00, "limegreen"],   # 1 TP
        [0.33, "royalblue"],   # 2 TN
        [0.66, "orange"],      # 3 FP
        [1.00, "red"],         # 4 FN
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=False,
            colorbar=dict(title=base),
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=1,
            zmax=4,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(
                title="error",
                tickvals=[1, 2, 3, 4],
                ticktext=["TP", "TN", "FP", "FN"],
                x=1.12,
            ),
        )
    )

    fig.update_layout(
        title=title or f"Pixel errors — {image_key}",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return show_or_return(fig, show)


def plot_pixel_fp_fn_overlay(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    target_class: str = "peanut",
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.75,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """Overlay only false positives and false negatives."""
    background = background_image(
        image_db,
        image_key,
        base=base,
        band=band,
    )

    err = make_pixel_error_map(
        image_key=image_key,
        image_db=image_db,
        pixel_df=pixel_df,
        target_class=target_class,
    )

    overlay = np.zeros_like(err, dtype=float)
    overlay[err == 3] = 1  # FP
    overlay[err == 4] = 2  # FN
    overlay[overlay == 0] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=True,
            colorbar=dict(title=base),
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=1,
            zmax=2,
            colorscale=[
                [0.00, "red"],
                [1.00, "orange"],
            ],
            opacity=opacity,
            colorbar=dict(
                title="error",
                tickvals=[1, 2],
                ticktext=["FP", "FN"],
                x=1.12,
            ),
        )
    )

    fig.update_layout(
        title=title or f"False positive / false negative pixels — {image_key}",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return show_or_return(fig, show)


def plot_pixel_prediction_overlay(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    target_class: str = "peanut",
    pred_col: str | None = None,
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.60,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """Overlay binary pixel-level target predictions on an image."""
    background = background_image(
        image_db,
        image_key,
        base=base,
        band=band,
    )

    pred_map = make_pixel_prediction_map(
        image_key=image_key,
        image_db=image_db,
        pixel_df=pixel_df,
        target_class=target_class,
        pred_col=pred_col,
    )

    overlay = pred_map.astype(float)
    overlay[overlay == 0] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=False,
            colorbar=dict(title=base),
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=1,
            zmax=1,
            colorscale=[[0.0, "crimson"], [1.0, "crimson"]],
            opacity=opacity,
            colorbar=dict(
                title=f"predicted {target_class}",
                tickvals=[1],
                ticktext=[target_class],
                x=1.12,
            ),
        )
    )

    fig.update_layout(
        title=title or f"Pixel-level prediction — {image_key}",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return show_or_return(fig, show)