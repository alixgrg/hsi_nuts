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


def _decision_color(label: str) -> str:
    """
    Stable colors for binary and 3-way SIMCA decisions.
    """
    lab = str(label).lower()

    if lab in {"almond", "non_target", "non_peanut", "almond_only"}:
        return "royalblue"

    if lab in {"uncertain", "ambiguous"}:
        return "purple"

    if lab in {"peanut", "target", "peanut_only"}:
        return "limegreen"

    return "lightgray"


def _discrete_colorscale(colors: list[str]) -> list[list[object]]:
    """
    Build a discrete Plotly colorscale from categorical colors.
    """
    n = len(colors)

    if n == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    scale = []

    for i, color in enumerate(colors):
        left = i / n
        right = (i + 1) / n
        scale.append([left, color])
        scale.append([right, color])

    return scale


def _apply_categorical_overlay_scale(
    fig: go.Figure,
    tickvals: list[int],
    ticktext: list[str],
    colorbar_title: str = "decision",
) -> go.Figure:
    """
    Force the last Heatmap trace to behave as a categorical overlay.
    """
    colors = [_decision_color(label) for label in ticktext]

    fig.data[-1].update(
        zmin=min(tickvals),
        zmax=max(tickvals),
        colorscale=_discrete_colorscale(colors),
        colorbar=dict(
            title=colorbar_title,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            x=1.12,
        ),
    )

    return fig


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
            "non_peanut": 2,
            "non_target": 2,
            "peanut_only": 3,
            "peanut": 3,
            "target": 3,
            "ambiguous": 4,
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

    fig = plot_image_overlay(
        img["image_ref"],
        decision_map,
        title=title or f"Object decisions — {image_key}",
        background_title="image_ref",
        overlay_title="decision",
        overlay_colorscale="Viridis",  # overwritten below
        overlay_colorbar=colorbar,
        alpha=0.55,
        width=width,
        height=height,
        show=False,
    )

    tickvals = sorted([c for c in code_to_name if c != 0])
    ticktext = [code_to_name[c] for c in tickvals]

    fig = _apply_categorical_overlay_scale(
        fig,
        tickvals=tickvals,
        ticktext=ticktext,
        colorbar_title="decision",
    )

    return show_or_return(fig, show)


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


def plot_pixel_three_way_decision_overlay(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    decision_col: str = "decision_3way",
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.60,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    """
    Overlay pixel-level view of objectwise 3-way decisions.

    Stable categorical colors:
    - non_target_label / non_target / almond -> blue
    - uncertain -> purple
    - target_class / target / peanut -> green
    """
    background = background_image(
        image_db,
        image_key,
        base=base,
        band=band,
    )

    shape = background.shape
    overlay = np.zeros(shape, dtype=float)

    decision_to_code = {
        str(non_target_label): 1,
        "non_target": 1,
        "non_peanut": 1,
        "almond": 1,
        "almond_only": 1,

        str(uncertain_label): 2,
        "uncertain": 2,
        "ambiguous": 2,

        str(target_class): 3,
        "target": 3,
        "peanut": 3,
        "peanut_only": 3,
    }

    sub = pixel_df[
        pixel_df[source_col].astype(str).eq(str(image_key))
    ].copy()

    if len(sub) > 0:
        rows = sub[row_col].astype(int).to_numpy()
        cols = sub[col_col].astype(int).to_numpy()

        decisions = (
            sub[decision_col]
            .astype(str)
            .map(decision_to_code)
            .fillna(0)
            .to_numpy()
        )

        overlay[rows, cols] = decisions

    overlay[overlay == 0] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=False,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=1,
            zmax=3,
            colorscale=_discrete_colorscale(
                [
                    _decision_color(non_target_label),
                    _decision_color(uncertain_label),
                    _decision_color(target_class),
                ]
            ),
            opacity=opacity,
            colorbar=dict(
                title="3-way decision",
                tickmode="array",
                tickvals=[1, 2, 3],
                ticktext=[
                    str(non_target_label),
                    str(uncertain_label),
                    str(target_class),
                ],
                x=1.12,
            ),
        )
    )

    fig.update_layout(
        title=title or f"Pixel view of 3-way decision — {image_key}",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )

    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return show_or_return(fig, show)


def three_way_confusion_table(
        df: pd.DataFrame,
        true_col: str,
        decision_col: str = "decision_3way",
        confidence_col: str | None = "three_way_confidence",
        group_cols=(),
        target_class: str = "peanut",
        non_target_label: str = "almond",
        uncertain_label: str = "uncertain",
    ) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()

        group_cols = [col for col in list(group_cols) if col in df.columns]
        required = [true_col, decision_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for 3-way confusion table: {missing}")

        d = df[df[true_col].notna() & df[decision_col].notna()].copy()
        if len(d) == 0:
            return pd.DataFrame()

        d["true_label_3way"] = np.where(
            d[true_col].astype(bool),
            target_class,
            non_target_label,
        )

        decisions = [non_target_label, uncertain_label, target_class]
        rows = []

        grouped = d.groupby(group_cols, dropna=False) if group_cols else [((), d)]
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            base = {col: value for col, value in zip(group_cols, key)}
            n_group = len(group)

            for true_label in [non_target_label, target_class]:
                true_group = group[group["true_label_3way"].astype(str).eq(str(true_label))]
                n_true = len(true_group)

                for decision in decisions:
                    cell = true_group[true_group[decision_col].astype(str).eq(str(decision))]
                    row = dict(base)
                    row.update({
                        "true_label_3way": true_label,
                        "decision_3way": decision,
                        "n": int(len(cell)),
                        "n_true_label": int(n_true),
                        "n_group": int(n_group),
                        "row_rate": len(cell) / max(n_true, 1),
                        "global_rate": len(cell) / max(n_group, 1),
                    })

                    if confidence_col is not None and confidence_col in cell.columns:
                        conf = pd.to_numeric(cell[confidence_col], errors="coerce")
                        row["mean_confidence"] = float(conf.mean()) if conf.notna().any() else np.nan
                        row["median_confidence"] = float(conf.median()) if conf.notna().any() else np.nan
                    else:
                        row["mean_confidence"] = np.nan
                        row["median_confidence"] = np.nan

                    rows.append(row)

        return pd.DataFrame(rows)


def plot_confusion_heatmap_from_long(
    confusion_df: pd.DataFrame,
    true_col_name: str,
    decision_col_name: str,
    title: str,
    count_col: str = "n",
    rate_col: str = "row_rate",
    confidence_col: str = "mean_confidence",
    width: int = 750,
    height: int = 550,
    show: bool = True,
):
    if confusion_df is None or len(confusion_df) == 0:
        raise ValueError("Empty confusion table.")

    d = confusion_df.copy()

    pivot_count = d.pivot_table(
        index=true_col_name,
        columns=decision_col_name,
        values=count_col,
        aggfunc="sum",
        fill_value=0,
    )
    pivot_rate = d.pivot_table(
        index=true_col_name,
        columns=decision_col_name,
        values=rate_col,
        aggfunc="mean",
        fill_value=0,
    )

    if confidence_col in d.columns:
        pivot_conf = d.pivot_table(
            index=true_col_name,
            columns=decision_col_name,
            values=confidence_col,
            aggfunc="mean",
        )
    else:
        pivot_conf = None

    text = []
    for i in pivot_count.index:
        row_text = []
        for j in pivot_count.columns:
            n = pivot_count.loc[i, j]
            rate = pivot_rate.loc[i, j]
            if pivot_conf is not None and i in pivot_conf.index and j in pivot_conf.columns and pd.notna(pivot_conf.loc[i, j]):
                row_text.append(f"{int(n)}<br>{rate:.1%}<br>conf={pivot_conf.loc[i, j]:.2f}")
            else:
                row_text.append(f"{int(n)}<br>{rate:.1%}")
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_rate.to_numpy(dtype=float),
            x=pivot_rate.columns.astype(str),
            y=pivot_rate.index.astype(str),
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title="row rate"),
            hovertemplate=(
                "True: %{y}<br>"
                "Decision: %{x}<br>"
                "Rate: %{z:.2%}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Decision",
        yaxis_title="True label",
        width=width,
        height=height,
    )

    return show_or_return(fig, show=show)


def plot_three_way_confusion_heatmap(
    confusion_df: pd.DataFrame,
    true_col: str = "true_label_3way",
    decision_col: str = "decision_3way",
    count_col: str = "n",
    rate_col: str = "row_rate",
    confidence_col: str = "mean_confidence",
    title: str = "3-way confusion table",
    width: int = 750,
    height: int = 550,
    show: bool = True,
):
    return plot_confusion_heatmap_from_long(
        confusion_df=confusion_df,
        true_col_name=true_col,
        decision_col_name=decision_col,
        title=title,
        count_col=count_col,
        rate_col=rate_col,
        confidence_col=confidence_col,
        width=width,
        height=height,
        show=show,
    )

def plot_binary_confusion_heatmap(
    confusion_df: pd.DataFrame,
    title: str = "2-way confusion table",
    show: bool = True,
):
    return plot_confusion_heatmap_from_long(
        confusion_df=confusion_df,
        true_col_name="true_label_2way",
        decision_col_name="predicted_label_2way",
        title=title,
        show=show,
    )