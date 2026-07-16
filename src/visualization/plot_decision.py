from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.decision.maps import (
    make_object_error_map,
    make_object_fp_fn_map,
    make_pixel_error_map,
    make_pixel_prediction_map,
)
from src.visualization.common import (
    ERROR_COLOR_MAP,
    apply_project_theme,
    background_image,
    class_color,
    crop_arrays_to_foreground,
    discrete_colorscale,
    normalize_class_label,
    show_or_return,
)
from src.visualization.plot_images import plot_image_overlay


DEFAULT_DECISION_CATEGORIES = {
    1: {"label": "unknown", "color": "lightgray"},
    2: {"label": "almond", "color": "royalblue"},
    3: {"label": "peanut", "color": "limegreen"},
    4: {"label": "uncertain", "color": "purple"},
}

DEFAULT_DECISION_TO_CODE = {
    "unknown": 1,
    "missing": 1,
    "almond": 2,
    "almond_only": 2,
    "non_peanut": 2,
    "non_target": 2,
    "peanut": 3,
    "peanut_only": 3,
    "target": 3,
    "ambiguous": 4,
    "uncertain": 4,
}


def _category_spec(
    categories: Mapping[int, Mapping[str, str]] | None,
) -> dict[int, dict[str, str]]:
    source = DEFAULT_DECISION_CATEGORIES if categories is None else categories
    out: dict[int, dict[str, str]] = {}
    for code, value in source.items():
        if isinstance(value, Mapping):
            label = str(value.get("label", code))
            color = str(value.get("color", class_color(label)))
        else:
            label = str(value)
            color = class_color(label)
        out[int(code)] = {"label": label, "color": color}
    return out


def _apply_categorical_overlay_scale(
    fig: go.Figure,
    categories: Mapping[int, Mapping[str, str]],
    colorbar_title: str = "decision",
) -> go.Figure:
    codes = sorted(categories)
    colors = [categories[code]["color"] for code in codes]
    labels = [categories[code]["label"] for code in codes]
    fig.data[-1].update(
        zmin=min(codes),
        zmax=max(codes),
        colorscale=discrete_colorscale(colors),
        colorbar=dict(
            title=colorbar_title,
            tickmode="array",
            tickvals=codes,
            ticktext=labels,
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
    categories: Mapping[int, Mapping[str, str]] | None = None,
    code_to_name: Mapping[int, str] | None = None,
    title: str | None = None,
    width: int = 850,
    height: int = 750,
    show: bool = True,
    base: str = "image_ref",
    band: int | None = None,
    opacity: float = 0.55,
    crop_to_objects: bool = True,
    padding: int = 5,
):
    """Overlay object-level binary or 3-way decisions on an image.

    Labels and colours are stored separately, so changing a legend label can no
    longer accidentally turn a category grey.
    """
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")
    img = image_db[image_key]
    if "labels" not in img:
        raise KeyError(f"Image {image_key!r} has no labels image.")

    labels_img = np.asarray(img["labels"])
    background = background_image(image_db, image_key, base=base, band=band)
    decision_to_code = dict(DEFAULT_DECISION_TO_CODE if decision_to_code is None else decision_to_code)
    category_map = _category_spec(categories)
    if code_to_name is not None:
        # Backward compatibility: display names change, colours stay code-based.
        for code, name in code_to_name.items():
            if int(code) in category_map:
                category_map[int(code)]["label"] = str(name)

    decision_map = np.zeros_like(labels_img, dtype=float)
    sub = results_df[results_df[source_col].astype(str).eq(str(image_key))]
    for _, row in sub.iterrows():
        object_id = str(row[object_id_col])
        obj = object_db.get(object_id)
        if obj is None or "label_id" not in obj:
            continue
        raw_decision = str(row[decision_col]).strip().lower()
        normalized = normalize_class_label(raw_decision)
        code = decision_to_code.get(raw_decision, decision_to_code.get(normalized, 1))
        decision_map[labels_img == int(obj["label_id"])] = int(code)

    used_codes = sorted(int(code) for code in np.unique(decision_map) if code != 0)
    shown_categories = {
        code: category_map.get(code, {"label": str(code), "color": "lightgray"})
        for code in used_codes
    }
    if not shown_categories:
        shown_categories = category_map

    fig = plot_image_overlay(
        background,
        decision_map,
        title=title or f"Object decisions — {image_key}",
        background_title=base,
        overlay_title="decision",
        overlay_colorscale="Viridis",
        alpha=opacity,
        width=width,
        height=height,
        zmin=min(shown_categories),
        zmax=max(shown_categories),
        crop_to_foreground=crop_to_objects,
        foreground_mask=labels_img > 0,
        padding=padding,
        show=False,
    )
    _apply_categorical_overlay_scale(fig, shown_categories, "decision")
    return show_or_return(fig, show)


def _plot_coded_overlay(
    background: np.ndarray,
    overlay: np.ndarray,
    *,
    categories: Mapping[int, Mapping[str, str]],
    title: str,
    colorbar_title: str,
    opacity: float,
    width: int,
    height: int,
    crop_mask: np.ndarray | None = None,
    crop_to_objects: bool = True,
    padding: int = 5,
    show: bool = True,
):
    background = np.asarray(background)
    overlay = np.asarray(overlay, dtype=float)
    overlay[overlay == 0] = np.nan
    if crop_to_objects and crop_mask is not None:
        (background, overlay), _ = crop_arrays_to_foreground(
            [background, overlay], np.asarray(crop_mask, dtype=bool), padding=padding
        )

    codes = sorted(categories)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=background, colorscale="Gray", showscale=False)
    )
    fig.add_trace(
        go.Heatmap(
            z=overlay,
            zmin=min(codes),
            zmax=max(codes),
            colorscale=discrete_colorscale([categories[code]["color"] for code in codes]),
            opacity=opacity,
            colorbar=dict(
                title=colorbar_title,
                tickmode="array",
                tickvals=codes,
                ticktext=[categories[code]["label"] for code in codes],
                x=1.12,
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
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    apply_project_theme(fig)
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
    error_cases: Sequence[str] = ("TP", "TN", "FP", "FN"),
    crop_to_objects: bool = True,
    padding: int = 5,
    show: bool = True,
):
    """Overlay selected object-level TP/TN/FP/FN cases."""
    background = background_image(image_db, image_key, base=base, band=band)
    err = make_object_error_map(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        object_df=object_df,
        target_class=target_class,
        pred_col=pred_col,
        true_col=true_col,
    )
    code_by_case = {"TP": 1, "TN": 2, "FP": 3, "FN": 4}
    keep = [case for case in error_cases if case in code_by_case]
    filtered = np.zeros_like(err, dtype=float)
    categories = {}
    for display_code, case in enumerate(keep, start=1):
        filtered[err == code_by_case[case]] = display_code
        categories[display_code] = {"label": case, "color": ERROR_COLOR_MAP[case]}
    if not categories:
        raise ValueError("error_cases must include at least one of TP, TN, FP, FN.")
    crop_mask = np.asarray(image_db[image_key].get("labels", err)) > 0
    return _plot_coded_overlay(
        background,
        filtered,
        categories=categories,
        title=title or f"Object-level errors — {image_key}",
        colorbar_title="object error",
        opacity=opacity,
        width=width,
        height=height,
        crop_mask=crop_mask,
        crop_to_objects=crop_to_objects,
        padding=padding,
        show=show,
    )


def plot_object_fp_fn_overlay(*args, **kwargs):
    """Deprecated compatibility wrapper for an FP/FN-only object map."""
    warnings.warn(
        "plot_object_fp_fn_overlay is deprecated; use plot_object_error_overlay "
        "with error_cases=('FP', 'FN').",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs["error_cases"] = ("FP", "FN")
    return plot_object_error_overlay(*args, **kwargs)


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
    error_cases: Sequence[str] = ("TP", "TN", "FP", "FN"),
    crop_to_objects: bool = True,
    padding: int = 5,
    show: bool = True,
):
    """Overlay selected pixel-level TP/TN/FP/FN cases."""
    background = background_image(image_db, image_key, base=base, band=band)
    err = make_pixel_error_map(
        image_key=image_key,
        image_db=image_db,
        pixel_df=pixel_df,
        target_class=target_class,
    )
    code_by_case = {"TP": 1, "TN": 2, "FP": 3, "FN": 4}
    keep = [case for case in error_cases if case in code_by_case]
    filtered = np.zeros_like(err, dtype=float)
    categories = {}
    for display_code, case in enumerate(keep, start=1):
        filtered[err == code_by_case[case]] = display_code
        categories[display_code] = {"label": case, "color": ERROR_COLOR_MAP[case]}
    if not categories:
        raise ValueError("error_cases must include at least one of TP, TN, FP, FN.")
    crop_mask = np.asarray(image_db[image_key].get("labels", err)) > 0
    return _plot_coded_overlay(
        background,
        filtered,
        categories=categories,
        title=title or f"Pixel errors — {image_key}",
        colorbar_title="pixel error",
        opacity=opacity,
        width=width,
        height=height,
        crop_mask=crop_mask,
        crop_to_objects=crop_to_objects,
        padding=padding,
        show=show,
    )


def plot_pixel_fp_fn_overlay(*args, **kwargs):
    """Deprecated compatibility wrapper for an FP/FN-only pixel map."""
    warnings.warn(
        "plot_pixel_fp_fn_overlay is deprecated; use plot_pixel_error_overlay "
        "with error_cases=('FP', 'FN').",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs["error_cases"] = ("FP", "FN")
    return plot_pixel_error_overlay(*args, **kwargs)


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
    crop_to_objects: bool = True,
    padding: int = 5,
    show: bool = True,
):
    """Overlay binary target predictions using the stable target colour."""
    background = background_image(image_db, image_key, base=base, band=band)
    pred_map = make_pixel_prediction_map(
        image_key=image_key,
        image_db=image_db,
        pixel_df=pixel_df,
        target_class=target_class,
        pred_col=pred_col,
    )
    crop_mask = np.asarray(image_db[image_key].get("labels", pred_map)) > 0
    return _plot_coded_overlay(
        background,
        pred_map,
        categories={1: {"label": str(target_class), "color": class_color(target_class)}},
        title=title or f"Pixel-level prediction — {image_key}",
        colorbar_title=f"predicted {target_class}",
        opacity=opacity,
        width=width,
        height=height,
        crop_mask=crop_mask,
        crop_to_objects=crop_to_objects,
        padding=padding,
        show=show,
    )


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
    crop_to_objects: bool = True,
    padding: int = 5,
    show: bool = True,
):
    """Overlay pixel-level views of objectwise 3-way decisions."""
    background = background_image(image_db, image_key, base=base, band=band)
    overlay = np.zeros(background.shape, dtype=float)
    sub = pixel_df[pixel_df[source_col].astype(str).eq(str(image_key))].copy()
    if not sub.empty:
        rows = sub[row_col].astype(int).to_numpy()
        cols = sub[col_col].astype(int).to_numpy()
        normalized = [
            normalize_class_label(
                value,
                target_class=target_class,
                non_target_label=non_target_label,
                uncertain_label=uncertain_label,
            )
            for value in sub[decision_col]
        ]
        code_lookup = {
            str(non_target_label): 1,
            str(uncertain_label): 2,
            str(target_class): 3,
        }
        codes = np.asarray([code_lookup.get(value, 0) for value in normalized])
        valid = (
            (rows >= 0)
            & (rows < overlay.shape[0])
            & (cols >= 0)
            & (cols < overlay.shape[1])
        )
        overlay[rows[valid], cols[valid]] = codes[valid]

    crop_mask = np.asarray(image_db[image_key].get("labels", overlay)) > 0
    categories = {
        1: {"label": str(non_target_label), "color": class_color(non_target_label)},
        2: {"label": str(uncertain_label), "color": class_color(uncertain_label)},
        3: {"label": str(target_class), "color": class_color(target_class)},
    }
    return _plot_coded_overlay(
        background,
        overlay,
        categories=categories,
        title=title or f"Pixel view of 3-way decision — {image_key}",
        colorbar_title="3-way decision",
        opacity=opacity,
        width=width,
        height=height,
        crop_mask=crop_mask,
        crop_to_objects=crop_to_objects,
        padding=padding,
        show=show,
    )


def _coerce_true_target(series: pd.Series, target_class: str) -> pd.Series:
    values = series.astype("object")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    numeric_bool = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    out.loc[numeric_bool] = values.loc[numeric_bool].astype(bool)
    text = values.astype(str).str.strip().str.lower()
    out.loc[text.isin({"true", "1", "yes", "y", "target", "peanut", str(target_class).lower()})] = True
    out.loc[text.isin({"false", "0", "no", "n", "non_target", "almond", "non_peanut"})] = False
    return out


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
    """Build a complete long-format 3-way confusion table."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    group_cols = [column for column in list(group_cols) if column in df.columns]
    missing = [column for column in (true_col, decision_col) if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for 3-way confusion table: {missing}")

    d = df[df[true_col].notna() & df[decision_col].notna()].copy()
    true_target = _coerce_true_target(d[true_col], target_class)
    d = d[true_target.notna()].copy()
    true_target = true_target.loc[d.index].astype(bool)
    d["true_label_3way"] = np.where(true_target, target_class, non_target_label)
    d[decision_col] = [
        normalize_class_label(
            value,
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
        )
        for value in d[decision_col]
    ]

    decisions = [non_target_label, uncertain_label, target_class]
    rows = []
    grouped = d.groupby(group_cols, dropna=False) if group_cols else [((), d)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        n_group = len(group)
        for true_label in [non_target_label, target_class]:
            true_group = group[group["true_label_3way"].astype(str).eq(str(true_label))]
            n_true = len(true_group)
            for decision in decisions:
                cell = true_group[true_group[decision_col].astype(str).eq(str(decision))]
                row = dict(base)
                row.update(
                    true_label_3way=true_label,
                    decision_3way=decision,
                    n=int(len(cell)),
                    n_true_label=int(n_true),
                    n_group=int(n_group),
                    row_rate=len(cell) / n_true if n_true else np.nan,
                    global_rate=len(cell) / n_group if n_group else np.nan,
                )
                if confidence_col is not None and confidence_col in cell.columns:
                    conf = pd.to_numeric(cell[confidence_col], errors="coerce")
                    row["mean_confidence"] = conf.mean()
                    row["median_confidence"] = conf.median()
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
    true_order: Sequence[str] | None = None,
    decision_order: Sequence[str] | None = None,
    display: str = "both",
    width: int = 750,
    height: int = 550,
    show: bool = True,
):
    """Plot a confusion heatmap with fixed category order and a [0, 1] scale."""
    if confusion_df is None or len(confusion_df) == 0:
        raise ValueError("Empty confusion table.")
    d = confusion_df.copy()

    # Sum counts first; derive rates from aggregated counts to avoid averaging rates.
    counts = d.pivot_table(
        index=true_col_name,
        columns=decision_col_name,
        values=count_col,
        aggfunc="sum",
        fill_value=0,
    )
    if true_order is not None:
        counts = counts.reindex(index=list(true_order), fill_value=0)
    if decision_order is not None:
        counts = counts.reindex(columns=list(decision_order), fill_value=0)
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    rates = counts.div(row_totals, axis=0).fillna(0.0)

    confidence = None
    if confidence_col in d.columns:
        confidence = d.pivot_table(
            index=true_col_name,
            columns=decision_col_name,
            values=confidence_col,
            aggfunc="mean",
        ).reindex(index=counts.index, columns=counts.columns)

    text = []
    for true_label in counts.index:
        row = []
        for decision in counts.columns:
            n = int(counts.loc[true_label, decision])
            rate = float(rates.loc[true_label, decision])
            if display == "count":
                value = f"{n}"
            elif display == "rate":
                value = f"{rate:.1%}"
            else:
                value = f"{n}<br>{rate:.1%}"
            if confidence is not None and pd.notna(confidence.loc[true_label, decision]):
                value += f"<br>conf={confidence.loc[true_label, decision]:.2f}"
            row.append(value)
        text.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=rates.to_numpy(dtype=float),
            x=rates.columns.astype(str),
            y=rates.index.astype(str),
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="row rate"),
            hovertemplate=(
                "True: %{y}<br>Decision: %{x}<br>Rate: %{z:.2%}<extra></extra>"
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
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_three_way_confusion_heatmap(
    confusion_df: pd.DataFrame,
    true_col: str = "true_label_3way",
    decision_col: str = "decision_3way",
    count_col: str = "n",
    rate_col: str = "row_rate",
    confidence_col: str = "mean_confidence",
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
    title: str = "3-way confusion table",
    width: int = 750,
    height: int = 550,
    show: bool = True,
):
    return plot_confusion_heatmap_from_long(
        confusion_df,
        true_col_name=true_col,
        decision_col_name=decision_col,
        title=title,
        count_col=count_col,
        rate_col=rate_col,
        confidence_col=confidence_col,
        true_order=[non_target_label, target_class],
        decision_order=[non_target_label, uncertain_label, target_class],
        width=width,
        height=height,
        show=show,
    )


def plot_binary_confusion_heatmap(
    confusion_df: pd.DataFrame,
    true_col: str = "true_label_2way",
    decision_col: str = "predicted_label_2way",
    target_class: str = "peanut",
    non_target_label: str = "almond",
    title: str = "2-way confusion table",
    show: bool = True,
):
    return plot_confusion_heatmap_from_long(
        confusion_df,
        true_col_name=true_col,
        decision_col_name=decision_col,
        title=title,
        true_order=[non_target_label, target_class],
        decision_order=[non_target_label, target_class],
        show=show,
    )
