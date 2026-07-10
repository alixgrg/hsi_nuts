from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.decision.aggregation import add_object_metadata
from src.decision.metrics import binary_detection_metrics
from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    DEFAULT_NON_TARGET_LABEL,
    predicted_col as make_predicted_col,
    true_col as make_true_col,
    pixel_ratio_col,
    true_pixel_ratio_total_col,
    n_predicted_pixels_col,
)

def _object_distance_map(obj: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return distance-to-background inside an object's crop mask."""
    if "bbox" not in obj or "mask" not in obj:
        raise KeyError("Object must contain 'bbox' and 'mask'.")

    mask = np.asarray(obj["mask"], dtype=bool)
    dist = ndi.distance_transform_edt(mask)

    return dist, tuple(int(v) for v in obj["bbox"])


def add_border_flags_to_pixel_df(
    pixel_df: pd.DataFrame,
    object_db: dict,
    border_width: int = 2,
    object_id_col: str = "object_id",
    row_col: str = "row",
    col_col: str = "col",
) -> pd.DataFrame:
    """
    Add distance_to_border, is_border_pixel and is_core_pixel to a pixel table.

    Core definition:
        distance_to_border > border_width
    """
    if border_width < 0:
        raise ValueError("border_width must be >= 0.")

    df = pixel_df.copy()
    df["distance_to_border"] = np.nan

    for object_id, idx in df.groupby(object_id_col).groups.items():
        obj = object_db.get(str(object_id))

        if obj is None:
            continue

        dist, bbox = _object_distance_map(obj)
        min_row, min_col, max_row, max_col = bbox

        rows = df.loc[idx, row_col].astype(int).to_numpy()
        cols = df.loc[idx, col_col].astype(int).to_numpy()

        rr = rows - min_row
        cc = cols - min_col

        inside = (
            (rr >= 0)
            & (rr < dist.shape[0])
            & (cc >= 0)
            & (cc < dist.shape[1])
        )

        vals = np.full(len(idx), np.nan, dtype=float)
        vals[inside] = dist[rr[inside], cc[inside]]

        df.loc[idx, "distance_to_border"] = vals

    df["is_border_pixel"] = df["distance_to_border"] <= float(border_width)
    df["is_core_pixel"] = df["distance_to_border"] > float(border_width)

    if border_width == 0:
        df["is_border_pixel"] = False
        df["is_core_pixel"] = True

    return df


def aggregate_pixel_predictions_to_objects_core(
    pixel_df: pd.DataFrame,
    object_db: dict,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    object_threshold: float = 0.75,
    truth_threshold: float = 0.50,
    border_width: int = 2,
    min_core_pixels: int = 20,
    fallback_to_all_pixels: bool = True,
    pred_col: str | None = None,
    true_pixel_col: str | None = None,
    truth_available_col: str = "truth_available",
    object_id_col: str = "object_id",
    source_col: str = "source_image",
) -> pd.DataFrame:
    """
    Aggregate pixel predictions into object decisions after excluding border pixels.

    Decision rule
    -------------
    predicted_target_object =
        mean(predicted_target_pixel on core pixels) >= object_threshold
    """
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "pixel")
    if true_pixel_col is None:
        true_pixel_col = make_true_col(target_class, "pixel")
    if pred_col not in pixel_df.columns:
        raise ValueError(f"pixel_df must contain {pred_col!r}.")

    df = add_border_flags_to_pixel_df(
        pixel_df=pixel_df,
        object_db=object_db,
        border_width=border_width,
        object_id_col=object_id_col,
    )

    df[pred_col] = df[pred_col].astype(bool)

    parts = []

    for (object_id, source_image), group in df.groupby(
        [object_id_col, source_col],
        sort=False,
    ):
        core = group[group["is_core_pixel"]].copy()

        use_core = True
        decision_group = core

        if len(core) < int(min_core_pixels):
            if fallback_to_all_pixels:
                decision_group = group.copy()
                use_core = False
            else:
                decision_group = core.copy()

        n_total = int(len(group))
        n_core = int(group["is_core_pixel"].sum())
        n_border = int(group["is_border_pixel"].sum())
        n_decision = int(len(decision_group))

        if n_decision > 0:
            target_ratio = float(decision_group[pred_col].mean())
            n_pred = int(decision_group[pred_col].sum())
        else:
            target_ratio = np.nan
            n_pred = 0

        pred_object = bool(target_ratio >= object_threshold) if np.isfinite(target_ratio) else False

        row = {
            object_id_col: object_id,
            source_col: source_image,
            "n_pixels_total": n_total,
            "n_pixels_core": n_core,
            "n_pixels_border": n_border,
            "n_pixels_decision": n_decision,
            "decision_used_core": bool(use_core),
            "border_width": int(border_width),
            "min_core_pixels": int(min_core_pixels),
            n_predicted_pixels_col(target_class): n_pred,
            pixel_ratio_col(target_class): target_ratio,
            make_predicted_col(target_class, "object"): pred_object,
            "predicted_label_object": target_class if pred_object else non_target_label,
            "object_threshold": float(object_threshold),
        }

        if true_pixel_col in group.columns:
            if truth_available_col in group.columns:
                truth_group = group[group[truth_available_col].astype(bool)].copy()
            else:
                truth_group = group

            if len(truth_group) > 0:
                true_ratio_total = float(truth_group[true_pixel_col].astype(bool).mean())
                true_object = bool(true_ratio_total >= truth_threshold)
            else:
                true_ratio_total = np.nan
                true_object = np.nan

            row[true_pixel_ratio_total_col(target_class)] = true_ratio_total
            row[make_true_col(target_class, "object")] = true_object
            row["true_label_object"] = (
                target_class
                if true_object is True
                else non_target_label
                if true_object is False
                else np.nan
            )

        parts.append(row)

    out = pd.DataFrame(parts)
    if out.empty:
        return out

    out = add_object_metadata(
        out,
        object_db=object_db,
        object_id_col=object_id_col,
    )

    return out


def border_width_object_threshold_grid(
    pixel_df: pd.DataFrame,
    object_db: dict,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    border_widths=(0, 1, 2, 3, 4),
    object_thresholds=(0.60, 0.70, 0.75, 0.80, 0.85, 0.90),
    min_core_pixels: int = 20,
    fallback_to_all_pixels: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate combinations of border_width and object_threshold."""
    rows = []
    tables = {}

    true_col = make_true_col(target_class, "object")
    pred_col = make_predicted_col(target_class, "object")

    for bw in border_widths:
        for thr in object_thresholds:
            obj_df = aggregate_pixel_predictions_to_objects_core(
                pixel_df=pixel_df,
                object_db=object_db,
                target_class=target_class,
                non_target_label=non_target_label,
                object_threshold=float(thr),
                border_width=int(bw),
                min_core_pixels=int(min_core_pixels),
                fallback_to_all_pixels=fallback_to_all_pixels,
            )

            key = (int(bw), float(thr))
            tables[key] = obj_df

            if true_col in obj_df.columns:
                metrics = binary_detection_metrics(
                    obj_df,
                    true_col=true_col,
                    pred_col=pred_col,
                    target_class=target_class,
                    non_target_class=non_target_label,
                )

                metrics.update({
                    "border_width": int(bw),
                    "object_threshold": float(thr),
                    "min_core_pixels": int(min_core_pixels),
                    "mean_core_fraction": float(
                        (obj_df["n_pixels_core"] / obj_df["n_pixels_total"]).mean()
                    ),
                    "fallback_object_rate": float(
                        (~obj_df["decision_used_core"]).mean()
                    ),
                })

                rows.append(metrics)

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        summary = summary.sort_values(
            ["balanced_accuracy", "target_sensitivity", "non_target_specificity"],
            ascending=False,
        ).reset_index(drop=True)

    return summary, tables


def summarize_pixel_errors_by_border_zone(
    pixel_df: pd.DataFrame,
    object_db: dict,
    target_class: str = DEFAULT_TARGET_CLASS,
    border_width: int = 2,
    pred_col: str | None = None,
    true_col: str | None = None,
) -> pd.DataFrame:
    """Summarize TP/TN/FP/FN separately for border and core pixels."""
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "pixel")
    if true_col is None:
        true_col = make_true_col(target_class, "pixel")

    df = add_border_flags_to_pixel_df(
        pixel_df=pixel_df,
        object_db=object_db,
        border_width=border_width,
    )

    rows = []

    for zone_name, zone_mask in {
        "border": df["is_border_pixel"],
        "core": df["is_core_pixel"],
    }.items():
        g = df[zone_mask].copy()

        if "truth_available" in g.columns:
            g = g[g["truth_available"].astype(bool)]

        if len(g) == 0:
            continue

        y_true = g[true_col].astype(bool).to_numpy()
        y_pred = g[pred_col].astype(bool).to_numpy()

        tp = int(np.sum(y_true & y_pred))
        tn = int(np.sum((~y_true) & (~y_pred)))
        fp = int(np.sum((~y_true) & y_pred))
        fn = int(np.sum(y_true & (~y_pred)))

        rows.append({
            "zone": zone_name,
            "border_width": int(border_width),
            "n_pixels": int(len(g)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "fp_rate": fp / max(int(np.sum(~y_true)), 1),
            "fn_rate": fn / max(int(np.sum(y_true)), 1),
            "pixel_accuracy": (tp + tn) / max(len(g), 1),
        })

    return pd.DataFrame(rows)


def summarize_border_diagnostics_by_config(
    pixel_df: pd.DataFrame,
    object_db: dict,
    target_class: str = DEFAULT_TARGET_CLASS,
    border_widths=(1, 2, 3),
    config_cols=("selected_config_id",),
) -> pd.DataFrame:
    """
    Summarize border/core pixel errors for several configs and border widths.

    This wraps summarize_pixel_errors_by_border_zone while preserving
    configuration metadata.
    """
    if pixel_df is None or len(pixel_df) == 0:
        return pd.DataFrame()

    if isinstance(config_cols, str):
        config_cols = [config_cols]
    else:
        config_cols = list(config_cols)

    config_cols = [col for col in config_cols if col in pixel_df.columns]

    if not config_cols:
        config_cols = ["selected_config_id"] if "selected_config_id" in pixel_df.columns else []

    rows = []

    if config_cols:
        grouped = pixel_df.groupby(config_cols, dropna=False)
    else:
        grouped = [((), pixel_df)]

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        meta = {
            col: value
            for col, value in zip(config_cols, key)
        }

        for border_width in border_widths:
            summary = summarize_pixel_errors_by_border_zone(
                pixel_df=group,
                object_db=object_db,
                target_class=target_class,
                border_width=int(border_width),
            )

            if summary is None or len(summary) == 0:
                continue

            summary = summary.copy()

            for col, value in meta.items():
                summary[col] = value

            summary["n_errors"] = summary["fp"].astype(int) + summary["fn"].astype(int)
            summary["error_rate"] = summary["n_errors"] / summary["n_pixels"].clip(lower=1)

            rows.append(summary)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)

    ordered_cols = (
        config_cols
        + [
            "border_width",
            "zone",
            "n_pixels",
            "tp",
            "tn",
            "fp",
            "fn",
            "n_errors",
            "error_rate",
            "fp_rate",
            "fn_rate",
            "pixel_accuracy",
        ]
    )

    ordered_cols = [col for col in ordered_cols if col in out.columns]
    other_cols = [col for col in out.columns if col not in ordered_cols]

    return out[ordered_cols + other_cols].sort_values(
        config_cols + ["border_width", "zone"] if config_cols else ["border_width", "zone"]
    ).reset_index(drop=True)
