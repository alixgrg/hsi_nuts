# src/border_decision.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


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
    Add distance-to-border, is_border_pixel and is_core_pixel to a pixel table.

    Definition
    ----------
    For each object mask, distance_to_border is the Euclidean distance, in pixels,
    from each object pixel to the nearest background pixel. A pixel is considered
    core if distance_to_border > border_width.

    Examples
    --------
    border_width=0 keeps all object pixels.
    border_width=1 removes the outer one-pixel rim.
    border_width=2 removes a slightly thicker rim.
    """
    df = pixel_df.copy()
    df["distance_to_border"] = np.nan

    if border_width < 0:
        raise ValueError("border_width must be >= 0.")

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
            (rr >= 0) & (rr < dist.shape[0])
            & (cc >= 0) & (cc < dist.shape[1])
        )
        vals = np.full(len(idx), np.nan, dtype=float)
        vals[inside] = dist[rr[inside], cc[inside]]
        df.loc[idx, "distance_to_border"] = vals

    df["is_border_pixel"] = df["distance_to_border"] <= float(border_width)
    df["is_core_pixel"] = df["distance_to_border"] > float(border_width)

    # If border_width=0, every detected object pixel should be kept.
    if border_width == 0:
        df["is_border_pixel"] = False
        df["is_core_pixel"] = True

    return df


def binary_detection_metrics(
    df: pd.DataFrame,
    true_col: str = "true_peanut_object",
    pred_col: str = "predicted_peanut_object",
) -> dict:
    """Compute binary object-level peanut detection metrics."""
    d = df.dropna(subset=[true_col, pred_col]).copy()
    if len(d) == 0:
        return {
            "n": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "tn": 0,
            "peanut_sensitivity": np.nan,
            "almond_specificity": np.nan,
            "balanced_accuracy": np.nan,
        }

    y_true = d[true_col].astype(bool).to_numpy()
    y_pred = d[pred_col].astype(bool).to_numpy()

    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    ba = 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan

    return {
        "n": int(len(d)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "peanut_sensitivity": sens,
        "almond_specificity": spec,
        "balanced_accuracy": ba,
    }


def aggregate_pixel_predictions_to_objects_core(
    pixel_df: pd.DataFrame,
    object_db: dict,
    object_threshold: float = 0.75,
    truth_threshold: float = 0.50,
    border_width: int = 2,
    min_core_pixels: int = 20,
    fallback_to_all_pixels: bool = True,
    pred_col: str = "predicted_peanut_pixel",
    true_pixel_col: str = "true_peanut_pixel",
    object_id_col: str = "object_id",
    source_col: str = "source_image",
) -> pd.DataFrame:
    """
    Aggregate pixel predictions into object decisions after excluding border pixels.

    Decision rule
    -------------
    predicted_peanut_object = mean(predicted_peanut_pixel on core pixels) >= object_threshold

    If an object has fewer than min_core_pixels core pixels, fallback_to_all_pixels=True
    uses all its pixels to avoid unstable decisions on very small objects.
    """
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
    for (object_id, source_image), group in df.groupby([object_id_col, source_col], sort=False):
        core = group[group["is_core_pixel"]].copy()
        use_core = True
        decision_group = core

        if len(core) < int(min_core_pixels):
            if fallback_to_all_pixels:
                decision_group = group.copy()
                use_core = False
            else:
                # Keep the object but mark the ratio as NaN.
                decision_group = core.copy()

        n_total = int(len(group))
        n_core = int(group["is_core_pixel"].sum())
        n_border = int(group["is_border_pixel"].sum())
        n_decision = int(len(decision_group))

        if n_decision > 0:
            peanut_ratio = float(decision_group[pred_col].mean())
            n_pred = int(decision_group[pred_col].sum())
        else:
            peanut_ratio = np.nan
            n_pred = 0

        row = {
            "object_id": object_id,
            "source_image": source_image,
            "n_pixels_total": n_total,
            "n_pixels_core": n_core,
            "n_pixels_border": n_border,
            "n_pixels_decision": n_decision,
            "decision_used_core": bool(use_core),
            "border_width": int(border_width),
            "min_core_pixels": int(min_core_pixels),
            "n_predicted_peanut_pixels": n_pred,
            "peanut_pixel_ratio": peanut_ratio,
            "predicted_peanut_object": bool(peanut_ratio >= object_threshold) if np.isfinite(peanut_ratio) else False,
            "predicted_label_object": "peanut" if np.isfinite(peanut_ratio) and peanut_ratio >= object_threshold else "non_peanut",
            "object_threshold": float(object_threshold),
        }

        if true_pixel_col in group.columns:
            if n_decision > 0:
                true_ratio_decision = float(decision_group[true_pixel_col].astype(bool).mean())
            else:
                true_ratio_decision = np.nan
            true_ratio_total = float(group[true_pixel_col].astype(bool).mean())
            row["true_peanut_pixel_ratio_decision"] = true_ratio_decision
            row["true_peanut_pixel_ratio_total"] = true_ratio_total
            # For truth, total object ratio is generally safer, because border exclusion is a decision rule,
            # not a change in ground truth.
            row["true_peanut_object"] = bool(true_ratio_total >= truth_threshold)
            row["true_label_object"] = "peanut" if row["true_peanut_object"] else "non_peanut"

        obj = object_db.get(str(object_id), {})
        centroid = obj.get("centroid", (np.nan, np.nan))
        row.update({
            "area_pixels": obj.get("area_pixels", np.nan),
            "batch": obj.get("batch", None),
            "sample_kind": obj.get("sample_kind", None),
            "object_nut_type": obj.get("object_nut_type", None),
            "centroid_row": centroid[0] if len(centroid) > 0 else np.nan,
            "centroid_col": centroid[1] if len(centroid) > 1 else np.nan,
        })
        parts.append(row)

    return pd.DataFrame(parts)


def border_width_object_threshold_grid(
    pixel_df: pd.DataFrame,
    object_db: dict,
    border_widths=(0, 1, 2, 3, 4),
    object_thresholds=(0.60, 0.70, 0.75, 0.80, 0.85, 0.90),
    min_core_pixels: int = 20,
    fallback_to_all_pixels: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate combinations of border_width and object_threshold."""
    rows = []
    tables = {}

    for bw in border_widths:
        for thr in object_thresholds:
            obj_df = aggregate_pixel_predictions_to_objects_core(
                pixel_df=pixel_df,
                object_db=object_db,
                object_threshold=float(thr),
                border_width=int(bw),
                min_core_pixels=int(min_core_pixels),
                fallback_to_all_pixels=fallback_to_all_pixels,
            )
            key = (int(bw), float(thr))
            tables[key] = obj_df

            if "true_peanut_object" in obj_df.columns:
                metrics = binary_detection_metrics(obj_df)
                metrics.update({
                    "border_width": int(bw),
                    "object_threshold": float(thr),
                    "min_core_pixels": int(min_core_pixels),
                    "mean_core_fraction": float((obj_df["n_pixels_core"] / obj_df["n_pixels_total"]).mean()),
                    "fallback_object_rate": float((~obj_df["decision_used_core"]).mean()),
                })
                rows.append(metrics)

    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        summary = summary.sort_values(
            ["balanced_accuracy", "peanut_sensitivity", "almond_specificity"],
            ascending=False,
        ).reset_index(drop=True)
    return summary, tables


def summarize_pixel_errors_by_border_zone(
    pixel_df: pd.DataFrame,
    object_db: dict,
    border_width: int = 2,
    pred_col: str = "predicted_peanut_pixel",
    true_col: str = "true_peanut_pixel",
) -> pd.DataFrame:
    """Summarize TP/TN/FP/FN separately for border and core pixels."""
    df = add_border_flags_to_pixel_df(pixel_df, object_db, border_width=border_width)
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
