from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.metrics import binary_detection_metrics


def add_object_metadata(
    object_df: pd.DataFrame,
    object_db: dict,
    object_id_col: str = "object_id",
) -> pd.DataFrame:
    """Attach object metadata to an object-level dataframe."""
    if object_df.empty:
        return object_df.copy()
    df = object_df.copy()

    rows = []
    for obj_id in df[object_id_col].astype(str):
        obj = object_db.get(str(obj_id), {})
        centroid = obj.get("centroid", (np.nan, np.nan))
        if centroid is None:
            centroid = (np.nan, np.nan)
        try:
            centroid_row = centroid[0]
            centroid_col = centroid[1]
        except Exception:
            centroid_row = np.nan
            centroid_col = np.nan

        rows.append({
            "area_pixels": obj.get("area_pixels", np.nan),
            "batch": obj.get("batch", None),
            "sample_kind": obj.get("sample_kind", None),
            "object_nut_type": obj.get("object_nut_type", None),
            "centroid_row": centroid_row,
            "centroid_col": centroid_col,
        })

    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(rows)],
        axis=1,
    )


def aggregate_pixel_predictions_to_objects(
    pixel_df: pd.DataFrame,
    object_db: dict | None = None,
    target_class: str = "peanut",
    object_threshold: float = 0.75,
    truth_threshold: float = 0.50,
    min_truth_available_ratio: float = 0.50,
    pred_col: str | None = None,
    true_pixel_col: str | None = None,
    truth_available_col: str = "truth_available",
    object_id_col: str = "object_id",
    source_col: str = "source_image",
) -> pd.DataFrame:
    """
    Aggregate pixel predictions into object decisions.

    Decision rule
    -------------
    predicted_target_object =
        mean(predicted_target_pixel over object pixels) >= object_threshold
    """
    if pred_col is None:
        pred_col = f"predicted_{target_class}_pixel"

    if true_pixel_col is None:
        true_pixel_col = f"true_{target_class}_pixel"

    if pred_col not in pixel_df.columns:
        raise ValueError(f"pixel_df must contain {pred_col!r}.")

    df = pixel_df.copy()
    df[pred_col] = df[pred_col].astype(bool)

    agg_dict = {
        "n_pixels_projected": (pred_col, "size"),
        f"n_predicted_{target_class}_pixels": (pred_col, "sum"),
        f"{target_class}_pixel_ratio": (pred_col, "mean"),
    }

    # Optional SIMCA diagnostic columns.
    optional_mean_cols = [
        "H",
        "Q",
        "rule_statistic",
        "rule_limit",
        "distance_to_border",
    ]

    for col in optional_mean_cols:
        if col in df.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")

    if true_pixel_col in df.columns:
        if truth_available_col in df.columns:
            df["_truth_for_ratio"] = np.where(
                df[truth_available_col].astype(bool),
                df[true_pixel_col].astype(bool),
                np.nan,
            )
            agg_dict[f"true_{target_class}_pixel_ratio"] = ("_truth_for_ratio", "mean")
            agg_dict["truth_available_ratio"] = (truth_available_col, "mean")
        else:
            agg_dict[f"true_{target_class}_pixel_ratio"] = (true_pixel_col, "mean")

    out = df.groupby(
        [object_id_col, source_col],
        as_index=False,
    ).agg(**agg_dict)

    ratio_col = f"{target_class}_pixel_ratio"
    pred_object_col = f"predicted_{target_class}_object"

    out[pred_object_col] = out[ratio_col] >= float(object_threshold)
    out["predicted_label_object"] = np.where(
        out[pred_object_col],
        target_class,
        f"non_{target_class}",
    )
    out["object_threshold"] = float(object_threshold)

    true_ratio_col = f"true_{target_class}_pixel_ratio"
    true_object_col = f"true_{target_class}_object"

    if true_ratio_col in out.columns:
        if "truth_available_ratio" in out.columns:
            valid_truth = out["truth_available_ratio"] >= float(min_truth_available_ratio)
        else:
            valid_truth = np.ones(len(out), dtype=bool)
        valid_truth = pd.Series(valid_truth, index=out.index).fillna(False).astype(bool)

        out[true_object_col] = pd.Series(pd.NA, index=out.index, dtype="object")
        truth_values = out.loc[valid_truth, true_ratio_col].to_numpy(dtype=float)>= float(truth_threshold)
        out.loc[valid_truth, true_object_col] = truth_values.tolist()
        out["true_label_object"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out.loc[out[true_object_col] == True, "true_label_object"] = target_class
        out.loc[out[true_object_col] == False, "true_label_object"] = f"non_{target_class}"

    if object_db is not None:
        out = add_object_metadata(
            out,
            object_db=object_db,
            object_id_col=object_id_col,
        )

    return out


def object_threshold_grid(
    pixel_df: pd.DataFrame,
    object_db: dict | None = None,
    target_class: str = "peanut",
    thresholds=(0.30, 0.50, 0.70, 0.80, 0.90),
    truth_threshold: float = 0.50,
    min_truth_available_ratio: float = 0.50,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate several object thresholds."""
    rows = []
    object_tables = {}

    true_col = f"true_{target_class}_object"
    pred_col = f"predicted_{target_class}_object"

    for thr in thresholds:
        obj_df = aggregate_pixel_predictions_to_objects(
            pixel_df=pixel_df,
            object_db=object_db,
            target_class=target_class,
            object_threshold=float(thr),
            truth_threshold=truth_threshold,
            min_truth_available_ratio=min_truth_available_ratio,
        )

        object_tables[float(thr)] = obj_df

        if true_col in obj_df.columns:
            metrics = binary_detection_metrics(
                obj_df,
                true_col=true_col,
                pred_col=pred_col,
            )
            metrics["object_threshold"] = float(thr)
            rows.append(metrics)

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        summary = summary.sort_values(
            ["balanced_accuracy", "target_sensitivity", "non_target_specificity"],
            ascending=False,
        ).reset_index(drop=True)

    return summary, object_tables