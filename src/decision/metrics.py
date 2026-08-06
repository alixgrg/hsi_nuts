from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    DEFAULT_NON_TARGET_LABEL,
    predicted_col as make_predicted_col,
    true_col as make_true_col,
)


def apply_locked_margin_decision(
    margin,
    decision_mode,
    *,
    direct_2way_threshold=0.0,
    three_way_lower_threshold=np.nan,
    three_way_upper_threshold=np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply prespecified 2-way/3-way thresholds to SIMCA margins.

    Every argument is broadcast with NumPy.  The lower 3-way boundary belongs
    to the uncertain interval and the upper boundary belongs to the target
    interval, matching the frozen 03B/03C convention.
    """
    arrays = np.broadcast_arrays(
        np.asarray(margin, dtype=float),
        np.asarray(decision_mode, dtype=str),
        np.asarray(direct_2way_threshold, dtype=float),
        np.asarray(three_way_lower_threshold, dtype=float),
        np.asarray(three_way_upper_threshold, dtype=float),
    )
    margin_array, mode, direct, lower, upper = arrays
    unknown = ~np.isin(mode, ("2way", "3way"))
    if unknown.any():
        raise ValueError(
            "Unknown decision modes: "
            f"{sorted(set(mode[unknown].ravel().tolist()))}"
        )
    if not np.isfinite(margin_array).all():
        raise ValueError("Every SIMCA margin must be finite.")
    mode_2way = mode == "2way"
    mode_3way = mode == "3way"
    if (mode_2way & ~np.isfinite(direct)).any():
        raise ValueError("Every 2-way threshold must be finite.")
    valid_three_way = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    if (mode_3way & ~valid_three_way).any():
        raise ValueError("Every 3-way threshold must be finite with lower < upper.")
    uncertain = mode_3way & (margin_array >= lower) & (margin_array < upper)
    target = np.where(mode_3way, margin_array >= upper, margin_array >= direct)
    if (target & uncertain).any():
        raise RuntimeError("A decision cannot be both target and uncertain.")
    return np.asarray(target, dtype=bool), np.asarray(uncertain, dtype=bool)


def binary_mask_agreement(mask_a, mask_b, roi=None) -> dict:
    """Compute independent pixel agreement constraints inside an ROI."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(
            f"Masks must be aligned 2D arrays, got {a.shape} and {b.shape}."
        )
    valid = (
        np.ones(a.shape, dtype=bool)
        if roi is None
        else np.asarray(roi, dtype=bool)
    )
    if valid.shape != a.shape:
        raise ValueError("ROI shape does not match mask shape.")
    av = a[valid]
    bv = b[valid]
    if av.size == 0:
        raise ValueError("ROI contains no pixels.")
    intersection = int(np.count_nonzero(av & bv))
    union = int(np.count_nonzero(av | bv))
    positive_sum = int(np.count_nonzero(av) + np.count_nonzero(bv))
    return {
        "n_roi_pixels": int(av.size),
        "pixel_agreement": float(np.mean(av == bv)),
        "dice": (
            float(2 * intersection / positive_sum)
            if positive_sum
            else 1.0
        ),
        "iou": float(intersection / union) if union else 1.0,
    }


def pairwise_component_iou(labels_a, labels_b) -> np.ndarray:
    """Build all component IoUs without looping over pixels."""
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Component label images must be aligned 2D arrays.")
    n_a = int(a.max(initial=0))
    n_b = int(b.max(initial=0))
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=float)
    joint = np.bincount(
        (a.ravel() * (n_b + 1) + b.ravel()),
        minlength=(n_a + 1) * (n_b + 1),
    ).reshape(n_a + 1, n_b + 1)
    intersections = joint[1:, 1:].astype(float)
    area_a = np.bincount(a.ravel(), minlength=n_a + 1)[1:, None]
    area_b = np.bincount(b.ravel(), minlength=n_b + 1)[None, 1:]
    unions = area_a + area_b - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )


def component_agreement(
    mask_a,
    mask_b,
    *,
    connectivity: int = 2,
) -> dict:
    """Match connected components optimally and report unmatched rates."""
    labels_a = label(np.asarray(mask_a, dtype=bool), connectivity=connectivity)
    labels_b = label(np.asarray(mask_b, dtype=bool), connectivity=connectivity)
    pairwise = pairwise_component_iou(labels_a, labels_b)
    n_a, n_b = pairwise.shape
    if n_a and n_b:
        row_ind, col_ind = linear_sum_assignment(-pairwise)
        matched_ious = pairwise[row_ind, col_ind]
        positive = matched_ious > 0
        matched_count = int(np.count_nonzero(positive))
        mean_iou = (
            float(matched_ious[positive].mean())
            if matched_count
            else 0.0
        )
    else:
        matched_count = 0
        mean_iou = 1.0 if n_a == n_b == 0 else 0.0
    unmatched = (n_a - matched_count) + (n_b - matched_count)
    denominator = n_a + n_b
    return {
        "n_components_a": int(n_a),
        "n_components_b": int(n_b),
        "n_components_matched": matched_count,
        "mean_matched_component_iou": mean_iou,
        "unmatched_component_rate": (
            float(unmatched / denominator) if denominator else 0.0
        ),
    }


def _component_detection_analysis(
    truth_mask,
    prediction_mask,
    *,
    valid_mask=None,
    connectivity: int = 2,
    truth_component_labels=None,
    prediction_component_labels=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label aligned masks once and return their pairwise IoU matrix."""
    truth = np.asarray(truth_mask, dtype=bool)
    prediction = np.asarray(prediction_mask, dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Component masks must be aligned 2D arrays.")
    valid = (
        np.ones(truth.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != truth.shape:
        raise ValueError("valid_mask is not aligned with component masks.")
    def canonical_labels(values, foreground) -> np.ndarray:
        source = np.asarray(values)
        if source.shape != truth.shape:
            raise ValueError("Explicit component labels are not aligned.")
        if not np.issubdtype(source.dtype, np.integer):
            numeric = source.astype(float)
            if not np.isfinite(numeric).all() or not np.equal(
                numeric, np.floor(numeric)
            ).all():
                raise ValueError("Component labels must be finite integers.")
            source = numeric.astype(np.int64)
        source = source.astype(np.int64, copy=False)
        if (source < 0).any():
            raise ValueError("Component labels must be non-negative.")
        source = np.where(foreground & valid, source, 0)
        unique = np.unique(source)
        positive = unique[unique > 0]
        if not len(positive):
            return np.zeros(source.shape, dtype=np.int64)
        positions = np.searchsorted(positive, source)
        return np.where(source > 0, positions + 1, 0).astype(np.int64)

    truth_labels = (
        label(truth & valid, connectivity=int(connectivity))
        if truth_component_labels is None
        else canonical_labels(truth_component_labels, truth)
    )
    prediction_labels = (
        label(prediction & valid, connectivity=int(connectivity))
        if prediction_component_labels is None
        else canonical_labels(prediction_component_labels, prediction)
    )
    pairwise = pairwise_component_iou(truth_labels, prediction_labels)
    return truth_labels, prediction_labels, pairwise


def _fragment_table_from_analysis(
    truth_labels: np.ndarray,
    pairwise: np.ndarray,
    *,
    area_upper_bounds,
    area_labels,
    min_iou: float = 0.0,
) -> pd.DataFrame:
    n_truth = int(truth_labels.max(initial=0))
    if n_truth == 0:
        return pd.DataFrame(
            columns=[
                "fragment_id",
                "area_pixels",
                "area_class",
                "detected",
                "best_iou",
            ]
        )
    areas = np.bincount(truth_labels.ravel(), minlength=n_truth + 1)[1:]
    bounds = np.asarray(tuple(area_upper_bounds), dtype=float)
    labels_array = np.asarray(tuple(area_labels), dtype=object)
    if len(labels_array) != len(bounds) + 1:
        raise ValueError("area_labels must have len(area_upper_bounds) + 1.")
    classes = labels_array[np.searchsorted(bounds, areas, side="left")]
    best_iou = (
        pairwise.max(axis=1)
        if pairwise.shape[1]
        else np.zeros(n_truth, dtype=float)
    )
    return pd.DataFrame(
        {
            "fragment_id": np.arange(1, n_truth + 1, dtype=int),
            "area_pixels": areas.astype(int),
            "area_class": classes,
            "detected": best_iou > float(min_iou),
            "best_iou": best_iou,
        }
    )


def component_detection_metrics(
    truth_mask,
    prediction_mask,
    *,
    valid_mask=None,
    connectivity: int = 2,
    return_fragment_table: bool = False,
    area_upper_bounds=None,
    area_labels=None,
    truth_component_labels=None,
    prediction_component_labels=None,
    min_iou: float = 0.0,
) -> dict | tuple[dict, pd.DataFrame]:
    """Compute component metrics, optionally reusing labels for fragments."""
    truth_labels, prediction_labels, pairwise = _component_detection_analysis(
        truth_mask,
        prediction_mask,
        valid_mask=valid_mask,
        connectivity=connectivity,
        truth_component_labels=truth_component_labels,
        prediction_component_labels=prediction_component_labels,
    )
    overlap = pairwise > float(min_iou)
    n_truth, n_prediction = pairwise.shape
    detected_truth = overlap.any(axis=1) if n_truth else np.zeros(0, dtype=bool)
    matched_prediction = (
        overlap.any(axis=0) if n_prediction else np.zeros(0, dtype=bool)
    )
    split_truth = (
        overlap.sum(axis=1) > 1 if n_truth else np.zeros(0, dtype=bool)
    )
    merged_prediction = (
        overlap.sum(axis=0) > 1
        if n_prediction
        else np.zeros(0, dtype=bool)
    )
    metrics = {
        "n_truth_components": int(n_truth),
        "n_predicted_components": int(n_prediction),
        "n_detected_truth_components": int(detected_truth.sum()),
        "n_matched_predicted_components": int(matched_prediction.sum()),
        "n_split_truth_components": int(split_truth.sum()),
        "n_merged_predicted_components": int(merged_prediction.sum()),
        "component_recall": (
            float(detected_truth.mean()) if n_truth else np.nan
        ),
        "component_precision": (
            float(matched_prediction.mean()) if n_prediction else np.nan
        ),
        "split_rate": float(split_truth.mean()) if n_truth else 0.0,
        "merge_rate": (
            float(merged_prediction.mean()) if n_prediction else 0.0
        ),
    }
    if not return_fragment_table:
        return metrics
    if area_upper_bounds is None or area_labels is None:
        raise ValueError(
            "area_upper_bounds and area_labels are required for fragments."
        )
    return metrics, _fragment_table_from_analysis(
        truth_labels,
        pairwise,
        area_upper_bounds=area_upper_bounds,
        area_labels=area_labels,
        min_iou=min_iou,
    )


def fragment_detection_table(
    truth_mask,
    prediction_mask,
    *,
    area_upper_bounds,
    area_labels,
    valid_mask=None,
    connectivity: int = 2,
    truth_component_labels=None,
    prediction_component_labels=None,
) -> pd.DataFrame:
    """Return one vectorized detection record per true fragment."""
    truth_labels, _, pairwise = _component_detection_analysis(
        truth_mask,
        prediction_mask,
        valid_mask=valid_mask,
        connectivity=connectivity,
        truth_component_labels=truth_component_labels,
        prediction_component_labels=prediction_component_labels,
    )
    return _fragment_table_from_analysis(
        truth_labels,
        pairwise,
        area_upper_bounds=area_upper_bounds,
        area_labels=area_labels,
    )


def component_detection_table(
    truth_mask,
    prediction_mask,
    *,
    valid_mask=None,
    connectivity: int = 2,
    truth_component_labels=None,
    prediction_component_labels=None,
    area_upper_bounds=(),
    area_labels=("all",),
    min_iou: float = 0.0,
) -> pd.DataFrame:
    """Return compact truth/prediction component geometry and best matches."""
    truth_labels, prediction_labels, pairwise = _component_detection_analysis(
        truth_mask,
        prediction_mask,
        valid_mask=valid_mask,
        connectivity=connectivity,
        truth_component_labels=truth_component_labels,
        prediction_component_labels=prediction_component_labels,
    )
    bounds = np.asarray(tuple(area_upper_bounds), dtype=float)
    class_labels = np.asarray(tuple(area_labels), dtype=object)
    if len(class_labels) != len(bounds) + 1:
        raise ValueError("area_labels must have len(area_upper_bounds) + 1.")

    def geometry(labels_image, role: str, iou_matrix: np.ndarray) -> pd.DataFrame:
        n_components = int(labels_image.max(initial=0))
        if n_components == 0:
            return pd.DataFrame()
        flat = labels_image.ravel()
        component_ids = np.arange(1, n_components + 1, dtype=int)
        areas = np.bincount(flat, minlength=n_components + 1)[1:]
        row_grid, col_grid = np.indices(labels_image.shape)
        row_sums = np.bincount(
            flat, weights=row_grid.ravel(), minlength=n_components + 1
        )[1:]
        col_sums = np.bincount(
            flat, weights=col_grid.ravel(), minlength=n_components + 1
        )[1:]
        slices = ndi.find_objects(labels_image, max_label=n_components)
        bboxes = np.asarray(
            [
                (
                    section[0].start,
                    section[1].start,
                    section[0].stop,
                    section[1].stop,
                )
                if section is not None
                else (0, 0, 0, 0)
                for section in slices
            ],
            dtype=int,
        )
        if iou_matrix.shape[1]:
            best_index = iou_matrix.argmax(axis=1)
            best_iou = iou_matrix[np.arange(n_components), best_index]
            best_match = np.where(best_iou > float(min_iou), best_index + 1, 0)
            overlap_count = (iou_matrix > float(min_iou)).sum(axis=1)
        else:
            best_match = np.zeros(n_components, dtype=int)
            best_iou = np.zeros(n_components, dtype=float)
            overlap_count = np.zeros(n_components, dtype=int)
        return pd.DataFrame(
            {
                "component_role": role,
                "component_id": component_ids,
                "area_pixels": areas.astype(int),
                "area_class": class_labels[
                    np.searchsorted(bounds, areas, side="left")
                ],
                "centroid_row": row_sums / areas,
                "centroid_col": col_sums / areas,
                "bbox_min_row": bboxes[:, 0],
                "bbox_min_col": bboxes[:, 1],
                "bbox_max_row": bboxes[:, 2],
                "bbox_max_col": bboxes[:, 3],
                "best_match_component_id": best_match.astype(int),
                "best_iou": best_iou.astype(float),
                "overlap_count": overlap_count.astype(int),
                "detected_or_matched": best_match > 0,
                "split_or_merge": overlap_count > 1,
            }
        )

    truth_table = geometry(truth_labels, "truth", pairwise)
    predicted_table = geometry(
        prediction_labels,
        "predicted",
        pairwise.T,
    )
    return pd.concat(
        [truth_table, predicted_table], ignore_index=True, sort=False
    )

def binary_detection_metrics(
    df: pd.DataFrame,
    true_col: str | None = None,
    pred_col: str | None = None,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> dict:
    """
    Compute binary detection metrics.

    Positive class = True = target class
    Negative class = False = non-target class
    """
    if true_col is None:
        true_col = make_true_col(target_class, "object")
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "object")

    d = df.dropna(subset=[true_col, pred_col]).copy()
    empty = {
        "target_class": target_class,
        "non_target_class": non_target_class,
        "n": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "target_sensitivity": np.nan,
        "non_target_specificity": np.nan,
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "f1_score": np.nan,
        "fn_rate": np.nan,
        "fp_rate": np.nan,
    }
    if len(d) == 0:
        return empty
    true_bool = coerce_binary_series(d[true_col], target_class=target_class, non_target_class=non_target_class)
    pred_bool = coerce_binary_series(d[pred_col], target_class=target_class, non_target_class=non_target_class)
    valid = true_bool.notna() & pred_bool.notna()
    d = d.loc[valid].copy()
    true_bool = true_bool.loc[valid].astype(bool)
    pred_bool = pred_bool.loc[valid].astype(bool)
    if len(d) == 0:
        return empty
    y_true = true_bool.to_numpy()
    y_pred = pred_bool.to_numpy()
    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    n = int(len(d))

    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    ba = 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan
    acc = (tp + tn) / n if n > 0 else np.nan
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = (
        2.0 * precision * sens / (precision + sens)
        if np.isfinite(precision)
        and np.isfinite(sens)
        and (precision + sens) > 0
        else np.nan
    )
    fn_rate = fn / (tp + fn) if tp + fn > 0 else np.nan
    fp_rate = fp / (fp + tn) if fp + tn > 0 else np.nan

    out = {
        "target_class": target_class,
        "non_target_class": non_target_class,
        "n": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "target_sensitivity": sens,
        "non_target_specificity": spec,
        "balanced_accuracy": ba,
        "accuracy": acc,
        "precision": precision,
        "f1_score": f1,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
    }
    return out


def metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    true_col: str | None = None,
    pred_col: str | None = None,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> pd.DataFrame:
    """Compute binary metrics by group, for example by batch or image."""
    rows = []

    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    for group_value, group_df in df.groupby(group_col, dropna=False):
        row = binary_detection_metrics(
            group_df,
            true_col=true_col,
            pred_col=pred_col,
            target_class=target_class,
            non_target_class=non_target_class,
        )
        row[group_col] = group_value
        rows.append(row)

    return pd.DataFrame(rows)


def add_detection_score(
    metrics_df: pd.DataFrame,
    sensitivity_weight: float = 10.0,
    specificity_weight: float = 1.0,
    score_col: str = "detection_score",
) -> pd.DataFrame:
    """
    Add a scalar score useful for model ranking.

    High score favors target sensitivity first, then non-target specificity.
    """
    df = metrics_df.copy()

    sens = df["target_sensitivity"].astype(float)
    spec = df["non_target_specificity"].astype(float)

    df[score_col] = sensitivity_weight * sens + specificity_weight * spec

    return df


def add_binary_confusion_case(
    df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    level: str = "object",
    true_col: str | None = None,
    pred_col: str | None = None,
    output_col: str | None = None,
    truth_available_col: str = "truth_available",
    truth_available_ratio_col: str = "truth_available_ratio",
    min_truth_available_ratio: float = 0.50,
) -> pd.DataFrame:
    """
    Add TP/TN/FP/FN labels to a pixel-level or object-level dataframe.

    level:
        "pixel" or "object"
    """
    if level not in {"pixel", "object"}:
        raise ValueError("level must be 'pixel' or 'object'.")
    if true_col is None:
        true_col = make_true_col(target_class, level)
    if pred_col is None:
        pred_col = make_predicted_col(target_class, level)
    if output_col is None:
        output_col = f"{level}_error_case"

    required = [true_col, pred_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    out = df.copy()

    valid = out[true_col].notna() & out[pred_col].notna()

    if level == "pixel" and truth_available_col in out.columns:
        valid = valid & coerce_binary_series(out[truth_available_col]).fillna(False).astype(bool)

    if level == "object" and truth_available_ratio_col in out.columns:
        truth_ratio = pd.to_numeric(
            out[truth_available_ratio_col],
            errors="coerce",
        )
        valid = valid & (
            truth_ratio >= float(min_truth_available_ratio)
        )

    truth_nullable = coerce_binary_series(out[true_col], target_class=target_class)
    pred_nullable = coerce_binary_series(out[pred_col], target_class=target_class)
    valid = valid & truth_nullable.notna() & pred_nullable.notna()
    truth = truth_nullable.fillna(False).astype(bool)
    pred = pred_nullable.fillna(False).astype(bool)

    out[output_col] = "unavailable"

    out.loc[valid & truth & pred, output_col] = "TP"
    out.loc[valid & (~truth) & (~pred), output_col] = "TN"
    out.loc[valid & (~truth) & pred, output_col] = "FP"
    out.loc[valid & truth & (~pred), output_col] = "FN"

    out[f"{level}_is_error"] = out[output_col].isin(["FP", "FN"])
    out[f"{level}_is_fp"] = out[output_col].eq("FP")
    out[f"{level}_is_fn"] = out[output_col].eq("FN")

    return out


def _normalise_group_cols(group_cols):
    if group_cols is None:
        return ["source_image"]
    if isinstance(group_cols, str):
        return [group_cols]
    return list(group_cols)


def summarize_pixel_errors_by_image(
    pixel_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    group_cols=("source_image",),
    sort_worst_first: bool = True,
    truth_available_col: str = "truth_available",
) -> pd.DataFrame:
    """
    Summarize pixel-level TP/TN/FP/FN by image, model, or any group columns.

    Examples
    --------
    group_cols=("source_image",)
    group_cols=("selected_config_id", "source_image")
    group_cols=("selected_config_id", "matrix_family", "source_image")
    """
    group_cols = _normalise_group_cols(group_cols)

    true_col = make_true_col(target_class, "pixel")
    pred_col = make_predicted_col(target_class, "pixel")

    required = group_cols + [true_col, pred_col]
    missing = [col for col in required if col not in pixel_df.columns]
    if missing:
        raise KeyError(f"Missing columns in pixel_df: {missing}")

    rows = []

    for key, group in pixel_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        g = group.copy()

        if truth_available_col in g.columns:
            available = coerce_binary_series(g[truth_available_col]).fillna(False).astype(bool)
            g = g[available]
        if len(g) == 0:
            continue

        metrics = binary_detection_metrics(
            g,
            true_col=true_col,
            pred_col=pred_col,
            target_class=target_class,
            non_target_class=non_target_label,
        )

        row = {
            col: value
            for col, value in zip(group_cols, key)
        }
        row.update(metrics)
        row["n_truth_pixels"] = metrics["n"]
        row["pixel_accuracy"] = metrics["accuracy"]
        row["pixel_balanced_accuracy"] = metrics["balanced_accuracy"]
        row["pixel_fn_rate"] = metrics["fn_rate"]
        row["pixel_fp_rate"] = metrics["fp_rate"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    if sort_worst_first:
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["fn_rate", "fp_rate", "balanced_accuracy"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
    else:
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["fn_rate", "fp_rate", "balanced_accuracy"],
                ascending=[True, True, False],
            )
            .reset_index(drop=True)
        )


def summarize_object_errors_by_image(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    group_cols=("source_image",),
    truth_available_ratio_col: str = "truth_available_ratio",
    sort_worst_first: bool = True,
    min_truth_available_ratio: float = 0.50,
) -> pd.DataFrame:
    """
    Summarize object-level TP/TN/FP/FN by image, model, or any group columns.
    """
    group_cols = _normalise_group_cols(group_cols)

    true_col = make_true_col(target_class, "object")
    pred_col = make_predicted_col(target_class, "object")

    required = group_cols + [true_col, pred_col]
    missing = [col for col in required if col not in object_df.columns]
    if missing:
        raise KeyError(f"Missing columns in object_df: {missing}")

    rows = []
    for key, group in object_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        g = group.copy()

        if truth_available_ratio_col in g.columns:
            truth_ratio = pd.to_numeric(
                g[truth_available_ratio_col],
                errors="coerce",
            )
            g = g[
                truth_ratio >= float(min_truth_available_ratio)
            ].copy()

        if len(g) == 0:
            continue

        metrics = binary_detection_metrics(
            g,
            true_col=true_col,
            pred_col=pred_col,
            target_class=target_class,
            non_target_class=non_target_label,
        )

        row = {
            col: value
            for col, value in zip(group_cols, key)
        }
        row.update(metrics)
        row["n_truth_objects"] = metrics["n"]
        row["object_accuracy"] = metrics["accuracy"]
        row["object_balanced_accuracy"] = metrics["balanced_accuracy"]
        row["object_fn_rate"] = metrics["fn_rate"]
        row["object_fp_rate"] = metrics["fp_rate"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    if sort_worst_first:
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["fn_rate", "fp_rate", "balanced_accuracy"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
    else:
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["fn_rate", "fp_rate", "balanced_accuracy"],
                ascending=[True, True, False],
            )
            .reset_index(drop=True)
        )


def coerce_binary_series(
    series: pd.Series,
    *,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> pd.Series:
    """Convert bool/numeric/string labels to pandas nullable booleans.

    Unlike ``astype(bool)``, strings such as ``"False"`` are correctly mapped to
    False instead of being treated as truthy.
    """
    values = series.astype("object")
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")

    is_bool = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    if is_bool.any():
        out.loc[is_bool] = values.loc[is_bool].astype(bool)

    text = values.astype(str).str.strip().str.lower()
    true_tokens = {
        "true", "1", "yes", "y", "positive", "target", "peanut",
        str(target_class).lower(),
    }
    false_tokens = {
        "false", "0", "no", "n", "negative", "non_target", "non-target",
        "almond", "non_peanut", str(non_target_class).lower(),
    }
    out.loc[text.isin(true_tokens)] = True
    out.loc[text.isin(false_tokens)] = False
    return out


def binary_confusion_table(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    group_cols=(),
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    confidence_col: str | None = "binary_confidence",
) -> pd.DataFrame:
    """Build a complete long-format binary confusion table."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    group_cols = [column for column in list(group_cols) if column in df.columns]
    missing = [column for column in (true_col, pred_col) if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for binary confusion table: {missing}")

    d = df[df[true_col].notna() & df[pred_col].notna()].copy()
    true_bool = coerce_binary_series(
        d[true_col], target_class=target_class, non_target_class=non_target_label
    )
    pred_bool = coerce_binary_series(
        d[pred_col], target_class=target_class, non_target_class=non_target_label
    )
    valid = true_bool.notna() & pred_bool.notna()
    d = d.loc[valid].copy()
    true_bool = true_bool.loc[valid].astype(bool)
    pred_bool = pred_bool.loc[valid].astype(bool)

    d["true_label_2way"] = np.where(true_bool, target_class, non_target_label)
    d["predicted_label_2way"] = np.where(pred_bool, target_class, non_target_label)

    rows = []
    grouped = d.groupby(group_cols, dropna=False) if group_cols else [((), d)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        n_group = len(group)
        for true_label in (non_target_label, target_class):
            true_group = group[group["true_label_2way"].eq(true_label)]
            n_true = len(true_group)
            for decision in (non_target_label, target_class):
                cell = true_group[true_group["predicted_label_2way"].eq(decision)]
                row = dict(base)
                row.update(
                    true_label_2way=true_label,
                    predicted_label_2way=decision,
                    n=int(len(cell)),
                    n_true_label=int(n_true),
                    n_group=int(n_group),
                    row_rate=len(cell) / n_true if n_true else np.nan,
                    global_rate=len(cell) / n_group if n_group else np.nan,
                )
                if confidence_col is not None and confidence_col in cell.columns:
                    confidence = pd.to_numeric(
                        cell[confidence_col],
                        errors="coerce",
                    )
                    row["mean_confidence"] = (
                        float(confidence.mean())
                        if confidence.notna().any()
                        else np.nan
                    )
                    row["median_confidence"] = (
                        float(confidence.median())
                        if confidence.notna().any()
                        else np.nan
                    )
                else:
                    row["mean_confidence"] = np.nan
                    row["median_confidence"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def binary_confusion_counts_vectorized(truth, predictions) -> pd.DataFrame:
    """Compute one or many binary confusion matrices without row loops."""
    y_true = np.asarray(truth, dtype=bool).reshape(-1)
    y_pred = np.asarray(predictions, dtype=bool)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_pred.ndim != 2 or y_pred.shape[0] != y_true.size:
        raise ValueError(
            "predictions must have shape (n,) or (n, n_candidates)."
        )
    target = y_true[:, None]
    tp = np.sum(target & y_pred, axis=0)
    fn = np.sum(target & ~y_pred, axis=0)
    fp = np.sum(~target & y_pred, axis=0)
    tn = np.sum(~target & ~y_pred, axis=0)
    return pd.DataFrame(
        {
            "candidate_index": np.arange(y_pred.shape[1], dtype=int),
            "n": y_true.size,
            "tp": tp.astype(int),
            "fn": fn.astype(int),
            "fp": fp.astype(int),
            "tn": tn.astype(int),
        }
    )


def summarize_binary_metrics_vectorized(
    predictions,
    truth=None,
    *,
    truth_col: str = "truth",
    prediction_col: str = "prediction",
    group_levels: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarise binary metrics at micro and requested macro group levels."""
    if isinstance(predictions, pd.DataFrame):
        frame = predictions.copy()
        if truth is None:
            if truth_col not in frame or prediction_col not in frame:
                raise KeyError(
                    f"Missing {truth_col!r} or {prediction_col!r}."
                )
            y_true = frame[truth_col].to_numpy(dtype=bool)
        else:
            y_true = np.asarray(truth, dtype=bool)
        y_pred = frame[prediction_col].to_numpy(dtype=bool)
    else:
        frame = None
        y_pred = np.asarray(predictions, dtype=bool)
        if truth is None:
            raise ValueError("truth is required for array predictions.")
        y_true = np.asarray(truth, dtype=bool)

    def metrics(values_true, values_pred) -> dict[str, float | int]:
        counts = binary_confusion_counts_vectorized(
            values_true, values_pred
        ).iloc[0]
        n_target = int(counts["tp"] + counts["fn"])
        n_non_target = int(counts["tn"] + counts["fp"])
        miss = counts["fn"] / n_target if n_target else np.nan
        false_accept = counts["fp"] / n_non_target if n_non_target else np.nan
        sensitivity = 1.0 - miss if np.isfinite(miss) else np.nan
        specificity = (
            1.0 - false_accept if np.isfinite(false_accept) else np.nan
        )
        return {
            "n": int(counts["n"]),
            "target_miss_rate": float(miss),
            "false_accept_rate": float(false_accept),
            "balanced_accuracy": float(
                np.nanmean([sensitivity, specificity])
            ),
        }

    rows = [{"aggregation_level": "micro", **metrics(y_true, y_pred)}]
    if frame is None:
        return pd.DataFrame(rows)
    for group_col in group_levels:
        if group_col not in frame:
            raise KeyError(f"Missing grouping column: {group_col}")
        grouped_rows = [
            metrics(y_true[index], y_pred[index])
            for index in frame.groupby(group_col, sort=False).indices.values()
        ]
        grouped = pd.DataFrame(grouped_rows)
        rows.append(
            {
                "aggregation_level": f"macro_{group_col}",
                "n": int(len(grouped)),
                "target_miss_rate": float(grouped["target_miss_rate"].mean()),
                "false_accept_rate": float(grouped["false_accept_rate"].mean()),
                "balanced_accuracy": float(grouped["balanced_accuracy"].mean()),
            }
        )
        rows.append(
            {
                "aggregation_level": f"worst_{group_col}",
                "n": int(len(grouped)),
                "target_miss_rate": float(grouped["target_miss_rate"].max()),
                "false_accept_rate": float(grouped["false_accept_rate"].max()),
                "balanced_accuracy": float(grouped["balanced_accuracy"].min()),
            }
        )
    return pd.DataFrame(rows)
