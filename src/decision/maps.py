from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.metrics import coerce_binary_series
from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    predicted_col as make_predicted_col,
    true_col as make_true_col,
)


def make_pixel_error_map(
    image_key: str,
    image_db: dict,
    pixel_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    pred_col: str | None = None,
    true_col: str | None = None,
    truth_available_col: str = "truth_available",
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
) -> np.ndarray:
    """
    Create a pixel error map.

    Codes
    -----
    0 : background / no truth
    1 : TP
    2 : TN
    3 : FP
    4 : FN
    """
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "pixel")
    if true_col is None:
        true_col = make_true_col(target_class, "pixel")
    if true_col not in pixel_df.columns:
        raise KeyError(f"Missing true column in pixel_df: {true_col}")

    shape = image_db[image_key]["image_ref"].shape
    err = np.zeros(shape, dtype=np.uint8)

    sub = pixel_df[pixel_df[source_col].astype(str) == str(image_key)]

    if sub.empty:
        return err

    rows = sub[row_col].astype(int).to_numpy()
    cols = sub[col_col].astype(int).to_numpy()

    pred_s = coerce_binary_series(sub[pred_col], target_class=target_class)
    truth_s = coerce_binary_series(sub[true_col], target_class=target_class)
    valid_binary = pred_s.notna() & truth_s.notna()
    pred = pred_s.fillna(False).astype(bool).to_numpy()
    truth = truth_s.fillna(False).astype(bool).to_numpy()

    if truth_available_col in sub.columns:
        available = sub[truth_available_col].astype(bool).to_numpy()
    else:
        available = np.ones(len(sub), dtype=bool)

    codes = np.zeros(len(sub), dtype=np.uint8)

    available = available & valid_binary.to_numpy()

    codes[available & truth & pred] = 1
    codes[available & (~truth) & (~pred)] = 2
    codes[available & (~truth) & pred] = 3
    codes[available & truth & (~pred)] = 4

    err[rows, cols] = codes

    return err


def _fill_object_region_on_map(
    out: np.ndarray,
    image_key: str,
    image_db: dict,
    object_db: dict,
    object_id: str,
    value: int,
) -> bool:
    """
    Fill one object's spatial region in an output map.

    Priority:
    1. image labels + object label_id;
    2. object mask_global;
    3. object bbox + local mask.
    """
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    obj = object_db.get(str(object_id))
    if obj is None:
        return False

    img = image_db[image_key]

    # Preferred path: label image + label_id.
    if "labels" in img and "label_id" in obj:
        labels_img = np.asarray(img["labels"])
        label_id = int(obj["label_id"])

        mask = labels_img == label_id

        if mask.shape == out.shape and np.any(mask):
            out[mask] = int(value)
            return True

    # Fallback: global object mask.
    if "mask_global" in obj:
        mask = np.asarray(obj["mask_global"], dtype=bool)

        if mask.shape == out.shape and np.any(mask):
            out[mask] = int(value)
            return True

    # Fallback: local mask inserted into bbox.
    if "bbox" in obj and "mask" in obj:
        min_row, min_col, max_row, max_col = [
            int(v) for v in obj["bbox"]
        ]

        mask = np.asarray(obj["mask"], dtype=bool)
        view = out[min_row:max_row, min_col:max_col]

        if view.shape == mask.shape and np.any(mask):
            view[mask] = int(value)
            return True

    return False


def make_object_error_map(
    image_key: str,
    image_db: dict,
    object_db: dict,
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    pred_col: str | None = None,
    true_col: str | None = None,
    source_col: str = "source_image",
    object_id_col: str = "object_id",
    truth_available_ratio_col: str = "truth_available_ratio",
    min_truth_available_ratio: float = 0.50,
) -> np.ndarray:
    """
    Create an object-level TP/TN/FP/FN map.

    Codes
    -----
    0 : background / unavailable truth
    1 : TP
    2 : TN
    3 : FP
    4 : FN
    """
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "object")
    if true_col is None:
        true_col = make_true_col(target_class, "object")
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    required_cols = [source_col, object_id_col, pred_col, true_col]
    missing = [col for col in required_cols if col not in object_df.columns]

    if missing:
        raise KeyError(f"Missing column(s) in object_df: {missing}")

    if "labels" in image_db[image_key]:
        shape = image_db[image_key]["labels"].shape
    else:
        shape = image_db[image_key]["image_ref"].shape

    err = np.zeros(shape, dtype=np.uint8)

    sub = object_df[
        object_df[source_col].astype(str).eq(str(image_key))
    ].copy()

    if sub.empty:
        return err

    for _, row in sub.iterrows():
        object_id = str(row[object_id_col])

        if pd.isna(row[true_col]) or pd.isna(row[pred_col]):
            continue

        if truth_available_ratio_col in row.index:
            value = row[truth_available_ratio_col]

            if pd.notna(value):
                if float(value) < float(min_truth_available_ratio):
                    continue

        truth_s = coerce_binary_series(pd.Series([row[true_col]]), target_class=target_class)
        pred_s = coerce_binary_series(pd.Series([row[pred_col]]), target_class=target_class)
        if truth_s.isna().iloc[0] or pred_s.isna().iloc[0]:
            continue
        truth = bool(truth_s.iloc[0])
        pred = bool(pred_s.iloc[0])

        if truth and pred:
            code = 1  # TP
        elif (not truth) and (not pred):
            code = 2  # TN
        elif (not truth) and pred:
            code = 3  # FP
        else:
            code = 4  # FN

        _fill_object_region_on_map(
            out=err,
            image_key=image_key,
            image_db=image_db,
            object_db=object_db,
            object_id=object_id,
            value=code,
        )

    return err


def make_object_fp_fn_map(
    image_key: str,
    image_db: dict,
    object_db: dict,
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    pred_col: str | None = None,
    true_col: str | None = None,
    source_col: str = "source_image",
    object_id_col: str = "object_id",
    truth_available_ratio_col: str = "truth_available_ratio",
    min_truth_available_ratio: float = 0.50,
) -> np.ndarray:
    """
    Create an object-level FP/FN-only map.

    Codes
    -----
    0 : background / TP / TN / unavailable truth
    1 : FP
    2 : FN
    """
    err = make_object_error_map(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        object_df=object_df,
        target_class=target_class,
        pred_col=pred_col,
        true_col=true_col,
        source_col=source_col,
        object_id_col=object_id_col,
        truth_available_ratio_col=truth_available_ratio_col,
        min_truth_available_ratio=min_truth_available_ratio,
    )

    out = np.zeros_like(err, dtype=np.uint8)
    out[err == 3] = 1  # FP
    out[err == 4] = 2  # FN

    return out


def make_pixel_prediction_map(
    image_key: str,
    image_db: dict,
    pixel_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    pred_col: str | None = None,
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
) -> np.ndarray:
    """
    Create a binary map of pixels predicted as target class.
    """
    if pred_col is None:
        pred_col = make_predicted_col(target_class, "pixel")
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    for required_col in [source_col, row_col, col_col, pred_col]:
        if required_col not in pixel_df.columns:
            raise KeyError(f"Missing column in pixel_df: {required_col}")

    shape = image_db[image_key]["image_ref"].shape
    pred_map = np.zeros(shape, dtype=np.uint8)

    sub = pixel_df[pixel_df[source_col].astype(str) == str(image_key)]

    if sub.empty:
        return pred_map

    rows = sub[row_col].astype(int).to_numpy()
    cols = sub[col_col].astype(int).to_numpy()
    pred_s = coerce_binary_series(sub[pred_col], target_class=target_class)
    pred = pred_s.fillna(False).astype(bool).to_numpy()

    pred_map[rows[pred], cols[pred]] = 1

    return pred_map

def assign_object_decisions_to_pixels(
    pixel_df: pd.DataFrame,
    object_df: pd.DataFrame,
    decision_col: str = "decision_3way",
    object_id_col: str = "object_id",
    source_col: str = "source_image",
    output_col: str | None = None,
    validate: str = "many_to_one",
) -> pd.DataFrame:
    """Attach object-level decisions to every corresponding pixel row.

    This replaces notebook-local helpers such as
    ``assign_object_three_way_decision_to_pixels``.
    """
    output_col = decision_col if output_col is None else output_col
    required_pixel = [object_id_col]
    required_object = [object_id_col, decision_col]
    if source_col in pixel_df.columns and source_col in object_df.columns:
        required_pixel.append(source_col)
        required_object.append(source_col)
    missing_pixel = [column for column in required_pixel if column not in pixel_df.columns]
    missing_object = [column for column in required_object if column not in object_df.columns]
    if missing_pixel:
        raise KeyError(f"Missing columns in pixel_df: {missing_pixel}")
    if missing_object:
        raise KeyError(f"Missing columns in object_df: {missing_object}")

    keys = [object_id_col]
    if source_col in required_pixel:
        keys.append(source_col)
    lookup = object_df[keys + [decision_col]].drop_duplicates(keys).rename(
        columns={decision_col: output_col}
    )
    out = pixel_df.copy()
    if output_col in out.columns:
        out = out.drop(columns=[output_col])
    return out.merge(lookup, on=keys, how="left", validate=validate)


def make_pixel_categorical_map(
    image_key: str,
    image_db: dict,
    pixel_df: pd.DataFrame,
    value_col: str,
    value_to_code: dict,
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
    default_code: int = 0,
) -> np.ndarray:
    """Create a categorical pixel map from any dataframe column."""
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")
    required = [source_col, row_col, col_col, value_col]
    missing = [column for column in required if column not in pixel_df.columns]
    if missing:
        raise KeyError(f"Missing columns in pixel_df: {missing}")

    shape = np.asarray(image_db[image_key]["image_ref"]).shape
    out = np.full(shape, int(default_code), dtype=int)
    sub = pixel_df[pixel_df[source_col].astype(str).eq(str(image_key))]
    if sub.empty:
        return out
    rows = sub[row_col].astype(int).to_numpy()
    cols = sub[col_col].astype(int).to_numpy()
    valid = (
        (rows >= 0)
        & (rows < shape[0])
        & (cols >= 0)
        & (cols < shape[1])
    )
    codes = sub[value_col].map(value_to_code).fillna(default_code).astype(int).to_numpy()
    out[rows[valid], cols[valid]] = codes[valid]
    return out
