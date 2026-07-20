from __future__ import annotations

import numpy as np
import pandas as pd


QC_FLAG_COLUMNS = ["record_type", "record_id", "flag_type", "warning"]

DEFAULT_REQUIRED_IMAGE_FIELDS = (
    "cube",
    "image_ref",
    "mask",
    "labels",
    "clean_key",
    "sample_kind",
    "nut_type",
    "n_objects",
    "object_ids",
)

DEFAULT_REQUIRED_OBJECT_FIELDS = (
    "object_id",
    "source_clean_key",
    "sample_kind",
    "object_nut_type",
    "batch",
    "split",
    "bbox",
    "centroid",
    "area_pixels",
    "mask",
    "mask_global",
    "positions_global",
    "spectra",
    "mean_spectrum",
    "median_spectrum",
    "std_spectrum",
)

IMAGE_QC_COLUMNS = (
    "clean_key",
    "image_id",
    "sample_kind",
    "nut_type",
    "batch",
    "position_set",
    "description",
    "is_pure",
    "is_mixture",
    "is_position_reference",
    "is_unknown",
    "height",
    "width",
    "n_bands",
    "n_pixels_image",
    "n_objects_recorded",
    "n_labels_positive",
    "max_label",
    "threshold",
    "mask_area_pixels",
    "mask_area_ratio",
    "has_wavelengths",
    "data_mode",
)

OBJECT_QC_COLUMNS = (
    "object_id",
    "source_clean_key",
    "source_image",
    "sample_kind",
    "image_nut_type",
    "object_nut_type",
    "batch",
    "position_set",
    "split",
    "is_pure",
    "is_mixture",
    "is_position_reference",
    "is_unknown",
    "label_id",
    "object_index",
    "area_pixels",
    "n_pixels",
    "n_bands",
    "spectra_shape",
    "mean_spectrum_length",
    "centroid_row",
    "centroid_col",
    "bbox",
    "bbox_height",
    "bbox_width",
    "bbox_area",
    "data_mode",
)


def build_image_qc_table(image_db: dict) -> pd.DataFrame:
    """Build the image QC summary table used by notebook 01."""
    rows = []
    for image_key, img in image_db.items():
        cube = np.asarray(img["cube"])
        labels = np.asarray(img["labels"])
        unique_labels = np.unique(labels)
        object_labels = unique_labels[unique_labels > 0]
        wavelengths = img.get("wavelengths")
        has_wavelengths = wavelengths is not None and len(np.asarray(wavelengths)) > 0

        rows.append(
            {
                "clean_key": image_key,
                "image_id": img.get("image_id"),
                "sample_kind": img.get("sample_kind"),
                "nut_type": img.get("nut_type"),
                "batch": img.get("batch"),
                "position_set": img.get("position_set"),
                "description": img.get("description"),
                "is_pure": bool(img.get("is_pure", False)),
                "is_mixture": bool(img.get("is_mixture", False)),
                "is_position_reference": bool(img.get("is_position_reference", False)),
                "is_unknown": bool(img.get("is_unknown", False)),
                "height": cube.shape[0],
                "width": cube.shape[1],
                "n_bands": cube.shape[2],
                "n_pixels_image": cube.shape[0] * cube.shape[1],
                "n_objects_recorded": int(img.get("n_objects", 0)),
                "n_labels_positive": int(len(object_labels)),
                "max_label": int(labels.max()) if labels.size else 0,
                "threshold": img.get("threshold"),
                "mask_area_pixels": int(np.asarray(img["mask"]).sum()) if "mask" in img else np.nan,
                "mask_area_ratio": (
                    float(np.asarray(img["mask"]).sum() / labels.size)
                    if "mask" in img and labels.size > 0
                    else np.nan
                ),
                "has_wavelengths": bool(has_wavelengths),
                "data_mode": img.get("data_mode"),
            }
        )

    out = pd.DataFrame(rows, columns=IMAGE_QC_COLUMNS)
    if out.empty:
        return out
    return (
        out.sort_values(
            ["sample_kind", "nut_type", "batch", "position_set", "clean_key"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_image_qc_warnings(image_qc_df: pd.DataFrame) -> pd.DataFrame:
    """Return image-level QC warnings from an image QC summary table."""
    warnings = []
    for _, row in image_qc_df.iterrows():
        if row["n_objects_recorded"] == 0:
            warnings.append({"clean_key": row["clean_key"], "warning": "No object detected"})
        if row["n_objects_recorded"] != row["n_labels_positive"]:
            warnings.append(
                {
                    "clean_key": row["clean_key"],
                    "warning": (
                        f"n_objects_recorded={row['n_objects_recorded']} differs from "
                        f"n_labels_positive={row['n_labels_positive']}"
                    ),
                }
            )
        if pd.isna(row["mask_area_ratio"]) or row["mask_area_ratio"] <= 0:
            warnings.append(
                {
                    "clean_key": row["clean_key"],
                    "warning": "Empty or invalid mask area ratio",
                }
            )
    return pd.DataFrame(warnings, columns=["clean_key", "warning"])


def build_object_qc_table(object_db: dict) -> pd.DataFrame:
    """Build the object QC summary table used by notebook 01."""
    rows = []
    for object_id, obj in object_db.items():
        centroid = obj.get("centroid", (np.nan, np.nan))
        bbox = obj.get("bbox", None)
        spectra = np.asarray(obj.get("spectra"))
        mean_spectrum = np.asarray(obj.get("mean_spectrum"))

        if bbox is not None:
            min_row, min_col, max_row, max_col = bbox
            bbox_height = int(max_row - min_row)
            bbox_width = int(max_col - min_col)
        else:
            bbox_height = np.nan
            bbox_width = np.nan

        rows.append(
            {
                "object_id": object_id,
                "source_clean_key": obj.get("source_clean_key"),
                "source_image": obj.get("source_image"),
                "sample_kind": obj.get("sample_kind"),
                "image_nut_type": obj.get("image_nut_type"),
                "object_nut_type": obj.get("object_nut_type"),
                "batch": obj.get("batch"),
                "position_set": obj.get("position_set"),
                "split": obj.get("split"),
                "is_pure": bool(obj.get("is_pure", False)),
                "is_mixture": bool(obj.get("is_mixture", False)),
                "is_position_reference": bool(obj.get("is_position_reference", False)),
                "is_unknown": bool(obj.get("is_unknown", False)),
                "label_id": obj.get("label_id"),
                "object_index": obj.get("object_index"),
                "area_pixels": obj.get("area_pixels"),
                "n_pixels": obj.get("n_pixels"),
                "n_bands": obj.get("n_bands"),
                "spectra_shape": spectra.shape if spectra is not None else None,
                "mean_spectrum_length": len(mean_spectrum) if mean_spectrum is not None else np.nan,
                "centroid_row": centroid[0] if centroid is not None else np.nan,
                "centroid_col": centroid[1] if centroid is not None else np.nan,
                "bbox": bbox,
                "bbox_height": bbox_height,
                "bbox_width": bbox_width,
                "bbox_area": (
                    bbox_height * bbox_width
                    if np.isfinite(bbox_height) and np.isfinite(bbox_width)
                    else np.nan
                ),
                "data_mode": obj.get("data_mode"),
            }
        )

    out = pd.DataFrame(rows, columns=OBJECT_QC_COLUMNS)
    if out.empty:
        return out
    return (
        out.sort_values(
            ["sample_kind", "object_nut_type", "batch", "source_clean_key", "object_index"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_object_qc_warnings(object_qc_df: pd.DataFrame) -> pd.DataFrame:
    """Return object-level QC warnings from an object QC summary table."""
    warnings = []
    for _, row in object_qc_df.iterrows():
        if row["area_pixels"] is None or pd.isna(row["area_pixels"]) or row["area_pixels"] <= 0:
            warnings.append({"object_id": row["object_id"], "warning": "Invalid object area"})
        if row["n_pixels"] != row["area_pixels"]:
            warnings.append(
                {
                    "object_id": row["object_id"],
                    "warning": f"n_pixels={row['n_pixels']} differs from area_pixels={row['area_pixels']}",
                }
            )
        if row["mean_spectrum_length"] != row["n_bands"]:
            warnings.append(
                {
                    "object_id": row["object_id"],
                    "warning": (
                        f"mean_spectrum_length={row['mean_spectrum_length']} differs "
                        f"from n_bands={row['n_bands']}"
                    ),
                }
            )
        if row["bbox_area"] < row["area_pixels"]:
            warnings.append(
                {
                    "object_id": row["object_id"],
                    "warning": "bbox_area smaller than object area",
                }
            )
    return pd.DataFrame(warnings, columns=["object_id", "warning"])


def check_missing_required_fields(
    image_db: dict,
    object_db: dict,
    required_image_fields=DEFAULT_REQUIRED_IMAGE_FIELDS,
    required_object_fields=DEFAULT_REQUIRED_OBJECT_FIELDS,
) -> pd.DataFrame:
    """Return records missing required image or object fields."""
    rows = []
    for image_key, img in image_db.items():
        missing = [field for field in required_image_fields if field not in img]
        if missing:
            rows.append(
                {
                    "record_type": "image",
                    "record_id": image_key,
                    "missing_fields": missing,
                }
            )
    for object_id, obj in object_db.items():
        missing = [field for field in required_object_fields if field not in obj]
        if missing:
            rows.append(
                {
                    "record_type": "object",
                    "record_id": object_id,
                    "missing_fields": missing,
                }
            )
    return pd.DataFrame(rows, columns=["record_type", "record_id", "missing_fields"])


def build_object_shape_check_tables(object_db: dict, image_db: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all object shape checks and the failing subset."""
    rows = []
    for object_id, obj in object_db.items():
        source_key = obj.get("source_clean_key")
        img = image_db.get(source_key)
        spectra = np.asarray(obj["spectra"])
        mean_spectrum = np.asarray(obj["mean_spectrum"])
        median_spectrum = np.asarray(obj["median_spectrum"])
        std_spectrum = np.asarray(obj["std_spectrum"])
        positions_global = np.asarray(obj["positions_global"])

        row = {
            "object_id": object_id,
            "source_clean_key": source_key,
            "spectra_shape": spectra.shape,
            "positions_shape": positions_global.shape,
            "mean_spectrum_length": len(mean_spectrum),
            "median_spectrum_length": len(median_spectrum),
            "std_spectrum_length": len(std_spectrum),
            "n_pixels_recorded": obj.get("n_pixels"),
            "area_pixels": obj.get("area_pixels"),
            "n_bands_recorded": obj.get("n_bands"),
            "ok_spectra_pixels": spectra.shape[0] == obj.get("n_pixels"),
            "ok_positions_pixels": positions_global.shape[0] == obj.get("n_pixels"),
            "ok_mean_length": len(mean_spectrum) == obj.get("n_bands"),
            "ok_median_length": len(median_spectrum) == obj.get("n_bands"),
            "ok_std_length": len(std_spectrum) == obj.get("n_bands"),
        }
        if img is not None:
            cube = np.asarray(img["cube"])
            row["image_n_bands"] = cube.shape[2]
            row["ok_object_image_bands"] = obj.get("n_bands") == cube.shape[2]
        else:
            row["image_n_bands"] = np.nan
            row["ok_object_image_bands"] = False
        rows.append(row)

    shape_check_df = pd.DataFrame(rows)
    if shape_check_df.empty:
        return shape_check_df, shape_check_df.copy()

    ok_cols = [
        "ok_spectra_pixels",
        "ok_positions_pixels",
        "ok_mean_length",
        "ok_median_length",
        "ok_std_length",
        "ok_object_image_bands",
    ]
    bad_shape_df = shape_check_df[~shape_check_df[ok_cols].all(axis=1)].copy()
    return shape_check_df, bad_shape_df


def build_qc_flags_table(
    image_warnings_df: pd.DataFrame | None = None,
    object_warnings_df: pd.DataFrame | None = None,
    missing_fields_df: pd.DataFrame | None = None,
    bad_shape_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine notebook 01 QC warnings into one canonical flag table."""
    parts = []

    if image_warnings_df is not None and len(image_warnings_df) > 0:
        tmp = image_warnings_df.copy()
        tmp["record_type"] = "image"
        tmp = tmp.rename(columns={"clean_key": "record_id"})
        tmp["flag_type"] = "image_warning"
        parts.append(tmp[QC_FLAG_COLUMNS])

    if object_warnings_df is not None and len(object_warnings_df) > 0:
        tmp = object_warnings_df.copy()
        tmp["record_type"] = "object"
        tmp = tmp.rename(columns={"object_id": "record_id"})
        tmp["flag_type"] = "object_warning"
        parts.append(tmp[QC_FLAG_COLUMNS])

    if missing_fields_df is not None and len(missing_fields_df) > 0:
        tmp = missing_fields_df.copy()
        tmp["flag_type"] = "missing_fields"
        tmp["warning"] = tmp["missing_fields"].astype(str)
        parts.append(tmp[QC_FLAG_COLUMNS])

    if bad_shape_df is not None and len(bad_shape_df) > 0:
        tmp = bad_shape_df.copy()
        tmp["record_type"] = "object"
        tmp = tmp.rename(columns={"object_id": "record_id"})
        tmp["flag_type"] = "bad_shape"
        tmp["warning"] = "Object shape consistency check failed"
        parts.append(tmp[QC_FLAG_COLUMNS])

    if not parts:
        return pd.DataFrame(columns=QC_FLAG_COLUMNS)
    return pd.concat(parts, ignore_index=True, sort=False)
