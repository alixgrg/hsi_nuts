from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.protocol_governance import canonical_json, sha256_payload

from src.spectra.band_selection import spectral_pixel_validity_report

QC_ALERT_COLUMNS = list(expcfg.QC_ALERT_OUTPUT_COLUMNS)
QC_FLAG_COLUMNS = QC_ALERT_COLUMNS

QC_REVIEW_KEY_COLUMNS = (
    "record_type",
    "record_id",
    "flag_type",
)

QC_REVIEW_COLUMNS = expcfg.QC_REVIEW_OUTPUT_COLUMNS

QC_RESOLVED_FLAG_COLUMNS = (
    *QC_FLAG_COLUMNS,
    *[
        column
        for column in QC_REVIEW_COLUMNS
        if column not in QC_REVIEW_KEY_COLUMNS
    ],
)

SPECTRAL_INTEGRITY_COLUMNS = (
    "record_type",
    "record_id",
    "shape",
    "dtype",
    "is_numeric",
    "expected_ndim_ok",
    "n_valid_rows",
    "n_nan",
    "n_inf",
    "finite_rate",
    "axis_length",
    "axis_matches_reference",
    "is_valid",
    "error",
)

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
    "wavelengths",
)

DEFAULT_REQUIRED_OBJECT_FIELDS = (
    "object_id",
    "source_clean_key",
    "sample_kind",
    "object_nut_type",
    "batch",
    "bbox",
    "centroid",
    "area_pixels",
    "positions_global",
    "spectra",
    "mean_spectrum",
    "median_spectrum",
    "std_spectrum",
    "wavelengths",
)

IMAGE_QC_COLUMNS = expcfg.IMAGE_QC_OUTPUT_COLUMNS
OBJECT_QC_COLUMNS = expcfg.OBJECT_QC_OUTPUT_COLUMNS


def _shape_text(shape) -> str:
    return "x".join(str(int(value)) for value in tuple(shape))


def _finite_counts(array: np.ndarray) -> tuple[int, int, int, float]:
    if array.ndim == 0:
        row_view = array.reshape(1, 1)
    elif array.ndim == 1:
        row_view = array.reshape(1, -1)
    else:
        row_view = array.reshape(-1, array.shape[-1])
    finite = np.isfinite(array)
    return (
        int(np.isfinite(row_view).all(axis=1).sum()),
        int(np.isnan(array).sum()),
        int(np.isinf(array).sum()),
        float(finite.mean()) if finite.size else np.nan,
    )


def _zero_variance_band_rate(array, *, epsilon: float) -> float:
    array = np.asarray(array)
    if array.ndim < 2 or not np.issubdtype(array.dtype, np.number):
        return np.nan
    rows = array.reshape(-1, array.shape[-1])
    if rows.shape[0] == 0:
        return np.nan
    variances = np.var(rows.astype(float), axis=0)
    return float(np.mean(variances <= float(epsilon)))


def build_spectral_integrity_table(
    records: Mapping,
    *,
    record_type: str,
    array_getter: Callable,
    axis_getter: Callable | None = None,
    expected_ndim: int | None = None,
    reference_axis=None,
) -> pd.DataFrame:
    """Build a reusable numeric/axis integrity table for images or objects."""
    rows = []
    reference_axis = (
        None
        if reference_axis is None
        else np.asarray(reference_axis, dtype=float)
    )

    for record_id, record in records.items():
        try:
            array = np.asarray(array_getter(record))
            is_numeric = bool(np.issubdtype(array.dtype, np.number))
            expected_ndim_ok = expected_ndim is None or array.ndim == expected_ndim
            if is_numeric:
                n_valid_rows, n_nan, n_inf, finite_rate = _finite_counts(array)
            else:
                n_valid_rows, n_nan, n_inf, finite_rate = 0, np.nan, np.nan, 0.0

            axis = axis_getter(record) if axis_getter is not None else None
            if axis is None:
                axis_length = np.nan
                axis_matches_reference = np.nan
            else:
                axis = np.asarray(axis, dtype=float)
                axis_length = int(len(axis))
                axis_matches_reference = bool(
                    reference_axis is None or np.array_equal(axis, reference_axis)
                )

            axis_ok = (
                True
                if pd.isna(axis_matches_reference)
                else bool(axis_matches_reference)
            )
            rows.append(
                {
                    "record_type": record_type,
                    "record_id": str(record_id),
                    "shape": _shape_text(array.shape),
                    "dtype": str(array.dtype),
                    "is_numeric": is_numeric,
                    "expected_ndim_ok": bool(expected_ndim_ok),
                    "n_valid_rows": n_valid_rows,
                    "n_nan": n_nan,
                    "n_inf": n_inf,
                    "finite_rate": finite_rate,
                    "axis_length": axis_length,
                    "axis_matches_reference": axis_matches_reference,
                    "is_valid": bool(
                        is_numeric
                        and expected_ndim_ok
                        and finite_rate == 1.0
                        and axis_ok
                    ),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "record_type": record_type,
                    "record_id": str(record_id),
                    "is_valid": False,
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows, columns=SPECTRAL_INTEGRITY_COLUMNS)


def _first_axis(records: Mapping):
    for record in records.values():
        axis = record.get("wavelengths")
        if axis is not None and len(np.asarray(axis)) > 0:
            return np.asarray(axis, dtype=float)
    return None


def build_image_qc_table(
    image_db: dict,
    *,
    qc_policy=expcfg.QC_POLICY,
) -> pd.DataFrame:
    """Build the compact image QC output used by notebooks 00 and 01."""
    reference_axis = _first_axis(image_db)
    rows = []

    for image_key, img in image_db.items():
        cube = np.asarray(img.get("cube"))
        labels = np.asarray(img.get("labels"))
        mask = np.asarray(img.get("mask"), dtype=bool)
        wavelengths = img.get("wavelengths")
        axis_matches = bool(
            wavelengths is not None
            and reference_axis is not None
            and np.array_equal(np.asarray(wavelengths, dtype=float), reference_axis)
        )
        numeric = np.issubdtype(cube.dtype, np.number)
        if numeric:
            n_valid, n_nan, n_inf, _ = _finite_counts(cube)
        else:
            n_valid, n_nan, n_inf = 0, np.nan, np.nan

        n_labels = int(len(np.unique(labels[labels > 0]))) if labels.size else 0
        n_objects = int(img.get("n_objects", 0))
        mask_area_ratio = float(mask.mean()) if mask.size else np.nan
        invalid_rate = (
            float((n_nan + n_inf) / cube.size)
            if numeric and cube.size
            else 1.0
        )

        status = str(img.get("image_status", "accepted"))
        fatal = (
            cube.ndim != 3
            or not numeric
            or invalid_rate > float(qc_policy.get("max_invalid_pixel_rate", 0.0))
            or not axis_matches
            or (
                bool(qc_policy.get("exclude_empty_mask", True))
                and (not mask.size or int(mask.sum()) == 0)
            )
        )
        warning = n_objects != n_labels
        if status != "corrected_segmentation":
            status = "excluded" if fatal else ("warning" if warning else "accepted")

        rows.append(
            {
                "clean_key": str(image_key),
                "sample_kind": img.get("sample_kind"),
                "nut_type": img.get("nut_type"),
                "batch": img.get("batch"),
                "height": int(cube.shape[0]) if cube.ndim == 3 else np.nan,
                "width": int(cube.shape[1]) if cube.ndim == 3 else np.nan,
                "n_bands": int(cube.shape[2]) if cube.ndim == 3 else np.nan,
                "n_objects": n_objects,
                "n_labels": n_labels,
                "mask_area_ratio": mask_area_ratio,
                "n_valid_pixels": n_valid,
                "n_nan": n_nan,
                "n_inf": n_inf,
                "zero_variance_band_rate": _zero_variance_band_rate(
                    cube,
                    epsilon=float(
                        qc_policy.get(
                            "zero_variance_epsilon",
                            expcfg.QC_ZERO_VARIANCE_EPSILON,
                        )
                    ),
                ),
                "axis_matches_reference": axis_matches,
                "image_status": status,
            }
        )

    out = pd.DataFrame(rows, columns=IMAGE_QC_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(
        ["sample_kind", "nut_type", "batch", "clean_key"],
        na_position="last",
    ).reset_index(drop=True)


def _flag(
    *,
    record_type,
    record_id,
    flag_type,
    warning,
    severity="warning",
    qc_status="warning",
    exclusion_reason="",
    requires_segmentation_review=False,
    evidence=None,
):
    key_payload = {
        "record_type": str(record_type),
        "record_id": str(record_id),
        "flag_type": str(flag_type),
    }
    evidence_payload = (
        {"warning": str(warning)}
        if evidence is None
        else dict(evidence)
    )
    return {
        "alert_id": sha256_payload(key_payload),
        "record_type": str(record_type),
        "record_id": str(record_id),
        "flag_type": str(flag_type),
        "severity": str(severity),
        "qc_status": str(qc_status),
        "exclusion_reason": str(exclusion_reason),
        "requires_segmentation_review": bool(requires_segmentation_review),
        "warning": str(warning),
        "evidence_json": canonical_json(evidence_payload),
    }


def _alerts_from_mask(
    frame,
    mask,
    *,
    record_type,
    id_col,
    flag_type,
    warning,
    severity="warning",
    qc_status="warning",
    exclusion_reason="",
    requires_segmentation_review=False,
    evidence_cols=(),
):
    selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    if selected.empty:
        return pd.DataFrame(columns=QC_ALERT_COLUMNS)
    ids = selected[id_col].astype(str)
    warnings_series = (
        warning.loc[selected.index].astype(str)
        if isinstance(warning, pd.Series)
        else pd.Series(str(warning), index=selected.index)
    )
    out = pd.DataFrame(index=selected.index)
    out["record_type"] = str(record_type)
    out["record_id"] = ids
    out["flag_type"] = str(flag_type)
    out["alert_id"] = [
        sha256_payload(
            {
                "record_type": str(record_type),
                "record_id": record_id,
                "flag_type": str(flag_type),
            }
        )
        for record_id in ids
    ]
    out["severity"] = str(severity)
    out["qc_status"] = str(qc_status)
    out["exclusion_reason"] = str(exclusion_reason)
    out["requires_segmentation_review"] = bool(
        requires_segmentation_review
    )
    out["warning"] = warnings_series
    evidence_records = (
        selected.loc[:, list(evidence_cols)].to_dict("records")
        if evidence_cols
        else [{} for _ in range(len(selected))]
    )
    out["evidence_json"] = [
        canonical_json(
            {
                column: (
                    None
                    if isinstance(value, (float, np.floating))
                    and not np.isfinite(value)
                    else (
                        value.item()
                        if isinstance(value, np.generic)
                        else value
                    )
                )
                for column, value in row.items()
            }
        )
        for row in evidence_records
    ]
    return out.loc[:, QC_ALERT_COLUMNS].reset_index(drop=True)


def build_image_qc_warnings(image_qc_df: pd.DataFrame) -> pd.DataFrame:
    """Build image alerts from vectorized boolean masks."""
    n_objects = pd.to_numeric(image_qc_df["n_objects"], errors="coerce")
    n_labels = pd.to_numeric(image_qc_df["n_labels"], errors="coerce")
    n_nonfinite = (
        pd.to_numeric(image_qc_df["n_nan"], errors="coerce").fillna(0)
        + pd.to_numeric(image_qc_df["n_inf"], errors="coerce").fillna(0)
    )
    mismatch_warning = (
        "n_objects="
        + n_objects.astype("Int64").astype(str)
        + " differs from n_labels="
        + n_labels.astype("Int64").astype(str)
        + "."
    )
    parts = [
        _alerts_from_mask(
            image_qc_df,
            n_objects.eq(0),
            record_type="image",
            id_col="clean_key",
            flag_type="empty_segmentation",
            severity="error",
            qc_status="excluded",
            exclusion_reason="no_object_detected",
            requires_segmentation_review=True,
            warning="No object detected.",
            evidence_cols=("n_objects", "n_labels", "mask_area_ratio"),
        ),
        _alerts_from_mask(
            image_qc_df,
            n_objects.ne(n_labels),
            record_type="image",
            id_col="clean_key",
            flag_type="object_label_count_mismatch",
            requires_segmentation_review=True,
            warning=mismatch_warning,
            evidence_cols=("n_objects", "n_labels"),
        ),
        _alerts_from_mask(
            image_qc_df,
            n_nonfinite.gt(0),
            record_type="image",
            id_col="clean_key",
            flag_type="non_finite_cube",
            severity="error",
            qc_status="excluded",
            exclusion_reason="non_finite_cube",
            warning="Cube contains NaN or infinite values.",
            evidence_cols=("n_nan", "n_inf"),
        ),
        _alerts_from_mask(
            image_qc_df,
            ~image_qc_df["axis_matches_reference"].fillna(False).astype(bool),
            record_type="image",
            id_col="clean_key",
            flag_type="wavelength_axis_mismatch",
            severity="error",
            qc_status="excluded",
            exclusion_reason="wavelength_axis_mismatch",
            warning="Wavelength axis differs from the canonical axis.",
            evidence_cols=("n_bands", "axis_matches_reference"),
        ),
        _alerts_from_mask(
            image_qc_df,
            image_qc_df["image_status"].eq("corrected_segmentation"),
            record_type="image",
            id_col="clean_key",
            flag_type="corrected_segmentation",
            severity="info",
            qc_status="corrected_segmentation",
            requires_segmentation_review=True,
            warning=(
                "Segmentation was corrected; notebooks 00-02 must be rerun."
            ),
            evidence_cols=("image_status",),
        ),
    ]
    return pd.concat(parts, ignore_index=True).loc[:, QC_ALERT_COLUMNS]


def _bbox_distance(left, right) -> float:
    lmin_r, lmin_c, lmax_r, lmax_c = map(float, left)
    rmin_r, rmin_c, rmax_r, rmax_c = map(float, right)
    row_gap = max(rmin_r - lmax_r, lmin_r - rmax_r, 0.0)
    col_gap = max(rmin_c - lmax_c, lmin_c - rmax_c, 0.0)
    return float(np.hypot(row_gap, col_gap))


def _nearest_bbox_distances(object_db: Mapping) -> dict[str, float]:
    """Compute all within-image nearest bounding-box distances vectorially."""
    by_image: dict[str, list[tuple[str, tuple]]] = {}
    for object_id, obj in object_db.items():
        bbox = obj.get("bbox")
        if bbox is None:
            continue
        source = str(obj.get("source_clean_key", obj.get("source_image")))
        by_image.setdefault(source, []).append((str(object_id), tuple(bbox)))

    nearest = {}
    for items in by_image.values():
        ids = [item[0] for item in items]
        boxes = np.asarray([item[1] for item in items], dtype=float)
        if len(boxes) == 1:
            nearest[ids[0]] = np.inf
            continue
        min_r, min_c, max_r, max_c = boxes.T
        zeros = np.zeros((len(boxes), len(boxes)), dtype=float)
        row_gap = np.maximum.reduce(
            [
                min_r[None, :] - max_r[:, None],
                min_r[:, None] - max_r[None, :],
                zeros,
            ]
        )
        col_gap = np.maximum.reduce(
            [
                min_c[None, :] - max_c[:, None],
                min_c[:, None] - max_c[None, :],
                zeros,
            ]
        )
        distances = np.hypot(row_gap, col_gap)
        np.fill_diagonal(distances, np.inf)
        nearest.update(
            zip(ids, np.min(distances, axis=1).astype(float), strict=True)
        )
    return nearest


def build_object_qc_table(
    object_db: dict,
    image_db: dict | None = None,
    *,
    pixel_qc_df: pd.DataFrame | None = None,
    include_geometry: bool = True,
    border_margin: int = 0,
    merge_warning_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Build compact object QC with geometry and spectral integrity flags."""
    thresholds = dict(
        expcfg.SEGMENTATION_MERGE_WARNING_THRESHOLDS
        if merge_warning_thresholds is None
        else merge_warning_thresholds
    )
    min_fill = float(thresholds.get("min_fill_ratio", 0.45))
    min_separation = float(thresholds.get("min_separation_pixels", 2.0))
    min_area = int(
        thresholds.get(
            "min_area_pixels",
            expcfg.QC_POLICY["min_area_pixels"],
        )
    )

    nearest_by_object = _nearest_bbox_distances(object_db)

    pixel_qc_by_object = {}
    if pixel_qc_df is not None and not pixel_qc_df.empty:
        for object_id, group in pixel_qc_df.groupby(
            "object_id",
            sort=False,
        ):
            n_total = len(group)
            n_analysis_valid = int(
                group["analysis_valid"].sum()
            )
            pixel_qc_by_object[str(object_id)] = {
                "n_analysis_valid_pixels": n_analysis_valid,
                "n_analysis_invalid_pixels": (
                    n_total - n_analysis_valid
                ),
                "analysis_invalid_pixel_rate": (
                    float((n_total - n_analysis_valid) / n_total)
                    if n_total
                    else np.nan
                ),
                "n_all_zero_pixels": int(
                    group["all_zero_spectrum"].sum()
                ),
                "n_nonpositive_pixels": int(
                    group["has_nonpositive_reflectance"].sum()
                ),
            }

    rows = []
    for object_id, obj in object_db.items():
        source = str(obj.get("source_clean_key", obj.get("source_image")))
        spectra = np.asarray(obj.get("spectra"))
        numeric = np.issubdtype(spectra.dtype, np.number)
        if numeric:
            n_valid, n_nan, n_inf, _ = _finite_counts(spectra)
        else:
            n_valid, n_nan, n_inf = 0, np.nan, np.nan

        area = int(obj.get("area_pixels", 0))
        n_pixels = int(obj.get("n_pixels", spectra.shape[0] if spectra.ndim else 0))
        n_bands = int(obj.get("n_bands", spectra.shape[1] if spectra.ndim == 2 else 0))
        bbox = obj.get("bbox")
        bbox_area = np.nan
        fill_ratio = np.nan
        touches_border = False
        nearest = np.inf

        if include_geometry and bbox is not None:
            min_row, min_col, max_row, max_col = map(int, bbox)
            bbox_area = int(max(0, max_row - min_row) * max(0, max_col - min_col))
            fill_ratio = float(area / bbox_area) if bbox_area > 0 else np.nan
            image_shape = None
            if image_db is not None and source in image_db:
                labels = np.asarray(image_db[source].get("labels"))
                image_shape = labels.shape if labels.ndim == 2 else None
            if image_shape is None and obj.get("mask_global") is not None:
                mask_global = np.asarray(obj.get("mask_global"))
                image_shape = mask_global.shape if mask_global.ndim == 2 else None
            if image_shape is not None:
                height, width = image_shape
                touches_border = bool(
                    min_row <= border_margin
                    or min_col <= border_margin
                    or max_row >= height - border_margin
                    or max_col >= width - border_margin
                )
            nearest = nearest_by_object.get(str(object_id), np.inf)

        possible_merged = bool(
            include_geometry
            and (
                (np.isfinite(fill_ratio) and fill_ratio < min_fill)
                or (np.isfinite(nearest) and nearest < min_separation)
            )
        )
        too_small = area < min_area
        requires_review = bool(possible_merged or too_small)
        pixel_stats = pixel_qc_by_object.get(str(object_id), {
                    "n_analysis_valid_pixels": n_valid,
                    "n_analysis_invalid_pixels": 0,
                    "analysis_invalid_pixel_rate": 0.0,
                    "n_all_zero_pixels": 0,
                    "n_nonpositive_pixels": 0,
                })
        fatal = (
            not numeric
            or spectra.ndim != 2
            or n_nan + n_inf > 0
            or n_pixels != area
            or spectra.shape != (n_pixels, n_bands)
            or pixel_stats["n_analysis_valid_pixels"] ==0
        )

        status = str(obj.get("object_status", "accepted"))
        if obj.get("image_status") == "corrected_segmentation":
            status = "corrected_segmentation"
        elif fatal or too_small:
            status = "excluded"
        elif possible_merged or touches_border:
            status = "warning"
        else:
            status = "accepted"

        mean_spectrum = np.asarray(obj.get("mean_spectrum"), dtype=float)
        median_spectrum = np.asarray(obj.get("median_spectrum"), dtype=float)
        std_spectrum = np.asarray(obj.get("std_spectrum"), dtype=float)        
        rows.append(
            {
                "object_id": str(object_id),
                "source_image": source,
                "sample_kind": obj.get("sample_kind"),
                "object_nut_type": obj.get("object_nut_type"),
                "batch": obj.get("batch"),
                "area_pixels": area,
                "n_pixels": n_pixels,
                "n_bands": n_bands,
                "n_valid_pixels": n_valid,
                "n_nan": n_nan,
                "n_inf": n_inf,
                "zero_variance_band_rate": _zero_variance_band_rate(
                    spectra,
                    epsilon=expcfg.QC_ZERO_VARIANCE_EPSILON,
                ),
                "spectral_robust_distance": np.nan,
                "spectral_outlier": False,
                "mean_spectrum_mean": (
                    float(np.nanmean(mean_spectrum)) if mean_spectrum.size else np.nan
                ),
                "median_spectrum_mean": (
                    float(np.nanmean(median_spectrum)) if median_spectrum.size else np.nan
                ),
                "std_spectrum_mean": (
                    float(np.nanmean(std_spectrum)) if std_spectrum.size else np.nan
                ),
                "bbox_fill_ratio": fill_ratio,
                "touches_border": touches_border,
                "nearest_object_distance": nearest,
                "possible_merged_object": possible_merged,
                "too_small": too_small,
                "requires_segmentation_review": requires_review,
                "object_status": status,
                **pixel_stats,
            }
        )

    out = pd.DataFrame(rows, columns=OBJECT_QC_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(
        ["sample_kind", "object_nut_type", "batch", "source_image", "object_id"],
        na_position="last",
    ).reset_index(drop=True)


def add_robust_spectral_qc(
    object_qc_df: pd.DataFrame,
    object_db: Mapping,
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
    *,
    group_cols=expcfg.QC_SPECTRAL_GROUP_COLUMNS,
    threshold=expcfg.QC_SPECTRAL_OUTLIER_DISTANCE_THRESHOLD,
    epsilon=expcfg.QC_ZERO_VARIANCE_EPSILON,
) -> pd.DataFrame:
    """Add median-band robust spectral distances within scientific groups."""
    out = object_qc_df.copy()
    missing = [
        column
        for column in ("object_id", *tuple(group_cols))
        if column not in out.columns
    ]
    if missing:
        raise KeyError(f"Object QC table is missing spectral groups: {missing}")
    if out.empty:
        return out

    spectra = []
    for object_id in out["object_id"].astype(str):
        if object_id not in object_db:
            raise KeyError(f"Object {object_id!r} is absent from object_db.")
        # spectra.append(
        #     np.asarray(object_db[object_id]["mean_spectrum"], dtype=float)
        # )
        obj = object_db[object_id]
        X_obj = np.asarray(obj["spectra"], dtype=float)
        validity = spectral_pixel_validity_report(
            X_obj,
            policy=pixel_validity_policy,
        )
        valid_mask = validity["valid_mask"]
        if not valid_mask.any():
            raise ValueError(
                f"Object {object_id!r} has no valid spectral pixel."
            )
        spectra.append(
            np.mean(X_obj[valid_mask], axis=0)
        )

    lengths = {len(row) for row in spectra}
    if len(lengths) != 1:
        raise ValueError("Object mean spectra have inconsistent lengths.")
    X = np.vstack(spectra)
    distances = np.full(len(out), np.nan, dtype=float)

    grouped = out.groupby(list(group_cols), dropna=False, sort=False).indices
    for indices in grouped.values():
        idx = np.asarray(indices, dtype=int)
        group = X[idx]
        median = np.median(group, axis=0)
        absolute_deviation = np.abs(group - median)
        mad = np.median(absolute_deviation, axis=0)
        robust_z = np.divide(
            absolute_deviation,
            mad[None, :],
            out=np.where(
                absolute_deviation <= float(epsilon),
                0.0,
                np.inf,
            ),
            where=mad[None, :] > float(epsilon),
        )
        distances[idx] = np.median(robust_z, axis=1)

    out["spectral_robust_distance"] = distances
    out["spectral_outlier"] = (
        np.isfinite(distances) & (distances > float(threshold))
    ) | np.isinf(distances)
    return out


def build_segmentation_diagnostics_table(
    object_db: dict,
    image_db: dict,
    *,
    border_margin: int = 0,
    merge_warning_thresholds: dict | None = None,
) -> pd.DataFrame:
    object_qc = build_object_qc_table(
        object_db,
        image_db=image_db,
        include_geometry=True,
        border_margin=border_margin,
        merge_warning_thresholds=merge_warning_thresholds,
    ).set_index("object_id")
    rows = []
    for object_id, obj in object_db.items():
        qc = object_qc.loc[str(object_id)]
        bbox = obj.get("bbox")
        if bbox is None:
            bbox_area = np.nan
        else:
            min_row, min_col, max_row, max_col = map(int, bbox)
            bbox_area = int(max_row - min_row) * int(max_col - min_col)
        if bool(qc["too_small"]):
            segmentation_status = "excluded"
            segmentation_action = "exclude_too_small"
        elif bool(qc["possible_merged_object"]) or bool(qc["touches_border"]):
            segmentation_status = "warning"
            segmentation_action = (
                "review_possible_merge"
                if bool(qc["possible_merged_object"])
                else "review_border"
            )
        else:
            segmentation_status = "accepted"
            segmentation_action = "accept"
        image = image_db.get(str(qc["source_image"]), {})
        if image.get("segmentation_source") == "documented_override":
            segmentation_action = "documented_override"
        rows.append(
            {
                "clean_key": qc["source_image"],
                "label_id": obj.get("label_id"),
                "area_pixels": qc["area_pixels"],
                "bbox_area": bbox_area,
                "fill_ratio": qc["bbox_fill_ratio"],
                "touches_border": qc["touches_border"],
                "nearest_object_distance": qc["nearest_object_distance"],
                "segmentation_action": segmentation_action,
                "segmentation_status": segmentation_status,
            }
        )
    return pd.DataFrame(rows, columns=expcfg.SEGMENTATION_DIAGNOSTIC_COLUMNS)


def build_object_qc_warnings(object_qc_df: pd.DataFrame) -> pd.DataFrame:
    """Build object alerts from vectorized boolean masks."""
    n_nonfinite = (
        pd.to_numeric(object_qc_df["n_nan"], errors="coerce").fillna(0)
        + pd.to_numeric(object_qc_df["n_inf"], errors="coerce").fillna(0)
    )
    parts = [
        _alerts_from_mask(
            object_qc_df,
            n_nonfinite.gt(0),
            record_type="object",
            id_col="object_id",
            flag_type="non_finite_spectra",
            severity="error",
            qc_status="excluded",
            exclusion_reason="non_finite_spectra",
            warning="Object spectra contain NaN or infinite values.",
            evidence_cols=("n_nan", "n_inf"),
        ),
        _alerts_from_mask(
            object_qc_df,
            object_qc_df["too_small"].fillna(False),
            record_type="object",
            id_col="object_id",
            flag_type="too_small",
            severity="error",
            qc_status="excluded",
            exclusion_reason="area_below_minimum",
            requires_segmentation_review=True,
            warning="Object area is below the configured minimum.",
            evidence_cols=("area_pixels", "too_small"),
        ),
        _alerts_from_mask(
            object_qc_df,
            object_qc_df["possible_merged_object"].fillna(False),
            record_type="object",
            id_col="object_id",
            flag_type="possible_merged_object",
            requires_segmentation_review=True,
            warning="Geometry suggests a close or merged object.",
            evidence_cols=(
                "bbox_fill_ratio",
                "nearest_object_distance",
            ),
        ),
        _alerts_from_mask(
            object_qc_df,
            object_qc_df["touches_border"].fillna(False),
            record_type="object",
            id_col="object_id",
            flag_type="touches_border",
            warning="Object touches the image border.",
            evidence_cols=("touches_border",),
        ),
        _alerts_from_mask(
            object_qc_df,
            object_qc_df["spectral_outlier"].fillna(False),
            record_type="object",
            id_col="object_id",
            flag_type="robust_spectral_outlier",
            warning="Robust group-wise spectral distance exceeds the threshold.",
            evidence_cols=("spectral_robust_distance",),
        ),
    ]
    return pd.concat(parts, ignore_index=True).loc[:, QC_ALERT_COLUMNS]


def check_missing_required_fields(
    image_db: dict,
    object_db: dict,
    required_image_fields=DEFAULT_REQUIRED_IMAGE_FIELDS,
    required_object_fields=DEFAULT_REQUIRED_OBJECT_FIELDS,
) -> pd.DataFrame:
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


def build_object_shape_check_tables(
    object_db: dict,
    image_db: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact dimensional/axis checks and their failing subset."""
    rows = []
    reference_axis = _first_axis(image_db)
    for object_id, obj in object_db.items():
        source = obj.get("source_clean_key")
        img = image_db.get(source)
        spectra = np.asarray(obj.get("spectra"))
        positions = np.asarray(obj.get("positions_global"))
        axis = np.asarray(obj.get("wavelengths"))
        n_pixels = int(obj.get("n_pixels", -1))
        n_bands = int(obj.get("n_bands", -1))
        checks = {
            "spectra": spectra.ndim == 2 and spectra.shape == (n_pixels, n_bands),
            "positions": positions.ndim == 2 and positions.shape == (n_pixels, 2),
            "summary_lengths": all(
                len(np.asarray(obj.get(field))) == n_bands
                for field in ("mean_spectrum", "median_spectrum", "std_spectrum")
            ),
            "object_axis": axis.ndim == 1
            and len(axis) == n_bands
            and reference_axis is not None
            and np.array_equal(axis.astype(float), reference_axis),
            "image_bands": (
                img is not None
                and np.asarray(img.get("cube")).ndim == 3
                and np.asarray(img.get("cube")).shape[2] == n_bands
            ),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "object_id": str(object_id),
                "source_image": source,
                "n_pixels": n_pixels,
                "n_bands": n_bands,
                "all_shapes_valid": not failed_checks,
                "failed_checks": failed_checks,
            }
        )
    checks_df = pd.DataFrame(
        rows,
        columns=[
            "object_id",
            "source_image",
            "n_pixels",
            "n_bands",
            "all_shapes_valid",
            "failed_checks",
        ],
    )
    if checks_df.empty:
        return checks_df, checks_df.copy()
    return checks_df, checks_df.loc[~checks_df["all_shapes_valid"]].copy()


def build_qc_alerts_table(
    image_warnings_df: pd.DataFrame | None = None,
    object_warnings_df: pd.DataFrame | None = None,
    missing_fields_df: pd.DataFrame | None = None,
    bad_shape_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    parts = []
    for record_type, frame in (
        ("image", image_warnings_df),
        ("object", object_warnings_df),
    ):
        if frame is not None and not frame.empty:
            if set(QC_ALERT_COLUMNS).issubset(frame.columns):
                parts.append(frame.loc[:, QC_ALERT_COLUMNS].copy())
                continue
            id_col = "clean_key" if record_type == "image" else "object_id"
            parts.append(
                _alerts_from_mask(
                    frame,
                    np.ones(len(frame), dtype=bool),
                    record_type=record_type,
                    id_col=id_col,
                    flag_type=f"{record_type}_warning",
                    warning=frame.get(
                        "warning",
                        pd.Series("", index=frame.index),
                    ),
                    evidence_cols=tuple(
                        column
                        for column in frame.columns
                        if column not in {id_col, "warning"}
                    ),
                )
            )

    if missing_fields_df is not None and not missing_fields_df.empty:
        for record_type, group in missing_fields_df.groupby(
            "record_type",
            dropna=False,
        ):
            parts.append(
                _alerts_from_mask(
                    group,
                    np.ones(len(group), dtype=bool),
                    record_type=str(record_type),
                    id_col="record_id",
                    flag_type="missing_fields",
                    severity="error",
                    qc_status="excluded",
                    exclusion_reason="missing_required_fields",
                    warning=(
                        "Missing required fields: "
                        + group["missing_fields"].astype(str)
                    ),
                    evidence_cols=("missing_fields",),
                )
            )

    if bad_shape_df is not None and not bad_shape_df.empty:
        parts.append(
            _alerts_from_mask(
                bad_shape_df,
                np.ones(len(bad_shape_df), dtype=bool),
                record_type="object",
                id_col="object_id",
                flag_type="bad_shape",
                severity="error",
                qc_status="excluded",
                exclusion_reason="dimensional_inconsistency",
                warning=(
                    "Failed checks: "
                    + bad_shape_df["failed_checks"].astype(str)
                )
                ,
                evidence_cols=("failed_checks",),
            )
        )

    if not parts:
        return pd.DataFrame(columns=QC_ALERT_COLUMNS)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.loc[:, QC_ALERT_COLUMNS].drop_duplicates().reset_index(drop=True)
    if out["alert_id"].duplicated().any():
        duplicates = out.loc[
            out["alert_id"].duplicated(keep=False),
            ["alert_id", "record_type", "record_id", "flag_type"],
        ].to_dict("records")
        raise RuntimeError(f"Duplicate QC alert identifiers: {duplicates}")
    return out


def build_qc_flags_table(*args, **kwargs) -> pd.DataFrame:
    """Deprecated compatibility wrapper; canonical output is qc_alerts."""
    warnings.warn(
        "build_qc_flags_table is deprecated; use build_qc_alerts_table.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_qc_alerts_table(*args, **kwargs)


def build_qc_review_table(
    qc_flags_df: pd.DataFrame,
    *,
    overrides: tuple[Mapping, ...] | list[Mapping] | None = None,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Build and validate the auditable manual-review table for QC flags."""
    missing = [
        column for column in QC_REVIEW_KEY_COLUMNS
        if column not in qc_flags_df
    ]
    if missing:
        raise KeyError(f"QC flags are missing review keys: {missing}")

    review = (
        qc_flags_df.loc[:, list(QC_REVIEW_KEY_COLUMNS)]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    review["review_status"] = "pending"
    review["review_decision"] = ""
    review["reviewer"] = ""
    review["review_date"] = ""
    review["review_comment"] = ""
    review["review_evidence"] = ""

    for override in tuple(overrides or ()):
        unknown = set(override).difference(QC_REVIEW_COLUMNS)
        if unknown:
            raise KeyError(f"Unknown QC review columns: {sorted(unknown)}")
        mask = pd.Series(True, index=review.index)
        for key in QC_REVIEW_KEY_COLUMNS:
            if key not in override:
                raise KeyError(
                    f"QC review override is missing key {key!r}: {override}"
                )
            mask &= review[key].astype(str).eq(str(override[key]))
        if int(mask.sum()) != 1:
            raise KeyError(
                "QC review override must match exactly one flag: "
                f"{override}"
            )
        for column, value in override.items():
            if column not in QC_REVIEW_KEY_COLUMNS:
                review.loc[mask, column] = value

    reviewed = review["review_status"].eq(expcfg.QC_REVIEW_REQUIRED_STATUS)
    invalid_decision = reviewed & ~review["review_decision"].isin(
        expcfg.QC_REVIEW_ALLOWED_DECISIONS
    )
    if invalid_decision.any():
        invalid = review.loc[
            invalid_decision,
            [*QC_REVIEW_KEY_COLUMNS, "review_decision"],
        ].to_dict("records")
        raise ValueError(f"Reviewed QC flags have invalid decisions: {invalid}")
    if require_complete:
        validate_qc_review_closure(review)
    return review.loc[:, list(QC_REVIEW_COLUMNS)]


def merge_existing_reviews_or_initialize(
    qc_alerts_df: pd.DataFrame,
    review_source=None,
) -> pd.DataFrame:
    """Initialize pending reviews, then merge any versioned human decisions."""
    review = build_qc_review_table(qc_alerts_df, require_complete=False)
    if review_source is None:
        return review
    if isinstance(review_source, pd.DataFrame):
        existing = review_source.copy()
    else:
        path = Path(review_source)
        if not path.exists():
            return review
        existing = pd.read_parquet(path)
    missing = [
        column for column in QC_REVIEW_COLUMNS if column not in existing.columns
    ]
    if missing:
        raise KeyError(f"Existing QC review is missing columns: {missing}")
    if existing.duplicated(list(QC_REVIEW_KEY_COLUMNS)).any():
        raise RuntimeError("Existing QC review contains duplicate alert keys.")
    payload_columns = [
        column
        for column in QC_REVIEW_COLUMNS
        if column not in QC_REVIEW_KEY_COLUMNS
    ]
    merged = review.merge(
        existing.loc[:, list(QC_REVIEW_COLUMNS)],
        on=list(QC_REVIEW_KEY_COLUMNS),
        how="left",
        suffixes=("", "_existing"),
        validate="one_to_one",
    )
    for column in payload_columns:
        existing_column = f"{column}_existing"
        available = merged[existing_column].notna()
        merged.loc[available, column] = merged.loc[
            available,
            existing_column,
        ]
        merged = merged.drop(columns=existing_column)
    return merged.loc[:, list(QC_REVIEW_COLUMNS)]


def validate_qc_review_closure(qc_review_df: pd.DataFrame) -> bool:
    """Require a documented, final decision for every QC alert."""
    missing = [
        column for column in QC_REVIEW_COLUMNS if column not in qc_review_df
    ]
    if missing:
        raise KeyError(f"QC review table is missing columns: {missing}")
    if qc_review_df.duplicated(list(QC_REVIEW_KEY_COLUMNS)).any():
        raise RuntimeError("QC review table contains duplicate alert keys.")
    reviewed = qc_review_df["review_status"].eq(
        expcfg.QC_REVIEW_REQUIRED_STATUS
    )
    valid_decision = qc_review_df["review_decision"].isin(
        expcfg.QC_REVIEW_ALLOWED_DECISIONS
    )
    required_text = (
        qc_review_df.loc[
            :,
            [
                "reviewer",
                "review_date",
                "review_comment",
                "review_evidence",
            ],
        ]
        .fillna("")
        .astype(str)
        .apply(lambda series: series.str.strip().ne(""))
        .all(axis=1)
    )
    valid_date = pd.to_datetime(
        qc_review_df["review_date"],
        errors="coerce",
    ).notna()
    complete = reviewed & valid_decision & required_text & valid_date
    if not complete.all():
        failures = qc_review_df.loc[
            ~complete,
            list(QC_REVIEW_COLUMNS),
        ].to_dict("records")
        raise RuntimeError(
            "QC review closure is blocking; every reviewed alert requires "
            "a valid decision, reviewer, date, justification and evidence: "
            f"{failures}"
        )
    return True


def apply_qc_reviews(
    qc_flags_df: pd.DataFrame,
    qc_review_df: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Apply explicit review decisions while preserving every raw QC flag."""
    if qc_flags_df.empty:
        return pd.DataFrame(columns=QC_RESOLVED_FLAG_COLUMNS)
    missing = [
        column for column in QC_REVIEW_COLUMNS
        if column not in qc_review_df
    ]
    if missing:
        raise KeyError(f"QC review table is missing columns: {missing}")
    if qc_review_df.duplicated(list(QC_REVIEW_KEY_COLUMNS)).any():
        raise RuntimeError("QC review table contains duplicate flag keys.")

    out = qc_flags_df.merge(
        qc_review_df.loc[:, list(QC_REVIEW_COLUMNS)],
        on=list(QC_REVIEW_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    out["review_status"] = out["review_status"].fillna("pending")
    for column in (
        "review_decision",
        "reviewer",
        "review_date",
        "review_comment",
        "review_evidence",
    ):
        out[column] = out[column].fillna("")

    reviewed = out["review_status"].eq(expcfg.QC_REVIEW_REQUIRED_STATUS)
    if require_complete:
        validate_qc_review_closure(
            out.loc[:, list(QC_REVIEW_COLUMNS)]
        )

    accepted = reviewed & out["review_decision"].eq("accept_as_is")
    excluded = reviewed & out["review_decision"].eq("exclude")
    corrected = reviewed & out["review_decision"].eq(
        "correct_segmentation"
    )
    out.loc[accepted, "qc_status"] = "accepted_after_review"
    out.loc[accepted, "requires_segmentation_review"] = False
    out.loc[excluded, "qc_status"] = "excluded"
    out.loc[excluded & out["exclusion_reason"].eq(""), "exclusion_reason"] = (
        "manual_visual_review"
    )
    out.loc[excluded, "requires_segmentation_review"] = False
    out.loc[corrected, "qc_status"] = "corrected_segmentation"
    out.loc[corrected, "requires_segmentation_review"] = True
    return out.loc[:, list(QC_RESOLVED_FLAG_COLUMNS)]


def build_qc_merge_review_figure(
    qc_flags_df: pd.DataFrame,
    object_db: Mapping,
    image_db: Mapping,
    object_qc_df: pd.DataFrame,
):
    """Return a persistent visual audit figure for possible-merge flags."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    flagged = qc_flags_df.loc[
        qc_flags_df["flag_type"].eq("possible_merged_object")
    ].copy()
    if flagged.empty:
        return None
    qc_by_object = object_qc_df.set_index("object_id")
    figure, axes = plt.subplots(
        len(flagged),
        3,
        figsize=(13, 3.3 * len(flagged)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, object_id in enumerate(
        flagged["record_id"].astype(str)
    ):
        obj = object_db[object_id]
        image = image_db[obj["source_clean_key"]]
        image_ref = np.asarray(image["image_ref"], dtype=float)
        labels = np.asarray(image["labels"])
        label_id = int(obj["label_id"])
        min_row, min_col, max_row, max_col = map(int, obj["bbox"])
        margin = 18
        rows = slice(
            max(0, min_row - margin),
            min(image_ref.shape[0], max_row + margin),
        )
        columns = slice(
            max(0, min_col - margin),
            min(image_ref.shape[1], max_col + margin),
        )
        crop = np.squeeze(image_ref[rows, columns])
        if crop.ndim == 3:
            crop = crop[..., :3]
        label_crop = labels[rows, columns]
        object_mask = label_crop == label_id

        axis = axes[row_index, 0]
        axis.imshow(crop, cmap="gray")
        axis.contour(
            object_mask,
            levels=[0.5],
            colors="red",
            linewidths=2,
        )
        axis.set_title(f"{object_id}\nlocal view + target outline")
        axis.axis("off")

        axis = axes[row_index, 1]
        axis.imshow(crop, cmap="gray")
        overlay = np.ma.masked_where(label_crop == 0, label_crop)
        axis.imshow(
            overlay,
            cmap="turbo",
            alpha=0.45,
            interpolation="nearest",
        )
        axis.contour(
            object_mask,
            levels=[0.5],
            colors="white",
            linewidths=2,
        )
        axis.set_title(f"{obj['source_clean_key']} - neighboring labels")
        axis.axis("off")

        axis = axes[row_index, 2]
        axis.imshow(
            object_mask,
            cmap=ListedColormap(["white", "#c51b7d"]),
            interpolation="nearest",
        )
        qc = qc_by_object.loc[object_id]
        diagnostics = (
            f"area={int(qc['area_pixels'])} px\n"
            f"fill_ratio={float(qc['bbox_fill_ratio']):.3f}\n"
            f"nearest={float(qc['nearest_object_distance']):.2f} px\n"
            f"border={bool(qc['touches_border'])}\n"
            f"label={label_id}"
        )
        axis.set_title("binary mask")
        axis.text(
            1.03,
            0.5,
            diagnostics,
            transform=axis.transAxes,
            va="center",
            fontsize=10,
        )
        axis.axis("off")
    figure.suptitle(
        "Visual review of QC flags - possible merged objects",
        fontsize=16,
    )
    return figure


def build_qc_visual_review_report(
    qc_alerts_df: pd.DataFrame,
    object_db: Mapping,
    image_db: Mapping,
    object_qc_df: pd.DataFrame,
    output_path,
) -> Path:
    """Write one systematic multi-page PDF for the complete QC review."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    object_qc = object_qc_df.copy()
    with PdfPages(output_path) as pdf:
        figure, axis = plt.subplots(figsize=(11.7, 8.3))
        axis.axis("off")
        summary_lines = [
            "HSI Nuts - systematic QC visual review",
            f"Images: {len(image_db)}",
            f"Objects: {len(object_db)}",
            f"Alerts: {len(qc_alerts_df)}",
            "",
            "Alert counts:",
        ]
        if qc_alerts_df.empty:
            summary_lines.append("  none")
        else:
            summary_lines.extend(
                f"  {name}: {count}"
                for name, count in qc_alerts_df["flag_type"]
                .value_counts()
                .sort_index()
                .items()
            )
        axis.text(0.03, 0.97, "\n".join(summary_lines), va="top")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        # Representative images by class and batch.
        grouped_images = {}
        for image_id, image in image_db.items():
            key = (image.get("nut_type"), image.get("batch"))
            grouped_images.setdefault(key, (image_id, image))
        representatives = list(grouped_images.values())[:12]
        if representatives:
            ncols = 3
            nrows = int(np.ceil(len(representatives) / ncols))
            figure, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(11.7, max(4.0, 3.2 * nrows)),
                squeeze=False,
            )
            for axis, (image_id, image) in zip(
                axes.ravel(),
                representatives,
            ):
                axis.imshow(np.asarray(image["image_ref"]), cmap="gray")
                labels = np.asarray(image.get("labels"))
                if labels.ndim == 2 and np.any(labels > 0):
                    axis.contour(labels > 0, levels=[0.5], colors="red")
                axis.set_title(
                    f"{image_id}\n{image.get('nut_type')} / "
                    f"batch {image.get('batch')}"
                )
                axis.axis("off")
            for axis in axes.ravel()[len(representatives):]:
                axis.axis("off")
            figure.suptitle("Representative images by class and batch")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        # Small, median and large objects plus flagged geometries.
        if not object_qc.empty:
            ordered = object_qc.sort_values("area_pixels")
            picks = ordered.iloc[
                sorted(
                    set(
                        [
                            0,
                            len(ordered) // 2,
                            len(ordered) - 1,
                        ]
                    )
                )
            ]
            figure, axes = plt.subplots(
                len(picks),
                2,
                figsize=(11.7, 3.2 * len(picks)),
                squeeze=False,
            )
            for row_index, row in enumerate(picks.itertuples()):
                obj = object_db[str(row.object_id)]
                source = obj["source_clean_key"]
                image = image_db[source]
                labels = np.asarray(image["labels"])
                label_mask = labels == int(obj["label_id"])
                axes[row_index, 0].imshow(
                    np.asarray(image["image_ref"]),
                    cmap="gray",
                )
                axes[row_index, 0].contour(
                    label_mask,
                    levels=[0.5],
                    colors="red",
                )
                axes[row_index, 0].set_title(
                    f"{row.object_id} - area={row.area_pixels}"
                )
                axes[row_index, 0].axis("off")
                axes[row_index, 1].plot(
                    np.asarray(obj["mean_spectrum"], dtype=float)
                )
                axes[row_index, 1].set_title(
                    "Mean spectrum "
                    f"(robust distance={row.spectral_robust_distance:.3g})"
                )
            figure.suptitle("Small, median and large object review")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        # One evidence page per alert keeps the review exhaustive.
        for alert in qc_alerts_df.itertuples():
            figure, axes = plt.subplots(1, 2, figsize=(11.7, 5.8))
            if alert.record_type == "object" and alert.record_id in object_db:
                obj = object_db[alert.record_id]
                image = image_db[obj["source_clean_key"]]
                labels = np.asarray(image["labels"])
                mask = labels == int(obj["label_id"])
                axes[0].imshow(np.asarray(image["image_ref"]), cmap="gray")
                axes[0].contour(mask, levels=[0.5], colors="red")
                axes[0].axis("off")
                axes[1].plot(np.asarray(obj["mean_spectrum"], dtype=float))
                axes[1].set_title("Mean spectrum")
            elif alert.record_type == "image" and alert.record_id in image_db:
                image = image_db[alert.record_id]
                axes[0].imshow(np.asarray(image["image_ref"]), cmap="gray")
                axes[0].axis("off")
                axes[1].imshow(np.asarray(image["labels"]))
                axes[1].axis("off")
            else:
                axes[0].axis("off")
                axes[1].axis("off")
            figure.suptitle(
                f"{alert.flag_type}: {alert.record_id}\n"
                f"{alert.warning}\nEvidence: {alert.evidence_json}"
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return output_path


def build_qc_exclusion_report(qc_flags_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(expcfg.QC_EXCLUSION_OUTPUT_COLUMNS)
    if qc_flags_df.empty:
        return pd.DataFrame(columns=columns)
    mask = qc_flags_df["qc_status"].isin(
        ["excluded", "corrected_segmentation"]
    )
    return (
        qc_flags_df.loc[mask, columns]
        .drop_duplicates()
        .sort_values(["record_type", "record_id"])
        .reset_index(drop=True)
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    records = json.loads(
        frame.sort_index(axis=1).to_json(
            orient="records",
            date_format="iso",
        )
    )
    return sha256_payload(records)


def build_qc_protocol(
    qc_alerts_df: pd.DataFrame,
    qc_review_df: pd.DataFrame,
    exclusion_manifest: pd.DataFrame,
    *,
    protocol_version=expcfg.PROTOCOL_VERSION,
    pixel_exclusion_manifest=None,
    qc_policy=None,
    spectral_pixel_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
) -> pd.DataFrame:
    """Build the one-row QC closure contract."""
    policy = expcfg.QC_POLICY if qc_policy is None else qc_policy
    pixel_exclusion_manifest = (
        pd.DataFrame()
        if pixel_exclusion_manifest is None
        else pixel_exclusion_manifest
    )
    n_pending = int(
        (~qc_review_df["review_status"].eq(
            expcfg.QC_REVIEW_REQUIRED_STATUS
        )).sum()
    )
    return pd.DataFrame(
        [
            {
                "protocol_version": str(protocol_version),
                "qc_policy_hash": sha256_payload(policy),
                "spectral_pixel_policy_hash": sha256_payload(spectral_pixel_policy),
                "pixel_exclusion_hash": _frame_hash(pixel_exclusion_manifest),
                "alerts_hash": _frame_hash(qc_alerts_df),
                "review_hash": _frame_hash(qc_review_df),
                "n_alerts": int(len(qc_alerts_df)),
                "n_pending": n_pending,
                "n_excluded": int(len(exclusion_manifest)),
                "n_pixel_excluded": int(len(pixel_exclusion_manifest)),
                "closure_status": (
                    "closed" if n_pending == 0 else "pending"
                ),
            }
        ],
        columns=expcfg.QC_PROTOCOL_OUTPUT_COLUMNS,
    )


def qc_requires_new_cycle(qc_flags_df: pd.DataFrame) -> bool:
    if qc_flags_df.empty:
        return False
    if "review_status" in qc_flags_df:
        pending_review = ~qc_flags_df["review_status"].eq(
            expcfg.QC_REVIEW_REQUIRED_STATUS
        )
    else:
        pending_review = qc_flags_df[
            "requires_segmentation_review"
        ].fillna(False)
    return bool(
        pending_review.any()
        or qc_flags_df["qc_status"].eq("corrected_segmentation").any()
    )


def build_terminal_band_qc_table(
    object_db: Mapping,
    *,
    raw_band_indices=None,
    policy=None,
) -> pd.DataFrame:
    """
    Diagnose reflectance integrity band by band, with emphasis on
    terminal retained wavelengths.

    All-zero spectra can be excluded from this diagnostic because they
    represent pixel-level no-data rather than a band-specific defect.
    """
    policy = dict(
        expcfg.TERMINAL_BAND_QC_POLICY
        if policy is None
        else policy
    )

    spectra_parts = []
    reference_wavelengths = None

    for object_id, obj in object_db.items():
        spectra = np.asarray(obj.get("spectra"), dtype=float)

        if spectra.ndim != 2:
            raise ValueError(
                f"Object {object_id!r}: spectra must be 2D, "
                f"got shape={spectra.shape}."
            )

        wavelengths = np.asarray(obj.get("wavelengths"), dtype=float)

        if reference_wavelengths is None:
            reference_wavelengths = wavelengths
        elif not np.array_equal(wavelengths, reference_wavelengths):
            raise ValueError(
                f"Inconsistent wavelength axis for object {object_id!r}."
            )

        spectra_parts.append(spectra)

    if not spectra_parts:
        return pd.DataFrame()

    X = np.vstack(spectra_parts)

    finite_rows = np.isfinite(X).all(axis=1)

    if policy.get(
        "exclude_all_zero_pixels_from_diagnostics",
        True,
    ):
        all_zero_rows = np.all(X == 0.0, axis=1)
    else:
        all_zero_rows = np.zeros(len(X), dtype=bool)

    diagnostic_rows = finite_rows & ~all_zero_rows
    X_qc = X[diagnostic_rows]

    if len(X_qc) == 0:
        raise ValueError(
            "No valid spectrum is available for terminal-band QC."
        )

    n_bands = X_qc.shape[1]
    n_terminal = min(
        int(policy.get("n_terminal_bands", 5)),
        n_bands,
    )

    if raw_band_indices is None:
        raw_band_indices = np.arange(n_bands)
    else:
        raw_band_indices = np.asarray(raw_band_indices, dtype=int)

    if len(raw_band_indices) != n_bands:
        raise ValueError(
            "raw_band_indices must match the retained spectral axis."
        )

    rows = []

    for j in range(n_bands):
        values = X_qc[:, j]

        n_negative = int(np.count_nonzero(values < 0.0))
        n_zero = int(np.count_nonzero(values == 0.0))
        n_nonpositive = int(np.count_nonzero(values <= 0.0))

        is_terminal = j >= n_bands - n_terminal

        failed = bool(
            is_terminal
            and policy.get("flag_any_negative_reflectance", True)
            and n_negative > 0
        )

        rows.append(
            {
                "processed_band_index": int(j),
                "raw_band_index": int(raw_band_indices[j]),
                "wavelength_nm": float(reference_wavelengths[j]),
                "n_pixels_evaluated": int(len(values)),
                "n_negative": n_negative,
                "negative_rate": float(n_negative / len(values)),
                "n_zero": n_zero,
                "zero_rate": float(n_zero / len(values)),
                "n_nonpositive": n_nonpositive,
                "min_reflectance": float(np.min(values)),
                "q001_reflectance": float(
                    np.quantile(values, 0.001)
                ),
                "q01_reflectance": float(
                    np.quantile(values, 0.01)
                ),
                "is_terminal": is_terminal,
                "terminal_qc_status": (
                    "fail" if failed else "pass"
                ),
            }
        )

    return pd.DataFrame(rows)


def assert_terminal_bands_valid(
    terminal_band_qc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Block database freezing when a retained terminal band fails QC."""
    failed = terminal_band_qc_df.loc[
        terminal_band_qc_df["is_terminal"]
        & terminal_band_qc_df["terminal_qc_status"].eq("fail")
    ]

    if not failed.empty:
        details = failed[
            [
                "raw_band_index",
                "wavelength_nm",
                "n_negative",
                "min_reflectance",
            ]
        ].to_dict("records")

        raise RuntimeError(
            "Retained terminal spectral bands failed QC. "
            f"Review N_STOP_END before freezing the database: {details}"
        )

    return terminal_band_qc_df


def build_pixel_spectral_qc_table(
    object_db: Mapping,
    *,
    policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
)->pd.DataFrame:
    """
    Diagnose reflectance integrity pixel by pixel, with emphasis on
    spectral outliers.

    All-zero spectra can be excluded from this diagnostic because they
    represent pixel-level no-data rather than a band-specific defect.
    """
    rows = []

    for object_id, obj in object_db.items():
        spectra = np.asarray(obj.get("spectra"), dtype=float)
        positions = np.asarray(obj.get("positions_global"))

        if spectra.ndim != 2:
            raise ValueError(
                f"Object {object_id!r}: spectra must be 2D, "
                f"got shape={spectra.shape}."
            )
        if (
            positions.ndim !=2
            or positions.shape[1] !=2
            or len(positions) != len(spectra)
        ):
            raise ValueError(
                f"Object {object_id!r}: Inconsistent shape for positions."
            )
        
        report = spectral_pixel_validity_report(spectra, policy=policy)

        finite_values = np.isfinite(spectra)
        min_reflectance = np.min(
            np.where(finite_values, spectra, np.inf), axis=1
        )
        min_reflectance[~np.isfinite(min_reflectance)] = np.nan

        for pixel_index in range(len(spectra)):
            rows.append(
                {
                    "object_id": str(object_id),
                    "source_image": str(obj.get("source_clean_key", obj.get("source_image"))),
                    "batch": obj.get("batch"),
                    "label": obj.get("object_nut_type"),
                    "pixel_index": int(pixel_index),
                    "row": int(positions[pixel_index, 0]),
                    "col": int(positions[pixel_index, 1]),
                    "n_bands": int(spectra.shape[1]),
                    "n_zero": int(report['n_zero'][pixel_index]),
                    "n_nonpositive": int(report["n_nonpositive"][pixel_index]),
                    "zero_fraction": float(
                        report["n_zero"][pixel_index]
                        / spectra.shape[1]
                    ),
                    "min_reflectance": float(
                        min_reflectance[pixel_index]
                    ),
                    "finite": bool(
                        report["finite"][pixel_index]
                    ),
                    "all_zero_spectrum": bool(
                        report["all_zero"][pixel_index]
                    ),
                    "has_nonpositive_reflectance": bool(
                        report["has_nonpositive"][pixel_index]
                    ),
                    "analysis_valid": bool(
                        report["valid_mask"][pixel_index]
                    ),
                    "invalid_reason": str(
                        report["reason"][pixel_index]
                    ),
                }
            )

    return pd.DataFrame(rows, columns=expcfg.PIXEL_SPECTRAL_QC_COLUMNS)
