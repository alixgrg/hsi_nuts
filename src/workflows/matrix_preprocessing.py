from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.matrices.redim_matrix import select_balanced_pixel_indices
from src.spectra.band_selection import spectral_pixel_validity_report
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.preprocessing_configs import (
    normalize_preprocessing_configs,
    preprocessing_derivative,
    preprocessing_name_from_steps,
    validate_preprocessing_steps,
)
from src.utils import filter_records, require_columns
from src.workflows.protocol_split import (
    PROTOCOL_SPLIT_CHECK_COLUMNS as PROTOCOL_CHECK_COLUMNS,
    PROTOCOL_SPLIT_MANIFEST_COLUMNS as PROTOCOL_MANIFEST_COLUMNS,
    build_protocol_manifest,
)


MATRIX_COVERAGE_COLUMNS = expcfg.MATRIX_COVERAGE_COLUMNS
BALANCED_SAMPLING_COLUMNS = expcfg.M_FEASIBILITY_COLUMNS
PIXEL_SAMPLING_DIAGNOSTIC_COLUMNS = (
    expcfg.PIXEL_SAMPLING_DIAGNOSTIC_COLUMNS
)
PREPROCESSING_ERROR_COLUMNS = expcfg.PREPROCESSING_ERROR_COLUMNS


def wavelength_axis_id(wavelengths) -> str:
    axis = np.ascontiguousarray(np.asarray(wavelengths, dtype="<f8"))
    payload = len(axis).to_bytes(8, byteorder="little", signed=False) + axis.tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_wavelength_config(
    image_db,
    object_db,
    *,
    wavelength_mode: str,
    protocol_version: str = expcfg.PROTOCOL_VERSION,
    spectral_config_id: str | None = None,
    n_remove_start: int = expcfg.N_REMOVE_START,
    n_stop_end: int | None = expcfg.N_STOP_END,
    window_min_nm: float | None = None,
    window_max_nm: float | None = None,
    locked: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Validate every image/object axis and return one locked config row."""
    axes = []
    for record_type, records in (("image", image_db), ("object", object_db)):
        for record_id, record in records.items():
            axis = record.get("wavelengths")
            if axis is None:
                if strict:
                    raise ValueError(f"{record_type} {record_id!r} has no wavelength axis.")
                continue
            axes.append((record_type, str(record_id), np.asarray(axis, dtype=float)))
    if not axes:
        raise ValueError("No wavelength axis found in the database.")

    reference = axes[0][2]
    strictly_increasing = bool(np.all(np.diff(reference) > 0))
    unique_axis = bool(np.unique(reference).size == reference.size)
    if strict and not strictly_increasing:
        raise ValueError("The wavelength axis must be strictly increasing.")
    mismatches = [
        f"{record_type}:{record_id}"
        for record_type, record_id, axis in axes
        if not np.array_equal(axis, reference)
    ]
    if strict and mismatches:
        raise ValueError(f"Inconsistent wavelength axes: {mismatches[:10]}")

    image_band_mismatches = [
        str(record_id)
        for record_id, record in image_db.items()
        if np.asarray(record.get("cube")).ndim != 3
        or np.asarray(record.get("cube")).shape[2] != len(reference)
    ]
    object_band_mismatches = [
        str(record_id)
        for record_id, record in object_db.items()
        if np.asarray(record.get("spectra")).ndim != 2
        or np.asarray(record.get("spectra")).shape[1] != len(reference)
    ]
    if strict and (image_band_mismatches or object_band_mismatches):
        raise ValueError(
            "Spectral band-count mismatch: "
            f"images={image_band_mismatches[:10]}, objects={object_band_mismatches[:10]}"
        )

    axis_id = wavelength_axis_id(reference)
    if spectral_config_id is None:
        spectral_config_id = hashlib.sha256(
            (
                f"{protocol_version}|{wavelength_mode}|{axis_id}|"
                f"{int(n_remove_start)}|{n_stop_end}|"
                f"{window_min_nm}|{window_max_nm}"
            ).encode("utf-8")
        ).hexdigest()
    return pd.DataFrame(
        [
            {
                "protocol_version": str(protocol_version),
                "spectral_config_id": str(spectral_config_id),
                "wavelength_mode": str(wavelength_mode),
                "wavelength_axis_id": axis_id,
                "n_remove_start": int(n_remove_start),
                "n_stop_end": (
                    None if n_stop_end is None else int(n_stop_end)
                ),
                "window_min_nm": window_min_nm,
                "window_max_nm": window_max_nm,
                "n_bands": int(len(reference)),
                "min_wavelength_nm": float(np.min(reference)),
                "max_wavelength_nm": float(np.max(reference)),
                "n_images_checked": int(len(image_db)),
                "n_objects_checked": int(len(object_db)),
                "all_axes_match": not mismatches,
                "strictly_increasing": strictly_increasing,
                "unique_axis": unique_axis,
                "locked": bool(locked),
            }
        ]
    )


def assert_wavelength_lock(
    existing: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Reject a spectral-axis/configuration change under one protocol version."""
    required = {
        "protocol_version",
        "spectral_config_id",
        "wavelength_axis_id",
        "locked",
    }
    for name, frame in (("existing", existing), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing or len(frame) != 1:
            raise ValueError(
                f"{name} wavelength lock must contain one row and columns "
                f"{sorted(required)}; missing={sorted(missing)}."
            )
    old = existing.iloc[0]
    new = candidate.iloc[0]
    if str(old["protocol_version"]) != str(new["protocol_version"]):
        return
    if (
        str(old["spectral_config_id"]) != str(new["spectral_config_id"])
        or str(old["wavelength_axis_id"]) != str(new["wavelength_axis_id"])
    ):
        raise RuntimeError(
            "Spectral axis/configuration changed without a new protocol version."
        )


def summarize_matrix_output(
    X,
    y,
    meta,
    matrix_method: str,
    filters: dict | None = None,
    balanced_pixel_strategy: str | None = None,
    *,
    matrix_id: str | None = None,
    protocol_role: str = "unspecified",
    wavelengths=None,
    zero_variance_epsilon: float = expcfg.PREPROCESSING_ZERO_VARIANCE_EPSILON,
):
    """Return one compact matrix summary row and aligned metadata dataframe."""
    del filters  # kept for backward-compatible calls; filters are encoded by role.
    X = np.asarray(X)
    y = np.asarray(y)
    meta_df = pd.DataFrame(meta)
    numeric = np.issubdtype(X.dtype, np.number)
    n_nan = int(np.isnan(X).sum()) if numeric else np.nan
    n_inf = int(np.isinf(X).sum()) if numeric else np.nan
    n_classes = int(pd.Series(y).dropna().nunique())
    matrix_rank = (
        int(np.linalg.matrix_rank(X))
        if X.ndim == 2 and X.shape[0] and X.shape[1] and numeric
        else 0
    )
    n_zero_variance_bands = (
        int(
            np.count_nonzero(
                np.var(X, axis=0) <= float(zero_variance_epsilon)
            )
        )
        if X.ndim == 2 and X.shape[0] and numeric
        else 0
    )
    status = (
        "accepted"
        if X.ndim == 2
        and X.shape[0] > 0
        and X.shape[1] > 0
        and numeric
        and n_nan == 0
        and n_inf == 0
        and n_classes >= 2
        else "invalid"
    )
    if matrix_id is None:
        suffix = (
            f"_{balanced_pixel_strategy}"
            if balanced_pixel_strategy is not None
            else ""
        )
        matrix_id = f"{protocol_role}_{matrix_method}{suffix}"
    row = {
        "matrix_id": str(matrix_id),
        "protocol_role": str(protocol_role),
        "matrix_method": str(matrix_method),
        "balanced_pixel_strategy": balanced_pixel_strategy,
        "n_observations": int(X.shape[0]) if X.ndim else 0,
        "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
        "n_classes": n_classes,
        "n_objects": (
            int(meta_df["object_id"].nunique())
            if "object_id" in meta_df
            else 0
        ),
        "n_images": (
            int(meta_df["source_image"].nunique())
            if "source_image" in meta_df
            else 0
        ),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "matrix_rank": matrix_rank,
        "rank_ratio": (
            float(matrix_rank / min(X.shape))
            if X.ndim == 2 and min(X.shape) > 0
            else 0.0
        ),
        "n_zero_variance_bands": n_zero_variance_bands,
        "wavelength_axis_id": (
            wavelength_axis_id(wavelengths)
            if wavelengths is not None
            else None
        ),
        "status": status,
    }
    return row, meta_df


def build_matrix_coverage_table(meta, *, matrix_id: str) -> pd.DataFrame:
    meta_df = pd.DataFrame(meta)
    require_columns(
        meta_df,
        expcfg.MATRIX_REQUIRED_METADATA,
        name=f"{matrix_id} metadata",
    )
    return (
        meta_df.groupby(
            ["object_id", "source_image", "batch", "label", "sample_kind"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_rows")
        .assign(matrix_id=str(matrix_id))
        .loc[:, list(MATRIX_COVERAGE_COLUMNS)]
    )


def _balance_ratio(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False)
    if counts.empty or counts.max() == 0:
        return np.nan
    return float(counts.min() / counts.max())


def _mean_selection_overlap(selection_runs) -> float:
    overlaps = []
    for left, right in combinations(selection_runs, 2):
        for object_id in set(left) & set(right):
            union = left[object_id] | right[object_id]
            if union:
                overlaps.append(len(left[object_id] & right[object_id]) / len(union))
    return float(np.mean(overlaps)) if overlaps else np.nan


def _balance_ratio_from_counts(counts: Mapping[object, int]) -> float:
    positive = [
        int(count)
        for count in counts.values()
        if int(count) > 0
    ]
    if not positive:
        return np.nan
    return float(min(positive) / max(positive))


def _evaluate_balanced_sampling_run(
    selected_objects,
    *,
    m: int,
    strategy: str,
    seed: int,
    replace: bool,
    under_m_policy: str,
    pixel_validity_policy,
):
    """Evaluate one sampling seed without rebuilding the spectral matrix."""
    diagnostic_rows = []
    selections: dict[str, set[int]] = {}
    class_counts: dict[object, int] = {}
    image_counts: dict[object, int] = {}
    n_rows = 0

    try:
        for object_id, obj in selected_objects:
            indices, diagnostic = select_balanced_pixel_indices(
                obj,
                m=m,
                random_state=seed,
                object_id=str(object_id),
                replace=replace,
                balanced_pixel_strategy=strategy,
                under_m_policy=under_m_policy,
                return_diagnostics=True,
                pixel_validity_policy=pixel_validity_policy,
            )

            diagnostic_rows.append(
                {
                    "m": int(m),
                    "strategy": str(strategy),
                    "seed": int(seed),
                    **{
                        key: diagnostic[key]
                        for key in (
                            "object_id",
                            "n_raw",
                            "n_available",
                            "n_invalid",
                            "n_selected",
                            "selection_hash",
                            "status",
                        )
                    },
                }
            )

            if indices is None or len(indices) == 0:
                continue

            indices = np.asarray(indices, dtype=int)
            object_key = str(object_id)
            selections[object_key] = set(indices.tolist())

            n_selected = int(indices.size)
            n_rows += n_selected

            label = obj.get("object_nut_type")
            image = obj.get(
                "source_clean_key",
                obj.get("source_image"),
            )
            class_counts[label] = (
                class_counts.get(label, 0)
                + n_selected
            )
            image_counts[image] = (
                image_counts.get(image, 0)
                + n_selected
            )

        if n_rows == 0:
            raise ValueError(
                "No balanced pixels were selected."
            )
        if len(class_counts) < 2:
            raise ValueError(
                "Balanced sampling produced fewer than two classes."
            )

        metrics = {
            "n_rows": n_rows,
            "n_classes": len(class_counts),
            "n_images": len(image_counts),
            "class_balance_ratio": (
                _balance_ratio_from_counts(class_counts)
            ),
            "image_balance_ratio": (
                _balance_ratio_from_counts(image_counts)
            ),
        }
        return metrics, selections, diagnostic_rows, None

    except Exception as exc:
        return (
            None,
            None,
            diagnostic_rows,
            repr(exc),
        )


def evaluate_balanced_sampling_grid(
    object_db,
    *,
    filters,
    m_values=expcfg.BALANCED_SAMPLING_M_VALUES,
    strategies=expcfg.BALANCED_PIXEL_STRATEGIES,
    seeds=expcfg.BALANCED_SAMPLING_SEEDS,
    replace=False,
    under_m_policy=expcfg.BALANCED_SAMPLING_UNDER_M_POLICY,
    min_eligible_rate=expcfg.BALANCED_SAMPLING_MIN_ELIGIBLE_RATE,
    return_diagnostics: bool = False,
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate technical feasibility and seed stability of candidate m values.

    Each object/strategy/seed selection is computed exactly once. Summary
    metrics are derived directly from selected indices and object metadata.
    """
    selected = list(
        filter_records(
            object_db,
            **(filters or {}),
        )
    )
    total = len(selected)

    available_by_object = {}
    for object_id, obj in selected:
        spectra = np.asarray(
            obj.get("spectra"),
            dtype=float,
        )
        validity = spectral_pixel_validity_report(
            spectra,
            policy=pixel_validity_policy,
        )
        available_by_object[str(object_id)] = int(
            validity["valid_mask"].sum()
        )

    rows = []
    diagnostic_rows = []

    for m_value in m_values:
        m = int(m_value)
        under_ids = [
            object_id
            for object_id, n_available
            in available_by_object.items()
            if n_available < m
        ]
        eligible_rate = (
            float((total - len(under_ids)) / total)
            if total
            else 0.0
        )

        for strategy_value in strategies:
            strategy = str(strategy_value)
            run_metrics = []
            selection_runs = []
            errors = []

            for seed_value in seeds:
                seed = int(seed_value)
                (
                    metrics,
                    selections,
                    run_diagnostics,
                    error,
                ) = _evaluate_balanced_sampling_run(
                    selected,
                    m=m,
                    strategy=strategy,
                    seed=seed,
                    replace=bool(replace),
                    under_m_policy=str(under_m_policy),
                    pixel_validity_policy=pixel_validity_policy,
                )

                diagnostic_rows.extend(run_diagnostics)

                if error is not None:
                    errors.append(error)
                    continue

                run_metrics.append(metrics)
                selection_runs.append(selections)

            metrics_df = pd.DataFrame(run_metrics)

            if errors or metrics_df.empty:
                status = "invalid"
            elif eligible_rate < float(min_eligible_rate):
                status = "warning"
            else:
                status = "accepted"

            rows.append(
                {
                    "m": m,
                    "strategy": strategy,
                    "under_m_policy": str(under_m_policy),
                    "n_objects_total": total,
                    "n_objects_under_m": len(under_ids),
                    "eligible_rate": eligible_rate,
                    "n_rows": (
                        int(round(metrics_df["n_rows"].median()))
                        if not metrics_df.empty
                        else 0
                    ),
                    "n_classes": (
                        int(metrics_df["n_classes"].min())
                        if not metrics_df.empty
                        else 0
                    ),
                    "n_images": (
                        int(metrics_df["n_images"].min())
                        if not metrics_df.empty
                        else 0
                    ),
                    "class_balance_ratio": (
                        float(
                            metrics_df[
                                "class_balance_ratio"
                            ].min()
                        )
                        if not metrics_df.empty
                        else np.nan
                    ),
                    "image_balance_ratio": (
                        float(
                            metrics_df[
                                "image_balance_ratio"
                            ].min()
                        )
                        if not metrics_df.empty
                        else np.nan
                    ),
                    "selection_stability": (
                        _mean_selection_overlap(
                            selection_runs
                        )
                    ),
                    "status": status,
                }
            )

    feasibility = pd.DataFrame(
        rows,
        columns=BALANCED_SAMPLING_COLUMNS,
    )
    diagnostics = pd.DataFrame(
        diagnostic_rows,
        columns=PIXEL_SAMPLING_DIAGNOSTIC_COLUMNS,
    )

    if return_diagnostics:
        return feasibility, diagnostics
    return feasibility


def _preprocessing_summary_row(
    X_before,
    X_after,
    *,
    preprocessing_name,
    steps,
    sg_window_length,
    sg_polyorder,
    saturation_bounds,
    zero_variance_epsilon,
    repeatability_error,
    matrix_id="unspecified",
    fit_role="calibration",
    eval_role="validation",
    wavelength_axis_identifier=None,
):
    X_before = np.asarray(X_before)
    X_after = np.asarray(X_after)
    deriv = preprocessing_derivative(steps)
    n_nan = int(np.isnan(X_after).sum())
    n_inf = int(np.isinf(X_after).sum())
    variances = np.var(X_after, axis=0)
    if saturation_bounds is None:
        saturation_rate = np.nan
    else:
        lower, upper = map(float, saturation_bounds)
        saturation_rate = float(np.mean((X_after <= lower) | (X_after >= upper)))
    name_coherent = preprocessing_name_from_steps(steps) == str(preprocessing_name)
    band_unchanged = X_before.shape[1] == X_after.shape[1]
    zero_variance_rate = float(
        np.mean(variances <= float(zero_variance_epsilon))
    )
    saturation_ok = np.isnan(saturation_rate) or saturation_rate == 0.0
    status = (
        "accepted"
        if n_nan == 0
        and n_inf == 0
        and band_unchanged
        and name_coherent
        and np.isfinite(repeatability_error)
        and repeatability_error <= expcfg.PREPROCESSING_REPEATABILITY_TOLERANCE
        and zero_variance_rate
        <= expcfg.PREPROCESSING_MAX_ZERO_VARIANCE_BAND_RATE
        and saturation_ok
        else "invalid"
    )
    return {
        "matrix_id": str(matrix_id),
        "fit_role": str(fit_role),
        "eval_role": str(eval_role),
        "wavelength_axis_id": wavelength_axis_identifier,
        "preprocessing": str(preprocessing_name),
        "steps": " + ".join(steps),
        "sg_window_length": (
            int(sg_window_length) if deriv is not None else np.nan
        ),
        "sg_polyorder": int(sg_polyorder) if deriv is not None else np.nan,
        "deriv": int(deriv) if deriv is not None else np.nan,
        "status": status,
        "n_features_before": int(X_before.shape[1]),
        "n_features_after": int(X_after.shape[1]),
        "band_count_unchanged": bool(band_unchanged),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "zero_variance_band_rate": zero_variance_rate,
        "saturation_rate": saturation_rate,
        "global_min": float(np.nanmin(X_after)),
        "global_max": float(np.nanmax(X_after)),
        "repeatability_error": float(repeatability_error),
        "name_steps_coherent": bool(name_coherent),
    }


def evaluate_preprocessing_grid(
    X_fit,
    X_eval=None,
    *,
    preprocessing_configs,
    sg_windows=expcfg.SG_WINDOW_CHOICES,
    sg_polyorder=expcfg.SG_POLYORDER,
    wavelengths=None,
    matrix_id: str = "unspecified",
    fit_role: str = "calibration",
    eval_role: str = "validation",
    saturation_bounds=expcfg.PREPROCESSING_SATURATION_BOUNDS,
    zero_variance_epsilon=expcfg.PREPROCESSING_ZERO_VARIANCE_EPSILON,
):
    """Fit on calibration only, transform evaluation data, and audit each chain."""
    X_fit = np.asarray(X_fit, dtype=float)
    X_eval = X_fit if X_eval is None else np.asarray(X_eval, dtype=float)
    configs = normalize_preprocessing_configs(preprocessing_configs)
    rows = []
    outputs = {}
    errors = []

    for preprocessing_name, steps in configs.items():
        deriv = preprocessing_derivative(steps)
        candidate_windows = tuple(sg_windows) if deriv is not None else (None,)
        for window in candidate_windows:
            output_key = (
                str(preprocessing_name)
                if window is None
                else f"{preprocessing_name}__sg{int(window)}"
            )
            try:
                effective_window = (
                    int(expcfg.SG_DEFAULT_WINDOW)
                    if window is None
                    else int(window)
                )
                validate_preprocessing_steps(
                    steps,
                    n_features=X_fit.shape[1],
                    sg_window_length=effective_window if deriv is not None else None,
                    sg_polyorder=int(sg_polyorder) if deriv is not None else None,
                )
                preprocessor = SpectralPreprocessor(
                    steps=steps,
                    sg_window_length=effective_window,
                    sg_polyorder=int(sg_polyorder),
                    absorbance_nonpositive_policy=(
                        expcfg.PREPROCESSING_ABSORBANCE_NONPOSITIVE_POLICY
                    ),
                )
                X_fit_processed = preprocessor.fit_transform(
                    X_fit,
                    wavelengths=wavelengths,
                )
                X_eval_processed = preprocessor.transform(X_eval)
                X_eval_repeat = preprocessor.transform(X_eval)
                repeatability_error = float(
                    np.max(np.abs(X_eval_processed - X_eval_repeat))
                )
                combined = np.vstack([X_fit_processed, X_eval_processed])
                row = _preprocessing_summary_row(
                    X_fit,
                    combined,
                    preprocessing_name=preprocessing_name,
                    steps=steps,
                    sg_window_length=effective_window,
                    sg_polyorder=sg_polyorder,
                    saturation_bounds=saturation_bounds,
                    zero_variance_epsilon=zero_variance_epsilon,
                    repeatability_error=repeatability_error,
                    matrix_id=matrix_id,
                    fit_role=fit_role,
                    eval_role=eval_role,
                    wavelength_axis_identifier=(
                        wavelength_axis_id(wavelengths)
                        if wavelengths is not None
                        else None
                    ),
                )
                outputs[output_key] = {
                    "preprocessor": preprocessor,
                    "X_fit": X_fit_processed,
                    "X_eval": X_eval_processed,
                    "steps": tuple(steps),
                }
                if row["status"] != "accepted":
                    errors.append(
                        {
                            "matrix_id": str(matrix_id),
                            "fit_role": str(fit_role),
                            "eval_role": str(eval_role),
                            "wavelength_axis_id": (
                                wavelength_axis_id(wavelengths)
                                if wavelengths is not None
                                else None
                            ),
                            "preprocessing": str(preprocessing_name),
                            "sg_window_length": (
                                int(window) if window is not None else np.nan
                            ),
                            "error_type": "ValidationError",
                            "error": "Numeric preprocessing audit failed.",
                        }
                    )
                    outputs.pop(output_key, None)
            except Exception as exc:
                row = {
                    "matrix_id": str(matrix_id),
                    "fit_role": str(fit_role),
                    "eval_role": str(eval_role),
                    "wavelength_axis_id": (
                        wavelength_axis_id(wavelengths)
                        if wavelengths is not None
                        else None
                    ),
                    "preprocessing": str(preprocessing_name),
                    "steps": " + ".join(steps),
                    "sg_window_length": (
                        int(window) if window is not None else np.nan
                    ),
                    "sg_polyorder": (
                        int(sg_polyorder) if window is not None else np.nan
                    ),
                    "deriv": int(deriv) if deriv is not None else np.nan,
                    "status": "invalid",
                }
                errors.append(
                    {
                        "matrix_id": str(matrix_id),
                        "fit_role": str(fit_role),
                        "eval_role": str(eval_role),
                        "wavelength_axis_id": (
                            wavelength_axis_id(wavelengths)
                            if wavelengths is not None
                            else None
                        ),
                        "preprocessing": str(preprocessing_name),
                        "sg_window_length": (
                            int(window) if window is not None else np.nan
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            rows.append(row)

    summary = pd.DataFrame(
        rows,
        columns=expcfg.PREPROCESSING_SUMMARY_REQUIRED_COLUMNS,
    )
    error_df = pd.DataFrame(errors, columns=PREPROCESSING_ERROR_COLUMNS)
    return summary, outputs, error_df


def summarize_preprocessing_output(
    X_preprocessed,
    *,
    preprocessing_name: str,
    steps: Sequence[str],
    sg_window_length: int,
    sg_polyorder: int,
) -> dict:
    """Backward-compatible one-matrix wrapper around the compact audit schema."""
    X_preprocessed = np.asarray(X_preprocessed, dtype=float)
    return _preprocessing_summary_row(
        X_preprocessed,
        X_preprocessed,
        preprocessing_name=preprocessing_name,
        steps=tuple(steps),
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        saturation_bounds=expcfg.PREPROCESSING_SATURATION_BOUNDS,
        zero_variance_epsilon=expcfg.PREPROCESSING_ZERO_VARIANCE_EPSILON,
        repeatability_error=0.0,
        matrix_id="unspecified",
        fit_role="calibration",
        eval_role="validation",
        wavelength_axis_identifier=None,
    )
