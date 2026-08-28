"""Exhaustive evaluation of the locked eight-track SIMCA domain.

Notebook 04A does not fit models or tune thresholds.  It reapplies the locked
03B decisions to the OOF predictions, optionally applies the immutable 03C
spatial operation, then separates technical validity, acceptability,
eligibility, equivalence and Pareto membership.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.decision.metrics import apply_locked_margin_decision
from src.workflows.simca_candidates import stable_frame_signature
from src.workflows.simca_selection_utils import pareto_front_by_group
from src.workflows.spatial_postprocessing_calibration import (
    apply_spatial_postprocessing,
)
from src.workflows.simca_thresholds_calibration import (
    build_pixel_vote_table,
)

def _json_list(values: Sequence) -> str:
    return json.dumps(sorted({str(value) for value in values}), ensure_ascii=False)


def _safe_divide(numerator, denominator):
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=np.broadcast_to(denominator_array, result.shape) > 0,
    )
    return result


def _frame_signature(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash an ordered entity/value matrix without serialising predictions."""
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise ValueError("A prediction signature needs at least one column.")
    digest = hashlib.sha256()
    digest.update("\x1f".join(available).encode("utf-8"))
    for column in available:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            digest.update(numeric.astype("<f8", copy=False).tobytes())
        else:
            hashes = pd.util.hash_pandas_object(
                series.astype("string"), index=False, categorize=True
            ).to_numpy(dtype="<u8", copy=False)
            digest.update(hashes.tobytes())
    return digest.hexdigest()


def _validate_unique_ids(domain: pd.DataFrame) -> None:
    required = {
        "domain_config_id",
        "calibration_id",
        "projection_config_id",
        "evaluation_track",
        "track_id",
        "decision_mode",
        "projection_level",
        "random_state",
    }
    missing = sorted(required - set(domain.columns))
    if missing:
        raise KeyError(f"Missing calibration-domain columns: {missing}")
    if domain["domain_config_id"].astype(str).duplicated().any():
        raise RuntimeError("domain_config_id must be unique in calibration_domain.")
    unknown_tracks = sorted(
        set(domain["evaluation_track"].astype(str))
        - set(expcfg.SIMCA_EVALUATION_TRACKS)
    )
    if unknown_tracks:
        raise RuntimeError(f"Unknown evaluation tracks in 03B: {unknown_tracks}")


def validate_exhaustive_grid_inputs(
    calibration_domain: pd.DataFrame,
    oof_object_predictions: pd.DataFrame,
    oof_pixel_predictions: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
) -> None:
    """Validate that every 03B configuration has a matching OOF source."""
    _validate_unique_ids(calibration_domain)
    eligibility_required = {"evaluation_track", "eligibility_status"}
    missing = sorted(eligibility_required - set(projection_eligibility.columns))
    if missing:
        raise KeyError(f"Missing projection-eligibility columns: {missing}")
    if projection_eligibility["evaluation_track"].astype(str).duplicated().any():
        raise RuntimeError("Projection eligibility must contain one row per track.")

    prediction_required = {
        "projection_config_id",
        "fold_id",
        "random_state",
        "source_image",
        "object_id",
        "truth",
        *expcfg.SIMCA_GRID_REQUIRED_FINITE_PREDICTION_COLUMNS,
    }
    for level, predictions in (
        ("object_projection", oof_object_predictions),
        ("pixel_projection", oof_pixel_predictions),
    ):
        missing = sorted(prediction_required - set(predictions.columns))
        if missing:
            raise KeyError(f"Missing {level} OOF columns: {missing}")
        expected = calibration_domain.loc[
            calibration_domain["projection_level"].astype(str).eq(level),
            ["projection_config_id", "random_state"],
        ].drop_duplicates()
        observed = predictions[["projection_config_id", "random_state"]].drop_duplicates()
        coverage = expected.merge(
            observed,
            on=["projection_config_id", "random_state"],
            how="left",
            indicator=True,
        )
        if coverage["_merge"].ne("both").any():
            examples = coverage.loc[
                coverage["_merge"].ne("both"),
                ["projection_config_id", "random_state"],
            ].head(10)
            raise RuntimeError(
                f"03B {level} OOF coverage is incomplete: {examples.to_dict('records')}"
            )


def _apply_locked_spatial_operation(
    observations: pd.DataFrame,
    target: np.ndarray,
    uncertain: np.ndarray,
    image_db: Mapping,
    parameters: Mapping,
) -> np.ndarray:
    required = {"source_image", "row", "col"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise KeyError(f"Spatial post-processing needs columns: {missing}")
    cleaned_target = np.asarray(target, dtype=bool).copy()
    for image_key, positions in observations.groupby(
        "source_image", sort=False
    ).indices.items():
        image_key = str(image_key)
        if image_key not in image_db:
            raise KeyError(f"Image {image_key!r} is absent from the HDF5 database.")
        shape = np.asarray(image_db[image_key]["labels"]).shape
        group = observations.iloc[positions]
        rows = pd.to_numeric(group["row"], errors="raise").astype(int).to_numpy()
        cols = pd.to_numeric(group["col"], errors="raise").astype(int).to_numpy()
        if (
            (rows < 0).any()
            or (cols < 0).any()
            or (rows >= shape[0]).any()
            or (cols >= shape[1]).any()
        ):
            raise RuntimeError(f"OOF coordinates fall outside image {image_key}.")
        if pd.DataFrame({"row": rows, "col": cols}).duplicated().any():
            raise RuntimeError(f"Duplicated OOF coordinates in image {image_key}.")
        valid_map = np.zeros(shape, dtype=bool)
        target_map = np.zeros(shape, dtype=bool)
        uncertain_map = np.zeros(shape, dtype=bool)
        valid_map[rows, cols] = True
        target_map[rows, cols] = np.asarray(target, dtype=bool)[positions]
        uncertain_map[rows, cols] = np.asarray(uncertain, dtype=bool)[positions]
        cleaned, preserved_uncertain = apply_spatial_postprocessing(
            target_map,
            uncertain_map,
            valid_map,
            connectivity=int(parameters["connectivity"]),
            morphology_operation=str(parameters["morphology_operation"]),
            morphology_radius=int(parameters["morphology_radius"]),
            min_area_pixels=int(parameters["min_area_pixels"]),
        )
        if not np.array_equal(
            preserved_uncertain[rows, cols],
            np.asarray(uncertain, dtype=bool)[positions],
        ):
            raise RuntimeError("The locked spatial operation modified uncertainty.")
        cleaned_target[positions] = cleaned[rows, cols]
    return cleaned_target


def summarize_grouped_decisions(
    observations: pd.DataFrame,
    target_decision: np.ndarray,
    uncertain_decision: np.ndarray,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    group_columns = tuple(group_columns)
    internal_group_columns = group_columns
    work = observations[
        list(dict.fromkeys([*group_columns, "object_id", "truth"]))
    ].copy()
    if not group_columns:
        work["__all_observations__"] = "all"
        internal_group_columns = ("__all_observations__",)
    work["truth"] = work["truth"].astype(bool)
    work["target_decision"] = np.asarray(target_decision, dtype=bool)
    work["uncertain_decision"] = np.asarray(uncertain_decision, dtype=bool)
    work["non_target_decision"] = ~work["target_decision"] & ~work["uncertain_decision"]
    work["tp"] = work["truth"] & work["target_decision"]
    work["fn"] = work["truth"] & work["non_target_decision"]
    work["fp"] = ~work["truth"] & work["target_decision"]
    work["tn"] = ~work["truth"] & work["non_target_decision"]
    work["target_uncertain"] = work["truth"] & work["uncertain_decision"]
    work["non_target_uncertain"] = ~work["truth"] & work["uncertain_decision"]
    grouped = work.groupby(
        list(internal_group_columns), dropna=False, sort=False
    )
    metrics = grouped.agg(
        n_observations=("truth", "size"),
        n_target=("truth", "sum"),
        n_uncertain=("uncertain_decision", "sum"),
        n_target_uncertain=("target_uncertain", "sum"),
        n_non_target_uncertain=("non_target_uncertain", "sum"),
        tp=("tp", "sum"),
        fn=("fn", "sum"),
        fp=("fp", "sum"),
        tn=("tn", "sum"),
    ).reset_index()
    metrics["n_non_target"] = metrics["n_observations"] - metrics["n_target"]
    metrics["target_miss_rate"] = _safe_divide(metrics["fn"], metrics["n_target"])
    metrics["false_accept_rate"] = _safe_divide(metrics["fp"], metrics["n_non_target"])
    metrics["uncertain_rate"] = _safe_divide(
        metrics["n_uncertain"], metrics["n_observations"]
    )
    metrics["target_uncertain_rate"] = _safe_divide(
        metrics["n_target_uncertain"], metrics["n_target"]
    )
    metrics["non_target_uncertain_rate"] = _safe_divide(
        metrics["n_non_target_uncertain"], metrics["n_non_target"]
    )
    metrics["coverage_rate"] = 1.0 - metrics["uncertain_rate"]
    sensitivity = _safe_divide(metrics["tp"], metrics["tp"] + metrics["fn"])
    specificity = _safe_divide(metrics["tn"], metrics["tn"] + metrics["fp"])
    metrics["decided_balanced_accuracy"] = np.where(
        np.isfinite(sensitivity) & np.isfinite(specificity),
        0.5 * (sensitivity + specificity),
        np.nan,
    )
    no_uncertainty = metrics["n_uncertain"].eq(0)
    metrics["balanced_accuracy"] = np.where(
        no_uncertainty, metrics["decided_balanced_accuracy"], np.nan
    )

    object_rates = work.groupby(
        [*internal_group_columns, "object_id"], dropna=False, sort=False
    ).agg(
        object_truth=("truth", "first"),
        object_n_target=("truth", "sum"),
        object_fn=("fn", "sum"),
    ).reset_index()
    target_objects = object_rates.loc[object_rates["object_truth"].astype(bool)].copy()
    target_objects["object_target_miss_rate"] = _safe_divide(
        target_objects["object_fn"], target_objects["object_n_target"]
    )
    object_summary = target_objects.groupby(
        list(internal_group_columns), dropna=False, sort=False
    ).agg(
        n_target_objects=("object_id", "nunique"),
        macro_object_target_miss_rate=("object_target_miss_rate", "mean"),
    ).reset_index()
    metrics = metrics.merge(
        object_summary,
        on=list(internal_group_columns),
        how="left",
        validate="one_to_one",
    )
    metrics["n_target_objects"] = metrics["n_target_objects"].fillna(0).astype(int)
    return metrics.drop(columns=["__all_observations__"], errors="ignore")


# Internal compatibility alias used by the exhaustive 04A evaluator.
_summarize_grouped_decisions = summarize_grouped_decisions


def _technical_audit_row(row: Mapping, *, status: str, exc: Exception | None = None) -> dict:
    return {
        "domain_config_id": str(row["domain_config_id"]),
        "calibration_id": str(row["calibration_id"]),
        "evaluation_track": str(row["evaluation_track"]),
        "track_id": str(row["track_id"]),
        "technical_status": status,
        "calculable": status == "calculable",
        "acceptability_status": "not_evaluated",
        "eligibility_status": "not_evaluated",
        "duplicate_status": "not_evaluated",
        "representative_calibration_id": str(row["calibration_id"]),
        "pareto_eligible": False,
        "error_type": "" if exc is None else type(exc).__name__,
        "error_message": "" if exc is None else str(exc),
    }


def evaluate_locked_oof_domain(
    calibration_domain: pd.DataFrame,
    oof_object_predictions: pd.DataFrame,
    oof_pixel_predictions: pd.DataFrame,
    *,
    image_db: Mapping | None = None,
    spatial_lock: Mapping | None = None,
    spatial_supported_tracks: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every domain row or retain a row-level technical error."""
    _validate_unique_ids(calibration_domain)
    supported_spatial = set(map(str, spatial_supported_tracks))
    if supported_spatial and (image_db is None or spatial_lock is None):
        raise ValueError("Locked spatial tracks require image_db and spatial_lock.")
    spatial_parameters = (
        dict(spatial_lock["selected_parameters"]) if spatial_lock is not None else {}
    )
    lookups = {}
    for level, predictions in (
        ("object_projection", oof_object_predictions),
        ("pixel_projection", oof_pixel_predictions),
    ):
        key_frame = predictions[["projection_config_id", "random_state"]].copy()
        key_frame["projection_config_id"] = key_frame["projection_config_id"].astype(str)
        key_frame["random_state"] = pd.to_numeric(
            key_frame["random_state"], errors="raise"
        ).astype(int)
        lookups[level] = {
            (str(key[0]), int(key[1])): np.asarray(indices, dtype=int)
            for key, indices in key_frame.groupby(
                ["projection_config_id", "random_state"], sort=False
            ).indices.items()
        }

    metric_parts: list[pd.DataFrame] = []
    signatures: list[dict] = []
    audit_rows: list[dict] = []
    required_finite = list(expcfg.SIMCA_GRID_REQUIRED_FINITE_PREDICTION_COLUMNS)
    for row in calibration_domain.to_dict("records"):
        try:
            level = str(row["projection_level"])
            predictions = (
                oof_object_predictions if level == "object_projection" else oof_pixel_predictions
            )
            key = (str(row["projection_config_id"]), int(row["random_state"]))
            positions = lookups.get(level, {}).get(key)
            if positions is None or len(positions) == 0:
                raise RuntimeError(f"No OOF observations for projection/seed {key}.")
            observations = predictions.iloc[positions].copy()
            finite = observations[required_finite].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(finite).all():
                raise RuntimeError("Non-finite OOF diagnostic or rule metric.")
            truth = observations["truth"].astype(bool)
            if truth.nunique(dropna=False) != 2:
                raise RuntimeError("The OOF configuration does not cover both classes.")
            target, uncertain = apply_locked_margin_decision(
                pd.to_numeric(observations["simca_margin"], errors="coerce").to_numpy(),
                str(row["decision_mode"]),
                direct_2way_threshold=row.get("direct_2way_threshold", np.nan),
                three_way_lower_threshold=row.get("three_way_lower_threshold", np.nan),
                three_way_upper_threshold=row.get("three_way_upper_threshold", np.nan),
            )
            map_variant = "raw"
            if level == "pixel_projection" and str(row["evaluation_track"]) in supported_spatial:
                target = _apply_locked_spatial_operation(
                    observations,
                    target,
                    uncertain,
                    image_db=image_db,
                    parameters=spatial_parameters,
                )
                map_variant = "locked_postprocessed"

            fold = _summarize_grouped_decisions(
                observations,
                target,
                uncertain,
                group_columns=("fold_id",),
            )
            fold["aggregation_level"] = "fold"
            fold["group_id"] = fold["fold_id"].astype(str)
            image = _summarize_grouped_decisions(
                observations,
                target,
                uncertain,
                group_columns=("fold_id", "source_image"),
            )
            image["aggregation_level"] = "source_image"
            image["group_id"] = image.pop("source_image").astype(str)
            metrics = pd.concat([fold, image], ignore_index=True, sort=False)
            for name in (
                "domain_config_id",
                "calibration_id",
                "evaluation_track",
                "track_id",
                "decision_mode",
                "projection_level",
            ):
                metrics[name] = row[name]
            metrics["map_variant"] = map_variant
            metrics["random_state"] = int(row["random_state"])
            metrics["status"] = "calculable"
            metric_parts.append(metrics)

            entity_columns = ["fold_id", "source_image", "object_id"]
            if level == "pixel_projection":
                entity_columns += ["row", "col"]
            signature_frame = observations[entity_columns + ["simca_margin"]].copy()
            signature_frame["target_decision"] = target
            signature_frame["uncertain_decision"] = uncertain
            signatures.append(
                {
                    "domain_config_id": str(row["domain_config_id"]),
                    "calibration_id": str(row["calibration_id"]),
                    "evaluation_track": str(row["evaluation_track"]),
                    "random_state": int(row["random_state"]),
                    "score_signature": _frame_signature(
                        signature_frame, [*entity_columns, "simca_margin"]
                    ),
                    "decision_signature": _frame_signature(
                        signature_frame,
                        [*entity_columns, "target_decision", "uncertain_decision"],
                    ),
                }
            )
            audit_rows.append(_technical_audit_row(row, status="calculable"))
        except Exception as exc:  # every failed configuration remains explicit
            audit_rows.append(_technical_audit_row(row, status="technical_error", exc=exc))

    fold_metrics = (
        pd.concat(metric_parts, ignore_index=True, sort=False)
        if metric_parts
        else pd.DataFrame(columns=expcfg.SIMCA_GRID_FOLD_METRIC_COLUMNS)
    )
    fold_metrics = fold_metrics.reindex(columns=expcfg.SIMCA_GRID_FOLD_METRIC_COLUMNS)
    audit = pd.DataFrame(audit_rows).reindex(
        columns=expcfg.SIMCA_GRID_TECHNICAL_AUDIT_COLUMNS
    )
    signature_table = pd.DataFrame(signatures)
    if len(audit) != len(calibration_domain):
        raise RuntimeError("Technical audit lost calibration-domain rows.")
    return fold_metrics, audit, signature_table


def _weighted_rate(frame: pd.DataFrame, rate: str, weight: str) -> float:
    values = pd.to_numeric(frame[rate], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame[weight], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def _finite_values(values) -> np.ndarray:
    """Return only finite numeric values from an arbitrary 1D input."""
    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)

    return numeric[
        np.isfinite(numeric)
    ]


def finite_mean(values) -> float:
    """Finite-value mean shared by 04A/04C and notebook 05."""
    numeric = _finite_values(values)

    return (
        float(numeric.mean())
        if numeric.size
        else np.nan
    )


def finite_min(values) -> float:
    """Finite-value minimum; NaN when no finite value exists."""
    numeric = _finite_values(values)

    return (
        float(numeric.min())
        if numeric.size
        else np.nan
    )


def finite_max(values) -> float:
    """Finite-value maximum; NaN when no finite value exists."""
    numeric = _finite_values(values)

    return (
        float(numeric.max())
        if numeric.size
        else np.nan
    )


def finite_std(
    values,
    *,
    ddof: int = 1,
) -> float:
    """Finite-value standard deviation with explicit degrees of freedom."""
    ddof = int(ddof)

    if ddof < 0:
        raise ValueError(
            "ddof must be non-negative."
        )

    numeric = _finite_values(values)

    if numeric.size <= ddof:
        return np.nan

    return float(
        numeric.std(
            ddof=ddof
        )
    )


def _constraint_status(metrics: Mapping) -> tuple[str, str]:
    mode = str(metrics["decision_mode"])
    constraints = expcfg.SIMCA_SEARCH_CONSTRAINTS[mode]
    failures = []
    checks = (
        ("worst_fold_target_miss_rate", "max_fn_rate", "<="),
        ("worst_fold_false_accept_rate", "max_fp_rate", "<="),
        (
            "decided_balanced_accuracy" if mode == "3way" else "balanced_accuracy",
            "min_balanced_accuracy",
            ">=",
        ),
        ("fold_metric_std", "max_fold_metric_std", "<="),
    )
    if mode == "3way":
        checks += (
            ("uncertain_rate", "max_uncertain_rate", "<="),
            ("coverage_rate", "min_coverage", ">="),
        )
    for metric_name, constraint_name, operator in checks:
        value = float(metrics.get(metric_name, np.nan))
        limit = float(constraints[constraint_name])
        passed = np.isfinite(value) and (value <= limit if operator == "<=" else value >= limit)
        if not passed:
            rendered = "non_finite" if not np.isfinite(value) else f"{value:.6g}"
            failures.append(f"{metric_name}={rendered}{operator}{limit:.6g}")
    return (
        ("acceptable", "")
        if not failures
        else ("calculable_not_acceptable", ";".join(failures))
    )


def aggregate_grid_metrics(
    fold_metrics: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate folds/seeds/images at calibration_id without selecting rows."""
    eligibility = projection_eligibility[
        ["evaluation_track", "eligibility_status"]
    ].drop_duplicates()
    rows = []
    for calibration_id, group in fold_metrics.groupby("calibration_id", sort=False):
        folds = group.loc[group["aggregation_level"].eq("fold")].copy()
        images = group.loc[group["aggregation_level"].eq("source_image")].copy()
        if folds.empty:
            continue
        first = folds.iloc[0]
        miss = _weighted_rate(folds, "target_miss_rate", "n_target")
        false_accept = _weighted_rate(folds, "false_accept_rate", "n_non_target")
        uncertainty = _weighted_rate(folds, "uncertain_rate", "n_observations")
        target_uncertainty = _weighted_rate(
            folds, "target_uncertain_rate", "n_target"
        )
        non_target_uncertainty = _weighted_rate(
            folds, "non_target_uncertain_rate", "n_non_target"
        )
        balanced_2way = (
            1.0 - 0.5 * (miss + false_accept)
            if np.isfinite(miss) and np.isfinite(false_accept)
            else np.nan
        )
        decided_sensitivity = (
            (1.0 - miss - target_uncertainty) / (1.0 - target_uncertainty)
            if np.isfinite(miss)
            and np.isfinite(target_uncertainty)
            and target_uncertainty < 1.0
            else np.nan
        )
        decided_specificity = (
            (1.0 - false_accept - non_target_uncertainty)
            / (1.0 - non_target_uncertainty)
            if np.isfinite(false_accept)
            and np.isfinite(non_target_uncertainty)
            and non_target_uncertainty < 1.0
            else np.nan
        )
        decided_balanced = (
            0.5 * (decided_sensitivity + decided_specificity)
            if np.isfinite(decided_sensitivity)
            and np.isfinite(decided_specificity)
            else np.nan
        )
        macro_image_miss = finite_mean(images["target_miss_rate"])
        macro_image_false = finite_mean(images["false_accept_rate"])
        macro_image_balanced = (
            1.0 - 0.5 * (macro_image_miss + macro_image_false)
            if np.isfinite(macro_image_miss) and np.isfinite(macro_image_false)
            else np.nan
        )
        macro_object_miss = _weighted_rate(
            images,
            "macro_object_target_miss_rate",
            "n_target_objects",
        )
        stability_column = (
            "decided_balanced_accuracy"
            if str(first["decision_mode"]) == "3way"
            else "balanced_accuracy"
        )
        stability = pd.to_numeric(folds[stability_column], errors="coerce")
        row = {
            "calibration_id": str(calibration_id),
            "evaluation_track": str(first["evaluation_track"]),
            "track_id": str(first["track_id"]),
            "decision_mode": str(first["decision_mode"]),
            "projection_level": str(first["projection_level"]),
            "map_variant": str(first["map_variant"]),
            "n_domain_configurations": int(folds["domain_config_id"].nunique()),
            "n_seeds": int(folds["random_state"].nunique()),
            "n_folds": int(folds["fold_id"].nunique()),
            "n_images": int(images["group_id"].nunique()),
            "n_observations": int(folds["n_observations"].sum()),
            "target_miss_rate": miss,
            "false_accept_rate": false_accept,
            "uncertain_rate": uncertainty,
            "coverage_rate": 1.0 - uncertainty if np.isfinite(uncertainty) else np.nan,
            "balanced_accuracy": (
                balanced_2way if str(first["decision_mode"]) == "2way" else np.nan
            ),
            "decided_balanced_accuracy": decided_balanced,
            "macro_image_target_miss_rate": macro_image_miss,
            "macro_image_false_accept_rate": macro_image_false,
            "macro_image_balanced_accuracy": macro_image_balanced,
            "macro_object_target_miss_rate": macro_object_miss,
            "worst_fold_target_miss_rate": finite_max(folds["target_miss_rate"]),
            "worst_fold_false_accept_rate": finite_max(folds["false_accept_rate"]),
            "fold_metric_std": float(stability.std(ddof=0)) if stability.notna().any() else np.nan,
            "technical_status": "calculable",
        }
        row["acceptability_status"], row["failure_reason"] = _constraint_status(row)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_GRID_THRESHOLD_METRIC_COLUMNS)
    result = result.merge(
        eligibility,
        on="evaluation_track",
        how="left",
        validate="many_to_one",
    )
    result["eligibility_status"] = result["eligibility_status"].fillna(
        "unsupported_missing_domain_diagnostic"
    )
    return result.reindex(columns=expcfg.SIMCA_GRID_THRESHOLD_METRIC_COLUMNS)


def _aggregate_signatures(signatures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calibration_id, group in signatures.groupby("calibration_id", sort=False):
        group = group.sort_values(["random_state", "domain_config_id"], kind="mergesort")
        score = hashlib.sha256("|".join(group["score_signature"]).encode()).hexdigest()
        decision = hashlib.sha256("|".join(group["decision_signature"]).encode()).hexdigest()
        rows.append(
            {
                "calibration_id": str(calibration_id),
                "evaluation_track": str(group["evaluation_track"].iloc[0]),
                "score_signature": score,
                "decision_signature": decision,
            }
        )
    return pd.DataFrame(rows)


def build_configuration_catalog(calibration_domain: pd.DataFrame) -> pd.DataFrame:
    """Collapse seed repetitions while keeping all provenance identifiers."""
    rows = []
    for calibration_id, group in calibration_domain.groupby("calibration_id", sort=False):
        first = group.sort_values("domain_config_id", kind="mergesort").iloc[0].to_dict()
        # Keep one explicit runtime row for consumers that still require a
        # scalar seed/id, while the JSON fields below retain every repetition.
        first["domain_config_id"] = str(first["domain_config_id"])
        first["random_state"] = int(first["random_state"])
        first["calibration_id"] = str(calibration_id)
        first["domain_config_ids"] = _json_list(group["domain_config_id"])
        first["random_states"] = json.dumps(
            sorted(pd.to_numeric(group["random_state"], errors="raise").astype(int).unique().tolist())
        )
        first["n_seed_repetitions"] = int(group["random_state"].nunique())
        rows.append(first)
    return pd.DataFrame(rows)


def build_duplicate_groups(
    calibration_domain: pd.DataFrame,
    signatures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect exact configs and exact OOF outputs with lexical representatives."""
    catalog = build_configuration_catalog(calibration_domain)
    aggregated_signatures = _aggregate_signatures(signatures)
    catalog = catalog.merge(
        aggregated_signatures,
        on=["calibration_id", "evaluation_track"],
        how="left",
        validate="one_to_one",
    )
    catalog["representative_calibration_id"] = catalog["calibration_id"].astype(str)
    catalog["duplicate_status"] = "unique"
    group_rows = []

    exact_columns = [
        column
        for column in expcfg.SIMCA_GRID_EXACT_CONFIGURATION_COLUMNS
        if column in catalog.columns
    ]
    if exact_columns:
        for _, group in catalog.groupby(
            ["evaluation_track", *exact_columns], dropna=False, sort=False
        ):
            if len(group) < 2:
                continue
            members = sorted(group["calibration_id"].astype(str))
            representative = members[0]
            group_id = "exact_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:16]
            group_rows.append(
                {
                    "duplicate_group_id": group_id,
                    "duplicate_kind": "exact_configuration",
                    "evaluation_track": str(group["evaluation_track"].iloc[0]),
                    "representative_calibration_id": representative,
                    "member_calibration_ids": json.dumps(members),
                    "n_members": len(members),
                    "score_signature": "",
                    "decision_signature": "",
                    "reason": "identical_locked_scientific_configuration",
                }
            )
            mask = catalog["calibration_id"].astype(str).isin(members)
            catalog.loc[mask, "representative_calibration_id"] = representative
            catalog.loc[mask, "duplicate_status"] = np.where(
                catalog.loc[mask, "calibration_id"].astype(str).eq(representative),
                "exact_configuration_representative",
                "exact_configuration_duplicate",
            )

    representatives = catalog.loc[
        catalog["calibration_id"].astype(str).eq(
            catalog["representative_calibration_id"].astype(str)
        )
        & catalog["score_signature"].notna()
        & catalog["decision_signature"].notna()
    ]
    for _, group in representatives.groupby(
        ["evaluation_track", "score_signature", "decision_signature"],
        dropna=False,
        sort=False,
    ):
        if len(group) < 2:
            continue
        members = sorted(group["calibration_id"].astype(str))
        representative = members[0]
        group_id = "prediction_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:16]
        group_rows.append(
            {
                "duplicate_group_id": group_id,
                "duplicate_kind": "identical_oof_prediction_vector",
                "evaluation_track": str(group["evaluation_track"].iloc[0]),
                "representative_calibration_id": representative,
                "member_calibration_ids": json.dumps(members),
                "n_members": len(members),
                "score_signature": str(group["score_signature"].iloc[0]),
                "decision_signature": str(group["decision_signature"].iloc[0]),
                "reason": "identical_entity_score_and_locked_decision_vectors",
            }
        )
        mask = catalog["calibration_id"].astype(str).isin(members)
        catalog.loc[mask, "representative_calibration_id"] = representative
        catalog.loc[mask, "duplicate_status"] = np.where(
            catalog.loc[mask, "calibration_id"].astype(str).eq(representative),
            "prediction_representative",
            "prediction_duplicate",
        )

    duplicate_groups = pd.DataFrame(group_rows).reindex(
        columns=expcfg.SIMCA_GRID_DUPLICATE_GROUP_COLUMNS
    )
    return catalog, duplicate_groups


def _annotate_pareto(
    metrics: pd.DataFrame,
    candidate_mask: pd.Series,
    flag_column: str,
) -> pd.Series:
    flags = pd.Series(False, index=metrics.index, dtype=bool)
    for track, indices in metrics.loc[candidate_mask].groupby(
        "evaluation_track", sort=False
    ).groups.items():
        spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[str(track)]
        objectives = [*spec["pareto_minimize"], *spec["pareto_maximize"]]
        group = metrics.loc[indices]
        finite = group[objectives].apply(pd.to_numeric, errors="coerce")
        valid_indices = finite.index[
            np.isfinite(finite.to_numpy(dtype=float)).all(axis=1)
        ]
        if len(valid_indices) == 0:
            continue
        front = pareto_front_by_group(
            metrics.loc[valid_indices],
            group_cols=("evaluation_track",),
            minimize_cols=spec["pareto_minimize"],
            maximize_cols=spec["pareto_maximize"],
            epsilon=float(expcfg.SIMCA_GRID_PARETO_EPSILON),
        )
        front_ids = set(front["calibration_id"].astype(str))
        flags.loc[valid_indices] = metrics.loc[valid_indices, "calibration_id"].astype(str).isin(front_ids)
    flags.name = flag_column
    return flags


def build_pareto_reference(
    threshold_metrics: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build exhaustive diagnostic and protocol Pareto flags in all 8 tracks."""
    reference = threshold_metrics.merge(
        catalog[
            [
                "calibration_id",
                "duplicate_status",
                "representative_calibration_id",
            ]
        ],
        on="calibration_id",
        how="left",
        validate="one_to_one",
    )
    reference["is_duplicate_representative"] = reference["calibration_id"].astype(str).eq(
        reference["representative_calibration_id"].astype(str)
    )
    diagnostic_pool = reference["technical_status"].eq("calculable") & reference[
        "is_duplicate_representative"
    ]
    supported = reference["eligibility_status"].isin(
        expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
    )
    protocol_pool = (
        diagnostic_pool
        & supported
        & reference["acceptability_status"].eq("acceptable")
    )
    reference["diagnostic_pareto_front"] = _annotate_pareto(
        reference, diagnostic_pool, "diagnostic_pareto_front"
    )
    reference["protocol_pareto_front"] = _annotate_pareto(
        reference, protocol_pool, "protocol_pareto_front"
    )
    reference["pareto_exclusion_reason"] = ""
    reference.loc[~reference["is_duplicate_representative"], "pareto_exclusion_reason"] = (
        "exact_or_prediction_duplicate"
    )
    reference.loc[
        reference["is_duplicate_representative"]
        & ~reference["acceptability_status"].eq("acceptable"),
        "pareto_exclusion_reason",
    ] = "calculable_not_acceptable"
    reference.loc[
        reference["is_duplicate_representative"]
        & reference["acceptability_status"].eq("acceptable")
        & ~supported,
        "pareto_exclusion_reason",
    ] = "unsupported_evaluation_track_retained"
    reference["row_type"] = "configuration"

    summaries = []
    for track in expcfg.SIMCA_EVALUATION_TRACKS:
        spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[track]
        present = reference["evaluation_track"].astype(str).eq(track)
        if present.any():
            reason = (
                "track_supported"
                if reference.loc[present, "eligibility_status"].isin(
                    expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
                ).any()
                else "unsupported_track_retained_as_scientific_result"
            )
            eligibility_status = str(reference.loc[present, "eligibility_status"].iloc[0])
        else:
            reason = "no_calibrated_configuration_in_03B_domain"
            eligibility_status = "unsupported_internal_calibration"
        summaries.append(
            {
                "row_type": "track_summary",
                "calibration_id": "",
                "evaluation_track": track,
                "track_id": spec["track_id"],
                "technical_status": "no_configuration" if not present.any() else "audited",
                "acceptability_status": "not_applicable",
                "eligibility_status": eligibility_status,
                "is_duplicate_representative": False,
                "diagnostic_pareto_front": False,
                "protocol_pareto_front": False,
                "pareto_exclusion_reason": reason,
            }
        )
    reference = pd.concat([reference, pd.DataFrame(summaries)], ignore_index=True, sort=False)
    return reference.reindex(columns=expcfg.SIMCA_GRID_PARETO_REFERENCE_COLUMNS)


def finalize_grid_audit(
    technical_audit: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_status = threshold_metrics[
        ["calibration_id", "acceptability_status", "eligibility_status"]
    ]
    catalog_status = catalog[
        [
            "calibration_id",
            "duplicate_status",
            "representative_calibration_id",
        ]
    ]
    audit = technical_audit.drop(
        columns=[
            "acceptability_status",
            "eligibility_status",
            "duplicate_status",
            "representative_calibration_id",
            "pareto_eligible",
        ],
        errors="ignore",
    ).merge(aggregate_status, on="calibration_id", how="left", validate="many_to_one")
    audit = audit.merge(catalog_status, on="calibration_id", how="left", validate="many_to_one")
    audit["acceptability_status"] = audit["acceptability_status"].fillna("not_calculable")
    audit["eligibility_status"] = audit["eligibility_status"].fillna(
        "unsupported_missing_domain_diagnostic"
    )
    audit["duplicate_status"] = audit["duplicate_status"].fillna("not_evaluated")
    audit["representative_calibration_id"] = audit[
        "representative_calibration_id"
    ].fillna(audit["calibration_id"])
    audit["pareto_eligible"] = (
        audit["calculable"].astype(bool)
        & audit["acceptability_status"].eq("acceptable")
        & audit["eligibility_status"].isin(
            expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
        )
        & audit["calibration_id"].astype(str).eq(
            audit["representative_calibration_id"].astype(str)
        )
    )
    return audit.reindex(columns=expcfg.SIMCA_GRID_TECHNICAL_AUDIT_COLUMNS)


def _wilson_interval(
    successes,
    totals,
    *,
    confidence_level: float = expcfg.SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson interval for binomial validation rates."""
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    valid = np.isfinite(successes) & np.isfinite(totals) & (totals > 0)
    low = np.full(np.broadcast_shapes(successes.shape, totals.shape), np.nan)
    high = np.full(low.shape, np.nan)
    if not valid.any():
        return low, high
    z = NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)
    n = totals[valid]
    p = successes[valid] / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))
    half /= denominator
    low[valid] = np.maximum(0.0, center - half)
    high[valid] = np.minimum(1.0, center + half)
    return low, high


def _add_validation_rate_intervals(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    specifications = (
        ("target_miss_rate", "n_target"),
        ("false_accept_rate", "n_non_target"),
        ("uncertain_rate", "n_observations"),
        ("coverage_rate", "n_observations"),
    )
    for metric, total_column in specifications:
        rate = pd.to_numeric(out[metric], errors="coerce").to_numpy(dtype=float)
        total = pd.to_numeric(
            out[total_column], errors="coerce"
        ).to_numpy(dtype=float)
        successes = np.rint(rate * total)
        low, high = _wilson_interval(successes, total)
        out[f"{metric}_ci_low"] = low
        out[f"{metric}_ci_high"] = high
    out["balanced_accuracy_ci_low"] = 0.5 * (
        1.0 - out["target_miss_rate_ci_high"]
        + 1.0 - out["false_accept_rate_ci_high"]
    )
    out["balanced_accuracy_ci_high"] = 0.5 * (
        1.0 - out["target_miss_rate_ci_low"]
        + 1.0 - out["false_accept_rate_ci_low"]
    )
    invalid = ~np.isfinite(pd.to_numeric(out["balanced_accuracy"], errors="coerce"))
    out.loc[
        invalid, ["balanced_accuracy_ci_low", "balanced_accuracy_ci_high"]
    ] = np.nan
    tp = pd.to_numeric(out["tp"], errors="coerce").to_numpy(dtype=float)
    fn = pd.to_numeric(out["fn"], errors="coerce").to_numpy(dtype=float)
    tn = pd.to_numeric(out["tn"], errors="coerce").to_numpy(dtype=float)
    fp = pd.to_numeric(out["fp"], errors="coerce").to_numpy(dtype=float)
    sensitivity_low, sensitivity_high = _wilson_interval(tp, tp + fn)
    specificity_low, specificity_high = _wilson_interval(tn, tn + fp)
    out["decided_balanced_accuracy_ci_low"] = 0.5 * (
        sensitivity_low + specificity_low
    )
    out["decided_balanced_accuracy_ci_high"] = 0.5 * (
        sensitivity_high + specificity_high
    )
    invalid_decided = ~np.isfinite(
        pd.to_numeric(out["decided_balanced_accuracy"], errors="coerce")
    )
    out.loc[
        invalid_decided,
        [
            "decided_balanced_accuracy_ci_low",
            "decided_balanced_accuracy_ci_high",
        ],
    ] = np.nan
    return out


def _finite_mean_interval(values) -> tuple[float, float, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return np.nan, np.nan, np.nan
    mean = float(numeric.mean())
    if len(numeric) == 1:
        return mean, mean, mean
    z = NormalDist().inv_cdf(
        0.5 + expcfg.SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL / 2.0
    )
    half = z * float(numeric.std(ddof=1)) / np.sqrt(len(numeric))
    return mean, max(0.0, mean - half), min(1.0, mean + half)


def _macro_image_decision_metrics(
    by_image: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate class-conditional image metrics for pure or mixed images."""
    result: dict[str, float] = {}
    for metric in (
        "target_miss_rate",
        "false_accept_rate",
        "uncertain_rate",
        "target_uncertain_rate",
        "non_target_uncertain_rate",
        "coverage_rate",
    ):
        mean, low, high = _finite_mean_interval(by_image[metric])
        key = f"macro_image_{metric}"
        result[key] = mean
        result[f"{key}_ci_low"] = low
        result[f"{key}_ci_high"] = high

    miss = result["macro_image_target_miss_rate"]
    false_accept = result["macro_image_false_accept_rate"]
    miss_low = result["macro_image_target_miss_rate_ci_low"]
    miss_high = result["macro_image_target_miss_rate_ci_high"]
    false_low = result["macro_image_false_accept_rate_ci_low"]
    false_high = result["macro_image_false_accept_rate_ci_high"]
    if np.isfinite(miss) and np.isfinite(false_accept):
        result["macro_image_balanced_accuracy"] = float(
            np.clip(1.0 - 0.5 * (miss + false_accept), 0.0, 1.0)
        )
        result["macro_image_balanced_accuracy_ci_low"] = float(
            np.clip(1.0 - 0.5 * (miss_high + false_high), 0.0, 1.0)
        )
        result["macro_image_balanced_accuracy_ci_high"] = float(
            np.clip(1.0 - 0.5 * (miss_low + false_low), 0.0, 1.0)
        )
    else:
        result["macro_image_balanced_accuracy"] = np.nan
        result["macro_image_balanced_accuracy_ci_low"] = np.nan
        result["macro_image_balanced_accuracy_ci_high"] = np.nan

    decided_sensitivity = _safe_divide(
        by_image["tp"],
        by_image["tp"] + by_image["fn"],
    )
    decided_specificity = _safe_divide(
        by_image["tn"],
        by_image["tn"] + by_image["fp"],
    )
    sensitivity, sensitivity_low, sensitivity_high = _finite_mean_interval(
        decided_sensitivity
    )
    specificity, specificity_low, specificity_high = _finite_mean_interval(
        decided_specificity
    )
    if np.isfinite(sensitivity) and np.isfinite(specificity):
        result["macro_image_decided_balanced_accuracy"] = float(
            0.5 * (sensitivity + specificity)
        )
        result["macro_image_decided_balanced_accuracy_ci_low"] = float(
            0.5 * (sensitivity_low + specificity_low)
        )
        result["macro_image_decided_balanced_accuracy_ci_high"] = float(
            0.5 * (sensitivity_high + specificity_high)
        )
    else:
        result["macro_image_decided_balanced_accuracy"] = np.nan
        result["macro_image_decided_balanced_accuracy_ci_low"] = np.nan
        result["macro_image_decided_balanced_accuracy_ci_high"] = np.nan
    return result


def _equivalence_group_ids(
    tracks: pd.Series,
    signatures: pd.Series,
    *,
    prefix: str,
) -> pd.Series:
    values = []
    for track, signature in zip(tracks.astype(str), signatures.astype(str)):
        if not signature:
            values.append("")
            continue
        digest = hashlib.sha256(f"{track}|{signature}".encode("utf-8")).hexdigest()
        values.append(f"{prefix}_{digest[:16]}")
    return pd.Series(values, index=tracks.index, dtype="string")


def evaluate_locked_validation_predictions(
    validation_executions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    *,
    technical_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply frozen 03B policies to batch-3 projections and return long metrics.

    The function never creates a validation-specific scientific identifier.
    Continuous predictions are addressed by ``projection_id`` and decision
    policies by ``(model_id, random_state, decision_scope)``.
    """
    required_executions = set(expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS)
    required_thresholds = set(expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS)
    for frame, required, name in (
        (validation_executions, required_executions, "validation_executions"),
        (selected_thresholds, required_thresholds, "selected_thresholds"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing columns: {missing}")

    if validation_executions.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    if expcfg.INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY != "safe_reject":
        raise RuntimeError(
            "The current grouped-decision metrics implement the frozen "
            "safe_reject uncertainty policy only."
        )

    executions = validation_executions.copy()
    executions["model_id"] = executions["model_id"].astype(str)
    executions["projection_id"] = executions["projection_id"].astype(str)
    executions["random_state"] = pd.to_numeric(
        executions["random_state"], errors="raise"
    ).astype(int)
    run_keys = ["model_id", "random_state"]
    if executions.duplicated(run_keys).any():
        raise RuntimeError("Validation executions duplicate (model_id, random_state).")

    thresholds = selected_thresholds.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS)
    ].copy()
    thresholds["model_id"] = thresholds["model_id"].astype(str)
    thresholds["random_state"] = pd.to_numeric(
        thresholds["random_state"], errors="raise"
    ).astype(int)
    thresholds["decision_scope"] = thresholds["decision_scope"].astype(str)
    threshold_key = [*run_keys, "decision_scope"]
    if thresholds.duplicated(threshold_key).any():
        raise RuntimeError("Selected validation thresholds duplicate a natural key.")

    expected_scope_rows = []
    for row in executions.itertuples(index=False):
        expected_scope_rows.append(
            {
                "model_id": str(row.model_id),
                "random_state": int(row.random_state),
                "decision_scope": "direct",
            }
        )
        if str(row.projection_level) == "pixel_projection":
            expected_scope_rows.append(
                {
                    "model_id": str(row.model_id),
                    "random_state": int(row.random_state),
                    "decision_scope": "pixel_to_object",
                }
            )
    expected_scopes = pd.DataFrame(expected_scope_rows).drop_duplicates()
    scope_coverage = expected_scopes.merge(
        thresholds[threshold_key].drop_duplicates().assign(_present=True),
        on=threshold_key,
        how="outer",
        indicator=True,
    )
    if not scope_coverage["_merge"].eq("both").all():
        missing_scopes = scope_coverage.loc[
            scope_coverage["_merge"].eq("left_only"), threshold_key
        ].to_dict("records")
        extra_scopes = scope_coverage.loc[
            scope_coverage["_merge"].eq("right_only"), threshold_key
        ].to_dict("records")
        raise RuntimeError(
            "Selected threshold scopes do not match the 04C execution registry: "
            f"missing={missing_scopes[:10]}, extra={extra_scopes[:10]}."
        )

    threshold_lookup = {
        (str(row.model_id), int(row.random_state), str(row.decision_scope)): row
        for row in thresholds.itertuples(index=False)
    }

    prediction_required = {
        "projection_id",
        "source_image",
        "object_id",
        "truth",
        "simca_margin",
    }
    prediction_lookups: dict[str, dict[str, np.ndarray]] = {}
    for level, frame in (
        ("object_projection", object_predictions),
        ("pixel_projection", pixel_predictions),
    ):
        if frame.empty:
            prediction_lookups[level] = {}
            continue
        missing = sorted(prediction_required - set(frame.columns))
        if missing:
            raise KeyError(f"{level} validation predictions are missing: {missing}")
        if frame.duplicated(
            [
                "projection_id",
                "source_image",
                "object_id",
                *(["row", "col"] if level == "pixel_projection" else []),
            ]
        ).any():
            raise RuntimeError(f"{level} validation predictions contain duplicates.")
        prediction_lookups[level] = {
            str(projection_id): np.asarray(indices, dtype=int)
            for projection_id, indices in frame.groupby(
                "projection_id", sort=False
            ).indices.items()
        }

    event_lookup: dict[str, list[dict]] = {}
    if technical_events is not None and len(technical_events):
        required_events = set(expcfg.SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS)
        missing = sorted(required_events - set(technical_events.columns))
        if missing:
            raise KeyError(f"technical_events is missing columns: {missing}")
        for projection_id, group in technical_events.groupby(
            "projection_id", sort=False, dropna=False
        ):
            event_lookup[str(projection_id)] = group.to_dict("records")

    diagnostic_columns = (
        "pca_score_pc1",
        "pca_score_pc2",
        "H",
        "Q",
        "rule_statistic",
        "rule_limit",
        "normalized_ratio",
        "simca_margin",
    )
    base_metric_names = (
        "n_observations",
        "n_target",
        "n_non_target",
        "n_target_objects",
        "n_uncertain",
        "n_target_uncertain",
        "n_non_target_uncertain",
        "tp",
        "fn",
        "fp",
        "tn",
        "target_miss_rate",
        "false_accept_rate",
        "uncertain_rate",
        "target_uncertain_rate",
        "non_target_uncertain_rate",
        "coverage_rate",
        "balanced_accuracy",
        "decided_balanced_accuracy",
        "macro_object_target_miss_rate",
    )
    overall_only_metric_names = (
        "macro_image_target_miss_rate",
        "macro_image_false_accept_rate",
        "macro_image_uncertain_rate",
        "macro_image_target_uncertain_rate",
        "macro_image_non_target_uncertain_rate",
        "macro_image_coverage_rate",
        "macro_image_balanced_accuracy",
        "macro_image_decided_balanced_accuracy",
    )

    def _technical_rows(
        execution: Mapping[str, object],
        scopes: Sequence[str],
        exc: Exception,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_id": str(execution["model_id"]),
                    "random_state": int(execution["random_state"]),
                    "track_id": str(execution["track_id"]),
                    "decision_scope": str(scope),
                    "map_variant": "raw",
                    "aggregation_level": "overall",
                    "group_id": "all",
                    "metric": "technical_calculability",
                    "value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "status": "technical_failure",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                for scope in scopes
            ]
        )

    def _wide_to_long(
        wide: pd.DataFrame,
        *,
        model_id: str,
        random_state: int,
        track_id: str,
        decision_scope: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in wide.to_dict("records"):
            aggregation_level = str(record["aggregation_level"])
            metric_names = list(base_metric_names)
            if aggregation_level == "overall":
                metric_names.extend(overall_only_metric_names)
            for metric_name in metric_names:
                if metric_name not in record:
                    continue
                raw_value = record.get(metric_name, np.nan)
                value = pd.to_numeric(
                    pd.Series([raw_value]), errors="coerce"
                ).iloc[0]
                ci_low = pd.to_numeric(
                    pd.Series([record.get(f"{metric_name}_ci_low", np.nan)]),
                    errors="coerce",
                ).iloc[0]
                ci_high = pd.to_numeric(
                    pd.Series([record.get(f"{metric_name}_ci_high", np.nan)]),
                    errors="coerce",
                ).iloc[0]
                rows.append(
                    {
                        "model_id": str(model_id),
                        "random_state": int(random_state),
                        "track_id": str(track_id),
                        "decision_scope": str(decision_scope),
                        "map_variant": "raw",
                        "aggregation_level": aggregation_level,
                        "group_id": str(record["group_id"]),
                        "metric": str(metric_name),
                        "value": float(value) if pd.notna(value) else np.nan,
                        "ci_low": float(ci_low) if pd.notna(ci_low) else np.nan,
                        "ci_high": float(ci_high) if pd.notna(ci_high) else np.nan,
                        "status": "calculable",
                        "error_type": "",
                        "error_message": "",
                    }
                )
        return pd.DataFrame(rows)

    def _evaluate_scope(
        observations: pd.DataFrame,
        *,
        score_col: str,
        threshold_row,
        decision_mode: str,
        model_id: str,
        random_state: int,
        track_id: str,
        decision_scope: str,
    ) -> pd.DataFrame:
        scores = pd.to_numeric(observations[score_col], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{score_col} contains non-finite values.")
        truth = pd.to_numeric(observations["truth"], errors="coerce")
        if not truth.isin((0, 1)).all():
            raise RuntimeError("Validation truth must be binary.")
        if truth.astype(bool).nunique() != 2:
            raise RuntimeError(
                "Batch-3 validation observations do not cover both classes."
            )
        lower = float(threshold_row.lower_threshold)
        upper = float(threshold_row.upper_threshold)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise RuntimeError("A selected decision threshold is non-finite.")
        if decision_mode == "2way" and not np.isclose(lower, upper):
            raise RuntimeError("A 2-way selected threshold must have lower == upper.")
        if decision_mode == "3way" and not lower < upper:
            raise RuntimeError("A 3-way selected threshold must have lower < upper.")
        target, uncertain = apply_locked_margin_decision(
            scores,
            decision_mode,
            direct_2way_threshold=lower,
            three_way_lower_threshold=lower,
            three_way_upper_threshold=upper,
        )
        overall = summarize_grouped_decisions(
            observations,
            target,
            uncertain,
            group_columns=(),
        )
        overall["aggregation_level"] = "overall"
        overall["group_id"] = "all"
        by_image = summarize_grouped_decisions(
            observations,
            target,
            uncertain,
            group_columns=("source_image",),
        )
        by_image["aggregation_level"] = "source_image"
        by_image["group_id"] = by_image.pop("source_image").astype(str)
        macro_image_metrics = _macro_image_decision_metrics(by_image)
        for column, value in macro_image_metrics.items():
            overall[column] = value
        wide = pd.concat([overall, by_image], ignore_index=True, sort=False)
        wide = _add_validation_rate_intervals(wide)
        return _wide_to_long(
            wide,
            model_id=model_id,
            random_state=random_state,
            track_id=track_id,
            decision_scope=decision_scope,
        )

    metric_parts: list[pd.DataFrame] = []
    for execution in executions.to_dict("records"):
        model_id = str(execution["model_id"])
        random_state = int(execution["random_state"])
        track_id = str(execution["track_id"])
        projection_id = str(execution["projection_id"])
        projection_level = str(execution["projection_level"])
        decision_mode = str(execution["decision_mode"])
        expected_scopes = (
            ("direct", "pixel_to_object")
            if projection_level == "pixel_projection"
            else ("direct",)
        )
        try:
            if projection_id in event_lookup:
                first = event_lookup[projection_id][0]
                raise RuntimeError(
                    f"{first.get('stage', 'projection')}: "
                    f"{first.get('error_type', 'technical_error')}: "
                    f"{first.get('error_message', '')}"
                )
            if projection_level not in prediction_lookups:
                raise RuntimeError(f"Unknown projection level: {projection_level!r}.")
            positions = prediction_lookups[projection_level].get(projection_id)
            if positions is None or not len(positions):
                raise RuntimeError(
                    f"No batch-3 predictions for projection_id={projection_id!r}."
                )
            predictions = (
                object_predictions
                if projection_level == "object_projection"
                else pixel_predictions
            )
            observations = predictions.iloc[positions].copy()
            available_diagnostics = [
                column for column in diagnostic_columns if column in observations.columns
            ]
            if len(available_diagnostics) != len(diagnostic_columns):
                missing = sorted(set(diagnostic_columns) - set(available_diagnostics))
                raise RuntimeError(f"Validation diagnostics are missing: {missing}")
            numeric = observations[list(diagnostic_columns)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise RuntimeError("A validation diagnostic is non-finite.")

            direct_threshold = threshold_lookup[
                (model_id, random_state, "direct")
            ]
            metric_parts.append(
                _evaluate_scope(
                    observations,
                    score_col="simca_margin",
                    threshold_row=direct_threshold,
                    decision_mode=decision_mode,
                    model_id=model_id,
                    random_state=random_state,
                    track_id=track_id,
                    decision_scope="direct",
                )
            )

            if projection_level == "pixel_projection":
                object_votes = build_pixel_vote_table(
                    observations,
                    group_columns=("source_image", "object_id"),
                )
                secondary_threshold = threshold_lookup[
                    (model_id, random_state, "pixel_to_object")
                ]
                metric_parts.append(
                    _evaluate_scope(
                        object_votes,
                        score_col="pixel_target_ratio",
                        threshold_row=secondary_threshold,
                        decision_mode=decision_mode,
                        model_id=model_id,
                        random_state=random_state,
                        track_id=track_id,
                        decision_scope="pixel_to_object",
                    )
                )
        except Exception as exc:
            # If direct evaluation succeeded but the secondary aggregation failed,
            # do not duplicate a direct technical-failure row.
            completed_scopes = {
                str(part["decision_scope"].iloc[0])
                for part in metric_parts
                if len(part)
                and str(part["model_id"].iloc[0]) == model_id
                and int(part["random_state"].iloc[0]) == random_state
            }
            failed_scopes = [
                scope for scope in expected_scopes if scope not in completed_scopes
            ]
            metric_parts.append(
                _technical_rows(execution, failed_scopes or expected_scopes, exc)
            )

    metrics = (
        pd.concat(metric_parts, ignore_index=True, sort=False)
        if metric_parts
        else pd.DataFrame(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    )
    metrics = metrics.reindex(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    natural_key = [
        "model_id",
        "random_state",
        "decision_scope",
        "map_variant",
        "aggregation_level",
        "group_id",
        "metric",
    ]
    if len(metrics) and metrics.duplicated(natural_key).any():
        duplicate_rows = metrics.loc[
            metrics.duplicated(natural_key, keep=False), natural_key
        ].head(20)
        raise RuntimeError(
            "Duplicate natural keys in validation_metrics: "
            f"{duplicate_rows.to_dict('records')}"
        )
    return metrics.sort_values(natural_key, kind="mergesort").reset_index(drop=True)


def build_validation_guardrails(
    validation_executions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    *,
    spatial_component_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the frozen 04C guardrails to each execution/decision scope.

    Supported 04A models receive blocking checks. ``diagnostic_only`` models
    receive exactly the same diagnostics, but those checks cannot eliminate a
    model from the protocol-supported path.
    """
    required_executions = set(expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS)
    required_metrics = set(expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    for frame, required, name in (
        (validation_executions, required_executions, "validation_executions"),
        (validation_metrics, required_metrics, "validation_metrics"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing columns: {missing}")

    executions = validation_executions.copy()
    executions["model_id"] = executions["model_id"].astype(str)
    executions["random_state"] = pd.to_numeric(
        executions["random_state"], errors="raise"
    ).astype(int)
    run_keys = ["model_id", "random_state"]
    if executions.duplicated(run_keys).any():
        raise RuntimeError("Validation executions duplicate (model_id, random_state).")

    metrics = validation_metrics.copy()
    metrics["model_id"] = metrics["model_id"].astype(str)
    metrics["random_state"] = pd.to_numeric(
        metrics["random_state"], errors="raise"
    ).astype(int)
    metrics["decision_scope"] = metrics["decision_scope"].astype(str)
    metric_key = [
        "model_id",
        "random_state",
        "decision_scope",
        "map_variant",
        "aggregation_level",
        "group_id",
        "metric",
    ]
    if metrics.duplicated(metric_key).any():
        raise RuntimeError("validation_metrics duplicates its natural metric key.")

    # Exact identity check: metrics may not introduce an execution that is absent
    # from the canonical 04C registry.
    metric_runs = metrics[run_keys].drop_duplicates()
    run_coverage = metric_runs.merge(
        executions[run_keys].drop_duplicates().assign(_present=True),
        on=run_keys,
        how="left",
        validate="one_to_one",
    )
    if run_coverage["_present"].isna().any():
        raise RuntimeError("validation_metrics contains an unknown execution key.")

    def metric_value(
        model_id: str,
        random_state: int,
        decision_scope: str,
        aggregation_level: str,
        group_id: str,
        metric: str,
    ) -> tuple[float, float, float, str, str, str]:
        subset = metrics.loc[
            metrics["model_id"].eq(str(model_id))
            & metrics["random_state"].eq(int(random_state))
            & metrics["decision_scope"].eq(str(decision_scope))
            & metrics["map_variant"].astype(str).eq("raw")
            & metrics["aggregation_level"].astype(str).eq(str(aggregation_level))
            & metrics["group_id"].astype(str).eq(str(group_id))
            & metrics["metric"].astype(str).eq(str(metric))
        ]
        if len(subset) != 1:
            return np.nan, np.nan, np.nan, "missing", "", ""
        row = subset.iloc[0]
        return (
            float(pd.to_numeric(pd.Series([row["value"]]), errors="coerce").iloc[0]),
            float(pd.to_numeric(pd.Series([row["ci_low"]]), errors="coerce").iloc[0]),
            float(pd.to_numeric(pd.Series([row["ci_high"]]), errors="coerce").iloc[0]),
            str(row["status"]),
            str(row["error_type"]),
            str(row["error_message"]),
        )

    # Verify the long representation preserves coverage = 1 - uncertainty
    # wherever both quantities are identifiable.
    complement_pairs = (
        ("uncertain_rate", "coverage_rate"),
        ("macro_image_uncertain_rate", "macro_image_coverage_rate"),
    )
    for uncertain_metric, coverage_metric in complement_pairs:
        pair = metrics.loc[
            metrics["metric"].isin([uncertain_metric, coverage_metric])
        ].pivot_table(
            index=[
                "model_id",
                "random_state",
                "decision_scope",
                "map_variant",
                "aggregation_level",
                "group_id",
            ],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        if {uncertain_metric, coverage_metric}.issubset(pair.columns):
            uncertainty = pd.to_numeric(
                pair[uncertain_metric], errors="coerce"
            ).to_numpy(dtype=float)
            coverage = pd.to_numeric(
                pair[coverage_metric], errors="coerce"
            ).to_numpy(dtype=float)
            finite = np.isfinite(uncertainty) & np.isfinite(coverage)
            if (finite & ~np.isclose(uncertainty + coverage, 1.0, atol=1e-7)).any():
                raise RuntimeError(
                    f"{coverage_metric} must equal 1 - {uncertain_metric}."
                )

    spatial_lookup: dict[tuple[str, int], float] = {}
    if spatial_component_metrics is not None and len(spatial_component_metrics):
        required_spatial = {
            "model_id",
            "random_state",
            "aggregation_level",
            "map_variant",
            "smallest_fragment_recall",
        }
        missing = sorted(required_spatial - set(spatial_component_metrics.columns))
        if missing:
            raise KeyError(f"spatial_component_metrics is missing: {missing}")
        spatial_overall = spatial_component_metrics.loc[
            spatial_component_metrics["aggregation_level"].astype(str).eq("overall")
            & spatial_component_metrics["map_variant"].astype(str).eq(
                "locked_postprocessed"
            )
        ].copy()
        spatial_key = ["model_id", "random_state"]
        if spatial_overall.duplicated(spatial_key).any():
            raise RuntimeError(
                "Spatial overall metrics duplicate (model_id, random_state)."
            )
        spatial_lookup = {
            (str(row.model_id), int(row.random_state)): float(
                row.smallest_fragment_recall
            )
            for row in spatial_overall.itertuples(index=False)
        }

    rows: list[dict[str, object]] = []
    status_by_scope: dict[tuple[str, int, str], str] = {}

    for execution in executions.to_dict("records"):
        model_id = str(execution["model_id"])
        random_state = int(execution["random_state"])
        track_id = str(execution["track_id"])
        decision_mode = str(execution["decision_mode"])
        projection_level = str(execution["projection_level"])
        eligibility_status = str(execution["eligibility_status"])
        downstream_status = str(execution["downstream_status"])
        if downstream_status not in {"supported", "diagnostic_only"}:
            raise RuntimeError(
                f"Unknown downstream_status={downstream_status!r}."
            )
        if decision_mode not in expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS:
            raise RuntimeError(f"Unknown decision mode: {decision_mode!r}.")
        if projection_level not in expcfg.SIMCA_CONCAT_REFIT_PRIMARY_DECISION_SCOPES:
            raise RuntimeError(f"Unknown projection level: {projection_level!r}.")

        decision_scopes = (
            ("direct", "pixel_to_object")
            if projection_level == "pixel_projection"
            else ("direct",)
        )
        primary_decision_scopes = set(
            map(
                str,
                expcfg.SIMCA_CONCAT_REFIT_PRIMARY_DECISION_SCOPES[
                    projection_level
                ],
            )
        )
        for decision_scope in decision_scopes:
            is_protocol_supported = downstream_status == "supported"
            is_primary_scope = decision_scope in primary_decision_scopes
            blocking_failures: list[bool] = []
            technical_failure = False

            # The presence of the explicit technical-calculability row takes
            # precedence over performance guardrails.
            technical = metrics.loc[
                metrics["model_id"].eq(model_id)
                & metrics["random_state"].eq(random_state)
                & metrics["decision_scope"].eq(decision_scope)
                & metrics["aggregation_level"].astype(str).eq("overall")
                & metrics["metric"].astype(str).eq("technical_calculability")
            ]
            if len(technical):
                first = technical.iloc[0]
                technical_failure = True
                rows.append(
                    {
                        "model_id": model_id,
                        "random_state": random_state,
                        "track_id": track_id,
                        "decision_scope": decision_scope,
                        "eligibility_status": eligibility_status,
                        "downstream_status": downstream_status,
                        "candidate_status": "technical_failure",
                        "rule_id": "technical_calculability",
                        "scope": "overall",
                        "metric": "technical_calculability",
                        "severity": "blocking",
                        "n_independent_units": 0,
                        "min_independent_units": np.nan,
                        "observed_value": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "comparator": "is",
                        "threshold": np.nan,
                        "check_status": "technical_error",
                        "is_blocking": bool(
                            is_protocol_supported and is_primary_scope
                        ),
                        "reason_code": "validation_technical_failure",
                        "reason": (
                            f"{first.get('error_type', '')}: "
                            f"{first.get('error_message', '')}"
                        ).strip(": "),
                    }
                )
                status_by_scope[(model_id, random_state, decision_scope)] = (
                    "technical_failure"
                )
                continue

            constraints = expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS[
                decision_mode
            ]
            specs = expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS[
                decision_mode
            ]
            for spec in specs:
                projection_levels = tuple(
                    map(str, spec.get("projection_levels", ()))
                )
                if projection_levels and projection_level not in projection_levels:
                    continue
                applicable_scopes = tuple(
                    map(str, spec.get("decision_scopes", ()))
                )
                if applicable_scopes and decision_scope not in applicable_scopes:
                    continue

                rule_id = str(spec["rule_id"])
                scope = str(spec["scope"])
                metric_name = str(spec["metric"])
                constraint_name = str(spec["limit_key"])
                comparator = str(spec["comparator"])
                severity = str(spec["severity"])
                min_independent_units_raw = spec.get("min_independent_units")
                min_independent_units = (
                    int(min_independent_units_raw)
                    if min_independent_units_raw is not None
                    else None
                )

                # Direct pixel decisions use equal-image macro summaries for
                # their primary overall endpoint. Explicit macro metrics, such
                # as macro-object miss, are already fully qualified.
                reported_metric = (
                    f"macro_image_{metric_name}"
                    if scope == "overall"
                    and decision_scope == "direct"
                    and projection_level == "pixel_projection"
                    and not metric_name.startswith("macro_")
                    else metric_name
                )
                n_independent_units = np.nan
                if scope == "overall":
                    observed, ci_low, ci_high, status, error_type, error_message = (
                        metric_value(
                            model_id,
                            random_state,
                            decision_scope,
                            "overall",
                            "all",
                            reported_metric,
                        )
                    )
                else:
                    group = metrics.loc[
                        metrics["model_id"].eq(model_id)
                        & metrics["random_state"].eq(random_state)
                        & metrics["decision_scope"].eq(decision_scope)
                        & metrics["map_variant"].astype(str).eq("raw")
                        & metrics["aggregation_level"].astype(str).eq(
                            "source_image"
                        )
                        & metrics["metric"].astype(str).eq(reported_metric)
                        & metrics["status"].astype(str).eq("calculable")
                    ].copy()
                    values = pd.to_numeric(group["value"], errors="coerce")
                    finite = values[np.isfinite(values)]
                    n_independent_units = int(
                        group.loc[finite.index, "group_id"].astype(str).nunique()
                    )
                    if finite.empty:
                        observed = ci_low = ci_high = np.nan
                        status = "missing"
                        error_type = ""
                        error_message = ""
                    else:
                        index = (
                            finite.idxmax()
                            if comparator == "<="
                            else finite.idxmin()
                        )
                        selected = group.loc[index]
                        observed = float(selected["value"])
                        ci_low = float(
                            pd.to_numeric(
                                pd.Series([selected["ci_low"]]),
                                errors="coerce",
                            ).iloc[0]
                        )
                        ci_high = float(
                            pd.to_numeric(
                                pd.Series([selected["ci_high"]]),
                                errors="coerce",
                            ).iloc[0]
                        )
                        status = str(selected["status"])
                        error_type = str(selected["error_type"])
                        error_message = str(selected["error_message"])

                enough_independent_units = bool(
                    min_independent_units is None
                    or (
                        np.isfinite(n_independent_units)
                        and int(n_independent_units) >= min_independent_units
                    )
                )
                rule_can_block = bool(
                    severity == "blocking"
                    or (
                        severity == "conditional_blocking"
                        and enough_independent_units
                    )
                )
                is_blocking = bool(
                    is_protocol_supported
                    and is_primary_scope
                    and rule_can_block
                )
                threshold = float(constraints[constraint_name])
                if status != "calculable" or not np.isfinite(observed):
                    passed = False
                    check_status = (
                        "technical_error" if is_blocking else "not_evaluable"
                    )
                    reason_code = (
                        "required_guardrail_metric_non_finite"
                        if is_blocking
                        else "supporting_guardrail_metric_non_finite"
                    )
                    reason = (
                        error_message
                        or f"Guardrail metric {reported_metric} is unavailable."
                    )
                    technical_failure = bool(technical_failure or is_blocking)
                else:
                    passed = bool(
                        observed <= threshold
                        if comparator == "<="
                        else observed >= threshold
                    )
                    check_status = "pass" if passed else "fail"
                    direction = "above" if comparator == "<=" else "below"
                    reason_code = (
                        f"{rule_id}_within_limit"
                        if passed
                        else f"{rule_id}_{direction}_limit"
                    )
                    if (
                        severity == "conditional_blocking"
                        and not enough_independent_units
                    ):
                        reason = (
                            "Point-estimate alert only: too few independent "
                            "source images for a blocking worst-image rule."
                        )
                    else:
                        reason = f"Prespecified {severity} point-estimate guardrail."
                if is_blocking:
                    blocking_failures.append(not passed)
                rows.append(
                    {
                        "model_id": model_id,
                        "random_state": random_state,
                        "track_id": track_id,
                        "decision_scope": decision_scope,
                        "eligibility_status": eligibility_status,
                        "downstream_status": downstream_status,
                        "candidate_status": "pending",
                        "rule_id": rule_id,
                        "scope": scope,
                        "metric": reported_metric,
                        "severity": severity,
                        "n_independent_units": n_independent_units,
                        "min_independent_units": (
                            min_independent_units
                            if min_independent_units is not None
                            else np.nan
                        ),
                        "observed_value": observed,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "comparator": comparator,
                        "threshold": threshold,
                        "check_status": check_status,
                        "is_blocking": is_blocking,
                        "reason_code": reason_code,
                        "reason": reason,
                    }
                )

            # Spatial morphology acts on the direct pixel map only. The
            # pixel-to-object scope has no map and therefore no spatial check.
            if decision_scope == "direct" and projection_level == "pixel_projection":
                observed = spatial_lookup.get((model_id, random_state), np.nan)
                fragment_threshold = (
                    expcfg.SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN
                )
                if fragment_threshold is None:
                    is_blocking = False
                    severity = "diagnostic"
                    comparator = "not_thresholded"
                    threshold = np.nan
                    check_status = (
                        "diagnostic_only"
                        if np.isfinite(observed)
                        else "not_evaluable"
                    )
                    reason_code = "fragment_guardrail_not_prespecified"
                    reason = (
                        "Smallest-fragment recall is reported diagnostically; "
                        "no prespecified blocking threshold exists."
                    )
                else:
                    threshold = float(fragment_threshold)
                    severity = "blocking"
                    comparator = ">="
                    is_blocking = bool(
                        is_protocol_supported and is_primary_scope
                    )
                    if not np.isfinite(observed):
                        check_status = "technical_error"
                        technical_failure = True
                        passed = False
                        reason_code = "smallest_fragment_recall_missing"
                        reason = "Required spatial guardrail metric is unavailable."
                    else:
                        passed = bool(observed >= threshold)
                        check_status = "pass" if passed else "fail"
                        reason_code = (
                            "smallest_fragment_recall_within_limit"
                            if passed
                            else "smallest_fragment_recall_below_limit"
                        )
                        reason = "Prospectively frozen fragment guardrail."
                    if is_blocking:
                        blocking_failures.append(not passed)
                rows.append(
                    {
                        "model_id": model_id,
                        "random_state": random_state,
                        "track_id": track_id,
                        "decision_scope": decision_scope,
                        "eligibility_status": eligibility_status,
                        "downstream_status": downstream_status,
                        "candidate_status": "pending",
                        "rule_id": "smallest_fragment_recall",
                        "scope": "smallest_fragment_class",
                        "metric": "smallest_fragment_recall",
                        "severity": severity,
                        "n_independent_units": np.nan,
                        "min_independent_units": np.nan,
                        "observed_value": observed,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "comparator": comparator,
                        "threshold": threshold,
                        "check_status": check_status,
                        "is_blocking": is_blocking,
                        "reason_code": reason_code,
                        "reason": reason,
                    }
                )

            if technical_failure:
                candidate_status = "technical_failure"
            elif downstream_status == "diagnostic_only":
                candidate_status = "diagnostic_only"
            elif any(blocking_failures):
                candidate_status = "calculable_but_not_acceptable"
            else:
                candidate_status = "pass"
            status_by_scope[(model_id, random_state, decision_scope)] = candidate_status

    guardrails = pd.DataFrame(rows)
    if guardrails.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_VALIDATION_GUARDRAIL_COLUMNS)

    # Fill status only after all checks have been evaluated so every row for one
    # natural decision scope carries the same final status.
    guardrails["candidate_status"] = [
        status_by_scope.get(
            (str(model_id), int(random_state), str(decision_scope)),
            str(candidate_status),
        )
        for model_id, random_state, decision_scope, candidate_status in zip(
            guardrails["model_id"],
            guardrails["random_state"],
            guardrails["decision_scope"],
            guardrails["candidate_status"],
        )
    ]
    guardrails = guardrails.reindex(
        columns=expcfg.SIMCA_VALIDATION_GUARDRAIL_COLUMNS
    )
    natural_key = [
        "model_id",
        "random_state",
        "decision_scope",
        "rule_id",
        "scope",
        "metric",
    ]
    if guardrails.duplicated(natural_key).any():
        raise RuntimeError("validation_guardrails duplicates its natural key.")
    return guardrails.sort_values(natural_key, kind="mergesort").reset_index(drop=True)


def _assert_uncertainty_coverage_complementarity(
    validation_metrics: pd.DataFrame,
) -> None:
    """Ensure coverage remains the exact reported complement of uncertainty."""
    for uncertain_column, coverage_column in (
        ("uncertain_rate", "coverage_rate"),
        ("macro_image_uncertain_rate", "macro_image_coverage_rate"),
    ):
        if not {
            uncertain_column,
            coverage_column,
        }.issubset(validation_metrics.columns):
            continue
        uncertainty = pd.to_numeric(
            validation_metrics[uncertain_column], errors="coerce"
        ).to_numpy(dtype=float)
        coverage = pd.to_numeric(
            validation_metrics[coverage_column], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(uncertainty) & np.isfinite(coverage)
        inconsistent = finite & ~np.isclose(
            uncertainty + coverage,
            1.0,
            rtol=0.0,
            atol=1e-7,
        )
        if inconsistent.any():
            raise ValueError(
                f"{coverage_column} must equal 1 - {uncertain_column}."
            )


def run_exhaustive_locked_grid_evaluation(
    calibration_domain: pd.DataFrame,
    oof_object_predictions: pd.DataFrame,
    oof_pixel_predictions: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
    *,
    image_db: Mapping | None = None,
    spatial_lock: Mapping | None = None,
    spatial_supported_tracks: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    """Run task 27-28 and return the compact notebook 04A output tables."""
    validate_exhaustive_grid_inputs(
        calibration_domain,
        oof_object_predictions,
        oof_pixel_predictions,
        projection_eligibility,
    )
    fold_metrics, raw_audit, signatures = evaluate_locked_oof_domain(
        calibration_domain,
        oof_object_predictions,
        oof_pixel_predictions,
        image_db=image_db,
        spatial_lock=spatial_lock,
        spatial_supported_tracks=spatial_supported_tracks,
    )
    threshold_metrics = aggregate_grid_metrics(fold_metrics, projection_eligibility)
    catalog, duplicate_groups = build_duplicate_groups(calibration_domain, signatures)
    technical_audit = finalize_grid_audit(raw_audit, threshold_metrics, catalog)
    pareto_reference = build_pareto_reference(threshold_metrics, catalog)
    representatives = set(
        catalog.loc[
            catalog["calibration_id"].astype(str).eq(
                catalog["representative_calibration_id"].astype(str)
            ),
            "calibration_id",
        ].astype(str)
    )
    calculable_calibrations = set(threshold_metrics["calibration_id"].astype(str))
    configurations = catalog.loc[
        catalog["calibration_id"].astype(str).isin(
            representatives.intersection(calculable_calibrations)
        )
    ].reset_index(drop=True)
    calculable_not_acceptable = threshold_metrics.loc[
        threshold_metrics["acceptability_status"].eq("calculable_not_acceptable")
    ].reset_index(drop=True)
    if technical_audit["domain_config_id"].astype(str).nunique() != len(calibration_domain):
        raise RuntimeError("Each 03B domain configuration must have exactly one audit row.")
    if technical_audit["domain_config_id"].astype(str).duplicated().any():
        raise RuntimeError("Technical audit contains duplicated domain_config_id values.")
    return {
        "configurations": configurations,
        "fold_metrics": fold_metrics,
        "threshold_metrics": threshold_metrics,
        "pareto_reference": pareto_reference,
        "technical_audit": technical_audit,
        "duplicate_groups": duplicate_groups,
        "calculable_not_acceptable": calculable_not_acceptable,
    }


__all__ = [
    "finite_max",
    "finite_mean",
    "finite_min",
    "finite_std",
    "aggregate_grid_metrics",
    "build_configuration_catalog",
    "build_duplicate_groups",
    "build_pareto_reference",
    "evaluate_locked_oof_domain",
    "evaluate_locked_validation_predictions",
    "finalize_grid_audit",
    "build_validation_guardrails",
    "run_exhaustive_locked_grid_evaluation",
    "summarize_grouped_decisions",
    "validate_exhaustive_grid_inputs",
]
