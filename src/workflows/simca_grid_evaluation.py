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


def _finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _finite_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else np.nan


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
        macro_image_miss = _finite_mean(images["target_miss_rate"])
        macro_image_false = _finite_mean(images["false_accept_rate"])
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
            "worst_fold_target_miss_rate": _finite_max(folds["target_miss_rate"]),
            "worst_fold_false_accept_rate": _finite_max(folds["false_accept_rate"]),
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
    candidate_pool: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    *,
    technical_errors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply frozen decisions and compute task-31 batch-3 metrics.

    Every candidate yields overall and per-image rows, or one explicit error
    row. Equivalent predictions and decisions are tagged but never removed.
    """
    required_candidates = {
        "validation_candidate_id",
        "calibration_id",
        "domain_config_id",
        "projection_config_id",
        "evaluation_track",
        "track_id",
        "projection_level",
        "decision_mode",
        "random_state",
        "direct_2way_threshold",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    }
    missing = sorted(required_candidates - set(candidate_pool.columns))
    if missing:
        raise KeyError(f"Missing validation-candidate columns: {missing}")
    error_lookup: dict[str, list[dict]] = {}
    if technical_errors is not None and len(technical_errors):
        for projection_id, group in technical_errors.groupby(
            "projection_config_id", sort=False
        ):
            error_lookup[str(projection_id)] = group.to_dict("records")

    lookups = {}
    for level, frame in (
        ("object_projection", object_predictions),
        ("pixel_projection", pixel_predictions),
    ):
        if frame.empty:
            lookups[level] = {}
            continue
        required_predictions = {
            "projection_config_id",
            "random_state",
            "source_image",
            "object_id",
            "truth",
            "simca_margin",
        }
        missing = sorted(required_predictions - set(frame.columns))
        if missing:
            raise KeyError(f"Missing {level} validation columns: {missing}")
        lookups[level] = {
            (str(key[0]), int(key[1])): np.asarray(indices, dtype=int)
            for key, indices in frame.groupby(
                ["projection_config_id", "random_state"], sort=False
            ).indices.items()
        }

    metric_parts: list[pd.DataFrame] = []
    completed_calibrations: set[str] = set()
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
    for candidate in candidate_pool.to_dict("records"):
        calibration_id = str(candidate["calibration_id"])
        validation_candidate_id = str(candidate["validation_candidate_id"])
        projection_id = str(candidate["projection_config_id"])
        base = {
            "validation_candidate_id": validation_candidate_id,
            "calibration_id": calibration_id,
            "domain_config_id": str(candidate["domain_config_id"]),
            "evaluation_track": str(candidate["evaluation_track"]),
            "track_id": str(candidate["track_id"]),
            "decision_mode": str(candidate["decision_mode"]),
            "projection_level": str(candidate["projection_level"]),
            "random_state": int(candidate["random_state"]),
            "map_variant": "raw",
        }
        try:
            if projection_id in error_lookup:
                first = error_lookup[projection_id][0]
                raise RuntimeError(
                    f"{first.get('stage', 'projection')}: "
                    f"{first.get('error_type', 'technical_error')}: "
                    f"{first.get('error_message', '')}"
                )
            level = str(candidate["projection_level"])
            predictions = (
                object_predictions
                if level == "object_projection"
                else pixel_predictions
            )
            key = (projection_id, int(candidate["random_state"]))
            positions = lookups.get(level, {}).get(key)
            if positions is None or not len(positions):
                raise RuntimeError(f"No batch-3 predictions for {key}.")
            observations = predictions.iloc[positions].copy()
            numeric = observations[list(diagnostic_columns)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise RuntimeError("A validation diagnostic is non-finite.")
            if observations["truth"].astype(bool).nunique() != 2:
                raise RuntimeError("Batch-3 predictions do not cover both classes.")
            target, uncertain = apply_locked_margin_decision(
                observations["simca_margin"].to_numpy(dtype=float),
                str(candidate["decision_mode"]),
                direct_2way_threshold=candidate["direct_2way_threshold"],
                three_way_lower_threshold=candidate["three_way_lower_threshold"],
                three_way_upper_threshold=candidate["three_way_upper_threshold"],
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
            metrics = pd.concat([overall, by_image], ignore_index=True, sort=False)
            for column, value in base.items():
                metrics[column] = value
            entities = ["source_image", "object_id"]
            if level == "pixel_projection":
                entities.extend(["row", "col"])
            prediction_signature = stable_frame_signature(
                observations,
                [*entities, *diagnostic_columns],
                sort_columns=entities,
                round_decimals=expcfg.SIMCA_CONCAT_REFIT_SIGNATURE_ROUND_DECIMALS,
            )
            decisions = observations[entities].copy()
            decisions["target_decision"] = target
            decisions["uncertain_decision"] = uncertain
            decision_signature = stable_frame_signature(
                decisions,
                [*entities, "target_decision", "uncertain_decision"],
                sort_columns=entities,
                round_decimals=None,
            )
            metrics["prediction_signature"] = prediction_signature
            metrics["decision_signature"] = decision_signature
            metrics["status"] = "calculable"
            metrics["error_type"] = ""
            metrics["error_message"] = ""
            metric_parts.append(_add_validation_rate_intervals(metrics))
            completed_calibrations.add(calibration_id)
        except Exception as exc:
            metric_parts.append(
                pd.DataFrame(
                    [{
                        **base,
                        "aggregation_level": "overall",
                        "group_id": "all",
                        "status": "technical_failure",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }]
                )
            )

    metrics = (
        pd.concat(metric_parts, ignore_index=True, sort=False)
        if metric_parts
        else pd.DataFrame(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    )
    metrics["prediction_signature"] = metrics.get(
        "prediction_signature", pd.Series("", index=metrics.index)
    ).fillna("").astype(str)
    metrics["decision_signature"] = metrics.get(
        "decision_signature", pd.Series("", index=metrics.index)
    ).fillna("").astype(str)
    metrics["prediction_equivalence_group_id"] = _equivalence_group_ids(
        metrics["evaluation_track"],
        metrics["prediction_signature"],
        prefix="pred_eq",
    )
    metrics["decision_equivalence_group_id"] = _equivalence_group_ids(
        metrics["evaluation_track"],
        metrics["decision_signature"],
        prefix="decision_eq",
    )

    missing_tracks = sorted(
        set(expcfg.SIMCA_EVALUATION_TRACKS)
        - set(candidate_pool["evaluation_track"].astype(str))
    )
    if missing_tracks:
        missing_rows = []
        for track in missing_tracks:
            spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[track]
            missing_rows.append(
                {
                    "calibration_id": "",
                    "validation_candidate_id": "",
                    "domain_config_id": "",
                    "evaluation_track": track,
                    "track_id": spec["track_id"],
                    "decision_mode": spec["decision_mode"],
                    "projection_level": spec["projection_level"],
                    "map_variant": "raw",
                    "aggregation_level": "overall",
                    "group_id": "all",
                    "status": "not_evaluable_no_calibrated_candidate",
                    "error_type": "EmptyCalibratedDomain",
                    "error_message": (
                        "No frozen 03B/04A candidate is available for this track."
                    ),
                }
            )
        metrics = pd.concat(
            [metrics, pd.DataFrame(missing_rows)], ignore_index=True, sort=False
        )
    return metrics.reindex(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)


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


def build_validation_guardrails(
    candidate_pool: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    *,
    spatial_component_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply prespecified validation constraints without a composite score."""
    _assert_uncertainty_coverage_complementarity(validation_metrics)
    overall = validation_metrics.loc[
        validation_metrics["aggregation_level"].astype(str).eq("overall")
    ].copy()
    image = validation_metrics.loc[
        validation_metrics["aggregation_level"].astype(str).eq("source_image")
        & validation_metrics["status"].astype(str).eq("calculable")
    ].copy()
    overall_lookup = {
        str(row["validation_candidate_id"]): row
        for row in overall.to_dict("records")
        if str(row.get("validation_candidate_id", ""))
    }
    image_lookup = {
        str(validation_candidate_id): group
        for validation_candidate_id, group in image.groupby(
            "validation_candidate_id", sort=False
        )
    }
    spatial_lookup: dict[str, float] = {}
    if spatial_component_metrics is not None and len(spatial_component_metrics):
        spatial_overall = spatial_component_metrics.loc[
            spatial_component_metrics["aggregation_level"].astype(str).eq("overall")
            & spatial_component_metrics["map_variant"].astype(str).eq(
                "locked_postprocessed"
            )
        ]
        spatial_lookup = {
            str(row["validation_candidate_id"]): float(
                row["smallest_fragment_recall"]
            )
            for row in spatial_overall.to_dict("records")
        }

    rows: list[dict] = []
    candidate_statuses: dict[str, str] = {}
    for candidate in candidate_pool.to_dict("records"):
        validation_candidate_id = str(candidate["validation_candidate_id"])
        calibration_id = str(candidate["calibration_id"])
        metric_row = overall_lookup.get(validation_candidate_id)
        common = {
            "validation_candidate_id": validation_candidate_id,
            "calibration_id": calibration_id,
            "evaluation_track": str(candidate["evaluation_track"]),
            "track_id": str(candidate["track_id"]),
            "random_state": int(candidate["random_state"]),
            "eligibility_status": str(candidate["eligibility_status"]),
        }
        if metric_row is None or str(metric_row.get("status")) != "calculable":
            candidate_statuses[validation_candidate_id] = "technical_failure"
            rows.append(
                {
                    **common,
                    "scope": "overall",
                    "metric": "technical_calculability",
                    "comparator": "is",
                    "check_status": "fail",
                    "reason": (
                        "No calculable batch-3 metric was produced for the frozen candidate."
                    ),
                }
            )
            continue
        mode = str(candidate["decision_mode"])
        constraints = expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS[mode]
        pixel_primary = str(candidate["projection_level"]) == "pixel_projection"
        candidate_checks: list[bool] = []
        technical_guardrail_error = False
        for scope in expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_SCOPES:
            scope_specs = expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS[
                mode
            ][scope]
            for metric, constraint_name, comparator in scope_specs:
                if scope == "overall":
                    reported_metric = (
                        f"macro_image_{metric}" if pixel_primary else metric
                    )
                    observed = float(metric_row.get(reported_metric, np.nan))
                    ci_low = float(
                        metric_row.get(f"{reported_metric}_ci_low", np.nan)
                    )
                    ci_high = float(
                        metric_row.get(f"{reported_metric}_ci_high", np.nan)
                    )
                else:
                    reported_metric = metric
                    group = image_lookup.get(
                        validation_candidate_id, pd.DataFrame()
                    )
                    values = pd.to_numeric(
                        group.get(metric, pd.Series(dtype=float)), errors="coerce"
                    )
                    values = values[np.isfinite(values)]
                    if values.empty:
                        observed = ci_low = ci_high = np.nan
                    else:
                        index = values.idxmax() if comparator == "<=" else values.idxmin()
                        observed = float(values.loc[index])
                        ci_low = float(group.loc[index].get(f"{metric}_ci_low", np.nan))
                        ci_high = float(group.loc[index].get(f"{metric}_ci_high", np.nan))
                threshold = float(constraints[constraint_name])
                if not np.isfinite(observed):
                    passed = False
                    check_status = "technical_error"
                    reason = "required_guardrail_metric_non_finite"
                    technical_guardrail_error = True
                else:
                    passed = bool(
                        observed <= threshold
                        if comparator == "<="
                        else observed >= threshold
                    )
                    check_status = "pass" if passed else "fail"
                    reason = "prespecified_point_estimate_guardrail"
                candidate_checks.append(passed)
                rows.append(
                    {
                        **common,
                        "scope": scope,
                        "metric": reported_metric,
                        "observed_value": observed,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "comparator": comparator,
                        "threshold": threshold,
                        "check_status": check_status,
                        "reason": reason,
                        "prediction_equivalence_group_id": metric_row.get(
                            "prediction_equivalence_group_id", ""
                        ),
                        "decision_equivalence_group_id": metric_row.get(
                            "decision_equivalence_group_id", ""
                        ),
                    }
                )
        if str(candidate["projection_level"]) == "pixel_projection":
            observed = spatial_lookup.get(validation_candidate_id, np.nan)
            fragment_threshold = expcfg.SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN
            if fragment_threshold is None:
                fragment_status = "diagnostic_only_no_prespecified_threshold"
                passed = True
                comparator = "not_thresholded"
                threshold = np.nan
            else:
                threshold = float(fragment_threshold)
                comparator = ">="
                passed = bool(np.isfinite(observed) and observed >= threshold)
                fragment_status = "pass" if passed else "fail"
                candidate_checks.append(passed)
            rows.append(
                {
                    **common,
                    "scope": "smallest_fragment_class",
                    "metric": "smallest_fragment_recall",
                    "observed_value": observed,
                    "comparator": comparator,
                    "threshold": threshold,
                    "check_status": fragment_status,
                    "reason": (
                        "reported_without_post_batch3_threshold_choice"
                        if fragment_threshold is None
                        else "prospectively_frozen_fragment_guardrail"
                    ),
                    "prediction_equivalence_group_id": metric_row.get(
                        "prediction_equivalence_group_id", ""
                    ),
                    "decision_equivalence_group_id": metric_row.get(
                        "decision_equivalence_group_id", ""
                    ),
                }
            )
        if str(candidate["eligibility_status"]) in (
            expcfg.SIMCA_CONCAT_REFIT_UNSUPPORTED_ELIGIBILITY_STATUSES
        ):
            candidate_statuses[validation_candidate_id] = (
                "unsupported_domain_shift_diagnostic"
            )
        elif technical_guardrail_error:
            candidate_statuses[validation_candidate_id] = "technical_failure"
        elif all(candidate_checks):
            candidate_statuses[validation_candidate_id] = "pass"
        else:
            candidate_statuses[validation_candidate_id] = (
                "calculable_but_not_acceptable"
            )

    guardrails = pd.DataFrame(rows)
    if len(guardrails):
        guardrails["candidate_status"] = guardrails[
            "validation_candidate_id"
        ].map(candidate_statuses)
    missing_tracks = sorted(
        set(expcfg.SIMCA_EVALUATION_TRACKS)
        - set(candidate_pool["evaluation_track"].astype(str))
    )
    missing_rows = [
        {
            "calibration_id": "",
            "validation_candidate_id": "",
            "evaluation_track": track,
            "track_id": expcfg.SIMCA_EVALUATION_TRACK_IDS[track],
            "random_state": np.nan,
            "eligibility_status": "unsupported_internal_calibration",
            "candidate_status": "not_evaluable_no_calibrated_candidate",
            "scope": "overall",
            "metric": "calibrated_candidate_availability",
            "comparator": "is",
            "check_status": "not_evaluable",
            "reason": "03B produced no calibrated candidate for this track.",
        }
        for track in missing_tracks
    ]
    if missing_rows:
        guardrails = pd.concat(
            [guardrails, pd.DataFrame(missing_rows)],
            ignore_index=True,
            sort=False,
        )
    return guardrails.reindex(columns=expcfg.SIMCA_VALIDATION_GUARDRAIL_COLUMNS)


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
