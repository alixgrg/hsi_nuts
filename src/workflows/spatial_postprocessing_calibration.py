"""OOF-only spatial post-processing calibration for protocol notebook 03C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import zlib

import numpy as np
import pandas as pd
from skimage import morphology

from src import experiment_config as expcfg
from src.decision.metrics import (
    apply_locked_margin_decision,
    component_detection_table,
    component_detection_metrics,
)
from src.decision.truth import pure_image_class_truth
from src.protocol_governance import sha256_dataframe


_REMOVE_SMALL_OBJECTS_HAS_MAX_SIZE = (
    "max_size" in inspect.signature(morphology.remove_small_objects).parameters
)


def _candidate_id(payload: Mapping) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "spatial_" + hashlib.sha256(encoded).hexdigest()[:16]


def _payload_hash(payload: Mapping) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_spatial_candidate_grid(
    *,
    connectivities: Sequence[int] = expcfg.SPATIAL_CALIBRATION_CONNECTIVITIES,
    operations: Sequence[str] = (
        expcfg.SPATIAL_CALIBRATION_MORPHOLOGY_OPERATIONS
    ),
    radii: Sequence[int] = expcfg.SPATIAL_CALIBRATION_MORPHOLOGY_RADII,
    min_areas: Sequence[int] = expcfg.SPATIAL_CALIBRATION_MIN_AREAS,
) -> pd.DataFrame:
    """Build the deduplicated, predeclared spatial grid."""
    allowed_operations = {"none", "opening", "closing", "opening_closing"}
    rows = []
    for connectivity in sorted(set(map(int, connectivities))):
        if connectivity not in {1, 2}:
            raise ValueError("2D connectivity must be 1 or 2.")
        for operation in dict.fromkeys(map(str, operations)):
            if operation not in allowed_operations:
                raise ValueError(f"Unknown morphology operation: {operation}")
            active_radii = (0,) if operation == "none" else tuple(
                radius for radius in sorted(set(map(int, radii))) if radius > 0
            )
            for radius in active_radii:
                for min_area in sorted(set(map(int, min_areas))):
                    if min_area < 0:
                        raise ValueError("min_area must be non-negative.")
                    payload = {
                        "connectivity": connectivity,
                        "morphology_operation": operation,
                        "morphology_radius": radius,
                        "min_area_pixels": min_area,
                    }
                    rows.append({"spatial_candidate_id": _candidate_id(payload), **payload})
    result = pd.DataFrame(rows).drop_duplicates("spatial_candidate_id")
    if result.empty:
        raise RuntimeError("The spatial calibration grid is empty.")
    return result.reset_index(drop=True)


def _validate_spatial_candidate_grid(candidate_grid: pd.DataFrame) -> pd.DataFrame:
    """Validate identities and parameters before any candidate is evaluated."""
    required = [
        "spatial_candidate_id",
        "connectivity",
        "morphology_operation",
        "morphology_radius",
        "min_area_pixels",
    ]
    missing = sorted(set(required) - set(candidate_grid.columns))
    if missing:
        raise KeyError(f"Missing spatial candidate columns: {missing}")
    grid = candidate_grid[required].copy()
    if grid.empty:
        raise RuntimeError("The spatial calibration grid is empty.")
    if grid["spatial_candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Spatial candidate identifiers must be unique.")
    grid["connectivity"] = pd.to_numeric(
        grid["connectivity"], errors="raise"
    ).astype(int)
    grid["morphology_radius"] = pd.to_numeric(
        grid["morphology_radius"], errors="raise"
    ).astype(int)
    grid["min_area_pixels"] = pd.to_numeric(
        grid["min_area_pixels"], errors="raise"
    ).astype(int)
    allowed_operations = {"none", "opening", "closing", "opening_closing"}
    for row in grid.itertuples(index=False):
        operation = str(row.morphology_operation)
        connectivity = int(row.connectivity)
        radius = int(row.morphology_radius)
        min_area = int(row.min_area_pixels)
        if connectivity not in {1, 2}:
            raise ValueError("2D connectivity must be 1 or 2.")
        if operation not in allowed_operations:
            raise ValueError(f"Unknown morphology operation: {operation}")
        if (operation == "none" and radius != 0) or (
            operation != "none" and radius <= 0
        ):
            raise ValueError("Morphology operation and radius are inconsistent.")
        if min_area < 0:
            raise ValueError("min_area must be non-negative.")
        payload = {
            "connectivity": connectivity,
            "morphology_operation": operation,
            "morphology_radius": radius,
            "min_area_pixels": min_area,
        }
        if str(row.spatial_candidate_id) != _candidate_id(payload):
            raise RuntimeError(
                "A spatial candidate identifier does not match its parameters."
            )
    return grid.reset_index(drop=True)


def apply_spatial_postprocessing(
    target_mask,
    uncertain_mask,
    valid_mask,
    *,
    connectivity: int,
    morphology_operation: str,
    morphology_radius: int,
    min_area_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply morphology to positives while preserving uncertainty exactly."""
    valid = np.asarray(valid_mask, dtype=bool)
    uncertain = np.asarray(uncertain_mask, dtype=bool) & valid
    target = np.asarray(target_mask, dtype=bool) & valid & ~uncertain
    operation = str(morphology_operation)
    radius = int(morphology_radius)
    if operation == "none":
        if radius != 0:
            raise ValueError("The 'none' operation requires radius=0.")
        cleaned = target.copy()
    else:
        if radius <= 0:
            raise ValueError("Morphological operations require a positive radius.")
        footprint = morphology.disk(radius)
        if operation == "opening":
            cleaned = morphology.opening(target, footprint=footprint)
        elif operation == "closing":
            cleaned = morphology.closing(target, footprint=footprint)
        elif operation == "opening_closing":
            cleaned = morphology.closing(
                morphology.opening(target, footprint=footprint),
                footprint=footprint,
            )
        else:
            raise ValueError(f"Unknown morphology operation: {operation}")
    cleaned &= valid & ~uncertain
    if int(min_area_pixels) > 1:
        if _REMOVE_SMALL_OBJECTS_HAS_MAX_SIZE:
            cleaned = morphology.remove_small_objects(
                cleaned,
                max_size=int(min_area_pixels) - 1,
                connectivity=int(connectivity),
            )
        else:
            cleaned = morphology.remove_small_objects(
                cleaned,
                min_size=int(min_area_pixels),
                connectivity=int(connectivity),
            )
    return np.asarray(cleaned, dtype=bool), uncertain


def build_spatial_calibration_input(
    oof_pixels: pd.DataFrame,
    selected_executions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    image_db: dict,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    allowed_batches: Sequence[int] = expcfg.SPATIAL_CALIBRATION_ALLOWED_BATCHES,
    required_classes: Sequence[str] = expcfg.SPATIAL_CALIBRATION_REQUIRED_CLASSES,
) -> pd.DataFrame:
    """Build compact pixel maps from normalized selected 03B tables."""
    required_oof = {
        "projection_id",
        "source_image",
        "batch",
        "row",
        "col",
        "simca_margin",
    }
    missing = sorted(required_oof - set(oof_pixels.columns))
    if missing:
        raise KeyError(f"Missing OOF pixel columns: {missing}")
    mapping_columns = [
        "model_id",
        "random_state",
        "track_id",
        "projection_id",
        "decision_mode",
        "projection_level",
    ]
    missing = sorted(set(mapping_columns) - set(selected_executions.columns))
    if missing:
        raise KeyError(f"Missing selected-execution columns: {missing}")
    mapping = selected_executions.loc[
        selected_executions["projection_level"].astype(str).eq("pixel_projection"),
        mapping_columns,
    ].copy()
    if mapping.empty:
        raise RuntimeError("No selected pixel-projection execution is available.")
    execution_keys = ["model_id", "random_state"]
    if mapping.duplicated(execution_keys).any():
        raise RuntimeError("Selected spatial execution keys must be unique.")

    required_thresholds = set(
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
    )
    missing = sorted(required_thresholds - set(selected_thresholds.columns))
    if missing:
        raise KeyError(f"Missing selected-threshold columns: {missing}")
    direct = selected_thresholds.loc[
        selected_thresholds["decision_scope"].astype(str).eq("direct"),
        [*execution_keys, "lower_threshold", "upper_threshold"],
    ].copy()
    if direct.duplicated(execution_keys).any():
        raise RuntimeError("A selected spatial execution has duplicate direct thresholds.")
    mapping = mapping.merge(
        direct,
        on=execution_keys,
        how="left",
        validate="one_to_one",
    )
    lower = pd.to_numeric(mapping["lower_threshold"], errors="coerce")
    upper = pd.to_numeric(mapping["upper_threshold"], errors="coerce")
    if not np.isfinite(np.column_stack([lower, upper])).all():
        raise RuntimeError("A selected spatial execution has no finite direct threshold.")

    out = oof_pixels.merge(
        mapping,
        on="projection_id",
        how="inner",
        validate="many_to_many",
    )
    if out.empty:
        raise RuntimeError("No OOF pixel row matches the selected 03B executions.")
    observed_keys = out[execution_keys].drop_duplicates()
    missing_executions = mapping[execution_keys].merge(
        observed_keys,
        on=execution_keys,
        how="left",
        indicator=True,
    )
    if missing_executions["_merge"].eq("left_only").any():
        raise RuntimeError("A selected pixel execution has no OOF pixel prediction.")
    batches = set(pd.to_numeric(out["batch"], errors="raise").astype(int))
    if not batches.issubset(set(map(int, allowed_batches))):
        raise RuntimeError(f"Forbidden batch in spatial calibration: {sorted(batches)}")

    truth_cache = {}
    observed_classes = set()
    for image_key in sorted(out["source_image"].astype(str).unique()):
        truth = pure_image_class_truth(
            image_key,
            image_db,
            target_class=str(target_class),
            allowed_batches=allowed_batches,
        )
        truth_cache[image_key] = truth
        observed_classes.add(str(truth.provenance["class_name"]))
    missing_classes = sorted(set(map(str, required_classes)) - observed_classes)
    if missing_classes:
        raise RuntimeError(
            "Spatial OOF truth must cover both pure classes; missing "
            f"{missing_classes}."
        )

    out["true_target"] = False
    truth_available = np.zeros(len(out), dtype=bool)
    for image_key, indices in out.groupby("source_image", sort=False).groups.items():
        truth = truth_cache[str(image_key)]
        row = pd.to_numeric(out.loc[indices, "row"], errors="raise").astype(int).to_numpy()
        col = pd.to_numeric(out.loc[indices, "col"], errors="raise").astype(int).to_numpy()
        shape = truth.truth_mask.shape
        inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        if not inside.all():
            raise RuntimeError(f"OOF coordinates outside image {image_key}.")
        out.loc[indices, "true_target"] = truth.truth_mask[row, col]
        truth_available[out.index.get_indexer(indices)] = truth.available_mask[row, col]
    if not truth_available.all():
        raise RuntimeError(
            "An OOF projected pixel falls outside the pure-image segmented ROI."
        )
    two_way = out["decision_mode"].astype(str).eq("2way").to_numpy()
    three_way = out["decision_mode"].astype(str).eq("3way").to_numpy()
    if not (two_way | three_way).all():
        unknown = sorted(
            set(out.loc[~(two_way | three_way), "decision_mode"].astype(str))
        )
        raise RuntimeError(f"Unknown spatial decision modes: {unknown}")
    lower_threshold = pd.to_numeric(
        out["lower_threshold"], errors="coerce"
    ).to_numpy(dtype=float)
    upper_threshold = pd.to_numeric(
        out["upper_threshold"], errors="coerce"
    ).to_numpy(dtype=float)
    try:
        target, uncertain = apply_locked_margin_decision(
            pd.to_numeric(out["simca_margin"], errors="coerce").to_numpy(),
            out["decision_mode"].astype(str).to_numpy(),
            direct_2way_threshold=np.where(two_way, lower_threshold, np.nan),
            three_way_lower_threshold=np.where(
                three_way, lower_threshold, np.nan
            ),
            three_way_upper_threshold=np.where(
                three_way, upper_threshold, np.nan
            ),
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Unknown decision modes"):
            message = message.replace(
                "Unknown decision modes", "Unknown spatial decision modes", 1
            )
        elif "2-way threshold" in message:
            message = "Invalid selected 2-way direct threshold."
        elif "3-way threshold" in message:
            message = "Invalid selected 3-way direct thresholds."
        raise RuntimeError(message) from exc
    out["raw_uncertain"] = uncertain
    out["raw_target"] = target
    coordinate_keys = [
        "model_id",
        "random_state",
        "source_image",
        "row",
        "col",
    ]
    duplicated_coordinates = out.duplicated(coordinate_keys, keep=False)
    if duplicated_coordinates.any():
        examples = out.loc[
            duplicated_coordinates, coordinate_keys
        ].head(10)
        raise RuntimeError(
            "A spatial map contains duplicated pixel coordinates: "
            f"{examples.to_dict('records')}"
        )
    out["truth_level"] = expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE
    return out.reindex(columns=expcfg.SPATIAL_CALIBRATION_INPUT_COLUMNS)


def _maps_for_group(group: pd.DataFrame, image_db: dict) -> dict[str, np.ndarray]:
    image_key = str(group["source_image"].iloc[0])
    if group["source_image"].astype(str).nunique() != 1:
        raise RuntimeError("A spatial map group mixes several source images.")
    if group.duplicated(["row", "col"], keep=False).any():
        raise RuntimeError(f"Spatial coordinates are duplicated for image {image_key}.")
    shape = np.asarray(image_db[image_key]["labels"]).shape
    row = pd.to_numeric(group["row"], errors="raise").astype(int).to_numpy()
    col = pd.to_numeric(group["col"], errors="raise").astype(int).to_numpy()
    valid = np.zeros(shape, dtype=bool)
    target = np.zeros(shape, dtype=bool)
    uncertain = np.zeros(shape, dtype=bool)
    truth = np.zeros(shape, dtype=bool)
    valid[row, col] = True
    target[row, col] = group["raw_target"].astype(bool).to_numpy()
    uncertain[row, col] = group["raw_uncertain"].astype(bool).to_numpy()
    truth[row, col] = group["true_target"].astype(bool).to_numpy()
    return {"valid": valid, "target": target, "uncertain": uncertain, "truth": truth}


def _evaluate_maps(
    maps_by_image: list[dict[str, np.ndarray]],
    *,
    connectivity: int,
    candidate: Mapping | None,
) -> tuple[dict, pd.DataFrame]:
    totals = {
        "valid": 0,
        "truth": 0,
        "prediction": 0,
        "intersection": 0,
        "union": 0,
        "uncertain": 0,
        "truth_components": 0,
        "predicted_components": 0,
        "detected_truth_components": 0,
        "matched_predicted_components": 0,
        "split_truth_components": 0,
        "merged_predicted_components": 0,
    }
    fragments = []
    for image_maps in maps_by_image:
        valid = image_maps["valid"]
        uncertain = image_maps["uncertain"] & valid
        evaluable = valid & ~uncertain
        truth = image_maps["truth"] & evaluable
        if candidate is None:
            prediction = image_maps["target"] & evaluable
        else:
            prediction, uncertain = apply_spatial_postprocessing(
                image_maps["target"],
                uncertain,
                valid,
                connectivity=int(candidate["connectivity"]),
                morphology_operation=str(candidate["morphology_operation"]),
                morphology_radius=int(candidate["morphology_radius"]),
                min_area_pixels=int(candidate["min_area_pixels"]),
            )
            prediction &= evaluable
        components, fragment = component_detection_metrics(
            truth,
            prediction,
            valid_mask=evaluable,
            connectivity=int(connectivity),
            return_fragment_table=True,
            area_upper_bounds=(
                expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_UPPER_BOUNDS
            ),
            area_labels=expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS,
        )
        totals["valid"] += int(valid.sum())
        totals["truth"] += int(truth.sum())
        totals["prediction"] += int(prediction.sum())
        totals["intersection"] += int((truth & prediction).sum())
        totals["union"] += int((truth | prediction).sum())
        totals["uncertain"] += int(uncertain.sum())
        for key in (
            "truth_components",
            "predicted_components",
            "detected_truth_components",
            "matched_predicted_components",
            "split_truth_components",
            "merged_predicted_components",
        ):
            totals[key] += int(components[f"n_{key}"])
        if not fragment.empty:
            fragments.append(fragment)
    positive_sum = totals["truth"] + totals["prediction"]
    tp = totals["intersection"]
    metrics = {
        "n_images": int(len(maps_by_image)),
        "n_valid_pixels": totals["valid"],
        "dice": 2.0 * tp / positive_sum if positive_sum else 1.0,
        "iou": tp / totals["union"] if totals["union"] else 1.0,
        "pixel_precision": tp / totals["prediction"] if totals["prediction"] else np.nan,
        "pixel_recall": tp / totals["truth"] if totals["truth"] else np.nan,
        "component_precision": (
            totals["matched_predicted_components"] / totals["predicted_components"]
            if totals["predicted_components"] else np.nan
        ),
        "component_recall": (
            totals["detected_truth_components"] / totals["truth_components"]
            if totals["truth_components"] else np.nan
        ),
        "split_rate": (
            totals["split_truth_components"] / totals["truth_components"]
            if totals["truth_components"] else 0.0
        ),
        "merge_rate": (
            totals["merged_predicted_components"] / totals["predicted_components"]
            if totals["predicted_components"] else 0.0
        ),
        "uncertain_pixel_rate": (
            totals["uncertain"] / totals["valid"] if totals["valid"] else np.nan
        ),
    }
    fragment_table = (
        pd.concat(fragments, ignore_index=True)
        if fragments
        else pd.DataFrame(columns=["area_class", "detected", "best_iou"])
    )
    ordered_labels = tuple(
        map(str, expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS)
    )
    observed_labels = set(fragment_table["area_class"].astype(str))
    smallest = next(
        (label_name for label_name in ordered_labels if label_name in observed_labels),
        None,
    )
    smallest_rows = (
        fragment_table.loc[fragment_table["area_class"].eq(smallest)]
        if smallest is not None
        else fragment_table.iloc[0:0]
    )
    metrics["smallest_fragment_recall"] = (
        float(smallest_rows["detected"].mean()) if len(smallest_rows) else np.nan
    )
    return metrics, fragment_table


def _summarize_fragment_classes(
    fragments: pd.DataFrame,
    *,
    metadata: Mapping,
) -> list[dict]:
    bounds = tuple(map(int, expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_UPPER_BOUNDS))
    labels = tuple(map(str, expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS))
    minima = (1,) + tuple(value + 1 for value in bounds)
    maxima: tuple[int | None, ...] = bounds + (None,)
    observed_labels = set(fragments["area_class"].astype(str))
    smallest_observed = next(
        (label_name for label_name in labels if label_name in observed_labels),
        None,
    )
    rows = []
    for label_name, minimum, maximum in zip(labels, minima, maxima):
        group = fragments.loc[fragments["area_class"].eq(label_name)]
        rows.append(
            {
                **metadata,
                "area_class": label_name,
                "min_area_pixels": int(minimum),
                "max_area_pixels": (
                    float(maximum) if maximum is not None else np.nan
                ),
                "n_truth_fragments": int(len(group)),
                "n_detected_fragments": int(group["detected"].sum()) if len(group) else 0,
                "fragment_recall": float(group["detected"].mean()) if len(group) else np.nan,
                "mean_best_iou": float(group["best_iou"].mean()) if len(group) else np.nan,
                "is_smallest_class": bool(label_name == smallest_observed),
            }
        )
    return rows


def _select_candidate_within_track(
    metrics: pd.DataFrame,
    *,
    track_id: str,
    tolerance: float,
) -> str:
    """Select one spatial candidate using only executions from one track."""
    track_id = str(track_id)
    candidates = metrics.loc[
        metrics["map_variant"].astype(str).eq("postprocessed")
        & metrics["track_id"].astype(str).eq(track_id)
    ].copy()
    if candidates.empty:
        raise RuntimeError(
            f"No post-processed spatial candidate is available for track {track_id}."
        )

    maximize = list(expcfg.SPATIAL_CALIBRATION_SELECTION_MAXIMIZE)
    minimize = list(expcfg.SPATIAL_CALIBRATION_SELECTION_MINIMIZE)
    selection_metrics = [*maximize, *minimize]
    parameter_columns = list(expcfg.SPATIAL_CALIBRATION_PARAMETER_COLUMNS)
    required = {
        "spatial_candidate_id", "model_id", "random_state", "track_id",
        *selection_metrics, *parameter_columns,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise KeyError(f"Missing spatial selection columns: {missing}")

    candidate_keys = ["spatial_candidate_id", "model_id", "random_state"]
    if candidates.duplicated(candidate_keys).any():
        raise RuntimeError(
            f"Track {track_id} contains duplicate candidate/execution metrics."
        )

    parameter_counts = candidates.groupby(
        "spatial_candidate_id", sort=False
    )[parameter_columns].nunique(dropna=False)
    if parameter_counts.gt(1).any().any():
        raise RuntimeError(
            f"A spatial candidate has conflicting parameters inside track {track_id}."
        )

    execution_keys = ["model_id", "random_state"]
    n_expected_executions = len(candidates[execution_keys].drop_duplicates())
    coverage = (
        candidates[candidate_keys].drop_duplicates()
        .groupby("spatial_candidate_id", sort=False).size()
    )
    incomplete = coverage[coverage.ne(n_expected_executions)]
    if len(incomplete):
        raise RuntimeError(
            f"Spatial candidates do not cover the same selected executions in "
            f"track {track_id}: {incomplete.to_dict()}"
        )

    numeric = candidates[selection_metrics].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    row_counts = candidates.groupby("spatial_candidate_id", sort=False).size()
    complete_counts = (
        candidates.loc[finite].groupby("spatial_candidate_id", sort=False).size()
        .reindex(row_counts.index, fill_value=0)
    )
    selectable_ids = row_counts.index[complete_counts.eq(row_counts)]
    if len(selectable_ids) == 0:
        raise RuntimeError(
            f"No spatial candidate has complete finite metrics in track {track_id}."
        )
    candidates.loc[:, selection_metrics] = numeric
    candidates = candidates.loc[
        candidates["spatial_candidate_id"].isin(selectable_ids)
    ].copy()

    aggregation = {metric: (metric, "mean") for metric in selection_metrics}
    aggregation.update(
        {parameter: (parameter, "first") for parameter in parameter_columns}
    )
    summary = candidates.groupby(
        "spatial_candidate_id", as_index=False, sort=False
    ).agg(**aggregation)

    active = summary.copy()
    for column in maximize:
        best = float(active[column].max())
        active = active.loc[active[column].ge(best - float(tolerance))]
        if active.empty:
            raise RuntimeError(
                f"Spatial maximization removed every candidate for {track_id} at {column}."
            )
    for column in minimize:
        best = float(active[column].min())
        active = active.loc[active[column].le(best + float(tolerance))]
        if active.empty:
            raise RuntimeError(
                f"Spatial minimization removed every candidate for {track_id} at {column}."
            )

    operation_complexity = active["morphology_operation"].astype(str).map(
        expcfg.SPATIAL_CALIBRATION_OPERATION_COMPLEXITY
    )
    if operation_complexity.isna().any():
        unknown = sorted(set(active.loc[operation_complexity.isna(), "morphology_operation"].astype(str)))
        raise RuntimeError(
            f"Unknown morphology operations in track {track_id}: {unknown}"
        )
    active = active.assign(operation_complexity=operation_complexity.astype(int))
    active = active.sort_values(
        ["min_area_pixels", "operation_complexity", "morphology_radius",
         "connectivity", "spatial_candidate_id"],
        kind="mergesort",
    )
    return str(active.iloc[0]["spatial_candidate_id"])

def _select_candidates_by_track(
    metrics: pd.DataFrame,
    *,
    tolerance: float,
    track_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Select exactly one spatial candidate independently inside each track."""
    observed_tracks = tuple(
        dict.fromkeys(
            metrics.loc[
                metrics["map_variant"].astype(str).eq("postprocessed"),
                "track_id",
            ].astype(str)
        )
    )
    if track_ids is None:
        selected_tracks = observed_tracks
    else:
        selected_tracks = tuple(dict.fromkeys(map(str, track_ids)))

    if not selected_tracks:
        raise RuntimeError("No spatial track is available for within-track selection.")

    missing_tracks = sorted(set(selected_tracks) - set(observed_tracks))
    extra_tracks = sorted(set(observed_tracks) - set(selected_tracks))
    if missing_tracks or extra_tracks:
        raise RuntimeError(
            "Spatial selection track coverage mismatch: "
            f"missing={missing_tracks}, extra={extra_tracks}."
        )

    return {
        track_id: _select_candidate_within_track(
            metrics,
            track_id=track_id,
            tolerance=float(tolerance),
        )
        for track_id in selected_tracks
    }



def calibrate_spatial_postprocessing(
    spatial_input: pd.DataFrame,
    image_db: dict,
    *,
    protocol_hash: str,
    candidate_grid: pd.DataFrame | None = None,
    tolerance: float = expcfg.SPATIAL_CALIBRATION_SELECTION_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate OOF maps and freeze one independent spatial lock per track."""
    grid = _validate_spatial_candidate_grid(
        build_spatial_candidate_grid()
        if candidate_grid is None
        else candidate_grid.copy()
    )
    required_input = set(expcfg.SPATIAL_CALIBRATION_INPUT_COLUMNS)
    missing_input = sorted(required_input - set(spatial_input.columns))
    if missing_input:
        raise KeyError(f"Missing compact spatial input columns: {missing_input}")
    if spatial_input.empty:
        raise RuntimeError("Spatial calibration input is empty.")

    spatial_input = spatial_input.reindex(
        columns=expcfg.SPATIAL_CALIBRATION_INPUT_COLUMNS
    ).copy()
    spatial_input["model_id"] = spatial_input["model_id"].astype(str)
    spatial_input["track_id"] = spatial_input["track_id"].astype(str)
    spatial_input["random_state"] = pd.to_numeric(
        spatial_input["random_state"], errors="raise"
    ).astype(int)

    observed_tracks = tuple(
        dict.fromkeys(spatial_input["track_id"].astype(str))
    )
    allowed_pixel_tracks = set(map(str, expcfg.SPATIAL_CALIBRATION_PIXEL_TRACK_IDS))
    unexpected_tracks = sorted(set(observed_tracks) - allowed_pixel_tracks)
    if unexpected_tracks:
        raise RuntimeError(
            "Spatial calibration received non-pixel-projection tracks: "
            f"{unexpected_tracks}."
        )

    group_ids = ["model_id", "random_state", "track_id"]
    metric_rows: list[dict] = []
    fragment_rows: list[dict] = []

    for key, configuration in spatial_input.groupby(group_ids, sort=False):
        metadata = dict(
            zip(group_ids, key if isinstance(key, tuple) else (key,))
        )
        maps = [
            _maps_for_group(group, image_db)
            for _, group in configuration.groupby("source_image", sort=False)
        ]

        for connectivity in sorted(grid["connectivity"].astype(int).unique()):
            raw_id = f"raw_c{connectivity}"
            raw_metrics, raw_fragments = _evaluate_maps(
                maps,
                connectivity=int(connectivity),
                candidate=None,
            )
            metric_rows.append(
                {
                    "spatial_candidate_id": raw_id,
                    "map_variant": "raw",
                    **metadata,
                    "connectivity": int(connectivity),
                    "morphology_operation": "none",
                    "morphology_radius": 0,
                    "min_area_pixels": 0,
                    **raw_metrics,
                    "is_locked_candidate": False,
                    "truth_level": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
                    "protocol_version": expcfg.PROTOCOL_VERSION,
                    "protocol_hash": str(protocol_hash),
                }
            )
            fragment_rows.extend(
                _summarize_fragment_classes(
                    raw_fragments,
                    metadata={
                        "spatial_candidate_id": raw_id,
                        **metadata,
                        "is_locked_candidate": False,
                        "truth_level": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
                        "protocol_version": expcfg.PROTOCOL_VERSION,
                        "protocol_hash": str(protocol_hash),
                    },
                )
            )

        for candidate in grid.to_dict("records"):
            candidate_metrics, candidate_fragments = _evaluate_maps(
                maps,
                connectivity=int(candidate["connectivity"]),
                candidate=candidate,
            )
            candidate_metadata = {
                "spatial_candidate_id": str(candidate["spatial_candidate_id"]),
                **metadata,
            }
            metric_rows.append(
                {
                    **candidate_metadata,
                    "map_variant": "postprocessed",
                    "connectivity": int(candidate["connectivity"]),
                    "morphology_operation": str(
                        candidate["morphology_operation"]
                    ),
                    "morphology_radius": int(candidate["morphology_radius"]),
                    "min_area_pixels": int(candidate["min_area_pixels"]),
                    **candidate_metrics,
                    "is_locked_candidate": False,
                    "truth_level": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
                    "protocol_version": expcfg.PROTOCOL_VERSION,
                    "protocol_hash": str(protocol_hash),
                }
            )
            fragment_rows.extend(
                _summarize_fragment_classes(
                    candidate_fragments,
                    metadata={
                        **candidate_metadata,
                        "is_locked_candidate": False,
                        "truth_level": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
                        "protocol_version": expcfg.PROTOCOL_VERSION,
                        "protocol_hash": str(protocol_hash),
                    },
                )
            )

    metrics = pd.DataFrame(metric_rows)
    fragments = pd.DataFrame(fragment_rows)

    selected_by_track = _select_candidates_by_track(
        metrics,
        tolerance=float(tolerance),
        track_ids=observed_tracks,
    )

    selected_metric_id = metrics["track_id"].astype(str).map(selected_by_track)
    metrics["is_locked_candidate"] = (
        metrics["map_variant"].astype(str).eq("postprocessed")
        & metrics["spatial_candidate_id"].astype(str).eq(selected_metric_id)
    )

    selected_fragment_id = fragments["track_id"].astype(str).map(
        selected_by_track
    )
    fragments["is_locked_candidate"] = (
        fragments["spatial_candidate_id"].astype(str).eq(
            selected_fragment_id
        )
    )

    metrics = metrics.reindex(columns=expcfg.SPATIAL_CALIBRATION_METRIC_COLUMNS)
    fragments = fragments.reindex(columns=expcfg.FRAGMENT_SIZE_CLASS_COLUMNS)

    metric_keys = [
        "track_id",
        "spatial_candidate_id",
        "map_variant",
        "model_id",
        "random_state",
    ]
    fragment_keys = [
        "track_id",
        "spatial_candidate_id",
        "model_id",
        "random_state",
        "area_class",
    ]
    if metrics.duplicated(metric_keys).any():
        raise RuntimeError("Spatial metric natural keys are not unique.")
    if fragments.duplicated(fragment_keys).any():
        raise RuntimeError("Fragment-class natural keys are not unique.")

    selected_parameters_by_track: dict[str, dict] = {}
    selected_counts_by_track: dict[str, dict[str, int]] = {}

    for track_id in observed_tracks:
        selected_id = str(selected_by_track[track_id])
        selected = grid.loc[
            grid["spatial_candidate_id"].astype(str).eq(selected_id)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"The spatial lock is not unique inside track {track_id}."
            )
        payload = selected.iloc[0].to_dict()
        selected_parameters_by_track[str(track_id)] = {
            "spatial_candidate_id": str(payload["spatial_candidate_id"]),
            "connectivity": int(payload["connectivity"]),
            "morphology_operation": str(payload["morphology_operation"]),
            "morphology_radius": int(payload["morphology_radius"]),
            "min_area_pixels": int(payload["min_area_pixels"]),
        }

        track_input = spatial_input.loc[
            spatial_input["track_id"].astype(str).eq(str(track_id))
        ]
        selected_counts_by_track[str(track_id)] = {
            "models": int(track_input["model_id"].nunique()),
            "executions": int(
                len(
                    track_input[
                        ["model_id", "random_state"]
                    ].drop_duplicates()
                )
            ),
        }

    lock = {
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "protocol_hash": str(protocol_hash),
        "rule_version": expcfg.SPATIAL_CALIBRATION_RULE_VERSION,
        "truth_source": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
        "allowed_batches": list(
            map(int, expcfg.SPATIAL_CALIBRATION_ALLOWED_BATCHES)
        ),
        "forbidden_batches": list(
            map(int, expcfg.SPATIAL_CALIBRATION_FORBIDDEN_BATCHES)
        ),
        "uncertain_pixel_policy": "preserve_as_distinct_immutable_layer",
        "selection_scope": expcfg.SPATIAL_CALIBRATION_SELECTION_SCOPE,
        "selection_policy": expcfg.SPATIAL_CALIBRATION_SELECTION_POLICY,
        "within_track_aggregation": (
            expcfg.SPATIAL_CALIBRATION_WITHIN_TRACK_AGGREGATION
        ),
        "selection_tolerance": float(tolerance),
        "selection_maximize": list(
            map(str, expcfg.SPATIAL_CALIBRATION_SELECTION_MAXIMIZE)
        ),
        "selection_minimize": list(
            map(str, expcfg.SPATIAL_CALIBRATION_SELECTION_MINIMIZE)
        ),
        "spatial_track_ids": list(map(str, observed_tracks)),
        "n_selected_models": int(spatial_input["model_id"].nunique()),
        "n_selected_executions": int(
            len(
                spatial_input[
                    ["model_id", "random_state"]
                ].drop_duplicates()
            )
        ),
        "selected_counts_by_track": selected_counts_by_track,
        "selected_parameters_by_track": selected_parameters_by_track,
        "area_minimum_version": expcfg.SPATIAL_CALIBRATION_RULE_VERSION,
        "spatial_input_sha256": sha256_dataframe(
            spatial_input.reindex(
                columns=expcfg.SPATIAL_CALIBRATION_INPUT_COLUMNS
            )
        ),
        "candidate_grid_sha256": sha256_dataframe(grid),
        "spatial_calibration_metrics_sha256": sha256_dataframe(metrics),
        "fragment_size_classes_sha256": sha256_dataframe(fragments),
    }
    lock["lock_sha256"] = _payload_hash(lock)
    return metrics, fragments, lock


def verify_spatial_postprocessing_lock(
    lock: Mapping,
    metrics: pd.DataFrame,
    fragments: pd.DataFrame,
) -> None:
    """Verify the per-track 03C lock and its persisted calibration outputs."""
    payload = dict(lock)
    lock_hash = payload.pop("lock_sha256", None)
    expected_hash = _payload_hash(payload)
    if str(lock_hash) != expected_hash:
        raise RuntimeError("The spatial post-processing lock was modified.")

    if str(lock.get("protocol_version", "")) != str(expcfg.PROTOCOL_VERSION):
        raise RuntimeError(
            "The spatial post-processing lock belongs to another protocol version."
        )
    if str(lock.get("rule_version", "")) != str(
        expcfg.SPATIAL_CALIBRATION_RULE_VERSION
    ):
        raise RuntimeError(
            "The spatial post-processing lock uses another rule version."
        )
    if str(lock.get("truth_source", "")) != str(
        expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE
    ):
        raise RuntimeError(
            "The spatial post-processing lock uses another truth source."
        )
    if str(lock.get("selection_scope", "")) != str(
        expcfg.SPATIAL_CALIBRATION_SELECTION_SCOPE
    ):
        raise RuntimeError("The spatial lock is not explicitly within-track.")
    if str(lock.get("selection_policy", "")) != str(
        expcfg.SPATIAL_CALIBRATION_SELECTION_POLICY
    ):
        raise RuntimeError("Unexpected spatial selection policy.")
    if str(lock.get("within_track_aggregation", "")) != str(
        expcfg.SPATIAL_CALIBRATION_WITHIN_TRACK_AGGREGATION
    ):
        raise RuntimeError("Unexpected within-track aggregation policy.")

    legacy_fields = sorted(
        {"selected_parameters", "selection_weighting"}.intersection(lock)
    )
    if legacy_fields:
        raise RuntimeError(
            f"Legacy global spatial-lock fields are forbidden: {legacy_fields}."
        )

    if str(lock["spatial_calibration_metrics_sha256"]) != sha256_dataframe(metrics):
        raise RuntimeError(
            "spatial_calibration_metrics.parquet changed after lock."
        )
    if str(lock["fragment_size_classes_sha256"]) != sha256_dataframe(fragments):
        raise RuntimeError("fragment_size_classes.parquet changed after lock.")

    selected_parameters_by_track = lock.get("selected_parameters_by_track")
    if not isinstance(selected_parameters_by_track, Mapping):
        raise RuntimeError(
            "The spatial lock has no selected_parameters_by_track mapping."
        )
    selected_parameters_by_track = {
        str(track_id): dict(parameters)
        for track_id, parameters in selected_parameters_by_track.items()
    }

    spatial_track_ids = tuple(map(str, lock.get("spatial_track_ids", ())))
    if not spatial_track_ids:
        raise RuntimeError("The spatial lock has no spatial_track_ids.")
    if len(set(spatial_track_ids)) != len(spatial_track_ids):
        raise RuntimeError("spatial_track_ids contains duplicates.")
    if set(selected_parameters_by_track) != set(spatial_track_ids):
        raise RuntimeError(
            "Each spatial track must have exactly one locked parameter payload."
        )

    allowed_pixel_tracks = set(map(str, expcfg.SPATIAL_CALIBRATION_PIXEL_TRACK_IDS))
    if not set(spatial_track_ids).issubset(allowed_pixel_tracks):
        raise RuntimeError(
            "The spatial lock contains a non-pixel-projection track."
        )

    metric_tracks = set(metrics["track_id"].astype(str))
    fragment_tracks = set(fragments["track_id"].astype(str))
    if metric_tracks != set(spatial_track_ids):
        raise RuntimeError(
            "Spatial metric tracks do not match the locked track universe."
        )
    if fragment_tracks != set(spatial_track_ids):
        raise RuntimeError(
            "Fragment-class tracks do not match the locked track universe."
        )

    if metrics.loc[
        metrics["map_variant"].astype(str).eq("raw"),
        "is_locked_candidate",
    ].fillna(False).astype(bool).any():
        raise RuntimeError("A raw map was incorrectly marked as locked.")

    required_parameter_keys = set(
        expcfg.SPATIAL_CALIBRATION_REQUIRED_LOCK_PARAMETER_KEYS
    )
    counts_by_track = lock.get("selected_counts_by_track", {})
    if counts_by_track and not isinstance(counts_by_track, Mapping):
        raise RuntimeError("selected_counts_by_track must be a mapping.")

    for track_id in spatial_track_ids:
        parameters = selected_parameters_by_track[track_id]
        missing_parameters = sorted(
            required_parameter_keys - set(parameters)
        )
        if missing_parameters:
            raise KeyError(
                f"Track {track_id} spatial lock is missing parameters: "
                f"{missing_parameters}."
            )

        candidate_payload = {
            parameter: parameters[parameter]
            for parameter in expcfg.SPATIAL_CALIBRATION_PARAMETER_COLUMNS
        }
        expected_candidate_id = _candidate_id(candidate_payload)
        selected_id = str(parameters["spatial_candidate_id"])
        if selected_id != expected_candidate_id:
            raise RuntimeError(
                f"Track {track_id} spatial candidate id does not match its parameters."
            )

        track_metrics = metrics.loc[
            metrics["track_id"].astype(str).eq(track_id)
        ].copy()
        locked_metrics = track_metrics.loc[
            track_metrics["is_locked_candidate"].fillna(False).astype(bool)
        ]
        if locked_metrics.empty:
            raise RuntimeError(
                f"Track {track_id} has no locked spatial metric rows."
            )
        if not locked_metrics["map_variant"].astype(str).eq(
            "postprocessed"
        ).all():
            raise RuntimeError(
                f"Track {track_id} has a non-postprocessed locked row."
            )
        locked_ids = set(
            locked_metrics["spatial_candidate_id"].astype(str)
        )
        if locked_ids != {selected_id}:
            raise RuntimeError(
                f"Track {track_id} locked candidate does not match metrics."
            )

        execution_keys = ["model_id", "random_state"]
        expected_executions = set(
            track_metrics[execution_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        locked_executions = set(
            locked_metrics[execution_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if locked_executions != expected_executions:
            raise RuntimeError(
                f"Track {track_id} lock does not cover every selected execution."
            )
        if locked_metrics.duplicated(execution_keys).any():
            raise RuntimeError(
                f"Track {track_id} has more than one locked candidate row per execution."
            )

        parameter_columns = list(
            expcfg.SPATIAL_CALIBRATION_PARAMETER_COLUMNS
        )
        parameter_values = locked_metrics[parameter_columns].drop_duplicates()
        if len(parameter_values) != 1:
            raise RuntimeError(
                f"Track {track_id} locked metric rows disagree on parameters."
            )
        row = parameter_values.iloc[0]
        for parameter in parameter_columns:
            expected_value = parameters[parameter]
            observed_value = row[parameter]
            if parameter in {
                "connectivity",
                "morphology_radius",
                "min_area_pixels",
            }:
                if int(observed_value) != int(expected_value):
                    raise RuntimeError(
                        f"Track {track_id} locked parameter mismatch: {parameter}."
                    )
            elif str(observed_value) != str(expected_value):
                raise RuntimeError(
                    f"Track {track_id} locked parameter mismatch: {parameter}."
                )

        track_fragments = fragments.loc[
            fragments["track_id"].astype(str).eq(track_id)
        ]
        locked_fragments = track_fragments.loc[
            track_fragments["is_locked_candidate"]
            .fillna(False)
            .astype(bool)
        ]
        if locked_fragments.empty:
            raise RuntimeError(
                f"Track {track_id} has no locked fragment-class rows."
            )
        fragment_ids = set(
            locked_fragments["spatial_candidate_id"].astype(str)
        )
        if fragment_ids != {selected_id}:
            raise RuntimeError(
                f"Track {track_id} fragment lock does not match metrics."
            )

        if counts_by_track:
            raw_counts = counts_by_track.get(track_id)
            if not isinstance(raw_counts, Mapping):
                raise RuntimeError(
                    f"Track {track_id} has no selected_counts_by_track payload."
                )
            observed_models = int(
                locked_metrics["model_id"].astype(str).nunique()
            )
            observed_executions = int(
                len(locked_metrics[execution_keys].drop_duplicates())
            )
            if int(raw_counts.get("models", -1)) != observed_models:
                raise RuntimeError(
                    f"Track {track_id} locked model count mismatch."
                )
            if int(raw_counts.get("executions", -1)) != observed_executions:
                raise RuntimeError(
                    f"Track {track_id} locked execution count mismatch."
                )


def encode_boolean_map(
    mask,
    *,
    compression_level: int = expcfg.SIMCA_CONCAT_REFIT_MAP_COMPRESSION_LEVEL,
) -> bytes:
    """Encode a 2D boolean map using the versioned compact 04C contract."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("Only aligned 2D boolean maps can be encoded.")
    packed = np.packbits(values.ravel(order="C"), bitorder="little")
    return zlib.compress(packed.tobytes(), level=int(compression_level))


def decode_boolean_map(payload: bytes, shape: Sequence[int]) -> np.ndarray:
    """Decode a ``packbits_zlib_v1`` map (mainly for audit/tests)."""
    height, width = map(int, shape)
    packed = np.frombuffer(zlib.decompress(bytes(payload)), dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")[: height * width]
    return unpacked.reshape((height, width)).astype(bool)


def build_locked_spatial_validation_outputs(
    validation_executions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    image_db: Mapping[str, Mapping],
    spatial_lock: Mapping,
) -> dict[str, pd.DataFrame]:
    """Build locked batch-3 pixel maps and spatial diagnostics for 04C.

    The function reuses the canonical 03B identities. One scientific execution
    is addressed by ``(model_id, random_state)`` and continuous pixel
    predictions by ``projection_id``. Only the locked ``direct`` decision scope
    is spatially reconstructed; ``pixel_to_object`` is an object-level decision
    and therefore has no pixel map.
    """
    required_executions = set(expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS)
    required_thresholds = set(
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
    )
    required_pixels = set(expcfg.SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS)
    for frame, required, name in (
        (validation_executions, required_executions, "validation_executions"),
        (selected_thresholds, required_thresholds, "selected_thresholds"),
        (pixel_predictions, required_pixels, "validation_pixel_predictions"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing spatial columns: {missing}")

    # ------------------------------------------------------------------
    # Verify the immutable 03C spatial lock before using any parameter.
    # ------------------------------------------------------------------
    lock_payload = dict(spatial_lock)
    lock_hash = str(lock_payload.pop("lock_sha256", ""))
    if not lock_hash or lock_hash != _payload_hash(lock_payload):
        raise RuntimeError("The spatial post-processing lock was modified.")
    if str(spatial_lock.get("protocol_version", "")) != str(
        expcfg.PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "The spatial post-processing lock belongs to another protocol version."
        )
    if str(spatial_lock.get("truth_source", "")) != str(
        expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE
    ):
        raise RuntimeError(
            "The spatial post-processing lock uses another truth source."
        )
    if expcfg.SIMCA_CONCAT_REFIT_MAP_ENCODING != "packbits_zlib_v1":
        raise RuntimeError(
            "encode_boolean_map currently implements packbits_zlib_v1 only."
        )

    if "selected_parameters" in spatial_lock or "selection_weighting" in spatial_lock:
        raise RuntimeError(
            "Legacy global spatial-lock fields are forbidden in 04C."
        )
    if str(spatial_lock.get("selection_scope", "")) != str(
        expcfg.SPATIAL_CALIBRATION_SELECTION_SCOPE
    ):
        raise RuntimeError("04C requires the within-track 03C spatial lock.")

    raw_parameters_by_track = spatial_lock.get("selected_parameters_by_track")
    if not isinstance(raw_parameters_by_track, Mapping):
        raise RuntimeError(
            "The spatial lock has no selected_parameters_by_track mapping."
        )
    parameters_by_track = {
        str(track_id): dict(parameters)
        for track_id, parameters in raw_parameters_by_track.items()
    }
    required_parameters = set(
        expcfg.SPATIAL_CALIBRATION_REQUIRED_LOCK_PARAMETER_KEYS
    )
    for track_id, parameters in parameters_by_track.items():
        missing_parameters = sorted(required_parameters - set(parameters))
        if missing_parameters:
            raise KeyError(
                f"Track {track_id} spatial lock is missing parameters: "
                f"{missing_parameters}."
            )
        connectivity = int(parameters["connectivity"])
        if connectivity not in {1, 2}:
            raise RuntimeError(
                f"Track {track_id} locked 2D connectivity must be 1 or 2."
            )

    # ------------------------------------------------------------------
    # Normalize the canonical execution and threshold registries.
    # ------------------------------------------------------------------
    executions = validation_executions.copy()
    executions["model_id"] = executions["model_id"].astype(str)
    executions["projection_id"] = executions["projection_id"].astype(str)
    executions["track_id"] = executions["track_id"].astype(str)
    executions["random_state"] = pd.to_numeric(
        executions["random_state"], errors="raise"
    ).astype(int)
    run_keys = ["model_id", "random_state"]
    if executions.duplicated(run_keys).any():
        raise RuntimeError(
            "Validation executions duplicate the natural (model_id, random_state) key."
        )

    pixel_execution_mask = executions["projection_level"].astype(str).eq(
        "pixel_projection"
    )
    supported_statuses = set(
        map(str, expcfg.SIMCA_CONCAT_REFIT_SUPPORTED_ELIGIBILITY_STATUSES)
    )
    supported_pixel_mask = (
        pixel_execution_mask
        & executions["eligibility_status"].astype(str).isin(supported_statuses)
        & executions["downstream_status"].astype(str).eq("supported")
    )
    pixel_executions = executions.loc[
        supported_pixel_mask,
        [
            "model_id",
            "random_state",
            "projection_id",
            "track_id",
            "decision_mode",
        ],
    ].copy()

    lock_tracks = set(map(str, parameters_by_track))
    supported_pixel_tracks = set(pixel_executions["track_id"].astype(str))
    if supported_pixel_tracks != lock_tracks:
        raise RuntimeError(
            "04C supported pixel-track universe does not match the 03C spatial lock: "
            f"supported={sorted(supported_pixel_tracks)}, locked={sorted(lock_tracks)}."
        )

    if pixel_executions.empty:
        return {
            "pixel_maps_manifest": pd.DataFrame(
                columns=expcfg.SIMCA_PIXEL_MAP_MANIFEST_COLUMNS
            ),
            "spatial_components": pd.DataFrame(
                columns=expcfg.SIMCA_SPATIAL_COMPONENT_COLUMNS
            ),
            "spatial_component_metrics": pd.DataFrame(
                columns=expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS
            ),
        }

    thresholds = selected_thresholds.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS)
    ].copy()
    thresholds["model_id"] = thresholds["model_id"].astype(str)
    thresholds["random_state"] = pd.to_numeric(
        thresholds["random_state"], errors="raise"
    ).astype(int)
    thresholds["decision_scope"] = thresholds["decision_scope"].astype(str)
    direct = thresholds.loc[
        thresholds["decision_scope"].eq("direct"),
        [*run_keys, "lower_threshold", "upper_threshold"],
    ].copy()
    if direct.duplicated(run_keys).any():
        raise RuntimeError(
            "A pixel validation execution has duplicate direct thresholds."
        )

    threshold_coverage = pixel_executions[run_keys].merge(
        direct[run_keys].assign(_threshold_present=True),
        on=run_keys,
        how="left",
        validate="one_to_one",
    )
    if threshold_coverage["_threshold_present"].isna().any():
        missing_keys = threshold_coverage.loc[
            threshold_coverage["_threshold_present"].isna(), run_keys
        ].to_dict("records")
        raise RuntimeError(
            "A pixel validation execution has no locked direct threshold: "
            f"{missing_keys[:10]}"
        )

    pixel_executions = pixel_executions.merge(
        direct,
        on=run_keys,
        how="left",
        validate="one_to_one",
    )
    lower = pd.to_numeric(
        pixel_executions["lower_threshold"], errors="coerce"
    ).to_numpy(dtype=float)
    upper = pd.to_numeric(
        pixel_executions["upper_threshold"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([lower, upper])).all():
        raise RuntimeError("A selected direct threshold is non-finite.")
    two_way = pixel_executions["decision_mode"].astype(str).eq("2way").to_numpy()
    three_way = pixel_executions["decision_mode"].astype(str).eq("3way").to_numpy()
    if not (two_way | three_way).all():
        unknown = sorted(
            set(
                pixel_executions.loc[
                    ~(two_way | three_way), "decision_mode"
                ].astype(str)
            )
        )
        raise RuntimeError(f"Unknown spatial decision modes: {unknown}")
    if not np.isclose(lower[two_way], upper[two_way]).all():
        raise RuntimeError("A selected 2-way direct threshold is inconsistent.")
    if not (lower[three_way] < upper[three_way]).all():
        raise RuntimeError("A selected 3-way direct threshold is inconsistent.")

    # ------------------------------------------------------------------
    # Continuous predictions are stored once per projection_id.
    # ------------------------------------------------------------------
    predictions = pixel_predictions.copy()
    predictions["projection_id"] = predictions["projection_id"].astype(str)
    duplicate_prediction_key = [
        "projection_id",
        "source_image",
        "object_id",
        "row",
        "col",
    ]
    if predictions.duplicated(duplicate_prediction_key).any():
        raise RuntimeError(
            "validation_pixel_predictions duplicates its natural observation key."
        )
    prediction_lookup = {
        str(projection_id): np.asarray(indices, dtype=int)
        for projection_id, indices in predictions.groupby(
            "projection_id", sort=False, dropna=False
        ).indices.items()
    }

    manifests: list[dict[str, object]] = []
    component_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    bounds = tuple(expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_UPPER_BOUNDS)
    labels = tuple(map(str, expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS))
    allowed_batches = set(map(int, expcfg.SIMCA_CONCAT_REFIT_PROJECTION_BATCHES))

    # ------------------------------------------------------------------
    # Reconstruct direct pixel decisions and apply the immutable 03C lock.
    # ------------------------------------------------------------------
    for execution in pixel_executions.to_dict("records"):
        model_id = str(execution["model_id"])
        random_state = int(execution["random_state"])
        track_id = str(execution["track_id"])
        projection_id = str(execution["projection_id"])
        decision_mode = str(execution["decision_mode"])
        if track_id not in parameters_by_track:
            raise RuntimeError(
                f"No track-specific spatial lock is available for {track_id}."
            )
        parameters = parameters_by_track[track_id]
        connectivity = int(parameters["connectivity"])

        positions = prediction_lookup.get(projection_id)
        if positions is None or not len(positions):
            # A missing projection is represented by validation technical events
            # and will become a technical failure in the guardrail table.
            continue

        observations = predictions.iloc[positions].copy()
        margin = pd.to_numeric(
            observations["simca_margin"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(margin).all():
            raise RuntimeError(
                f"Non-finite validation margin for projection_id={projection_id!r}."
            )

        direct_target, direct_uncertain = apply_locked_margin_decision(
            margin,
            decision_mode,
            direct_2way_threshold=float(execution["lower_threshold"]),
            three_way_lower_threshold=float(execution["lower_threshold"]),
            three_way_upper_threshold=float(execution["upper_threshold"]),
        )
        observations["__raw_target"] = np.asarray(direct_target, dtype=bool)
        observations["__uncertain"] = np.asarray(direct_uncertain, dtype=bool)

        for image_key, image_positions in observations.groupby(
            "source_image", sort=False
        ).indices.items():
            image_key = str(image_key)
            if image_key not in image_db:
                raise KeyError(
                    f"Validation image is absent from HDF5: {image_key!r}"
                )
            group = observations.iloc[image_positions].copy()
            if group.duplicated(["row", "col"]).any():
                raise RuntimeError(
                    f"Duplicated validation pixel coordinates for {image_key}."
                )

            batch_values = pd.to_numeric(
                group["batch"], errors="raise"
            ).astype(int)
            observed_batches = set(batch_values.tolist())
            if len(observed_batches) != 1 or not observed_batches.issubset(
                allowed_batches
            ):
                raise RuntimeError(
                    "Spatial validation observations must belong to exactly one "
                    f"allowed batch; image={image_key!r}, batches={sorted(observed_batches)}."
                )

            truth_result = pure_image_class_truth(
                image_key,
                dict(image_db),
                target_class=expcfg.TARGET_CLASS,
                allowed_batches=expcfg.SIMCA_CONCAT_REFIT_PROJECTION_BATCHES,
            )
            if str(truth_result.truth_level) != str(
                expcfg.SIMCA_CONCAT_REFIT_TRUTH_SOURCE
            ):
                raise RuntimeError(
                    "Validation spatial truth does not match the frozen 04C truth source."
                )

            shape = tuple(map(int, truth_result.truth_mask.shape))
            image_labels = np.asarray(image_db[image_key]["labels"])
            if image_labels.shape != shape:
                raise RuntimeError(
                    f"HDF5 label shape disagrees with truth for {image_key!r}."
                )

            row = pd.to_numeric(group["row"], errors="raise").astype(int).to_numpy()
            col = pd.to_numeric(group["col"], errors="raise").astype(int).to_numpy()
            inside = (
                (row >= 0)
                & (col >= 0)
                & (row < shape[0])
                & (col < shape[1])
            )
            if not inside.all():
                raise RuntimeError(f"Validation coordinates outside {image_key!r}.")

            valid = np.zeros(shape, dtype=bool)
            raw_target = np.zeros(shape, dtype=bool)
            uncertain_map = np.zeros(shape, dtype=bool)
            valid[row, col] = True
            raw_target[row, col] = group["__raw_target"].astype(bool).to_numpy()
            uncertain_map[row, col] = group["__uncertain"].astype(bool).to_numpy()

            valid &= np.asarray(truth_result.available_mask, dtype=bool)
            raw_target &= valid
            uncertain_map &= valid
            raw_target &= ~uncertain_map
            truth = np.asarray(truth_result.truth_mask, dtype=bool) & valid

            post_target, preserved_uncertain = apply_spatial_postprocessing(
                raw_target,
                uncertain_map,
                valid,
                connectivity=connectivity,
                morphology_operation=str(parameters["morphology_operation"]),
                morphology_radius=int(parameters["morphology_radius"]),
                min_area_pixels=int(parameters["min_area_pixels"]),
            )
            if not np.array_equal(preserved_uncertain, uncertain_map):
                raise RuntimeError("Locked morphology modified uncertainty.")

            manifests.append(
                {
                    "model_id": model_id,
                    "random_state": random_state,
                    "track_id": track_id,
                    "source_image": image_key,
                    "batch": int(batch_values.iloc[0]),
                    "height": int(shape[0]),
                    "width": int(shape[1]),
                    "map_encoding": expcfg.SIMCA_CONCAT_REFIT_MAP_ENCODING,
                    "valid_mask": encode_boolean_map(valid),
                    "raw_target_mask": encode_boolean_map(raw_target),
                    "uncertain_mask": encode_boolean_map(uncertain_map),
                    "postprocessed_target_mask": encode_boolean_map(post_target),
                    "truth_mask": encode_boolean_map(truth),
                    "truth_level": str(truth_result.truth_level),
                    "spatial_lock_sha256": lock_hash,
                }
            )

            # Uncertain pixels stay a distinct non-evaluable layer, exactly as
            # in the 03C spatial calibration code.
            evaluable = valid & ~uncertain_map
            evaluable_truth = truth & evaluable
            truth_component_labels = np.where(evaluable_truth, image_labels, 0)

            for map_variant, raw_prediction in (
                ("raw", raw_target),
                ("locked_postprocessed", post_target),
            ):
                prediction = np.asarray(raw_prediction, dtype=bool) & evaluable
                component_metrics = component_detection_metrics(
                    evaluable_truth,
                    prediction,
                    valid_mask=evaluable,
                    connectivity=connectivity,
                    truth_component_labels=truth_component_labels,
                    min_iou=expcfg.SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU,
                )
                components = component_detection_table(
                    evaluable_truth,
                    prediction,
                    valid_mask=evaluable,
                    connectivity=connectivity,
                    truth_component_labels=truth_component_labels,
                    area_upper_bounds=bounds,
                    area_labels=labels,
                    min_iou=expcfg.SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU,
                )
                if len(components):
                    components = components.copy()
                    components["model_id"] = model_id
                    components["random_state"] = random_state
                    components["track_id"] = track_id
                    components["source_image"] = image_key
                    components["map_variant"] = map_variant
                    components["truth_level"] = str(truth_result.truth_level)
                    component_parts.append(
                        components.reindex(
                            columns=expcfg.SIMCA_SPATIAL_COMPONENT_COLUMNS
                        )
                    )

                intersection = int(
                    np.count_nonzero(evaluable_truth & prediction)
                )
                union = int(np.count_nonzero(evaluable_truth | prediction))
                n_truth_pixels = int(np.count_nonzero(evaluable_truth))
                n_prediction_pixels = int(np.count_nonzero(prediction))

                truth_components = (
                    components.loc[
                        components["component_role"].astype(str).eq("truth")
                    ].copy()
                    if len(components) and "component_role" in components.columns
                    else pd.DataFrame()
                )
                observed_classes = set(
                    truth_components.get(
                        "area_class", pd.Series(dtype="string")
                    ).astype(str)
                )
                smallest = next(
                    (
                        label_name
                        for label_name in labels
                        if label_name in observed_classes
                    ),
                    None,
                )
                smallest_rows = (
                    truth_components.loc[
                        truth_components["area_class"].astype(str).eq(smallest)
                    ]
                    if smallest is not None
                    else truth_components.iloc[0:0]
                )

                metric_rows.append(
                    {
                        "model_id": model_id,
                        "random_state": random_state,
                        "track_id": track_id,
                        "source_image": image_key,
                        "aggregation_level": "source_image",
                        "map_variant": map_variant,
                        "n_valid_pixels": int(evaluable.sum()),
                        "dice": (
                            2.0 * intersection
                            / (n_truth_pixels + n_prediction_pixels)
                            if n_truth_pixels + n_prediction_pixels
                            else 1.0
                        ),
                        "iou": intersection / union if union else 1.0,
                        "pixel_precision": (
                            intersection / n_prediction_pixels
                            if n_prediction_pixels
                            else np.nan
                        ),
                        "pixel_recall": (
                            intersection / n_truth_pixels
                            if n_truth_pixels
                            else np.nan
                        ),
                        **component_metrics,
                        "smallest_fragment_recall": (
                            float(
                                pd.to_numeric(
                                    smallest_rows["detected_or_matched"],
                                    errors="coerce",
                                ).mean()
                            )
                            if len(smallest_rows)
                            else np.nan
                        ),
                        "truth_level": str(truth_result.truth_level),
                        # Internal counters used only for exact pooled metrics.
                        "__truth_pixels": n_truth_pixels,
                        "__prediction_pixels": n_prediction_pixels,
                        "__intersection": intersection,
                        "__union": union,
                    }
                )

    # ------------------------------------------------------------------
    # Aggregate source-image spatial metrics to one execution-level row.
    # ------------------------------------------------------------------
    image_metrics = pd.DataFrame(metric_rows)
    components = (
        pd.concat(component_parts, ignore_index=True, sort=False)
        if component_parts
        else pd.DataFrame(columns=expcfg.SIMCA_SPATIAL_COMPONENT_COLUMNS)
    )
    overall_rows: list[dict[str, object]] = []

    if len(image_metrics):
        count_columns = (
            "n_truth_components",
            "n_predicted_components",
            "n_detected_truth_components",
            "n_matched_predicted_components",
            "n_split_truth_components",
            "n_merged_predicted_components",
        )
        missing_counts = sorted(set(count_columns) - set(image_metrics.columns))
        if missing_counts:
            raise RuntimeError(
                "Spatial component metrics are missing pooled counters: "
                f"{missing_counts}"
            )

        group_columns = [
            "model_id",
            "random_state",
            "track_id",
            "map_variant",
        ]
        for key, group in image_metrics.groupby(
            group_columns, sort=False, dropna=False
        ):
            model_id, random_state, track_id, map_variant = key
            totals = group[
                [
                    "n_valid_pixels",
                    "__truth_pixels",
                    "__prediction_pixels",
                    "__intersection",
                    "__union",
                    *count_columns,
                ]
            ].sum(numeric_only=True)

            truth_total = float(totals["__truth_pixels"])
            prediction_total = float(totals["__prediction_pixels"])
            intersection = float(totals["__intersection"])
            union = float(totals["__union"])
            truth_component_total = float(totals["n_truth_components"])
            predicted_component_total = float(totals["n_predicted_components"])

            truth_components = (
                components.loc[
                    components["model_id"].astype(str).eq(str(model_id))
                    & pd.to_numeric(
                        components["random_state"], errors="coerce"
                    ).eq(int(random_state))
                    & components["track_id"].astype(str).eq(str(track_id))
                    & components["map_variant"].astype(str).eq(str(map_variant))
                    & components["component_role"].astype(str).eq("truth")
                ].copy()
                if len(components)
                else pd.DataFrame()
            )
            observed_classes = set(
                truth_components.get(
                    "area_class", pd.Series(dtype="string")
                ).astype(str)
            )
            smallest = next(
                (label_name for label_name in labels if label_name in observed_classes),
                None,
            )
            smallest_rows = (
                truth_components.loc[
                    truth_components["area_class"].astype(str).eq(smallest)
                ]
                if smallest is not None
                else truth_components.iloc[0:0]
            )

            truth_levels = set(group["truth_level"].astype(str))
            if len(truth_levels) != 1:
                raise RuntimeError(
                    "One spatial execution/map variant mixes several truth levels."
                )

            overall_rows.append(
                {
                    "model_id": str(model_id),
                    "random_state": int(random_state),
                    "track_id": str(track_id),
                    "source_image": "all",
                    "aggregation_level": "overall",
                    "map_variant": str(map_variant),
                    "n_valid_pixels": int(totals["n_valid_pixels"]),
                    "dice": (
                        2.0 * intersection / (truth_total + prediction_total)
                        if truth_total + prediction_total
                        else 1.0
                    ),
                    "iou": intersection / union if union else 1.0,
                    "pixel_precision": (
                        intersection / prediction_total
                        if prediction_total
                        else np.nan
                    ),
                    "pixel_recall": (
                        intersection / truth_total if truth_total else np.nan
                    ),
                    "n_truth_components": int(truth_component_total),
                    "n_predicted_components": int(predicted_component_total),
                    "component_precision": (
                        float(totals["n_matched_predicted_components"])
                        / predicted_component_total
                        if predicted_component_total
                        else np.nan
                    ),
                    "component_recall": (
                        float(totals["n_detected_truth_components"])
                        / truth_component_total
                        if truth_component_total
                        else np.nan
                    ),
                    "split_rate": (
                        float(totals["n_split_truth_components"])
                        / truth_component_total
                        if truth_component_total
                        else 0.0
                    ),
                    "merge_rate": (
                        float(totals["n_merged_predicted_components"])
                        / predicted_component_total
                        if predicted_component_total
                        else 0.0
                    ),
                    "smallest_fragment_recall": (
                        float(
                            pd.to_numeric(
                                smallest_rows["detected_or_matched"],
                                errors="coerce",
                            ).mean()
                        )
                        if len(smallest_rows)
                        else np.nan
                    ),
                    "truth_level": next(iter(truth_levels)),
                }
            )

    metrics = (
        pd.concat(
            [image_metrics, pd.DataFrame(overall_rows)],
            ignore_index=True,
            sort=False,
        )
        if len(image_metrics) or overall_rows
        else pd.DataFrame(columns=expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS)
    )
    metrics = metrics.reindex(columns=expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS)
    manifests_df = pd.DataFrame(manifests).reindex(
        columns=expcfg.SIMCA_PIXEL_MAP_MANIFEST_COLUMNS
    )
    components = components.reindex(columns=expcfg.SIMCA_SPATIAL_COMPONENT_COLUMNS)

    # ------------------------------------------------------------------
    # Final compact-contract checks: no legacy IDs and unique natural keys.
    # ------------------------------------------------------------------
    forbidden_legacy_ids = {
        "validation_candidate_id",
        "calibration_id",
        "domain_config_id",
        "evaluation_config_id",
        "data_config_id",
        "fit_config_id",
        "projection_config_id",
    }
    for frame, name in (
        (manifests_df, "pixel_maps_manifest"),
        (components, "spatial_components"),
        (metrics, "spatial_component_metrics"),
    ):
        leaked = sorted(forbidden_legacy_ids.intersection(frame.columns))
        if leaked:
            raise RuntimeError(f"Legacy identifiers leaked into {name}: {leaked}")

    if len(manifests_df) and manifests_df.duplicated(
        ["track_id", "model_id", "random_state", "source_image"]
    ).any():
        raise RuntimeError("pixel_maps_manifest duplicates its natural map key.")
    if len(components) and components.duplicated(
        [
            "track_id",
            "model_id",
            "random_state",
            "source_image",
            "map_variant",
            "component_role",
            "component_id",
        ]
    ).any():
        raise RuntimeError("spatial_components duplicates its natural component key.")
    if len(metrics) and metrics.duplicated(
        [
            "track_id",
            "model_id",
            "random_state",
            "source_image",
            "aggregation_level",
            "map_variant",
        ]
    ).any():
        raise RuntimeError(
            "spatial_component_metrics duplicates its natural metric key."
        )

    return {
        "pixel_maps_manifest": manifests_df.sort_values(
            ["track_id", "model_id", "random_state", "source_image"],
            kind="mergesort",
        ).reset_index(drop=True),
        "spatial_components": components.sort_values(
            [
                "track_id",
                "model_id",
                "random_state",
                "source_image",
                "map_variant",
                "component_role",
                "component_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True),
        "spatial_component_metrics": metrics.sort_values(
            [
                "track_id",
                "model_id",
                "random_state",
                "aggregation_level",
                "source_image",
                "map_variant",
            ],
            kind="mergesort",
        ).reset_index(drop=True),
    }


__all__ = [
    "apply_spatial_postprocessing",
    "build_spatial_calibration_input",
    "build_spatial_candidate_grid",
    "build_locked_spatial_validation_outputs",
    "calibrate_spatial_postprocessing",
    "decode_boolean_map",
    "encode_boolean_map",
    "verify_spatial_postprocessing_lock",
]