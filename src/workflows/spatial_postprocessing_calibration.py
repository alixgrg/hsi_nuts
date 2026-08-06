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
    calibration_domain: pd.DataFrame,
    image_db: dict,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    allowed_batches: Sequence[int] = expcfg.SPATIAL_CALIBRATION_ALLOWED_BATCHES,
    required_classes: Sequence[str] = expcfg.SPATIAL_CALIBRATION_REQUIRED_CLASSES,
) -> pd.DataFrame:
    """Attach locked thresholds and exact pure-image pixel truth to OOF rows."""
    required_oof = {
        "projection_config_id",
        "fold_id",
        "source_image",
        "object_id",
        "batch",
        "row",
        "col",
        "simca_margin",
    }
    missing = sorted(required_oof - set(oof_pixels.columns))
    if missing:
        raise KeyError(f"Missing OOF pixel columns: {missing}")
    mapping_columns = [
        "domain_config_id",
        "evaluation_track",
        "track_id",
        "projection_config_id",
        "decision_mode",
        "direct_2way_threshold",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    ]
    has_oof_seed = "random_state" in oof_pixels.columns
    has_domain_seed = "random_state" in calibration_domain.columns
    if has_oof_seed != has_domain_seed:
        raise KeyError(
            "random_state must be present in both OOF pixels and the domain, "
            "or absent from both legacy inputs."
        )
    if has_oof_seed:
        mapping_columns.append("random_state")
    missing = sorted(set(mapping_columns) - set(calibration_domain.columns))
    if missing:
        raise KeyError(f"Missing spatial-domain columns: {missing}")
    mapping = calibration_domain.loc[
        calibration_domain["projection_level"].astype(str).eq("pixel_projection"),
        mapping_columns,
    ].drop_duplicates()
    if mapping.empty:
        raise RuntimeError("No calibrated pixel-projection track is available.")
    if mapping["domain_config_id"].astype(str).duplicated().any():
        raise RuntimeError("A spatial domain_config_id has conflicting metadata.")
    merge_keys = ["projection_config_id"]
    if has_oof_seed:
        merge_keys.append("random_state")
    out = oof_pixels.merge(
        mapping,
        on=merge_keys,
        how="inner",
        validate="many_to_many",
    )
    if out.empty:
        raise RuntimeError("No OOF pixel row matches the calibrated 03B domain.")
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
    out["truth_available"] = False
    for image_key, indices in out.groupby("source_image", sort=False).groups.items():
        truth = truth_cache[str(image_key)]
        row = pd.to_numeric(out.loc[indices, "row"], errors="raise").astype(int).to_numpy()
        col = pd.to_numeric(out.loc[indices, "col"], errors="raise").astype(int).to_numpy()
        shape = truth.truth_mask.shape
        inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        if not inside.all():
            raise RuntimeError(f"OOF coordinates outside image {image_key}.")
        out.loc[indices, "true_target"] = truth.truth_mask[row, col]
        out.loc[indices, "truth_available"] = truth.available_mask[row, col]
    if not out["truth_available"].astype(bool).all():
        raise RuntimeError(
            "An OOF projected pixel falls outside the pure-image segmented ROI."
        )
    try:
        target, uncertain = apply_locked_margin_decision(
            pd.to_numeric(out["simca_margin"], errors="coerce").to_numpy(),
            out["decision_mode"].astype(str).to_numpy(),
            direct_2way_threshold=pd.to_numeric(
                out["direct_2way_threshold"], errors="coerce"
            ).to_numpy(),
            three_way_lower_threshold=pd.to_numeric(
                out["three_way_lower_threshold"], errors="coerce"
            ).to_numpy(),
            three_way_upper_threshold=pd.to_numeric(
                out["three_way_upper_threshold"], errors="coerce"
            ).to_numpy(),
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Unknown decision modes"):
            message = message.replace(
                "Unknown decision modes", "Unknown spatial decision modes", 1
            )
        elif "2-way threshold" in message:
            message = "Invalid locked 2-way threshold in calibration_domain."
        elif "3-way threshold" in message:
            message = "Invalid locked 3-way threshold in calibration_domain."
        raise RuntimeError(message) from exc
    out["raw_uncertain"] = uncertain
    out["raw_target"] = target
    coordinate_keys = ["domain_config_id", "source_image", "row", "col"]
    duplicated_coordinates = out.duplicated(coordinate_keys, keep=False)
    if duplicated_coordinates.any():
        examples = out.loc[
            duplicated_coordinates, coordinate_keys + ["object_id"]
        ].head(10)
        raise RuntimeError(
            "A spatial map contains duplicated pixel coordinates: "
            f"{examples.to_dict('records')}"
        )
    out["truth_level"] = expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE
    return out


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
    for index, (label_name, minimum, maximum) in enumerate(
        zip(labels, minima, maxima)
    ):
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


def _select_global_candidate(
    metrics: pd.DataFrame,
    *,
    tolerance: float,
) -> str:
    candidates = metrics.loc[metrics["map_variant"].eq("postprocessed")].copy()
    selection_metrics = [
        "smallest_fragment_recall",
        "component_recall",
        "pixel_recall",
        "dice",
        "iou",
        "component_precision",
        "split_rate",
        "merge_rate",
    ]
    selection_parameters = [
        "min_area_pixels",
        "morphology_radius",
        "morphology_operation",
        "connectivity",
    ]
    required = {
        "spatial_candidate_id",
        "evaluation_track",
        *selection_metrics,
        *selection_parameters,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise KeyError(f"Missing spatial selection columns: {missing}")
    if candidates.empty:
        raise RuntimeError("No spatial candidate can be locked.")
    parameter_counts = candidates.groupby("spatial_candidate_id")[
        selection_parameters
    ].nunique(dropna=False)
    if parameter_counts.gt(1).any().any():
        raise RuntimeError("A candidate identifier has conflicting parameters.")
    expected_tracks = set(candidates["evaluation_track"].astype(str))
    track_coverage = candidates.groupby("spatial_candidate_id")[
        "evaluation_track"
    ].nunique()
    if not track_coverage.eq(len(expected_tracks)).all():
        raise RuntimeError("Spatial candidates do not cover identical track sets.")
    expected_configurations = candidates[
        ["domain_config_id", "evaluation_track"]
    ].drop_duplicates()
    configuration_coverage = candidates.groupby("spatial_candidate_id")[
        "domain_config_id"
    ].nunique()
    if not configuration_coverage.eq(
        expected_configurations["domain_config_id"].nunique()
    ).all():
        raise RuntimeError(
            "Spatial candidates do not cover identical domain configurations."
        )
    numeric_selection = candidates[selection_metrics].apply(
        pd.to_numeric, errors="coerce"
    )
    candidates[selection_metrics] = numeric_selection
    complete_candidate = (
        np.isfinite(numeric_selection.to_numpy(dtype=float))
        .all(axis=1)
    )
    # An id is selectable only when every configuration contributes every
    # predeclared metric; partial skip-na means would favor candidates with
    # missing difficult cases.
    row_counts = candidates.groupby("spatial_candidate_id").size()
    complete_counts = candidates.loc[complete_candidate].groupby(
        "spatial_candidate_id"
    ).size()
    selectable_ids = [
        candidate_id
        for candidate_id, count in row_counts.items()
        if int(complete_counts.get(candidate_id, 0)) == int(count)
    ]
    candidates = candidates.loc[
        candidates["spatial_candidate_id"].isin(selectable_ids)
    ].copy()
    if candidates.empty:
        raise RuntimeError(
            "No spatial candidate has complete finite metrics on every configuration."
        )

    # Two-stage macro aggregation prevents tracks with many retained domain
    # configurations from dominating the global lock (E4/E7/E8 can have very
    # different configuration counts).
    by_track = candidates.groupby(
        ["spatial_candidate_id", "evaluation_track"], as_index=False
    ).agg(
        smallest_fragment_recall=("smallest_fragment_recall", "mean"),
        component_recall=("component_recall", "mean"),
        pixel_recall=("pixel_recall", "mean"),
        dice=("dice", "mean"),
        iou=("iou", "mean"),
        component_precision=("component_precision", "mean"),
        split_rate=("split_rate", "mean"),
        merge_rate=("merge_rate", "mean"),
        min_area_pixels=("min_area_pixels", "first"),
        morphology_radius=("morphology_radius", "first"),
        morphology_operation=("morphology_operation", "first"),
        connectivity=("connectivity", "first"),
    )
    summary = by_track.groupby("spatial_candidate_id", as_index=False).agg(
        smallest_fragment_recall=("smallest_fragment_recall", "mean"),
        component_recall=("component_recall", "mean"),
        pixel_recall=("pixel_recall", "mean"),
        dice=("dice", "mean"),
        iou=("iou", "mean"),
        component_precision=("component_precision", "mean"),
        split_rate=("split_rate", "mean"),
        merge_rate=("merge_rate", "mean"),
        min_area_pixels=("min_area_pixels", "first"),
        morphology_radius=("morphology_radius", "first"),
        morphology_operation=("morphology_operation", "first"),
        connectivity=("connectivity", "first"),
    )
    active = summary.copy()
    for column in (
        "smallest_fragment_recall",
        "component_recall",
        "pixel_recall",
        "dice",
        "iou",
        "component_precision",
    ):
        finite = pd.to_numeric(active[column], errors="coerce")
        if finite.notna().any():
            best = float(finite.max())
            active = active.loc[finite.ge(best - float(tolerance))].copy()
    for column in ("split_rate", "merge_rate"):
        finite = pd.to_numeric(active[column], errors="coerce")
        if finite.notna().any():
            best = float(finite.min())
            active = active.loc[finite.le(best + float(tolerance))].copy()
    active["operation_complexity"] = active["morphology_operation"].map(
        {"none": 0, "opening": 1, "closing": 1, "opening_closing": 2}
    )
    active = active.sort_values(
        [
            "min_area_pixels",
            "operation_complexity",
            "morphology_radius",
            "connectivity",
            "spatial_candidate_id",
        ],
        kind="mergesort",
    )
    return str(active.iloc[0]["spatial_candidate_id"])


def calibrate_spatial_postprocessing(
    spatial_input: pd.DataFrame,
    image_db: dict,
    *,
    protocol_hash: str,
    candidate_grid: pd.DataFrame | None = None,
    tolerance: float = expcfg.SPATIAL_CALIBRATION_SELECTION_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate raw/post maps and freeze one global OOF spatial candidate."""
    grid = _validate_spatial_candidate_grid(
        build_spatial_candidate_grid()
        if candidate_grid is None
        else candidate_grid.copy()
    )
    group_ids = [
        "domain_config_id",
        "evaluation_track",
        "track_id",
        "projection_config_id",
    ]
    metric_rows: list[dict] = []
    fragment_rows: list[dict] = []
    for key, configuration in spatial_input.groupby(group_ids, sort=False):
        metadata = dict(zip(group_ids, key if isinstance(key, tuple) else (key,)))
        maps = [
            _maps_for_group(group, image_db)
            for _, group in configuration.groupby("source_image", sort=False)
        ]
        for connectivity in sorted(grid["connectivity"].astype(int).unique()):
            raw_id = f"raw_c{connectivity}"
            raw_metrics, raw_fragments = _evaluate_maps(
                maps, connectivity=int(connectivity), candidate=None
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
            metrics, fragments = _evaluate_maps(
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
                    "morphology_operation": str(candidate["morphology_operation"]),
                    "morphology_radius": int(candidate["morphology_radius"]),
                    "min_area_pixels": int(candidate["min_area_pixels"]),
                    **metrics,
                    "is_locked_candidate": False,
                    "truth_level": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
                    "protocol_version": expcfg.PROTOCOL_VERSION,
                    "protocol_hash": str(protocol_hash),
                }
            )
            fragment_rows.extend(
                _summarize_fragment_classes(
                    fragments,
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
    selected_id = _select_global_candidate(metrics, tolerance=float(tolerance))
    metrics["is_locked_candidate"] = metrics["spatial_candidate_id"].eq(selected_id)
    fragments["is_locked_candidate"] = fragments["spatial_candidate_id"].eq(selected_id)
    metrics = metrics.reindex(columns=expcfg.SPATIAL_CALIBRATION_METRIC_COLUMNS)
    fragments = fragments.reindex(columns=expcfg.FRAGMENT_SIZE_CLASS_COLUMNS)
    selected = grid.loc[grid["spatial_candidate_id"].eq(selected_id)]
    if len(selected) != 1:
        raise RuntimeError("The global spatial lock is not unique.")
    selected_payload = selected.iloc[0].to_dict()
    lock = {
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "protocol_hash": str(protocol_hash),
        "rule_version": expcfg.SPATIAL_CALIBRATION_RULE_VERSION,
        "truth_source": expcfg.SPATIAL_CALIBRATION_TRUTH_SOURCE,
        "allowed_batches": list(map(int, expcfg.SPATIAL_CALIBRATION_ALLOWED_BATCHES)),
        "forbidden_batches": list(map(int, expcfg.SPATIAL_CALIBRATION_FORBIDDEN_BATCHES)),
        "uncertain_pixel_policy": "preserve_as_distinct_immutable_layer",
        "selection_policy": (
            "global_track_balanced_lexicographic_plateau_then_minimum_complexity"
        ),
        "selection_weighting": (
            "equal_evaluation_track_after_equal_domain_configuration"
        ),
        "selection_tolerance": float(tolerance),
        "selected_parameters": selected_payload,
        "area_minimum_version": expcfg.SPATIAL_CALIBRATION_RULE_VERSION,
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
    """Block when a locked parameter or calibration output has changed."""
    payload = dict(lock)
    lock_hash = payload.pop("lock_sha256", None)
    expected_hash = _payload_hash(payload)
    if str(lock_hash) != expected_hash:
        raise RuntimeError("The spatial post-processing lock was modified.")
    if str(lock["spatial_calibration_metrics_sha256"]) != sha256_dataframe(metrics):
        raise RuntimeError("spatial_calibration_metrics.parquet changed after lock.")
    if str(lock["fragment_size_classes_sha256"]) != sha256_dataframe(fragments):
        raise RuntimeError("fragment_size_classes.parquet changed after lock.")
    selected_id = str(lock["selected_parameters"]["spatial_candidate_id"])
    selected = metrics.loc[
        metrics["is_locked_candidate"].astype(bool), "spatial_candidate_id"
    ].astype(str).unique()
    if selected.tolist() != [selected_id]:
        raise RuntimeError("Locked spatial candidate does not match metrics.")


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
    candidate_pool: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    image_db: Mapping[str, Mapping],
    spatial_lock: Mapping,
) -> dict[str, pd.DataFrame]:
    """Build task-32 raw/post maps, components and fragment diagnostics."""
    required_candidates = {
        "validation_candidate_id",
        "calibration_id",
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
    required_pixels = {
        "projection_config_id",
        "random_state",
        "source_image",
        "object_id",
        "batch",
        "row",
        "col",
        "simca_margin",
    }
    for frame, required, name in (
        (candidate_pool, required_candidates, "candidate_pool"),
        (pixel_predictions, required_pixels, "validation_pixel_predictions"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing spatial columns: {missing}")
    payload = dict(spatial_lock)
    lock_hash = str(payload.pop("lock_sha256", ""))
    if lock_hash != _payload_hash(payload):
        raise RuntimeError("The spatial post-processing lock was modified.")
    parameters = dict(spatial_lock["selected_parameters"])
    connectivity = int(parameters["connectivity"])
    pixel_candidates = candidate_pool.loc[
        candidate_pool["projection_level"].astype(str).eq("pixel_projection")
    ].copy()
    prediction_lookup = {
        (str(key[0]), int(key[1])): np.asarray(indices, dtype=int)
        for key, indices in pixel_predictions.groupby(
            ["projection_config_id", "random_state"], sort=False
        ).indices.items()
    }
    manifests: list[dict] = []
    component_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    bounds = tuple(expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_UPPER_BOUNDS)
    labels = tuple(expcfg.SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS)
    for candidate in pixel_candidates.to_dict("records"):
        validation_candidate_id = str(candidate["validation_candidate_id"])
        calibration_id = str(candidate["calibration_id"])
        projection_id = str(candidate["projection_config_id"])
        key = (projection_id, int(candidate["random_state"]))
        positions = prediction_lookup.get(key)
        if positions is None or not len(positions):
            # The technical failure is already retained in validation_metrics.
            continue
        observations = pixel_predictions.iloc[positions].copy()
        target, uncertain = apply_locked_margin_decision(
            observations["simca_margin"].to_numpy(dtype=float),
            str(candidate["decision_mode"]),
            direct_2way_threshold=candidate["direct_2way_threshold"],
            three_way_lower_threshold=candidate["three_way_lower_threshold"],
            three_way_upper_threshold=candidate["three_way_upper_threshold"],
        )
        observations["__raw_target"] = target
        observations["__uncertain"] = uncertain
        for image_key, image_positions in observations.groupby(
            "source_image", sort=False
        ).indices.items():
            image_key = str(image_key)
            if image_key not in image_db:
                raise KeyError(f"Validation image is absent from HDF5: {image_key}")
            group = observations.iloc[image_positions]
            if group.duplicated(["row", "col"]).any():
                raise RuntimeError(
                    f"Duplicated validation pixel coordinates for {image_key}."
                )
            batch_values = pd.to_numeric(group["batch"], errors="raise").astype(int)
            if set(batch_values) != set(
                map(int, expcfg.SIMCA_CONCAT_REFIT_PROJECTION_BATCHES)
            ):
                raise RuntimeError("Spatial validation must use batch 3 only.")
            truth_result = pure_image_class_truth(
                image_key,
                dict(image_db),
                target_class=expcfg.TARGET_CLASS,
                allowed_batches=expcfg.SIMCA_CONCAT_REFIT_PROJECTION_BATCHES,
            )
            shape = truth_result.truth_mask.shape
            row = pd.to_numeric(group["row"], errors="raise").astype(int).to_numpy()
            col = pd.to_numeric(group["col"], errors="raise").astype(int).to_numpy()
            inside = (
                (row >= 0)
                & (col >= 0)
                & (row < shape[0])
                & (col < shape[1])
            )
            if not inside.all():
                raise RuntimeError(f"Validation coordinates outside {image_key}.")
            valid = np.zeros(shape, dtype=bool)
            raw_target = np.zeros(shape, dtype=bool)
            uncertain_map = np.zeros(shape, dtype=bool)
            valid[row, col] = True
            raw_target[row, col] = group["__raw_target"].astype(bool).to_numpy()
            uncertain_map[row, col] = group["__uncertain"].astype(bool).to_numpy()
            valid &= truth_result.available_mask
            raw_target &= valid
            uncertain_map &= valid
            truth = truth_result.truth_mask & valid
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
                    "validation_candidate_id": validation_candidate_id,
                    "calibration_id": calibration_id,
                    "evaluation_track": str(candidate["evaluation_track"]),
                    "track_id": str(candidate["track_id"]),
                    "projection_config_id": projection_id,
                    "random_state": int(candidate["random_state"]),
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
                    "margin_source": (
                        "validation_pixel_predictions.parquet#"
                        f"projection_config_id={projection_id}"
                    ),
                    "truth_level": truth_result.truth_level,
                    "spatial_lock_sha256": lock_hash,
                }
            )

            image_labels = np.asarray(image_db[image_key]["labels"])
            truth_component_labels = np.where(truth, image_labels, 0)
            evaluable = valid & ~uncertain_map
            for map_variant, prediction in (
                ("raw", raw_target),
                ("locked_postprocessed", post_target),
            ):
                prediction = np.asarray(prediction, dtype=bool) & evaluable
                evaluable_truth = truth & evaluable
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
                    for column, value in (
                        ("calibration_id", calibration_id),
                        ("validation_candidate_id", validation_candidate_id),
                        ("evaluation_track", str(candidate["evaluation_track"])),
                        ("track_id", str(candidate["track_id"])),
                        ("random_state", int(candidate["random_state"])),
                        ("source_image", image_key),
                        ("map_variant", map_variant),
                        ("truth_level", truth_result.truth_level),
                    ):
                        components[column] = value
                    component_parts.append(components)
                intersection = int(np.count_nonzero(evaluable_truth & prediction))
                union = int(np.count_nonzero(evaluable_truth | prediction))
                n_truth_pixels = int(np.count_nonzero(evaluable_truth))
                n_prediction_pixels = int(np.count_nonzero(prediction))
                truth_components = components.loc[
                    components.get("component_role", pd.Series(dtype=str)).eq("truth")
                ] if len(components) else pd.DataFrame()
                observed_classes = set(
                    truth_components.get("area_class", pd.Series(dtype=str)).astype(str)
                )
                smallest = next(
                    (label_name for label_name in labels if label_name in observed_classes),
                    None,
                )
                smallest_rows = (
                    truth_components.loc[truth_components["area_class"].eq(smallest)]
                    if smallest is not None
                    else truth_components.iloc[0:0]
                )
                metric_rows.append(
                    {
                        "validation_candidate_id": validation_candidate_id,
                        "calibration_id": calibration_id,
                        "evaluation_track": str(candidate["evaluation_track"]),
                        "track_id": str(candidate["track_id"]),
                        "random_state": int(candidate["random_state"]),
                        "source_image": image_key,
                        "aggregation_level": "source_image",
                        "map_variant": map_variant,
                        "n_valid_pixels": int(evaluable.sum()),
                        "dice": (
                            2.0 * intersection / (n_truth_pixels + n_prediction_pixels)
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
                            float(smallest_rows["detected_or_matched"].mean())
                            if len(smallest_rows)
                            else np.nan
                        ),
                        "truth_level": truth_result.truth_level,
                        "__truth_pixels": n_truth_pixels,
                        "__prediction_pixels": n_prediction_pixels,
                        "__intersection": intersection,
                        "__union": union,
                    }
                )

    image_metrics = pd.DataFrame(metric_rows)
    overall_rows: list[dict] = []
    if len(image_metrics):
        components_all = (
            pd.concat(component_parts, ignore_index=True, sort=False)
            if component_parts
            else pd.DataFrame()
        )
        for keys, group in image_metrics.groupby(
            [
                "validation_candidate_id",
                "calibration_id",
                "evaluation_track",
                "track_id",
                "random_state",
                "map_variant",
            ],
            sort=False,
        ):
            (
                validation_candidate_id,
                calibration_id,
                evaluation_track,
                track_id,
                random_state,
                map_variant,
            ) = keys
            totals = group[
                [
                    "n_valid_pixels",
                    "__truth_pixels",
                    "__prediction_pixels",
                    "__intersection",
                    "__union",
                    "n_truth_components",
                    "n_predicted_components",
                    "n_detected_truth_components",
                    "n_matched_predicted_components",
                    "n_split_truth_components",
                    "n_merged_predicted_components",
                ]
            ].sum(numeric_only=True)
            truth_total = float(totals["__truth_pixels"])
            prediction_total = float(totals["__prediction_pixels"])
            intersection = float(totals["__intersection"])
            union = float(totals["__union"])
            truth_component_total = float(totals["n_truth_components"])
            predicted_component_total = float(totals["n_predicted_components"])
            truth_components = components_all.loc[
                components_all["validation_candidate_id"].astype(str).eq(
                    str(validation_candidate_id)
                )
                & components_all["map_variant"].astype(str).eq(str(map_variant))
                & components_all["component_role"].astype(str).eq("truth")
            ] if len(components_all) else pd.DataFrame()
            observed_classes = set(
                truth_components.get("area_class", pd.Series(dtype=str)).astype(str)
            )
            smallest = next(
                (label_name for label_name in labels if label_name in observed_classes),
                None,
            )
            smallest_rows = (
                truth_components.loc[truth_components["area_class"].eq(smallest)]
                if smallest is not None
                else truth_components.iloc[0:0]
            )
            overall_rows.append(
                {
                    "validation_candidate_id": validation_candidate_id,
                    "calibration_id": calibration_id,
                    "evaluation_track": evaluation_track,
                    "track_id": track_id,
                    "random_state": int(random_state),
                    "source_image": "all",
                    "aggregation_level": "overall",
                    "map_variant": map_variant,
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
                        float(smallest_rows["detected_or_matched"].mean())
                        if len(smallest_rows)
                        else np.nan
                    ),
                    "truth_level": expcfg.SIMCA_CONCAT_REFIT_TRUTH_SOURCE,
                }
            )
    components = (
        pd.concat(component_parts, ignore_index=True, sort=False)
        if component_parts
        else pd.DataFrame(columns=expcfg.SIMCA_SPATIAL_COMPONENT_COLUMNS)
    )
    metrics = pd.concat(
        [image_metrics, pd.DataFrame(overall_rows)],
        ignore_index=True,
        sort=False,
    ) if len(image_metrics) or overall_rows else pd.DataFrame()
    return {
        "pixel_maps_manifest": pd.DataFrame(manifests).reindex(
            columns=expcfg.SIMCA_PIXEL_MAP_MANIFEST_COLUMNS
        ),
        "spatial_components": components.reindex(
            columns=expcfg.SIMCA_SPATIAL_COMPONENT_COLUMNS
        ),
        "spatial_component_metrics": metrics.reindex(
            columns=expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS
        ),
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
