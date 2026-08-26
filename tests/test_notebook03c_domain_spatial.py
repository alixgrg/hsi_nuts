import json
from pathlib import Path
from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.decision.truth import pure_image_class_truth
from src.workflows.projection_domain_audit import (
    build_projection_eligibility,
    build_projection_shift_diagnostics,
    summarize_projection_shift,
)
from src.workflows.simca_calibration_registry import (
    build_selected_execution_registry,
)
from src.workflows.spatial_postprocessing_calibration import (
    apply_spatial_postprocessing,
    build_spatial_calibration_input,
    build_spatial_candidate_grid,
    calibrate_spatial_postprocessing,
    verify_spatial_postprocessing_lock,
)
from src.workflows import spatial_postprocessing_calibration as spatial_module


ROOT = Path(__file__).resolve().parents[1]


def _image_db():
    labels = np.array([[1, 1], [1, 1]], dtype=int)
    return {
        "almond1": {
            "labels": labels,
            "is_pure": True,
            "sample_kind": "pure",
            "nut_type": "almond",
            "batch": 1,
        },
        "peanut2": {
            "labels": labels,
            "is_pure": True,
            "sample_kind": "pure",
            "nut_type": "peanut",
            "batch": 2,
        },
    }


def _pixel_oof():
    rows = []
    for image, object_id, batch, margins in (
        ("almond1", "a1", 1, [-1.0, -0.8, -0.5, -0.2]),
        ("peanut2", "p2", 2, [0.8, 0.7, 0.6, 0.5]),
    ):
        for (row, col), margin in zip(np.ndindex(2, 2), margins):
            rows.append(
                {
                    "projection_id": "proj_pixel",
                    "projection_level": "pixel_projection",
                    "projection_matrix_method": "pixel",
                    "fold_id": batch - 1,
                    "source_image": image,
                    "object_id": object_id,
                    "batch": batch,
                    "object_area": 4,
                    "size_bin": "small",
                    "truth": image.startswith("peanut"),
                    "pca_score_pc1": margin,
                    "pca_score_pc2": margin / 2,
                    "H": abs(margin),
                    "Q": abs(margin) / 2,
                    "rule_limit": 1.0,
                    "normalized_ratio": 1.0 - margin,
                    "simca_margin": margin,
                    "row": row,
                    "col": col,
                }
            )
    return pd.DataFrame(rows)


def _pixel_executions():
    return pd.DataFrame(
        [
            {
                "model_id": "model_pixel",
                "random_state": 0,
                "track_id": "E3",
                "projection_id": "proj_pixel",
                "projection_level": "pixel_projection",
                "decision_mode": "2way",
            }
        ]
    )


def _selected_thresholds():
    return pd.DataFrame(
        [
            {
                "model_id": "model_pixel",
                "random_state": 0,
                "decision_scope": "direct",
                "lower_quantile": np.nan,
                "upper_quantile": np.nan,
                "vote_threshold": np.nan,
                "lower_threshold": 0.0,
                "upper_threshold": 0.0,
            },
            {
                "model_id": "model_pixel",
                "random_state": 0,
                "decision_scope": "pixel_to_object",
                "lower_quantile": np.nan,
                "upper_quantile": np.nan,
                "vote_threshold": 0.75,
                "lower_threshold": 0.75,
                "upper_threshold": 0.75,
            },
        ]
    )


def test_pure_image_truth_is_exact_inside_segmentation():
    image_db = _image_db()
    almond = pure_image_class_truth("almond1", image_db)
    peanut = pure_image_class_truth("peanut2", image_db)
    assert almond.truth_level == "pure_image_class_exact"
    assert not almond.truth_mask.any()
    assert peanut.truth_mask.all()
    assert almond.available_mask.all() and peanut.available_mask.all()


def test_selected_execution_registry_filters_intermediate_thresholds():
    catalog_row = {
        column: pd.NA
        for column in expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS
    }
    catalog_row.update(
        {
            "model_id": "model_pixel",
            "evaluation_track": "object_train__pixel_projection__2way",
            "track_id": "E3",
            "parent_track": "object_train__pixel_projection",
            "decision_mode": "2way",
            "matrix_family": "object",
            "matrix_method": "object_mean",
            "projection_level": "pixel_projection",
            "projection_matrix_method": "pixel",
            "m": np.nan,
            "balanced_pixel_strategy": "not_applicable",
            "preprocessing": "snv",
            "preprocessing_steps": "snv",
            "rule_family": "simple",
            "rule_variant": "simple_classical",
            "limit_source": "theoretical_train_fit",
            "n_components": 2,
            "alpha": 0.01,
            "sg_window_length": 11,
            "sg_polyorder": 2,
            "position_dilation_radius": 0,
        }
    )
    thresholds = pd.concat(
        [
            _selected_thresholds(),
            _selected_thresholds().assign(model_id="intermediate_model"),
        ],
        ignore_index=True,
    )
    executions, filtered_thresholds = build_selected_execution_registry(
        pd.DataFrame([catalog_row]),
        pd.DataFrame(
            [{"model_id": "model_pixel", "selection_status": "selected"}]
        ),
        pd.DataFrame(
            [
                {
                    "model_id": "model_pixel",
                    "random_state": 0,
                    "fit_id": "fit_pixel",
                    "projection_id": "proj_pixel",
                }
            ]
        ),
        thresholds,
        track_contracts=pd.DataFrame(
            [
                {
                    "track_id": "E3",
                    "decision_mode": "2way",
                    "projection_level": "pixel_projection",
                }
            ]
        ),
    )
    assert list(executions.columns) == list(
        expcfg.DOMAIN_SPATIAL_SELECTED_EXECUTION_COLUMNS
    )
    assert executions[["model_id", "random_state"]].value_counts().iloc[0] == 1
    assert set(filtered_thresholds["model_id"]) == {"model_pixel"}
    assert set(filtered_thresholds["decision_scope"]) == {
        "direct",
        "pixel_to_object",
    }


def test_uncertainty_layer_is_immutable():
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    uncertain = np.array([[0, 1], [0, 0]], dtype=bool)
    cleaned, preserved = apply_spatial_postprocessing(
        target,
        uncertain,
        np.ones((2, 2), dtype=bool),
        connectivity=1,
        morphology_operation="none",
        morphology_radius=0,
        min_area_pixels=0,
    )
    np.testing.assert_array_equal(preserved, uncertain)
    assert not cleaned[0, 1]


def test_spatial_calibration_emits_raw_post_and_verifiable_lock():
    image_db = _image_db()
    spatial_input = build_spatial_calibration_input(
        _pixel_oof(), _pixel_executions(), _selected_thresholds(), image_db
    )
    grid = build_spatial_candidate_grid(
        connectivities=(1,),
        operations=("none",),
        radii=(0,),
        min_areas=(0,),
    )
    metrics, fragments, lock = calibrate_spatial_postprocessing(
        spatial_input,
        image_db,
        protocol_hash="protocol-test",
        candidate_grid=grid,
    )
    assert set(metrics["map_variant"]) == {"raw", "postprocessed"}
    assert metrics["is_locked_candidate"].any()
    assert set(metrics["truth_level"]) == {"pure_image_class_exact"}
    assert json.dumps(lock)
    verify_spatial_postprocessing_lock(lock, metrics, fragments)
    metrics_buffer = BytesIO()
    fragments_buffer = BytesIO()
    metrics.to_parquet(metrics_buffer, index=False)
    fragments.to_parquet(fragments_buffer, index=False)
    metrics_buffer.seek(0)
    fragments_buffer.seek(0)
    verify_spatial_postprocessing_lock(
        lock,
        pd.read_parquet(metrics_buffer),
        pd.read_parquet(fragments_buffer),
    )


def test_domain_diagnostics_and_status_cover_the_track():
    pixels = _pixel_oof()
    pixels["fold_id"] = 0
    train = pd.DataFrame(
        {
            "pca_score_pc1": [-1.0, -0.5, 0.5, 1.0],
            "pca_score_pc2": [-0.5, -0.25, 0.25, 0.5],
            "H": [0.2, 0.4, 0.6, 0.8],
            "Q": [0.1, 0.2, 0.3, 0.4],
            "rule_limit": 1.0,
            "normalized_ratio": [0.2, 0.4, 0.6, 0.8],
            "simca_margin": [0.8, 0.6, 0.4, 0.2],
        }
    )
    shifts = []
    for fold_id, group in pixels.groupby("fold_id"):
        fold_train = train.assign(fold_id=int(fold_id))
        shifts.append(
            summarize_projection_shift(fold_train, group)
        )
    projection_shift = pd.concat(shifts, ignore_index=True)
    object_db = {
        "a1": {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=bool)},
        "p2": {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=bool)},
    }
    diagnostics = build_projection_shift_diagnostics(
        pd.DataFrame(),
        pixels,
        _pixel_executions(),
        projection_shift,
        object_db=object_db,
        protocol_hash="protocol-test",
        min_stratum_n=1,
    )
    eligibility = build_projection_eligibility(
        diagnostics,
        _pixel_executions(),
        protocol_hash="protocol-test",
        expected_track_ids=["E3"],
        thresholds={
            "warning_max_abs_standardized_shift": 100.0,
            "unsupported_max_abs_standardized_shift": 200.0,
            "warning_out_of_domain_rate": 1.0,
            "unsupported_out_of_domain_rate": 2.0,
            "warning_target_rejection_rate": 1.0,
            "unsupported_target_rejection_rate": 2.0,
        },
    )
    assert set(diagnostics["stratum_type"]) == set(
        expcfg.PROJECTION_DOMAIN_DIAGNOSTIC_DIMENSIONS
    )
    assert eligibility.loc[0, "eligibility_status"] == "eligible"

    unsupported_track = "E4"
    eligibility_with_unsupported = build_projection_eligibility(
        diagnostics,
        _pixel_executions(),
        protocol_hash="protocol-test",
        expected_track_ids=["E3", unsupported_track],
        thresholds={
            "warning_max_abs_standardized_shift": 100.0,
            "unsupported_max_abs_standardized_shift": 200.0,
            "warning_out_of_domain_rate": 1.0,
            "unsupported_out_of_domain_rate": 2.0,
            "warning_target_rejection_rate": 1.0,
            "unsupported_target_rejection_rate": 2.0,
        },
    )
    e4 = eligibility_with_unsupported.loc[
        eligibility_with_unsupported["track_id"].eq(unsupported_track)
    ]
    assert e4["eligibility_status"].eq(
        "unsupported_internal_calibration"
    ).all()
    assert e4["n_selected_runs"].eq(0).all()


def test_tiny_nonzero_q_scale_remains_finite_without_reduce_warning():
    pixels = _pixel_oof()
    pixels["Q"] = pd.to_numeric(pixels["Q"]) * 1e-12
    train = pd.DataFrame(
        {
            "pca_score_pc1": [-1.0, -0.5, 0.5, 1.0],
            "pca_score_pc2": [-0.5, -0.25, 0.25, 0.5],
            "H": [0.2, 0.4, 0.6, 0.8],
            "Q": [0.5e-12, 1.0e-12, 1.5e-12, 2.0e-12],
            "rule_limit": 1.0,
            "normalized_ratio": [0.2, 0.4, 0.6, 0.8],
            "simca_margin": [0.8, 0.6, 0.4, 0.2],
        }
    )
    shifts = []
    for fold_id, group in pixels.groupby("fold_id"):
        shifts.append(
            summarize_projection_shift(
                train.assign(fold_id=int(fold_id)), group
            )
        )
    object_db = {
        "a1": {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=bool)},
        "p2": {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=bool)},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = build_projection_shift_diagnostics(
            pd.DataFrame(),
            pixels,
            _pixel_executions(),
            pd.concat(shifts, ignore_index=True),
            object_db=object_db,
            protocol_hash="protocol-test",
            min_stratum_n=1,
        )
    supported = diagnostics["n_observations"].gt(0)
    assert np.isfinite(diagnostics.loc[supported, "q_standardized_shift"]).all()
    assert not any(
        "invalid value encountered in reduce" in str(item.message)
        for item in caught
    )


def test_eligibility_diagnostics_compare_target_projection_only():
    pixels = _pixel_oof()
    # Deliberately extreme non-target values must remain descriptive; they do
    # not measure support of target projections against a target train model.
    non_target = ~pixels["truth"].astype(bool)
    pixels.loc[non_target, "normalized_ratio"] = 100.0
    pixels.loc[non_target, "simca_margin"] = -99.0
    train = pd.DataFrame(
        {
            "pca_score_pc1": [-1.0, -0.5, 0.5, 1.0],
            "pca_score_pc2": [-0.5, -0.25, 0.25, 0.5],
            "H": [0.2, 0.4, 0.6, 0.8],
            "Q": [0.1, 0.2, 0.3, 0.4],
            "rule_limit": 1.0,
            "normalized_ratio": [0.2, 0.4, 0.6, 0.8],
            "simca_margin": [0.8, 0.6, 0.4, 0.2],
        }
    )
    shifts = [
        summarize_projection_shift(
            train.assign(fold_id=int(fold_id)), group
        )
        for fold_id, group in pixels.groupby("fold_id")
    ]
    object_db = {
        "a1": {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=bool)},
        "p2": {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=bool)},
    }
    diagnostics = build_projection_shift_diagnostics(
        pd.DataFrame(),
        pixels,
        _pixel_executions(),
        pd.concat(shifts, ignore_index=True),
        object_db=object_db,
        protocol_hash="protocol-test",
        min_stratum_n=1,
    )
    overall = diagnostics.loc[diagnostics["stratum_type"].eq("overall")].iloc[0]
    assert overall["n_observations"] == overall["n_target"] == 4
    assert overall["out_of_domain_rate"] == 0.0
    non_target_description = diagnostics.loc[
        diagnostics["stratum_type"].eq("truth_class")
        & diagnostics["stratum_value"].eq("non_target")
    ]
    assert len(non_target_description) == 1
    assert non_target_description.iloc[0]["out_of_domain_rate"] == 1.0


def test_spatial_input_rejects_unknown_modes_threshold_gaps_and_duplicate_pixels():
    image_db = _image_db()
    unknown = _pixel_executions().copy()
    unknown["decision_mode"] = "unexpected"
    with pytest.raises(RuntimeError, match="Unknown spatial decision modes"):
        build_spatial_calibration_input(
            _pixel_oof(), unknown, _selected_thresholds(), image_db
        )

    missing_threshold = _selected_thresholds().copy()
    missing_threshold.loc[
        missing_threshold["decision_scope"].eq("direct"), "lower_threshold"
    ] = np.nan
    with pytest.raises(RuntimeError, match="no finite direct threshold"):
        build_spatial_calibration_input(
            _pixel_oof(), _pixel_executions(), missing_threshold, image_db
        )

    duplicated = pd.concat(
        [_pixel_oof(), _pixel_oof().iloc[[0]]], ignore_index=True
    )
    with pytest.raises(RuntimeError, match="duplicated pixel coordinates"):
        build_spatial_calibration_input(
            duplicated, _pixel_executions(), _selected_thresholds(), image_db
        )


def test_uncertain_pixels_are_preserved_but_excluded_from_scored_truth():
    maps = {
        "valid": np.ones((2, 2), dtype=bool),
        "truth": np.ones((2, 2), dtype=bool),
        "target": np.array([[True, False], [False, False]], dtype=bool),
        "uncertain": np.array([[False, True], [True, True]], dtype=bool),
    }
    metrics, _ = spatial_module._evaluate_maps(
        [maps], connectivity=1, candidate=None
    )
    assert metrics["pixel_recall"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["uncertain_pixel_rate"] == 0.75


def test_global_spatial_selection_equal_weights_tracks_not_configuration_counts():
    rows = []
    for candidate_id, e4_score, e7_score in (
        ("candidate_a", 1.0, 0.0),
        ("candidate_b", 0.6, 0.6),
    ):
        for index in range(10):
            rows.append(
                {
                    "spatial_candidate_id": candidate_id,
                    "map_variant": "postprocessed",
                    "model_id": f"e4_{index}",
                    "random_state": 0,
                    "track_id": "E4",
                    "smallest_fragment_recall": e4_score,
                    "component_recall": 1.0,
                    "pixel_recall": 1.0,
                    "dice": 1.0,
                    "iou": 1.0,
                    "component_precision": 1.0,
                    "split_rate": 0.0,
                    "merge_rate": 0.0,
                    "min_area_pixels": 0,
                    "morphology_radius": 0,
                    "morphology_operation": "none",
                    "connectivity": 1,
                }
            )
        rows.append(
            {
                "spatial_candidate_id": candidate_id,
                "map_variant": "postprocessed",
                "model_id": "e7_0",
                "random_state": 0,
                "track_id": "E7",
                "smallest_fragment_recall": e7_score,
                "component_recall": 1.0,
                "pixel_recall": 1.0,
                "dice": 1.0,
                "iou": 1.0,
                "component_precision": 1.0,
                "split_rate": 0.0,
                "merge_rate": 0.0,
                "min_area_pixels": 0,
                "morphology_radius": 0,
                "morphology_operation": "none",
                "connectivity": 1,
            }
        )
    selected = spatial_module._select_global_candidate(
        pd.DataFrame(rows), tolerance=0.005
    )
    assert selected == "candidate_b"


def test_notebook_03c_and_downstream_guards_are_materialized():
    notebook = json.loads(
        (ROOT / "notebooks" / "03C_projection_spatial_calibration.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for filename in expcfg.DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES.values():
        assert filename in source or "output_paths" in source
    assert "build_projection_shift_diagnostics" in source
    assert "build_spatial_calibration_input" in source
    assert "build_selected_execution_registry" in source
    assert "validate_internal_calibration_manifest" in source
    assert "DOMAIN_SPATIAL_REQUIRED_03B_ARTIFACTS" in source
    assert "SPATIAL_CALIBRATION_FORBIDDEN_BATCHES" in source
    assert "Eligibility diagnostics must use target projections only" in source
    assert "The spatial lock is not track-balanced" in source
    assert source.count("optimize=False") == 4
    assert "persisted_spatial_metrics" in source
    assert "persisted_fragment_size_classes" in source
    assert "natural_execution_key" in source
    assert "audit_manifest" in source
    assert "spatial_calibration_domain" not in source
    assert "calibration_domain" not in source
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), "notebook03c", "exec")
