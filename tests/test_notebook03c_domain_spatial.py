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
                    "projection_config_id": "proj_pixel",
                    "fit_config_id": "fit_pixel",
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


def _pixel_domain():
    return pd.DataFrame(
        [
            {
                "domain_config_id": "domain_pixel",
                "evaluation_track": "pixel_matrix__pixel_projection__2way",
                "track_id": "E3",
                "projection_config_id": "proj_pixel",
                "fit_config_id": "fit_pixel",
                "projection_level": "pixel_projection",
                "projection_matrix_method": "pixel",
                "decision_mode": "2way",
                "direct_2way_threshold": 0.0,
                "three_way_lower_threshold": np.nan,
                "three_way_upper_threshold": np.nan,
            }
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
        _pixel_oof(), _pixel_domain(), image_db
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
            "fit_config_id": "fit_pixel",
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
        _pixel_domain(),
        projection_shift,
        object_db=object_db,
        protocol_hash="protocol-test",
        min_stratum_n=1,
    )
    eligibility = build_projection_eligibility(
        diagnostics,
        _pixel_domain(),
        protocol_hash="protocol-test",
        expected_tracks=["pixel_matrix__pixel_projection__2way"],
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

    e3_track = "object_train__pixel_projection__2way"
    eligibility_with_unsupported = build_projection_eligibility(
        diagnostics,
        _pixel_domain(),
        protocol_hash="protocol-test",
        expected_tracks=[
            "pixel_matrix__pixel_projection__2way",
            e3_track,
        ],
        unsupported_tracks={e3_track: "internal_calibration:risk_constraints"},
        thresholds={
            "warning_max_abs_standardized_shift": 100.0,
            "unsupported_max_abs_standardized_shift": 200.0,
            "warning_out_of_domain_rate": 1.0,
            "unsupported_out_of_domain_rate": 2.0,
            "warning_target_rejection_rate": 1.0,
            "unsupported_target_rejection_rate": 2.0,
        },
    )
    e3 = eligibility_with_unsupported.loc[
        eligibility_with_unsupported["evaluation_track"].eq(e3_track)
    ]
    assert e3["track_id"].eq("E3").all()
    assert e3["eligibility_status"].eq(
        "unsupported_internal_calibration"
    ).all()
    assert e3["n_projection_configurations"].eq(0).all()


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
            "fit_config_id": "fit_pixel",
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
            _pixel_domain(),
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
            "fit_config_id": "fit_pixel",
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
        _pixel_domain(),
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
    unknown = _pixel_domain().copy()
    unknown["decision_mode"] = "unexpected"
    with pytest.raises(RuntimeError, match="Unknown spatial decision modes"):
        build_spatial_calibration_input(_pixel_oof(), unknown, image_db)

    missing_threshold = _pixel_domain().copy()
    missing_threshold["direct_2way_threshold"] = np.nan
    with pytest.raises(RuntimeError, match="Invalid locked 2-way threshold"):
        build_spatial_calibration_input(
            _pixel_oof(), missing_threshold, image_db
        )

    duplicated = pd.concat(
        [_pixel_oof(), _pixel_oof().iloc[[0]]], ignore_index=True
    )
    with pytest.raises(RuntimeError, match="duplicated pixel coordinates"):
        build_spatial_calibration_input(duplicated, _pixel_domain(), image_db)


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
                    "domain_config_id": f"e4_{index}",
                    "evaluation_track": "E4_track",
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
                "domain_config_id": "e7_0",
                "evaluation_track": "E7_track",
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
    assert "SPATIAL_CALIBRATION_FORBIDDEN_BATCHES" in source
    assert "Eligibility diagnostics must use target projections only" in source
    assert "spatial_calibration_domain" in source
    assert "The spatial lock is not track-balanced" in source
    assert source.count("optimize=False") >= 2
    assert "persisted_spatial_metrics" in source
    assert "persisted_fragment_size_classes" in source
    for name in ("04A_simca_grid_search.ipynb", "04B_simca_optuna_search.ipynb"):
        downstream = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        assert (
            "PROJECTION_ELIGIBILITY_PATH" in downstream
            or "projection_eligibility" in downstream
        )
        assert "verify_spatial_postprocessing_lock" in downstream
        assert (
            "SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES" in downstream
            or "unsupported_domain_shift" in downstream
        )
        assert (
            "unsupported_tracks" in downstream
            or "unsupported_internal_calibration" in downstream
        )
