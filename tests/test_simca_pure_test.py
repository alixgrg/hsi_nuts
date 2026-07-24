from pathlib import Path

import pandas as pd
import pytest

from src.workflows.simca_pure_test import (
    build_3way_outputs,
    build_pure_test_guardrails,
    build_pure_test_projection_filters,
    select_pure_test_candidate_panel,
    validate_pure_test_guardrails,
)


def _candidate_row(selected_config_id, matrix_family="object_matrix"):
    return {
        "selected_config_id": selected_config_id,
        "candidate_id": f"cand_{selected_config_id}",
        "candidate_sources": "grid",
        "target_class": "peanut",
        "non_target_label": "almond",
        "matrix_family": matrix_family,
        "matrix_method": "object_mean" if matrix_family == "object_matrix" else "balanced_pixels",
        "training_matrix_id": "object_mean" if matrix_family == "object_matrix" else "balanced_pixel_m40",
        "preprocessing": "snv",
        "preprocessing_steps": "snv",
        "model_family": "empirical_cv_rule",
        "rule_variant": "simple_emp_cv",
        "rule_for_refit": "simple_emp_cv",
        "n_components": 4,
        "alpha": 0.01,
        "object_threshold": 0.75,
    }


def test_pure_test_guardrails_accept_canonical_split(tmp_path):
    thresholds_path = tmp_path / "validation_3way_selected_thresholds.parquet"
    pd.DataFrame({"selected_config_id": ["sel_001"]}).to_parquet(thresholds_path)

    object_db = {
        "obj_train": {
            "sample_kind": "pure",
            "object_nut_type": "peanut",
            "batch": 1,
        },
        "obj_test": {
            "sample_kind": "pure",
            "object_nut_type": "almond",
            "batch": 4,
        },
    }
    projection_filters = build_pure_test_projection_filters(["almond", "peanut"], [4])

    guardrails = build_pure_test_guardrails(
        train_batches=[1, 2, 3],
        test_batches=[4],
        train_filters={"sample_kind": ["pure"], "object_nut_type": "peanut", "batch": [1, 2, 3]},
        projection_filters=projection_filters,
        thresholds_path=thresholds_path,
        object_db=object_db,
        target_class="peanut",
        reference_classes=["almond", "peanut"],
    )

    assert guardrails["passed"].all()
    validate_pure_test_guardrails(guardrails)


def test_pure_test_guardrails_reject_noncanonical_test_batch(tmp_path):
    guardrails = build_pure_test_guardrails(
        train_batches=[1, 2, 3],
        test_batches=[3],
        train_filters={"batch": [1, 2, 3]},
        projection_filters={"sample_kind": ["pure"], "batch": [3]},
        thresholds_path=tmp_path / "missing.parquet",
    )

    with pytest.raises(RuntimeError, match="Pure-test guardrail failure"):
        validate_pure_test_guardrails(guardrails)


def test_select_pure_test_candidate_panel_restores_04c_configs_and_thresholds():
    candidate_panel = pd.DataFrame(
        [
            _candidate_row("sel_001", matrix_family="object_matrix"),
            _candidate_row("sel_002", matrix_family="pixel_matrix"),
            _candidate_row("sel_003", matrix_family="object_matrix"),
        ]
    )
    track_scoring_flags = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_002",
                "selection_track": "pixel_matrix_2way",
                "review_rank_in_track": 1,
                "review_flag_count": 0,
                "robustness_score": 0.8,
            },
            {
                "selected_config_id": "sel_001",
                "selection_track": "object_matrix_2way",
                "review_rank_in_track": 1,
                "review_flag_count": 0,
                "robustness_score": 0.9,
            },
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "three_way_lower_threshold": 0.30,
                "three_way_upper_threshold": 0.75,
            },
            {
                "selected_config_id": "sel_002",
                "three_way_lower_threshold": 0.25,
                "three_way_upper_threshold": 0.80,
            },
        ]
    )

    panel, selected_thresholds = select_pure_test_candidate_panel(
        candidate_panel,
        track_scoring_flags,
        thresholds,
    )

    assert panel["selected_config_id"].tolist() == ["sel_002", "sel_001"]
    assert panel["matrix_family"].tolist() == ["pixel_matrix", "object_matrix"]
    assert set(selected_thresholds["selected_config_id"]) == {"sel_001", "sel_002"}


def test_select_pure_test_candidate_panel_requires_fixed_3way_thresholds():
    candidate_panel = pd.DataFrame([_candidate_row("sel_001")])
    track_scoring_flags = pd.DataFrame(
        [{"selected_config_id": "sel_001", "selection_track": "object_matrix_3way"}]
    )
    thresholds = pd.DataFrame(
        columns=["selected_config_id", "three_way_lower_threshold", "three_way_upper_threshold"]
    )

    with pytest.raises(RuntimeError, match="Missing fixed 3-way validation thresholds"):
        select_pure_test_candidate_panel(candidate_panel, track_scoring_flags, thresholds)


def test_build_3way_outputs_applies_fixed_thresholds_by_config():
    object_df = pd.DataFrame(
        [
            {
                **_candidate_row("sel_001"),
                "source_image": "batch4_img1",
                "peanut_pixel_ratio": 0.90,
                "true_peanut_object": "peanut",
            },
            {
                **_candidate_row("sel_001"),
                "source_image": "batch4_img1",
                "peanut_pixel_ratio": 0.20,
                "true_peanut_object": "almond",
            },
            {
                **_candidate_row("sel_001"),
                "source_image": "batch4_img2",
                "peanut_pixel_ratio": 0.50,
                "true_peanut_object": "almond",
            },
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "three_way_lower_threshold": 0.30,
                "three_way_upper_threshold": 0.80,
            }
        ]
    )

    metrics, image_metrics, objects_3way = build_3way_outputs(
        object_df,
        thresholds,
        target_class="peanut",
        non_target_label="almond",
        evaluation_stage="pure_test_batch_4",
    )

    assert metrics.loc[0, "three_way_lower_threshold"] == 0.30
    assert metrics.loc[0, "three_way_upper_threshold"] == 0.80
    assert metrics.loc[0, "target_miss_rate"] == 0.0
    assert metrics.loc[0, "uncertain_rate"] == pytest.approx(1 / 3)
    assert set(objects_3way["decision_3way"]) == {"peanut", "almond", "uncertain"}
    assert set(image_metrics["source_image"]) == {"batch4_img1", "batch4_img2"}
