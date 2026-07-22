import numpy as np
import pandas as pd
import pytest

from src.workflows.simca_robustness import (
    add_simca_robustness_scores,
    build_ablation_diagnostics,
    build_border_core_diagnostics,
    build_border_core_skip_table,
    build_duplicated_candidate_review,
    build_pareto_diagnostics,
    build_random_state_stability_panel,
    select_track_primary_or_available_metrics,
    summarize_duplicated_candidate_review,
    summarize_random_state_stability_metrics,
    validate_no_pure_test_inputs,
)


def _metric_row(**overrides):
    row = {
        "selected_config_id": "cfg_a",
        "candidate_id": "cand_a",
        "selection_track": "object_matrix_2way",
        "matrix_family": "object_matrix",
        "matrix_method": "object_mean",
        "training_matrix_id": "object_mean",
        "decision_mode": "2way",
        "evaluation_stage": "validation_batch_3_refit",
        "metric_level": "object",
        "target_class": "peanut",
        "non_target_label": "almond",
        "preprocessing": "snv",
        "preprocessing_steps": "snv",
        "rule_variant": "simple_emp_cv",
        "rule_for_refit": "simple_emp_cv",
        "limit_source": "empirical_cv",
        "n_components": 4,
        "alpha": 0.01,
        "object_threshold": 0.75,
        "sg_window_length": 11,
        "sg_polyorder": 2,
        "position_dilation_radius": 3,
        "n": 20,
        "tp": 9,
        "fn": 1,
        "fp": 2,
        "tn": 8,
        "target_sensitivity": 0.90,
        "non_target_specificity": 0.80,
        "balanced_accuracy": 0.85,
        "accuracy": 0.85,
        "precision": 0.82,
        "f1_score": 0.86,
        "fn_rate": 0.10,
        "fp_rate": 0.20,
    }
    row.update(overrides)
    return row


def test_validate_no_pure_test_inputs_rejects_forbidden_stage():
    df = pd.DataFrame([_metric_row(evaluation_stage="pure_test_batch_4")])

    with pytest.raises(ValueError, match="must not consume pure-test"):
        validate_no_pure_test_inputs(df)


def test_select_track_primary_falls_back_to_secondary_when_primary_absent():
    df = pd.DataFrame(
        [
            _metric_row(
                selected_config_id="cfg_3way",
                selection_track="pixel_matrix_3way",
                matrix_family="pixel_matrix",
                matrix_method="balanced_pixels",
                decision_mode="3way",
                metric_level="object",
                target_miss_rate=0.0,
                non_target_false_accept_rate=0.1,
                uncertain_rate=0.2,
                coverage_rate=0.8,
                screening_sensitivity=1.0,
                decided_balanced_accuracy=0.9,
            )
        ]
    )

    selected = select_track_primary_or_available_metrics(df)

    assert len(selected) == 1
    assert selected.loc[0, "metric_level"] == "object"


def test_add_simca_robustness_scores_adds_flags_and_track_rank():
    df = pd.DataFrame(
        [
            _metric_row(selected_config_id="bad_2way", fn_rate=0.20, fp_rate=0.30, balanced_accuracy=0.70),
            _metric_row(selected_config_id="good_2way", fn_rate=0.00, fp_rate=0.05, balanced_accuracy=0.95),
            _metric_row(
                selected_config_id="uncertain_3way",
                selection_track="object_matrix_3way",
                decision_mode="3way",
                target_miss_rate=0.00,
                non_target_false_accept_rate=0.05,
                uncertain_rate=0.40,
                coverage_rate=0.60,
                screening_sensitivity=1.0,
                decided_balanced_accuracy=0.90,
            ),
        ]
    )

    scored = add_simca_robustness_scores(df)

    bad_flags = scored.loc[scored["selected_config_id"].eq("bad_2way"), "robustness_flags"].iloc[0]
    three_way_flags = scored.loc[scored["selected_config_id"].eq("uncertain_3way"), "robustness_flags"].iloc[0]
    good_rank = scored.loc[scored["selected_config_id"].eq("good_2way"), "robustness_rank_in_track"].iloc[0]

    assert "high_fn_rate" in bad_flags
    assert "high_fp_rate" in bad_flags
    assert "low_balanced_accuracy" in bad_flags
    assert "high_uncertain_rate" in three_way_flags
    assert int(good_rank) == 1


def test_build_pareto_diagnostics_marks_dominated_rows():
    df = pd.DataFrame(
        [
            _metric_row(selected_config_id="front_a", fn_rate=0.05, fp_rate=0.05, balanced_accuracy=0.95),
            _metric_row(selected_config_id="dominated", fn_rate=0.20, fp_rate=0.20, balanced_accuracy=0.70),
            _metric_row(selected_config_id="tradeoff", fn_rate=0.00, fp_rate=0.40, balanced_accuracy=0.80),
        ]
    )

    front, annotated, audit = build_pareto_diagnostics(df, decision_mode="2way")

    assert set(front["selected_config_id"]) == {"front_a", "tradeoff"}
    assert not bool(annotated.loc[annotated["selected_config_id"].eq("dominated"), "is_pareto_2way"].iloc[0])
    assert int(audit["n_dominated"].sum()) == 1


def test_build_ablation_diagnostics_summarizes_factor_values():
    df = pd.DataFrame(
        [
            _metric_row(selected_config_id="cfg_4", n_components=4, fn_rate=0.0, fp_rate=0.1, balanced_accuracy=0.95),
            _metric_row(selected_config_id="cfg_8", n_components=8, fn_rate=0.1, fp_rate=0.2, balanced_accuracy=0.85),
        ]
    )

    ablation = build_ablation_diagnostics(df, factor_cols=("n_components",))

    assert set(ablation["factor"]) == {"n_components"}
    assert set(ablation["factor_value"]) == {"4", "8"}
    assert "robustness_score_mean" in ablation.columns


def test_build_random_state_stability_panel_prefers_balanced_pixels_when_limited():
    candidate_panel = pd.DataFrame(
        [
            _metric_row(selected_config_id="object_cfg", matrix_method="object_mean"),
            _metric_row(
                selected_config_id="balanced_cfg",
                matrix_family="pixel_matrix",
                matrix_method="balanced_pixels",
                training_matrix_id="balanced_pixel_random_m40",
            ),
        ]
    )
    metrics = pd.DataFrame(
        [
            _metric_row(selected_config_id="object_cfg", matrix_method="object_mean", robustness_score=10.0),
            _metric_row(
                selected_config_id="balanced_cfg",
                matrix_family="pixel_matrix",
                matrix_method="balanced_pixels",
                training_matrix_id="balanced_pixel_random_m40",
                robustness_score=0.0,
            ),
        ]
    )

    panel = build_random_state_stability_panel(
        candidate_panel,
        metrics,
        max_per_track=1,
        prefer_balanced_pixels=True,
    )

    assert panel.loc[0, "selected_config_id"] == "balanced_cfg"
    assert panel.loc[0, "stability_panel_reason"] == "top_robustness_candidates_per_track"


def test_summarize_random_state_stability_metrics_flags_unstable_metrics():
    stability_metrics = pd.DataFrame(
        [
            _metric_row(selected_config_id="cfg_a", random_state=0, fn_rate=0.00, fp_rate=0.10, balanced_accuracy=0.95),
            _metric_row(selected_config_id="cfg_a", random_state=1, fn_rate=0.10, fp_rate=0.30, balanced_accuracy=0.80),
        ]
    )

    summary = summarize_random_state_stability_metrics(stability_metrics)

    assert len(summary) == 1
    assert "unstable_fn_rate" in summary.loc[0, "stability_flags"]
    assert "unstable_fp_rate" in summary.loc[0, "stability_flags"]


def test_duplicated_candidate_review_uses_optional_refit_status():
    groups = pd.DataFrame(
        [
            {
                "metric_equivalence_group_id": "grp_1",
                "varied_parameter_group": "n_components",
                "n_metric_equivalent_candidates": 2,
                "kept_candidate_id": "kept",
                "dropped_candidate_ids": "dropped",
            }
        ]
    )
    dropped = pd.DataFrame(
        [
            {
                "metric_equivalence_group_id": "grp_1",
                "candidate_id": "dropped",
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "metric_equivalence_group_id": "grp_1",
                "n_refit_candidates": 2,
                "all_post_refit_metrics_equal": True,
                "all_post_refit_metrics_match_pre_refit": True,
            }
        ]
    )

    review = build_duplicated_candidate_review(groups, dropped, comparison)
    summary = summarize_duplicated_candidate_review(review)

    assert review.loc[0, "duplicated_refit_status"] == "verified_equal"
    assert not bool(review.loc[0, "needs_duplicate_manual_review"])
    assert int(summary.loc[0, "n_dropped_candidates"]) == 1


def test_duplicated_candidate_review_marks_not_run_without_comparison():
    groups = pd.DataFrame(
        [
            {
                "metric_equivalence_group_id": "grp_1",
                "varied_parameter_group": "rule",
                "n_metric_equivalent_candidates": 2,
            }
        ]
    )

    review = build_duplicated_candidate_review(groups)

    assert review.loc[0, "duplicated_refit_status"] == "not_run"
    assert bool(review.loc[0, "needs_duplicate_manual_review"])


def test_border_core_skip_table_documents_required_04c_setting():
    skip = build_border_core_skip_table("missing pixels", pixel_batch_dir="results/04C/pixels")

    assert skip.loc[0, "border_core_status"] == "skipped"
    assert skip.loc[0, "required_04c_setting"] == "SAVE_BATCH_PIXEL_TABLES=True"


def test_build_border_core_diagnostics_on_mini_fixture():
    rows = []
    object_db = {}
    for object_id, start_col, is_target in [("o1", 0, True), ("o2", 4, False)]:
        object_db[object_id] = {
            "bbox": (0, start_col, 3, start_col + 3),
            "mask": np.ones((3, 3), dtype=bool),
            "centroid": (1, start_col + 1),
            "area_pixels": 9,
            "batch": 3,
            "sample_kind": "pure",
            "object_nut_type": "peanut" if is_target else "almond",
        }
        for row in range(3):
            for col in range(start_col, start_col + 3):
                rows.append(
                    {
                        "selected_config_id": "cfg_a",
                        "candidate_id": "cand_a",
                        "selection_track": "object_matrix_2way",
                        "matrix_family": "object_matrix",
                        "matrix_method": "object_mean",
                        "preprocessing": "snv",
                        "rule_variant": "simple_emp_cv",
                        "n_components": 4,
                        "object_id": object_id,
                        "source_image": "img_1",
                        "row": row,
                        "col": col,
                        "predicted_peanut_pixel": is_target,
                        "true_peanut_pixel": is_target,
                        "truth_available": True,
                        "target_class": "peanut",
                        "non_target_label": "almond",
                        "object_threshold": 0.75,
                    }
                )
    pixel_df = pd.DataFrame(rows)

    diagnostics, status = build_border_core_diagnostics(
        pixel_df,
        object_db,
        border_widths=(0, 1),
        object_thresholds=(0.75,),
        min_core_pixels=1,
    )

    assert status.empty
    assert set(diagnostics["zone"]) == {"all_pixels", "core_without_border"}
    assert diagnostics["balanced_accuracy"].notna().all()
