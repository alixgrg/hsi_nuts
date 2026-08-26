import json
from pathlib import Path

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
    build_planned_contrast_results,
    build_random_state_stability_panel,
    build_risk_coverage_curves,
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


def _selection_unit_row(**overrides):
    row = {
        "evaluation_track": "object_train__object_projection__2way",
        "track_id": "E1",
        "calibration_id": "cal_a",
        "selection_unit_id": "cal_a",
        "decision_mode": "2way",
        "projection_level": "object_projection",
        "matrix_method": "object_mean",
        "balanced_pixel_strategy": "not_applicable",
        "target_miss_rate": 0.05,
        "false_accept_rate": 0.10,
        "balanced_accuracy": 0.925,
        "safety_target_miss_rate": 0.05,
        "safety_false_accept_rate": 0.10,
        "safety_balanced_accuracy": 0.925,
        "pareto_pool_status": "eligible",
        "pareto_pool_reason": "",
        "all_member_seeds_calculable": True,
        "eligibility_status": "eligible",
    }
    row.update(overrides)
    return row


def test_validate_no_pure_test_inputs_rejects_forbidden_stage():
    df = pd.DataFrame([_metric_row(evaluation_stage="pure_test_batch_4")])

    with pytest.raises(ValueError, match="must not consume pure-test"):
        validate_no_pure_test_inputs(df)


def test_notebook05_orchestrates_frozen_upstream_outputs_without_legacy_scoring():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "05_simca_validation_robustness.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "build_locked_validation_candidate_pool" in source
    assert "projection_eligibility" in source
    assert "ablation_plan" in source
    assert "validation_metrics" in source
    assert "select_final_models_lexicographic" in source
    assert "SIMCA_ROBUSTNESS_PERSISTED_CANDIDATE_COLUMNS" in source
    assert "add_simca_robustness_scores" not in source
    assert "select_top_with_diversity" not in source
    assert '"batch4_loaded": False' in source


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


def test_add_simca_robustness_scores_is_flags_only_without_rank_or_score():
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
    assert "high_fn_rate" in bad_flags
    assert "high_fp_rate" in bad_flags
    assert "low_balanced_accuracy" in bad_flags
    assert "high_uncertain_rate" in three_way_flags
    assert "robustness_score" not in scored
    assert "robustness_rank_in_track" not in scored


def test_build_pareto_diagnostics_marks_dominated_rows():
    df = pd.DataFrame(
        [
            _selection_unit_row(calibration_id="front_a", selection_unit_id="front_a", target_miss_rate=0.05, false_accept_rate=0.05),
            _selection_unit_row(calibration_id="dominated", selection_unit_id="dominated", target_miss_rate=0.20, false_accept_rate=0.20),
            _selection_unit_row(calibration_id="tradeoff", selection_unit_id="tradeoff", target_miss_rate=0.00, false_accept_rate=0.40),
        ]
    )

    front, annotated, audit = build_pareto_diagnostics(df)

    assert set(front["calibration_id"]) == {"front_a", "tradeoff"}
    assert not bool(annotated.loc[annotated["calibration_id"].eq("dominated"), "is_protocol_pareto"].iloc[0])
    assert int((audit["n_protocol_eligible"] - audit["n_protocol_pareto"]).sum()) == 1


def test_build_ablation_diagnostics_uses_only_frozen_paired_plan():
    df = pd.DataFrame(
        [
            {
                **_selection_unit_row(calibration_id="cfg_4"),
                "aggregation_level": "overall",
                "status": "calculable",
                "random_state": 0,
                "target_miss_rate": 0.0,
                "false_accept_rate": 0.1,
            },
            {
                **_selection_unit_row(calibration_id="cfg_8"),
                "aggregation_level": "overall",
                "status": "calculable",
                "random_state": 0,
                "target_miss_rate": 0.1,
                "false_accept_rate": 0.2,
            },
        ]
    )
    plan = pd.DataFrame(
        [
            {
                "ablation_id": "abl_1",
                "evaluation_track": "object_train__object_projection__2way",
                "track_id": "E1",
                "contrast_type": "paired_variant",
                "factor": "n_components",
                "reference_config_id": "cfg_4",
                "ablated_config_id": "cfg_8",
                "reference_level": "4",
                "ablated_level": "8",
                "eligibility_status": "eligible",
                "plan_status": "planned",
                "unsupported_reason": "",
                "registration_status": "frozen",
                "preregistered": True,
            }
        ]
    )

    ablation = build_ablation_diagnostics(
        df,
        ablation_plan_df=plan,
        metric_cols=("target_miss_rate", "false_accept_rate"),
    )

    assert set(ablation["ablation_id"]) == {"abl_1"}
    assert set(ablation["effect_status"]) == {"estimated_paired"}
    assert set(ablation["metric"]) == {"target_miss_rate", "false_accept_rate"}
    assert "robustness_score" not in ablation.columns


def test_threshold_ablation_reuses_saved_margins_without_refit():
    metrics = pd.DataFrame(
        [
            {
                **_selection_unit_row(calibration_id="cfg_threshold"),
                "aggregation_level": "overall",
                "status": "calculable",
                "random_state": 0,
                "target_miss_rate": 0.0,
            },
            {
                **_selection_unit_row(calibration_id="cfg_threshold"),
                "aggregation_level": "source_image",
                "status": "calculable",
                "random_state": 0,
                "group_id": "img_target",
                "target_miss_rate": 0.0,
            },
        ]
    )
    plan = pd.DataFrame(
        [
            {
                "ablation_id": "threshold_low",
                "evaluation_track": "object_train__object_projection__2way",
                "track_id": "E1",
                "contrast_type": "threshold_sensitivity",
                "factor": "direct_2way_threshold",
                "reference_config_id": "cfg_threshold",
                "ablated_config_id": "",
                "reference_level": "0.0",
                "ablated_level": -0.2,
                "eligibility_status": "eligible",
                "plan_status": "planned",
                "unsupported_reason": "",
                "registration_status": "frozen",
                "preregistered": True,
            },
            {
                "ablation_id": "threshold_high",
                "evaluation_track": "object_train__object_projection__2way",
                "track_id": "E1",
                "contrast_type": "threshold_sensitivity",
                "factor": "direct_2way_threshold",
                "reference_config_id": "cfg_threshold",
                "ablated_config_id": "",
                "reference_level": "0.0",
                "ablated_level": 0.2,
                "eligibility_status": "eligible",
                "plan_status": "planned",
                "unsupported_reason": "",
                "registration_status": "frozen",
                "preregistered": True,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "calibration_id": "cfg_threshold",
                "projection_config_id": "projection_0",
                "projection_level": "object_projection",
                "decision_mode": "2way",
                "direct_2way_threshold": 0.0,
                "three_way_lower_threshold": np.nan,
                "three_way_upper_threshold": np.nan,
            }
        ]
    )
    predictions = pd.DataFrame(
        {
            "projection_config_id": ["projection_0"] * 4,
            "source_image": ["img_target", "img_target", "img_non_target", "img_non_target"],
            "object_id": ["t1", "t2", "n1", "n2"],
            "truth": [True, True, False, False],
            "simca_margin": [0.3, 0.1, -0.1, -0.3],
        }
    )

    effects = build_ablation_diagnostics(
        metrics,
        ablation_plan_df=plan,
        candidate_pool_df=candidates,
        object_predictions_df=predictions,
        pixel_predictions_df=predictions.iloc[0:0].copy(),
        metric_cols=("target_miss_rate",),
    )

    assert effects["effect_status"].eq("estimated_paired").all()
    values = effects.set_index("ablation_id")["ablated_value"]
    assert values["threshold_low"] == pytest.approx(0.0)
    assert values["threshold_high"] == pytest.approx(0.5)


def test_build_random_state_stability_panel_keeps_all_stochastic_pareto_units():
    candidate_panel = pd.DataFrame(
        [
            _selection_unit_row(calibration_id="object_cfg", matrix_method="object_mean"),
            _selection_unit_row(
                calibration_id="balanced_cfg",
                matrix_method="balanced_pixels",
                balanced_pixel_strategy="random",
            ),
        ]
    )
    metrics = candidate_panel.copy()
    metrics["is_protocol_pareto"] = True

    panel = build_random_state_stability_panel(
        candidate_panel,
        metrics,
    )

    assert panel["calibration_id"].tolist() == ["balanced_cfg"]
    assert panel.loc[0, "stability_panel_reason"] == "all_protocol_pareto_stochastic_units"
    with pytest.raises(ValueError, match="forbids a per-track"):
        build_random_state_stability_panel(candidate_panel, metrics, max_per_track=1)


def test_summarize_random_state_stability_metrics_flags_unstable_metrics():
    stability_metrics = pd.DataFrame(
        [
            {
                **_selection_unit_row(
                    calibration_id="cfg_a",
                    matrix_method="balanced_pixels",
                    balanced_pixel_strategy="random",
                ),
                "aggregation_level": "overall",
                "random_state": 0,
                "target_miss_rate": 0.00,
                "false_accept_rate": 0.10,
                "balanced_accuracy": 0.95,
            },
            {
                **_selection_unit_row(
                    calibration_id="cfg_a",
                    matrix_method="balanced_pixels",
                    balanced_pixel_strategy="random",
                ),
                "aggregation_level": "overall",
                "random_state": 1,
                "target_miss_rate": 0.10,
                "false_accept_rate": 0.30,
                "balanced_accuracy": 0.80,
            },
        ]
    )

    summary = summarize_random_state_stability_metrics(
        stability_metrics, expected_random_states=(0, 1)
    )

    assert len(summary) == 1
    assert "unstable_target_miss" in summary.loc[0, "stability_flags"]
    assert "unstable_false_accept" in summary.loc[0, "stability_flags"]
    assert summary.loc[0, "stability_status"] == "unstable"


def test_deterministic_stability_is_explicitly_not_applicable():
    metrics = pd.DataFrame(
        [
            {
                **_selection_unit_row(calibration_id="deterministic"),
                "aggregation_level": "overall",
                "random_state": 0,
            }
        ]
    )

    summary = summarize_random_state_stability_metrics(metrics)

    assert summary.loc[0, "stability_status"] == "not_applicable_deterministic"


def test_risk_coverage_supports_frozen_h4_point_contrasts():
    selected = pd.DataFrame(
        [
            {
                "evaluation_track": "object_train__object_projection__2way",
                "track_id": "E1",
                "calibration_id": "cal_2way",
                "decision_mode": "2way",
            },
            {
                "evaluation_track": "object_train__object_projection__3way",
                "track_id": "E2",
                "calibration_id": "cal_3way",
                "decision_mode": "3way",
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "calibration_id": "cal_2way",
                "projection_config_id": "projection_2way",
                "projection_level": "object_projection",
                "decision_mode": "2way",
                "direct_2way_threshold": 0.0,
                "three_way_lower_threshold": np.nan,
                "three_way_upper_threshold": np.nan,
                "random_state": 0,
            },
            {
                "calibration_id": "cal_3way",
                "projection_config_id": "projection_3way",
                "projection_level": "object_projection",
                "decision_mode": "3way",
                "direct_2way_threshold": np.nan,
                "three_way_lower_threshold": -0.2,
                "three_way_upper_threshold": 0.2,
                "random_state": 0,
            },
        ]
    )
    base_predictions = pd.DataFrame(
        {
            "source_image": ["target", "target", "non_target", "non_target"],
            "object_id": ["t1", "t2", "n1", "n2"],
            "truth": [True, True, False, False],
            "simca_margin": [0.4, 0.1, -0.1, -0.4],
        }
    )
    predictions = pd.concat(
        [
            base_predictions.assign(projection_config_id="projection_2way"),
            base_predictions.assign(projection_config_id="projection_3way"),
        ],
        ignore_index=True,
    )
    curves = build_risk_coverage_curves(
        selected,
        candidates,
        predictions,
        predictions.iloc[0:0].copy(),
        coverage_grid=(0.5, 0.75, 1.0),
    )
    contrasts = pd.DataFrame(
        [
            {
                "contrast_id": "H4_test",
                "hypothesis_id": "H4",
                "left_track": "object_train__object_projection__3way",
                "right_track": "object_train__object_projection__2way",
                "metric": "selective_risk_auc",
            }
        ]
    )

    results = build_planned_contrast_results(contrasts, selected, curves)

    assert set(curves["decision_mode"]) == {"2way", "3way"}
    assert curves["n_seeds"].eq(1).all()
    assert curves["selective_risk_auc"].notna().all()
    assert np.isfinite(results.loc[0, "effect_estimate"])
    assert results.loc[0, "inference_status"] == "not_estimable_insufficient_independent_images"


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
