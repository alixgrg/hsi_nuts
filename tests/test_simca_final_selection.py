import pandas as pd
import pytest

from src.workflows.simca_final_selection import (
    assign_pareto_tiers,
    build_final_selection_guardrails,
    build_final_selection_pool,
    apply_rate_threshold_filter_by_track,
    pareto_front_mask,
    select_final_models_lexicographic,
    select_final_models_by_track,
    select_top_with_diversity,
    validate_final_selection_guardrails,
)


def _metric_row(
    selected_config_id,
    selection_track="object_matrix_2way",
    preprocessing="snv",
    rule_for_refit="simple_emp_cv",
    strategy="not_applicable",
    fn_rate=0.0,
    fp_rate=0.05,
    balanced_accuracy=0.975,
):
    matrix_family, decision_mode = selection_track.rsplit("_", 1)
    matrix_method = "object_mean" if matrix_family == "object_matrix" else "balanced_pixels"
    metric_level = "pixel" if selection_track == "pixel_matrix_2way" else "object"
    row = {
        "selected_config_id": selected_config_id,
        "candidate_id": f"cand_{selected_config_id}",
        "candidate_sources": "grid",
        "selection_track": selection_track,
        "matrix_family": matrix_family,
        "decision_mode": decision_mode,
        "metric_level": metric_level,
        "target_class": "peanut",
        "non_target_label": "almond",
        "model_family": "empirical_cv_rule",
        "matrix_method": matrix_method,
        "training_matrix_id": matrix_method,
        "m_effective": 40 if matrix_family == "pixel_matrix" else pd.NA,
        "balanced_pixel_strategy_effective": strategy,
        "preprocessing": preprocessing,
        "preprocessing_steps": preprocessing,
        "rule_variant": rule_for_refit,
        "rule_for_refit": rule_for_refit,
        "n_components": 4,
        "alpha": 0.01,
        "object_threshold": 0.75,
        "n": 20,
        "tp": 10,
        "fn": int(round(fn_rate * 10)),
        "fp": int(round(fp_rate * 10)),
        "tn": 10,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "balanced_accuracy": balanced_accuracy,
    }
    if decision_mode == "3way":
        row.update(
            {
                "target_miss_rate": fn_rate,
                "non_target_false_accept_rate": fp_rate,
                "uncertain_rate": 0.10,
                "coverage_rate": 0.90,
                "decided_balanced_accuracy": balanced_accuracy,
                "three_way_lower_threshold": 0.30,
                "three_way_upper_threshold": 0.80,
            }
        )
    return row


def _review_row(selected_config_id, selection_track="object_matrix_2way", flags=""):
    return {
        "selected_config_id": selected_config_id,
        "candidate_id": f"cand_{selected_config_id}",
        "selection_track": selection_track,
        "matrix_family": selection_track.rsplit("_", 1)[0],
        "decision_mode": selection_track.rsplit("_", 1)[1],
        "metric_level": "object",
        "target_class": "peanut",
        "non_target_label": "almond",
        "review_flags": flags,
        "robustness_flags": "",
        "stability_flags": "",
        "fn_rate": 0.0,
        "fp_rate": 0.05,
        "balanced_accuracy": 0.975,
    }


def test_final_selection_guardrails_require_pure_test_guardrails_passed():
    guardrails = build_final_selection_guardrails(
        track_scoring_flags_df=pd.DataFrame([_review_row("sel_001")]),
        pure_test_metrics_df=pd.DataFrame([_metric_row("sel_001")]),
        pure_test_guardrails_df=pd.DataFrame(
            [{"check_name": "pure_test", "passed": False, "status": "failed"}]
        ),
        candidate_panel_df=pd.DataFrame([_metric_row("sel_001")]),
        expected_tracks=["object_matrix_2way"],
    )

    with pytest.raises(RuntimeError, match="guardrail failure"):
        validate_final_selection_guardrails(guardrails)


def test_pareto_front_mask_uses_rates_without_score():
    df = pd.DataFrame(
        {
            "selected_config_id": ["a", "b", "c", "d"],
            "fn_rate": [0.00, 0.05, 0.00, 0.10],
            "fp_rate": [0.20, 0.05, 0.30, 0.30],
        }
    )

    mask = pareto_front_mask(df, ["fn_rate", "fp_rate"])

    assert df.loc[mask, "selected_config_id"].tolist() == ["a", "b"]


def test_lexicographic_selection_prioritizes_target_miss_without_score_or_diversity():
    candidates = pd.DataFrame(
        [
            {
                "calibration_id": "safer",
                "evaluation_track": "object_train__object_projection__2way",
                "track_id": "E1",
                "decision_mode": "2way",
                "final_candidate_status": "eligible_for_final_selection",
                "eligibility_status": "eligible",
                "safety_target_miss_rate": 0.00,
                "worst_image_target_miss_rate": 0.00,
                "worst_seed_target_miss_rate": 0.00,
                "safety_false_accept_rate": 0.20,
                "safety_uncertain_rate": 0.00,
                "stability_status": "not_applicable_deterministic",
                "preprocessing_steps": "snv",
                "n_components": 5,
            },
            {
                "calibration_id": "more_specific",
                "evaluation_track": "object_train__object_projection__2way",
                "track_id": "E1",
                "decision_mode": "2way",
                "final_candidate_status": "eligible_for_final_selection",
                "eligibility_status": "eligible",
                "safety_target_miss_rate": 0.03,
                "worst_image_target_miss_rate": 0.03,
                "worst_seed_target_miss_rate": 0.03,
                "safety_false_accept_rate": 0.00,
                "safety_uncertain_rate": 0.00,
                "stability_status": "not_applicable_deterministic",
                "preprocessing_steps": "snv",
                "n_components": 4,
            },
        ]
    )

    selected, locked, protocol = select_final_models_lexicographic(
        candidates,
        expected_tracks=("object_train__object_projection__2way",),
    )

    assert selected["calibration_id"].tolist() == ["safer"]
    assert set(locked["calibration_id"]) == {"safer"}
    assert protocol.loc[0, "selection_method"] == "hard_guardrails_then_lexicographic"
    assert not any("score" in column for column in selected.columns)


def test_assign_pareto_tiers_repeats_non_dominated_fronts():
    df = pd.DataFrame(
        {
            "selected_config_id": ["a", "b", "c"],
            "fn_rate": [0.00, 0.05, 0.10],
            "fp_rate": [0.20, 0.10, 0.30],
        }
    )

    out = assign_pareto_tiers(df, ["fn_rate", "fp_rate"])

    assert out.loc[out["selected_config_id"].eq("a"), "pareto_tier"].iloc[0] == 1
    assert out.loc[out["selected_config_id"].eq("b"), "pareto_tier"].iloc[0] == 1
    assert out.loc[out["selected_config_id"].eq("c"), "pareto_tier"].iloc[0] == 2


def test_build_final_selection_pool_does_not_create_score_based_selection_columns():
    pure_metrics = pd.DataFrame(
        [
            _metric_row("good", fn_rate=0.0),
            _metric_row("flagged", fn_rate=0.1),
        ]
    )
    review = pd.DataFrame(
        [
            _review_row("good"),
            _review_row("flagged", flags="high_fn_rate"),
        ]
    )

    pool = build_final_selection_pool(
        pure_test_metrics_df=pure_metrics,
        track_scoring_flags_df=review,
        candidate_panel_df=pure_metrics,
        apply_previous_flag_filter=False,
    )

    assert "selection_score" not in pool.columns
    assert "eligibility_status" not in pool.columns
    assert pool["preselection_status"].tolist() == ["candidate", "candidate"]
    assert pool.loc[pool["selected_config_id"].eq("flagged"), "previous_flags"].iloc[0] == "high_fn_rate"


def test_build_final_selection_pool_can_optionally_filter_previous_flags():
    pure_metrics = pd.DataFrame([_metric_row("flagged", fn_rate=0.1)])
    review = pd.DataFrame([_review_row("flagged", flags="high_fn_rate")])

    pool = build_final_selection_pool(
        pure_test_metrics_df=pure_metrics,
        track_scoring_flags_df=review,
        candidate_panel_df=pure_metrics,
        apply_previous_flag_filter=True,
        previous_flags_to_filter=("high_fn_rate",),
    )

    assert pool.loc[0, "preselection_status"] == "filtered"
    assert pool.loc[0, "filter_reason"] == "previous_flag_filter"


def test_rate_threshold_filter_is_strict_and_skips_none_dimensions():
    pool = pd.DataFrame(
        [
            {
                "selected_config_id": "good",
                "selection_track": "object_matrix_2way",
                "preselection_status": "candidate",
                "fn_rate": 0.10,
                "fp_rate": 0.10,
            },
            {
                "selected_config_id": "equal_fn",
                "selection_track": "object_matrix_2way",
                "preselection_status": "candidate",
                "fn_rate": 0.20,
                "fp_rate": 0.01,
            },
            {
                "selected_config_id": "equal_fp",
                "selection_track": "object_matrix_2way",
                "preselection_status": "candidate",
                "fn_rate": 0.01,
                "fp_rate": 0.20,
            },
        ]
    )

    out = apply_rate_threshold_filter_by_track(
        pool,
        fn_rate_max=0.20,
        fp_rate_max=0.20,
        uncertain_rate_max=None,
        track_order=["object_matrix_2way"],
    )

    assert out.loc[out["selected_config_id"].eq("good"), "rate_threshold_passed"].iloc[0]
    assert not out.loc[out["selected_config_id"].eq("equal_fn"), "rate_threshold_passed"].iloc[0]
    assert not out.loc[out["selected_config_id"].eq("equal_fp"), "rate_threshold_passed"].iloc[0]
    assert out.loc[out["selected_config_id"].eq("equal_fn"), "rate_threshold_reason"].iloc[0] == "fn_rate>=0.2"
    assert out.loc[out["selected_config_id"].eq("equal_fp"), "rate_threshold_reason"].iloc[0] == "fp_rate>=0.2"


def test_select_top_with_diversity_avoids_single_preprocessing_when_possible():
    df = pd.DataFrame(
        [
            _metric_row("a", preprocessing="snv", balanced_accuracy=0.99),
            _metric_row("b", preprocessing="snv", balanced_accuracy=0.98),
            _metric_row("c", preprocessing="msc", balanced_accuracy=0.90),
        ]
    )
    df["pareto_tier"] = [1, 1, 2]
    df["pareto_rank_in_track"] = [1, 2, 3]

    selected = select_top_with_diversity(
        df,
        top_n=2,
        diversity_columns=["preprocessing"],
    )

    assert selected["selected_config_id"].tolist() == ["a", "c"]


def test_select_final_models_by_track_can_deduplicate_across_tracks_and_refill():
    pure_metrics = pd.DataFrame(
        [
            _metric_row("shared", selection_track="object_matrix_2way", fn_rate=0.0, fp_rate=0.01),
            _metric_row("shared", selection_track="object_matrix_3way", fn_rate=0.0, fp_rate=0.01),
            # Incomparable with ``shared``: worse target miss, better false accept.
            _metric_row("replacement", selection_track="object_matrix_3way", fn_rate=0.02, fp_rate=0.00),
        ]
    )
    review = pd.DataFrame(
        [
            _review_row("shared", selection_track="object_matrix_2way"),
            _review_row("shared", selection_track="object_matrix_3way"),
            _review_row("replacement", selection_track="object_matrix_3way"),
        ]
    )
    pool = build_final_selection_pool(
        pure_test_metrics_df=pure_metrics,
        track_scoring_flags_df=review,
        candidate_panel_df=pure_metrics,
    )

    selected, annotated_pool, summary = select_final_models_by_track(
        pool,
        top_n_per_track=1,
        apply_diversity=False,
        deduplicate_across_tracks=True,
        cross_track_dedup_col="selected_config_id",
        track_order=["object_matrix_2way", "object_matrix_3way"],
        require_all_tracks=True,
    )

    assert selected["selected_config_id"].tolist() == ["shared", "replacement"]
    assert selected["selection_track"].tolist() == ["object_matrix_2way", "object_matrix_3way"]
    assert selected["selection_reason"].notna().all()
    assert annotated_pool["is_final_selected"].sum() == 2
    assert not summary.empty
