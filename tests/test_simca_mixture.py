import pandas as pd
import pytest

from src.workflows.simca_mixture import (
    build_mixture_guardrails,
    build_mixture_projection_filters,
    prepare_mixture_outputs,
    restore_mixture_selected_configs,
    validate_mixture_guardrails,
)
from src.workflows.simca_tables import SIMCA_TABLE_COLUMNS, compact_simca_table


def _final_selected_row(selected_config_id="cfg_001", selection_track="object_matrix_2way"):
    matrix_family, decision_mode = selection_track.rsplit("_", 1)
    return {
        "selected_config_id": selected_config_id,
        "candidate_id": f"cand_{selected_config_id}",
        "selection_track": selection_track,
        "assigned_selection_track": selection_track,
        "matrix_family": matrix_family,
        "decision_mode": decision_mode,
        "metric_level": "object" if matrix_family == "object_matrix" else "pixel",
        "matrix_method": "object_mean" if matrix_family == "object_matrix" else "balanced_pixels",
        "preprocessing": "snv",
        "rule_for_refit": "simple_emp_cv",
        "n_components": 3,
        "alpha": 0.01,
        "object_threshold": 0.75,
        "balanced_pixel_strategy_effective": "random",
        "final_rank_in_track": 1,
        "pareto_tier": 1,
        "pareto_rank_in_track": 1,
        "is_pareto_front": True,
        "selection_reason": "pareto_tier=1",
        "previous_flags": "",
        "selection_status": "selected",
    }


def _candidate_row(selected_config_id="cfg_001"):
    return {
        "selected_config_id": selected_config_id,
        "candidate_id": f"cand_{selected_config_id}",
        "target_class": "peanut",
        "non_target_label": "almond",
        "candidate_sources": "06B",
        "matrix_family": "object_matrix",
        "matrix_method": "object_mean",
        "training_matrix_id": "object_mean",
        "preprocessing": "snv",
        "preprocessing_steps": "snv",
        "model_family": "empirical_cv_rule",
        "rule_variant": "simple_emp_cv",
        "selected_rule_name": "simple_emp_cv",
        "rule_for_refit": "simple_emp_cv",
        "n_components": 3,
        "alpha": 0.01,
        "object_threshold": 0.75,
    }


def test_restore_mixture_selected_configs_joins_full_candidate_panel_and_thresholds():
    selected = pd.DataFrame([_final_selected_row()])
    candidates = pd.DataFrame([_candidate_row()])
    thresholds = pd.DataFrame(
        [
            {
                "selected_config_id": "cfg_001",
                "three_way_lower_threshold": 0.25,
                "three_way_upper_threshold": 0.80,
            }
        ]
    )

    restored, selected_thresholds = restore_mixture_selected_configs(
        final_selected_models_df=selected,
        candidate_panel_df=candidates,
        thresholds_df=thresholds,
    )

    assert restored.loc[0, "assigned_selection_track"] == "object_matrix_2way"
    assert restored.loc[0, "preprocessing_steps"] == "snv"
    assert restored.loc[0, "model_family"] == "empirical_cv_rule"
    assert restored.loc[0, "three_way_lower_threshold"] == 0.25
    assert selected_thresholds["selected_config_id"].tolist() == ["cfg_001"]


def test_restore_mixture_selected_configs_requires_thresholds():
    with pytest.raises(RuntimeError, match="Missing fixed 3-way thresholds"):
        restore_mixture_selected_configs(
            final_selected_models_df=pd.DataFrame([_final_selected_row()]),
            candidate_panel_df=pd.DataFrame([_candidate_row()]),
            thresholds_df=pd.DataFrame(
                [{"selected_config_id": "other", "three_way_lower_threshold": 0.1, "three_way_upper_threshold": 0.9}]
            ),
        )


def test_prepare_mixture_outputs_keeps_only_assigned_track_metrics():
    outputs = {
        "2way_object_metrics": pd.DataFrame(
            [
                {
                    "selected_config_id": "cfg_001",
                    "selection_track": "object_matrix_2way",
                    "assigned_selection_track": "object_matrix_2way",
                    "matrix_family": "object_matrix",
                    "decision_mode": "2way",
                    "metric_level": "object",
                    "fn_rate": 0.1,
                    "fp_rate": 0.2,
                    "balanced_accuracy": 0.85,
                },
                {
                    "selected_config_id": "cfg_001",
                    "selection_track": "object_matrix_2way",
                    "assigned_selection_track": "object_matrix_3way",
                    "matrix_family": "object_matrix",
                    "decision_mode": "2way",
                    "metric_level": "object",
                    "fn_rate": 0.3,
                    "fp_rate": 0.4,
                    "balanced_accuracy": 0.65,
                },
            ]
        ),
        "2way_pixel_metrics": pd.DataFrame(),
        "3way_object_metrics": pd.DataFrame(),
    }

    prepared = prepare_mixture_outputs(outputs, keep_only_assigned_track_metrics=True)

    assert prepared["2way_object_metrics"]["selected_config_id"].tolist() == ["cfg_001"]
    assert len(prepared["metrics_long"]) == 1
    assert prepared["summary"].loc[0, "n_models"] == 1


def test_mixture_guardrails_require_mixture_projection_filter():
    guardrails = build_mixture_guardrails(
        selected_configs_df=pd.DataFrame([_candidate_row()]),
        final_selected_models_df=pd.DataFrame([_final_selected_row()]),
        candidate_panel_df=pd.DataFrame([_candidate_row()]),
        thresholds_df=pd.DataFrame([{"selected_config_id": "cfg_001"}]),
        object_db={},
        train_batches=[1, 2, 3, 4],
        projection_filters={"sample_kind": ["pure"]},
        expected_tracks=["object_matrix_2way"],
    )

    with pytest.raises(RuntimeError, match="projection_filters_select_mixtures"):
        validate_mixture_guardrails(guardrails)


def test_mixture_projection_filters_and_compact_contracts_are_score_free():
    assert build_mixture_projection_filters() == {"sample_kind": ["mixture"]}
    assert "selection_score" not in SIMCA_TABLE_COLUMNS["mixture_metrics"]

    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "cfg_001",
                "selection_track": "object_matrix_2way",
                "assigned_selection_track": "object_matrix_2way",
                "matrix_family": "object_matrix",
                "decision_mode": "2way",
                "metric_level": "object",
                "fn_rate": 0.0,
                "fp_rate": 0.1,
                "selection_score": 99,
                "all_empty": None,
            }
        ]
    )

    compact = compact_simca_table(raw, table_kind="mixture_metrics")

    assert "selection_score" not in compact.columns
    assert "all_empty" not in compact.columns
