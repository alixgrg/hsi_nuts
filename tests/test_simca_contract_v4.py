import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.workflows.simca_grid_evaluation import (
    _macro_image_decision_metrics,
    build_validation_guardrails,
)
from src.workflows.simca_robustness import (
    _execution_guardrail_status,
    aggregate_repeated_execution_metrics,
    build_robustness_review_guardrails,
    summarize_random_state_stability_metrics,
)


def _with_schema(columns, rows):
    return pd.DataFrame(rows).reindex(columns=columns)


def _execution(*, track_id="E1", decision_mode="2way", projection_level="object_projection", downstream_status="supported"):
    return _with_schema(
        expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS,
        [
            {
                "model_id": "model_test",
                "random_state": 0,
                "fit_id": "fit_test",
                "projection_id": "projection_test",
                "track_id": track_id,
                "decision_mode": decision_mode,
                "projection_level": projection_level,
                "eligibility_status": (
                    "eligible"
                    if downstream_status == "supported"
                    else "unsupported_domain_shift"
                ),
                "downstream_status": downstream_status,
            }
        ],
    )


def _metric(metric, value, *, aggregation_level="overall", group_id="all", decision_scope="direct"):
    return {
        "model_id": "model_test",
        "random_state": 0,
        "track_id": "E1",
        "decision_scope": decision_scope,
        "map_variant": "raw",
        "aggregation_level": aggregation_level,
        "group_id": group_id,
        "metric": metric,
        "value": value,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "status": "calculable",
        "error_type": "",
        "error_message": "",
    }


def _two_way_metrics(*, false_accept_rate, n_images=3, worst_image_target_miss=0.20):
    rows = [
        _metric("target_miss_rate", 0.0),
        _metric("false_accept_rate", false_accept_rate),
    ]
    for index in range(n_images):
        rows.extend(
            [
                _metric(
                    "target_miss_rate",
                    worst_image_target_miss if index == 0 else 0.0,
                    aggregation_level="source_image",
                    group_id=f"image_{index}",
                ),
                _metric(
                    "false_accept_rate",
                    0.0,
                    aggregation_level="source_image",
                    group_id=f"image_{index}",
                ),
            ]
        )
    return _with_schema(expcfg.SIMCA_VALIDATION_METRIC_COLUMNS, rows)


def test_04c_warning_and_hard_false_accept_thresholds_are_distinct():
    guardrails = build_validation_guardrails(
        _execution(),
        _two_way_metrics(false_accept_rate=0.35),
    )

    assert set(guardrails["candidate_status"]) == {"pass"}
    warning = guardrails.loc[
        guardrails["rule_id"].eq("overall_false_accept_warning")
    ].iloc[0]
    hard = guardrails.loc[
        guardrails["rule_id"].eq("overall_false_accept_hard")
    ].iloc[0]
    assert warning["check_status"] == "fail"
    assert not bool(warning["is_blocking"])
    assert hard["check_status"] == "pass"
    assert bool(hard["is_blocking"])

    rejected = build_validation_guardrails(
        _execution(),
        _two_way_metrics(false_accept_rate=0.41),
    )
    assert set(rejected["candidate_status"]) == {
        "calculable_but_not_acceptable"
    }


def test_worst_image_target_miss_blocks_only_with_five_images():
    three_images = build_validation_guardrails(
        _execution(),
        _two_way_metrics(false_accept_rate=0.0, n_images=3),
    )
    rule = three_images.loc[
        three_images["rule_id"].eq("worst_image_target_miss_conditional")
    ].iloc[0]
    assert rule["n_independent_units"] == 3
    assert rule["check_status"] == "fail"
    assert not bool(rule["is_blocking"])
    assert set(three_images["candidate_status"]) == {"pass"}

    five_images = build_validation_guardrails(
        _execution(),
        _two_way_metrics(false_accept_rate=0.0, n_images=5),
    )
    rule = five_images.loc[
        five_images["rule_id"].eq("worst_image_target_miss_conditional")
    ].iloc[0]
    assert rule["n_independent_units"] == 5
    assert bool(rule["is_blocking"])
    assert set(five_images["candidate_status"]) == {
        "calculable_but_not_acceptable"
    }


def test_diagnostic_pixel_track_has_no_blocking_guardrail():
    executions = _execution(
        track_id="E4",
        decision_mode="3way",
        projection_level="pixel_projection",
        downstream_status="diagnostic_only",
    )
    metrics = pd.DataFrame(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)

    guardrails = build_validation_guardrails(executions, metrics)

    assert set(guardrails["candidate_status"]) == {"diagnostic_only"}
    assert not guardrails["is_blocking"].astype(bool).any()


def test_secondary_scope_failure_is_propagated_as_a_warning():
    rows = []
    for decision_scope, is_blocking, check_status in (
        ("direct", True, "pass"),
        ("pixel_to_object", False, "fail"),
    ):
        rows.append(
            {
                "model_id": "model_test",
                "random_state": 0,
                "track_id": "E7",
                "decision_scope": decision_scope,
                "eligibility_status": "eligible",
                "downstream_status": "supported",
                "candidate_status": "pass",
                "rule_id": "overall_target_miss_hard",
                "scope": "overall",
                "metric": "target_miss_rate",
                "severity": "blocking",
                "check_status": check_status,
                "is_blocking": is_blocking,
            }
        )
    guardrails = _with_schema(expcfg.SIMCA_VALIDATION_GUARDRAIL_COLUMNS, rows)

    status = _execution_guardrail_status(guardrails)
    direct = status.loc[status["decision_scope"].eq("direct")].iloc[0]
    secondary = status.loc[
        status["decision_scope"].eq("pixel_to_object")
    ].iloc[0]

    assert bool(direct["all_blocking_checks_pass"])
    assert not bool(direct["has_supporting_warning"])
    assert bool(secondary["all_blocking_checks_pass"])
    assert bool(secondary["has_supporting_warning"])


def test_macro_image_metrics_separate_target_and_non_target_uncertainty():
    by_image = pd.DataFrame(
        {
            "target_miss_rate": [0.0, 0.2],
            "false_accept_rate": [0.1, 0.0],
            "uncertain_rate": [0.3, 0.5],
            "target_uncertain_rate": [0.2, 0.4],
            "non_target_uncertain_rate": [0.4, 0.6],
            "coverage_rate": [0.7, 0.5],
            "tp": [8, 7],
            "fn": [0, 1],
            "tn": [9, 8],
            "fp": [1, 0],
        }
    )

    result = _macro_image_decision_metrics(by_image)

    assert result["macro_image_target_uncertain_rate"] == pytest.approx(0.3)
    assert result["macro_image_non_target_uncertain_rate"] == pytest.approx(0.5)


def _selection_members():
    rows = []
    for random_state, miss, false_accept in ((0, 0.0, 0.1), (1, 0.1, 0.3)):
        rows.append(
            {
                "model_id": "model_mean",
                "track_id": "E6",
                "random_state": random_state,
                "decision_scope": "direct",
                "is_stochastic": True,
                "target_miss_rate": miss,
                "false_accept_rate": false_accept,
            }
        )
    return _with_schema(expcfg.SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS, rows)


def test_pareto_seed_aggregation_uses_equal_weight_mean():
    aggregated = aggregate_repeated_execution_metrics(
        _selection_members(), include_statistics=True
    )

    assert aggregated.loc[0, "direct__target_miss_rate"] == 0.05
    assert aggregated.loc[0, "direct__false_accept_rate"] == 0.20
    assert aggregated.loc[0, "max__direct__target_miss_rate"] == 0.10


def _disagreement(*, global_rate, target_rate):
    return _with_schema(
        expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS,
        [
            {
                "model_id": "model_mean",
                "track_id": "E6",
                "decision_scope": "direct",
                "n_random_states": 2,
                "n_entities": 20,
                "n_target_entities": 10,
                "entity_seed_coverage_complete": True,
                "decision_disagreement_rate": global_rate,
                "target_decision_disagreement_rate": target_rate,
                "disagreement_status": "calculable",
            }
        ],
    )


def test_only_target_decision_disagreement_is_blocking():
    stable_members = _selection_members()
    stable_members["target_miss_rate"] = 0.05
    stable_members["false_accept_rate"] = 0.20
    supporting = summarize_random_state_stability_metrics(
        stable_members,
        decision_disagreement=_disagreement(global_rate=0.20, target_rate=0.0),
        expected_random_states=(0, 1),
    )
    assert set(supporting["model_stability_status"]) == {
        "robust_with_supporting_warnings"
    }
    assert not supporting["blocking_stability_failed"].astype(bool).any()
    assert "decision_disagreement_rate" in supporting.loc[
        0, "supporting_stability_flags"
    ]

    blocking = summarize_random_state_stability_metrics(
        stable_members,
        decision_disagreement=_disagreement(global_rate=0.20, target_rate=0.06),
        expected_random_states=(0, 1),
    )
    assert set(blocking["model_stability_status"]) == {"unstable_blocking"}
    assert blocking["blocking_stability_failed"].astype(bool).all()


def _review_unit(*, supported, warning=False):
    row = {
        "model_id": "model_review",
        "track_id": "E5" if supported else "E3",
        "eligibility_status": "eligible" if supported else "unsupported_domain_shift",
        "downstream_status": "supported" if supported else "diagnostic_only",
        "is_stochastic": False,
        "seed_requirement_satisfied": True,
        "all_execution_calculable": True,
        "all_execution_protocol_supported": supported,
        "all_04c_blocking_guardrails_pass": True,
        "any_04c_supporting_warning": warning,
        "model_diagnostic_eligible": True,
        "model_protocol_eligible_pre_stability": supported,
        "diagnostic_pareto_eligible": True,
        "is_diagnostic_pareto": True,
        "protocol_pareto_eligible": supported,
        "is_protocol_pareto": supported,
    }
    return _with_schema(expcfg.SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS, [row])


def test_diagnostic_review_is_nonblocking_and_never_enters_pure_test():
    checks, review, pure = build_robustness_review_guardrails(
        _review_unit(supported=False), None
    )

    assert review.loc[0, "review_status"] == "diagnostic_only"
    assert not checks["is_blocking"].astype(bool).any()
    assert "within_track_diagnostic_pareto" in set(checks["check_name"])
    assert pure.empty


def test_04c_warning_propagates_to_review_without_exclusion():
    _, review, pure = build_robustness_review_guardrails(
        _review_unit(supported=True, warning=True), None
    )

    assert review.loc[0, "review_status"] == "eligible_with_warning"
    assert "04c_supporting_guardrail_warning" in review.loc[0, "review_flags"]
    assert len(pure) == 1
