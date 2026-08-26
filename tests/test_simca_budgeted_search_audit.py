import math

import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.workflows.simca_budgeted_search_audit import (
    build_evaluable_model_universe,
    run_categorical_tpe_coverage_benchmark,
)


def _row(columns, **values):
    row = {column: None for column in columns}
    row.update(values)
    return row


def _benchmark_inputs():
    catalog_rows = []
    metric_rows = []
    selected_rows = []
    reference_rows = []
    track_rows = []

    for track_index, evaluation_track in enumerate(
        expcfg.SIMCA_EVALUATION_TRACKS
    ):
        track_id = expcfg.SIMCA_EVALUATION_TRACK_IDS[evaluation_track]
        track_rows.append(
            _row(
                expcfg.INTERNAL_CALIBRATION_TRACK_CONTRACT_COLUMNS,
                track_id=track_id,
                evaluation_track=evaluation_track,
                decision_mode=("2way" if track_index % 2 == 0 else "3way"),
                projection_level="object_projection",
            )
        )
        objective_spec = expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES[
            track_id
        ]
        objective_names = (
            *objective_spec["minimize"],
            *objective_spec["maximize"],
        )
        for model_index in range(2):
            model_id = f"{track_id.lower()}_model_{model_index}"
            catalog_rows.append(
                _row(
                    expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
                    model_id=model_id,
                    evaluation_track=evaluation_track,
                    track_id=track_id,
                    decision_mode=(
                        "2way" if track_index % 2 == 0 else "3way"
                    ),
                    matrix_family="object_matrix",
                    projection_level="object_projection",
                )
            )
            metric_rows.extend(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "value": 0.1 + 0.01 * model_index,
                }
                for metric in objective_names
            )
            if model_index == 0:
                selected_rows.append(
                    {"model_id": model_id, "selection_status": "selected"}
                )
                reference_rows.append(
                    {
                        "model_id": model_id,
                        "track_id": track_id,
                        "n_selected_runs": 1,
                        "n_decision_scopes": 1,
                        "eligibility_status": "eligible",
                        "downstream_status": (
                            "diagnostic_only"
                            if track_id in {"E3", "E4"}
                            else "supported"
                        ),
                        "max_abs_metric_difference": 0.0,
                    }
                )

    return {
        "model_catalog": pd.DataFrame(catalog_rows),
        "model_metrics": pd.DataFrame(
            metric_rows,
            columns=expcfg.INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS,
        ),
        "selected_models": pd.DataFrame(
            selected_rows,
            columns=expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS,
        ),
        "track_contracts": pd.DataFrame(track_rows),
        "model_reference": pd.DataFrame(
            reference_rows,
            columns=expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS,
        ),
    }


def test_evaluable_universe_keeps_one_existing_identity_per_model():
    inputs = _benchmark_inputs()
    universe = build_evaluable_model_universe(
        inputs["model_catalog"],
        inputs["model_metrics"],
        inputs["track_contracts"],
    )

    assert list(universe.columns) == ["model_id", "track_id"]
    assert len(universe) == 16
    assert universe["model_id"].is_unique
    assert set(universe["track_id"]) == set(
        expcfg.SIMCA_EVALUATION_TRACK_IDS.values()
    )


def test_categorical_tpe_outputs_are_compact_deterministic_and_score_free():
    inputs = _benchmark_inputs()
    first = run_categorical_tpe_coverage_benchmark(
        **inputs,
        trial_budget=5,
        n_startup_trials=2,
        random_state=17,
    )
    second = run_categorical_tpe_coverage_benchmark(
        **inputs,
        trial_budget=5,
        n_startup_trials=2,
        random_state=17,
    )

    sampled = first["sampled_models"]
    summary = first["search_efficiency"]
    assert tuple(sampled.columns) == (
        expcfg.SIMCA_OPTUNA_BENCHMARK_SAMPLE_COLUMNS
    )
    assert tuple(summary.columns) == (
        expcfg.SIMCA_OPTUNA_BENCHMARK_SUMMARY_COLUMNS
    )
    assert len(sampled) == 8 * 5
    assert len(summary) == 8
    assert not sampled.duplicated(["track_id", "trial_number"]).any()
    assert sampled.equals(second["sampled_models"])
    assert summary.equals(second["search_efficiency"])

    forbidden_identity_columns = {
        "calibration_id",
        "domain_config_id",
        "study_name",
        "ablation_id",
        "search_plan_hash",
    }
    assert not forbidden_identity_columns.intersection(sampled.columns)
    assert not forbidden_identity_columns.intersection(summary.columns)
    assert not any("score" in column for column in (*sampled, *summary))

    expected_recall = 1.0 - (1.0 - 1.0 / 2.0) ** 5
    assert all(
        math.isclose(value, expected_recall, rel_tol=0.0, abs_tol=1e-12)
        for value in summary["uniform_expected_selected_recall"]
    )
    assert summary.loc[
        summary["track_id"].isin(["E3", "E4"]), "downstream_status"
    ].eq("diagnostic_only").all()


def test_benchmark_rejects_a_changed_04a_selected_universe():
    inputs = _benchmark_inputs()
    inputs["model_reference"] = inputs["model_reference"].copy()
    inputs["model_reference"].loc[0, "model_id"] = "changed_by_04a"

    with pytest.raises(RuntimeError, match="changed the 03B"):
        run_categorical_tpe_coverage_benchmark(
            **inputs,
            trial_budget=2,
            n_startup_trials=1,
        )
