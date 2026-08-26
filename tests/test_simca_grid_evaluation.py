import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.workflows.simca_calibration_selection import (
    aggregate_threshold_candidates,
    build_model_metrics,
)
from src.workflows.simca_selected_model_audit import (
    run_selected_model_reference_audit,
)


def _frame_with_schema(columns, rows):
    return pd.DataFrame(
        [{column: row.get(column) for column in columns} for row in rows],
        columns=columns,
    )


def _synthetic_inputs(tmp_path):
    model_id = "model_test"
    model_catalog = _frame_with_schema(
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        [
            {
                "model_id": model_id,
                "track_id": "E1",
                "decision_mode": "2way",
                "projection_level": "object_projection",
            }
        ],
    )
    selected_models = pd.DataFrame(
        {"model_id": [model_id], "selection_status": ["selected"]}
    )
    selected_runs = pd.DataFrame(
        {
            "model_id": [model_id],
            "random_state": [0],
            "fit_id": ["fit_test"],
            "projection_id": ["projection_test"],
        }
    )
    selected_thresholds = pd.DataFrame(
        {
            "model_id": [model_id],
            "random_state": [0],
            "decision_scope": ["direct"],
            "lower_quantile": [float("nan")],
            "upper_quantile": [float("nan")],
            "vote_threshold": [float("nan")],
            "lower_threshold": [0.0],
            "upper_threshold": [0.0],
        }
    )
    fold_values = (
        {
            "target_miss_rate": 0.0,
            "false_accept_rate": 0.5,
            "balanced_accuracy": 0.75,
            "n_observations": 4,
            "n_target": 2,
            "n_non_target": 2,
            "max_unit_target_miss_rate": 0.0,
            "max_unit_false_accept_rate": 0.5,
        },
        {
            "target_miss_rate": 0.5,
            "false_accept_rate": 0.0,
            "balanced_accuracy": 0.75,
            "n_observations": 4,
            "n_target": 2,
            "n_non_target": 2,
            "max_unit_target_miss_rate": 0.5,
            "max_unit_false_accept_rate": 0.0,
        },
    )
    metric_rows = []
    for fold_id, values in enumerate(fold_values):
        for metric, value in values.items():
            metric_rows.append(
                {
                    "model_id": model_id,
                    "random_state": 0,
                    "evaluation_fold": fold_id,
                    "decision_scope": "direct",
                    "lower_quantile": float("nan"),
                    "upper_quantile": float("nan"),
                    "vote_threshold": float("nan"),
                    "lower_threshold": 0.0,
                    "upper_threshold": 0.0,
                    "metric": metric,
                    "value": value,
                }
            )
    threshold_metrics = _frame_with_schema(
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS,
        metric_rows,
    )
    threshold_metrics_path = tmp_path / "threshold_metrics.parquet"
    threshold_metrics.to_parquet(threshold_metrics_path, index=False)
    model_metrics = build_model_metrics(
        aggregate_threshold_candidates(threshold_metrics)
    )
    track_contracts = pd.DataFrame(
        {
            "track_id": ["E1"],
            "decision_mode": ["2way"],
            "projection_level": ["object_projection"],
        }
    )
    projection_eligibility = _frame_with_schema(
        expcfg.PROJECTION_ELIGIBILITY_COLUMNS,
        [
            {
                "track_id": "E1",
                "n_selected_models": 1,
                "n_selected_runs": 1,
                "eligibility_status": "eligible",
            }
        ],
    )
    return {
        "model_catalog": model_catalog,
        "selected_models": selected_models,
        "selected_runs": selected_runs,
        "selected_threshold_rows": selected_thresholds,
        "model_metrics": model_metrics,
        "track_contracts": track_contracts,
        "projection_eligibility": projection_eligibility,
        "threshold_metrics_path": threshold_metrics_path,
    }


def test_04a_audits_selected_natural_keys_without_new_ids(tmp_path):
    outputs = run_selected_model_reference_audit(
        **_synthetic_inputs(tmp_path)
    )

    assert set(outputs) == {"model_reference", "fold_metrics"}
    reference = outputs["model_reference"]
    folds = outputs["fold_metrics"]
    assert tuple(reference.columns) == expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS
    assert tuple(folds.columns) == expcfg.SIMCA_GRID_SELECTED_FOLD_METRIC_COLUMNS
    assert reference.to_dict(orient="records") == [
        {
            "model_id": "model_test",
            "track_id": "E1",
            "n_selected_runs": 1,
            "n_decision_scopes": 1,
            "eligibility_status": "eligible",
            "downstream_status": "supported",
            "max_abs_metric_difference": 0.0,
        }
    ]
    assert len(folds) == expcfg.INTERNAL_CALIBRATION_N_SPLITS
    assert not folds.duplicated(
        ["model_id", "random_state", "decision_scope", "fold_id"]
    ).any()
    forbidden_ids = {
        "calibration_id",
        "domain_config_id",
        "fit_id",
        "projection_id",
        "duplicate_group_id",
    }
    assert not forbidden_ids.intersection(reference.columns)
    assert not forbidden_ids.intersection(folds.columns)


def test_04a_blocks_when_03b_model_metrics_cannot_be_reproduced(tmp_path):
    inputs = _synthetic_inputs(tmp_path)
    inputs["model_metrics"] = inputs["model_metrics"].copy()
    inputs["model_metrics"].loc[
        inputs["model_metrics"]["metric"].eq("direct.target_miss_rate"),
        "value",
    ] += 0.01

    with pytest.raises(RuntimeError, match="not reproducible"):
        run_selected_model_reference_audit(**inputs)
