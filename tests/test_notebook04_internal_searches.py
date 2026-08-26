import ast
import json
from pathlib import Path

import pandas as pd

from src import experiment_config as expcfg
from src.workflows.simca_optuna import (
    build_preregistered_ablation_plan,
    build_optuna_study_registry,
    build_optuna_search_efficiency_audit,
    evaluate_config_binary_multiseed,
    select_binary_threshold_pareto,
    suggest_simca_config,
)
from src.workflows.protocol_audit import assert_no_forbidden_score_columns


class _CategoricalTrial:
    def __init__(self, selected):
        self.selected = selected
        self.calls = []

    def suggest_categorical(self, name, choices):
        self.calls.append((name, tuple(choices)))
        assert self.selected in choices
        return self.selected


def test_optuna_suggests_an_exact_calibrated_domain_identifier():
    domain = pd.DataFrame(
        [
            {
                "domain_config_id": "domain_a",
                "preprocessing": "raw",
                "preprocessing_steps": "raw",
            },
            {
                "domain_config_id": "domain_b",
                "preprocessing": "snv",
                "preprocessing_steps": "snv",
            },
        ]
    )
    trial = _CategoricalTrial("domain_b")

    selected = suggest_simca_config(trial, allowed_domain=domain)

    assert selected["domain_config_id"] == "domain_b"
    assert selected["preprocessing_steps"] == ("snv",)
    assert trial.calls == [
        ("domain_config_id", ("domain_a", "domain_b"))
    ]


def test_optuna_objective_contract_is_derived_for_each_evaluation_track():
    assert set(expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS) == set(
        expcfg.SIMCA_EVALUATION_TRACKS
    )
    for track, track_spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.items():
        objective_spec = expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS[track]
        assert objective_spec["objective_names"] == (
            *track_spec["pareto_minimize"],
            *track_spec["pareto_maximize"],
        )
        assert objective_spec["directions"] == (
            *("minimize" for _ in track_spec["pareto_minimize"]),
            *("maximize" for _ in track_spec["pareto_maximize"]),
        )


def test_study_registry_keeps_all_eight_tracks_including_empty_domain():
    domain = pd.DataFrame(
        {
            "evaluation_track": [expcfg.SIMCA_EVALUATION_TRACKS[0]],
            "domain_config_id": ["domain_e1"],
        }
    )
    eligibility = pd.DataFrame(
        {
            "evaluation_track": list(expcfg.SIMCA_EVALUATION_TRACKS),
            "eligibility_status": [
                "eligible",
                "eligible_with_warning",
                "unsupported_internal_calibration",
                "unsupported_domain_shift",
                "eligible",
                "eligible",
                "eligible_with_warning",
                "unsupported_domain_shift",
            ],
        }
    )
    registry = build_optuna_study_registry(
        domain,
        eligibility,
        results_tag="test",
        search_plan_hash="a" * 64,
    )
    assert list(registry["evaluation_track"]) == list(
        expcfg.SIMCA_EVALUATION_TRACKS
    )
    assert registry.loc[registry["track_id"].eq("E3"), "study_status"].item() == (
        "not_runnable_no_domain"
    )
    assert registry.loc[registry["track_id"].eq("E4"), "study_scope"].item() == (
        "unsupported_empty"
    )


def test_optuna_reuses_exact_grid_metrics_without_refitting():
    domain = pd.DataFrame(
        {
            "domain_config_id": ["domain_a", "domain_a"],
            "decision_mode": ["2way", "2way"],
            "random_state": [0, 1],
        }
    )
    metrics = pd.DataFrame(
        {
            "domain_config_id": ["domain_a"],
            "decision_mode": ["2way"],
            "fn_rate_max": [0.05],
            "fp_rate_max": [0.10],
            "balanced_accuracy_mean": [0.90],
            "fold_metric_std": [0.02],
            "coverage_rate_mean": [1.0],
            "uncertain_rate_max": [0.0],
            "status": ["acceptable"],
        }
    )

    folds, summary = evaluate_config_binary_multiseed(
        cfg={"domain_config_id": "domain_a"},
        object_db=None,
        image_db=None,
        calibration_folds=None,
        allowed_domain=domain,
        decision_mode="2way",
        seeds=(0, 1),
        precomputed_metrics=metrics,
        constraints={
            "max_fn_rate": 0.10,
            "max_fp_rate": 0.20,
            "min_balanced_accuracy": 0.80,
            "max_fold_metric_std": 0.10,
        },
    )

    assert folds.empty
    assert summary["status"] == "acceptable"
    assert summary["balanced_accuracy_mean"] == 0.90


def test_ablation_plan_matches_existing_counterparts_without_score():
    track = "object_train__object_projection__2way"
    configurations = pd.DataFrame(
        [
            {
                "calibration_id": "mean",
                "domain_config_id": "domain_mean",
                "evaluation_track": track,
                "track_id": "E1",
                "decision_mode": "2way",
                "matrix_family": "object_matrix",
                "matrix_method": "object_mean",
                "projection_level": "object_projection",
                "projection_matrix_method": "object_mean",
                "preprocessing": "absorbance_sg_d1",
                "preprocessing_steps": "absorbance+sg_d1",
                "direct_2way_threshold": 0.0,
            },
            {
                "calibration_id": "median",
                "domain_config_id": "domain_median",
                "evaluation_track": track,
                "track_id": "E1",
                "decision_mode": "2way",
                "matrix_family": "object_matrix",
                "matrix_method": "object_median",
                "projection_level": "object_projection",
                "projection_matrix_method": "object_median",
                "preprocessing": "absorbance_sg_d1",
                "preprocessing_steps": "absorbance+sg_d1",
                "direct_2way_threshold": 0.0,
            },
        ]
    )
    pareto = pd.DataFrame(
        {
            "row_type": ["configuration"],
            "calibration_id": ["mean"],
            "protocol_pareto_front": [True],
        }
    )
    eligibility = pd.DataFrame(
        {"evaluation_track": [track], "eligibility_status": ["eligible"]}
    )
    plan = build_preregistered_ablation_plan(
        configurations,
        pareto,
        eligibility,
        protocol_hash="protocol",
        search_plan_hash="search",
    )
    paired = plan.loc[plan["factor"].eq("matrix_method_object")]
    assert ((paired["reference_config_id"] == "mean") & (paired["ablated_config_id"] == "median")).any()
    assert plan["preregistered"].all()
    assert not any("score" in column for column in plan.columns)


def test_binary_pareto_has_no_unconstrained_fallback():
    result = select_binary_threshold_pareto(
        pd.DataFrame(
            {
                "object_threshold": [0.5, 0.6],
                "fn_rate": [0.2, 0.3],
                "fp_rate": [0.1, 0.0],
                "balanced_accuracy": [0.8, 0.7],
            }
        ),
        max_fn_rate=0.0,
        max_fp_rate=0.5,
    )

    assert result == {
        "status": "calculable_but_not_acceptable",
        "selected_threshold": None,
        "metrics": None,
    }


def test_optuna_efficiency_audit_uses_exhaustive_pareto_without_score():
    track = "object_train__object_projection__2way"
    exhaustive = pd.DataFrame(
        {
            "domain_config_id": ["da", "db", "dc"],
            "calibration_id": ["a", "b", "c"],
            "evaluation_track": [track] * 3,
            "decision_mode": ["2way"] * 3,
        }
    )
    pareto_reference = pd.DataFrame(
        {
            "row_type": ["configuration"] * 3,
            "calibration_id": ["a", "b", "c"],
            "evaluation_track": [track] * 3,
            "diagnostic_pareto_front": [True, True, False],
            "protocol_pareto_front": [True, True, False],
        }
    )
    trials = pd.DataFrame(
        {
            "domain_config_id": ["da", "da", "dc"],
            "calibration_id": ["a", "a", "c"],
            "evaluation_track": [track] * 3,
            "state": ["COMPLETE"] * 3,
            "status": ["acceptable"] * 3,
        }
    )
    registry = pd.DataFrame(
        {
            "evaluation_track": [track],
            "track_id": ["E1"],
            "decision_mode": ["2way"],
            "study_name": ["study_e1"],
            "study_scope": ["protocol"],
            "eligibility_status": ["eligible"],
            "study_status": ["runnable"],
            "trial_budget": [3],
        }
    )
    audit = build_optuna_search_efficiency_audit(
        exhaustive,
        trials,
        pareto_reference=pareto_reference,
        study_registry=registry,
    )
    row = audit.iloc[0]
    assert row["n_domain_configurations"] == 3
    assert row["n_unique_configurations_sampled"] == 2
    assert row["n_exhaustive_pareto_configurations"] == 2
    assert row["n_exhaustive_pareto_recovered"] == 1
    assert abs(row["duplicate_trial_rate"] - 1 / 3) < 1e-12
    assert abs(row["uniform_recall_expectation"] - (1 - (2 / 3) ** 3)) < 1e-12
    assert not any("score" in column for column in audit.columns)


def test_notebooks_04_use_only_internal_calibration_contract():
    for filename in (
        "04A_simca_grid_search.ipynb",
        "04B_simca_optuna_search.ipynb",
    ):
        path = Path("notebooks") / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        assert "build_calibration_domain_from_03b" not in source
        if filename.startswith("04A"):
            assert "selected_models" in source
            assert "threshold_metrics" in source
            assert "calibration_domain" not in source
        else:
            assert "model_metrics" in source
            assert "model_reference" in source
            assert "calibration_domain" not in source
            assert "grid_configurations" not in source
        assert "sha256_file(" in source
        assert "SIMCA_VALIDATION_BATCHES" not in source
        assert "run_simca_rule_variant_grid" not in source
        assert "OBJECT_THRESHOLDS =" not in source
        assert "THREE_WAY_LOWER_THRESHOLDS =" not in source
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            cell_source = "\n".join(
                line
                for line in "".join(cell.get("source", [])).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            ast.parse(cell_source, filename=f"{filename}:cell-{cell_index}")
            if filename.startswith("04B"):
                assert cell.get("execution_count") is None
                assert not cell.get("outputs")

    grid_source = Path("notebooks/04A_simca_grid_search.ipynb").read_text(
        encoding="utf-8"
    )
    optuna_source = Path(
        "notebooks/04B_simca_optuna_search.ipynb"
    ).read_text(encoding="utf-8")
    assert "run_internal_calibration" not in grid_source
    assert "run_selected_model_reference_audit" in grid_source
    assert "oof_object_predictions" not in grid_source
    assert "oof_pixel_predictions" not in grid_source
    assert "03B_selected_models" in grid_source
    assert "retained_as_diagnostic_only" in grid_source
    assert "weighted_score_used" in grid_source
    assert "run_categorical_tpe_coverage_benchmark" in optuna_source
    assert "SIMCA_OPTUNA_BENCHMARK_ROLE" in optuna_source
    assert "make_optuna_binary_pareto_objective" not in optuna_source
    assert "grid_threshold_metrics_df" not in optuna_source


def test_search_output_contracts_stay_compact():
    assert len(expcfg.SIMCA_GRID_SELECTED_FOLD_METRIC_COLUMNS) == 17
    assert len(expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS) == 7
    assert len(expcfg.SIMCA_OPTUNA_BENCHMARK_SAMPLE_COLUMNS) == 5
    assert len(expcfg.SIMCA_OPTUNA_BENCHMARK_SUMMARY_COLUMNS) == 12
    assert set(expcfg.SIMCA_GRID_SEARCH_OUTPUT_FILENAMES) == {
        "model_reference",
        "fold_metrics",
        "audit_manifest",
    }
    assert set(expcfg.SIMCA_OPTUNA_OUTPUT_FILENAMES) == {
        "sampled_models",
        "search_efficiency",
        "audit_manifest",
    }


def test_active_notebooks_have_no_local_uppercase_scientific_literals():
    filenames = (
        "00_building_database.ipynb",
        "01_database_quality_check.ipynb",
        "02_matrices_preprocessing.ipynb",
        "03_pca_exploration_selection.ipynb",
        "03B_internal_calibration.ipynb",
        "04A_simca_grid_search.ipynb",
        "04B_simca_optuna_search.ipynb",
    )
    offenders = []
    for filename in filenames:
        notebook = json.loads(
            (Path("notebooks") / filename).read_text(encoding="utf-8")
        )
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "\n".join(
                line
                for line in "".join(cell.get("source", [])).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            tree = ast.parse(
                source,
                filename=f"{filename}:cell-{cell_index}",
            )
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                if not isinstance(
                    node.value,
                    (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set),
                ):
                    continue
                for target in targets:
                    provenance_literals = {
                        "CHECKPOINT_PATHS",
                        "REVIEWED_PDF_SHA256",
                    }
                    if (
                        isinstance(target, ast.Name)
                        and target.id.isupper()
                        and target.id not in provenance_literals
                    ):
                        offenders.append(
                            (filename, cell_index, target.id)
                        )
    assert offenders == []


def test_protocol_choices_are_centralized_and_score_free():
    assert expcfg.M_BALANCED_PIXELS == 10
    assert expcfg.PCA_BALANCED_M_VALUES == expcfg.BALANCED_SAMPLING_STUDY_M_VALUES
    assert expcfg.INTERNAL_CALIBRATION_OBJECT_THRESHOLDS == (0.75, 0.80)
    assert expcfg.INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD == 0.0
    assert expcfg.INTERNAL_CALIBRATION_THRESHOLD_CROSSFIT is True
    assert expcfg.INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES == (
        0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0
    )
    assert expcfg.INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES == (
        0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0
    )
    for columns in (
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS,
        expcfg.SIMCA_GRID_SELECTED_FOLD_METRIC_COLUMNS,
        expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS,
        expcfg.SIMCA_OPTUNA_BENCHMARK_SAMPLE_COLUMNS,
        expcfg.SIMCA_OPTUNA_BENCHMARK_SUMMARY_COLUMNS,
    ):
        assert not set(columns).intersection(
            expcfg.ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS
        )


def test_score_column_guard_rejects_legacy_weighted_outputs():
    clean = assert_no_forbidden_score_columns(
        {"metrics": pd.DataFrame({"fn_rate": [0.1]})}
    )
    assert clean["score_free"].all()
    try:
        assert_no_forbidden_score_columns(
            {"metrics": pd.DataFrame({"selection_score": [1.0]})}
        )
    except RuntimeError as exc:
        assert "selection_score" in str(exc)
    else:
        raise AssertionError("Legacy selection_score should be rejected.")
