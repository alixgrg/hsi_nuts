import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.workflows.simca_candidates import (
    add_selection_track,
    add_simca_candidate_ids,
    build_pca_preprocessing_configs_by_matrix_family,
    deduplicate_simca_candidates,
    deduplicate_metric_equivalent_simca_candidates,
    deduplicate_simca_refit_configs,
    filter_simca_candidates_by_pca_preprocessing,
    selection_track_from_parts,
    validate_simca_candidate_contract,
    validate_simca_evaluation_contract,
)


def _base_candidate(**overrides):
    row = {
        "matrix_family": "object_matrix",
        "matrix_method": "object_mean",
        "training_matrix_id": "object_mean",
        "preprocessing": "absorbance_sg_d1",
        "preprocessing_steps": ("absorbance", "sg_d1"),
        "model_family": "empirical_cv_rule",
        "rule": "simple",
        "rule_variant": "simple_emp_cv",
        "selected_rule_name": "simple_emp_cv",
        "rule_for_refit": "simple_emp_cv",
        "limit_source": "empirical_cv",
        "m": 40,
        "m_effective": 40,
        "balanced_pixel_strategy": "not_applicable",
        "balanced_pixel_strategy_effective": "random",
        "n_components": 5,
        "alpha": 0.01,
        "object_threshold": 0.75,
        "sg_window_length": 11,
        "sg_polyorder": 2,
        "position_dilation_radius": 3,
        "target_class": "peanut",
        "non_target_label": "almond",
        "n": 10,
        "tp": 8,
        "fn": 2,
        "fp": 1,
        "tn": 9,
        "fn_rate": 0.2,
        "fp_rate": 0.1,
        "balanced_accuracy": 0.85,
    }
    row.update(overrides)
    return row


def test_selection_tracks_are_explicit_and_validated():
    assert expcfg.SIMCA_SELECTION_TRACKS == (
        "object_matrix_2way",
        "object_matrix_3way",
        "pixel_matrix_2way",
        "pixel_matrix_3way",
    )
    assert selection_track_from_parts("object_matrix", "2way") == "object_matrix_2way"
    assert selection_track_from_parts("pixel_matrix", "three_way") == "pixel_matrix_3way"

    df = pd.DataFrame(
        [
            {"matrix_family": "object_matrix", "decision_mode": "2way"},
            {"matrix_family": "pixel_matrix", "decision_mode": "3way"},
        ]
    )
    tracked = add_selection_track(df)

    assert tracked["selection_track"].tolist() == [
        "object_matrix_2way",
        "pixel_matrix_3way",
    ]

    with pytest.raises(ValueError, match="Unknown SIMCA selection track"):
        selection_track_from_parts("object_matrix", "4way")


def test_candidate_key_is_stable_and_ignores_source_columns():
    df = pd.DataFrame(
        [
            _base_candidate(candidate_source="grid"),
            _base_candidate(
                preprocessing_steps='["absorbance", "sg_d1"]',
                candidate_source="optuna",
            ),
            _base_candidate(object_threshold=0.80, candidate_source="grid"),
        ]
    )

    keyed = add_simca_candidate_ids(df)

    assert keyed.loc[0, "candidate_id"] == keyed.loc[1, "candidate_id"]
    assert keyed.loc[0, "candidate_id"] != keyed.loc[2, "candidate_id"]


def test_deduplicate_candidates_merges_sources_and_counts_duplicates():
    df = pd.DataFrame(
        [
            _base_candidate(candidate_source="grid"),
            _base_candidate(candidate_source="optuna"),
            _base_candidate(
                matrix_method="object_median",
                training_matrix_id="object_median",
                candidate_source="grid",
            ),
        ]
    )

    deduped = deduplicate_simca_candidates(df)

    assert len(deduped) == 2
    first = deduped.sort_values("n_duplicate_rows", ascending=False).iloc[0]
    assert first["candidate_sources"] == "grid,optuna"
    assert int(first["n_candidate_sources"]) == 2
    assert int(first["n_duplicate_rows"]) == 2


def test_deduplicate_candidates_drops_previous_provenance_counters_before_merge():
    df = pd.DataFrame(
        [
            _base_candidate(
                candidate_source="grid",
                candidate_sources="grid",
                n_candidate_sources=99,
                n_duplicate_rows=99,
            ),
            _base_candidate(
                candidate_source="optuna",
                candidate_sources="optuna",
                n_candidate_sources=99,
                n_duplicate_rows=99,
            ),
        ]
    )

    deduped = deduplicate_simca_candidates(df)

    assert len(deduped) == 1
    assert not any(col.endswith("_x") or col.endswith("_y") for col in deduped.columns)
    assert deduped.loc[0, "candidate_sources"] == "grid,optuna"
    assert int(deduped.loc[0, "n_candidate_sources"]) == 2
    assert int(deduped.loc[0, "n_duplicate_rows"]) == 2


def test_deduplicate_refit_configs_keeps_first_and_reports_dropped_rows():
    df = pd.DataFrame(
        [
            _base_candidate(
                candidate_id="grid_candidate",
                candidate_source="grid",
                candidate_sources="grid",
                model_family="empirical_cv_rule",
                m=40,
                m_effective=40,
                balanced_pixel_strategy="not_applicable",
                balanced_pixel_strategy_effective="random",
                fn_rate=0.0,
            ),
            _base_candidate(
                candidate_id="optuna_candidate",
                candidate_source="optuna",
                candidate_sources="optuna",
                model_family="rule_variant_grid",
                m=40,
                m_effective=40,
                balanced_pixel_strategy="random",
                balanced_pixel_strategy_effective="random",
                fn_rate=0.1,
            ),
            _base_candidate(
                candidate_id="other_candidate",
                candidate_source="grid",
                candidate_sources="grid",
                n_components=6,
                m=40,
                m_effective=40,
                balanced_pixel_strategy="not_applicable",
                balanced_pixel_strategy_effective="random",
            ),
        ]
    )

    kept, dropped, summary = deduplicate_simca_refit_configs(df)

    assert len(kept) == 2
    assert len(dropped) == 1
    assert kept.loc[0, "candidate_id"] == "grid_candidate"
    assert dropped.loc[0, "candidate_id"] == "optuna_candidate"

    duplicate_summary = summary.loc[summary["n_refit_config_candidates"].eq(2)].iloc[0]
    assert duplicate_summary["refit_config_candidate_ids"] == "grid_candidate,optuna_candidate"
    assert duplicate_summary["refit_config_candidate_sources"] == "grid,optuna"


def test_deduplicate_metric_equivalent_candidates_collapses_one_parameter_group():
    df = pd.DataFrame(
        [
            _base_candidate(candidate_id="simple_candidate", n_components=4),
            _base_candidate(candidate_id="complex_candidate", n_components=8),
            _base_candidate(candidate_id="different_metric", n_components=8, fn=3, fn_rate=0.3),
        ]
    )

    kept, dropped, summary = deduplicate_metric_equivalent_simca_candidates(df)

    assert kept["candidate_id"].tolist() == ["simple_candidate", "different_metric"]
    assert dropped["candidate_id"].tolist() == ["complex_candidate"]
    assert summary.loc[0, "varied_parameter_group"] == "n_components"
    assert summary.loc[0, "kept_candidate_id"] == "simple_candidate"


def test_deduplicate_metric_equivalent_candidates_keeps_two_parameter_differences():
    df = pd.DataFrame(
        [
            _base_candidate(candidate_id="baseline", n_components=4, object_threshold=0.75),
            _base_candidate(candidate_id="two_differences", n_components=8, object_threshold=0.80),
        ]
    )

    kept, dropped, summary = deduplicate_metric_equivalent_simca_candidates(df)

    assert kept["candidate_id"].tolist() == ["baseline", "two_differences"]
    assert dropped.empty
    assert summary.empty


def test_deduplicate_metric_equivalent_candidates_respects_protected_family():
    df = pd.DataFrame(
        [
            _base_candidate(candidate_id="object_candidate", matrix_family="object_matrix"),
            _base_candidate(
                candidate_id="pixel_candidate",
                matrix_family="pixel_matrix",
                matrix_method="balanced_pixels",
                training_matrix_id="balanced_pixel_random_m40",
            ),
        ]
    )

    kept, dropped, summary = deduplicate_metric_equivalent_simca_candidates(df)

    assert kept["candidate_id"].tolist() == ["object_candidate", "pixel_candidate"]
    assert dropped.empty
    assert summary.empty


def test_pca_preprocessing_configs_are_kept_separate_by_matrix_family():
    pca_shortlist = pd.DataFrame(
        [
            {
                "matrix_family": "object_matrix",
                "preprocessing": "absorbance_sg_d1",
                "preprocessing_steps": "absorbance+sg_d1",
            },
            {
                "matrix_family": "object_matrix",
                "preprocessing": "shared_snv",
                "preprocessing_steps": "snv",
            },
            {
                "matrix_family": "pixel_matrix",
                "preprocessing": "snv_sg_smooth",
                "preprocessing_steps": "snv+sg_smooth",
            },
            {
                "matrix_family": "pixel_matrix",
                "preprocessing": "shared_snv",
                "preprocessing_steps": "snv",
            },
        ]
    )

    configs = build_pca_preprocessing_configs_by_matrix_family(pca_shortlist)

    assert set(configs["object_matrix"]) == {"absorbance_sg_d1", "shared_snv"}
    assert set(configs["pixel_matrix"]) == {"snv_sg_smooth", "shared_snv"}

    candidates = pd.DataFrame(
        [
            {"matrix_family": "object_matrix", "preprocessing": "absorbance_sg_d1"},
            {"matrix_family": "pixel_matrix", "preprocessing": "snv_sg_smooth"},
            {"matrix_family": "object_matrix", "preprocessing": "shared_snv"},
            {"matrix_family": "pixel_matrix", "preprocessing": "shared_snv"},
            {"matrix_family": "pixel_matrix", "preprocessing": "absorbance_sg_d1"},
        ]
    )

    filtered = filter_simca_candidates_by_pca_preprocessing(
        candidates,
        pca_shortlist,
        strict=False,
    )

    assert len(filtered) == 4
    assert not (
        filtered["matrix_family"].eq("pixel_matrix")
        & filtered["preprocessing"].eq("absorbance_sg_d1")
    ).any()

    with pytest.raises(ValueError, match="not selected for their matrix family"):
        filter_simca_candidates_by_pca_preprocessing(candidates, pca_shortlist, strict=True)


def test_candidate_and_evaluation_contracts_validate_required_columns():
    candidate = deduplicate_simca_candidates(pd.DataFrame([_base_candidate()]))
    validate_simca_candidate_contract(candidate)

    evaluation = add_selection_track(
        pd.DataFrame(
            [
                {
                    "candidate_id": candidate.loc[0, "candidate_id"],
                    "matrix_family": "object_matrix",
                    "decision_mode": "2way",
                    "evaluation_stage": "validation",
                    "metric_level": "object",
                    "fn_rate": 0.0,
                    "fp_rate": 0.1,
                    "balanced_accuracy": 0.95,
                }
            ]
        )
    )
    validate_simca_evaluation_contract(evaluation)

    bad = evaluation.copy()
    bad.loc[0, "selection_track"] = "pixel_matrix_2way"
    with pytest.raises(ValueError, match="selection_track must match"):
        validate_simca_evaluation_contract(bad)
