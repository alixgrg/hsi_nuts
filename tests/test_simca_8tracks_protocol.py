from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.workflows.simca_internal_calibration as calibration_module
from src import experiment_config as expcfg
from src.protocol_governance import build_simca_track_contracts
from src.workflows.simca_internal_calibration import (
    build_calibration_domain_8tracks,
    build_calibration_folds,
    build_internal_calibrated_hyperparameters_8tracks,
    build_internal_calibration_configurations,
    build_reference_object_table,
    evaluate_crossfitted_three_way_thresholds,
    evaluate_internal_2way_tracks,
    expand_projection_configurations,
    load_selected_oof_predictions_from_checkpoint_8tracks,
    resolve_internal_calibration_checkpoint_run_8tracks,
    run_internal_calibration_8tracks,
    hash_internal_calibration_configuration,
    summarize_internal_calibration_checkpoint_8tracks,
)


def _object_db() -> dict:
    rng = np.random.default_rng(12)
    records = {}
    for class_name, shift in (("almond", 0.1), ("peanut", 0.7)):
        for batch in (1, 2):
            source_image = f"{class_name}_b{batch}"
            for object_index in range(3):
                object_id = f"{source_image}_{object_index}"
                spectra = shift + rng.normal(0.0, 0.02, size=(12, 6))
                positions = np.column_stack(
                    [np.arange(12), np.arange(12)]
                )
                records[object_id] = {
                    "object_id": object_id,
                    "source_clean_key": source_image,
                    "source_image": source_image,
                    "sample_kind": "pure",
                    "object_nut_type": class_name,
                    "batch": batch,
                    "area_pixels": 12,
                    "n_pixels": 12,
                    "spectra": spectra,
                    "mean_spectrum": spectra.mean(axis=0),
                    "median_spectrum": np.median(spectra, axis=0),
                    "positions_global": positions,
                    "centroid": positions.mean(axis=0),
                    "wavelengths": np.linspace(900.0, 950.0, 6),
                }
    return records


def _configurations() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.DataFrame(
        {
            "matrix_family": ["object_matrix", "pixel_matrix"],
            "preprocessing": ["raw", "raw"],
            "preprocessing_steps": ["raw", "raw"],
        }
    )
    fits = build_internal_calibration_configurations(
        selected,
        matrix_methods=("object_mean", "balanced_pixels"),
        m_values=(10,),
        pixel_strategies=("center",),
        n_components_values=(1,),
        rule_variants=("simple_chi2",),
        alpha_values=(0.01,),
        sg_windows=(5,),
        sg_polyorders=(2,),
        dilation_radii=(0,),
        random_seeds=(0,),
    )
    contracts = build_simca_track_contracts()
    return expand_projection_configurations(fits, contracts), contracts


def test_track_contract_materialises_e1_to_e8_and_separates_fragment_metrics():
    contracts = build_simca_track_contracts()
    assert contracts["track_id"].tolist() == [f"E{i}" for i in range(1, 9)]
    assert contracts["evaluation_track"].is_unique
    assert contracts["direct_2way_threshold"].eq(0.0).all()
    pixel_contracts = contracts.loc[
        contracts["projection_level"].eq("pixel_projection")
    ]
    assert pixel_contracts["primary_unit"].eq("source_image").all()
    assert pixel_contracts["calibration_primary_metrics_json"].str.contains(
        "small_fragment_recall"
    ).sum() == 0
    assert pixel_contracts["final_evaluation_metrics_json"].str.contains(
        "small_fragment_recall"
    ).all()


def test_reference_table_enforces_qc_allowed_object_ids():
    object_db = _object_db()
    allowed = sorted(object_db)[:-1]
    reference = build_reference_object_table(
        object_db,
        allowed_object_ids=allowed,
    )
    assert set(reference["object_id"]) == set(allowed)
    assert reference["object_id"].is_unique


def test_projection_expansion_has_three_stable_identity_levels():
    expanded, _ = _configurations()
    assert set(expanded["track_id"]) == {f"E{i}" for i in range(1, 9)}
    assert expanded["evaluation_config_id"].is_unique
    assert expanded["fit_config_id"].nunique() == 2
    assert (
        expanded.groupby("projection_config_id")["decision_mode"]
        .nunique()
        .max()
        == 2
    )
    e5_e6 = expanded.loc[expanded["track_id"].isin(["E5", "E6"])]
    assert set(e5_e6["projection_matrix_method"]) == {
        "object_mean",
        "object_median",
    }


def test_8track_runner_shares_fits_and_emits_signed_oof_margins():
    object_db = _object_db()
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(reference, n_splits=2)
    expanded, _ = _configurations()
    result = run_internal_calibration_8tracks(
        object_db=object_db,
        folds=folds,
        configurations=expanded,
        wavelengths=np.linspace(900.0, 950.0, 6),
        verbose=False,
    )
    assert result["technical_errors"].empty
    assert not result["oof_object_predictions"].empty
    assert not result["oof_pixel_predictions"].empty
    for name in ("oof_object_predictions", "oof_pixel_predictions"):
        table = result[name]
        assert np.allclose(
            table["simca_margin"],
            1.0 - table["normalized_ratio"],
        )
        assert table["direct_2way_decision"].eq(
            table["simca_margin"].ge(0.0)
        ).all()
        assert set(table["fold_id"]) == {0, 1}
    assert len(result["fit_diagnostics"]) == 4
    metrics, votes = evaluate_internal_2way_tracks(
        result["oof_object_predictions"],
        result["oof_pixel_predictions"],
        expanded,
    )
    assert set(metrics["track_id"]) == {"E1", "E3", "E5", "E7"}
    assert set(votes["track_id"]) == {"E3", "E7"}
    assert set(votes["secondary_object_threshold"]) == {0.75, 0.80}


def test_8track_runner_filters_projection_rows_invalid_for_absorbance():
    object_db = _object_db()
    object_id = sorted(object_db)[0]
    object_db[object_id]["spectra"][0, 0] = 0.0
    object_db[object_id]["mean_spectrum"] = object_db[object_id][
        "spectra"
    ].mean(axis=0)
    object_db[object_id]["median_spectrum"] = np.median(
        object_db[object_id]["spectra"],
        axis=0,
    )
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(reference, n_splits=2)
    selected = pd.DataFrame(
        {
            "matrix_family": ["object_matrix", "pixel_matrix"],
            "preprocessing": ["absorbance", "raw"],
            "preprocessing_steps": ["absorbance", "raw"],
        }
    )
    fits = build_internal_calibration_configurations(
        selected,
        matrix_methods=("object_mean", "balanced_pixels"),
        m_values=(10,),
        pixel_strategies=("center",),
        n_components_values=(1,),
        rule_variants=("simple_chi2",),
        alpha_values=(0.01,),
        sg_windows=(5,),
        sg_polyorders=(2,),
        dilation_radii=(0,),
        random_seeds=(0,),
    )
    expanded = expand_projection_configurations(
        fits,
        build_simca_track_contracts(),
    )
    result = run_internal_calibration_8tracks(
        object_db=object_db,
        folds=folds,
        configurations=expanded,
        wavelengths=np.linspace(900.0, 950.0, 6),
        verbose=False,
    )
    assert not result["oof_pixel_predictions"].empty
    assert result["rule_diagnostics"]["status"].eq("ok").all()
    filters = result["technical_errors"].loc[
        result["technical_errors"]["audit_type"].eq(
            "projection_input_filter"
        )
    ]
    assert not filters.empty
    assert (
        pd.to_numeric(filters["n_initial"])
        - pd.to_numeric(filters["n_technical_valid"])
    ).eq(1).all()
    assert filters["failure_reason"].str.contains(
        "nonpositive_absorbance_rows=1",
        regex=False,
    ).all()


def test_notebook_03b_has_the_nine_protocol_sections_and_no_old_k_pass():
    notebook = json.loads(
        Path("notebooks/03B_internal_calibration.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for section in range(1, 10):
        assert f"## {section} —" in source
    assert "build_simca_track_contracts" in source
    assert "run_internal_calibration_8tracks" in source
    assert "summarize_internal_calibration_checkpoint_8tracks" in source
    assert "select_smallest_plateau_components" not in source
    assert "build_exact_oof_prediction_equivalence" not in source
    assert "build_calibration_domain_from_03b" not in source
    assert "small_fragment_recall" not in source


def test_active_03b_configuration_is_margin_based_and_centralized():
    assert expcfg.INTERNAL_CALIBRATION_M_VALUES == (10, 20)
    assert expcfg.INTERNAL_CALIBRATION_UNDER_M_POLICY == "exclude"
    assert expcfg.INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD == 0.0
    assert expcfg.INTERNAL_CALIBRATION_THRESHOLD_CROSSFIT is True
    assert expcfg.INTERNAL_CALIBRATION_THREE_WAY_LOWER_THRESHOLDS == ()
    assert expcfg.INTERNAL_CALIBRATION_THREE_WAY_UPPER_THRESHOLDS == ()


def test_crossfitted_three_way_thresholds_keep_evaluation_fold_held_out():
    scores = pd.DataFrame(
        {
            "projection_config_id": ["projection"] * 8,
            "fit_config_id": ["fit"] * 8,
            "fold_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "random_state": [0] * 8,
            "truth": [True, True, False, False] * 2,
            "simca_margin": [0.9, 0.6, -0.6, -0.9, 0.8, 0.5, -0.5, -0.8],
        }
    )
    configurations = pd.DataFrame(
        [
            {
                "evaluation_config_id": "evaluation",
                "projection_config_id": "projection",
                "evaluation_track": "object_train__object_projection__3way",
                "track_id": "E2",
                "projection_level": "object_projection",
                "decision_mode": "3way",
                "random_state": 0,
            }
        ]
    )
    thresholds, _ = evaluate_crossfitted_three_way_thresholds(
        scores,
        pd.DataFrame(),
        configurations,
    )
    assert set(thresholds["evaluation_fold"]) == {-1, 0, 1}
    selected = thresholds.loc[thresholds["selected"]]
    assert selected["three_way_lower_threshold"].lt(0.0).all()
    assert selected["three_way_upper_threshold"].gt(0.0).all()
    assert selected["score_type"].eq("simca_margin").all()


def test_internal_hash_does_not_silently_drop_identifier_fields():
    left = hash_internal_calibration_configuration(
        {"fit_config_id": "fit_a", "n_components": 3}
    )
    right = hash_internal_calibration_configuration(
        {"fit_config_id": "fit_b", "n_components": 3}
    )
    assert left != right


def test_pixel_three_way_studies_direct_and_derived_decision_scopes():
    rows = []
    for fold_id in (0, 1):
        for object_id, truth, margin in (
            (f"target_{fold_id}", True, 0.8),
            (f"non_target_{fold_id}", False, -0.8),
        ):
            for pixel_index in range(4):
                rows.append(
                    {
                        "projection_config_id": "pixel_projection",
                        "fit_config_id": "fit",
                        "fold_id": fold_id,
                        "random_state": 0,
                        "source_image": f"image_{fold_id}_{truth}",
                        "object_id": object_id,
                        "truth": truth,
                        "simca_margin": margin,
                        "direct_2way_decision": truth,
                        "row": pixel_index,
                        "col": pixel_index,
                    }
                )
    configurations = pd.DataFrame(
        [
            {
                "evaluation_config_id": "pixel_evaluation",
                "projection_config_id": "pixel_projection",
                "evaluation_track": "object_train__pixel_projection__3way",
                "track_id": "E4",
                "projection_level": "pixel_projection",
                "decision_mode": "3way",
                "random_state": 0,
            }
        ]
    )
    thresholds, study = evaluate_crossfitted_three_way_thresholds(
        pd.DataFrame(),
        pd.DataFrame(rows),
        configurations,
    )
    assert set(thresholds["decision_scope"]) == {
        "direct",
        "derived_pixel_to_object",
    }
    assert set(thresholds["score_type"]) == {
        "simca_margin",
        "pixel_vote_ratio",
    }
    assert set(study["decision_scope"]) == {
        "direct",
        "derived_pixel_to_object",
    }


def test_eight_track_selection_returns_explicit_audit_and_locked_domain():
    configuration_rows = []
    metric_rows = []
    threshold_rows = []
    for evaluation_track in expcfg.SIMCA_EVALUATION_TRACKS:
        spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[evaluation_track]
        track_id = spec["track_id"]
        pixel_training = spec["training_matrix_family"] == "pixel_matrix"
        projection_method = spec["allowed_projection_methods"][0]
        evaluation_id = f"evaluation_{track_id}"
        configuration_rows.append(
            {
                "evaluation_config_id": evaluation_id,
                "projection_config_id": f"projection_{track_id}",
                "fit_config_id": f"fit_{track_id}",
                "source_config_id": f"source_{track_id}",
                "evaluation_track": evaluation_track,
                "track_id": track_id,
                "parent_track": spec["parent_track"],
                "decision_mode": spec["decision_mode"],
                "decision_score_type": "simca_margin",
                "matrix_family": spec["training_matrix_family"],
                "matrix_method": (
                    "balanced_pixels" if pixel_training else "object_mean"
                ),
                "projection_level": spec["projection_level"],
                "projection_matrix_method": projection_method,
                "m": 10 if pixel_training else np.nan,
                "balanced_pixel_strategy": (
                    "center" if pixel_training else "not_applicable"
                ),
                "preprocessing": "raw",
                "preprocessing_steps": "raw",
                "rule_family": "simple",
                "rule_variant": "simple_chi2",
                "limit_source": "theoretical_train_fit",
                "n_components": 3,
                "alpha": 0.01,
                "sg_window_length": 5,
                "sg_polyorder": 2,
                "position_dilation_radius": 0,
                "random_state": 0,
                "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
                "protocol_version": expcfg.PROTOCOL_VERSION,
            }
        )
        if spec["decision_mode"] == "2way":
            for fold_id in (0, 1):
                metric_rows.append(
                    {
                        "evaluation_config_id": evaluation_id,
                        "evaluation_track": evaluation_track,
                        "track_id": track_id,
                        "fold_id": fold_id,
                        "random_state": 0,
                        "aggregation_level": "macro_source_image",
                        "n": 4,
                        "target_miss_rate": 0.0,
                        "false_accept_rate": 0.0,
                        "balanced_accuracy": 1.0,
                        "metric_role": "calibration_primary",
                    }
                )
        else:
            threshold_rows.append(
                {
                    "evaluation_config_id": evaluation_id,
                    "evaluation_track": evaluation_track,
                    "track_id": track_id,
                    "evaluation_fold": -1,
                    "random_state": 0,
                    "decision_scope": "direct",
                    "score_type": "simca_margin",
                    "three_way_lower_threshold": -0.2,
                    "three_way_upper_threshold": 0.2,
                    "target_miss_rate": 0.0,
                    "false_accept_rate": 0.0,
                    "uncertain_rate": 0.1,
                    "coverage_rate": 0.9,
                    "decided_balanced_accuracy": 1.0,
                    "feasible": True,
                    "pareto_front": True,
                    "selected": True,
                    "failure_reason": "",
                }
            )
    configurations = pd.DataFrame(configuration_rows)
    calibrated, audit = build_internal_calibrated_hyperparameters_8tracks(
        configurations,
        pd.DataFrame(metric_rows),
        pd.DataFrame(threshold_rows),
    )
    assert set(calibrated["track_id"]) == {f"E{i}" for i in range(1, 9)}
    assert audit["track_status"].eq("calibrated").all()
    domain = build_calibration_domain_8tracks(
        calibrated,
        configurations,
        pca_shortlist_id="shortlist",
        protocol_hash="protocol_hash",
    )
    assert tuple(domain.columns) == expcfg.SIMCA_CALIBRATION_DOMAIN_COLUMNS
    assert set(domain["evaluation_track"]) == set(
        expcfg.SIMCA_EVALUATION_TRACKS
    )
    assert "object_threshold" not in domain

    unsupported_metrics = pd.DataFrame(metric_rows)
    unsupported_metrics.loc[
        unsupported_metrics["track_id"].eq("E3"),
        "target_miss_rate",
    ] = 0.25
    calibrated_supported, audit_supported = (
        build_internal_calibrated_hyperparameters_8tracks(
            configurations,
            unsupported_metrics,
            pd.DataFrame(threshold_rows),
            allowed_unsupported_track_ids=("E3",),
        )
    )
    e3_audit = audit_supported.loc[audit_supported["track_id"].eq("E3")]
    assert e3_audit["track_status"].eq("unsupported").all()
    assert e3_audit["failure_reason"].eq("risk_constraints").all()
    assert "E3" not in set(calibrated_supported["track_id"])
    supported_domain = build_calibration_domain_8tracks(
        calibrated_supported,
        configurations,
        pca_shortlist_id="shortlist",
        protocol_hash="protocol_hash",
        unsupported_track_ids=("E3",),
    )
    assert set(supported_domain["track_id"]) == {
        "E1", "E2", "E4", "E5", "E6", "E7", "E8"
    }


def test_8track_checkpoint_validates_shards_and_resumes_without_refit(
    tmp_path,
    monkeypatch,
):
    object_db = _object_db()
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(reference, n_splits=2)
    expanded, _ = _configurations()
    context = {
        "protocol_hash": "protocol",
        "pca_shortlist_id": "shortlist",
        "track_contract_hash": "tracks",
        "fold_contract_hash": "folds",
        "configuration_hash": "configuration",
    }
    kwargs = {
        "object_db": object_db,
        "folds": folds,
        "configurations": expanded,
        "wavelengths": np.linspace(900.0, 950.0, 6),
        "verbose": False,
        "checkpoint_dir": tmp_path / "checkpoints",
        "checkpoint_context": context,
        "resume_from_checkpoint": True,
    }
    first = run_internal_calibration_8tracks(**kwargs)
    markers = list((tmp_path / "checkpoints").rglob("markers/*.json"))
    assert len(markers) == expanded["matrix_family"].nunique()
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    assert marker["schema_version"] == expcfg.RESULTS_SCHEMA_VERSION
    assert marker["protocol_hash"] == "protocol"
    assert marker["shards"]
    assert {
        "row_count",
        "columns",
        "file_sha256",
        "completed_fit_config_ids",
    }.issubset(marker["shards"][0])

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("A validated checkpoint was recomputed.")

    monkeypatch.setattr(
        calibration_module,
        "fit_simca_bundle_from_matrix",
        fail_if_refit,
    )
    resumed = run_internal_calibration_8tracks(**kwargs)
    assert len(resumed["fit_diagnostics"]) == len(first["fit_diagnostics"])
    assert len(resumed["oof_pixel_predictions"]) == len(
        first["oof_pixel_predictions"]
    )

    streamed_state = run_internal_calibration_8tracks(
        **kwargs,
        materialize_checkpoint_results=False,
    )
    assert streamed_state["oof_pixel_predictions"].empty
    streamed = summarize_internal_calibration_checkpoint_8tracks(
        streamed_state["checkpoint_run_dir"],
        expanded,
        verbose=False,
    )
    assert not streamed["metrics_2way"].empty
    assert not streamed["thresholds_3way"].empty
    assert streamed["oof_pixel_predictions"].empty

    complete_run = resolve_internal_calibration_checkpoint_run_8tracks(
        tmp_path / "checkpoints",
        checkpoint_context=context,
        expected_fit_config_ids=expanded["fit_config_id"].astype(str).unique(),
    )
    assert complete_run == Path(streamed_state["checkpoint_run_dir"])

    selected_domain = (
        expanded.drop_duplicates("projection_config_id")
        .groupby("projection_level", sort=False, as_index=False)
        .head(1)
        .loc[
            :,
            ["projection_config_id", "fit_config_id", "projection_level"],
        ]
    )
    selected_objects, selected_pixels = (
        load_selected_oof_predictions_from_checkpoint_8tracks(
            streamed_state["checkpoint_run_dir"],
            selected_domain,
        )
    )
    expected_object_ids = set(
        selected_domain.loc[
            ~selected_domain["projection_level"].eq("pixel_projection"),
            "projection_config_id",
        ].astype(str)
    )
    expected_pixel_ids = set(
        selected_domain.loc[
            selected_domain["projection_level"].eq("pixel_projection"),
            "projection_config_id",
        ].astype(str)
    )
    assert set(selected_objects["projection_config_id"].astype(str)) == (
        expected_object_ids
    )
    assert set(selected_pixels["projection_config_id"].astype(str)) == (
        expected_pixel_ids
    )
