from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.workflows.simca_internal_calibration as internal_calibration_module
from src import experiment_config as expcfg
from src.decision.uncertainty import select_three_way_threshold_pareto
from src.models.simca import SIMCAClassModel
from src.models.simca_rules import compute_rule_variant_stat_limit
from src.workflows.simca_internal_calibration import (
    build_calibration_folds,
    build_calibration_domain_from_03b,
    build_exact_oof_prediction_equivalence,
    build_internal_calibrated_hyperparameters,
    build_internal_calibration_configurations,
    build_reference_object_table,
    compute_train_only_rule_thresholds,
    evaluate_internal_object_thresholds,
    evaluate_internal_three_way_thresholds,
    hash_internal_calibration_configuration,
    run_internal_calibration,
    select_smallest_plateau_components,
    summarize_internal_three_way_threshold_study,
    validate_simca_configuration,
)


def _reference_frame() -> pd.DataFrame:
    rows = []
    object_index = 0
    for class_name in ("almond", "peanut"):
        for batch in (1, 2):
            for group_index in range(5):
                source_image = f"{class_name}_b{batch}_g{group_index}"
                for within_group in range(2):
                    object_index += 1
                    rows.append(
                        {
                            "source_image": source_image,
                            "object_id": f"obj_{object_index:03d}",
                            "class_name": class_name,
                            "batch": batch,
                            "object_area": (
                                20
                                + 5 * group_index
                                + within_group
                                + (10 if class_name == "peanut" else 0)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _configuration(
    *,
    rule_variant: str = "simple_chi2",
    config_id: str = "cfg",
    fit_config_id: str = "fit",
) -> dict:
    return {
        "config_id": config_id,
        "fit_config_id": fit_config_id,
        "matrix_family": "object_matrix",
        "matrix_method": "object_mean",
        "m": np.nan,
        "balanced_pixel_strategy": "not_applicable",
        "preprocessing": "raw",
        "preprocessing_steps": "raw",
        "rule_family": rule_variant.split("_")[0],
        "rule_variant": rule_variant,
        "limit_source": (
            "calibration_train_only"
            if rule_variant.endswith("_emp_cv")
            else "theoretical_train_fit"
        ),
        "n_components": 1,
        "alpha": 0.01,
        "sg_window_length": 5,
        "sg_polyorder": 2,
        "position_dilation_radius": 0,
        "random_state": 0,
    }


def _internal_object_db() -> dict:
    rng = np.random.default_rng(123)
    out = {}
    for class_name, class_shift in (("almond", 0.15), ("peanut", 0.65)):
        for batch in (1, 2):
            source_image = f"{class_name}_b{batch}"
            for object_index in range(3):
                object_id = f"{source_image}_obj{object_index}"
                pixels = (
                    class_shift
                    + 0.01 * batch
                    + 0.02 * object_index
                    + rng.normal(0.0, 0.01, size=(8, 6))
                )
                positions = np.column_stack(
                    [np.arange(8), np.arange(8)]
                ).astype(int)
                out[object_id] = {
                    "object_id": object_id,
                    "source_clean_key": source_image,
                    "source_image": source_image,
                    "sample_kind": "pure",
                    "object_nut_type": class_name,
                    "batch": batch,
                    "area_pixels": 8,
                    "n_pixels": 8,
                    "spectra": pixels,
                    "mean_spectrum": pixels.mean(axis=0),
                    "median_spectrum": np.median(pixels, axis=0),
                    "std_spectrum": pixels.std(axis=0),
                    "positions_global": positions,
                    "centroid": positions.mean(axis=0),
                    "wavelengths": np.linspace(900.0, 950.0, 6),
                }
    return out


def test_build_calibration_folds_keeps_groups_disjoint_and_balanced():
    folds, diagnostics = build_calibration_folds(
        _reference_frame(),
        n_splits=5,
        random_state=42,
    )
    assert folds.groupby("source_image")["fold_id"].nunique().max() == 1
    assert set(folds["fold_id"]) == set(range(5))
    assert diagnostics["coverage_complete"].all()
    assert tuple(folds.columns) == expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS
    assert (
        tuple(diagnostics.columns)
        == expcfg.INTERNAL_CALIBRATION_FOLD_DIAGNOSTIC_COLUMNS
    )


def test_configuration_grid_uses_preprocessings_by_family_without_variant_split():
    pca_selected = pd.DataFrame(
        {
            "matrix_family": [
                "object_matrix",
                "object_matrix",
                "pixel_matrix",
                "pixel_matrix",
            ],
            "preprocessing": ["raw", "snv", "raw", "raw"],
            "preprocessing_steps": ["raw", "snv", "raw", "raw"],
        }
    )
    configs = build_internal_calibration_configurations(
        pca_selected,
        matrix_methods=("object_mean", "balanced_pixels"),
        m_values=(5,),
        pixel_strategies=("random", "center"),
        n_components_values=(3,),
        rule_variants=("simple_chi2",),
        alpha_values=(0.01,),
        sg_windows=(5,),
        sg_polyorders=(2,),
        dilation_radii=(0,),
        random_seeds=(0, 1),
    )
    object_configs = configs.loc[
        configs["matrix_family"].eq("object_matrix")
    ]
    pixel_configs = configs.loc[configs["matrix_family"].eq("pixel_matrix")]
    assert set(object_configs["preprocessing"]) == {"raw", "snv"}
    assert set(pixel_configs["preprocessing"]) == {"raw"}
    assert set(pixel_configs["balanced_pixel_strategy"]) == {
        "random",
        "center",
    }
    assert (
        pixel_configs.loc[
            pixel_configs["balanced_pixel_strategy"].eq("random"),
            "random_state",
        ].nunique()
        == 2
    )
    assert (
        pixel_configs.loc[
            pixel_configs["balanced_pixel_strategy"].eq("center"),
            "random_state",
        ].nunique()
        == 1
    )
    assert configs["config_id"].is_unique


def test_internal_calibration_runtime_grid_disables_dilation_and_uses_three_seeds():
    assert expcfg.INTERNAL_CALIBRATION_AVAILABLE_DILATION_RADII == (0, 2, 3, 5)
    assert expcfg.INTERNAL_CALIBRATION_DILATION_RADII == (0,)
    assert expcfg.INTERNAL_CALIBRATION_RANDOM_SEEDS == (0, 1, 2)
    assert expcfg.INTERNAL_CALIBRATION_RISK_PROFILE == "exploratory"
    assert (
        expcfg.INTERNAL_CALIBRATION_MAX_FN_RATE
        == expcfg.INTERNAL_CALIBRATION_RISK_PROFILES["exploratory"][
            "max_fn_rate"
        ]
    )


def test_validate_simca_configuration_reports_all_requested_limits():
    config = {
        "matrix_method": "balanced_pixels",
        "preprocessing_steps": "sg_d1",
        "n_components": 4,
        "m": 20,
        "sg_window_length": 4,
        "sg_polyorder": 4,
        "position_dilation_radius": -1,
    }
    result = validate_simca_configuration(
        config,
        X_train=np.ones((4, 3)),
        y_train=("peanut",) * 4,
        n_target_observations=4,
        n_features=3,
        numeric_rank=2,
        n_pixels_by_object=(10, 12),
        available_classes=("peanut",),
    )
    assert not result["valid"]
    assert {
        "MISSING_NON_TARGET_CLASS",
        "N_COMPONENTS_EXCEED_DIMENSION",
        "N_COMPONENTS_EXCEED_TARGET_N",
        "N_COMPONENTS_EXCEED_RANK",
        "SG_WINDOW_NOT_ODD",
        "SG_WINDOW_NOT_GREATER_THAN_POLYORDER",
        "SG_WINDOW_EXCEEDS_SPECTRUM",
        "M_EXCEEDS_AVAILABLE_PIXELS",
        "NEGATIVE_DILATION_RADIUS",
    }.issubset(result["technical_error_codes"])
    assert all(
        set(error) == {"code", "message", "parameter"}
        for error in result["technical_errors"]
    )


def test_combined_index_empirical_limit_is_train_only():
    rng = np.random.default_rng(4)
    model = SIMCAClassModel(
        class_name="peanut",
        n_components=2,
        alpha=0.01,
    ).fit(rng.normal(size=(30, 6)))
    thresholds = compute_train_only_rule_thresholds(model, alpha=0.01)
    stat, limit = compute_rule_variant_stat_limit(
        H=model.H_train_,
        Q=model.Q_train_,
        model=model,
        variant_name="combined_index_emp_cv",
        cv_thresholds=thresholds,
    )
    assert np.isfinite(stat).all()
    assert limit == thresholds["combined_index_emp_cv"]
    assert 0.0 <= np.mean(stat >= limit) <= 0.1


def test_three_way_selection_can_preserve_infeasibility():
    grid = pd.DataFrame(
        {
            "three_way_lower_threshold": [0.2, 0.3],
            "three_way_upper_threshold": [0.7, 0.8],
            "target_miss_rate": [0.2, 0.3],
            "non_target_false_accept_rate": [0.2, 0.3],
            "uncertain_rate": [0.4, 0.5],
            "coverage_rate": [0.6, 0.5],
            "decided_balanced_accuracy": [0.8, 0.7],
        }
    )
    selected = select_three_way_threshold_pareto(
        grid,
        max_target_miss_rate=0.05,
        max_false_accept_rate=0.05,
        max_uncertain_rate=0.2,
        min_coverage=0.8,
        allow_infeasible_fallback=False,
    )
    assert not selected["feasible"]
    assert (
        selected["selection_status"]
        == "technically_calculable_but_not_acceptable"
    )
    assert pd.isna(selected["three_way_lower_threshold"])


def test_internal_calibration_produces_strict_oof_tables():
    object_db = _internal_object_db()
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(
        reference,
        n_splits=2,
        random_state=4,
        require_complete_coverage=True,
    )
    configs = pd.DataFrame(
        [
            _configuration(),
            _configuration(
                rule_variant="combined_index_emp_cv",
                config_id="cfg_combined",
            ),
        ]
    )
    result = run_internal_calibration(
        object_db=object_db,
        image_db={},
        folds=folds,
        configurations=configs,
        wavelengths=np.linspace(900.0, 950.0, 6),
        keep_oof_pixels=True,
        keep_oof_objects=True,
        verbose=False,
    )
    assert result["errors"].empty
    assert not result["oof_pixels"].empty
    assert not result["oof_objects"].empty
    assert set(result["oof_pixels"]["fold_id"]) == {0, 1}
    assert set(result["oof_pixels"]["batch"]) == {1, 2}
    assert (
        tuple(result["oof_pixels"].columns)
        == expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS
    )
    assert (
        tuple(result["rule_diagnostics"].columns)
        == expcfg.INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS
    )
    assert result["rule_diagnostics"]["rule_limit"].notna().all()
    assert (
        tuple(result["sampling_diagnostics"].columns)
        == expcfg.INTERNAL_CALIBRATION_SAMPLING_DIAGNOSTIC_COLUMNS
    )


def test_balanced_pixel_sampling_is_sketched_compactly_across_seeds():
    object_db = _internal_object_db()
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(
        reference,
        n_splits=2,
        random_state=4,
        require_complete_coverage=True,
    )
    configurations = []
    for seed in (0, 1):
        configurations.append(
            {
                **_configuration(
                    config_id=f"seed_{seed}",
                    fit_config_id=f"fit_seed_{seed}",
                ),
                "matrix_family": "pixel_matrix",
                "matrix_method": "balanced_pixels",
                "m": 5,
                "balanced_pixel_strategy": "random",
                "random_state": seed,
            }
        )
    result = run_internal_calibration(
        object_db=object_db,
        image_db={},
        folds=folds,
        configurations=pd.DataFrame(configurations),
        wavelengths=np.linspace(900.0, 950.0, 6),
        verbose=False,
    )
    sampling = result["sampling_diagnostics"]
    assert len(sampling) == 4
    assert sampling["sampling_group_id"].nunique() == 1
    assert sampling["sampling_minhash"].str.len().gt(0).all()


def test_internal_calibration_reuses_matrix_and_preprocessing_across_k(monkeypatch):
    object_db = _internal_object_db()
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(
        reference,
        n_splits=2,
        random_state=4,
        require_complete_coverage=True,
    )
    configurations = pd.DataFrame(
        [
            _configuration(config_id="k1", fit_config_id="fit_k1"),
            {
                **_configuration(config_id="k2", fit_config_id="fit_k2"),
                "n_components": 2,
            },
        ]
    )
    original_build_matrix = internal_calibration_module.build_matrix
    calls = []

    def counted_build_matrix(*args, **kwargs):
        calls.append(str(kwargs.get("matrix_method")))
        return original_build_matrix(*args, **kwargs)

    monkeypatch.setattr(
        internal_calibration_module,
        "build_matrix",
        counted_build_matrix,
    )
    run_internal_calibration(
        object_db=object_db,
        image_db={},
        folds=folds,
        configurations=configurations,
        wavelengths=np.linspace(900.0, 950.0, 6),
        keep_oof_pixels=False,
        keep_oof_objects=False,
        verbose=False,
    )

    assert calls.count("object_mean") == 2
    assert calls.count("pixel") == 2


def test_internal_calibration_checkpoint_resumes_without_recomputing(
    tmp_path,
    monkeypatch,
):
    object_db = _internal_object_db()
    reference = build_reference_object_table(object_db)
    folds, _ = build_calibration_folds(
        reference,
        n_splits=2,
        random_state=4,
        require_complete_coverage=True,
    )
    configurations = pd.DataFrame([_configuration()])
    checkpoint_dir = tmp_path / "checkpoint"
    kwargs = {
        "object_db": object_db,
        "image_db": {},
        "folds": folds,
        "configurations": configurations,
        "wavelengths": np.linspace(900.0, 950.0, 6),
        "keep_oof_pixels": False,
        "keep_oof_objects": False,
        "verbose": False,
        "checkpoint_dir": checkpoint_dir,
        "resume_from_checkpoint": True,
        "checkpoint_every_n_data_configs": 1,
    }
    first = run_internal_calibration(**kwargs)
    assert len(list(checkpoint_dir.rglob("manifest.json"))) == 1
    assert len(list(checkpoint_dir.rglob("markers/*.json"))) == 1

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("A completed checkpoint was recomputed.")

    monkeypatch.setattr(
        internal_calibration_module,
        "build_matrix",
        fail_if_recomputed,
    )
    resumed = run_internal_calibration(**kwargs)

    assert len(resumed["fold_metrics"]) == len(first["fold_metrics"])
    assert len(resumed["rule_diagnostics"]) == len(
        first["rule_diagnostics"]
    )
    assert resumed["errors"].empty


def test_threshold_grids_and_component_plateau_are_score_free():
    rows = []
    for config_id, k in (("k3", 3), ("k4", 4)):
        for fold_id in (0, 1):
            rows.append(
                {
                    "config_id": config_id,
                    "fold_id": fold_id,
                    "fn_rate": 0.0,
                    "fp_rate": 0.0,
                    "balanced_accuracy": 1.0,
                }
            )
    configurations = pd.DataFrame(
        [
            {
                **_configuration(config_id="k3", fit_config_id="fit3"),
                "n_components": 3,
            },
            {
                **_configuration(config_id="k4", fit_config_id="fit4"),
                "n_components": 4,
            },
        ]
    )
    component_summary, selected_components = select_smallest_plateau_components(
        pd.DataFrame(rows),
        configurations,
    )
    assert selected_components["n_components"].tolist() == [3]
    assert "selection_score" not in component_summary

    oof = pd.DataFrame(
        {
            "config_id": ["k3"] * 4,
            "fold_id": [0, 1, 0, 1],
            "source_image": ["t1", "t2", "n1", "n2"],
            "object_id": ["t1", "t2", "n1", "n2"],
            "batch": [1, 2, 1, 2],
            "target_pixel_ratio": [0.95, 0.85, 0.15, 0.05],
            "true_target_object": [True, True, False, False],
        }
    )
    grid_2way = evaluate_internal_object_thresholds(
        oof,
        thresholds=(0.5, 0.75, 0.8),
    )
    grid_3way = evaluate_internal_three_way_thresholds(
        oof,
        lower_thresholds=(0.1, 0.2),
        upper_thresholds=(0.75, 0.9),
    )
    assert grid_2way["selected"].sum() == 1
    assert grid_3way["selected"].sum() == 1
    assert "selection_score" not in grid_2way
    assert "selection_score" not in grid_3way
    study = summarize_internal_three_way_threshold_study(grid_3way)
    assert tuple(study.columns) == (
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_STUDY_COLUMNS
    )
    assert study["n_configurations"].gt(0).all()
    assert study["feasible_configuration_rate"].between(0.0, 1.0).all()


def test_exact_oof_prediction_deduplication_is_bitwise_and_lossless():
    base = pd.DataFrame(
        {
            "fold_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "source_image": [
                "image_0",
                "image_0",
                "image_0",
                "image_0",
                "image_1",
                "image_1",
                "image_1",
                "image_1",
            ],
            "object_id": [f"object_{index}" for index in range(8)],
            "batch": [1, 1, 1, 1, 2, 2, 2, 2],
            "target_pixel_ratio": [
                0.90,
                0.40,
                0.60,
                0.10,
                0.80,
                0.20,
                0.75,
                np.nan,
            ],
            "true_target_object": [
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
            ],
        }
    )
    one_ulp_ratios = base["target_pixel_ratio"].to_numpy(copy=True)
    one_ulp_ratios[2] = np.nextafter(one_ulp_ratios[2], 1.0)
    oof = pd.concat(
        [
            base.assign(config_id="config_a"),
            base.sample(frac=1.0, random_state=4).assign(
                config_id="config_b"
            ),
            base.assign(
                config_id="config_c",
                target_pixel_ratio=one_ulp_ratios,
            ),
        ],
        ignore_index=True,
    )

    equivalence = build_exact_oof_prediction_equivalence(oof)
    mapping = equivalence.set_index("config_id")
    assert (
        mapping.loc["config_a", "representative_config_id"]
        == mapping.loc["config_b", "representative_config_id"]
    )
    assert (
        mapping.loc["config_a", "prediction_signature"]
        == mapping.loc["config_b", "prediction_signature"]
    )
    assert (
        mapping.loc["config_c", "prediction_signature"]
        != mapping.loc["config_a", "prediction_signature"]
    )
    assert mapping.loc["config_a", "equivalence_size"] == 2
    assert mapping.loc["config_c", "equivalence_size"] == 1

    grid_2way = evaluate_internal_object_thresholds(
        oof,
        thresholds=(0.50,),
        prediction_equivalence=equivalence,
        max_fn_rate=1.0,
        max_fp_rate=1.0,
        min_balanced_accuracy=0.0,
    ).set_index("config_id")
    assert set(grid_2way.index) == {
        "config_a",
        "config_b",
        "config_c",
    }
    assert grid_2way.loc["config_a", "n"] == 7
    assert grid_2way.loc["config_a", "fn"] == 2
    assert grid_2way.loc["config_a", "fp"] == 2
    assert grid_2way.loc["config_a", "decision_rate"] == 7 / 8
    pd.testing.assert_series_equal(
        grid_2way.loc["config_a"],
        grid_2way.loc["config_b"],
        check_names=False,
    )

    grid_3way = evaluate_internal_three_way_thresholds(
        oof,
        lower_thresholds=(0.20,),
        upper_thresholds=(0.80,),
        prediction_equivalence=equivalence,
        max_target_miss_rate=1.0,
        max_false_accept_rate=1.0,
        max_uncertain_rate=1.0,
        min_coverage=0.0,
    ).set_index("config_id")
    assert grid_3way.loc["config_a", "n"] == 8
    # The 3-way protocol is inclusive at the lower boundary:
    # ratio <= lower threshold is classified as non-target.
    assert grid_3way.loc["config_a", "target_miss_rate"] == 0.25
    assert (
        grid_3way.loc[
            "config_a",
            "non_target_false_accept_rate",
        ]
        == 0.0
    )
    assert grid_3way.loc["config_a", "uncertain_rate"] == 4 / 8
    assert grid_3way.loc["config_a", "coverage_rate"] == 4 / 8
    assert (
        grid_3way.loc["config_a", "decided_balanced_accuracy"]
        == (2 / 3 + 1.0) / 2
    )
    pd.testing.assert_series_equal(
        grid_3way.loc["config_a"],
        grid_3way.loc["config_b"],
        check_names=False,
    )


def test_component_selection_aggregates_seeds_before_choosing_k():
    configurations = []
    metrics = []
    for seed in (0, 1, 2):
        for k in (3, 4):
            config_id = f"s{seed}_k{k}"
            configurations.append(
                {
                    **_configuration(
                        config_id=config_id,
                        fit_config_id=f"fit_{config_id}",
                    ),
                    "n_components": k,
                    "random_state": seed,
                }
            )
            for fold_id in (0, 1):
                metrics.append(
                    {
                        "config_id": config_id,
                        "fold_id": fold_id,
                        "fn_rate": 0.10 + 0.01 * seed,
                        "fp_rate": 0.10,
                        "balanced_accuracy": 0.90 - 0.005 * (k - 3),
                    }
                )
    summary, selected = select_smallest_plateau_components(
        pd.DataFrame(metrics),
        pd.DataFrame(configurations),
        expected_n_folds=2,
        tolerance=0.02,
    )
    assert set(selected["n_components"]) == {3}
    assert set(selected["random_state"]) == {0, 1, 2}
    assert summary.loc[
        summary["n_components"].eq(3),
        "n_seeds_evaluated",
    ].eq(3).all()


def test_calibrated_hyperparameters_are_seed_agnostic_and_track_separated():
    configurations = []
    object_rows = []
    for family, method, strategy in (
        ("object_matrix", "object_mean", "not_applicable"),
        ("pixel_matrix", "balanced_pixels", "random"),
    ):
        seeds = (0,) if family == "object_matrix" else (0, 1, 2)
        preprocessings = (
            ("raw", "snv")
            if family == "object_matrix"
            else ("raw",)
        )
        for preprocessing in preprocessings:
            for seed in seeds:
                config_id = f"{family}_{preprocessing}_{seed}"
                configurations.append(
                    {
                        **_configuration(
                            config_id=config_id,
                            fit_config_id=f"fit_{config_id}",
                        ),
                        "matrix_family": family,
                        "matrix_method": method,
                        "m": np.nan if family == "object_matrix" else 5,
                        "balanced_pixel_strategy": strategy,
                        "preprocessing": preprocessing,
                        "preprocessing_steps": preprocessing,
                        "n_components": 3,
                        "random_state": seed,
                        "component_acceptable": True,
                        "component_plateau": True,
                    }
                )
                ratios = (
                    0.95,
                    0.85,
                    0.15,
                    0.05,
                    0.90,
                    0.80,
                    0.20,
                    0.10,
                )
                for index, ratio in enumerate(ratios):
                    object_rows.append(
                        {
                            "config_id": config_id,
                            "fold_id": index // 4,
                            "source_image": (
                                "t1",
                                "t1",
                                "n1",
                                "n1",
                                "t2",
                                "t2",
                                "n2",
                                "n2",
                            )[index],
                            "object_id": f"o{index}",
                            "batch": 1 + index // 4,
                            "target_pixel_ratio": ratio,
                            "true_target_object": index % 4 < 2,
                        }
                    )
    oof = pd.DataFrame(object_rows)
    thresholds_2way = evaluate_internal_object_thresholds(
        oof,
        thresholds=(0.50, 0.75, 0.80),
    )
    thresholds_3way = evaluate_internal_three_way_thresholds(
        oof,
        lower_thresholds=(0.10, 0.20),
        upper_thresholds=(0.75, 0.90),
    )
    diagnostics = pd.DataFrame(
        [
            {
                "config_id": config["config_id"],
                "fold_id": fold_id,
                "rule_limit": 1.0,
                "train_rejection_rate": 0.01,
                "validation_target_rejection_rate": 0.02,
                "n_train_target": 20,
                "n_validation_target": 4,
            }
            for config in configurations
            for fold_id in (0, 1)
        ]
    )
    calibrated = build_internal_calibrated_hyperparameters(
        pd.DataFrame(configurations),
        thresholds_2way,
        thresholds_3way,
        rule_diagnostics=diagnostics,
        oof_objects=oof,
    )
    assert set(calibrated["calibration_track"]) == {
        "object_matrix_2way",
        "object_matrix_3way",
        "pixel_matrix_2way",
        "pixel_matrix_3way",
    }
    assert "pareto_front" not in calibrated
    assert "random_state" not in calibrated
    assert calibrated["calibration_id"].is_unique
    assert calibrated["calibration_status"].eq(
        "calibrated_for_downstream_search"
    ).all()
    assert calibrated.loc[
        calibrated["calibration_track"].str.startswith("pixel_matrix"),
        "n_seeds_evaluated",
    ].eq(3).all()
    object_calibrations = calibrated.loc[
        calibrated["calibration_track"].str.startswith("object_matrix")
    ]
    assert object_calibrations.groupby("calibration_track").size().eq(2).all()
    assert set(object_calibrations["preprocessing"]) == {"raw", "snv"}
    calibration_audit = calibrated.attrs["calibration_audit"]
    assert calibration_audit
    assert pd.DataFrame(calibration_audit)["stage"].iloc[-1] == (
        "calibrated_for_downstream_search"
    )

    pca_shortlist = calibrated[
        ["calibration_track", "preprocessing"]
    ].copy()
    pca_shortlist["matrix_family"] = pca_shortlist[
        "calibration_track"
    ].str.replace(r"_(2way|3way)$", "", regex=True)
    domain = build_calibration_domain_from_03b(
        calibrated,
        pca_selected_preprocessings=pca_shortlist[
            ["matrix_family", "preprocessing"]
        ],
        random_seeds=(0, 1, 2),
    )
    assert domain["config_id"].is_unique
    assert domain["domain_config_id"].nunique() == len(calibrated)
    assert domain.loc[
        domain["calibration_track"].str.startswith("pixel_matrix")
    ].groupby("domain_config_id").size().eq(3).all()
    assert domain.loc[
        domain["calibration_track"].str.startswith("object_matrix")
    ].groupby("domain_config_id").size().eq(1).all()

    legacy_calibrated = calibrated.drop(
        columns=[
            "source_config_id",
            "model_group_id",
            "random_states_json",
            "position_dilation_radius",
        ]
    )
    legacy_domain = build_calibration_domain_from_03b(
        legacy_calibrated,
        pca_selected_preprocessings=pca_shortlist[
            ["matrix_family", "preprocessing"]
        ],
        random_seeds=(0, 1, 2),
    )
    assert legacy_domain["position_dilation_radius"].eq(0).all()
    assert legacy_domain["source_config_id"].notna().all()
    assert legacy_domain["model_group_id"].notna().all()


def test_configuration_hash_is_order_independent():
    left = hash_internal_calibration_configuration({"b": 2, "a": 1})
    right = hash_internal_calibration_configuration({"a": 1, "b": 2})
    assert left == right


def test_notebook_03b_has_all_8track_sections_and_no_saved_outputs():
    path = Path("notebooks/03B_internal_calibration.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    for section in range(1, 10):
        assert f"## {section} —" in source
    assert "INTERNAL_CALIBRATION_OUTPUT_FILENAMES.items()" in source
    assert "INTERNAL_CALIBRATION_BATCHES" in source
    assert "INTERNAL_CALIBRATION_FORBIDDEN_BATCHES" in source
    assert "build_internal_calibration_configurations" in source
    assert "build_internal_calibrated_hyperparameters_8tracks" in source
    assert "run_internal_calibration_8tracks" in source
    assert "build_simca_track_contracts" in source
    assert "expand_projection_configurations" in source
    assert "evaluate_internal_2way_tracks" in source
    assert "evaluate_crossfitted_three_way_thresholds" in source
    assert "build_exact_oof_prediction_equivalence" not in source
    assert "select_smallest_plateau_components" not in source
    assert "build_internal_selected_candidates" not in source
    assert "INTERNAL_CALIBRATION_PANEL_MAX_PER_TRACK" not in source
    assert "INTERNAL_CALIBRATION_PANEL_DIVERSITY_COLUMNS" not in source

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        code = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(code, filename=f"{path.name}:cell-{index}")
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")
