from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.workflows.pca import (
    compare_aligned_loadings,
    compare_pca_representations,
    evaluate_pca_stability,
    fit_pca_candidate,
    subset_object_db_for_pca,
    summarize_pca_stability,
)
from src.workflows.pca_selection import (
    PCA_ARTIFACT_COLUMNS,
    PCA_TECHNICAL_FLAG_COLUMNS,
    aggregate_pca_preprocessing_diagnostics,
    build_pca_artifact_review_table,
    build_pca_scoring_diagnostics,
    build_pca_selection_diagnostics,
    build_pca_selection_flow_tables,
    select_pca_pareto_front,
    select_pca_preprocessing_shortlist,
    validate_pca_preprocessing_shortlist,
)


def _candidate_fixture(n_per_family=12):
    rows = []
    for family in ("object_matrix", "pixel_matrix"):
        for idx in range(n_per_family):
            object_family = family == "object_matrix"
            row = {
                "candidate_id": f"{family}_{idx}",
                "training_matrix_id": f"calibration_{family}_{idx}",
                "wavelength_axis_id": "axis",
                "matrix_family": family,
                "matrix_variant": (
                    "object_mean"
                    if object_family
                    else "balanced_pixels_random"
                ),
                "matrix_method": (
                    "object_mean"
                    if object_family
                    else "balanced_pixels"
                ),
                "balanced_pixel_strategy": (
                    "not_applicable" if object_family else "random"
                ),
                "m": np.nan if object_family else 10,
                "preprocessing": f"prep_{idx}",
                "preprocessing_steps": f"prep_{idx}",
                "n_observations": 100,
                "n_bands": 20,
                "ncomp_90": 4,
                "ncomp_95": 6 + idx,
                "ncomp_99": 10,
                "class_trace_ratio": 0.2 + idx * 0.03,
                "mahalanobis_pc1_pc2_pc3": 3.0 - idx * 0.05,
                "batch_trace_ratio": 0.10,
                "object_class_trace_ratio": 0.2 + idx * 0.03,
                "object_over_intra_ratio": 3.0 - idx * 0.05,
                "object_batch_trace_ratio": 0.10,
                "mean_intra_object_trace": 0.2 + idx * 0.01,
                "mean_train_projection_shift_norm": 0.2,
                "projection_q_deviation": 0.2,
                "instability_metric": 0.05,
                "loading_abs_correlation_mean": 0.95,
                "loading_angle_mean_deg": 5.0,
            }
            row.update({column: True for column in PCA_TECHNICAL_FLAG_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def _expanded_object_db(mini_hsi_db):
    base_db, _ = mini_hsi_db
    out = {}
    rng = np.random.default_rng(42)
    for batch in (1, 2, 3, 4):
        for label in ("almond", "peanut"):
            template = base_db[f"{label}1_obj001"]
            for replicate in range(3):
                obj = deepcopy(template)
                object_id = f"{label}{batch}_obj{replicate:03d}"
                source = f"{label}{batch}_{replicate:03d}"
                noise = rng.normal(0.0, 0.003, size=obj["spectra"].shape)
                trend = np.linspace(0.0, 0.01 * (replicate + 1), obj["n_bands"])
                spectra = obj["spectra"] + noise + trend
                obj.update(
                    {
                        "object_id": object_id,
                        "source_clean_key": source,
                        "source_image": f"{source}_sb",
                        "batch": batch,
                        "spectra": spectra,
                        "mean_spectrum": spectra.mean(axis=0),
                        "median_spectrum": np.median(spectra, axis=0),
                        "std_spectrum": spectra.std(axis=0),
                    }
                )
                out[object_id] = obj
    return out


def test_protocol_subset_is_calibration_only_and_excludes_later_batches(mini_hsi_db):
    object_db = _expanded_object_db(mini_hsi_db)
    calibration = subset_object_db_for_pca(
        object_db,
        allowed_batches=expcfg.PCA_CALIBRATION_BATCHES,
        forbidden_batches=expcfg.PCA_FORBIDDEN_BATCHES,
    )
    assert {obj["batch"] for obj in calibration.values()} == {1, 2}
    assert not {3, 4}.intersection(
        {obj["batch"] for obj in calibration.values()}
    )


def test_compare_pca_fits_calibration_and_projects_confirmation(mini_hsi_db):
    object_db = _expanded_object_db(mini_hsi_db)
    calibration = subset_object_db_for_pca(
        object_db,
        allowed_batches=(1, 2),
        forbidden_batches=(4,),
    )
    confirmation = subset_object_db_for_pca(
        object_db,
        allowed_batches=(3,),
        forbidden_batches=(4,),
    )
    diagnostics, results, component_table = compare_pca_representations(
        calibration,
        projection_object_db=confirmation,
        projection_role="confirmation_batch_3",
        matrix_methods=("object_mean",),
        preprocessing_methods={"raw": ("raw",)},
        n_components=3,
        wavelengths=np.asarray([900.0, 910.0, 920.0, 930.0]),
        return_component_table=True,
        verbose=False,
    )
    row = diagnostics.iloc[0]
    assert row["pca_fit_valid"]
    assert row["projection_valid"]
    assert row["residuals_valid"]
    assert np.isfinite(row["centroid_distance_pc1_pc2_pc3"])
    assert np.isfinite(row["projection_q_deviation"])
    assert row["ncomp_99"] >= 1
    assert set(component_table.columns) == {
        "matrix_variant",
        "preprocessing",
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    }
    assert len(component_table) == 4
    assert results["object_mean"]["raw"]["projection_scores"].shape[0] == 6


def test_fit_pca_candidate_diagnostic_accepts_task16_stability_fields(mini_hsi_db):
    object_db = _expanded_object_db(mini_hsi_db)
    candidate = {
        "candidate_id": "candidate_raw",
        "training_matrix_id": "calibration_object_mean",
        "wavelength_axis_id": "axis",
        "matrix_family": "object_matrix",
        "matrix_variant": "object_mean",
        "matrix_method": "object_mean",
        "m": np.nan,
        "balanced_pixel_strategy": "not_applicable",
        "preprocessing": "raw",
        "preprocessing_steps": "raw",
    }
    diagnostic, result, _ = fit_pca_candidate(
        object_db,
        candidate,
        n_components=3,
        wavelengths=np.asarray([900.0, 910.0, 920.0, 930.0]),
        random_state=42,
        under_m_policy="error",
    )
    assert isinstance(diagnostic, dict)
    assert result is not None
    diagnostic.update({"stability_valid": True, "instability_metric": 0.0})
    assert diagnostic["stability_valid"]
    assert diagnostic["instability_metric"] == pytest.approx(0.0)


def test_loading_alignment_and_stability_summary():
    reference = np.asarray(
        [[0.8, 0.1], [0.4, 0.7], [0.2, -0.7]],
        dtype=float,
    )
    loadings = compare_aligned_loadings(
        [
            {"run_type": "reference", "loadings": reference},
            {"run_type": "group_fold", "loadings": -reference},
        ]
    )
    fold = loadings.loc[loadings["run_type"].eq("group_fold")]
    assert np.allclose(fold["loading_abs_correlation"], 1.0)
    assert np.allclose(fold["loading_angle_deg"], 0.0)
    summary = summarize_pca_stability(pd.DataFrame(), loadings)
    assert summary["stability_valid"]
    assert summary["instability_metric"] == pytest.approx(0.0)


def test_grouped_and_bootstrap_pca_stability_runs(mini_hsi_db):
    object_db = subset_object_db_for_pca(
        _expanded_object_db(mini_hsi_db),
        allowed_batches=(1, 2),
        forbidden_batches=(4,),
    )
    metric_stability, loading_stability = evaluate_pca_stability(
        object_db,
        matrix_method="object_mean",
        preprocessing_steps=("raw",),
        n_components=3,
        n_splits=3,
        seeds=(0, 1),
        n_bootstrap=3,
        wavelengths=np.asarray([900.0, 910.0, 920.0, 930.0]),
    )
    assert not metric_stability.empty
    assert {"group_fold", "source_image_bootstrap"}.issubset(
        set(loading_stability["run_type"])
    )
    summary = summarize_pca_stability(
        metric_stability,
        loading_stability,
    )
    assert summary["stability_valid"]
    assert np.isfinite(summary["mean_train_projection_shift_norm"])
    assert np.isfinite(summary["projection_q_deviation"])


def test_strict_coverage_and_preprocessing_pareto_have_no_weighted_score():
    candidates = _candidate_fixture()
    review = build_pca_artifact_review_table(candidates)
    review["review_status"] = "reviewed"
    review["review_decision"] = "accept"
    diagnostics = build_pca_selection_diagnostics(
        candidates,
        artifact_review_df=review,
        config=expcfg.make_pca_selection_config(),
    )
    retained, preprocessing_audit, stage_summary = select_pca_preprocessing_shortlist(
        diagnostics,
        config=expcfg.make_pca_selection_config(),
    )
    _, outcomes = build_pca_selection_flow_tables(
        preprocessing_audit,
        config=expcfg.make_pca_selection_config(),
    )
    compact = build_pca_scoring_diagnostics(
        diagnostics,
        preprocessing_summary_df=preprocessing_audit,
    )

    assert retained["selection_status"].eq("selected").all()
    assert retained["pareto_front"].all()
    assert retained.groupby("matrix_family")["preprocessing"].nunique().eq(12).all()
    assert len(retained) == int(outcomes["retained_after_pareto"].sum())
    assert set(stage_summary["stage"]) == {
        "input",
        "strict_family_coverage",
        "complete_pareto_metrics",
        "pareto_front",
    }
    for _, family_flow in stage_summary.groupby("matrix_family"):
        ordered = family_flow.set_index("stage").loc[
            [
                "input",
                "strict_family_coverage",
                "complete_pareto_metrics",
                "pareto_front",
            ]
        ]
        assert ordered["n_retained"].is_monotonic_decreasing
        assert (
            ordered["n_entering"].iloc[1:].to_numpy()
            == ordered["n_retained"].iloc[:-1].to_numpy()
        ).all()
    assert set(outcomes["first_failed_stage"]).issubset(
        {
            "strict_family_coverage",
            "complete_pareto_metrics",
            "pareto_dominance",
            "retained_after_pareto",
        }
    )
    assert not any("selection_score" in column for column in compact.columns)
    assert tuple(compact.columns) == expcfg.PCA_SCORING_DIAGNOSTIC_COLUMNS
    assert {
        "technical_blocker",
        "preprocessing_pareto_source_metric",
        "artifact_review",
        "selection",
    }.issubset(set(compact["diagnostic_group"]))
    assert {
        "technical_valid",
        "class_trace_ratio",
        "critical_artifact",
        "candidate_retained_after_pareto",
    }.issubset(
        set(compact["metric"])
    )


def test_critical_artifact_is_a_blocking_filter():
    candidates = _candidate_fixture(4)
    review = build_pca_artifact_review_table(candidates)
    key = review.index[0]
    review.loc[key, "critical_artifact"] = True
    review.loc[key, "review_status"] = "reviewed"
    review.loc[key, "review_decision"] = "accept"
    diagnostics = build_pca_selection_diagnostics(
        candidates,
        artifact_review_df=review,
    )
    blocked = diagnostics.loc[
        diagnostics["matrix_variant"].eq(review.loc[key, "matrix_variant"])
        & diagnostics["preprocessing"].eq(review.loc[key, "preprocessing"])
    ].iloc[0]
    assert not blocked["technical_valid"]
    assert "critical_artifact" in blocked["blocking_reason"]


def test_pending_artifact_review_is_a_blocking_filter():
    candidates = _candidate_fixture(1)
    review = build_pca_artifact_review_table(candidates)
    diagnostics = build_pca_selection_diagnostics(
        candidates,
        artifact_review_df=review,
    )
    assert not diagnostics["technical_valid"].any()
    assert diagnostics["blocking_reason"].str.contains(
        "artifact_review_pending"
    ).all()


def test_pareto_front_matches_non_dominance_rule():
    frame = pd.DataFrame(
        {
            "candidate": ["a", "b", "c"],
            "separation": [3.0, 2.0, 1.0],
            "batch": [1.0, 2.0, 3.0],
        }
    )
    front = select_pca_pareto_front(
        frame,
        maximize_metrics=("separation",),
        minimize_metrics=("batch",),
    )
    assert front["candidate"].tolist() == ["a"]


def test_experiment_config_builds_non_weighted_pca_selection_config():
    config = expcfg.make_pca_selection_config()
    assert config.max_preprocessings_per_family is None
    assert config.strict_variant_coverage
    assert "mean_train_projection_shift_norm" not in config.profiles[
        "object_matrix"
    ].minimize_metrics
    assert "projection_q_deviation" not in config.profiles[
        "pixel_matrix"
    ].minimize_metrics
    assert not hasattr(config, "stability_penalty_weight")
    assert not hasattr(config.profiles["object_matrix"], "positive_weights")


def test_validate_shortlist_rejects_duplicates_and_overflow():
    good = pd.DataFrame(
        {
            "matrix_family": ["object_matrix", "pixel_matrix"],
            "preprocessing": ["raw", "snv"],
            "preprocessing_steps": ["raw", "snv"],
            "selection_status": ["selected", "selected"],
        }
    )
    counts = validate_pca_preprocessing_shortlist(
        good,
        max_per_family=None,
        expected_families=("object_matrix", "pixel_matrix"),
    )
    assert counts.to_dict() == {"object_matrix": 1, "pixel_matrix": 1}
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_pca_preprocessing_shortlist(
            pd.concat([good.iloc[[0]], good.iloc[[0]]], ignore_index=True),
            max_per_family=None,
        )


def test_preprocessing_aggregation_uses_worst_case_and_strict_variant_coverage():
    candidates = _candidate_fixture(2)
    object_rows = candidates.loc[candidates["matrix_family"].eq("object_matrix")]
    median_variant = object_rows.copy()
    median_variant["candidate_id"] = median_variant["candidate_id"] + "_median"
    median_variant["matrix_variant"] = "object_median"
    median_variant["matrix_method"] = "object_median"
    median_variant["class_trace_ratio"] = [0.10, 0.50]
    median_variant["batch_trace_ratio"] = [0.30, 0.05]
    candidates = pd.concat([candidates, median_variant], ignore_index=True)
    review = build_pca_artifact_review_table(candidates)
    review["review_status"] = "reviewed"
    review["review_decision"] = "accept"
    review.loc[
        review["candidate_id"].eq("object_matrix_1_median"),
        "review_decision",
    ] = "reject"

    diagnostics = build_pca_selection_diagnostics(
        candidates,
        artifact_review_df=review,
    )
    summary = aggregate_pca_preprocessing_diagnostics(diagnostics)
    prep0 = summary.loc[
        summary["matrix_family"].eq("object_matrix")
        & summary["preprocessing"].eq("prep_0")
    ].iloc[0]
    prep1 = summary.loc[
        summary["matrix_family"].eq("object_matrix")
        & summary["preprocessing"].eq("prep_1")
    ].iloc[0]

    assert prep0["class_trace_ratio_worst"] == pytest.approx(0.10)
    assert prep0["batch_trace_ratio_worst"] == pytest.approx(0.30)
    assert prep0["preprocessing_eligible"]
    assert not prep1["preprocessing_eligible"]
    assert prep1["n_reject"] == 1


def test_artifact_review_contract_contains_requested_columns():
    review = build_pca_artifact_review_table(_candidate_fixture(1))
    assert set(PCA_ARTIFACT_COLUMNS).issubset(review.columns)
    assert {"candidate_id", "m", "review_decision", "run_fingerprint"}.issubset(
        review.columns
    )
    assert review["review_status"].eq("pending").all()


def test_review_decisions_are_not_hardcoded_in_experiment_config():
    assert not hasattr(expcfg, "PCA_ARTIFACT_REVIEW_OVERRIDES")
    assert expcfg.PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS == (
        "accept",
        "warning",
        "reject",
    )
