import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.workflows.pca import add_pca_selection_score
from src.workflows.pca_selection import (
    add_pca_selection_scores,
    build_pca_scoring_diagnostics,
    make_pca_selection_config,
    select_pca_preprocessing_shortlist,
    validate_pca_preprocessing_shortlist,
)


def _pca_summary_fixture(n_per_family=7):
    rows = []
    preprocessings = [f"prep_{idx}" for idx in range(n_per_family)]
    for idx, preprocessing in enumerate(preprocessings):
        rows.append(
            {
                "matrix_family": "object_matrix",
                "matrix_variant": "object_mean",
                "matrix_method": "object_mean",
                "balanced_pixel_strategy": "not_applicable",
                "preprocessing": preprocessing,
                "preprocessing_steps": preprocessing,
                "class_trace_ratio": 0.20 + idx * 0.03,
                "mahalanobis_pc1_pc2_pc3": 1.0 + idx * 0.10,
                "batch_trace_ratio": 0.03 + (n_per_family - idx) * 0.001,
                "mean_train_projection_shift_norm": 0.20 + idx * 0.01,
                "projection_q_deviation": 0.30 + idx * 0.005,
                "ncomp_95": 4 + idx,
                "object_class_trace_ratio": np.nan,
                "object_over_intra_ratio": np.nan,
                "object_batch_trace_ratio": np.nan,
                "mean_intra_object_trace": np.nan,
            }
        )
        rows.append(
            {
                "matrix_family": "pixel_matrix",
                "matrix_variant": "balanced_pixels_random",
                "matrix_method": "balanced_pixels",
                "balanced_pixel_strategy": "random",
                "preprocessing": preprocessing,
                "preprocessing_steps": preprocessing,
                "class_trace_ratio": np.nan,
                "mahalanobis_pc1_pc2_pc3": np.nan,
                "batch_trace_ratio": np.nan,
                "mean_train_projection_shift_norm": 0.25 + idx * 0.01,
                "projection_q_deviation": 0.25 + idx * 0.006,
                "ncomp_95": 5 + idx,
                "object_class_trace_ratio": 0.15 + idx * 0.02,
                "object_over_intra_ratio": 1.20 + idx * 0.05,
                "object_batch_trace_ratio": 0.02 + (n_per_family - idx) * 0.001,
                "mean_intra_object_trace": 0.40 + idx * 0.02,
            }
        )
    return pd.DataFrame(rows)


def test_pca_selection_scores_are_family_specific_and_stability_penalized():
    config = make_pca_selection_config(
        stability_bootstrap_iterations=20,
        stability_penalty_weight=0.5,
        random_state=123,
    )
    scored = add_pca_selection_scores(_pca_summary_fixture(), config=config)

    required = {
        "selection_score",
        "selection_score_without_stability",
        "object_matrix_score",
        "pixel_matrix_score",
        "selection_score_stability_std",
        "selection_score_rank_std",
        "selection_flag",
        "pca_validation_warning",
    }
    assert required.issubset(scored.columns)

    object_rows = scored["matrix_family"].eq("object_matrix")
    pixel_rows = scored["matrix_family"].eq("pixel_matrix")
    assert scored.loc[object_rows, "object_matrix_score"].notna().all()
    assert scored.loc[object_rows, "pixel_matrix_score"].isna().all()
    assert scored.loc[pixel_rows, "pixel_matrix_score"].notna().all()
    assert scored.loc[pixel_rows, "object_matrix_score"].isna().all()
    assert (
        scored["selection_score"]
        <= scored["selection_score_without_stability"] + 1e-12
    ).all()


def test_pca_preprocessing_shortlist_is_limited_per_family():
    config = make_pca_selection_config(
        max_preprocessings_per_family=5,
        stability_bootstrap_iterations=10,
        random_state=123,
    )
    scored = add_pca_selection_scores(_pca_summary_fixture(), config=config)
    scored = scored.sort_values("selection_score", ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1))

    selected, candidate_pool, counts = select_pca_preprocessing_shortlist(
        scored,
        config=config,
    )

    assert counts.to_dict() == {"object_matrix": 5, "pixel_matrix": 5}
    assert len(selected) == 10
    assert candidate_pool.groupby("matrix_family").size().to_dict() == {
        "object_matrix": 7,
        "pixel_matrix": 7,
    }
    assert selected["selection_reason"].str.contains("score_stability_std=").all()
    assert selected.groupby("matrix_family")["family_selection_rank"].max().max() == 5


def test_validate_pca_preprocessing_shortlist_raises_on_family_overflow():
    bad = pd.DataFrame(
        {
            "matrix_family": ["object_matrix"] * 6 + ["pixel_matrix"] * 5,
            "preprocessing": [f"prep_{idx}" for idx in range(11)],
        }
    )

    with pytest.raises(RuntimeError, match="max 5 rows"):
        validate_pca_preprocessing_shortlist(
            bad,
            max_per_family=5,
            expected_families=("object_matrix", "pixel_matrix"),
        )


def test_experiment_config_builds_canonical_pca_selection_config():
    config = expcfg.make_pca_selection_config()

    assert config.max_preprocessings_per_family == expcfg.MAX_PCA_PREPROCESSINGS_PER_FAMILY
    assert config.expected_families == expcfg.PCA_SELECTION_EXPECTED_FAMILIES
    assert config.group_cols == expcfg.PCA_SELECTION_GROUP_COLS
    assert config.stability_bootstrap_iterations == expcfg.PCA_SELECTION_BOOTSTRAP_ITERATIONS
    assert config.stability_penalty_weight == expcfg.PCA_SELECTION_STABILITY_PENALTY_WEIGHT
    assert config.random_state == expcfg.RANDOM_STATE
    assert set(config.profiles) == set(expcfg.PCA_SELECTION_PROFILES)
    assert (
        config.profiles["object_matrix"].positive_weights["class_trace_ratio"]
        == expcfg.PCA_SELECTION_PROFILES["object_matrix"]["positive_weights"]["class_trace_ratio"]
    )


def test_pca_scoring_diagnostics_table_is_separate_and_explanatory():
    config = make_pca_selection_config(
        stability_bootstrap_iterations=10,
        random_state=123,
    )
    scored = add_pca_selection_scores(_pca_summary_fixture(), config=config)
    scored = scored.sort_values("selection_score", ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1))

    diagnostics = build_pca_scoring_diagnostics(scored, config=config)

    required = {
        "rank",
        "matrix_family",
        "matrix_variant",
        "preprocessing",
        "selection_score",
        "selection_score_without_stability",
        "selection_score_stability_penalty",
        "selection_score_stability_std",
        "selection_flag",
        "pca_validation_warning",
        "contrib_plus_class_trace_ratio",
        "contrib_minus_batch_trace_ratio",
    }
    assert required.issubset(diagnostics.columns)
    assert diagnostics["rank"].tolist() == sorted(diagnostics["rank"].tolist())
    assert len(diagnostics) == len(scored)


def test_legacy_pca_selection_score_wrapper_delegates_to_selection_module():
    scored = add_pca_selection_score(
        _pca_summary_fixture(),
        group_col="matrix_variant",
    )

    assert "selection_score" in scored.columns
    assert "selection_score_without_stability" in scored.columns
    assert "object_matrix_score" in scored.columns
    assert "pixel_matrix_score" in scored.columns
    assert scored["selection_score_stability_penalty"].eq(0.0).all()
