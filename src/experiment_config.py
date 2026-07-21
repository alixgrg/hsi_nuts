"""Central experiment configuration for the HSI nuts workflow.

The notebooks keep local variable names for readability, but those variables
should be initialized from this module so the train/validation/test protocol and
main search grids stay consistent across the project.
"""

from __future__ import annotations

from typing import Any


# Spectral/result namespace
DEFAULT_WAVELENGTH_MODE = "non_noisy_all"
DEFAULT_RESULTS_TAG = "non_noisy_all"


# Detection task
TARGET_CLASS = "peanut"
NON_TARGET_LABEL = "almond"
REFERENCE_CLASSES = ("almond", TARGET_CLASS)


# Dataset split protocol
PCA_ALLOWED_BATCHES = [1, 2, 3]
SIMCA_TRAIN_BATCHES = [1, 2]
SIMCA_VALIDATION_BATCHES = [3]
PURE_TEST_TRAIN_BATCHES = [1, 2, 3]
PURE_TEST_BATCHES = [4]
MIXTURE_FINAL_TRAIN_BATCHES = [1, 2, 3, 4]


# Main search-grid reduction agreed after the audit
SIMCA_ALPHA_VALUES = [0.01]
SIMCA_OBJECT_THRESHOLDS = [0.75, 0.80]


# SIMCA matrix families and final selection tracks
SIMCA_OBJECT_MATRIX_METHODS = ("object_mean", "object_median")
SIMCA_PIXEL_MATRIX_METHODS = ("balanced_pixels", "all_pixels", "pixel")
SIMCA_MATRIX_FAMILIES = ("object_matrix", "pixel_matrix")
SIMCA_DECISION_MODES = ("2way", "3way")

SIMCA_MATRIX_METHOD_FAMILY = {
    **{method: "object_matrix" for method in SIMCA_OBJECT_MATRIX_METHODS},
    **{method: "pixel_matrix" for method in SIMCA_PIXEL_MATRIX_METHODS},
}

SIMCA_SELECTION_TRACKS = (
    "object_matrix_2way",
    "object_matrix_3way",
    "pixel_matrix_2way",
    "pixel_matrix_3way",
)

SIMCA_SELECTION_TRACK_SPECS = {
    "object_matrix_2way": {
        "matrix_family": "object_matrix",
        "decision_mode": "2way",
        "primary_metric_level": "object",
        "secondary_metric_level": "pixel",
    },
    "object_matrix_3way": {
        "matrix_family": "object_matrix",
        "decision_mode": "3way",
        "primary_metric_level": "object",
        "secondary_metric_level": "pixel",
    },
    "pixel_matrix_2way": {
        "matrix_family": "pixel_matrix",
        "decision_mode": "2way",
        "primary_metric_level": "pixel",
        "secondary_metric_level": "object",
    },
    "pixel_matrix_3way": {
        "matrix_family": "pixel_matrix",
        "decision_mode": "3way",
        "primary_metric_level": "pixel",
        "secondary_metric_level": "object",
    },
}


# Notebook 02 result-table contracts
MATRIX_SUMMARY_REQUIRED_COLUMNS = (
    "matrix_method",
    "balanced_pixel_strategy",
    "filters",
    "n_observations",
    "n_features",
    "n_labels",
    "labels",
    "has_metadata",
    "n_unique_objects",
    "n_unique_images",
    "n_nan_values",
    "nan_rate",
    "global_min",
    "global_max",
    "global_mean",
    "global_std",
)

PREPROCESSING_SUMMARY_REQUIRED_COLUMNS = (
    "preprocessing",
    "steps",
    "n_observations",
    "n_features",
    "global_mean",
    "global_std",
    "global_min",
    "global_max",
    "nan_rate",
    "sg_window_length",
    "sg_polyorder",
)


# SIMCA candidate/result-table contracts
SIMCA_PCA_SHORTLIST_REQUIRED_COLUMNS = (
    "matrix_family",
    "preprocessing",
    "preprocessing_steps",
)

SIMCA_CANDIDATE_ID_COLUMNS = (
    "matrix_family",
    "matrix_method",
    "training_matrix_id",
    "balanced_pixel_strategy",
    "balanced_pixel_strategy_effective",
    "m",
    "m_effective",
    "preprocessing",
    "preprocessing_steps",
    "model_family",
    "rule",
    "rule_variant",
    "selected_rule_name",
    "rule_for_refit",
    "limit_source",
    "decision_mode",
    "n_components",
    "alpha",
    "object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "lower_threshold",
    "upper_threshold",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "target_class",
    "non_target_label",
)

SIMCA_REFIT_CONFIG_DEDUP_COLUMNS = (
    "target_class",
    "non_target_label",
    "object_threshold",
    "matrix_family",
    "matrix_method",
    "training_matrix_id",
    "m_effective",
    "balanced_pixel_strategy_effective",
    "preprocessing",
    "preprocessing_steps",
    "rule_for_refit",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
)

SIMCA_METRIC_EQUIVALENCE_METRIC_COLUMNS = (
    "n",
    "tp",
    "fn",
    "fp",
    "tn",
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
)

SIMCA_METRIC_EQUIVALENCE_PROTECTED_COLUMNS = (
    "target_class",
    "non_target_label",
    "matrix_family",
)

SIMCA_METRIC_EQUIVALENCE_PARAMETER_GROUPS = {
    "object_threshold": ("object_threshold",),
    "matrix_method": ("matrix_method", "training_matrix_id"),
    "pixel_sampling": (
        "m_effective",
        "balanced_pixel_strategy_effective",
    ),
    "preprocessing": ("preprocessing", "preprocessing_steps"),
    "rule": ("rule_for_refit", "limit_source"),
    "n_components": ("n_components",),
    "alpha": ("alpha",),
    "savgol": ("sg_window_length", "sg_polyorder"),
    "position_dilation": ("position_dilation_radius",),
}

SIMCA_CANDIDATE_CONFIG_REQUIRED_COLUMNS = (
    "candidate_id",
    "candidate_sources",
    "matrix_family",
    "matrix_method",
    "preprocessing",
    "preprocessing_steps",
    "model_family",
    "rule_variant",
    "n_components",
    "alpha",
    "object_threshold",
    "target_class",
    "non_target_label",
)

SIMCA_CANDIDATE_EVALUATION_REQUIRED_COLUMNS = (
    "candidate_id",
    "selection_track",
    "matrix_family",
    "decision_mode",
    "evaluation_stage",
    "metric_level",
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
)

SIMCA_FINAL_MODEL_SELECTION_REQUIRED_COLUMNS = (
    "candidate_id",
    "selection_track",
    "matrix_family",
    "decision_mode",
    "rank",
    "selection_score",
    "selection_reason",
)


# PCA shortlist policy
MAX_PCA_PREPROCESSINGS_PER_FAMILY = 5


# PCA scoring policy
PCA_SELECTION_EXPECTED_FAMILIES = ("object_matrix", "pixel_matrix")
PCA_SELECTION_GROUP_COLS = ("matrix_variant",)
PCA_SELECTION_ROBUST_SCALING = True
PCA_SELECTION_CLIP_QUANTILES = (0.05, 0.95)
PCA_SELECTION_QUALITY_LOWER_QUANTILE = 0.25
PCA_SELECTION_QUALITY_UPPER_QUANTILE = 0.75
PCA_SELECTION_VALIDATION_UPPER_QUANTILE = 0.75
PCA_SELECTION_BOOTSTRAP_ITERATIONS = 100
PCA_SELECTION_STABILITY_PENALTY_WEIGHT = 0.25
PCA_SELECTION_EPS = 1e-12

PCA_SELECTION_PROFILES = {
    "object_matrix": {
        "positive_weights": {
            "class_trace_ratio": 3.0,
            "mahalanobis_pc1_pc2_pc3": 1.0,
        },
        "negative_weights": {
            "batch_trace_ratio": 2.0,
            "mean_train_projection_shift_norm": 1.5,
            "projection_q_deviation": 1.5,
            "ncomp_95": 0.3,
        },
        "separation_metric": "class_trace_ratio",
        "batch_metric": "batch_trace_ratio",
        "projection_metric": "projection_q_deviation",
        "validation_metric": "mean_train_projection_shift_norm",
    },
    "pixel_matrix": {
        "positive_weights": {
            "object_class_trace_ratio": 3.0,
            "object_over_intra_ratio": 1.0,
        },
        "negative_weights": {
            "object_batch_trace_ratio": 2.0,
            "mean_intra_object_trace": 1.0,
            "mean_train_projection_shift_norm": 1.2,
            "projection_q_deviation": 1.2,
            "ncomp_95": 0.3,
        },
        "separation_metric": "object_class_trace_ratio",
        "batch_metric": "object_batch_trace_ratio",
        "projection_metric": "projection_q_deviation",
        "validation_metric": "mean_train_projection_shift_norm",
    },
}


# Shared runtime defaults
RANDOM_STATE = 42
REPLACE_BALANCED_PIXELS = False
CV_N_SPLITS = 5
CV_GROUP_COL = "object_id"
M_BALANCED_PIXELS = 40
BALANCED_PIXEL_STRATEGIES = ["random", "center"]


def make_pca_selection_config(**overrides: Any):
    """Return the canonical PCA selection configuration for this project."""
    from src.workflows.pca_selection import (
        PCASelectionProfile,
        make_pca_selection_config as _make_pca_selection_config,
    )

    profiles = {
        family: PCASelectionProfile(
            positive_weights=dict(profile["positive_weights"]),
            negative_weights=dict(profile["negative_weights"]),
            separation_metric=str(profile["separation_metric"]),
            batch_metric=str(profile["batch_metric"]),
            projection_metric=str(profile["projection_metric"]),
            validation_metric=str(profile["validation_metric"]),
        )
        for family, profile in PCA_SELECTION_PROFILES.items()
    }
    config_kwargs = {
        "profiles": profiles,
        "group_cols": PCA_SELECTION_GROUP_COLS,
        "max_preprocessings_per_family": MAX_PCA_PREPROCESSINGS_PER_FAMILY,
        "expected_families": PCA_SELECTION_EXPECTED_FAMILIES,
        "robust": PCA_SELECTION_ROBUST_SCALING,
        "clip_quantiles": PCA_SELECTION_CLIP_QUANTILES,
        "quality_lower_quantile": PCA_SELECTION_QUALITY_LOWER_QUANTILE,
        "quality_upper_quantile": PCA_SELECTION_QUALITY_UPPER_QUANTILE,
        "validation_upper_quantile": PCA_SELECTION_VALIDATION_UPPER_QUANTILE,
        "stability_bootstrap_iterations": PCA_SELECTION_BOOTSTRAP_ITERATIONS,
        "stability_penalty_weight": PCA_SELECTION_STABILITY_PENALTY_WEIGHT,
        "random_state": RANDOM_STATE,
        "eps": PCA_SELECTION_EPS,
    }
    config_kwargs.update(overrides)
    return _make_pca_selection_config(**config_kwargs)
