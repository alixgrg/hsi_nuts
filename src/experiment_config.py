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
MIXTURE_APPLICATION_EVALUATION_STAGE = "mixture_application"
MIXTURE_APPLICATION_BATCH_SIZE = 10
MIXTURE_APPLICATION_MAX_MODELS_PER_TRACK = None
MIXTURE_APPLICATION_SAVE_BATCH_METRIC_TABLES = True
MIXTURE_APPLICATION_SAVE_BATCH_OBJECT_TABLES = False
MIXTURE_APPLICATION_SAVE_BATCH_PIXEL_TABLES = False
MIXTURE_APPLICATION_SAVE_BATCH_3WAY_OBJECT_TABLES = False
MIXTURE_APPLICATION_SAVE_COMBINED_OBJECT_TABLES = True
MIXTURE_APPLICATION_SAVE_COMBINED_PIXEL_TABLES = True
MIXTURE_APPLICATION_SAVE_COMBINED_3WAY_OBJECT_TABLES = True
MIXTURE_APPLICATION_KEEP_ONLY_ASSIGNED_TRACK_METRICS = True
MIXTURE_APPLICATION_DIAGNOSTIC_TOP_IMAGES = 3


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
    "final_rank_in_track",
    "pareto_tier",
    "selection_reason",
)


# SIMCA notebook 06B final multi-model selection
# None keeps every model that survives both Pareto filters. Set an integer only
# when an explicit display/export cap is desired.
SIMCA_FINAL_TOP_N_PER_TRACK = None
SIMCA_FINAL_APPLY_DIVERSITY = False
SIMCA_FINAL_DIVERSITY_COLUMNS = (
    "preprocessing",
    "rule_for_refit",
    "balanced_pixel_strategy_effective",
)
SIMCA_FINAL_DEDUPLICATE_ACROSS_TRACKS = False
SIMCA_FINAL_CROSS_TRACK_DEDUP_COL = "selected_config_id"

SIMCA_FINAL_2WAY_PARETO_MINIMIZE_COLUMNS = ("fn_rate", "fp_rate")
SIMCA_FINAL_3WAY_PARETO_MINIMIZE_COLUMNS = (
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
)
SIMCA_FINAL_TIEBREAK_2WAY_COLUMNS = (
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
    "selected_config_id",
)
SIMCA_FINAL_TIEBREAK_3WAY_COLUMNS = (
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "decided_balanced_accuracy",
    "selected_config_id",
)

SIMCA_FINAL_APPLY_PREVIOUS_FLAG_FILTER = False
SIMCA_FINAL_PREVIOUS_FLAG_COLUMNS = (
    "review_flags",
    "robustness_flags",
    "stability_flags",
)
SIMCA_FINAL_PREVIOUS_FLAGS_TO_FILTER = ()
SIMCA_FINAL_EXCLUDE_PURE_TEST_ERRORS = True
SIMCA_FINAL_FN_RATE_MAX = 0.5
SIMCA_FINAL_FP_RATE_MAX = 0.9
SIMCA_FINAL_UNCERTAIN_RATE_MAX = 0.6


# SIMCA notebook 05 robustness diagnostics
SIMCA_ROBUSTNESS_RANDOM_STATES = (0, 1, 2, 3, 4, 5, 10, 20, 42, 100)
SIMCA_ROBUSTNESS_MAX_STABILITY_CANDIDATES_PER_TRACK = 12
SIMCA_ROBUSTNESS_PREFER_BALANCED_PIXELS_FOR_STABILITY = True
SIMCA_ROBUSTNESS_BORDER_WIDTHS = (0, 1, 2, 3, 4)
SIMCA_ROBUSTNESS_MIN_CORE_PIXELS = 20
SIMCA_ROBUSTNESS_PARETO_EPSILON = 1e-12

SIMCA_ROBUSTNESS_WARNING_THRESHOLDS = {
    "fn_rate": 0.05,
    "fp_rate": 0.20,
    "balanced_accuracy": 0.80,
    "target_miss_rate": 0.05,
    "non_target_false_accept_rate": 0.20,
    "uncertain_rate": 0.25,
    "coverage_rate": 0.75,
    "decided_balanced_accuracy": 0.80,
    "std_fn_rate": 0.03,
    "std_fp_rate": 0.05,
    "std_balanced_accuracy": 0.03,
}

SIMCA_ROBUSTNESS_2WAY_SCORE_WEIGHTS = {
    "fn_rate": -10.0,
    "fp_rate": -2.0,
    "balanced_accuracy": 2.0,
}

SIMCA_ROBUSTNESS_3WAY_SCORE_WEIGHTS = {
    "target_miss_rate": -10.0,
    "non_target_false_accept_rate": -2.0,
    "uncertain_rate": -0.75,
    "coverage_rate": 1.0,
    "screening_sensitivity": 1.0,
    "decided_balanced_accuracy": 2.0,
}

SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS = (
    "matrix_method",
    "training_matrix_id",
    "preprocessing",
    "rule_for_refit",
    "limit_source",
    "n_components",
    "object_threshold",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "balanced_pixel_strategy_effective",
)

SIMCA_ROBUSTNESS_2WAY_PARETO_MINIMIZE_COLUMNS = ("fn_rate", "fp_rate")
SIMCA_ROBUSTNESS_2WAY_PARETO_MAXIMIZE_COLUMNS = ("balanced_accuracy",)
SIMCA_ROBUSTNESS_3WAY_PARETO_MINIMIZE_COLUMNS = (
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
)
SIMCA_ROBUSTNESS_3WAY_PARETO_MAXIMIZE_COLUMNS = (
    "coverage_rate",
    "screening_sensitivity",
    "decided_balanced_accuracy",
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
