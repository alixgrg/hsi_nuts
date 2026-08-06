"""Central experiment configuration for the HSI nuts workflow.

The notebooks keep local variable names for readability, but those variables
should be initialized from this module so the train/validation/test protocol and
main search grids stay consistent across the project.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Canonical paths for notebooks 00-03
# ---------------------------------------------------------------------------
RAW_MAT_RELATIVE_PATH = (
    "HSI Data",
    "NIR camera UCO (889-1702 nm)",
    "NIR_uco_sb.mat",
)
PROTOCOL_ARTIFACT_RELATIVE_DIR = ("docs", "protocol")
DATABASE_H5_RELATIVE_PATH = ("HSI Data", "processed", "nir_uco_database.h5")
DATABASE_RESULTS_RELATIVE_DIR = ("results", "00_database")
QC_RESULTS_RELATIVE_DIR = ("results", "01_quality_check")
SPATIAL_GT_RESULTS_RELATIVE_DIR = ("results", "01B_spatial_ground_truth")
MATRIX_RESULTS_DIR_PREFIX = "02_matrices"
PCA_RESULTS_DIR_PREFIX = "03_pca"
INTERNAL_CALIBRATION_RESULTS_DIR_PREFIX = "03B_internal_calibration_8tracks_v3"
DOMAIN_SPATIAL_CALIBRATION_RESULTS_DIR_PREFIX = (
    "03C_projection_spatial_calibration"
)
SIMCA_GRID_SEARCH_RESULTS_DIR_PREFIX = "04A_simca_grid_search_8tracks_v3"
SIMCA_OPTUNA_RESULTS_DIR_PREFIX = "04B_simca_optuna_search_8tracks_v3"
SIMCA_CONCAT_REFIT_RESULTS_DIR_PREFIX = "04C_simca_concat_refit_8tracks_v3"

PROTOCOL_OUTPUT_FILENAMES = {
    "manifest": "protocol_manifest.parquet",
    "checks": "protocol_checks.parquet",
    "inference_plan": "inference_plan.json",
    "planned_contrasts": "planned_contrasts.parquet",
    "lock": "protocol_lock.json",
}
DATABASE_OUTPUT_FILENAMES = {
    "raw_image_manifest": "raw_image_manifest.parquet",
    "metadata_parsing_errors": "metadata_parsing_errors.parquet",
    "image_summary": "image_summary.parquet",
    "object_summary": "object_summary.parquet",
    "segmentation_diagnostics": "segmentation_diagnostics.parquet",
    "manifest": "database_manifest.parquet",
}
QC_OUTPUT_FILENAMES = {
    "image_summary": "image_qc_summary.parquet",
    "object_summary": "object_qc_summary.parquet",
    "alerts": "qc_alerts.parquet",
    "review": "qc_review.parquet",
    "exclusion_manifest": "exclusion_manifest.parquet",
    "protocol": "qc_protocol.parquet",
    "split_manifest": "protocol_split_manifest.parquet",
    "split_diagnostics": "split_diagnostics.parquet",
    "visual_review": "qc_visual_review_report.pdf",
}
SPATIAL_GT_OUTPUT_FILENAMES = {
    "manifest": "spatial_ground_truth_manifest.parquet",
    "components": "fragment_reference_components.parquet",
    "agreement": "annotation_agreement.parquet",
    "adjudication": "annotation_adjudication.parquet",
    "lock": "spatial_ground_truth_lock.json",
    "annotation_protocol": "spatial_annotation_protocol.json",
}
MATRIX_OUTPUT_FILENAMES = {
    "wavelength_config": "wavelength_config.parquet",
    "m_feasibility": "m_feasibility.parquet",
    "pixel_sampling_diagnostics": "pixel_sampling_diagnostics.parquet",
    "matrix_summary": "matrix_summary.parquet",
    "matrix_coverage": "matrix_coverage.parquet",
    "matrix_errors": "matrix_errors.parquet",
    "preprocessing_validation": "preprocessing_validation.parquet",
    "preprocessing_errors": "preprocessing_errors.parquet",
}
PCA_OUTPUT_FILENAMES = {
    "summary": "pca_summary.parquet",
    "diagnostics": "pca_scoring_diagnostics.parquet",
    "preprocessing_summary": "pca_preprocessing_summary.parquet",
    "selected": "pca_selected_preprocessings.parquet",
    "artifact_review": "pca_artifact_review.parquet",
    "visual_review": "pca_visual_review.pdf",
}
INTERNAL_CALIBRATION_OUTPUT_FILENAMES = {
    "track_contracts": "track_contracts.parquet",
    "folds": "internal_calibration_folds.parquet",
    "fold_diagnostics": "fold_diagnostics.parquet",
    "fit_diagnostics": "fit_diagnostics.parquet",
    "rule_diagnostics": "rule_diagnostics.parquet",
    "oof_object_predictions": "oof_object_predictions.parquet",
    "oof_pixel_predictions": "oof_pixel_predictions.parquet",
    "projection_shift": "projection_shift.parquet",
    "oof_2way_metrics": "oof_2way_metrics.parquet",
    "pixel_to_object_thresholds_2way": (
        "pixel_to_object_thresholds_2way.parquet"
    ),
    "thresholds_3way": "thresholds_3way.parquet",
    "thresholds_3way_study": "thresholds_3way_study.parquet",
    "calibrated_hyperparameters": "calibrated_hyperparameters.parquet",
    "calibration_audit": "calibration_audit.parquet",
    "calibration_domain": "calibration_domain.parquet",
    "checkpoint_manifest": "checkpoint_manifest.json",
}
DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES = {
    "projection_shift_diagnostics": "projection_shift_diagnostics.parquet",
    "projection_eligibility": "projection_eligibility.parquet",
    "spatial_calibration_metrics": "spatial_calibration_metrics.parquet",
    "fragment_size_classes": "fragment_size_classes.parquet",
    "spatial_postprocessing_lock": "spatial_postprocessing_lock.json",
}
SIMCA_GRID_SEARCH_OUTPUT_FILENAMES = {
    "configurations": "grid_configurations.parquet",
    "fold_metrics": "grid_fold_metrics.parquet",
    "threshold_metrics": "grid_threshold_metrics.parquet",
    "pareto_reference": "grid_pareto_reference.parquet",
    "technical_audit": "technical_audit.parquet",
    "duplicate_groups": "duplicate_groups.parquet",
    "calculable_not_acceptable": "calculable_not_acceptable.parquet",
    "protocol": "grid_protocol.json",
}
SIMCA_OPTUNA_OUTPUT_FILENAMES = {
    "trials": "optuna_trials.parquet",
    "pareto_candidates": "optuna_pareto_candidates.parquet",
    "search_efficiency": "optuna_search_efficiency.parquet",
    "errors": "optuna_errors.parquet",
    "ablation_plan": "preregistered_ablation_plan.parquet",
    "protocol": "optuna_protocol.json",
}
SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES = {
    "object_predictions": "validation_object_predictions.parquet",
    "pixel_predictions": "validation_pixel_predictions.parquet",
    "metrics": "validation_metrics.parquet",
    "pixel_maps_manifest": "pixel_maps_manifest.parquet",
    "spatial_components": "spatial_components.parquet",
    "spatial_component_metrics": "spatial_component_metrics.parquet",
    "guardrails": "validation_guardrails.parquet",
    "protocol": "validation_protocol.json",
}


# ---------------------------------------------------------------------------
# Canonical low-level database construction
# ---------------------------------------------------------------------------
DEFAULT_WAVELENGTH_MODE = "non_noisy_all"
DEFAULT_RESULTS_TAG = "non_noisy_all"
SPECTRAL_START_NM = 889.0
SPECTRAL_END_NM = 1702.0
N_BANDS_RAW = 69
N_REMOVE_START = 6
N_STOP_END = None
DATA_MODE = "reflectance"

OBJECT_MIN_AREA = 10
DATABASE_FORCED_SPLIT = None
DATABASE_SKIP_UNKNOWN = False
DATABASE_SELECTED_KEYS = None
DATABASE_INCLUDE_HEAVY_OBJECT_ARRAYS = False
DATABASE_OVERWRITE_OUTPUTS = True
SEGMENTATION_OVERRIDE_RELATIVE_DIR = (
    "docs",
    "protocol",
    "segmentation_overrides",
)

SEGMENTATION_KWARGS = {
    "reference_method": "max",
    "threshold_method": "fixed",
    "tau_min": 0.02,
    "opening_radius": 0,
    "closing_radius": 1,
    "fill_holes": True,
    "min_distance": 10,
    "min_area": OBJECT_MIN_AREA,
    "use_watershed": False,
}
SEGMENTATION_MERGE_WARNING_THRESHOLDS = {
    "min_fill_ratio": 0.45,
    "min_separation_pixels": 2.0,
}

IMAGE_STATUS_VALUES = (
    "accepted",
    "warning",
    "excluded",
    "corrected_segmentation",
)

DATABASE_RUN_QC_PLOTS = True
DATABASE_N_QC_IMAGES = 3
DATABASE_N_QC_OBJECTS = 20


# ---------------------------------------------------------------------------
# Notebook 01 quality-control policy
# ---------------------------------------------------------------------------
QC_POLICY = {
    "min_area_pixels": OBJECT_MIN_AREA,
    "max_invalid_pixel_rate": 0.0,
    "exclude_empty_mask": True,
    "merge_warning_thresholds": SEGMENTATION_MERGE_WARNING_THRESHOLDS,
    "zero_variance_epsilon": 1e-12,
    "spectral_outlier_distance_threshold": 6.0,
}
QC_SPECTRAL_GROUP_COLUMNS = (
    "sample_kind",
    "object_nut_type",
    "batch",
)
QC_SPECTRAL_OUTLIER_DISTANCE_THRESHOLD = 6.0
QC_ZERO_VARIANCE_EPSILON = 1e-12
QC_BORDER_MARGIN = 0
QC_RUN_SEGMENTATION_PLOTS = True
QC_RUN_OBJECT_PLOTS = True
QC_RUN_SPECTRAL_PLOTS = True
QC_RUN_RAW_DATABASE_COMPARISON = False
QC_RECONSTRUCT_HEAVY_OBJECT_ARRAYS = True
QC_EXAMPLE_IMAGES_PER_BATCH = 2
QC_OBJECTS_IN_GRID = 25
QC_MAX_OBJECT_SPECTRA_PER_GROUP = 80
QC_REVIEW_ALLOWED_DECISIONS = (
    "accept_as_is",
    "exclude",
    "correct_segmentation",
)
QC_REVIEW_REQUIRED_STATUS = "reviewed"
QC_REVIEW_DECISIONS_RELATIVE_PATH = (
    "docs",
    "protocol",
    "qc_review_decisions.parquet",
)


# ---------------------------------------------------------------------------
# Notebook 01B spatial ground-truth and annotation lock
# ---------------------------------------------------------------------------
SPATIAL_GT_ALLOWED_LEVELS = (
    "object_exact",
    "pixel_annotated",
    "weak_object_label",
    "indirect",
)
SPATIAL_GT_PRIMARY_PIXEL_LEVELS = ("pixel_annotated",)
SPATIAL_GT_ANNOTATION_RELATIVE_DIR = (
    "HSI Data",
    "annotations",
    "spatial_gt_v1",
)
SPATIAL_GT_ANNOTATION_PROTOCOL_RELATIVE_PATH = (
    "docs",
    "protocol",
    "spatial_annotation_protocol.json",
)
SPATIAL_GT_ANNOTATION_TOOL = "matplotlib_lasso_mask_editor"
SPATIAL_GT_ANNOTATION_TOOL_VERSION = "1.0"
SPATIAL_GT_ANNOTATION_PROTOCOL_VERSION = "spatial_gt_v1"
SPATIAL_GT_DOUBLE_ANNOTATION_POLICY = "all_selected_images"
SPATIAL_GT_DOUBLE_ANNOTATION_FRACTION = 1.00
SPATIAL_GT_ANNOTATION_FRACTION = 0.30
SPATIAL_GT_TEST_BATCHES = (4,)
SPATIAL_GT_TARGET_CLASS = "peanut"
SPATIAL_GT_ANNOTATED_CLASS = "peanut"
SPATIAL_GT_POSITIVE_VALUE = 1
SPATIAL_GT_POSITIVE_CLASS = "peanut"
SPATIAL_GT_POSITIVE_DEFINITION = (
    "Peanut tissue occupies at least 50% of the pixel footprint."
)
SPATIAL_GT_NEGATIVE_VALUE = 0
SPATIAL_GT_NEGATIVE_DEFINITION = (
    "No visible peanut tissue within a valid annotated pixel."
)
SPATIAL_GT_OUTSIDE_ROI_DEFINITION = (
    "Not evaluated; never counted as a valid negative."
)
SPATIAL_GT_MASK_SEMANTICS_ID = "binary_peanut_presence"
SPATIAL_GT_BOUNDARY_POLICY_ID = "majority_pixel_area"
SPATIAL_GT_AMBIGUITY_POLICY_ID = "separate_validity_mask"
SPATIAL_GT_ROI_SOURCE = "image_db.labels>0"
SPATIAL_GT_COMPONENT_CONNECTIVITY = 2
SPATIAL_AGREEMENT_MIN_DICE = 0.85
SPATIAL_AGREEMENT_MIN_IOU = 0.75
SPATIAL_AGREEMENT_MAX_UNMATCHED_COMPONENT_RATE = 0.10


# ---------------------------------------------------------------------------
# Compact output contracts for notebooks 00-02
# ---------------------------------------------------------------------------
DATABASE_IMAGE_SUMMARY_COLUMNS = (
    "clean_key",
    "sample_kind",
    "nut_type",
    "batch",
    "image_status",
    "n_objects",
    "height",
    "width",
    "n_bands",
    "mask_area_ratio",
)

DATABASE_OBJECT_SUMMARY_COLUMNS = (
    "object_id",
    "source_image",
    "sample_kind",
    "object_nut_type",
    "batch",
    "object_status",
    "area_pixels",
    "n_pixels",
    "n_bands",
)

SEGMENTATION_DIAGNOSTIC_COLUMNS = (
    "clean_key",
    "label_id",
    "area_pixels",
    "bbox_area",
    "fill_ratio",
    "touches_border",
    "nearest_object_distance",
    "segmentation_action",
    "segmentation_status",
)

RAW_IMAGE_MANIFEST_COLUMNS = (
    "original_key",
    "clean_key",
    "sample_kind",
    "scientific_role",
    "nut_type",
    "batch",
    "components_json",
    "height",
    "width",
    "n_bands",
    "dtype",
    "n_nan",
    "n_inf",
    "metadata_status",
)

METADATA_PARSING_ERROR_COLUMNS = (
    "original_key",
    "clean_key",
    "metadata_status",
    "metadata_error",
)

DATABASE_MANIFEST_COLUMNS = (
    "database_id",
    "wavelength_mode",
    "data_mode",
    "n_images",
    "n_objects",
    "n_bands",
    "wavelength_min_nm",
    "wavelength_max_nm",
    "hdf5_valid",
    "validation_failures",
    "h5_schema_version",
    "protocol_version",
    "database_content_sha256",
    "h5_file_sha256",
)

IMAGE_QC_OUTPUT_COLUMNS = (
    "clean_key",
    "sample_kind",
    "nut_type",
    "batch",
    "height",
    "width",
    "n_bands",
    "n_objects",
    "n_labels",
    "mask_area_ratio",
    "n_valid_pixels",
    "n_nan",
    "n_inf",
    "zero_variance_band_rate",
    "axis_matches_reference",
    "image_status",
)

OBJECT_QC_OUTPUT_COLUMNS = (
    "object_id",
    "source_image",
    "sample_kind",
    "object_nut_type",
    "batch",
    "area_pixels",
    "n_pixels",
    "n_bands",
    "n_valid_pixels",
    "n_nan",
    "n_inf",
    "zero_variance_band_rate",
    "spectral_robust_distance",
    "spectral_outlier",
    "mean_spectrum_mean",
    "median_spectrum_mean",
    "std_spectrum_mean",
    "bbox_fill_ratio",
    "touches_border",
    "nearest_object_distance",
    "possible_merged_object",
    "too_small",
    "requires_segmentation_review",
    "object_status",
)

QC_ALERT_OUTPUT_COLUMNS = (
    "alert_id",
    "record_type",
    "record_id",
    "flag_type",
    "severity",
    "qc_status",
    "exclusion_reason",
    "requires_segmentation_review",
    "warning",
    "evidence_json",
)
QC_REVIEW_OUTPUT_COLUMNS = (
    "record_type",
    "record_id",
    "flag_type",
    "review_status",
    "review_decision",
    "reviewer",
    "review_date",
    "review_comment",
    "review_evidence",
)
QC_EXCLUSION_OUTPUT_COLUMNS = (
    "record_type",
    "record_id",
    "qc_status",
    "exclusion_reason",
    "requires_segmentation_review",
)
QC_PROTOCOL_OUTPUT_COLUMNS = (
    "protocol_version",
    "qc_policy_hash",
    "alerts_hash",
    "review_hash",
    "n_alerts",
    "n_pending",
    "n_excluded",
    "closure_status",
)
PROTOCOL_SPLIT_MANIFEST_COLUMNS = (
    "source_image",
    "object_id",
    "batch",
    "label",
    "sample_kind",
    "protocol_role",
    "cv_group",
    "qc_eligibility",
)
PROTOCOL_SPLIT_CHECK_COLUMNS = ("check", "passed", "detail")
SPLIT_DIAGNOSTIC_COLUMNS = (
    "protocol_role",
    "label",
    "batch",
    "n_objects",
    "n_images",
    "area_min",
    "area_median",
    "area_max",
)
SPATIAL_GT_MANIFEST_COLUMNS = (
    "reference_id",
    "source_image",
    "source_class",
    "annotator_id",
    "truth_level",
    "target_class",
    "annotated_class",
    "positive_value",
    "positive_class",
    "positive_definition",
    "negative_value",
    "negative_definition",
    "outside_roi_definition",
    "mask_semantics_id",
    "boundary_policy_id",
    "ambiguity_policy_id",
    "annotation_tool",
    "annotation_tool_version",
    "annotation_protocol_version",
    "annotation_protocol_sha256",
    "annotation_date",
    "roi_source",
    "roi_mask_path",
    "roi_sha256",
    "target_mask_path",
    "target_mask_sha256",
    "validity_mask_path",
    "validity_mask_sha256",
    "metadata_path",
    "metadata_sha256",
    "n_roi_pixels",
    "n_valid_pixels",
    "n_positive_pixels",
    "n_ambiguous_pixels",
    "status",
)
SPATIAL_GT_COMPONENT_COLUMNS = (
    "reference_id",
    "component_id",
    "area_pixels",
    "centroid_row",
    "centroid_col",
    "bbox_json",
)
SPATIAL_GT_AGREEMENT_COLUMNS = (
    "source_image",
    "target_class",
    "reference_id_a",
    "reference_id_b",
    "n_roi_pixels",
    "n_pairwise_valid_pixels",
    "pairwise_valid_coverage",
    "ambiguous_rate_a",
    "ambiguous_rate_b",
    "validity_agreement",
    "pixel_agreement",
    "dice",
    "iou",
    "n_components_a",
    "n_components_b",
    "n_components_matched",
    "mean_matched_component_iou",
    "unmatched_component_rate",
    "dice_passed",
    "iou_passed",
    "unmatched_rate_passed",
    "status",
)
SPATIAL_GT_ADJUDICATION_COLUMNS = (
    "source_image",
    "status",
    "adjudicator",
    "date",
    "justification",
    "reference_id",
)


# ---------------------------------------------------------------------------
# Frozen scientific protocol: tasks 01-02
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = "8tracks_v3"
RESULTS_SCHEMA_VERSION = "8tracks_v2"
PROTOCOL_STATUS = "frozen"
PROTOCOL_FREEZE_DATE = "2026-08-03"
PROTOCOL_REGISTRATION_MODE = "prospective_amendment_tasks25_26"
PROTOCOL_PRIOR_RESULTS_STATUS = "legacy_exploratory"
PROTOCOL_TEST_BLINDING_CLAIM = "not_asserted"
PROTOCOL_AMENDMENT_JUSTIFICATION = (
    "2026-08-03: tasks 25-26 added before batch-3 inspection: fixed "
    "train-projection eligibility thresholds, explicit unsupported tracks, "
    "and one globally calibrated spatial post-processing lock using only "
    "pure-image OOF maps from batches 1-2."
)
PROTOCOL_AMENDMENT_POLICY = (
    "Any scientific change after the freeze requires a new protocol version, "
    "a dated justification, and regeneration of every protocol artifact."
)


# Detection task
TARGET_CLASS = "peanut"
NON_TARGET_LABEL = "almond"
REFERENCE_CLASSES = ("almond", TARGET_CLASS)


# Dataset split protocol
SIMCA_TRAIN_BATCHES = (1, 2)
SIMCA_VALIDATION_BATCHES = (3,)
PURE_TEST_TRAIN_BATCHES = (1, 2, 3)
PURE_TEST_BATCHES = (4,)
MIXTURE_FINAL_TRAIN_BATCHES = (1, 2, 3, 4)
PROTOCOL_CALIBRATION_BATCHES = SIMCA_TRAIN_BATCHES
PROTOCOL_VALIDATION_BATCHES = SIMCA_VALIDATION_BATCHES
PROTOCOL_TEST_BATCHES = PURE_TEST_BATCHES
PROTOCOL_EXPECTED_CLASSES = REFERENCE_CLASSES
PCA_CALIBRATION_BATCHES = tuple(PROTOCOL_CALIBRATION_BATCHES)
# Notebook 03 is calibration-only. Batches 3 and 4 must never enter a PCA
# fit, projection, stability metric or selection diagnostic.
PCA_FORBIDDEN_BATCHES = tuple(
    (*PROTOCOL_VALIDATION_BATCHES, *PROTOCOL_TEST_BATCHES)
)
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


# SIMCA matrix families, parent tracks and eight evaluation tracks
SIMCA_OBJECT_MATRIX_METHODS = ("object_mean", "object_median")
SIMCA_PIXEL_MATRIX_METHODS = ("balanced_pixels", "all_pixels", "pixel")
SIMCA_MATRIX_FAMILIES = ("object_matrix", "pixel_matrix")
SIMCA_PROJECTION_LEVELS = ("object_projection", "pixel_projection")
SIMCA_DECISION_MODES = ("2way", "3way")

SIMCA_MATRIX_METHOD_FAMILY = {
    **{method: "object_matrix" for method in SIMCA_OBJECT_MATRIX_METHODS},
    **{method: "pixel_matrix" for method in SIMCA_PIXEL_MATRIX_METHODS},
}

SIMCA_PARENT_TRACKS = (
    "object_matrix_2way",
    "object_matrix_3way",
    "pixel_matrix_2way",
    "pixel_matrix_3way",
)

SIMCA_PARENT_TRACK_SPECS = {
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

# Backward-compatible names used by the current four-parent-track notebooks.
# New eight-track code must use SIMCA_EVALUATION_TRACKS and
# SIMCA_EVALUATION_TRACK_SPECS for projection-aware evaluation and Pareto.
SIMCA_SELECTION_TRACKS = SIMCA_PARENT_TRACKS
SIMCA_SELECTION_TRACK_SPECS = SIMCA_PARENT_TRACK_SPECS

SIMCA_EVALUATION_TRACKS = (
    "object_train__object_projection__2way",
    "object_train__object_projection__3way",
    "object_train__pixel_projection__2way",
    "object_train__pixel_projection__3way",
    "pixel_train__object_projection__2way",
    "pixel_train__object_projection__3way",
    "pixel_train__pixel_projection__2way",
    "pixel_train__pixel_projection__3way",
)

SIMCA_EVALUATION_TRACK_IDS = {
    "object_train__object_projection__2way": "E1",
    "object_train__object_projection__3way": "E2",
    "object_train__pixel_projection__2way": "E3",
    "object_train__pixel_projection__3way": "E4",
    "pixel_train__object_projection__2way": "E5",
    "pixel_train__object_projection__3way": "E6",
    "pixel_train__pixel_projection__2way": "E7",
    "pixel_train__pixel_projection__3way": "E8",
}

SIMCA_EVALUATION_TRACK_SPECS = {
    "object_train__object_projection__2way": {
        "track_id": "E1",
        "parent_track": "object_matrix_2way",
        "training_matrix_family": "object_matrix",
        "projection_level": "object_projection",
        "decision_mode": "2way",
        "primary_unit": "object",
        "decision_score_type": "simca_margin",
        "primary_metrics": (
            "target_miss_rate",
            "false_accept_rate",
            "balanced_accuracy",
        ),
        "pareto_minimize": ("target_miss_rate", "false_accept_rate"),
        "pareto_maximize": ("balanced_accuracy",),
        "secondary_object_aggregation_thresholds": (),
        "secondary_object_aggregation_policy": "not_applicable",
    },
    "object_train__object_projection__3way": {
        "track_id": "E2",
        "parent_track": "object_matrix_3way",
        "training_matrix_family": "object_matrix",
        "projection_level": "object_projection",
        "decision_mode": "3way",
        "primary_unit": "object",
        "decision_score_type": "simca_margin",
        "primary_metrics": (
            "target_miss_rate",
            "false_accept_rate",
            "uncertain_rate",
            "coverage_rate",
            "decided_balanced_accuracy",
        ),
        "pareto_minimize": (
            "target_miss_rate",
            "false_accept_rate",
            "uncertain_rate",
        ),
        "pareto_maximize": ("coverage_rate", "decided_balanced_accuracy"),
        "secondary_object_aggregation_thresholds": (),
        "secondary_object_aggregation_policy": "not_applicable",
    },
    "object_train__pixel_projection__2way": {
        "track_id": "E3",
        "parent_track": "object_matrix_2way",
        "training_matrix_family": "object_matrix",
        "projection_level": "pixel_projection",
        "decision_mode": "2way",
        "primary_unit": "source_image",
        "decision_score_type": "simca_margin",
        "calibration_primary_metrics": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
        ),
        "final_evaluation_metrics": (
            "small_fragment_recall",
            "fragment_precision",
        ),
        "pareto_minimize": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
        ),
        "pareto_maximize": ("macro_image_balanced_accuracy",),
        "secondary_object_aggregation_thresholds": (0.75, 0.80),
        "secondary_object_aggregation_policy": "fixed_2way_pixel_vote",
    },
    "object_train__pixel_projection__3way": {
        "track_id": "E4",
        "parent_track": "object_matrix_3way",
        "training_matrix_family": "object_matrix",
        "projection_level": "pixel_projection",
        "decision_mode": "3way",
        "primary_unit": "source_image",
        "decision_score_type": "simca_margin",
        "calibration_primary_metrics": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
            "uncertain_rate",
            "coverage_rate",
            "decided_balanced_accuracy",
        ),
        "final_evaluation_metrics": (
            "small_fragment_recall",
            "fragment_precision",
        ),
        "pareto_minimize": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
            "uncertain_rate",
        ),
        "pareto_maximize": (
            "coverage_rate",
            "decided_balanced_accuracy",
        ),
        "secondary_object_aggregation_thresholds": (),
        "secondary_object_aggregation_policy": "oof_calibrated_3way_vote",
    },
    "pixel_train__object_projection__2way": {
        "track_id": "E5",
        "parent_track": "pixel_matrix_2way",
        "training_matrix_family": "pixel_matrix",
        "projection_level": "object_projection",
        "decision_mode": "2way",
        "primary_unit": "object",
        "decision_score_type": "simca_margin",
        "primary_metrics": (
            "target_miss_rate",
            "false_accept_rate",
            "balanced_accuracy",
        ),
        "pareto_minimize": ("target_miss_rate", "false_accept_rate"),
        "pareto_maximize": ("balanced_accuracy",),
        "secondary_object_aggregation_thresholds": (),
        "secondary_object_aggregation_policy": "not_applicable",
    },
    "pixel_train__object_projection__3way": {
        "track_id": "E6",
        "parent_track": "pixel_matrix_3way",
        "training_matrix_family": "pixel_matrix",
        "projection_level": "object_projection",
        "decision_mode": "3way",
        "primary_unit": "object",
        "decision_score_type": "simca_margin",
        "primary_metrics": (
            "target_miss_rate",
            "false_accept_rate",
            "uncertain_rate",
            "coverage_rate",
            "decided_balanced_accuracy",
        ),
        "pareto_minimize": (
            "target_miss_rate",
            "false_accept_rate",
            "uncertain_rate",
        ),
        "pareto_maximize": ("coverage_rate", "decided_balanced_accuracy"),
        "secondary_object_aggregation_thresholds": (),
        "secondary_object_aggregation_policy": "not_applicable",
    },
    "pixel_train__pixel_projection__2way": {
        "track_id": "E7",
        "parent_track": "pixel_matrix_2way",
        "training_matrix_family": "pixel_matrix",
        "projection_level": "pixel_projection",
        "decision_mode": "2way",
        "primary_unit": "source_image",
        "decision_score_type": "simca_margin",
        "calibration_primary_metrics": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
        ),
        "final_evaluation_metrics": (
            "small_fragment_recall",
            "fragment_precision",
        ),
        "pareto_minimize": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
        ),
        "pareto_maximize": ("macro_image_balanced_accuracy",),
        "secondary_object_aggregation_thresholds": (0.75, 0.80),
        "secondary_object_aggregation_policy": "fixed_2way_pixel_vote",
    },
    "pixel_train__pixel_projection__3way": {
        "track_id": "E8",
        "parent_track": "pixel_matrix_3way",
        "training_matrix_family": "pixel_matrix",
        "projection_level": "pixel_projection",
        "decision_mode": "3way",
        "primary_unit": "source_image",
        "decision_score_type": "simca_margin",
        "calibration_primary_metrics": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
            "uncertain_rate",
            "coverage_rate",
            "decided_balanced_accuracy",
        ),
        "final_evaluation_metrics": (
            "small_fragment_recall",
            "fragment_precision",
        ),
        "pareto_minimize": (
            "macro_image_target_miss_rate",
            "macro_image_false_accept_rate",
            "macro_object_target_miss_rate",
            "uncertain_rate",
        ),
        "pareto_maximize": (
            "coverage_rate",
            "decided_balanced_accuracy",
        ),
        "secondary_object_aggregation_thresholds": (),
        "secondary_object_aggregation_policy": "oof_calibrated_3way_vote",
    },
}

# Projection and decision metadata are derived once here and materialised by
# ``build_simca_track_contracts``.  Fragment metrics require independent batch-4
# annotations and therefore never enter the 03B calibration Pareto.
for _evaluation_track, _spec in SIMCA_EVALUATION_TRACK_SPECS.items():
    _is_pixel_projection = _spec["projection_level"] == "pixel_projection"
    _is_object_training = _spec["training_matrix_family"] == "object_matrix"
    if _is_pixel_projection:
        _projection_policy = "all_pixels"
        _projection_methods = ("all_pixels",)
    elif _is_object_training:
        _projection_policy = "match_object_training_method"
        _projection_methods = ("object_mean", "object_median")
    else:
        _projection_policy = "compare_object_mean_and_median"
        _projection_methods = ("object_mean", "object_median")
    _spec["projection_matrix_policy"] = _projection_policy
    _spec["allowed_projection_methods"] = _projection_methods
    _spec["higher_is_target"] = True
    _spec["direct_2way_threshold"] = 0.0
    _spec["constraint_profile_id"] = "internal_calibration_risk_profile"
    _spec["limit_alpha"] = 0.01
    if "calibration_primary_metrics" not in _spec:
        _spec["calibration_primary_metrics"] = tuple(_spec["primary_metrics"])
        _spec["final_evaluation_metrics"] = ()
    _spec["primary_metrics"] = tuple(_spec["calibration_primary_metrics"])
del _evaluation_track, _spec, _is_pixel_projection, _is_object_training
del _projection_policy, _projection_methods

# Prespecified inference contract. Rate-scale tolerances are absolute
# differences (five percentage points); the domain-shift tolerance is in
# standardized units. Changing either value requires a new protocol version.
PROTOCOL_PRIMARY_HYPOTHESES = (
    {
        "hypothesis_id": "H1",
        "name": "object_training_family",
        "question": (
            "At fixed object projection and decision mode, does the training "
            "matrix family change object-level performance?"
        ),
        "scope": "primary",
    },
    {
        "hypothesis_id": "H2",
        "name": "pixel_training_family",
        "question": (
            "At fixed pixel projection and decision mode, does the training "
            "matrix family change pixel and small-fragment performance?"
        ),
        "scope": "primary",
    },
    {
        "hypothesis_id": "H3",
        "name": "train_projection_transfer",
        "question": (
            "Does the interaction between training family and projection "
            "level create a measurable standardized domain shift?"
        ),
        "scope": "primary",
    },
    {
        "hypothesis_id": "H4",
        "name": "selective_rejection",
        "question": (
            "At fixed training family and projection level, does 3-way "
            "decision improve the risk-coverage trade-off over 2-way?"
        ),
        "scope": "primary",
    },
)

PROTOCOL_RATE_PRACTICAL_TOLERANCE = 0.05
PROTOCOL_STANDARDIZED_SHIFT_TOLERANCE = 0.20
PROTOCOL_CONFIDENCE_LEVEL = 0.95
PROTOCOL_BOOTSTRAP_GROUP_COL = "source_image"
PROTOCOL_BOOTSTRAP_N_RESAMPLES = 2000
PROTOCOL_BOOTSTRAP_RANDOM_STATE = 42
PROTOCOL_MULTIPLICITY_METHOD = "holm_within_hypothesis_family"
PROTOCOL_SELECTION_PRIORITY = (
    "risk_guardrails",
    "worst_image",
    "stability",
    "simplicity",
    "compute_cost",
)
PROTOCOL_SELECTION_POLICY = {
    "technical_filters_first": True,
    "hard_constraints_before_pareto": True,
    "pareto_scope": "evaluation_track",
    "weighted_scores_allowed": False,
    "tie_break": "lexicographic",
    "tie_break_priority": PROTOCOL_SELECTION_PRIORITY,
}


# Notebook 02 result-table contracts
MATRIX_SUMMARY_REQUIRED_COLUMNS = (
    "matrix_id",
    "protocol_role",
    "matrix_method",
    "balanced_pixel_strategy",
    "n_observations",
    "n_features",
    "n_classes",
    "n_objects",
    "n_images",
    "n_nan",
    "n_inf",
    "matrix_rank",
    "rank_ratio",
    "n_zero_variance_bands",
    "wavelength_axis_id",
    "status",
)

MATRIX_COVERAGE_COLUMNS = (
    "matrix_id",
    "object_id",
    "source_image",
    "batch",
    "label",
    "sample_kind",
    "n_rows",
)
MATRIX_ERROR_COLUMNS = (
    "matrix_id",
    "protocol_role",
    "matrix_method",
    "balanced_pixel_strategy",
    "m",
    "error_type",
    "error",
)
M_FEASIBILITY_COLUMNS = (
    "m",
    "strategy",
    "under_m_policy",
    "n_objects_total",
    "n_objects_under_m",
    "eligible_rate",
    "n_rows",
    "n_classes",
    "n_images",
    "class_balance_ratio",
    "image_balance_ratio",
    "selection_stability",
    "status",
)
PIXEL_SAMPLING_DIAGNOSTIC_COLUMNS = (
    "m",
    "strategy",
    "seed",
    "object_id",
    "n_available",
    "n_selected",
    "selection_hash",
    "status",
)

PREPROCESSING_SUMMARY_REQUIRED_COLUMNS = (
    "matrix_id",
    "fit_role",
    "eval_role",
    "wavelength_axis_id",
    "preprocessing",
    "steps",
    "sg_window_length",
    "sg_polyorder",
    "deriv",
    "status",
    "n_features_before",
    "n_features_after",
    "band_count_unchanged",
    "n_nan",
    "n_inf",
    "zero_variance_band_rate",
    "saturation_rate",
    "global_min",
    "global_max",
    "repeatability_error",
    "name_steps_coherent",
)
PREPROCESSING_ERROR_COLUMNS = (
    "matrix_id",
    "fit_role",
    "eval_role",
    "wavelength_axis_id",
    "preprocessing",
    "sg_window_length",
    "error_type",
    "error",
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

SIMCA_EXACT_CONFIG_COLUMNS = (
    "evaluation_track",
    "track_id",
    "parent_track",
    "target_class",
    "non_target_label",
    "matrix_family",
    "matrix_method",
    "projection_level",
    "projection_matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "fit_config_id",
    "projection_config_id",
    "rule_variant",
    "limit_source",
    "decision_mode",
    "n_components",
    "alpha",
    "random_state",
    "direct_2way_threshold",
    "secondary_object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
)
# Backward-compatible name used by the shared exact-refit deduplicator.
SIMCA_REFIT_CONFIG_DEDUP_COLUMNS = SIMCA_EXACT_CONFIG_COLUMNS

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


# Shared runtime defaults
RANDOM_STATE = 42
REPLACE_BALANCED_PIXELS = False
CV_N_SPLITS = 5
CV_GROUP_COL = "source_image"
M_BALANCED_PIXELS = 10
BALANCED_PIXEL_STRATEGIES = ("random", "center")


# Notebook 02: spectral lock, matrix construction and technical diagnostics
USE_WAVELENGTH_WINDOW = False
WAVELENGTH_WINDOW_MIN_NM = 1225.0
WAVELENGTH_WINDOW_MAX_NM = 1675.0

MATRIX_METHODS_TO_CHECK = (
    "object_mean",
    "object_median",
    "balanced_pixels",
    "all_pixels",
)
MATRIX_BUILD_PROTOCOL_ROLES = ("calibration", "validation")
MATRIX_REQUIRED_METADATA = (
    "object_id",
    "source_image",
    "batch",
    "label",
    "sample_kind",
)

BALANCED_SAMPLING_M_VALUES = (5, 10, 20, 40, 60, 80, 100)
BALANCED_SAMPLING_SEEDS = (42, 43, 44)
BALANCED_SAMPLING_UNDER_M_POLICY = "exclude"
BALANCED_SAMPLING_MIN_ELIGIBLE_RATE = 0.95
BALANCED_SAMPLING_STUDY_M_VALUES = (10, 20)

SG_WINDOW_CHOICES = (5, 7, 9, 11, 13, 21)
SG_DEFAULT_WINDOW = 11
SG_POLYORDER = 2
PREPROCESSING_ZERO_VARIANCE_EPSILON = 1e-12
PREPROCESSING_MAX_ZERO_VARIANCE_BAND_RATE = 0.25
PREPROCESSING_SATURATION_BOUNDS = (-1e6, 1e6)
PREPROCESSING_REPEATABILITY_TOLERANCE = 1e-12
PREPROCESSING_ABSORBANCE_NONPOSITIVE_POLICY = "error"
PREPROCESSING_MATRIX_SPECS = (
    ("object_mean", None, None),
    ("object_median", None, None),
    ("all_pixels", None, None),
    ("balanced_pixels", "random", 10),
    ("balanced_pixels", "center", 10),
    ("balanced_pixels", "random", 20),
    ("balanced_pixels", "center", 20),
)

PREPROCESSING_CONFIGS_TO_COMPARE = {
    "raw": ("raw",),
    "absorbance": ("absorbance",),
    "snv": ("snv",),
    "msc": ("msc",),
    "vector_norm": ("vector_norm",),
    "sg_smooth": ("sg_smooth",),
    "sg_d1": ("sg_d1",),
    "sg_d2": ("sg_d2",),
    "absorbance_snv": ("absorbance", "snv"),
    "absorbance_msc": ("absorbance", "msc"),
    "absorbance_sg_smooth": ("absorbance", "sg_smooth"),
    "absorbance_sg_d1": ("absorbance", "sg_d1"),
    "absorbance_sg_d2": ("absorbance", "sg_d2"),
    "snv_sg_smooth": ("snv", "sg_smooth"),
    "snv_sg_d1": ("snv", "sg_d1"),
    "snv_sg_d2": ("snv", "sg_d2"),
    "absorbance_snv_sg_smooth": ("absorbance", "snv", "sg_smooth"),
    "absorbance_snv_sg_d1": ("absorbance", "snv", "sg_d1"),
    "absorbance_snv_sg_d2": ("absorbance", "snv", "sg_d2"),
}
PREPROCESSING_PLOT_NAMES = (
    "raw",
    "snv",
    "msc",
    "sg_smooth",
    "sg_d1",
    "sg_d2",
    "absorbance_snv",
    "absorbance_msc",
)
MATRIX_MAX_SPECTRA_TO_PLOT = 80


# ---------------------------------------------------------------------------
# Notebook 03: PCA exploration, stability and preprocessing Pareto shortlist
# ---------------------------------------------------------------------------
PCA_SAMPLE_KIND = "pure"
PCA_N_COMPONENTS = 20
PCA_DIAGNOSTIC_N_COMPONENTS = 3
PCA_MATRIX_METHODS = (
    "object_mean",
    "object_median",
    "all_pixels",
    "balanced_pixels",
)
PCA_BALANCED_M_VALUES = tuple(BALANCED_SAMPLING_STUDY_M_VALUES)
PCA_BALANCED_STRATEGIES = tuple(BALANCED_PIXEL_STRATEGIES)
PCA_SG_WINDOW_LENGTH = SG_DEFAULT_WINDOW
PCA_SG_POLYORDER = SG_POLYORDER
PCA_BALANCED_UNDER_M_POLICY = BALANCED_SAMPLING_UNDER_M_POLICY

PCA_STABILITY_SEEDS = (0, 1, 2, 3, 4)
PCA_STABILITY_REFERENCE_SEED = 0
PCA_STABILITY_N_SPLITS = 2
PCA_STABILITY_N_BOOTSTRAP = 100
PCA_STABILITY_N_COMPONENTS = 10
PCA_STABILITY_GROUP_COL = CV_GROUP_COL
PCA_STABILITY_BOOTSTRAP_GROUP_COL = "source_image"

MAX_PCA_PREPROCESSINGS_PER_FAMILY = None
PCA_SELECTION_EXPECTED_FAMILIES = ("object_matrix", "pixel_matrix")

# Pareto is applied after strict aggregation across every matrix variant of a
# preprocessing. Projection diagnostics remain reported but are not objectives.
PCA_SELECTION_PROFILES = {
    "object_matrix": {
        "maximize_metrics": (
            "class_trace_ratio",
        ),
        "minimize_metrics": (
            "batch_trace_ratio",
            "instability_metric",
            "ncomp_95",
        ),
    },
    "pixel_matrix": {
        "maximize_metrics": (
            "object_class_trace_ratio",
        ),
        "minimize_metrics": (
            "object_batch_trace_ratio",
            "instability_metric",
            "ncomp_95",
        ),
    },
}

PCA_SUMMARY_COLUMNS = (
    "candidate_id",
    "training_matrix_id",
    "matrix_family",
    "matrix_variant",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "component",
    "explained_variance_ratio",
    "cumulative_explained_variance_ratio",
)
PCA_DIAGNOSTIC_ID_COLUMNS = (
    "candidate_id",
    "training_matrix_id",
    "wavelength_axis_id",
    "matrix_family",
    "matrix_variant",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
)
PCA_SCORING_DIAGNOSTIC_COLUMNS = (
    *PCA_DIAGNOSTIC_ID_COLUMNS,
    "diagnostic_group",
    "metric",
    "value",
    "threshold",
    "constraint",
    "passed",
    "pareto_goal",
    "selection_status",
    "detail",
)
PCA_SELECTED_PREPROCESSING_COLUMNS = (
    "shortlist_id",
    "protocol_hash",
    "input_fingerprint",
    "review_hash",
    "matrix_family",
    "preprocessing",
    "preprocessing_steps",
    "selection_status",
    "selection_reason",
)

PCA_PREPROCESSING_SUMMARY_ID_COLUMNS = (
    "matrix_family",
    "preprocessing",
    "preprocessing_steps",
)

PCA_RUN_SPECTRA_CHECK_PLOTS = True
PCA_RUN_DETAILED_PLOTS = True
PCA_MAX_SPECTRA_TO_PLOT = 50
PCA_MAX_ROWS_TO_DISPLAY = 20
PCA_ARTIFACT_REVIEW_REQUIRED_STATUS = "reviewed"
PCA_ARTIFACT_REVIEW_ALLOWED_STATUSES = ("pending", "reviewed")
PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS = ("accept", "warning", "reject")
PCA_ARTIFACT_REVIEW_COLUMNS = (
    "candidate_id",
    "matrix_family",
    "matrix_variant",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "review_status",
    "review_decision",
    "artifact_codes",
    "critical_artifact",
    "review_comment",
    "reviewer",
    "review_date",
    "review_evidence",
    "run_fingerprint",
)

def make_pca_selection_config(**overrides: Any):
    """Return the canonical preprocessing-level PCA Pareto configuration."""
    from src.workflows.pca_selection import (
        PCASelectionProfile,
        make_pca_selection_config as _make_pca_selection_config,
    )

    profiles = {
        family: PCASelectionProfile(
            maximize_metrics=tuple(profile["maximize_metrics"]),
            minimize_metrics=tuple(profile["minimize_metrics"]),
        )
        for family, profile in PCA_SELECTION_PROFILES.items()
    }
    config_kwargs = {
        "profiles": profiles,
        "expected_families": PCA_SELECTION_EXPECTED_FAMILIES,
        "max_preprocessings_per_family": MAX_PCA_PREPROCESSINGS_PER_FAMILY,
    }
    config_kwargs.update(overrides)
    return _make_pca_selection_config(**config_kwargs)


# ---------------------------------------------------------------------------
# Notebook 03B: internal hyperparameter calibration on batches 1-2.
# Final ranking, Pareto filtering, diversity and per-track quotas belong to
# downstream model-selection notebooks and are intentionally absent here.
# ---------------------------------------------------------------------------
INTERNAL_CALIBRATION_BATCHES = tuple(PROTOCOL_CALIBRATION_BATCHES)
INTERNAL_CALIBRATION_FORBIDDEN_BATCHES = (
    *PROTOCOL_VALIDATION_BATCHES,
    *PROTOCOL_TEST_BATCHES,
)
INTERNAL_CALIBRATION_GROUP_COL = "source_image"
INTERNAL_CALIBRATION_LABEL_COL = "class_name"
INTERNAL_CALIBRATION_BATCH_COL = "batch"
INTERNAL_CALIBRATION_OBJECT_SIZE_COL = "object_area"
# Four pure calibration images are available (class x batch). Two grouped
# folds are therefore the largest image-level split with complete coverage.
INTERNAL_CALIBRATION_N_SPLITS = 2
INTERNAL_CALIBRATION_FOLD_RANDOM_STATE = 42
INTERNAL_CALIBRATION_RANDOM_SEEDS = (0, 1, 2)
INTERNAL_CALIBRATION_SIZE_N_BINS = 3

INTERNAL_CALIBRATION_MATRIX_METHODS = (
    "object_mean",
    "object_median",
    "balanced_pixels",
)
INTERNAL_CALIBRATION_M_VALUES = tuple(PCA_BALANCED_M_VALUES)
INTERNAL_CALIBRATION_PIXEL_STRATEGIES = ("random", "center")
INTERNAL_CALIBRATION_N_COMPONENTS_VALUES = tuple(range(3, 13))
INTERNAL_CALIBRATION_ALPHA_VALUES = (0.01,)
INTERNAL_CALIBRATION_SG_WINDOWS = tuple(SG_WINDOW_CHOICES)
INTERNAL_CALIBRATION_SG_POLYORDERS = (SG_POLYORDER,)
# Dilation is not identifiable on the pure-reference batches used in 03B.
# Keep the complete candidate set available for a later mixture/application
# stage, but run internal calibration with the neutral radius only.
INTERNAL_CALIBRATION_AVAILABLE_DILATION_RADII = (0, 2, 3, 5)
INTERNAL_CALIBRATION_DILATION_RADII = (0,)
INTERNAL_CALIBRATION_UNDER_M_POLICY = "exclude"
INTERNAL_CALIBRATION_RULE_VARIANTS = (
    "simple_chi2",
    "simple_emp_cv",
    "alternative_chi2_fixed2",
    "alternative_empHQ_emp_cv",
    "combined_index_chi2",
    "combined_index_emp_cv",
    "data_driven_chi2",
    "data_driven_emp_cv",
)

INTERNAL_CALIBRATION_OBJECT_THRESHOLDS = tuple(
    map(float, SIMCA_OBJECT_THRESHOLDS)
)
INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD = 0.0
INTERNAL_CALIBRATION_PIXEL_VOTE_CENTER = 0.5
INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES = (
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    1.00,
)
INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES = (
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    1.00,
)
INTERNAL_CALIBRATION_THRESHOLD_CROSSFIT = True
# Deprecated positive ratio grids are retained only for reading legacy
# exploratory artefacts.  The active 8-track path calibrates signed margins.
INTERNAL_CALIBRATION_THREE_WAY_LOWER_THRESHOLDS = ()
INTERNAL_CALIBRATION_THREE_WAY_UPPER_THRESHOLDS = ()

# Feasibility is explicit: no fallback to an unconstrained "best compromise".
# The exploratory profile is deliberately permissive so 03B first documents
# the feasible domain. Switching one name activates the stricter final profile.
INTERNAL_CALIBRATION_RISK_PROFILE = "exploratory"
INTERNAL_CALIBRATION_RISK_PROFILES = {
    "exploratory": {
        "max_fn_rate": 0.20,
        "max_fp_rate": 0.50,
        "min_balanced_accuracy": 0.60,
        "min_decision_rate": 0.75,
        "max_target_miss_rate": 0.25,
        "max_false_accept_rate": 0.30,
        "max_uncertain_rate": 0.60,
        "min_coverage": 0.40,
        "max_image_fn_rate": 1.00,
        "max_image_fp_rate": 1.00,
        "max_fold_fn_rate": 1.00,
        "max_fold_fp_rate": 1.00,
        "min_fold_balanced_accuracy": 0.00,
        "max_image_target_miss_rate": 1.00,
        "max_image_false_accept_rate": 1.00,
        "max_image_uncertain_rate": 1.00,
        "min_image_coverage": 0.00,
        "max_fold_balanced_accuracy_std": 0.50,
        "max_limit_relative_std": 1.00,
        "max_train_validation_rejection_gap": 0.50,
        "max_train_rejection_alpha_gap": 0.50,
        "plateau_tolerance": 0.05,
    },
    "final_strict": {
        "max_fn_rate": 0.10,
        "max_fp_rate": 0.20,
        "min_balanced_accuracy": 0.80,
        "min_decision_rate": 0.90,
        "max_target_miss_rate": 0.10,
        "max_false_accept_rate": 0.10,
        "max_uncertain_rate": 0.30,
        "min_coverage": 0.70,
        "max_image_fn_rate": 0.20,
        "max_image_fp_rate": 0.30,
        "max_fold_fn_rate": 0.20,
        "max_fold_fp_rate": 0.30,
        "min_fold_balanced_accuracy": 0.70,
        "max_image_target_miss_rate": 0.20,
        "max_image_false_accept_rate": 0.20,
        "max_image_uncertain_rate": 0.50,
        "min_image_coverage": 0.50,
        "max_fold_balanced_accuracy_std": 0.08,
        "max_limit_relative_std": 0.35,
        "max_train_validation_rejection_gap": 0.15,
        "max_train_rejection_alpha_gap": 0.10,
        "plateau_tolerance": 0.02,
    },
}
if INTERNAL_CALIBRATION_RISK_PROFILE not in INTERNAL_CALIBRATION_RISK_PROFILES:
    raise ValueError(
        "Unknown INTERNAL_CALIBRATION_RISK_PROFILE: "
        f"{INTERNAL_CALIBRATION_RISK_PROFILE!r}"
    )
_INTERNAL_CALIBRATION_RISK = INTERNAL_CALIBRATION_RISK_PROFILES[
    INTERNAL_CALIBRATION_RISK_PROFILE
]
INTERNAL_CALIBRATION_MAX_FN_RATE = _INTERNAL_CALIBRATION_RISK["max_fn_rate"]
INTERNAL_CALIBRATION_MAX_FP_RATE = _INTERNAL_CALIBRATION_RISK["max_fp_rate"]
INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY = _INTERNAL_CALIBRATION_RISK[
    "min_balanced_accuracy"
]
INTERNAL_CALIBRATION_MIN_DECISION_RATE = _INTERNAL_CALIBRATION_RISK[
    "min_decision_rate"
]
INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_target_miss_rate"
]
INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_false_accept_rate"
]
INTERNAL_CALIBRATION_MAX_UNCERTAIN_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_uncertain_rate"
]
INTERNAL_CALIBRATION_MIN_COVERAGE = _INTERNAL_CALIBRATION_RISK["min_coverage"]
INTERNAL_CALIBRATION_MAX_IMAGE_FN_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_image_fn_rate"
]
INTERNAL_CALIBRATION_MAX_IMAGE_FP_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_image_fp_rate"
]
INTERNAL_CALIBRATION_MAX_FOLD_FN_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_fold_fn_rate"
]
INTERNAL_CALIBRATION_MAX_FOLD_FP_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_fold_fp_rate"
]
INTERNAL_CALIBRATION_MIN_FOLD_BALANCED_ACCURACY = _INTERNAL_CALIBRATION_RISK[
    "min_fold_balanced_accuracy"
]
INTERNAL_CALIBRATION_MAX_IMAGE_TARGET_MISS_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_image_target_miss_rate"
]
INTERNAL_CALIBRATION_MAX_IMAGE_FALSE_ACCEPT_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_image_false_accept_rate"
]
INTERNAL_CALIBRATION_MAX_IMAGE_UNCERTAIN_RATE = _INTERNAL_CALIBRATION_RISK[
    "max_image_uncertain_rate"
]
INTERNAL_CALIBRATION_MIN_IMAGE_COVERAGE = _INTERNAL_CALIBRATION_RISK[
    "min_image_coverage"
]
INTERNAL_CALIBRATION_MAX_FOLD_BALANCED_ACCURACY_STD = (
    _INTERNAL_CALIBRATION_RISK["max_fold_balanced_accuracy_std"]
)
INTERNAL_CALIBRATION_MAX_LIMIT_RELATIVE_STD = _INTERNAL_CALIBRATION_RISK[
    "max_limit_relative_std"
]
INTERNAL_CALIBRATION_MAX_TRAIN_VALIDATION_REJECTION_GAP = (
    _INTERNAL_CALIBRATION_RISK["max_train_validation_rejection_gap"]
)
INTERNAL_CALIBRATION_MAX_TRAIN_REJECTION_ALPHA_GAP = (
    _INTERNAL_CALIBRATION_RISK["max_train_rejection_alpha_gap"]
)
INTERNAL_CALIBRATION_REFERENCE_OBJECT_THRESHOLD = float(
    INTERNAL_CALIBRATION_OBJECT_THRESHOLDS[0]
)
INTERNAL_CALIBRATION_PREFERRED_OBJECT_THRESHOLDS = (
    INTERNAL_CALIBRATION_OBJECT_THRESHOLDS
)
# The reference-threshold pass only finds a k plateau. Risk acceptance is
# applied after each model receives its calibrated 2-way/3-way threshold.
INTERNAL_CALIBRATION_PROVISIONAL_MAX_FN_RATE = 1.00
INTERNAL_CALIBRATION_PROVISIONAL_MAX_FP_RATE = 1.00
INTERNAL_CALIBRATION_PROVISIONAL_MIN_BALANCED_ACCURACY = 0.00
INTERNAL_CALIBRATION_PERFORMANCE_PLATEAU_TOLERANCE = (
    _INTERNAL_CALIBRATION_RISK["plateau_tolerance"]
)

INTERNAL_CALIBRATION_RUN = True
INTERNAL_CALIBRATION_MAX_CONFIGS = None
INTERNAL_CALIBRATION_VERBOSE = True
INTERNAL_CALIBRATION_KEEP_FINAL_OOF_PIXELS = False
INTERNAL_CALIBRATION_MAX_ROWS_TO_DISPLAY = 30
INTERNAL_CALIBRATION_CHECKPOINT_ENABLED = True
INTERNAL_CALIBRATION_CHECKPOINT_DIRNAME = "_ckpt"
INTERNAL_CALIBRATION_CHECKPOINT_EVERY_N_DATA_CONFIGS = 5
INTERNAL_CALIBRATION_RESUME_FROM_CHECKPOINT = True


# ---------------------------------------------------------------------------
# Notebook 03C: train-to-projection domain audit and spatial calibration
# ---------------------------------------------------------------------------
# These rules are part of the frozen protocol. They must not be modified after
# any batch-3 result has been inspected.
PROJECTION_DOMAIN_AUDIT_RULE_VERSION = "projection_domain_v1"
PROJECTION_DOMAIN_AUDIT_ALLOWED_BATCHES = (1, 2)
PROJECTION_DOMAIN_AUDIT_FORBIDDEN_BATCHES = (3, 4)
PROJECTION_DOMAIN_BORDER_WIDTH = 2
PROJECTION_DOMAIN_MIN_STRATUM_N = 5
PROJECTION_DOMAIN_DIAGNOSTIC_DIMENSIONS = (
    "overall",
    "fold",
    "size_bin",
    "border_core",
    "truth_class",
    "source_image",
)
PROJECTION_DOMAIN_ELIGIBILITY_DIMENSIONS = ("overall", "fold")
PROJECTION_DOMAIN_ELIGIBILITY_THRESHOLDS = {
    "warning_max_abs_standardized_shift": 1.5,
    "unsupported_max_abs_standardized_shift": 3.0,
    "warning_out_of_domain_rate": 0.10,
    "unsupported_out_of_domain_rate": 0.25,
    "warning_target_rejection_rate": 0.10,
    "unsupported_target_rejection_rate": 0.25,
}

SPATIAL_CALIBRATION_RULE_VERSION = "spatial_postprocessing_v1"
SPATIAL_CALIBRATION_ALLOWED_BATCHES = (1, 2)
SPATIAL_CALIBRATION_FORBIDDEN_BATCHES = (3, 4)
SPATIAL_CALIBRATION_TRUTH_SOURCE = "pure_image_class_exact"
SPATIAL_CALIBRATION_REQUIRED_TRUTH_LEVELS = (
    "pure_image_class_exact",
)
SPATIAL_CALIBRATION_REQUIRED_CLASSES = ("almond", "peanut")
SPATIAL_CALIBRATION_CONNECTIVITIES = (1, 2)
SPATIAL_CALIBRATION_MORPHOLOGY_OPERATIONS = (
    "none",
    "opening",
    "closing",
    "opening_closing",
)
SPATIAL_CALIBRATION_MORPHOLOGY_RADII = (0, 1, 2)
SPATIAL_CALIBRATION_MIN_AREAS = (0, 2, 5, 10, 20)
SPATIAL_CALIBRATION_FRAGMENT_AREA_UPPER_BOUNDS = (4, 9, 24, 49)
SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS = (
    "tiny_1_4",
    "very_small_5_9",
    "small_10_24",
    "medium_25_49",
    "large_50_plus",
)
SPATIAL_CALIBRATION_SELECTION_TOLERANCE = 0.005

INTERNAL_CALIBRATION_FOLD_COLUMNS = (
    "source_image",
    "object_id",
    "class_name",
    "batch",
    "object_area",
    "size_bin",
    "fold_id",
)
INTERNAL_CALIBRATION_FOLD_DIAGNOSTIC_COLUMNS = (
    "fold_id",
    "n_groups",
    "n_objects",
    "n_target_objects",
    "n_non_target_objects",
    "n_batch_1_objects",
    "n_batch_2_objects",
    "median_object_size",
    "coverage_complete",
)
INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS = (
    "config_id",
    "fold_id",
    "source_image",
    "object_id",
    "batch",
    "row",
    "col",
    "true_target_pixel",
    "predicted_target_pixel",
    "rule_statistic",
    "rule_limit",
)
INTERNAL_CALIBRATION_OOF_OBJECT_COLUMNS = (
    "config_id",
    "fold_id",
    "source_image",
    "object_id",
    "batch",
    "target_pixel_ratio",
    "true_target_object",
    "n_pixels_projected",
)
INTERNAL_CALIBRATION_FOLD_METRIC_COLUMNS = (
    "config_id",
    "fold_id",
    "n_objects",
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
    "pixel_fn_rate",
    "pixel_fp_rate",
    "pixel_balanced_accuracy",
)
INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS = (
    "config_id",
    "fold_id",
    "rule_limit",
    "train_rejection_rate",
    "validation_target_rejection_rate",
    "n_train_target",
    "n_validation_target",
)
INTERNAL_CALIBRATION_SAMPLING_DIAGNOSTIC_COLUMNS = (
    "data_config_id",
    "sampling_group_id",
    "fold_id",
    "random_state",
    "sampling_minhash",
)
INTERNAL_CALIBRATION_THRESHOLD_2WAY_COLUMNS = (
    "config_id",
    "object_threshold",
    "n",
    "n_folds",
    "fn",
    "fp",
    "target_sensitivity",
    "non_target_specificity",
    "balanced_accuracy",
    "fn_rate",
    "fp_rate",
    "decision_rate",
    "max_image_fn_rate",
    "max_image_fp_rate",
    "max_fold_fn_rate",
    "max_fold_fp_rate",
    "min_fold_balanced_accuracy",
    "std_fold_balanced_accuracy",
    "threshold_sensitivity",
    "feasible",
    "selected",
    "selection_status",
)
INTERNAL_CALIBRATION_THRESHOLD_3WAY_COLUMNS = (
    "config_id",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "n",
    "n_folds",
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "decided_balanced_accuracy",
    "max_image_target_miss_rate",
    "max_image_false_accept_rate",
    "max_image_uncertain_rate",
    "min_image_coverage",
    "max_fold_target_miss_rate",
    "max_fold_false_accept_rate",
    "max_fold_uncertain_rate",
    "min_fold_coverage",
    "std_fold_decided_balanced_accuracy",
    "uncertain_zone_width",
    "feasible",
    "selected",
    "selection_status",
)
INTERNAL_CALIBRATION_THRESHOLD_3WAY_STUDY_COLUMNS = (
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "n_configurations",
    "n_feasible_configurations",
    "feasible_configuration_rate",
    "n_selected_configurations",
    "selected_configuration_rate",
    "median_target_miss_rate",
    "p90_target_miss_rate",
    "median_non_target_false_accept_rate",
    "p90_non_target_false_accept_rate",
    "median_uncertain_rate",
    "p90_uncertain_rate",
    "median_coverage_rate",
    "median_decided_balanced_accuracy",
    "pair_status",
)
INTERNAL_CALIBRATION_ERROR_COLUMNS = (
    "scope_id",
    "fold_id",
    "status",
    "technical_errors",
    "error_type",
    "error_message",
    "n_affected_configurations",
)
INTERNAL_CALIBRATION_CALIBRATED_HYPERPARAMETER_COLUMNS = (
    "calibration_id",
    "source_config_id",
    "model_group_id",
    "random_states_json",
    "calibration_track",
    "decision_mode",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "fn_rate",
    "fp_rate",
    "balanced_accuracy",
    "uncertain_rate",
    "coverage_rate",
    "fold_balanced_accuracy_std",
    "limit_relative_std",
    "train_validation_rejection_gap",
    "seed_prediction_agreement",
    "seed_sampling_agreement",
    "n_seeds_evaluated",
    "calibration_status",
)

# Canonical 8-track schemas.  Hyperparameters remain in the configuration and
# calibrated-domain tables; pixel rows carry only stable identifiers.
INTERNAL_CALIBRATION_TRACK_CONTRACT_COLUMNS = (
    "track_id",
    "evaluation_track",
    "parent_track",
    "training_matrix_family",
    "projection_level",
    "projection_matrix_policy",
    "allowed_projection_methods_json",
    "primary_unit",
    "decision_mode",
    "decision_score_type",
    "higher_is_target",
    "direct_2way_threshold",
    "constraint_profile_id",
    "calibration_primary_metrics_json",
    "final_evaluation_metrics_json",
    "pareto_minimize_json",
    "pareto_maximize_json",
    "protocol_version",
    "schema_version",
)
INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_V2_COLUMNS = (
    "fit_config_id",
    "fold_id",
    "random_state",
    "raw_rank",
    "preprocessed_rank",
    "n_train_target",
    "n_features",
    "n_components",
    "matrix_build_seconds",
    "preprocessing_seconds",
    "fit_seconds",
    "status",
    "error_code",
    "error_message",
)
INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_V2_COLUMNS = (
    "projection_config_id",
    "fit_config_id",
    "fold_id",
    "random_state",
    "rule_variant",
    "limit_method",
    "limit_alpha",
    "q_limit",
    "t2_limit",
    "rule_limit",
    "train_rejection_rate",
    "oof_target_rejection_rate",
    "status",
    "error_code",
)
INTERNAL_CALIBRATION_OOF_BASE_V2_COLUMNS = (
    "projection_config_id",
    "fit_config_id",
    "fold_id",
    "random_state",
    "training_matrix_family",
    "projection_level",
    "projection_matrix_method",
    "source_image",
    "object_id",
    "batch",
    "object_area",
    "size_bin",
    "truth",
    "truth_level",
    "pca_score_pc1",
    "pca_score_pc2",
    "H",
    "Q",
    "rule_statistic",
    "rule_limit",
    "normalized_ratio",
    "simca_margin",
    "direct_2way_decision",
)
INTERNAL_CALIBRATION_OOF_OBJECT_V2_COLUMNS = (
    *INTERNAL_CALIBRATION_OOF_BASE_V2_COLUMNS,
)
INTERNAL_CALIBRATION_OOF_PIXEL_V2_COLUMNS = (
    *INTERNAL_CALIBRATION_OOF_BASE_V2_COLUMNS,
    "row",
    "col",
)
INTERNAL_CALIBRATION_PROJECTION_SHIFT_COLUMNS = (
    "projection_config_id",
    "fit_config_id",
    "projection_level",
    "projection_matrix_method",
    "fold_id",
    "n_train",
    "n_projection",
    "train_pc1_mean",
    "train_pc1_std",
    "train_pc2_mean",
    "train_pc2_std",
    "train_h_mean",
    "train_h_std",
    "train_q_mean",
    "train_q_std",
    "train_rule_limit_mean",
    "train_rule_limit_std",
    "train_normalized_ratio_mean",
    "train_normalized_ratio_std",
    "train_margin_mean",
    "train_margin_std",
    "pca_pc1_standardized_shift",
    "pca_pc2_standardized_shift",
    "pca_centroid_shift",
    "h_standardized_shift",
    "q_standardized_shift",
    "rule_limit_standardized_shift",
    "normalized_ratio_standardized_shift",
    "margin_standardized_shift",
    "projection_out_of_domain_rate",
    "projection_target_rejection_rate",
)

PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS = (
    "evaluation_track",
    "track_id",
    "projection_config_id",
    "fit_config_id",
    "projection_level",
    "projection_matrix_method",
    "fold_id",
    "stratum_type",
    "stratum_value",
    "n_observations",
    "n_target",
    "pca_centroid_shift",
    "t2_standardized_shift",
    "q_standardized_shift",
    "rule_limit_standardized_shift",
    "normalized_ratio_standardized_shift",
    "simca_margin_standardized_shift",
    "out_of_domain_rate",
    "target_rejection_rate",
    "target_margin_displacement",
    "diagnostic_status",
    "protocol_version",
    "protocol_hash",
)
PROJECTION_ELIGIBILITY_COLUMNS = (
    "evaluation_track",
    "track_id",
    "n_projection_configurations",
    "n_diagnostics",
    "max_abs_standardized_shift",
    "max_out_of_domain_rate",
    "max_target_rejection_rate",
    "eligibility_status",
    "eligibility_reason",
    "rule_version",
    "thresholds_json",
    "protocol_version",
    "protocol_hash",
)
SPATIAL_CALIBRATION_METRIC_COLUMNS = (
    "spatial_candidate_id",
    "map_variant",
    "domain_config_id",
    "evaluation_track",
    "track_id",
    "projection_config_id",
    "connectivity",
    "morphology_operation",
    "morphology_radius",
    "min_area_pixels",
    "n_images",
    "n_valid_pixels",
    "dice",
    "iou",
    "pixel_precision",
    "pixel_recall",
    "component_precision",
    "component_recall",
    "split_rate",
    "merge_rate",
    "uncertain_pixel_rate",
    "smallest_fragment_recall",
    "is_locked_candidate",
    "truth_level",
    "protocol_version",
    "protocol_hash",
)
FRAGMENT_SIZE_CLASS_COLUMNS = (
    "spatial_candidate_id",
    "domain_config_id",
    "evaluation_track",
    "track_id",
    "projection_config_id",
    "area_class",
    "min_area_pixels",
    "max_area_pixels",
    "n_truth_fragments",
    "n_detected_fragments",
    "fragment_recall",
    "mean_best_iou",
    "is_smallest_class",
    "is_locked_candidate",
    "truth_level",
    "protocol_version",
    "protocol_hash",
)
INTERNAL_CALIBRATION_2WAY_METRIC_V2_COLUMNS = (
    "evaluation_config_id",
    "evaluation_track",
    "track_id",
    "fold_id",
    "random_state",
    "aggregation_level",
    "n",
    "target_miss_rate",
    "false_accept_rate",
    "balanced_accuracy",
    "metric_role",
)
INTERNAL_CALIBRATION_PIXEL_TO_OBJECT_2WAY_COLUMNS = (
    "evaluation_config_id",
    "evaluation_track",
    "track_id",
    "fold_id",
    "random_state",
    "secondary_object_threshold",
    "n_objects",
    "target_miss_rate",
    "false_accept_rate",
    "balanced_accuracy",
)
INTERNAL_CALIBRATION_THRESHOLD_3WAY_V2_COLUMNS = (
    "evaluation_config_id",
    "evaluation_track",
    "track_id",
    "evaluation_fold",
    "random_state",
    "decision_scope",
    "score_type",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "decided_balanced_accuracy",
    "feasible",
    "pareto_front",
    "selected",
    "failure_reason",
)
INTERNAL_CALIBRATION_AUDIT_V2_COLUMNS = (
    "audit_type",
    "evaluation_track",
    "track_id",
    "n_initial",
    "n_technical_valid",
    "n_risk_feasible",
    "n_k_plateau",
    "n_m_plateau",
    "n_seed_consensus",
    "n_pareto",
    "track_status",
    "failure_reason",
)


# ---------------------------------------------------------------------------
# Notebooks 04A/04B: searches restricted to the calibrated 03B domain
# ---------------------------------------------------------------------------
SIMCA_SEARCH_ALLOWED_DILATION_RADII = INTERNAL_CALIBRATION_DILATION_RADII
SIMCA_SEARCH_RANDOM_SEEDS = INTERNAL_CALIBRATION_RANDOM_SEEDS
SIMCA_SEARCH_ERROR_GRANULARITY = "configuration"
SIMCA_SEARCH_KEEP_OOF_PIXELS = False
SIMCA_SEARCH_KEEP_OOF_OBJECTS = True

SIMCA_GRID_SEARCH_RUN = True
SIMCA_GRID_SEARCH_VERBOSE = True
SIMCA_GRID_SEARCH_CHECKPOINT_DIRNAME = "_checkpoints"
SIMCA_GRID_SEARCH_CHECKPOINT_EVERY_N_DATA_CONFIGS = 5
SIMCA_GRID_SEARCH_RESUME_FROM_CHECKPOINT = True
# 04A evaluates the complete locked 03B domain. These switches control only
# execution/persistence; objectives and constraints remain those declared in
# SIMCA_EVALUATION_TRACK_SPECS and SIMCA_SEARCH_CONSTRAINTS.
SIMCA_GRID_APPLY_SPATIAL_LOCK = True
SIMCA_GRID_PARETO_EPSILON = 0.0
SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES = (
    "eligible",
    "eligible_with_warning",
)
SIMCA_GRID_REQUIRED_FINITE_PREDICTION_COLUMNS = (
    "H",
    "Q",
    "rule_statistic",
    "rule_limit",
    "normalized_ratio",
    "simca_margin",
)
SIMCA_GRID_EXACT_CONFIGURATION_COLUMNS = (
    "evaluation_track",
    "decision_mode",
    "matrix_family",
    "matrix_method",
    "projection_level",
    "projection_matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing_steps",
    "rule_family",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "direct_2way_threshold",
    "secondary_object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
)

SIMCA_OPTUNA_RUN = True
SIMCA_OPTUNA_PURPOSE = "budgeted_search_efficiency_benchmark"
SIMCA_OPTUNA_LOAD_EXISTING_STUDY = True
SIMCA_OPTUNA_N_TRIALS_PER_TRACK = 100
SIMCA_OPTUNA_N_STARTUP_TRIALS = 20
SIMCA_OPTUNA_N_JOBS = 1
SIMCA_OPTUNA_RANDOM_STATE = RANDOM_STATE
SIMCA_OPTUNA_STORAGE_TIMEOUT_SECONDS = 30.0
SIMCA_OPTUNA_SHOW_PROGRESS_BAR = True
SIMCA_OPTUNA_STUDY_NAME_TEMPLATE = "04B_{track_id}_{results_tag}_{plan_hash}"
SIMCA_OPTUNA_STORAGE_FILENAME = "optuna_studies.sqlite3"
SIMCA_OPTUNA_SAMPLER_NAME = "TPESampler"
SIMCA_OPTUNA_SAMPLER_MULTIVARIATE = True
SIMCA_OPTUNA_MIN_PARETO_RECALL = 0.80
SIMCA_OPTUNA_UNIFORM_RECALL_DELTA_TOLERANCE = 0.05
SIMCA_OPTUNA_SUPPORTED_ELIGIBILITY_STATUSES = (
    "eligible",
    "eligible_with_warning",
)
SIMCA_OPTUNA_TECHNICAL_PRUNE_STATUSES = (
    "technical_invalid",
    "fit_or_projection_error",
    "non_finite_objective",
    "impossible_violation",
)
# The benchmark is deliberately lookup-only. 04A has already evaluated the
# complete locked 03B domain on the exact folds, seeds and thresholds; 04B must
# never reload spectra, refit a model or inspect batches 3-4 in its objective.
SIMCA_OPTUNA_REUSE_GRID_METRICS = True

# Prospective addendum for the next 8-track batch-3 run. Legacy 04C/05 outputs
# already exist and are therefore disclosed rather than silently ignored.
SIMCA_ABLATION_REGISTRATION_STATUS = (
    "prospective_for_8tracks_v3_rerun_after_legacy_exploratory_batch3_processing"
)
SIMCA_ABLATION_THRESHOLD_PERTURBATION = 0.05
SIMCA_ABLATION_PRIMARY_METRICS = (
    "target_miss_rate",
    "false_accept_rate",
    "balanced_accuracy",
    "uncertain_rate",
    "coverage_rate",
    "decided_balanced_accuracy",
    "worst_image",
    "seed_stability",
    "compute_cost",
    "dice",
    "iou",
    "fragment_recall",
    "split_rate",
    "merge_rate",
)
SIMCA_ABLATION_FACTOR_SPECS = (
    {
        "factor": "matrix_method_object",
        "contrast_type": "paired_variant",
        "factor_columns": ("matrix_method", "projection_matrix_method"),
        "reference_levels": ("object_mean",),
        "ablated_levels": ("object_median",),
        "fit_changed": True,
        "projection_changed": True,
        "decision_changed": False,
        "spatial_processing_changed": False,
    },
    {
        "factor": "sampling_m",
        "contrast_type": "paired_variant",
        "factor_columns": ("m",),
        "reference_levels": (20,),
        "ablated_levels": (10,),
        "fit_changed": True,
        "projection_changed": False,
        "decision_changed": False,
        "spatial_processing_changed": False,
    },
    {
        "factor": "sampling_strategy",
        "contrast_type": "paired_variant",
        "factor_columns": ("balanced_pixel_strategy",),
        "reference_levels": ("center",),
        "ablated_levels": ("random",),
        "fit_changed": True,
        "projection_changed": False,
        "decision_changed": False,
        "spatial_processing_changed": False,
    },
    {
        "factor": "limit_source",
        "contrast_type": "strict_ablation",
        "factor_columns": ("limit_source", "rule_variant"),
        "reference_levels": ("calibration_train_only",),
        "ablated_levels": ("theoretical_train_fit",),
        "fit_changed": False,
        "projection_changed": False,
        "decision_changed": True,
        "spatial_processing_changed": False,
    },
    {
        "factor": "rule_family",
        "contrast_type": "paired_variant",
        "factor_columns": ("rule_family", "rule_variant"),
        "reference_levels": ("combined_index",),
        "ablated_levels": ("simple",),
        "fit_changed": False,
        "projection_changed": False,
        "decision_changed": True,
        "spatial_processing_changed": False,
    },
    {
        "factor": "spectral_sg_window",
        "contrast_type": "paired_variant",
        "factor_columns": ("sg_window_length",),
        "reference_levels": (11, 11),
        "ablated_levels": (9, 13),
        "fit_changed": True,
        "projection_changed": False,
        "decision_changed": False,
        "spatial_processing_changed": False,
    },
)
SIMCA_ABLATION_PREPROCESSING_FACTORS = (
    "spectral_absorbance",
    "spectral_snv_msc",
    "spectral_sg",
    "spectral_sg_derivative",
    "spectral_sg_window",
)
SIMCA_ABLATION_INTERACTION_SPECS = (
    "preprocessing_x_matrix_method",
    "m_x_sampling_strategy",
    "rule_x_limit_source",
    "morphology_x_min_area",
    "border_policy_x_morphology",
)


# ---------------------------------------------------------------------------
# Notebook 04C: locked batch-3 validation and spatial reconstruction
# ---------------------------------------------------------------------------
# 04C uses batches 1-2 for the final calibration refit and projects batch 3
# once. Batch 4 is deliberately outside this notebook.
SIMCA_CONCAT_REFIT_TRAIN_BATCHES = tuple(PROTOCOL_CALIBRATION_BATCHES)
SIMCA_CONCAT_REFIT_PROJECTION_BATCHES = tuple(PROTOCOL_VALIDATION_BATCHES)
SIMCA_CONCAT_REFIT_FORBIDDEN_BATCHES = tuple(PROTOCOL_TEST_BATCHES)
SIMCA_CONCAT_REFIT_RUN = True
SIMCA_CONCAT_REFIT_VERBOSE = False
SIMCA_CONCAT_REFIT_BATCH_SIZE = 10
SIMCA_CONCAT_REFIT_MAX_CANDIDATES = None
SIMCA_CONCAT_REFIT_RECONSTRUCT_HEAVY_OBJECT_ARRAYS = False
SIMCA_CONCAT_REFIT_CHECKPOINT_ENABLED = True
SIMCA_CONCAT_REFIT_RESUME_FROM_CHECKPOINT = True
SIMCA_CONCAT_REFIT_CHECKPOINT_DIRNAME = "_checkpoints"
SIMCA_CONCAT_REFIT_SIGNATURE_ROUND_DECIMALS = 12
SIMCA_CONCAT_REFIT_CANDIDATE_POLICY = (
    "supported_protocol_pareto_plus_unsupported_diagnostic_pareto"
)
SIMCA_CONCAT_REFIT_SUPPORTED_ELIGIBILITY_STATUSES = (
    "eligible",
    "eligible_with_warning",
)
SIMCA_CONCAT_REFIT_UNSUPPORTED_ELIGIBILITY_STATUSES = (
    "unsupported_domain_shift",
)
SIMCA_CONCAT_REFIT_BORDER_WIDTH = PROJECTION_DOMAIN_BORDER_WIDTH
SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL = 0.95
SIMCA_CONCAT_REFIT_MAP_ENCODING = "packbits_zlib_v1"
SIMCA_CONCAT_REFIT_MAP_COMPRESSION_LEVEL = 6
SIMCA_CONCAT_REFIT_TRUTH_SOURCE = "pure_image_class_exact"
SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU = 0.0
SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN = None
SIMCA_CONCAT_REFIT_GUARDRAIL_SCOPES = ("overall", "worst_image")
SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS = {
    "2way": {
        "max_fn_rate": INTERNAL_CALIBRATION_MAX_FN_RATE,
        "max_fp_rate": INTERNAL_CALIBRATION_MAX_FP_RATE,
        "min_balanced_accuracy": (
            INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY
        ),
        "max_image_fn_rate": INTERNAL_CALIBRATION_MAX_IMAGE_FN_RATE,
        "max_image_fp_rate": INTERNAL_CALIBRATION_MAX_IMAGE_FP_RATE,
    },
    "3way": {
        "max_fn_rate": INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE,
        "max_fp_rate": INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE,
        "max_uncertain_rate": INTERNAL_CALIBRATION_MAX_UNCERTAIN_RATE,
        "min_balanced_accuracy": (
            INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY
        ),
        "max_image_target_miss_rate": (
            INTERNAL_CALIBRATION_MAX_IMAGE_TARGET_MISS_RATE
        ),
        "max_image_false_accept_rate": (
            INTERNAL_CALIBRATION_MAX_IMAGE_FALSE_ACCEPT_RATE
        ),
        "max_image_uncertain_rate": (
            INTERNAL_CALIBRATION_MAX_IMAGE_UNCERTAIN_RATE
        ),
    },
}
SIMCA_CONCAT_REFIT_EVALUATION_RULE_VERSION = (
    "04c_validation_metrics_v3_scope_specific_guardrails"
)
SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS = {
    "2way": {
        "overall": (
            ("target_miss_rate", "max_fn_rate", "<="),
            ("false_accept_rate", "max_fp_rate", "<="),
            ("balanced_accuracy", "min_balanced_accuracy", ">="),
        ),
        "worst_image": (
            ("target_miss_rate", "max_image_fn_rate", "<="),
            ("false_accept_rate", "max_image_fp_rate", "<="),
        ),
    },
    "3way": {
        "overall": (
            ("target_miss_rate", "max_fn_rate", "<="),
            ("false_accept_rate", "max_fp_rate", "<="),
            ("uncertain_rate", "max_uncertain_rate", "<="),
            (
                "decided_balanced_accuracy",
                "min_balanced_accuracy",
                ">=",
            ),
        ),
        "worst_image": (
            (
                "target_miss_rate",
                "max_image_target_miss_rate",
                "<=",
            ),
            (
                "false_accept_rate",
                "max_image_false_accept_rate",
                "<=",
            ),
            (
                "uncertain_rate",
                "max_image_uncertain_rate",
                "<=",
            ),
        ),
    },
}
SIMCA_CONCAT_REFIT_EVALUATION_AMENDMENT = {
    "amendment_type": "guardrail_scope_mapping_correction",
    "reason": (
        "Worst-image controls use the image-specific limits frozen in the "
        "active risk profile. Coverage remains a reported metric but is not "
        "a second blocking alias of uncertainty."
    ),
    "thresholds_changed": False,
    "numeric_thresholds_changed": False,
    "threshold_mapping_corrected": True,
    "candidate_pool_changed": False,
    "model_parameters_changed": False,
    "predictions_changed": False,
    "batch3_used_to_choose_thresholds": False,
}
SIMCA_CONCAT_REFIT_EVALUATION_RULE_KEYS = (
    "SIMCA_CONCAT_REFIT_EVALUATION_RULE_VERSION",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS",
    "SIMCA_CONCAT_REFIT_EVALUATION_AMENDMENT",
    "SIMCA_CONCAT_REFIT_TRUTH_SOURCE",
    "SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL",
)
SIMCA_CONCAT_REFIT_PIXEL_TRACKS = tuple(
    track
    for track, spec in SIMCA_EVALUATION_TRACK_SPECS.items()
    if spec["projection_level"] == "pixel_projection"
)
SIMCA_CONCAT_REFIT_VALIDATION_PLAN_KEYS = (
    "SIMCA_CONCAT_REFIT_TRAIN_BATCHES",
    "SIMCA_CONCAT_REFIT_PROJECTION_BATCHES",
    "SIMCA_CONCAT_REFIT_FORBIDDEN_BATCHES",
    "SIMCA_CONCAT_REFIT_CANDIDATE_POLICY",
    "SIMCA_CONCAT_REFIT_BORDER_WIDTH",
    "SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL",
    "SIMCA_CONCAT_REFIT_MAP_ENCODING",
    "SIMCA_CONCAT_REFIT_TRUTH_SOURCE",
    "SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU",
    "SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_SCOPES",
    "SIMCA_SEARCH_CONSTRAINTS",
)

SIMCA_CONCAT_REFIT_CANDIDATE_COLUMNS = (
    "validation_candidate_id",
    "calibration_id",
    "domain_config_id",
    "evaluation_config_id",
    "data_config_id",
    "fit_config_id",
    "projection_config_id",
    "evaluation_track",
    "track_id",
    "parent_track",
    "decision_mode",
    "matrix_family",
    "matrix_method",
    "projection_level",
    "projection_matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "random_state",
    "sg_window_length",
    "sg_polyorder",
    "direct_2way_threshold",
    "secondary_object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "position_dilation_radius",
    "calibration_status",
    "eligibility_status",
    "candidate_front",
    "visited_by_optuna",
    "optuna_pareto",
)
SIMCA_VALIDATION_PREDICTION_BASE_COLUMNS = (
    "projection_config_id",
    "fit_config_id",
    "random_state",
    "training_matrix_family",
    "projection_level",
    "projection_matrix_method",
    "source_image",
    "object_id",
    "batch",
    "object_area",
    "truth",
    "truth_level",
    "pca_score_pc1",
    "pca_score_pc2",
    "H",
    "Q",
    "rule_statistic",
    "rule_limit",
    "normalized_ratio",
    "simca_margin",
)
SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS = (
    *SIMCA_VALIDATION_PREDICTION_BASE_COLUMNS,
)
SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS = (
    *SIMCA_VALIDATION_PREDICTION_BASE_COLUMNS,
    "row",
    "col",
    "distance_to_border",
    "is_border_pixel",
    "is_core_pixel",
)
SIMCA_VALIDATION_METRIC_COLUMNS = (
    "validation_candidate_id",
    "calibration_id",
    "domain_config_id",
    "evaluation_track",
    "track_id",
    "decision_mode",
    "projection_level",
    "random_state",
    "map_variant",
    "aggregation_level",
    "group_id",
    "n_observations",
    "n_target",
    "n_non_target",
    "n_target_objects",
    "n_uncertain",
    "n_target_uncertain",
    "n_non_target_uncertain",
    "tp",
    "fn",
    "fp",
    "tn",
    "balanced_accuracy",
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "decided_balanced_accuracy",
    "macro_object_target_miss_rate",
    "macro_image_target_miss_rate",
    "macro_image_false_accept_rate",
    "macro_image_uncertain_rate",
    "macro_image_coverage_rate",
    "macro_image_balanced_accuracy",
    "macro_image_decided_balanced_accuracy",
    "macro_image_target_miss_rate_ci_low",
    "macro_image_target_miss_rate_ci_high",
    "macro_image_false_accept_rate_ci_low",
    "macro_image_false_accept_rate_ci_high",
    "macro_image_uncertain_rate_ci_low",
    "macro_image_uncertain_rate_ci_high",
    "macro_image_coverage_rate_ci_low",
    "macro_image_coverage_rate_ci_high",
    "macro_image_balanced_accuracy_ci_low",
    "macro_image_balanced_accuracy_ci_high",
    "macro_image_decided_balanced_accuracy_ci_low",
    "macro_image_decided_balanced_accuracy_ci_high",
    "target_miss_rate_ci_low",
    "target_miss_rate_ci_high",
    "false_accept_rate_ci_low",
    "false_accept_rate_ci_high",
    "uncertain_rate_ci_low",
    "uncertain_rate_ci_high",
    "coverage_rate_ci_low",
    "coverage_rate_ci_high",
    "balanced_accuracy_ci_low",
    "balanced_accuracy_ci_high",
    "decided_balanced_accuracy_ci_low",
    "decided_balanced_accuracy_ci_high",
    "prediction_signature",
    "decision_signature",
    "prediction_equivalence_group_id",
    "decision_equivalence_group_id",
    "status",
    "error_type",
    "error_message",
)
SIMCA_PIXEL_MAP_MANIFEST_COLUMNS = (
    "validation_candidate_id",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "projection_config_id",
    "random_state",
    "source_image",
    "batch",
    "height",
    "width",
    "map_encoding",
    "valid_mask",
    "raw_target_mask",
    "uncertain_mask",
    "postprocessed_target_mask",
    "truth_mask",
    "margin_source",
    "truth_level",
    "spatial_lock_sha256",
)
SIMCA_SPATIAL_COMPONENT_COLUMNS = (
    "validation_candidate_id",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "random_state",
    "source_image",
    "map_variant",
    "component_role",
    "component_id",
    "area_pixels",
    "area_class",
    "centroid_row",
    "centroid_col",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
    "best_match_component_id",
    "best_iou",
    "overlap_count",
    "detected_or_matched",
    "split_or_merge",
    "truth_level",
)
SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS = (
    "validation_candidate_id",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "random_state",
    "source_image",
    "aggregation_level",
    "map_variant",
    "n_valid_pixels",
    "dice",
    "iou",
    "pixel_precision",
    "pixel_recall",
    "n_truth_components",
    "n_predicted_components",
    "component_precision",
    "component_recall",
    "split_rate",
    "merge_rate",
    "smallest_fragment_recall",
    "truth_level",
)
SIMCA_VALIDATION_GUARDRAIL_COLUMNS = (
    "validation_candidate_id",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "random_state",
    "eligibility_status",
    "candidate_status",
    "scope",
    "metric",
    "observed_value",
    "ci_low",
    "ci_high",
    "comparator",
    "threshold",
    "check_status",
    "reason",
    "prediction_equivalence_group_id",
    "decision_equivalence_group_id",
)

# Generic candidate-audit helpers remain public for notebooks 04A/04B and for
# downstream compatibility. 04C no longer uses them to remove candidates.
SIMCA_TECHNICAL_AUDIT_COLUMNS = (
    "candidate_id",
    "technical_status",
    "technical_failure_type",
    "technical_failure_message",
)
SIMCA_OUTPUT_SIGNATURE_COLUMNS = (
    "candidate_id",
    "prediction_signature",
    "decision_signature",
)

SIMCA_SEARCH_CONSTRAINTS = {
    "2way": {
        "max_fn_rate": INTERNAL_CALIBRATION_MAX_FN_RATE,
        "max_fp_rate": INTERNAL_CALIBRATION_MAX_FP_RATE,
        "min_balanced_accuracy": (
            INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY
        ),
        "max_fold_metric_std": (
            INTERNAL_CALIBRATION_MAX_FOLD_BALANCED_ACCURACY_STD
        ),
    },
    "3way": {
        "max_fn_rate": INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE,
        "max_fp_rate": INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE,
        "max_uncertain_rate": INTERNAL_CALIBRATION_MAX_UNCERTAIN_RATE,
        "min_coverage": INTERNAL_CALIBRATION_MIN_COVERAGE,
        "min_balanced_accuracy": (
            INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY
        ),
        "max_fold_metric_std": (
            INTERNAL_CALIBRATION_MAX_FOLD_BALANCED_ACCURACY_STD
        ),
    },
}
SIMCA_OPTUNA_OBJECTIVE_SPECS = {
    _track: {
        "objective_names": (
            *tuple(_spec["pareto_minimize"]),
            *tuple(_spec["pareto_maximize"]),
        ),
        "directions": (
            *tuple("minimize" for _ in _spec["pareto_minimize"]),
            *tuple("maximize" for _ in _spec["pareto_maximize"]),
        ),
    }
    for _track, _spec in SIMCA_EVALUATION_TRACK_SPECS.items()
}
# Legacy entry kept byte-for-byte compatible with the task-01 frozen bundle.
# Active 04B objectives are defined once, per track, in
# SIMCA_OPTUNA_OBJECTIVE_SPECS and protected by the child search-plan hash.
SIMCA_OPTUNA_DIRECTIONS = {
    "2way": ("minimize", "minimize", "minimize", "minimize"),
    "3way": (
        "minimize",
        "minimize",
        "minimize",
        "minimize",
        "minimize",
        "minimize",
    ),
}
SIMCA_OPTUNA_SEARCH_EFFICIENCY_COLUMNS = (
    "evaluation_track",
    "track_id",
    "decision_mode",
    "study_name",
    "study_scope",
    "eligibility_status",
    "study_status",
    "n_domain_configurations",
    "trial_budget",
    "n_trials",
    "n_complete_trials",
    "n_pruned_trials",
    "n_technical_errors",
    "n_unique_configurations_sampled",
    "duplicate_trial_rate",
    "domain_coverage_rate",
    "pareto_reference_scope",
    "n_exhaustive_pareto_configurations",
    "n_exhaustive_pareto_recovered",
    "exhaustive_pareto_recall",
    "uniform_recall_expectation",
    "pareto_recall_delta_vs_uniform",
    "pareto_recall_lift_vs_uniform",
    "budget_status",
    "optuna_conclusion",
    "exhaustive_reference_retained",
)

# Scientific selections in the rewritten 00-04B protocol must use hard
# constraints, explicit lexicographic priorities, or Pareto dominance.
ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS = (
    "selection_score",
    "robustness_score",
    "three_way_score",
    "weighted_score",
    "detection_score",
)

SIMCA_CALIBRATION_DOMAIN_COLUMNS = (
    "domain_config_id",
    "calibration_id",
    "evaluation_config_id",
    "projection_config_id",
    "fit_config_id",
    "source_config_id",
    "track_id",
    "evaluation_track",
    "parent_track",
    "decision_mode",
    "decision_score_type",
    "matrix_family",
    "matrix_method",
    "projection_level",
    "projection_matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "rule_family",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "direct_2way_threshold",
    "secondary_object_threshold",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "random_state",
    "calibration_status",
    "schema_version",
    "protocol_version",
    "protocol_hash",
    "pca_shortlist_id",
)
SIMCA_GRID_FOLD_METRIC_COLUMNS = (
    "domain_config_id",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "decision_mode",
    "projection_level",
    "map_variant",
    "aggregation_level",
    "group_id",
    "fold_id",
    "random_state",
    "n_observations",
    "n_target",
    "n_non_target",
    "n_target_objects",
    "macro_object_target_miss_rate",
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "status",
)
SIMCA_GRID_THRESHOLD_METRIC_COLUMNS = (
    "calibration_id",
    "evaluation_track",
    "track_id",
    "decision_mode",
    "projection_level",
    "map_variant",
    "n_domain_configurations",
    "n_seeds",
    "n_folds",
    "n_images",
    "n_observations",
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_image_target_miss_rate",
    "macro_image_false_accept_rate",
    "macro_image_balanced_accuracy",
    "macro_object_target_miss_rate",
    "worst_fold_target_miss_rate",
    "worst_fold_false_accept_rate",
    "fold_metric_std",
    "technical_status",
    "acceptability_status",
    "eligibility_status",
    "failure_reason",
)
SIMCA_GRID_TECHNICAL_AUDIT_COLUMNS = (
    "domain_config_id",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "technical_status",
    "calculable",
    "acceptability_status",
    "eligibility_status",
    "duplicate_status",
    "representative_calibration_id",
    "pareto_eligible",
    "error_type",
    "error_message",
)
SIMCA_GRID_DUPLICATE_GROUP_COLUMNS = (
    "duplicate_group_id",
    "duplicate_kind",
    "evaluation_track",
    "representative_calibration_id",
    "member_calibration_ids",
    "n_members",
    "score_signature",
    "decision_signature",
    "reason",
)
SIMCA_GRID_PARETO_REFERENCE_COLUMNS = (
    "row_type",
    "calibration_id",
    "evaluation_track",
    "track_id",
    "technical_status",
    "acceptability_status",
    "eligibility_status",
    "is_duplicate_representative",
    "diagnostic_pareto_front",
    "protocol_pareto_front",
    "pareto_exclusion_reason",
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_image_target_miss_rate",
    "macro_image_false_accept_rate",
    "macro_image_balanced_accuracy",
    "macro_object_target_miss_rate",
)
SIMCA_OPTUNA_TRIAL_COLUMNS = (
    "trial_number",
    "study_name",
    "evaluation_track",
    "track_id",
    "decision_mode",
    "study_scope",
    "study_seed",
    "domain_config_id",
    "calibration_id",
    "state",
    "status",
    "eligibility_status",
    "objective_values_json",
    "prune_reason",
    "error_type",
    "error_message",
    "evaluation_cache_hit",
    "is_duplicate_domain_config",
    "duplicate_of_trial_number",
    "datetime_start_utc",
    "datetime_complete_utc",
    "duration_seconds",
    "evaluation_source",
    "search_plan_hash",
)
SIMCA_OPTUNA_PARETO_COLUMNS = (
    "trial_number",
    "study_name",
    "evaluation_track",
    "track_id",
    "decision_mode",
    "study_scope",
    "domain_config_id",
    "calibration_id",
    "eligibility_status",
    "objective_values_json",
    "status",
    "diagnostic_optuna_front",
    "protocol_optuna_front",
    "downstream_eligible",
    "search_plan_hash",
)
SIMCA_OPTUNA_ERROR_COLUMNS = (
    "trial_number",
    "study_name",
    "evaluation_track",
    "domain_config_id",
    "calibration_id",
    "state",
    "prune_reason",
    "error_type",
    "error_message",
    "duration_seconds",
    "search_plan_hash",
)
SIMCA_ABLATION_PLAN_COLUMNS = (
    "ablation_id",
    "evaluation_track",
    "track_id",
    "reference_config_id",
    "ablated_config_id",
    "contrast_type",
    "factor",
    "reference_level",
    "ablated_level",
    "fit_changed",
    "projection_changed",
    "decision_changed",
    "spatial_processing_changed",
    "interaction_formula",
    "pairing_keys_json",
    "metric_set_json",
    "preregistered",
    "registration_status",
    "eligibility_status",
    "plan_status",
    "unsupported_reason",
    "protocol_hash",
    "search_plan_hash",
)


# Curated scientific settings serialized by the task-01 protocol freezer.
# The list is explicit so adding an unrelated runtime/display option does not
# silently change the scientific protocol hash.
PROTOCOL_CONFIGURATION_KEYS = (
    "PROTOCOL_VERSION",
    "RESULTS_SCHEMA_VERSION",
    "PROTOCOL_STATUS",
    "PROTOCOL_FREEZE_DATE",
    "PROTOCOL_REGISTRATION_MODE",
    "PROTOCOL_PRIOR_RESULTS_STATUS",
    "PROTOCOL_TEST_BLINDING_CLAIM",
    "PROTOCOL_AMENDMENT_JUSTIFICATION",
    "PROTOCOL_AMENDMENT_POLICY",
    "TARGET_CLASS",
    "NON_TARGET_LABEL",
    "REFERENCE_CLASSES",
    "PROTOCOL_CALIBRATION_BATCHES",
    "PROTOCOL_VALIDATION_BATCHES",
    "PROTOCOL_TEST_BATCHES",
    "CV_GROUP_COL",
    "DEFAULT_WAVELENGTH_MODE",
    "SPECTRAL_START_NM",
    "SPECTRAL_END_NM",
    "N_BANDS_RAW",
    "N_REMOVE_START",
    "N_STOP_END",
    "USE_WAVELENGTH_WINDOW",
    "WAVELENGTH_WINDOW_MIN_NM",
    "WAVELENGTH_WINDOW_MAX_NM",
    "DATA_MODE",
    "DATABASE_SKIP_UNKNOWN",
    "SEGMENTATION_OVERRIDE_RELATIVE_DIR",
    "SEGMENTATION_KWARGS",
    "QC_POLICY",
    "QC_SPECTRAL_GROUP_COLUMNS",
    "QC_SPECTRAL_OUTLIER_DISTANCE_THRESHOLD",
    "QC_ZERO_VARIANCE_EPSILON",
    "QC_REVIEW_ALLOWED_DECISIONS",
    "QC_REVIEW_REQUIRED_STATUS",
    "SPATIAL_GT_ALLOWED_LEVELS",
    "SPATIAL_GT_PRIMARY_PIXEL_LEVELS",
    "SPATIAL_GT_ANNOTATION_TOOL",
    "SPATIAL_GT_ANNOTATION_TOOL_VERSION",
    "SPATIAL_GT_ANNOTATION_PROTOCOL_VERSION",
    "SPATIAL_GT_DOUBLE_ANNOTATION_POLICY",
    "SPATIAL_GT_DOUBLE_ANNOTATION_FRACTION",
    "SPATIAL_GT_ANNOTATION_FRACTION",
    "SPATIAL_GT_TEST_BATCHES",
    "SPATIAL_GT_TARGET_CLASS",
    "SPATIAL_GT_ANNOTATED_CLASS",
    "SPATIAL_GT_POSITIVE_VALUE",
    "SPATIAL_GT_POSITIVE_CLASS",
    "SPATIAL_GT_POSITIVE_DEFINITION",
    "SPATIAL_GT_NEGATIVE_VALUE",
    "SPATIAL_GT_NEGATIVE_DEFINITION",
    "SPATIAL_GT_OUTSIDE_ROI_DEFINITION",
    "SPATIAL_GT_MASK_SEMANTICS_ID",
    "SPATIAL_GT_BOUNDARY_POLICY_ID",
    "SPATIAL_GT_AMBIGUITY_POLICY_ID",
    "SPATIAL_GT_ROI_SOURCE",
    "SPATIAL_GT_COMPONENT_CONNECTIVITY",
    "SPATIAL_AGREEMENT_MIN_DICE",
    "SPATIAL_AGREEMENT_MIN_IOU",
    "SPATIAL_AGREEMENT_MAX_UNMATCHED_COMPONENT_RATE",
    "SIMCA_OBJECT_MATRIX_METHODS",
    "SIMCA_PIXEL_MATRIX_METHODS",
    "SIMCA_ALPHA_VALUES",
    "SIMCA_OBJECT_THRESHOLDS",
    "SIMCA_MATRIX_FAMILIES",
    "SIMCA_PROJECTION_LEVELS",
    "SIMCA_DECISION_MODES",
    "SIMCA_PARENT_TRACKS",
    "SIMCA_PARENT_TRACK_SPECS",
    "SIMCA_EVALUATION_TRACKS",
    "SIMCA_EVALUATION_TRACK_IDS",
    "SIMCA_EVALUATION_TRACK_SPECS",
    "BALANCED_SAMPLING_M_VALUES",
    "BALANCED_SAMPLING_UNDER_M_POLICY",
    "BALANCED_SAMPLING_SEEDS",
    "BALANCED_SAMPLING_MIN_ELIGIBLE_RATE",
    "BALANCED_SAMPLING_STUDY_M_VALUES",
    "PREPROCESSING_CONFIGS_TO_COMPARE",
    "SG_WINDOW_CHOICES",
    "SG_DEFAULT_WINDOW",
    "SG_POLYORDER",
    "PREPROCESSING_ZERO_VARIANCE_EPSILON",
    "PREPROCESSING_MAX_ZERO_VARIANCE_BAND_RATE",
    "PREPROCESSING_SATURATION_BOUNDS",
    "PREPROCESSING_REPEATABILITY_TOLERANCE",
    "PREPROCESSING_ABSORBANCE_NONPOSITIVE_POLICY",
    "PREPROCESSING_MATRIX_SPECS",
    "PCA_CALIBRATION_BATCHES",
    "PCA_FORBIDDEN_BATCHES",
    "PCA_SAMPLE_KIND",
    "PCA_N_COMPONENTS",
    "PCA_DIAGNOSTIC_N_COMPONENTS",
    "PCA_MATRIX_METHODS",
    "PCA_BALANCED_M_VALUES",
    "PCA_BALANCED_STRATEGIES",
    "PCA_SG_WINDOW_LENGTH",
    "PCA_SG_POLYORDER",
    "PCA_BALANCED_UNDER_M_POLICY",
    "PCA_STABILITY_SEEDS",
    "PCA_STABILITY_REFERENCE_SEED",
    "PCA_STABILITY_N_SPLITS",
    "PCA_STABILITY_N_BOOTSTRAP",
    "PCA_STABILITY_N_COMPONENTS",
    "PCA_STABILITY_GROUP_COL",
    "PCA_STABILITY_BOOTSTRAP_GROUP_COL",
    "MAX_PCA_PREPROCESSINGS_PER_FAMILY",
    "PCA_SELECTION_EXPECTED_FAMILIES",
    "PCA_SELECTION_PROFILES",
    "PCA_ARTIFACT_REVIEW_REQUIRED_STATUS",
    "PCA_ARTIFACT_REVIEW_ALLOWED_STATUSES",
    "PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS",
    "INTERNAL_CALIBRATION_BATCHES",
    "INTERNAL_CALIBRATION_FORBIDDEN_BATCHES",
    "INTERNAL_CALIBRATION_GROUP_COL",
    "INTERNAL_CALIBRATION_N_SPLITS",
    "INTERNAL_CALIBRATION_FOLD_RANDOM_STATE",
    "INTERNAL_CALIBRATION_SIZE_N_BINS",
    "INTERNAL_CALIBRATION_RANDOM_SEEDS",
    "INTERNAL_CALIBRATION_MATRIX_METHODS",
    "INTERNAL_CALIBRATION_M_VALUES",
    "INTERNAL_CALIBRATION_PIXEL_STRATEGIES",
    "INTERNAL_CALIBRATION_N_COMPONENTS_VALUES",
    "INTERNAL_CALIBRATION_ALPHA_VALUES",
    "INTERNAL_CALIBRATION_SG_WINDOWS",
    "INTERNAL_CALIBRATION_SG_POLYORDERS",
    "INTERNAL_CALIBRATION_AVAILABLE_DILATION_RADII",
    "INTERNAL_CALIBRATION_DILATION_RADII",
    "INTERNAL_CALIBRATION_UNDER_M_POLICY",
    "INTERNAL_CALIBRATION_RULE_VARIANTS",
    "INTERNAL_CALIBRATION_OBJECT_THRESHOLDS",
    "INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD",
    "INTERNAL_CALIBRATION_PIXEL_VOTE_CENTER",
    "INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES",
    "INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES",
    "INTERNAL_CALIBRATION_THRESHOLD_CROSSFIT",
    "INTERNAL_CALIBRATION_THREE_WAY_LOWER_THRESHOLDS",
    "INTERNAL_CALIBRATION_THREE_WAY_UPPER_THRESHOLDS",
    "INTERNAL_CALIBRATION_RISK_PROFILE",
    "INTERNAL_CALIBRATION_RISK_PROFILES",
    "PROJECTION_DOMAIN_AUDIT_RULE_VERSION",
    "PROJECTION_DOMAIN_AUDIT_ALLOWED_BATCHES",
    "PROJECTION_DOMAIN_AUDIT_FORBIDDEN_BATCHES",
    "PROJECTION_DOMAIN_BORDER_WIDTH",
    "PROJECTION_DOMAIN_MIN_STRATUM_N",
    "PROJECTION_DOMAIN_DIAGNOSTIC_DIMENSIONS",
    "PROJECTION_DOMAIN_ELIGIBILITY_DIMENSIONS",
    "PROJECTION_DOMAIN_ELIGIBILITY_THRESHOLDS",
    "SPATIAL_CALIBRATION_RULE_VERSION",
    "SPATIAL_CALIBRATION_ALLOWED_BATCHES",
    "SPATIAL_CALIBRATION_FORBIDDEN_BATCHES",
    "SPATIAL_CALIBRATION_REQUIRED_TRUTH_LEVELS",
    "SPATIAL_CALIBRATION_TRUTH_SOURCE",
    "SPATIAL_CALIBRATION_REQUIRED_CLASSES",
    "SPATIAL_CALIBRATION_CONNECTIVITIES",
    "SPATIAL_CALIBRATION_MORPHOLOGY_OPERATIONS",
    "SPATIAL_CALIBRATION_MORPHOLOGY_RADII",
    "SPATIAL_CALIBRATION_MIN_AREAS",
    "SPATIAL_CALIBRATION_FRAGMENT_AREA_UPPER_BOUNDS",
    "SPATIAL_CALIBRATION_FRAGMENT_AREA_LABELS",
    "SPATIAL_CALIBRATION_SELECTION_TOLERANCE",
    "SIMCA_OPTUNA_PURPOSE",
    "SIMCA_OPTUNA_N_TRIALS_PER_TRACK",
    "SIMCA_OPTUNA_N_STARTUP_TRIALS",
    "SIMCA_OPTUNA_RANDOM_STATE",
    "SIMCA_OPTUNA_DIRECTIONS",
    "ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS",
    "PROTOCOL_PRIMARY_HYPOTHESES",
    "PROTOCOL_RATE_PRACTICAL_TOLERANCE",
    "PROTOCOL_STANDARDIZED_SHIFT_TOLERANCE",
    "PROTOCOL_CONFIDENCE_LEVEL",
    "PROTOCOL_BOOTSTRAP_GROUP_COL",
    "PROTOCOL_BOOTSTRAP_N_RESAMPLES",
    "PROTOCOL_BOOTSTRAP_RANDOM_STATE",
    "PROTOCOL_MULTIPLICITY_METHOD",
    "PROTOCOL_SELECTION_POLICY",
)
