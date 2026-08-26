"""Central experiment configuration for the HSI nuts workflow.

The notebooks keep local variable names for readability, but those variables
should be initialized from this module so the train/validation/test protocol and
main search grids stay consistent across the project.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Frozen scientific protocol: tasks 01-02
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = "8tracks_v5"
RESULTS_SCHEMA_VERSION = "8tracks_v5"
PROTOCOL_STATUS = "frozen"
PROTOCOL_FREEZE_DATE = "2026-08-12"
PROTOCOL_REGISTRATION_MODE = "prospective_rebuild_before_internal_calibration"
PROTOCOL_PRIOR_RESULTS_STATUS = "legacy_exploratory"
PROTOCOL_TEST_BLINDING_CLAIM = "batch3_results_precede_spectral_validity_amendment"
PROTOCOL_AMENDMENT_JUSTIFICATION = (
    "2026-08-07: spectral acquisition validity amendment introduced after "
    "inspection of the raw reflectance domain. All 20 negative reflectance "
    "values were confined to the terminal 1702 nm band, indicating an "
    "unreliable terminal acquisition band. In addition, 1197 exact zero "
    "values were identified as 19 complete all-zero pixel spectra across "
    "the retained spectral axis, consistent with pixel-level no-data rather "
    "than valid reflectance measurements. The amendment therefore removes "
    "the 1702 nm terminal band globally and excludes spectrally invalid "
    "pixels before object aggregation, pixel sampling, matrix construction "
    "and preprocessing. No clipping or value imputation is introduced. "
    "The absorbance transformation retains its strict R > 0 requirement. "
    "These changes are motivated by physical and data-integrity diagnostics, "
    "not by downstream classification performance. All downstream models "
    "and selections must be recomputed under this amended protocol."
    "2026-08-12: before rerunning notebook 03B, the internal "
    "calibration architecture was rebuilt. Random seeds are now treated "
    "as repeated executions of one model rather than selectable "
    "hyperparameters. Three-way threshold policies are compared by their "
    "prespecified quantile coordinates using cross-fitted predictions. "
    "False-negative risk is controlled by stricter hard constraints and "
    "a lexicographic FN-first threshold policy. Pareto selection remains "
    "unweighted. No batch-3 or batch-4 observation was used to define "
    "these rules."
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
INTERNAL_CALIBRATION_CONSTRAINT_PROFILE_ID = "fn-priority_v1"

# ---------------------------------------------------------------------------
# Canonical paths for notebooks 00-03
# ---------------------------------------------------------------------------
RAW_MAT_RELATIVE_PATH = (
    "HSI Data",
    "NIR camera UCO (889-1702 nm)",
    "NIR_uco_sb.mat",
)
PROTOCOL_ARTIFACT_RELATIVE_DIR = ("docs", "protocol", PROTOCOL_VERSION,)
DATABASE_H5_RELATIVE_PATH = ("HSI Data", "processed", f"nir_uco_database_{PROTOCOL_VERSION}.h5")
DATABASE_RESULTS_RELATIVE_DIR = ("results", f"00_database_{PROTOCOL_VERSION}")
QC_RESULTS_RELATIVE_DIR = ("results", f"01_quality_check_{PROTOCOL_VERSION}")
SPATIAL_GT_RESULTS_RELATIVE_DIR = ("results", f"01B_spatial_ground_truth_{PROTOCOL_VERSION}")
MATRIX_RESULTS_DIR_PREFIX = f"02_matrices_{PROTOCOL_VERSION}"
PCA_RESULTS_DIR_PREFIX = f"03_pca_{PROTOCOL_VERSION}"
INTERNAL_CALIBRATION_RESULTS_DIR_PREFIX = f"03B_internal_calibration_{PROTOCOL_VERSION}"
DOMAIN_SPATIAL_CALIBRATION_RESULTS_DIR_PREFIX = (
    f"03C_projection_spatial_calibration_{PROTOCOL_VERSION}"
)
SIMCA_GRID_SEARCH_RESULTS_DIR_PREFIX = f"04A_simca_grid_search_{PROTOCOL_VERSION}"
SIMCA_OPTUNA_RESULTS_DIR_PREFIX = f"04B_simca_optuna_search_{PROTOCOL_VERSION}"
SIMCA_CONCAT_REFIT_RESULTS_DIR_PREFIX = f"04C_simca_concat_refit_{PROTOCOL_VERSION}"
SIMCA_ROBUSTNESS_RESULTS_DIR_PREFIX = f"05_simca_validation_robustness_{PROTOCOL_VERSION}"

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
    "terminal_band_qc": "terminal_band_qc.parquet",
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
    "pixel_spectral_qc": "pixel_spectral_qc.parquet",
    "pixel_exclusions": "pixel_exclusions.parquet",
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
    "candidate_registry": "pca_candidate_registry.parquet",
    "summary": "pca_summary.parquet",
    "diagnostics": "pca_scoring_diagnostics.parquet",
    "preprocessing_summary": "pca_preprocessing_summary.parquet",
    "selected": "pca_selected_preprocessings.parquet",
    "artifact_review": "pca_artifact_review.parquet",
    "visual_review": "pca_visual_review.pdf",
    "selection_audit": "pca_selection_audit.parquet",
}
INTERNAL_CALIBRATION_OUTPUT_FILENAMES = {
    "track_contracts": "track_contracts.parquet",
    "folds": "calibration_folds.parquet",
    "fold_diagnostics": "fold_diagnostics.parquet",
    "model_catalog": "model_catalog.parquet",
    "candidate_runs": "candidate_runs.parquet",
    "fit_diagnostics": "fit_diagnostics.parquet",
    "rule_diagnostics": "rule_diagnostics.parquet",
    "projection_shift": "projection_shift.parquet",
    "oof_object_predictions": "oof_object_predictions.parquet",
    "oof_pixel_predictions": "oof_pixel_predictions.parquet",
    "threshold_metrics": "threshold_metrics.parquet",
    "model_metrics": "model_metrics.parquet",
    "selected_models": "selected_models.parquet",
    "selected_runs": "selected_runs.parquet",
    "selected_thresholds": "selected_thresholds.parquet",
    "selection_audit": "selection_audit.parquet",
    "technical_events": "technical_events.parquet",
    "checkpoint_manifest": "checkpoint_manifest.json",
}
DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES = {
    "projection_shift_diagnostics": "projection_shift_diagnostics.parquet",
    "projection_eligibility": "projection_eligibility.parquet",
    "spatial_calibration_metrics": "spatial_calibration_metrics.parquet",
    "fragment_size_classes": "fragment_size_classes.parquet",
    "spatial_postprocessing_lock": "spatial_postprocessing_lock.json",
    "audit_manifest": "audit_manifest.json",
}
SIMCA_GRID_SEARCH_OUTPUT_FILENAMES = {
    "model_reference": "selected_model_reference.parquet",
    "fold_metrics": "selected_run_fold_metrics.parquet",
    "audit_manifest": "audit_manifest.json",
}
SIMCA_OPTUNA_OUTPUT_FILENAMES = {
    "sampled_models": "categorical_tpe_sampled_models.parquet",
    "search_efficiency": "categorical_tpe_coverage.parquet",
    "audit_manifest": "audit_manifest.json",
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
    "technical_events": "validation_technical_events.parquet",
}
SIMCA_ROBUSTNESS_OUTPUT_FILENAMES = {
    "selection_units": "validation_selection_units.parquet",
    "selection_members": "validation_selection_members.parquet",
    "pareto_candidates": "validation_pareto_candidates.parquet",
    "pareto_audit": "validation_pareto_audit.parquet",
    "seed_executions": "robustness_seed_executions.parquet",
    "seed_thresholds": "robustness_seed_thresholds.parquet",
    "seed_metrics": "robustness_seed_metrics.parquet",
    "stability_summary": "model_seed_stability.parquet",
    "seed_disagreement": "seed_decision_disagreement.parquet",
    "ablation_plan": "robustness_ablation_plan.parquet",
    "ablation_diagnostics": "robustness_ablation_diagnostics.parquet",
    "statistical_uncertainty": "statistical_uncertainty.parquet",
    "risk_coverage": "risk_coverage_curves.parquet",
    "review_guardrails": "robustness_review_guardrails.parquet",
    "track_scoring_flags": "track_scoring_flags.parquet",
    "pure_test_candidates": "pure_test_candidate_registry.parquet",
    "protocol": "robustness_review_protocol.json",
    "lock_manifest": "robustness_review_lock.json",
    "threshold_sensitivity_plan": "threshold_sensitivity_plan.parquet",
    "threshold_sensitivity_metrics": "threshold_sensitivity_metrics.parquet",
    "threshold_sensitivity_decisions": "threshold_sensitivity_decisions.parquet",
    "threshold_stability": "threshold_calibration_stability.parquet",
    "source_image_influence": "source_image_influence.parquet",
    "fold_sensitivity_plan": "calibration_fold_sensitivity_plan.parquet",
    "fold_sensitivity_assignments": "calibration_fold_sensitivity_assignments.parquet",
    "fold_sensitivity_thresholds": "calibration_fold_sensitivity_thresholds.parquet",
    "fold_sensitivity_metrics": "calibration_fold_sensitivity_metrics.parquet",
    "fold_sensitivity_decisions": "calibration_fold_sensitivity_decisions.parquet",
    "fold_sensitivity_technical_events": "calibration_fold_sensitivity_technical_events.parquet",
    "pareto_robustness_replicates": "pareto_robustness_replicates.parquet",
    "pareto_robustness_summary": "pareto_robustness_summary.parquet",
    "pareto_robustness_audit": "pareto_robustness_audit.parquet",
    "spatial_sensitivity_plan": "spatial_sensitivity_plan.parquet",
    "spatial_sensitivity_metrics": "spatial_sensitivity_metrics.parquet",
    "ablation_coverage": "robustness_ablation_coverage.parquet",
}


# ---------------------------------------------------------------------------
# Spectral acquisition validity
# ---------------------------------------------------------------------------

TERMINAL_BAND_QC_POLICY = {
    "rule_version": "terminal_band_qc_v1",
    "n_terminal_bands": 5,
    "exclude_all_zero_pixels_from_diagnostics": True,
    "negative_reflectance_rule": "any_negative_blocks_terminal_band",
    "flag_any_negative_reflectance": True,
    "action_on_failure": "exclude_terminal_band_before_database_freeze",
}


SPECTRAL_PIXEL_VALIDITY_POLICY = {
    "rule_version": "spectral_pixel_validity_v1",
    "require_finite": True,
    "exclude_all_zero": True,
    "all_zero_atol": 0.0,
    "require_strictly_positive": True,
    "invalid_pixel_action": "exclude_from_all_spectral_representations",
    "object_action": (
        "retain_object_if_at_least_one_analysis_valid_pixel_remains"
    ),
    "imputation": "forbidden",
    "clipping": "forbidden",
}


# ---------------------------------------------------------------------------
# Canonical low-level database construction
# ---------------------------------------------------------------------------
DEFAULT_WAVELENGTH_MODE = "non_noisy_all"
DEFAULT_RESULTS_TAG = "px_qc_v1"
SPECTRAL_START_NM = 889.0
SPECTRAL_END_NM = 1702.0
N_BANDS_RAW = 69
N_REMOVE_START = 6
N_STOP_END = 67
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
    "8tracks_v4",
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
    "n_analysis_valid_pixels",
    "n_analysis_invalid_pixels",
    "analysis_invalid_pixel_rate",
    "n_all_zero_pixels",
    "n_nonpositive_pixels",
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

PIXEL_SPECTRAL_QC_COLUMNS = (
    "object_id",
    "source_image",
    "batch",
    "label",
    "pixel_index",
    "row",
    "col",
    "n_bands",
    "n_zero",
    "n_nonpositive",
    "zero_fraction",
    "min_reflectance",
    "finite",
    "all_zero_spectrum",
    "has_nonpositive_reflectance",
    "analysis_valid",
    "invalid_reason",
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
    "spectral_pixel_policy_hash",
    "pixel_exclusion_hash",
    "alerts_hash",
    "review_hash",
    "n_alerts",
    "n_pending",
    "n_excluded",
    "n_pixel_excluded",
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

# LEGACY ------------------------------------------
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
# -----------------------------------------------------------

# Backward-compatible names used by the current four-parent-track notebooks.
# New eight-track code must use SIMCA_EVALUATION_TRACKS and
# SIMCA_EVALUATION_TRACK_SPECS for projection-aware evaluation and Pareto.
LEGACY_SIMCA_SELECTION_TRACKS = SIMCA_PARENT_TRACKS
LEGACY_SIMCA_SELECTION_TRACK_SPECS = SIMCA_PARENT_TRACK_SPECS

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
        "secondary_object_aggregation_thresholds": tuple(map(float, SIMCA_OBJECT_THRESHOLDS)),
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
        "secondary_object_aggregation_thresholds": tuple(map(float, SIMCA_OBJECT_THRESHOLDS)),
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
    _spec["constraint_profile_id"] = INTERNAL_CALIBRATION_CONSTRAINT_PROFILE_ID
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


SELECTION_ID_PREFIXES = {
    "pca_candidate": "pca_candidate",
    "pca_preprocessing": "pca_preproc",
    "model": "model",
    "fit": "fit",
    "projection": "projection",
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
    "n_raw",
    "n_available",
    "n_invalid",
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
    "selection_unit_id",
    "matrix_family",
    "preprocessing",
    "preprocessing_steps",
    "sg_window_length",
    "sg_polyorder",
    "wavelength_axis_id",
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


# ---------------------------------------------------------------------------
# Notebook 05: batch-3 robustness and pre-batch4 candidate review
# ---------------------------------------------------------------------------
# Notebook 05 is a child contract of the frozen 8tracks_v5 protocol.
#
# Scientific roles:
# - batch 3 is the locked validation set;
# - the initial Pareto comparison uses only the common 03B/04C execution panel;
# - additional seeds are evaluated only after that Pareto step and only for
#   stochastic Pareto models;
# - additional seeds never create a new model_id;
# - threshold-policy coordinates selected in 03B are frozen;
# - only seed-specific numeric thresholds are recalibrated from batches 1-2;
# - notebook 05 does not open batch 4 and does not perform final model
#   selection.
#
# These settings intentionally remain outside PROTOCOL_CONFIGURATION_KEYS:
# modifying them must change the notebook-05 child-contract hash, not the
# already frozen parent 8tracks_v5 hash used by notebooks 03B-04C.
SIMCA_ROBUSTNESS_CONTRACT_VERSION = "8tracks_v5_notebook05_v3"
SIMCA_ROBUSTNESS_CONTRACT_ROLE = "batch3_robustness_and_pre_batch4_candidate_review"
SIMCA_ROBUSTNESS_SELECTION_SCOPE = "within_track"
SIMCA_ROBUSTNESS_ALLOW_CROSS_TRACK_SELECTION = False
SIMCA_ROBUSTNESS_ALLOW_BATCH4_INPUTS = False
SIMCA_ROBUSTNESS_FINAL_MODEL_SELECTION_PERFORMED = False
# Canonical eight-track order. Do not duplicate the E1-E8 definition.
SIMCA_ROBUSTNESS_TRACK_IDS = tuple(
    SIMCA_EVALUATION_TRACK_IDS[track]
    for track in SIMCA_EVALUATION_TRACKS
)
if SIMCA_ROBUSTNESS_TRACK_IDS != tuple(
    f"E{index}" for index in range(1, 9)
):
    raise RuntimeError(
        "Notebook 05 requires exactly the eight evaluation tracks E1-E8."
    )
# ---------------------------------------------------------------------------
# Random-state contract
# ---------------------------------------------------------------------------
# The base panel is the panel already evaluated symmetrically by 04C.
# It is the only panel allowed to determine the validation Pareto front.
SIMCA_ROBUSTNESS_BASE_RANDOM_STATES = (0, 1, 2)
SIMCA_ROBUSTNESS_RANDOM_STATES = (0, 1, 2, 3, 4, 5, 10, 20, 42, 100)
SIMCA_ROBUSTNESS_ADDITIONAL_RANDOM_STATES = tuple(
    seed
    for seed in SIMCA_ROBUSTNESS_RANDOM_STATES
    if seed not in SIMCA_ROBUSTNESS_BASE_RANDOM_STATES
)
SIMCA_ROBUSTNESS_RUN_ADDITIONAL_SEEDS = True
SIMCA_ROBUSTNESS_RECOMPUTE_PARETO_AFTER_ADDITIONAL_SEEDS = False
SIMCA_ROBUSTNESS_RESELECT_THRESHOLD_POLICY_FOR_ADDITIONAL_SEEDS = False
SIMCA_ROBUSTNESS_RECALIBRATE_NUMERIC_THRESHOLDS_FOR_ADDITIONAL_SEEDS = True

if len(SIMCA_ROBUSTNESS_BASE_RANDOM_STATES) != len(
    set(SIMCA_ROBUSTNESS_BASE_RANDOM_STATES)
):
    raise RuntimeError(
        "SIMCA_ROBUSTNESS_BASE_RANDOM_STATES contains duplicates."
    )
if len(SIMCA_ROBUSTNESS_RANDOM_STATES) != len(
    set(SIMCA_ROBUSTNESS_RANDOM_STATES)
):
    raise RuntimeError(
        "SIMCA_ROBUSTNESS_RANDOM_STATES contains duplicates."
    )
if not set(SIMCA_ROBUSTNESS_BASE_RANDOM_STATES).issubset(
    SIMCA_ROBUSTNESS_RANDOM_STATES
):
    raise RuntimeError(
        "Every base random state must belong to the robustness panel."
    )
if set(SIMCA_ROBUSTNESS_ADDITIONAL_RANDOM_STATES).intersection(
    SIMCA_ROBUSTNESS_BASE_RANDOM_STATES
):
    raise RuntimeError(
        "Additional robustness seeds must exclude the base 04C seeds."
    )

# ---------------------------------------------------------------------------
# Stochastic-model definition
# ---------------------------------------------------------------------------
# Only random balanced-pixel sampling introduces stochastic sampling in the
# current protocol. Deterministic models must not be cloned across seeds.
SIMCA_STOCHASTIC_MATRIX_METHODS = ("balanced_pixels",)
SIMCA_STOCHASTIC_SAMPLING_STRATEGIES = ("random",)

# ---------------------------------------------------------------------------
# Upstream eligibility / downstream review contract
# ---------------------------------------------------------------------------
SIMCA_ROBUSTNESS_SUPPORTED_ELIGIBILITY_STATUSES = (
    "eligible",
    "eligible_with_warning",
)
SIMCA_ROBUSTNESS_SUPPORTED_DOWNSTREAM_STATUSES = ("supported",)
SIMCA_ROBUSTNESS_PROTOCOL_CANDIDATE_STATUSES = ("pass",)
SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT = "raw"
SIMCA_ROBUSTNESS_SPATIAL_MAP_VARIANT = "locked_postprocessed"
SIMCA_ROBUSTNESS_PARETO_EPSILON = 1e-12
SIMCA_ROBUSTNESS_REQUIRE_STABILITY_FOR_PURE_TEST = True
SIMCA_ROBUSTNESS_PURE_TEST_STABILITY_STATUSES = (
    "robust",
    "robust_with_supporting_warnings",
    "not_applicable_deterministic",
)


# Metrics retained from 04C. The robustness module also preserves any extra
# long-form metric present in validation_metrics; these lists define stable
# diagnostics and the spatial columns expected by downstream notebooks.
SIMCA_ROBUSTNESS_IMAGE_RISK_METRICS = (
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
)
SIMCA_ROBUSTNESS_IMAGE_PERFORMANCE_METRICS = (
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
)
SIMCA_ROBUSTNESS_SPATIAL_METRICS = (
    "dice",
    "iou",
    "pixel_precision",
    "pixel_recall",
    "component_precision",
    "component_recall",
    "split_rate",
    "merge_rate",
    "smallest_fragment_recall",
)

# Long 04C metrics materialized by notebook 05. Keeping this list in the
# configuration makes the execution/model summary schemas deterministic while
# the original long 04C tables remain the lossless source of record.
SIMCA_ROBUSTNESS_DECISION_SCOPES = ("direct", "pixel_to_object")
SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES = (
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
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_object_target_miss_rate",
    "macro_image_target_miss_rate",
    "macro_image_false_accept_rate",
    "macro_image_uncertain_rate",
    "macro_image_coverage_rate",
    "macro_image_balanced_accuracy",
    "macro_image_decided_balanced_accuracy",
)
SIMCA_ROBUSTNESS_WORST_IMAGE_METRIC_NAMES = (
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_object_target_miss_rate",
)

# Direction is declared once per metric. Notebook 05 uses it both to define a
# conservative across-seed value (max for a minimized risk, min for a
# maximized performance metric) and to identify the worst source-image value.
SIMCA_ROBUSTNESS_METRIC_DIRECTIONS = {
    "target_miss_rate": "minimize",
    "false_accept_rate": "minimize",
    "uncertain_rate": "minimize",
    "target_uncertain_rate": "minimize",
    "non_target_uncertain_rate": "minimize",
    "macro_object_target_miss_rate": "minimize",
    "macro_image_target_miss_rate": "minimize",
    "macro_image_false_accept_rate": "minimize",
    "macro_image_uncertain_rate": "minimize",
    "split_rate": "minimize",
    "merge_rate": "minimize",
    "coverage_rate": "maximize",
    "balanced_accuracy": "maximize",
    "decided_balanced_accuracy": "maximize",
    "macro_image_coverage_rate": "maximize",
    "macro_image_balanced_accuracy": "maximize",
    "macro_image_decided_balanced_accuracy": "maximize",
    "dice": "maximize",
    "iou": "maximize",
    "pixel_precision": "maximize",
    "pixel_recall": "maximize",
    "component_precision": "maximize",
    "component_recall": "maximize",
    "smallest_fragment_recall": "maximize",
}
SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_LIMITS = {
    "decision_disagreement_rate": 0.10,
    "target_decision_disagreement_rate": 0.05,
}
# SIMCA_ROBUSTNESS_FINAL_ADMISSIBLE_STABILITY_STATUSES = (
#     "robust",
#     "not_applicable_deterministic",
# )
SIMCA_ROBUSTNESS_MIN_IMAGES_FOR_CLUSTER_INTERVAL = 5
SIMCA_ROBUSTNESS_MIN_IMAGES_FOR_PRIMARY_INFERENCE = 20
SIMCA_ROBUSTNESS_RISK_COVERAGE_GRID = (
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 1.00,
)
SIMCA_ROBUSTNESS_SENSITIVITY_TOLERANCES = {
    "target_miss_rate": 0.02,
    "false_accept_rate": 0.05,
    "uncertain_rate": 0.05,
    "target_uncertain_rate": 0.05,
    "non_target_uncertain_rate": 0.05,
    "coverage_rate": 0.05,
    "balanced_accuracy": 0.03,
    "decided_balanced_accuracy": 0.03,
    "macro_object_target_miss_rate": 0.02,
    "macro_image_target_miss_rate": 0.02,
    "macro_image_false_accept_rate": 0.05,
    "macro_image_uncertain_rate": 0.05,
    "macro_image_coverage_rate": 0.05,
    "macro_image_balanced_accuracy": 0.03,
    "macro_image_decided_balanced_accuracy": 0.03,
    "dice": 0.03,
    "iou": 0.03,
    "pixel_precision": 0.05,
    "pixel_recall": 0.05,
    "component_precision": 0.05,
    "component_recall": 0.05,
    "smallest_fragment_recall": 0.05,
    "split_rate": 0.05,
    "merge_rate": 0.05,
}
SIMCA_ROBUSTNESS_ABLATION_METRICS = (
    "direct__target_miss_rate",
    "direct__false_accept_rate",
    "direct__uncertain_rate",
    "direct__balanced_accuracy",
    "direct__decided_balanced_accuracy",
    "pixel_to_object__target_miss_rate",
    "pixel_to_object__false_accept_rate",
    "pixel_to_object__uncertain_rate",
    "pixel_to_object__decided_balanced_accuracy",
    "spatial__dice",
    "spatial__iou",
    "spatial__component_precision",
    "spatial__component_recall",
    "spatial__smallest_fragment_recall",
    "spatial__split_rate",
    "spatial__merge_rate",
)
SIMCA_ROBUSTNESS_STABILITY_LIMITS = {
    "target_miss_rate": {"max_std": 0.02, "max_range": 0.05},
    "false_accept_rate": {"max_std": 0.05, "max_range": 0.15},
    "uncertain_rate": {"max_std": 0.05, "max_range": 0.15},
    "target_uncertain_rate": {"max_std": 0.05, "max_range": 0.15},
    "non_target_uncertain_rate": {"max_std": 0.05, "max_range": 0.15},
    "balanced_accuracy": {"max_std": 0.03, "max_range": 0.08},
    "decided_balanced_accuracy": {"max_std": 0.03, "max_range": 0.08},
    "macro_object_target_miss_rate": {"max_std": 0.02, "max_range": 0.05},
    "macro_image_target_miss_rate": {"max_std": 0.02, "max_range": 0.05},
    "macro_image_false_accept_rate": {"max_std": 0.05, "max_range": 0.15},
    "macro_image_uncertain_rate": {"max_std": 0.05, "max_range": 0.15},
    "macro_image_balanced_accuracy": {"max_std": 0.03, "max_range": 0.08},
    "macro_image_decided_balanced_accuracy": {"max_std": 0.03, "max_range": 0.08},
    "smallest_fragment_recall": {"max_std": 0.05, "max_range": 0.10},
    "component_precision": {"max_std": 0.05, "max_range": 0.10},
    "component_recall": {"max_std": 0.05, "max_range": 0.10},
}

# Pareto objectives are expressed on the model-level conservative seed
# aggregates produced by simca_robustness. Minimized metrics use the worst seed
# (maximum); maximized metrics use the worst seed (minimum).
SIMCA_ROBUSTNESS_PARETO_OBJECTIVES = {
    "E1": {
        "minimize": (
            "direct__target_miss_rate",
            "direct__false_accept_rate",
        ),
        "maximize": (),
    },
    "E2": {
        "minimize": (
            "direct__target_miss_rate",
            "direct__false_accept_rate",
            "direct__target_uncertain_rate",
            "direct__non_target_uncertain_rate",
        ),
        "maximize": ("direct__decided_balanced_accuracy",),
    },
    "E3": {
        "minimize": (
            "direct__macro_image_target_miss_rate",
            "direct__macro_image_false_accept_rate",
            "direct__macro_object_target_miss_rate",
            "pixel_to_object__target_miss_rate",
            "pixel_to_object__false_accept_rate",
        ),
        "maximize": (),
    },
    "E4": {
        "minimize": (
            "direct__macro_image_target_miss_rate",
            "direct__macro_image_false_accept_rate",
            "direct__macro_object_target_miss_rate",
            "direct__macro_image_uncertain_rate",
            "pixel_to_object__target_miss_rate",
            "pixel_to_object__false_accept_rate",
            "pixel_to_object__target_uncertain_rate",
            "pixel_to_object__non_target_uncertain_rate",
        ),
        "maximize": (
            "direct__macro_image_decided_balanced_accuracy",
            "pixel_to_object__decided_balanced_accuracy",
        ),
    },
    "E5": {
        "minimize": (
            "direct__target_miss_rate",
            "direct__false_accept_rate",
        ),
        "maximize": (),
    },
    "E6": {
        "minimize": (
            "direct__target_miss_rate",
            "direct__false_accept_rate",
            "direct__target_uncertain_rate",
            "direct__non_target_uncertain_rate",
        ),
        "maximize": ("direct__decided_balanced_accuracy",),
    },
    "E7": {
        "minimize": (
            "direct__macro_image_target_miss_rate",
            "direct__macro_image_false_accept_rate",
            "direct__macro_object_target_miss_rate",
            "pixel_to_object__target_miss_rate",
            "pixel_to_object__false_accept_rate",
        ),
        "maximize": (),
    },
    "E8": {
        "minimize": (
            "direct__macro_image_target_miss_rate",
            "direct__macro_image_false_accept_rate",
            "direct__macro_object_target_miss_rate",
            "direct__macro_image_uncertain_rate",
            "pixel_to_object__target_miss_rate",
            "pixel_to_object__false_accept_rate",
            "pixel_to_object__target_uncertain_rate",
            "pixel_to_object__non_target_uncertain_rate",
        ),
        "maximize": (
            "direct__macro_image_decided_balanced_accuracy",
            "pixel_to_object__decided_balanced_accuracy",
        ),
    },
}

if set(SIMCA_ROBUSTNESS_PARETO_OBJECTIVES) != set(
    SIMCA_ROBUSTNESS_TRACK_IDS
):
    missing = sorted(
        set(SIMCA_ROBUSTNESS_TRACK_IDS)
        - set(SIMCA_ROBUSTNESS_PARETO_OBJECTIVES)
    )
    extra = sorted(
        set(SIMCA_ROBUSTNESS_PARETO_OBJECTIVES)
        - set(SIMCA_ROBUSTNESS_TRACK_IDS)
    )
    raise RuntimeError(
        "Notebook-05 Pareto objectives must be defined exactly for E1-E8: "
        f"missing={missing}, extra={extra}."
    )
for track_id, objectives in SIMCA_ROBUSTNESS_PARETO_OBJECTIVES.items():
    minimize = tuple(objectives.get("minimize", ()))
    maximize = tuple(objectives.get("maximize", ()))
    if not minimize and not maximize:
        raise RuntimeError(
            f"{track_id} has no notebook-05 Pareto objective."
        )
    overlap = sorted(set(minimize).intersection(maximize))
    if overlap:
        raise RuntimeError(
            f"{track_id} contains metrics configured both for minimization "
            f"and maximization: {overlap}."
        )
    

# ---------------------------------------------------------------------------
# Notebook 05 v3 — stability authority and supporting robustness diagnostics
# ---------------------------------------------------------------------------
# This child-contract remains outside PROTOCOL_CONFIGURATION_KEYS. It cannot
# alter the already frozen parent 8tracks_v5 protocol used by notebooks 00-04C.
SIMCA_ROBUSTNESS_PARETO_SEED_AGGREGATION = (
    "worst_observed_seed_by_metric_direction"
)

# Only minimized official Pareto risks may block progression to batch 4.
# Maximized performance and spatial-quality metrics remain supporting warnings.
SIMCA_ROBUSTNESS_BLOCKING_STABILITY_METRICS_BY_TRACK = {
    track_id: tuple(map(str, spec.get("minimize", ())))
    for track_id, spec in SIMCA_ROBUSTNESS_PARETO_OBJECTIVES.items()
}
SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_IS_BLOCKING = True
SIMCA_ROBUSTNESS_STABILITY_REGISTRATION_STATUS = (
    "validation_selection_rule_v3_pre_batch4"
)

for _track_id, _metrics in (
    SIMCA_ROBUSTNESS_BLOCKING_STABILITY_METRICS_BY_TRACK.items()
):
    for _metric in _metrics:
        _metric_base = str(_metric).split("__")[-1]
        if _metric_base not in SIMCA_ROBUSTNESS_STABILITY_LIMITS:
            raise RuntimeError(
                f"{_track_id}: blocking stability metric {_metric!r} "
                "has no SIMCA_ROBUSTNESS_STABILITY_LIMITS entry."
            )

# Supporting analyses are fixed before the v3 execution and cannot modify the
# official Pareto, the 03C spatial lock, the frozen 03B threshold policy or the
# pure-test candidate eligibility.
SIMCA_ROBUSTNESS_SUPPORTING_DIAGNOSTIC_RULE_VERSION = (
    "notebook05_supporting_robustness_v1"
)
SIMCA_ROBUSTNESS_SUPPORTING_DIAGNOSTIC_REGISTRATION_STATUS = (
    "supporting_post_validation_defined_before_batch4"
)
SIMCA_ROBUSTNESS_UNCERTAINTY_SUMMARY_SEMANTICS = (
    "descriptive_seed_envelope_of_persisted_04c_intervals_not_cluster_ci"
)

# Local numeric-threshold sensitivity. These perturbations are symmetric and
# never used to retune the official decision policy.
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_REGISTRATION_STATUS = (
    SIMCA_ROBUSTNESS_SUPPORTING_DIAGNOSTIC_REGISTRATION_STATUS
)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DIRECT_2WAY_DELTAS = (-0.05, 0.05)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_VOTE_2WAY_DELTAS = (-0.05, 0.05)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_CENTER_SHIFT_FRACTIONS = (-0.10, 0.10)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_WIDTH_SCALES = (0.90, 1.10)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRICS = (
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_object_target_miss_rate",
    "macro_image_target_miss_rate",
    "macro_image_false_accept_rate",
    "macro_image_uncertain_rate",
    "macro_image_coverage_rate",
    "macro_image_balanced_accuracy",
    "macro_image_decided_balanced_accuracy",
)
SIMCA_ROBUSTNESS_THRESHOLD_STABILITY_WARNING_LIMITS = {
    "max_center_range_over_mean_width": 0.25,
    "max_band_width_cv": 0.25,
}

# Leave-one-source-image-out sensitivity computed from persisted 04C
# source-image metrics. Images, never pixels, are treated as independent units.
SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_METRICS = (
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_object_target_miss_rate",
)

# Calibration-fold sensitivity. Generator states identify candidate partitions;
# they are not SIMCA random states and are never persisted as scientific IDs.
SIMCA_ROBUSTNESS_RUN_FOLD_SENSITIVITY = True
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_RANDOM_STATES = tuple(range(64))
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_MAX_UNIQUE_ALTERNATIVES = 4
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRICS = (
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "macro_object_target_miss_rate",
    "macro_image_target_miss_rate",
    "macro_image_false_accept_rate",
    "macro_image_uncertain_rate",
    "macro_image_coverage_rate",
    "macro_image_balanced_accuracy",
    "macro_image_decided_balanced_accuracy",
)

# Supporting jackknife of the official front. It is restricted to the frozen
# base-seed panel and never replaces the official protocol Pareto.
SIMCA_ROBUSTNESS_RUN_PARETO_FRONT_SENSITIVITY = True

# Local sensitivity around the already frozen per-track 03C spatial candidate.
SIMCA_ROBUSTNESS_RUN_SPATIAL_SENSITIVITY = True
SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_METRICS = (
    "dice",
    "iou",
    "pixel_precision",
    "pixel_recall",
    "component_precision",
    "component_recall",
    "smallest_fragment_recall",
    "split_rate",
    "merge_rate",
)

# Exact one-factor counterfactuals. A conceptual factor may own several model
# columns; this prevents partial/ill-defined ablations of one representation.
SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS = {
    "matrix_representation": (
        "matrix_family",
        "matrix_method",
        "projection_matrix_method",
    ),
    "m": ("m",),
    "balanced_pixel_strategy": ("balanced_pixel_strategy",),
    "preprocessing": ("preprocessing", "preprocessing_steps"),
    "rule_variant": ("rule_family", "rule_variant", "limit_source"),
    "n_components": ("n_components",),
    "sg_window_length": ("sg_window_length",),
    "position_dilation_radius": ("position_dilation_radius",),
}

# Final selection is lexicographic *inside* each Pareto front. Tolerances define
# equivalence plateaus; remaining models are locked as equivalent alternatives,
# while model_id is used only as the deterministic primary tiebreak.
# SIMCA_FINAL_TRACK_PRIORITY = {
#     "E1": (
#         {"column": "direct__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_target_miss_risk"},
#         {"column": "direct__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_false_accept_risk"},
#         {"column": "direct__balanced_accuracy", "direction": "max", "tolerance": 0.02, "reason": "lower_balanced_accuracy"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E2": (
#         {"column": "direct__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_target_miss_risk"},
#         {"column": "direct__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_false_accept_risk"},
#         {"column": "direct__target_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_target_uncertainty"},
#         {"column": "direct__non_target_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_non_target_uncertainty"},
#         {"column": "direct__decided_balanced_accuracy", "direction": "max", "tolerance": 0.02, "reason": "lower_decided_balanced_accuracy"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E3": (
#         {"column": "direct__macro_image_target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_image_target_miss_risk"},
#         {"column": "pixel_to_object__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_object_target_miss_risk"},
#         {"column": "spatial__smallest_fragment_recall", "direction": "max", "tolerance": 0.05, "reason": "lower_small_fragment_recall"},
#         {"column": "direct__macro_image_false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_image_false_accept_risk"},
#         {"column": "pixel_to_object__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_object_false_accept_risk"},
#         {"column": "spatial__component_precision", "direction": "max", "tolerance": 0.05, "reason": "lower_component_precision"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E4": (
#         {"column": "direct__macro_image_target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_image_target_miss_risk"},
#         {"column": "pixel_to_object__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_object_target_miss_risk"},
#         {"column": "spatial__smallest_fragment_recall", "direction": "max", "tolerance": 0.05, "reason": "lower_small_fragment_recall"},
#         {"column": "direct__macro_image_false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_image_false_accept_risk"},
#         {"column": "pixel_to_object__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_object_false_accept_risk"},
#         {"column": "direct__macro_image_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_image_uncertainty"},
#         {"column": "pixel_to_object__target_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_target_uncertainty"},
#         {"column": "pixel_to_object__decided_balanced_accuracy", "direction": "max", "tolerance": 0.02, "reason": "lower_decided_balanced_accuracy"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E5": (
#         {"column": "direct__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_target_miss_risk"},
#         {"column": "direct__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_false_accept_risk"},
#         {"column": "direct__balanced_accuracy", "direction": "max", "tolerance": 0.02, "reason": "lower_balanced_accuracy"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E6": (
#         {"column": "direct__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_target_miss_risk"},
#         {"column": "direct__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_false_accept_risk"},
#         {"column": "direct__target_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_target_uncertainty"},
#         {"column": "direct__non_target_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_non_target_uncertainty"},
#         {"column": "direct__decided_balanced_accuracy", "direction": "max", "tolerance": 0.02, "reason": "lower_decided_balanced_accuracy"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E7": (
#         {"column": "direct__macro_image_target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_image_target_miss_risk"},
#         {"column": "pixel_to_object__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_object_target_miss_risk"},
#         {"column": "spatial__smallest_fragment_recall", "direction": "max", "tolerance": 0.05, "reason": "lower_small_fragment_recall"},
#         {"column": "direct__macro_image_false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_image_false_accept_risk"},
#         {"column": "pixel_to_object__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_object_false_accept_risk"},
#         {"column": "spatial__component_precision", "direction": "max", "tolerance": 0.05, "reason": "lower_component_precision"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
#     "E8": (
#         {"column": "direct__macro_image_target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_image_target_miss_risk"},
#         {"column": "pixel_to_object__target_miss_rate", "direction": "min", "tolerance": 0.01, "reason": "higher_object_target_miss_risk"},
#         {"column": "spatial__smallest_fragment_recall", "direction": "max", "tolerance": 0.05, "reason": "lower_small_fragment_recall"},
#         {"column": "direct__macro_image_false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_image_false_accept_risk"},
#         {"column": "pixel_to_object__false_accept_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_object_false_accept_risk"},
#         {"column": "direct__macro_image_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_image_uncertainty"},
#         {"column": "pixel_to_object__target_uncertain_rate", "direction": "min", "tolerance": 0.05, "reason": "higher_target_uncertainty"},
#         {"column": "pixel_to_object__decided_balanced_accuracy", "direction": "max", "tolerance": 0.02, "reason": "lower_decided_balanced_accuracy"},
#         #{"column": "stability_rank", "direction": "min", "tolerance": 0.0, "reason": "lower_seed_stability"},
#         {"column": "preprocessing_step_count", "direction": "min", "tolerance": 0.0, "reason": "higher_preprocessing_complexity"},
#         {"column": "n_components", "direction": "min", "tolerance": 0.0, "reason": "higher_model_complexity"},
#     ),
# }


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

BALANCED_SAMPLING_M_VALUES = (5, 10, 20, 30, 40, 50, 60, 80, 100)
BALANCED_SAMPLING_SEEDS = (42, 43, 44)
BALANCED_SAMPLING_UNDER_M_POLICY = "exclude"
BALANCED_SAMPLING_MIN_ELIGIBLE_RATE = 0.9
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

PCA_CANDIDATE_REGISTRY_COLUMNS = (
    "candidate_id",
    "selection_unit_id",
    "training_matrix_id",
    "candidate_matrix_id",
    "wavelength_axis_id",
    "matrix_family",
    "matrix_variant",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "sg_window_length",
    "sg_polyorder",
)

PCA_SUMMARY_COLUMNS = (
    "candidate_id",
    "selection_unit_id",
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
    "selection_unit_id",
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
    "selection_unit_id",
    "protocol_hash",
    "input_fingerprint",
    "review_hash",
    "matrix_family",
    "preprocessing",
    "preprocessing_steps",
    "sg_window_length",
    "sg_polyorder",
    "wavelength_axis_id",
    "selection_status",
    "selection_reason",
)
PCA_TECHNICAL_FLAG_COLUMNS = (
    "matrix_nonempty",
    "finite_values",
    "sg_valid",
    "variance_valid",
    "pca_fit_valid",
    "projection_valid",
    "residuals_valid",
    "stability_valid",
)
PCA_ARTIFACT_COLUMNS = (
    "critical_artifact",
)
PCA_REVIEW_METADATA_COLUMNS = (
    "review_decision",
    "artifact_codes",
    "reviewer",
    "review_date",
    "review_evidence",
    "run_fingerprint",
)
PCA_SELECTION_STRICT_VARIANT_COVERAGE = True

SELECTION_AUDIT_COLUMNS = (
    "stage",
    "substage",
    "entity_type",
    "entity_id",
    "related_entity_id",
    "track_id",
    "decision",
    "reason_code",
    "metric",
    "observed_value",
    "operator",
    "reference_value",
    "reference_source",
    "mechanism",
    "detail",
)

PCA_PREPROCESSING_SUMMARY_ID_COLUMNS = (
    "selection_unit_id",
    "matrix_family",
    "preprocessing",
    "preprocessing_steps",
    "sg_window_length",
    "sg_polyorder",
    "wavelength_axis_id",
)

PCA_AUDIT_STAGE = "03"

PCA_AUDIT_REQUIRED_SUBSTAGES = (
    "candidate_generation",
    "technical_check",
    "technical_fit_outcome",
    "artifact_review",
    "candidate_admissibility",
    "strict_variant_coverage",
    "pareto_metric_completeness",
    "pareto_objective",
    "pareto_selection",
)

PCA_AUDIT_OPTIONAL_SUBSTAGES = (
    "pareto_dominance",
)

PCA_AUDIT_ALLOWED_SUBSTAGES = (
    *PCA_AUDIT_REQUIRED_SUBSTAGES,
    *PCA_AUDIT_OPTIONAL_SUBSTAGES,
)

PCA_AUDIT_ALLOWED_DECISIONS = {
    "candidate_generation": ("entered",),
    "technical_check": ("kept", "eliminated"),
    "technical_fit_outcome": ("kept", "eliminated"),
    "artifact_review": (
        "kept",
        "warning",
        "eliminated",
        "not_applicable",
    ),
    "candidate_admissibility": ("kept", "eliminated"),
    "strict_variant_coverage": ("kept", "eliminated"),
    "pareto_metric_completeness": ("kept", "eliminated"),
    "pareto_objective": ("evaluated", "not_evaluated"),
    "pareto_dominance": ("eliminated",),
    "pareto_selection": ("kept", "eliminated"),
}

PCA_AUDIT_CANDIDATE_SUBSTAGES = (
    "candidate_generation",
    "technical_check",
    "technical_fit_outcome",
    "artifact_review",
    "candidate_admissibility",
)

PCA_AUDIT_PREPROCESSING_SUBSTAGES = (
    "strict_variant_coverage",
    "pareto_metric_completeness",
    "pareto_objective",
    "pareto_dominance",
    "pareto_selection",
)

PCA_TECHNICAL_AUDIT_RULES = {
    "matrix_nonempty": {
        "metric": "n_observations",
        "operator": ">",
        "reference_value": 0.0,
        "reference_source": "PCA technical contract",
        "mechanism": "hard_constraint",
        "reason_code_prefix": "matrix_nonempty",
    },
    "variance_valid": {
        "metric": "zero_variance_band_rate",
        "operator": "<=",
        "reference_config": (
            "PREPROCESSING_MAX_ZERO_VARIANCE_BAND_RATE"
        ),
        "reference_source": (
            "PREPROCESSING_MAX_ZERO_VARIANCE_BAND_RATE"
        ),
        "mechanism": "hard_threshold",
        "reason_code_prefix": "zero_variance_band_rate",
    },
}

PCA_TECHNICAL_AUDIT_FALLBACK_RULE = {
    "operator": "==",
    "reference_value": 1.0,
    "reference_source": "PCA technical contract",
    "mechanism": "hard_constraint",
}

PCA_PARETO_ATOL = 1e-12

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

SIMCA_MODEL_PARAMETER_COLUMNS = (
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
    "rule_family",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
)

# La graine n'appartient pas à l'identité scientifique du modèle.
SIMCA_MODEL_ID_COLUMNS = SIMCA_MODEL_PARAMETER_COLUMNS

# Un fit est une exécution technique d'un modèle PCA-SIMCA.
SIMCA_FIT_ID_COLUMNS = (
    "matrix_family",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "random_state",
)

# Une projection partageable ne dépend ni du track ni du mode 2-way/3-way.
SIMCA_PROJECTION_ID_COLUMNS = (
    "fit_id",
    "projection_level",
    "projection_matrix_method",
    "rule_variant",
    "limit_source",
)

# Table large utilisée uniquement en mémoire par le runner.
INTERNAL_CALIBRATION_EXECUTION_COLUMNS = (
    "model_id",
    "fit_id",
    "projection_id",
    "random_state",
    *SIMCA_MODEL_PARAMETER_COLUMNS,
)

# Tables persistées compactes.
INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS = (
    "model_id",
    *SIMCA_MODEL_PARAMETER_COLUMNS,
)

INTERNAL_CALIBRATION_CANDIDATE_RUN_COLUMNS = (
    "model_id",
    "random_state",
    "fit_id",
    "projection_id",
)

INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_COLUMNS = (
    "fit_id",
    "fold_id",
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

INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS = (
    "projection_id",
    "fold_id",
    "rule_limit",
    "q_limit",
    "t2_limit",
    "train_rejection_rate",
    "oof_target_rejection_rate",
    "status",
    "error_code",
)

INTERNAL_CALIBRATION_OOF_BASE_COLUMNS = (
    "projection_id",
    "fold_id",
    "source_image",
    "object_id",
    "batch",
    "object_area",
    "size_bin",
    "truth",
    "pca_score_pc1",
    "pca_score_pc2",
    "H",
    "Q",
    "rule_statistic",
    "rule_limit",
    "normalized_ratio",
    "simca_margin",
)

INTERNAL_CALIBRATION_OOF_OBJECT_COLUMNS = (
    *INTERNAL_CALIBRATION_OOF_BASE_COLUMNS,
)

INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS = (
    *INTERNAL_CALIBRATION_OOF_BASE_COLUMNS,
    "row",
    "col",
)

INTERNAL_CALIBRATION_PROJECTION_SHIFT_COLUMNS = (
    "projection_id",
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

# Format long : une métrique par ligne.
INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS = (
    "model_id",
    "random_state",
    "evaluation_fold",
    "decision_scope",
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
    "lower_threshold",
    "upper_threshold",
    "metric",
    "value",
)

INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS = (
    "model_id",
    "metric",
    "value",
)

INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS = (
    "model_id",
    "selection_status",
)

INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS = (
    "model_id",
    "random_state",
    "fit_id",
    "projection_id",
)

INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS = (
    "model_id",
    "random_state",
    "decision_scope",
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
    "lower_threshold",
    "upper_threshold",
)

INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS = (
    "selection_level",
    "model_id",
    "decision_scope",
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
    "stage",
    "decision",
    "reason_code",
    "metric",
    "observed_value",
    "operator",
    "reference_value",
    "related_model_id",
)

INTERNAL_CALIBRATION_TECHNICAL_EVENT_COLUMNS = (
    "fit_id",
    "projection_id",
    "fold_id",
    "stage",
    "status",
    "reason_code",
    "n_initial",
    "n_valid",
    "n_filtered",
    "error_type",
    "error_message",
)


# ---------------------------------------------------------------------------
# Notebook-05 compact natural keys
# ---------------------------------------------------------------------------
# One scientific execution. track_id is functionally determined by model_id
# through model_catalog and therefore is not duplicated in the compact seed
# registry persisted to disk.
SIMCA_ROBUSTNESS_SEED_EXECUTION_KEY_COLUMNS = (
    "model_id",
    "random_state",
)
SIMCA_ROBUSTNESS_SEED_EXECUTION_COLUMNS = (
    "model_id",
    "random_state",
    "fit_id",
    "projection_id",
)
# One numeric decision threshold for one execution and decision scope.
SIMCA_ROBUSTNESS_SEED_THRESHOLD_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "decision_scope",
)
SIMCA_ROBUSTNESS_SEED_THRESHOLD_COLUMNS = (
    INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
)
SIMCA_ROBUSTNESS_EXECUTION_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
)
SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
)
SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS = (
    "model_id",
    "track_id",
)


# ---------------------------------------------------------------------------
# Notebook-05 persisted table schemas
# ---------------------------------------------------------------------------
# These schemas belong only to the notebook-05 child contract.
# They are deliberately outside PROTOCOL_CONFIGURATION_KEYS.
# ---------------------------------------------------------------------------
SIMCA_ROBUSTNESS_MODEL_INVARIANT_COLUMNS = (
    "model_id",
    "track_id",
    "evaluation_track",
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
    "rule_family",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "eligibility_status",
    "downstream_status",
)
SIMCA_ROBUSTNESS_MEMBER_METRIC_COLUMNS = tuple(
    SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES
)
SIMCA_ROBUSTNESS_MEMBER_WORST_IMAGE_COLUMNS = tuple(
    f"worst_image__{metric}"
    for metric in SIMCA_ROBUSTNESS_WORST_IMAGE_METRIC_NAMES
)
SIMCA_ROBUSTNESS_MEMBER_SPATIAL_COLUMNS = tuple(
    f"spatial__{metric}"
    for metric in SIMCA_ROBUSTNESS_SPATIAL_METRICS
)
SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS = (
    "model_id",
    "random_state",
    "fit_id",
    "projection_id",
    "track_id",
    "decision_scope",
    "evaluation_track",
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
    "rule_family",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "eligibility_status",
    "downstream_status",
    "candidate_status",
    "scope_calculable",
    "scope_protocol_pass",
    "all_blocking_checks_pass",
    "n_guardrail_checks",
    "n_blocking_checks",
    "n_blocking_failures",
    "n_technical_errors",
    "is_stochastic",
    "preprocessing_step_count",
    *SIMCA_ROBUSTNESS_MEMBER_METRIC_COLUMNS,
    *SIMCA_ROBUSTNESS_MEMBER_WORST_IMAGE_COLUMNS,
    *SIMCA_ROBUSTNESS_MEMBER_SPATIAL_COLUMNS,
)
SIMCA_ROBUSTNESS_SCOPED_VALUE_COLUMNS = tuple(
    f"{scope}__{metric}"
    for scope in SIMCA_ROBUSTNESS_DECISION_SCOPES
    for metric in SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES
)
SIMCA_ROBUSTNESS_SCOPED_WORST_IMAGE_COLUMNS = tuple(
    f"{scope}__worst_image__{metric}"
    for scope in SIMCA_ROBUSTNESS_DECISION_SCOPES
    for metric in SIMCA_ROBUSTNESS_WORST_IMAGE_METRIC_NAMES
)
SIMCA_ROBUSTNESS_SPATIAL_VALUE_COLUMNS = tuple(
    f"spatial__{metric}"
    for metric in SIMCA_ROBUSTNESS_SPATIAL_METRICS
)
SIMCA_ROBUSTNESS_MODEL_VALUE_COLUMNS = (
    *SIMCA_ROBUSTNESS_SCOPED_VALUE_COLUMNS,
    *SIMCA_ROBUSTNESS_SCOPED_WORST_IMAGE_COLUMNS,
    *SIMCA_ROBUSTNESS_SPATIAL_VALUE_COLUMNS,
)
SIMCA_ROBUSTNESS_MODEL_STATISTIC_COLUMNS = tuple(
    f"{statistic}__{metric}"
    for statistic in (
        "mean",
        "std",
        "min",
        "max",
    )
    for metric in SIMCA_ROBUSTNESS_MODEL_VALUE_COLUMNS
)
SIMCA_ROBUSTNESS_SELECTION_UNIT_COLUMNS = (
    *SIMCA_ROBUSTNESS_MODEL_INVARIANT_COLUMNS,
    "is_stochastic",
    "preprocessing_step_count",
    "n_random_states",
    "n_expected_random_states",
    "observed_random_states_json",
    "missing_random_states_json",
    "all_expected_random_states_present",
    "seed_requirement_satisfied",
    "all_execution_calculable",
    "all_execution_protocol_supported",
    "all_04c_blocking_guardrails_pass",
    "model_diagnostic_eligible",
    "model_protocol_eligible_pre_stability",
    *SIMCA_ROBUSTNESS_MODEL_VALUE_COLUMNS,
    *SIMCA_ROBUSTNESS_MODEL_STATISTIC_COLUMNS,
)
SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS = (
    *SIMCA_ROBUSTNESS_SELECTION_UNIT_COLUMNS,
    "diagnostic_pareto_eligible",
    "is_diagnostic_pareto",
    "diagnostic_dominated_by_model_id",
    "protocol_pareto_eligible",
    "is_protocol_pareto",
    "protocol_dominated_by_model_id",
    "pareto_status",
    "pareto_exclusion_reason",
)
SIMCA_ROBUSTNESS_PARETO_AUDIT_COLUMNS = (
    "track_id",
    "model_id",
    "pool_type",
    "is_candidate",
    "is_pareto",
    "dominated_by_model_id",
    "reason_code",
    "pareto_minimize_json",
    "pareto_maximize_json",
    "objective_values_json",
)
SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS = (
    "model_id",
    "track_id",
    "decision_scope",
    "n_random_states",
    "n_entities",
    "n_target_entities",
    "entity_seed_coverage_complete",
    "decision_disagreement_rate",
    "target_decision_disagreement_rate",
    "disagreement_status",
)
SIMCA_ROBUSTNESS_STABILITY_SUMMARY_COLUMNS = (
    "model_id",
    "track_id",
    "metric",
    "metric_base",
    "metric_direction",
    "stability_role",
    "is_stochastic",
    "n_random_states",
    "n_expected_random_states",
    "observed_random_states_json",
    "missing_random_states_json",
    "all_expected_random_states_present",
    "n_finite_values",
    "mean",
    "std",
    "min",
    "max",
    "range",
    "worst_value",
    "worst_random_state",
    "max_std_limit",
    "max_range_limit",
    "std_limit_exceeded",
    "range_limit_exceeded",
    "blocking_metric_failure",
    "supporting_metric_warning",
    "stability_metric_status",
    "decision_disagreement_rate",
    "target_decision_disagreement_rate",
    "blocking_stability_failed",
    "supporting_stability_warning",
    "blocking_stability_flags",
    "supporting_stability_flags",
    "stability_flags",
    "stability_flag_count",
    "model_stability_status",
)
# IMPORTANT:
# observed_value remains numeric.
# Categorical values are kept in observed_status.
SIMCA_ROBUSTNESS_REVIEW_GUARDRAIL_COLUMNS = (
    "model_id",
    "track_id",
    "check_scope",
    "check_name",
    "observed_value",
    "observed_status",
    "comparator",
    "threshold_value",
    "threshold_statuses",
    "check_status",
    "is_blocking",
    "reason_code",
    "reason",
)
SIMCA_ROBUSTNESS_TRACK_REVIEW_COLUMNS = (
    "model_id",
    "track_id",
    "review_status",
    "hard_exclusion",
    "robustness_flags",
    "stability_flags",
    "review_flags",
    "review_flag_count",
    "selection_influence",
)
SIMCA_ROBUSTNESS_PURE_TEST_CANDIDATE_COLUMNS = (
    *SIMCA_ROBUSTNESS_MODEL_INVARIANT_COLUMNS,
    "is_stochastic",
    "is_protocol_pareto",
    "model_stability_status",
    "review_status",
    "review_flags",
    "candidate_role",
)
SIMCA_ROBUSTNESS_STATISTICAL_UNCERTAINTY_COLUMNS = (
    "model_id",
    "track_id",
    "decision_scope",
    "metric",
    "estimate",
    "interval_envelope_low",
    "interval_envelope_high",
    "n_random_states",
    "n_independent_images",
    "interval_status",
    "selection_influence",
)
SIMCA_ROBUSTNESS_RISK_COVERAGE_COLUMNS = (
    "model_id",
    "track_id",
    "requested_coverage",
    "attained_coverage",
    "target_miss_rate",
    "false_accept_rate",
    "mean_n_decided",
    "n_random_states",
    "selective_risk_auc",
    "coverage_at_target_miss_guardrail",
    "curve_role",
)
SIMCA_ROBUSTNESS_ABLATION_PLAN_COLUMNS = (
    "track_id",
    "reference_model_id",
    "ablated_model_id",
    "factor",
    "selection_influence",
)
SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS = (
    "track_id",
    "reference_model_id",
    "ablated_model_id",
    "factor",
    "metric",
    "n_paired_random_states",
    "reference_value",
    "ablated_value",
    "effect",
    "effect_std",
    "effect_min",
    "effect_max",
    "practical_tolerance",
    "effect_status",
    "directional_status",
    "selection_influence",
)

SIMCA_ROBUSTNESS_ABLATION_COVERAGE_COLUMNS = (
    "track_id",
    "factor",
    "n_reference_pareto_models",
    "n_exact_counterfactual_pairs",
    "n_reference_models_with_counterfactual",
    "reference_coverage_rate",
    "coverage_status",
    "selection_influence",
)

SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_PLAN_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "perturbation_type",
    "perturbation_value",
    "alternative_lower_threshold",
    "alternative_upper_threshold",
    "plan_status",
    "selection_influence",
)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRIC_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "perturbation_type",
    "perturbation_value",
    "metric",
    "reference_value",
    "alternative_value",
    "delta",
    "practical_tolerance",
    "effect_status",
    "selection_influence",
)
SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DECISION_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "perturbation_type",
    "perturbation_value",
    "n_entities",
    "n_target_entities",
    "decision_flip_rate",
    "target_decision_flip_rate",
    "selection_influence",
)
SIMCA_ROBUSTNESS_THRESHOLD_STABILITY_COLUMNS = (
    "model_id",
    "track_id",
    "decision_scope",
    "decision_mode",
    "n_random_states",
    "policy_coordinates_invariant",
    "fixed_numeric_threshold_invariant",
    "lower_threshold_mean",
    "lower_threshold_std",
    "lower_threshold_range",
    "upper_threshold_mean",
    "upper_threshold_std",
    "upper_threshold_range",
    "threshold_center_mean",
    "threshold_center_std",
    "threshold_center_range",
    "uncertainty_band_width_mean",
    "uncertainty_band_width_std",
    "uncertainty_band_width_range",
    "band_width_cv",
    "center_range_over_mean_width",
    "numeric_stability_warning",
    "stability_status",
    "selection_influence",
)

SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "metric",
    "omitted_source_image",
    "n_source_images",
    "n_finite_source_images_full",
    "n_finite_source_images_retained",
    "full_macro_image_value",
    "leave_one_image_out_value",
    "delta",
    "absolute_delta",
    "influence_status",
    "selection_influence",
)

SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_PLAN_COLUMNS = (
    "generator_random_state",
    "reference_partition_sha256",
    "alternative_partition_sha256",
    "n_source_images",
    "n_folds",
    "coverage_complete",
    "plan_status",
    "selection_influence",
)
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_THRESHOLD_COLUMNS = (
    "alternative_partition_sha256",
    *INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
)
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRIC_COLUMNS = (
    "alternative_partition_sha256",
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "metric",
    "reference_value",
    "alternative_value",
    "delta",
    "practical_tolerance",
    "effect_status",
    "selection_influence",
)
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_DECISION_COLUMNS = (
    "alternative_partition_sha256",
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "n_entities",
    "n_target_entities",
    "decision_flip_rate",
    "target_decision_flip_rate",
    "selection_influence",
)
SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_TECHNICAL_EVENT_COLUMNS = (
    "alternative_partition_sha256",
    *INTERNAL_CALIBRATION_TECHNICAL_EVENT_COLUMNS,
)

SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_REPLICATE_COLUMNS = (
    "model_id",
    "track_id",
    "reference_is_protocol_pareto",
    "omitted_random_state",
    "n_random_states_used",
    "replicate_is_pareto",
    "dominated_by_model_id",
    "selection_influence",
)
SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_SUMMARY_COLUMNS = (
    "model_id",
    "track_id",
    "reference_is_protocol_pareto",
    "n_replicates",
    "n_pareto_replicates",
    "pareto_membership_frequency",
    "front_stability_status",
    "selection_influence",
)
SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_AUDIT_COLUMNS = (
    "track_id",
    "omitted_random_state",
    "n_candidate_models",
    "n_reference_pareto",
    "n_replicate_pareto",
    "pareto_jaccard_vs_reference",
    "selection_influence",
)

SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_PLAN_COLUMNS = (
    "track_id",
    "factor",
    "alternative_spatial_candidate_id",
    "connectivity",
    "morphology_operation",
    "morphology_radius",
    "min_area_pixels",
    "selection_influence",
)
SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "factor",
    "alternative_spatial_candidate_id",
    "metric",
    "reference_value",
    "alternative_value",
    "delta",
    "practical_tolerance",
    "effect_status",
    "directional_status",
    "selection_influence",
)

SIMCA_ROBUSTNESS_REVIEW_ELIGIBLE_STATUSES = (
    "eligible_for_pure_test",
    "eligible_with_warning",
)
SIMCA_ROBUSTNESS_REVIEW_HARD_EXCLUSION_STATUSES = (
    "excluded_technical",
    "excluded_04c_guardrail",
    "excluded_unstable",
    "excluded_missing_seed",
)

SIMCA_ROBUSTNESS_ABLATION_REGISTRATION_STATUS = (
    "supporting_post_validation_defined"
)
SIMCA_ROBUSTNESS_ABLATION_FACTORS = tuple(
    SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS
)

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
if tuple(INTERNAL_CALIBRATION_RANDOM_SEEDS) != tuple(
    SIMCA_ROBUSTNESS_BASE_RANDOM_STATES
):
    raise RuntimeError(
        "SIMCA_ROBUSTNESS_BASE_RANDOM_STATES must match the 03B execution "
        "seeds used by the frozen 04C validation."
    )
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
INTERNAL_CALIBRATION_THRESHOLD_BLOCK_SIZE = 16

# "safe_reject": an uncertain target is rejected or sent for confirmation.
# "counts_as_miss": an uncertain target may operationally become negative.
INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY = "safe_reject"

if INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY not in {
    "safe_reject",
    "counts_as_miss",
}:
    raise ValueError(
        "INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY must be "
        "'safe_reject' or 'counts_as_miss'."
    )

# No track is allowed to disappear silently.
INTERNAL_CALIBRATION_ALLOWED_UNSUPPORTED_TRACK_IDS = ()

# Selection-only amendment.  It does not change model, fit or projection
# identities and therefore does not invalidate the expensive OOF checkpoints.
INTERNAL_CALIBRATION_SELECTION_PARENT_PROFILE_ID = (
    INTERNAL_CALIBRATION_CONSTRAINT_PROFILE_ID
)
# Immutable execution lineage of the PCA shortlist and completed 03B fits.
# This value is intentionally excluded from PROTOCOL_CONFIGURATION_KEYS: it
# links a selection-only amendment to its parent execution instead of changing
# the scientific inputs or identities of that execution.
INTERNAL_CALIBRATION_SELECTION_PARENT_PROTOCOL_HASH = (
    "5d66e659d7da4e69fa647123058bcea08d33fe0155b6c59d71001820dbc78f9e"
)
INTERNAL_CALIBRATION_SELECTION_AMENDMENT_SCOPE = "selection_only"
INTERNAL_CALIBRATION_SELECTION_PROFILE_ID = (
    "fn-priority_v1_scope-aware-amendment-1"
)
INTERNAL_CALIBRATION_SELECTION_AMENDMENT_REASON = (
    "E3 uses scope-specific attainable FN limits. Balanced accuracy is only "
    "enforced where both classes are identifiable at the primary evaluation "
    "unit; direct pixel-projection scopes use pure source images and therefore "
    "have undefined image-level balanced accuracy."
)

# Accuracy is identifiable either for direct object projections or after a
# pixel projection has been aggregated back to the object level. Contexts are
# OR-ed; fields inside one context are AND-ed by the vectorized selector.
INTERNAL_CALIBRATION_ACCURACY_CONTEXTS = (
    {
        "projection_level": "object_projection",
        "decision_scope": "direct",
    },
    {
        "projection_level": "pixel_projection",
        "decision_scope": "pixel_to_object",
    },
)

# Only E3 receives numerical feasibility overrides. All unlisted metrics,
# scopes and tracks retain the generic constraints below.
INTERNAL_CALIBRATION_THRESHOLD_OVERRIDES = {
    "E3": {
        "direct": {
            "target_miss_rate": 0.18,
            "worst_target_miss_rate": 0.25,
            "worst_unit_target_miss_rate": 0.25,
        },
        "pixel_to_object": {
            "target_miss_rate": 0.25,
            "worst_target_miss_rate": 0.35,
            "worst_unit_target_miss_rate": 0.35,
        },
    },
}

# Feasibility is explicit: no fallback to an unconstrained "best compromise".
# The exploratory profile is deliberately permissive so 03B first documents
# the feasible domain. Switching one name activates the stricter final profile.
INTERNAL_CALIBRATION_THRESHOLD_CONSTRAINTS = {
    "2way": (
        {
            "metric": "target_miss_rate",
            "operator": "<=",
            "value": 0.05,
            "reason": "mean_fn_above_limit",
        },
        {
            "metric": "worst_target_miss_rate",
            "operator": "<=",
            "value": 0.10,
            "reason": "worst_fn_above_limit",
        },
        {
            "metric": "worst_unit_target_miss_rate",
            "operator": "<=",
            "value": 0.10,
            "reason": "worst_unit_fn_above_limit",
        },
        {
            "metric": "false_accept_rate",
            "operator": "<=",
            "value": 0.30,
            "reason": "mean_fp_above_limit",
        },
        {
            "metric": "worst_false_accept_rate",
            "operator": "<=",
            "value": 0.50,
            "reason": "worst_fp_above_limit",
        },
        {
            "metric": "balanced_accuracy",
            "operator": ">=",
            "value": 0.75,
            "reason": "balanced_accuracy_below_limit",
            "applies_when": INTERNAL_CALIBRATION_ACCURACY_CONTEXTS,
        },
    ),
    "3way": (
        {
            "metric": "target_miss_rate",
            "operator": "<=",
            "value": 0.05,
            "reason": "mean_fn_above_limit",
        },
        {
            "metric": "worst_target_miss_rate",
            "operator": "<=",
            "value": 0.10,
            "reason": "worst_fn_above_limit",
        },
        {
            "metric": "worst_unit_target_miss_rate",
            "operator": "<=",
            "value": 0.10,
            "reason": "worst_unit_fn_above_limit",
        },
        {
            "metric": "target_uncertain_rate",
            "operator": "<=",
            "value": 0.15,
            "reason": "target_uncertainty_above_limit",
        },
        {
            "metric": "false_accept_rate",
            "operator": "<=",
            "value": 0.30,
            "reason": "mean_fp_above_limit",
        },
        {
            "metric": "worst_false_accept_rate",
            "operator": "<=",
            "value": 0.50,
            "reason": "worst_fp_above_limit",
        },
        {
            "metric": "uncertain_rate",
            "operator": "<=",
            "value": 0.30,
            "reason": "uncertainty_above_limit",
        },
        {
            "metric": "coverage_rate",
            "operator": ">=",
            "value": 0.70,
            "reason": "coverage_below_limit",
        },
        {
            "metric": "decided_balanced_accuracy",
            "operator": ">=",
            "value": 0.75,
            "reason": "decided_balanced_accuracy_below_limit",
            "applies_when": INTERNAL_CALIBRATION_ACCURACY_CONTEXTS,
        },
    ),
}
INTERNAL_CALIBRATION_FN_PLATEAU_TOLERANCE = 0.01
INTERNAL_CALIBRATION_FP_PLATEAU_TOLERANCE = 0.02
INTERNAL_CALIBRATION_THRESHOLD_PRIORITY = (
    {
        "metric": "target_miss_rate",
        "direction": "min",
        "tolerance": INTERNAL_CALIBRATION_FN_PLATEAU_TOLERANCE,
        "reason": "outside_fn_plateau",
    },
    {
        "metric": "false_accept_rate",
        "direction": "min",
        "tolerance": INTERNAL_CALIBRATION_FP_PLATEAU_TOLERANCE,
        "reason": "outside_fp_plateau",
    },
    {
        "metric": "target_uncertain_rate",
        "direction": "min",
        "tolerance": 0.02,
        "applies_to": ("3way",),
        "reason": "outside_target_uncertainty_plateau",
    },
    {
        "metric": "uncertain_rate",
        "direction": "min",
        "tolerance": 0.02,
        "applies_to": ("3way",),
        "reason": "outside_uncertainty_plateau",
    },
    {
        "metric": "decided_balanced_accuracy",
        "direction": "max",
        "tolerance": 0.01,
        "applies_to": ("3way",),
        "applies_when": INTERNAL_CALIBRATION_ACCURACY_CONTEXTS,
        "reason": "outside_accuracy_plateau",
    },
)
INTERNAL_CALIBRATION_THRESHOLD_TIEBREAK = (
    {
        "column": "vote_threshold",
        "direction": "min",
    },
    {
        "column": "lower_quantile",
        "direction": "max",
    },
    {
        "column": "upper_quantile",
        "direction": "min",
    },
)
INTERNAL_CALIBRATION_MODEL_PRIORITY = (
    {
        "metric": "safety.target_miss_rate",
        "direction": "min",
        "tolerance": INTERNAL_CALIBRATION_FN_PLATEAU_TOLERANCE,
        "reason": "outside_model_fn_plateau",
    },
    {
        "metric": "safety.false_accept_rate",
        "direction": "min",
        "tolerance": INTERNAL_CALIBRATION_FP_PLATEAU_TOLERANCE,
        "reason": "outside_model_fp_plateau",
    },
)
INTERNAL_CALIBRATION_COMPLEXITY_SELECTION = {
    "parameter_order": ("n_components", "m"),
}
INTERNAL_CALIBRATION_PARETO_OBJECTIVES = {
    "E1": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
        ),
        "maximize": (),
    },
    "E2": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
            "direct.target_uncertain_rate",
            "direct.non_target_uncertain_rate",
        ),
        "maximize": (
            "direct.decided_balanced_accuracy",
        ),
    },
    "E3": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
            "pixel_to_object.target_miss_rate",
            "pixel_to_object.false_accept_rate",
        ),
        "maximize": (),
    },
    "E4": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
            "direct.target_uncertain_rate",
            "direct.non_target_uncertain_rate",
            "pixel_to_object.target_miss_rate",
            "pixel_to_object.false_accept_rate",
            "pixel_to_object.target_uncertain_rate",
            "pixel_to_object.non_target_uncertain_rate",
        ),
        "maximize": (
            "pixel_to_object.decided_balanced_accuracy",
        ),
    },
    "E5": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
        ),
        "maximize": (),
    },
    "E6": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
            "direct.target_uncertain_rate",
            "direct.non_target_uncertain_rate",
        ),
        "maximize": (
            "direct.decided_balanced_accuracy",
        ),
    },
    "E7": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
            "pixel_to_object.target_miss_rate",
            "pixel_to_object.false_accept_rate",
        ),
        "maximize": (),
    },
    "E8": {
        "minimize": (
            "direct.target_miss_rate",
            "direct.false_accept_rate",
            "direct.target_uncertain_rate",
            "direct.non_target_uncertain_rate",
            "pixel_to_object.target_miss_rate",
            "pixel_to_object.false_accept_rate",
            "pixel_to_object.target_uncertain_rate",
            "pixel_to_object.non_target_uncertain_rate",
        ),
        "maximize": (
            "pixel_to_object.decided_balanced_accuracy",
        ),
    },
}

INTERNAL_CALIBRATION_RUN = True
INTERNAL_CALIBRATION_MAX_CONFIGS = None
INTERNAL_CALIBRATION_VERBOSE = True
INTERNAL_CALIBRATION_KEEP_FINAL_OOF_PIXELS = False
INTERNAL_CALIBRATION_MAX_ROWS_TO_DISPLAY = 30
INTERNAL_CALIBRATION_MAX_PLOT_CANDIDATES_PER_TRACK_SCOPE = 5_000
INTERNAL_CALIBRATION_CHECKPOINT_ENABLED = True
INTERNAL_CALIBRATION_CHECKPOINT_DIRNAME = "_ckpt"
INTERNAL_CALIBRATION_CHECKPOINT_EVERY_N_DATA_CONFIGS = 5
INTERNAL_CALIBRATION_RESUME_FROM_CHECKPOINT = True
INTERNAL_CALIBRATION_THRESHOLD_CANDIDATE_CACHE_FILENAME = (
    "_threshold_candidates.parquet"
)
INTERNAL_CALIBRATION_REUSE_THRESHOLD_CANDIDATE_CACHE = True
INTERNAL_CALIBRATION_REBUILD_THRESHOLD_CANDIDATE_CACHE = False


# ---------------------------------------------------------------------------
# Notebook 03C: train-to-projection domain audit and spatial calibration
# ---------------------------------------------------------------------------
# These rules are part of the frozen protocol. They must not be modified after
# any batch-3 result has been inspected.
DOMAIN_SPATIAL_REQUIRED_03B_ARTIFACTS = (
    "track_contracts",
    "model_catalog",
    "selected_models",
    "selected_runs",
    "selected_thresholds",
    "oof_object_predictions",
    "oof_pixel_predictions",
    "projection_shift",
)
DOMAIN_SPATIAL_SELECTED_EXECUTION_COLUMNS = (
    "model_id",
    "random_state",
    "projection_id",
    "track_id",
    "decision_mode",
    "projection_level",
)

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
PROJECTION_DOMAIN_ELIGIBILITY_STATUSES = (
    "eligible",
    "eligible_with_warning",
    "unsupported_domain_shift",
    "unsupported_internal_calibration",
)
PROJECTION_DOMAIN_SPATIAL_SUPPORTED_STATUSES = (
    "eligible",
    "eligible_with_warning",
)
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
SPATIAL_CALIBRATION_PIXEL_TRACK_IDS = tuple(
    SIMCA_EVALUATION_TRACK_IDS[evaluation_track]
    for evaluation_track, spec in SIMCA_EVALUATION_TRACK_SPECS.items()
    if str(spec["projection_level"]) == "pixel_projection"
)
SPATIAL_CALIBRATION_SELECTION_SCOPE = "within_track"
SPATIAL_CALIBRATION_SELECTION_POLICY = (
    "within_track_lexicographic_plateau_then_minimum_complexity"
)
SPATIAL_CALIBRATION_WITHIN_TRACK_AGGREGATION = "equal_selected_execution"
SPATIAL_CALIBRATION_SELECTION_TOLERANCE = 0.005
SPATIAL_CALIBRATION_SELECTION_MAXIMIZE = (
    "smallest_fragment_recall",
    "component_recall",
    "pixel_recall",
    "dice",
    "iou",
    "component_precision",
)
SPATIAL_CALIBRATION_SELECTION_MINIMIZE = (
    "split_rate",
    "merge_rate",
)
SPATIAL_CALIBRATION_PARAMETER_COLUMNS = (
    "connectivity",
    "morphology_operation",
    "morphology_radius",
    "min_area_pixels",
)
SPATIAL_CALIBRATION_REQUIRED_LOCK_PARAMETER_KEYS = (
    "spatial_candidate_id",
    *SPATIAL_CALIBRATION_PARAMETER_COLUMNS,
)
SPATIAL_CALIBRATION_OPERATION_COMPLEXITY = {
    "none": 0,
    "opening": 1,
    "closing": 1,
    "opening_closing": 2,
}
SPATIAL_CALIBRATION_INPUT_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "source_image",
    "batch",
    "row",
    "col",
    "raw_target",
    "raw_uncertain",
    "true_target",
    "truth_level",
)

INTERNAL_CALIBRATION_FOLD_COLUMNS = (
    "source_image",
    "object_id",
    "class_name",
    "batch",
    "object_area",
    "size_bin",
    "fold_id",
)

SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_ASSIGNMENT_COLUMNS = (
    "alternative_partition_sha256",
    *INTERNAL_CALIBRATION_FOLD_COLUMNS,
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

PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
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
    "track_id",
    "n_selected_models",
    "n_selected_runs",
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
    "model_id",
    "random_state",
    "track_id",
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
    "model_id",
    "random_state",
    "track_id",
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

# ---------------------------------------------------------------------------
# Notebook 04A: reference audit of the models selected in 03B
# ---------------------------------------------------------------------------
SIMCA_GRID_REQUIRED_03B_ARTIFACTS = (
    "track_contracts",
    "model_catalog",
    "selected_models",
    "selected_runs",
    "selected_thresholds",
    "threshold_metrics",
    "model_metrics",
    "selection_audit",
)
SIMCA_GRID_REQUIRED_03C_ARTIFACTS = (
    "projection_eligibility",
    "spatial_postprocessing_lock",
    "audit_manifest",
)
SIMCA_GRID_THRESHOLD_METRIC_BATCH_SIZE = 250_000
# Upstream selected model metrics are persisted as float32. This tolerance is
# only a serialization-consistency guard and is not a scientific decision
# margin.
SIMCA_GRID_REFERENCE_METRIC_ATOL = 1e-7

# Legacy 04B/04C search settings are retained until those notebooks are
# migrated. Notebook 04A no longer creates a second model-search domain.
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
SIMCA_OPTUNA_BENCHMARK_RULE_VERSION = "categorical_tpe_coverage_v1"
SIMCA_OPTUNA_BENCHMARK_ROLE = "diagnostic_negative_control"
SIMCA_OPTUNA_BENCHMARK_PARAMETER = "model_id"
SIMCA_OPTUNA_REQUIRED_03B_ARTIFACTS = (
    "track_contracts",
    "model_catalog",
    "model_metrics",
    "selected_models",
    "selection_audit",
)
SIMCA_OPTUNA_REQUIRED_04A_ARTIFACTS = (
    "model_reference",
    "audit_manifest",
)
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
# One categorical parameter has no multivariate structure. This benchmark is
# a negative control of identifier coverage, not a hyperparameter optimizer.
SIMCA_OPTUNA_SAMPLER_MULTIVARIATE = False
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
SIMCA_OPTUNA_REUSE_INTERNAL_METRICS = True

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
#SIMCA_CONCAT_REFIT_MAX_CANDIDATES = None
SIMCA_CONCAT_REFIT_RECONSTRUCT_HEAVY_OBJECT_ARRAYS = False
SIMCA_CONCAT_REFIT_CHECKPOINT_ENABLED = True
SIMCA_CONCAT_REFIT_RESUME_FROM_CHECKPOINT = True
SIMCA_CONCAT_REFIT_CHECKPOINT_DIRNAME = "_checkpoints"
# SIMCA_CONCAT_REFIT_SIGNATURE_ROUND_DECIMALS = 12
# SIMCA_CONCAT_REFIT_CANDIDATE_POLICY = (
#     "supported_protocol_pareto_plus_unsupported_diagnostic_pareto"
# )
SIMCA_CONCAT_REFIT_EXECUTION_POLICY = (
    "selected_03B_executions_no_04C_reselection"
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
# ---------------------------------------------------------------------------
# 04C validation guardrails
# ---------------------------------------------------------------------------

SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU = 0.0

# Keep the spatial-fragment criterion diagnostic in batch 3.
# The pure-reference spatial truth used here is not an independently
# annotated mixture-fragment ground truth.
SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN = None

SIMCA_CONCAT_REFIT_GUARDRAIL_SCOPES = (
    "overall",
    "worst_image",
)

SIMCA_CONCAT_REFIT_GUARDRAIL_PROFILE_ID = (
    "04c_fn_priority_strict_v1"
)

# Explicit 04C validation limits.
#
# They are intentionally stored independently from the 03B calibration
# constraints so that a later change in calibration rules cannot silently
# modify the batch-3 validation contract.
SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS = {
    "2way": {
        # Overall validation risk
        "max_fn_rate": 0.05,
        "max_fp_rate": 0.30,
        "min_balanced_accuracy": 0.75,

        # Worst validation image
        "max_image_fn_rate": 0.10,
        "max_image_fp_rate": 0.50,
    },

    "3way": {
        # Overall validation risk
        "max_fn_rate": 0.05,
        "max_fp_rate": 0.30,
        "max_uncertain_rate": 0.30,
        "min_balanced_accuracy": 0.75,

        # Worst validation image
        "max_image_target_miss_rate": 0.10,
        "max_image_false_accept_rate": 0.50,
        "max_image_uncertain_rate": 0.30,
    },
}

SIMCA_CONCAT_REFIT_EVALUATION_RULE_VERSION = (
    "04c_validation_metrics_v5_compact_ids_strict_guardrails"
)

SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS = {
    "2way": {
        "overall": (
            (
                "target_miss_rate",
                "max_fn_rate",
                "<=",
            ),
            (
                "false_accept_rate",
                "max_fp_rate",
                "<=",
            ),
            (
                "balanced_accuracy",
                "min_balanced_accuracy",
                ">=",
            ),
        ),
        "worst_image": (
            (
                "target_miss_rate",
                "max_image_fn_rate",
                "<=",
            ),
            (
                "false_accept_rate",
                "max_image_fp_rate",
                "<=",
            ),
        ),
    },

    "3way": {
        "overall": (
            (
                "target_miss_rate",
                "max_fn_rate",
                "<=",
            ),
            (
                "false_accept_rate",
                "max_fp_rate",
                "<=",
            ),
            (
                "uncertain_rate",
                "max_uncertain_rate",
                "<=",
            ),
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
    "amendment_type": "04c_guardrail_strengthening",
    "amendment_date": "2026-08-24",

    "reason": (
        "04C reuses the canonical 03B model_id, fit_id and projection_id, "
        "stores validation metrics in long format, evaluates the locked "
        "direct and pixel_to_object decision scopes, and applies a stricter "
        "FN-priority batch-3 acceptability profile aligned with the generic "
        "03B safety constraints. The selected model population, fitted model "
        "definitions, projection definitions and 03B-selected decision "
        "thresholds are unchanged."
    ),

    # Scientific decision thresholds selected in 03B are unchanged.
    "selected_decision_thresholds_changed": False,

    # Validation acceptance limits ARE changed.
    "guardrail_thresholds_changed": True,
    "numeric_thresholds_changed": True,

    # Architecture migration.
    "threshold_mapping_corrected": True,
    "execution_population_changed": False,
    "identifier_schema_changed": True,
    "decision_scopes_expanded": True,

    # Model outputs are not altered by this amendment.
    "model_parameters_changed": False,
    "continuous_predictions_changed": False,

    # IMPORTANT:
    # Leave False only if these numerical limits were defined without looking
    # at the current rebuilt-v5 batch-3 validation results.
    "batch3_used_to_choose_thresholds": False,
}

SIMCA_CONCAT_REFIT_EVALUATION_RULE_KEYS = (
    "SIMCA_CONCAT_REFIT_EVALUATION_RULE_VERSION",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_PROFILE_ID",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_SCOPES",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS",
    "SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS",
    "SIMCA_CONCAT_REFIT_EVALUATION_AMENDMENT",
    "SIMCA_CONCAT_REFIT_TRUTH_SOURCE",
    "SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL",
    "SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU",
    "SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN",
    "INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD",
    "INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY",
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
    "SIMCA_CONCAT_REFIT_EXECUTION_POLICY",
    "SIMCA_CONCAT_REFIT_BORDER_WIDTH",
    "SIMCA_CONCAT_REFIT_MAP_ENCODING",
    "SIMCA_CONCAT_REFIT_MAP_COMPRESSION_LEVEL",
    "SIMCA_CONCAT_REFIT_TRUTH_SOURCE",
)

# SIMCA_CONCAT_REFIT_CANDIDATE_COLUMNS = (
#     "validation_candidate_id",
#     "calibration_id",
#     "domain_config_id",
#     "evaluation_config_id",
#     "data_config_id",
#     "fit_config_id",
#     "projection_config_id",
#     "evaluation_track",
#     "track_id",
#     "parent_track",
#     "decision_mode",
#     "matrix_family",
#     "matrix_method",
#     "projection_level",
#     "projection_matrix_method",
#     "m",
#     "balanced_pixel_strategy",
#     "preprocessing",
#     "preprocessing_steps",
#     "rule_variant",
#     "limit_source",
#     "n_components",
#     "alpha",
#     "random_state",
#     "sg_window_length",
#     "sg_polyorder",
#     "direct_2way_threshold",
#     "secondary_object_threshold",
#     "three_way_lower_threshold",
#     "three_way_upper_threshold",
#     "position_dilation_radius",
#     "calibration_status",
#     "eligibility_status",
#     "candidate_front",
#     "visited_by_optuna",
#     "optuna_pareto",
# )
SIMCA_VALIDATION_EXECUTION_COLUMNS = (
    "model_id",
    "random_state",
    "fit_id",
    "projection_id",
    "track_id",
    "decision_mode",
    "projection_level",
    "matrix_method",
    "projection_matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing_steps",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "eligibility_status",
    "downstream_status",
)
SIMCA_VALIDATION_PREDICTION_BASE_COLUMNS = (
    "projection_id",
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
SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS = (
    "fit_id",
    "projection_id",
    "stage",
    "error_type",
    "error_message",
)
SIMCA_VALIDATION_METRIC_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "map_variant",
    "aggregation_level",
    "group_id",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "status",
    "error_type",
    "error_message",
)
SIMCA_PIXEL_MAP_MANIFEST_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
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
    "truth_level",
    "spatial_lock_sha256",
)
SIMCA_SPATIAL_COMPONENT_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
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
    "model_id",
    "random_state",
    "track_id",
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
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "eligibility_status",
    "downstream_status",
    "candidate_status",
    "scope",
    "metric",
    "observed_value",
    "ci_low",
    "ci_high",
    "comparator",
    "threshold",
    "check_status",
    "is_blocking",
    "reason_code",
    "reason",
)
# -----------------------------------------------------------------------
# Notebook 05 result schemas: model_id is the scientific identity and
# (model_id, random_state) is the execution identity. decision_scope is
# materialized in column prefixes in the wide execution/model summaries.
# -----------------------------------------------------------------------
# SIMCA_ROBUSTNESS_EXECUTION_KEY_COLUMNS = (
#     "model_id",
#     "random_state",
#     "track_id",
# )
# SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS = (
#     *SIMCA_ROBUSTNESS_EXECUTION_KEY_COLUMNS,
#     "decision_scope",
# )
# SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS = ("model_id", "track_id")
# SIMCA_ROBUSTNESS_MODEL_INVARIANT_COLUMNS = (
#     "model_id",
#     "track_id",
#     "decision_mode",
#     "projection_level",
#     "matrix_method",
#     "projection_matrix_method",
#     "m",
#     "balanced_pixel_strategy",
#     "preprocessing_steps",
#     "rule_variant",
#     "limit_source",
#     "n_components",
#     "alpha",
#     "sg_window_length",
#     "sg_polyorder",
#     "position_dilation_radius",
#     "eligibility_status",
#     "downstream_status",
# )
# SIMCA_ROBUSTNESS_SCOPED_VALUE_COLUMNS = tuple(
#     f"{scope}__{metric}"
#     for scope in SIMCA_ROBUSTNESS_DECISION_SCOPES
#     for metric in SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES
# )
# SIMCA_ROBUSTNESS_WORST_IMAGE_VALUE_COLUMNS = tuple(
#     f"{scope}__worst_image__{metric}"
#     for scope in SIMCA_ROBUSTNESS_DECISION_SCOPES
#     for metric in SIMCA_ROBUSTNESS_WORST_IMAGE_METRIC_NAMES
# )
# SIMCA_ROBUSTNESS_SPATIAL_VALUE_COLUMNS = tuple(
#     f"spatial__{metric}" for metric in SIMCA_ROBUSTNESS_SPATIAL_METRICS
# )
# SIMCA_ROBUSTNESS_EXECUTION_VALUE_COLUMNS = (
#     *SIMCA_ROBUSTNESS_SCOPED_VALUE_COLUMNS,
#     *SIMCA_ROBUSTNESS_WORST_IMAGE_VALUE_COLUMNS,
#     *SIMCA_ROBUSTNESS_SPATIAL_VALUE_COLUMNS,
# )
# SIMCA_ROBUSTNESS_EXECUTION_GUARDRAIL_COLUMNS = tuple(
#     f"{scope}__guardrail__{field}"
#     for scope in SIMCA_ROBUSTNESS_DECISION_SCOPES
#     for field in (
#         "candidate_status",
#         "scope_calculable",
#         "scope_protocol_pass",
#         "all_blocking_checks_pass",
#         "n_guardrail_checks",
#         "n_blocking_checks",
#         "n_blocking_failures",
#         "n_technical_errors",
#     )
# )
# SIMCA_ROBUSTNESS_EXECUTION_METRIC_COLUMNS = (
#     *SIMCA_VALIDATION_EXECUTION_COLUMNS,
#     "is_stochastic",
#     "preprocessing_step_count",
#     "execution_calculable",
#     "execution_protocol_supported",
#     "all_04c_blocking_guardrails_pass",
#     *SIMCA_ROBUSTNESS_EXECUTION_GUARDRAIL_COLUMNS,
#     *SIMCA_ROBUSTNESS_EXECUTION_VALUE_COLUMNS,
# )
# SIMCA_ROBUSTNESS_MODEL_SUMMARY_STAT_COLUMNS = tuple(
#     f"{stat}__{metric}"
#     for metric in SIMCA_ROBUSTNESS_EXECUTION_VALUE_COLUMNS
#     for stat in ("mean", "std", "min", "max")
# )
# SIMCA_ROBUSTNESS_MODEL_SUMMARY_COLUMNS = (
#     *SIMCA_ROBUSTNESS_MODEL_INVARIANT_COLUMNS,
#     "is_stochastic",
#     "preprocessing_step_count",
#     "n_random_states",
#     "n_expected_random_states",
#     "observed_random_states_json",
#     "missing_random_states_json",
#     "all_expected_random_states_present",
#     "seed_requirement_satisfied",
#     "all_execution_calculable",
#     "all_execution_protocol_supported",
#     "all_04c_blocking_guardrails_pass",
#     "model_diagnostic_eligible",
#     "model_protocol_eligible_pre_stability",
#     *SIMCA_ROBUSTNESS_EXECUTION_VALUE_COLUMNS,
#     *SIMCA_ROBUSTNESS_MODEL_SUMMARY_STAT_COLUMNS,
# )
# SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS = (
#     "model_id",
#     "track_id",
#     "decision_scope",
#     "n_random_states",
#     "n_entities",
#     "n_target_entities",
#     "entity_seed_coverage_complete",
#     "decision_disagreement_rate",
#     "target_decision_disagreement_rate",
#     "disagreement_status",
# )
# SIMCA_ROBUSTNESS_STABILITY_SUMMARY_COLUMNS = (
#     "model_id",
#     "track_id",
#     "metric",
#     "metric_base",
#     "metric_direction",
#     "is_stochastic",
#     "n_random_states",
#     "n_expected_random_states",
#     "observed_random_states_json",
#     "missing_random_states_json",
#     "all_expected_random_states_present",
#     "n_finite_values",
#     "mean",
#     "std",
#     "min",
#     "max",
#     "range",
#     "max_std_limit",
#     "max_range_limit",
#     "std_limit_exceeded",
#     "range_limit_exceeded",
#     "stability_metric_status",
#     "decision_disagreement_rate",
#     "target_decision_disagreement_rate",
#     "stability_flags",
#     "stability_flag_count",
#     "model_stability_status",
# )
# SIMCA_ROBUSTNESS_FINAL_SAFETY_COLUMNS = (
#     "model_id",
#     "track_id",
#     "eligibility_status",
#     "downstream_status",
#     "is_stochastic",
#     "n_random_states",
#     "n_expected_random_states",
#     "all_expected_random_states_present",
#     "seed_requirement_satisfied",
#     "all_execution_calculable",
#     "all_04c_blocking_guardrails_pass",
#     "model_stability_status",
#     "stability_flags",
#     "is_finally_admissible",
#     "final_admissibility_status",
#     "final_admissibility_reason",
# )
# SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS = (
#     *SIMCA_ROBUSTNESS_MODEL_SUMMARY_COLUMNS,
#     *tuple(
#         column
#         for column in SIMCA_ROBUSTNESS_FINAL_SAFETY_COLUMNS
#         if column not in SIMCA_ROBUSTNESS_MODEL_SUMMARY_COLUMNS
#     ),
#     "diagnostic_pareto_eligible",
#     "is_diagnostic_pareto",
#     "diagnostic_dominated_by_model_id",
#     "protocol_pareto_eligible",
#     "is_protocol_pareto",
#     "protocol_dominated_by_model_id",
#     "pareto_exclusion_reason",
# )
# SIMCA_ROBUSTNESS_PARETO_AUDIT_COLUMNS = (
#     "track_id",
#     "model_id",
#     "pool_type",
#     "is_candidate",
#     "is_pareto",
#     "dominated_by_model_id",
#     "reason_code",
#     "pareto_minimize_json",
#     "pareto_maximize_json",
#     "objective_values_json",
# )
# SIMCA_ROBUSTNESS_FINAL_SELECTION_AUDIT_COLUMNS = (
#     "model_id",
#     "track_id",
#     "is_finally_admissible",
#     "final_admissibility_status",
#     "final_admissibility_reason",
#     "is_protocol_pareto",
#     "selection_role",
#     "selection_status",
#     "elimination_step",
#     "elimination_criterion",
#     "selection_reason",
# )
# SIMCA_ROBUSTNESS_FINAL_MODEL_BASE_COLUMNS = (
#     "model_id",
#     "track_id",
#     "decision_mode",
#     "projection_level",
#     "matrix_method",
#     "projection_matrix_method",
#     "m",
#     "balanced_pixel_strategy",
#     "preprocessing_steps",
#     "rule_variant",
#     "limit_source",
#     "n_components",
#     "alpha",
#     "sg_window_length",
#     "sg_polyorder",
#     "position_dilation_radius",
#     "eligibility_status",
#     "downstream_status",
#     "is_stochastic",
#     "model_stability_status",
#     "stability_flags",
#     *SIMCA_ROBUSTNESS_EXECUTION_VALUE_COLUMNS,
#     "selection_role",
#     "selection_status",
#     "selection_reason",
#     "final_rank_in_track",
# )
# SIMCA_ROBUSTNESS_FINAL_SELECTED_MODEL_COLUMNS = (
#     *SIMCA_ROBUSTNESS_FINAL_MODEL_BASE_COLUMNS,
# )
# SIMCA_ROBUSTNESS_LOCKED_MODEL_COLUMNS = (
#     *SIMCA_ROBUSTNESS_FINAL_MODEL_BASE_COLUMNS,
# )
# SIMCA_ROBUSTNESS_STATISTICAL_UNCERTAINTY_COLUMNS = (
#     "model_id",
#     "track_id",
#     "decision_scope",
#     "metric",
#     "estimate",
#     "ci_low",
#     "ci_high",
#     "n_random_states",
#     "n_independent_images",
#     "support_status",
#     "analysis_stage",
#     "inferential_role",
# )
# SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS = (
#     "ablation_id",
#     "track_id",
#     "reference_model_id",
#     "ablated_model_id",
#     "contrast_type",
#     "factor",
#     "metric",
#     "reference_value",
#     "ablated_value",
#     "effect",
#     "practical_tolerance",
#     "effect_status",
#     "diagnostic_role",
# )

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
SIMCA_OPTUNA_BENCHMARK_SAMPLE_COLUMNS = (
    "track_id",
    "trial_number",
    "model_id",
    "is_repeat",
    "is_selected_reference",
)
SIMCA_OPTUNA_BENCHMARK_SUMMARY_COLUMNS = (
    "track_id",
    "downstream_status",
    "n_evaluable_models",
    "n_selected_reference_models",
    "trial_budget",
    "n_unique_models_sampled",
    "duplicate_trial_rate",
    "model_coverage_rate",
    "n_selected_reference_recovered",
    "selected_reference_recall",
    "uniform_expected_selected_recall",
    "recall_delta_vs_uniform",
)

# Legacy schemas retained for the not-yet-migrated compatibility functions in
# simca_optuna.py. Notebook 04B writes only the benchmark schemas above.
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
    "secondary_three_way_lower_threshold",
    "secondary_three_way_upper_threshold",
    "random_state",
    "calibration_status",
    "schema_version",
    "protocol_version",
    "protocol_hash",
    "pca_selection_fingerprint",
)
SIMCA_GRID_SELECTED_FOLD_METRIC_COLUMNS = (
    "model_id",
    "random_state",
    "decision_scope",
    "fold_id",
    "n_observations",
    "n_target",
    "n_non_target",
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "max_unit_target_miss_rate",
    "max_unit_false_accept_rate",
)
SIMCA_GRID_MODEL_REFERENCE_COLUMNS = (
    "model_id",
    "track_id",
    "n_selected_runs",
    "n_decision_scopes",
    "eligibility_status",
    "downstream_status",
    "max_abs_metric_difference",
)

# Legacy schemas used by the not-yet-migrated 04B/04C workflows.
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



# ---------------------------------------------------------------------------
# Canonical tabular identity and persistence contracts (notebooks 00-05)
# ---------------------------------------------------------------------------
# These constants do not create any new identifier in persisted data. They
# only name the natural keys already produced by notebooks 00-04C and by the
# notebook-05 child workflow. `simca_tables.py` consumes these contracts but
# defines no schema or identifier constant of its own.
#
# The identity progression is intentionally minimal:
#   00-01 : clean_key / object_id / reference_id
#   02    : matrix_id
#   03    : candidate_id / selection_unit_id
#   03B+  : model_id (scientific model), random_state (repetition),
#           fit_id and projection_id (technical identities)
# No validation_candidate_id, calibration_id, execution_id or seed_id is
# introduced by the active 03B-05 workflow.

# Dataset / QC identities.
DATABASE_IMAGE_KEY_COLUMNS = ("clean_key",)
DATABASE_OBJECT_KEY_COLUMNS = ("object_id",)
DATABASE_SEGMENTATION_KEY_COLUMNS = ("clean_key", "label_id")
QC_PIXEL_KEY_COLUMNS = ("object_id", "pixel_index")
QC_REVIEW_KEY_COLUMNS = ("record_type", "record_id", "flag_type")
QC_EXCLUSION_KEY_COLUMNS = ("record_type", "record_id")
PROTOCOL_SPLIT_KEY_COLUMNS = ("object_id",)

# Independent spatial-ground-truth identities.
SPATIAL_GT_REFERENCE_KEY_COLUMNS = ("reference_id",)
SPATIAL_GT_COMPONENT_KEY_COLUMNS = ("reference_id", "component_id")
SPATIAL_GT_AGREEMENT_KEY_COLUMNS = (
    "source_image",
    "reference_id_a",
    "reference_id_b",
)
SPATIAL_GT_ADJUDICATION_KEY_COLUMNS = ("source_image",)

# Matrix / PCA identities.
MATRIX_KEY_COLUMNS = ("matrix_id",)
MATRIX_COVERAGE_KEY_COLUMNS = ("matrix_id", "object_id")
BALANCED_SAMPLING_KEY_COLUMNS = ("m", "strategy")
PIXEL_SAMPLING_DIAGNOSTIC_KEY_COLUMNS = (
    "m",
    "strategy",
    "seed",
    "object_id",
)
PREPROCESSING_VALIDATION_KEY_COLUMNS = (
    "matrix_id",
    "fit_role",
    "eval_role",
    "preprocessing",
    "sg_window_length",
    "sg_polyorder",
    "deriv",
)
PCA_CANDIDATE_KEY_COLUMNS = ("candidate_id",)
PCA_COMPONENT_KEY_COLUMNS = ("candidate_id", "component")
PCA_DIAGNOSTIC_KEY_COLUMNS = (
    "candidate_id",
    "diagnostic_group",
    "metric",
)
PCA_SELECTION_UNIT_KEY_COLUMNS = ("selection_unit_id",)

# Canonical SIMCA identities from 03B onward. These are aliases of the
# scientific structure already used by the notebooks; they do not add IDs.
SIMCA_SCIENTIFIC_MODEL_KEY_COLUMNS = ("model_id",)
SIMCA_EXECUTION_KEY_COLUMNS = ("model_id", "random_state")
SIMCA_DECISION_EXECUTION_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "decision_scope",
)
SIMCA_TECHNICAL_FIT_KEY_COLUMNS = ("fit_id",)
SIMCA_TECHNICAL_PROJECTION_KEY_COLUMNS = ("projection_id",)

# 03B natural table keys.
INTERNAL_CALIBRATION_TRACK_CONTRACT_KEY_COLUMNS = ("track_id",)
INTERNAL_CALIBRATION_FOLD_KEY_COLUMNS = ("object_id",)
INTERNAL_CALIBRATION_FOLD_DIAGNOSTIC_KEY_COLUMNS = ("fold_id",)
INTERNAL_CALIBRATION_MODEL_CATALOG_KEY_COLUMNS = SIMCA_SCIENTIFIC_MODEL_KEY_COLUMNS
INTERNAL_CALIBRATION_CANDIDATE_RUN_KEY_COLUMNS = SIMCA_EXECUTION_KEY_COLUMNS
INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_KEY_COLUMNS = ("fit_id", "fold_id")
INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_KEY_COLUMNS = (
    "projection_id",
    "fold_id",
)
INTERNAL_CALIBRATION_PROJECTION_SHIFT_KEY_COLUMNS = (
    "projection_id",
    "fold_id",
)
INTERNAL_CALIBRATION_OOF_OBJECT_KEY_COLUMNS = (
    "projection_id",
    "fold_id",
    "source_image",
    "object_id",
)
INTERNAL_CALIBRATION_OOF_PIXEL_KEY_COLUMNS = (
    *INTERNAL_CALIBRATION_OOF_OBJECT_KEY_COLUMNS,
    "row",
    "col",
)
INTERNAL_CALIBRATION_THRESHOLD_METRIC_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "evaluation_fold",
    "decision_scope",
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
    "metric",
)
INTERNAL_CALIBRATION_MODEL_METRIC_KEY_COLUMNS = ("model_id", "metric")
INTERNAL_CALIBRATION_SELECTED_MODEL_KEY_COLUMNS = SIMCA_SCIENTIFIC_MODEL_KEY_COLUMNS
INTERNAL_CALIBRATION_SELECTED_RUN_KEY_COLUMNS = SIMCA_EXECUTION_KEY_COLUMNS
INTERNAL_CALIBRATION_SELECTED_THRESHOLD_KEY_COLUMNS = (
    SIMCA_DECISION_EXECUTION_KEY_COLUMNS
)

# 03C natural table keys.
PROJECTION_SHIFT_DIAGNOSTIC_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "fold_id",
    "stratum_type",
    "stratum_value",
)
PROJECTION_ELIGIBILITY_KEY_COLUMNS = ("track_id",)
SPATIAL_CALIBRATION_METRIC_KEY_COLUMNS = (
    "spatial_candidate_id",
    "model_id",
    "random_state",
    "track_id",
    "map_variant",
)
FRAGMENT_SIZE_CLASS_KEY_COLUMNS = (
    "spatial_candidate_id",
    "model_id",
    "random_state",
    "track_id",
    "area_class",
)

# 04A / 04B natural table keys.
SIMCA_GRID_MODEL_REFERENCE_KEY_COLUMNS = SIMCA_SCIENTIFIC_MODEL_KEY_COLUMNS
SIMCA_GRID_SELECTED_FOLD_METRIC_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "decision_scope",
    "fold_id",
)
SIMCA_OPTUNA_BENCHMARK_SAMPLE_KEY_COLUMNS = ("track_id", "trial_number")
SIMCA_OPTUNA_BENCHMARK_SUMMARY_KEY_COLUMNS = ("track_id",)

# 04C natural table keys. fit_id/projection_id are reused from 03B; 04C does
# not mint another scientific identity.
SIMCA_VALIDATION_EXECUTION_KEY_COLUMNS = SIMCA_EXECUTION_KEY_COLUMNS
SIMCA_VALIDATION_OBJECT_PREDICTION_KEY_COLUMNS = (
    "projection_id",
    "source_image",
    "object_id",
)
SIMCA_VALIDATION_PIXEL_PREDICTION_KEY_COLUMNS = (
    *SIMCA_VALIDATION_OBJECT_PREDICTION_KEY_COLUMNS,
    "row",
    "col",
)
SIMCA_VALIDATION_METRIC_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "map_variant",
    "aggregation_level",
    "group_id",
    "metric",
)
SIMCA_PIXEL_MAP_MANIFEST_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "source_image",
)
SIMCA_SPATIAL_COMPONENT_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "source_image",
    "map_variant",
    "component_role",
    "component_id",
)
SIMCA_SPATIAL_COMPONENT_METRIC_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "source_image",
    "aggregation_level",
    "map_variant",
)
SIMCA_VALIDATION_GUARDRAIL_KEY_COLUMNS = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
    "scope",
    "metric",
)

# ---------------------------------------------------------------------------
# Table registry consumed by src.workflows.simca_tables.
# ---------------------------------------------------------------------------
# `columns=()` means that the upstream notebook owns a dynamic/legacy column
# set that is not fully declared in experiment_config. In that case the table
# helper validates/protects the declared natural key but does not invent or
# drop columns. Exact schemas are used wherever notebooks 00-04C already
# define one centrally.

def _table_contract(columns, key_columns=(), *, unique_key=True):
    return {
        "columns": tuple(columns),
        "key_columns": tuple(key_columns),
        "unique_key": bool(unique_key),
    }


PIPELINE_TABLE_CONTRACTS = {
    # Notebook 00 — database construction.
    "00_raw_image_manifest": _table_contract(
        RAW_IMAGE_MANIFEST_COLUMNS,
        DATABASE_IMAGE_KEY_COLUMNS,
    ),
    "00_metadata_parsing_errors": _table_contract(
        METADATA_PARSING_ERROR_COLUMNS,
        (),
        unique_key=False,
    ),
    "00_image_summary": _table_contract(
        DATABASE_IMAGE_SUMMARY_COLUMNS,
        DATABASE_IMAGE_KEY_COLUMNS,
    ),
    "00_object_summary": _table_contract(
        DATABASE_OBJECT_SUMMARY_COLUMNS,
        DATABASE_OBJECT_KEY_COLUMNS,
    ),
    "00_segmentation_diagnostics": _table_contract(
        SEGMENTATION_DIAGNOSTIC_COLUMNS,
        DATABASE_SEGMENTATION_KEY_COLUMNS,
    ),
    "00_database_manifest": _table_contract(
        DATABASE_MANIFEST_COLUMNS,
        ("database_id",),
    ),
    "00_terminal_band_qc": _table_contract(
        (),
        (),
        unique_key=False,
    ),

    # Notebook 01 — quality control and frozen split.
    "01_image_qc_summary": _table_contract(
        IMAGE_QC_OUTPUT_COLUMNS,
        DATABASE_IMAGE_KEY_COLUMNS,
    ),
    "01_object_qc_summary": _table_contract(
        OBJECT_QC_OUTPUT_COLUMNS,
        DATABASE_OBJECT_KEY_COLUMNS,
    ),
    "01_qc_alerts": _table_contract(
        QC_ALERT_OUTPUT_COLUMNS,
        ("alert_id",),
    ),
    "01_qc_review": _table_contract(
        QC_REVIEW_OUTPUT_COLUMNS,
        QC_REVIEW_KEY_COLUMNS,
    ),
    "01_exclusion_manifest": _table_contract(
        QC_EXCLUSION_OUTPUT_COLUMNS,
        QC_EXCLUSION_KEY_COLUMNS,
    ),
    "01_qc_protocol": _table_contract(
        QC_PROTOCOL_OUTPUT_COLUMNS,
        ("protocol_version",),
    ),
    "01_protocol_split_manifest": _table_contract(
        PROTOCOL_SPLIT_MANIFEST_COLUMNS,
        PROTOCOL_SPLIT_KEY_COLUMNS,
    ),
    "01_split_diagnostics": _table_contract(
        SPLIT_DIAGNOSTIC_COLUMNS,
        ("protocol_role", "label", "batch"),
    ),
    "01_pixel_spectral_qc": _table_contract(
        PIXEL_SPECTRAL_QC_COLUMNS,
        QC_PIXEL_KEY_COLUMNS,
    ),
    "01_pixel_exclusions": _table_contract(
        PIXEL_SPECTRAL_QC_COLUMNS,
        QC_PIXEL_KEY_COLUMNS,
    ),

    # Notebook 01B — independent spatial ground truth.
    "01b_spatial_ground_truth_manifest": _table_contract(
        SPATIAL_GT_MANIFEST_COLUMNS,
        SPATIAL_GT_REFERENCE_KEY_COLUMNS,
    ),
    "01b_fragment_reference_components": _table_contract(
        SPATIAL_GT_COMPONENT_COLUMNS,
        SPATIAL_GT_COMPONENT_KEY_COLUMNS,
    ),
    "01b_annotation_agreement": _table_contract(
        SPATIAL_GT_AGREEMENT_COLUMNS,
        SPATIAL_GT_AGREEMENT_KEY_COLUMNS,
    ),
    "01b_annotation_adjudication": _table_contract(
        SPATIAL_GT_ADJUDICATION_COLUMNS,
        SPATIAL_GT_ADJUDICATION_KEY_COLUMNS,
    ),

    # Notebook 02 — matrix and preprocessing contracts.
    "02_wavelength_config": _table_contract(
        (),
        ("wavelength_axis_id",),
    ),
    "02_m_feasibility": _table_contract(
        M_FEASIBILITY_COLUMNS,
        BALANCED_SAMPLING_KEY_COLUMNS,
    ),
    "02_pixel_sampling_diagnostics": _table_contract(
        PIXEL_SAMPLING_DIAGNOSTIC_COLUMNS,
        PIXEL_SAMPLING_DIAGNOSTIC_KEY_COLUMNS,
    ),
    "02_matrix_summary": _table_contract(
        MATRIX_SUMMARY_REQUIRED_COLUMNS,
        MATRIX_KEY_COLUMNS,
    ),
    "02_matrix_coverage": _table_contract(
        MATRIX_COVERAGE_COLUMNS,
        MATRIX_COVERAGE_KEY_COLUMNS,
    ),
    "02_matrix_errors": _table_contract(
        MATRIX_ERROR_COLUMNS,
        MATRIX_KEY_COLUMNS,
        unique_key=False,
    ),
    "02_preprocessing_validation": _table_contract(
        PREPROCESSING_SUMMARY_REQUIRED_COLUMNS,
        PREPROCESSING_VALIDATION_KEY_COLUMNS,
    ),
    "02_preprocessing_errors": _table_contract(
        PREPROCESSING_ERROR_COLUMNS,
        (
            "matrix_id",
            "fit_role",
            "eval_role",
            "preprocessing",
            "sg_window_length",
            "error_type",
        ),
        unique_key=False,
    ),

    # Notebook 03 — PCA candidates / selection units.
    "03_pca_candidate_registry": _table_contract(
        PCA_CANDIDATE_REGISTRY_COLUMNS,
        PCA_CANDIDATE_KEY_COLUMNS,
    ),
    "03_pca_summary": _table_contract(
        PCA_SUMMARY_COLUMNS,
        PCA_COMPONENT_KEY_COLUMNS,
    ),
    "03_pca_scoring_diagnostics": _table_contract(
        PCA_SCORING_DIAGNOSTIC_COLUMNS,
        PCA_DIAGNOSTIC_KEY_COLUMNS,
        unique_key=False,
    ),
    "03_pca_preprocessing_summary": _table_contract(
        (),
        PCA_PREPROCESSING_SUMMARY_ID_COLUMNS,
    ),
    "03_pca_selected_preprocessings": _table_contract(
        PCA_SELECTED_PREPROCESSING_COLUMNS,
        PCA_SELECTION_UNIT_KEY_COLUMNS,
    ),
    "03_pca_artifact_review": _table_contract(
        PCA_ARTIFACT_REVIEW_COLUMNS,
        PCA_CANDIDATE_KEY_COLUMNS,
    ),
    "03_pca_selection_audit": _table_contract(
        SELECTION_AUDIT_COLUMNS,
        (),
        unique_key=False,
    ),

    # Notebook 03B — canonical SIMCA model / execution / fit / projection IDs.
    "03b_track_contracts": _table_contract(
        INTERNAL_CALIBRATION_TRACK_CONTRACT_COLUMNS,
        INTERNAL_CALIBRATION_TRACK_CONTRACT_KEY_COLUMNS,
    ),
    "03b_calibration_folds": _table_contract(
        INTERNAL_CALIBRATION_FOLD_COLUMNS,
        INTERNAL_CALIBRATION_FOLD_KEY_COLUMNS,
    ),
    "03b_fold_diagnostics": _table_contract(
        INTERNAL_CALIBRATION_FOLD_DIAGNOSTIC_COLUMNS,
        INTERNAL_CALIBRATION_FOLD_DIAGNOSTIC_KEY_COLUMNS,
    ),
    "03b_model_catalog": _table_contract(
        INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        INTERNAL_CALIBRATION_MODEL_CATALOG_KEY_COLUMNS,
    ),
    "03b_candidate_runs": _table_contract(
        INTERNAL_CALIBRATION_CANDIDATE_RUN_COLUMNS,
        INTERNAL_CALIBRATION_CANDIDATE_RUN_KEY_COLUMNS,
    ),
    "03b_fit_diagnostics": _table_contract(
        INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_COLUMNS,
        INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_KEY_COLUMNS,
    ),
    "03b_rule_diagnostics": _table_contract(
        INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS,
        INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_KEY_COLUMNS,
    ),
    "03b_projection_shift": _table_contract(
        INTERNAL_CALIBRATION_PROJECTION_SHIFT_COLUMNS,
        INTERNAL_CALIBRATION_PROJECTION_SHIFT_KEY_COLUMNS,
    ),
    "03b_oof_object_predictions": _table_contract(
        INTERNAL_CALIBRATION_OOF_OBJECT_COLUMNS,
        INTERNAL_CALIBRATION_OOF_OBJECT_KEY_COLUMNS,
    ),
    "03b_oof_pixel_predictions": _table_contract(
        INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS,
        INTERNAL_CALIBRATION_OOF_PIXEL_KEY_COLUMNS,
    ),
    "03b_threshold_metrics": _table_contract(
        INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS,
        INTERNAL_CALIBRATION_THRESHOLD_METRIC_KEY_COLUMNS,
    ),
    "03b_model_metrics": _table_contract(
        INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS,
        INTERNAL_CALIBRATION_MODEL_METRIC_KEY_COLUMNS,
    ),
    "03b_selected_models": _table_contract(
        INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS,
        INTERNAL_CALIBRATION_SELECTED_MODEL_KEY_COLUMNS,
    ),
    "03b_selected_runs": _table_contract(
        INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS,
        INTERNAL_CALIBRATION_SELECTED_RUN_KEY_COLUMNS,
    ),
    "03b_selected_thresholds": _table_contract(
        INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
        INTERNAL_CALIBRATION_SELECTED_THRESHOLD_KEY_COLUMNS,
    ),
    "03b_selection_audit": _table_contract(
        INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS,
        (),
        unique_key=False,
    ),
    "03b_technical_events": _table_contract(
        INTERNAL_CALIBRATION_TECHNICAL_EVENT_COLUMNS,
        (),
        unique_key=False,
    ),

    # Notebook 03C — projection-domain and spatial calibration audit.
    "03c_projection_shift_diagnostics": _table_contract(
        PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS,
        PROJECTION_SHIFT_DIAGNOSTIC_KEY_COLUMNS,
    ),
    "03c_projection_eligibility": _table_contract(
        PROJECTION_ELIGIBILITY_COLUMNS,
        PROJECTION_ELIGIBILITY_KEY_COLUMNS,
    ),
    "03c_spatial_calibration_metrics": _table_contract(
        SPATIAL_CALIBRATION_METRIC_COLUMNS,
        SPATIAL_CALIBRATION_METRIC_KEY_COLUMNS,
    ),
    "03c_fragment_size_classes": _table_contract(
        FRAGMENT_SIZE_CLASS_COLUMNS,
        FRAGMENT_SIZE_CLASS_KEY_COLUMNS,
    ),

    # Notebook 04A — reference audit of the 03B-selected population.
    "04a_model_reference": _table_contract(
        SIMCA_GRID_MODEL_REFERENCE_COLUMNS,
        SIMCA_GRID_MODEL_REFERENCE_KEY_COLUMNS,
    ),
    "04a_selected_run_fold_metrics": _table_contract(
        SIMCA_GRID_SELECTED_FOLD_METRIC_COLUMNS,
        SIMCA_GRID_SELECTED_FOLD_METRIC_KEY_COLUMNS,
    ),

    # Notebook 04B — lookup-only categorical TPE benchmark.
    "04b_sampled_models": _table_contract(
        SIMCA_OPTUNA_BENCHMARK_SAMPLE_COLUMNS,
        SIMCA_OPTUNA_BENCHMARK_SAMPLE_KEY_COLUMNS,
    ),
    "04b_search_efficiency": _table_contract(
        SIMCA_OPTUNA_BENCHMARK_SUMMARY_COLUMNS,
        SIMCA_OPTUNA_BENCHMARK_SUMMARY_KEY_COLUMNS,
    ),

    # Notebook 04C — locked batch-3 validation. No new scientific ID.
    "04c_validation_object_predictions": _table_contract(
        SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS,
        SIMCA_VALIDATION_OBJECT_PREDICTION_KEY_COLUMNS,
    ),
    "04c_validation_pixel_predictions": _table_contract(
        SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS,
        SIMCA_VALIDATION_PIXEL_PREDICTION_KEY_COLUMNS,
    ),
    "04c_validation_metrics": _table_contract(
        SIMCA_VALIDATION_METRIC_COLUMNS,
        SIMCA_VALIDATION_METRIC_KEY_COLUMNS,
    ),
    "04c_pixel_maps_manifest": _table_contract(
        SIMCA_PIXEL_MAP_MANIFEST_COLUMNS,
        SIMCA_PIXEL_MAP_MANIFEST_KEY_COLUMNS,
    ),
    "04c_spatial_components": _table_contract(
        SIMCA_SPATIAL_COMPONENT_COLUMNS,
        SIMCA_SPATIAL_COMPONENT_KEY_COLUMNS,
    ),
    "04c_spatial_component_metrics": _table_contract(
        SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS,
        SIMCA_SPATIAL_COMPONENT_METRIC_KEY_COLUMNS,
    ),
    "04c_validation_guardrails": _table_contract(
        SIMCA_VALIDATION_GUARDRAIL_COLUMNS,
        SIMCA_VALIDATION_GUARDRAIL_KEY_COLUMNS,
    ),
    "04c_validation_technical_events": _table_contract(
        SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS,
        (),
        unique_key=False,
    ),

    # Notebook 05 child workflow.
    "05_selection_units": _table_contract(
        SIMCA_ROBUSTNESS_SELECTION_UNIT_COLUMNS,
        SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS,
    ),
    "05_selection_members": _table_contract(
        SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS,
        SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS,
    ),
    "05_pareto_candidates": _table_contract(
        SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS,
        SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS,
    ),
    "05_pareto_audit": _table_contract(
        SIMCA_ROBUSTNESS_PARETO_AUDIT_COLUMNS,
        (),
        unique_key=False,
    ),
    "05_seed_executions": _table_contract(
        SIMCA_ROBUSTNESS_SEED_EXECUTION_COLUMNS,
        SIMCA_ROBUSTNESS_SEED_EXECUTION_KEY_COLUMNS,
    ),
    "05_seed_thresholds": _table_contract(
        SIMCA_ROBUSTNESS_SEED_THRESHOLD_COLUMNS,
        SIMCA_ROBUSTNESS_SEED_THRESHOLD_KEY_COLUMNS,
    ),
    "05_seed_metrics": _table_contract(
        SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS,
        SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS,
    ),
    "05_stability_summary": _table_contract(
        SIMCA_ROBUSTNESS_STABILITY_SUMMARY_COLUMNS,
        ("model_id", "track_id", "metric"),
    ),
    "05_seed_disagreement": _table_contract(
        SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS,
        ("model_id", "track_id", "decision_scope"),
    ),
    "05_ablation_plan": _table_contract(
        SIMCA_ROBUSTNESS_ABLATION_PLAN_COLUMNS,
        ("track_id", "reference_model_id", "ablated_model_id", "factor"),
    ),
    "05_ablation_diagnostics": _table_contract(
        SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS,
        (
            "track_id",
            "reference_model_id",
            "ablated_model_id",
            "factor",
            "metric",
        ),
    ),
    "05_statistical_uncertainty": _table_contract(
        SIMCA_ROBUSTNESS_STATISTICAL_UNCERTAINTY_COLUMNS,
        ("model_id", "track_id", "decision_scope", "metric"),
    ),
    "05_risk_coverage": _table_contract(
        SIMCA_ROBUSTNESS_RISK_COVERAGE_COLUMNS,
        ("model_id", "track_id", "requested_coverage"),
    ),
    "05_review_guardrails": _table_contract(
        SIMCA_ROBUSTNESS_REVIEW_GUARDRAIL_COLUMNS,
        ("model_id", "track_id", "check_scope", "check_name"),
    ),
    "05_track_review": _table_contract(
        SIMCA_ROBUSTNESS_TRACK_REVIEW_COLUMNS,
        SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS,
    ),
    "05_pure_test_candidates": _table_contract(
        SIMCA_ROBUSTNESS_PURE_TEST_CANDIDATE_COLUMNS,
        SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS,
    ),
    "05_threshold_sensitivity_plan": _table_contract(
        SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_PLAN_COLUMNS,
        (
            "model_id",
            "random_state",
            "decision_scope",
            "perturbation_type",
            "perturbation_value",
        ),
    ),
    "05_threshold_sensitivity_metrics": _table_contract(
        SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRIC_COLUMNS,
        (
            "model_id",
            "random_state",
            "decision_scope",
            "perturbation_type",
            "perturbation_value",
            "metric",
        ),
    ),
    "05_threshold_sensitivity_decisions": _table_contract(
        SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DECISION_COLUMNS,
        (
            "model_id",
            "random_state",
            "decision_scope",
            "perturbation_type",
            "perturbation_value",
        ),
    ),
    "05_threshold_stability": _table_contract(
        SIMCA_ROBUSTNESS_THRESHOLD_STABILITY_COLUMNS,
        ("model_id", "track_id", "decision_scope"),
    ),
    "05_source_image_influence": _table_contract(
        SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_COLUMNS,
        (
            "model_id",
            "random_state",
            "track_id",
            "decision_scope",
            "metric",
            "omitted_source_image",
        ),
    ),
    "05_fold_sensitivity_plan": _table_contract(
        SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_PLAN_COLUMNS,
        ("reference_partition_sha256", "alternative_partition_sha256"),
    ),
    "05_fold_sensitivity_assignments": _table_contract(
        SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_ASSIGNMENT_COLUMNS,
        ("alternative_partition_sha256", "object_id"),
    ),
    "05_fold_sensitivity_thresholds": _table_contract(
        SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_THRESHOLD_COLUMNS,
        (
            "alternative_partition_sha256",
            "model_id",
            "random_state",
            "decision_scope",
        ),
    ),
    "05_fold_sensitivity_metrics": _table_contract(
        SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRIC_COLUMNS,
        (
            "alternative_partition_sha256",
            "model_id",
            "random_state",
            "decision_scope",
            "metric",
        ),
    ),
    "05_fold_sensitivity_decisions": _table_contract(
        SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_DECISION_COLUMNS,
        (
            "alternative_partition_sha256",
            "model_id",
            "random_state",
            "decision_scope",
        ),
    ),
    "05_fold_sensitivity_technical_events": _table_contract(
        SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_TECHNICAL_EVENT_COLUMNS,
        (),
        unique_key=False,
    ),
    "05_pareto_robustness_replicates": _table_contract(
        SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_REPLICATE_COLUMNS,
        ("model_id", "track_id", "omitted_random_state"),
    ),
    "05_pareto_robustness_summary": _table_contract(
        SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_SUMMARY_COLUMNS,
        ("model_id", "track_id"),
    ),
    "05_pareto_robustness_audit": _table_contract(
        SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_AUDIT_COLUMNS,
        ("track_id", "omitted_random_state"),
    ),
    "05_spatial_sensitivity_plan": _table_contract(
        SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_PLAN_COLUMNS,
        ("track_id", "factor", "alternative_spatial_candidate_id"),
    ),
    "05_spatial_sensitivity_metrics": _table_contract(
        SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_COLUMNS,
        (
            "model_id",
            "random_state",
            "track_id",
            "factor",
            "alternative_spatial_candidate_id",
            "metric",
        ),
    ),
    "05_ablation_coverage": _table_contract(
        SIMCA_ROBUSTNESS_ABLATION_COVERAGE_COLUMNS,
        ("track_id", "factor"),
    ),
}


# Resolve canonical persisted parquet names without repeating filename strings.
PIPELINE_TABLE_KIND_BY_FILE_NAME = {
    # 00
    DATABASE_OUTPUT_FILENAMES["raw_image_manifest"]: "00_raw_image_manifest",
    DATABASE_OUTPUT_FILENAMES["metadata_parsing_errors"]: "00_metadata_parsing_errors",
    DATABASE_OUTPUT_FILENAMES["image_summary"]: "00_image_summary",
    DATABASE_OUTPUT_FILENAMES["object_summary"]: "00_object_summary",
    DATABASE_OUTPUT_FILENAMES["segmentation_diagnostics"]: "00_segmentation_diagnostics",
    DATABASE_OUTPUT_FILENAMES["manifest"]: "00_database_manifest",
    DATABASE_OUTPUT_FILENAMES["terminal_band_qc"]: "00_terminal_band_qc",
    # 01
    QC_OUTPUT_FILENAMES["image_summary"]: "01_image_qc_summary",
    QC_OUTPUT_FILENAMES["object_summary"]: "01_object_qc_summary",
    QC_OUTPUT_FILENAMES["alerts"]: "01_qc_alerts",
    QC_OUTPUT_FILENAMES["review"]: "01_qc_review",
    QC_OUTPUT_FILENAMES["exclusion_manifest"]: "01_exclusion_manifest",
    QC_OUTPUT_FILENAMES["protocol"]: "01_qc_protocol",
    QC_OUTPUT_FILENAMES["split_manifest"]: "01_protocol_split_manifest",
    QC_OUTPUT_FILENAMES["split_diagnostics"]: "01_split_diagnostics",
    QC_OUTPUT_FILENAMES["pixel_spectral_qc"]: "01_pixel_spectral_qc",
    QC_OUTPUT_FILENAMES["pixel_exclusions"]: "01_pixel_exclusions",
    # 01B
    SPATIAL_GT_OUTPUT_FILENAMES["manifest"]: "01b_spatial_ground_truth_manifest",
    SPATIAL_GT_OUTPUT_FILENAMES["components"]: "01b_fragment_reference_components",
    SPATIAL_GT_OUTPUT_FILENAMES["agreement"]: "01b_annotation_agreement",
    SPATIAL_GT_OUTPUT_FILENAMES["adjudication"]: "01b_annotation_adjudication",
    # 02
    MATRIX_OUTPUT_FILENAMES["wavelength_config"]: "02_wavelength_config",
    MATRIX_OUTPUT_FILENAMES["m_feasibility"]: "02_m_feasibility",
    MATRIX_OUTPUT_FILENAMES["pixel_sampling_diagnostics"]: "02_pixel_sampling_diagnostics",
    MATRIX_OUTPUT_FILENAMES["matrix_summary"]: "02_matrix_summary",
    MATRIX_OUTPUT_FILENAMES["matrix_coverage"]: "02_matrix_coverage",
    MATRIX_OUTPUT_FILENAMES["matrix_errors"]: "02_matrix_errors",
    MATRIX_OUTPUT_FILENAMES["preprocessing_validation"]: "02_preprocessing_validation",
    MATRIX_OUTPUT_FILENAMES["preprocessing_errors"]: "02_preprocessing_errors",
    # 03
    PCA_OUTPUT_FILENAMES["candidate_registry"]: "03_pca_candidate_registry",
    PCA_OUTPUT_FILENAMES["summary"]: "03_pca_summary",
    PCA_OUTPUT_FILENAMES["diagnostics"]: "03_pca_scoring_diagnostics",
    PCA_OUTPUT_FILENAMES["preprocessing_summary"]: "03_pca_preprocessing_summary",
    PCA_OUTPUT_FILENAMES["selected"]: "03_pca_selected_preprocessings",
    PCA_OUTPUT_FILENAMES["artifact_review"]: "03_pca_artifact_review",
    PCA_OUTPUT_FILENAMES["selection_audit"]: "03_pca_selection_audit",
    # 03B
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["track_contracts"]: "03b_track_contracts",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["folds"]: "03b_calibration_folds",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["fold_diagnostics"]: "03b_fold_diagnostics",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["model_catalog"]: "03b_model_catalog",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["candidate_runs"]: "03b_candidate_runs",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["fit_diagnostics"]: "03b_fit_diagnostics",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["rule_diagnostics"]: "03b_rule_diagnostics",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["projection_shift"]: "03b_projection_shift",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["oof_object_predictions"]: "03b_oof_object_predictions",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["oof_pixel_predictions"]: "03b_oof_pixel_predictions",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["threshold_metrics"]: "03b_threshold_metrics",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["model_metrics"]: "03b_model_metrics",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["selected_models"]: "03b_selected_models",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["selected_runs"]: "03b_selected_runs",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["selected_thresholds"]: "03b_selected_thresholds",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["selection_audit"]: "03b_selection_audit",
    INTERNAL_CALIBRATION_OUTPUT_FILENAMES["technical_events"]: "03b_technical_events",
    # 03C
    DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES["projection_shift_diagnostics"]: "03c_projection_shift_diagnostics",
    DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES["projection_eligibility"]: "03c_projection_eligibility",
    DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES["spatial_calibration_metrics"]: "03c_spatial_calibration_metrics",
    DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES["fragment_size_classes"]: "03c_fragment_size_classes",
    # 04A
    SIMCA_GRID_SEARCH_OUTPUT_FILENAMES["model_reference"]: "04a_model_reference",
    SIMCA_GRID_SEARCH_OUTPUT_FILENAMES["fold_metrics"]: "04a_selected_run_fold_metrics",
    # 04B
    SIMCA_OPTUNA_OUTPUT_FILENAMES["sampled_models"]: "04b_sampled_models",
    SIMCA_OPTUNA_OUTPUT_FILENAMES["search_efficiency"]: "04b_search_efficiency",
    # 04C
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["object_predictions"]: "04c_validation_object_predictions",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["pixel_predictions"]: "04c_validation_pixel_predictions",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["metrics"]: "04c_validation_metrics",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["pixel_maps_manifest"]: "04c_pixel_maps_manifest",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["spatial_components"]: "04c_spatial_components",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["spatial_component_metrics"]: "04c_spatial_component_metrics",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["guardrails"]: "04c_validation_guardrails",
    SIMCA_CONCAT_REFIT_OUTPUT_FILENAMES["technical_events"]: "04c_validation_technical_events",
    # 05
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["selection_units"]: "05_selection_units",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["selection_members"]: "05_selection_members",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["pareto_candidates"]: "05_pareto_candidates",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["pareto_audit"]: "05_pareto_audit",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["seed_executions"]: "05_seed_executions",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["seed_thresholds"]: "05_seed_thresholds",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["seed_metrics"]: "05_seed_metrics",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["stability_summary"]: "05_stability_summary",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["seed_disagreement"]: "05_seed_disagreement",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["ablation_plan"]: "05_ablation_plan",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["ablation_diagnostics"]: "05_ablation_diagnostics",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["statistical_uncertainty"]: "05_statistical_uncertainty",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["risk_coverage"]: "05_risk_coverage",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["review_guardrails"]: "05_review_guardrails",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["track_scoring_flags"]: "05_track_review",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["pure_test_candidates"]: "05_pure_test_candidates",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["threshold_sensitivity_plan"]: "05_threshold_sensitivity_plan",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["threshold_sensitivity_metrics"]: "05_threshold_sensitivity_metrics",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["threshold_sensitivity_decisions"]: "05_threshold_sensitivity_decisions",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["threshold_stability"]: "05_threshold_stability",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["source_image_influence"]: "05_source_image_influence",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["fold_sensitivity_plan"]: "05_fold_sensitivity_plan",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["fold_sensitivity_assignments"]: "05_fold_sensitivity_assignments",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["fold_sensitivity_thresholds"]: "05_fold_sensitivity_thresholds",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["fold_sensitivity_metrics"]: "05_fold_sensitivity_metrics",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["fold_sensitivity_decisions"]: "05_fold_sensitivity_decisions",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["fold_sensitivity_technical_events"]: "05_fold_sensitivity_technical_events",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["pareto_robustness_replicates"]: "05_pareto_robustness_replicates",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["pareto_robustness_summary"]: "05_pareto_robustness_summary",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["pareto_robustness_audit"]: "05_pareto_robustness_audit",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["spatial_sensitivity_plan"]: "05_spatial_sensitivity_plan",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["spatial_sensitivity_metrics"]: "05_spatial_sensitivity_metrics",
    SIMCA_ROBUSTNESS_OUTPUT_FILENAMES["ablation_coverage"]: "05_ablation_coverage",
}

# The active 00-05 workflow does not require dynamically named result-table
# suffixes. The empty mapping is explicit so the table utility has one central
# resolution contract and does not carry legacy suffix definitions locally.
PIPELINE_TABLE_KIND_BY_FILE_SUFFIX = {}

# Backward-compatible names for callers that already import the SIMCA_TABLE_*
# registry from experiment_config. The source of truth remains the single
# PIPELINE_TABLE_CONTRACTS mapping above.
SIMCA_TABLE_CONTRACTS = PIPELINE_TABLE_CONTRACTS
SIMCA_TABLE_COLUMNS = {
    name: tuple(contract["columns"])
    for name, contract in PIPELINE_TABLE_CONTRACTS.items()
}
SIMCA_TABLE_KEY_COLUMNS = {
    name: tuple(contract["key_columns"])
    for name, contract in PIPELINE_TABLE_CONTRACTS.items()
}
SIMCA_TABLE_KIND_BY_FILE_NAME = PIPELINE_TABLE_KIND_BY_FILE_NAME
SIMCA_TABLE_KIND_BY_FILE_SUFFIX = PIPELINE_TABLE_KIND_BY_FILE_SUFFIX

# Curated scientific settings serialized by the protocol freezer.
#
# Include parameters that can alter:
# - the analysed data population,
# - scientific representations,
# - candidate generation,
# - eligibility/admissibility,
# - model-selection objectives,
# - calibration/decision rules,
# - inferential conclusions.
#
# Exclude runtime, display, persistence, output-schema and audit-format
# parameters that cannot change a scientific decision.

PROTOCOL_CONFIGURATION_KEYS = (
    # ------------------------------------------------------------------
    # Protocol identity and registration
    # ------------------------------------------------------------------
    "PROTOCOL_VERSION",
    "RESULTS_SCHEMA_VERSION",
    "PROTOCOL_STATUS",
    "PROTOCOL_FREEZE_DATE",
    "PROTOCOL_REGISTRATION_MODE",
    "PROTOCOL_PRIOR_RESULTS_STATUS",
    "PROTOCOL_TEST_BLINDING_CLAIM",
    "PROTOCOL_AMENDMENT_JUSTIFICATION",
    "PROTOCOL_AMENDMENT_POLICY",

    # ------------------------------------------------------------------
    # Scientific identity and data roles
    # ------------------------------------------------------------------
    "TARGET_CLASS",
    "NON_TARGET_LABEL",
    "REFERENCE_CLASSES",
    "PROTOCOL_CALIBRATION_BATCHES",
    "PROTOCOL_VALIDATION_BATCHES",
    "PROTOCOL_TEST_BATCHES",

    # ------------------------------------------------------------------
    # Shared stochastic / grouping rules actually used scientifically
    # ------------------------------------------------------------------
    "CV_GROUP_COL",
    "RANDOM_STATE",
    "REPLACE_BALANCED_PIXELS",
    "M_BALANCED_PIXELS",
    "BALANCED_PIXEL_STRATEGIES",

    # ------------------------------------------------------------------
    # Spectral acquisition and database construction
    # ------------------------------------------------------------------
    "DEFAULT_WAVELENGTH_MODE",
    "SPECTRAL_START_NM",
    "SPECTRAL_END_NM",
    "N_BANDS_RAW",
    "N_REMOVE_START",
    "N_STOP_END",
    "TERMINAL_BAND_QC_POLICY",
    "SPECTRAL_PIXEL_VALIDITY_POLICY",
    "USE_WAVELENGTH_WINDOW",
    "WAVELENGTH_WINDOW_MIN_NM",
    "WAVELENGTH_WINDOW_MAX_NM",
    "DATA_MODE",
    "DATABASE_SKIP_UNKNOWN",
    "SEGMENTATION_OVERRIDE_RELATIVE_DIR",
    "SEGMENTATION_KWARGS",

    # ------------------------------------------------------------------
    # Notebook 01 - quality control
    # ------------------------------------------------------------------
    "QC_POLICY",
    "QC_SPECTRAL_GROUP_COLUMNS",
    "QC_SPECTRAL_OUTLIER_DISTANCE_THRESHOLD",
    "QC_ZERO_VARIANCE_EPSILON",
    "QC_REVIEW_ALLOWED_DECISIONS",
    "QC_REVIEW_REQUIRED_STATUS",

    # ------------------------------------------------------------------
    # Notebook 01B - spatial ground truth
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Global SIMCA representation / track definitions
    # ------------------------------------------------------------------
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
    "SIMCA_MODEL_ID_COLUMNS",
    "SIMCA_FIT_ID_COLUMNS",
    "SIMCA_PROJECTION_ID_COLUMNS",

    # ------------------------------------------------------------------
    # Notebook 02 - matrix construction and balanced sampling
    # ------------------------------------------------------------------
    "BALANCED_SAMPLING_M_VALUES",
    "BALANCED_SAMPLING_UNDER_M_POLICY",
    "BALANCED_SAMPLING_SEEDS",
    "BALANCED_SAMPLING_MIN_ELIGIBLE_RATE",
    "BALANCED_SAMPLING_STUDY_M_VALUES",

    # ------------------------------------------------------------------
    # Notebook 02 - preprocessing
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Notebook 03 - PCA exploration and selection
    # ------------------------------------------------------------------
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

    # Stability
    "PCA_STABILITY_SEEDS",
    "PCA_STABILITY_REFERENCE_SEED",
    "PCA_STABILITY_N_SPLITS",
    "PCA_STABILITY_N_BOOTSTRAP",
    "PCA_STABILITY_N_COMPONENTS",
    "PCA_STABILITY_GROUP_COL",
    "PCA_STABILITY_BOOTSTRAP_GROUP_COL",

    # Scientific PCA selection contract
    "MAX_PCA_PREPROCESSINGS_PER_FAMILY",
    "PCA_SELECTION_EXPECTED_FAMILIES",
    "PCA_SELECTION_PROFILES",
    "PCA_TECHNICAL_FLAG_COLUMNS",
    "PCA_SELECTION_STRICT_VARIANT_COVERAGE",

    # Human-review decision contract
    "PCA_ARTIFACT_REVIEW_REQUIRED_STATUS",
    "PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS",

    # ------------------------------------------------------------------
    # Notebook 03B - internal calibration
    # ------------------------------------------------------------------
    "INTERNAL_CALIBRATION_BATCHES",
    "INTERNAL_CALIBRATION_FORBIDDEN_BATCHES",
    "INTERNAL_CALIBRATION_GROUP_COL",
    "INTERNAL_CALIBRATION_LABEL_COL",
    "INTERNAL_CALIBRATION_BATCH_COL",
    "INTERNAL_CALIBRATION_OBJECT_SIZE_COL",
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
    "INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY",
    "INTERNAL_CALIBRATION_THRESHOLD_CONSTRAINTS",
    "INTERNAL_CALIBRATION_THRESHOLD_OVERRIDES",
    "INTERNAL_CALIBRATION_ACCURACY_CONTEXTS",
    "INTERNAL_CALIBRATION_THRESHOLD_PRIORITY",
    "INTERNAL_CALIBRATION_COMPLEXITY_SELECTION",
    "INTERNAL_CALIBRATION_PARETO_OBJECTIVES",
    "INTERNAL_CALIBRATION_ALLOWED_UNSUPPORTED_TRACK_IDS",
    "INTERNAL_CALIBRATION_CONSTRAINT_PROFILE_ID",
    "INTERNAL_CALIBRATION_SELECTION_PARENT_PROFILE_ID",
    "INTERNAL_CALIBRATION_SELECTION_PROFILE_ID",
    "INTERNAL_CALIBRATION_SELECTION_AMENDMENT_REASON",
    "INTERNAL_CALIBRATION_FN_PLATEAU_TOLERANCE",
    "INTERNAL_CALIBRATION_FP_PLATEAU_TOLERANCE",
    "INTERNAL_CALIBRATION_THRESHOLD_TIEBREAK",
    "INTERNAL_CALIBRATION_MODEL_PRIORITY",
    "INTERNAL_CALIBRATION_THRESHOLD_CANDIDATE_CACHE_FILENAME",
    "INTERNAL_CALIBRATION_REUSE_THRESHOLD_CANDIDATE_CACHE",
    "INTERNAL_CALIBRATION_REBUILD_THRESHOLD_CANDIDATE_CACHE",

    # ------------------------------------------------------------------
    # Notebook 03C - projection-domain audit
    # ------------------------------------------------------------------
    "PROJECTION_DOMAIN_AUDIT_RULE_VERSION",
    "PROJECTION_DOMAIN_AUDIT_ALLOWED_BATCHES",
    "PROJECTION_DOMAIN_AUDIT_FORBIDDEN_BATCHES",
    "PROJECTION_DOMAIN_BORDER_WIDTH",
    "PROJECTION_DOMAIN_MIN_STRATUM_N",
    "PROJECTION_DOMAIN_DIAGNOSTIC_DIMENSIONS",
    "PROJECTION_DOMAIN_ELIGIBILITY_DIMENSIONS",
    "PROJECTION_DOMAIN_ELIGIBILITY_THRESHOLDS",

    # ------------------------------------------------------------------
    # Notebook 03C - spatial calibration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Notebook 04B - preregistered search-efficiency experiment
    # ------------------------------------------------------------------
    "SIMCA_OPTUNA_PURPOSE",
    "SIMCA_OPTUNA_N_TRIALS_PER_TRACK",
    "SIMCA_OPTUNA_N_STARTUP_TRIALS",
    "SIMCA_OPTUNA_RANDOM_STATE",

    # ------------------------------------------------------------------
    # Primary inference and model-selection policy
    # ------------------------------------------------------------------
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
