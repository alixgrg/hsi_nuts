"""
Public API for the src package.

This file intentionally uses lazy imports:
- `import src` stays lightweight;
- old notebooks using `from src import ...` still work;
- broken optional modules do not break the whole package at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__version__ = "0.2.1"


_PUBLIC_API: dict[str, tuple[str, str]] = {
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    "load_mat_file": ("src.io.dataload", "load_mat_file"),
    "load_nir_uco_h5": ("src.io.database_h5", "load_nir_uco_h5"),
    "save_nir_uco_h5": ("src.io.database_h5", "save_nir_uco_h5"),
    "validate_nir_uco_h5": ("src.io.database_h5", "validate_nir_uco_h5"),
    "database_content_hash": (
        "src.io.database_h5",
        "database_content_hash",
    ),
    "build_database_manifest": (
        "src.io.database_h5",
        "build_database_manifest",
    ),

    # ------------------------------------------------------------------
    # Data / database / segmentation
    # ------------------------------------------------------------------
    "NIR_UCO_NAME_CONFIG": ("src.data.database", "NIR_UCO_NAME_CONFIG"),
    "make_reference_image": ("src.data.segmentation", "make_reference_image"),
    "make_binary_mask": ("src.data.segmentation", "make_binary_mask"),
    "clean_mask": ("src.data.segmentation", "clean_mask"),
    "label_objects_with_watershed": (
        "src.data.segmentation",
        "label_objects_with_watershed",
    ),
    "segment_objects": ("src.data.segmentation", "segment_objects"),
    "parse_image_key": ("src.data.database", "parse_image_key"),
    "infer_split_from_metadata": (
        "src.data.database",
        "infer_split_from_metadata",
    ),
    "infer_object_nut_type_from_metadata": (
        "src.data.database",
        "infer_object_nut_type_from_metadata",
    ),
    "segmentation_metadata": (
        "src.data.database",
        "segmentation_metadata",
    ),
    "preprocess_nir_uco_cube": ("src.data.database", "preprocess_nir_uco_cube"),
    "extract_objects_from_labeled_image": (
        "src.data.database",
        "extract_objects_from_labeled_image",
    ),
    "build_minimal_nir_uco_object_database": (
        "src.data.database",
        "build_minimal_nir_uco_object_database",
    ),
    "is_hyperspectral_cube": ("src.data.database", "is_hyperspectral_cube"),
    "detect_known_image_keys": ("src.data.database", "detect_known_image_keys"),
    "resolve_selected_keys": ("src.data.database", "resolve_selected_keys"),
    "build_raw_image_manifest": (
        "src.data.database",
        "build_raw_image_manifest",
    ),
    "validate_raw_image_manifest": (
        "src.data.database",
        "validate_raw_image_manifest",
    ),
    "validate_extracted_object": (
        "src.data.database",
        "validate_extracted_object",
    ),

    # ------------------------------------------------------------------
    # Decision labels
    # ------------------------------------------------------------------
    "DEFAULT_TARGET_CLASS": ("src.decision.labels", "DEFAULT_TARGET_CLASS"),
    "DEFAULT_NON_TARGET_LABEL": ("src.decision.labels", "DEFAULT_NON_TARGET_LABEL"),
    "UNCERTAIN_LABEL": ("src.decision.labels", "UNCERTAIN_LABEL"),
    "predicted_col": ("src.decision.labels", "predicted_col"),
    "true_col": ("src.decision.labels", "true_col"),
    "pixel_ratio_col": ("src.decision.labels", "pixel_ratio_col"),
    "true_pixel_ratio_col": ("src.decision.labels", "true_pixel_ratio_col"),
    "true_pixel_ratio_total_col": (
        "src.decision.labels",
        "true_pixel_ratio_total_col",
    ),
    "n_predicted_pixels_col": (
        "src.decision.labels",
        "n_predicted_pixels_col",
    ),

    # ------------------------------------------------------------------
    # Decision metrics
    # ------------------------------------------------------------------
    "binary_detection_metrics": (
        "src.decision.metrics",
        "binary_detection_metrics",
    ),
    "metrics_by_group": ("src.decision.metrics", "metrics_by_group"),
    "add_binary_confusion_case": (
        "src.decision.metrics",
        "add_binary_confusion_case",
    ),
    "summarize_pixel_errors_by_image": (
        "src.decision.metrics",
        "summarize_pixel_errors_by_image",
    ),
    "summarize_object_errors_by_image": (
        "src.decision.metrics",
        "summarize_object_errors_by_image",
    ),
    "binary_mask_agreement": (
        "src.decision.metrics",
        "binary_mask_agreement",
    ),
    "component_agreement": (
        "src.decision.metrics",
        "component_agreement",
    ),

    # ------------------------------------------------------------------
    # Decision truth
    # ------------------------------------------------------------------
    "expected_position_key_for_mixture": (
        "src.decision.truth",
        "expected_position_key_for_mixture",
    ),
    "union_object_masks": ("src.decision.truth", "union_object_masks"),
    "target_truth_map_for_image": (
        "src.decision.truth",
        "target_truth_map_for_image",
    ),
    "peanut_truth_map_for_image": (
        "src.decision.truth",
        "peanut_truth_map_for_image",
    ),
    "add_pixel_truth_labels": (
        "src.decision.truth",
        "add_pixel_truth_labels",
    ),
    "TruthResult": ("src.decision.truth", "TruthResult"),
    "select_annotation_subset": (
        "src.decision.truth",
        "select_annotation_subset",
    ),
    "select_double_annotation_images": (
        "src.decision.truth",
        "select_double_annotation_images",
    ),
    "validate_reference_mask": (
        "src.decision.truth",
        "validate_reference_mask",
    ),
    "validate_reference_annotation": (
        "src.decision.truth",
        "validate_reference_annotation",
    ),
    "build_spatial_ground_truth_manifest": (
        "src.decision.truth",
        "build_spatial_ground_truth_manifest",
    ),
    "extract_reference_components": (
        "src.decision.truth",
        "extract_reference_components",
    ),
    "resolve_truth_for_image": (
        "src.decision.truth",
        "resolve_truth_for_image",
    ),
    "build_annotation_agreement_table": (
        "src.decision.truth",
        "build_annotation_agreement_table",
    ),
    "build_spatial_ground_truth_lock": (
        "src.decision.truth",
        "build_spatial_ground_truth_lock",
    ),
    "verify_spatial_ground_truth_lock": (
        "src.decision.truth",
        "verify_spatial_ground_truth_lock",
    ),

    # ------------------------------------------------------------------
    # Decision aggregation
    # ------------------------------------------------------------------
    "add_object_metadata": (
        "src.decision.aggregation",
        "add_object_metadata",
    ),
    "aggregate_pixel_predictions_to_objects": (
        "src.decision.aggregation",
        "aggregate_pixel_predictions_to_objects",
    ),
    "object_threshold_grid": (
        "src.decision.aggregation",
        "object_threshold_grid",
    ),

    # ------------------------------------------------------------------
    # Border-aware decision
    # ------------------------------------------------------------------
    "add_border_flags_to_pixel_df": (
        "src.decision.border",
        "add_border_flags_to_pixel_df",
    ),
    "aggregate_pixel_predictions_to_objects_core": (
        "src.decision.border",
        "aggregate_pixel_predictions_to_objects_core",
    ),
    "border_width_object_threshold_grid": (
        "src.decision.border",
        "border_width_object_threshold_grid",
    ),
    "summarize_pixel_errors_by_border_zone": (
        "src.decision.border",
        "summarize_pixel_errors_by_border_zone",
    ),

    # ------------------------------------------------------------------
    # Decision maps
    # ------------------------------------------------------------------
    "make_pixel_error_map": ("src.decision.maps", "make_pixel_error_map"),
    "make_pixel_prediction_map": (
        "src.decision.maps",
        "make_pixel_prediction_map",
    ),
    "make_object_error_map": ("src.decision.maps", "make_object_error_map"),
    "make_object_fp_fn_map": ("src.decision.maps", "make_object_fp_fn_map"),

    # ------------------------------------------------------------------
    # Three-way / uncertainty
    # ------------------------------------------------------------------
    "add_three_way_object_decision": (
        "src.decision.uncertainty",
        "add_three_way_object_decision",
    ),
    "summarize_three_way_decision": (
        "src.decision.uncertainty",
        "summarize_three_way_decision",
    ),
    "evaluate_three_way_object_decision": (
        "src.decision.uncertainty",
        "evaluate_three_way_object_decision",
    ),
    "three_way_object_threshold_grid": (
        "src.decision.uncertainty",
        "three_way_object_threshold_grid",
    ),
    "three_way_object_threshold_grid_by_group": (
        "src.decision.uncertainty",
        "three_way_object_threshold_grid_by_group",
    ),

    # ------------------------------------------------------------------
    # Matrices
    # ------------------------------------------------------------------
    "MatrixSpec": ("src.matrices.matrix_registry", "MatrixSpec"),
    "MatrixOutput": ("src.matrices.matrix_registry", "MatrixOutput"),
    "build_matrix": ("src.matrices.matrix_registry", "build_matrix"),
    "build_matrix_output": (
        "src.matrices.matrix_registry",
        "build_matrix_output",
    ),
    "get_matrix_spec": ("src.matrices.matrix_registry", "get_matrix_spec"),
    "matrix_method_to_args": (
        "src.matrices.matrix_registry",
        "matrix_method_to_args",
    ),
    "available_matrix_methods": (
        "src.matrices.matrix_registry",
        "available_matrix_methods",
    ),
    "object_db_to_matrix": ("src.matrices.redim_matrix", "object_db_to_matrix"),
    "select_balanced_pixel_indices": (
        "src.matrices.redim_matrix",
        "select_balanced_pixel_indices",
    ),

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    "PCAModel": ("src.models.pca", "PCAModel"),
    "pca_from_cov": ("src.models.pca", "pca_from_cov"),
    "SIMCAClassModel": ("src.models.simca", "SIMCAClassModel"),
    "SIMCAClassifier": ("src.models.simca", "SIMCAClassifier"),
    "BaseSIMCARule": ("src.models.simca_rules", "BaseSIMCARule"),
    "SimpleSIMCARule": ("src.models.simca_rules", "SimpleSIMCARule"),
    "AltSIMCARule": ("src.models.simca_rules", "AltSIMCARule"),
    "CombinedIndexSIMCARule": (
        "src.models.simca_rules",
        "CombinedIndexSIMCARule",
    ),
    "DataDrivenSIMCARule": ("src.models.simca_rules", "DataDrivenSIMCARule"),
    "make_simca_rule": ("src.models.simca_rules", "make_simca_rule"),
    "compute_rule_variant_stat_limit": (
        "src.models.simca_rules",
        "compute_rule_variant_stat_limit",
    ),
    "accept_rule_variant": (
        "src.models.simca_rules",
        "accept_rule_variant",
    ),

    # ------------------------------------------------------------------
    # Spectra / preprocessing / wavelength selection
    # ------------------------------------------------------------------
    "SpectralPreprocessor": (
        "src.spectra.preprocessing",
        "SpectralPreprocessor",
    ),
    "center_X": ("src.spectra.preprocessing", "center_X"),
    "snv": ("src.spectra.preprocessing", "snv"),
    "vector_normalize": ("src.spectra.preprocessing", "vector_normalize"),
    "msc_fit": ("src.spectra.preprocessing", "msc_fit"),
    "msc_transform": ("src.spectra.preprocessing", "msc_transform"),
    "savgol_derivative": ("src.spectra.preprocessing", "savgol_derivative"),
    "reflectance_to_absorbance": (
        "src.spectra.preprocessing",
        "reflectance_to_absorbance",
    ),
    "preprocessing_input_validity_report": (
        "src.spectra.preprocessing",
        "preprocessing_input_validity_report",
    ),
    "VALID_PREPROCESSING_STEPS": (
        "src.spectra.preprocessing_configs",
        "VALID_PREPROCESSING_STEPS",
    ),
    "PREPROCESSING_ALIASES": (
        "src.spectra.preprocessing_configs",
        "PREPROCESSING_ALIASES",
    ),
    "DEFAULT_PREPROCESSING_CONFIGS": (
        "src.spectra.preprocessing_configs",
        "DEFAULT_PREPROCESSING_CONFIGS",
    ),
    "SIMCA_SEARCH_PREPROCESSING_CONFIGS": (
        "src.spectra.preprocessing_configs",
        "SIMCA_SEARCH_PREPROCESSING_CONFIGS",
    ),
    "preprocessing_name_from_steps": (
        "src.spectra.preprocessing_configs",
        "preprocessing_name_from_steps",
    ),
    "validate_preprocessing_steps": (
        "src.spectra.preprocessing_configs",
        "validate_preprocessing_steps",
    ),
    "resolve_preprocessing_steps": (
        "src.spectra.preprocessing_configs",
        "resolve_preprocessing_steps",
    ),
    "normalize_preprocessing_configs": (
        "src.spectra.preprocessing_configs",
        "normalize_preprocessing_configs",
    ),
    "preprocessing_derivative": (
        "src.spectra.preprocessing_configs",
        "preprocessing_derivative",
    ),
    "wavelength_selection_summary": (
        "src.spectra.band_selection",
        "wavelength_selection_summary",
    ),
    "select_wavelength_range_from_database": (
        "src.spectra.band_selection",
        "select_wavelength_range_from_database",
    ),

    # ------------------------------------------------------------------
    # PCA workflow
    # ------------------------------------------------------------------
    "OBJECT_MATRIX_METHODS": ("src.workflows.pca", "OBJECT_MATRIX_METHODS"),
    "PIXEL_MATRIX_METHODS": ("src.workflows.pca", "PIXEL_MATRIX_METHODS"),
    "pca_matrix_family_from_method": (
        "src.workflows.pca",
        "pca_matrix_family_from_method",
    ),
    "pca_matrix_variant_from_method": (
        "src.workflows.pca",
        "pca_matrix_variant_from_method",
    ),
    "binary_class_separation_scores": (
        "src.workflows.pca",
        "binary_class_separation_scores",
    ),
    "mahalanobis_centroid_distance": (
        "src.workflows.pca",
        "mahalanobis_centroid_distance",
    ),
    "n_components_for_cumulative_variance": (
        "src.workflows.pca",
        "n_components_for_cumulative_variance",
    ),
    "trace_ratio_by_group": ("src.workflows.pca", "trace_ratio_by_group"),
    "pca_distance_summary": ("src.workflows.pca", "pca_distance_summary"),
    "train_projection_shift_by_label": (
        "src.workflows.pca",
        "train_projection_shift_by_label",
    ),
    "pixel_object_score_metrics": (
        "src.workflows.pca",
        "pixel_object_score_metrics",
    ),
    "compute_pca_summary_metrics": (
        "src.workflows.pca",
        "compute_pca_summary_metrics",
    ),
    "compare_pca_representations": (
        "src.workflows.pca",
        "compare_pca_representations",
    ),
    "build_pca_candidate_plan": (
        "src.workflows.pca",
        "build_pca_candidate_plan",
    ),
    "fit_pca_candidate": (
        "src.workflows.pca",
        "fit_pca_candidate",
    ),
    "subset_object_db_for_pca": (
        "src.workflows.pca",
        "subset_object_db_for_pca",
    ),
    "pca_component_variance_table": (
        "src.workflows.pca",
        "pca_component_variance_table",
    ),
    "compute_group_compactness": (
        "src.workflows.pca",
        "compute_group_compactness",
    ),
    "compute_group_centroid_displacements": (
        "src.workflows.pca",
        "compute_group_centroid_displacements",
    ),
    "compare_aligned_loadings": (
        "src.workflows.pca",
        "compare_aligned_loadings",
    ),
    "evaluate_pca_stability": (
        "src.workflows.pca",
        "evaluate_pca_stability",
    ),
    "summarize_pca_stability": (
        "src.workflows.pca",
        "summarize_pca_stability",
    ),
    "DEFAULT_PCA_SELECTION_CONFIG": (
        "src.workflows.pca_selection",
        "DEFAULT_PCA_SELECTION_CONFIG",
    ),
    "PCASelectionConfig": (
        "src.workflows.pca_selection",
        "PCASelectionConfig",
    ),
    "PCASelectionProfile": (
        "src.workflows.pca_selection",
        "PCASelectionProfile",
    ),
    "make_pca_selection_config": (
        "src.workflows.pca_selection",
        "make_pca_selection_config",
    ),
    "build_pca_artifact_review_table": (
        "src.workflows.pca_selection",
        "build_pca_artifact_review_table",
    ),
    "apply_pca_artifact_review_decisions": (
        "src.workflows.pca_selection",
        "apply_pca_artifact_review_decisions",
    ),
    "aggregate_pca_preprocessing_diagnostics": (
        "src.workflows.pca_selection",
        "aggregate_pca_preprocessing_diagnostics",
    ),
    "build_pca_run_fingerprint": (
        "src.workflows.pca_selection",
        "build_pca_run_fingerprint",
    ),
    "freeze_pca_shortlist": (
        "src.workflows.pca_selection",
        "freeze_pca_shortlist",
    ),
    "hash_pca_review_table": (
        "src.workflows.pca_selection",
        "hash_pca_review_table",
    ),
    "hash_pca_input_artifacts": (
        "src.workflows.pca_selection",
        "hash_pca_input_artifacts",
    ),
    "pca_input_artifact_paths": (
        "src.workflows.pca_selection",
        "pca_input_artifact_paths",
    ),
    "pca_input_fingerprint": (
        "src.workflows.pca_selection",
        "pca_input_fingerprint",
    ),
    "validate_pca_artifact_review": (
        "src.workflows.pca_selection",
        "validate_pca_artifact_review",
    ),
    "build_pca_selection_flow_tables": (
        "src.workflows.pca_selection",
        "build_pca_selection_flow_tables",
    ),
    "build_pca_selection_diagnostics": (
        "src.workflows.pca_selection",
        "build_pca_selection_diagnostics",
    ),
    "build_pca_scoring_diagnostics": (
        "src.workflows.pca_selection",
        "build_pca_scoring_diagnostics",
    ),
    "add_pca_relative_quality_flags": (
        "src.workflows.pca_selection",
        "add_pca_relative_quality_flags",
    ),
    "format_pca_selection_reason": (
        "src.workflows.pca_selection",
        "format_pca_selection_reason",
    ),
    "select_pca_preprocessing_shortlist": (
        "src.workflows.pca_selection",
        "select_pca_preprocessing_shortlist",
    ),
    "select_pca_pareto_front": (
        "src.workflows.pca_selection",
        "select_pca_pareto_front",
    ),
    "validate_pca_preprocessing_shortlist": (
        "src.workflows.pca_selection",
        "validate_pca_preprocessing_shortlist",
    ),
    "build_reference_object_table": (
        "src.workflows.simca_internal_calibration",
        "build_reference_object_table",
    ),
    "build_calibration_folds": (
        "src.workflows.simca_internal_calibration",
        "build_calibration_folds",
    ),
    "build_internal_calibration_configurations": (
        "src.workflows.simca_internal_calibration",
        "build_internal_calibration_configurations",
    ),
    "expand_projection_configurations": (
        "src.workflows.simca_internal_calibration",
        "expand_projection_configurations",
    ),
    "hash_internal_calibration_configuration": (
        "src.workflows.simca_internal_calibration",
        "hash_internal_calibration_configuration",
    ),
    "validate_simca_configuration": (
        "src.workflows.simca_internal_calibration",
        "validate_simca_configuration",
    ),
    "compute_train_only_rule_thresholds": (
        "src.workflows.simca_internal_calibration",
        "compute_train_only_rule_thresholds",
    ),
    "run_internal_calibration": (
        "src.workflows.simca_internal_calibration",
        "run_internal_calibration",
    ),
    "run_internal_calibration_8tracks": (
        "src.workflows.simca_internal_calibration",
        "run_internal_calibration_8tracks",
    ),
    "evaluate_internal_2way_tracks": (
        "src.workflows.simca_internal_calibration",
        "evaluate_internal_2way_tracks",
    ),
    "evaluate_crossfitted_three_way_thresholds": (
        "src.workflows.simca_internal_calibration",
        "evaluate_crossfitted_three_way_thresholds",
    ),
    "build_internal_calibrated_hyperparameters_8tracks": (
        "src.workflows.simca_internal_calibration",
        "build_internal_calibrated_hyperparameters_8tracks",
    ),
    "build_calibration_domain_8tracks": (
        "src.workflows.simca_internal_calibration",
        "build_calibration_domain_8tracks",
    ),
    "select_smallest_plateau_components": (
        "src.workflows.simca_internal_calibration",
        "select_smallest_plateau_components",
    ),
    "build_exact_oof_prediction_equivalence": (
        "src.workflows.simca_internal_calibration",
        "build_exact_oof_prediction_equivalence",
    ),
    "evaluate_internal_object_thresholds": (
        "src.workflows.simca_internal_calibration",
        "evaluate_internal_object_thresholds",
    ),
    "evaluate_internal_three_way_thresholds": (
        "src.workflows.simca_internal_calibration",
        "evaluate_internal_three_way_thresholds",
    ),
    "build_internal_calibrated_hyperparameters": (
        "src.workflows.simca_internal_calibration",
        "build_internal_calibrated_hyperparameters",
    ),
    "build_calibration_domain_from_03b": (
        "src.workflows.simca_internal_calibration",
        "build_calibration_domain_from_03b",
    ),
    "build_image_qc_table": (
        "src.workflows.quality_check",
        "build_image_qc_table",
    ),
    "build_image_qc_warnings": (
        "src.workflows.quality_check",
        "build_image_qc_warnings",
    ),
    "build_object_qc_table": (
        "src.workflows.quality_check",
        "build_object_qc_table",
    ),
    "build_object_qc_warnings": (
        "src.workflows.quality_check",
        "build_object_qc_warnings",
    ),
    "build_object_shape_check_tables": (
        "src.workflows.quality_check",
        "build_object_shape_check_tables",
    ),
    "build_qc_flags_table": (
        "src.workflows.quality_check",
        "build_qc_flags_table",
    ),
    "build_qc_alerts_table": (
        "src.workflows.quality_check",
        "build_qc_alerts_table",
    ),
    "add_robust_spectral_qc": (
        "src.workflows.quality_check",
        "add_robust_spectral_qc",
    ),
    "merge_existing_reviews_or_initialize": (
        "src.workflows.quality_check",
        "merge_existing_reviews_or_initialize",
    ),
    "validate_qc_review_closure": (
        "src.workflows.quality_check",
        "validate_qc_review_closure",
    ),
    "build_qc_visual_review_report": (
        "src.workflows.quality_check",
        "build_qc_visual_review_report",
    ),
    "build_qc_protocol": (
        "src.workflows.quality_check",
        "build_qc_protocol",
    ),
    "build_qc_exclusion_report": (
        "src.workflows.quality_check",
        "build_qc_exclusion_report",
    ),
    "build_segmentation_diagnostics_table": (
        "src.workflows.quality_check",
        "build_segmentation_diagnostics_table",
    ),
    "build_spectral_integrity_table": (
        "src.workflows.quality_check",
        "build_spectral_integrity_table",
    ),
    "check_missing_required_fields": (
        "src.workflows.quality_check",
        "check_missing_required_fields",
    ),
    "qc_requires_new_cycle": (
        "src.workflows.quality_check",
        "qc_requires_new_cycle",
    ),
    "build_matrix_coverage_table": (
        "src.workflows.matrix_preprocessing",
        "build_matrix_coverage_table",
    ),
    "build_protocol_manifest": (
        "src.workflows.protocol_split",
        "build_protocol_manifest",
    ),
    "build_grouped_folds": (
        "src.workflows.protocol_split",
        "build_grouped_folds",
    ),
    "build_split_diagnostics": (
        "src.workflows.protocol_split",
        "build_split_diagnostics",
    ),
    "assert_no_split_leakage": (
        "src.workflows.protocol_split",
        "assert_no_split_leakage",
    ),
    "eligible_object_ids": (
        "src.workflows.protocol_split",
        "eligible_object_ids",
    ),
    "build_wavelength_config": (
        "src.workflows.matrix_preprocessing",
        "build_wavelength_config",
    ),
    "assert_wavelength_lock": (
        "src.workflows.matrix_preprocessing",
        "assert_wavelength_lock",
    ),
    "evaluate_balanced_sampling_grid": (
        "src.workflows.matrix_preprocessing",
        "evaluate_balanced_sampling_grid",
    ),
    "evaluate_preprocessing_grid": (
        "src.workflows.matrix_preprocessing",
        "evaluate_preprocessing_grid",
    ),
    "summarize_matrix_output": (
        "src.workflows.matrix_preprocessing",
        "summarize_matrix_output",
    ),
    "summarize_preprocessing_output": (
        "src.workflows.matrix_preprocessing",
        "summarize_preprocessing_output",
    ),
    "validate_required_columns": (
        "src.workflows.matrix_preprocessing",
        "validate_required_columns",
    ),
    "build_inference_plan": (
        "src.protocol_governance",
        "build_inference_plan",
    ),
    "build_planned_contrasts": (
        "src.protocol_governance",
        "build_planned_contrasts",
    ),
    "build_protocol_configuration": (
        "src.protocol_governance",
        "build_protocol_configuration",
    ),
    "build_scientific_protocol_manifest": (
        "src.protocol_governance",
        "build_scientific_protocol_manifest",
    ),
    "freeze_protocol": (
        "src.protocol_governance",
        "freeze_protocol",
    ),
    "validate_protocol_contract": (
        "src.protocol_governance",
        "validate_protocol_contract",
    ),
    "verify_frozen_protocol": (
        "src.protocol_governance",
        "verify_frozen_protocol",
    ),

    # ------------------------------------------------------------------
    # SIMCA workflow
    # ------------------------------------------------------------------
    "uses_sg": ("src.workflows.simca", "uses_sg"),
    "valid_sg_parameter_pairs": (
        "src.workflows.simca",
        "valid_sg_parameter_pairs",
    ),
    "matrix_family_from_method": (
        "src.workflows.simca",
        "matrix_family_from_method",
    ),
    "balanced_strategy_grid_for_matrix": (
        "src.workflows.simca",
        "balanced_strategy_grid_for_matrix",
    ),
    "standard_grid_sort": ("src.workflows.simca", "standard_grid_sort"),
    "make_target_train_filters": (
        "src.workflows.simca",
        "make_target_train_filters",
    ),
    "fit_one_class_simca": ("src.workflows.simca", "fit_one_class_simca"),
    "fit_simca_bundle_from_matrix": (
        "src.workflows.simca",
        "fit_simca_bundle_from_matrix",
    ),
    "prepare_simca_projection": (
        "src.workflows.simca",
        "prepare_simca_projection",
    ),
    "project_simca_bundle": (
        "src.workflows.simca",
        "project_simca_bundle",
    ),
    "predict_pixels_with_simca": (
        "src.workflows.simca",
        "predict_pixels_with_simca",
    ),
    "run_single_simca_pixel_projection": (
        "src.workflows.simca",
        "run_single_simca_pixel_projection",
    ),
    "run_simca_pixel_projection_grid": (
        "src.workflows.simca",
        "run_simca_pixel_projection_grid",
    ),
    "calibrate_simca_thresholds_cv": (
        "src.workflows.simca",
        "calibrate_simca_thresholds_cv",
    ),
    "fit_final_simca_model": (
        "src.workflows.simca",
        "fit_final_simca_model",
    ),
    "project_pixels_with_rule_variants": (
        "src.workflows.simca",
        "project_pixels_with_rule_variants",
    ),
    "summarize_cv_calibration": (
        "src.workflows.simca",
        "summarize_cv_calibration",
    ),
    "run_simca_rule_variant_grid": (
        "src.workflows.simca",
        "run_simca_rule_variant_grid",
    ),
    "refit_best_grid_row": (
        "src.workflows.simca",
        "refit_best_grid_row",
    ),
    "refit_empirical_cv_rule_row": (
        "src.workflows.simca",
        "refit_empirical_cv_rule_row",
    ),
    "refit_selected_simca_row": (
        "src.workflows.simca",
        "refit_selected_simca_row",
    ),
    "refit_selected_simca_configs": (
        "src.workflows.simca",
        "refit_selected_simca_configs",
    ),

    # ------------------------------------------------------------------
    # SIMCA selection
    # ------------------------------------------------------------------
    "SIMCA_RULE_METADATA": (
        "src.workflows.simca_selection_utils",
        "SIMCA_RULE_METADATA",
    ),
    "detection_selection_score": (
        "src.workflows.simca_selection_utils",
        "detection_selection_score",
    ),
    "add_detection_selection_score": (
        "src.workflows.simca_selection_utils",
        "add_detection_selection_score",
    ),
    "sort_detection_selection": (
        "src.workflows.simca_selection_utils",
        "sort_detection_selection",
    ),
    "infer_model_family_from_rule_token": (
        "src.workflows.simca_selection_utils",
        "infer_model_family_from_rule_token",
    ),
    "normalize_simca_rule_columns": (
        "src.workflows.simca_selection_utils",
        "normalize_simca_rule_columns",
    ),
    "fill_selected_config_defaults": (
        "src.workflows.simca_selection_utils",
        "fill_selected_config_defaults",
    ),
    "ensure_candidate_columns": (
        "src.workflows.simca_selection_utils",
        "ensure_candidate_columns",
    ),
    "select_top_models": (
        "src.workflows.simca_selection_utils",
        "select_top_models",
    ),
    "add_reference_selection_scores": (
        "src.workflows.simca_selection_utils",
        "add_reference_selection_scores",
    ),
    "select_top_by_score": (
        "src.workflows.simca_selection_utils",
        "select_top_by_score",
    ),
    "pareto_front": (
        "src.workflows.simca_selection_utils",
        "pareto_front",
    ),
    "summarize_parameter_tendencies": (
        "src.workflows.simca_selection_utils",
        "summarize_parameter_tendencies",
    ),
    "infer_matrix_family": (
        "src.workflows.simca_candidates",
        "infer_matrix_family",
    ),
    "selection_track_from_parts": (
        "src.workflows.simca_candidates",
        "selection_track_from_parts",
    ),
    "add_selection_track": (
        "src.workflows.simca_candidates",
        "add_selection_track",
    ),
    "candidate_identity_payload": (
        "src.workflows.simca_candidates",
        "candidate_identity_payload",
    ),
    "simca_candidate_key": (
        "src.workflows.simca_candidates",
        "simca_candidate_key",
    ),
    "add_simca_candidate_ids": (
        "src.workflows.simca_candidates",
        "add_simca_candidate_ids",
    ),
    "deduplicate_simca_candidates": (
        "src.workflows.simca_candidates",
        "deduplicate_simca_candidates",
    ),
    "deduplicate_metric_equivalent_simca_candidates": (
        "src.workflows.simca_candidates",
        "deduplicate_metric_equivalent_simca_candidates",
    ),
    "deduplicate_simca_refit_configs": (
        "src.workflows.simca_candidates",
        "deduplicate_simca_refit_configs",
    ),
    "audit_simca_candidate_technical_status": (
        "src.workflows.simca_candidates",
        "audit_simca_candidate_technical_status",
    ),
    "build_simca_output_signatures": (
        "src.workflows.simca_candidates",
        "build_simca_output_signatures",
    ),
    "normalize_simca_candidate_columns": (
        "src.workflows.simca_candidates",
        "normalize_simca_candidate_columns",
    ),
    "build_pca_preprocessing_configs_by_matrix_family": (
        "src.workflows.simca_candidates",
        "build_pca_preprocessing_configs_by_matrix_family",
    ),
    "allowed_pca_preprocessing_pairs": (
        "src.workflows.simca_candidates",
        "allowed_pca_preprocessing_pairs",
    ),
    "filter_simca_candidates_by_pca_preprocessing": (
        "src.workflows.simca_candidates",
        "filter_simca_candidates_by_pca_preprocessing",
    ),
    "validate_simca_candidates_match_pca_preprocessing": (
        "src.workflows.simca_candidates",
        "validate_simca_candidates_match_pca_preprocessing",
    ),
    "validate_simca_candidate_contract": (
        "src.workflows.simca_candidates",
        "validate_simca_candidate_contract",
    ),
    "validate_simca_evaluation_contract": (
        "src.workflows.simca_candidates",
        "validate_simca_evaluation_contract",
    ),
    "validate_simca_selection_tracks": (
        "src.workflows.simca_candidates",
        "validate_simca_selection_tracks",
    ),
    "validate_simca_table_columns": (
        "src.workflows.simca_candidates",
        "validate_simca_table_columns",
    ),

    # ------------------------------------------------------------------
    # SIMCA robustness diagnostics
    # ------------------------------------------------------------------
    "validate_no_pure_test_inputs": (
        "src.workflows.simca_robustness",
        "validate_no_pure_test_inputs",
    ),
    "validate_simca_robustness_inputs": (
        "src.workflows.simca_robustness",
        "validate_simca_robustness_inputs",
    ),
    "select_track_primary_or_available_metrics": (
        "src.workflows.simca_robustness",
        "select_track_primary_or_available_metrics",
    ),
    "add_simca_robustness_scores": (
        "src.workflows.simca_robustness",
        "add_simca_robustness_scores",
    ),
    "build_pareto_diagnostics": (
        "src.workflows.simca_robustness",
        "build_pareto_diagnostics",
    ),
    "build_ablation_diagnostics": (
        "src.workflows.simca_robustness",
        "build_ablation_diagnostics",
    ),
    "build_random_state_stability_panel": (
        "src.workflows.simca_robustness",
        "build_random_state_stability_panel",
    ),
    "summarize_random_state_stability_metrics": (
        "src.workflows.simca_robustness",
        "summarize_random_state_stability_metrics",
    ),
    "build_border_core_skip_table": (
        "src.workflows.simca_robustness",
        "build_border_core_skip_table",
    ),
    "build_border_core_diagnostics": (
        "src.workflows.simca_robustness",
        "build_border_core_diagnostics",
    ),
    "build_duplicated_candidate_review": (
        "src.workflows.simca_robustness",
        "build_duplicated_candidate_review",
    ),
    "summarize_duplicated_candidate_review": (
        "src.workflows.simca_robustness",
        "summarize_duplicated_candidate_review",
    ),
    "build_track_scoring_table": (
        "src.workflows.simca_robustness",
        "build_track_scoring_table",
    ),

    # ------------------------------------------------------------------
    # SIMCA table schemas
    # ------------------------------------------------------------------
    "SIMCA_TABLE_COLUMNS": (
        "src.workflows.simca_tables",
        "SIMCA_TABLE_COLUMNS",
    ),
    "TABLE_KIND_BY_FILE_NAME": (
        "src.workflows.simca_tables",
        "TABLE_KIND_BY_FILE_NAME",
    ),
    "TABLE_KIND_BY_FILE_SUFFIX": (
        "src.workflows.simca_tables",
        "TABLE_KIND_BY_FILE_SUFFIX",
    ),
    "canonicalize_simca_columns": (
        "src.workflows.simca_tables",
        "canonicalize_simca_columns",
    ),
    "resolve_merge_suffix_columns": (
        "src.workflows.simca_tables",
        "resolve_merge_suffix_columns",
    ),
    "drop_all_na_columns": (
        "src.workflows.simca_tables",
        "drop_all_na_columns",
    ),
    "compact_simca_table": (
        "src.workflows.simca_tables",
        "compact_simca_table",
    ),
    "compact_simca_table_for_path": (
        "src.workflows.simca_tables",
        "compact_simca_table_for_path",
    ),
    "schema_diagnostics": (
        "src.workflows.simca_tables",
        "schema_diagnostics",
    ),
    "build_schema_manifest": (
        "src.workflows.simca_tables",
        "build_schema_manifest",
    ),

    # Backward-compatible aliases for older notebooks.
    # Prefer add_detection_selection_score / sort_detection_selection in new notebooks.
    "add_simca_selection_score": (
        "src.workflows.simca_selection_utils",
        "add_detection_selection_score",
    ),
    "sort_simca_selection": (
        "src.workflows.simca_selection_utils",
        "sort_detection_selection",
    ),

    # ------------------------------------------------------------------
    # Optuna workflow
    # ------------------------------------------------------------------
    "make_simca_optuna_objective": (
        "src.workflows.simca_optuna",
        "make_simca_optuna_objective",
    ),
    "run_optuna_simca_pixel_optimization": (
        "src.workflows.simca_optuna",
        "run_optuna_simca_pixel_optimization",
    ),
    "optuna_trials_dataframe": (
        "src.workflows.simca_optuna",
        "optuna_trials_dataframe",
    ),
    "optuna_trials_to_candidate_configs": (
        "src.workflows.simca_optuna",
        "optuna_trials_to_candidate_configs",
    ),
    "best_completed_trial_row": (
        "src.workflows.simca_optuna",
        "best_completed_trial_row",
    ),
    "refit_optuna_best_trial": (
        "src.workflows.simca_optuna",
        "refit_optuna_best_trial",
    ),
    "close_optuna_study": (
        "src.workflows.simca_optuna",
        "close_optuna_study",
    ),
    "build_optuna_search_plan_hash": (
        "src.workflows.simca_optuna",
        "build_optuna_search_plan_hash",
    ),
    "build_optuna_study_registry": (
        "src.workflows.simca_optuna",
        "build_optuna_study_registry",
    ),
    "build_optuna_pareto_candidates": (
        "src.workflows.simca_optuna",
        "build_optuna_pareto_candidates",
    ),
    "build_optuna_search_efficiency_audit": (
        "src.workflows.simca_optuna",
        "build_optuna_search_efficiency_audit",
    ),
    "build_preregistered_ablation_plan": (
        "src.workflows.simca_optuna",
        "build_preregistered_ablation_plan",
    ),

    # ------------------------------------------------------------------
    # Visualization: common
    # ------------------------------------------------------------------
    "show_or_return": ("src.visualization.common", "show_or_return"),
    "make_customdata": ("src.visualization.common", "make_customdata"),
    "ordered_unique": ("src.visualization.common", "ordered_unique"),
    "make_dynamic_color_map": (
        "src.visualization.common",
        "make_dynamic_color_map",
    ),
    "background_image": ("src.visualization.common", "background_image"),
    "validate_columns": ("src.visualization.common", "validate_columns"),

    # ------------------------------------------------------------------
    # Visualization: generic
    # ------------------------------------------------------------------
    "plot_bar_values": ("src.visualization.plot_generic", "plot_bar_values"),
    "plot_counts_by_group": (
        "src.visualization.plot_generic",
        "plot_counts_by_group",
    ),
    "plot_lines_from_dataframe": (
        "src.visualization.plot_generic",
        "plot_lines_from_dataframe",
    ),
    "plot_distribution_with_curve": (
        "src.visualization.plot_generic",
        "plot_distribution_with_curve",
    ),

    # ------------------------------------------------------------------
    # Visualization: images
    # ------------------------------------------------------------------
    "plot_hypercube_band_slider": (
        "src.visualization.plot_images",
        "plot_hypercube_band_slider",
    ),
    "plot_image2d": ("src.visualization.plot_images", "plot_image2d"),
    "plot_image_overlay": (
        "src.visualization.plot_images",
        "plot_image_overlay",
    ),
    "plot_label_overlay_from_image_db": (
        "src.visualization.plot_images",
        "plot_label_overlay_from_image_db",
    ),

    # ------------------------------------------------------------------
    # Visualization: spectra
    # ------------------------------------------------------------------
    "mean_spectrum_from_cube": (
        "src.visualization.plot_spectra",
        "mean_spectrum_from_cube",
    ),
    "extract_spectral_matrix": (
        "src.visualization.plot_spectra",
        "extract_spectral_matrix",
    ),
    "plot_spectra": ("src.visualization.plot_spectra", "plot_spectra"),
    "plot_spectral_distribution": (
        "src.visualization.plot_spectra",
        "plot_spectral_distribution",
    ),
    "plot_object_spectra": (
        "src.visualization.plot_spectra",
        "plot_object_spectra",
    ),

    # ------------------------------------------------------------------
    # Visualization: objects
    # ------------------------------------------------------------------
    "plot_object_view": (
        "src.visualization.plot_objects",
        "plot_object_view",
    ),
    "plot_object_grid": (
        "src.visualization.plot_objects",
        "plot_object_grid",
    ),
    "plot_object_areas": (
        "src.visualization.plot_objects",
        "plot_object_areas",
    ),

    # ------------------------------------------------------------------
    # Visualization: scores / PCA / diagnostics
    # ------------------------------------------------------------------
    "plot_scores": ("src.visualization.plot_scores", "plot_scores"),
    "build_scores_dataframe": (
        "src.visualization.plot_scores",
        "build_scores_dataframe",
    ),
    "sample_scores_dataframe": (
        "src.visualization.plot_scores",
        "sample_scores_dataframe",
    ),
    "plot_scores_density": (
        "src.visualization.plot_scores",
        "plot_scores_density",
    ),
    "plot_scores_distribution": (
        "src.visualization.plot_scores",
        "plot_scores_distribution",
    ),
    "summarize_scores_by_object": (
        "src.visualization.plot_scores",
        "summarize_scores_by_object",
    ),
    "plot_object_score_summary": (
        "src.visualization.plot_scores",
        "plot_object_score_summary",
    ),
    "plot_metric_by_index": (
        "src.visualization.plot_diagnostics",
        "plot_metric_by_index",
    ),
    "plot_xy_diagnostic": (
        "src.visualization.plot_diagnostics",
        "plot_xy_diagnostic",
    ),
    "plot_metric_heatmap": (
        "src.visualization.plot_diagnostics",
        "plot_metric_heatmap",
    ),
    "plot_explained_variance": (
        "src.visualization.plot_pca",
        "plot_explained_variance",
    ),
    "plot_loadings": ("src.visualization.plot_pca", "plot_loadings"),
    "plot_biplot": ("src.visualization.plot_pca", "plot_biplot"),
    "plot_pca_metric_t2": (
        "src.visualization.plot_pca",
        "plot_pca_metric_t2",
    ),
    "plot_pca_metric_q": (
        "src.visualization.plot_pca",
        "plot_pca_metric_q",
    ),
    "plot_pca_diagnostic": (
        "src.visualization.plot_pca",
        "plot_pca_diagnostic",
    ),
    "plot_pca_metric_heatmap": (
        "src.visualization.plot_pca",
        "plot_pca_metric_heatmap",
    ),
    "plot_pca_metric_tradeoff": (
        "src.visualization.plot_pca",
        "plot_pca_metric_tradeoff",
    ),
    "plot_pca_metric_ranking": (
        "src.visualization.plot_pca",
        "plot_pca_metric_ranking",
    ),

    # ------------------------------------------------------------------
    # Visualization: SIMCA / decisions
    # ------------------------------------------------------------------
    "plot_simca_distance": (
        "src.visualization.plot_simca",
        "plot_simca_distance",
    ),
    "plot_simca_rule_metric": (
        "src.visualization.plot_simca",
        "plot_simca_rule_metric",
    ),
    "plot_decision_counts": (
        "src.visualization.plot_simca",
        "plot_decision_counts",
    ),
    "plot_object_decision_map": (
        "src.visualization.plot_decision",
        "plot_object_decision_map",
    ),
    "plot_object_error_overlay": (
        "src.visualization.plot_decision",
        "plot_object_error_overlay",
    ),
    "plot_object_fp_fn_overlay": (
        "src.visualization.plot_decision",
        "plot_object_fp_fn_overlay",
    ),
    "plot_pixel_error_overlay": (
        "src.visualization.plot_decision",
        "plot_pixel_error_overlay",
    ),
    "plot_pixel_fp_fn_overlay": (
        "src.visualization.plot_decision",
        "plot_pixel_fp_fn_overlay",
    ),
    "plot_pixel_prediction_overlay": (
        "src.visualization.plot_decision",
        "plot_pixel_prediction_overlay",
    ),

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------
    "as_2d_array": ("src.utils", "as_2d_array"),
    "as_1d_array": ("src.utils", "as_1d_array"),
    "as_list": ("src.utils", "as_list"),
    "check_same_length": ("src.utils", "check_same_length"),
    "is_float_like": ("src.utils", "is_float_like"),
    "safe_positive": ("src.utils", "safe_positive"),
    "safe_divide": ("src.utils", "safe_divide"),
    "mask_value_to_nan": ("src.utils", "mask_value_to_nan"),
    "filter_records": ("src.utils", "filter_records"),
    "filter_dataframe_by_values": ("src.utils", "filter_dataframe_by_values"),
    "wavelength_axis": ("src.utils", "wavelength_axis"),
    "make_wavelengths": ("src.utils", "make_wavelengths"),
    "save_pickle": ("src.utils", "save_pickle"),
    "ensure_parent_dir": ("src.utils", "ensure_parent_dir"),
    "optimize_dataframe_for_parquet": (
        "src.utils",
        "optimize_dataframe_for_parquet",
    ),
    "make_dataframe_parquet_safe": (
        "src.utils",
        "make_dataframe_parquet_safe",
    ),
    "save_parquet": ("src.utils", "save_parquet"),
    "save_parquet_if_nonempty": ("src.utils", "save_parquet_if_nonempty"),
    "load_parquet": ("src.utils", "load_parquet"),
    "save_empty_parquet": ("src.utils", "save_empty_parquet"),
    "row_value": ("src.utils", "row_value"),
    "row_int": ("src.utils", "row_int"),
    "row_float": ("src.utils", "row_float"),
    "row_str": ("src.utils", "row_str"),
    "is_missing_value": ("src.utils", "is_missing_value"),
    "first_available_value": ("src.utils", "first_available_value"),
    "list_result_files": ("src.utils", "list_result_files"),
}


# ---------------------------------------------------------------------------
# API additions introduced by the decision/reporting/visualization refactor.
#
# Keep these imports lazy: importing ``src`` must not import pandas, scipy,
# scikit-image or Plotly until one of the corresponding attributes is used.
# ---------------------------------------------------------------------------
_PUBLIC_API.update(
    {
        # Utilities
        "to_numeric_metrics": ("src.utils", "to_numeric_metrics"),
        "parse_preprocessing_steps": (
            "src.utils",
            "parse_preprocessing_steps",
        ),
        "merge_config_metadata": (
            "src.utils",
            "merge_config_metadata",
        ),

        # Decision metrics
        "coerce_binary_series": (
            "src.decision.metrics",
            "coerce_binary_series",
        ),
        "binary_confusion_table": (
            "src.decision.metrics",
            "binary_confusion_table",
        ),

        # Binary confidence
        "add_binary_confidence": (
            "src.decision.confidence",
            "add_binary_confidence",
        ),
        "add_binary_object_confidence": (
            "src.decision.confidence",
            "add_binary_object_confidence",
        ),
        "add_binary_pixel_confidence": (
            "src.decision.confidence",
            "add_binary_pixel_confidence",
        ),

        # Border diagnostics
        "summarize_border_diagnostics_by_config": (
            "src.decision.border",
            "summarize_border_diagnostics_by_config",
        ),

        # Decision maps
        "assign_object_decisions_to_pixels": (
            "src.decision.maps",
            "assign_object_decisions_to_pixels",
        ),
        "make_pixel_categorical_map": (
            "src.decision.maps",
            "make_pixel_categorical_map",
        ),

        # Three-way decision / uncertainty
        "select_three_way_threshold_one_config": (
            "src.decision.uncertainty",
            "select_three_way_threshold_one_config",
        ),
        "select_three_way_threshold_pareto": (
            "src.decision.uncertainty",
            "select_three_way_threshold_pareto",
        ),
        "calibrate_three_way_thresholds_by_config": (
            "src.decision.uncertainty",
            "calibrate_three_way_thresholds_by_config",
        ),
        "apply_three_way_thresholds_by_config": (
            "src.decision.uncertainty",
            "apply_three_way_thresholds_by_config",
        ),
        "evaluate_three_way_by_config": (
            "src.decision.uncertainty",
            "evaluate_three_way_by_config",
        ),
        "add_three_way_confidence": (
            "src.decision.uncertainty",
            "add_three_way_confidence",
        ),
        # Canonical calculation function. Do not re-export the duplicate
        # implementation from src.visualization.plot_decision.
        "three_way_confusion_table": (
            "src.decision.uncertainty",
            "three_way_confusion_table",
        ),

        # Reporting selection
        "choose_diagnostic_configs": (
            "src.reporting.selection",
            "choose_diagnostic_configs",
        ),
        "choose_images_for_config": (
            "src.reporting.selection",
            "choose_images_for_config",
        ),
        "choose_images_for_config_3way": (
            "src.reporting.selection",
            "choose_images_for_config_3way",
        ),
        "sample_for_qt2_plot": (
            "src.reporting.selection",
            "sample_for_qt2_plot",
        ),

        # Visualization common
        "CLASS_COLOR_MAP": (
            "src.visualization.common",
            "CLASS_COLOR_MAP",
        ),
        "ERROR_COLOR_MAP": (
            "src.visualization.common",
            "ERROR_COLOR_MAP",
        ),
        "BINARY_CLASS_ORDER": (
            "src.visualization.common",
            "BINARY_CLASS_ORDER",
        ),
        "THREE_WAY_CLASS_ORDER": (
            "src.visualization.common",
            "THREE_WAY_CLASS_ORDER",
        ),
        "ERROR_ORDER": (
            "src.visualization.common",
            "ERROR_ORDER",
        ),
        "normalize_class_label": (
            "src.visualization.common",
            "normalize_class_label",
        ),
        "normalize_class_array": (
            "src.visualization.common",
            "normalize_class_array",
        ),
        "class_color": (
            "src.visualization.common",
            "class_color",
        ),
        "class_color_map": (
            "src.visualization.common",
            "class_color_map",
        ),
        "color_with_alpha": (
            "src.visualization.common",
            "color_with_alpha",
        ),
        "discrete_colorscale": (
            "src.visualization.common",
            "discrete_colorscale",
        ),
        "apply_project_theme": (
            "src.visualization.common",
            "apply_project_theme",
        ),
        "foreground_bbox": (
            "src.visualization.common",
            "foreground_bbox",
        ),
        "crop_to_foreground": (
            "src.visualization.common",
            "crop_to_foreground",
        ),
        "crop_arrays_to_foreground": (
            "src.visualization.common",
            "crop_arrays_to_foreground",
        ),
        "sanitize_filename": (
            "src.visualization.common",
            "sanitize_filename",
        ),
        "make_config_display_name": (
            "src.visualization.common",
            "make_config_display_name",
        ),
        "save_figure_bundle": (
            "src.visualization.common",
            "save_figure_bundle",
        ),

        # Visualization objects
        "plot_object_area_distribution": (
            "src.visualization.plot_objects",
            "plot_object_area_distribution",
        ),

        # Visualization decisions
        "plot_pixel_three_way_decision_overlay": (
            "src.visualization.plot_decision",
            "plot_pixel_three_way_decision_overlay",
        ),
        "plot_confusion_heatmap_from_long": (
            "src.visualization.plot_decision",
            "plot_confusion_heatmap_from_long",
        ),
        "plot_three_way_confusion_heatmap": (
            "src.visualization.plot_decision",
            "plot_three_way_confusion_heatmap",
        ),
        "plot_binary_confusion_heatmap": (
            "src.visualization.plot_decision",
            "plot_binary_confusion_heatmap",
        ),

        # Model-selection plots
        "plot_detection_pareto": (
            "src.visualization.plot_model_selection",
            "plot_detection_pareto",
        ),
        "plot_three_way_tradeoff": (
            "src.visualization.plot_model_selection",
            "plot_three_way_tradeoff",
        ),
        "plot_parameter_tendencies": (
            "src.visualization.plot_model_selection",
            "plot_parameter_tendencies",
        ),
        "plot_validation_test_shift": (
            "src.visualization.plot_model_selection",
            "plot_validation_test_shift",
        ),
        "plot_model_metric_ranking": (
            "src.visualization.plot_model_selection",
            "plot_model_metric_ranking",
        ),

        # Reporting plots
        "plot_per_image_performance": (
            "src.visualization.plot_reporting",
            "plot_per_image_performance",
        ),
        "plot_true_vs_predicted_object_counts": (
            "src.visualization.plot_reporting",
            "plot_true_vs_predicted_object_counts",
        ),
        "plot_stage_metric_comparison": (
            "src.visualization.plot_reporting",
            "plot_stage_metric_comparison",
        ),
        "plot_mixture_diagnostic_panel": (
            "src.visualization.plot_reporting",
            "plot_mixture_diagnostic_panel",
        ),

        # Robustness plots
        "plot_ablation_deltas": (
            "src.visualization.plot_robustness",
            "plot_ablation_deltas",
        ),
        "plot_ablation_heatmap": (
            "src.visualization.plot_robustness",
            "plot_ablation_heatmap",
        ),
        "plot_stability_intervals": (
            "src.visualization.plot_robustness",
            "plot_stability_intervals",
        ),
        "plot_border_core_metrics": (
            "src.visualization.plot_robustness",
            "plot_border_core_metrics",
        ),
        "plot_truth_dilation_sensitivity": (
            "src.visualization.plot_robustness",
            "plot_truth_dilation_sensitivity",
        ),
    }
)

# This symbol no longer exists in plot_objects; use
# ``plot_object_area_distribution`` instead.
_PUBLIC_API.pop("plot_object_areas", None)


# ---------------------------------------------------------------------------
# API additions from plot_scores / plot_simca / plot_spectra / tables.
# ---------------------------------------------------------------------------
_PUBLIC_API.update(
    {
        # Score plots and pixel-score summaries
        "build_scores_dataframe": (
            "src.visualization.plot_scores",
            "build_scores_dataframe",
        ),
        "sample_scores_dataframe": (
            "src.visualization.plot_scores",
            "sample_scores_dataframe",
        ),
        "plot_scores_density": (
            "src.visualization.plot_scores",
            "plot_scores_density",
        ),
        "plot_scores_distribution": (
            "src.visualization.plot_scores",
            "plot_scores_distribution",
        ),
        "summarize_scores_by_object": (
            "src.visualization.plot_scores",
            "summarize_scores_by_object",
        ),
        "plot_object_score_summary": (
            "src.visualization.plot_scores",
            "plot_object_score_summary",
        ),

        # SIMCA diagnostics
        "plot_simca_q_t2_dataframe": (
            "src.visualization.plot_simca",
            "plot_simca_q_t2_dataframe",
        ),

        # Spectral reporting
        "plot_spectra_by_batch": (
            "src.visualization.plot_spectra",
            "plot_spectra_by_batch",
        ),

        # Compact reporting tables
        "build_database_inventory_table": (
            "src.visualization.tables",
            "build_database_inventory_table",
        ),
        "build_preprocessing_shortlist_table": (
            "src.visualization.tables",
            "build_preprocessing_shortlist_table",
        ),
        "build_candidate_model_table": (
            "src.visualization.tables",
            "build_candidate_model_table",
        ),
        "build_frozen_reference_table": (
            "src.visualization.tables",
            "build_frozen_reference_table",
        ),
        "build_per_image_error_table": (
            "src.visualization.tables",
            "build_per_image_error_table",
        ),
        "build_presentation_summary_table": (
            "src.visualization.tables",
            "build_presentation_summary_table",
        ),
    }
)


# Deprecated compatibility wrappers are intentionally not part of the root API.
# They remain importable from their original modules during the transition.
for _deprecated_name in (
    "plot_object_areas",
    "plot_object_fp_fn_overlay",
    "plot_pixel_fp_fn_overlay",
    "plot_decision_counts",
    "choose_images_for_config_2way",
):
    _PUBLIC_API.pop(_deprecated_name, None)
del _deprecated_name


def __getattr__(name: str) -> Any:
    """Lazily import public objects when accessed as ``src.<name>``."""
    if name not in _PUBLIC_API:
        raise AttributeError(f"module 'src' has no attribute {name!r}")

    module_name, attr_name = _PUBLIC_API[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PUBLIC_API.keys()))


__all__ = sorted(_PUBLIC_API.keys())
