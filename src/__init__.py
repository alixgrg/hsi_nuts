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


__version__ = "0.1.0"


_PUBLIC_API: dict[str, tuple[str, str]] = {
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    "load_mat_file": ("src.io.dataload", "load_mat_file"),
    "load_nir_uco_h5": ("src.io.database_h5", "load_nir_uco_h5"),
    "save_nir_uco_h5": ("src.io.database_h5", "save_nir_uco_h5"),

    # ------------------------------------------------------------------
    # Data / database / segmentation
    # ------------------------------------------------------------------
    "segment_objects": ("src.data.segmentation", "segment_objects"),
    "parse_image_key": ("src.data.database", "parse_image_key"),
    "preprocess_nir_uco_cube": ("src.data.database", "preprocess_nir_uco_cube"),
    "extract_objects_from_labeled_image": (
        "src.data.database",
        "extract_objects_from_labeled_image",
    ),
    "build_minimal_nir_uco_object_database": (
        "src.data.database",
        "build_minimal_nir_uco_object_database",
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
    "add_detection_score": ("src.decision.metrics", "add_detection_score"),
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
    "build_matrix": ("src.matrices.matrix_registry", "build_matrix"),
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

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    "PCAModel": ("src.models.pca", "PCAModel"),
    "pca_from_cov": ("src.models.pca", "pca_from_cov"),
    "SIMCAClassModel": ("src.models.simca", "SIMCAClassModel"),
    "SIMCAClassifier": ("src.models.simca", "SIMCAClassifier"),
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
    "normalize_preprocessing_configs": (
        "src.spectra.preprocessing_configs",
        "normalize_preprocessing_configs",
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
    "add_pca_selection_score": (
        "src.workflows.pca",
        "add_pca_selection_score",
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
    "run_simca_empirical_rule_grid": (
        "src.workflows.simca",
        "run_simca_empirical_rule_grid",
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