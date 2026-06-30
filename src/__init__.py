"""
Public API for the src package.

The PCA public workflow is now based on PCAModel. The low-level pca_from_cov
function remains inside src.pca for implementation/debugging but is not exported
here to avoid using raw PCA dictionaries in notebooks.
"""

# Dataload
from .io.dataload import load_mat_file

from src.data.segmentation import segment_objects
from src.data.database import (
    parse_image_key, 
    preprocess_nir_uco_cube, 
    extract_objects_from_labeled_image, 
    build_minimal_nir_uco_object_database,
)

from src.decision.metrics import (
    binary_detection_metrics,
    metrics_by_group,
    add_detection_score,
    summarize_pixel_errors_by_image,
)

from src.decision.truth import (
    expected_position_key_for_mixture,
    union_object_masks,
    target_truth_map_for_image,
    peanut_truth_map_for_image,
    add_pixel_truth_labels,
)

from src.decision.aggregation import (
    add_object_metadata,
    aggregate_pixel_predictions_to_objects,
    object_threshold_grid,
)

from src.decision.border import (
    add_border_flags_to_pixel_df,
    aggregate_pixel_predictions_to_objects_core,
    border_width_object_threshold_grid,
    summarize_pixel_errors_by_border_zone,
)

from src.decision.maps import (
    make_pixel_error_map,
    make_pixel_prediction_map,
    make_object_error_map,
)

from src.decision.uncertainty import (
    add_three_way_object_decision,
    summarize_three_way_decision,
)

from src.io.database_h5 import load_nir_uco_h5
from src.io.dataload import load_mat_file

from src.matrices.matrix_registry import (
    build_matrix,
    get_matrix_spec,
    matrix_method_to_args,
    available_matrix_methods
)
from src.matrices.redim_matrix import object_db_to_matrix

from src.models.pca import PCAModel, pca_from_cov
from src.models.simca import SIMCAClassModel, SIMCAClassifier
from src.models.simca_rules import make_simca_rule, compute_rule_variant_stat_limit, accept_rule_variant

# Preprocessing
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.band_selection import wavelength_selection_summary, select_wavelength_range_from_database

from src.workflows.pixel_projection import (
    build_training_matrix,
    build_projection_pixel_matrix,
    fit_one_class_peanut_simca,
    predict_pixels_with_simca,
    fit_one_class_simca,
)

from src.workflows.simca_pixel_grid import (
    make_peanut_train_filters,
    run_single_simca_pixel_projection,
    run_simca_pixel_projection_grid,
    refit_best_grid_row,
)

from src.workflows.simca_cv_calibration import (
    calibrate_simca_thresholds_cv,
    fit_final_simca_model,
    project_pixels_with_rule_variants,
    summarize_cv_calibration,
    run_simca_empirical_rule_grid,
)

from src.workflows.simca_optuna import (
    make_simca_optuna_objective,
    run_optuna_simca_pixel_optimization,
    optuna_trials_dataframe,
    best_completed_trial_row,
    refit_optuna_best_trial,
    close_optuna_study,
)

from src.workflows.pca_comparison import (
    compare_pca_representations,
    add_pca_selection_score,
)

from src.workflows.pca_diagnostic import (
    class_separation_scores,
    compute_pca_summary_metrics,
)

# Plotting
from src.visualization.plot_generic import (
    plot_bar_values,
    plot_counts_by_group,
    plot_lines_from_dataframe,
    plot_distribution_with_curve,
)

from src.visualization.plot_images import (
    plot_hypercube_band_slider,
    plot_image2d,
    plot_image_overlay,
    plot_label_overlay_from_image_db,
)

from src.visualization.plot_spectra import (
    mean_spectrum_from_cube,
    extract_spectral_matrix,
    plot_spectra,
    plot_spectral_distribution,
    plot_object_spectra,
)

from src.visualization.plot_objects import (
    plot_object_view,
    plot_object_grid,
    plot_object_areas,
)

from src.visualization.plot_scores import (
    plot_scores,
    build_scores_dataframe,
    sample_scores_dataframe,
    plot_scores_density,
    plot_scores_distribution,
    summarize_scores_by_object,
    plot_object_score_summary,
)

from src.visualization.plot_diagnostics import (
    plot_metric_by_index,
    plot_xy_diagnostic,
)

from src.visualization.plot_pca import (
    plot_explained_variance,
    plot_loadings,
    plot_biplot,
    plot_pca_metric_t2,
    plot_pca_metric_q,
    plot_pca_diagnostic,
    plot_pca_metric_heatmap,
    plot_pca_metric_tradeoff,
    plot_pca_metric_ranking,
)

from src.visualization.plot_simca import (
    plot_simca_distance,
    plot_simca_rule_metric,
    plot_decision_counts,
)

from src.visualization.plot_decision import (
    plot_object_decision_map,
    plot_object_error_overlay,
    plot_pixel_error_overlay,
    plot_pixel_fp_fn_overlay,
    plot_object_fp_fn_overlay,
)

# Utilities
from .utils import (
    as_2d_array,
    as_1d_array,
    as_list,
    check_same_length,
    is_float_like,
    safe_positive,
    safe_divide,
    mask_value_to_nan,
    filter_records,
    wavelength_axis,
    make_wavelengths,
    save_pickle,
)

__all__ = [
    "load_mat_file",
    "load_nir_uco_h5",
    "segment_objects",
    "preprocess_nir_uco_cube",
    "binary_detection_metrics",
    "metrics_by_group",
    "add_detection_score",
    "summarize_pixel_errors_by_image",
    "parse_image_key",
    "extract_objects_from_labeled_image",
    "build_minimal_nir_uco_object_database",
    "object_db_to_matrix",
    "expected_position_key_for_mixture",
    "union_object_masks",
    "target_truth_map_for_image",
    "peanut_truth_map_for_image",
    "add_pixel_truth_labels",
    "add_object_metadata",
    "aggregate_pixel_predictions_to_objects",
    "object_threshold_grid",
    "add_border_flags_to_pixel_df",
    "aggregate_pixel_predictions_to_objects_core",
    "border_width_object_threshold_grid",
    "summarize_pixel_errors_by_border_zone",
    "make_pixel_error_map",
    "make_pixel_prediction_map",
    "make_object_error_map",
    "add_three_way_object_decision",
    "summarize_three_way_decision",
    "build_matrix",
    "get_matrix_spec",
    "matrix_method_to_args",
    "available_matrix_methods",
    "normalize_preprocessing_configs",
    "SpectralPreprocessor",
    "wavelength_selection_summary",
    "select_wavelength_range_from_database",
    "PCAModel",
    "pca_from_cov",
    "class_separation_scores",
    "compare_pca_representations",
    "compute_pca_summary_metrics",
    "add_pca_selection_score",
    "make_simca_rule", 
    "compute_rule_variant_stat_limit", 
    "accept_rule_variant",
    "SIMCAClassModel",
    "SIMCAClassifier",
    "build_training_matrix",
    "build_projection_pixel_matrix",
    "fit_one_class_peanut_simca",
    "predict_pixels_with_simca",
    "fit_one_class_simca",
    "make_peanut_train_filters",
    "run_single_simca_pixel_projection",
    "run_simca_pixel_projection_grid",
    "refit_best_grid_row",
    "calibrate_simca_thresholds_cv",
    "fit_final_simca_model",
    "project_pixels_with_rule_variants",
    "summarize_cv_calibration",
    "run_simca_empirical_rule_grid",
    "make_simca_optuna_objective",
    "run_optuna_simca_pixel_optimization",
    "optuna_trials_dataframe",
    "best_completed_trial_row",
    "refit_optuna_best_trial",
    "close_optuna_study",
    "mean_spectrum_from_cube",
    "extract_spectral_matrix",
    "plot_hypercube_band_slider",
    "plot_image2d",
    "plot_image_overlay",
    "plot_label_overlay_from_image_db",
    "plot_spectra",
    "plot_spectral_distribution",
    "plot_object_spectra",
    "plot_object_view",
    "plot_object_grid",
    "plot_object_areas",
    "plot_explained_variance",
    "plot_scores",
    "build_scores_dataframe",
    "sample_scores_dataframe",
    "plot_scores_density",
    "plot_scores_distribution",
    "summarize_scores_by_object",
    "plot_object_score_summary",
    "plot_loadings",
    "plot_biplot",
    "plot_metric_by_index",
    "plot_xy_diagnostic",
    "plot_pca_metric_t2",
    "plot_pca_metric_q",
    "plot_pca_diagnostic",
    "plot_pca_metric_heatmap",
    "plot_pca_metric_tradeoff",
    "plot_pca_metric_ranking",
    "plot_bar_values",
    "plot_counts_by_group",
    "plot_lines_from_dataframe",
    "plot_object_decision_map",
    "plot_distribution_with_curve",
    "plot_decision_counts",
    "plot_simca_distance",
    "plot_simca_rule_metric",
    "plot_pixel_error_overlay",
    "plot_pixel_fp_fn_overlay",
    "plot_object_fp_fn_overlay",
    "plot_object_error_overlay",
    "as_2d_array",
    "as_1d_array",
    "as_list",
    "check_same_length",
    "is_float_like",
    "safe_positive",
    "safe_divide",
    "mask_value_to_nan",
    "filter_records",
    "wavelength_axis",
    "make_wavelengths",
    "save_pickle",
]
