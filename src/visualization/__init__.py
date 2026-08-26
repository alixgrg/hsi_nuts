from src.visualization.common import (
    background_image,
    make_customdata,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
    validate_columns,
)

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
    plot_metric_heatmap,
    plot_metric_by_index,
    plot_xy_diagnostic,
)

from src.visualization.plot_pca import (
    build_pca_visual_review_pdf,
    plot_explained_variance,
    plot_loadings,
    plot_biplot,
    plot_pca_metric_t2,
    plot_pca_metric_q,
    plot_pca_diagnostic,
    plot_pca_metric_heatmap,
    plot_pca_metric_tradeoff,
    plot_pca_metric_ranking,
    plot_pca_review_panel,
)

from src.visualization.plot_simca import (
    plot_simca_distance,
    plot_simca_rule_metric,
    plot_decision_counts,
)

from src.visualization.plot_model_selection import (
    plot_detection_pareto,
    plot_model_metric_ranking,
    plot_parameter_tendencies,
    plot_selection_reasons,
    plot_three_way_tradeoff,
    plot_threshold_tradeoff,
    plot_validation_test_shift,
)

from src.visualization.plot_decision import (
    plot_object_decision_map,
    plot_object_error_overlay,
    plot_object_fp_fn_overlay,
    plot_pixel_error_overlay,
    plot_pixel_fp_fn_overlay,
    plot_pixel_prediction_overlay,
)

__all__ = [
    "background_image",
    "build_scores_dataframe",
    "build_pca_visual_review_pdf",
    "extract_spectral_matrix",
    "make_customdata",
    "make_dynamic_color_map",
    "mean_spectrum_from_cube",
    "ordered_unique",
    "plot_bar_values",
    "plot_biplot",
    "plot_counts_by_group",
    "plot_decision_counts",
    "plot_detection_pareto",
    "plot_distribution_with_curve",
    "plot_explained_variance",
    "plot_hypercube_band_slider",
    "plot_image2d",
    "plot_image_overlay",
    "plot_label_overlay_from_image_db",
    "plot_lines_from_dataframe",
    "plot_loadings",
    "plot_metric_by_index",
    "plot_metric_heatmap",
    "plot_model_metric_ranking",
    "plot_object_decision_map",
    "plot_object_error_overlay",
    "plot_object_fp_fn_overlay",
    "plot_object_grid",
    "plot_object_score_summary",
    "plot_object_spectra",
    "plot_object_view",
    "plot_parameter_tendencies",
    "plot_pca_diagnostic",
    "plot_pca_metric_heatmap",
    "plot_pca_metric_q",
    "plot_pca_metric_ranking",
    "plot_pca_metric_t2",
    "plot_pca_metric_tradeoff",
    "plot_pca_review_panel",
    "plot_pixel_error_overlay",
    "plot_pixel_fp_fn_overlay",
    "plot_pixel_prediction_overlay",
    "plot_scores",
    "plot_scores_density",
    "plot_scores_distribution",
    "plot_selection_reasons",
    "plot_simca_distance",
    "plot_simca_rule_metric",
    "plot_spectra",
    "plot_spectral_distribution",
    "plot_three_way_tradeoff",
    "plot_threshold_tradeoff",
    "plot_validation_test_shift",
    "plot_xy_diagnostic",
    "sample_scores_dataframe",
    "show_or_return",
    "summarize_scores_by_object",
    "validate_columns",
]
