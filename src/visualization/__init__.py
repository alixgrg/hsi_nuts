from src.visualization.common import (
    show_or_return,
    make_customdata,
    make_dynamic_color_map,
    background_image,
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
    plot_pixel_error_overlay,
    plot_pixel_fp_fn_overlay,
    plot_object_fp_fn_overlay,
    plot_object_error_overlay,
)