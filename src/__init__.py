"""
Public API for the src package.

This file exposes the stable functions/classes used in the notebooks.

Example
-------
from src import load_mat_file, SIMCAClassifier, plot_spectra, plot_scores
"""

# Dataload
from .dataload import (
    load_mat_file,
)

# Segmentation
from .segmentation import (
    make_reference_image,
    make_binary_mask,
    clean_mask,
    label_objects_with_watershed,
    segment_objects,
)

# Database
from .database import (
    NIR_UCO_NAME_CONFIG,
    parse_image_key,
    extract_objects_from_labeled_image,
    build_minimal_nir_uco_object_database,
)

# Redim Matrix
from .redim_matrix import (
    object_db_to_object_matrix,
    object_db_to_object_matrix_by_sources,
    object_db_to_pixel_matrix,
    object_db_to_balanced_px_matrix,
)

# Preprocessing
from .preprocessing import (
    center_X,
    snv,
    vector_normalize,
    msc_fit,
    msc_transform,
    savgol_derivative,
    reflectance_to_absorbance,
)

# Pca
from .pca import (
    pca_from_cov,
    pca_sklearn,
)

# Pca Comparison
from .pca_comparison import (
    apply_preprocessing_for_pca,
    class_separation_scores,
    mahalanobis_centroid_distance,
    build_matrix_for_pca_method,
    compare_pca_representations,
)

# Stats
from .stats import (
    mean_spectrum,
    hotelling_t2,
    q_residuals,
)

# Simca
from .simca import (
    SIMCAClassModel,
    BaseSIMCARule,
    SimpleSIMCARule,
    AltSIMCARule,
    CombinedIndexSIMCARule,
    DataDrivenSIMCARule,
    SIMCAClassifier,
)

# Plotting
from .plotting import (
    select_objects,
    plot_hypercube_band_slider,
    plot_image2d,
    plot_image_overlay,
    plot_label_overlay_from_image_db,
    plot_spectra,
    plot_spectral_distribution,
    plot_object_spectra,
    plot_object_view,
    plot_object_grid,
    plot_object_areas,
    plot_explained_variance,
    plot_scores,
    plot_loadings,
    plot_biplot,
    plot_metric_by_index,
    plot_xy_diagnostic,
    plot_pca_metric_t2,
    plot_pca_metric_q,
    plot_pca_diagnostic,
    plot_bar_values,
    plot_counts_by_group,
    plot_lines_from_dataframe,
    plot_object_decision_map,
    plot_distribution_with_curve,
    plot_decision_counts,
    plot_simca_distance,
    plot_simca_rule_metric,
)

__all__ = [
    "load_mat_file",
    "make_reference_image",
    "make_binary_mask",
    "clean_mask",
    "label_objects_with_watershed",
    "segment_objects",
    "NIR_UCO_NAME_CONFIG",
    "parse_image_key",
    "extract_objects_from_labeled_image",
    "build_minimal_nir_uco_object_database",
    "object_db_to_object_matrix",
    "object_db_to_object_matrix_by_sources",
    "object_db_to_pixel_matrix",
    "object_db_to_balanced_px_matrix",
    "center_X",
    "snv",
    "vector_normalize",
    "msc_fit",
    "msc_transform",
    "savgol_derivative",
    "reflectance_to_absorbance",
    "pca_from_cov",
    "pca_sklearn",
    "apply_preprocessing_for_pca",
    "class_separation_scores",
    "mahalanobis_centroid_distance",
    "build_matrix_for_pca_method",
    "compare_pca_representations",
    "mean_spectrum",
    "hotelling_t2",
    "q_residuals",
    "SIMCAClassModel",
    "BaseSIMCARule",
    "SimpleSIMCARule",
    "AltSIMCARule",
    "CombinedIndexSIMCARule",
    "DataDrivenSIMCARule",
    "SIMCAClassifier",
    "select_objects",
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
    "plot_loadings",
    "plot_biplot",
    "plot_metric_by_index",
    "plot_xy_diagnostic",
    "plot_pca_metric_t2",
    "plot_pca_metric_q",
    "plot_pca_diagnostic",
    "plot_bar_values",
    "plot_counts_by_group",
    "plot_lines_from_dataframe",
    "plot_object_decision_map",
    "plot_distribution_with_curve",
    "plot_decision_counts",
    "plot_simca_distance",
    "plot_simca_rule_metric",
]
