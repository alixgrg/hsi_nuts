"""
Public API for the src package.

This file allows imports such as:

    from src import load_mat_file
    from src import parse_image_key
    from src import build_minimal_nir_uco_object_database
    from src import segment_objects
"""

# Data loading
from .dataload import load_mat_file

# Segmentation
from .segmentation import (
    make_reference_image,
    make_binary_mask,
    clean_mask,
    label_objects_with_watershed,
    segment_objects,
)

# Object database construction
from .database import (
    NIR_UCO_NAME_CONFIG,
    parse_image_key,
    extract_objects_from_labeled_image,
    build_minimal_nir_uco_object_database,
)

# Plotting
from .plotting import (
    # Raw data visualization
    plot_bands_slider,
    plot_mean_spectra,
    plot_spectral_distribution,
    mean_spectrum,
    plot_mean_spectra_from_excel,
    plot_two_classes,
    plot_loadings,
    # Object database visualization
    plot_db_image,
    plot_db_labels_overlay,
    plot_db_object,
    plot_db_object_grid,
    plot_db_object_spectra,
    plot_db_object_areas,
    # PCA visualization
    plot_pca_explained_variance,
    plot_pca_scores_2d,
    plot_pca_scores_3d,
    plot_pca_loadings,
    plot_pca_biplot_2d,
    plot_pca_hotelling_t2,
    plot_pca_q_residuals,
    plot_pca_q_vs_t2,
)

from .redim_matrix import (
    object_db_to_object_matrix, 
    object_db_to_pixel_matrix, 
    object_db_to_balanced_px_matrix
)

from .pca import (
    pca_from_cov,
    pca_sklearn,
)

from .stats import (
    mean_spectrum,
    hotelling_t2,
    q_residuals,
)

from .preprocessing import (
    center_X,
    snv, 
    vector_normalize,
    msc_fit,
    msc_transform,
    savgol_derivative,
)


__all__ = [
    # dataload
    "load_mat_file",

    # segmentation
    "make_reference_image",
    "make_binary_mask",
    "clean_mask",
    "label_objects_with_watershed",
    "segment_objects",

    # database
    "NIR_UCO_NAME_CONFIG",
    "parse_image_key",
    "extract_objects_from_labeled_image",
    "build_minimal_nir_uco_object_database",

    # plotting
    #    Raw data 
    "plot_bands_slider",
    "plot_mean_spectra",
    "plot_spectral_distribution",
    "plot_mean_spectra_from_excel",
    "plot_two_classes",
    "plot_loadings",
    #   Object database visualization
    "plot_db_image",
    "plot_db_labels_overlay",
    "plot_db_object",
    "plot_db_object_grid",
    "plot_db_object_spectra",
    "plot_db_object_areas",
    #   PCA
    "plot_pca_explained_variance",
    "plot_pca_scores_2d",
    "plot_pca_scores_3d",
    "plot_pca_loadings",
    "plot_pca_biplot_2d",
    "plot_pca_hotelling_t2",
    "plot_pca_q_residuals",
    "plot_pca_q_vs_t2",

    # redim_matrix
    "object_db_to_object_matrix",
    "object_db_to_pixel_matrix",
    "object_db_to_balanced_px_matrix",

    # pca
    "pca_from_cov",
    "pca_sklearn",

    # stats
    "mean_spectrum",
    "hotelling_t2",
    "q_residuals",

    # preprocessing
    "center_X",
    "snv",
    "vector_normalize",
    "msc_fit",
    "msc_transform",
    "savgol_derivative",
]