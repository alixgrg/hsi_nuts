"""
Public API for the src package.

The PCA public workflow is now based on PCAModel. The low-level pca_from_cov
function remains inside src.pca for implementation/debugging but is not exported
here to avoid using raw PCA dictionaries in notebooks.
"""

# Dataload
from .dataload import load_mat_file

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

# Matrix construction / feature extraction
from .redim_matrix import object_db_to_matrix

# Preprocessing
from .preprocessing import (
    SpectralPreprocessor,
    center_X,
    snv,
    vector_normalize,
    msc_fit,
    msc_transform,
    savgol_derivative,
    reflectance_to_absorbance,
)

# PCA
from .pca import PCAModel

# PCA comparison
from .pca_comparison import (
    class_separation_scores,
    mahalanobis_centroid_distance,
    compare_pca_representations,
)

# SIMCA
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
    "object_db_to_matrix",
    "SpectralPreprocessor",
    "center_X",
    "snv",
    "vector_normalize",
    "msc_fit",
    "msc_transform",
    "savgol_derivative",
    "reflectance_to_absorbance",
    "PCAModel",
    "class_separation_scores",
    "mahalanobis_centroid_distance",
    "compare_pca_representations",
    "SIMCAClassModel",
    "BaseSIMCARule",
    "SimpleSIMCARule",
    "AltSIMCARule",
    "CombinedIndexSIMCARule",
    "DataDrivenSIMCARule",
    "SIMCAClassifier",
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
]
