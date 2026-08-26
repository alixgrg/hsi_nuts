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
    summarize_border_diagnostics_by_config,
)
from src.decision.labels import (
    DEFAULT_NON_TARGET_LABEL,
    DEFAULT_TARGET_CLASS,
    UNCERTAIN_LABEL,
    n_predicted_pixels_col,
    pixel_ratio_col,
    predicted_col,
    true_col,
    true_pixel_ratio_col,
    true_pixel_ratio_total_col,
)
from src.decision.maps import (
    make_object_error_map,
    make_object_fp_fn_map,
    make_pixel_error_map,
    make_pixel_prediction_map,
)
from src.decision.metrics import (
    add_binary_confusion_case,
    apply_locked_margin_decision,
    binary_confusion_counts_vectorized,
    binary_mask_agreement,
    binary_detection_metrics,
    component_agreement,
    component_detection_table,
    component_detection_metrics,
    fragment_detection_table,
    metrics_by_group,
    summarize_object_errors_by_image,
    summarize_binary_metrics_vectorized,
    summarize_pixel_errors_by_image,
)
from src.decision.truth import (
    TruthResult,
    add_pixel_truth_labels,
    build_annotation_agreement_table,
    build_spatial_ground_truth_lock,
    build_spatial_ground_truth_manifest,
    expected_position_key_for_mixture,
    extract_reference_components,
    peanut_truth_map_for_image,
    pure_image_class_truth,
    resolve_truth_for_image,
    select_annotation_subset,
    select_double_annotation_images,
    target_truth_map_for_image,
    union_object_masks,
    validate_annotation_adjudication,
    validate_reference_annotation,
    validate_reference_mask,
    verify_spatial_ground_truth_lock,
)
_LAZY_EXPORTS = {
    "add_three_way_object_decision": ("src.decision.uncertainty", "add_three_way_object_decision"),
    "evaluate_three_way_object_decision": ("src.decision.uncertainty", "evaluate_three_way_object_decision"),
    "summarize_three_way_decision": ("src.decision.uncertainty", "summarize_three_way_decision"),
    "three_way_object_threshold_grid": ("src.decision.uncertainty", "three_way_object_threshold_grid"),
    "three_way_object_threshold_grid_by_group": (
        "src.decision.uncertainty",
        "three_way_object_threshold_grid_by_group",
    ),
    "select_three_way_threshold_one_config": (
        "src.decision.uncertainty",
        "select_three_way_threshold_one_config",
    ),
    "select_three_way_threshold_pareto": ("src.decision.uncertainty", "select_three_way_threshold_pareto"),
    "calibrate_three_way_thresholds_by_config": (
        "src.decision.uncertainty",
        "calibrate_three_way_thresholds_by_config",
    ),
    "apply_three_way_thresholds_by_config": (
        "src.decision.uncertainty",
        "apply_three_way_thresholds_by_config",
    ),
    "evaluate_three_way_by_config": ("src.decision.uncertainty", "evaluate_three_way_by_config"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_NON_TARGET_LABEL",
    "DEFAULT_TARGET_CLASS",
    "UNCERTAIN_LABEL",
    "add_binary_confusion_case",
    "apply_locked_margin_decision",
    "add_border_flags_to_pixel_df",
    "add_object_metadata",
    "add_pixel_truth_labels",
    #"add_three_way_object_decision",
    "aggregate_pixel_predictions_to_objects",
    "aggregate_pixel_predictions_to_objects_core",
    "binary_detection_metrics",
    "binary_confusion_counts_vectorized",
    "binary_mask_agreement",
    "component_agreement",
    "component_detection_table",
    "component_detection_metrics",
    "fragment_detection_table",
    "border_width_object_threshold_grid",
    #"evaluate_three_way_object_decision",
    "expected_position_key_for_mixture",
    "make_object_error_map",
    "make_object_fp_fn_map",
    "make_pixel_error_map",
    "make_pixel_prediction_map",
    "metrics_by_group",
    "n_predicted_pixels_col",
    "object_threshold_grid",
    "peanut_truth_map_for_image",
    "pure_image_class_truth",
    "pixel_ratio_col",
    "predicted_col",
    "summarize_object_errors_by_image",
    "summarize_binary_metrics_vectorized",
    "summarize_pixel_errors_by_border_zone",
    "summarize_pixel_errors_by_image",
    #"summarize_three_way_decision",
    "target_truth_map_for_image",
    #"three_way_object_threshold_grid",
    #"three_way_object_threshold_grid_by_group",
    "true_col",
    "true_pixel_ratio_col",
    "true_pixel_ratio_total_col",
    "union_object_masks",
    "TruthResult",
    "build_annotation_agreement_table",
    "build_spatial_ground_truth_lock",
    "build_spatial_ground_truth_manifest",
    "extract_reference_components",
    "resolve_truth_for_image",
    "select_annotation_subset",
    "select_double_annotation_images",
    "validate_annotation_adjudication",
    "validate_reference_annotation",
    "validate_reference_mask",
    "verify_spatial_ground_truth_lock",
    "summarize_border_diagnostics_by_config",
    #"select_three_way_threshold_one_config",
    #"select_three_way_threshold_pareto",
    #"calibrate_three_way_thresholds_by_config",
    #"apply_three_way_thresholds_by_config",
    #"evaluate_three_way_by_config",
]
