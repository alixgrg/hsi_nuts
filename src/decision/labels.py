from __future__ import annotations

DEFAULT_TARGET_CLASS = "target"
DEFAULT_NON_TARGET_LABEL = "non_target"
UNCERTAIN_LABEL = "uncertain"


def predicted_col(target_class: str, level: str) -> str:
    return f"predicted_{target_class}_{level}"


def true_col(target_class: str, level: str) -> str:
    return f"true_{target_class}_{level}"


def pixel_ratio_col(target_class: str) -> str:
    return f"{target_class}_pixel_ratio"


def true_pixel_ratio_col(target_class: str) -> str:
    return f"true_{target_class}_pixel_ratio"


def true_pixel_ratio_total_col(target_class: str) -> str:
    return f"true_{target_class}_pixel_ratio_total"


def n_predicted_pixels_col(target_class: str) -> str:
    return f"n_predicted_{target_class}_pixels"