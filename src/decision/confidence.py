from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.labels import DEFAULT_TARGET_CLASS, pixel_ratio_col


def add_binary_confidence(
    df: pd.DataFrame,
    score_col: str,
    threshold,
    *,
    higher_is_target: bool = True,
    output_margin_col: str = "binary_margin",
    output_confidence_col: str = "binary_confidence",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Add a normalized confidence based on distance to a binary threshold.

    ``threshold`` can be a scalar or the name of a dataframe column. Confidence
    is normalized separately on each side of the threshold and clipped to [0, 1].
    """
    if score_col not in df.columns:
        raise KeyError(f"Missing score column: {score_col}")
    out = df.copy()
    score = pd.to_numeric(out[score_col], errors="coerce")
    if isinstance(threshold, str):
        if threshold not in out.columns:
            raise KeyError(f"Missing threshold column: {threshold}")
        threshold_values = pd.to_numeric(out[threshold], errors="coerce")
    else:
        threshold_values = pd.Series(float(threshold), index=out.index)

    signed = score - threshold_values
    if not higher_is_target:
        signed = -signed
    margin = signed.abs()

    # Scores are usually ratios or normalized statistics. Use the available
    # interval on each side when it is meaningful, otherwise fall back to a
    # relative threshold distance.
    upper_scale = (1.0 - threshold_values).abs().clip(lower=eps)
    lower_scale = threshold_values.abs().clip(lower=eps)
    scale = pd.Series(
        np.where(signed >= 0, upper_scale, lower_scale),
        index=out.index,
        dtype=float,
    )
    fallback = threshold_values.abs().clip(lower=1.0, upper=np.inf)
    scale = scale.where(np.isfinite(scale) & (scale > eps), fallback)

    out[output_margin_col] = signed
    out[output_confidence_col] = (margin / scale).clip(lower=0.0, upper=1.0)
    out["binary_confidence_bin"] = pd.cut(
        out[output_confidence_col],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["low", "medium", "high"],
    ).astype("object")
    return out


def add_binary_object_confidence(
    object_df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    ratio_col: str | None = None,
    threshold_col: str = "object_threshold",
    threshold: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Add confidence to object decisions based on the target-pixel ratio."""
    ratio_col = pixel_ratio_col(target_class) if ratio_col is None else ratio_col
    threshold_arg = threshold_col if threshold is None and threshold_col in object_df.columns else threshold
    if threshold_arg is None:
        raise ValueError("Provide threshold or an existing threshold_col.")
    return add_binary_confidence(
        object_df,
        score_col=ratio_col,
        threshold=threshold_arg,
        higher_is_target=True,
        **kwargs,
    )


def add_binary_pixel_confidence(
    pixel_df: pd.DataFrame,
    score_col: str = "rule_statistic",
    threshold_col: str = "rule_limit",
    threshold: float | None = None,
    accepted_when_below: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Add confidence to pixel SIMCA decisions from a rule statistic.

    SIMCA acceptance commonly occurs when the statistic is below its limit;
    therefore ``accepted_when_below=True`` is the default.
    """
    threshold_arg = threshold_col if threshold is None and threshold_col in pixel_df.columns else threshold
    if threshold_arg is None:
        raise ValueError("Provide threshold or an existing threshold_col.")
    return add_binary_confidence(
        pixel_df,
        score_col=score_col,
        threshold=threshold_arg,
        higher_is_target=not accepted_when_below,
        **kwargs,
    )
