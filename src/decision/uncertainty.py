from __future__ import annotations

import numpy as np
import pandas as pd


def add_three_way_object_decision(
    object_df: pd.DataFrame,
    target_class: str = "peanut",
    ratio_col: str | None = None,
    lower_threshold: float = 0.40,
    upper_threshold: float = 0.75,
    output_col: str = "decision_3way",
) -> pd.DataFrame:
    """
    Add a three-way decision:
        non_peanut
        uncertain
        peanut

    Useful when false negatives are very costly and uncertain objects
    should be inspected manually.
    """
    if ratio_col is None:
        ratio_col = f"{target_class}_pixel_ratio"

    if ratio_col not in object_df.columns:
        raise KeyError(f"Missing ratio column: {ratio_col}")

    df = object_df.copy()

    ratio = df[ratio_col].astype(float)

    df[output_col] = np.where(
        ratio >= upper_threshold,
        target_class,
        np.where(
            ratio < lower_threshold,
            f"non_{target_class}",
            "uncertain",
        ),
    )

    df["three_way_lower_threshold"] = float(lower_threshold)
    df["three_way_upper_threshold"] = float(upper_threshold)

    return df


def summarize_three_way_decision(
    object_df: pd.DataFrame,
    decision_col: str = "decision_3way",
) -> pd.DataFrame:
    """Summarize three-way decision counts and rates."""
    counts = object_df[decision_col].value_counts(dropna=False).rename("n").reset_index()
    counts = counts.rename(columns={"index": decision_col})
    counts["rate"] = counts["n"] / len(object_df) if len(object_df) > 0 else np.nan

    return counts