from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    DEFAULT_NON_TARGET_LABEL,
)




def add_simca_selection_score(
    df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> pd.DataFrame:
    """Add a scalar score with FN-first hierarchy for SIMCA selection tables."""
    out = df.copy()

    sens_col = f"{target_class}_sensitivity"
    spec_col = f"{non_target_class}_specificity"

    sens = out.get(sens_col, np.nan)
    spec = out.get(spec_col, np.nan)
    if "fn_rate" not in out.columns:
        out["fn_rate"] = 1.0 - pd.Series(sens, index=out.index).astype(float)
    if "fp_rate" not in out.columns:
        out["fp_rate"] = 1.0 - pd.Series(spec, index=out.index).astype(float)
    f1 = out["f1_score"] if "f1_score" in out.columns else 0.0
    acc = out["accuracy"] if "accuracy" in out.columns else 0.0
    ba = out["balanced_accuracy"] if "balanced_accuracy" in out.columns else 0.0
    out["selection_score"] = (
        -10.0 * out["fn_rate"].astype(float)
        -1.0 * out["fp_rate"].astype(float)
        +0.05 * pd.Series(acc, index=out.index).astype(float).fillna(0.0)
        +0.02 * pd.Series(f1, index=out.index).astype(float).fillna(0.0)
        +0.02 * pd.Series(ba, index=out.index).astype(float).fillna(0.0)
    )

    return out


def sort_simca_selection(
    df: pd.DataFrame,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_class: str = DEFAULT_NON_TARGET_LABEL,
) -> pd.DataFrame:
    """Sort a SIMCA result table using FN-first project hierarchy."""
    if df.empty:
        return df.copy()

    out = add_simca_selection_score(
        df,
        target_class=target_class,
        non_target_class=non_target_class,
    )

    sort_cols = [
        "fn_rate",
        "fp_rate",
        "accuracy",
        "f1_score",
        "balanced_accuracy",
        "selection_score",
    ]
    sort_cols = [c for c in sort_cols if c in out.columns]

    ascending = [c in {"fn_rate", "fp_rate"} for c in sort_cols]
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
