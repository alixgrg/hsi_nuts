from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Raise when a result table misses required contract columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"{table_name} is missing required columns: {missing}")


def summarize_matrix_output(
    X,
    y,
    meta,
    matrix_method: str,
    filters: dict,
    balanced_pixel_strategy: str | None = None,
):
    """Return the matrix summary row and row-level metadata dataframe used by notebook 02."""
    X = np.asarray(X)
    y = np.asarray(y)
    meta_df = pd.DataFrame(meta)
    numeric = np.issubdtype(X.dtype, np.number)

    row = {
        "matrix_method": matrix_method,
        "balanced_pixel_strategy": balanced_pixel_strategy,
        "filters": json.dumps(filters),
        "n_observations": int(X.shape[0]),
        "n_features": int(X.shape[1]) if X.ndim == 2 else np.nan,
        "n_labels": int(len(np.unique(y))) if len(y) > 0 else 0,
        "labels": ", ".join(map(str, sorted(pd.Series(y).dropna().unique()))),
        "has_metadata": bool(len(meta_df) == len(y)),
        "n_unique_objects": (
            int(meta_df["object_id"].nunique())
            if "object_id" in meta_df.columns
            else np.nan
        ),
        "n_unique_images": (
            int(meta_df["source_image"].nunique())
            if "source_image" in meta_df.columns
            else np.nan
        ),
        "n_nan_values": int(np.isnan(X).sum()) if numeric else np.nan,
        "nan_rate": float(np.isnan(X).mean()) if numeric else np.nan,
        "global_min": float(np.nanmin(X)) if X.size else np.nan,
        "global_max": float(np.nanmax(X)) if X.size else np.nan,
        "global_mean": float(np.nanmean(X)) if X.size else np.nan,
        "global_std": float(np.nanstd(X)) if X.size else np.nan,
    }
    return row, meta_df


def summarize_preprocessing_output(
    X_preprocessed,
    *,
    preprocessing_name: str,
    steps: Sequence[str],
    sg_window_length: int,
    sg_polyorder: int,
) -> dict:
    """Return the preprocessing summary row used by notebook 02."""
    X_preprocessed = np.asarray(X_preprocessed)
    return {
        "preprocessing": preprocessing_name,
        "steps": " + ".join(steps),
        "n_observations": int(X_preprocessed.shape[0]),
        "n_features": int(X_preprocessed.shape[1]),
        "global_mean": float(np.nanmean(X_preprocessed)),
        "global_std": float(np.nanstd(X_preprocessed)),
        "global_min": float(np.nanmin(X_preprocessed)),
        "global_max": float(np.nanmax(X_preprocessed)),
        "nan_rate": float(np.mean(~np.isfinite(X_preprocessed))),
        "sg_window_length": int(sg_window_length),
        "sg_polyorder": int(sg_polyorder),
    }
