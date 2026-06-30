"""
Small generic utilities used across the src package.

Keep this module intentionally small: it should contain only helpers that are
not specific to PCA, SIMCA, spectroscopy, segmentation, or plotting.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle
import json
from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path


def as_2d_array(X, dtype=float) -> np.ndarray:
    """
    Convert X to a 2D numpy array.

    - 1D input becomes shape (1, n_features)
    - 2D input is returned as-is after conversion
    - other dimensions raise an error
    """
    arr = np.asarray(X, dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape}.")
    return arr


def as_1d_array(values, n: int | None = None, default: Any = None, dtype=None) -> np.ndarray:
    """
    Convert values to a 1D numpy array, optionally checking its length.

    If values is None and n is provided, returns an array filled with default.
    """
    if values is None:
        if n is None:
            return np.asarray([], dtype=dtype)
        return np.asarray([default] * n, dtype=dtype)

    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        arr = arr.ravel()
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"Expected {n} values, got {arr.shape[0]}.")
    return arr


def as_list(x, none_as_empty: bool = True) -> list:
    """
    Normalize scalars / tuples / arrays to a Python list.
    """
    if x is None:
        return [] if none_as_empty else [None]
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]


def check_same_length(**arrays) -> int:
    """
    Check that all non-None arrays have the same first dimension.

    Returns the common length.
    """
    lengths = {
        name: len(value)
        for name, value in arrays.items()
        if value is not None
    }
    if not lengths:
        return 0
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Arrays do not have the same length: {lengths}")
    return next(iter(unique_lengths))


def is_float_like(x) -> bool:
    """
    True if x can be safely converted to float.
    """
    try:
        float(x)
        return True
    except Exception:
        return False


def safe_positive(x, eps: float = 1e-12) -> np.ndarray:
    """
    Clip values from below by eps.
    Useful to avoid division by zero in distances or normalizations.
    """
    return np.maximum(np.asarray(x, dtype=float), eps)


def safe_divide(numerator, denominator, eps: float = 1e-12):
    """
    Divide numerator by denominator after clipping denominator away from zero.
    """
    return np.asarray(numerator, dtype=float) / safe_positive(denominator, eps=eps)


def mask_value_to_nan(arr, mask_value=0) -> np.ndarray:
    """
    Convert a chosen mask value to NaN, useful before plotting overlays.
    """
    out = np.asarray(arr, dtype=float).copy()
    out[out == mask_value] = np.nan
    return out


def filter_records(
    records: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[str, str] | None = None,
    return_items: bool = True,
    **filters,
):
    """
    Filter a dictionary of record dictionaries by field values.

    Parameters
    ----------
    records:
        Mapping such as object_db.
    aliases:
        Optional mapping from user-facing filter names to real record fields.
        Example: {"nut_type": "object_nut_type"}.
    return_items:
        If True, return [(record_id, record), ...]. If False, return only ids.
    **filters:
        field=value or field=[allowed_values]. None filters are ignored.

    Examples
    --------
    filter_records(object_db, nut_type="almond", split="train_minimal")
    """
    aliases = aliases or {}
    selected = []

    for record_id, record in records.items():
        ok = True
        for key, allowed in filters.items():
            if allowed is None:
                continue
            field = aliases.get(key, key)
            value = record.get(field)
            if isinstance(allowed, (list, tuple, set, np.ndarray)):
                ok = ok and value in allowed
            else:
                ok = ok and value == allowed
            if not ok:
                break
        if ok:
            selected.append((record_id, record) if return_items else record_id)

    return selected


def wavelength_axis(n_features: int, wavelengths=None, default_label: str = "Band index"):
    """
    Return an x-axis and label for spectral plots.
    """
    if wavelengths is None:
        return np.arange(n_features), default_label
    return np.asarray(wavelengths), "Wavelength (nm)"

def make_wavelengths(start_nm:int, end_nm:int, original_bands:int, n_remove_start:int, n_stop_end:int):
    """
    Build wavelength axis after removing the first and last noisy bands.
    Raw data: 69 bands from 889 to 1702 nm.
    Processed data: bands [n_remove_start:n_stop_end] only.
    """
    full_axis = np.linspace(float(start_nm), float(end_nm), int(original_bands))
    return full_axis[int(n_remove_start):int(n_stop_end)]


def save_pickle(obj, path):
    """Save a Python object with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def ensure_parent_dir(path):
    """Create parent directory and return Path."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def optimize_dataframe_for_parquet(
    df,
    float_dtype: str = "float32",
    object_to_category: bool = True,
    category_max_ratio: float = 0.50,
):
    """
    Reduce DataFrame memory before Parquet export.

    - float64 -> float32
    - integer columns -> smaller integer dtype
    - low-cardinality object columns -> category
    """
    out = df.copy()

    for col in out.columns:
        s = out[col]

        if pd.api.types.is_float_dtype(s):
            out[col] = s.astype(float_dtype)

        elif pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")

        elif object_to_category and pd.api.types.is_object_dtype(s):
            s_non_na = s.dropna()

            simple_types = (
                str,
                int,
                float,
                bool,
                np.integer,
                np.floating,
                np.bool_,
            )

            only_simple_scalars = s_non_na.map(
                lambda x: isinstance(x, simple_types)
            ).all()

            if only_simple_scalars:
                n = len(s)
                nunique = s.nunique(dropna=True)

                if n > 0 and nunique / n <= category_max_ratio:
                    try:
                        out[col] = s.astype("category")
                    except Exception:
                        pass

    return out


def make_dataframe_parquet_safe(df):
    """
    Convert complex object/category columns to JSON strings before Parquet export.

    This avoids pyarrow errors with nested dictionaries/lists/arrays.
    Compatible with recent pandas versions.
    """

    def _json_default_for_parquet(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        return str(x)

    def _safe_value(x):
        if x is None:
            return None

        if isinstance(x, (dict, list, tuple, set, np.ndarray)):
            return json.dumps(
                x,
                default=_json_default_for_parquet,
                ensure_ascii=False,
            )

        if isinstance(x, Path):
            return str(x)

        if isinstance(x, np.generic):
            return x.item()

        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        return x

    out = df.copy()
    complex_types = (dict, list, tuple, set, np.ndarray, Path)

    for col in out.columns:
        s = out[col]

        is_object = pd.api.types.is_object_dtype(s)
        is_category = isinstance(s.dtype, pd.CategoricalDtype)

        if is_object or is_category:
            s_obj = s.astype("object")

            has_complex_values = s_obj.map(
                lambda x: isinstance(x, complex_types)
            ).any()

            if has_complex_values:
                out[col] = s_obj.map(_safe_value).astype("string")

    return out


def save_parquet(
    df,
    path,
    index: bool = False,
    compression: str = "zstd",
    optimize: bool = True,
):
    """
    Save a pandas DataFrame as compressed Parquet.

    Use this for all result tables:
    - summaries
    - metrics
    - grid-search outputs
    - errors
    - selected configurations
    """
    path = ensure_parent_dir(path)

    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")

    df_safe = make_dataframe_parquet_safe(df)

    df_to_save = (
        optimize_dataframe_for_parquet(df_safe)
        if optimize
        else df_safe.copy()
    )

    df_to_save.to_parquet(
        path,
        index=index,
        compression=compression,
    )

    return path


def load_parquet(path):
    """Load a pandas DataFrame saved with save_parquet."""
    path = Path(path)
    return pd.read_parquet(path)


def save_empty_parquet(path):
    """Create an empty parquet table when a result dataframe is empty."""
    return save_parquet(
        pd.DataFrame(),
        path=path,
        index=False,
        optimize=False,
    )


def row_value(row: pd.Series, col: str, default=None):
    if col not in row.index:
        return default
    value = row[col]
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def row_int(row: pd.Series, col: str, default: int) -> int:
    return int(row_value(row, col, default))


def row_float(row: pd.Series, col: str, default: float) -> float:
    return float(row_value(row, col, default))


def row_str(row: pd.Series, col: str, default: str) -> str:
    return str(row_value(row, col, default))


