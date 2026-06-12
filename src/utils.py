"""
Small generic utilities used across the src package.

Keep this module intentionally small: it should contain only helpers that are
not specific to PCA, SIMCA, spectroscopy, segmentation, or plotting.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any

import numpy as np


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

def make_wavelengths(start_nm:int, end_nm:int, original_bands:int, n_remove:int):
    """
    Build wavelength axis after removing the first noisy bands.
    Raw data: 69 bands from 889 to 1702 nm.
    Processed data: bands [n_remove:] only.
    """
    full_axis = np.linspace(float(start_nm), float(end_nm), int(original_bands))
    return full_axis[int(n_remove):]


def save_pickle(obj, path):
    """Save a Python object with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path
