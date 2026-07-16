"""
Small generic utilities used across the src package.

Keep this module intentionally small: it should contain only helpers that are
not specific to PCA, SIMCA, spectroscopy, segmentation, or plotting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

def make_wavelengths(start_nm:int, end_nm:int, original_bands:int, n_remove_start:int, n_stop_end:int=None):
    """
    Build wavelength axis after removing the first and last noisy bands.
    Raw data: 69 bands from 889 to 1702 nm.
    Processed data: bands [n_remove_start:n_stop_end] only.
    """
    full_axis = np.linspace(float(start_nm), float(end_nm), int(original_bands))
    return full_axis[int(n_remove_start):] if n_stop_end is None else full_axis[int(n_remove_start):int(n_stop_end)]


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


def save_parquet_if_nonempty(
    df,
    path,
    index: bool = False,
    compression: str = "zstd",
    optimize: bool = True,
):
    """Save a parquet file only if the dataframe is not empty."""
    if df is None or len(df) == 0:
        return None
    return save_parquet(
        df,
        path=path,
        index=index,
        compression=compression,
        optimize=optimize,
    )


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


def filter_dataframe_by_values(
    df: pd.DataFrame,
    filters: dict,
    strict: bool = True,
) -> pd.DataFrame:
    """Filter a dataframe with dict-style equality / isin filters."""
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    for col, allowed in filters.items():
        if col not in out.columns:
            if strict:
                raise KeyError(f"Column not found in dataframe: {col}")
            continue
        if allowed is None:
            continue
        allowed_values = list(allowed) if isinstance(allowed, (list, tuple, set, np.ndarray, pd.Index)) else [allowed]
        mask = mask & out[col].isin(allowed_values)

    return out.loc[mask].copy()


def list_result_files(directory: str | Path) -> pd.DataFrame:
    """List result files with sizes to monitor output bloat in notebooks."""
    root = Path(directory)
    rows = []

    for path in root.rglob("*"):
        if path.is_file():
            rows.append(
                {
                    "file": str(path.relative_to(root)),
                    "suffixes": "".join(path.suffixes),
                    "size_mb": path.stat().st_size / 1024**2,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["file", "suffixes", "size_mb"])

    return pd.DataFrame(rows).sort_values("size_mb", ascending=False).reset_index(drop=True)


def is_missing_value(x: Any) -> bool:
    """Robust scalar missing-value test used for row/config helpers."""
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def first_available_value(row: pd.Series, columns: list[str], default=None):
    """Return the first non-missing value found in a row among candidate columns."""
    for col in columns:
        if col in row.index:
            value = row[col]
            if not is_missing_value(value):
                return value
    return default


def to_numeric_metrics(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def parse_preprocessing_steps(value) -> list[str]:
    """Normalize preprocessing definitions stored as strings, lists or JSON."""
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except Exception:
            pass

    normalized = (
        text.lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("savitzky_golay_smooth", "sg_smooth")
        .replace("savitzky_golay_d1", "sg_d1")
        .replace("savitzky_golay_d2", "sg_d2")
    )

    raw_parts = [
        part
        for part in normalized.split("_")
        if part
    ]

    steps = []
    i = 0

    while i < len(raw_parts):
        current = raw_parts[i]

        if (
            current == "sg"
            and i + 1 < len(raw_parts)
            and raw_parts[i + 1] in {"smooth", "d1", "d2"}
        ):
            steps.append(f"sg_{raw_parts[i + 1]}")
            i += 2
            continue

        steps.append(current)
        i += 1

    return steps


def merge_config_metadata(
    results_df: pd.DataFrame,
    config_df: pd.DataFrame,
    id_col: str = "selected_config_id",
    columns: Sequence[str] | None = None,
    overwrite: bool = False,
    validate: str | None = "many_to_one",
) -> pd.DataFrame:
    """Attach one configuration table to result rows without notebook helpers.

    Existing result columns are preserved by default. Set ``overwrite=True`` to
    replace them with values from ``config_df``.
    """
    if id_col not in results_df.columns:
        raise KeyError(f"Missing {id_col!r} in results_df.")
    if id_col not in config_df.columns:
        raise KeyError(f"Missing {id_col!r} in config_df.")

    right = config_df.drop_duplicates(id_col).copy()
    if columns is not None:
        keep = [id_col] + [column for column in columns if column in right.columns and column != id_col]
        right = right[keep]

    overlapping = [column for column in right.columns if column in results_df.columns and column != id_col]
    if overwrite:
        left = results_df.drop(columns=overlapping)
        return left.merge(right, on=id_col, how="left", validate=validate)

    # Only merge columns not already present.
    right = right[[column for column in right.columns if column == id_col or column not in results_df.columns]]
    return results_df.merge(right, on=id_col, how="left", validate=validate)
