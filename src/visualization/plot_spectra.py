from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import as_1d_array, filter_records, is_float_like, wavelength_axis
from src.visualization.common import (
    apply_project_theme,
    color_with_alpha,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
)


def mean_spectrum_from_cube(cube: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return a mean spectrum from a hyperspectral cube, optionally inside a mask."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("Expected cube with shape (H, W, B).")
    pixels = cube.reshape(-1, cube.shape[2])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != cube.shape[:2]:
            raise ValueError("mask shape must match cube spatial dimensions.")
        pixels = pixels[mask.ravel()]
    if len(pixels) == 0:
        return np.full(cube.shape[2], np.nan)
    return np.nanmean(pixels, axis=0)


def _detect_spectral_columns(df: pd.DataFrame) -> list[Any]:
    """Detect wavelength-like dataframe columns instead of assuming ``columns[3:]``."""
    numeric_like = [column for column in df.columns if is_float_like(column)]
    if numeric_like:
        return numeric_like

    common_metadata = {
        "object_id",
        "source_image",
        "source_clean_key",
        "batch",
        "label",
        "object_nut_type",
        "sample_kind",
        "subset",
        "row",
        "col",
        "area_pixels",
    }
    candidate = [
        column
        for column in df.columns
        if column not in common_metadata and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not candidate:
        raise ValueError(
            "No spectral columns could be detected. Pass spectral_cols explicitly."
        )
    return candidate


def extract_spectral_matrix(
    data,
    keys: Sequence[str] | None = None,
    spectral_cols: Sequence[Any] | None = None,
    label_col: str | None = None,
    name_col: str | None = None,
    spectrum_field: str = "mean_spectrum",
):
    """Accept arrays, mappings of cubes/objects/spectra, or a DataFrame."""
    if isinstance(data, pd.DataFrame):
        df = data
        if spectral_cols is None:
            spectral_cols = _detect_spectral_columns(df)
        spectral_cols = list(spectral_cols)
        X = df.loc[:, spectral_cols].to_numpy(dtype=float)
        labels = df[label_col].to_numpy() if label_col and label_col in df.columns else None
        names = df[name_col].to_numpy() if name_col and name_col in df.columns else None
        wavelengths = (
            np.asarray(spectral_cols, dtype=float)
            if all(is_float_like(column) for column in spectral_cols)
            else None
        )
        return X, labels, names, wavelengths

    if isinstance(data, Mapping):
        keys = list(data.keys()) if keys is None else list(keys)
        X_list, labels, names = [], [], []
        inferred_wavelengths = None
        for key in keys:
            item = data[key]
            if isinstance(item, Mapping) and spectrum_field in item:
                X_list.append(np.asarray(item[spectrum_field], dtype=float))
                labels.append(item.get("object_nut_type", item.get("label", "all")))
                names.append(str(key))
                if inferred_wavelengths is None and item.get("wavelengths") is not None:
                    inferred_wavelengths = np.asarray(item["wavelengths"])
                continue

            arr = np.asarray(item)
            if arr.ndim == 3:
                X_list.append(mean_spectrum_from_cube(arr))
                labels.append(str(key))
                names.append(str(key))
            elif arr.ndim == 1:
                X_list.append(arr.astype(float))
                labels.append(str(key))
                names.append(str(key))
            else:
                raise ValueError(
                    f"Mapping item {key!r} must be a cube, spectrum, or object dict."
                )
        if not X_list:
            raise ValueError("No spectra found in mapping.")
        return np.vstack(X_list), np.asarray(labels), np.asarray(names), inferred_wavelengths

    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("Array data must be 1D or 2D.")
    return X, None, None, None


def _stratified_indices(
    labels: np.ndarray,
    max_spectra: int,
    random_state: int,
) -> np.ndarray:
    """Sample approximately the same number of spectra from each label."""
    rng = np.random.default_rng(random_state)
    groups = ordered_unique(labels)
    if not groups:
        return np.arange(min(len(labels), max_spectra))
    per_group = max(1, int(np.ceil(max_spectra / len(groups))))
    selected: list[int] = []
    for group in groups:
        idx = np.flatnonzero(labels == group)
        if len(idx) > per_group:
            idx = rng.choice(idx, size=per_group, replace=False)
        selected.extend(idx.tolist())
    if len(selected) > max_spectra:
        selected = rng.choice(np.asarray(selected), size=max_spectra, replace=False).tolist()
    return np.asarray(sorted(selected), dtype=int)


def _summary_curves(X: np.ndarray, reducer: str, ci: float = 0.95):
    if reducer in {"mean", "mean_std"}:
        center = np.nanmean(X, axis=0)
        if reducer == "mean":
            return center, None, None
        spread = np.nanstd(X, axis=0)
        return center, center - spread, center + spread
    if reducer == "median_iqr":
        center = np.nanmedian(X, axis=0)
        return center, np.nanquantile(X, 0.25, axis=0), np.nanquantile(X, 0.75, axis=0)
    if reducer == "mean_ci":
        center = np.nanmean(X, axis=0)
        n_eff = np.sum(np.isfinite(X), axis=0).clip(min=1)
        sem = np.nanstd(X, axis=0, ddof=1) / np.sqrt(n_eff)
        # Normal approximation is sufficient for visual reporting.
        z = 1.96 if np.isclose(ci, 0.95) else 1.0
        return center, center - z * sem, center + z * sem
    raise ValueError(
        "reducer must be 'none', 'mean', 'mean_std', 'median_iqr', or 'mean_ci'."
    )


def _add_summary_band(
    fig: go.Figure,
    x,
    lower,
    upper,
    color: str,
    name: str,
    row: int | None = None,
    col: int | None = None,
):
    kwargs = {} if row is None else {"row": row, "col": col}
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        **kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=color_with_alpha(color, 0.18),
            name=f"{name} spread",
            showlegend=False,
            hoverinfo="skip",
        ),
        **kwargs,
    )


def plot_spectra(
    data,
    wavelengths=None,
    labels=None,
    names=None,
    keys: Sequence[str] | None = None,
    spectral_cols: Sequence[Any] | None = None,
    label_col: str | None = None,
    name_col: str | None = None,
    spectrum_field: str = "mean_spectrum",
    mode: str = "lines",
    reducer: str = "none",
    show_std: bool = False,
    max_spectra: int | None = None,
    random_state: int = 42,
    stratified_sampling: bool = True,
    color_map: dict[str, str] | None = None,
    category_order: Sequence[str] | None = None,
    facet_values=None,
    facet_name: str = "group",
    facet_col_wrap: int = 3,
    ci: float = 0.95,
    title: str = "Spectra",
    y_title: str = "Value",
    width: int = 950,
    height: int = 550,
    show: bool = True,
):
    """Generic spectral plot with stable colours and true uncertainty bands."""
    X, inferred_labels, inferred_names, inferred_wavelengths = extract_spectral_matrix(
        data,
        keys=keys,
        spectral_cols=spectral_cols,
        label_col=label_col,
        name_col=name_col,
        spectrum_field=spectrum_field,
    )
    if wavelengths is None:
        wavelengths = inferred_wavelengths
    if labels is None:
        labels = inferred_labels
    if names is None:
        names = inferred_names

    n, p = X.shape
    labels = as_1d_array(labels, n, "all").astype(str)
    names = as_1d_array(names, n, "").astype(str)
    facets = as_1d_array(facet_values, n, "all").astype(str)

    if max_spectra is not None and reducer == "none" and n > int(max_spectra):
        if stratified_sampling:
            idx = _stratified_indices(labels, int(max_spectra), random_state)
        else:
            idx = np.random.default_rng(random_state).choice(
                n, size=int(max_spectra), replace=False
            )
        X, labels, names, facets = X[idx], labels[idx], names[idx], facets[idx]
        n = len(idx)

    x, x_title = wavelength_axis(p, wavelengths)
    groups = (
        [str(value) for value in category_order]
        if category_order is not None
        else ordered_unique(labels)
    )
    if color_map is None:
        color_map = make_dynamic_color_map(groups)

    facet_groups = ordered_unique(facets)
    use_facets = facet_values is not None and len(facet_groups) > 1
    if use_facets:
        n_cols = min(max(1, int(facet_col_wrap)), len(facet_groups))
        n_rows = ceil(len(facet_groups) / n_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{facet_name}={value}" for value in facet_groups],
            shared_xaxes=True,
        )
    else:
        fig = go.Figure()
        n_cols = n_rows = 1

    for facet_index, facet_value in enumerate(facet_groups):
        facet_mask = facets == facet_value
        row = facet_index // n_cols + 1
        col = facet_index % n_cols + 1
        trace_kwargs = {} if not use_facets else {"row": row, "col": col}

        if reducer == "none":
            for index in np.flatnonzero(facet_mask):
                label = labels[index]
                trace_name = names[index] or f"spectrum {index}"
                if label != "all" and label not in trace_name:
                    trace_name = f"{label} | {trace_name}"
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=X[index],
                        mode=mode,
                        name=trace_name,
                        legendgroup=label,
                        line=dict(color=color_map.get(label)),
                        opacity=0.40,
                        showlegend=not use_facets or facet_index == 0,
                    ),
                    **trace_kwargs,
                )
        else:
            effective_reducer = "mean_std" if show_std and reducer == "mean" else reducer
            for label in groups:
                group_mask = facet_mask & (labels == label)
                if not np.any(group_mask):
                    continue
                center, lower, upper = _summary_curves(X[group_mask], effective_reducer, ci=ci)
                color = color_map.get(label, "lightgray")
                if lower is not None and upper is not None:
                    _add_summary_band(
                        fig,
                        x,
                        lower,
                        upper,
                        color,
                        label,
                        None if not use_facets else row,
                        None if not use_facets else col,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=center,
                        mode=mode,
                        name=label,
                        legendgroup=label,
                        line=dict(color=color, width=2.5),
                        showlegend=not use_facets or facet_index == 0,
                    ),
                    **trace_kwargs,
                )
        if use_facets:
            fig.update_xaxes(title_text=x_title, row=row, col=col)
            fig.update_yaxes(title_text=y_title, row=row, col=col)

    fig.update_layout(
        title=title,
        xaxis_title=None if use_facets else x_title,
        yaxis_title=None if use_facets else y_title,
        width=width,
        height=max(height, 340 * n_rows),
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_spectral_distribution(
    cube,
    title: str = "Spectral distribution",
    n_pixels: int = 2000,
    wavelengths=None,
    mask: np.ndarray | None = None,
    random_state: int = 42,
    y_title: str = "Reflectance",
    reducer: str = "mean_std",
    show: bool = True,
):
    """Sample cube pixels inside an optional foreground mask and plot a summary."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("cube must have shape (H, W, B).")

    pixels = cube.reshape(-1, cube.shape[2])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != cube.shape[:2]:
            raise ValueError("mask shape must match cube spatial dimensions.")
        pixels = pixels[mask.ravel()]
    if len(pixels) == 0:
        raise ValueError("No pixels available after applying the mask.")

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(pixels), size=min(int(n_pixels), len(pixels)), replace=False)
    return plot_spectra(
        pixels[idx],
        wavelengths=wavelengths,
        reducer=reducer,
        title=title,
        y_title=y_title,
        show=show,
    )


def plot_object_spectra(
    object_db: Mapping[str, Mapping[str, Any]],
    source_image: str | None = None,
    nut_type: str | None = None,
    spectrum_field: str = "mean_spectrum",
    reducer: str = "none",
    title: str | None = None,
    show: bool = True,
    **kwargs,
):
    """Plot spectra from objects in ``object_db``."""
    objects = filter_records(
        object_db,
        source_clean_key=source_image,
        object_nut_type=nut_type,
    )
    if not objects:
        raise ValueError("No object found with these filters.")

    X = np.vstack([obj[spectrum_field] for _, obj in objects])
    labels = np.asarray([obj.get("object_nut_type", "unknown") for _, obj in objects])
    names = np.asarray([object_id for object_id, _ in objects])
    wavelengths = objects[0][1].get("wavelengths")

    if title is None:
        suffix = ""
        if source_image is not None:
            suffix += f" — source={source_image}"
        if nut_type is not None:
            suffix += f" — type={nut_type}"
        title = f"Object spectra{suffix}"

    return plot_spectra(
        X,
        wavelengths=wavelengths,
        labels=labels,
        names=names,
        reducer=reducer,
        title=title,
        show=show,
        **kwargs,
    )


def plot_spectra_by_batch(
    X,
    labels,
    batches,
    wavelengths=None,
    preprocessing_name: str | None = None,
    reducer: str = "mean_std",
    title: str | None = None,
    show: bool = True,
    **kwargs,
):
    """Convenience wrapper moved out of notebooks for class-by-batch spectra."""
    labels = np.asarray(labels).astype(str)
    batches = np.asarray(batches).astype(str)
    if len(labels) != len(batches) or len(labels) != len(X):
        raise ValueError("X, labels and batches must have the same number of rows.")
    if title is None:
        suffix = f" — {preprocessing_name}" if preprocessing_name else ""
        title = f"Spectra by batch{suffix}"
    return plot_spectra(
        X,
        wavelengths=wavelengths,
        labels=labels,
        facet_values=batches,
        facet_name="batch",
        reducer=reducer,
        title=title,
        show=show,
        **kwargs,
    )
