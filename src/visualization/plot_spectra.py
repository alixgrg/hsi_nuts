from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils import (
    as_1d_array,
    filter_records,
    is_float_like,
    wavelength_axis,
)
from src.visualization.common import show_or_return


def mean_spectrum_from_cube(cube: np.ndarray) -> np.ndarray:
    """Return mean spectrum from a hyperspectral cube."""
    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("Expected cube with shape (H, W, B).")

    return np.nanmean(cube.reshape(-1, cube.shape[2]), axis=0)


def extract_spectral_matrix(
    data,
    keys: Sequence[str] | None = None,
    spectral_cols: Sequence[Any] | None = None,
    label_col: str | None = None,
    name_col: str | None = None,
    spectrum_field: str = "mean_spectrum",
):
    """
    Accept array, dict of cubes/objects/spectra, or DataFrame.

    Returns
    -------
    X, labels, names, wavelengths
    """
    if isinstance(data, pd.DataFrame):
        df = data

        if spectral_cols is None:
            spectral_cols = df.columns[3:]

        X = df.loc[:, spectral_cols].to_numpy(dtype=float)
        labels = df[label_col].to_numpy() if label_col and label_col in df.columns else None
        names = df[name_col].to_numpy() if name_col and name_col in df.columns else None

        wavelengths = (
            np.asarray(spectral_cols, dtype=float)
            if all(is_float_like(c) for c in spectral_cols)
            else None
        )

        return X, labels, names, wavelengths

    if isinstance(data, Mapping):
        if keys is None:
            keys = list(data.keys())

        X_list, labels, names = [], [], []

        for key in keys:
            item = data[key]

            if isinstance(item, Mapping) and spectrum_field in item:
                X_list.append(np.asarray(item[spectrum_field], dtype=float))
                labels.append(item.get("object_nut_type", item.get("label", "all")))
                names.append(str(key))

            else:
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
                        f"Mapping item {key!r} must be a cube, "
                        "spectrum, or object dict."
                    )

        return np.vstack(X_list), np.asarray(labels), np.asarray(names), None

    X = np.asarray(data, dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError("Array data must be 1D or 2D.")

    return X, None, None, None


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
    title: str = "Spectra",
    y_title: str = "Value",
    width: int = 950,
    height: int = 550,
    show: bool = True,
):
    """
    Generic spectral plot.

    data can be ndarray, DataFrame, object_db, image_db, cube dict, etc.
    reducer:
        - "none"
        - "mean"
        - "mean_std"
    """
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

    if max_spectra is not None and reducer == "none":
        X = X[:max_spectra]
        if labels is not None:
            labels = np.asarray(labels)[:max_spectra]
        if names is not None:
            names = np.asarray(names)[:max_spectra]

    n, p = X.shape

    x, x_title = wavelength_axis(p, wavelengths)
    labels = as_1d_array(labels, n, "all").astype(str)
    names = as_1d_array(names, n, "").astype(str)

    fig = go.Figure()

    if reducer == "none":
        for i in range(n):
            trace_name = names[i] if names[i] else f"spectrum {i}"

            if labels[i] != "all" and labels[i] not in trace_name:
                trace_name = f"{labels[i]} | {trace_name}"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=X[i],
                    mode=mode,
                    name=trace_name,
                )
            )

    elif reducer in {"mean", "mean_std"}:
        for lab in np.unique(labels):
            Xg = X[labels == lab]
            mu = np.nanmean(Xg, axis=0)
            sd = np.nanstd(Xg, axis=0)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mu,
                    mode=mode,
                    name=str(lab),
                )
            )

            if reducer == "mean_std" or show_std:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mu + sd,
                        mode="lines",
                        name=f"{lab} +1 std",
                        line=dict(dash="dash"),
                        opacity=0.35,
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mu - sd,
                        mode="lines",
                        name=f"{lab} -1 std",
                        line=dict(dash="dash"),
                        opacity=0.35,
                        showlegend=False,
                    )
                )

    else:
        raise ValueError("reducer must be 'none', 'mean' or 'mean_std'.")

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=width,
        height=height,
    )

    return show_or_return(fig, show)


def plot_spectral_distribution(
    cube,
    title: str = "Spectral distribution",
    n_pixels: int = 2000,
    wavelengths=None,
    random_state: int = 42,
    y_title: str = "Reflectance",
    show: bool = True,
):
    """Sample pixels from a cube and plot mean ± std spectrum."""
    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("cube must have shape (H, W, B).")

    rng = np.random.default_rng(random_state)
    pixels = cube.reshape(-1, cube.shape[2])

    idx = rng.choice(
        pixels.shape[0],
        size=min(int(n_pixels), pixels.shape[0]),
        replace=False,
    )

    return plot_spectra(
        pixels[idx],
        wavelengths=wavelengths,
        reducer="mean_std",
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
):
    """Plot spectra from objects in object_db."""
    objects = filter_records(
        object_db,
        source_clean_key=source_image,
        object_nut_type=nut_type,
    )

    if not objects:
        raise ValueError("No object found with these filters.")

    X = np.vstack([obj[spectrum_field] for _, obj in objects])
    labels = np.asarray([obj.get("object_nut_type", "unknown") for _, obj in objects])
    names = np.asarray([oid for oid, _ in objects])
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
    )