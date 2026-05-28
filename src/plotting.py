from typing import Sequence, Any, Mapping
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.stats import hotelling_t2, q_residuals
from src.utils import (
    mask_value_to_nan, 
    as_1d_array, 
    wavelength_axis, 
    is_float_like,
    filter_records,
)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------



def _show_or_return(fig: go.Figure, show: bool = True):
    if show:
        fig.show()
    return fig


def _customdata(n: int, **metadata) -> tuple[np.ndarray, str]:
    names = [k for k, v in metadata.items() if v is not None]
    if not names:
        return np.empty((n, 0), dtype=str), ""
    cols = [as_1d_array(metadata[k], n, "").astype(str) for k in names]
    data = np.stack(cols, axis=1)
    hover = "".join(f"{name}: %{{customdata[{i}]}}<br>" for i, name in enumerate(names))
    return data, hover


def _mean_spectrum_from_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("Expected a hyperspectral cube with shape (H, W, B).")
    return np.nanmean(cube.reshape(-1, cube.shape[2]), axis=0)

def _extract_spectral_matrix(
    data,
    keys: Sequence[str] | None = None,
    spectral_cols: Sequence[Any] | None = None,
    label_col: str | None = None,
    name_col: str | None = None,
    spectrum_field: str = "mean_spectrum",
):
    """Accept an array, a dict of cubes/objects, or a dataframe and return X, labels, names."""
    if isinstance(data, pd.DataFrame):
        df = data
        if spectral_cols is None:
            # Default used in your Excel-like tables: metadata in first 3 columns.
            spectral_cols = df.columns[3:]
        X = df.loc[:, spectral_cols].to_numpy(dtype=float)
        labels = df[label_col].to_numpy() if label_col and label_col in df.columns else None
        names = df[name_col].to_numpy() if name_col and name_col in df.columns else None
        wavelengths = np.asarray(spectral_cols, dtype=float) if all(is_float_like(c) for c in spectral_cols) else None
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
                    X_list.append(_mean_spectrum_from_cube(arr))
                    labels.append(str(key))
                    names.append(str(key))
                elif arr.ndim == 1:
                    X_list.append(arr.astype(float))
                    labels.append(str(key))
                    names.append(str(key))
                else:
                    raise ValueError(f"Mapping item {key!r} must be a cube, spectrum, or object dict.")
        return np.vstack(X_list), np.asarray(labels), np.asarray(names), None

    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X, None, None, None



# -----------------------------------------------------------------------------
# Images and hyperspectral cubes
# -----------------------------------------------------------------------------

# Generic version of plot_band_slider
def plot_hypercube_band_slider(
    cube: np.ndarray,
    wavelengths=None,
    title: str = "Hyperspectral image",
    value_name: str = "Value",
    colorscale: str = "Viridis",
    width: int = 700,
    height: int = 700,
    show: bool = True,
):
    """Interactive slider over spectral bands of a HSI cube, shape (H, W, B)."""
    cube = np.asarray(cube)
    n_bands = cube.shape[2]
    wavelengths = None if wavelengths is None else np.asarray(wavelengths)

    fig = go.Figure()
    for i in range(n_bands):
        label = f"band {i}" if wavelengths is None else f"{wavelengths[i]:.1f} nm"
        fig.add_trace(
            go.Heatmap(
                z=cube[:, :, i],
                visible=(i == 0),
                colorscale=colorscale,
                colorbar=dict(title=value_name),
                hovertemplate=(
                    "row: %{y}<br>col: %{x}<br>"
                    f"{label}<br>{value_name}: %{{z}}<extra></extra>"
                ),
            )
        )

    steps = []
    for i in range(n_bands):
        label = str(i) if wavelengths is None else f"{wavelengths[i]:.0f}"
        title_i = f"{title} — band {i}" + ("" if wavelengths is None else f" — {wavelengths[i]:.1f} nm")
        steps.append(
            dict(
                method="update",
                args=[{"visible": [j == i for j in range(n_bands)]}, {"title": title_i}],
                label=label,
            )
        )
    fig.update_layout(
        title=f"{title} — band 0",
        sliders=[dict(active=0, currentvalue={"prefix": "Band: "}, steps=steps)],
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )
    return _show_or_return(fig, show)


# Generic version of plot_db_image
def plot_image2d(
    z: np.ndarray,
    title: str = "Image",
    colorscale: str = "Viridis",
    colorbar_title: str = "Value",
    width: int = 800,
    height: int = 700,
    reverse_y: bool = True,
    show: bool = True,
):
    """Generic 2D heatmap."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
    )
    if reverse_y:
        fig.update_yaxes(autorange="reversed", scaleanchor="x")
    return _show_or_return(fig, show)

# Generic version of plot_db_labels_overlay, part of plot_simca_object_map
def plot_image_overlay(
    background: np.ndarray,
    overlay: np.ndarray,
    title: str = "Image overlay",
    background_colorscale: str = "Gray",
    overlay_colorscale: Any = "Turbo",
    background_title: str = "Background",
    overlay_title: str = "Overlay",
    overlay_mask_value=0,
    alpha: float = 0.45,
    width: int = 850,
    height: int = 750,
    overlay_colorbar: dict | None = None,
    show: bool = True,
):
    """Generic image + semi-transparent overlay."""
    overlay_plot = mask_value_to_nan(overlay, mask_value=overlay_mask_value)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale=background_colorscale,
            showscale=True,
            colorbar=dict(title=background_title),
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=overlay_plot,
            colorscale=overlay_colorscale,
            opacity=alpha,
            showscale=True,
            colorbar=overlay_colorbar or dict(title=overlay_title, x=1.12),
            hovertemplate="row: %{y}<br>col: %{x}<br>overlay: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )
    return _show_or_return(fig, show)


def plot_label_overlay_from_image_db(
    image_db: Mapping[str, Mapping[str, Any]],
    image_id: str,
    base: str = "image_ref",
    band: int = 0,
    title: str | None = None,
    show: bool = True,
):
    img = image_db[image_id]
    if base == "band":
        background = img["cube"][:, :, band]
    else:
        background = img.get(base, img.get("image_ref"))
    return plot_image_overlay(background, img["labels"], title=title or f"Labels overlay — {image_id}", overlay_title="label", show=show)




# -----------------------------------------------------------------------------
# Spectra
# -----------------------------------------------------------------------------

# Generic version of plot_mean_spectra, plot_mean_spectra_from_excel, plot_two_classes, part of plot_db_object_spectra, part of plot_spectral_distribution
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

    data can be:
    - ndarray (n_spectra, n_bands) or (n_bands,)
    - dict of cubes/spectra/object records, optionally with keys=[...]
    - dataframe, with spectral_cols=[...] and optional label_col/name_col

    reducer: "none", "mean", or "mean_std".
    """
    X, inferred_labels, inferred_names, inferred_wavelengths = _extract_spectral_matrix(
        data, keys=keys, spectral_cols=spectral_cols, label_col=label_col, name_col=name_col, spectrum_field=spectrum_field
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
            fig.add_trace(go.Scatter(x=x, y=X[i], mode=mode, name=trace_name))
    elif reducer in {"mean", "mean_std"}:
        for lab in np.unique(labels):
            Xg = X[labels == lab]
            mu = np.nanmean(Xg, axis=0)
            sd = np.nanstd(Xg, axis=0)
            fig.add_trace(go.Scatter(x=x, y=mu, mode=mode, name=str(lab)))
            if reducer == "mean_std" or show_std:
                fig.add_trace(go.Scatter(x=x, y=mu + sd, mode="lines", name=f"{lab} +1 std", line=dict(dash="dash"), opacity=0.35, showlegend=False))
                fig.add_trace(go.Scatter(x=x, y=mu - sd, mode="lines", name=f"{lab} -1 std", line=dict(dash="dash"), opacity=0.35, showlegend=False))
    else:
        raise ValueError("reducer must be 'none', 'mean' or 'mean_std'.")

    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height)
    return _show_or_return(fig, show)

# Merge both ?
def plot_spectral_distribution(cube, title="Spectral distribution", n_pixels=2000, wavelengths=None, random_state=42, show=True):
    rng = np.random.default_rng(random_state)
    pixels = np.asarray(cube).reshape(-1, cube.shape[2])
    idx = rng.choice(pixels.shape[0], size=min(n_pixels, pixels.shape[0]), replace=False)
    return plot_spectra(pixels[idx], wavelengths=wavelengths, reducer="mean_std", title=title, y_title="Reflectance", show=show)


def plot_object_spectra(
    object_db: Mapping[str, Mapping[str, Any]],
    source_image: str | None = None,
    nut_type: str | None = None,
    spectrum_field: str = "mean_spectrum",
    reducer: str = "none",
    title: str | None = None,
    show: bool = True,
):
    objects = filter_records(object_db, source_clean_key=source_image, object_nut_type=nut_type)
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
    return plot_spectra(X, wavelengths=wavelengths, labels=labels, names=names, reducer=reducer, title=title, show=show)


# -----------------------------------------------------------------------------
# Object database functions
# -----------------------------------------------------------------------------

def plot_object_view(
    object_db_or_obj,
    object_id: str | None = None,
    spectrum_field: str = "mean_spectrum",
    show_spectrum: bool = True,
    show_std: bool = True,
    height: int = 500,
    width: int = 1000,
    show: bool = True,
):
    obj = object_db_or_obj[object_id] if object_id is not None else object_db_or_obj
    object_label = object_id or "object"

    fig = make_subplots(rows=1, cols=2 if show_spectrum else 1, subplot_titles=(f"{object_label} — crop", spectrum_field) if show_spectrum else (f"{object_label} — crop",))
    fig.add_trace(go.Heatmap(z=obj["image_ref_crop"], colorscale="Gray", showscale=True, colorbar=dict(title="Image ref")), row=1, col=1)
    fig.add_trace(go.Heatmap(z=mask_value_to_nan(obj["mask"], 0), colorscale="Reds", opacity=0.35, showscale=False), row=1, col=1)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)

    if show_spectrum:
        spectrum = np.asarray(obj[spectrum_field])
        x, x_title = wavelength_axis(spectrum.shape[0], obj.get("wavelengths"))
        fig.add_trace(go.Scatter(x=x, y=spectrum, mode="lines", name=spectrum_field), row=1, col=2)
        if show_std and spectrum_field == "mean_spectrum" and "std_spectrum" in obj:
            std = np.asarray(obj["std_spectrum"])
            fig.add_trace(go.Scatter(x=x, y=spectrum + std, mode="lines", name="+1 std", line=dict(dash="dash"), opacity=0.5), row=1, col=2)
            fig.add_trace(go.Scatter(x=x, y=spectrum - std, mode="lines", name="-1 std", line=dict(dash="dash"), opacity=0.5), row=1, col=2)
        fig.update_xaxes(title_text=x_title, row=1, col=2)
        fig.update_yaxes(title_text=obj.get("data_mode", "value"), row=1, col=2)

    fig.update_layout(
        title=f"{object_label} | type={obj.get('object_nut_type')} | source={obj.get('source_clean_key')} | area={obj.get('area_pixels')}",
        height=height, width=width,
    )
    return _show_or_return(fig, show)

# Generic version of plot_db_object_grid
def plot_object_grid(
    objects_or_db,
    source_image: str | None = None,
    nut_type: str | None = None,
    title: str = "Object grid",
    max_objects: int = 40,
    n_cols: int = 5,
    height_per_row: int = 220,
    width: int = 1100,
    show: bool = True,
):
    if isinstance(objects_or_db, Mapping):
        objects = filter_records(objects_or_db, source_clean_key=source_image, object_nut_type=nut_type)
    else:
        objects = list(objects_or_db)
    selected = objects[:max_objects]
    if not selected:
        raise ValueError("No object to plot.")

    n_rows = int(np.ceil(len(selected) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{oid}<br>area={obj.get('area_pixels')}" for oid, obj in selected])
    for idx, (_, obj) in enumerate(selected):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.add_trace(go.Heatmap(z=obj["image_ref_crop"], colorscale="Gray", showscale=False), row=row, col=col)
        fig.add_trace(go.Heatmap(z=mask_value_to_nan(obj["mask"], 0), colorscale="Reds", opacity=0.35, showscale=False), row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)
    fig.update_layout(title=title, height=height_per_row * n_rows, width=width)
    return _show_or_return(fig, show)


def plot_object_areas(object_db, source_image=None, nut_type=None, show=True):
    objects = filter_records(object_db, source_clean_key=source_image, object_nut_type=nut_type)
    if not objects:
        raise ValueError("No object found with these filters.")
    labels = [oid for oid, _ in objects]
    areas = [obj["area_pixels"] for _, obj in objects]
    suffix = ""
    if source_image is not None:
        suffix += f" — {source_image}"
    if nut_type is not None:
        suffix += f" — {nut_type}"
    return plot_bar_values(labels, areas, title=f"Object areas{suffix}", x_title="object_id", y_title="area_pixels", show=show)




# -----------------------------------------------------------------------------
# PCA / score-space functions
# -----------------------------------------------------------------------------

def _pca_scores_from_args(scores=None, pca_res=None):
    if scores is not None:
        return np.asarray(scores, dtype=float)
    if pca_res is not None:
        return np.asarray(pca_res["scores"], dtype=float)
    raise ValueError("Provide scores or pca_res.")


def _pca_loadings_from_args(loadings=None, pca_res=None):
    if loadings is not None:
        return np.asarray(loadings, dtype=float)
    if pca_res is not None:
        return np.asarray(pca_res["loadings"], dtype=float)
    raise ValueError("Provide loadings or pca_res.")

# Previously plot_pca_explained_variance
def plot_explained_variance(
    explained_variance_ratio,
    cumulative_explained_variance_ratio=None,
    n_components_to_show: int | None = None,
    title: str = "Explained variance",
    width: int = 850,
    height: int = 500,
    show: bool = True,
):
    evr = np.asarray(explained_variance_ratio, dtype=float)
    cum = np.cumsum(evr) if cumulative_explained_variance_ratio is None else np.asarray(cumulative_explained_variance_ratio, dtype=float)
    if n_components_to_show is not None:
        evr = evr[:n_components_to_show]
        cum = cum[:n_components_to_show]
    pcs = np.arange(1, len(evr) + 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pcs, y=evr, name="Explained variance", hovertemplate="PC%{x}<br>variance: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=pcs, y=cum, mode="lines+markers", name="Cumulative variance", hovertemplate="PC%{x}<br>cumulative: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Component", yaxis_title="Variance ratio", width=width, height=height)
    return _show_or_return(fig, show)


# Merged version of plot_pca_scores_2d, plot_pca_scores_3d
def plot_scores(
    scores=None,
    pca_res=None,
    dims: Sequence[int] | None = None,
    pcx: int = 1,
    pcy: int = 2,
    pcz: int | None = None,
    labels=None,
    color_values=None,
    color_by: str = "label",
    object_ids=None,
    source_images=None,
    batches=None,
    areas=None,
    title: str | None = None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    **metadata,
):
    T = _pca_scores_from_args(scores, pca_res)
    if dims is None:
        dims = (pcx, pcy) if pcz is None else (pcx, pcy, pcz)
    dims = tuple(dims)
    idx = [d - 1 for d in dims]
    n = T.shape[0]

    if color_values is None:
        if color_by == "source_image":
            color_values = source_images
        elif color_by == "batch":
            color_values = batches
        else:
            color_values = labels
    groups = as_1d_array(color_values, n, "all").astype(str)
    meta = dict(object_id=object_ids, label=labels, source_image=source_images, batch=batches, area=areas)
    meta.update(metadata)
    custom, hover_meta = _customdata(n, **meta)

    fig = go.Figure()
    if len(dims) == 2:
        if title is None:
            title = f"Scores: C{dims[0]} vs C{dims[1]}"
        for group in np.unique(groups):
            mask = groups == group
            fig.add_trace(go.Scatter(
                x=T[mask, idx[0]], y=T[mask, idx[1]], mode="markers", name=str(group), customdata=custom[mask],
                marker=dict(size=9, opacity=0.8),
                hovertemplate=f"C{dims[0]}: %{{x:.4f}}<br>C{dims[1]}: %{{y:.4f}}<br>" + hover_meta + "<extra></extra>",
            ))
        fig.update_layout(title=title, xaxis_title=f"C{dims[0]}", yaxis_title=f"C{dims[1]}", width=width, height=height)
    elif len(dims) == 3:
        if title is None:
            title = f"Scores: C{dims[0]} vs C{dims[1]} vs C{dims[2]}"
        for group in np.unique(groups):
            mask = groups == group
            fig.add_trace(go.Scatter3d(
                x=T[mask, idx[0]], y=T[mask, idx[1]], z=T[mask, idx[2]], mode="markers", name=str(group), customdata=custom[mask],
                marker=dict(size=5, opacity=0.85),
                hovertemplate=f"C{dims[0]}: %{{x:.4f}}<br>C{dims[1]}: %{{y:.4f}}<br>C{dims[2]}: %{{z:.4f}}<br>" + hover_meta + "<extra></extra>",
            ))
        fig.update_layout(title=title, width=width, height=height, scene=dict(xaxis_title=f"C{dims[0]}", yaxis_title=f"C{dims[1]}", zaxis_title=f"C{dims[2]}"))
    else:
        raise ValueError("dims must contain 2 or 3 components.")
    return _show_or_return(fig, show)


# Generic version of plot_pca_loadings, and old plot_loadings
def plot_loadings(
    loadings: np.ndarray,
    pca_res=None,
    wavelengths=None,
    components: Sequence[int] = (1, 2, 3),
    component_names: Sequence[str] | None = None,
    title: str = "Loadings",
    width: int = 900,
    height: int = 500,
    show: bool = True,
):
    if component_names is None and pca_res is not None and not isinstance(pca_res, Mapping):
        maybe_names = list(pca_res) if isinstance(pca_res, (list, tuple, np.ndarray)) else None
        if maybe_names and all(isinstance(v, str) for v in maybe_names):
            component_names = maybe_names
            pca_res = None
            components = tuple(range(1, len(component_names) + 1))
    P = _pca_loadings_from_args(loadings, pca_res)
    x, x_title = wavelength_axis(P.shape[0], wavelengths)
    fig = go.Figure()
    for k, comp in enumerate(components):
        j = comp - 1
        name = component_names[k] if component_names is not None and k < len(component_names) else f"{component_prefix}{comp}"
        fig.add_trace(go.Scatter(x=x, y=P[:, j], mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="Loading", width=width, height=height)
    return _show_or_return(fig, show)

# Generic version of plot_pca_biplot_2D
def plot_biplot(
    scores: np.ndarray=None,
    pca_res=None,
    loadings: np.ndarray=None,
    dims: Sequence[int] = (1, 2),
    labels=None,
    color_by: str = "label",
    color_values=None,
    object_ids=None,
    source_images=None,
    wavelengths=None,
    n_loadings: int = 10,
    loading_scale: float = 1.0,
    title: str | None = None,
    show: bool = True,
    **metadata,
):
    T = _pca_scores_from_args(scores, pca_res)
    P = _pca_loadings_from_args(loadings, pca_res)
    if len(dims) != 2:
        raise ValueError("Biplot is implemented only in 2D.")
    fig = plot_scores(
        T,
        dims=dims, 
        labels=labels, 
        color_values=color_values, 
        color_by=color_by, 
        object_ids=object_ids, 
        source_images=source_images, 
        title=title or f"Biplot: C{dims[0]} vs C{dims[1]}", 
        show=False, 
        **metadata
    )
    ix, iy = dims[0] - 1, dims[1] - 1
    strength = np.sqrt(P[:, ix] ** 2 + P[:, iy] ** 2)
    top_idx = np.argsort(strength)[-n_loadings:]
    score_range = max(np.nanmax(np.abs(T[:, ix])), np.nanmax(np.abs(T[:, iy])))
    for j in top_idx:
        x_end = P[j, ix] * score_range * loading_scale
        y_end = P[j, iy] * score_range * loading_scale
        label = f"band {j}" if wavelengths is None else f"{np.asarray(wavelengths)[j]:.1f} nm"
        fig.add_annotation(x=x_end, y=y_end, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, text=label)
    return _show_or_return(fig, show)


# -----------------------------------------------------------------------------
# Diagnostic / metric functions
# -----------------------------------------------------------------------------

# Generic version of plot_pca_hotelling_t2, plot_pca_q_residuals, plot_simca_rule_statistic
def plot_metric_by_index(values, labels=None, title="Metric by observation", y_title="Metric", hline=None, object_ids=None, source_images=None, width=900, height=500, show=True, **metadata):
    values = np.asarray(values, dtype=float)
    n = len(values)
    labels = as_1d_array(labels, n, "all").astype(str)
    meta = dict(object_id=object_ids, source_image=source_images)
    meta.update(metadata)
    custom, hover_meta = _customdata(n, **meta)
    x = np.arange(n)
    fig = go.Figure()
    for lab in np.unique(labels):
        mask = labels == lab
        fig.add_trace(go.Scatter(x=x[mask], y=values[mask], mode="markers", name=str(lab), customdata=custom[mask], marker=dict(size=8, opacity=0.8), hovertemplate="index: %{x}<br>value: %{y:.4f}<br>" + hover_meta + "<extra></extra>"))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash")
    fig.update_layout(title=title, xaxis_title="Observation index", yaxis_title=y_title, width=width, height=height)
    return _show_or_return(fig, show)


# Generic version of plot_pca_q_vs_t2, plot_simca_distance_plot
# Add option for log ?
def plot_xy_diagnostic(
    x,
    y,
    labels=None,
    title: str = "Diagnostic plot",
    x_title: str = "x",
    y_title: str = "y",
    vline: float | None = None,
    hline: float | None = None,
    line_traces: Sequence[go.Scatter] | None = None,
    object_ids=None,
    source_images=None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    **metadata,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    labels = as_1d_array(labels, n, "all").astype(str)
    meta = dict(object_id=object_ids, source_image=source_images)
    meta.update(metadata)
    custom, hover_meta = _customdata(n, **meta)
    fig = go.Figure()
    for lab in np.unique(labels):
        mask = labels == lab
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name=str(lab), customdata=custom[mask], marker=dict(size=9, opacity=0.8), hovertemplate=f"{x_title}: %{{x:.4f}}<br>{y_title}: %{{y:.4f}}<br>" + hover_meta + "<extra></extra>"))
    if line_traces:
        for tr in line_traces:
            fig.add_trace(tr)
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash")
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash")
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height)
    return _show_or_return(fig, show)


# Wrappers
def plot_pca_metric_t2(pca_res, labels=None, object_ids=None, source_images=None, n_components=None, title="Hotelling T²", show=True):
    return plot_metric_by_index(hotelling_t2(pca_res, n_components), labels=labels, object_ids=object_ids, source_images=source_images, title=title, y_title="Hotelling T²", show=show)

def plot_pca_metric_q(X_centered, pca_res, labels=None, object_ids=None, source_images=None, n_components=None, title="Q residuals", show=True):
    Q, _ = q_residuals(X_centered, pca_res, n_components)
    return plot_metric_by_index(Q, labels=labels, object_ids=object_ids, source_images=source_images, title=title, y_title="Q residual", show=show)

def plot_pca_diagnostic(X_centered, pca_res, labels=None, object_ids=None, source_images=None, n_components=None, title="PCA diagnostic: Q residuals vs Hotelling T²", show=True):
    Q, _ = q_residuals(X_centered, pca_res, n_components)
    T2 = hotelling_t2(pca_res, n_components)
    return plot_xy_diagnostic(T2, Q, labels=labels, object_ids=object_ids, source_images=source_images, title=title, x_title="Hotelling T²", y_title="Q residual", show=show)


# -----------------------------------------------------------------------------
# Generic bars, counts and line summaries
# -----------------------------------------------------------------------------

# New Generic function
def plot_bar_values(
    x,
    y,
    title: str = "Bar plot",
    x_title: str = "x",
    y_title: str = "y",
    width: int = 1000,
    height: int = 500,
    show: bool = True,
):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, hovertemplate="%{x}<br>%{y}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height)
    return _show_or_return(fig, show)


# New Generic function
def plot_counts_by_group(
    df: pd.DataFrame,
    group_col: str,
    category_col: str,
    title: str = "Counts",
    width: int = 850,
    height: int = 500,
    show: bool = True,
):
    counts = df.groupby([group_col, category_col]).size().reset_index(name="count")
    fig = go.Figure()
    for group in counts[group_col].unique():
        sub = counts[counts[group_col] == group]
        fig.add_trace(go.Bar(x=sub[category_col], y=sub["count"], name=str(group), hovertemplate=f"{group_col}: %{{fullData.name}}<br>{category_col}: %{{x}}<br>count: %{{y}}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=category_col, yaxis_title="Count", barmode="group", width=width, height=height)
    return _show_or_return(fig, show)


# New Generic function
def plot_lines_from_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
    names: Sequence[str] | None = None,
    title: str = "Metrics",
    x_title: str | None = None,
    y_title: str = "Value",
    hlines: Sequence[tuple[float, str, str]] | None = None,
    percent_y: bool = False,
    width: int = 900,
    height: int = 550,
    show: bool = True,
):
    fig = go.Figure()
    if names is None:
        names = y_cols
    for col, name in zip(y_cols, names):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode="markers+lines", name=name))
    if hlines:
        for y, dash, text in hlines:
            fig.add_hline(y=y, line_dash=dash, annotation_text=text, annotation_position="top left")
    fig.update_layout(title=title, xaxis_title=x_title or x_col, yaxis_title=y_title, width=width, height=height)
    if percent_y:
        fig.update_yaxes(tickformat=".0%")
    return _show_or_return(fig, show)


# -----------------------------------------------------------------------------
# SIMCA specific helpers built on generic plots
# -----------------------------------------------------------------------------

def plot_object_decision_map(
    image_db,
    object_db,
    results_df: pd.DataFrame,
    image_id: str,
    decision_col: str = "simca_case",
    object_id_col: str = "object_id",
    source_col: str = "source_image",
    decision_to_code: Mapping[str, int] | None = None,
    code_to_name: Mapping[int, str] | None = None,
    title: str | None = None,
    width: int = 850,
    height: int = 750,
    show: bool = True,
):
    img = image_db[image_id]
    labels_img = img["labels"]
    if decision_to_code is None:
        decision_to_code = {"unknown": 1, "almond_only": 2, "peanut_only": 3, "ambiguous": 4}
    if code_to_name is None:
        code_to_name = {0: "background", 1: "unknown", 2: "almond_only", 3: "peanut_only", 4: "ambiguous"}

    decision_map = np.zeros_like(labels_img, dtype=float)
    sub = results_df[results_df[source_col] == image_id]
    for _, row in sub.iterrows():
        obj_id = row[object_id_col]
        if obj_id not in object_db:
            continue
        label_id = object_db[obj_id]["label_id"]
        decision_map[labels_img == label_id] = decision_to_code.get(row[decision_col], 1)

    tickvals = sorted([c for c in code_to_name if c != 0])
    colorbar = dict(title="decision", tickvals=tickvals, ticktext=[code_to_name[c] for c in tickvals], x=1.12)
    colorscale = [[0.00, "lightgray"], [0.25, "royalblue"], [0.50, "crimson"], [0.75, "orange"], [1.00, "purple"]]
    return plot_image_overlay(
        img["image_ref"], decision_map, title=title or f"Object decisions — {image_id}",
        background_title="image_ref", overlay_title="decision", overlay_colorscale=colorscale,
        overlay_colorbar=colorbar, alpha=0.55, width=width, height=height, show=show,
    )


def plot_distribution_with_curve(values, curve_x=None, curve_y=None, nbins=30, title="Distribution", x_title="Value", curve_name="theoretical", width=850, height=500, show=True):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.asarray(values), histnorm="probability density", nbinsx=nbins, name="empirical", opacity=0.65))
    if curve_x is not None and curve_y is not None:
        fig.add_trace(go.Scatter(x=curve_x, y=curve_y, mode="lines", name=curve_name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="Density", width=width, height=height)
    return _show_or_return(fig, show)

# Wrappers
def plot_decision_counts(results_df, true_label_col="true_label", decision_col="simca_case", title="SIMCA prediction counts", show=True):
    return plot_counts_by_group(results_df, group_col=true_label_col, category_col=decision_col, title=title, show=show)


def plot_simca_distance(simca_results, class_name, labels=None, object_ids=None, source_images=None, normalized=True, title=None, width=850, height=650, show=True):
    res = simca_results[class_name]
    if normalized:
        x = np.asarray(res["H_norm_limit"])
        y = np.asarray(res["Q_norm_limit"])
        x_title, y_title = "H / H_limit", "Q / Q_limit"
        vline, hline = 1.0, 1.0
    else:
        x = np.asarray(res["H"])
        y = np.asarray(res["Q"])
        x_title, y_title = "H", "Q"
        vline, hline = res["H_limit"], res["Q_limit"]
    accepted = np.asarray(res.get("accepted", [""] * len(x))).astype(str)
    rule_stat = np.asarray(res.get("rule_statistic", [""] * len(x))).astype(str)
    return plot_xy_diagnostic(
        x, y, labels=labels, object_ids=object_ids, source_images=source_images,
        accepted=accepted, rule_statistic=rule_stat,
        title=title or f"SIMCA distance — class={class_name}", x_title=x_title, y_title=y_title,
        vline=vline, hline=hline, width=width, height=height, show=show,
    )


def plot_simca_rule_metric(simca_results, class_name, labels=None, object_ids=None, source_images=None, title=None, show=True):
    res = simca_results[class_name]
    return plot_metric_by_index(
        res["rule_statistic"], labels=labels, object_ids=object_ids, source_images=source_images,
        hline=res.get("rule_limit"), title=title or f"SIMCA rule statistic — class={class_name}", y_title="Rule statistic", show=show,
    )

