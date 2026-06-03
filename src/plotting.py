from typing import Sequence, Any, Mapping
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pc

#from src.stats import hotelling_t2, q_residuals
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
        return None
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

# def _pca_scores_from_args(scores=None, pca_res=None):
#     if scores is not None:
#         return np.asarray(scores, dtype=float)
#     if pca_res is not None:
#         return np.asarray(pca_res["scores"], dtype=float)
#     raise ValueError("Provide scores or pca_res.")


# def _pca_loadings_from_args(loadings=None, pca_res=None):
#     if loadings is not None:
#         return np.asarray(loadings, dtype=float)
#     if pca_res is not None:
#         return np.asarray(pca_res["loadings"], dtype=float)
#     raise ValueError("Provide loadings or pca_res.")

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
    scores,
    #pca_res=None,
    dims: Sequence[int] | None = None,
    pcx: int = 1,
    pcy: int = 2,
    pcz: int | None = None,
    labels=None,
    color_values=None,
    color_by: str = "label",
    color_sequence=None,
    continuous_colorscale: str = "Turbo",
    object_ids=None,
    source_images=None,
    batches=None,
    areas=None,
    subset = None,
    title: str | None = None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    symbol_values=None,
    symbol_by: str | None = None,
    contour_values=None,
    contour_by: str | None = None,
    use_open_symbol_for_contour: bool = True,
    marker_size: int = 9,
    marker_opacity: float = 0.85,
    **metadata,
):
    if dims is None:
        dims = (pcx, pcy) if pcz is None else (pcx, pcy, pcz)
    dims = tuple(dims)
    idx = [d - 1 for d in dims]
    n = scores.shape[0]

    if color_values is None:
        if color_by == "source_image":
            color_values = source_images
        elif color_by == "batch":
            color_values = batches
        elif color_by == "subset":
            color_values = subset
        else:
            color_values = labels
    color_groups = as_1d_array(color_values, n, "all").astype(str)

    def ordered_unique(values):
        return list(dict.fromkeys(np.asarray(values).astype(str)))

    def make_dynamic_color_map(groups, color_sequence=None, continuous_colorscale="Turbo"):
        unique_groups = ordered_unique(groups)
        n_groups = len(unique_groups)
        if color_sequence is None:
            color_sequence = (
                pc.qualitative.Plotly
                + pc.qualitative.D3
                + pc.qualitative.G10
                + pc.qualitative.T10
                + pc.qualitative.Alphabet
            )
        if n_groups <= len(color_sequence):
            colors = color_sequence[:n_groups]
        else:
            colors = pc.sample_colorscale(
                continuous_colorscale,
                np.linspace(0, 1, n_groups),
            )
        return {
            group: colors[i]
            for i, group in enumerate(unique_groups)
        }

    color_map = make_dynamic_color_map(
        color_groups,
        color_sequence=color_sequence,
        continuous_colorscale=continuous_colorscale,
    )

    if symbol_values is None:
        if symbol_by == "batch":
            symbol_values = batches
        elif symbol_by == "source_image":
            symbol_values = source_images
        elif symbol_by == "subset":
            symbol_values = subset
        elif symbol_by == "label":
            symbol_values = labels
        else:
            symbol_values = np.array(["all"] * n)
    symbol_groups = as_1d_array(symbol_values, n, "all").astype(str)

    if contour_values is None:
        if contour_by == "subset":
            contour_values = subset
        elif contour_by == "batch":
            contour_values = batches
        elif contour_by == "source_image":
            contour_values = source_images
        elif contour_by == "label":
            contour_values = labels
        else:
            contour_values = np.array(["filled"] * n)
    contour_groups = as_1d_array(contour_values, n, "filled").astype(str)

    meta = dict(
        object_id=object_ids,
        label=labels,
        source_image=source_images,
        batch=batches,
        subset=subset,
        area=areas,
    )
    meta.update(metadata)
    custom, hover_meta = _customdata(n, **meta)

    base_symbols = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "star",
    ]
    unique_symbol_groups = np.unique(symbol_groups)
    symbol_map = {
        group: base_symbols[i % len(base_symbols)]
        for i, group in enumerate(unique_symbol_groups)
    }

    def is_open_contour(contour_group):
        contour_group_lower = str(contour_group).lower()
        open_keywords = [
            "projection",
            "test",
            "validation",
            "val",
            "external",
            "projected",
        ]
        return any(key in contour_group_lower for key in open_keywords)

    def symbol_with_contour(base_symbol, contour_group):
        if not use_open_symbol_for_contour:
            return base_symbol
        if is_open_contour(contour_group):
            return f"{base_symbol}-open"
        return base_symbol

    def contour_line_width(contour_group):
        return 2.0 if is_open_contour(contour_group) else 0.8

    def contour_line_color(contour_group, color_group):
        # Open markers: outline keeps the class color.
        # Filled markers: black outline improves readability.
        if is_open_contour(contour_group):
            return color_map[color_group]
        return "black"

    fig = go.Figure()
    combined_groups = np.array([
        f"{c}|||{s}|||{k}"
        for c, s, k in zip(color_groups, symbol_groups, contour_groups)
    ])
    unique_combined = np.unique(combined_groups)
    if len(dims) == 2:
        if title is None:
            title = f"Scores: C{dims[0]} vs C{dims[1]}"
        for combined in unique_combined:
            color_group, symbol_group, contour_group = combined.split("|||")
            mask = combined_groups == combined
            base_symbol = symbol_map[symbol_group]
            marker_symbol = symbol_with_contour(base_symbol, contour_group)
            trace_name = f"{color_group} | batch={symbol_group} | set={contour_group}"
            fig.add_trace(
                go.Scatter(
                    x=scores[mask, idx[0]],
                    y=scores[mask, idx[1]],
                    mode="markers",
                    name=trace_name,
                    customdata=custom[mask],
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        symbol=marker_symbol,
                        color=color_map[color_group],
                        line=dict(
                            width=contour_line_width(contour_group),
                            color=contour_line_color(contour_group, color_group),
                        ),
                    ),
                    hovertemplate=(
                        f"C{dims[0]}: %{{x:.4f}}<br>"
                        f"C{dims[1]}: %{{y:.4f}}<br>"
                        + hover_meta
                        + "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=f"C{dims[0]}",
            yaxis_title=f"C{dims[1]}",
            width=width,
            height=height,
            legend_title_text="Color | Batch | Set",
        )
    elif len(dims) == 3:
        if title is None:
            title = f"Scores: C{dims[0]} vs C{dims[1]} vs C{dims[2]}"
        for combined in unique_combined:
            color_group, symbol_group, contour_group = combined.split("|||")
            mask = combined_groups == combined
            # Scatter3d supports fewer marker symbols than Scatter.
            # We keep the batch symbol, but open-symbol rendering is less reliable in 3D.
            base_symbol = symbol_map[symbol_group]
            marker_symbol = base_symbol
            trace_name = f"{color_group} | batch={symbol_group} | set={contour_group}"
            fig.add_trace(
                go.Scatter3d(
                    x=scores[mask, idx[0]],
                    y=scores[mask, idx[1]],
                    z=scores[mask, idx[2]],
                    mode="markers",
                    name=trace_name,
                    customdata=custom[mask],
                    marker=dict(
                        size=max(marker_size - 3, 4),
                        opacity=marker_opacity,
                        symbol=marker_symbol,
                        color=color_map[color_group],
                        line=dict(
                            width=contour_line_width(contour_group),
                            color=contour_line_color(contour_group, color_group),
                        ),
                    ),
                    hovertemplate=(
                        f"C{dims[0]}: %{{x:.4f}}<br>"
                        f"C{dims[1]}: %{{y:.4f}}<br>"
                        f"C{dims[2]}: %{{z:.4f}}<br>"
                        + hover_meta
                        + "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            legend_title_text="Color | Batch | Set",
            scene=dict(
                xaxis_title=f"C{dims[0]}",
                yaxis_title=f"C{dims[1]}",
                zaxis_title=f"C{dims[2]}",
            ),
        )
    else:
        raise ValueError("dims must contain 2 or 3 components.")
    return _show_or_return(fig, show)


# SCORES FOR PIXEL-LEVEL PCA

def build_scores_dataframe(
    scores,
    labels=None,
    meta=None,
    subset=None,
    dims=(1, 2, 3),
    score_prefix="C",
):
    """
    Build a tidy DataFrame from PCA / SIMCA / latent scores.

    Parameters
    ----------
    scores : array-like, shape (n_samples, n_components)
        Score matrix.
    labels : array-like, optional
        Class labels, e.g. nut type.
    meta : DataFrame or dict, optional
        Metadata associated with each observation.
    subset : array-like, optional
        Set information, e.g. train / projection / test.
    dims : tuple
        Components to include. Components are 1-indexed.
    score_prefix : str
        Prefix for component columns, e.g. "C" or "PC".

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with score columns and metadata.
    """
    scores = np.asarray(scores)
    n = scores.shape[0]
    if meta is None:
        df = pd.DataFrame(index=np.arange(n))
    elif isinstance(meta, dict):
        df = pd.DataFrame(meta).copy()
    else:
        df = meta.copy()
    if len(df) != n:
        raise ValueError(
            f"Metadata length ({len(df)}) does not match scores length ({n})."
        )
    for d in dims:
        idx = d - 1
        if idx >= scores.shape[1]:
            raise ValueError(
                f"Component {d} requested, but scores only have "
                f"{scores.shape[1]} components."
            )
        df[f"{score_prefix}{d}"] = scores[:, idx]
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError(
                f"Labels length ({len(labels)}) does not match scores length ({n})."
            )
        df["label"] = labels.astype(str)
    if subset is not None:
        subset = np.asarray(subset)
        if len(subset) != n:
            raise ValueError(
                f"Subset length ({len(subset)}) does not match scores length ({n})."
            )
        df["subset"] = subset.astype(str)
    # Convert useful metadata to string for robust plotting
    for col in ["object_id", "source_image", "batch", "label", "subset"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def sample_scores_dataframe(
    df,
    group_cols=None,
    n_per_group=500,
    random_state=0,
    keep_group_cols=True,
):
    """
    Balanced sampling of a score DataFrame.

    This version preserves all original columns, including grouping columns
    such as label, subset and batch.
    """
    if group_cols is None:
        group_cols = []

    df = df.copy()
    # Keep only existing columns
    group_cols = [col for col in group_cols if col in df.columns]
    # If no valid grouping columns, simple random sample
    if len(group_cols) == 0:
        return (
            df.sample(
                n=min(len(df), n_per_group),
                random_state=random_state,
            )
            .reset_index(drop=True)
        )

    sampled_parts = []
    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        sampled_group = group.sample(
            n=min(len(group), n_per_group),
            random_state=random_state,
        )
        sampled_parts.append(sampled_group)
    df_sample = pd.concat(sampled_parts, axis=0, ignore_index=True)
    # Safety check: grouping columns should still be present
    if keep_group_cols:
        missing_cols = [col for col in group_cols if col not in df_sample.columns]
        if missing_cols:
            raise RuntimeError(
                f"Sampling removed grouping columns: {missing_cols}. "
                "Check the input DataFrame and groupby logic."
            )
    return df_sample

def plot_scores_density(
    df,
    x="C1",
    y="C2",
    color_by="label",
    facet_col=None,
    facet_row=None,
    mode="contour",
    nbinsx=80,
    nbinsy=80,
    title=None,
    width=950,
    height=650,
    show=True,
):
    """
    Plot 2D density of scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Score DataFrame.
    x, y : str
        Score columns.
    color_by : str
        Column used for contours color.
    facet_col, facet_row : str, optional
        Columns used for faceting.
    mode : {"contour", "heatmap"}
        Density visualization mode.
    nbinsx, nbinsy : int
        Number of bins for heatmap mode.
    """
    if title is None:
        title = f"Score density: {x} vs {y}"
    if mode == "contour":
        fig = px.density_contour(
            df,
            x=x,
            y=y,
            color=color_by if color_by in df.columns else None,
            facet_col=facet_col if facet_col in df.columns else None,
            facet_row=facet_row if facet_row in df.columns else None,
            title=title,
        )
        fig.update_traces(contours_coloring="none")
    elif mode == "heatmap":
        fig = px.density_heatmap(
            df,
            x=x,
            y=y,
            facet_col=facet_col if facet_col in df.columns else None,
            facet_row=facet_row if facet_row in df.columns else None,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            title=title,
        )
    else:
        raise ValueError("mode must be either 'contour' or 'heatmap'.")
    fig.update_layout(width=width, height=height)
    if show:
        fig.show()
    return fig


def plot_scores_distribution(
    df,
    score_col="C1",
    x_by="label",
    color_by="label",
    facet_col=None,
    facet_row=None,
    kind="violin",
    box=True,
    points=False,
    title=None,
    width=950,
    height=600,
    show=True,
):
    """
    Plot score distributions by class, batch, subset, etc.

    Parameters
    ----------
    df : pandas.DataFrame
        Score DataFrame.
    score_col : str
        Score column to plot.
    x_by : str
        Grouping column on x-axis.
    color_by : str
        Color grouping column.
    facet_col, facet_row : str, optional
        Faceting columns.
    kind : {"violin", "box", "histogram"}
        Type of distribution plot.
    """
    if title is None:
        title = f"Distribution of {score_col}"
    facet_col_arg = facet_col if facet_col in df.columns else None
    facet_row_arg = facet_row if facet_row in df.columns else None
    color_arg = color_by if color_by in df.columns else None
    if kind == "violin":
        fig = px.violin(
            df,
            x=x_by,
            y=score_col,
            color=color_arg,
            facet_col=facet_col_arg,
            facet_row=facet_row_arg,
            box=box,
            points="all" if points else False,
            title=title,
        )
    elif kind == "box":
        fig = px.box(
            df,
            x=x_by,
            y=score_col,
            color=color_arg,
            facet_col=facet_col_arg,
            facet_row=facet_row_arg,
            points="all" if points else False,
            title=title,
        )
    elif kind == "histogram":
        fig = px.histogram(
            df,
            x=score_col,
            color=color_arg,
            facet_col=facet_col_arg,
            facet_row=facet_row_arg,
            marginal="box",
            histnorm="probability density",
            opacity=0.65,
            title=title,
        )
    else:
        raise ValueError("kind must be 'violin', 'box', or 'histogram'.")
    fig.update_layout(width=width, height=height)
    if show:
        fig.show()
    return fig


def summarize_scores_by_object(
    df,
    score_cols=("C1", "C2", "C3"),
    object_col="object_id",
    extra_group_cols=None,
):
    """
    Aggregate pixel-level scores into object-level summaries.

    Parameters
    ----------
    df : pandas.DataFrame
        Pixel score DataFrame.
    score_cols : tuple[str]
        Score columns to summarize.
    object_col : str
        Object identifier column.
    extra_group_cols : list[str], optional
        Additional columns to preserve, e.g. label, subset, batch, source_image.

    Returns
    -------
    summary : pandas.DataFrame
        One row per object.
    """
    if object_col not in df.columns:
        raise ValueError(f"Column '{object_col}' not found in DataFrame.")
    if extra_group_cols is None:
        extra_group_cols = ["label", "subset", "batch", "source_image"]
    group_cols = [object_col] + [
        col for col in extra_group_cols
        if col in df.columns and col != object_col
    ]
    score_cols = [col for col in score_cols if col in df.columns]
    agg_dict = {}
    for col in score_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")
        agg_dict[f"{col}_median"] = (col, "median")
        agg_dict[f"{col}_q05"] = (col, lambda s: s.quantile(0.05))
        agg_dict[f"{col}_q95"] = (col, lambda s: s.quantile(0.95))
    # Pixel count
    first_score = score_cols[0]
    agg_dict["n_pixels"] = (first_score, "size")
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )
    return summary


def plot_object_score_summary(
    df_object,
    x="C1_mean",
    y="C2_mean",
    color_by="label",
    symbol_by="batch",
    facet_col="subset",
    size_by="n_pixels",
    error_x=None,
    error_y=None,
    hover_cols=None,
    title=None,
    width=950,
    height=650,
    show=True,
):
    """
    Plot object-level summary of pixel scores.

    Parameters
    ----------
    df_object : pandas.DataFrame
        Output of summarize_scores_by_object().
    x, y : str
        Columns used as axes.
    color_by : str
        Color column.
    symbol_by : str
        Symbol column.
    facet_col : str, optional
        Facet column.
    size_by : str, optional
        Marker size column.
    error_x, error_y : str, optional
        Error bar columns.
    """
    if title is None:
        title = f"Object summary of pixel scores: {x} vs {y}"
    if hover_cols is None:
        hover_cols = [
            col for col in [
                "object_id",
                "source_image",
                "batch",
                "subset",
                "label",
                "n_pixels",
                "C1_std",
                "C2_std",
            ]
            if col in df_object.columns
        ]
    fig = px.scatter(
        df_object,
        x=x,
        y=y,
        color=color_by if color_by in df_object.columns else None,
        symbol=symbol_by if symbol_by in df_object.columns else None,
        facet_col=facet_col if facet_col in df_object.columns else None,
        size=size_by if size_by in df_object.columns else None,
        error_x=error_x if error_x in df_object.columns else None,
        error_y=error_y if error_y in df_object.columns else None,
        hover_data=hover_cols,
        title=title,
    )
    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=1, color="black"),
        )
    )
    fig.update_layout(width=width, height=height)
    if show:
        fig.show()
    return fig

# Generic version of plot_pca_loadings, and old plot_loadings
def plot_loadings(
    loadings: np.ndarray,
    #pca_res=None,
    wavelengths=None,
    components: Sequence[int] = (1, 2, 3),
    component_names: Sequence[str] | None = None,
    title: str = "Loadings",
    width: int = 900,
    height: int = 500,
    show: bool = True,
):
    x, x_title = wavelength_axis(loadings.shape[0], wavelengths)
    fig = go.Figure()
    for k, comp in enumerate(components):
        j = comp - 1
        if j >= loadings.shape[1]:
            continue
        name = component_names[k] if component_names is not None and k < len(component_names) else f"PC{comp}"
        fig.add_trace(go.Scatter(x=x, y=loadings[:, j], mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="Loading", width=width, height=height)
    return _show_or_return(fig, show)


# Generic version of plot_pca_biplot_2D
def plot_biplot(
    scores: np.ndarray,
    loadings: np.ndarray,
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
    scores = np.asarray(scores, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    if len(dims) != 2:
        raise ValueError("Biplot is implemented only in 2D.")
    fig = plot_scores(
        scores,
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
    strength = np.sqrt(loadings[:, ix] ** 2 + loadings[:, iy] ** 2)
    top_idx = np.argsort(strength)[-n_loadings:]
    score_range = max(np.nanmax(np.abs(scores[:, ix])), np.nanmax(np.abs(scores[:, iy])))
    for j in top_idx:
        x_end = loadings[j, ix] * score_range * loading_scale
        y_end = loadings[j, iy] * score_range * loading_scale
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
def plot_pca_metric_t2(pca_model, X=None, labels=None, object_ids=None, source_images=None, n_components=None, title="Hotelling T²", show=True):
    T2 = pca_model.hotelling_t2(X, n_components=n_components)
    return plot_metric_by_index(T2, labels=labels, object_ids=object_ids, source_images=source_images, title=title, y_title="Hotelling T²", show=show)

def plot_pca_metric_q(pca_model, X=None, labels=None, object_ids=None, source_images=None, n_components=None, title="Q residuals", show=True):
    if X is None:
        X = pca_model.inverse_transform(pca_model.scores_)
    Q, _ = pca_model.q_residuals(X, n_components=n_components)
    return plot_metric_by_index(Q, labels=labels, object_ids=object_ids, source_images=source_images, title=title, y_title="Q residual", show=show)


def plot_pca_diagnostic(pca_model, X=None, labels=None, object_ids=None, source_images=None, n_components=None, title="PCA diagnostic: Q residuals vs Hotelling T²", show=True):
    if X is None:
        X = pca_model.inverse_transform(pca_model.scores_)
    T2, Q = pca_model.distances(X, n_components=n_components)
    return plot_xy_diagnostic(T2, Q, labels=labels, object_ids=object_ids, source_images=source_images, title=title, x_title="Hotelling T²", y_title="Q residual", show=show)


def plot_pca_metric_heatmap(
    summary_df,
    metric,
    index_col="preprocessing",
    column_col="matrix_method",
    title=None,
    colorscale="Viridis",
    width=850,
    height=600,
    show=True,
):
    """
    Plot a heatmap for one PCA comparison metric.

    Useful to compare preprocessing x matrix representation.
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found in summary_df.")

    table = summary_df.pivot_table(
        index=index_col,
        columns=column_col,
        values=metric,
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=table.values,
            x=table.columns.astype(str),
            y=table.index.astype(str),
            colorscale=colorscale,
            colorbar=dict(title=metric),
            hovertemplate=(
                f"{column_col}: %{{x}}<br>"
                f"{index_col}: %{{y}}<br>"
                f"{metric}: %{{z:.4f}}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or f"PCA metric heatmap — {metric}",
        xaxis_title=column_col,
        yaxis_title=index_col,
        width=width,
        height=height,
    )

    return _show_or_return(fig, show)


def plot_pca_metric_tradeoff(
    summary_df,
    x_metric="batch_trace_ratio",
    y_metric="class_trace_ratio",
    color_by="matrix_method",
    symbol_by="preprocessing",
    size_by=None,
    hover_cols=None,
    title=None,
    width=900,
    height=650,
    show=True,
):
    """
    Plot trade-off between two PCA metrics.

    Typical use:
        x = batch effect
        y = class separation
    """
    if x_metric not in summary_df.columns:
        raise ValueError(f"x_metric '{x_metric}' not found.")

    if y_metric not in summary_df.columns:
        raise ValueError(f"y_metric '{y_metric}' not found.")

    if hover_cols is None:
        hover_cols = [
            col for col in [
                "matrix_method",
                "preprocessing",
                "class_trace_ratio",
                "batch_trace_ratio",
                "class_over_batch_ratio",
                "ncomp_90",
                "ncomp_95",
                "train_q_mean",
                "projection_q_mean",
                "projection_train_q_ratio",
            ]
            if col in summary_df.columns
        ]

    fig = px.scatter(
        summary_df,
        x=x_metric,
        y=y_metric,
        color=color_by if color_by in summary_df.columns else None,
        symbol=symbol_by if symbol_by in summary_df.columns else None,
        size=size_by if size_by in summary_df.columns else None,
        hover_data=hover_cols,
        title=title or f"PCA trade-off: {y_metric} vs {x_metric}",
    )

    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=1, color="black"),
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=x_metric,
        yaxis_title=y_metric,
    )

    return _show_or_return(fig, show)


def plot_pca_metric_ranking(
    summary_df,
    metric,
    group_col="matrix_method",
    label_col="preprocessing",
    ascending=False,
    top_n=None,
    title=None,
    width=1000,
    height=600,
    show=True,
):
    """
    Plot ranking of preprocessing methods according to one metric.
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found.")

    df = summary_df.copy()

    df = df.sort_values(metric, ascending=ascending)

    if top_n is not None:
        df = (
            df.groupby(group_col, group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    fig = px.bar(
        df,
        x=label_col,
        y=metric,
        color=group_col if group_col in df.columns else None,
        facet_col=group_col if group_col in df.columns else None,
        hover_data=[
            col for col in [
                "matrix_method",
                "preprocessing",
                "class_trace_ratio",
                "batch_trace_ratio",
                "class_over_batch_ratio",
                "ncomp_90",
                "ncomp_95",
            ]
            if col in df.columns
        ],
        title=title or f"Ranking by {metric}",
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=label_col,
        yaxis_title=metric,
    )

    fig.update_xaxes(tickangle=45)

    return _show_or_return(fig, show)




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

