import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

from src.stats import mean_spectrum, hotelling_t2, q_residuals
from src.simca import simca_limits_from_alpha

# RAW DATA PLOTTING FUNCTIONS
def plot_bands_slider(cube, title="Image hyperspectrale"):
    n_bands = cube.shape[2]

    fig = go.Figure()

    for i in range(n_bands):
        fig.add_trace(
            go.Heatmap(
                z=cube[:, :, i],
                visible=(i == 0),
                colorscale="Viridis",
                colorbar=dict(title="Réflectance")
            )
        )

    steps = []
    for i in range(n_bands):
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": [j == i for j in range(n_bands)]},
                    {"title": f"{title} — bande {i}"}
                ],
                label=str(i)
            )
        )

    fig.update_layout(
        title=f"{title} — bande 0",
        sliders=[dict(active=0, currentvalue={"prefix": "Bande : "}, steps=steps)],
        width=700,
        height=700
    )

    fig.show()

def plot_mean_spectra(data, keys):
    fig = go.Figure()

    for key in keys:
        spectrum = mean_spectrum(data[key])
        fig.add_trace(
            go.Scatter(
                y=spectrum,
                mode="lines",
                name=key
            )
        )

    fig.update_layout(
        title="Spectres moyens",
        xaxis_title="Bande spectrale",
        yaxis_title="Réflectance moyenne"
    )

    fig.show()

def plot_mean_spectra_from_excel(df, title="Spectres moyens par classe"):
    spectral_cols = df.columns[3:]
    wavelengths = spectral_cols.astype(float)

    fig = go.Figure()

    for cls in df["Class"].unique():
        sub = df[df["Class"] == cls]
        mean_spectrum = sub[spectral_cols].mean(axis=0)

        fig.add_trace(
            go.Scatter(
                x=wavelengths,
                y=mean_spectrum,
                mode="lines",
                name=cls
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Longueur d’onde, nm",
        yaxis_title="Absorbance"
    )

    fig.show()

def plot_spectral_distribution(cube, title="Distribution spectrale", n_pixels=2000):
    _, _, b = cube.shape
    pixels = cube.reshape(-1, b)

    idx = np.random.choice(pixels.shape[0], size=min(n_pixels, pixels.shape[0]), replace=False)
    sample = pixels[idx]

    mean = np.nanmean(sample, axis=0)
    std = np.nanstd(sample, axis=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=mean,
        mode="lines",
        name="Moyenne"
    ))

    fig.add_trace(go.Scatter(
        y=mean + std,
        mode="lines",
        name="+1 écart-type",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        y=mean - std,
        mode="lines",
        name="-1 écart-type",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Bande",
        yaxis_title="Réflectance"
    )

    fig.show()

def plot_two_classes(df, class_a="Almond", class_b="Peanut"):
    spectral_cols = df.columns[3:]
    wavelengths = spectral_cols.astype(float)

    mean_a = df[df["Class"] == class_a][spectral_cols].mean(axis=0)
    mean_b = df[df["Class"] == class_b][spectral_cols].mean(axis=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=mean_a,
        mode="lines",
        name=class_a
    ))

    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=mean_b,
        mode="lines",
        name=class_b
    ))

    fig.update_layout(
        title=f"{class_a} vs {class_b}",
        xaxis_title="Longueur d’onde, nm",
        yaxis_title="Absorbance"
    )

    fig.show()

def plot_loadings(loadings, lv=["LV 1 (53.85%)"]):
    fig = go.Figure()

    if not isinstance(lv, list):
        lv = [lv]

    fig.add_trace(
        go.Scatter(
            x=loadings["Wavelength"],
            y=loadings[lv[0]],
            mode="lines+markers",
            name=lv[0]
        )
    )

    if len(lv) > 1:
        for component in lv[1:]:
            fig.add_trace(
                go.Scatter(
                    x=loadings["Wavelength"],
                    y=loadings[component],
                    mode="lines+markers",
                    name=component
                )
            )

    fig.update_layout(
        title=f"Loadings",
        xaxis_title="Longueur d’onde, nm",
        yaxis_title="Loading"
    )

    fig.show()


# DB OBJECT PLOTTING FUNCTIONS

def _masked_array_for_plot(arr, mask_value=0):
    """
    Replace background values by NaN so Plotly does not display them.
    Useful for label overlays.
    """
    arr = np.asarray(arr).astype(float)
    arr[arr == mask_value] = np.nan
    return arr


def plot_db_image(
    image_db,
    image_id,
    image_type="image_ref",
    band=0,
    title=None,
    colorscale="Viridis",
    height=700,
    width=800,
):
    """
    Plot one image stored in image_db.

    Parameters
    ----------
    image_db : dict
        Image-level database.
    image_id : str
        Key in image_db, for example "almond1" or "peanut2".
    image_type : str
        One of:
        - "image_ref" : 2D reference image used for segmentation
        - "mask"      : binary object/background mask
        - "labels"    : labelled objects image
        - "band"      : one spectral band from cube
    band : int
        Spectral band index if image_type="band".
    """
    img = image_db[image_id]

    if image_type == "image_ref":
        z = img["image_ref"]
        colorbar_title = "Image ref"

    elif image_type == "mask":
        z = img["mask"].astype(int)
        colorbar_title = "Mask"

    elif image_type == "labels":
        z = img["labels"]
        colorbar_title = "Label"

    elif image_type == "band":
        z = img["cube"][:, :, band]
        colorbar_title = img.get("data_mode", "value")

    else:
        raise ValueError("image_type must be 'image_ref', 'mask', 'labels' or 'band'.")

    if title is None:
        title = f"{image_id} — {image_type}"
        if image_type == "band":
            title += f" {band}"

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
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )

    fig.show()


def plot_db_labels_overlay(
    image_db,
    image_id,
    base="image_ref",
    band=0,
    alpha=0.45,
    height=700,
    width=800,
):
    """
    Plot image with object labels overlay.

    Parameters
    ----------
    base : str
        - "image_ref" : use image_ref as background
        - "band"      : use cube[:, :, band] as background
    """
    img = image_db[image_id]

    if base == "image_ref":
        background = img["image_ref"]
        base_title = "image_ref"
    elif base == "band":
        background = img["cube"][:, :, band]
        base_title = f"band {band}"
    else:
        raise ValueError("base must be 'image_ref' or 'band'.")

    labels = _masked_array_for_plot(img["labels"], mask_value=0)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=background,
            colorscale="Gray",
            showscale=True,
            colorbar=dict(title=base_title),
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=labels,
            colorscale="Turbo",
            opacity=alpha,
            showscale=True,
            colorbar=dict(title="Object label", x=1.12),
            hovertemplate="row: %{y}<br>col: %{x}<br>label: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{image_id} — labels overlay — {img['n_objects']} objects",
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )

    fig.show()


def plot_db_object(
    object_db,
    object_id,
    show_spectrum=True,
    spectrum_field="mean_spectrum",
    height=500,
    width=1000,
):
    """
    Plot one extracted object:
    - crop reference image
    - object mask overlay
    - optional mean spectrum
    """
    obj = object_db[object_id]

    if show_spectrum:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"{object_id} — crop",
                f"{spectrum_field}",
            ),
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"{object_id} — crop",),
        )

    fig.add_trace(
        go.Heatmap(
            z=obj["image_ref_crop"],
            colorscale="Gray",
            showscale=True,
            colorbar=dict(title="Image ref"),
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    mask_overlay = obj["mask"].astype(float)
    mask_overlay[mask_overlay == 0] = np.nan

    fig.add_trace(
        go.Heatmap(
            z=mask_overlay,
            colorscale="Reds",
            opacity=0.35,
            showscale=False,
            hovertemplate="row: %{y}<br>col: %{x}<br>mask: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if show_spectrum:
        wavelengths = obj.get("wavelengths", None)

        if wavelengths is None:
            x = np.arange(obj[spectrum_field].shape[0])
            x_title = "band"
        else:
            x = wavelengths
            x_title = "wavelength (nm)"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=obj[spectrum_field],
                mode="lines",
                name=spectrum_field,
                hovertemplate=f"{x_title}: %{{x}}<br>value: %{{y}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        if "std_spectrum" in obj and spectrum_field == "mean_spectrum":
            mean = obj["mean_spectrum"]
            std = obj["std_spectrum"]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean + std,
                    mode="lines",
                    name="+1 std",
                    line=dict(dash="dash"),
                    opacity=0.5,
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean - std,
                    mode="lines",
                    name="-1 std",
                    line=dict(dash="dash"),
                    opacity=0.5,
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(title_text=x_title, row=1, col=2)
        fig.update_yaxes(title_text=obj.get("data_mode", "value"), row=1, col=2)

    fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)

    fig.update_layout(
        title=(
            f"{object_id} | "
            f"type={obj.get('object_nut_type')} | "
            f"source={obj.get('source_clean_key')} | "
            f"area={obj.get('area_pixels')}"
        ),
        height=height,
        width=width,
    )

    fig.show()


def plot_db_object_grid(
    object_db,
    source_image,
    max_objects=40,
    n_cols=5,
    height_per_row=220,
    width=1100,
):
    """
    Plot crops of all objects extracted from one source image.
    """
    selected = [
        (obj_id, obj)
        for obj_id, obj in object_db.items()
        if obj.get("source_clean_key") == source_image
    ]

    selected = selected[:max_objects]

    if len(selected) == 0:
        print(f"No object found for source_image={source_image}")
        return

    n_objects = len(selected)
    n_rows = int(np.ceil(n_objects / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{obj_id}<br>area={obj['area_pixels']}"
            for obj_id, obj in selected
        ],
    )

    for idx, (obj_id, obj) in enumerate(selected):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        fig.add_trace(
            go.Heatmap(
                z=obj["image_ref_crop"],
                colorscale="Gray",
                showscale=False,
                hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        mask_overlay = obj["mask"].astype(float)
        mask_overlay[mask_overlay == 0] = np.nan

        fig.add_trace(
            go.Heatmap(
                z=mask_overlay,
                colorscale="Reds",
                opacity=0.35,
                showscale=False,
                hovertemplate="row: %{y}<br>col: %{x}<br>mask<extra></extra>",
            ),
            row=row,
            col=col,
        )

        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title=f"Objects extracted from {source_image} — {n_objects} objects",
        height=height_per_row * n_rows,
        width=width,
    )

    fig.show()


def plot_db_object_spectra(
    object_db,
    source_image=None,
    nut_type=None,
    spectrum_field="mean_spectrum",
    show_std=False,
    max_objects=100,
    title=None,
):
    """
    Plot spectra of extracted objects.

    You can filter by:
    - source_image, for example "almond1"
    - nut_type, for example "almond" or "peanut"
    """
    selected = []

    for obj_id, obj in object_db.items():
        if source_image is not None and obj.get("source_clean_key") != source_image:
            continue

        if nut_type is not None and obj.get("object_nut_type") != nut_type:
            continue

        selected.append((obj_id, obj))

    selected = selected[:max_objects]

    if len(selected) == 0:
        print("No objects found with these filters.")
        return

    fig = go.Figure()

    for obj_id, obj in selected:
        wavelengths = obj.get("wavelengths", None)

        if wavelengths is None:
            x = np.arange(obj[spectrum_field].shape[0])
            x_title = "band"
        else:
            x = wavelengths
            x_title = "wavelength (nm)"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=obj[spectrum_field],
                mode="lines",
                name=obj_id,
                hovertemplate=(
                    f"object: {obj_id}<br>"
                    f"{x_title}: %{{x}}<br>"
                    "value: %{y}<extra></extra>"
                ),
            )
        )

        if show_std and spectrum_field == "mean_spectrum":
            mean = obj["mean_spectrum"]
            std = obj["std_spectrum"]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean + std,
                    mode="lines",
                    name=f"{obj_id} + std",
                    line=dict(dash="dash"),
                    opacity=0.25,
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean - std,
                    mode="lines",
                    name=f"{obj_id} - std",
                    line=dict(dash="dash"),
                    opacity=0.25,
                    showlegend=False,
                )
            )

    if title is None:
        title = "Object spectra"
        if source_image is not None:
            title += f" — {source_image}"
        if nut_type is not None:
            title += f" — {nut_type}"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=selected[0][1].get("data_mode", "value"),
        height=550,
        width=950,
    )

    fig.show()


def plot_db_object_areas(object_db, source_image=None, nut_type=None):
    """
    Plot histogram of object areas.
    Useful to diagnose min_area and segmentation quality.
    """
    areas = []
    labels = []

    for obj_id, obj in object_db.items():
        if source_image is not None and obj.get("source_clean_key") != source_image:
            continue

        if nut_type is not None and obj.get("object_nut_type") != nut_type:
            continue

        areas.append(obj["area_pixels"])
        labels.append(obj_id)

    if len(areas) == 0:
        print("No objects found with these filters.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=areas,
            hovertemplate="object: %{x}<br>area: %{y}<extra></extra>",
        )
    )

    title = "Object areas"
    if source_image is not None:
        title += f" — {source_image}"
    if nut_type is not None:
        title += f" — {nut_type}"

    fig.update_layout(
        title=title,
        xaxis_title="object_id",
        yaxis_title="area_pixels",
        height=500,
        width=1000,
    )

    fig.show()

    
# PCA PLOTTING FUNCTIONS

def plot_pca_explained_variance(
    pca_res,
    n_components_to_show=None,
    title="PCA explained variance",
):
    """
    Plot explained variance ratio and cumulative explained variance.

    Parameters
    ----------
    pca_res : dict
        Output of pca_from_cov or pca_sklearn.
        Must contain:
        - explained_variance_ratio
        - cumulative_explained_variance_ratio
    n_components_to_show : int or None
        Number of PCs to show. If None, show all.
    """
    evr = np.asarray(pca_res["explained_variance_ratio"])
    cum = np.asarray(pca_res["cumulative_explained_variance_ratio"])

    if n_components_to_show is not None:
        evr = evr[:n_components_to_show]
        cum = cum[:n_components_to_show]

    pcs = np.arange(1, len(evr) + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pcs,
            y=evr,
            name="Explained variance",
            hovertemplate="PC%{x}<br>variance: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pcs,
            y=cum,
            mode="lines+markers",
            name="Cumulative variance",
            hovertemplate="PC%{x}<br>cumulative: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Principal component",
        yaxis_title="Variance ratio",
        width=850,
        height=500,
    )

    fig.show()


def plot_pca_scores_2d(
    pca_res,
    labels=None,
    object_ids=None,
    source_images=None,
    batches=None,
    areas=None,
    pcx=1,
    pcy=2,
    color_by="label",
    title=None,
):
    """
    Plot PCA scores PCx vs PCy.

    Parameters
    ----------
    pca_res : dict
        PCA result dict with key "scores".
    labels : array-like or None
        Class labels, for example y.
    object_ids : array-like or None
        Object IDs.
    source_images : array-like or None
        Source image names.
    batches : array-like or None
        Batch numbers.
    areas : array-like or None
        Object areas.
    color_by : str
        "label", "source_image", "batch", or "none".
    """
    T = np.asarray(pca_res["scores"])

    ix = pcx - 1
    iy = pcy - 1

    if title is None:
        title = f"PCA scores: PC{pcx} vs PC{pcy}"

    if color_by == "label" and labels is not None:
        color_values = np.asarray(labels)
        color_name = "label"
    elif color_by == "source_image" and source_images is not None:
        color_values = np.asarray(source_images)
        color_name = "source image"
    elif color_by == "batch" and batches is not None:
        color_values = np.asarray(batches).astype(str)
        color_name = "batch"
    else:
        color_values = np.array(["all"] * T.shape[0])
        color_name = "all"

    custom_cols = []

    if object_ids is not None:
        custom_cols.append(np.asarray(object_ids).astype(str))
    else:
        custom_cols.append(np.array([""] * T.shape[0]))

    if labels is not None:
        custom_cols.append(np.asarray(labels).astype(str))
    else:
        custom_cols.append(np.array([""] * T.shape[0]))

    if source_images is not None:
        custom_cols.append(np.asarray(source_images).astype(str))
    else:
        custom_cols.append(np.array([""] * T.shape[0]))

    if batches is not None:
        custom_cols.append(np.asarray(batches).astype(str))
    else:
        custom_cols.append(np.array([""] * T.shape[0]))

    if areas is not None:
        custom_cols.append(np.asarray(areas).astype(str))
    else:
        custom_cols.append(np.array([""] * T.shape[0]))

    customdata = np.stack(custom_cols, axis=1)

    fig = go.Figure()

    for value in np.unique(color_values):
        mask = color_values == value

        fig.add_trace(
            go.Scatter(
                x=T[mask, ix],
                y=T[mask, iy],
                mode="markers",
                name=f"{color_name}: {value}",
                customdata=customdata[mask],
                marker=dict(size=9, opacity=0.8),
                hovertemplate=(
                    f"PC{pcx}: %{{x:.4f}}<br>"
                    f"PC{pcy}: %{{y:.4f}}<br>"
                    "object_id: %{customdata[0]}<br>"
                    "label: %{customdata[1]}<br>"
                    "source: %{customdata[2]}<br>"
                    "batch: %{customdata[3]}<br>"
                    "area: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"PC{pcx}",
        yaxis_title=f"PC{pcy}",
        width=850,
        height=650,
    )

    fig.show()


def plot_pca_scores_3d(
    pca_res,
    labels=None,
    object_ids=None,
    source_images=None,
    pcx=1,
    pcy=2,
    pcz=3,
    color_by="label",
    title=None,
):
    """
    3D PCA scores plot.
    """
    T = np.asarray(pca_res["scores"])

    ix = pcx - 1
    iy = pcy - 1
    iz = pcz - 1

    if title is None:
        title = f"PCA scores: PC{pcx} vs PC{pcy} vs PC{pcz}"

    if color_by == "label" and labels is not None:
        color_values = np.asarray(labels)
        color_name = "label"
    elif color_by == "source_image" and source_images is not None:
        color_values = np.asarray(source_images)
        color_name = "source image"
    else:
        color_values = np.array(["all"] * T.shape[0])
        color_name = "all"

    if object_ids is None:
        object_ids = np.array([""] * T.shape[0])
    if labels is None:
        labels = np.array([""] * T.shape[0])
    if source_images is None:
        source_images = np.array([""] * T.shape[0])

    customdata = np.stack(
        [
            np.asarray(object_ids).astype(str),
            np.asarray(labels).astype(str),
            np.asarray(source_images).astype(str),
        ],
        axis=1,
    )

    fig = go.Figure()

    for value in np.unique(color_values):
        mask = color_values == value

        fig.add_trace(
            go.Scatter3d(
                x=T[mask, ix],
                y=T[mask, iy],
                z=T[mask, iz],
                mode="markers",
                name=f"{color_name}: {value}",
                customdata=customdata[mask],
                marker=dict(size=5, opacity=0.85),
                hovertemplate=(
                    f"PC{pcx}: %{{x:.4f}}<br>"
                    f"PC{pcy}: %{{y:.4f}}<br>"
                    f"PC{pcz}: %{{z:.4f}}<br>"
                    "object_id: %{customdata[0]}<br>"
                    "label: %{customdata[1]}<br>"
                    "source: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        width=900,
        height=700,
        scene=dict(
            xaxis_title=f"PC{pcx}",
            yaxis_title=f"PC{pcy}",
            zaxis_title=f"PC{pcz}",
        ),
    )

    fig.show()


def plot_pca_loadings(
    pca_res,
    wavelengths=None,
    components=(1, 2, 3),
    title="PCA loadings",
):
    """
    Plot PCA loadings as a function of wavelength or band index.

    Parameters
    ----------
    pca_res : dict
        PCA result dict with key "loadings".
    wavelengths : array-like or None
        Wavelength axis. If None, use band index.
    components : tuple
        Components to plot, e.g. (1, 2, 3).
    """
    P = np.asarray(pca_res["loadings"])

    if wavelengths is None:
        x = np.arange(P.shape[0])
        x_title = "Band index"
    else:
        x = np.asarray(wavelengths)
        x_title = "Wavelength (nm)"

    fig = go.Figure()

    for comp in components:
        idx = comp - 1

        fig.add_trace(
            go.Scatter(
                x=x,
                y=P[:, idx],
                mode="lines+markers",
                name=f"PC{comp}",
                hovertemplate=(
                    f"{x_title}: %{{x}}<br>"
                    f"loading PC{comp}: %{{y:.5f}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Loading",
        width=900,
        height=500,
    )

    fig.show()


def plot_pca_biplot_2d(
    pca_res,
    labels=None,
    object_ids=None,
    wavelengths=None,
    pcx=1,
    pcy=2,
    n_loadings=10,
    loading_scale=1.0,
    title=None,
):
    """
    Simplified PCA biplot:
    - scores as points
    - strongest loading vectors as arrows
    """
    T = np.asarray(pca_res["scores"])
    P = np.asarray(pca_res["loadings"])

    ix = pcx - 1
    iy = pcy - 1

    if title is None:
        title = f"PCA biplot: PC{pcx} vs PC{pcy}"

    fig = go.Figure()

    if labels is None:
        labels = np.array(["all"] * T.shape[0])
    else:
        labels = np.asarray(labels)

    if object_ids is None:
        object_ids = np.array([""] * T.shape[0])
    else:
        object_ids = np.asarray(object_ids).astype(str)

    for cls in np.unique(labels):
        mask = labels == cls
        fig.add_trace(
            go.Scatter(
                x=T[mask, ix],
                y=T[mask, iy],
                mode="markers",
                name=str(cls),
                customdata=object_ids[mask],
                marker=dict(size=8, opacity=0.75),
                hovertemplate=(
                    f"PC{pcx}: %{{x:.4f}}<br>"
                    f"PC{pcy}: %{{y:.4f}}<br>"
                    "object: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    loading_strength = np.sqrt(P[:, ix] ** 2 + P[:, iy] ** 2)
    top_idx = np.argsort(loading_strength)[-n_loadings:]

    score_range = max(
        np.nanmax(np.abs(T[:, ix])),
        np.nanmax(np.abs(T[:, iy])),
    )

    for j in top_idx:
        x_end = P[j, ix] * score_range * loading_scale
        y_end = P[j, iy] * score_range * loading_scale

        if wavelengths is None:
            label = f"band {j}"
        else:
            label = f"{wavelengths[j]:.1f} nm"

        fig.add_annotation(
            x=x_end,
            y=y_end,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            text=label,
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"PC{pcx}",
        yaxis_title=f"PC{pcy}",
        width=850,
        height=700,
    )

    fig.show()


def plot_pca_hotelling_t2(
    pca_res,
    object_ids=None,
    labels=None,
    source_images=None,
    n_components=None,
    title="PCA Hotelling T²",
):
    """
    Plot Hotelling T² for each observation.
    """
    t2 = hotelling_t2(pca_res, n_components=n_components)

    n = len(t2)
    x = np.arange(n)

    if object_ids is None:
        object_ids = np.array([str(i) for i in x])
    if labels is None:
        labels = np.array([""] * n)
    if source_images is None:
        source_images = np.array([""] * n)

    customdata = np.stack(
        [
            np.asarray(object_ids).astype(str),
            np.asarray(labels).astype(str),
            np.asarray(source_images).astype(str),
        ],
        axis=1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=t2,
            mode="markers",
            customdata=customdata,
            marker=dict(size=8),
            hovertemplate=(
                "index: %{x}<br>"
                "T²: %{y:.4f}<br>"
                "object: %{customdata[0]}<br>"
                "label: %{customdata[1]}<br>"
                "source: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Observation index",
        yaxis_title="Hotelling T²",
        width=900,
        height=500,
    )

    fig.show()


def plot_pca_q_residuals(
    X_centered,
    pca_res,
    object_ids=None,
    labels=None,
    source_images=None,
    n_components=None,
    title="PCA Q residuals",
):
    """
    Plot Q residuals for each observation.
    """
    Q, _ = q_residuals(
        X_centered,
        pca_res,
        n_components=n_components,
    )

    n = len(Q)
    x = np.arange(n)

    if object_ids is None:
        object_ids = np.array([str(i) for i in x])
    if labels is None:
        labels = np.array([""] * n)
    if source_images is None:
        source_images = np.array([""] * n)

    customdata = np.stack(
        [
            np.asarray(object_ids).astype(str),
            np.asarray(labels).astype(str),
            np.asarray(source_images).astype(str),
        ],
        axis=1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=Q,
            mode="markers",
            customdata=customdata,
            marker=dict(size=8),
            hovertemplate=(
                "index: %{x}<br>"
                "Q: %{y:.4f}<br>"
                "object: %{customdata[0]}<br>"
                "label: %{customdata[1]}<br>"
                "source: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Observation index",
        yaxis_title="Q residual",
        width=900,
        height=500,
    )

    fig.show()


def plot_pca_q_vs_t2(
    X_centered,
    pca_res,
    object_ids=None,
    labels=None,
    source_images=None,
    n_components=None,
    title="PCA diagnostic plot: Q residuals vs Hotelling T²",
):
    """
    Plot Q residuals against Hotelling T².
    Useful for detecting outliers.
    """
    Q, _ = q_residuals(
        X_centered,
        pca_res,
        n_components=n_components,
    )

    T2 = hotelling_t2(
        pca_res,
        n_components=n_components,
    )

    n = len(Q)

    if object_ids is None:
        object_ids = np.array([str(i) for i in range(n)])
    if labels is None:
        labels = np.array(["all"] * n)
    if source_images is None:
        source_images = np.array([""] * n)

    labels = np.asarray(labels)
    customdata = np.stack(
        [
            np.asarray(object_ids).astype(str),
            np.asarray(labels).astype(str),
            np.asarray(source_images).astype(str),
        ],
        axis=1,
    )

    fig = go.Figure()

    for cls in np.unique(labels):
        mask = labels == cls

        fig.add_trace(
            go.Scatter(
                x=T2[mask],
                y=Q[mask],
                mode="markers",
                name=str(cls),
                customdata=customdata[mask],
                marker=dict(size=9, opacity=0.8),
                hovertemplate=(
                    "T²: %{x:.4f}<br>"
                    "Q: %{y:.4f}<br>"
                    "object: %{customdata[0]}<br>"
                    "label: %{customdata[1]}<br>"
                    "source: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hotelling T²",
        yaxis_title="Q residual",
        width=800,
        height=650,
    )

    fig.show()



# SIMCA PLOTTING FUNCTIONS

def plot_simca_distance_plot(
    simca_results,
    class_name,
    labels=None,
    object_ids=None,
    source_images=None,
    normalized=True,
    title=None,
    width=850,
    height=650,
):
    """
    Plot SIMCA H/Q distance plot for one target class.

    If normalized=True:
        x = H / H_limit
        y = Q / Q_limit

    For Simple SIMCA, accepted objects are in the lower-left square:
        H/H_limit < 1 and Q/Q_limit < 1.
    """
    res = simca_results[class_name]

    if normalized:
        x = np.asarray(res["H_norm_limit"])
        y = np.asarray(res["Q_norm_limit"])
        x_title = "H / H_limit"
        y_title = "Q / Q_limit"
        vline = 1.0
        hline = 1.0
    else:
        x = np.asarray(res["H"])
        y = np.asarray(res["Q"])
        x_title = "H"
        y_title = "Q"
        vline = res["H_limit"]
        hline = res["Q_limit"]

    n = len(x)

    if labels is None:
        labels = np.array(["all"] * n)
    else:
        labels = np.asarray(labels)

    if object_ids is None:
        object_ids = np.array([""] * n)
    else:
        object_ids = np.asarray(object_ids)

    if source_images is None:
        source_images = np.array([""] * n)
    else:
        source_images = np.asarray(source_images)

    accepted = np.asarray(res["accepted"]).astype(str)

    customdata = np.stack(
        [
            object_ids.astype(str),
            labels.astype(str),
            source_images.astype(str),
            accepted,
            np.asarray(res["rule_statistic"]).astype(str),
        ],
        axis=1,
    )

    fig = go.Figure()

    for lab in np.unique(labels):
        mask = labels == lab

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=str(lab),
                customdata=customdata[mask],
                marker=dict(size=9, opacity=0.8),
                hovertemplate=(
                    f"{x_title}: %{{x:.4f}}<br>"
                    f"{y_title}: %{{y:.4f}}<br>"
                    "object: %{customdata[0]}<br>"
                    "true label: %{customdata[1]}<br>"
                    "source: %{customdata[2]}<br>"
                    "accepted: %{customdata[3]}<br>"
                    "rule statistic: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=vline, line_dash="dash")
    fig.add_hline(y=hline, line_dash="dash")

    if title is None:
        title = f"SIMCA distance plot — target class: {class_name}"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_prediction_counts(
    results_df,
    true_label_col="true_label",
    decision_col="simca_case",
    title="SIMCA prediction counts",
    width=850,
    height=500,
):
    """
    Barplot of SIMCA decision cases by true label.

    Expected decision cases:
    - almond_only
    - peanut_only
    - ambiguous
    - unknown
    """
    import pandas as pd

    counts = (
        results_df
        .groupby([true_label_col, decision_col])
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()

    for label in counts[true_label_col].unique():
        sub = counts[counts[true_label_col] == label]

        fig.add_trace(
            go.Bar(
                x=sub[decision_col],
                y=sub["count"],
                name=str(label),
                hovertemplate=(
                    "true label: %{fullData.name}<br>"
                    "decision: %{x}<br>"
                    "count: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="SIMCA decision",
        yaxis_title="Count",
        barmode="group",
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_rule_statistic(
    simca_results,
    class_name,
    labels=None,
    object_ids=None,
    source_images=None,
    title=None,
    width=900,
    height=500,
):
    """
    Plot the rule statistic for one target class.

    For:
    - AltSIMCA: statistic = H/Hlim + Q/Qlim
    - CombinedIndex: statistic = C
    - DataDriven: statistic = D
    """
    res = simca_results[class_name]

    stat = np.asarray(res["rule_statistic"])
    limit = res["rule_limit"]

    n = len(stat)
    x = np.arange(n)

    if labels is None:
        labels = np.array(["all"] * n)
    else:
        labels = np.asarray(labels)

    if object_ids is None:
        object_ids = np.array([""] * n)
    else:
        object_ids = np.asarray(object_ids)

    if source_images is None:
        source_images = np.array([""] * n)
    else:
        source_images = np.asarray(source_images)

    accepted = np.asarray(res["accepted"]).astype(str)

    customdata = np.stack(
        [
            object_ids.astype(str),
            labels.astype(str),
            source_images.astype(str),
            accepted,
        ],
        axis=1,
    )

    fig = go.Figure()

    for lab in np.unique(labels):
        mask = labels == lab

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=stat[mask],
                mode="markers",
                name=str(lab),
                customdata=customdata[mask],
                marker=dict(size=8, opacity=0.8),
                hovertemplate=(
                    "index: %{x}<br>"
                    "statistic: %{y:.4f}<br>"
                    "object: %{customdata[0]}<br>"
                    "true label: %{customdata[1]}<br>"
                    "source: %{customdata[2]}<br>"
                    "accepted: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=limit, line_dash="dash")

    if title is None:
        title = (
            f"SIMCA rule statistic — class={class_name} — "
            f"rule={res['rule_name']}"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Observation index",
        yaxis_title="Rule statistic",
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_object_map(
    image_db,
    object_db,
    results_df,
    image_id,
    decision_col="simca_case",
    object_id_col="object_id",
    title=None,
    width=850,
    height=750,
):
    """
    Overlay object-level SIMCA decisions on the image labels.

    This function colors each object label according to its SIMCA decision.

    decision_col values expected:
    - almond_only
    - peanut_only
    - ambiguous
    - unknown
    """
    img = image_db[image_id]
    labels_img = img["labels"]

    decision_to_code = {
        "unknown": 1,
        "almond_only": 2,
        "peanut_only": 3,
        "ambiguous": 4,
    }

    code_to_name = {
        0: "background",
        1: "unknown",
        2: "almond_only",
        3: "peanut_only",
        4: "ambiguous",
    }

    decision_map = np.zeros_like(labels_img, dtype=float)

    sub = results_df[results_df["source_image"] == image_id]

    for _, row in sub.iterrows():
        obj_id = row[object_id_col]
        decision = row[decision_col]

        if obj_id not in object_db:
            continue

        label_id = object_db[obj_id]["label_id"]
        code = decision_to_code.get(decision, 1)

        decision_map[labels_img == label_id] = code

    decision_map_masked = decision_map.astype(float)
    decision_map_masked[decision_map_masked == 0] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=img["image_ref"],
            colorscale="Gray",
            showscale=True,
            colorbar=dict(title="image_ref"),
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=decision_map_masked,
            colorscale=[
                [0.00, "lightgray"],
                [0.25, "royalblue"],
                [0.50, "crimson"],
                [0.75, "orange"],
                [1.00, "purple"],
            ],
            opacity=0.55,
            showscale=True,
            colorbar=dict(
                title="decision",
                tickvals=[1, 2, 3, 4],
                ticktext=[
                    code_to_name[1],
                    code_to_name[2],
                    code_to_name[3],
                    code_to_name[4],
                ],
                x=1.12,
            ),
            hovertemplate="row: %{y}<br>col: %{x}<br>decision code: %{z}<extra></extra>",
        )
    )

    if title is None:
        title = f"SIMCA object decisions — {image_id}"

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="column",
        yaxis_title="row",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )

    fig.show()





def plot_simca_distance_log_plot(
    simca_model,
    simca_results,
    class_name,
    labels=None,
    object_ids=None,
    source_images=None,
    title=None,
    width=850,
    height=650,
):
    """
    SIMCA distance-distance plot in the paper style:

        x = ln(1 + H/H0)
        y = ln(1 + Q/Q0)

    This corresponds to the style used in Fig. 8 of the SIMCA paper.

    Parameters
    ----------
    simca_model : SIMCAClassifier
        Fitted SIMCA classifier.
    simca_results : dict
        Output of simca.predict(...)[2].
    class_name : str
        Target class model to visualize, e.g. "peanut" or "almond".
    """
    model = simca_model.models_[class_name]
    res = simca_results[class_name]

    H = np.asarray(res["H"], dtype=float)
    Q = np.asarray(res["Q"], dtype=float)

    x = np.log1p(H / model.H0_)
    y = np.log1p(Q / model.Q0_)

    n = len(H)

    if labels is None:
        labels = np.array(["all"] * n)
    else:
        labels = np.asarray(labels)

    if object_ids is None:
        object_ids = np.array([""] * n)
    else:
        object_ids = np.asarray(object_ids)

    if source_images is None:
        source_images = np.array([""] * n)
    else:
        source_images = np.asarray(source_images)

    accepted = np.asarray(res["accepted"]).astype(str)

    customdata = np.stack(
        [
            object_ids.astype(str),
            labels.astype(str),
            source_images.astype(str),
            accepted,
            H.astype(str),
            Q.astype(str),
        ],
        axis=1,
    )

    fig = go.Figure()

    for lab in np.unique(labels):
        mask = labels == lab

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=str(lab),
                customdata=customdata[mask],
                marker=dict(size=9, opacity=0.80),
                hovertemplate=(
                    "ln(1+H/H0): %{x:.4f}<br>"
                    "ln(1+Q/Q0): %{y:.4f}<br>"
                    "object: %{customdata[0]}<br>"
                    "true label: %{customdata[1]}<br>"
                    "source: %{customdata[2]}<br>"
                    "accepted: %{customdata[3]}<br>"
                    "H: %{customdata[4]}<br>"
                    "Q: %{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )

    # Add decision boundary if possible
    rule_name = res["rule_name"]
    rule_limit = res["rule_limit"]

    h_rel_grid = np.linspace(0, max(3, np.nanmax(H / model.H0_) * 1.05), 400)

    if rule_name == "data_driven":
        # D = NH * H/H0 + NQ * Q/Q0 < limit
        q_rel_boundary = (rule_limit - model.NH_ * h_rel_grid) / model.NQ_
        mask = q_rel_boundary >= 0

        fig.add_trace(
            go.Scatter(
                x=np.log1p(h_rel_grid[mask]),
                y=np.log1p(q_rel_boundary[mask]),
                mode="lines",
                name="decision boundary",
                line=dict(dash="dash"),
            )
        )

    elif rule_name in ["alternative", "combined_index"]:
        # H/Hlim + Q/Qlim < limit
        H_lim = res["H_limit"]
        Q_lim = res["Q_limit"]

        H_grid = h_rel_grid * model.H0_
        q_boundary = Q_lim * (rule_limit - H_grid / H_lim)
        q_rel_boundary = q_boundary / model.Q0_

        mask = q_rel_boundary >= 0

        fig.add_trace(
            go.Scatter(
                x=np.log1p(h_rel_grid[mask]),
                y=np.log1p(q_rel_boundary[mask]),
                mode="lines",
                name="decision boundary",
                line=dict(dash="dash"),
            )
        )

    elif rule_name == "simple":
        # Simple SIMCA has a rectangular boundary
        H_lim = res["H_limit"]
        Q_lim = res["Q_limit"]

        fig.add_vline(
            x=np.log1p(H_lim / model.H0_),
            line_dash="dash",
        )
        fig.add_hline(
            y=np.log1p(Q_lim / model.Q0_),
            line_dash="dash",
        )

    if title is None:
        title = (
            f"SIMCA log distance plot — class={class_name} — "
            f"rule={rule_name}"
        )

    fig.update_layout(
        title=title,
        xaxis_title="ln(1 + H/H0)",
        yaxis_title="ln(1 + Q/Q0)",
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_distance_distribution_fit(
    simca_model,
    class_name,
    distance="H",
    nbins=30,
    title=None,
    width=850,
    height=500,
):
    """
    Histogram of H/H0 or Q/Q0 with scaled chi-square density.

    This corresponds to the spirit of Fig. 2 in the SIMCA paper.

    If:
        N * H/H0 ~ chi2(N)
    then the density of R = H/H0 is:
        f_R(r) = N * chi2.pdf(N*r, N)
    """
    model = simca_model.models_[class_name]

    if distance.upper() == "H":
        values = model.H_train_ / model.H0_
        N = model.NH_
        x_title = "H / H0"
        dist_name = "score distance"
    elif distance.upper() == "Q":
        values = model.Q_train_ / model.Q0_
        N = model.NQ_
        x_title = "Q / Q0"
        dist_name = "orthogonal distance"
    else:
        raise ValueError("distance must be 'H' or 'Q'.")

    values = np.asarray(values, dtype=float)

    x_grid = np.linspace(0, max(values.max() * 1.15, 0.1), 500)
    pdf_grid = N * chi2.pdf(N * x_grid, N)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            histnorm="probability density",
            nbinsx=nbins,
            name="empirical",
            opacity=0.65,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf_grid,
            mode="lines",
            name=f"scaled chi-square, DoF={N:.2f}",
        )
    )

    if title is None:
        title = f"{dist_name} distribution fit — class={class_name}"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Density",
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_sensitivity_by_rule(
    simca_summary_df,
    sensitivity_col="peanut_sensitivity",
    expected=0.95,
    tolerance=0.02,
    title=None,
    width=850,
    height=500,
):
    """
    Plot sensitivity by SIMCA rule.

    Similar in spirit to Fig. 3B in the SIMCA paper.
    """
    df = simca_summary_df.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["rule"],
            y=df[sensitivity_col],
            mode="markers+text",
            text=[f"{100*v:.1f}%" for v in df[sensitivity_col]],
            textposition="top center",
            marker=dict(size=12),
            name=sensitivity_col,
        )
    )

    fig.add_hline(
        y=expected,
        line_dash="solid",
        annotation_text=f"expected = {100*expected:.1f}%",
        annotation_position="top left",
    )

    fig.add_hline(
        y=expected - tolerance,
        line_dash="dash",
        annotation_text=f"-{100*tolerance:.1f}%",
        annotation_position="bottom left",
    )

    fig.add_hline(
        y=expected + tolerance,
        line_dash="dash",
        annotation_text=f"+{100*tolerance:.1f}%",
        annotation_position="top left",
    )

    if title is None:
        title = f"Sensitivity by SIMCA rule — {sensitivity_col}"

    fig.update_layout(
        title=title,
        xaxis_title="SIMCA rule",
        yaxis_title="Sensitivity",
        yaxis_tickformat=".0%",
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_extreme_plot(
    model,
    X_target,
    rule_name="data_driven",
    alphas=None,
    title=None,
    width=700,
    height=650,
):
    """
    Extreme plot: observed extremes vs expected extremes.

    This corresponds to Fig. 4 in the SIMCA paper.

    Parameters
    ----------
    model : SIMCAClassModel
        Fitted target class model, e.g. simca.models_["peanut"].
    X_target : ndarray, shape (N, B)
        Target-class samples to evaluate.
        For training extreme plot, use training target samples.
        For projection/test extreme plot, use projected target samples.
    rule_name : str
        "simple", "alternative", "combined_index", or "data_driven".
    alphas : array-like
        Significance values to scan.
    """
    if alphas is None:
        alphas = np.linspace(0.005, 0.25, 60)

    X_target = np.asarray(X_target, dtype=float)
    H, Q, _, _ = model.compute_distances(X_target)

    n = len(H)

    expected = []
    observed = []

    for alpha in alphas:
        accepted, _, _ = _simca_accept_for_rule_alpha(
            H=H,
            Q=Q,
            model=model,
            rule_name=rule_name,
            alpha=alpha,
        )

        n_exp = alpha * n
        n_obs = np.sum(~accepted)

        expected.append(n_exp)
        observed.append(n_obs)

    expected = np.asarray(expected)
    observed = np.asarray(observed)

    # Tolerance corridor as in the paper:
    # n ± 2 sqrt(n(1-n/I)), with n=Iexp
    lower = expected - 2.0 * np.sqrt(expected * (1.0 - expected / n))
    upper = expected + 2.0 * np.sqrt(expected * (1.0 - expected / n))
    lower = np.maximum(lower, 0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([expected, expected[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            mode="lines",
            name="95% tolerance corridor",
            line=dict(width=0),
            opacity=0.25,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=expected,
            y=expected,
            mode="lines",
            name="ideal diagonal",
            line=dict(dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=expected,
            y=observed,
            mode="markers+lines",
            name=f"{rule_name}",
            marker=dict(size=7),
        )
    )

    if title is None:
        title = f"Extreme plot — class={model.class_name} — rule={rule_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Expected extremes",
        yaxis_title="Observed extremes",
        width=width,
        height=height,
    )

    fig.show()


def plot_simca_complexity_metrics(
    complexity_df,
    target_class="peanut",
    title=None,
    width=900,
    height=550,
):
    """
    Plot SIMCA performance metrics versus number of PCs.

    Similar in spirit to Fig. 7 in the SIMCA paper.

    Expected columns:
    - A
    - peanut_sensitivity
    - peanut_specificity_vs_almond
    - almond_sensitivity
    - almond_specificity_vs_peanut
    - ambiguous_rate
    - unknown_rate
    """
    df = complexity_df.copy()

    fig = go.Figure()

    if target_class == "peanut":
        sns_col = "peanut_sensitivity"
        spc_col = "peanut_specificity_vs_almond"
    elif target_class == "almond":
        sns_col = "almond_sensitivity"
        spc_col = "almond_specificity_vs_peanut"
    else:
        raise ValueError("target_class must be 'peanut' or 'almond'.")

    curves = [
        (sns_col, "Sensitivity"),
        (spc_col, "Specificity"),
        ("ambiguous_rate", "Ambiguous rate"),
        ("unknown_rate", "Unknown rate"),
    ]

    for col, name in curves:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["A"],
                    y=df[col],
                    mode="markers+lines",
                    name=name,
                )
            )

    fig.add_hline(
        y=0.95,
        line_dash="dash",
        annotation_text="expected 95%",
        annotation_position="top left",
    )

    if title is None:
        title = f"SIMCA complexity plot — target={target_class}"

    fig.update_layout(
        title=title,
        xaxis_title="Number of PCs A",
        yaxis_title="Metric",
        yaxis_tickformat=".0%",
        width=width,
        height=height,
    )

    fig.show()