from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skimage import morphology

from src.database import parse_image_key
from src.redim_matrix import object_db_to_matrix
from src.preprocessing import SpectralPreprocessor
from src.pca import PCAModel
from src.simca import (
    SIMCAClassModel,
    SimpleSIMCARule,
    AltSIMCARule,
    CombinedIndexSIMCARule,
    DataDrivenSIMCARule,
)


def matrix_method_to_args(matrix_method: str) -> dict:
    """Map user-facing matrix methods to object_db_to_matrix arguments."""
    if matrix_method == "object_mean":
        return {"level": "object", "spectrum_field": "mean_spectrum"}
    if matrix_method == "object_median":
        return {"level": "object", "spectrum_field": "median_spectrum"}
    if matrix_method == "balanced_pixels":
        return {"level": "balanced_pixel", "spectrum_field": "mean_spectrum"}
    if matrix_method in {"all_pixels", "pixel"}:
        return {"level": "pixel", "spectrum_field": "mean_spectrum"}
    raise ValueError(
        "matrix_method must be one of: "
        "'object_mean', 'object_median', 'balanced_pixels', 'all_pixels'."
    )


def build_training_matrix(
    object_db,
    matrix_method: str,
    filters: dict,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
):
    """Build a training matrix according to matrix_method."""
    args = matrix_method_to_args(matrix_method)
    return object_db_to_matrix(
        object_db=object_db,
        level=args["level"],
        spectrum_field=args["spectrum_field"],
        filters=filters,
        m=m,
        random_state=random_state,
        replace=replace,
    )


def build_projection_pixel_matrix(object_db, filters: dict | None = None):
    """Build the projection matrix at pixel level."""
    return object_db_to_matrix(
        object_db=object_db,
        level="pixel",
        filters=filters or {},
    )


def make_simca_rule(rule_name: str = "alternative"):
    rule_name = str(rule_name).lower()
    if rule_name == "simple":
        return SimpleSIMCARule()
    if rule_name in {"alternative", "alt"}:
        return AltSIMCARule(threshold=2.0)
    if rule_name == "combined_index":
        return CombinedIndexSIMCARule()
    if rule_name == "data_driven":
        return DataDrivenSIMCARule()
    raise ValueError(
        "rule_name must be one of: simple, alternative, combined_index, data_driven."
    )


def fit_pca_for_pixel_projection(
    object_db,
    matrix_method: str,
    train_filters: dict,
    preprocessing_steps=("absorbance", "snv", "sg_d1"),
    n_components: int = 5,
    wavelengths=None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
):
    """Fit PCA on object_mean or balanced_pixels, then later project pixels."""
    X_train_raw, y_train, meta_train = build_training_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
    )

    preprocessor = SpectralPreprocessor(
        steps=preprocessing_steps,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )
    X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)
    pca = PCAModel(n_components=n_components, center=True).fit(X_train)

    return {
        "matrix_method": matrix_method,
        "preprocessor": preprocessor,
        "pca": pca,
        "X_train_raw": X_train_raw,
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
    }


def project_pixels_with_pca(object_db, pca_bundle: dict, projection_filters: dict | None = None):
    """Project all pixels into a fitted PCA model."""
    X_pixel_raw, y_pixel, meta_pixel = build_projection_pixel_matrix(
        object_db=object_db,
        filters=projection_filters,
    )
    X_pixel = pca_bundle["preprocessor"].transform(X_pixel_raw)
    scores = pca_bundle["pca"].transform(X_pixel)

    df = pd.DataFrame(meta_pixel)
    df["label"] = y_pixel.astype(str)
    for k in range(scores.shape[1]):
        df[f"PC{k+1}"] = scores[:, k]
    return df, scores, X_pixel


def fit_one_class_peanut_simca(
    object_db,
    matrix_method: str,
    train_filters: dict,
    preprocessing_steps=("absorbance", "snv", "sg_d1"),
    n_components: int = 5,
    alpha: float = 0.01,
    rule_name: str = "alternative",
    wavelengths=None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
):
    """
    Fit a one-class SIMCA peanut model.
    train_filters should select only peanut observations.
    """
    X_train_raw, y_train, meta_train = build_training_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
    )

    preprocessor = SpectralPreprocessor(
        steps=preprocessing_steps,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )
    X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)

    model = SIMCAClassModel(
        class_name="peanut",
        n_components=n_components,
        alpha=alpha,
    )
    model.fit(X_train)

    rule = make_simca_rule(rule_name)
    rule.fit(model)

    return {
        "matrix_method": matrix_method,
        "preprocessor": preprocessor,
        "model": model,
        "rule": rule,
        "X_train_raw": X_train_raw,
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
    }


def predict_pixels_with_simca(object_db, simca_bundle: dict, projection_filters: dict | None = None):
    """Apply a fitted one-class peanut SIMCA model to all selected pixels."""
    X_pixel_raw, y_pixel, meta_pixel = build_projection_pixel_matrix(
        object_db=object_db,
        filters=projection_filters,
    )

    X_pixel = simca_bundle["preprocessor"].transform(X_pixel_raw)
    model = simca_bundle["model"]
    rule = simca_bundle["rule"]

    values = model.decision_values(X_pixel)
    accepted = rule.accept(values["H"], values["Q"], model)
    rule_statistic = rule.statistic(values["H"], values["Q"], model)
    rule_limit = rule.limit(model)

    df = pd.DataFrame(meta_pixel)
    df["label"] = y_pixel.astype(str)
    df["predicted_peanut_pixel"] = accepted.astype(bool)
    df["predicted_label_pixel"] = np.where(accepted, "peanut", "non_peanut")
    df["H"] = values["H"]
    df["Q"] = values["Q"]
    df["H_norm_limit"] = values["H_norm_limit"]
    df["Q_norm_limit"] = values["Q_norm_limit"]
    df["rule_statistic"] = rule_statistic
    df["rule_limit"] = float(rule_limit)
    df["rule_name"] = rule.name
    df["matrix_method"] = simca_bundle["matrix_method"]

    for k in range(values["scores"].shape[1]):
        df[f"T{k+1}"] = values["scores"][:, k]

    return df, values, X_pixel


def expected_position_key_for_mixture(mixture_clean_key: str) -> str:
    """Convert mixture key to matching position-reference image, e.g. alm3pea2 -> pea2_pos3."""
    meta = parse_image_key(mixture_clean_key)
    if not meta["is_mixture"]:
        raise ValueError(f"Not a mixture key: {mixture_clean_key}")
    components = meta["components"]
    almond_batch = components["almond"]["batch"]
    peanut_batch = components["peanut"]["batch"]
    return f"pea{peanut_batch}_pos{almond_batch}"


def union_object_masks(object_db, source_clean_key: str, shape):
    """Build a binary mask from all objects extracted in one image."""
    out = np.zeros(shape, dtype=bool)
    for _, obj in object_db.items():
        if obj.get("source_clean_key") != source_clean_key:
            continue
        if "mask_global" in obj:
            out |= obj["mask_global"].astype(bool)
        else:
            min_row, min_col, max_row, max_col = obj["bbox"]
            out[min_row:max_row, min_col:max_col] |= obj["mask"].astype(bool)
    return out


def peanut_truth_map_for_image(
    image_key: str,
    image_db,
    object_db,
    dilation_radius: int = 3,
):
    """
    Build a pixel-level peanut truth map for one image.
    Pure images use the known class; mixtures use the matching peaP_posA image.
    """
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    img = image_db[image_key]
    shape = img["labels"].shape
    object_area = img["labels"] > 0
    truth = np.zeros(shape, dtype=bool)
    available = object_area.copy()

    if img.get("is_pure", False):
        truth[object_area] = img.get("nut_type") == "peanut"
        return truth, available

    if img.get("is_position_reference", False):
        truth[object_area] = img.get("nut_type") == "peanut"
        return truth, available

    if img.get("is_mixture", False):
        pos_key = expected_position_key_for_mixture(image_key)
        if pos_key not in image_db:
            return truth, np.zeros(shape, dtype=bool)

        ref_mask = union_object_masks(
            object_db=object_db,
            source_clean_key=pos_key,
            shape=shape,
        )
        if dilation_radius and dilation_radius > 0:
            ref_mask = morphology.binary_dilation(
                ref_mask,
                footprint=morphology.disk(dilation_radius),
            )

        truth[object_area] = ref_mask[object_area]
        return truth, available

    return truth, np.zeros(shape, dtype=bool)


def add_pixel_truth_labels(
    pixel_df: pd.DataFrame,
    image_db,
    object_db,
    dilation_radius: int = 3,
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
):
    """Add true_peanut_pixel and truth_available columns to a pixel dataframe."""
    df = pixel_df.copy()
    df["true_peanut_pixel"] = False
    df["truth_available"] = False

    cache = {}
    for image_key in df[source_col].astype(str).unique():
        cache[image_key] = peanut_truth_map_for_image(
            image_key=image_key,
            image_db=image_db,
            object_db=object_db,
            dilation_radius=dilation_radius,
        )

    for image_key, idx in df.groupby(source_col).groups.items():
        truth, available = cache[str(image_key)]
        rows = df.loc[idx, row_col].astype(int).to_numpy()
        cols = df.loc[idx, col_col].astype(int).to_numpy()
        df.loc[idx, "true_peanut_pixel"] = truth[rows, cols]
        df.loc[idx, "truth_available"] = available[rows, cols]

    return df


def aggregate_pixel_predictions_to_objects(
    pixel_df: pd.DataFrame,
    object_db=None,
    object_threshold: float = 0.75,
    truth_threshold: float = 0.50,
):
    """
    Aggregate pixel decisions to object decisions.

    predicted_peanut_object = predicted_peanut_pixels / object_pixels >= object_threshold
    """
    if "predicted_peanut_pixel" not in pixel_df.columns:
        raise ValueError("pixel_df must contain 'predicted_peanut_pixel'.")

    df = pixel_df.copy()
    df["predicted_peanut_pixel"] = df["predicted_peanut_pixel"].astype(bool)

    agg_dict = {
        "n_pixels_projected": ("predicted_peanut_pixel", "size"),
        "n_predicted_peanut_pixels": ("predicted_peanut_pixel", "sum"),
        "peanut_pixel_ratio": ("predicted_peanut_pixel", "mean"),
        "H_mean": ("H", "mean"),
        "Q_mean": ("Q", "mean"),
        "rule_statistic_mean": ("rule_statistic", "mean"),
    }
    if "true_peanut_pixel" in df.columns:
        agg_dict["true_peanut_pixel_ratio"] = ("true_peanut_pixel", "mean")
    if "truth_available" in df.columns:
        agg_dict["truth_available_ratio"] = ("truth_available", "mean")

    out = df.groupby(["object_id", "source_image"], as_index=False).agg(**agg_dict)
    out["predicted_peanut_object"] = out["peanut_pixel_ratio"] >= object_threshold
    out["predicted_label_object"] = np.where(out["predicted_peanut_object"], "peanut", "non_peanut")
    out["object_threshold"] = float(object_threshold)

    if "true_peanut_pixel_ratio" in out.columns:
        out["true_peanut_object"] = out["true_peanut_pixel_ratio"] >= truth_threshold
        out["true_label_object"] = np.where(out["true_peanut_object"], "peanut", "non_peanut")

    if object_db is not None:
        extra_rows = []
        for obj_id in out["object_id"]:
            obj = object_db.get(obj_id, {})
            centroid = obj.get("centroid", (np.nan, np.nan))
            extra_rows.append({
                "area_pixels": obj.get("area_pixels", np.nan),
                "batch": obj.get("batch", None),
                "sample_kind": obj.get("sample_kind", None),
                "object_nut_type": obj.get("object_nut_type", None),
                "centroid_row": centroid[0],
                "centroid_col": centroid[1],
            })
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(extra_rows)], axis=1)

    return out


def binary_detection_metrics(df: pd.DataFrame, true_col: str, pred_col: str):
    """Compute binary peanut detection metrics."""
    d = df.dropna(subset=[true_col, pred_col]).copy()
    if len(d) == 0:
        return {"n": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0,
                "peanut_sensitivity": np.nan, "almond_specificity": np.nan,
                "balanced_accuracy": np.nan}

    y_true = d[true_col].astype(bool).to_numpy()
    y_pred = d[pred_col].astype(bool).to_numpy()

    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    sens = tp / (tp + fn) if tp + fn > 0 else np.nan
    spec = tn / (tn + fp) if tn + fp > 0 else np.nan
    ba = 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan

    return {"n": int(len(d)), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "peanut_sensitivity": sens, "almond_specificity": spec,
            "balanced_accuracy": ba}


def object_threshold_grid(pixel_df, object_db=None, thresholds=(0.3, 0.5, 0.7, 0.8, 0.9)):
    """Evaluate several object thresholds."""
    rows = []
    object_tables = {}
    for thr in thresholds:
        obj_df = aggregate_pixel_predictions_to_objects(
            pixel_df=pixel_df,
            object_db=object_db,
            object_threshold=thr,
        )
        object_tables[thr] = obj_df
        if "true_peanut_object" in obj_df.columns:
            metrics = binary_detection_metrics(
                obj_df,
                true_col="true_peanut_object",
                pred_col="predicted_peanut_object",
            )
            metrics["object_threshold"] = thr
            rows.append(metrics)
    return pd.DataFrame(rows), object_tables


def make_pixel_error_map(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    pred_col: str = "predicted_peanut_pixel",
    true_col: str = "true_peanut_pixel",
    truth_available_col: str = "truth_available",
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
):
    """
    Create a pixel map with codes:
    0 background / no truth; 1 TP; 2 TN; 3 FP; 4 FN.
    """
    shape = image_db[image_key]["image_ref"].shape
    err = np.zeros(shape, dtype=np.uint8)
    sub = pixel_df[pixel_df[source_col].astype(str) == str(image_key)]
    if sub.empty:
        return err

    rows = sub[row_col].astype(int).to_numpy()
    cols = sub[col_col].astype(int).to_numpy()
    pred = sub[pred_col].astype(bool).to_numpy()
    truth = sub[true_col].astype(bool).to_numpy()
    available = (
        sub[truth_available_col].astype(bool).to_numpy()
        if truth_available_col in sub.columns
        else np.ones(len(sub), dtype=bool)
    )

    codes = np.zeros(len(sub), dtype=np.uint8)
    codes[available & truth & pred] = 1
    codes[available & (~truth) & (~pred)] = 2
    codes[available & (~truth) & pred] = 3
    codes[available & truth & (~pred)] = 4
    err[rows, cols] = codes
    return err


def make_pixel_prediction_map(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    pred_col: str = "predicted_peanut_pixel",
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
):
    """Create a binary map of pixels predicted as peanut."""
    shape = image_db[image_key]["image_ref"].shape
    pred_map = np.zeros(shape, dtype=np.uint8)
    sub = pixel_df[pixel_df[source_col].astype(str) == str(image_key)]
    if sub.empty:
        return pred_map

    rows = sub[row_col].astype(int).to_numpy()
    cols = sub[col_col].astype(int).to_numpy()
    pred = sub[pred_col].astype(bool).to_numpy()
    pred_map[rows[pred], cols[pred]] = 1
    return pred_map


def _background_image(image_db, image_key: str, base: str = "image_ref", band: int | None = None):
    img = image_db[image_key]
    if base == "band":
        if band is None:
            band = img["cube"].shape[2] // 2
        return img["cube"][:, :, band]
    return img.get(base, img["image_ref"])


def plot_pixel_error_overlay(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.60,
    show: bool = True,
):
    """Overlay TP/TN/FP/FN pixel errors on the source/reference image."""
    background = _background_image(image_db, image_key, base=base, band=band)
    err = make_pixel_error_map(image_key, image_db, pixel_df)
    overlay = err.astype(float)
    overlay[overlay == 0] = np.nan

    colorscale = [
        [0.00, "limegreen"],   # 1 TP
        [0.33, "royalblue"],   # 2 TN
        [0.66, "orange"],      # 3 FP
        [1.00, "red"],         # 4 FN
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=background, colorscale="Gray", showscale=True,
                             colorbar=dict(title=base)))
    fig.add_trace(go.Heatmap(
        z=overlay,
        zmin=1,
        zmax=4,
        colorscale=colorscale,
        opacity=opacity,
        colorbar=dict(title="error", tickvals=[1, 2, 3, 4],
                      ticktext=["TP", "TN", "FP", "FN"], x=1.12),
    ))
    fig.update_layout(
        title=title or f"Pixel SIMCA errors — {image_key}",
        width=850,
        height=750,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    if show:
        fig.show()
    return fig


def plot_pixel_fp_fn_overlay(
    image_key: str,
    image_db,
    pixel_df: pd.DataFrame,
    base: str = "image_ref",
    band: int | None = None,
    title: str | None = None,
    opacity: float = 0.75,
    show: bool = True,
):
    """Overlay only false positives and false negatives: red=FP, orange=FN."""
    background = _background_image(image_db, image_key, base=base, band=band)
    err = make_pixel_error_map(image_key, image_db, pixel_df)

    overlay = np.zeros_like(err, dtype=float)
    overlay[err == 3] = 1  # FP
    overlay[err == 4] = 2  # FN
    overlay[overlay == 0] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=background, colorscale="Gray", showscale=True,
                             colorbar=dict(title=base)))
    fig.add_trace(go.Heatmap(
        z=overlay,
        zmin=1,
        zmax=2,
        colorscale=[[0.00, "red"], [1.00, "orange"]],
        opacity=opacity,
        colorbar=dict(title="error", tickvals=[1, 2],
                      ticktext=["FP", "FN"], x=1.12),
    ))
    fig.update_layout(
        title=title or f"False positive / false negative pixels — {image_key}",
        width=850,
        height=750,
        xaxis_title="column",
        yaxis_title="row",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    if show:
        fig.show()
    return fig
