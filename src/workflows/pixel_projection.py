from __future__ import annotations

import numpy as np
import pandas as pd

from src.spectra.preprocessing import SpectralPreprocessor
from src.models.simca import SIMCAClassModel
from src.models.simca_rules import make_simca_rule
from src.matrices.matrix_registry import build_matrix


def build_training_matrix(
    object_db,
    matrix_method: str,
    filters: dict,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    balanced_pixel_strategy: str = "random",
):
    """Build a training matrix according to matrix_method."""
    return build_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )


def build_projection_pixel_matrix(object_db, filters: dict | None = None, balanced_pixel_strategy: str = "random"):
    """Build the projection matrix at pixel level."""
    return build_matrix(
        object_db=object_db,
        matrix_method="pixel",
        filters=filters or {},
        balanced_pixel_strategy=balanced_pixel_strategy,
    )



# def fit_pca_for_pixel_projection(
#     object_db,
#     matrix_method: str,
#     train_filters: dict,
#     preprocessing_steps=("absorbance", "snv", "sg_d1"),
#     n_components: int = 5,
#     wavelengths=None,
#     m: int = 40,
#     random_state: int = 42,
#     replace: bool = False,
#     sg_window_length: int = 9,
#     sg_polyorder: int = 2,
#     balanced_pixel_strategy: str = "random",
# ):
#     """Fit PCA on object_mean or balanced_pixels, then later project pixels."""
#     X_train_raw, y_train, meta_train = build_training_matrix(
#         object_db=object_db,
#         matrix_method=matrix_method,
#         filters=train_filters,
#         m=m,
#         random_state=random_state,
#         replace=replace,
#         balanced_pixel_strategy=balanced_pixel_strategy,
#     )

#     preprocessor = SpectralPreprocessor(
#         steps=preprocessing_steps,
#         sg_window_length=sg_window_length,
#         sg_polyorder=sg_polyorder,
#     )
#     X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)
#     pca = PCAModel(n_components=n_components, center=True).fit(X_train)

#     return {
#         "matrix_method": matrix_method,
#         "preprocessor": preprocessor,
#         "pca": pca,
#         "X_train_raw": X_train_raw,
#         "X_train": X_train,
#         "y_train": y_train,
#         "meta_train": meta_train,
#     }


# def project_pixels_with_pca(object_db, pca_bundle: dict, projection_filters: dict | None = None):
#     """Project all pixels into a fitted PCA model."""
#     X_pixel_raw, y_pixel, meta_pixel = build_projection_pixel_matrix(
#         object_db=object_db,
#         filters=projection_filters,
#     )
#     X_pixel = pca_bundle["preprocessor"].transform(X_pixel_raw)
#     scores = pca_bundle["pca"].transform(X_pixel)

#     df = pd.DataFrame(meta_pixel)
#     df["label"] = y_pixel.astype(str)
#     for k in range(scores.shape[1]):
#         df[f"PC{k+1}"] = scores[:, k]
#     return df, scores, X_pixel


def fit_one_class_simca(
    object_db,
    matrix_method: str,
    train_filters: dict,
    target_class: str = "peanut",
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
    balanced_pixel_strategy: str = "random",
):
    X_train_raw, y_train, meta_train = build_training_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )

    preprocessor = SpectralPreprocessor(
        steps=preprocessing_steps,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )

    X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)

    model = SIMCAClassModel(
        class_name=target_class,
        n_components=n_components,
        alpha=alpha,
    )
    model.fit(X_train)

    rule = make_simca_rule(rule_name)
    rule.fit(model)

    return {
        "target_class": target_class,
        "matrix_method": matrix_method,
        "preprocessor": preprocessor,
        "model": model,
        "rule": rule,
        "X_train_raw": X_train_raw,
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
    }


def fit_one_class_peanut_simca(*args, **kwargs):
    kwargs.setdefault("target_class", "peanut")
    return fit_one_class_simca(*args, **kwargs)


# def predict_pixels_with_simca(object_db, simca_bundle: dict, projection_filters: dict | None = None):
#     """Apply a fitted one-class peanut SIMCA model to all selected pixels."""
#     X_pixel_raw, y_pixel, meta_pixel = build_projection_pixel_matrix(
#         object_db=object_db,
#         filters=projection_filters,
#     )

#     X_pixel = simca_bundle["preprocessor"].transform(X_pixel_raw)
#     model = simca_bundle["model"]
#     rule = simca_bundle["rule"]
#     target_class = simca_bundle.get("target_class", model.class_name)
#     pred_col = f"predicted_{target_class}_pixel"

#     values = model.decision_values(X_pixel)
#     accepted = rule.accept(values["H"], values["Q"], model)
#     rule_statistic = rule.statistic(values["H"], values["Q"], model)
#     rule_limit = rule.limit(model)
    

#     df = pd.DataFrame(meta_pixel)
#     df["label"] = y_pixel.astype(str)
#     df[pred_col] = accepted.astype(bool)
#     df["predicted_label_pixel"] = np.where(accepted,target_class, f"non_{target_class}")
#     # Backward compatibility for current notebooks
#     if target_class == "peanut":
#         df["predicted_peanut_pixel"] = df[pred_col]
#     df["predicted_label_pixel"] = np.where(accepted, "peanut", "non_peanut")
#     df["H"] = values["H"]
#     df["Q"] = values["Q"]
#     df["H_norm_limit"] = values["H_norm_limit"]
#     df["Q_norm_limit"] = values["Q_norm_limit"]
#     df["rule_statistic"] = rule_statistic
#     df["rule_limit"] = float(rule_limit)
#     df["rule_name"] = rule.name
#     df["matrix_method"] = simca_bundle["matrix_method"]
    

#     for k in range(values["scores"].shape[1]):
#         df[f"T{k+1}"] = values["scores"][:, k]

#     return df, values, X_pixel


def predict_pixels_with_simca(
    object_db,
    simca_bundle: dict,
    projection_filters: dict | None = None,
    target_class: str | None = None,
):
    """Apply a fitted one-class SIMCA model to selected pixels."""
    X_pixel_raw, y_pixel, meta_pixel = build_projection_pixel_matrix(
        object_db=object_db,
        filters=projection_filters,
    )

    X_pixel = simca_bundle["preprocessor"].transform(X_pixel_raw)
    model = simca_bundle["model"]
    rule = simca_bundle["rule"]

    if target_class is None:
        target_class = simca_bundle.get("target_class", model.class_name)

    pred_col = f"predicted_{target_class}_pixel"

    values = model.decision_values(X_pixel)
    accepted = rule.accept(values["H"], values["Q"], model)
    rule_statistic = rule.statistic(values["H"], values["Q"], model)
    rule_limit = rule.limit(model)

    df = pd.DataFrame(meta_pixel)
    df["label"] = y_pixel.astype(str)

    df[pred_col] = accepted.astype(bool)
    df["predicted_label_pixel"] = np.where(
        accepted,
        target_class,
        f"non_{target_class}",
    )

    # Backward compatibility for existing notebooks
    if target_class == "peanut":
        df["predicted_peanut_pixel"] = df[pred_col]

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