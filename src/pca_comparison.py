import numpy as np
import pandas as pd

from src.redim_matrix import (
    object_db_to_object_matrix,
    object_db_to_pixel_matrix,
    object_db_to_balanced_px_matrix,
)
from src.preprocessing import (
    center_X,
    snv,
    vector_normalize,
    msc_fit,
    msc_transform,
    savgol_derivative,
    reflectance_to_absorbance,
)
from src.pca import pca_from_cov


def apply_preprocessing_for_pca(
    X,
    method="raw",
    wavelengths=None,
    sg_window_length=9,
    sg_polyorder=2,
):
    """
    Apply one spectral preprocessing method before centering + PCA.

    Returns
    -------
    X_pre : ndarray
        Preprocessed matrix.
    info : dict
        Information needed to reuse the preprocessing later.
    """
    X = np.asarray(X, dtype=float)
    info = {"method": method}

    if method == "raw":
        X_pre = X.copy()
    elif method == "absorbance":
        X_pre = reflectance_to_absorbance(X)
    elif method == "snv":
        X_pre = snv(X)
    elif method == "absorbance_snv":
        X_abs = reflectance_to_absorbance(X)
        X_pre = snv(X_abs)
    elif method == "vector_norm":
        X_pre = vector_normalize(X)
    elif method == "msc":
        ref = msc_fit(X)
        X_pre = msc_transform(X, ref)
        info["msc_reference"] = ref

    elif method == "sg_d1":
        if wavelengths is None:
            delta = 1.0
        else:
            delta = float(np.mean(np.diff(wavelengths)))
        X_pre = savgol_derivative(
            X,
            window_length=sg_window_length,
            polyorder=sg_polyorder,
            deriv=1,
            delta=delta,
        )
        info["sg_window_length"] = sg_window_length
        info["sg_polyorder"] = sg_polyorder
        info["sg_deriv"] = 1
        info["delta"] = delta

    elif method == "sg_d2":
        if wavelengths is None:
            delta = 1.0
        else:
            delta = float(np.mean(np.diff(wavelengths)))
        X_pre = savgol_derivative(
            X,
            window_length=max(sg_window_length, 11),
            polyorder=max(sg_polyorder, 3),
            deriv=2,
            delta=delta,
        )
        info["sg_window_length"] = max(sg_window_length, 11)
        info["sg_polyorder"] = max(sg_polyorder, 3)
        info["sg_deriv"] = 2
        info["delta"] = delta
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    return X_pre, info


def class_separation_scores(T, y, n_components=3):
    """
    Compute simple class separation diagnostics in PCA score space.

    This assumes two classes, e.g. almond / peanut.

    Metrics:
    - centroid_distance_pc1_pc2:
        distance between class centroids in PC1-PC2 space

    - fisher_pc1, fisher_pc2, fisher_pc3:
        between-class squared distance divided by within-class variance
        for each PC.

    These are not classification accuracies.
    They are just quick diagnostics.
    """
    T = np.asarray(T)
    y = np.asarray(y)
    classes = np.unique(y)

    if len(classes) != 2:
        return {
            "centroid_distance_pc1_pc2": np.nan,
            "fisher_pc1": np.nan,
            "fisher_pc2": np.nan,
            "fisher_pc3": np.nan,
            "mahalanobis_pc1_pc2": np.nan,
            "mahalanobis_pc1_pc2_pc3": np.nan,
        }
    
    c0, c1 = classes
    T0 = T[y == c0]
    T1 = T[y == c1]

    # Centroid distance in PC1-PC2
    if T.shape[1] >= 2:
        m0_2d = np.mean(T0[:, :2], axis=0)
        m1_2d = np.mean(T1[:, :2], axis=0)
        centroid_dist = float(np.linalg.norm(m0_2d - m1_2d))
    else:
        centroid_dist = np.nan
    # Univariate Fisher scores for PC1, PC2, PC3
    fisher_values = []
    for a in range(3):
        if T.shape[1] <= a:
            fisher_values.append(np.nan)
            continue
        mean0 = np.mean(T0[:, a])
        mean1 = np.mean(T1[:, a])
        var0 = np.var(T0[:, a], ddof=1) if T0.shape[0] > 1 else 0.0
        var1 = np.var(T1[:, a], ddof=1) if T1.shape[0] > 1 else 0.0
        fisher = (mean0 - mean1) ** 2 / (var0 + var1 + 1e-12)
        fisher_values.append(float(fisher))
    # Mahalanobis distances
    mahal_2d = mahalanobis_centroid_distance(T, y, n_components=2,reg=1e-6)
    mahal_3d = mahalanobis_centroid_distance(T, y, n_components=3,reg=1e-6)

    return {
        "centroid_distance_pc1_pc2": centroid_dist,
        "fisher_pc1": fisher_values[0],
        "fisher_pc2": fisher_values[1],
        "fisher_pc3": fisher_values[2],
        "mahalanobis_pc1_pc2": mahal_2d,
        "mahalanobis_pc1_pc2_pc3": mahal_3d,
    }



def mahalanobis_centroid_distance(T, y, n_components=3, reg=1e-6):
    """
    Compute regularized Mahalanobis distance between two class centroids
    in PCA score space.

    Parameters
    ----------
    T : ndarray, shape (N, A)
        PCA scores.
    y : ndarray, shape (N,)
        Class labels.
    n_components : int
        Number of PCA components to use.
    reg : float
        Regularization added to covariance diagonal.

    Returns
    -------
    distance : float
        Mahalanobis distance between class centroids.
    """
    T = np.asarray(T)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        return np.nan
    T_a = T[:, :n_components]
    c0, c1 = classes
    T0 = T_a[y == c0]
    T1 = T_a[y == c1]

    if T0.shape[0] < 2 or T1.shape[0] < 2:
        return np.nan

    mu0 = np.mean(T0, axis=0)
    mu1 = np.mean(T1, axis=0)
    S0 = np.cov(T0, rowvar=False)
    S1 = np.cov(T1, rowvar=False)
    n0 = T0.shape[0]
    n1 = T1.shape[0]
    S_pool = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    S_pool = np.atleast_2d(S_pool)
    S_reg = S_pool + reg * np.eye(S_pool.shape[0])
    diff = mu0 - mu1
    d2 = diff.T @ np.linalg.pinv(S_reg) @ diff

    return float(np.sqrt(d2))


def build_matrix_for_pca_method(
    object_db,
    matrix_method="object_mean",
    m=40,
    allowed_splits=None,
    allowed_labels=None,
    random_state=42,
    replace=False,
):
    """
    Build X and metadata for one PCA representation.

    matrix_method options:
    - "object_mean"
    - "object_median"
    - "all_pixels"
    - "balanced_pixels"
    """
    if matrix_method == "object_mean":
        X, y, obs_ids, source_images, batches, areas = object_db_to_object_matrix(
            object_db,
            spectrum_field="mean_spectrum",
            allowed_splits=allowed_splits,
            allowed_labels=allowed_labels,
        )
        metadata = {
            "observation_ids": obs_ids,
            "source_images": source_images,
            "batches": batches,
            "areas": areas,
            "positions": None,
        }

    elif matrix_method == "object_median":
        X, y, obs_ids, source_images, batches, areas = object_db_to_object_matrix(
            object_db,
            spectrum_field="median_spectrum",
            allowed_splits=allowed_splits,
            allowed_labels=allowed_labels,
        )
        metadata = {
            "observation_ids": obs_ids,
            "source_images": source_images,
            "batches": batches,
            "areas": areas,
            "positions": None,
        }

    elif matrix_method == "all_pixels":
        X, y, obs_ids, source_images, positions = object_db_to_pixel_matrix(
            object_db,
            allowed_splits=allowed_splits,
            allowed_labels=allowed_labels,
        )
        metadata = {
            "observation_ids": obs_ids,
            "source_images": source_images,
            "batches": None,
            "areas": None,
            "positions": positions,
        }

    elif matrix_method == "balanced_pixels":
        X, y, obs_ids, source_images, positions = object_db_to_balanced_px_matrix(
            object_db,
            m=m,
            allowed_splits=allowed_splits,
            allowed_labels=allowed_labels,
            random_state=random_state,
            replace=replace,
        )
        metadata = {
            "observation_ids": obs_ids,
            "source_images": source_images,
            "batches": None,
            "areas": None,
            "positions": positions,
        }

    else:
        raise ValueError(
            "matrix_method must be one of: "
            "'object_mean', 'object_median', 'all_pixels', 'balanced_pixels'."
        )
    return X, y, metadata


def compare_pca_representations(
    object_db,
    matrix_methods=("object_mean", "all_pixels", "balanced_pixels"),
    preprocessing_methods=("raw", "snv", "vector_norm", "msc", "sg_d1"),
    allowed_splits=("train_minimal",),
    allowed_labels=("almond", "peanut"),
    n_components=5,
    m=40,
    wavelengths=None,
    random_state=42,
    replace=False,
    sg_window_length=9,
    sg_polyorder=2,
):
    """
    Systematically compare PCA results across:
    - several matrix construction methods
    - several spectral preprocessing methods

    Returns
    -------
    summary_df : pandas.DataFrame
        Compact comparison table.

    results : dict
        Full results for plotting and further analysis.
        Access example:
            results["object_mean"]["snv"]["pca"]
            results["balanced_pixels"]["sg_d1"]["X_centered"]
    """
    allowed_splits = list(allowed_splits) if allowed_splits is not None else None
    allowed_labels = list(allowed_labels) if allowed_labels is not None else None
    rows = []
    results = {}

    for matrix_method in matrix_methods:
        print(f"\n=== Matrix method: {matrix_method} ===")
        X_raw, y, metadata = build_matrix_for_pca_method(
            object_db=object_db,
            matrix_method=matrix_method,
            m=m,
            allowed_splits=allowed_splits,
            allowed_labels=allowed_labels,
            random_state=random_state,
            replace=replace,
        )
        results[matrix_method] = {}
        label_values, label_counts = np.unique(y, return_counts=True)
        label_count_dict = {
            str(label): int(count)
            for label, count in zip(label_values, label_counts)
        }
        print(f"X shape: {X_raw.shape}")
        print(f"Labels: {label_count_dict}")

        for preproc in preprocessing_methods:
            print(f"  - preprocessing: {preproc}")
            X_pre, preproc_info = apply_preprocessing_for_pca(
                X_raw,
                method=preproc,
                wavelengths=wavelengths,
                sg_window_length=sg_window_length,
                sg_polyorder=sg_polyorder,
            )
            X_c, mu = center_X(X_pre)
            pca_res = pca_from_cov(
                X_c,
                n_components=n_components,
            )
            evr = pca_res["explained_variance_ratio"]
            cum = pca_res["cumulative_explained_variance_ratio"]
            T = pca_res["scores"]
            sep = class_separation_scores(
                T,
                y,
                n_components=n_components,
            )
            row = {
                "matrix_method": matrix_method,
                "preprocessing": preproc,
                "n_observations": int(X_raw.shape[0]),
                "n_bands": int(X_raw.shape[1]),
                "n_components": int(n_components),
                "m": int(m) if matrix_method == "balanced_pixels" else np.nan,
                "n_almond": label_count_dict.get("almond", 0),
                "n_peanut": label_count_dict.get("peanut", 0),
                "evr_pc1": float(evr[0]) if len(evr) > 0 else np.nan,
                "evr_pc2": float(evr[1]) if len(evr) > 1 else np.nan,
                "evr_pc3": float(evr[2]) if len(evr) > 2 else np.nan,
                "cum_pc2": float(cum[1]) if len(cum) > 1 else np.nan,
                "cum_pc3": float(cum[2]) if len(cum) > 2 else np.nan,
                "centroid_distance_pc1_pc2": sep["centroid_distance_pc1_pc2"],
                "fisher_pc1": sep["fisher_pc1"],
                "fisher_pc2": sep["fisher_pc2"],
                "fisher_pc3": sep["fisher_pc3"],
                "mahalanobis_pc1_pc2": sep["mahalanobis_pc1_pc2"],
                "mahalanobis_pc1_pc2_pc3": sep["mahalanobis_pc1_pc2_pc3"],
            }
            rows.append(row)
            results[matrix_method][preproc] = {
                "X_raw": X_raw,
                "X_preprocessed": X_pre,
                "X_centered": X_c,
                "center_mean": mu,
                "y": y,
                "metadata": metadata,
                "preprocessing_info": preproc_info,
                "pca": pca_res,
                "summary": row,
            }
    summary_df = pd.DataFrame(rows)

    # Useful sorting: strongest quick PC1-PC2 class separation first
    summary_df = summary_df.sort_values(
        by=["fisher_pc1", "fisher_pc2", "centroid_distance_pc1_pc2"],
        ascending=False,
    ).reset_index(drop=True)
    return summary_df, results