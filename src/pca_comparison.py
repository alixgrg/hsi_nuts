import numpy as np
import pandas as pd

from src.redim_matrix import object_db_to_matrix
from src.preprocessing import SpectralPreprocessor
from src.pca import pca_from_cov, PCAModel


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


def n_components_for_cumulative_variance(cumulative_variance, threshold=0.90):
    """
    Return the smallest number of components needed to reach a cumulative
    explained variance threshold.
    """
    cumulative_variance = np.asarray(cumulative_variance, dtype=float)

    if cumulative_variance.size == 0:
        return np.nan

    idx = np.where(cumulative_variance >= threshold)[0]

    if len(idx) == 0:
        return np.nan

    return int(idx[0] + 1)


def trace_ratio_by_group(T, groups, n_components=3, eps=1e-12):
    """
    Compute between-group / within-group variance ratio in score space.

    R = trace(S_between) / trace(S_within)

    Parameters
    ----------
    T : ndarray, shape (n_samples, n_components_total)
        PCA scores.
    groups : array-like, shape (n_samples,)
        Group labels, e.g. class labels or batch labels.
    n_components : int
        Number of PCA components used.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    ratio : float
        Between-group / within-group trace ratio.
    """
    T = np.asarray(T, dtype=float)[:, :n_components]
    groups = np.asarray(groups).astype(str)

    valid = (
        np.isfinite(T).all(axis=1)
        & (groups != "None")
        & (groups != "nan")
        & (groups != "unknown")
    )

    T = T[valid]
    groups = groups[valid]

    unique_groups = np.unique(groups)

    if T.shape[0] < 2 or len(unique_groups) < 2:
        return np.nan

    global_mean = np.mean(T, axis=0)

    ss_between = 0.0
    ss_within = 0.0

    for group in unique_groups:
        Tg = T[groups == group]

        if Tg.shape[0] == 0:
            continue

        mean_g = np.mean(Tg, axis=0)

        ss_between += Tg.shape[0] * np.sum((mean_g - global_mean) ** 2)
        ss_within += np.sum((Tg - mean_g) ** 2)

    return float(ss_between / (ss_within + eps))


def pca_distance_summary(
    pca_model,
    X,
    n_components=None,
    prefix="train",
):
    """
    Compute summary statistics for PCA Q-residuals and Hotelling T².

    Parameters
    ----------
    pca_model : PCAModel
        Fitted PCA model.
    X : ndarray
        Data matrix to evaluate.
    n_components : int or None
        Number of components used for reconstruction/distances.
    prefix : str
        Prefix used in returned metric names.

    Returns
    -------
    metrics : dict
        Summary metrics.
    """
    if n_components is None:
        n_components = pca_model.loadings_.shape[1]

    T2 = pca_model.hotelling_t2(X, n_components=n_components)
    Q, _ = pca_model.q_residuals(X, n_components=n_components)

    return {
        f"{prefix}_q_mean": float(np.mean(Q)),
        f"{prefix}_q_median": float(np.median(Q)),
        f"{prefix}_q_q95": float(np.quantile(Q, 0.95)),
        f"{prefix}_t2_mean": float(np.mean(T2)),
        f"{prefix}_t2_median": float(np.median(T2)),
        f"{prefix}_t2_q95": float(np.quantile(T2, 0.95)),
    }


def train_projection_shift_by_label(
    T_train,
    T_projection,
    y_train,
    y_projection,
    n_components=3,
    eps=1e-12,
):
    """
    Measure the shift between train and projection scores for each label.

    For each class c:
        shift_c = ||mean_train_c - mean_projection_c||

    Normalized version:
        shift_norm_c = shift_c / sqrt(trace(cov_train_c))

    Returns mean and max shifts across common labels.
    """
    T_train = np.asarray(T_train, dtype=float)[:, :n_components]
    T_projection = np.asarray(T_projection, dtype=float)[:, :n_components]

    y_train = np.asarray(y_train).astype(str)
    y_projection = np.asarray(y_projection).astype(str)

    common_labels = sorted(set(y_train).intersection(set(y_projection)))

    if len(common_labels) == 0:
        return {
            "mean_train_projection_shift": np.nan,
            "max_train_projection_shift": np.nan,
            "mean_train_projection_shift_norm": np.nan,
            "max_train_projection_shift_norm": np.nan,
        }

    shifts = []
    shifts_norm = []

    for label in common_labels:
        Ttr = T_train[y_train == label]
        Tpr = T_projection[y_projection == label]

        if Ttr.shape[0] == 0 or Tpr.shape[0] == 0:
            continue

        mu_tr = np.mean(Ttr, axis=0)
        mu_pr = np.mean(Tpr, axis=0)

        shift = np.linalg.norm(mu_tr - mu_pr)

        if Ttr.shape[0] > 1:
            cov_tr = np.cov(Ttr, rowvar=False)
            cov_tr = np.atleast_2d(cov_tr)
            scale = np.sqrt(np.trace(cov_tr))
        else:
            scale = eps

        shifts.append(shift)
        shifts_norm.append(shift / (scale + eps))

    if len(shifts) == 0:
        return {
            "mean_train_projection_shift": np.nan,
            "max_train_projection_shift": np.nan,
            "mean_train_projection_shift_norm": np.nan,
            "max_train_projection_shift_norm": np.nan,
        }

    return {
        "mean_train_projection_shift": float(np.mean(shifts)),
        "max_train_projection_shift": float(np.max(shifts)),
        "mean_train_projection_shift_norm": float(np.mean(shifts_norm)),
        "max_train_projection_shift_norm": float(np.max(shifts_norm)),
    }


def pixel_object_score_metrics(
    T,
    y,
    metadata,
    n_components=3,
    object_col="object_id",
    batch_col="batch",
    eps=1e-12,
):
    """
    Compute object-level metrics from pixel-level PCA scores.

    This is useful for all_pixels and balanced_pixels.

    Metrics:
    - object_class_trace_ratio:
        class separation after averaging pixel scores by object
    - object_batch_trace_ratio:
        batch effect after averaging pixel scores by object
    - mean_intra_object_trace:
        average dispersion of pixels inside each object
    - object_over_intra_ratio:
        between-object variance / intra-object variance
    """
    T = np.asarray(T, dtype=float)[:, :n_components]
    y = np.asarray(y).astype(str)

    if object_col not in metadata:
        return {
            "object_class_trace_ratio": np.nan,
            "object_batch_trace_ratio": np.nan,
            "mean_intra_object_trace": np.nan,
            "object_over_intra_ratio": np.nan,
        }

    object_ids = np.asarray(metadata[object_col]).astype(str)
    batches = (
        np.asarray(metadata[batch_col]).astype(str)
        if batch_col in metadata
        else np.array(["unknown"] * len(object_ids))
    )

    df = pd.DataFrame({
        "object_id": object_ids,
        "label": y,
        "batch": batches,
    })

    for a in range(n_components):
        df[f"C{a+1}"] = T[:, a]

    score_cols = [f"C{a+1}" for a in range(n_components)]

    # Object-level mean scores
    df_obj = (
        df.groupby(["object_id", "label", "batch"], as_index=False)
        .agg({col: "mean" for col in score_cols})
    )

    T_obj = df_obj[score_cols].to_numpy(dtype=float)
    y_obj = df_obj["label"].to_numpy()
    batch_obj = df_obj["batch"].to_numpy()

    object_class_ratio = trace_ratio_by_group(
        T_obj,
        y_obj,
        n_components=n_components,
        eps=eps,
    )

    object_batch_ratio = trace_ratio_by_group(
        T_obj,
        batch_obj,
        n_components=n_components,
        eps=eps,
    )

    # Intra-object dispersion
    intra_traces = []

    for _, group in df.groupby("object_id"):
        Tg = group[score_cols].to_numpy(dtype=float)

        if Tg.shape[0] <= 1:
            continue

        cov_g = np.cov(Tg, rowvar=False)
        cov_g = np.atleast_2d(cov_g)
        intra_traces.append(np.trace(cov_g))

    if len(intra_traces) == 0:
        mean_intra = np.nan
    else:
        mean_intra = float(np.mean(intra_traces))

    if T_obj.shape[0] > 1:
        cov_obj = np.cov(T_obj, rowvar=False)
        cov_obj = np.atleast_2d(cov_obj)
        between_object_trace = float(np.trace(cov_obj))
    else:
        between_object_trace = np.nan

    object_over_intra = (
        between_object_trace / (mean_intra + eps)
        if np.isfinite(mean_intra)
        else np.nan
    )

    return {
        "object_class_trace_ratio": float(object_class_ratio),
        "object_batch_trace_ratio": float(object_batch_ratio),
        "mean_intra_object_trace": float(mean_intra),
        "object_over_intra_ratio": float(object_over_intra),
    }


def compute_pca_summary_metrics(
    pca_model,
    X_train,
    T_train,
    y_train,
    metadata_train=None,
    X_projection=None,
    T_projection=None,
    y_projection=None,
    metadata_projection=None,
    n_components=3,
    matrix_method=None,
):
    """
    Compute a full set of PCA comparison metrics.

    Can be used for:
    - object_mean
    - object_median
    - balanced_pixels
    - all_pixels
    """
    y_train = np.asarray(y_train).astype(str)
    T_train = np.asarray(T_train, dtype=float)

    metadata_train = metadata_train or {}

    evr = pca_model.explained_variance_ratio_
    cum = pca_model.cumulative_explained_variance_ratio_

    batches_train = (
        metadata_train.get("batch")
        if "batch" in metadata_train
        else metadata_train.get("batches", None)
    )

    metrics = {
        "evr_pc1": float(evr[0]) if len(evr) > 0 else np.nan,
        "evr_pc2": float(evr[1]) if len(evr) > 1 else np.nan,
        "evr_pc3": float(evr[2]) if len(evr) > 2 else np.nan,
        "cum_pc2": float(cum[1]) if len(cum) > 1 else np.nan,
        "cum_pc3": float(cum[2]) if len(cum) > 2 else np.nan,
        "ncomp_90": n_components_for_cumulative_variance(cum, threshold=0.90),
        "ncomp_95": n_components_for_cumulative_variance(cum, threshold=0.95),
    }

    # Class separation
    metrics["class_trace_ratio"] = trace_ratio_by_group(
        T_train,
        y_train,
        n_components=n_components,
    )

    # Batch effect
    if batches_train is not None:
        metrics["batch_trace_ratio"] = trace_ratio_by_group(
            T_train,
            batches_train,
            n_components=n_components,
        )
    else:
        metrics["batch_trace_ratio"] = np.nan

    metrics["class_over_batch_ratio"] = (
        metrics["class_trace_ratio"] / (metrics["batch_trace_ratio"] + 1e-12)
        if np.isfinite(metrics["batch_trace_ratio"])
        else np.nan
    )

    # Q and T² on train
    metrics.update(
        pca_distance_summary(
            pca_model,
            X_train,
            n_components=n_components,
            prefix="train",
        )
    )

    # Projection metrics
    if X_projection is not None:
        metrics.update(
            pca_distance_summary(
                pca_model,
                X_projection,
                n_components=n_components,
                prefix="projection",
            )
        )

        metrics["projection_train_q_ratio"] = (
            metrics["projection_q_mean"] / (metrics["train_q_mean"] + 1e-12)
        )

    if (
        T_projection is not None
        and y_projection is not None
    ):
        metrics.update(
            train_projection_shift_by_label(
                T_train=T_train,
                T_projection=T_projection,
                y_train=y_train,
                y_projection=y_projection,
                n_components=n_components,
            )
        )

    # Pixel/object metrics
    if matrix_method in {"all_pixels", "balanced_pixels"}:
        metrics.update(
            pixel_object_score_metrics(
                T=T_train,
                y=y_train,
                metadata=metadata_train,
                n_components=n_components,
            )
        )
    else:
        metrics.update({
            "object_class_trace_ratio": np.nan,
            "object_batch_trace_ratio": np.nan,
            "mean_intra_object_trace": np.nan,
            "object_over_intra_ratio": np.nan,
        })

    return metrics



def _preproc_to_steps(preproc):
    if isinstance(preproc, str):
        aliases = {
            "absorbance_snv": ("absorbance", "snv"),
            "absorbance_sg_d1": ("absorbance", "sg_d1"),
            "absorbance_sg_d2": ("absorbance", "sg_d2"),
        }
        return list(aliases.get(preproc, (preproc,)))
    return list(preproc)


def _preproc_name(preproc):
    if isinstance(preproc, str):
        return preproc
    return "+".join(preproc)


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
    level_by_method = {
        "object_mean": "object",
        "object_median": "object",
        "all_pixels": "pixel",
        "balanced_pixels": "balanced_pixel",
    }
    spectrum_field_by_method = {
        "object_mean": "mean_spectrum",
        "object_median": "median_spectrum",
    }

    for matrix_method in matrix_methods:
        if matrix_method not in level_by_method:
            raise ValueError(f"Unknown matrix_method: {matrix_method}")
        print(f"\n=== Matrix method: {matrix_method} ===")
        X_raw, y, metadata = object_db_to_matrix(
            object_db=object_db,
            level=level_by_method[matrix_method],
            spectrum_field=spectrum_field_by_method[matrix_method],
            filters={"split": allowed_splits, "object_nut_type": allowed_labels},
            m=m,
            random_state=random_state,
            replace=replace,
        )
        results[matrix_method] = {}
        metadata = dict(metadata)
        metadata.setdefault("observation_ids", metadata.get("object_id"))
        metadata.setdefault("source_images", metadata.get("source_image"))
        metadata.setdefault("batches", metadata.get("batch"))
        metadata.setdefault("areas", metadata.get("area"))
        label_values, label_counts = np.unique(y, return_counts=True)
        label_count_dict = {
            str(label): int(count)
            for label, count in zip(label_values, label_counts)
        }
        print(f"X shape: {X_raw.shape}")
        print(f"Labels: {label_count_dict}")

        for preproc in preprocessing_methods:
            preproc_name = _preproc_name(preproc)
            steps = _preproc_to_steps(preproc)
            print(f"  - preprocessing: {preproc_name}")
            prep = SpectralPreprocessor(steps, sg_window_length=sg_window_length, sg_polyorder=sg_polyorder)
            X_pre = prep.fit_transform(X_raw, wavelengths=wavelengths)
            pca = PCAModel(n_components=n_components, center=True).fit(X_pre)
            T = pca.scores_
            evr = pca.explained_variance_ratio_
            cum = pca.cumulative_explained_variance_ratio_
            sep = class_separation_scores(T, y, n_components=n_components)
            extra_metrics = compute_pca_summary_metrics(
                pca_model=pca,
                X_train=X_pre,
                T_train=T,
                y_train=y,
                metadata_train=metadata,
                n_components=min(n_components, 3),
                matrix_method=matrix_method,
            )
            row = {
                "matrix_method": matrix_method,
                "preprocessing": preproc_name,
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
                **extra_metrics,
            }
            rows.append(row)
            results[matrix_method][preproc_name] = {
                "X_raw": X_raw,
                "X_preprocessed": X_pre,
                "pca": pca,
                "covariance": pca.covariance_,
                "eigenvalues": pca.eigenvalues_,
                "eigenvectors": pca.eigenvectors_,
                "scores": pca.scores_,
                "loadings": pca.loadings_,
                "explained_variance_ratio" : evr,
                "cum = pca.cumulative_explained_variance_ratio": cum,
                "y": y,
                "metadata": metadata,
                "preprocessing_info": prep,
                "summary": row,
            }
    summary_df = pd.DataFrame(rows)

    # Useful sorting: strongest quick PC1-PC2 class separation first
    summary_df = summary_df.sort_values(
        by=["fisher_pc1", "fisher_pc2", "centroid_distance_pc1_pc2"],
        ascending=False,
    ).reset_index(drop=True)
    return summary_df, results


def add_pca_selection_score(
    summary_df,
    matrix_method_col="matrix_method",
    score_col="selection_score",
    profile="auto",
    group_col=None,
    robust=True,
    clip_quantiles=(0.05, 0.95),
    eps=1e-12,
):
    """
    Add an adapted PCA selection score depending on the matrix representation.

    The score is designed for the NIR UCO almond/peanut project.

    Main idea
    ---------
    For object-level matrices:
        prioritize class separation, low batch effect, and projection stability.

    For pixel-level matrices:
        prioritize object-level class separation after pixel aggregation,
        low object-level batch effect, controlled intra-object dispersion,
        and projection stability.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        PCA metrics table.
    matrix_method_col : str
        Column containing matrix representation names.
    score_col : str
        Name of the output score column.
    profile : {"auto", "object", "pixel"}
        If "auto", choose profile from matrix_method.
    group_col : str or None
        If provided, normalization is done within each group.
        Recommended:
            group_col="matrix_method"
        when comparing preprocessings within each matrix representation.
    robust : bool
        If True, use median/IQR scaling instead of mean/std.
    clip_quantiles : tuple
        Quantile clipping before scaling to reduce outlier influence.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    df : pandas.DataFrame
        Copy of input with adapted score and diagnostic columns.
    """
    df = summary_df.copy()

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    object_positive = {
        # Main objective: separate almond vs peanut
        "class_trace_ratio": 3.0,
        # Secondary discriminant diagnostic
        "mahalanobis_pc1_pc2_pc3": 1.0,
    }
    object_negative = {
        # Avoid learning batch
        "batch_trace_ratio": 2.0,
        # Avoid unstable projection behavior
        "mean_train_projection_shift_norm": 1.5,
        "projection_q_deviation": 1.5,
        # Complexity should matter, but not dominate
        "ncomp_95": 0.3,
    }
    pixel_positive = {
        # Main objective for pixel matrices:
        # object-level separation after aggregating pixel scores
        "object_class_trace_ratio": 3.0,

        # Useful but secondary: objects should be separated relative to
        # their internal pixel dispersion
        "object_over_intra_ratio": 1.0,
    }
    pixel_negative = {
        # Avoid object-level batch structure
        "object_batch_trace_ratio": 2.0,
        # Penalize excessive intra-object pixel dispersion
        "mean_intra_object_trace": 1.0,
        # Projection stability
        "mean_train_projection_shift_norm": 1.2,
        "projection_q_deviation": 1.2,
        # Complexity should not dominate
        "ncomp_95": 0.3,
    }

    def get_profile_for_matrix(matrix_method):
        matrix_method = str(matrix_method)
        if profile == "object":
            return object_positive, object_negative
        if profile == "pixel":
            return pixel_positive, pixel_negative
        if profile != "auto":
            raise ValueError("profile must be 'auto', 'object', or 'pixel'.")
        if matrix_method in {"object_mean", "object_median"}:
            return object_positive, object_negative
        if matrix_method in {"all_pixels", "balanced_pixels"}:
            return pixel_positive, pixel_negative
        # Default: object profile
        return object_positive, object_negative

    # ------------------------------------------------------------------
    # Robust scaling
    # ------------------------------------------------------------------
    def scale_metric(values):
        values = pd.Series(values, dtype="float64")
        if values.notna().sum() <= 1:
            return pd.Series(np.zeros(len(values)), index=values.index)
        lo, hi = values.quantile(clip_quantiles[0]), values.quantile(clip_quantiles[1])
        values_clip = values.clip(lo, hi)
        if robust:
            center = values_clip.median()
            scale = values_clip.quantile(0.75) - values_clip.quantile(0.25)
        else:
            center = values_clip.mean()
            scale = values_clip.std(ddof=0)
        if not np.isfinite(scale) or scale < eps:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values_clip - center) / (scale + eps)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------
    def compute_scores_for_group(g):
        g = g.copy()
        scores = np.zeros(len(g), dtype=float)
        # Store contributions for interpretation
        contribution_cols = []
        for idx, row in g.iterrows():
            pos_weights, neg_weights = get_profile_for_matrix(
                row[matrix_method_col] if matrix_method_col in g.columns else "object_mean"
            )
        # Union of all metrics needed in this group
        all_pos_metrics = set()
        all_neg_metrics = set()
        for mm in g[matrix_method_col].unique() if matrix_method_col in g.columns else ["object_mean"]:
            pos, neg = get_profile_for_matrix(mm)
            all_pos_metrics.update(pos.keys())
            all_neg_metrics.update(neg.keys())
        scaled = {}
        for metric in sorted(all_pos_metrics | all_neg_metrics):
            if metric in g.columns:
                scaled[metric] = scale_metric(g[metric])
            else:
                scaled[metric] = pd.Series(np.zeros(len(g)), index=g.index)
        for _, idx in enumerate(g.index):
            mm = g.loc[idx, matrix_method_col] if matrix_method_col in g.columns else "object_mean"
            pos_weights, neg_weights = get_profile_for_matrix(mm)
            score_i = 0.0
            for metric, weight in pos_weights.items():
                contrib = weight * scaled[metric].loc[idx]
                score_i += contrib
                col = f"contrib_plus_{metric}"
                if col not in g.columns:
                    g[col] = 0.0
                    contribution_cols.append(col)
                g.loc[idx, col] = contrib
            for metric, weight in neg_weights.items():
                contrib = -weight * scaled[metric].loc[idx]
                score_i += contrib
                col = f"contrib_minus_{metric}"
                if col not in g.columns:
                    g[col] = 0.0
                    contribution_cols.append(col)
                g.loc[idx, col] = contrib
            g.loc[idx, score_col] = score_i
        return g

    if group_col is None:
        df = compute_scores_for_group(df)
    else:
        df = (
            df.groupby(group_col, group_keys=False)
            .apply(compute_scores_for_group)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Add a readable decision flag
    # ------------------------------------------------------------------
    def quality_flag(row):
        mm = row.get(matrix_method_col, "object_mean")
        if mm in {"object_mean", "object_median"}:
            sep = row.get("class_trace_ratio", np.nan)
            batch = row.get("batch_trace_ratio", np.nan)
            qdev = row.get("projection_q_deviation", np.nan)
            if pd.notna(sep) and sep < 0.10:
                return "weak_class_separation"
            if pd.notna(qdev) and qdev > 1.0:
                return "unstable_projection"
            if pd.notna(batch) and batch > 0.01:
                return "batch_sensitive"
            return "candidate"
        if mm in {"all_pixels", "balanced_pixels"}:
            sep = row.get("object_class_trace_ratio", np.nan)
            batch = row.get("object_batch_trace_ratio", np.nan)
            qdev = row.get("projection_q_deviation", np.nan)
            if pd.notna(sep) and sep < 0.05:
                return "weak_object_separation"
            if pd.notna(qdev) and qdev > 1.0:
                return "unstable_projection"
            if pd.notna(batch) and batch > 0.005:
                return "batch_sensitive"
            return "candidate"
        return "unknown"

    df["selection_flag"] = df.apply(quality_flag, axis=1)

    return df