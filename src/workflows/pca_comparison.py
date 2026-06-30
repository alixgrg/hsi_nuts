import numpy as np
import pandas as pd

from src.matrices.matrix_registry import build_matrix
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.models.pca import PCAModel
from src.workflows.pca_diagnostic import (
    class_separation_scores,
    compute_pca_summary_metrics,
)


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
    balanced_pixel_strategy="random",
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
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_methods)

    for matrix_method in matrix_methods:
        print(f"\n=== Matrix method: {matrix_method} ===")
        X_raw, y, metadata = build_matrix(
            object_db=object_db,
            matrix_method=matrix_method,
            filters={
                "split": allowed_splits,
                "object_nut_type": allowed_labels,
            },
            m=m,
            random_state=random_state,
            replace=replace,
            balanced_pixel_strategy=balanced_pixel_strategy,
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

        for preproc_name, steps in preprocessing_configs.items():
            steps = tuple(steps)
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