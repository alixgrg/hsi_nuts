from __future__ import annotations

import numpy as np
import pandas as pd

from src.matrices.matrix_registry import build_matrix
from src.models.pca import PCAModel
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.preprocessing_configs import normalize_preprocessing_configs


OBJECT_MATRIX_METHODS = {"object_mean", "object_median"}
PIXEL_MATRIX_METHODS = {"balanced_pixels", "all_pixels", "pixel"}


def pca_matrix_family_from_method(matrix_method: str) -> str:
    """Return the PCA matrix family used for summaries and selection."""
    matrix_method = str(matrix_method)

    if matrix_method in OBJECT_MATRIX_METHODS:
        return "object_matrix"

    if matrix_method in PIXEL_MATRIX_METHODS:
        return "pixel_matrix"

    return "unknown_matrix_family"


def pca_matrix_variant_from_method(
    matrix_method: str,
    balanced_pixel_strategy: str = "random",
) -> str:
    """
    Return a stable matrix variant label.

    Examples
    --------
    object_mean -> object_mean
    object_median -> object_median
    balanced_pixels + random -> balanced_pixels_random
    balanced_pixels + center -> balanced_pixels_center
    all_pixels -> all_pixels
    """
    matrix_method = str(matrix_method)

    if matrix_method == "balanced_pixels":
        return f"balanced_pixels_{balanced_pixel_strategy}"

    return matrix_method


# -----------------------------------------------------------------------------
# PCA diagnostics
# -----------------------------------------------------------------------------


def binary_class_separation_scores(T, y, n_components: int = 3) -> dict:
    """Compute quick two-class separation diagnostics in PCA score space."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(y).astype(str)
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

    if T.shape[1] >= 2:
        centroid_dist = float(np.linalg.norm(np.mean(T0[:, :2], axis=0) - np.mean(T1[:, :2], axis=0)))
    else:
        centroid_dist = np.nan

    fisher_values = []
    for axis in range(3):
        if T.shape[1] <= axis:
            fisher_values.append(np.nan)
            continue
        mean0 = np.mean(T0[:, axis])
        mean1 = np.mean(T1[:, axis])
        var0 = np.var(T0[:, axis], ddof=1) if T0.shape[0] > 1 else 0.0
        var1 = np.var(T1[:, axis], ddof=1) if T1.shape[0] > 1 else 0.0
        fisher_values.append(float((mean0 - mean1) ** 2 / (var0 + var1 + 1e-12)))

    return {
        "centroid_distance_pc1_pc2": centroid_dist,
        "fisher_pc1": fisher_values[0],
        "fisher_pc2": fisher_values[1],
        "fisher_pc3": fisher_values[2],
        "mahalanobis_pc1_pc2": mahalanobis_centroid_distance(T, y, n_components=2),
        "mahalanobis_pc1_pc2_pc3": mahalanobis_centroid_distance(T, y, n_components=3),
    }


def mahalanobis_centroid_distance(T, y, n_components: int = 3, reg: float = 1e-6) -> float:
    """Compute regularized Mahalanobis distance between two class centroids."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(y).astype(str)
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
    S0 = np.atleast_2d(np.cov(T0, rowvar=False))
    S1 = np.atleast_2d(np.cov(T1, rowvar=False))
    n0 = T0.shape[0]
    n1 = T1.shape[0]
    S_pool = ((n0 - 1) * S0 + (n1 - 1) * S1) / max(n0 + n1 - 2, 1)
    S_reg = S_pool + float(reg) * np.eye(S_pool.shape[0])
    diff = mu0 - mu1
    d2 = diff.T @ np.linalg.pinv(S_reg) @ diff
    return float(np.sqrt(max(d2, 0.0)))


def n_components_for_cumulative_variance(cumulative_variance, threshold: float = 0.90):
    """Return the smallest number of components needed to reach a variance threshold."""
    cumulative_variance = np.asarray(cumulative_variance, dtype=float)
    if cumulative_variance.size == 0:
        return np.nan
    idx = np.where(cumulative_variance >= float(threshold))[0]
    return int(idx[0] + 1) if len(idx) else np.nan


def trace_ratio_by_group(T, groups, n_components: int = 3, eps: float = 1e-12) -> float:
    """Compute between-group / within-group variance trace ratio in score space."""
    T = np.asarray(T, dtype=float)[:, :n_components]
    groups = np.asarray(groups).astype(str)

    valid = np.isfinite(T).all(axis=1) & ~np.isin(groups, ["None", "nan", "unknown"])
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


def pca_distance_summary(pca_model, X, n_components=None, prefix: str = "train") -> dict:
    """Compute summary statistics for PCA Q-residuals and Hotelling T²."""
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
    n_components: int = 3,
    eps: float = 1e-12,
) -> dict:
    """Measure train-vs-projection centroid shift for labels shared by both sets."""
    T_train = np.asarray(T_train, dtype=float)[:, :n_components]
    T_projection = np.asarray(T_projection, dtype=float)[:, :n_components]
    y_train = np.asarray(y_train).astype(str)
    y_projection = np.asarray(y_projection).astype(str)

    common_labels = sorted(set(y_train).intersection(set(y_projection)))
    if not common_labels:
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
        shift = np.linalg.norm(np.mean(Ttr, axis=0) - np.mean(Tpr, axis=0))
        if Ttr.shape[0] > 1:
            scale = np.sqrt(np.trace(np.atleast_2d(np.cov(Ttr, rowvar=False))))
        else:
            scale = eps
        shifts.append(shift)
        shifts_norm.append(shift / (scale + eps))

    if not shifts:
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
    n_components: int = 3,
    object_col: str = "object_id",
    batch_col: str = "batch",
    eps: float = 1e-12,
) -> dict:
    """Compute object-level diagnostics from pixel-level PCA scores."""
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
    batches = np.asarray(metadata.get(batch_col, ["unknown"] * len(object_ids))).astype(str)

    df = pd.DataFrame({"object_id": object_ids, "label": y, "batch": batches})
    for axis in range(n_components):
        df[f"C{axis + 1}"] = T[:, axis]
    score_cols = [f"C{axis + 1}" for axis in range(n_components)]

    df_obj = df.groupby(["object_id", "label", "batch"], as_index=False).agg({col: "mean" for col in score_cols})
    T_obj = df_obj[score_cols].to_numpy(dtype=float)
    y_obj = df_obj["label"].to_numpy()
    batch_obj = df_obj["batch"].to_numpy()

    intra_traces = []
    for _, group in df.groupby("object_id"):
        Tg = group[score_cols].to_numpy(dtype=float)
        if Tg.shape[0] > 1:
            intra_traces.append(np.trace(np.atleast_2d(np.cov(Tg, rowvar=False))))

    mean_intra = float(np.mean(intra_traces)) if intra_traces else np.nan
    between_object_trace = float(np.trace(np.atleast_2d(np.cov(T_obj, rowvar=False)))) if T_obj.shape[0] > 1 else np.nan

    return {
        "object_class_trace_ratio": trace_ratio_by_group(T_obj, y_obj, n_components=n_components, eps=eps),
        "object_batch_trace_ratio": trace_ratio_by_group(T_obj, batch_obj, n_components=n_components, eps=eps),
        "mean_intra_object_trace": mean_intra,
        "object_over_intra_ratio": between_object_trace / (mean_intra + eps) if np.isfinite(mean_intra) else np.nan,
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
    n_components: int = 3,
    matrix_method=None,
) -> dict:
    """Compute PCA comparison metrics for object-level or pixel-level matrices."""
    y_train = np.asarray(y_train).astype(str)
    T_train = np.asarray(T_train, dtype=float)
    metadata_train = metadata_train or {}

    evr = pca_model.explained_variance_ratio_
    cum = pca_model.cumulative_explained_variance_ratio_
    batches_train = metadata_train.get("batch", metadata_train.get("batches", None))

    metrics = {
        "evr_pc1": float(evr[0]) if len(evr) > 0 else np.nan,
        "evr_pc2": float(evr[1]) if len(evr) > 1 else np.nan,
        "evr_pc3": float(evr[2]) if len(evr) > 2 else np.nan,
        "cum_pc2": float(cum[1]) if len(cum) > 1 else np.nan,
        "cum_pc3": float(cum[2]) if len(cum) > 2 else np.nan,
        "ncomp_90": n_components_for_cumulative_variance(cum, threshold=0.90),
        "ncomp_95": n_components_for_cumulative_variance(cum, threshold=0.95),
        "class_trace_ratio": trace_ratio_by_group(T_train, y_train, n_components=n_components),
    }

    metrics["batch_trace_ratio"] = (
        trace_ratio_by_group(T_train, batches_train, n_components=n_components)
        if batches_train is not None
        else np.nan
    )
    metrics["class_over_batch_ratio"] = (
        metrics["class_trace_ratio"] / (metrics["batch_trace_ratio"] + 1e-12)
        if np.isfinite(metrics["batch_trace_ratio"])
        else np.nan
    )
    metrics.update(pca_distance_summary(pca_model, X_train, n_components=n_components, prefix="train"))

    if X_projection is not None:
        metrics.update(pca_distance_summary(pca_model, X_projection, n_components=n_components, prefix="projection"))
        metrics["projection_train_q_ratio"] = metrics["projection_q_mean"] / (metrics["train_q_mean"] + 1e-12)

    if T_projection is not None and y_projection is not None:
        metrics.update(
            train_projection_shift_by_label(
                T_train=T_train,
                T_projection=T_projection,
                y_train=y_train,
                y_projection=y_projection,
                n_components=n_components,
            )
        )

    if matrix_method in {"all_pixels", "balanced_pixels", "pixel"}:
        metrics.update(pixel_object_score_metrics(T=T_train, y=y_train, metadata=metadata_train, n_components=n_components))
    else:
        metrics.update(
            {
                "object_class_trace_ratio": np.nan,
                "object_batch_trace_ratio": np.nan,
                "mean_intra_object_trace": np.nan,
                "object_over_intra_ratio": np.nan,
            }
        )

    return metrics


# -----------------------------------------------------------------------------
# PCA workflow
# -----------------------------------------------------------------------------


def compare_pca_representations(
    object_db,
    matrix_methods=("object_mean", "all_pixels", "balanced_pixels"),
    preprocessing_methods=("raw", "snv", "vector_norm", "msc", "sg_d1"),
    filters: dict | None = None,
    allowed_splits=None,
    allowed_labels=None,
    label_col: str = "object_nut_type",
    n_components: int = 5,
    m: int = 40,
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    balanced_pixel_strategy: str = "random",
    verbose: bool = True,
):
    """Compare PCA representations across matrix methods and preprocessing configs."""
    if filters is None:
        filters = {}
        if allowed_splits is not None:
            filters["split"] = list(allowed_splits) if not isinstance(allowed_splits, str) else [allowed_splits]
        if allowed_labels is not None:
            filters[label_col] = list(allowed_labels) if not isinstance(allowed_labels, str) else [allowed_labels]

    rows = []
    results = {}
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_methods)

    for matrix_method in matrix_methods:
        matrix_method = str(matrix_method)
        matrix_family = pca_matrix_family_from_method(matrix_method)
        balanced_pixel_strategy_effective = str(balanced_pixel_strategy)
        balanced_pixel_strategy_label = (
            balanced_pixel_strategy_effective
            if matrix_method == "balanced_pixels"
            else "not_applicable"
        )
        matrix_variant = pca_matrix_variant_from_method(
            matrix_method=matrix_method,
            balanced_pixel_strategy=balanced_pixel_strategy_effective,
        )
        if verbose:
            print(f"\n=== Matrix method: {matrix_method} ===")

        X_raw, y, metadata = build_matrix(
            object_db=object_db,
            matrix_method=matrix_method,
            filters=filters,
            m=m,
            random_state=random_state,
            replace=replace,
            balanced_pixel_strategy=balanced_pixel_strategy,
        )

        metadata = dict(metadata)
        metadata.setdefault("observation_ids", metadata.get("object_id"))
        metadata.setdefault("source_images", metadata.get("source_image"))
        metadata.setdefault("batches", metadata.get("batch"))
        metadata.setdefault("areas", metadata.get("area"))

        label_values, label_counts = np.unique(y, return_counts=True)
        label_count_dict = {str(label): int(count) for label, count in zip(label_values, label_counts)}

        if verbose:
            print(f"X shape: {X_raw.shape}")
            print(f"Labels: {label_count_dict}")

        results[matrix_method] = {}

        for preprocessing_name, steps in preprocessing_configs.items():
            steps = tuple(steps)
            if verbose:
                print(f"  - preprocessing: {preprocessing_name}")

            preprocessor = SpectralPreprocessor(steps, sg_window_length=sg_window_length, sg_polyorder=sg_polyorder)
            X_pre = preprocessor.fit_transform(X_raw, wavelengths=wavelengths)
            pca = PCAModel(n_components=n_components, center=True).fit(X_pre)
            T = pca.scores_
            evr = pca.explained_variance_ratio_
            cum = pca.cumulative_explained_variance_ratio_

            sep = binary_class_separation_scores(T, y, n_components=n_components)
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
                "matrix_family": matrix_family,
                "matrix_variant": matrix_variant,
                "balanced_pixel_strategy": balanced_pixel_strategy_label,
                "balanced_pixel_strategy_effective": balanced_pixel_strategy_effective,
                "matrix_method": matrix_method,
                "preprocessing": str(preprocessing_name),
                "preprocessing_steps": "+".join(steps),
                "n_observations": int(X_raw.shape[0]),
                "n_bands": int(X_raw.shape[1]),
                "n_components": int(n_components),
                "m": int(m) if matrix_method == "balanced_pixels" else np.nan,
                "m_effective": int(m) if matrix_method == "balanced_pixels" else np.nan,
                "label_counts": label_count_dict,
                "evr_pc1": float(evr[0]) if len(evr) > 0 else np.nan,
                "evr_pc2": float(evr[1]) if len(evr) > 1 else np.nan,
                "evr_pc3": float(evr[2]) if len(evr) > 2 else np.nan,
                "cum_pc2": float(cum[1]) if len(cum) > 1 else np.nan,
                "cum_pc3": float(cum[2]) if len(cum) > 2 else np.nan,
                **sep,
                **extra_metrics,
            }
            for label, count in label_count_dict.items():
                row[f"n_label_{label}"] = count

            rows.append(row)
            results[matrix_method][preprocessing_name] = {
                "X_raw": X_raw,
                "X_preprocessed": X_pre,
                "pca": pca,
                "covariance": pca.covariance_,
                "eigenvalues": pca.eigenvalues_,
                "eigenvectors": pca.eigenvectors_,
                "scores": pca.scores_,
                "loadings": pca.loadings_,
                "explained_variance_ratio": evr,
                "cumulative_explained_variance_ratio": cum,
                "y": y,
                "metadata": metadata,
                "preprocessor": preprocessor,
                "summary": row,
            }

    summary_df = pd.DataFrame(rows)
    sort_cols = ["fisher_pc1", "fisher_pc2", "centroid_distance_pc1_pc2"]
    sort_cols = [col for col in sort_cols if col in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    return summary_df, results


def add_pca_selection_score(
    summary_df: pd.DataFrame,
    matrix_method_col: str = "matrix_method",
    score_col: str = "selection_score",
    profile: str = "auto",
    group_col=None,
    robust: bool = True,
    clip_quantiles=(0.05, 0.95),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Add a generic PCA representation selection score.

    Object matrices prioritize class separation, low batch structure, and stability.
    Pixel matrices prioritize object-level separation and controlled intra-object spread.
    """
    df = summary_df.copy()
    # Ensure stable metadata columns for downstream notebooks.
    if "matrix_family" not in df.columns and matrix_method_col in df.columns:
        df["matrix_family"] = df[matrix_method_col].apply(pca_matrix_family_from_method)
    if "balanced_pixel_strategy" not in df.columns:
        df["balanced_pixel_strategy"] = np.where(
            df[matrix_method_col].astype(str).eq("balanced_pixels"),
            "random",
            "not_applicable",
        )
    if "balanced_pixel_strategy_effective" not in df.columns:
        df["balanced_pixel_strategy_effective"] = np.where(
            df[matrix_method_col].astype(str).eq("balanced_pixels"),
            df["balanced_pixel_strategy"].astype(str),
            "random",
        )
    if "matrix_variant" not in df.columns and matrix_method_col in df.columns:
        df["matrix_variant"] = df.apply(
            lambda row: pca_matrix_variant_from_method(
                matrix_method=row[matrix_method_col],
                balanced_pixel_strategy=row.get("balanced_pixel_strategy_effective", row.get("balanced_pixel_strategy", "random")),
            ),
            axis=1,
        )

    object_positive = {"class_trace_ratio": 3.0, "mahalanobis_pc1_pc2_pc3": 1.0}
    object_negative = {"batch_trace_ratio": 2.0, "mean_train_projection_shift_norm": 1.5, "projection_q_deviation": 1.5, "ncomp_95": 0.3}
    pixel_positive = {"object_class_trace_ratio": 3.0, "object_over_intra_ratio": 1.0}
    pixel_negative = {"object_batch_trace_ratio": 2.0, "mean_intra_object_trace": 1.0, "mean_train_projection_shift_norm": 1.2, "projection_q_deviation": 1.2, "ncomp_95": 0.3}

    def profile_for_matrix(matrix_method):
        matrix_method = str(matrix_method)
        if profile == "object":
            return object_positive, object_negative
        if profile == "pixel":
            return pixel_positive, pixel_negative
        if profile != "auto":
            raise ValueError("profile must be 'auto', 'object', or 'pixel'.")
        if matrix_method in {"object_mean", "object_median"}:
            return object_positive, object_negative
        if matrix_method in {"all_pixels", "balanced_pixels", "pixel"}:
            return pixel_positive, pixel_negative
        return object_positive, object_negative

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

    def compute_scores_for_group(group):
        group = group.copy()
        matrix_values = group[matrix_method_col].unique() if matrix_method_col in group.columns else ["object_mean"]
        all_pos_metrics = set()
        all_neg_metrics = set()
        for matrix_value in matrix_values:
            pos, neg = profile_for_matrix(matrix_value)
            all_pos_metrics.update(pos)
            all_neg_metrics.update(neg)

        scaled = {
            metric: scale_metric(group[metric]) if metric in group.columns else pd.Series(np.zeros(len(group)), index=group.index)
            for metric in sorted(all_pos_metrics | all_neg_metrics)
        }

        for idx in group.index:
            matrix_value = group.loc[idx, matrix_method_col] if matrix_method_col in group.columns else "object_mean"
            pos, neg = profile_for_matrix(matrix_value)
            score = 0.0
            for metric, weight in pos.items():
                contrib = float(weight) * scaled[metric].loc[idx]
                group.loc[idx, f"contrib_plus_{metric}"] = contrib
                score += contrib
            for metric, weight in neg.items():
                contrib = -float(weight) * scaled[metric].loc[idx]
                group.loc[idx, f"contrib_minus_{metric}"] = contrib
                score += contrib
            group.loc[idx, score_col] = score
        return group

    if group_col is None:
        out = compute_scores_for_group(df)
    else:
        group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
        missing_group_cols = [
            col for col in group_cols
            if col not in df.columns
        ]
        if missing_group_cols:
            raise KeyError(
                f"Missing group column(s) in PCA summary: {missing_group_cols}"
            )
        parts = []
        for key, group in df.groupby(
            group_cols,
            group_keys=False,
            dropna=False,
            sort=False,
        ):
            scored_group = compute_scores_for_group(group)
            key_tuple = key if isinstance(key, tuple) else (key,)
            for col, value in zip(group_cols, key_tuple):
                if col not in scored_group.columns:
                    scored_group[col] = value
            parts.append(scored_group)
        out = (
            pd.concat(parts, ignore_index=False, sort=False)
            .sort_index()
            .reset_index(drop=True)
        )

    def quality_flag(row):
        matrix_method = str(row.get(matrix_method_col, "object_mean"))
        if matrix_method in {"object_mean", "object_median"}:
            if pd.notna(row.get("class_trace_ratio", np.nan)) and row.get("class_trace_ratio", np.nan) < 0.10:
                return "weak_class_separation"
            if pd.notna(row.get("projection_q_deviation", np.nan)) and row.get("projection_q_deviation", np.nan) > 1.0:
                return "unstable_projection"
            if pd.notna(row.get("batch_trace_ratio", np.nan)) and row.get("batch_trace_ratio", np.nan) > 0.01:
                return "batch_sensitive"
            return "candidate"
        if matrix_method in {"all_pixels", "balanced_pixels", "pixel"}:
            if pd.notna(row.get("object_class_trace_ratio", np.nan)) and row.get("object_class_trace_ratio", np.nan) < 0.05:
                return "weak_object_separation"
            if pd.notna(row.get("projection_q_deviation", np.nan)) and row.get("projection_q_deviation", np.nan) > 1.0:
                return "unstable_projection"
            if pd.notna(row.get("object_batch_trace_ratio", np.nan)) and row.get("object_batch_trace_ratio", np.nan) > 0.005:
                return "batch_sensitive"
            return "candidate"
        return "unknown"

    out["selection_flag"] = out.apply(quality_flag, axis=1)
    return out
