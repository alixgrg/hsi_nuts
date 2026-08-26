from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.matrices.matrix_registry import build_matrix, matrix_family_from_method
from src.models.pca import PCAModel
from src.protocol_governance import make_selection_id
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.preprocessing_configs import normalize_preprocessing_configs


OBJECT_MATRIX_METHODS = {"object_mean", "object_median"}
PIXEL_MATRIX_METHODS = {"balanced_pixels", "all_pixels", "pixel"}



def pca_matrix_variant_from_method(
    matrix_method: str,
    balanced_pixel_strategy: str = "random",
    m: int | None = None,
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
        suffix = "" if m is None else f"_m{int(m)}"
        return f"balanced_pixels_{balanced_pixel_strategy}{suffix}"

    return matrix_method


def _pca_preprocessing_identity_payload(row: Mapping) -> dict:
    steps = str(row["preprocessing_steps"])
    has_sg = "sg_" in steps

    sg_window = row.get("sg_window_length")
    sg_polyorder = row.get("sg_polyorder")

    return {
        "matrix_family": str(row["matrix_family"]),
        "preprocessing": str(row["preprocessing"]),
        "preprocessing_steps": steps,
        "sg_window_length": (
            int(sg_window)
            if has_sg and pd.notna(sg_window)
            else None
        ),
        "sg_polyorder": (
            int(sg_polyorder)
            if has_sg and pd.notna(sg_polyorder)
            else None
        ),
        "wavelength_axis_id": str(row["wavelength_axis_id"]),
    }


def build_pca_candidate_plan(
    matrix_summary: pd.DataFrame,
    m_feasibility: pd.DataFrame,
    preprocessing_validation: pd.DataFrame,
    *,
    allowed_m: Sequence[int],
    sg_window_length: int,
    matrix_methods: Sequence[str] = (
        "object_mean",
        "object_median",
        "all_pixels",
        "balanced_pixels",
    ),
    balanced_strategies: Sequence[str] = ("random", "center"),
) -> pd.DataFrame:
    """Build task-15 candidates exclusively from accepted task-14 outputs."""
    matrix_required = {
        "matrix_id",
        "protocol_role",
        "matrix_method",
        "wavelength_axis_id",
        "status",
    }
    preprocessing_required = {
        "matrix_id",
        "fit_role",
        "preprocessing",
        "steps",
        "sg_window_length",
        "status",
    }
    feasibility_required = {"m", "strategy", "status"}
    for name, frame, required in (
        ("matrix_summary", matrix_summary, matrix_required),
        ("m_feasibility", m_feasibility, feasibility_required),
        (
            "preprocessing_validation",
            preprocessing_validation,
            preprocessing_required,
        ),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing columns: {missing}")

    allowed_methods = set(map(str, matrix_methods))
    allowed_m = set(map(int, allowed_m))
    allowed_strategies = set(map(str, balanced_strategies))
    matrices = matrix_summary.loc[
        matrix_summary["status"].eq("accepted")
        & matrix_summary["protocol_role"].eq("calibration")
        & matrix_summary["matrix_method"].astype(str).isin(allowed_methods)
    ].copy()
    matrices["training_matrix_id"] = matrices["matrix_id"].astype(str)
    matrices["candidate_matrix_id"] = matrices["training_matrix_id"].str.replace(
        r"^calibration_", "", regex=True
    )
    matrices["m"] = pd.to_numeric(
        matrices["candidate_matrix_id"].str.extract(r"_m(\d+)$")[0],
        errors="coerce",
    )
    parsed_strategy = matrices["candidate_matrix_id"].str.extract(
        r"^balanced_pixels_([^_]+)_m\d+$"
    )[0]
    declared_strategy = matrices.get(
        "balanced_pixel_strategy",
        pd.Series(index=matrices.index, dtype=object),
    )
    matrices["balanced_pixel_strategy"] = declared_strategy.where(
        declared_strategy.notna(), parsed_strategy
    )
    matrices.loc[
        ~matrices["matrix_method"].eq("balanced_pixels"),
        "balanced_pixel_strategy",
    ] = "not_applicable"

    feasible = m_feasibility.loc[
        m_feasibility["status"].eq("accepted")
        & pd.to_numeric(m_feasibility["m"], errors="coerce").isin(allowed_m)
        & m_feasibility["strategy"].astype(str).isin(allowed_strategies),
        ["m", "strategy"],
    ].copy()
    feasible["m"] = pd.to_numeric(feasible["m"], errors="raise").astype(int)
    feasible_pairs = set(map(tuple, feasible.drop_duplicates().to_numpy()))
    balanced = matrices["matrix_method"].eq("balanced_pixels")
    balanced_pairs = pd.Series(
        [
            (int(m), str(strategy)) if pd.notna(m) else None
            for m, strategy in zip(
                matrices["m"], matrices["balanced_pixel_strategy"]
            )
        ],
        index=matrices.index,
    )
    matrices = matrices.loc[
        ~balanced | balanced_pairs.isin(feasible_pairs)
    ].copy()
    matrices.loc[balanced.reindex(matrices.index, fill_value=False), "m"] = (
        matrices.loc[
            balanced.reindex(matrices.index, fill_value=False), "m"
        ].astype(int)
    )

    accepted_preprocessing = preprocessing_validation.loc[
        preprocessing_validation["status"].eq("accepted")
        & preprocessing_validation["fit_role"].eq("calibration")
    ].copy()
    has_sg = accepted_preprocessing["preprocessing"].astype(str).str.contains(
        "sg_", regex=False
    )
    selected_window = pd.to_numeric(
        accepted_preprocessing["sg_window_length"], errors="coerce"
    ).eq(int(sg_window_length))
    accepted_preprocessing = accepted_preprocessing.loc[
        ~has_sg | selected_window
    ].copy()
    accepted_preprocessing["steps"] = accepted_preprocessing["steps"].map(
        lambda value: "+".join(
            token.strip() for token in str(value).split("+") if token.strip()
        )
    )
    accepted_preprocessing = accepted_preprocessing.rename(
        columns={
            "matrix_id": "candidate_matrix_id",
            "steps": "preprocessing_steps",
        }
    )
    keep_preprocessing = [
        "candidate_matrix_id",
        "preprocessing",
        "preprocessing_steps",
        "sg_window_length",
        "sg_polyorder",
    ]
    plan = matrices.merge(
        accepted_preprocessing[keep_preprocessing].drop_duplicates(),
        on="candidate_matrix_id",
        how="inner",
        validate="one_to_many",
    )
    plan["matrix_family"] = plan["matrix_method"].map(
        matrix_family_from_method
    )
    plan["matrix_variant"] = [
        pca_matrix_variant_from_method(
            method,
            balanced_pixel_strategy=strategy,
            m=(None if pd.isna(m) else int(m)),
        )
        for method, strategy, m in zip(
            plan["matrix_method"],
            plan["balanced_pixel_strategy"],
            plan["m"],
        )
    ]
    #plan["strategy"] = plan["balanced_pixel_strategy"]
    plan["selection_unit_id"] = [
        make_selection_id(
            "pca_preprocessing",
            _pca_preprocessing_identity_payload(row),
        )
        for row in plan.to_dict("records")
    ]
    selection_unit_identity_cols = [
        "matrix_family",
        "preprocessing",
        "preprocessing_steps",
        "sg_window_length",
        "sg_polyorder",
        "wavelength_axis_id",
    ]
    unit_check = (
        plan.groupby("selection_unit_id")[selection_unit_identity_cols]
        .nunique(dropna=False)
    )
    if unit_check.gt(1).any(axis=None):
        raise RuntimeError(
            "One PCA selection_unit_id maps to multiple scientific identities."
        )

    def _candidate_payload(row):
        m_value = row.get("m")

        return {
            "selection_unit_id": str(row["selection_unit_id"]),
            "training_matrix_id": str(row["training_matrix_id"]),
            "matrix_method": str(row["matrix_method"]),
            "m": (
                int(m_value)
                if pd.notna(m_value)
                else None
            ),
            "balanced_pixel_strategy": str(
                row["balanced_pixel_strategy"]
            ),
        }

    plan["candidate_id"] = [
        make_selection_id(
            "pca_candidate",
            _candidate_payload(row),
        )
        for row in plan.to_dict("records")
    ]
    # identity_columns = (
    #     "training_matrix_id",
    #     "matrix_method",
    #     "m",
    #     "balanced_pixel_strategy",
    #     "preprocessing",
    #     "preprocessing_steps",
    #     "sg_window_length",
    #     "sg_polyorder",
    #     "wavelength_axis_id",
    # )
    # plan["candidate_id"] = [
    #     "pca_" + make_selection_id(
    #         {
    #             key: (
    #                 None
    #                 if pd.isna(value)
    #                 else value.item()
    #                 if isinstance(value, np.generic)
    #                 else value
    #             )
    #             for key, value in zip(identity_columns, values)
    #         }
    #     )[:20]
    #     for values in plan.loc[:, identity_columns].itertuples(
    #         index=False, name=None
    #     )
    # ]
    if plan["candidate_id"].duplicated().any():
        raise RuntimeError("PCA candidate identities are not unique.")

    expected_matrix_ids = {
        "object_mean",
        "object_median",
        "all_pixels",
        *{
            f"balanced_pixels_{strategy}_m{m}"
            for m in allowed_m
            for strategy in allowed_strategies
            if (m, strategy) in feasible_pairs
        },
    }
    observed_matrix_ids = set(plan["candidate_matrix_id"].astype(str))
    missing_matrices = sorted(expected_matrix_ids - observed_matrix_ids)
    if missing_matrices:
        raise RuntimeError(
            "Accepted notebook-02 PCA matrices are incomplete: "
            f"{missing_matrices}"
        )
    columns = (
        "candidate_id",
        "selection_unit_id",
        "training_matrix_id",
        "candidate_matrix_id",
        "matrix_family",
        "matrix_variant",
        "matrix_method",
        "m",
        #"strategy",
        "balanced_pixel_strategy",
        "preprocessing",
        "preprocessing_steps",
        "sg_window_length",
        "sg_polyorder",
        "wavelength_axis_id",
    )
    return plan.loc[:, columns].sort_values(
        ["matrix_family", "matrix_variant", "preprocessing", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)


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
            "centroid_distance_pc1_pc2_pc3": np.nan,
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
    if T.shape[1] >= 3:
        centroid_dist_3 = float(
            np.linalg.norm(
                np.mean(T0[:, :3], axis=0) - np.mean(T1[:, :3], axis=0)
            )
        )
    else:
        centroid_dist_3 = np.nan

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
        "centroid_distance_pc1_pc2_pc3": centroid_dist_3,
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


def pca_component_variance_table(
    pca_model,
    *,
    matrix_variant: str,
    preprocessing: str,
) -> pd.DataFrame:
    """Return the complete, five-column explained-variance curve."""
    evr = np.asarray(pca_model.explained_variance_ratio_, dtype=float)
    cumulative = np.asarray(
        pca_model.cumulative_explained_variance_ratio_,
        dtype=float,
    )
    return pd.DataFrame(
        {
            "matrix_variant": str(matrix_variant),
            "preprocessing": str(preprocessing),
            "component": np.arange(1, len(evr) + 1, dtype=int),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )


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


def compute_group_compactness(T, groups, n_components: int = 3) -> dict:
    """Summarize within-group score dispersion without returning wide tables."""
    scores = np.asarray(T, dtype=float)[:, : int(n_components)]
    group_values = np.asarray(groups).astype(str)
    valid = np.isfinite(scores).all(axis=1) & ~np.isin(
        group_values,
        ["None", "nan", "unknown"],
    )
    scores = scores[valid]
    group_values = group_values[valid]
    rows = []
    for group in np.unique(group_values):
        group_scores = scores[group_values == group]
        if len(group_scores) == 0:
            continue
        centroid = group_scores.mean(axis=0)
        distances = np.linalg.norm(group_scores - centroid, axis=1)
        within_trace = (
            float(np.trace(np.atleast_2d(np.cov(group_scores, rowvar=False))))
            if len(group_scores) > 1
            else 0.0
        )
        rows.append(
            {
                "within_trace": within_trace,
                "mean_distance": float(np.mean(distances)),
                "q95_distance": float(np.quantile(distances, 0.95)),
            }
        )
    if not rows:
        return {
            "within_class_trace": np.nan,
            "mean_distance_to_class_centroid": np.nan,
            "q95_distance_to_class_centroid": np.nan,
        }
    summary = pd.DataFrame(rows)
    return {
        "within_class_trace": float(summary["within_trace"].mean()),
        "mean_distance_to_class_centroid": float(summary["mean_distance"].mean()),
        "q95_distance_to_class_centroid": float(summary["q95_distance"].mean()),
    }


def compute_group_centroid_displacements(
    T,
    groups,
    n_components: int = 3,
) -> pd.DataFrame:
    """Return one centroid displacement from the global centre per group."""
    scores = np.asarray(T, dtype=float)[:, : int(n_components)]
    group_values = np.asarray(groups).astype(str)
    valid = np.isfinite(scores).all(axis=1) & ~np.isin(
        group_values,
        ["None", "nan", "unknown"],
    )
    scores = scores[valid]
    group_values = group_values[valid]
    columns = ("group", "batch_centroid_shift_norm")
    if len(scores) == 0:
        return pd.DataFrame(columns=columns)
    global_centroid = scores.mean(axis=0)
    rows = []
    for group in np.unique(group_values):
        centroid = scores[group_values == group].mean(axis=0)
        rows.append(
            {
                "group": group,
                "batch_centroid_shift_norm": float(
                    np.linalg.norm(centroid - global_centroid)
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def pca_distance_summary(pca_model, X, n_components=None, prefix: str = "train") -> dict:
    """Compute summary statistics for PCA Q-residuals and Hotelling T2."""
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
    """Compute the compact scientific diagnostics required by notebook 03."""
    y_train = np.asarray(y_train).astype(str)
    T_train = np.asarray(T_train, dtype=float)
    metadata_train = {} if metadata_train is None else dict(metadata_train)

    evr = pca_model.explained_variance_ratio_
    cum = pca_model.cumulative_explained_variance_ratio_
    batches_train = metadata_train.get("batch", metadata_train.get("batches", None))

    metrics = {
        "ncomp_90": n_components_for_cumulative_variance(cum, threshold=0.90),
        "ncomp_95": n_components_for_cumulative_variance(cum, threshold=0.95),
        "ncomp_99": n_components_for_cumulative_variance(cum, threshold=0.99),
        "class_trace_ratio": trace_ratio_by_group(T_train, y_train, n_components=n_components),
    }
    metrics.update(
        binary_class_separation_scores(
            T_train,
            y_train,
            n_components=n_components,
        )
    )
    metrics.update(
        compute_group_compactness(
            T_train,
            y_train,
            n_components=n_components,
        )
    )

    metrics["batch_trace_ratio"] = (
        trace_ratio_by_group(T_train, batches_train, n_components=n_components)
        if batches_train is not None
        else np.nan
    )
    batch_displacements = (
        compute_group_centroid_displacements(
            T_train,
            batches_train,
            n_components=n_components,
        )
        if batches_train is not None
        else pd.DataFrame()
    )
    if len(batch_displacements):
        shifts = batch_displacements["batch_centroid_shift_norm"]
        metrics["mean_batch_centroid_shift_norm"] = float(shifts.mean())
        metrics["max_batch_centroid_shift_norm"] = float(shifts.max())
    else:
        metrics["mean_batch_centroid_shift_norm"] = np.nan
        metrics["max_batch_centroid_shift_norm"] = np.nan
    metrics.update(pca_distance_summary(pca_model, X_train, n_components=n_components, prefix="train"))

    if X_projection is not None:
        metrics.update(pca_distance_summary(pca_model, X_projection, n_components=n_components, prefix="projection"))
        metrics["projection_q_deviation"] = abs(
            metrics["projection_q_mean"] / (metrics["train_q_mean"] + 1e-12)
            - 1.0
        )
    else:
        for column in (
            "projection_q_mean",
            "projection_q_median",
            "projection_q_q95",
            "projection_t2_mean",
            "projection_t2_median",
            "projection_t2_q95",
        ):
            metrics[column] = np.nan
        metrics["projection_q_deviation"] = np.nan

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
    else:
        metrics.update(
            {
                "mean_train_projection_shift": np.nan,
                "max_train_projection_shift": np.nan,
                "mean_train_projection_shift_norm": np.nan,
                "max_train_projection_shift_norm": np.nan,
            }
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


def _normalise_batch_token(value) -> str:
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def subset_object_db_for_pca(
    object_db,
    *,
    sample_kind: str = "pure",
    reference_classes=("almond", "peanut"),
    allowed_batches=None,
    forbidden_batches=(),
    allowed_object_ids=None,
):
    """Return a protocol-safe PCA subset and reject forbidden test batches."""
    allowed = (
        None
        if allowed_batches is None
        else {_normalise_batch_token(value) for value in allowed_batches}
    )
    forbidden = {_normalise_batch_token(value) for value in forbidden_batches}
    allowed_ids = (
        None
        if allowed_object_ids is None
        else {str(value) for value in allowed_object_ids}
    )
    classes = {str(value) for value in reference_classes}
    selected = {}
    for object_id, obj in object_db.items():
        if allowed_ids is not None and str(object_id) not in allowed_ids:
            continue
        if str(obj.get("sample_kind")) != str(sample_kind):
            continue
        label = str(obj.get("object_nut_type", obj.get("label")))
        if label not in classes:
            continue
        batch = _normalise_batch_token(obj.get("batch"))
        if batch in forbidden:
            continue
        if allowed is not None and batch not in allowed:
            continue
        selected[object_id] = obj
    selected_batches = {
        _normalise_batch_token(obj.get("batch"))
        for obj in selected.values()
    }
    leaked = sorted(selected_batches & forbidden)
    if leaked:
        raise RuntimeError(f"Forbidden PCA batches leaked into the subset: {leaked}")
    return selected


def _metadata_aliases(metadata) -> dict:
    out = {key: np.asarray(value) for key, value in dict(metadata).items()}
    out.setdefault("observation_ids", out.get("object_id"))
    out.setdefault("source_images", out.get("source_image"))
    out.setdefault("batches", out.get("batch"))
    return out


def _invalid_pca_row(
    *,
    matrix_family,
    matrix_variant,
    matrix_method,
    balanced_pixel_strategy,
    preprocessing,
    steps,
    error,
):
    return {
        "matrix_family": matrix_family,
        "matrix_variant": matrix_variant,
        "matrix_method": matrix_method,
        "balanced_pixel_strategy": balanced_pixel_strategy,
        "preprocessing": str(preprocessing),
        "preprocessing_steps": "+".join(steps),
        "n_observations": 0,
        "n_bands": 0,
        "matrix_nonempty": False,
        "finite_values": False,
        "sg_valid": False,
        "variance_valid": False,
        "pca_fit_valid": False,
        "projection_valid": False,
        "residuals_valid": False,
        "technical_error": f"{type(error).__name__}: {error}",
    }


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
    under_m_policy: str | None = None,
    projection_object_db=None,
    projection_filters: dict | None = None,
    projection_role: str | None = None,
    max_zero_variance_band_rate: float = 0.25,
    zero_variance_epsilon: float = 1e-12,
    return_component_table: bool = False,
    verbose: bool = True,
):
    """Fit PCA on calibration and only transform the optional confirmation set."""
    if filters is None:
        filters = {}
        if allowed_splits is not None:
            filters["split"] = list(allowed_splits) if not isinstance(allowed_splits, str) else [allowed_splits]
        if allowed_labels is not None:
            filters[label_col] = list(allowed_labels) if not isinstance(allowed_labels, str) else [allowed_labels]

    rows = []
    component_tables = []
    results = {}
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_methods)

    for matrix_method in matrix_methods:
        matrix_method = str(matrix_method)
        matrix_family = matrix_family_from_method(matrix_method)
        balanced_pixel_strategy_effective = str(balanced_pixel_strategy)
        balanced_pixel_strategy_label = (
            balanced_pixel_strategy_effective
            if matrix_method == "balanced_pixels"
            else "not_applicable"
        )
        matrix_variant = pca_matrix_variant_from_method(
            matrix_method=matrix_method,
            balanced_pixel_strategy=balanced_pixel_strategy_effective,
            m=(m if matrix_method == "balanced_pixels" else None),
        )
        if verbose:
            print(f"\n=== Matrix method: {matrix_method} ===")

        try:
            X_raw, y, metadata = build_matrix(
                object_db=object_db,
                matrix_method=matrix_method,
                filters=filters,
                m=m,
                random_state=random_state,
                replace=replace,
                balanced_pixel_strategy=balanced_pixel_strategy,
                under_m_policy=under_m_policy,
                require_finite=False,
            )
            X_raw = np.asarray(X_raw, dtype=float)
            y = np.asarray(y)
            metadata = _metadata_aliases(metadata)
        except Exception as exc:
            for preprocessing_name, steps in preprocessing_configs.items():
                rows.append(
                    _invalid_pca_row(
                        matrix_family=matrix_family,
                        matrix_variant=matrix_variant,
                        matrix_method=matrix_method,
                        balanced_pixel_strategy=balanced_pixel_strategy_label,
                        preprocessing=preprocessing_name,
                        steps=tuple(steps),
                        error=exc,
                    )
                )
            results[matrix_method] = {}
            continue

        X_projection_raw = None
        y_projection = None
        metadata_projection = None
        projection_build_error = None
        if projection_object_db is not None:
            try:
                (
                    X_projection_raw,
                    y_projection,
                    metadata_projection,
                ) = build_matrix(
                    object_db=projection_object_db,
                    matrix_method=matrix_method,
                    filters=projection_filters or {},
                    m=m,
                    random_state=random_state,
                    replace=replace,
                    balanced_pixel_strategy=balanced_pixel_strategy,
                    under_m_policy=under_m_policy,
                    require_finite=False,
                )
                X_projection_raw = np.asarray(X_projection_raw, dtype=float)
                y_projection = np.asarray(y_projection)
                metadata_projection = _metadata_aliases(metadata_projection)
            except Exception as exc:
                projection_build_error = exc

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

            row = {
                "matrix_family": matrix_family,
                "matrix_variant": matrix_variant,
                "balanced_pixel_strategy": balanced_pixel_strategy_label,
                "matrix_method": matrix_method,
                "preprocessing": str(preprocessing_name),
                "preprocessing_steps": "+".join(steps),
                "n_observations": int(X_raw.shape[0]),
                "n_bands": int(X_raw.shape[1]),
                "matrix_nonempty": bool(X_raw.size),
                "finite_values": bool(np.isfinite(X_raw).all()),
                "sg_valid": False,
                "variance_valid": False,
                "pca_fit_valid": False,
                "projection_valid": projection_object_db is None,
                "residuals_valid": False,
                "projection_role": projection_role,
                "technical_error": "",
            }
            try:
                if not row["matrix_nonempty"]:
                    raise ValueError("Calibration matrix is empty.")
                if not row["finite_values"]:
                    raise ValueError("Calibration matrix contains non-finite values.")
                preprocessor = SpectralPreprocessor(
                    steps,
                    sg_window_length=sg_window_length,
                    sg_polyorder=sg_polyorder,
                )
                X_pre = preprocessor.fit_transform(X_raw, wavelengths=wavelengths)
                row["sg_valid"] = True
                if not np.isfinite(X_pre).all():
                    raise ValueError("Preprocessing produced non-finite values.")
                zero_variance_rate = float(
                    np.mean(
                        np.nanstd(X_pre, axis=0)
                        <= float(zero_variance_epsilon)
                    )
                )
                row["zero_variance_band_rate"] = zero_variance_rate
                row["variance_valid"] = (
                    zero_variance_rate <= float(max_zero_variance_band_rate)
                )
                if not row["variance_valid"]:
                    raise ValueError(
                        "Excessive near-zero variance band rate: "
                        f"{zero_variance_rate:.3f}."
                    )

                effective_components = min(
                    int(n_components),
                    int(X_pre.shape[1]),
                    max(int(X_pre.shape[0]) - 1, 1),
                )
                pca = PCAModel(
                    n_components=effective_components,
                    center=True,
                ).fit(X_pre)
                T = pca.transform(X_pre)
                row["pca_fit_valid"] = bool(
                    np.isfinite(T).all()
                    and np.isfinite(pca.loadings_).all()
                )
                if not row["pca_fit_valid"]:
                    raise ValueError("PCA scores or loadings are non-finite.")

                X_projection_pre = None
                T_projection = None
                if projection_object_db is not None:
                    if projection_build_error is not None:
                        raise ValueError(
                            "Projection matrix could not be built: "
                            f"{projection_build_error}"
                        )
                    if X_projection_raw is None or not X_projection_raw.size:
                        raise ValueError("Projection matrix is empty.")
                    if X_projection_raw.shape[1] != X_raw.shape[1]:
                        raise ValueError(
                            "Calibration and projection band counts differ."
                        )
                    if not np.isfinite(X_projection_raw).all():
                        raise ValueError("Projection matrix contains non-finite values.")
                    X_projection_pre = preprocessor.transform(X_projection_raw)
                    T_projection = pca.transform(X_projection_pre)
                    row["projection_valid"] = bool(
                        np.isfinite(X_projection_pre).all()
                        and np.isfinite(T_projection).all()
                    )
                    if not row["projection_valid"]:
                        raise ValueError("Projection preprocessing or scores failed.")

                train_q, _ = pca.q_residuals(
                    X_pre,
                    n_components=min(3, effective_components),
                )
                train_t2 = pca.hotelling_t2(
                    X_pre,
                    n_components=min(3, effective_components),
                )
                residual_arrays = [train_q, train_t2]
                if X_projection_pre is not None:
                    projection_q, _ = pca.q_residuals(
                        X_projection_pre,
                        n_components=min(3, effective_components),
                    )
                    projection_t2 = pca.hotelling_t2(
                        X_projection_pre,
                        n_components=min(3, effective_components),
                    )
                    residual_arrays.extend([projection_q, projection_t2])
                row["residuals_valid"] = all(
                    np.isfinite(values).all()
                    for values in residual_arrays
                )
                if not row["residuals_valid"]:
                    raise ValueError("PCA distances or Q residuals are non-finite.")

                row.update(
                    compute_pca_summary_metrics(
                        pca_model=pca,
                        X_train=X_pre,
                        T_train=T,
                        y_train=y,
                        metadata_train=metadata,
                        X_projection=X_projection_pre,
                        T_projection=T_projection,
                        y_projection=y_projection,
                        metadata_projection=metadata_projection,
                        n_components=min(3, effective_components),
                        matrix_method=matrix_method,
                    )
                )
                component_tables.append(
                    pca_component_variance_table(
                        pca,
                        matrix_variant=matrix_variant,
                        preprocessing=preprocessing_name,
                    )
                )
                results[matrix_method][preprocessing_name] = {
                    "X_raw": X_raw,
                    "X_preprocessed": X_pre,
                    "X_projection_raw": X_projection_raw,
                    "X_projection_preprocessed": X_projection_pre,
                    "pca": pca,
                    "scores": T,
                    "projection_scores": T_projection,
                    "loadings": pca.loadings_,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "cumulative_explained_variance_ratio": (
                        pca.cumulative_explained_variance_ratio_
                    ),
                    "y": y,
                    "y_projection": y_projection,
                    "metadata": metadata,
                    "metadata_projection": metadata_projection,
                    "preprocessor": preprocessor,
                    "summary": row,
                }
            except Exception as exc:
                row["technical_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    component_df = (
        pd.concat(component_tables, ignore_index=True)
        if component_tables
        else pd.DataFrame(
            columns=(
                "matrix_variant",
                "preprocessing",
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance_ratio",
            )
        )
    )
    if return_component_table:
        return summary_df.reset_index(drop=True), results, component_df
    return summary_df.reset_index(drop=True), results


def fit_pca_candidate(
    object_db,
    candidate: Mapping,
    *,
    n_components: int,
    wavelengths,
    random_state: int,
    under_m_policy: str | None,
    sg_window_length: int = 11,
    sg_polyorder: int = 2,
    max_zero_variance_band_rate: float = 0.25,
    zero_variance_epsilon: float = 1e-12,
) -> tuple[dict, dict | None, pd.DataFrame]:
    """Fit one immutable task-15 candidate without an external projection."""
    spec = dict(candidate)
    method = str(spec["matrix_method"])
    preprocessing = str(spec["preprocessing"])
    steps = tuple(
        token for token in str(spec["preprocessing_steps"]).split("+") if token
    )
    strategy = str(spec.get("balanced_pixel_strategy", "not_applicable"))
    effective_strategy = "random" if strategy == "not_applicable" else strategy
    m_value = spec.get("m")
    m = 1 if pd.isna(m_value) else int(m_value)
    diagnostics, registry, components = compare_pca_representations(
        object_db,
        matrix_methods=(method,),
        preprocessing_methods={preprocessing: steps},
        n_components=int(n_components),
        m=m,
        wavelengths=wavelengths,
        random_state=int(random_state),
        balanced_pixel_strategy=effective_strategy,
        under_m_policy=under_m_policy,
        sg_window_length=(
            int(spec["sg_window_length"])
            if pd.notna(spec.get("sg_window_length"))
            else int(sg_window_length)
        ),
        sg_polyorder=(
            int(spec["sg_polyorder"])
            if pd.notna(spec.get("sg_polyorder"))
            else int(sg_polyorder)
        ),
        max_zero_variance_band_rate=max_zero_variance_band_rate,
        zero_variance_epsilon=zero_variance_epsilon,
        return_component_table=True,
        verbose=False,
    )
    identity = {
        key: spec.get(key)
        for key in (
            "candidate_id",
            "selection_unit_id",
            "training_matrix_id",
            "wavelength_axis_id",
            "matrix_family",
            "matrix_variant",
            "matrix_method",
            "m",
            "balanced_pixel_strategy",
            "preprocessing",
            "preprocessing_steps",
            "sg_window_length",
            "sg_polyorder",
        )
    }
    for key, value in identity.items():
        diagnostics[key] = value
        components[key] = value
    result = registry.get(method, {}).get(preprocessing)
    if result is not None:
        result.update(identity)
        result["training_matrix_id"] = spec.get("training_matrix_id")
        result["candidate_id"] = spec.get("candidate_id")
    # Return an extensible mapping: ``Series.update`` silently ignores keys that
    # are not already present, which would discard the task-16 stability fields.
    return diagnostics.iloc[0].to_dict(), result, components


def compare_aligned_loadings(
    loading_rows,
    *,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Compare PCA axes and their sign/permutation-invariant subspaces."""
    if not loading_rows:
        return pd.DataFrame(
            columns=(
                "run_type",
                "seed",
                "fold",
                "strategy",
                "component",
                "loading_abs_correlation",
                "loading_angle_deg",
                "subspace_similarity",
                "subspace_instability",
                "max_principal_angle_deg",
            )
        )
    reference = np.asarray(loading_rows[0]["loadings"], dtype=float)
    max_components = reference.shape[1]
    if n_components is not None:
        max_components = min(max_components, int(n_components))
    output_rows = []
    for run in loading_rows:
        current = np.asarray(run["loadings"], dtype=float).copy()
        n_compare = min(max_components, current.shape[1])
        ref_basis, _ = np.linalg.qr(reference[:, :n_compare])
        cur_basis, _ = np.linalg.qr(current[:, :n_compare])
        singular_values = np.linalg.svd(
            ref_basis.T @ cur_basis,
            compute_uv=False,
        )
        singular_values = np.clip(singular_values, 0.0, 1.0)
        singular_values[np.isclose(singular_values, 1.0, atol=1e-12)] = 1.0
        principal_angles = np.degrees(np.arccos(singular_values))
        subspace_similarity = float(np.mean(singular_values**2))
        shared = {
            key: run.get(key)
            for key in (
                "candidate_id",
                "training_matrix_id",
                "matrix_method",
                "m",
                "balanced_pixel_strategy",
                "preprocessing",
            )
            if key in run
        }
        for component in range(n_compare):
            ref = reference[:, component]
            cur = current[:, component]
            corr = float(np.corrcoef(ref, cur)[0, 1])
            if not np.isfinite(corr):
                denominator = float(np.linalg.norm(ref) * np.linalg.norm(cur))
                corr = (
                    float(np.dot(ref, cur) / denominator)
                    if denominator > 0
                    else np.nan
                )
            if corr < 0:
                current[:, component] *= -1
                cur = current[:, component]
                corr = -corr
            denominator = float(np.linalg.norm(ref) * np.linalg.norm(cur))
            cosine = (
                float(np.dot(ref, cur) / denominator)
                if denominator > 0
                else np.nan
            )
            angle = float(
                np.degrees(np.arccos(np.clip(abs(cosine), 0.0, 1.0)))
            )
            output_rows.append(
                {
                    **shared,
                    "run_type": str(run.get("run_type", "unknown")),
                    "seed": int(run.get("seed", -1)),
                    "fold": int(run.get("fold", -1)),
                    "strategy": str(run.get("strategy", "not_applicable")),
                    "component": component + 1,
                    "loading_abs_correlation": float(abs(corr)),
                    "loading_angle_deg": angle,
                    "subspace_similarity": subspace_similarity,
                    "subspace_instability": float(1.0 - subspace_similarity),
                    "max_principal_angle_deg": float(
                        np.max(principal_angles)
                    ),
                }
            )
    return pd.DataFrame(output_rows)


def _summarize_stability_metrics(metric_rows) -> pd.DataFrame:
    if not metric_rows:
        return pd.DataFrame(
            columns=(
                "run_type",
                "metric",
                "mean",
                "std",
                "min",
                "max",
                "q05",
                "q95",
                "n_runs",
            )
        )
    frame = pd.DataFrame(metric_rows)
    id_columns = {
        "run_type",
        "seed",
        "fold",
        "strategy",
        "candidate_id",
        "training_matrix_id",
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "preprocessing",
    }
    rows = []
    for run_type, run_frame in frame.groupby("run_type", dropna=False):
        for column in run_frame.columns:
            if column in id_columns:
                continue
            values = pd.to_numeric(run_frame[column], errors="coerce")
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            rows.append(
                {
                    "run_type": str(run_type),
                    "metric": column,
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q05": float(values.quantile(0.05)),
                    "q95": float(values.quantile(0.95)),
                    "n_runs": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_group_indices(group_ids, rng) -> np.ndarray:
    group_ids = np.asarray(group_ids).astype(str)
    unique_groups = np.unique(group_ids)
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    parts = [np.flatnonzero(group_ids == group_id) for group_id in sampled]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def evaluate_pca_stability(
    object_db,
    *,
    matrix_method=None,
    preprocessing_steps=None,
    candidate: Mapping | None = None,
    fold_assignments: pd.DataFrame | None = None,
    train_filters=None,
    group_col="source_image",
    n_components=10,
    m=40,
    strategies=("random",),
    balanced_pixel_strategy=None,
    seeds=(0, 1, 2, 3, 4),
    reference_seed=0,
    n_splits=2,
    n_bootstrap=100,
    bootstrap_group_col="source_image",
    sg_window_length=11,
    sg_polyorder=2,
    wavelengths=None,
    under_m_policy=None,
):
    """Evaluate one candidate on common folds, seeds and image bootstraps."""
    spec = {} if candidate is None else dict(candidate)
    matrix_method = str(spec.get("matrix_method", matrix_method))
    if matrix_method in {"None", ""}:
        raise ValueError("matrix_method is required.")
    if preprocessing_steps is None:
        preprocessing_steps = tuple(
            token
            for token in str(spec.get("preprocessing_steps", "raw")).split("+")
            if token
        )
    else:
        preprocessing_steps = tuple(preprocessing_steps)
    m_value = spec.get("m", m)
    m = 1 if pd.isna(m_value) else int(m_value)
    strategy = spec.get(
        "balanced_pixel_strategy",
        balanced_pixel_strategy,
    )
    if strategy is None:
        strategy = tuple(strategies)[0] if matrix_method == "balanced_pixels" else "not_applicable"
    strategy = str(strategy)
    effective_strategy = "random" if strategy == "not_applicable" else strategy
    identity = {
        key: spec.get(key)
        for key in (
            "candidate_id",
            "training_matrix_id",
            "matrix_method",
            "m",
            "balanced_pixel_strategy",
            "preprocessing",
        )
    }
    identity.update(
        {
            "matrix_method": matrix_method,
            "m": np.nan if matrix_method != "balanced_pixels" else m,
            "balanced_pixel_strategy": strategy,
        }
    )
    metric_rows = []
    loading_rows = []
    canonical = None

    active_seeds = (
        tuple(map(int, seeds))
        if matrix_method == "balanced_pixels" and effective_strategy == "random"
        else (int(reference_seed),)
    )
    if int(reference_seed) not in active_seeds:
        active_seeds = (int(reference_seed), *active_seeds)
    common_group_to_fold = None

    for seed in active_seeds:
        X, y, metadata = build_matrix(
            object_db,
            matrix_method=matrix_method,
            filters=train_filters or {},
            m=m,
            random_state=int(seed),
            balanced_pixel_strategy=effective_strategy,
            under_m_policy=under_m_policy,
            require_finite=True,
            require_two_classes=True,
        )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        metadata = _metadata_aliases(metadata)
        if group_col not in metadata:
            raise KeyError(f"PCA matrix metadata has no {group_col!r} group.")
        groups = np.asarray(metadata[group_col]).astype(str)

        preprocessor = SpectralPreprocessor(
            preprocessing_steps,
            sg_window_length=sg_window_length,
            sg_polyorder=sg_polyorder,
        )
        X_pre = preprocessor.fit_transform(X, wavelengths=wavelengths)
        n_fit = min(int(n_components), X_pre.shape[1], max(len(X_pre) - 1, 1))
        full_pca = PCAModel(n_components=n_fit).fit(X_pre)
        run_type = "reference" if seed == int(reference_seed) else "seed_full"
        loading_rows.append(
            {
                **identity,
                "run_type": run_type,
                "seed": int(seed),
                "fold": -1,
                "strategy": effective_strategy,
                "loadings": full_pca.loadings_,
            }
        )
        if seed != int(reference_seed):
            continue
        canonical = (X, y, metadata, X_pre)

        if fold_assignments is None:
            from src.workflows.protocol_split import build_grouped_folds

            group_reference = pd.DataFrame(
                {
                    group_col: groups,
                    "label": y.astype(str),
                    "batch": np.asarray(metadata["batch"]).astype(str),
                }
            ).drop_duplicates(group_col)
            generated, _ = build_grouped_folds(
                group_reference,
                group_col=group_col,
                label_col="label",
                batch_col="batch",
                n_splits=min(int(n_splits), len(group_reference)),
                random_state=int(reference_seed),
                require_complete_coverage=False,
            )
            common_group_to_fold = generated.set_index(group_col)["fold_id"]
        else:
            required = {group_col, "fold_id"}
            missing = sorted(required.difference(fold_assignments.columns))
            if missing:
                raise KeyError(f"Fold assignment is missing columns: {missing}")
            common_group_to_fold = (
                fold_assignments[[group_col, "fold_id"]]
                .drop_duplicates()
                .set_index(group_col)["fold_id"]
            )
        observation_folds = pd.Series(groups).map(common_group_to_fold).to_numpy()
        if pd.isna(observation_folds).any():
            raise RuntimeError("At least one PCA observation has no common fold.")

        for fold in sorted(pd.unique(observation_folds)):
            valid_idx = np.flatnonzero(observation_folds == fold)
            fit_idx = np.flatnonzero(observation_folds != fold)
            if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
                raise RuntimeError(f"Fold {fold} does not preserve both classes.")
            fold_preprocessor = SpectralPreprocessor(
                preprocessing_steps,
                sg_window_length=sg_window_length,
                sg_polyorder=sg_polyorder,
            )
            X_fit = fold_preprocessor.fit_transform(
                X[fit_idx], wavelengths=wavelengths
            )
            X_valid = fold_preprocessor.transform(X[valid_idx])
            n_fold_components = min(
                int(n_components), X_fit.shape[1], max(len(X_fit) - 1, 1)
            )
            pca = PCAModel(n_components=n_fold_components).fit(X_fit)
            T_fit = pca.transform(X_fit)
            T_valid = pca.transform(X_valid)
            metrics = compute_pca_summary_metrics(
                pca_model=pca,
                X_train=X_fit,
                T_train=T_fit,
                y_train=y[fit_idx],
                metadata_train={
                    key: value[fit_idx]
                    for key, value in metadata.items()
                    if value is not None and len(value) == len(X)
                },
                X_projection=X_valid,
                T_projection=T_valid,
                y_projection=y[valid_idx],
                metadata_projection={
                    key: value[valid_idx]
                    for key, value in metadata.items()
                    if value is not None and len(value) == len(X)
                },
                n_components=min(3, n_fold_components),
                matrix_method=matrix_method,
            )
            metrics.update(
                {
                    **identity,
                    "run_type": "group_fold",
                    "seed": int(seed),
                    "fold": int(fold),
                    "strategy": effective_strategy,
                }
            )
            metric_rows.append(metrics)
            loading_rows.append(
                {
                    **identity,
                    "run_type": "group_fold",
                    "seed": int(seed),
                    "fold": int(fold),
                    "strategy": effective_strategy,
                    "loadings": pca.loadings_,
                }
            )

    if canonical is not None and int(n_bootstrap) > 0:
        X, _, metadata, X_pre = canonical
        bootstrap_groups = metadata.get(bootstrap_group_col)
        if bootstrap_groups is None:
            raise KeyError(
                f"PCA matrix metadata has no {bootstrap_group_col!r} bootstrap group."
            )
        rng = np.random.default_rng(int(reference_seed) + 10_000)
        preprocessing_is_row_wise = "msc" not in preprocessing_steps
        for bootstrap_id in range(int(n_bootstrap)):
            indices = _bootstrap_group_indices(bootstrap_groups, rng)
            if len(indices) < 2:
                continue
            if preprocessing_is_row_wise:
                X_boot = X_pre[indices]
            else:
                bootstrap_preprocessor = SpectralPreprocessor(
                    preprocessing_steps,
                    sg_window_length=sg_window_length,
                    sg_polyorder=sg_polyorder,
                )
                X_boot = bootstrap_preprocessor.fit_transform(
                    X[indices], wavelengths=wavelengths
                )
            n_fit = min(
                int(n_components), X_boot.shape[1], max(len(X_boot) - 1, 1)
            )
            pca = PCAModel(n_components=n_fit).fit(X_boot)
            loading_rows.append(
                {
                    **identity,
                    "run_type": "source_image_bootstrap",
                    "seed": int(reference_seed),
                    "fold": int(bootstrap_id),
                    "strategy": effective_strategy,
                    "loadings": pca.loadings_,
                }
            )

    return (
        _summarize_stability_metrics(metric_rows),
        compare_aligned_loadings(
            loading_rows,
            n_components=n_components,
        ),
    )


def summarize_pca_stability(
    metric_stability_df: pd.DataFrame,
    loading_stability_df: pd.DataFrame,
) -> dict:
    """Reduce primary fold and secondary seed/bootstrap stability separately."""
    loadings = loading_stability_df.copy()
    run_keys = [
        column
        for column in ("run_type", "seed", "fold", "strategy")
        if column in loadings
    ]
    run_level = (
        loadings.drop_duplicates(run_keys)
        if run_keys
        else pd.DataFrame()
    )

    def loading_stat(run_type, column, statistic="mean"):
        if run_level.empty or column not in run_level:
            return np.nan
        values = pd.to_numeric(
            run_level.loc[run_level["run_type"].eq(run_type), column],
            errors="coerce",
        )
        values = values[np.isfinite(values)]
        if values.empty:
            return np.nan
        return float(getattr(values, statistic)())

    def metric_stat(metric, statistic, run_type="group_fold"):
        if not len(metric_stability_df):
            return np.nan
        match = metric_stability_df.loc[
            metric_stability_df["metric"].eq(metric)
            & metric_stability_df.get(
                "run_type",
                pd.Series("group_fold", index=metric_stability_df.index),
            ).eq(run_type)
        ]
        return float(match.iloc[0][statistic]) if len(match) else np.nan

    group_projection_shift = metric_stat(
        "mean_train_projection_shift_norm",
        "mean",
    )
    group_q_deviation = metric_stat("projection_q_deviation", "mean")
    group_projection_shift_std = metric_stat(
        "mean_train_projection_shift_norm",
        "std",
    )
    group_instability = loading_stat(
        "group_fold", "subspace_instability"
    )
    group_correlation = loading_stat(
        "group_fold", "loading_abs_correlation"
    )
    group_angle = loading_stat("group_fold", "loading_angle_deg")
    seed_instability = loading_stat(
        "seed_full", "subspace_instability"
    )
    bootstrap_instability = loading_stat(
        "source_image_bootstrap", "subspace_instability"
    )
    n_group_runs = int(
        run_level.get("run_type", pd.Series(dtype=str)).eq("group_fold").sum()
    )
    return {
        "loading_abs_correlation_mean": group_correlation,
        "loading_angle_mean_deg": group_angle,
        "group_fold_subspace_instability": group_instability,
        "seed_subspace_instability": seed_instability,
        "bootstrap_subspace_instability": bootstrap_instability,
        "mean_train_projection_shift_norm": group_projection_shift,
        "projection_q_deviation": group_q_deviation,
        "group_fold_projection_shift_std": group_projection_shift_std,
        "score_stability_std": group_projection_shift_std,
        "instability_metric": group_instability,
        "stability_n_loading_comparisons": n_group_runs,
        "stability_valid": bool(
            n_group_runs > 0 and np.isfinite(group_instability)
        ),
    }
