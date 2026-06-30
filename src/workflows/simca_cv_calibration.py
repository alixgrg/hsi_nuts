from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

from src.spectra.preprocessing import SpectralPreprocessor
from src.decision.truth import add_pixel_truth_labels
from src.decision.aggregation import object_threshold_grid
from src.decision.metrics import binary_detection_metrics
from src.matrices.matrix_registry import build_matrix
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.models.simca import SIMCAClassModel
from src.models.simca_rules import compute_rule_variant_stat_limit
from src.utils import row_value

def _empirical_quantile(values, q):
    """
    Empirical quantile using 'higher' when available.
    This is conservative for thresholds.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))



def build_simca_matrix(
    object_db,
    matrix_method: str,
    filters: dict,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    balanced_pixel_strategy: str = "random",
):
    """
    Build X, y, metadata for SIMCA calibration.

    For cross-validation, metadata['object_id'] is used as group variable.
    """
    return build_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )


def _make_group_splitter(groups, n_splits=None):
    """
    If n_splits is None or >= number of groups, use Leave-One-Group-Out.
    Otherwise use GroupKFold.
    """
    groups = np.asarray(groups).astype(str)
    n_groups = len(np.unique(groups))

    if n_groups < 2:
        raise ValueError("Need at least two groups for group cross-validation.")

    if n_splits is None or int(n_splits) >= n_groups:
        return LeaveOneGroupOut()

    return GroupKFold(n_splits=int(n_splits))


def _fit_fold_simca(
    X_train_raw,
    preprocessing_steps,
    n_components,
    alpha,
    wavelengths=None,
    sg_window_length=9,
    sg_polyorder=2,
    target_class: str = "peanut",
):
    """
    Fit preprocessing + one-class SIMCA on one CV fold.
    """
    preprocessor = SpectralPreprocessor(
        steps=preprocessing_steps,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )

    X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)

    model = SIMCAClassModel(
        class_name=target_class,
        n_components=int(n_components),
        alpha=float(alpha),
    )
    model.fit(X_train)

    return preprocessor, model


def _fold_statistics_from_values(values, model):
    """
    Compute fold-level SIMCA statistics using the fold model.
    """
    H = np.asarray(values["H"], dtype=float)
    Q = np.asarray(values["Q"], dtype=float)

    H_norm_chi2 = H / model.H_limit_
    Q_norm_chi2 = Q / model.Q_limit_

    simple_chi2_stat = np.maximum(H_norm_chi2, Q_norm_chi2)
    alternative_chi2_stat = H_norm_chi2 + Q_norm_chi2

    data_driven_stat = (
        model.NQ_ * Q / max(model.Q0_, model.eps)
        + model.NH_ * H / max(model.H0_, model.eps)
    )
    data_driven_limit = chi2.ppf(1.0 - model.alpha, model.NQ_ + model.NH_)

    return {
        "H": H,
        "Q": Q,
        "H_norm_chi2": H_norm_chi2,
        "Q_norm_chi2": Q_norm_chi2,
        "simple_chi2_stat": simple_chi2_stat,
        "alternative_chi2_stat": alternative_chi2_stat,
        "data_driven_stat": data_driven_stat,
        "data_driven_chi2_limit": float(data_driven_limit),
        "H_chi2_limit_fold": float(model.H_limit_),
        "Q_chi2_limit_fold": float(model.Q_limit_),
        "H0_fold": float(model.H0_),
        "Q0_fold": float(model.Q0_),
        "NH_fold": float(model.NH_),
        "NQ_fold": float(model.NQ_),
    }


def calibrate_simca_thresholds_cv(
    object_db,
    train_filters: dict,
    matrix_method: str = "balanced_pixels",
    preprocessing_steps=("absorbance", "sg_d1"),
    n_components: int = 10,
    alpha: float = 0.05,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    group_col: str = "object_id",
    n_splits: int | None = None,
    balanced_pixel_strategy: str = 'random',
    target_class: str = "peanut",
):
    """
    Cross-validated empirical calibration of SIMCA thresholds.

    The folds are grouped by object_id to avoid pixel leakage.

    Returns
    -------
    cv_df : DataFrame
        Cross-validated target-class distances and statistics.

    thresholds : dict
        Empirical thresholds estimated from CV distances/statistics.
    """
    X_raw, y, meta = build_simca_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )

    if group_col not in meta:
        raise ValueError(f"metadata does not contain group_col={group_col!r}.")

    groups = np.asarray(meta[group_col]).astype(str)

    splitter = _make_group_splitter(groups, n_splits=n_splits)

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(X_raw, y, groups=groups),
        start=1,
    ):
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]

        if X_train_raw.shape[0] <= int(n_components):
            continue

        preprocessor, model = _fit_fold_simca(
            X_train_raw=X_train_raw,
            preprocessing_steps=preprocessing_steps,
            n_components=n_components,
            alpha=alpha,
            wavelengths=wavelengths,
            sg_window_length=sg_window_length,
            sg_polyorder=sg_polyorder,
            target_class=target_class
        )

        X_test = preprocessor.transform(X_test_raw)
        values = model.decision_values(X_test)
        stats = _fold_statistics_from_values(values, model)

        n_test = len(test_idx)

        for j in range(n_test):
            row = {
                "fold": int(fold_id),
                "row_index": int(test_idx[j]),
                "label": str(y[test_idx[j]]),
                "group": str(groups[test_idx[j]]),
            }

            for key, val in meta.items():
                if len(val) == len(X_raw):
                    row[key] = val[test_idx[j]]

            for stat_name, stat_values in stats.items():
                if np.ndim(stat_values) == 0:
                    row[stat_name] = float(stat_values)
                else:
                    row[stat_name] = float(stat_values[j])

            rows.append(row)

    cv_df = pd.DataFrame(rows)

    if cv_df.empty:
        raise RuntimeError("No CV distances were produced. Check n_components and groups.")

    q = 1.0 - float(alpha)

    H_emp_cv = _empirical_quantile(cv_df["H"], q)
    Q_emp_cv = _empirical_quantile(cv_df["Q"], q)

    cv_df["alternative_empHQ_stat"] = (
        cv_df["H"] / max(H_emp_cv, 1e-12)
        + cv_df["Q"] / max(Q_emp_cv, 1e-12)
    )

    thresholds = {
        "alpha": float(alpha),
        "quantile": float(q),

        # Empirical partial thresholds
        "H_emp_cv": H_emp_cv,
        "Q_emp_cv": Q_emp_cv,

        # Empirical thresholds on normalized statistics
        "simple_emp_cv": _empirical_quantile(cv_df["simple_chi2_stat"], q),
        "alternative_chi2_emp_cv": _empirical_quantile(cv_df["alternative_chi2_stat"], q),
        "alternative_empHQ_emp_cv": _empirical_quantile(cv_df["alternative_empHQ_stat"], q),
        "data_driven_emp_cv": _empirical_quantile(cv_df["data_driven_stat"], q),

        # Useful diagnostics
        "H_chi2_limit_fold_median": float(np.median(cv_df["H_chi2_limit_fold"])),
        "Q_chi2_limit_fold_median": float(np.median(cv_df["Q_chi2_limit_fold"])),
        "data_driven_chi2_limit_fold_median": float(np.median(cv_df["data_driven_chi2_limit"])),
        "n_cv_observations": int(len(cv_df)),
        "n_cv_groups": int(cv_df["group"].nunique()),
    }

    return cv_df, thresholds


def fit_final_simca_model(
    object_db,
    train_filters: dict,
    matrix_method: str = "balanced_pixels",
    preprocessing_steps=("absorbance", "sg_d1"),
    n_components: int = 10,
    alpha: float = 0.05,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    balanced_pixel_strategy: str = 'random',
    target_class: str = "peanut",
):
    """
    Fit final SIMCA model on all target training data.
    """
    X_train_raw, y_train, meta_train = build_simca_matrix(
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
        n_components=int(n_components),
        alpha=float(alpha),
    )
    model.fit(X_train)

    return {
        "matrix_method": matrix_method,
        "preprocessing_steps": tuple(preprocessing_steps),
        "preprocessor": preprocessor,
        "model": model,
        "X_train_raw": X_train_raw,
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
    }



def project_pixels_with_rule_variants(
    object_db,
    final_bundle: dict,
    projection_filters: dict,
    cv_thresholds: dict | None = None,
    rule_variants: Sequence[str] = (
        "simple_chi2",
        "alternative_chi2_fixed2",
        "alternative_chi2_emp_cv",
        "alternative_empHQ_emp_cv",
        "data_driven_chi2",
        "data_driven_emp_cv",
        "combined_index_chi2",
    ),
    balanced_pixel_strategy: str = "random",
    target_class: str = "peanut",
):
    """
    Project pixels and compute predictions for several SIMCA rule variants.

    Returns a wide pixel dataframe with:
        pred_<variant>
        stat_<variant>
        limit_<variant>
    """
    X_pixel_raw, y_pixel, meta_pixel = build_matrix(
        object_db=object_db,
        matrix_method="pixel",
        filters=projection_filters,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )

    X_pixel = final_bundle["preprocessor"].transform(X_pixel_raw)
    model = final_bundle["model"]
    values = model.decision_values(X_pixel)

    H = np.asarray(values["H"], dtype=float)
    Q = np.asarray(values["Q"], dtype=float)

    df = pd.DataFrame(meta_pixel)
    df["label"] = y_pixel.astype(str)
    df["H"] = H
    df["Q"] = Q
    df["H_norm_limit"] = H / model.H_limit_
    df["Q_norm_limit"] = Q / model.Q_limit_
    df["matrix_method"] = final_bundle["matrix_method"]
    df["target_class"] = target_class

    for k in range(values["scores"].shape[1]):
        df[f"T{k+1}"] = values["scores"][:, k]

    for variant in rule_variants:
        stat, limit = compute_rule_variant_stat_limit(
            H=H,
            Q=Q,
            model=model,
            variant_name=variant,
            cv_thresholds=cv_thresholds,
        )
        df[f"stat_{variant}"] = stat
        df[f"limit_{variant}"] = float(limit)
        df[f"pred_{variant}"] = stat < float(limit)

    return df, values, X_pixel


def summarize_cv_calibration(cv_df: pd.DataFrame, cv_thresholds: dict) -> pd.DataFrame:
    """
    Summarize how many target observations are rejected by each threshold
    on the cross-validated target distances.
    """
    alpha = float(cv_thresholds["alpha"])

    variants = {
        "simple_chi2": ("simple_chi2_stat", 1.0),
        "simple_emp_cv": ("simple_chi2_stat", cv_thresholds["simple_emp_cv"]),
        "alternative_chi2_fixed2": ("alternative_chi2_stat", 2.0),
        "alternative_chi2_emp_cv": ("alternative_chi2_stat", cv_thresholds["alternative_chi2_emp_cv"]),
        "alternative_empHQ_emp_cv": ("alternative_empHQ_stat", cv_thresholds["alternative_empHQ_emp_cv"]),
        "data_driven_emp_cv": ("data_driven_stat", cv_thresholds["data_driven_emp_cv"]),
    }

    rows = []
    for name, (stat_col, limit) in variants.items():
        stat = cv_df[stat_col].to_numpy(dtype=float)
        rejected = stat >= float(limit)

        rows.append({
            "rule_variant": name,
            "stat_col": stat_col,
            "limit": float(limit),
            "n": int(len(stat)),
            "n_rejected_target_cv": int(np.sum(rejected)),
            "rejection_rate_target_cv": float(np.mean(rejected)),
            "acceptance_rate_target_cv": float(1.0 - np.mean(rejected)),
            "expected_rejection_rate": alpha,
            "expected_acceptance_rate": 1.0 - alpha,
            "abs_rejection_error": float(abs(np.mean(rejected) - alpha)),
        })

    return pd.DataFrame(rows).sort_values("abs_rejection_error").reset_index(drop=True)


def _selection_score_from_metrics(metrics: dict) -> float:
    """
    Scalar orientation score for SIMCA hyperparameter search.

    The final selection should still be interpreted hierarchically:
    minimize FN rate first, then FP rate, then maximize F1 and accuracy.
    This score is mainly useful for plots and Optuna-style scalar ranking.

    Higher is better. The default cost ratio is FN:FP = 10:1 on rates.
    With rates in [0, 1], a 1 percentage point FN-rate increase is treated
    roughly like a 10 percentage point FP-rate increase.
    """
    fn_rate = float(metrics.get("fn_rate", np.nan))
    fp_rate = float(metrics.get("fp_rate", np.nan))
    f1 = float(metrics.get("f1_score", np.nan))
    acc = float(metrics.get("accuracy", np.nan))

    if not np.isfinite(fn_rate):
        sens = float(metrics.get("target_sensitivity", np.nan))
        fn_rate = 1.0 - sens if np.isfinite(sens) else np.nan
    if not np.isfinite(fp_rate):
        spec = float(metrics.get("non_target_specificity", np.nan))
        fp_rate = 1.0 - spec if np.isfinite(spec) else np.nan

    if not np.isfinite(fn_rate) or not np.isfinite(fp_rate):
        return np.nan

    f1_term = f1 if np.isfinite(f1) else 0.0
    acc_term = acc if np.isfinite(acc) else 0.0

    score = -(10.0 * fn_rate + 1.0 * fp_rate) + 0.05 * f1_term + 0.02 * acc_term
    return float(score)


def _metrics_by_batch(obj_df: pd.DataFrame) -> dict:
    """
    Compute compact object-level metrics by batch.

    Returns columns such as:
    - min_batch_balanced_accuracy
    - batch3_peanut_sensitivity
    - batch3_almond_specificity
    - batch3_fn
    - batch3_fp
    """
    if "batch" not in obj_df.columns:
        return {}

    rows = []

    for batch, group in obj_df.groupby("batch", dropna=False):
        if len(group) == 0:
            continue

        metrics = binary_detection_metrics(
            group,
            true_col="true_peanut_object",
            pred_col="predicted_peanut_object",
        )
        metrics["batch"] = batch
        rows.append(metrics)

    if len(rows) == 0:
        return {}

    batch_df = pd.DataFrame(rows)

    out = {
        "min_batch_balanced_accuracy": float(batch_df["balanced_accuracy"].min()),
        "min_batch_peanut_sensitivity": float(batch_df["peanut_sensitivity"].min()),
        "min_batch_almond_specificity": float(batch_df["almond_specificity"].min()),
    }

    for _, row in batch_df.iterrows():
        batch = row["batch"]

        if pd.isna(batch):
            batch_name = "unknown"
        else:
            try:
                batch_name = str(int(batch))
            except Exception:
                batch_name = str(batch)

        prefix = f"batch{batch_name}"

        out[f"{prefix}_balanced_accuracy"] = float(row["balanced_accuracy"])
        out[f"{prefix}_peanut_sensitivity"] = float(row["peanut_sensitivity"])
        out[f"{prefix}_almond_specificity"] = float(row["almond_specificity"])
        out[f"{prefix}_fn"] = int(row["fn"])
        out[f"{prefix}_fp"] = int(row["fp"])

    return out


def _cv_rule_diagnostics(cv_calibration_summary: pd.DataFrame, rule_variant: str) -> dict:
    """
    Extract calibration diagnostics for one rule variant.
    """
    if cv_calibration_summary is None or len(cv_calibration_summary) == 0:
        return {}

    sub = cv_calibration_summary[
        cv_calibration_summary["rule_variant"].astype(str).eq(str(rule_variant))
    ]

    if len(sub) == 0:
        return {}

    row = sub.iloc[0]

    return {
        "cv_target_rejection_rate": float(row["rejection_rate_target_cv"]),
        "cv_target_acceptance_rate": float(row["acceptance_rate_target_cv"]),
        "cv_expected_rejection_rate": float(row["expected_rejection_rate"]),
        "cv_abs_rejection_error": float(row["abs_rejection_error"]),
        "cv_rule_limit": float(row["limit"]),
    }


def run_simca_empirical_rule_grid(
    object_db,
    image_db,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    matrix_methods=("balanced_pixels",),
    rule_variants=(
        "simple_chi2",
        "simple_emp_cv",
        "alternative_chi2_fixed2",
        "alternative_chi2_emp_cv",
        "alternative_empHQ_fixed2",
        "alternative_empHQ_emp_cv",
        "data_driven_chi2",
        "data_driven_emp_cv",
        "combined_index_chi2",
    ),
    n_components_values=(5, 8, 10, 12, 15, 20),
    alpha_values=(0.05,),
    object_thresholds=(0.75,),
    m_values=(40,),
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length_values=(11,),
    sg_polyorder_values=(2,),
    position_dilation_radius_values=(3,),
    cv_n_splits: int | None = 5,
    group_col: str = "object_id",
    keep_pixel_tables: bool = False,
    keep_cv_tables: bool = False,
    verbose: bool = True,
    balanced_pixel_strategy_values=("random",),
    default_m: int = 40,
    default_sg_window_length: int = 11,
    default_sg_polyorder: int = 2,
    target_class: str = "peanut",
):
    """
    Grid search for SIMCA rule variants using empirical CV calibration.

    Hyperparameters tested:
    - matrix_method
    - preprocessing
    - rule_variant
    - n_components
    - alpha
    - object_threshold
    - m, only if matrix_method == "balanced_pixels"
    - balanced_pixel_strategy, only if matrix_method == "balanced_pixels"
    - sg_window_length / sg_polyorder, only if preprocessing uses SG
    - position_dilation_radius

    For each base configuration:
        1. calibrate empirical thresholds by grouped CV;
        2. fit final one-class SIMCA on all calibration data;
        3. project validation pixels once;
        4. apply all rule variants;
        5. aggregate to object-level for all object thresholds.
    """
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)

    # Import here to avoid circular imports at module load time.
    from src.workflows.simca_pixel_grid import (
        _balanced_strategy_grid_for_matrix,
        _matrix_family_from_method,
        _valid_sg_parameter_pairs,
        _standard_grid_sort,
    )

    summary_rows = []
    results = {}
    errors = []

    base_configs = []

    for matrix_method in matrix_methods:
        matrix_method = str(matrix_method)
        matrix_family = _matrix_family_from_method(matrix_method)

        matrix_param_configs = _balanced_strategy_grid_for_matrix(
            matrix_method=matrix_method,
            m_values=m_values,
            balanced_pixel_strategy_values=balanced_pixel_strategy_values,
            default_m=default_m,
        )

        for matrix_params in matrix_param_configs:
            for preprocessing_name, preprocessing_steps in preprocessing_configs.items():
                preprocessing_steps = tuple(preprocessing_steps)

                sg_pairs = _valid_sg_parameter_pairs(
                    preprocessing_steps=preprocessing_steps,
                    sg_window_length_values=sg_window_length_values,
                    sg_polyorder_values=sg_polyorder_values,
                    default_sg_window_length=default_sg_window_length,
                    default_sg_polyorder=default_sg_polyorder,
                )

                for sg_window_length, sg_polyorder in sg_pairs:
                    for n_components in n_components_values:
                        for alpha in alpha_values:
                            for position_dilation_radius in position_dilation_radius_values:
                                base_configs.append({
                                    "matrix_family": matrix_family,
                                    "matrix_method": matrix_method,
                                    "training_matrix_id": matrix_params["training_matrix_id"],
                                    "m": matrix_params["m"],
                                    "m_effective": matrix_params["m_effective"],
                                    "balanced_pixel_strategy": matrix_params["balanced_pixel_strategy"],
                                    "balanced_pixel_strategy_effective": matrix_params["balanced_pixel_strategy_effective"],
                                    "preprocessing": str(preprocessing_name),
                                    "preprocessing_steps": preprocessing_steps,
                                    "n_components": int(n_components),
                                    "alpha": float(alpha),
                                    "sg_window_length": int(sg_window_length),
                                    "sg_polyorder": int(sg_polyorder),
                                    "position_dilation_radius": int(position_dilation_radius),
                                })

    total_base = len(base_configs)

    for base_counter, cfg in enumerate(base_configs, start=1):
        base_key = (
            "empirical_cv_rule",
            cfg["training_matrix_id"],
            cfg["preprocessing"],
            cfg["n_components"],
            cfg["alpha"],
            cfg["sg_window_length"],
            cfg["sg_polyorder"],
            cfg["position_dilation_radius"],
        )

        if verbose:
            print(
                f"\n[{base_counter}/{total_base}] empirical_cv | "
                f"matrix={cfg['training_matrix_id']} | "
                f"preprocessing={cfg['preprocessing']} | "
                f"A={cfg['n_components']} | "
                f"alpha={cfg['alpha']} | "
                f"SG=({cfg['sg_window_length']},{cfg['sg_polyorder']}) | "
                f"dilation={cfg['position_dilation_radius']}"
            )

        try:
            cv_df, cv_thresholds = calibrate_simca_thresholds_cv(
                object_db=object_db,
                train_filters=train_filters,
                matrix_method=cfg["matrix_method"],
                preprocessing_steps=cfg["preprocessing_steps"],
                n_components=cfg["n_components"],
                alpha=cfg["alpha"],
                m=cfg["m_effective"],
                random_state=random_state,
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=cfg["sg_window_length"],
                sg_polyorder=cfg["sg_polyorder"],
                group_col=group_col,
                n_splits=cv_n_splits,
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
            )

            cv_calibration_summary = summarize_cv_calibration(
                cv_df=cv_df,
                cv_thresholds=cv_thresholds,
            )

            final_bundle = fit_final_simca_model(
                object_db=object_db,
                train_filters=train_filters,
                matrix_method=cfg["matrix_method"],
                preprocessing_steps=cfg["preprocessing_steps"],
                n_components=cfg["n_components"],
                alpha=cfg["alpha"],
                m=cfg["m_effective"],
                random_state=random_state,
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=cfg["sg_window_length"],
                sg_polyorder=cfg["sg_polyorder"],
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
            )

            pixel_variants_df, simca_values, X_pixel = project_pixels_with_rule_variants(
                object_db=object_db,
                final_bundle=final_bundle,
                projection_filters=projection_filters,
                cv_thresholds=cv_thresholds,
                rule_variants=rule_variants,
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
            )

            pixel_variants_df = add_pixel_truth_labels(
                pixel_df=pixel_variants_df,
                image_db=image_db,
                object_db=object_db,
                target_class=target_class,
                dilation_radius=cfg["position_dilation_radius"],
            )

            object_tables_by_rule = {}

            for rule_variant in rule_variants:
                pred_col = f"pred_{rule_variant}"
                stat_col = f"stat_{rule_variant}"
                limit_col = f"limit_{rule_variant}"

                if pred_col not in pixel_variants_df.columns:
                    errors.append({
                        **cfg,
                        "search_method": "grid_empirical_cv_rules",
                        "model_family": "empirical_cv_rule",
                        "rule_variant": rule_variant,
                        "error": f"Missing prediction column: {pred_col}",
                        "target_class": target_class,
                    })
                    continue

                tmp_pixel_df = pixel_variants_df.copy()

                target_pred_col = f"predicted_{target_class}_pixel"
                tmp_pixel_df[target_pred_col] = tmp_pixel_df[pred_col].astype(bool)

                if target_class == "peanut":
                    tmp_pixel_df["predicted_peanut_pixel"] = tmp_pixel_df[target_pred_col]

                tmp_pixel_df["predicted_label_pixel"] = np.where(
                    tmp_pixel_df[target_pred_col],
                    target_class,
                    f"non_{target_class}",
                )
                tmp_pixel_df["rule_statistic"] = tmp_pixel_df[stat_col]
                tmp_pixel_df["rule_limit"] = tmp_pixel_df[limit_col]
                tmp_pixel_df["rule_name"] = str(rule_variant)

                threshold_df, object_tables = object_threshold_grid(
                    pixel_df=tmp_pixel_df,
                    object_db=object_db,
                    target_class=target_class,
                    thresholds=object_thresholds,
                )

                if threshold_df is None or len(threshold_df) == 0:
                    errors.append({
                        **cfg,
                        "search_method": "grid_empirical_cv_rules",
                        "model_family": "empirical_cv_rule",
                        "rule_variant": rule_variant,
                        "error": "Empty threshold_df.",
                        "target_class": target_class,
                    })
                    continue

                object_tables_by_rule[str(rule_variant)] = object_tables

                cv_rule_metrics = _cv_rule_diagnostics(
                    cv_calibration_summary=cv_calibration_summary,
                    rule_variant=rule_variant,
                )

                threshold_df = threshold_df.copy()
                threshold_df["search_method"] = "grid_empirical_cv_rules"
                threshold_df["model_family"] = "empirical_cv_rule"
                threshold_df["matrix_family"] = cfg["matrix_family"]
                threshold_df["training_matrix_id"] = cfg["training_matrix_id"]
                threshold_df["matrix_method"] = cfg["matrix_method"]
                threshold_df["m"] = cfg["m"]
                threshold_df["m_effective"] = cfg["m_effective"]
                threshold_df["balanced_pixel_strategy"] = cfg["balanced_pixel_strategy"]
                threshold_df["balanced_pixel_strategy_effective"] = cfg["balanced_pixel_strategy_effective"]
                threshold_df["preprocessing"] = cfg["preprocessing"]
                threshold_df["preprocessing_steps"] = "+".join(cfg["preprocessing_steps"])
                threshold_df["rule_variant"] = str(rule_variant)
                threshold_df["rule"] = str(rule_variant)
                threshold_df["n_components"] = cfg["n_components"]
                threshold_df["alpha"] = cfg["alpha"]
                threshold_df["target_class"] = target_class
                threshold_df["sg_window_length"] = cfg["sg_window_length"]
                threshold_df["sg_polyorder"] = cfg["sg_polyorder"]
                threshold_df["position_dilation_radius"] = cfg["position_dilation_radius"]
                threshold_df["cv_n_splits"] = (
                    cv_n_splits if cv_n_splits is not None else "leave_one_group_out"
                )
                threshold_df["n_cv_observations"] = int(
                    cv_thresholds.get("n_cv_observations", len(cv_df))
                )
                threshold_df["n_cv_groups"] = int(
                    cv_thresholds.get("n_cv_groups", cv_df["group"].nunique())
                )

                threshold_df["H_emp_cv"] = float(cv_thresholds.get("H_emp_cv", np.nan))
                threshold_df["Q_emp_cv"] = float(cv_thresholds.get("Q_emp_cv", np.nan))
                threshold_df["simple_emp_cv_limit"] = float(cv_thresholds.get("simple_emp_cv", np.nan))
                threshold_df["alternative_chi2_emp_cv_limit"] = float(cv_thresholds.get("alternative_chi2_emp_cv", np.nan))
                threshold_df["alternative_empHQ_emp_cv_limit"] = float(cv_thresholds.get("alternative_empHQ_emp_cv", np.nan))
                threshold_df["data_driven_emp_cv_limit"] = float(cv_thresholds.get("data_driven_emp_cv", np.nan))

                for k_diag, v_diag in cv_rule_metrics.items():
                    threshold_df[k_diag] = v_diag

                threshold_df["selection_score"] = threshold_df.apply(
                    lambda r: _selection_score_from_metrics(r.to_dict()),
                    axis=1,
                )

                summary_rows.append(threshold_df)

            stored = {
                "cv_thresholds": cv_thresholds,
                "cv_calibration_summary": cv_calibration_summary,
                "object_tables_by_rule": object_tables_by_rule,
                "final_bundle": final_bundle,
            }

            if keep_cv_tables:
                stored["cv_df"] = cv_df

            if keep_pixel_tables:
                stored["pixel_variants_df"] = pixel_variants_df
                stored["simca_values"] = simca_values
                stored["X_pixel"] = X_pixel

            results[base_key] = stored

        except Exception as exc:
            errors.append({
                **cfg,
                "search_method": "grid_empirical_cv_rules",
                "model_family": "empirical_cv_rule",
                "rule_variant": "ALL",
                "error": repr(exc),
                "target_class": target_class,
            })

            if verbose:
                print("  -> ERROR:", repr(exc))

    summary_df = (
        pd.concat(summary_rows, ignore_index=True, sort=False)
        if summary_rows
        else pd.DataFrame()
    )

    errors_df = pd.DataFrame(errors)

    if len(summary_df) > 0:
        summary_df = _standard_grid_sort(summary_df)

    return summary_df, results, errors_df


def refit_empirical_cv_rule_row(
    object_db,
    image_db,
    best_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = "peanut",
    keep_cv_tables: bool = False,
):
    """
    Refit one empirical-CV SIMCA configuration from a selected row.
    Compatible with the new dynamic run_simca_empirical_rule_grid API.
    """
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)

    preproc_name = str(best_row["preprocessing"])

    if preproc_name in preprocessing_configs:
        steps = tuple(preprocessing_configs[preproc_name])
    else:
        steps = tuple(str(best_row["preprocessing_steps"]).split("+"))

    rule_variant = str(
        row_value(
            best_row,
            "rule_variant",
            row_value(best_row, "rule_for_refit", None),
        )
    )

    if rule_variant in {"None", "nan"}:
        raise ValueError("Could not infer rule_variant for empirical CV refit.")

    matrix_method = str(best_row["matrix_method"])

    object_threshold = float(row_value(best_row, "object_threshold", 0.75))
    n_components = int(row_value(best_row, "n_components", 5))
    alpha = float(row_value(best_row, "alpha", 0.05))

    m_effective = int(
        row_value(
            best_row,
            "m_effective",
            row_value(best_row, "m", 40),
        )
    )

    sg_window_length = int(row_value(best_row, "sg_window_length", 11))
    sg_polyorder = int(row_value(best_row, "sg_polyorder", 2))
    position_dilation_radius = int(row_value(best_row, "position_dilation_radius", 3))

    balanced_pixel_strategy_effective = str(
        row_value(
            best_row,
            "balanced_pixel_strategy_effective",
            row_value(best_row, "balanced_pixel_strategy", "random"),
        )
    )

    if balanced_pixel_strategy_effective == "not_applicable":
        balanced_pixel_strategy_effective = "random"

    summary_df, results, errors_df = run_simca_empirical_rule_grid(
        object_db=object_db,
        image_db=image_db,
        train_filters=train_filters,
        projection_filters=projection_filters,
        preprocessing_configs={preproc_name: steps},
        matrix_methods=[matrix_method],
        rule_variants=[rule_variant],
        n_components_values=[n_components],
        alpha_values=[alpha],
        object_thresholds=[object_threshold],
        m_values=[m_effective],
        random_state=random_state,
        replace=replace,
        wavelengths=wavelengths,
        sg_window_length_values=[sg_window_length],
        sg_polyorder_values=[sg_polyorder],
        position_dilation_radius_values=[position_dilation_radius],
        cv_n_splits=cv_n_splits,
        group_col=cv_group_col,
        keep_pixel_tables=True,
        keep_cv_tables=keep_cv_tables,
        verbose=False,
        balanced_pixel_strategy_values=[balanced_pixel_strategy_effective],
        default_m=m_effective,
        default_sg_window_length=sg_window_length,
        default_sg_polyorder=sg_polyorder,
        target_class=target_class,
    )

    if len(results) == 0:
        raise RuntimeError("No result returned by empirical CV refit.")

    first_key = next(iter(results.keys()))
    stored = results[first_key]

    object_tables = stored["object_tables_by_rule"][rule_variant]
    object_df = object_tables[object_threshold].copy()

    pixel_df = stored["pixel_variants_df"].copy()

    pred_col = f"pred_{rule_variant}"
    stat_col = f"stat_{rule_variant}"
    limit_col = f"limit_{rule_variant}"

    target_pred_col = f"predicted_{target_class}_pixel"

    pixel_df[target_pred_col] = pixel_df[pred_col].astype(bool)
    pixel_df["predicted_label_pixel"] = np.where(
        pixel_df[target_pred_col],
        target_class,
        f"non_{target_class}",
    )
    pixel_df["rule_statistic"] = pixel_df[stat_col]
    pixel_df["rule_limit"] = pixel_df[limit_col]
    pixel_df["rule_name"] = rule_variant

    if target_class == "peanut":
        pixel_df["predicted_peanut_pixel"] = pixel_df[target_pred_col]

    return {
        "summary_df": summary_df,
        "results": results,
        "errors_df": errors_df,
        "object_df": object_df,
        "pixel_df": pixel_df,
        "rule_variant": rule_variant,
    }