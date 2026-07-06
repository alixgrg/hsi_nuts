# src/simca_pixel_grid.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.workflows.legacy.pixel_projection import (
    fit_one_class_simca,
    predict_pixels_with_simca,
)
from src.decision.truth import add_pixel_truth_labels
from src.decision.aggregation import object_threshold_grid
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.utils import row_int, row_float, row_str
from src.workflows.legacy.simca_cv_calibration import refit_empirical_cv_rule_row
from src.decision.metrics import binary_detection_metrics, summarize_pixel_errors_by_image
from src.decision.labels import DEFAULT_TARGET_CLASS, DEFAULT_NON_TARGET_LABEL, predicted_col, true_col


def _as_list(x):
    """Normalize scalar / list-like parameter to a list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)):
        return list(x)
    return [x]


def _uses_sg(preprocessing_steps) -> bool:
    """Return True if preprocessing uses a Savitzky-Golay step."""
    return any(str(step).startswith("sg_") for step in tuple(preprocessing_steps))


def _valid_sg_parameter_pairs(
    preprocessing_steps,
    sg_window_length_values=(11,),
    sg_polyorder_values=(2,),
    default_sg_window_length=11,
    default_sg_polyorder=2,
):
    """
    Return valid SG parameter pairs.

    If the preprocessing does not use SG, do not expand the grid.
    """
    if not _uses_sg(preprocessing_steps):
        return [(int(default_sg_window_length), int(default_sg_polyorder))]

    pairs = []

    for window in _as_list(sg_window_length_values):
        for poly in _as_list(sg_polyorder_values):
            window = int(window)
            poly = int(poly)

            if window <= 0:
                continue
            if window % 2 == 0:
                continue
            if poly >= window:
                continue

            pairs.append((window, poly))

    if len(pairs) == 0:
        raise ValueError(
            "No valid Savitzky-Golay parameter pair. "
            "Require odd sg_window_length and sg_polyorder < sg_window_length."
        )

    return pairs


def _matrix_family_from_method(matrix_method: str) -> str:
    matrix_method = str(matrix_method)

    if matrix_method in {"object_mean", "object_median"}:
        return "object_matrix"

    if matrix_method in {"balanced_pixels", "all_pixels", "pixel"}:
        return "pixel_matrix"

    return "unknown_matrix_family"


def _balanced_strategy_grid_for_matrix(
    matrix_method: str,
    m_values=(40,),
    balanced_pixel_strategy_values=("random",),
    default_m: int = 40,
):
    """
    Return valid pixel-sampling configs for a given matrix_method.

    Important:
    - For balanced_pixels, expand over m and balanced_pixel_strategy.
    - For object_mean, object_median, all_pixels, pixel:
      do not expand over balanced_pixel_strategy.
    """
    matrix_method = str(matrix_method)

    if matrix_method == "balanced_pixels":
        configs = []

        for m in _as_list(m_values):
            for strategy in _as_list(balanced_pixel_strategy_values):
                configs.append({
                    "m": int(m),
                    "m_effective": int(m),
                    "balanced_pixel_strategy": str(strategy),
                    "balanced_pixel_strategy_effective": str(strategy),
                    "training_matrix_id": f"balanced_pixel_{strategy}_m{int(m)}",
                })

        return configs

    # For all other matrix types, m and balanced_pixel_strategy are not real hyperparameters.
    return [{
        "m": np.nan,
        "m_effective": int(default_m),
        "balanced_pixel_strategy": "not_applicable",
        "balanced_pixel_strategy_effective": "random",
        "training_matrix_id": str(matrix_method),
    }]


def _standard_grid_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    sort_cols = [
        "fn_rate",
        "fp_rate",
        "f1_score",
        "accuracy",
        "balanced_accuracy",
        "selection_score",
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]

    ascending = [
        True if c in {"fn_rate", "fp_rate"} else False
        for c in sort_cols
    ]

    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def make_target_train_filters(
    target_class: str = DEFAULT_TARGET_CLASS,
    train_batches=None,
    split=None,
) -> dict[str, Any]:
    """Build filters selecting only pure target-class objects for SIMCA training."""
    filters = {
        "sample_kind": ["pure"],
        "object_nut_type": [target_class],
    }

    if train_batches is not None:
        filters["batch"] = list(train_batches)

    if split is not None:
        filters["split"] = [split] if isinstance(split, str) else list(split)

    return filters




def run_single_simca_pixel_projection(
    object_db,
    image_db,
    matrix_method: str,
    preprocessing_name: str,
    preprocessing_steps,
    rule_name: str,
    train_filters: dict,
    projection_filters: dict,
    object_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    n_components: int = 5,
    alpha: float = 0.01,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    position_dilation_radius: int = 3,
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """
    Fit one-class SIMCA on pure target observations, project selected objects
    at pixel level, then aggregate pixel decisions to objects for several thresholds.
    """
    bundle = fit_one_class_simca(
        object_db=object_db,
        matrix_method=matrix_method,
        train_filters=train_filters,
        target_class=target_class,
        preprocessing_steps=tuple(preprocessing_steps),
        n_components=n_components,
        alpha=alpha,
        rule_name=rule_name,
        wavelengths=wavelengths,
        m=m,
        random_state=random_state,
        replace=replace,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )

    pixel_df, simca_values, X_pixel = predict_pixels_with_simca(
        object_db=object_db,
        simca_bundle=bundle,
        projection_filters=projection_filters,
        target_class=target_class,
        non_target_label=non_target_label,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )

    pixel_df = add_pixel_truth_labels(
        pixel_df=pixel_df,
        image_db=image_db,
        object_db=object_db,
        target_class=target_class,
        dilation_radius=position_dilation_radius,
    )

    threshold_df, object_tables = object_threshold_grid(
        pixel_df=pixel_df,
        object_db=object_db,
        thresholds=object_thresholds,
        target_class=target_class,
        non_target_label=non_target_label,
    )

    for df in [threshold_df]:
        if len(df) > 0:
            df["matrix_method"] = matrix_method
            df["preprocessing"] = preprocessing_name
            df["preprocessing_steps"] = "+".join(preprocessing_steps)
            df["rule"] = rule_name
            df["n_components"] = int(n_components)
            df["alpha"] = float(alpha)
            df["m"] = int(m) if matrix_method == "balanced_pixels" else np.nan
            df["n_train_observations"] = int(bundle["X_train"].shape[0])
            df["n_projected_pixels"] = int(len(pixel_df))
            df["balanced_pixel_strategy"] = balanced_pixel_strategy
            df["target_class"] = target_class
            df["sg_window_length"] = int(sg_window_length)
            df["sg_polyorder"] = int(sg_polyorder)
            df["position_dilation_radius"] = int(position_dilation_radius)

    return {
        "bundle": bundle,
        "pixel_df": pixel_df,
        "simca_values": simca_values,
        "X_pixel": X_pixel,
        "threshold_df": threshold_df,
        "object_tables": object_tables,
    }


def run_simca_pixel_projection_grid(
    object_db,
    image_db,
    matrix_methods=("object_mean", "balanced_pixels"),
    preprocessing_configs=None,
    rule_names=("simple", "alternative", "combined_index", "data_driven"),
    train_filters=None,
    projection_filters=None,
    object_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    n_components_values=(5,),
    alpha_values=(0.01,),
    m_values=(40,),
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length_values=(9,),
    sg_polyorder_values=(2,),
    position_dilation_radius_values=(3,),
    balanced_pixel_strategy_values=("random",),
    default_m: int = 40,
    default_sg_window_length: int = 11,
    default_sg_polyorder: int = 2,
    keep_pixel_tables: bool = False,
    verbose: bool = True,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """
    Grid search for standard SIMCA rules.

    Hyperparameters tested:
    - matrix_method
    - preprocessing
    - rule
    - n_components
    - alpha
    - m, only if matrix_method == "balanced_pixels"
    - balanced_pixel_strategy, only if matrix_method == "balanced_pixels"
    - sg_window_length / sg_polyorder, only if preprocessing uses SG
    - position_dilation_radius
    - object_thresholds

    Projection is always done at pixel level.
    """
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)

    if train_filters is None:
        train_filters = make_target_train_filters(target_class=target_class)

    if projection_filters is None:
        projection_filters = {"sample_kind": ["mixture"]}

    summary_parts = []
    stored_results = {}
    errors = []

    # Build valid base configurations first, so total is meaningful.
    grid_configs = []

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
            for preproc_name, steps in preprocessing_configs.items():
                steps = tuple(steps)

                sg_pairs = _valid_sg_parameter_pairs(
                    preprocessing_steps=steps,
                    sg_window_length_values=sg_window_length_values,
                    sg_polyorder_values=sg_polyorder_values,
                    default_sg_window_length=default_sg_window_length,
                    default_sg_polyorder=default_sg_polyorder,
                )

                for sg_window_length, sg_polyorder in sg_pairs:
                    for rule_name in rule_names:
                        for n_components in n_components_values:
                            for alpha in alpha_values:
                                for position_dilation_radius in position_dilation_radius_values:
                                    grid_configs.append({
                                        "matrix_family": matrix_family,
                                        "matrix_method": matrix_method,
                                        "training_matrix_id": matrix_params["training_matrix_id"],
                                        "m": matrix_params["m"],
                                        "m_effective": matrix_params["m_effective"],
                                        "balanced_pixel_strategy": matrix_params["balanced_pixel_strategy"],
                                        "balanced_pixel_strategy_effective": matrix_params["balanced_pixel_strategy_effective"],
                                        "preprocessing": preproc_name,
                                        "preprocessing_steps": steps,
                                        "rule": str(rule_name),
                                        "n_components": int(n_components),
                                        "alpha": float(alpha),
                                        "sg_window_length": int(sg_window_length),
                                        "sg_polyorder": int(sg_polyorder),
                                        "position_dilation_radius": int(position_dilation_radius),
                                    })

    total = len(grid_configs)

    for k, cfg in enumerate(grid_configs, start=1):
        if verbose:
            print(
                f"[{k}/{total}] standard | "
                f"matrix={cfg['training_matrix_id']} | "
                f"preproc={cfg['preprocessing']} | "
                f"rule={cfg['rule']} | "
                f"A={cfg['n_components']} | "
                f"alpha={cfg['alpha']} | "
                f"SG=({cfg['sg_window_length']},{cfg['sg_polyorder']}) | "
                f"dilation={cfg['position_dilation_radius']}"
            )

        key = (
            "standard_rule",
            cfg["training_matrix_id"],
            cfg["preprocessing"],
            cfg["rule"],
            cfg["n_components"],
            cfg["alpha"],
            cfg["sg_window_length"],
            cfg["sg_polyorder"],
            cfg["position_dilation_radius"],
        )

        try:
            res = run_single_simca_pixel_projection(
                object_db=object_db,
                image_db=image_db,
                matrix_method=cfg["matrix_method"],
                preprocessing_name=cfg["preprocessing"],
                preprocessing_steps=cfg["preprocessing_steps"],
                rule_name=cfg["rule"],
                train_filters=train_filters,
                projection_filters=projection_filters,
                object_thresholds=object_thresholds,
                n_components=cfg["n_components"],
                alpha=cfg["alpha"],
                m=cfg["m_effective"],
                random_state=random_state,
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=cfg["sg_window_length"],
                sg_polyorder=cfg["sg_polyorder"],
                position_dilation_radius=cfg["position_dilation_radius"],
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
                non_target_label=non_target_label,
            )

            threshold_df = res["threshold_df"].copy()

            if len(threshold_df) > 0:
                threshold_df["search_method"] = "grid_standard_rules"
                threshold_df["model_family"] = "standard_rule"
                threshold_df["matrix_family"] = cfg["matrix_family"]
                threshold_df["training_matrix_id"] = cfg["training_matrix_id"]
                threshold_df["matrix_method"] = cfg["matrix_method"]
                threshold_df["m"] = cfg["m"]
                threshold_df["m_effective"] = cfg["m_effective"]
                threshold_df["balanced_pixel_strategy"] = cfg["balanced_pixel_strategy"]
                threshold_df["balanced_pixel_strategy_effective"] = cfg["balanced_pixel_strategy_effective"]
                threshold_df["target_class"] = target_class
                threshold_df["non_target_label"] = non_target_label
                threshold_df["preprocessing"] = cfg["preprocessing"]
                threshold_df["sg_window_length"] = cfg["sg_window_length"]
                threshold_df["sg_polyorder"] = cfg["sg_polyorder"]
                threshold_df["position_dilation_radius"] = cfg["position_dilation_radius"]
                summary_parts.append(threshold_df)

            if keep_pixel_tables:
                stored_results[key] = res
            else:
                stored_results[key] = {
                    "threshold_df": threshold_df,
                    "object_tables": res["object_tables"],
                    "bundle": res["bundle"],
                }

            del res

        except Exception as exc:
            err = dict(cfg)
            err.update({
                "search_method": "grid_standard_rules",
                "model_family": "standard_rule",
                "target_class": target_class,
                "non_target_label": non_target_label,
                "error": repr(exc),
            })
            errors.append(err)

            if verbose:
                print("  -> ERROR:", repr(exc))

    summary_df = (
        pd.concat(summary_parts, ignore_index=True, sort=False)
        if summary_parts
        else pd.DataFrame()
    )

    if len(summary_df) > 0:
        summary_df = _standard_grid_sort(summary_df)

    errors_df = pd.DataFrame(errors)

    return summary_df, stored_results, errors_df


def refit_best_grid_row(
    object_db,
    image_db,
    best_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs=None,
    object_thresholds=None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    position_dilation_radius: int = 3,
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """
    Refit one standard-rule configuration from a grid summary row.

    This works for:
    - object matrices: object_mean, object_median
    - pixel matrices: balanced_pixels, all_pixels

    Projection is always done at pixel level.
    """
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)
    preproc_name = str(best_row["preprocessing"])
    if preproc_name not in preprocessing_configs:
        steps = tuple(str(best_row["preprocessing_steps"]).split("+"))
    else:
        steps = tuple(preprocessing_configs[preproc_name])
    if object_thresholds is None:
        object_thresholds = [row_float(best_row, "object_threshold", 0.75)]

    m = row_int(best_row, "m", m)
    sg_window_length = row_int(best_row, "sg_window_length", sg_window_length)
    sg_polyorder = row_int(best_row, "sg_polyorder", sg_polyorder)
    position_dilation_radius = row_int(
        best_row,
        "position_dilation_radius",
        position_dilation_radius,
    )
    balanced_pixel_strategy = row_str(
        best_row,
        "balanced_pixel_strategy",
        balanced_pixel_strategy,
    )
    target_class = row_str(best_row, "target_class", target_class)
    non_target_label = row_str(best_row, "non_target_label", non_target_label)
    rule_name = row_str(
        best_row,
        "rule_for_refit",
        row_str(best_row, "rule", "alternative"),
    )

    return run_single_simca_pixel_projection(
        object_db=object_db,
        image_db=image_db,
        matrix_method=str(best_row["matrix_method"]),
        preprocessing_name=preproc_name,
        preprocessing_steps=steps,
        rule_name=rule_name,
        train_filters=train_filters,
        projection_filters=projection_filters,
        object_thresholds=object_thresholds,
        n_components=int(best_row["n_components"]),
        alpha=float(best_row["alpha"]),
        m=m,
        random_state=random_state,
        replace=replace,
        wavelengths=wavelengths,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        position_dilation_radius=position_dilation_radius,
        balanced_pixel_strategy=balanced_pixel_strategy,
        target_class=target_class,
        non_target_label=non_target_label,
    )



def _attach_selected_metadata(df: pd.DataFrame, row: pd.Series, evaluation_split: str) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "selected_config_id",
        "matrix_family",
        "training_matrix_id",
        "matrix_method",
        "balanced_pixel_strategy",
        "balanced_pixel_strategy_effective",
        "model_family",
        "preprocessing",
        "preprocessing_steps",
        "rule",
        "rule_variant",
        "selected_rule_name",
        "rule_for_refit",
        "target_class",
        "n_components",
        "alpha",
        "object_threshold",
        "m",
        "m_effective",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
    ]:
        if col in row.index:
            out[col] = row[col]

    out["evaluation_split"] = evaluation_split
    return out


def refit_selected_simca_row(
    object_db,
    image_db,
    selected_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """
    Refit one selected SIMCA row.

    - standard_rule: uses refit_best_grid_row
    - empirical_cv_rule: uses refit_empirical_cv_rule_row
    """
    model_family = str(selected_row["model_family"])
    target_class = row_str(selected_row, "target_class", target_class)

    if model_family == "standard_rule":
        res = refit_best_grid_row(
            object_db=object_db,
            image_db=image_db,
            best_row=selected_row,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            object_thresholds=[float(selected_row["object_threshold"])],
            random_state=random_state,
            replace=replace,
            wavelengths=wavelengths,
            target_class=target_class,
            non_target_label=non_target_label,
        )

        object_df = res["object_tables"][float(selected_row["object_threshold"])].copy()
        pixel_df = res["pixel_df"].copy()

    elif model_family == "empirical_cv_rule":
        res = refit_empirical_cv_rule_row(
            object_db=object_db,
            image_db=image_db,
            best_row=selected_row,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            wavelengths=wavelengths,
            random_state=random_state,
            replace=replace,
            cv_n_splits=cv_n_splits,
            cv_group_col=cv_group_col,
            target_class=target_class,
            non_target_label=non_target_label,
        )

        object_df = res["object_df"].copy()
        pixel_df = res["pixel_df"].copy()

    else:
        raise ValueError(f"Unknown model_family={model_family!r}")

    object_df = _attach_selected_metadata(object_df, selected_row, "projection")
    pixel_df = _attach_selected_metadata(pixel_df, selected_row, "projection")

    return {
        "result": res,
        "object_df": object_df,
        "pixel_df": pixel_df,
    }


def refit_selected_simca_configs(
    selected_configs_df: pd.DataFrame,
    object_db,
    image_db,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    evaluation_split: str = "projection",
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """Refit several selected SIMCA configurations."""
    metric_rows = []
    object_parts = []
    pixel_parts = []
    pixel_error_parts = []
    errors = []

    for _, row in selected_configs_df.iterrows():
        config_id = row.get("selected_config_id", "unknown_config")
        print(f"[{evaluation_split}] {config_id}")

        try:
            out = refit_selected_simca_row(
                object_db=object_db,
                image_db=image_db,
                selected_row=row,
                train_filters=train_filters,
                projection_filters=projection_filters,
                preprocessing_configs=preprocessing_configs,
                wavelengths=wavelengths,
                random_state=random_state,
                replace=replace,
                cv_n_splits=cv_n_splits,
                cv_group_col=cv_group_col,
                target_class=target_class,
                non_target_label=non_target_label,
            )

            object_df = out["object_df"].copy()
            pixel_df = out["pixel_df"].copy()

            object_df["evaluation_split"] = evaluation_split
            pixel_df["evaluation_split"] = evaluation_split

            target_object_true_col = true_col(target_class, level="object")
            target_object_pred_col = predicted_col(target_class, level="object")

            if {target_object_true_col, target_object_pred_col}.issubset(object_df.columns):
                metrics = binary_detection_metrics(
                    object_df,
                    true_col=target_object_true_col,
                    pred_col=target_object_pred_col,
                    target_class=target_class,
                    non_target_class=non_target_label,
                )
            else:
                metrics = {}

            metric_row = row.to_dict()
            metric_row.update(metrics)
            metric_row["evaluation_split"] = evaluation_split
            metric_row["n_projected_objects"] = int(len(object_df))
            metric_row["n_projected_pixels"] = int(len(pixel_df))

            metric_rows.append(metric_row)
            object_parts.append(object_df)
            pixel_parts.append(pixel_df)

            target_pixel_true_col = true_col(target_class, "pixel")
            target_pixel_pred_col = predicted_col(target_class, "pixel")

            if {target_pixel_true_col, target_pixel_pred_col}.issubset(pixel_df.columns):
                pixel_err = summarize_pixel_errors_by_image(
                    pixel_df,
                    target_class=target_class,
                    non_target_label=non_target_label,
                    group_cols=("source_image",),
                )

                if len(pixel_err) > 0:
                    pixel_err["selected_config_id"] = config_id
                    pixel_err["evaluation_split"] = evaluation_split
                    pixel_error_parts.append(pixel_err)

        except Exception as exc:
            err = row.to_dict()
            err["evaluation_split"] = evaluation_split
            err["error"] = repr(exc)
            errors.append(err)
            print("  -> ERROR:", repr(exc))

    metrics_df = pd.DataFrame(metric_rows)

    objects_df = (
        pd.concat(object_parts, ignore_index=True, sort=False)
        if object_parts else pd.DataFrame()
    )

    pixels_df = (
        pd.concat(pixel_parts, ignore_index=True, sort=False)
        if pixel_parts else pd.DataFrame()
    )

    pixel_errors_df = (
        pd.concat(pixel_error_parts, ignore_index=True, sort=False)
        if pixel_error_parts else pd.DataFrame()
    )

    errors_df = pd.DataFrame(errors)

    return metrics_df, objects_df, pixels_df, pixel_errors_df, errors_df