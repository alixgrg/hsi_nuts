# src/simca_pixel_grid.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd

from src.pixel_projection import (
    fit_one_class_peanut_simca,
    predict_pixels_with_simca,
    add_pixel_truth_labels,
    object_threshold_grid,
    binary_detection_metrics,
)


DEFAULT_PREPROCESSING_CONFIGS = {
    "raw": ("raw",),
    "absorbance": ("absorbance",),
    "snv": ("snv",),
    "msc": ("msc",),
    "sg_d1": ("sg_d1",),
    "absorbance_snv": ("absorbance", "snv"),
    "absorbance_msc": ("absorbance", "msc"),
    "absorbance_sg_d1": ("absorbance", "sg_d1"),
    "absorbance_snv_sg_d1": ("absorbance", "snv", "sg_d1"),
}


def normalize_preprocessing_configs(configs=None) -> dict[str, tuple[str, ...]]:
    """
    Normalize preprocessing configurations.

    Accepted inputs
    ---------------
    None:
        Use DEFAULT_PREPROCESSING_CONFIGS.
    dict:
        {"name": ("absorbance", "snv"), ...}
    list/tuple of strings:
        ["raw", "absorbance_snv"] using aliases when known.
    list/tuple of tuples/lists:
        [("absorbance", "snv"), ...], named by joining steps with '_'.
    """
    if configs is None:
        return dict(DEFAULT_PREPROCESSING_CONFIGS)

    if isinstance(configs, Mapping):
        out = {}
        for name, steps in configs.items():
            if isinstance(steps, str):
                steps = DEFAULT_PREPROCESSING_CONFIGS.get(steps, (steps,))
            out[str(name)] = tuple(steps)
        return out

    out = {}
    for item in configs:
        if isinstance(item, str):
            steps = DEFAULT_PREPROCESSING_CONFIGS.get(item, (item,))
            name = item
        else:
            steps = tuple(item)
            name = "_".join(steps)
        out[name] = tuple(steps)
    return out


def make_peanut_train_filters(train_batches=None, split=None) -> dict[str, Any]:
    """Build filters selecting only pure peanut objects for SIMCA training."""
    filters = {
        "sample_kind": ["pure"],
        "object_nut_type": ["peanut"],
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
):
    """
    Fit one-class peanut SIMCA on pure peanut observations, project selected objects
    at pixel level, then aggregate pixel decisions to objects for several thresholds.
    """
    bundle = fit_one_class_peanut_simca(
        object_db=object_db,
        matrix_method=matrix_method,
        train_filters=train_filters,
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
    )

    pixel_df, simca_values, X_pixel = predict_pixels_with_simca(
        object_db=object_db,
        simca_bundle=bundle,
        projection_filters=projection_filters,
    )

    pixel_df = add_pixel_truth_labels(
        pixel_df=pixel_df,
        image_db=image_db,
        object_db=object_db,
        dilation_radius=position_dilation_radius,
    )

    threshold_df, object_tables = object_threshold_grid(
        pixel_df=pixel_df,
        object_db=object_db,
        thresholds=object_thresholds,
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
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    position_dilation_radius: int = 3,
    keep_pixel_tables: bool = False,
    verbose: bool = True,
):
    """
    Compare preprocessing configurations and SIMCA rules.

    Training is controlled by train_filters. For the intended use, pass:
        {"sample_kind": ["pure"], "object_nut_type": ["peanut"], "batch": [1, 2]}

    Projection is always done pixel-wise by predict_pixels_with_simca().
    """
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)
    if train_filters is None:
        train_filters = make_peanut_train_filters()
    if projection_filters is None:
        projection_filters = {"sample_kind": ["mixture"]}

    summary_parts = []
    stored_results = {}
    errors = []

    total = (
        len(matrix_methods)
        * len(preprocessing_configs)
        * len(rule_names)
        * len(tuple(n_components_values))
        * len(tuple(alpha_values))
    )
    k = 0

    for matrix_method in matrix_methods:
        for preproc_name, steps in preprocessing_configs.items():
            for rule_name in rule_names:
                for n_components in n_components_values:
                    for alpha in alpha_values:
                        k += 1
                        key = (
                            str(matrix_method),
                            str(preproc_name),
                            str(rule_name),
                            int(n_components),
                            float(alpha),
                        )
                        if verbose:
                            print(
                                f"[{k}/{total}] matrix={matrix_method} | "
                                f"preproc={preproc_name} | rule={rule_name} | "
                                f"A={n_components} | alpha={alpha}"
                            )
                        try:
                            res = run_single_simca_pixel_projection(
                                object_db=object_db,
                                image_db=image_db,
                                matrix_method=matrix_method,
                                preprocessing_name=preproc_name,
                                preprocessing_steps=steps,
                                rule_name=rule_name,
                                train_filters=train_filters,
                                projection_filters=projection_filters,
                                object_thresholds=object_thresholds,
                                n_components=n_components,
                                alpha=alpha,
                                m=m,
                                random_state=random_state,
                                replace=replace,
                                wavelengths=wavelengths,
                                sg_window_length=sg_window_length,
                                sg_polyorder=sg_polyorder,
                                position_dilation_radius=position_dilation_radius,
                            )
                            if len(res["threshold_df"]) > 0:
                                summary_parts.append(res["threshold_df"])

                            if keep_pixel_tables:
                                stored_results[key] = res
                            else:
                                # Keep only light objects needed for quick inspection.
                                stored_results[key] = {
                                    "threshold_df": res["threshold_df"],
                                    "object_tables": res["object_tables"],
                                    "bundle": res["bundle"],
                                }

                        except Exception as exc:
                            errors.append({
                                "matrix_method": matrix_method,
                                "preprocessing": preproc_name,
                                "rule": rule_name,
                                "n_components": n_components,
                                "alpha": alpha,
                                "error": repr(exc),
                            })
                            if verbose:
                                print("  -> ERROR:", repr(exc))

    summary_df = (
        pd.concat(summary_parts, ignore_index=True)
        if summary_parts
        else pd.DataFrame()
    )

    if len(summary_df) > 0:
        sort_cols = [
            col for col in [
                "balanced_accuracy",
                "peanut_sensitivity",
                "almond_specificity",
            ]
            if col in summary_df.columns
        ]
        if sort_cols:
            summary_df = summary_df.sort_values(
                sort_cols,
                ascending=False,
            ).reset_index(drop=True)

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
):
    """Refit one configuration from a row of the grid summary and keep pixel table."""
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)
    preproc_name = str(best_row["preprocessing"])
    if preproc_name not in preprocessing_configs:
        steps = tuple(str(best_row["preprocessing_steps"]).split("+"))
    else:
        steps = preprocessing_configs[preproc_name]

    if object_thresholds is None:
        object_thresholds = [float(best_row["object_threshold"])]

    return run_single_simca_pixel_projection(
        object_db=object_db,
        image_db=image_db,
        matrix_method=str(best_row["matrix_method"]),
        preprocessing_name=preproc_name,
        preprocessing_steps=steps,
        rule_name=str(best_row["rule"]),
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
    )


def summarize_pixel_errors_by_image(pixel_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pixel-level TP/TN/FP/FN by source image."""
    rows = []
    for image_key, group in pixel_df.groupby("source_image"):
        if "truth_available" in group.columns:
            group = group[group["truth_available"].astype(bool)]
        if len(group) == 0:
            continue
        y_true = group["true_peanut_pixel"].astype(bool).to_numpy()
        y_pred = group["predicted_peanut_pixel"].astype(bool).to_numpy()
        tp = int(np.sum(y_true & y_pred))
        tn = int(np.sum((~y_true) & (~y_pred)))
        fp = int(np.sum((~y_true) & y_pred))
        fn = int(np.sum(y_true & (~y_pred)))
        rows.append({
            "source_image": image_key,
            "n_pixels": int(len(group)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "fp_rate": fp / max(int(np.sum(~y_true)), 1),
            "fn_rate": fn / max(int(np.sum(y_true)), 1),
            "pixel_accuracy": (tp + tn) / max(len(group), 1),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["fn_rate", "fp_rate"],
        ascending=False,
    ).reset_index(drop=True)
