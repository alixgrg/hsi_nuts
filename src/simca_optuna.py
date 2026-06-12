# src/simca_optuna.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.simca_pixel_grid import (
    normalize_preprocessing_configs,
    make_peanut_train_filters,
    run_single_simca_pixel_projection,
)


def _require_optuna():
    try:
        import optuna
        return optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Install it with: conda install -c conda-forge optuna "
            "or pip install optuna."
        ) from exc


def _score_from_metrics(
    metrics: Mapping[str, Any],
    objective_metric: str = "weighted",
    sensitivity_weight: float = 0.0,
    specificity_weight: float = 0.0,
    balanced_accuracy_weight: float = 0.1,
    min_peanut_sensitivity: float = 0.7,
    min_almond_specificity: float =0.2,
    constraint_penalty: float = 2.0,
) -> float:
    """
    Build the scalar score maximized by Optuna.

    objective_metric="balanced_accuracy" uses the selected metric directly.
    objective_metric="weighted" uses:
        w_ba * balanced_accuracy + w_sens * peanut_sensitivity + w_spec * almond_specificity

    Optional constraints are applied as soft penalties.
    """
    ba = float(metrics.get("balanced_accuracy", np.nan))
    sens = float(metrics.get("peanut_sensitivity", np.nan))
    spec = float(metrics.get("almond_specificity", np.nan))

    if objective_metric == "weighted":
        score = (
            balanced_accuracy_weight * ba
            + sensitivity_weight * sens
            + specificity_weight * spec
        )
    else:
        score = float(metrics.get(objective_metric, np.nan))

    if not np.isfinite(score):
        return -np.inf

    if min_peanut_sensitivity is not None and np.isfinite(sens):
        score -= constraint_penalty * max(0.0, float(min_peanut_sensitivity) - sens)

    if min_almond_specificity is not None and np.isfinite(spec):
        score -= constraint_penalty * max(0.0, float(min_almond_specificity) - spec)

    return float(score)


def make_simca_optuna_objective(
    object_db,
    image_db,
    train_filters: dict | None = None,
    projection_filters: dict | None = None,
    matrix_methods: Sequence[str] = ("balanced_pixels"),
    preprocessing_configs: Mapping[str, Sequence[str]] | None = None,
    rule_names: Sequence[str] = ("alternative", "data_driven"),
    n_components_choices: Sequence[int] = (5, 8, 10, 15, 20),
    alpha_choices: Sequence[float] = (0.05, 0.01),
    object_threshold_low: float = 0.60,
    object_threshold_high: float = 0.95,
    object_threshold_step: float = 0.05,
    m_choices: Sequence[int] = (20, 40, 60, 80),
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_choices: Sequence[int] = (7, 9, 11, 13, 20),
    sg_polyorder_choices: Sequence[int] = (2, 3),
    position_dilation_radius_choices: Sequence[int] = (0, 2, 3, 5),
    objective_metric: str = "weighted",
    sensitivity_weight: float = 0.0,
    specificity_weight: float = 0.0,
    balanced_accuracy_weight: float = 0.1,
    min_peanut_sensitivity: float = 0.5,
    min_almond_specificity: float = 0.1,
    constraint_penalty: float = 2.0,
):
    """
    Create an Optuna objective for the SIMCA peanut pixel-projection workflow.

    The training filters should select only pure peanut observations, for example:
        {"sample_kind": ["pure"], "object_nut_type": ["peanut"], "batch": [1, 2]}

    Projection is still done pixel-wise inside run_single_simca_pixel_projection().
    """
    optuna = _require_optuna()

    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)
    if train_filters is None:
        train_filters = make_peanut_train_filters(train_batches=[1, 2])
    if projection_filters is None:
        projection_filters = {"sample_kind": ["pure"], "object_nut_type": ["almond", "peanut"], "batch": [3, 4]}

    def objective(trial):
        matrix_method = trial.suggest_categorical("matrix_method", list(matrix_methods))
        preprocessing_name = trial.suggest_categorical("preprocessing", list(preprocessing_configs.keys()))
        preprocessing_steps = preprocessing_configs[preprocessing_name]
        rule_name = trial.suggest_categorical("rule", list(rule_names))
        n_components = trial.suggest_categorical("n_components", list(n_components_choices))
        alpha = trial.suggest_categorical("alpha", list(alpha_choices))
        object_threshold = trial.suggest_float(
            "object_threshold",
            float(object_threshold_low),
            float(object_threshold_high),
            step=float(object_threshold_step),
        )

        if matrix_method == "balanced_pixels":
            m = trial.suggest_categorical("m", list(m_choices))
        else:
            m = int(m_choices[0]) if len(m_choices) else 40
            trial.set_user_attr("m", np.nan)

        uses_sg = any(str(step).startswith("sg_") for step in preprocessing_steps)
        if uses_sg:
            sg_window_length = trial.suggest_categorical("sg_window_length", list(sg_window_choices))
            sg_polyorder = trial.suggest_categorical("sg_polyorder", list(sg_polyorder_choices))
            if sg_polyorder >= sg_window_length:
                raise optuna.exceptions.TrialPruned()
        else:
            sg_window_length = 9
            sg_polyorder = 2

        position_dilation_radius = trial.suggest_categorical(
            "position_dilation_radius",
            list(position_dilation_radius_choices),
        )

        try:
            res = run_single_simca_pixel_projection(
                object_db=object_db,
                image_db=image_db,
                matrix_method=matrix_method,
                preprocessing_name=preprocessing_name,
                preprocessing_steps=preprocessing_steps,
                rule_name=rule_name,
                train_filters=train_filters,
                projection_filters=projection_filters,
                object_thresholds=[float(object_threshold)],
                n_components=int(n_components),
                alpha=float(alpha),
                m=int(m),
                random_state=int(random_state),
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=int(sg_window_length),
                sg_polyorder=int(sg_polyorder),
                position_dilation_radius=int(position_dilation_radius),
            )

            threshold_df = res["threshold_df"]
            if threshold_df is None or len(threshold_df) == 0:
                raise optuna.exceptions.TrialPruned()

            row = threshold_df.iloc[0].to_dict()
            score = _score_from_metrics(
                row,
                objective_metric=objective_metric,
                sensitivity_weight=sensitivity_weight,
                specificity_weight=specificity_weight,
                balanced_accuracy_weight=balanced_accuracy_weight,
                min_peanut_sensitivity=min_peanut_sensitivity,
                min_almond_specificity=min_almond_specificity,
                constraint_penalty=constraint_penalty,
            )

            if not np.isfinite(score):
                raise optuna.exceptions.TrialPruned()

            # Store useful metrics in the trial for later analysis.
            trial.set_user_attr("score", float(score))
            for col in [
                "balanced_accuracy",
                "peanut_sensitivity",
                "almond_specificity",
                "tp",
                "fn",
                "fp",
                "tn",
                "n",
                "n_train_observations",
                "n_projected_pixels",
                "preprocessing_steps",
            ]:
                if col in row:
                    value = row[col]
                    if isinstance(value, np.generic):
                        value = value.item()
                    trial.set_user_attr(col, value)

            if matrix_method == "balanced_pixels":
                trial.set_user_attr("m", int(m))
            trial.set_user_attr("sg_window_length", int(sg_window_length))
            trial.set_user_attr("sg_polyorder", int(sg_polyorder))

            return float(score)

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            raise optuna.exceptions.TrialPruned()

    return objective


def run_optuna_simca_pixel_optimization(
    object_db,
    image_db,
    train_filters: dict | None = None,
    projection_filters: dict | None = None,
    matrix_methods: Sequence[str] = ("balanced_pixels"),
    preprocessing_configs: Mapping[str, Sequence[str]] | None = None,
    rule_names: Sequence[str] = ("alternative", "data_driven"),
    n_components_choices: Sequence[int] = (5, 8, 10, 15, 20),
    alpha_choices: Sequence[float] = (0.05, 0.01),
    n_trials: int = 100,
    timeout: int | None = None,
    study_name: str = "simca_peanut_pixel_projection",
    storage_path: str | Path | None = None,
    load_if_exists: bool = True,
    random_state: int = 42,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
    **objective_kwargs,
):
    """
    Run Optuna optimization and return (study, trials_df).

    If storage_path is given, the study is saved as a SQLite database and can be resumed.
    Example: storage_path="outputs/optuna/simca_optuna.db".
    """
    optuna = _require_optuna()

    sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True)
    pruner = optuna.pruners.NopPruner()

    storage = None
    if storage_path is not None:
        storage_path = Path(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path.as_posix()}"

    objective = make_simca_optuna_objective(
        object_db=object_db,
        image_db=image_db,
        train_filters=train_filters,
        projection_filters=projection_filters,
        matrix_methods=matrix_methods,
        preprocessing_configs=preprocessing_configs,
        rule_names=rule_names,
        n_components_choices=n_components_choices,
        alpha_choices=alpha_choices,
        random_state=random_state,
        **objective_kwargs,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    return study, optuna_trials_dataframe(study)


def optuna_trials_dataframe(study) -> pd.DataFrame:
    """Return a readable dataframe with completed/pruned trials and flattened params/user attrs."""
    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state).split(".")[-1],
            "value": t.value,
        }
        for k, v in t.params.items():
            row[k] = v
        for k, v in t.user_attrs.items():
            # Avoid overwriting explicit params, except score/metrics are useful.
            if k not in row:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if "value" in df.columns:
        df = df.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
    return df


def best_completed_trial_row(trials_df: pd.DataFrame) -> pd.Series:
    """Return the best completed trial row from optuna_trials_dataframe()."""
    d = trials_df[trials_df["state"].eq("COMPLETE")].copy()
    if d.empty:
        raise ValueError("No completed Optuna trial found.")
    return d.sort_values("value", ascending=False).iloc[0]


def refit_optuna_best_trial(
    object_db,
    image_db,
    best_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs: Mapping[str, Sequence[str]] | None = None,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
):
    """Refit the best Optuna configuration and keep the full pixel table for maps."""
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)
    preprocessing_name = str(best_row["preprocessing"])
    preprocessing_steps = preprocessing_configs.get(
        preprocessing_name,
        tuple(str(best_row.get("preprocessing_steps", preprocessing_name)).split("+")),
    )

    matrix_method = str(best_row["matrix_method"])
    m = int(best_row.get("m", 40)) if matrix_method == "balanced_pixels" and pd.notna(best_row.get("m", np.nan)) else 40

    return run_single_simca_pixel_projection(
        object_db=object_db,
        image_db=image_db,
        matrix_method=matrix_method,
        preprocessing_name=preprocessing_name,
        preprocessing_steps=preprocessing_steps,
        rule_name=str(best_row["rule"]),
        train_filters=train_filters,
        projection_filters=projection_filters,
        object_thresholds=[float(best_row["object_threshold"])],
        n_components=int(best_row["n_components"]),
        alpha=float(best_row["alpha"]),
        m=m,
        random_state=int(random_state),
        replace=replace,
        wavelengths=wavelengths,
        sg_window_length=int(best_row.get("sg_window_length", 9)),
        sg_polyorder=int(best_row.get("sg_polyorder", 2)),
        position_dilation_radius=int(best_row.get("position_dilation_radius", 3)),
    )
