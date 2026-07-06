from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision.labels import DEFAULT_NON_TARGET_LABEL, DEFAULT_TARGET_CLASS
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.utils import row_str
from src.workflows.simca_selection_utils import detection_selection_score
from src.workflows.simca import make_target_train_filters, run_single_simca_pixel_projection


def _require_optuna():
    """Import Optuna lazily so this module remains importable without Optuna installed."""
    try:
        import optuna
        return optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Install it with: conda install -c conda-forge optuna "
            "or pip install optuna."
        ) from exc


def make_simca_optuna_objective(
    object_db,
    image_db,
    train_filters: dict | None = None,
    projection_filters: dict | None = None,
    matrix_methods: Sequence[str] = ("balanced_pixels",),
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
    sg_window_choices: Sequence[int] = (7, 9, 11, 13, 21),
    sg_polyorder_choices: Sequence[int] = (2,),
    position_dilation_radius_choices: Sequence[int] = (0, 2, 3, 5),
    objective_metric: str = "fn_fp_hierarchical",
    min_target_sensitivity: float | None = 0.5,
    min_non_target_specificity: float | None = 0.1,
    constraint_penalty: float = 2.0,
    balanced_pixel_strategy_choices: Sequence[str] = ("random", "center"),
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """Create an Optuna objective for the SIMCA pixel-projection workflow."""
    optuna = _require_optuna()
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)

    if train_filters is None:
        train_filters = make_target_train_filters(target_class=target_class, train_batches=[1, 2])
    if projection_filters is None:
        projection_filters = {"sample_kind": ["pure"], "batch": [3, 4]}

    def objective(trial):
        matrix_method = trial.suggest_categorical("matrix_method", list(matrix_methods))
        preprocessing_name = trial.suggest_categorical("preprocessing", list(preprocessing_configs.keys()))
        preprocessing_steps = tuple(preprocessing_configs[preprocessing_name])
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
            balanced_pixel_strategy = trial.suggest_categorical(
                "balanced_pixel_strategy",
                list(balanced_pixel_strategy_choices),
            )
        else:
            m = int(m_choices[0]) if len(m_choices) else 40
            balanced_pixel_strategy = "random"
            trial.set_user_attr("m", np.nan)
            trial.set_user_attr("balanced_pixel_strategy", "not_applicable")

        uses_sg = any(str(step).startswith("sg_") for step in preprocessing_steps)
        if uses_sg:
            sg_window_length = trial.suggest_categorical("sg_window_length", list(sg_window_choices))
            sg_polyorder = trial.suggest_categorical("sg_polyorder", list(sg_polyorder_choices))
            if int(sg_polyorder) >= int(sg_window_length) or int(sg_window_length) % 2 == 0:
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
                balanced_pixel_strategy=balanced_pixel_strategy,
                target_class=target_class,
                non_target_label=non_target_label,
            )

            threshold_df = res["threshold_df"]
            if threshold_df is None or len(threshold_df) == 0:
                raise optuna.exceptions.TrialPruned()

            row = threshold_df.iloc[0].to_dict()
            score = detection_selection_score(
                row,
                objective_metric=objective_metric,
                min_target_sensitivity=min_target_sensitivity,
                min_non_target_specificity=min_non_target_specificity,
                constraint_penalty=constraint_penalty,
            )
            if not np.isfinite(score):
                raise optuna.exceptions.TrialPruned()

            trial.set_user_attr("score", float(score))
            for col in [
                "balanced_accuracy",
                "target_sensitivity",
                "non_target_specificity",
                "fn_rate",
                "fp_rate",
                "tp",
                "fn",
                "fp",
                "tn",
                "n",
                "n_train_observations",
                "n_projected_pixels",
                "preprocessing_steps",
                "target_class",
                "non_target_label",
            ]:
                if col in row:
                    value = row[col]
                    if isinstance(value, np.generic):
                        value = value.item()
                    trial.set_user_attr(col, value)

            if matrix_method == "balanced_pixels":
                trial.set_user_attr("m", int(m))
                trial.set_user_attr("balanced_pixel_strategy", balanced_pixel_strategy)
            trial.set_user_attr("sg_window_length", int(sg_window_length))
            trial.set_user_attr("sg_polyorder", int(sg_polyorder))
            trial.set_user_attr("position_dilation_radius", int(position_dilation_radius))

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
    matrix_methods: Sequence[str] = ("balanced_pixels",),
    preprocessing_configs: Mapping[str, Sequence[str]] | None = None,
    rule_names: Sequence[str] = ("alternative", "data_driven"),
    n_components_choices: Sequence[int] = (5, 8, 10, 15, 20),
    alpha_choices: Sequence[float] = (0.05, 0.01),
    n_trials: int = 100,
    timeout: int | None = None,
    study_name: str | None = None,
    storage_path: str | Path | None = None,
    load_if_exists: bool = True,
    random_state: int = 42,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
    close_storage: bool = True,
    sqlite_timeout: float = 30.0,
    balanced_pixel_strategy_choices: Sequence[str] = ("random", "center"),
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    **objective_kwargs,
):
    """Run Optuna optimization and return ``(study, trials_df)``."""
    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True)
    pruner = optuna.pruners.NopPruner()

    if study_name is None:
        study_name = f"simca_{target_class}_pixel_projection"

    storage = None
    storage_obj = None
    if storage_path is not None:
        storage_path = Path(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_obj = optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path.as_posix()}",
            engine_kwargs={"connect_args": {"timeout": float(sqlite_timeout)}},
        )
        storage = storage_obj

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
        balanced_pixel_strategy_choices=balanced_pixel_strategy_choices,
        target_class=target_class,
        non_target_label=non_target_label,
        **objective_kwargs,
    )

    try:
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
    finally:
        if close_storage and storage_obj is not None:
            try:
                storage_obj.remove_session()
            except Exception as exc:
                print(f"[WARNING] Could not remove Optuna storage session: {exc!r}")


def optuna_trials_dataframe(study) -> pd.DataFrame:
    """Return a readable dataframe with flattened params and user attributes."""
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state).split(".")[-1],
            "value": trial.value,
        }
        for key, value in trial.params.items():
            row[key] = value
        for key, value in trial.user_attrs.items():
            if key not in row:
                row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if "value" in df.columns:
        df = df.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
    return df


def best_completed_trial_row(trials_df: pd.DataFrame) -> pd.Series:
    """Return the best completed trial row from ``optuna_trials_dataframe``."""
    completed = trials_df[trials_df["state"].eq("COMPLETE")].copy()
    if completed.empty:
        raise ValueError("No completed Optuna trial found.")
    return completed.sort_values("value", ascending=False).iloc[0]


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
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """Refit the best Optuna configuration and keep full pixel/object outputs."""
    preprocessing_configs = normalize_preprocessing_configs(preprocessing_configs)
    preprocessing_name = str(best_row["preprocessing"])
    preprocessing_steps = tuple(
        preprocessing_configs.get(
            preprocessing_name,
            tuple(str(best_row.get("preprocessing_steps", preprocessing_name)).split("+")),
        )
    )

    matrix_method = str(best_row["matrix_method"])
    m = int(best_row.get("m", 40)) if matrix_method == "balanced_pixels" and pd.notna(best_row.get("m", np.nan)) else 40
    target_class = row_str(best_row, "target_class", target_class)
    non_target_label = row_str(best_row, "non_target_label", non_target_label)

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
        balanced_pixel_strategy=str(best_row.get("balanced_pixel_strategy", "random")),
        target_class=target_class,
        non_target_label=non_target_label,
    )


def close_optuna_study(study) -> None:
    """Explicitly close Optuna RDB storage sessions, useful on Windows notebooks."""
    if study is None:
        return

    storage = getattr(study, "_storage", None)
    candidates = [storage]
    backend = getattr(storage, "_backend", None)
    if backend is not None:
        candidates.append(backend)

    for storage_candidate in candidates:
        if storage_candidate is None:
            continue
        if hasattr(storage_candidate, "remove_session"):
            try:
                storage_candidate.remove_session()
            except Exception as exc:
                print(f"[WARNING] remove_session failed: {exc!r}")
        engine = getattr(storage_candidate, "engine", None) or getattr(storage_candidate, "_engine", None)
        if engine is not None:
            try:
                engine.dispose()
            except Exception as exc:
                print(f"[WARNING] engine.dispose failed: {exc!r}")
