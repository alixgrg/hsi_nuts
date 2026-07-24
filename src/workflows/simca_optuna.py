from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from src.decision.labels import DEFAULT_NON_TARGET_LABEL, DEFAULT_TARGET_CLASS
from src.utils import parse_preprocessing_steps, row_str, row_value
from src.workflows.simca_selection_utils import (
    detection_selection_score,
    ensure_candidate_columns,
    normalize_simca_rule_columns,
    fill_selected_config_defaults,
    add_detection_selection_score,
    add_reference_selection_scores,
    sort_detection_selection,
    pareto_front_by_group,
)
from src.workflows.simca import (
    _normalize_preprocessing_configs_by_family,
    _preprocessing_configs_for_family,
    make_target_train_filters,
    matrix_family_from_method,
    run_single_simca_pixel_projection,
    run_simca_rule_variant_grid,
)


def _families_for_matrix_methods(matrix_methods: Sequence[str]) -> set[str]:
    return {matrix_family_from_method(str(method)) for method in matrix_methods}


def _preprocessing_param_name(matrix_family: str, matrix_methods: Sequence[str]) -> str:
    families = _families_for_matrix_methods(matrix_methods)
    return "preprocessing" if len(families) <= 1 else f"preprocessing_{matrix_family}"


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
    alpha_choices: Sequence[float] = (0.01, ),
    object_threshold_low: float = 0.75,
    object_threshold_high: float = 0.8,
    object_threshold_step: float = 0.05,
    m_choices: Sequence[int] = (5, 10, 20, 40, 60, 80, 100),
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_choices: Sequence[int] = (5, 7, 9, 11, 13, 21),
    sg_polyorder_choices: Sequence[int] = (2,),
    default_sg_window_length: Sequence[int] = (11,),
    default_sg_polyorder: Sequence[int] = (2,),
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
    preprocessing_configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)

    if train_filters is None:
        train_filters = make_target_train_filters(target_class=target_class, train_batches=[1, 2])
    if projection_filters is None:
        projection_filters = {
            "sample_kind": ["pure"],
            "object_nut_type": ["almond", target_class],
            "batch": [3],
        }

    def objective(trial):
        matrix_method = trial.suggest_categorical("matrix_method", list(matrix_methods))
        matrix_family = matrix_family_from_method(matrix_method)
        current_preprocessing_configs = _preprocessing_configs_for_family(
            preprocessing_configs_by_family,
            matrix_family,
        )
        if not current_preprocessing_configs:
            raise optuna.exceptions.TrialPruned()
        preprocessing_name = trial.suggest_categorical(
            _preprocessing_param_name(matrix_family, matrix_methods),
            list(current_preprocessing_configs.keys()),
        )
        preprocessing_steps = tuple(current_preprocessing_configs[preprocessing_name])
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

        if matrix_method == "balanced_pixels":
            training_matrix_id = f"balanced_pixel_{balanced_pixel_strategy}_m{int(m)}"
            m_effective = int(m)
            balanced_pixel_strategy_effective = str(balanced_pixel_strategy)
        else:
            training_matrix_id = str(matrix_method)
            m_effective = int(m)
            balanced_pixel_strategy_effective = "random"

        uses_sg = any(str(step).startswith("sg_") for step in preprocessing_steps)
        if uses_sg:
            sg_window_length = trial.suggest_categorical("sg_window_length", list(sg_window_choices))
            sg_polyorder = trial.suggest_categorical("sg_polyorder", list(sg_polyorder_choices))
            if int(sg_polyorder) >= int(sg_window_length) or int(sg_window_length) % 2 == 0:
                raise optuna.exceptions.TrialPruned()
        else:
            sg_window_length = int(default_sg_window_length[0])
            sg_polyorder = int(default_sg_polyorder[0])

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
            trial.set_user_attr("model_family", "standard_rule")
            trial.set_user_attr("matrix_family", matrix_family)
            trial.set_user_attr("training_matrix_id", training_matrix_id)
            trial.set_user_attr("m_effective", m_effective)
            trial.set_user_attr("balanced_pixel_strategy_effective", balanced_pixel_strategy_effective)
            trial.set_user_attr("selection_split", "validation_batch_3")
            trial.set_user_attr("selection_strategy", "04B2_optuna_challenge")

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
    """
    Return a readable dataframe with flattened params and user attributes.

    Compatible with:
    - single-objective Optuna studies: trial.value
    - multi-objective Optuna studies: trial.values

    For the 04B2 Pareto objective, value columns are:
    - value_0 = fn_rate_max          minimize
    - value_1 = fp_rate_mean         minimize
    - value_2 = balanced_accuracy_mean maximize
    """
    rows = []

    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state).split(".")[-1],
        }

        values = getattr(trial, "values", None)

        if values is not None:
            for i, value in enumerate(values):
                row[f"value_{i}"] = value
            # Convenience aliases for the current 04B2 objective.
            if len(values) >= 1:
                row["objective_fn_rate_max"] = values[0]
            if len(values) >= 2:
                row["objective_fp_rate_mean"] = values[1]
            if len(values) >= 3:
                row["objective_balanced_accuracy_mean"] = values[2]
            row["value"] = np.nan
        else:
            # Single-objective fallback.
            try:
                row["value"] = trial.value
            except RuntimeError:
                row["value"] = np.nan

        for key, value in trial.params.items():
            row[key] = value
        for key, value in trial.user_attrs.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    df = pd.DataFrame(rows)

    # Sorting rule:
    # - multi-objective: FN first, FP second, BA third
    # - single-objective: value descending
    if {"value_0", "value_1", "value_2"}.issubset(df.columns):
        df = df.sort_values(
            ["state", "value_0", "value_1", "value_2"],
            ascending=[True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)
    elif "value" in df.columns:
        df = df.sort_values(
            "value",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return df


def best_completed_trial_row(trials_df: pd.DataFrame) -> pd.Series:
    """
    Return a representative best completed trial.

    For multi-objective trials, this returns the first trial after
    FN-first / FP-second / BA-third sorting. This is only a convenience
    helper; final model selection should use Pareto filtering.
    """
    completed = trials_df[
        trials_df["state"].astype(str).eq("COMPLETE")
    ].copy()

    if completed.empty:
        raise ValueError("No completed Optuna trial found.")

    if {"value_0", "value_1", "value_2"}.issubset(completed.columns):
        return completed.sort_values(["value_0", "value_1", "value_2"], ascending=[True, True, False]).iloc[0]
        
    if "value" in completed.columns:
        return completed.sort_values("value", ascending=False).iloc[0]

    raise ValueError("No objective value column found in trials_df.")


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
    preprocessing_name = str(best_row["preprocessing"])
    matrix_method = str(best_row["matrix_method"])
    matrix_family = row_str(
        best_row,
        "matrix_family",
        matrix_family_from_method(matrix_method),
    )
    preprocessing_configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)
    current_preprocessing_configs = _preprocessing_configs_for_family(
        preprocessing_configs_by_family,
        matrix_family,
    )
    preprocessing_steps = tuple(
        current_preprocessing_configs.get(
            preprocessing_name,
            tuple(parse_preprocessing_steps(row_value(best_row, "preprocessing_steps", preprocessing_name))),
        )
    )

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


def optuna_trials_to_candidate_configs(
    trials_df: pd.DataFrame,
    n_per_matrix_family: int = 8,
    n_overall: int = 20,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    selection_split: str = "validation_batch_3",
    selection_strategy: str = "04B2_optuna_challenge",
) -> pd.DataFrame:
    """
    Convert completed Optuna trials into selected SIMCA candidate configs.

    The output is compatible with refit_selected_simca_configs.
    """
    if trials_df is None or len(trials_df) == 0:
        return pd.DataFrame()

    df = trials_df.copy()

    if "state" in df.columns:
        df = df[df["state"].astype(str).eq("COMPLETE")].copy()

    has_multi_objective = {"value_0", "value_1", "value_2"}.issubset(df.columns)
    if has_multi_objective:
        df = df[pd.to_numeric(df["value_0"], errors="coerce").notna()].copy()
    elif "value" in df.columns:
        df = df[pd.to_numeric(df["value"], errors="coerce").notna()].copy()

    if df.empty:
        return pd.DataFrame()

    if has_multi_objective:
        value_0 = pd.to_numeric(df["value_0"], errors="coerce")
        value_1 = pd.to_numeric(df["value_1"], errors="coerce")
        value_2 = pd.to_numeric(df["value_2"], errors="coerce")
        df["optuna_value"] = -10.0 * value_0 - value_1 + value_2
    else:
        df["optuna_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["optuna_trial_number"] = df["number"].astype(int)

    if "score" in df.columns:
        df["selection_score"] = pd.to_numeric(df["score"], errors="coerce")
    else:
        df["selection_score"] = df["optuna_value"]

    def _fill_numeric(target: str, sources: Sequence[str]) -> None:
        if target in df.columns and pd.to_numeric(df[target], errors="coerce").notna().any():
            return
        values = pd.Series(np.nan, index=df.index, dtype="float64")
        for source in sources:
            if source not in df.columns:
                continue
            source_values = pd.to_numeric(df[source], errors="coerce")
            values = values.where(values.notna(), source_values)
        df[target] = values

    _fill_numeric("fn_rate", ("fn_rate_max", "objective_fn_rate_max", "value_0"))
    _fill_numeric("fp_rate", ("fp_rate_mean", "objective_fp_rate_mean", "value_1"))
    _fill_numeric(
        "balanced_accuracy",
        ("balanced_accuracy_mean", "objective_balanced_accuracy_mean", "value_2"),
    )

    if "target_class" not in df.columns:
        df["target_class"] = target_class
    else:
        df["target_class"] = df["target_class"].fillna(target_class)

    if "non_target_label" not in df.columns:
        df["non_target_label"] = non_target_label
    else:
        df["non_target_label"] = df["non_target_label"].fillna(non_target_label)

    if "model_family" not in df.columns:
        df["model_family"] = "standard_rule"
    else:
        df["model_family"] = df["model_family"].fillna("standard_rule")

    if "matrix_family" not in df.columns:
        df["matrix_family"] = df["matrix_method"].apply(matrix_family_from_method)

    if "m" not in df.columns:
        df["m"] = np.nan

    df["m"] = pd.to_numeric(df["m"], errors="coerce")

    if "balanced_pixel_strategy" not in df.columns:
        df["balanced_pixel_strategy"] = np.where(
            df["matrix_method"].astype(str).eq("balanced_pixels"),
            "random",
            "not_applicable",
        )

    if "balanced_pixel_strategy_effective" not in df.columns:
        df["balanced_pixel_strategy_effective"] = np.where(
            df["matrix_method"].astype(str).eq("balanced_pixels"),
            df["balanced_pixel_strategy"].astype(str),
            "random",
        )

    if "m_effective" not in df.columns:
        df["m_effective"] = np.where(
            df["matrix_method"].astype(str).eq("balanced_pixels"),
            df["m"].fillna(40).astype(int),
            40,
        )

    if "training_matrix_id" not in df.columns:
        df["training_matrix_id"] = np.where(
            df["matrix_method"].astype(str).eq("balanced_pixels"),
            (
                "balanced_pixel_"
                + df["balanced_pixel_strategy_effective"].astype(str)
                + "_m"
                + df["m_effective"].astype(int).astype(str)
            ),
            df["matrix_method"].astype(str),
        )

    if "rule_variant" not in df.columns:
        df["rule_variant"] = df["rule"].astype(str)

    if "preprocessing_steps" not in df.columns:
        df["preprocessing_steps"] = df["preprocessing"].astype(str)

    if "object_threshold" not in df.columns and "object_threshold_median" in df.columns:
        df["object_threshold"] = pd.to_numeric(
            df["object_threshold_median"],
            errors="coerce",
        )

    for col, default in {
        "sg_window_length": 11,
        "sg_polyorder": 2,
        "position_dilation_radius": 3,
        "alpha": 0.05,
        "object_threshold": 0.75,
    }.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df = ensure_candidate_columns(df)
    df = normalize_simca_rule_columns(df)
    df = fill_selected_config_defaults(
        df,
        default_values={
            "target_class": target_class,
            "non_target_label": non_target_label,
            "sg_window_length": 11,
            "sg_polyorder": 2,
            "position_dilation_radius": 3,
            "m": 40,
            "alpha": 0.05,
            "object_threshold": 0.75,
        },
    )
    df = add_detection_selection_score(df)
    df = add_reference_selection_scores(df)

    df["selection_split"] = selection_split
    df["selection_strategy"] = selection_strategy
    df["candidate_source"] = selection_strategy

    df = (
        df.sort_values(
            ["fn_rate", "fp_rate", "selection_score", "optuna_value"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )

    parts = []

    if "matrix_family" in df.columns and n_per_matrix_family:
        parts.append(
            df.groupby("matrix_family", group_keys=False, dropna=False)
            .head(int(n_per_matrix_family))
        )

    if n_overall:
        parts.append(df.head(int(n_overall)))

    if parts:
        selected = (
            pd.concat(parts, ignore_index=True, sort=False)
            .drop_duplicates(subset=["optuna_trial_number"])
            .reset_index(drop=True)
        )
    else:
        selected = df.copy()

    selected = sort_detection_selection(selected, add_score=False)
    selected = selected.reset_index(drop=True)

    selected["selected_config_id"] = [
        f"optuna_{int(row.optuna_trial_number):04d}"
        for row in selected.itertuples()
    ]

    keep_cols = [
        "selected_config_id",
        "selection_split",
        "selection_strategy",
        "candidate_source",
        "optuna_trial_number",
        "optuna_value",
        "value_0",
        "value_1",
        "value_2",
        "objective_fn_rate_max",
        "objective_fp_rate_mean",
        "objective_balanced_accuracy_mean",

        "model_family",
        "matrix_family",
        "training_matrix_id",
        "matrix_method",
        "balanced_pixel_strategy",
        "balanced_pixel_strategy_effective",
        "m",
        "m_effective",

        "preprocessing",
        "preprocessing_steps",

        "rule",
        "rule_variant",
        "selected_rule_name",
        "rule_for_refit",
        "limit_source",

        "target_class",
        "non_target_label",

        "n",
        "tp",
        "fn",
        "fp",
        "tn",
        "balanced_accuracy",
        "target_sensitivity",
        "non_target_specificity",
        "fn_rate",
        "fp_rate",
        "f1_score",
        "accuracy",
        "precision",

        "selection_score",
        "score_conservative_target",
        "score_balanced_reference",
        "score_specificity_control",

        "n_components",
        "alpha",
        "object_threshold",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",

        "n_train_observations",
        "n_projected_pixels",
    ]

    keep_cols = [col for col in keep_cols if col in selected.columns]

    return selected[keep_cols].copy()



def suggest_simca_config(
    trial,
    matrix_methods,
    preprocessing_configs,
    rule_variants,
    n_components_choices,
    alpha_choices,
    m_choices,
    balanced_pixel_strategy_choices,
    sg_window_choices,
    sg_polyorder_choices,
    position_dilation_radius_choices,
):
    matrix_method = trial.suggest_categorical("matrix_method", list(matrix_methods))
    matrix_family = matrix_family_from_method(matrix_method)
    preprocessing_configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)
    current_preprocessing_configs = _preprocessing_configs_for_family(
        preprocessing_configs_by_family,
        matrix_family,
    )
    if not current_preprocessing_configs:
        raise optuna.exceptions.TrialPruned()

    preprocessing_name = trial.suggest_categorical(
        _preprocessing_param_name(matrix_family, matrix_methods),
        list(current_preprocessing_configs.keys()),
    )
    preprocessing_steps = tuple(current_preprocessing_configs[preprocessing_name])

    rule_variant = trial.suggest_categorical("rule_variant", list(rule_variants))
    n_components = trial.suggest_categorical("n_components", list(n_components_choices))
    alpha = trial.suggest_categorical("alpha", list(alpha_choices))

    if matrix_method == "balanced_pixels":
        m = trial.suggest_categorical("m", list(m_choices))
        balanced_pixel_strategy = trial.suggest_categorical(
            "balanced_pixel_strategy",
            list(balanced_pixel_strategy_choices),
        )
    else:
        m = 40
        balanced_pixel_strategy = "random"

    uses_sg = any(str(step).startswith("sg_") for step in preprocessing_steps)

    if uses_sg:
        sg_window_length = trial.suggest_categorical("sg_window_length", list(sg_window_choices))
        sg_polyorder = trial.suggest_categorical("sg_polyorder", list(sg_polyorder_choices))

        if int(sg_window_length) % 2 == 0 or int(sg_polyorder) >= int(sg_window_length):
            raise optuna.exceptions.TrialPruned()
    else:
        sg_window_length = 11
        sg_polyorder = 2

    position_dilation_radius = trial.suggest_categorical(
        "position_dilation_radius",
        list(position_dilation_radius_choices),
    )

    return {
        "matrix_method": matrix_method,
        "matrix_family": matrix_family,
        "preprocessing": preprocessing_name,
        "preprocessing_steps": preprocessing_steps,
        "rule_variant": rule_variant,
        "n_components": int(n_components),
        "alpha": float(alpha),
        "m": int(m),
        "balanced_pixel_strategy": balanced_pixel_strategy,
        "sg_window_length": int(sg_window_length),
        "sg_polyorder": int(sg_polyorder),
        "position_dilation_radius": int(position_dilation_radius),
    }


def select_binary_threshold_pareto(
    threshold_df: pd.DataFrame,
    max_fn_rate: float = 0.00,
    max_fp_rate: float | None = None,
) -> pd.Series:
    df = threshold_df.copy()

    for col in ["fn_rate", "fp_rate", "balanced_accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feasible = df[df["fn_rate"].fillna(1.0) <= float(max_fn_rate)].copy()

    if max_fp_rate is not None and len(feasible) > 0:
        feasible = feasible[feasible["fp_rate"].fillna(1.0) <= float(max_fp_rate)].copy()

    if feasible.empty:
        feasible = df.copy()

    front = pareto_front_by_group(
        feasible,
        group_cols=[],
        minimize_cols=["fn_rate", "fp_rate"],
        maximize_cols=["balanced_accuracy"],
    )

    return (
        front.sort_values(
            ["fn_rate", "fp_rate", "balanced_accuracy"],
            ascending=[True, True, False],
        )
        .iloc[0]
    )


def evaluate_config_binary_multiseed(
    cfg: dict,
    object_db,
    image_db,
    train_filters,
    projection_filters,
    preprocessing_configs,
    object_thresholds,
    seeds=(0, 1, 2, 3, 4),
    max_fn_rate=0.00,
    max_fp_rate=None,
    wavelengths=None,
    target_class="peanut",
    non_target_label="non_target",
):
    rows = []

    for seed in seeds:
        summary_df, _, errors_df = run_simca_rule_variant_grid(
            object_db=object_db,
            image_db=image_db,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs={
                cfg["preprocessing"]: cfg["preprocessing_steps"]
            },
            matrix_methods=[cfg["matrix_method"]],
            rule_variants=[cfg["rule_variant"]],
            n_components_values=[cfg["n_components"]],
            alpha_values=[cfg["alpha"]],
            object_thresholds=object_thresholds,
            m_values=[cfg["m"]],
            random_state=int(seed),
            wavelengths=wavelengths,
            sg_window_length_values=[cfg["sg_window_length"]],
            sg_polyorder_values=[cfg["sg_polyorder"]],
            position_dilation_radius_values=[cfg["position_dilation_radius"]],
            balanced_pixel_strategy_values=[cfg["balanced_pixel_strategy"]],
            target_class=target_class,
            non_target_label=non_target_label,
            verbose=False,
        )

        if summary_df is None or len(summary_df) == 0:
            continue

        best = select_binary_threshold_pareto(
            summary_df,
            max_fn_rate=max_fn_rate,
            max_fp_rate=max_fp_rate,
        )

        row = best.to_dict()
        row["random_state"] = int(seed)
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid seed evaluation.")

    eval_df = pd.DataFrame(rows)

    summary = {
        "fn_rate_mean": eval_df["fn_rate"].mean(),
        "fn_rate_max": eval_df["fn_rate"].max(),
        "fn_rate_std": eval_df["fn_rate"].std(ddof=0),
        "fp_rate_mean": eval_df["fp_rate"].mean(),
        "fp_rate_max": eval_df["fp_rate"].max(),
        "fp_rate_std": eval_df["fp_rate"].std(ddof=0),
        "balanced_accuracy_mean": eval_df["balanced_accuracy"].mean(),
        "object_threshold_median": eval_df["object_threshold"].median(),
    }

    return eval_df, summary


def make_optuna_binary_pareto_objective(
    object_db,
    image_db,
    train_filters,
    projection_filters,
    preprocessing_configs,
    matrix_methods,
    rule_variants,
    object_thresholds,
    seeds=(0, 1, 2),
    max_fn_rate=0.00,
    max_fp_rate=None,
    wavelengths=None,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    **search_space_kwargs,
):
    optuna = _require_optuna()
    preprocessing_configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)

    def objective(trial):
        cfg = suggest_simca_config(
            trial=trial,
            matrix_methods=matrix_methods,
            preprocessing_configs=preprocessing_configs_by_family,
            rule_variants=rule_variants,
            **search_space_kwargs,
        )

        eval_df, summary = evaluate_config_binary_multiseed(
            cfg=cfg,
            object_db=object_db,
            image_db=image_db,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            object_thresholds=object_thresholds,
            seeds=seeds,
            max_fn_rate=max_fn_rate,
            max_fp_rate=max_fp_rate,
            wavelengths=wavelengths,
            target_class=target_class,
            non_target_label=non_target_label,
        )

        for key, value in cfg.items():
            if key != "preprocessing_steps":
                trial.set_user_attr(key, value)
        trial.set_user_attr("preprocessing_steps", "+".join(cfg["preprocessing_steps"]))

        for key, value in summary.items():
            trial.set_user_attr(key, float(value))

        trial.set_user_attr("selection_strategy", "optuna_binary_pareto")
        trial.set_user_attr("model_family", "rule_variant_grid")

        return (
            float(summary["fn_rate_max"]),
            float(summary["fp_rate_mean"]),
            float(summary["balanced_accuracy_mean"]),
        )

    return objective
