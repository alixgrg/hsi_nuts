from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from src import experiment_config as expcfg
from src.decision.labels import DEFAULT_NON_TARGET_LABEL, DEFAULT_TARGET_CLASS
from src.utils import parse_preprocessing_steps, row_str, row_value
from src.workflows.simca_selection_utils import (
    ensure_candidate_columns,
    fill_selected_config_defaults,
    materialize_selection_metrics,
    normalize_simca_rule_columns,
    pareto_front_by_group,
    sort_detection_selection,
)
from src.workflows.simca import (
    _normalize_preprocessing_configs_by_family,
    _preprocessing_configs_for_family,
    make_target_train_filters,
    run_single_simca_pixel_projection,
    run_simca_rule_variant_grid,
)
from src.workflows.simca_internal_calibration import (
    validate_simca_configuration,
)
from src.matrices.matrix_registry import matrix_family_from_method

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
    min_target_sensitivity: float | None = 0.5,
    min_non_target_specificity: float | None = 0.1,
    balanced_pixel_strategy_choices: Sequence[str] = ("random", "center"),
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
):
    """Create a score-free, three-objective SIMCA Optuna objective."""
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

            metrics = materialize_selection_metrics(
                threshold_df.iloc[[0]],
                keep_source_columns=False,
            )
            row = metrics.iloc[0].to_dict()
            fn_rate = float(row["fn_rate"])
            fp_rate = float(row["fp_rate"])
            balanced_accuracy = float(row["balanced_accuracy"])
            if not np.isfinite(
                [fn_rate, fp_rate, balanced_accuracy]
            ).all():
                raise optuna.exceptions.TrialPruned()

            sensitivity = float(row["target_sensitivity"])
            specificity = float(row["non_target_specificity"])
            if (
                min_target_sensitivity is not None
                and sensitivity < float(min_target_sensitivity)
            ):
                trial.set_user_attr(
                    "prune_reason",
                    "min_target_sensitivity_not_met",
                )
                raise optuna.exceptions.TrialPruned()
            if (
                min_non_target_specificity is not None
                and specificity < float(min_non_target_specificity)
            ):
                trial.set_user_attr(
                    "prune_reason",
                    "min_non_target_specificity_not_met",
                )
                raise optuna.exceptions.TrialPruned()

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

            return fn_rate, fp_rate, balanced_accuracy

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
            directions=("minimize", "minimize", "maximize"),
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
    """Return every Optuna trial with timing and fixed-domain provenance."""
    rows = []

    for trial in study.trials:
        started = getattr(trial, "datetime_start", None)
        completed = getattr(trial, "datetime_complete", None)
        duration = getattr(trial, "duration", None)
        row = {
            "number": trial.number,
            "trial_number": trial.number,
            "study_name": str(getattr(study, "study_name", "")),
            "state": str(trial.state).split(".")[-1],
            "datetime_start_utc": (
                started.isoformat() if started is not None else ""
            ),
            "datetime_complete_utc": (
                completed.isoformat() if completed is not None else ""
            ),
            "duration_seconds": (
                float(duration.total_seconds()) if duration is not None else np.nan
            ),
        }

        values = getattr(trial, "values", None)

        if values is not None:
            row["objective_values_json"] = json.dumps(
                [
                    None if value is None else float(value)
                    for value in values
                ]
            )
            for i, value in enumerate(values):
                row[f"value_{i}"] = value
            row["value"] = np.nan
        else:
            row["objective_values_json"] = "[]"
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
    if df.empty:
        return df
    if "status" not in df:
        df["status"] = df["state"].astype(str).str.lower()
    domain_ids = df.get(
        "domain_config_id", pd.Series(pd.NA, index=df.index, dtype="string")
    ).astype("string")
    has_domain_id = domain_ids.notna() & domain_ids.ne("")
    df["is_duplicate_domain_config"] = False
    df.loc[has_domain_id, "is_duplicate_domain_config"] = domain_ids.loc[
        has_domain_id
    ].duplicated(keep="first")
    first_trial = (
        df.loc[has_domain_id]
        .groupby(domain_ids.loc[has_domain_id], sort=False)["trial_number"]
        .transform("min")
    )
    df["duplicate_of_trial_number"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    duplicate_mask = df["is_duplicate_domain_config"].astype(bool)
    df.loc[duplicate_mask, "duplicate_of_trial_number"] = first_trial.loc[
        duplicate_mask
    ].astype("Int64")
    return df.sort_values("trial_number", kind="mergesort").reset_index(drop=True)


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

    objective_columns = {"value_0", "value_1", "value_2"}
    if not objective_columns.issubset(completed.columns):
        raise ValueError(
            "Score-free selection requires three Optuna objective columns."
        )
    return completed.sort_values(
        ["value_0", "value_1", "value_2"],
        ascending=[True, True, False],
        kind="mergesort",
    ).iloc[0]


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

    objective_columns = {"value_0", "value_1", "value_2"}
    if not objective_columns.issubset(df.columns):
        raise ValueError(
            "Score-free candidate conversion requires a three-objective study."
        )
    df = df[pd.to_numeric(df["value_0"], errors="coerce").notna()].copy()

    if df.empty:
        return pd.DataFrame()

    df["optuna_trial_number"] = df["number"].astype(int)

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
    df = materialize_selection_metrics(
        df,
        keep_source_columns=False,
    )

    df["selection_split"] = selection_split
    df["selection_strategy"] = selection_strategy
    df["candidate_source"] = selection_strategy

    df = sort_detection_selection(
        df,
        materialize_metrics=False,
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

    selected = sort_detection_selection(
        selected,
        materialize_metrics=False,
    )
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
    matrix_methods=None,
    preprocessing_configs=None,
    rule_variants=None,
    n_components_choices=None,
    alpha_choices=None,
    m_choices=None,
    balanced_pixel_strategy_choices=None,
    sg_window_choices=None,
    sg_polyorder_choices=None,
    position_dilation_radius_choices=None,
    allowed_domain: pd.DataFrame | None = None,
):
    if allowed_domain is not None:
        if allowed_domain.empty:
            raise ValueError("allowed_domain cannot be empty.")
        if "domain_config_id" not in allowed_domain:
            raise KeyError("allowed_domain is missing domain_config_id.")
        choices = (
            allowed_domain.drop_duplicates("domain_config_id")
            .reset_index(drop=True)
        )
        domain_config_id = trial.suggest_categorical(
            "domain_config_id",
            choices["domain_config_id"].astype(str).tolist(),
        )
        config = choices.loc[
            choices["domain_config_id"].astype(str).eq(
                str(domain_config_id)
            )
        ].iloc[0].to_dict()
        config["preprocessing_steps"] = tuple(
            parse_preprocessing_steps(
                config.get(
                    "preprocessing_steps",
                    config.get("preprocessing", "raw"),
                )
            )
        )
        return config

    required_spaces = {
        "matrix_methods": matrix_methods,
        "preprocessing_configs": preprocessing_configs,
        "rule_variants": rule_variants,
        "n_components_choices": n_components_choices,
        "alpha_choices": alpha_choices,
        "m_choices": m_choices,
        "balanced_pixel_strategy_choices": (
            balanced_pixel_strategy_choices
        ),
        "sg_window_choices": sg_window_choices,
        "sg_polyorder_choices": sg_polyorder_choices,
        "position_dilation_radius_choices": (
            position_dilation_radius_choices
        ),
    }
    missing_spaces = [
        name for name, values in required_spaces.items() if values is None
    ]
    if missing_spaces:
        raise ValueError(
            f"Missing static Optuna search spaces: {missing_spaces}"
        )
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
) -> pd.Series | dict[str, object]:
    df = threshold_df.copy()

    for col in ["fn_rate", "fp_rate", "balanced_accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feasible = df[df["fn_rate"].fillna(1.0) <= float(max_fn_rate)].copy()

    if max_fp_rate is not None and len(feasible) > 0:
        feasible = feasible[feasible["fp_rate"].fillna(1.0) <= float(max_fp_rate)].copy()

    if feasible.empty:
        return {
            "status": "calculable_but_not_acceptable",
            "selected_threshold": None,
            "metrics": None,
        }

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
    calibration_folds: pd.DataFrame | None,
    allowed_domain: pd.DataFrame,
    decision_mode: str | None = None,
    seeds: Sequence[int] = expcfg.SIMCA_SEARCH_RANDOM_SEEDS,
    constraints: Mapping[str, float] | None = None,
    precomputed_metrics: pd.DataFrame | None = None,
    wavelengths=None,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
):
    if allowed_domain is None or allowed_domain.empty:
        raise ValueError("allowed_domain is required for internal Optuna.")
    domain_id = str(cfg["domain_config_id"])
    calibration_id = str(cfg.get("calibration_id", ""))
    runtime = allowed_domain.loc[
        allowed_domain["domain_config_id"].astype(str).eq(domain_id)
    ].copy()
    if runtime.empty:
        raise KeyError(f"Unknown domain_config_id: {domain_id}")
    active_seeds = set(map(int, seeds))
    if "random_states" in runtime.columns:
        configured_seeds = set()
        for value in runtime["random_states"]:
            configured_seeds.update(map(int, json.loads(str(value))))
        if not configured_seeds.intersection(active_seeds):
            runtime = runtime.iloc[0:0]
    elif "random_state" in runtime.columns:
        runtime = runtime.loc[
            pd.to_numeric(runtime["random_state"], errors="coerce")
            .astype(int)
            .isin(active_seeds)
        ]
    if runtime.empty:
        raise RuntimeError("No configured evaluation seed remains.")
    mode = str(
        decision_mode
        if decision_mode is not None
        else runtime["decision_mode"].iloc[0]
    ).lower()
    runtime = runtime.loc[runtime["decision_mode"].eq(mode)].copy()
    summary = {
        "domain_config_id": domain_id,
        "calibration_id": calibration_id,
        "decision_mode": mode,
        "status": "acceptable",
        "technical_error": "",
    }
    evaluation_track = str(
        cfg.get(
            "evaluation_track",
            runtime["evaluation_track"].iloc[0]
            if "evaluation_track" in runtime.columns
            else "",
        )
    )
    # Active 8-track path: 04A is the authoritative evaluator. Preserve its
    # three independent statuses and never recompute constraints in 04B.
    if precomputed_metrics is not None and evaluation_track:
        identifier_column = (
            "calibration_id"
            if calibration_id and "calibration_id" in precomputed_metrics.columns
            else "domain_config_id"
        )
        identifier_value = calibration_id if identifier_column == "calibration_id" else domain_id
        matching = precomputed_metrics.loc[
            precomputed_metrics[identifier_column].astype(str).eq(identifier_value)
            & precomputed_metrics["evaluation_track"].astype(str).eq(evaluation_track)
        ]
        if len(matching) != 1:
            summary.update(
                {
                    "status": "fit_or_projection_error",
                    "technical_error": "missing_or_duplicated_precomputed_metric",
                    "evaluation_track": evaluation_track,
                }
            )
            return pd.DataFrame(), summary
        summary.update(matching.iloc[0].to_dict())
        summary["evaluation_track"] = evaluation_track
        technical_status = str(summary.get("technical_status", "calculable"))
        acceptability_status = str(
            summary.get("acceptability_status", "acceptable")
        )
        if technical_status != "calculable":
            summary["status"] = "technical_invalid"
            summary["technical_error"] = str(
                summary.get("failure_reason", technical_status)
            )
        elif acceptability_status == "acceptable":
            summary["status"] = "acceptable"
        else:
            summary["status"] = "calculable_but_not_acceptable"
        return pd.DataFrame(), summary
    if precomputed_metrics is not None:
        if calibration_id and "calibration_id" in precomputed_metrics.columns:
            identifier_mask = precomputed_metrics["calibration_id"].astype(str).eq(
                calibration_id
            )
        else:
            identifier_mask = precomputed_metrics["domain_config_id"].astype(str).eq(
                domain_id
            )
        matching = precomputed_metrics.loc[
            identifier_mask
            & precomputed_metrics["decision_mode"].astype(str).eq(mode)
        ]
        if len(matching) != 1:
            summary["status"] = "fit_or_projection_error"
            summary["technical_error"] = (
                "missing_or_duplicated_precomputed_metric"
            )
            return pd.DataFrame(), summary
        aggregated = matching.iloc[0].to_dict()
        summary.update(aggregated)
        # 04A v3 names expose the scientific meaning directly.  These aliases
        # keep the budget benchmark API stable without duplicating metrics in
        # the persisted exhaustive table.
        aliases = {
            "fn_rate_max": "worst_fold_target_miss_rate",
            "fp_rate_max": "worst_fold_false_accept_rate",
            "uncertain_rate_max": "uncertain_rate",
            "coverage_rate_mean": "coverage_rate",
            "balanced_accuracy_mean": (
                "decided_balanced_accuracy" if mode == "3way" else "balanced_accuracy"
            ),
        }
        for legacy_name, source_name in aliases.items():
            if legacy_name not in summary and source_name in summary:
                summary[legacy_name] = summary[source_name]
        if "acceptability_status" in summary:
            summary["status"] = (
                "acceptable"
                if str(summary["acceptability_status"]) == "acceptable"
                else "calculable_but_not_acceptable"
            )
        eval_df = pd.DataFrame()
    else:
        raise ValueError(
            "precomputed_metrics is required: notebook 04A is the "
            "authoritative evaluator for the active eight-track protocol."
        )

    summary["coverage_mean"] = float(
        summary.get("coverage_rate_mean", np.nan)
    )
    summary["decided_balanced_accuracy_mean"] = float(
        summary.get("decided_balanced_accuracy", summary["balanced_accuracy_mean"])
    )
    active_constraints = dict(
        expcfg.SIMCA_SEARCH_CONSTRAINTS[mode]
        if constraints is None
        else constraints
    )
    acceptable = (
        summary["fn_rate_max"]
        <= float(active_constraints["max_fn_rate"])
        and summary["fp_rate_max"]
        <= float(active_constraints["max_fp_rate"])
        and summary["balanced_accuracy_mean"]
        >= float(active_constraints["min_balanced_accuracy"])
        and summary["fold_metric_std"]
        <= float(active_constraints["max_fold_metric_std"])
    )
    if mode == "3way":
        acceptable = (
            acceptable
            and summary["uncertain_rate_max"]
            <= float(active_constraints["max_uncertain_rate"])
            and summary["coverage_mean"]
            >= float(active_constraints["min_coverage"])
        )
    if not acceptable:
        summary["status"] = "calculable_but_not_acceptable"
    return eval_df, summary


def make_optuna_binary_pareto_objective(
    object_db,
    image_db,
    allowed_domain: pd.DataFrame,
    calibration_folds: pd.DataFrame | None,
    decision_mode: str,
    constraints: Mapping[str, float] | None = None,
    seeds: Sequence[int] = expcfg.SIMCA_SEARCH_RANDOM_SEEDS,
    precomputed_metrics: pd.DataFrame | None = None,
    wavelengths=None,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    evaluation_track: str | None = None,
    study_scope: str = "protocol",
    study_seed: int | None = None,
    search_plan_hash: str = "",
):
    optuna = _require_optuna()
    mode = str(decision_mode).lower()
    if mode not in {"2way", "3way"}:
        raise ValueError("decision_mode must be '2way' or '3way'.")
    domain = allowed_domain.loc[
        allowed_domain["decision_mode"].eq(mode)
    ].copy()
    if domain.empty:
        raise ValueError(f"No allowed-domain rows for {mode}.")
    evaluation_cache: dict[str, dict] = {}

    if evaluation_track is not None:
        track = str(evaluation_track)
        if track not in expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS:
            raise KeyError(f"Unknown evaluation_track: {track!r}")
        domain = domain.loc[
            domain["evaluation_track"].astype(str).eq(track)
        ].copy()
        if domain.empty:
            raise ValueError(f"No allowed-domain rows for {track!r}.")
        objective_names = tuple(
            expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS[track]["objective_names"]
        )

        def objective_8tracks(trial):
            cfg = suggest_simca_config(trial=trial, allowed_domain=domain)
            domain_id = str(cfg["domain_config_id"])
            calibration_id = str(cfg.get("calibration_id", domain_id))
            if calibration_id in evaluation_cache:
                summary = dict(evaluation_cache[calibration_id])
                cache_hit = True
            else:
                try:
                    _, summary = evaluate_config_binary_multiseed(
                        cfg=cfg,
                        object_db=None,
                        image_db=None,
                        calibration_folds=None,
                        allowed_domain=domain,
                        decision_mode=mode,
                        seeds=seeds,
                        constraints=constraints,
                        precomputed_metrics=precomputed_metrics,
                        wavelengths=None,
                        target_class=target_class,
                        non_target_label=non_target_label,
                    )
                except Exception as exc:
                    trial.set_user_attr("status", "fit_or_projection_error")
                    trial.set_user_attr("prune_reason", "metric_lookup_error")
                    trial.set_user_attr("error_type", type(exc).__name__)
                    trial.set_user_attr("error_message", str(exc))
                    raise optuna.TrialPruned() from exc
                evaluation_cache[calibration_id] = dict(summary)
                cache_hit = False

            attrs = {
                "calibration_id": calibration_id,
                "evaluation_track": track,
                "track_id": str(cfg.get("track_id", "")),
                "decision_mode": mode,
                "study_scope": str(study_scope),
                "study_seed": study_seed,
                "eligibility_status": str(cfg.get("eligibility_status", summary.get("eligibility_status", ""))),
                "status": str(summary.get("status", "fit_or_projection_error")),
                "evaluation_cache_hit": bool(cache_hit),
                "evaluation_source": "04A_exact_internal_metrics",
                "search_plan_hash": str(search_plan_hash),
                "objective_names_json": json.dumps(objective_names),
            }
            for key, value in attrs.items():
                trial.set_user_attr(key, value)

            if attrs["status"] in expcfg.SIMCA_OPTUNA_TECHNICAL_PRUNE_STATUSES:
                trial.set_user_attr("prune_reason", attrs["status"])
                trial.set_user_attr(
                    "error_message", str(summary.get("technical_error", ""))
                )
                raise optuna.TrialPruned()

            values = np.asarray(
                [pd.to_numeric(summary.get(name), errors="coerce") for name in objective_names],
                dtype=float,
            )
            if not np.isfinite(values).all():
                trial.set_user_attr("status", "non_finite_objective")
                trial.set_user_attr("prune_reason", "non_finite_objective")
                trial.set_user_attr(
                    "error_message",
                    json.dumps(
                        {
                            name: summary.get(name)
                            for name in objective_names
                            if not np.isfinite(
                                pd.to_numeric(summary.get(name), errors="coerce")
                            )
                        },
                        default=str,
                    ),
                )
                raise optuna.TrialPruned()
            trial.set_user_attr("prune_reason", "")
            trial.set_user_attr("error_type", "")
            trial.set_user_attr("error_message", "")
            return tuple(map(float, values))

        return objective_8tracks

    def objective(trial):
        cfg = suggest_simca_config(
            trial=trial,
            allowed_domain=domain,
        )
        validation = validate_simca_configuration(cfg)
        if not validation["is_valid"]:
            trial.set_user_attr("status", "technical_invalid")
            trial.set_user_attr(
                "technical_error",
                json.dumps(validation["technical_errors"]),
            )
            raise optuna.TrialPruned()
        domain_id = str(cfg["domain_config_id"])
        if domain_id in evaluation_cache:
            summary = dict(evaluation_cache[domain_id])
            trial.set_user_attr("evaluation_cache_hit", True)
        else:
            try:
                _, summary = evaluate_config_binary_multiseed(
                    cfg=cfg,
                    object_db=object_db,
                    image_db=image_db,
                    calibration_folds=calibration_folds,
                    allowed_domain=domain,
                    decision_mode=mode,
                    seeds=seeds,
                    constraints=constraints,
                    precomputed_metrics=precomputed_metrics,
                    wavelengths=wavelengths,
                    target_class=target_class,
                    non_target_label=non_target_label,
                )
            except Exception as exc:
                trial.set_user_attr("status", "fit_or_projection_error")
                trial.set_user_attr("error_type", type(exc).__name__)
                trial.set_user_attr("error_message", str(exc))
                raise optuna.TrialPruned() from exc
            evaluation_cache[domain_id] = dict(summary)
            trial.set_user_attr("evaluation_cache_hit", False)
        for key, value in cfg.items():
            if key in {
                "data_config_id",
                "sampling_group_id",
                "fit_config_id",
                "config_id",
            }:
                continue
            if key == "preprocessing_steps":
                value = "+".join(parse_preprocessing_steps(value))
            if isinstance(value, np.generic):
                value = value.item()
            if value is None or (
                isinstance(value, float) and not np.isfinite(value)
            ):
                value = None
            trial.set_user_attr(key, value)
        for key, value in summary.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            trial.set_user_attr(key, value)
        trial.set_user_attr(
            "selection_strategy",
            "optuna_internal_calibration_fixed_thresholds",
        )
        trial.set_user_attr(
            "evaluation_source",
            (
                "04A_exact_internal_metrics"
                if precomputed_metrics is not None
                else "direct_internal_calibration"
            ),
        )
        if summary["status"] != "acceptable":
            raise optuna.TrialPruned()
        if mode == "2way":
            return (
                float(summary["fn_rate_max"]),
                float(summary["fp_rate_max"]),
                -float(summary["balanced_accuracy_mean"]),
                float(summary["fold_metric_std"]),
            )
        return (
            float(summary["fn_rate_max"]),
            float(summary["fp_rate_max"]),
            float(summary["uncertain_rate_max"]),
            -float(summary["coverage_mean"]),
            -float(summary["decided_balanced_accuracy_mean"]),
            float(summary["fold_metric_std"]),
        )

    return objective


def build_optuna_search_efficiency_audit(
    exhaustive_metrics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    pareto_reference: pd.DataFrame | None = None,
    study_registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare budgeted Optuna coverage with the exhaustive 04A Pareto set."""
    schema = expcfg.SIMCA_OPTUNA_SEARCH_EFFICIENCY_COLUMNS
    if exhaustive_metrics is None or exhaustive_metrics.empty:
        if study_registry is None or study_registry.empty:
            return pd.DataFrame(columns=schema)

    if pareto_reference is not None and study_registry is not None:
        domain_df = exhaustive_metrics.copy()
        reference_df = pareto_reference.copy()
        if "row_type" in reference_df:
            reference_df = reference_df.loc[
                reference_df["row_type"].astype(str).eq("configuration")
            ].copy()
        rows: list[dict[str, object]] = []
        for registry_row in study_registry.itertuples(index=False):
            track = str(registry_row.evaluation_track)
            track_domain = domain_df.loc[
                domain_df["evaluation_track"].astype(str).eq(track)
            ].copy()
            track_trials = optuna_trials.loc[
                optuna_trials["evaluation_track"].astype(str).eq(track)
            ].copy() if not optuna_trials.empty else pd.DataFrame()
            track_reference = reference_df.loc[
                reference_df["evaluation_track"].astype(str).eq(track)
            ].copy()
            reference_scope = (
                "diagnostic_pareto_front"
                if str(registry_row.study_scope) == "diagnostic_only"
                else "protocol_pareto_front"
            )
            reference_ids = set(
                track_reference.loc[
                    track_reference.get(
                        reference_scope,
                        pd.Series(False, index=track_reference.index),
                    ).fillna(False).astype(bool),
                    "calibration_id",
                ].dropna().astype(str)
            )
            if track_trials.empty:
                sampled = track_trials
                complete = track_trials
                sampled_ids: set[str] = set()
                completed_ids: set[str] = set()
            else:
                sampled = track_trials.loc[
                    track_trials["domain_config_id"].notna()
                    & track_trials["domain_config_id"].astype(str).ne("")
                ].copy()
                complete = track_trials.loc[
                    track_trials["state"].astype(str).eq("COMPLETE")
                ].copy()
                sampled_ids = set(sampled["domain_config_id"].astype(str))
                completed_ids = set(
                    complete["calibration_id"].dropna().astype(str)
                )
            n_domain = int(track_domain["domain_config_id"].nunique())
            n_sampled_trials = int(len(sampled))
            n_unique_sampled = len(sampled_ids)
            n_trials = int(len(track_trials))
            n_complete = int(len(complete))
            n_pruned = int(
                track_trials["state"].astype(str).eq("PRUNED").sum()
            ) if not track_trials.empty else 0
            technical_statuses = set(expcfg.SIMCA_OPTUNA_TECHNICAL_PRUNE_STATUSES)
            n_technical_errors = int(
                track_trials["status"].astype(str).isin(technical_statuses).sum()
            ) if not track_trials.empty else 0
            recovered = len(reference_ids.intersection(completed_ids))
            recall = (
                float(recovered / len(reference_ids))
                if reference_ids
                else np.nan
            )
            uniform = (
                float(1.0 - (1.0 - 1.0 / n_domain) ** n_sampled_trials)
                if n_domain > 0 and n_sampled_trials > 0
                else np.nan
            )
            delta = (
                float(recall - uniform)
                if np.isfinite(recall) and np.isfinite(uniform)
                else np.nan
            )
            lift = (
                float(recall / uniform)
                if np.isfinite(recall) and np.isfinite(uniform) and uniform > 0
                else np.nan
            )
            if not reference_ids or n_domain == 0:
                budget_status = "not_estimable"
                conclusion = "not_estimable"
            elif recall < float(expcfg.SIMCA_OPTUNA_MIN_PARETO_RECALL):
                budget_status = "insufficient"
                conclusion = "insufficient"
            else:
                budget_status = "sufficient"
                tolerance = float(
                    expcfg.SIMCA_OPTUNA_UNIFORM_RECALL_DELTA_TOLERANCE
                )
                if delta > tolerance:
                    conclusion = "useful"
                elif delta < -tolerance:
                    conclusion = "insufficient"
                else:
                    conclusion = "neutral"
            rows.append(
                {
                    "evaluation_track": track,
                    "track_id": str(registry_row.track_id),
                    "decision_mode": str(registry_row.decision_mode),
                    "study_name": str(registry_row.study_name),
                    "study_scope": str(registry_row.study_scope),
                    "eligibility_status": str(registry_row.eligibility_status),
                    "study_status": str(registry_row.study_status),
                    "n_domain_configurations": n_domain,
                    "trial_budget": int(registry_row.trial_budget),
                    "n_trials": n_trials,
                    "n_complete_trials": n_complete,
                    "n_pruned_trials": n_pruned,
                    "n_technical_errors": n_technical_errors,
                    "n_unique_configurations_sampled": n_unique_sampled,
                    "duplicate_trial_rate": (
                        float(1.0 - n_unique_sampled / n_sampled_trials)
                        if n_sampled_trials
                        else np.nan
                    ),
                    "domain_coverage_rate": (
                        float(n_unique_sampled / n_domain) if n_domain else np.nan
                    ),
                    "pareto_reference_scope": reference_scope,
                    "n_exhaustive_pareto_configurations": len(reference_ids),
                    "n_exhaustive_pareto_recovered": recovered,
                    "exhaustive_pareto_recall": recall,
                    "uniform_recall_expectation": uniform,
                    "pareto_recall_delta_vs_uniform": delta,
                    "pareto_recall_lift_vs_uniform": lift,
                    "budget_status": budget_status,
                    "optuna_conclusion": conclusion,
                    "exhaustive_reference_retained": True,
                }
            )
        return pd.DataFrame(rows).reindex(columns=schema)

    exhaustive_metrics = exhaustive_metrics.copy()
    if "evaluation_track" in exhaustive_metrics.columns:
        exhaustive_metrics["calibration_track"] = exhaustive_metrics[
            "evaluation_track"
        ]
    if "calibration_id" in exhaustive_metrics.columns:
        exhaustive_metrics["domain_config_id"] = exhaustive_metrics[
            "calibration_id"
        ]
    aliases = {
        "status": "acceptability_status",
        "fn_rate_max": "worst_fold_target_miss_rate",
        "fp_rate_max": "worst_fold_false_accept_rate",
        "balanced_accuracy_mean": "balanced_accuracy",
        "uncertain_rate_max": "uncertain_rate",
        "coverage_rate_mean": "coverage_rate",
    }
    for legacy_name, source_name in aliases.items():
        if legacy_name not in exhaustive_metrics and source_name in exhaustive_metrics:
            exhaustive_metrics[legacy_name] = exhaustive_metrics[source_name]
    required_grid = {
        "domain_config_id",
        "calibration_track",
        "decision_mode",
        "status",
        "fn_rate_max",
        "fp_rate_max",
        "balanced_accuracy_mean",
        "fold_metric_std",
    }
    missing = sorted(required_grid.difference(exhaustive_metrics.columns))
    if missing:
        raise KeyError(
            "Exhaustive metrics are missing Optuna audit columns: "
            f"{missing}"
        )

    rows: list[dict[str, object]] = []
    for track, domain in exhaustive_metrics.groupby(
        "calibration_track",
        sort=True,
    ):
        modes = domain["decision_mode"].dropna().astype(str).unique()
        if len(modes) != 1:
            raise RuntimeError(
                f"Calibration track {track!r} mixes decision modes: {modes}"
            )
        mode = str(modes[0]).lower()
        acceptable = domain.loc[
            domain["status"].astype(str).eq("acceptable")
        ].copy()
        if str(track) in expcfg.SIMCA_EVALUATION_TRACK_SPECS:
            track_spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[str(track)]
            minimize = list(track_spec["pareto_minimize"])
            maximize = list(track_spec["pareto_maximize"])
        elif mode == "3way":
            required = {"uncertain_rate_max", "coverage_rate_mean"}
            missing_mode = sorted(required.difference(acceptable.columns))
            if missing_mode:
                raise KeyError(
                    f"3-way exhaustive metrics are missing: {missing_mode}"
                )
            minimize = [
                "fn_rate_max",
                "fp_rate_max",
                "uncertain_rate_max",
                "fold_metric_std",
            ]
            maximize = [
                "coverage_rate_mean",
                "balanced_accuracy_mean",
            ]
        else:
            minimize = [
                "fn_rate_max",
                "fp_rate_max",
                "fold_metric_std",
            ]
            maximize = ["balanced_accuracy_mean"]

        if acceptable.empty:
            exhaustive_pareto = acceptable
        else:
            exhaustive_pareto = pareto_front_by_group(
                acceptable,
                group_cols=[],
                minimize_cols=minimize,
                maximize_cols=maximize,
            )
        track_trials = optuna_trials.loc[
            optuna_trials["calibration_track"].astype(str).eq(str(track))
        ].copy()
        trial_id_column = (
            "calibration_id"
            if "calibration_id" in track_trials.columns
            else "domain_config_id"
        )
        evaluated_ids = set(
            track_trials[trial_id_column].dropna().astype(str)
        )
        pareto_ids = set(
            exhaustive_pareto["domain_config_id"].dropna().astype(str)
        )
        recovered = len(evaluated_ids.intersection(pareto_ids))
        n_domain = int(domain["domain_config_id"].nunique())
        n_unique = len(evaluated_ids)
        n_trials = int(len(track_trials))
        domain_coverage = (
            float(n_unique / n_domain) if n_domain else np.nan
        )
        pareto_recall = (
            float(recovered / len(pareto_ids))
            if pareto_ids
            else np.nan
        )
        uniform_expectation = domain_coverage
        lift = (
            float(pareto_recall / uniform_expectation)
            if np.isfinite(pareto_recall)
            and np.isfinite(uniform_expectation)
            and uniform_expectation > 0
            else np.nan
        )
        if not np.isfinite(lift):
            interpretation = "not_estimable"
        elif lift > 1.10:
            interpretation = "pareto_recall_above_uniform_expectation"
        elif lift < 0.90:
            interpretation = "pareto_recall_below_uniform_expectation"
        else:
            interpretation = "pareto_recall_near_uniform_expectation"
        rows.append(
            {
                "calibration_track": track,
                "decision_mode": mode,
                "n_domain_configurations": n_domain,
                "n_acceptable_domain_configurations": int(
                    acceptable["domain_config_id"].nunique()
                ),
                "n_trials": n_trials,
                "n_unique_configurations_evaluated": n_unique,
                "duplicate_trial_rate": (
                    float(1.0 - n_unique / n_trials)
                    if n_trials
                    else np.nan
                ),
                "domain_coverage_rate": domain_coverage,
                "n_exhaustive_pareto_configurations": len(pareto_ids),
                "n_exhaustive_pareto_recovered": recovered,
                "exhaustive_pareto_recall": pareto_recall,
                "uniform_recall_expectation": uniform_expectation,
                "pareto_recall_lift_vs_uniform": lift,
                "benchmark_interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows).loc[:, list(schema)]


def build_optuna_search_plan_hash(
    domain_configurations: pd.DataFrame,
    source_hashes: Mapping[str, str] | None = None,
) -> str:
    """Hash the immutable 04B domain, objectives, sampler and budget."""
    if domain_configurations is None:
        raise ValueError("domain_configurations is required.")
    domain_columns = [
        column
        for column in (
            "domain_config_id",
            "calibration_id",
            "evaluation_track",
            "decision_mode",
        )
        if column in domain_configurations.columns
    ]
    records = (
        domain_configurations.loc[:, domain_columns]
        .fillna("")
        .astype(str)
        .sort_values(domain_columns, kind="mergesort")
        .to_dict(orient="records")
    ) if domain_columns else []
    payload = {
        "domain": records,
        "source_hashes": dict(sorted((source_hashes or {}).items())),
        "objectives": expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS,
        "sampler": {
            "name": expcfg.SIMCA_OPTUNA_SAMPLER_NAME,
            "multivariate": expcfg.SIMCA_OPTUNA_SAMPLER_MULTIVARIATE,
            "n_startup_trials": expcfg.SIMCA_OPTUNA_N_STARTUP_TRIALS,
            "base_seed": expcfg.SIMCA_OPTUNA_RANDOM_STATE,
        },
        "trial_budget": expcfg.SIMCA_OPTUNA_N_TRIALS_PER_TRACK,
        "technical_prune_statuses": expcfg.SIMCA_OPTUNA_TECHNICAL_PRUNE_STATUSES,
        "minimum_pareto_recall": expcfg.SIMCA_OPTUNA_MIN_PARETO_RECALL,
        "uniform_delta_tolerance": (
            expcfg.SIMCA_OPTUNA_UNIFORM_RECALL_DELTA_TOLERANCE
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_optuna_study_registry(
    domain_configurations: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
    *,
    results_tag: str,
    search_plan_hash: str,
) -> pd.DataFrame:
    """Materialise one reproducible study record for each of the eight tracks."""
    eligibility = projection_eligibility.set_index("evaluation_track")
    rows = []
    for track_index, track in enumerate(expcfg.SIMCA_EVALUATION_TRACKS):
        spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[track]
        n_domain = int(
            domain_configurations["evaluation_track"].astype(str).eq(track).sum()
        )
        if track not in eligibility.index:
            raise RuntimeError(f"03C has no eligibility row for {track!r}.")
        eligibility_status = str(eligibility.loc[track, "eligibility_status"])
        supported = eligibility_status in set(
            expcfg.SIMCA_OPTUNA_SUPPORTED_ELIGIBILITY_STATUSES
        )
        if n_domain == 0:
            study_scope = "unsupported_empty"
            study_status = "not_runnable_no_domain"
        elif supported:
            study_scope = "protocol"
            study_status = "runnable"
        else:
            study_scope = "diagnostic_only"
            study_status = "runnable_diagnostic_only"
        study_seed = int(expcfg.SIMCA_OPTUNA_RANDOM_STATE + track_index)
        study_name = expcfg.SIMCA_OPTUNA_STUDY_NAME_TEMPLATE.format(
            track_id=spec["track_id"],
            results_tag=results_tag,
            plan_hash=search_plan_hash[:12],
        )
        rows.append(
            {
                "evaluation_track": track,
                "track_id": spec["track_id"],
                "decision_mode": spec["decision_mode"],
                "study_name": study_name,
                "study_scope": study_scope,
                "eligibility_status": eligibility_status,
                "study_status": study_status,
                "study_seed": study_seed,
                "trial_budget": int(expcfg.SIMCA_OPTUNA_N_TRIALS_PER_TRACK),
                "n_domain_configurations": n_domain,
                "objective_names_json": json.dumps(
                    expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS[track]["objective_names"]
                ),
                "directions_json": json.dumps(
                    expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS[track]["directions"]
                ),
                "search_plan_hash": search_plan_hash,
            }
        )
    return pd.DataFrame(rows)


def build_optuna_pareto_candidates(optuna_trials: pd.DataFrame) -> pd.DataFrame:
    """Build diagnostic and protocol Pareto flags without mixing studies."""
    schema = expcfg.SIMCA_OPTUNA_PARETO_COLUMNS
    if optuna_trials is None or optuna_trials.empty:
        return pd.DataFrame(columns=schema)
    candidate_parts = []
    for track, track_trials in optuna_trials.groupby("evaluation_track", sort=False):
        if str(track) not in expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS:
            continue
        objective_spec = expcfg.SIMCA_OPTUNA_OBJECTIVE_SPECS[str(track)]
        objective_names = tuple(objective_spec["objective_names"])
        value_columns = [f"value_{index}" for index in range(len(objective_names))]
        if not set(value_columns).issubset(track_trials.columns):
            continue
        unique_trials = (
            track_trials.sort_values("trial_number", kind="mergesort")
            .drop_duplicates(["study_name", "domain_config_id"], keep="first")
            .copy()
        )
        complete = unique_trials.loc[
            unique_trials["state"].astype(str).eq("COMPLETE")
        ].copy()
        if complete.empty:
            continue
        for name, column in zip(objective_names, value_columns):
            complete[name] = pd.to_numeric(complete[column], errors="coerce")
        finite = np.isfinite(
            complete.loc[:, list(objective_names)].to_numpy(dtype=float)
        ).all(axis=1)
        complete = complete.loc[finite].copy()
        if complete.empty:
            continue
        complete["diagnostic_optuna_front"] = False
        complete["protocol_optuna_front"] = False
        diagnostic = pareto_front_by_group(
            complete,
            group_cols=("study_name",),
            minimize_cols=expcfg.SIMCA_EVALUATION_TRACK_SPECS[str(track)]["pareto_minimize"],
            maximize_cols=expcfg.SIMCA_EVALUATION_TRACK_SPECS[str(track)]["pareto_maximize"],
        )
        diagnostic_ids = set(diagnostic["trial_number"].astype(int))
        complete["diagnostic_optuna_front"] = complete["trial_number"].astype(int).isin(
            diagnostic_ids
        )
        protocol_pool = complete.loc[
            complete["status"].astype(str).eq("acceptable")
            & complete["eligibility_status"].astype(str).isin(
                expcfg.SIMCA_OPTUNA_SUPPORTED_ELIGIBILITY_STATUSES
            )
        ].copy()
        if not protocol_pool.empty:
            protocol_front = pareto_front_by_group(
                protocol_pool,
                group_cols=("study_name",),
                minimize_cols=expcfg.SIMCA_EVALUATION_TRACK_SPECS[str(track)]["pareto_minimize"],
                maximize_cols=expcfg.SIMCA_EVALUATION_TRACK_SPECS[str(track)]["pareto_maximize"],
            )
            protocol_ids = set(protocol_front["trial_number"].astype(int))
            complete["protocol_optuna_front"] = complete["trial_number"].astype(int).isin(
                protocol_ids
            )
        complete["downstream_eligible"] = complete[
            "protocol_optuna_front"
        ].astype(bool)
        candidate_parts.append(
            complete.loc[
                complete["diagnostic_optuna_front"]
                | complete["protocol_optuna_front"]
            ].copy()
        )
    if not candidate_parts:
        return pd.DataFrame(columns=schema)
    return (
        pd.concat(candidate_parts, ignore_index=True, sort=False)
        .sort_values(["evaluation_track", "trial_number"], kind="mergesort")
        .reindex(columns=schema)
        .reset_index(drop=True)
    )


def _ablation_identifier(row: Mapping[str, object]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str)
    return "ablation_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_preregistered_ablation_plan(
    configurations: pd.DataFrame,
    pareto_reference: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
    *,
    protocol_hash: str,
    search_plan_hash: str,
    spatial_lock: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Freeze paired, score-free ablations before the next batch-3 run."""
    schema = expcfg.SIMCA_ABLATION_PLAN_COLUMNS
    reference = pareto_reference.copy()
    if "row_type" in reference:
        reference = reference.loc[reference["row_type"].eq("configuration")]
    reference_ids = set(
        reference.loc[
            reference["protocol_pareto_front"].fillna(False).astype(bool),
            "calibration_id",
        ].astype(str)
    )
    config = configurations.copy()
    config["calibration_id"] = config["calibration_id"].astype(str)
    references = config.loc[config["calibration_id"].isin(reference_ids)].copy()
    eligibility_map = projection_eligibility.set_index("evaluation_track")[
        "eligibility_status"
    ].astype(str).to_dict()
    metric_json = json.dumps(expcfg.SIMCA_ABLATION_PRIMARY_METRICS)
    exact_columns = [
        column
        for column in expcfg.SIMCA_GRID_EXACT_CONFIGURATION_COLUMNS
        if column in config.columns
    ]
    rows: list[dict[str, object]] = []

    def append_pair(
        reference_row: Mapping[str, object],
        ablated_row: Mapping[str, object] | None,
        *,
        contrast_type: str,
        factor: str,
        reference_level: object,
        ablated_level: object,
        fit_changed: bool,
        projection_changed: bool,
        decision_changed: bool,
        spatial_processing_changed: bool,
        pairing_keys: Sequence[str],
        plan_status: str = "planned",
        unsupported_reason: str = "",
        interaction_formula: str = "",
    ) -> None:
        base = {
            "evaluation_track": str(reference_row.get("evaluation_track", "")),
            "track_id": str(reference_row.get("track_id", "")),
            "reference_config_id": str(reference_row.get("calibration_id", "")),
            "ablated_config_id": (
                str(ablated_row.get("calibration_id", ""))
                if ablated_row is not None
                else ""
            ),
            "contrast_type": contrast_type,
            "factor": factor,
            "reference_level": str(reference_level),
            "ablated_level": str(ablated_level),
            "fit_changed": bool(fit_changed),
            "projection_changed": bool(projection_changed),
            "decision_changed": bool(decision_changed),
            "spatial_processing_changed": bool(spatial_processing_changed),
            "interaction_formula": interaction_formula,
            "pairing_keys_json": json.dumps(tuple(pairing_keys)),
            "metric_set_json": metric_json,
            "preregistered": True,
            "registration_status": expcfg.SIMCA_ABLATION_REGISTRATION_STATUS,
            "eligibility_status": eligibility_map.get(
                str(reference_row.get("evaluation_track", "")), ""
            ),
            "plan_status": plan_status,
            "unsupported_reason": unsupported_reason,
            "protocol_hash": protocol_hash,
            "search_plan_hash": search_plan_hash,
        }
        base["ablation_id"] = _ablation_identifier(base)
        rows.append(base)

    # Generic paired factors are matched on every protected configuration
    # column. Multiple exact counterparts are retained; no performance tie-break
    # or diversity filter is applied.
    for factor_spec in expcfg.SIMCA_ABLATION_FACTOR_SPECS:
        varied = tuple(factor_spec["factor_columns"])
        primary = varied[0]
        if primary not in config.columns:
            continue
        pairing_keys = [column for column in exact_columns if column not in varied]
        n_before = len(rows)
        for reference_level, ablated_level in zip(
            factor_spec["reference_levels"], factor_spec["ablated_levels"]
        ):
            left = references.loc[
                references[primary].astype(str).eq(str(reference_level))
            ].copy()
            right = config.loc[
                config[primary].astype(str).eq(str(ablated_level))
            ].copy()
            if left.empty or right.empty:
                continue
            pairs = left.merge(
                right,
                on=pairing_keys,
                how="inner",
                suffixes=("_reference", "_ablated"),
            )
            for pair in pairs.to_dict(orient="records"):
                reference_row = {
                    "evaluation_track": pair.get("evaluation_track", pair.get("evaluation_track_reference", "")),
                    "track_id": pair.get("track_id", pair.get("track_id_reference", "")),
                    "calibration_id": pair["calibration_id_reference"],
                }
                ablated_row = {"calibration_id": pair["calibration_id_ablated"]}
                append_pair(
                    reference_row,
                    ablated_row,
                    contrast_type=str(factor_spec["contrast_type"]),
                    factor=str(factor_spec["factor"]),
                    reference_level=reference_level,
                    ablated_level=ablated_level,
                    fit_changed=bool(factor_spec["fit_changed"]),
                    projection_changed=bool(factor_spec["projection_changed"]),
                    decision_changed=bool(factor_spec["decision_changed"]),
                    spatial_processing_changed=bool(factor_spec["spatial_processing_changed"]),
                    pairing_keys=pairing_keys,
                )
        if len(rows) == n_before:
            for track, track_reference in references.groupby("evaluation_track", sort=False):
                append_pair(
                    track_reference.iloc[0].to_dict(),
                    None,
                    contrast_type=str(factor_spec["contrast_type"]),
                    factor=str(factor_spec["factor"]),
                    reference_level="declared_reference_level",
                    ablated_level="declared_ablated_level",
                    fit_changed=bool(factor_spec["fit_changed"]),
                    projection_changed=bool(factor_spec["projection_changed"]),
                    decision_changed=bool(factor_spec["decision_changed"]),
                    spatial_processing_changed=bool(factor_spec["spatial_processing_changed"]),
                    pairing_keys=pairing_keys,
                    plan_status="unsupported_no_valid_counterpart",
                    unsupported_reason="no_exact_one_factor_counterpart_in_04A_domain",
                )

    # Spectral blocks: generate the expected chain after one declared removal,
    # then join it back to the already validated 04A domain.
    preprocessing_pairing = [
        column
        for column in exact_columns
        if column not in {"preprocessing", "preprocessing_steps"}
    ]
    proposals = []
    for ref in references.to_dict(orient="records"):
        steps = list(parse_preprocessing_steps(ref.get("preprocessing_steps", "")))
        spectral_variants = []
        if "absorbance" in steps:
            spectral_variants.append(("spectral_absorbance", [step for step in steps if step != "absorbance"]))
        if any(step in {"snv", "msc"} for step in steps):
            spectral_variants.append(("spectral_snv_msc", [step for step in steps if step not in {"snv", "msc"}]))
        if any(str(step).startswith("sg_") for step in steps):
            spectral_variants.append(("spectral_sg", [step for step in steps if not str(step).startswith("sg_")]))
        if "sg_smooth" in steps:
            spectral_variants.append(("spectral_sg_derivative", ["sg_d1" if step == "sg_smooth" else step for step in steps]))
        for factor, expected_steps in spectral_variants:
            if not expected_steps:
                continue
            proposal = {column: ref.get(column) for column in preprocessing_pairing}
            proposal.update(
                {
                    "reference_config_id": ref["calibration_id"],
                    "reference_track_id": ref["track_id"],
                    "factor": factor,
                    "reference_level": "+".join(steps),
                    "expected_steps": "+".join(expected_steps),
                }
            )
            proposals.append(proposal)
    if proposals:
        proposal_df = pd.DataFrame(proposals)
        right = config.copy()
        right["expected_steps"] = right["preprocessing_steps"].map(
            lambda value: "+".join(parse_preprocessing_steps(value))
        )
        pairs = proposal_df.merge(
            right,
            on=[*preprocessing_pairing, "expected_steps"],
            how="left",
            suffixes=("", "_ablated"),
        )
        for pair in pairs.to_dict(orient="records"):
            reference_row = {
                "evaluation_track": pair["evaluation_track"],
                "track_id": pair["reference_track_id"],
                "calibration_id": pair["reference_config_id"],
            }
            ablated_id = pair.get("calibration_id")
            has_counterpart = pd.notna(ablated_id)
            append_pair(
                reference_row,
                {"calibration_id": ablated_id} if has_counterpart else None,
                contrast_type=(
                    "paired_variant"
                    if pair["factor"] == "spectral_sg_derivative"
                    else "strict_ablation"
                ),
                factor=pair["factor"],
                reference_level=pair["reference_level"],
                ablated_level=pair["expected_steps"],
                fit_changed=True,
                projection_changed=False,
                decision_changed=False,
                spatial_processing_changed=False,
                pairing_keys=preprocessing_pairing,
                plan_status="planned" if has_counterpart else "unsupported_no_valid_counterpart",
                unsupported_reason=(
                    "" if has_counterpart else "expected_preprocessing_chain_absent_from_04A_domain"
                ),
            )

    # Fixed threshold sensitivity: values are perturbed, never re-optimised.
    delta = float(expcfg.SIMCA_ABLATION_THRESHOLD_PERTURBATION)
    for ref in references.to_dict(orient="records"):
        mode = str(ref.get("decision_mode"))
        threshold_columns = (
            ("direct_2way_threshold",)
            if mode == "2way"
            else ("three_way_lower_threshold", "three_way_upper_threshold")
        )
        for threshold_column in threshold_columns:
            value = pd.to_numeric(ref.get(threshold_column), errors="coerce")
            if not np.isfinite(value):
                continue
            for direction in (-1.0, 1.0):
                append_pair(
                    ref,
                    None,
                    contrast_type="threshold_sensitivity",
                    factor=threshold_column,
                    reference_level=float(value),
                    ablated_level=float(value + direction * delta),
                    fit_changed=False,
                    projection_changed=False,
                    decision_changed=True,
                    spatial_processing_changed=False,
                    pairing_keys=("evaluation_track", "calibration_id"),
                )

    # Spatial operations are frozen from the 03C lock and remain restricted to
    # supported pixel-projection references.
    selected_spatial = dict((spatial_lock or {}).get("selected_parameters", {}))
    for ref in references.loc[
        references["projection_level"].astype(str).eq("pixel_projection")
    ].to_dict(orient="records"):
        for factor, reference_level, ablated_level in (
            ("spatial_raw_map", "locked_spatial_chain", "raw_map"),
            ("spatial_morphology", selected_spatial.get("morphology_operation", "locked"), "none"),
            ("spatial_min_area", selected_spatial.get("min_area_pixels", "locked"), 0),
            ("border_policy", "locked_border_core_policy", "no_border_treatment"),
        ):
            append_pair(
                ref,
                None,
                contrast_type="strict_ablation",
                factor=factor,
                reference_level=reference_level,
                ablated_level=ablated_level,
                fit_changed=False,
                projection_changed=False,
                decision_changed=False,
                spatial_processing_changed=True,
                pairing_keys=("evaluation_track", "calibration_id"),
            )

    # A small, predeclared interaction family resolves the one-factor-at-a-time
    # limitation. Exact four-cell availability is checked again before notebook
    # 05 executes the contrast; no new interaction may be added after batch 3.
    for ref in references.to_dict(orient="records"):
        track_id = str(ref.get("track_id"))
        for interaction in expcfg.SIMCA_ABLATION_INTERACTION_SPECS:
            relevant = (
                interaction == "rule_x_limit_source"
                or interaction == "preprocessing_x_matrix_method"
                or (interaction == "m_x_sampling_strategy" and str(ref.get("matrix_family")) == "pixel_matrix")
                or (interaction in {"morphology_x_min_area", "border_policy_x_morphology"} and str(ref.get("projection_level")) == "pixel_projection")
            )
            if not relevant:
                continue
            append_pair(
                ref,
                None,
                contrast_type="interaction",
                factor=interaction,
                reference_level="four_cell_reference",
                ablated_level="four_cell_contrast",
                fit_changed=interaction not in {"morphology_x_min_area", "border_policy_x_morphology"},
                projection_changed=False,
                decision_changed=interaction == "rule_x_limit_source",
                spatial_processing_changed=interaction in {"morphology_x_min_area", "border_policy_x_morphology"},
                pairing_keys=("evaluation_track", "calibration_id"),
                plan_status="planned_four_cell_match_required",
                interaction_formula="(D-C)-(B-A)",
            )

    plan = pd.DataFrame(rows)
    if plan.empty:
        return pd.DataFrame(columns=schema)
    plan = plan.drop_duplicates("ablation_id").sort_values(
        ["evaluation_track", "contrast_type", "factor", "reference_config_id", "ablated_config_id"],
        kind="mergesort",
    )
    return plan.reindex(columns=schema).reset_index(drop=True)
