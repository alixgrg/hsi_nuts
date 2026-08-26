"""Diagnostic 04B coverage benchmark over the evaluable 03B model universe.

This module intentionally does not implement another scientific model search.
Optuna sees one categorical ``model_id`` parameter and looks up the frozen 03B
objectives. Such a sampler cannot learn similarity between model
hyperparameters, so its only defensible role is a negative control for
budgeted reference coverage. Its output must never alter ``selected_models``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import optuna
import pandas as pd

from src import experiment_config as expcfg
from src.utils import require_columns



def _ordered_track_ids() -> tuple[str, ...]:
    return tuple(
        str(expcfg.SIMCA_EVALUATION_TRACK_IDS[track])
        for track in expcfg.SIMCA_EVALUATION_TRACKS
    )


def _model_metric_matrix(model_metrics: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        model_metrics,
        expcfg.INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS,
        "model metrics",
    )
    work = model_metrics.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS)
    ].copy()
    work["model_id"] = work["model_id"].astype(str)
    work["metric"] = work["metric"].astype(str)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    if work.duplicated(["model_id", "metric"]).any():
        raise RuntimeError("model_metrics has duplicate natural keys.")
    wide = work.pivot(index="model_id", columns="metric", values="value")
    wide.columns.name = None
    return wide


def _validate_track_contracts(track_contracts: pd.DataFrame) -> None:
    require_columns(
        track_contracts,
        ("track_id", "decision_mode", "projection_level"),
        "track contracts",
    )
    observed = tuple(track_contracts["track_id"].astype(str))
    expected = _ordered_track_ids()
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise RuntimeError("Track contracts must contain exactly E1-E8 once.")


def build_evaluable_model_universe(
    model_catalog: pd.DataFrame,
    model_metrics: pd.DataFrame,
    track_contracts: pd.DataFrame,
) -> pd.DataFrame:
    """Return models with every configured 03B Pareto objective finite."""
    require_columns(
        model_catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        "model catalog",
    )
    _validate_track_contracts(track_contracts)
    catalog = model_catalog[["model_id", "track_id"]].copy()
    catalog["model_id"] = catalog["model_id"].astype(str)
    catalog["track_id"] = catalog["track_id"].astype(str)
    if catalog["model_id"].duplicated().any():
        raise RuntimeError("model_catalog.model_id must be unique.")

    metric_matrix = _model_metric_matrix(model_metrics)
    parts: list[pd.DataFrame] = []
    for track_id in _ordered_track_ids():
        if track_id not in expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES:
            raise KeyError(f"Missing 03B objectives for {track_id}.")
        objective_spec = expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES[
            track_id
        ]
        objective_columns = [
            *map(str, objective_spec["minimize"]),
            *map(str, objective_spec["maximize"]),
        ]
        missing_metrics = sorted(
            set(objective_columns) - set(metric_matrix.columns)
        )
        if missing_metrics:
            raise KeyError(
                f"Missing objective metrics for {track_id}: {missing_metrics}"
            )
        candidates = catalog.loc[catalog["track_id"].eq(track_id)].merge(
            metric_matrix[objective_columns],
            left_on="model_id",
            right_index=True,
            how="inner",
            validate="one_to_one",
        )
        finite = np.isfinite(
            candidates[objective_columns].to_numpy(dtype=float)
        ).all(axis=1)
        parts.append(candidates.loc[finite, ["model_id", "track_id"]])

    universe = pd.concat(parts, ignore_index=True, sort=False)
    if universe.empty or universe["model_id"].duplicated().any():
        raise RuntimeError("The evaluable model universe is empty or duplicated.")
    observed_tracks = set(universe["track_id"].astype(str))
    if observed_tracks != set(_ordered_track_ids()):
        missing = sorted(set(_ordered_track_ids()) - observed_tracks)
        raise RuntimeError(f"No evaluable model for tracks: {missing}")
    return universe.sort_values(
        ["track_id", "model_id"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_selected_reference(
    selected_models: pd.DataFrame,
    model_reference: pd.DataFrame,
    universe: pd.DataFrame,
) -> tuple[set[str], pd.DataFrame]:
    require_columns(
        selected_models,
        expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS,
        "selected models",
    )
    require_columns(
        model_reference,
        expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS,
        "04A model reference",
    )
    if not selected_models["selection_status"].astype(str).eq(
        "selected"
    ).all():
        raise RuntimeError("selected_models must contain selected rows only.")
    if selected_models["model_id"].astype(str).duplicated().any():
        raise RuntimeError("selected_models.model_id must be unique.")
    if model_reference["model_id"].astype(str).duplicated().any():
        raise RuntimeError("04A model_reference.model_id must be unique.")
    selected_ids = set(selected_models["model_id"].astype(str))
    reference_ids = set(model_reference["model_id"].astype(str))
    if selected_ids != reference_ids:
        raise RuntimeError("04A changed the 03B selected-model universe.")
    universe_ids = set(universe["model_id"].astype(str))
    if not selected_ids.issubset(universe_ids):
        missing = sorted(selected_ids - universe_ids)
        raise RuntimeError(
            f"Selected models lack finite benchmark objectives: {missing}"
        )

    track_status = model_reference[
        ["track_id", "downstream_status"]
    ].drop_duplicates()
    if track_status["track_id"].astype(str).duplicated().any():
        raise RuntimeError("A track has multiple 04A downstream statuses.")
    if set(track_status["track_id"].astype(str)) != set(
        _ordered_track_ids()
    ):
        raise RuntimeError("04A does not cover every track E1-E8.")
    return selected_ids, track_status


def run_categorical_tpe_coverage_benchmark(
    *,
    model_catalog: pd.DataFrame,
    model_metrics: pd.DataFrame,
    selected_models: pd.DataFrame,
    track_contracts: pd.DataFrame,
    model_reference: pd.DataFrame,
    trial_budget: int = expcfg.SIMCA_OPTUNA_N_TRIALS_PER_TRACK,
    n_startup_trials: int = expcfg.SIMCA_OPTUNA_N_STARTUP_TRIALS,
    random_state: int = expcfg.SIMCA_OPTUNA_RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Run a deterministic categorical-TPE negative-control benchmark."""
    if trial_budget < 1:
        raise ValueError("trial_budget must be positive.")
    if n_startup_trials < 1:
        raise ValueError("n_startup_trials must be positive.")
    universe = build_evaluable_model_universe(
        model_catalog,
        model_metrics,
        track_contracts,
    )
    selected_ids, track_status = _validate_selected_reference(
        selected_models,
        model_reference,
        universe,
    )
    metric_matrix = _model_metric_matrix(model_metrics)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_parts: list[pd.DataFrame] = []
    for track_offset, track_id in enumerate(_ordered_track_ids()):
        candidate_ids = (
            universe.loc[universe["track_id"].eq(track_id), "model_id"]
            .astype(str)
            .sort_values(kind="mergesort")
            .to_numpy()
        )
        objective_spec = expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES[
            track_id
        ]
        minimize = tuple(map(str, objective_spec["minimize"]))
        maximize = tuple(map(str, objective_spec["maximize"]))
        objective_columns = (*minimize, *maximize)
        directions = (
            *("minimize" for _ in minimize),
            *("maximize" for _ in maximize),
        )
        values = metric_matrix.loc[
            candidate_ids, list(objective_columns)
        ].to_numpy(dtype=float)
        position_by_model = {
            model_id: position
            for position, model_id in enumerate(candidate_ids)
        }
        candidate_choices = candidate_ids.tolist()

        sampler = optuna.samplers.TPESampler(
            seed=int(random_state) + track_offset,
            n_startup_trials=int(n_startup_trials),
            multivariate=bool(expcfg.SIMCA_OPTUNA_SAMPLER_MULTIVARIATE),
        )
        study = optuna.create_study(
            sampler=sampler,
            directions=directions,
        )

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            model_id = trial.suggest_categorical(
                expcfg.SIMCA_OPTUNA_BENCHMARK_PARAMETER,
                candidate_choices,
            )
            row = values[position_by_model[str(model_id)]]
            return tuple(map(float, row))

        study.optimize(
            objective,
            n_trials=int(trial_budget),
            n_jobs=1,
            show_progress_bar=False,
        )
        sampled = pd.DataFrame(
            {
                "track_id": track_id,
                "trial_number": np.arange(len(study.trials), dtype=np.int64),
                "model_id": [
                    str(trial.params[expcfg.SIMCA_OPTUNA_BENCHMARK_PARAMETER])
                    for trial in study.trials
                ],
            }
        )
        sampled["is_repeat"] = sampled["model_id"].duplicated()
        sampled["is_selected_reference"] = sampled["model_id"].isin(
            selected_ids
        )
        trial_parts.append(sampled)

    sampled_models = pd.concat(trial_parts, ignore_index=True, sort=False)
    sampled_models = sampled_models.reindex(
        columns=expcfg.SIMCA_OPTUNA_BENCHMARK_SAMPLE_COLUMNS
    )
    sample_key = ["track_id", "trial_number"]
    if sampled_models.duplicated(sample_key).any():
        raise RuntimeError("Benchmark trial sequence keys must be unique.")

    universe_counts = universe.groupby("track_id", as_index=False).agg(
        n_evaluable_models=("model_id", "nunique")
    )
    selected_by_track = model_reference.groupby(
        "track_id", as_index=False
    ).agg(n_selected_reference_models=("model_id", "nunique"))
    sampled_summary = sampled_models.groupby(
        "track_id", as_index=False
    ).agg(
        trial_budget=("trial_number", "size"),
        n_unique_models_sampled=("model_id", "nunique"),
    )
    recovered = (
        sampled_models.loc[sampled_models["is_selected_reference"]]
        .groupby("track_id", as_index=False)
        .agg(n_selected_reference_recovered=("model_id", "nunique"))
    )
    summary = (
        track_status.merge(universe_counts, on="track_id", validate="one_to_one")
        .merge(selected_by_track, on="track_id", validate="one_to_one")
        .merge(sampled_summary, on="track_id", validate="one_to_one")
        .merge(recovered, on="track_id", how="left", validate="one_to_one")
    )
    summary["n_selected_reference_recovered"] = summary[
        "n_selected_reference_recovered"
    ].fillna(0).astype(int)
    summary["duplicate_trial_rate"] = 1.0 - (
        summary["n_unique_models_sampled"] / summary["trial_budget"]
    )
    summary["model_coverage_rate"] = (
        summary["n_unique_models_sampled"] / summary["n_evaluable_models"]
    )
    summary["selected_reference_recall"] = (
        summary["n_selected_reference_recovered"]
        / summary["n_selected_reference_models"]
    )
    summary["uniform_expected_selected_recall"] = 1.0 - np.power(
        1.0 - (1.0 / summary["n_evaluable_models"]),
        summary["trial_budget"],
    )
    summary["recall_delta_vs_uniform"] = (
        summary["selected_reference_recall"]
        - summary["uniform_expected_selected_recall"]
    )
    summary = summary.reindex(
        columns=expcfg.SIMCA_OPTUNA_BENCHMARK_SUMMARY_COLUMNS
    )
    summary["track_id"] = pd.Categorical(
        summary["track_id"],
        categories=_ordered_track_ids(),
        ordered=True,
    )
    summary = summary.sort_values("track_id").reset_index(drop=True)
    summary["track_id"] = summary["track_id"].astype(str)
    if len(summary) != len(_ordered_track_ids()):
        raise RuntimeError("Benchmark summary must contain exactly E1-E8.")
    return {
        "sampled_models": sampled_models,
        "search_efficiency": summary,
    }


__all__ = [
    "build_evaluable_model_universe",
    "run_categorical_tpe_coverage_benchmark",
]
