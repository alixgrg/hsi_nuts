"""Compact 04A audit of the models and thresholds selected in notebook 03B.

Notebook 04A is deliberately not a second model-selection step. It streams
the cross-fitted rows of the policies already selected in 03B, reconstructs
the model-level metrics with the exact 03B aggregation code, and checks them
against ``model_metrics.parquet``. No model is fitted and no threshold or
identifier is created here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src import experiment_config as expcfg
from src.workflows.simca_calibration_registry import (
    build_selected_execution_registry,
)
from src.workflows.simca_calibration_selection import (
    aggregate_threshold_candidates,
    build_model_metrics,
)
from src.utils import require_columns

_POLICY_COLUMNS = (
    "model_id",
    "random_state",
    "decision_scope",
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
)
_POLICY_VALUE_COLUMNS = (
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
)
_POLICY_SENTINEL = -1.0
_POLICY_DECIMALS = 7
_COUNT_COLUMNS = (
    "n_observations",
    "n_target",
    "n_non_target",
)


def _normalized_policy_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable natural policy keys without creating a policy ID."""
    require_columns(frame, _POLICY_COLUMNS, "threshold policy table")
    out = frame.loc[:, list(_POLICY_COLUMNS)].copy()
    out["model_id"] = out["model_id"].astype(str)
    out["decision_scope"] = out["decision_scope"].astype(str)
    out["random_state"] = pd.to_numeric(
        out["random_state"], errors="raise"
    ).astype(int)
    for column in _POLICY_VALUE_COLUMNS:
        numeric = pd.to_numeric(
            out[column], errors="coerce"
        ).astype(np.float64)
        invalid = out[column].notna() & numeric.isna()
        if invalid.any():
            raise RuntimeError(f"Non-numeric values in {column}.")
        finite = numeric.dropna().to_numpy(dtype=float)
        if finite.size and not np.isfinite(finite).all():
            raise RuntimeError(f"Non-finite values in {column}.")
        out[column] = numeric.round(_POLICY_DECIMALS).fillna(
            _POLICY_SENTINEL
        )
    return out


def extract_selected_crossfit_metrics(
    threshold_metrics_path: str | Path,
    selected_thresholds: pd.DataFrame,
    *,
    batch_size: int = expcfg.SIMCA_GRID_THRESHOLD_METRIC_BATCH_SIZE,
) -> pd.DataFrame:
    """Stream only the selected 03B policy rows from threshold metrics.

    Filtering first on the 39 selected ``model_id`` values keeps the batch
    join small. The join then uses the natural run/scope/policy coordinates;
    no synthetic calibration or policy identifier is necessary.
    """
    source = Path(threshold_metrics_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    require_columns(
        selected_thresholds,
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
        "selected thresholds",
    )
    selected_keys = _normalized_policy_keys(selected_thresholds)
    if selected_keys.duplicated(list(_POLICY_COLUMNS)).any():
        raise RuntimeError("Selected natural policy keys must be unique.")
    selected_ids = frozenset(selected_keys["model_id"])

    columns = list(expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS)
    matched_parts: list[pd.DataFrame] = []
    dataset = ds.dataset(source, format="parquet")
    selected_filter = (
        ds.field("model_id").isin(sorted(selected_ids))
        & (ds.field("evaluation_fold") >= 0)
    )
    scanner = dataset.scanner(
        columns=columns,
        filter=selected_filter,
        batch_size=int(batch_size),
        use_threads=True,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue

        normalized = _normalized_policy_keys(frame)
        for column in _POLICY_COLUMNS:
            frame[column] = normalized[column].to_numpy(copy=True)
        matched = frame.merge(
            selected_keys,
            on=list(_POLICY_COLUMNS),
            how="inner",
            validate="many_to_one",
        )
        if not matched.empty:
            matched_parts.append(matched.reindex(columns=columns))

    if not matched_parts:
        raise RuntimeError("No cross-fitted metric matches a selected policy.")
    metrics = pd.concat(matched_parts, ignore_index=True, sort=False)
    for column in _POLICY_VALUE_COLUMNS:
        metrics[column] = pd.to_numeric(
            metrics[column], errors="raise"
        ).mask(metrics[column].eq(_POLICY_SENTINEL))
    metrics["evaluation_fold"] = pd.to_numeric(
        metrics["evaluation_fold"], errors="raise"
    ).astype(int)
    metrics["random_state"] = pd.to_numeric(
        metrics["random_state"], errors="raise"
    ).astype(int)
    metrics["value"] = pd.to_numeric(
        metrics["value"], errors="raise"
    ).astype(float)

    metric_key = [
        *_POLICY_COLUMNS,
        "evaluation_fold",
        "metric",
    ]
    if metrics.duplicated(metric_key).any():
        raise RuntimeError("Duplicate selected cross-fitted metric rows.")

    observed_keys = _normalized_policy_keys(metrics).drop_duplicates()
    coverage = selected_keys.merge(
        observed_keys.assign(_observed=True),
        on=list(_POLICY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if not coverage["_observed"].fillna(False).all():
        missing = coverage.loc[
            coverage["_observed"].isna(),
            ["model_id", "random_state", "decision_scope"],
        ]
        raise RuntimeError(
            "Selected policies without cross-fitted metrics: "
            f"{missing.to_dict(orient='records')}"
        )

    expected_folds = frozenset(
        range(int(expcfg.INTERNAL_CALIBRATION_N_SPLITS))
    )
    observed_folds = metrics.groupby(
        ["model_id", "random_state", "decision_scope"],
        sort=False,
        dropna=False,
    )["evaluation_fold"].agg(lambda values: frozenset(map(int, values)))
    if not observed_folds.map(lambda values: values == expected_folds).all():
        raise RuntimeError("A selected policy has incomplete fold coverage.")

    return metrics.sort_values(
        [
            "model_id",
            "random_state",
            "decision_scope",
            "evaluation_fold",
            "metric",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def build_selected_run_fold_metrics(
    selected_crossfit_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot the selected long metrics to one compact natural fold row."""
    require_columns(
        selected_crossfit_metrics,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS,
        "selected cross-fitted metrics",
    )
    index_columns = [
        "model_id",
        "random_state",
        "decision_scope",
        "evaluation_fold",
    ]
    duplicate_key = [*index_columns, "metric"]
    if selected_crossfit_metrics.duplicated(duplicate_key).any():
        raise RuntimeError("Duplicate metric for one selected run fold.")

    wide = selected_crossfit_metrics.pivot(
        index=index_columns,
        columns="metric",
        values="value",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"evaluation_fold": "fold_id"}).reindex(
        columns=expcfg.SIMCA_GRID_SELECTED_FOLD_METRIC_COLUMNS
    )
    if wide[list(_COUNT_COLUMNS)].isna().any().any():
        raise RuntimeError("Fold observation counts are incomplete.")
    for column in _COUNT_COLUMNS:
        numeric = pd.to_numeric(wide[column], errors="raise")
        if not np.isclose(numeric, np.rint(numeric)).all():
            raise RuntimeError(f"{column} must contain integer counts.")
        wide[column] = np.rint(numeric).astype(np.int64)

    natural_key = [
        "model_id",
        "random_state",
        "decision_scope",
        "fold_id",
    ]
    if wide.duplicated(natural_key).any():
        raise RuntimeError("Selected fold natural keys must be unique.")
    return wide.sort_values(natural_key, kind="mergesort").reset_index(
        drop=True
    )


def _metric_consistency_by_model(
    selected_crossfit_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    selected_model_ids: Sequence[str],
    *,
    atol: float,
) -> pd.DataFrame:
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")
    require_columns(
        model_metrics,
        expcfg.INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS,
        "model metrics",
    )
    candidates = aggregate_threshold_candidates(
        selected_crossfit_metrics
    )
    recomputed = build_model_metrics(candidates).rename(
        columns={"value": "recomputed_value"}
    )
    reference = model_metrics.loc[
        model_metrics["model_id"].astype(str).isin(
            set(map(str, selected_model_ids))
        )
    ].copy()
    reference["model_id"] = reference["model_id"].astype(str)
    reference = reference.rename(columns={"value": "reference_value"})
    key = ["model_id", "metric"]
    if reference.duplicated(key).any():
        raise RuntimeError("Reference model metrics contain duplicate keys.")

    comparison = reference.merge(
        recomputed,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        missing = comparison.loc[
            ~comparison["_merge"].eq("both"), [*key, "_merge"]
        ]
        raise RuntimeError(
            "03B model-metric keys cannot be reconstructed: "
            f"{missing.to_dict(orient='records')}"
        )
    reference_values = pd.to_numeric(
        comparison["reference_value"], errors="coerce"
    ).to_numpy(dtype=float)
    recomputed_values = pd.to_numeric(
        comparison["recomputed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_pair = np.isfinite(reference_values) & np.isfinite(
        recomputed_values
    )
    both_missing = np.isnan(reference_values) & np.isnan(recomputed_values)
    if not (finite_pair | both_missing).all():
        raise RuntimeError("Reference and reconstructed finite states differ.")
    comparison["absolute_difference"] = np.where(
        finite_pair,
        np.abs(reference_values - recomputed_values),
        0.0,
    )
    if comparison["absolute_difference"].gt(float(atol)).any():
        mismatch = comparison.nlargest(10, "absolute_difference")[
            [*key, "reference_value", "recomputed_value", "absolute_difference"]
        ]
        raise RuntimeError(
            "Selected 03B metrics are not reproducible at the configured "
            f"tolerance ({atol}): {mismatch.to_dict(orient='records')}"
        )
    return (
        comparison.groupby("model_id", as_index=False, sort=False)
        .agg(max_abs_metric_difference=("absolute_difference", "max"))
    )


def build_selected_model_reference(
    *,
    selected_executions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    model_metrics: pd.DataFrame,
    selected_crossfit_metrics: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
    atol: float = expcfg.SIMCA_GRID_REFERENCE_METRIC_ATOL,
) -> pd.DataFrame:
    """Build one audit row per selected model, retaining all eight tracks."""
    require_columns(
        selected_executions,
        expcfg.DOMAIN_SPATIAL_SELECTED_EXECUTION_COLUMNS,
        "selected executions",
    )
    require_columns(
        projection_eligibility,
        expcfg.PROJECTION_ELIGIBILITY_COLUMNS,
        "projection eligibility",
    )
    if projection_eligibility["track_id"].astype(str).duplicated().any():
        raise RuntimeError("Projection eligibility must have one row per track.")

    execution_key = ["model_id", "random_state"]
    if selected_executions.duplicated(execution_key).any():
        raise RuntimeError("Selected execution natural keys must be unique.")
    model_track = selected_executions[["model_id", "track_id"]].drop_duplicates()
    if model_track["model_id"].astype(str).duplicated().any():
        raise RuntimeError("A selected model maps to more than one track.")

    observed_counts = selected_executions.groupby(
        "track_id", as_index=False, sort=False
    ).agg(
        n_selected_models=("model_id", "nunique"),
        n_selected_runs=("model_id", "size"),
    )
    expected_counts = projection_eligibility[
        ["track_id", "n_selected_models", "n_selected_runs"]
    ].copy()
    count_check = expected_counts.merge(
        observed_counts,
        on="track_id",
        how="outer",
        suffixes=("_03c", "_observed"),
        validate="one_to_one",
    )
    for column in ("n_selected_models", "n_selected_runs"):
        if not count_check[f"{column}_03c"].eq(
            count_check[f"{column}_observed"]
        ).all():
            raise RuntimeError(f"03C {column} does not match selected 03B rows.")

    selected_ids = model_track["model_id"].astype(str).tolist()
    differences = _metric_consistency_by_model(
        selected_crossfit_metrics,
        model_metrics,
        selected_ids,
        atol=float(atol),
    )
    run_counts = selected_executions.groupby(
        "model_id", as_index=False, sort=False
    ).agg(n_selected_runs=("random_state", "nunique"))
    scope_counts = selected_thresholds.groupby(
        "model_id", as_index=False, sort=False
    ).agg(n_decision_scopes=("decision_scope", "nunique"))
    eligibility = projection_eligibility[
        ["track_id", "eligibility_status"]
    ].copy()

    reference = (
        model_track.merge(run_counts, on="model_id", validate="one_to_one")
        .merge(scope_counts, on="model_id", validate="one_to_one")
        .merge(eligibility, on="track_id", validate="many_to_one")
        .merge(differences, on="model_id", validate="one_to_one")
    )
    supported = reference["eligibility_status"].astype(str).isin(
        expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
    )
    reference["downstream_status"] = np.where(
        supported,
        "supported",
        "diagnostic_only",
    )
    reference = reference.reindex(
        columns=expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS
    )
    if reference["model_id"].astype(str).duplicated().any():
        raise RuntimeError("Model reference must have one row per model_id.")
    return reference.sort_values(
        ["track_id", "model_id"], kind="mergesort"
    ).reset_index(drop=True)


def run_selected_model_reference_audit(
    *,
    model_catalog: pd.DataFrame,
    selected_models: pd.DataFrame,
    selected_runs: pd.DataFrame,
    selected_threshold_rows: pd.DataFrame,
    model_metrics: pd.DataFrame,
    track_contracts: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
    threshold_metrics_path: str | Path,
    batch_size: int = expcfg.SIMCA_GRID_THRESHOLD_METRIC_BATCH_SIZE,
    atol: float = expcfg.SIMCA_GRID_REFERENCE_METRIC_ATOL,
) -> dict[str, pd.DataFrame]:
    """Run the complete, non-selecting 04A reference audit."""
    selected_executions, selected_thresholds = (
        build_selected_execution_registry(
            model_catalog,
            selected_models,
            selected_runs,
            selected_threshold_rows,
            track_contracts=track_contracts,
        )
    )
    selected_crossfit_metrics = extract_selected_crossfit_metrics(
        threshold_metrics_path,
        selected_thresholds,
        batch_size=int(batch_size),
    )
    fold_metrics = build_selected_run_fold_metrics(
        selected_crossfit_metrics
    )
    model_reference = build_selected_model_reference(
        selected_executions=selected_executions,
        selected_thresholds=selected_thresholds,
        model_metrics=model_metrics,
        selected_crossfit_metrics=selected_crossfit_metrics,
        projection_eligibility=projection_eligibility,
        atol=float(atol),
    )
    return {
        "model_reference": model_reference,
        "fold_metrics": fold_metrics,
    }


__all__ = [
    "build_selected_model_reference",
    "build_selected_run_fold_metrics",
    "extract_selected_crossfit_metrics",
    "run_selected_model_reference_audit",
]
