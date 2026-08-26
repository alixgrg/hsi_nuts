"""Notebook-05 SIMCA robustness and pre-batch4 review.

This module intentionally centralizes the complete notebook-05 robustness
workflow while preserving the upstream 00-04C scientific contracts.

Scientific contract
-------------------
- ``model_id`` is the scientific model identity.
- ``(model_id, random_state)`` is a repeated execution.
- ``fit_id`` and ``projection_id`` are reused from the frozen 03B/04C registry
  whenever the scientific execution already exists; no diagnostic surrogate
  model/execution IDs are introduced.
- the official validation Pareto is computed independently inside E1 ... E8
  and only from the common frozen base-seed panel.
- additional random states are a post-Pareto stress test for stochastic models
  only and never trigger model or threshold-policy reselection.
- threshold, source-image, calibration-fold, Pareto-front and spatial
  sensitivity analyses are supporting-only and cannot alter eligibility.
- notebook 05 never opens batch 4 and never performs final model selection.

Implementation policy
---------------------
The module reuses the canonical upstream kernels for threshold materialization,
04C metric evaluation, Pareto dominance and locked spatial evaluation. It does
not reimplement those scientific definitions locally.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.decision.metrics import apply_locked_margin_decision
from src.protocol_governance import (
    canonical_json,
    sha256_dataframe,
    sha256_payload,
)
from src.utils import (
    normalize_integer_sequence,
    parse_preprocessing_steps,
    require_columns,
)
from src.workflows.protocol_audit import assert_no_test_stage_inputs
from src.workflows.protocol_split import build_grouped_folds
from src.workflows.simca_calibration_registry import (
    build_additional_seed_execution_registry,
    stochastic_model_mask,
)
from src.workflows.simca_calibration_selection import (
    materialize_fixed_threshold_policy_for_runs,
)
from src.workflows.simca_candidates import validate_simca_table_columns
from src.workflows.simca_grid_evaluation import (
    evaluate_locked_validation_predictions,
    finite_max,
    finite_mean,
    finite_min,
    finite_std,
)
from src.workflows.simca_internal_calibration import (
    run_internal_calibration_8tracks,
)
from src.workflows.simca_selection_utils import pareto_front_with_witness
from src.workflows.simca_thresholds_calibration import build_pixel_vote_table
from src.workflows import spatial_postprocessing_calibration as spatial_cal
from src.workflows.spatial_postprocessing_calibration import (
    build_locked_spatial_validation_outputs,
    build_spatial_candidate_grid,
)


# ---------------------------------------------------------------------------
# 0. Canonical notebook-05 helper keys
# ---------------------------------------------------------------------------

_THRESHOLD_KEY = ("model_id", "random_state", "decision_scope")
_EXECUTION_KEY = ("model_id", "random_state")
_SCOPE_KEY = _THRESHOLD_KEY
_EXECUTION_VIEW = (
    "model_id",
    "random_state",
    "projection_id",
    "track_id",
    "decision_mode",
    "projection_level",
)
_SOURCE_IMAGE_GROUP_KEY = (
    "model_id",
    "random_state",
    "track_id",
    "decision_scope",
)


# ---------------------------------------------------------------------------
# 1. Small contract-aware helpers
# ---------------------------------------------------------------------------


def metric_base_name(metric: str) -> str:
    """Return the unscoped metric token used by config registries."""
    return str(metric).split("__")[-1]

def metric_direction(metric: str) -> str | None:
    """Return ``minimize``/``maximize`` from the canonical metric registry."""
    return expcfg.SIMCA_ROBUSTNESS_METRIC_DIRECTIONS.get(
        metric_base_name(metric)
    )

def practical_tolerance(metric: str) -> float:
    """Return the configured supporting-effect tolerance, or NaN if absent."""
    value = expcfg.SIMCA_ROBUSTNESS_SENSITIVITY_TOLERANCES.get(
        str(metric),
        expcfg.SIMCA_ROBUSTNESS_SENSITIVITY_TOLERANCES.get(
            metric_base_name(metric),
        ),
    )
    if value is None:
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan

def annotate_practical_effects(
    frame: pd.DataFrame,
    *,
    delta_col: str = "delta",
    metric_col: str = "metric",
    tolerance_col: str = "practical_tolerance",
    effect_status_col: str = "effect_status",
    directional_status_col: str | None = None,
    alternative_label: str = "alternative",
    reference_label: str = "reference",
) -> pd.DataFrame:
    """Attach one common interpretation to supporting metric deltas.

    The helper never filters rows and never changes model eligibility.  A
    missing practical tolerance is reported explicitly instead of being
    misclassified as a tolerance failure.
    """
    require_columns(frame, (metric_col, delta_col), "metric-effect table")
    out = frame.copy()
    delta = pd.to_numeric(out[delta_col], errors="coerce")
    tolerance = out[metric_col].astype(str).map(practical_tolerance).astype(float)
    finite_delta = np.isfinite(delta.to_numpy(dtype=float))
    finite_tolerance = np.isfinite(tolerance.to_numpy(dtype=float))
    within = finite_delta & finite_tolerance & (
        np.abs(delta.to_numpy(dtype=float)) <= tolerance.to_numpy(dtype=float)
    )
    outside = finite_delta & finite_tolerance & ~within

    out[tolerance_col] = tolerance
    out[effect_status_col] = np.select(
        [~finite_delta, within, outside],
        [
            "not_estimable_non_finite_metric",
            "within_practical_tolerance",
            "outside_practical_tolerance",
        ],
        default="descriptive_no_practical_tolerance",
    )

    if directional_status_col is not None:
        direction = out[metric_col].astype(str).map(metric_direction)
        delta_values = delta.to_numpy(dtype=float)
        out[directional_status_col] = np.select(
            [
                within,
                outside & direction.eq("minimize").to_numpy() & (delta_values < 0),
                outside & direction.eq("minimize").to_numpy() & (delta_values > 0),
                outside & direction.eq("maximize").to_numpy() & (delta_values > 0),
                outside & direction.eq("maximize").to_numpy() & (delta_values < 0),
            ],
            [
                "practically_equivalent",
                f"{alternative_label}_better",
                f"{reference_label}_better",
                f"{alternative_label}_better",
                f"{reference_label}_better",
            ],
            default="descriptive_change",
        )
    return out

def normalize_validation_executions(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    name: str = "validation_executions",
) -> pd.DataFrame:
    """Validate the 04C execution contract and return normalized columns only."""
    require_columns(frame, expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS, name)
    requested = tuple(
        map(
            str,
            columns
            if columns is not None
            else expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS,
        )
    )
    missing = [column for column in requested if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing requested columns: {missing}.")
    out = frame.loc[:, list(requested)].copy()
    if "model_id" in out:
        out["model_id"] = out["model_id"].astype(str)
    if "random_state" in out:
        out["random_state"] = pd.to_numeric(
            out["random_state"], errors="raise"
        ).astype(int)
    for column in (
        "fit_id",
        "projection_id",
        "track_id",
        "decision_mode",
        "projection_level",
        "decision_scope",
    ):
        if column in out:
            out[column] = out[column].astype(str)
    if {"model_id", "random_state"}.issubset(out.columns):
        if out.duplicated(list(_EXECUTION_KEY)).any():
            raise RuntimeError(f"{name} duplicates {_EXECUTION_KEY}.")
    return out


def normalize_threshold_registry(
    frame: pd.DataFrame,
    *,
    name: str = "selected_thresholds",
) -> pd.DataFrame:
    """Validate and normalize the canonical selected-threshold table.

    Notebook 05 always materializes numeric threshold/policy columns as
    float64 in its local working copy.

    This does not alter the persisted 03B/04C artifacts.  It only prevents
    dtype-dependent behaviour when supporting robustness analyses perturb
    numeric threshold values.
    """
    require_columns(
        frame,
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
        name,
    )

    out = frame.loc[
        :,
        list(expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS),
    ].copy()

    # ------------------------------------------------------------------
    # Natural identity.
    # ------------------------------------------------------------------
    out["model_id"] = out["model_id"].astype(str)

    out["random_state"] = pd.to_numeric(
        out["random_state"],
        errors="raise",
    ).astype(int)

    out["decision_scope"] = out["decision_scope"].astype(str)

    # ------------------------------------------------------------------
    # Canonical numeric representation for notebook 05.
    #
    # Policy coordinates may legitimately contain NaN depending on the
    # decision mode/scope.  Numeric thresholds themselves must be finite.
    # Explicit float64 conversion is intentional: threshold-sensitivity
    # perturbations are float64 and Pandas must not attempt a lossy assignment
    # into a float32 column loaded from Parquet.
    # ------------------------------------------------------------------
    numeric_columns = (
        "lower_quantile",
        "upper_quantile",
        "vote_threshold",
        "lower_threshold",
        "upper_threshold",
    )

    for column in numeric_columns:
        original = out[column]

        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid = original.notna() & numeric.isna()
        if invalid.any():
            examples = (
                original.loc[invalid]
                .astype(str)
                .drop_duplicates()
                .head(10)
                .tolist()
            )
            raise ValueError(
                f"{name}.{column} contains non-numeric values: "
                f"{examples}."
            )

        out[column] = numeric.astype("float64")

    # lower_threshold and upper_threshold are the materialized numerical
    # decision boundaries and must always exist for a selected policy.
    for column in (
        "lower_threshold",
        "upper_threshold",
    ):
        values = out[column].to_numpy(
            dtype=np.float64,
            copy=False,
        )

        if not np.isfinite(values).all():
            bad = out.loc[
                ~np.isfinite(values),
                [
                    "model_id",
                    "random_state",
                    "decision_scope",
                    column,
                ],
            ].head(10)

            raise RuntimeError(
                f"{name}.{column} contains non-finite materialized "
                "thresholds: "
                f"{bad.to_dict('records')}."
            )

    # ------------------------------------------------------------------
    # Natural key.
    # ------------------------------------------------------------------
    if out.duplicated(list(_THRESHOLD_KEY)).any():
        duplicates = (
            out.loc[
                out.duplicated(
                    list(_THRESHOLD_KEY),
                    keep=False,
                ),
                list(_THRESHOLD_KEY),
            ]
            .drop_duplicates()
            .head(10)
        )

        raise RuntimeError(
            f"{name} duplicates {_THRESHOLD_KEY}: "
            f"{duplicates.to_dict('records')}."
        )

    return out.reset_index(drop=True)

def assert_supporting_only(frame: pd.DataFrame, *, name: str) -> None:
    """Reject a supporting diagnostic that advertises selection influence."""
    if frame is None or frame.empty or "selection_influence" not in frame.columns:
        return
    if frame["selection_influence"].fillna(True).astype(bool).any():
        raise RuntimeError(f"{name} must be supporting-only.")

def _expected_track_ids() -> tuple[str, ...]:
    return tuple(map(str, expcfg.SIMCA_ROBUSTNESS_TRACK_IDS))

def _reindex_contract(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    requested = list(columns)
    out = frame.copy()
    missing = [column for column in requested if column not in out.columns]
    if missing:
        filler = pd.DataFrame(
            {column: pd.Series(pd.NA, index=out.index, dtype="object") for column in missing},
            index=out.index,
        )
        out = pd.concat([out, filler], axis=1)
    return out.loc[:, requested].reset_index(drop=True)

def _assert_group_invariant(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    name: str,
) -> None:
    columns = [column for column in value_columns if column in frame.columns]
    if frame.empty or not columns:
        return
    varying = (
        frame.groupby(list(group_columns), dropna=False, sort=False)[columns]
        .nunique(dropna=False)
        .gt(1)
    )
    if varying.any(axis=None):
        examples = varying.stack().loc[lambda values: values].index.tolist()[:10]
        raise RuntimeError(
            f"{name} has non-invariant values inside its natural groups: "
            f"{examples}."
        )

def _preprocessing_step_count(value: Any) -> int:
    steps = parse_preprocessing_steps(value)
    return int(
        sum(str(step).strip().lower() not in {"", "raw", "none"} for step in steps)
    )

def _score_columns(frame: pd.DataFrame) -> list[str]:
    forbidden = set(expcfg.ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS)
    return sorted(forbidden.intersection(frame.columns))

def _expected_scopes(projection_level: str) -> tuple[str, ...]:
    return (
        ("direct", "pixel_to_object")
        if str(projection_level) == "pixel_projection"
        else ("direct",)
    )

def _append_finite_long_part(
    parts: list[pd.DataFrame],
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    current = frame.copy()
    current["value"] = pd.to_numeric(current["value"], errors="coerce")
    current = current.loc[np.isfinite(current["value"].to_numpy(dtype=float))]
    if not current.empty:
        parts.append(current)

def _model_catalog_metadata(
    validation_executions: pd.DataFrame,
    model_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Return one canonical metadata row per model and verify 03B/04C identity."""
    validate_simca_table_columns(
        model_catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        table_name="03B model_catalog",
    )
    catalog = model_catalog.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS)
    ].copy()
    catalog["model_id"] = catalog["model_id"].astype(str)
    if catalog["model_id"].duplicated().any():
        raise RuntimeError("model_catalog.model_id must be unique.")

    execution_models = validation_executions.copy()
    execution_models["model_id"] = execution_models["model_id"].astype(str)
    execution_ids = set(execution_models["model_id"])
    catalog_ids = set(catalog["model_id"])
    missing = sorted(execution_ids - catalog_ids)
    if missing:
        raise RuntimeError(
            "04C contains model_id values absent from 03B model_catalog: "
            f"{missing[:20]}."
        )

    # Every column shared by the compact 04C registry and the 03B catalog must
    # still describe exactly the same scientific model.
    shared = [
        column
        for column in expcfg.SIMCA_MODEL_PARAMETER_COLUMNS
        if column in execution_models.columns and column in catalog.columns
    ]
    if shared:
        left = (
            execution_models[["model_id", *shared]]
            .drop_duplicates()
            .sort_values("model_id", kind="mergesort")
        )
        right = catalog[["model_id", *shared]].copy()
        comparison = left.merge(
            right,
            on="model_id",
            how="left",
            suffixes=("__04c", "__03b"),
            validate="one_to_one",
        )
        for column in shared:
            a = comparison[f"{column}__04c"].astype("string").fillna("<NA>")
            b = comparison[f"{column}__03b"].astype("string").fillna("<NA>")
            if not a.eq(b).all():
                bad = comparison.loc[
                    ~a.eq(b),
                    ["model_id", f"{column}__04c", f"{column}__03b"],
                ].head(10)
                raise RuntimeError(
                    f"03B/04C scientific metadata disagree for {column!r}: "
                    f"{bad.to_dict('records')}."
                )

    return catalog.loc[catalog["model_id"].isin(execution_ids)].reset_index(drop=True)



# ---------------------------------------------------------------------------
# 2. Strict 04C input validation and base selection members
# ---------------------------------------------------------------------------


def validate_robustness_inputs(
    validation_metrics: pd.DataFrame,
    validation_guardrails: pd.DataFrame,
    validation_executions: pd.DataFrame,
    model_catalog: pd.DataFrame,
    spatial_component_metrics: pd.DataFrame | None = None,
    *,
    require_complete_tracks: bool = True,
) -> dict[str, pd.DataFrame]:
    """Validate the current 03B/04C contracts; no legacy identifier fallback."""
    if validation_metrics is None or validation_metrics.empty:
        raise ValueError("04C validation_metrics is empty.")
    if validation_guardrails is None or validation_guardrails.empty:
        raise ValueError("04C validation_guardrails is empty.")
    if validation_executions is None or validation_executions.empty:
        raise ValueError("04C validation execution registry is empty.")

    metrics = assert_no_test_stage_inputs(validation_metrics.copy())
    guardrails = assert_no_test_stage_inputs(validation_guardrails.copy())
    executions = assert_no_test_stage_inputs(validation_executions.copy())
    spatial = (
        pd.DataFrame(columns=expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS)
        if spatial_component_metrics is None
        else assert_no_test_stage_inputs(spatial_component_metrics.copy())
    )

    validate_simca_table_columns(
        metrics,
        expcfg.SIMCA_VALIDATION_METRIC_COLUMNS,
        table_name="04C validation_metrics",
    )
    validate_simca_table_columns(
        guardrails,
        expcfg.SIMCA_VALIDATION_GUARDRAIL_COLUMNS,
        table_name="04C validation_guardrails",
    )
    validate_simca_table_columns(
        executions,
        expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS,
        table_name="04C validation_executions",
    )
    if not spatial.empty:
        validate_simca_table_columns(
            spatial,
            expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS,
            table_name="04C spatial_component_metrics",
        )

    for frame, name in (
        (metrics, "validation_metrics"),
        (guardrails, "validation_guardrails"),
        (executions, "validation_executions"),
        (spatial, "spatial_component_metrics"),
    ):
        forbidden = _score_columns(frame)
        if forbidden:
            raise RuntimeError(
                f"Composite score/rank columns are forbidden in notebook 05 "
                f"({name}): {forbidden}."
            )
        if frame.empty:
            continue
        frame["model_id"] = frame["model_id"].astype(str)
        frame["random_state"] = pd.to_numeric(
            frame["random_state"], errors="raise"
        ).astype(int)
        frame["track_id"] = frame["track_id"].astype(str)

    run_key = ["model_id", "random_state"]
    if executions.duplicated(run_key).any():
        raise RuntimeError(
            "04C validation_executions duplicates (model_id, random_state)."
        )

    metric_key = [
        "model_id",
        "random_state",
        "track_id",
        "decision_scope",
        "map_variant",
        "aggregation_level",
        "group_id",
        "metric",
    ]
    if metrics.duplicated(metric_key).any():
        raise RuntimeError("04C validation_metrics duplicates its natural key.")

    guardrail_key = [
        "model_id",
        "random_state",
        "track_id",
        "decision_scope",
        "scope",
        "metric",
    ]
    if guardrails.duplicated(guardrail_key).any():
        raise RuntimeError("04C validation_guardrails duplicates its natural key.")

    known_runs = set(executions[run_key].itertuples(index=False, name=None))
    for frame, name in (
        (metrics, "validation_metrics"),
        (guardrails, "validation_guardrails"),
        (spatial, "spatial_component_metrics"),
    ):
        if frame.empty:
            continue
        observed = set(
            frame[run_key].drop_duplicates().itertuples(index=False, name=None)
        )
        unknown = observed - known_runs
        if unknown:
            raise RuntimeError(
                f"{name} contains execution keys absent from the canonical "
                f"registry: {sorted(unknown)[:20]}."
            )

    _assert_group_invariant(
        executions,
        ["model_id"],
        [
            "track_id",
            "decision_mode",
            "projection_level",
            "matrix_method",
            "projection_matrix_method",
            "m",
            "balanced_pixel_strategy",
            "preprocessing_steps",
            "rule_variant",
            "limit_source",
            "n_components",
            "alpha",
            "sg_window_length",
            "sg_polyorder",
            "position_dilation_radius",
            "eligibility_status",
            "downstream_status",
        ],
        name="04C validation_executions",
    )

    expected_tracks = set(_expected_track_ids())
    observed_tracks = set(executions["track_id"].astype(str))
    extra_tracks = sorted(observed_tracks - expected_tracks)
    if extra_tracks:
        raise RuntimeError(f"Unknown track_id values in notebook 05: {extra_tracks}.")
    if require_complete_tracks and observed_tracks != expected_tracks:
        raise RuntimeError(
            "The base 04C review requires all E1-E8 tracks: "
            f"missing={sorted(expected_tracks - observed_tracks)}."
        )

    unknown_scopes = sorted(
        set(metrics["decision_scope"].astype(str))
        - set(expcfg.SIMCA_ROBUSTNESS_DECISION_SCOPES)
    )
    if unknown_scopes:
        raise RuntimeError(f"Unknown decision scopes in 04C metrics: {unknown_scopes}.")

    catalog = _model_catalog_metadata(executions, model_catalog)
    return {
        "metrics": metrics,
        "guardrails": guardrails,
        "executions": executions,
        "catalog": catalog,
        "spatial": spatial,
    }

def _execution_guardrail_status(guardrails: pd.DataFrame) -> pd.DataFrame:
    work = guardrails.copy()
    key = list(expcfg.SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS)
    _assert_group_invariant(
        work,
        key,
        ["eligibility_status", "downstream_status", "candidate_status"],
        name="04C guardrails",
    )

    work["_blocking"] = work["is_blocking"].fillna(False).astype(bool)
    work["_blocking_failure"] = (
        work["_blocking"] & ~work["check_status"].astype(str).eq("pass")
    )
    work["_technical_error"] = work["check_status"].astype(str).isin(
        {"technical_error", "technical_failure"}
    )

    out = (
        work.groupby(key, as_index=False, sort=False, dropna=False)
        .agg(
            eligibility_status=("eligibility_status", "first"),
            downstream_status=("downstream_status", "first"),
            candidate_status=("candidate_status", "first"),
            n_guardrail_checks=("metric", "size"),
            n_blocking_checks=("_blocking", "sum"),
            n_blocking_failures=("_blocking_failure", "sum"),
            n_technical_errors=("_technical_error", "sum"),
        )
    )
    for column in (
        "n_guardrail_checks",
        "n_blocking_checks",
        "n_blocking_failures",
        "n_technical_errors",
    ):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)

    out["scope_calculable"] = (
        ~out["candidate_status"].astype(str).eq("technical_failure")
        & out["n_technical_errors"].eq(0)
    )
    out["all_blocking_checks_pass"] = out["n_blocking_failures"].eq(0)
    out["scope_protocol_pass"] = (
        out["candidate_status"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_PROTOCOL_CANDIDATE_STATUSES
        )
        & out["eligibility_status"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_SUPPORTED_ELIGIBILITY_STATUSES
        )
        & out["downstream_status"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_SUPPORTED_DOWNSTREAM_STATUSES
        )
        & out["all_blocking_checks_pass"].astype(bool)
    )
    return out.sort_values(key, kind="mergesort").reset_index(drop=True)

def _overall_scope_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    key = list(expcfg.SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS)
    work = metrics.loc[
        metrics["map_variant"].astype(str).eq(
            expcfg.SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT
        )
        & metrics["aggregation_level"].astype(str).eq("overall")
        & metrics["group_id"].astype(str).eq("all")
        & metrics["metric"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=key)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    natural = [*key, "metric"]
    if work.duplicated(natural).any():
        raise RuntimeError("Overall validation metrics duplicate execution/scope/metric.")
    wide = work.pivot(index=key, columns="metric", values="value").reset_index()
    wide.columns.name = None
    return wide

def _worst_image_scope_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    key = list(expcfg.SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS)
    work = metrics.loc[
        metrics["map_variant"].astype(str).eq(
            expcfg.SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT
        )
        & metrics["aggregation_level"].astype(str).eq("source_image")
        & metrics["status"].astype(str).eq("calculable")
        & metrics["metric"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_WORST_IMAGE_METRIC_NAMES
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=key)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    group_key = [*key, "metric"]
    rows: list[pd.DataFrame] = []
    for direction, reducer in (
        ("minimize", finite_max),
        ("maximize", finite_min),
    ):
        names = {
            metric
            for metric, configured in expcfg.SIMCA_ROBUSTNESS_METRIC_DIRECTIONS.items()
            if configured == direction
        }
        current = work.loc[work["metric"].astype(str).isin(names)]
        if not current.empty:
            rows.append(
                current.groupby(
                    group_key, as_index=False, sort=False, dropna=False
                ).agg(value=("value", reducer))
            )
    known = {
        metric
        for metric in expcfg.SIMCA_ROBUSTNESS_METRIC_DIRECTIONS
    }
    other = work.loc[~work["metric"].astype(str).isin(known)]
    if not other.empty:
        rows.append(
            other.groupby(group_key, as_index=False, sort=False, dropna=False)
            .agg(value=("value", finite_mean))
        )
    if not rows:
        return pd.DataFrame(columns=key)
    reduced = pd.concat(rows, ignore_index=True, sort=False)
    reduced["metric"] = "worst_image__" + reduced["metric"].astype(str)
    wide = reduced.pivot(index=key, columns="metric", values="value").reset_index()
    wide.columns.name = None
    return wide

def _spatial_direct_metrics(spatial: pd.DataFrame) -> pd.DataFrame:
    run_key = list(expcfg.SIMCA_ROBUSTNESS_EXECUTION_KEY_COLUMNS)
    if spatial is None or spatial.empty:
        return pd.DataFrame(columns=[*run_key, "decision_scope"])
    work = spatial.loc[
        spatial["aggregation_level"].astype(str).eq("overall")
        & spatial["map_variant"].astype(str).eq(
            expcfg.SIMCA_ROBUSTNESS_SPATIAL_MAP_VARIANT
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=[*run_key, "decision_scope"])
    metrics = [
        metric
        for metric in expcfg.SIMCA_ROBUSTNESS_SPATIAL_METRICS
        if metric in work.columns
    ]
    if not metrics:
        return pd.DataFrame(columns=[*run_key, "decision_scope"])
    work[metrics] = work[metrics].apply(pd.to_numeric, errors="coerce")
    reduced = (
        work.groupby(run_key, as_index=False, sort=False, dropna=False)[metrics]
        .mean(numeric_only=True)
    )
    reduced = reduced.rename(
        columns={metric: f"spatial__{metric}" for metric in metrics}
    )
    reduced["decision_scope"] = "direct"
    return reduced

def _build_selection_members_from_validated(
    validated: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    metrics = validated["metrics"]
    guardrails = validated["guardrails"]
    executions = validated["executions"].copy()
    catalog = validated["catalog"].copy()
    spatial = validated["spatial"]

    scope_status = _execution_guardrail_status(guardrails)

    expected_rows: list[dict[str, Any]] = []
    for row in executions.to_dict("records"):
        for scope in _expected_scopes(str(row["projection_level"])):
            expected_rows.append(
                {
                    "model_id": str(row["model_id"]),
                    "random_state": int(row["random_state"]),
                    "track_id": str(row["track_id"]),
                    "decision_scope": scope,
                }
            )
    expected = pd.DataFrame(expected_rows)
    scope_key = list(expcfg.SIMCA_ROBUSTNESS_EXECUTION_SCOPE_KEY_COLUMNS)
    coverage = expected.merge(
        scope_status[scope_key].drop_duplicates(),
        on=scope_key,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not coverage["_merge"].eq("both").all():
        bad = coverage.loc[~coverage["_merge"].eq("both")].head(20)
        raise RuntimeError(
            "04C guardrail decision-scope coverage disagrees with the execution "
            f"registry: {bad.to_dict('records')}."
        )

    run_metadata = executions.merge(
        catalog,
        on="model_id",
        how="left",
        suffixes=("", "__catalog"),
        validate="many_to_one",
    )
    # Keep the 04C execution columns when duplicated; add only missing catalog
    # columns. This prevents accidental replacement of frozen execution state.
    duplicate_catalog_columns = [
        column
        for column in run_metadata.columns
        if column.endswith("__catalog")
    ]
    run_metadata = run_metadata.drop(columns=duplicate_catalog_columns)

    # eligibility/downstream state is taken from the 04C guardrail scope row,
    # not duplicated from the execution registry.
    run_metadata = run_metadata.drop(
        columns=["eligibility_status", "downstream_status"],
        errors="ignore",
    )
    member = scope_status.merge(
        run_metadata,
        on=["model_id", "random_state", "track_id"],
        how="left",
        validate="many_to_one",
    )
    member = member.merge(
        _overall_scope_metrics(metrics),
        on=scope_key,
        how="left",
        validate="one_to_one",
    )
    member = member.merge(
        _worst_image_scope_metrics(metrics),
        on=scope_key,
        how="left",
        validate="one_to_one",
    )
    spatial_direct = _spatial_direct_metrics(spatial)
    if not spatial_direct.empty:
        member = member.merge(
            spatial_direct,
            on=scope_key,
            how="left",
            validate="one_to_one",
        )

    member["is_stochastic"] = stochastic_model_mask(member).astype(bool)
    member["preprocessing_step_count"] = member["preprocessing_steps"].map(
        _preprocessing_step_count
    ).astype(int)

    if member.duplicated(scope_key).any():
        raise RuntimeError("Selection members duplicate their natural scope key.")
    return _reindex_contract(
        member.sort_values(scope_key, kind="mergesort"),
        expcfg.SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS,
    )

def build_seed_metrics(
    validation_metrics: pd.DataFrame,
    validation_guardrails: pd.DataFrame,
    validation_executions: pd.DataFrame,
    model_catalog: pd.DataFrame,
    spatial_component_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the scope-level metric table for any valid execution subset.

    This is used for the additional-seed stress test. It performs no Pareto
    filtering and no model selection.
    """
    validated = validate_robustness_inputs(
        validation_metrics,
        validation_guardrails,
        validation_executions,
        model_catalog,
        spatial_component_metrics,
        require_complete_tracks=False,
    )
    return _build_selection_members_from_validated(validated)

def selection_members_to_long_metrics(member: pd.DataFrame) -> pd.DataFrame:
    """Normalize selection-member metrics to one finite value per execution/metric."""
    model_seed = ["model_id", "track_id", "random_state"]
    parts: list[pd.DataFrame] = []

    regular = [
        column
        for column in expcfg.SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES
        if column in member.columns
    ]
    if regular:
        long = member.melt(
            id_vars=[*model_seed, "decision_scope"],
            value_vars=regular,
            var_name="metric_base",
            value_name="value",
        )
        long["metric"] = (
            long["decision_scope"].astype(str)
            + "__"
            + long["metric_base"].astype(str)
        )
        _append_finite_long_part(parts, long[[*model_seed, "metric", "value"]])

    worst = [
        column
        for column in expcfg.SIMCA_ROBUSTNESS_MEMBER_WORST_IMAGE_COLUMNS
        if column in member.columns
    ]
    if worst:
        long = member.melt(
            id_vars=[*model_seed, "decision_scope"],
            value_vars=worst,
            var_name="metric_base",
            value_name="value",
        )
        long["metric"] = (
            long["decision_scope"].astype(str)
            + "__"
            + long["metric_base"].astype(str)
        )
        _append_finite_long_part(parts, long[[*model_seed, "metric", "value"]])

    spatial = [
        column
        for column in expcfg.SIMCA_ROBUSTNESS_MEMBER_SPATIAL_COLUMNS
        if column in member.columns
    ]
    if spatial:
        direct = member.loc[member["decision_scope"].astype(str).eq("direct")]
        long = direct.melt(
            id_vars=model_seed,
            value_vars=spatial,
            var_name="metric",
            value_name="value",
        )
        _append_finite_long_part(parts, long[[*model_seed, "metric", "value"]])

    if not parts:
        return pd.DataFrame(columns=[*model_seed, "metric", "value"])
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    key = [*model_seed, "metric"]
    if out.duplicated(key).any():
        raise RuntimeError("Selection members duplicate an execution/metric value.")
    return out.reset_index(drop=True)

def aggregate_repeated_execution_metrics(
    selection_members: pd.DataFrame,
    *,
    include_statistics: bool = False,
) -> pd.DataFrame:
    """Aggregate repeated executions once, using the canonical metric directions.

    The conservative value is max across seeds for minimized metrics, min for
    maximized metrics, and mean only for directionless descriptive metrics.
    This kernel is shared by the official 05 selection units and the supporting
    Pareto jackknife, preventing two implementations of seed aggregation.
    """
    validate_simca_table_columns(
        selection_members,
        expcfg.SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS,
        table_name="selection members",
    )
    long = selection_members_to_long_metrics(selection_members)
    if long.empty:
        return pd.DataFrame(columns=["model_id", "track_id", "n_random_states"])

    model_key = ["model_id", "track_id"]
    stats = (
        long.groupby([*model_key, "metric"], as_index=False, sort=False, dropna=False)
        .agg(
            mean=("value", finite_mean),
            std=("value", finite_std),
            min=("value", finite_min),
            max=("value", finite_max),
            n_finite=("value", "size"),
        )
    )
    stats.loc[stats["n_finite"].eq(1), "std"] = 0.0
    direction = stats["metric"].map(metric_direction)
    stats["conservative"] = np.select(
        [direction.eq("minimize"), direction.eq("maximize")],
        [stats["max"], stats["min"]],
        default=stats["mean"],
    )

    n_states = (
        selection_members.groupby(model_key, as_index=False, sort=False, dropna=False)
        .agg(n_random_states=("random_state", "nunique"))
    )
    conservative = stats.pivot(
        index=model_key,
        columns="metric",
        values="conservative",
    )
    conservative.columns.name = None
    out = n_states.merge(
        conservative.reset_index(),
        on=model_key,
        how="left",
        validate="one_to_one",
    )

    if include_statistics:
        for statistic in ("mean", "std", "min", "max"):
            wide = stats.pivot(index=model_key, columns="metric", values=statistic)
            wide.columns = [f"{statistic}__{column}" for column in wide.columns]
            out = out.merge(
                wide.reset_index(),
                on=model_key,
                how="left",
                validate="one_to_one",
            )
    return out.reset_index(drop=True)

def build_selection_unit_metrics(
    validation_metrics: pd.DataFrame,
    validation_guardrails: pd.DataFrame,
    validation_executions: pd.DataFrame,
    model_catalog: pd.DataFrame,
    spatial_component_metrics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one model unit per E-track and preserve the complete base 04C panel."""
    validated = validate_robustness_inputs(
        validation_metrics,
        validation_guardrails,
        validation_executions,
        model_catalog,
        spatial_component_metrics,
        require_complete_tracks=True,
    )
    member = _build_selection_members_from_validated(validated)
    model_key = list(expcfg.SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS)

    run_status = (
        member.groupby(
            [*model_key, "random_state"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            execution_calculable=("scope_calculable", "all"),
            execution_protocol_supported=("scope_protocol_pass", "all"),
            all_04c_blocking_guardrails_pass=("all_blocking_checks_pass", "all"),
        )
    )

    invariant = [
        column
        for column in expcfg.SIMCA_ROBUSTNESS_MODEL_INVARIANT_COLUMNS
        if column not in model_key
    ]
    _assert_group_invariant(
        member,
        model_key,
        [*invariant, "is_stochastic", "preprocessing_step_count"],
        name="Notebook-05 selection members",
    )
    metadata = (
        member.groupby(model_key, as_index=False, sort=False, dropna=False)
        .agg(
            **{column: (column, "first") for column in invariant},
            is_stochastic=("is_stochastic", "first"),
            preprocessing_step_count=("preprocessing_step_count", "first"),
        )
    )

    run_summary = (
        run_status.groupby(model_key, as_index=False, sort=False, dropna=False)
        .agg(
            all_execution_calculable=("execution_calculable", "all"),
            all_execution_protocol_supported=("execution_protocol_supported", "all"),
            all_04c_blocking_guardrails_pass=("all_04c_blocking_guardrails_pass", "all"),
            _observed_states=(
                "random_state",
                lambda values: tuple(sorted(set(map(int, values)))),
            ),
        )
    )
    metadata = metadata.merge(run_summary, on=model_key, how="left", validate="one_to_one")

    # Every 04C model, deterministic or stochastic, must preserve the complete
    # frozen base execution panel. Deterministic rows are not interpreted as
    # independent stochastic repetitions; they are retained for lineage only.
    expected_states = set(map(int, expcfg.SIMCA_ROBUSTNESS_BASE_RANDOM_STATES))
    observed_sets = metadata["_observed_states"].map(lambda values: set(map(int, values)))
    metadata["n_expected_random_states"] = len(expected_states)
    metadata["observed_random_states_json"] = observed_sets.map(
        lambda values: canonical_json(sorted(values))
    )
    metadata["missing_random_states_json"] = observed_sets.map(
        lambda values: canonical_json(sorted(expected_states - values))
    )
    metadata["all_expected_random_states_present"] = observed_sets.map(
        lambda values: values == expected_states
    )
    metadata["seed_requirement_satisfied"] = metadata[
        "all_expected_random_states_present"
    ].astype(bool)
    metadata["model_diagnostic_eligible"] = metadata[
        "all_execution_calculable"
    ].astype(bool)
    metadata["model_protocol_eligible_pre_stability"] = (
        metadata["seed_requirement_satisfied"].astype(bool)
        & metadata["all_execution_protocol_supported"].astype(bool)
        & metadata["all_04c_blocking_guardrails_pass"].astype(bool)
        & metadata["eligibility_status"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_SUPPORTED_ELIGIBILITY_STATUSES
        )
        & metadata["downstream_status"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_SUPPORTED_DOWNSTREAM_STATUSES
        )
    )
    metadata = metadata.drop(columns="_observed_states")

    aggregated = aggregate_repeated_execution_metrics(
        member,
        include_statistics=True,
    )
    if aggregated.empty:
        raise RuntimeError("No finite base-04C metric is available for notebook 05.")
    unit = metadata.merge(
        aggregated,
        on=model_key,
        how="left",
        validate="one_to_one",
    )
    unit = _reindex_contract(unit, expcfg.SIMCA_ROBUSTNESS_SELECTION_UNIT_COLUMNS)
    if unit.duplicated(model_key).any():
        raise RuntimeError("Selection units duplicate (model_id, track_id).")
    return unit, member

def build_pareto_diagnostics(
    selection_units: pd.DataFrame,
    *,
    epsilon: float = expcfg.SIMCA_ROBUSTNESS_PARETO_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute diagnostic and protocol Pareto fronts independently in E1-E8."""
    if bool(expcfg.SIMCA_ROBUSTNESS_ALLOW_CROSS_TRACK_SELECTION):
        raise RuntimeError("Notebook 05 must forbid cross-track selection.")
    if bool(expcfg.SIMCA_ROBUSTNESS_RECOMPUTE_PARETO_AFTER_ADDITIONAL_SEEDS):
        raise RuntimeError(
            "The notebook-05 Pareto must remain frozen on the common base panel."
        )
    validate_simca_table_columns(
        selection_units,
        expcfg.SIMCA_ROBUSTNESS_SELECTION_UNIT_COLUMNS,
        table_name="validation selection units",
    )
    annotated = selection_units.copy()
    model_key = list(expcfg.SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS)
    if annotated.duplicated(model_key).any():
        raise RuntimeError("Pareto input duplicates its scientific model key.")

    for column, default in (
        ("diagnostic_pareto_eligible", False),
        ("is_diagnostic_pareto", False),
        ("diagnostic_dominated_by_model_id", ""),
        ("protocol_pareto_eligible", False),
        ("is_protocol_pareto", False),
        ("protocol_dominated_by_model_id", ""),
        ("pareto_status", "not_evaluated"),
        ("pareto_exclusion_reason", ""),
    ):
        annotated[column] = default

    audit_rows: list[dict[str, Any]] = []
    for track_id in _expected_track_ids():
        group = annotated.loc[annotated["track_id"].astype(str).eq(track_id)].copy()
        if group.empty:
            continue
        spec = expcfg.SIMCA_ROBUSTNESS_PARETO_OBJECTIVES[track_id]
        minimize = tuple(map(str, spec.get("minimize", ())))
        maximize = tuple(map(str, spec.get("maximize", ())))
        objectives = [*minimize, *maximize]
        missing = sorted(set(objectives) - set(group.columns))
        if missing:
            raise KeyError(f"Missing Pareto objectives for {track_id}: {missing}.")
        numeric = group[objectives].apply(pd.to_numeric, errors="coerce")
        finite = pd.Series(
            np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
            index=group.index,
        )
        pools = {
            "diagnostic": finite & group["model_diagnostic_eligible"].astype(bool),
            "protocol": finite
            & group["model_protocol_eligible_pre_stability"].astype(bool),
        }
        for pool_type, mask in pools.items():
            candidates = group.loc[mask].copy()
            if pool_type == "diagnostic":
                eligible_col = "diagnostic_pareto_eligible"
                front_col = "is_diagnostic_pareto"
                witness_col = "diagnostic_dominated_by_model_id"
            else:
                eligible_col = "protocol_pareto_eligible"
                front_col = "is_protocol_pareto"
                witness_col = "protocol_dominated_by_model_id"
            annotated.loc[candidates.index, eligible_col] = True
            if not candidates.empty:
                front, witness = pareto_front_with_witness(
                    candidates,
                    minimize_cols=minimize,
                    maximize_cols=maximize,
                    epsilon=float(epsilon),
                )
                annotated.loc[candidates.index, front_col] = front.to_numpy(bool)
                annotated.loc[candidates.index, witness_col] = witness.to_numpy(object)

            for index, row in group.iterrows():
                is_candidate = bool(mask.loc[index])
                is_pareto = bool(annotated.loc[index, front_col]) if is_candidate else False
                if not finite.loc[index]:
                    reason = "non_finite_pareto_objective"
                elif pool_type == "diagnostic" and not bool(row["model_diagnostic_eligible"]):
                    reason = "not_diagnostic_eligible"
                elif pool_type == "protocol" and not bool(
                    row["model_protocol_eligible_pre_stability"]
                ):
                    reason = "not_protocol_eligible_pre_stability"
                elif is_pareto:
                    reason = "pareto_front"
                else:
                    reason = "pareto_dominated"
                objective_values = {
                    column: (
                        None
                        if not np.isfinite(
                            pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                        )
                        else float(row[column])
                    )
                    for column in objectives
                }
                audit_rows.append(
                    {
                        "track_id": track_id,
                        "model_id": str(row["model_id"]),
                        "pool_type": pool_type,
                        "is_candidate": is_candidate,
                        "is_pareto": is_pareto,
                        "dominated_by_model_id": (
                            str(annotated.loc[index, witness_col]) if is_candidate else ""
                        ),
                        "reason_code": reason,
                        "pareto_minimize_json": canonical_json(list(minimize)),
                        "pareto_maximize_json": canonical_json(list(maximize)),
                        "objective_values_json": canonical_json(objective_values),
                    }
                )

        protocol_candidate = pools["protocol"]
        protocol_front = annotated.loc[group.index, "is_protocol_pareto"].astype(bool)
        annotated.loc[group.index, "pareto_status"] = np.where(
            protocol_front,
            "protocol_pareto",
            np.where(protocol_candidate, "protocol_dominated", "not_protocol_candidate"),
        )
        annotated.loc[group.index, "pareto_exclusion_reason"] = np.where(
            ~finite,
            "non_finite_pareto_objective",
            np.where(
                ~protocol_candidate,
                "not_protocol_eligible_pre_stability",
                np.where(protocol_front, "", "pareto_dominated"),
            ),
        )

    candidates = _reindex_contract(
        annotated,
        expcfg.SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS,
    )
    audit = _reindex_contract(
        pd.DataFrame(audit_rows),
        expcfg.SIMCA_ROBUSTNESS_PARETO_AUDIT_COLUMNS,
    )
    return candidates, audit



# Backward-compatible public name used by earlier notebook-05 drafts.
validate_simca_robustness_inputs = validate_robustness_inputs




# ---------------------------------------------------------------------------
# 3. Additional-seed execution, disagreement and stability
# ---------------------------------------------------------------------------


def build_robustness_seed_execution_registry(
    model_catalog: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    *,
    existing_executions: pd.DataFrame | None = None,
    random_states: Sequence[int] = expcfg.SIMCA_ROBUSTNESS_ADDITIONAL_RANDOM_STATES,
) -> pd.DataFrame:
    """Create only additional stochastic executions for protocol-Pareto models."""
    if not bool(expcfg.SIMCA_ROBUSTNESS_RUN_ADDITIONAL_SEEDS):
        return pd.DataFrame(columns=expcfg.INTERNAL_CALIBRATION_EXECUTION_COLUMNS)
    validate_simca_table_columns(
        pareto_candidates,
        expcfg.SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS,
        table_name="validation Pareto candidates",
    )
    requested = pareto_candidates.loc[
        pareto_candidates["is_protocol_pareto"].fillna(False).astype(bool),
        ["model_id", "track_id"],
    ].drop_duplicates()
    if requested.empty:
        return pd.DataFrame(columns=expcfg.INTERNAL_CALIBRATION_EXECUTION_COLUMNS)

    catalog = model_catalog.copy()
    validate_simca_table_columns(
        catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        table_name="03B model_catalog",
    )
    catalog["model_id"] = catalog["model_id"].astype(str)
    candidates = catalog.loc[catalog["model_id"].isin(set(requested["model_id"].astype(str)))].copy()
    candidates = candidates.loc[stochastic_model_mask(candidates)].copy()
    model_ids = candidates["model_id"].astype(str).drop_duplicates().tolist()
    if not model_ids:
        return pd.DataFrame(columns=expcfg.INTERNAL_CALIBRATION_EXECUTION_COLUMNS)

    states = normalize_integer_sequence(
        random_states,
        name="random_states",
        allow_empty=False,
    )
    return build_additional_seed_execution_registry(
        model_catalog,
        model_ids,
        states,
        stochastic_only=True,
        existing_executions=existing_executions,
    )

def _threshold_registry(thresholds: pd.DataFrame) -> pd.DataFrame:
    validate_simca_table_columns(
        thresholds,
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
        table_name="materialized thresholds",
    )
    out = thresholds.copy()
    out["model_id"] = out["model_id"].astype(str)
    out["random_state"] = pd.to_numeric(out["random_state"], errors="raise").astype(int)
    out["decision_scope"] = out["decision_scope"].astype(str)
    key = list(expcfg.SIMCA_ROBUSTNESS_SEED_THRESHOLD_KEY_COLUMNS)
    if out.duplicated(key).any():
        raise RuntimeError("Threshold registry duplicates its natural key.")
    return out

def compute_seed_decision_disagreement(
    validation_executions: pd.DataFrame,
    thresholds: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Measure disagreement across seeds for the same model and same entity."""
    validate_simca_table_columns(
        validation_executions,
        expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS,
        table_name="seed validation executions",
    )
    executions = validation_executions.copy()
    executions["model_id"] = executions["model_id"].astype(str)
    executions["random_state"] = pd.to_numeric(
        executions["random_state"], errors="raise"
    ).astype(int)
    executions["track_id"] = executions["track_id"].astype(str)
    policy = _threshold_registry(thresholds)
    policies = executions.merge(
        policy[
            [
                "model_id",
                "random_state",
                "decision_scope",
                "lower_threshold",
                "upper_threshold",
            ]
        ],
        on=["model_id", "random_state"],
        how="left",
        validate="one_to_many",
    )
    if policies[["lower_threshold", "upper_threshold"]].isna().any().any():
        raise RuntimeError("A seed execution has no materialized numeric threshold.")

    parts: list[pd.DataFrame] = []
    for row in policies.to_dict("records"):
        model_id = str(row["model_id"])
        random_state = int(row["random_state"])
        track_id = str(row["track_id"])
        scope = str(row["decision_scope"])
        projection_id = str(row["projection_id"])
        projection_level = str(row["projection_level"])
        decision_mode = str(row["decision_mode"])
        lower = float(row["lower_threshold"])
        upper = float(row["upper_threshold"])

        if scope == "direct":
            source = object_predictions if projection_level == "object_projection" else pixel_predictions
            if source is None or source.empty:
                continue
            observations = source.loc[
                source["projection_id"].astype(str).eq(projection_id)
            ].copy()
            if observations.empty:
                continue
            score = pd.to_numeric(observations["simca_margin"], errors="coerce")
            entity_columns = ["source_image", "object_id"]
            if projection_level == "pixel_projection":
                entity_columns.extend(["row", "col"])
        elif scope == "pixel_to_object" and projection_level == "pixel_projection":
            if pixel_predictions is None or pixel_predictions.empty:
                continue
            pixels = pixel_predictions.loc[
                pixel_predictions["projection_id"].astype(str).eq(projection_id)
            ].copy()
            if pixels.empty:
                continue
            observations = build_pixel_vote_table(
                pixels,
                group_columns=("source_image", "object_id"),
            )
            score = pd.to_numeric(observations["pixel_target_ratio"], errors="coerce")
            entity_columns = ["source_image", "object_id"]
        else:
            continue

        values = score.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite decision score for model={model_id}, seed={random_state}, "
                f"scope={scope}."
            )
        target, uncertain = apply_locked_margin_decision(
            values,
            decision_mode,
            direct_2way_threshold=lower,
            three_way_lower_threshold=lower,
            three_way_upper_threshold=upper,
        )
        current = observations[[*entity_columns, "truth"]].copy()
        current["model_id"] = model_id
        current["random_state"] = random_state
        current["track_id"] = track_id
        current["decision_scope"] = scope
        current["decision_code"] = np.where(
            target, 2, np.where(uncertain, 1, 0)
        ).astype(np.int8)
        parts.append(current)

    if not parts:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS)
    long = pd.concat(parts, ignore_index=True, sort=False)

    rows: list[dict[str, Any]] = []
    for (model_id, track_id, scope), group in long.groupby(
        ["model_id", "track_id", "decision_scope"],
        sort=False,
        dropna=False,
    ):
        entity_columns = ["source_image", "object_id"]
        if {"row", "col"}.issubset(group.columns) and group[["row", "col"]].notna().all().all():
            entity_columns.extend(["row", "col"])
        truth_check = group.groupby(entity_columns, dropna=False, sort=False)["truth"].nunique(dropna=False)
        if truth_check.gt(1).any():
            raise RuntimeError(
                f"Truth changes across seeds inside model={model_id}, scope={scope}."
            )
        entity = (
            group.groupby(entity_columns, as_index=False, dropna=False, sort=False)
            .agg(
                truth=("truth", "first"),
                n_seed_rows=("random_state", "nunique"),
                n_decisions=("decision_code", "nunique"),
            )
        )
        n_states = int(group["random_state"].nunique())
        complete = entity["n_seed_rows"].eq(n_states)
        comparable = entity.loc[complete].copy()
        target_entities = comparable.loc[comparable["truth"].astype(bool)]
        coverage_complete = bool(complete.all())
        rows.append(
            {
                "model_id": str(model_id),
                "track_id": str(track_id),
                "decision_scope": str(scope),
                "n_random_states": n_states,
                "n_entities": int(len(comparable)),
                "n_target_entities": int(len(target_entities)),
                "entity_seed_coverage_complete": coverage_complete,
                "decision_disagreement_rate": (
                    float(comparable["n_decisions"].gt(1).mean())
                    if len(comparable) else np.nan
                ),
                "target_decision_disagreement_rate": (
                    float(target_entities["n_decisions"].gt(1).mean())
                    if len(target_entities) else np.nan
                ),
                "disagreement_status": (
                    "calculable"
                    if coverage_complete and len(comparable)
                    else "not_estimable_incomplete_seed_entities"
                ),
            }
        )
    return _reindex_contract(
        pd.DataFrame(rows),
        expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS,
    )

def _stability_long(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    model_seed = ["model_id", "track_id", "random_state", "is_stochastic"]
    parts: list[pd.DataFrame] = []
    regular = [
        metric
        for metric in expcfg.SIMCA_ROBUSTNESS_VALIDATION_METRIC_NAMES
        if metric in seed_metrics.columns
        and metric in expcfg.SIMCA_ROBUSTNESS_STABILITY_LIMITS
    ]
    if regular:
        long = seed_metrics.melt(
            id_vars=[*model_seed, "decision_scope"],
            value_vars=regular,
            var_name="metric_base",
            value_name="value",
        )
        long["metric"] = long["decision_scope"].astype(str) + "__" + long["metric_base"].astype(str)
        parts.append(long[[*model_seed, "metric", "value"]])
    spatial = [
        f"spatial__{metric}"
        for metric in expcfg.SIMCA_ROBUSTNESS_SPATIAL_METRICS
        if metric in expcfg.SIMCA_ROBUSTNESS_STABILITY_LIMITS
        and f"spatial__{metric}" in seed_metrics.columns
    ]
    if spatial:
        direct = seed_metrics.loc[seed_metrics["decision_scope"].astype(str).eq("direct")]
        long = direct.melt(
            id_vars=model_seed,
            value_vars=spatial,
            var_name="metric",
            value_name="value",
        )
        parts.append(long[[*model_seed, "metric", "value"]])
    if not parts:
        return pd.DataFrame(columns=[*model_seed, "metric", "value"])
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.loc[np.isfinite(out["value"].to_numpy(dtype=float))].copy()

def summarize_random_state_stability_metrics(
    seed_metrics: pd.DataFrame,
    *,
    decision_disagreement: pd.DataFrame | None = None,
    expected_random_states: Sequence[int] = expcfg.SIMCA_ROBUSTNESS_RANDOM_STATES,
) -> pd.DataFrame:
    """Summarize stochastic seed stability with primary and supporting rules separated."""
    validate_simca_table_columns(
        seed_metrics,
        expcfg.SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS,
        table_name="robustness seed metrics",
    )
    work = seed_metrics.copy()
    work["random_state"] = pd.to_numeric(work["random_state"], errors="raise").astype(int)
    expected_stochastic = normalize_integer_sequence(
        expected_random_states,
        name="expected_random_states",
        allow_empty=False,
    )
    expected_deterministic = normalize_integer_sequence(
        expcfg.SIMCA_ROBUSTNESS_BASE_RANDOM_STATES,
        name="base_random_states",
        allow_empty=False,
    )
    long = _stability_long(work)
    if long.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_STABILITY_SUMMARY_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    for (model_id, track_id, metric), group in long.groupby(
        ["model_id", "track_id", "metric"], sort=False, dropna=False
    ):
        stochastic_values = group["is_stochastic"].dropna().astype(bool).unique()
        if len(stochastic_values) != 1:
            raise RuntimeError(f"Stochasticity changes inside model_id={model_id}.")
        is_stochastic = bool(stochastic_values[0])
        expected = expected_stochastic if is_stochastic else expected_deterministic
        observed = tuple(sorted(set(map(int, group["random_state"]))))
        values = pd.to_numeric(group["value"], errors="coerce")
        values = values[np.isfinite(values.to_numpy(float))]
        metric_name = str(metric)
        metric_base = metric_base_name(metric_name)
        direction = metric_direction(metric_name) or "descriptive"
        limits = expcfg.SIMCA_ROBUSTNESS_STABILITY_LIMITS.get(metric_base)
        if limits is None:
            raise KeyError(f"No stability limit is declared for {metric_base!r}.")

        mean = finite_mean(values)
        std = 0.0 if len(values) == 1 else finite_std(values)
        minimum = finite_min(values)
        maximum = finite_max(values)
        range_value = (
            float(maximum - minimum)
            if np.isfinite(minimum) and np.isfinite(maximum)
            else np.nan
        )
        if direction == "minimize":
            worst_value = maximum
        elif direction == "maximize":
            worst_value = minimum
        else:
            worst_value = np.nan
        worst_rows = group.loc[
            pd.to_numeric(group["value"], errors="coerce").eq(worst_value)
        ] if np.isfinite(worst_value) else group.iloc[0:0]
        worst_seed = (
            int(pd.to_numeric(worst_rows["random_state"], errors="raise").min())
            if len(worst_rows)
            else pd.NA
        )

        complete = set(observed) == set(expected)
        std_failed = bool(np.isfinite(std) and std > float(limits["max_std"]))
        range_failed = bool(
            np.isfinite(range_value) and range_value > float(limits["max_range"])
        )
        blocking_metrics = set(
            map(
                str,
                expcfg.SIMCA_ROBUSTNESS_BLOCKING_STABILITY_METRICS_BY_TRACK[
                    str(track_id)
                ],
            )
        )
        stability_role = (
            "blocking_primary_risk"
            if metric_name in blocking_metrics
            else "supporting_secondary"
        )
        if not complete:
            metric_status = "not_estimable_missing_seed"
        elif not is_stochastic:
            metric_status = "not_applicable_deterministic"
        elif std_failed or range_failed:
            metric_status = "unstable"
        else:
            metric_status = "robust"
        blocking_failure = bool(
            stability_role == "blocking_primary_risk"
            and metric_status == "unstable"
        )
        supporting_warning = bool(
            stability_role == "supporting_secondary"
            and metric_status == "unstable"
        )
        summary_rows.append(
            {
                "model_id": str(model_id),
                "track_id": str(track_id),
                "metric": metric_name,
                "metric_base": metric_base,
                "metric_direction": direction,
                "stability_role": stability_role,
                "is_stochastic": is_stochastic,
                "n_random_states": int(len(observed)),
                "n_expected_random_states": int(len(expected)),
                "observed_random_states_json": canonical_json(list(observed)),
                "missing_random_states_json": canonical_json(
                    sorted(set(expected) - set(observed))
                ),
                "all_expected_random_states_present": complete,
                "n_finite_values": int(len(values)),
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
                "range": range_value,
                "worst_value": worst_value,
                "worst_random_state": worst_seed,
                "max_std_limit": float(limits["max_std"]),
                "max_range_limit": float(limits["max_range"]),
                "std_limit_exceeded": std_failed,
                "range_limit_exceeded": range_failed,
                "blocking_metric_failure": blocking_failure,
                "supporting_metric_warning": supporting_warning,
                "stability_metric_status": metric_status,
            }
        )

    summary = pd.DataFrame(summary_rows)
    model_key = list(expcfg.SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS)
    if decision_disagreement is None or decision_disagreement.empty:
        disagreement = summary[model_key].drop_duplicates().copy()
        disagreement["decision_disagreement_rate"] = np.nan
        disagreement["target_decision_disagreement_rate"] = np.nan
    else:
        validate_simca_table_columns(
            decision_disagreement,
            expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_COLUMNS,
            table_name="seed decision disagreement",
        )
        disagreement = (
            decision_disagreement.groupby(
                model_key, as_index=False, sort=False, dropna=False
            )
            .agg(
                decision_disagreement_rate=("decision_disagreement_rate", finite_max),
                target_decision_disagreement_rate=(
                    "target_decision_disagreement_rate", finite_max
                ),
            )
        )

    model_status = (
        summary.groupby(model_key, as_index=False, sort=False, dropna=False)
        .agg(
            is_stochastic=("is_stochastic", "first"),
            all_expected_random_states_present=(
                "all_expected_random_states_present", "all"
            ),
            blocking_stability_failed=("blocking_metric_failure", "any"),
            supporting_stability_warning=("supporting_metric_warning", "any"),
        )
        .merge(disagreement, on=model_key, how="left", validate="one_to_one")
    )
    limits = expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_LIMITS
    stochastic = model_status["is_stochastic"].astype(bool)
    model_status["decision_disagreement_failed"] = stochastic & pd.to_numeric(
        model_status["decision_disagreement_rate"], errors="coerce"
    ).gt(float(limits["decision_disagreement_rate"]))
    model_status["target_decision_disagreement_failed"] = stochastic & pd.to_numeric(
        model_status["target_decision_disagreement_rate"], errors="coerce"
    ).gt(float(limits["target_decision_disagreement_rate"]))
    if bool(expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_IS_BLOCKING):
        model_status["blocking_stability_failed"] = (
            model_status["blocking_stability_failed"].astype(bool)
            | model_status["decision_disagreement_failed"].astype(bool)
            | model_status["target_decision_disagreement_failed"].astype(bool)
        )

    metric_flags = summary.loc[
        summary["std_limit_exceeded"].astype(bool)
        | summary["range_limit_exceeded"].astype(bool),
        [*model_key, "metric", "stability_role", "std_limit_exceeded", "range_limit_exceeded"],
    ].copy()
    flag_rows: list[dict[str, str]] = []
    for row in metric_flags.itertuples(index=False):
        if bool(row.std_limit_exceeded):
            flag_rows.append(
                {
                    "model_id": str(row.model_id),
                    "track_id": str(row.track_id),
                    "role": str(row.stability_role),
                    "flag": f"{row.metric}:std",
                }
            )
        if bool(row.range_limit_exceeded):
            flag_rows.append(
                {
                    "model_id": str(row.model_id),
                    "track_id": str(row.track_id),
                    "role": str(row.stability_role),
                    "flag": f"{row.metric}:range",
                }
            )
    flag_frame = pd.DataFrame(flag_rows)
    if flag_frame.empty:
        role_flags = model_status[model_key].copy()
        role_flags["blocking_stability_flags"] = ""
        role_flags["supporting_stability_flags"] = ""
    else:
        grouped_flags = (
            flag_frame.groupby([*model_key, "role"])["flag"]
            .agg(lambda values: ";".join(sorted(set(map(str, values)))))
            .unstack("role", fill_value="")
            .reset_index()
        )
        grouped_flags = grouped_flags.rename(
            columns={
                "blocking_primary_risk": "blocking_stability_flags",
                "supporting_secondary": "supporting_stability_flags",
            }
        )
        for column in ("blocking_stability_flags", "supporting_stability_flags"):
            if column not in grouped_flags:
                grouped_flags[column] = ""
        role_flags = model_status[model_key].merge(
            grouped_flags[
                [*model_key, "blocking_stability_flags", "supporting_stability_flags"]
            ],
            on=model_key,
            how="left",
            validate="one_to_one",
        ).fillna(
            {"blocking_stability_flags": "", "supporting_stability_flags": ""}
        )
    model_status = model_status.merge(
        role_flags,
        on=model_key,
        how="left",
        validate="one_to_one",
    )
    model_status.loc[
        model_status["decision_disagreement_failed"].astype(bool),
        "blocking_stability_flags",
    ] = model_status.loc[
        model_status["decision_disagreement_failed"].astype(bool),
        "blocking_stability_flags",
    ].map(lambda value: ";".join(filter(None, [str(value), "decision_disagreement_rate"])))
    model_status.loc[
        model_status["target_decision_disagreement_failed"].astype(bool),
        "blocking_stability_flags",
    ] = model_status.loc[
        model_status["target_decision_disagreement_failed"].astype(bool),
        "blocking_stability_flags",
    ].map(lambda value: ";".join(filter(None, [str(value), "target_decision_disagreement_rate"])))

    model_status["blocking_stability_flags"] = model_status[
        "blocking_stability_flags"
    ].map(lambda value: ";".join(sorted(set(filter(None, str(value).split(";"))))))
    model_status["supporting_stability_flags"] = model_status[
        "supporting_stability_flags"
    ].map(lambda value: ";".join(sorted(set(filter(None, str(value).split(";"))))))
    model_status["stability_flags"] = [
        ";".join(
            sorted(
                set(
                    filter(
                        None,
                        [*str(blocking).split(";"), *str(supporting).split(";")],
                    )
                )
            )
        )
        for blocking, supporting in zip(
            model_status["blocking_stability_flags"],
            model_status["supporting_stability_flags"],
        )
    ]
    model_status["stability_flag_count"] = model_status["stability_flags"].map(
        lambda value: 0 if not str(value) else len(str(value).split(";"))
    )
    model_status["model_stability_status"] = np.select(
        [
            ~model_status["all_expected_random_states_present"].astype(bool),
            ~model_status["is_stochastic"].astype(bool),
            model_status["blocking_stability_failed"].astype(bool),
            model_status["supporting_stability_warning"].astype(bool),
        ],
        [
            "not_estimable_missing_seed",
            "not_applicable_deterministic",
            "unstable_blocking",
            "robust_with_supporting_warnings",
        ],
        default="robust",
    )

    summary = summary.merge(
        model_status[
            [
                *model_key,
                "decision_disagreement_rate",
                "target_decision_disagreement_rate",
                "blocking_stability_failed",
                "supporting_stability_warning",
                "blocking_stability_flags",
                "supporting_stability_flags",
                "stability_flags",
                "stability_flag_count",
                "model_stability_status",
            ]
        ],
        on=model_key,
        how="left",
        validate="many_to_one",
    )
    return _reindex_contract(summary, expcfg.SIMCA_ROBUSTNESS_STABILITY_SUMMARY_COLUMNS)



# ---------------------------------------------------------------------------
# 4. Pre-batch4 review guardrails (filtering only, never ranking)
# ---------------------------------------------------------------------------


def build_robustness_review_guardrails(
    pareto_candidates: pd.DataFrame,
    stability_summary: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build pre-batch4 review; only primary seed instability is blocking."""
    validate_simca_table_columns(
        pareto_candidates,
        expcfg.SIMCA_ROBUSTNESS_PARETO_CANDIDATE_COLUMNS,
        table_name="validation Pareto candidates",
    )
    units = pareto_candidates.copy()
    model_key = list(expcfg.SIMCA_ROBUSTNESS_MODEL_KEY_COLUMNS)
    if units.duplicated(model_key).any():
        raise RuntimeError("Review input duplicates (model_id, track_id).")

    if stability_summary is None or stability_summary.empty:
        stability = units[model_key].drop_duplicates().copy()
        stability["model_stability_status"] = pd.NA
        stability["stability_flags"] = ""
        stability["blocking_stability_flags"] = ""
        stability["supporting_stability_flags"] = ""
        stability["blocking_stability_failed"] = False
        stability["supporting_stability_warning"] = False
    else:
        validate_simca_table_columns(
            stability_summary,
            expcfg.SIMCA_ROBUSTNESS_STABILITY_SUMMARY_COLUMNS,
            table_name="model seed stability",
        )
        invariant_columns = [
            "model_stability_status",
            "stability_flags",
            "blocking_stability_flags",
            "supporting_stability_flags",
            "blocking_stability_failed",
            "supporting_stability_warning",
        ]
        _assert_group_invariant(
            stability_summary,
            model_key,
            invariant_columns,
            name="stability summary",
        )
        stability = stability_summary[
            [*model_key, *invariant_columns]
        ].drop_duplicates(model_key)

    units = units.merge(
        stability,
        on=model_key,
        how="left",
        validate="one_to_one",
    )
    deterministic_missing = (
        ~units["is_stochastic"].astype(bool)
        & units["model_stability_status"].isna()
    )
    units.loc[
        deterministic_missing, "model_stability_status"
    ] = "not_applicable_deterministic"
    for column in (
        "stability_flags",
        "blocking_stability_flags",
        "supporting_stability_flags",
    ):
        units[column] = units[column].fillna("").astype(str)
    units["blocking_stability_failed"] = units[
        "blocking_stability_failed"
    ].fillna(False).astype(bool)
    units["supporting_stability_warning"] = units[
        "supporting_stability_warning"
    ].fillna(False).astype(bool)

    checks: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    allowed_stability = set(
        map(str, expcfg.SIMCA_ROBUSTNESS_PURE_TEST_STABILITY_STATUSES)
    )

    for row in units.to_dict("records"):
        model_id = str(row["model_id"])
        track_id = str(row["track_id"])
        supported = (
            str(row["eligibility_status"])
            in expcfg.SIMCA_ROBUSTNESS_SUPPORTED_ELIGIBILITY_STATUSES
            and str(row["downstream_status"])
            in expcfg.SIMCA_ROBUSTNESS_SUPPORTED_DOWNSTREAM_STATUSES
        )
        calculable = bool(row["all_execution_calculable"])
        guardrail_pass = bool(row["all_04c_blocking_guardrails_pass"])
        on_pareto = bool(row["is_protocol_pareto"])
        seed_ok = bool(row["seed_requirement_satisfied"])
        stability_status = str(row.get("model_stability_status", ""))
        blocking_stability_failed = bool(row["blocking_stability_failed"])
        supporting_warning = bool(row["supporting_stability_warning"])
        stability_ok = (
            stability_status in allowed_stability
            and not blocking_stability_failed
        )

        def add_check(
            scope: str,
            name: str,
            *,
            observed_value: float = np.nan,
            observed_status: str = "",
            comparator: str = "",
            threshold_value: float = np.nan,
            threshold_statuses: str = "",
            passed: bool,
            blocking: bool,
            reason_code: str,
            reason: str,
        ) -> None:
            checks.append(
                {
                    "model_id": model_id,
                    "track_id": track_id,
                    "check_scope": scope,
                    "check_name": name,
                    "observed_value": observed_value,
                    "observed_status": observed_status,
                    "comparator": comparator,
                    "threshold_value": threshold_value,
                    "threshold_statuses": threshold_statuses,
                    "check_status": "pass" if passed else "fail",
                    "is_blocking": bool(blocking),
                    "reason_code": reason_code,
                    "reason": reason,
                }
            )

        add_check(
            "upstream_support",
            "03c_04a_support",
            observed_status=f"{row['eligibility_status']}|{row['downstream_status']}",
            comparator="in",
            threshold_statuses="supported protocol path",
            passed=supported,
            blocking=True,
            reason_code="supported_upstream" if supported else "unsupported_upstream",
            reason="03C eligibility and 04A downstream support remain authoritative.",
        )
        add_check(
            "validation",
            "technical_calculability",
            observed_status="calculable" if calculable else "not_calculable",
            comparator="is",
            threshold_statuses="calculable",
            passed=calculable,
            blocking=True,
            reason_code="validation_calculable" if calculable else "validation_technical_failure",
            reason="Every required base 04C execution/scope must be calculable.",
        )
        add_check(
            "validation",
            "frozen_04c_guardrails",
            observed_status="pass" if guardrail_pass else "fail",
            comparator="is",
            threshold_statuses="pass",
            passed=guardrail_pass,
            blocking=True,
            reason_code="04c_guardrails_pass" if guardrail_pass else "04c_guardrail_failure",
            reason="No second independent batch-3 guardrail contract is introduced in 05.",
        )
        add_check(
            "validation_pareto",
            "within_track_protocol_pareto",
            observed_status="pareto" if on_pareto else "not_pareto",
            comparator="is",
            threshold_statuses="pareto",
            passed=on_pareto,
            blocking=True,
            reason_code="within_track_pareto" if on_pareto else "within_track_dominated",
            reason="Pareto membership is computed only inside the same E1-E8 track.",
        )
        if on_pareto and bool(row["is_stochastic"]):
            complete = stability_status != "not_estimable_missing_seed"
            add_check(
                "seed_robustness",
                "extended_seed_coverage",
                observed_status="complete" if complete else "incomplete",
                comparator="is",
                threshold_statuses="complete",
                passed=complete,
                blocking=True,
                reason_code="seed_panel_complete" if complete else "seed_panel_incomplete",
                reason="Stochastic Pareto models require the complete extended seed panel.",
            )
            add_check(
                "seed_robustness",
                "primary_seed_stability",
                observed_status=stability_status,
                comparator="in",
                threshold_statuses="|".join(sorted(allowed_stability)),
                passed=stability_ok,
                blocking=bool(expcfg.SIMCA_ROBUSTNESS_REQUIRE_STABILITY_FOR_PURE_TEST),
                reason_code="primary_seed_stability_acceptable" if stability_ok else "primary_seed_stability_failed",
                reason="Only prespecified primary risk metrics and decision disagreement can block on seed robustness.",
            )
            add_check(
                "seed_robustness",
                "secondary_seed_stability",
                observed_status=(
                    "warning" if supporting_warning else "no_warning"
                ),
                comparator="is",
                threshold_statuses="no_warning",
                passed=not supporting_warning,
                blocking=False,
                reason_code=(
                    "secondary_seed_stability_warning"
                    if supporting_warning
                    else "secondary_seed_stability_no_warning"
                ),
                reason="Secondary performance/spatial variability is reported as a warning only.",
            )

        flags: list[str] = []
        if not supported:
            review_status = "diagnostic_only"
            flags.append("unsupported_track_path")
        elif not calculable:
            review_status = "excluded_technical"
            flags.append("validation_technical_failure")
        elif not guardrail_pass:
            review_status = "excluded_04c_guardrail"
            flags.append("04c_guardrail_failure")
        elif not seed_ok:
            review_status = "excluded_missing_seed"
            flags.append("base_seed_coverage_incomplete")
        elif not on_pareto:
            review_status = "not_on_validation_pareto"
            flags.append("within_track_pareto_dominated")
        elif bool(row["is_stochastic"]) and stability_status == "not_estimable_missing_seed":
            review_status = "excluded_missing_seed"
            flags.append("robustness_seed_coverage_incomplete")
        elif (
            bool(row["is_stochastic"])
            and bool(expcfg.SIMCA_ROBUSTNESS_REQUIRE_STABILITY_FOR_PURE_TEST)
            and not stability_ok
        ):
            review_status = "excluded_unstable"
            flags.append("primary_seed_instability")
        elif (
            str(row["eligibility_status"]) == "eligible_with_warning"
            or supporting_warning
        ):
            review_status = "eligible_with_warning"
            if str(row["eligibility_status"]) == "eligible_with_warning":
                flags.append("upstream_eligibility_warning")
            if supporting_warning:
                flags.append("secondary_seed_stability_warning")
        else:
            review_status = "eligible_for_pure_test"

        stability_flags = str(row.get("stability_flags", ""))
        if stability_flags:
            flags.extend(flag for flag in stability_flags.split(";") if flag)
        flags = sorted(set(flags))
        review_rows.append(
            {
                "model_id": model_id,
                "track_id": track_id,
                "review_status": review_status,
                "hard_exclusion": review_status
                in expcfg.SIMCA_ROBUSTNESS_REVIEW_HARD_EXCLUSION_STATUSES,
                "robustness_flags": ";".join(flags),
                "stability_flags": stability_flags,
                "review_flags": ";".join(flags),
                "review_flag_count": len(flags),
                "selection_influence": False,
            }
        )

    guardrail_table = _reindex_contract(
        pd.DataFrame(checks),
        expcfg.SIMCA_ROBUSTNESS_REVIEW_GUARDRAIL_COLUMNS,
    )
    review_table = _reindex_contract(
        pd.DataFrame(review_rows),
        expcfg.SIMCA_ROBUSTNESS_TRACK_REVIEW_COLUMNS,
    )
    eligible = review_table.loc[
        review_table["review_status"].astype(str).isin(
            expcfg.SIMCA_ROBUSTNESS_REVIEW_ELIGIBLE_STATUSES
        )
    ].copy()
    pure = units.merge(
        eligible[["model_id", "track_id", "review_status", "review_flags"]],
        on=["model_id", "track_id"],
        how="inner",
        validate="one_to_one",
    )
    pure["candidate_role"] = "eligible_for_pure_test_batch4"
    pure = _reindex_contract(
        pure,
        expcfg.SIMCA_ROBUSTNESS_PURE_TEST_CANDIDATE_COLUMNS,
    )
    return guardrail_table, review_table, pure

def build_track_scoring_table(
    pareto_candidates: pd.DataFrame,
    stability_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    _, review, _ = build_robustness_review_guardrails(
        pareto_candidates,
        stability_summary,
    )
    return review



# ---------------------------------------------------------------------------
# 5. Supporting threshold sensitivity and threshold stability
# ---------------------------------------------------------------------------


def _cross_join_values(
    frame: pd.DataFrame,
    values: Sequence[float],
    *,
    column: str,
) -> pd.DataFrame:
    values_array = np.asarray(tuple(values), dtype=float)
    if (
        values_array.ndim != 1
        or values_array.size == 0
        or not np.isfinite(values_array).all()
    ):
        raise ValueError(f"{column} must contain at least one finite value.")
    return frame.merge(pd.DataFrame({column: values_array}), how="cross")

def build_threshold_sensitivity_plan(
    validation_executions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    *,
    model_ids: Sequence[str] | None = None,
    two_way_direct_deltas: Sequence[float] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DIRECT_2WAY_DELTAS
    ),
    two_way_vote_deltas: Sequence[float] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_VOTE_2WAY_DELTAS
    ),
    three_way_center_shift_fractions: Sequence[float] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_CENTER_SHIFT_FRACTIONS
    ),
    three_way_width_scales: Sequence[float] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_WIDTH_SCALES
    ),
) -> pd.DataFrame:
    """Create a compact, deterministic local numeric-threshold stress plan.

    No sensitivity-specific ID is introduced. The natural key is
    ``(model_id, random_state, decision_scope, perturbation_type,
    perturbation_value)``. The reference threshold is already persisted in the
    canonical seed-threshold registry and is therefore not duplicated here.
    """
    executions = normalize_validation_executions(
        validation_executions,
        columns=_EXECUTION_VIEW,
    )
    thresholds = normalize_threshold_registry(selected_thresholds)

    if model_ids is not None:
        allowed = set(map(str, model_ids))
        executions = executions.loc[executions["model_id"].isin(allowed)].copy()
        thresholds = thresholds.loc[thresholds["model_id"].isin(allowed)].copy()

    base = thresholds.merge(
        executions[
            [
                "model_id",
                "random_state",
                "track_id",
                "decision_mode",
                "projection_level",
            ]
        ],
        on=["model_id", "random_state"],
        how="inner",
        validate="many_to_one",
    )
    if base.empty:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_PLAN_COLUMNS
        )

    base["lower_threshold"] = pd.to_numeric(
        base["lower_threshold"], errors="raise"
    ).astype(float)
    base["upper_threshold"] = pd.to_numeric(
        base["upper_threshold"], errors="raise"
    ).astype(float)

    parts: list[pd.DataFrame] = []

    direct_2way = base.loc[
        base["decision_mode"].eq("2way")
        & base["decision_scope"].eq("direct")
    ].copy()
    if not direct_2way.empty:
        part = _cross_join_values(
            direct_2way,
            two_way_direct_deltas,
            column="perturbation_value",
        )
        part["perturbation_type"] = "direct_threshold_delta"
        alternative = part["lower_threshold"] + part["perturbation_value"]
        part["alternative_lower_threshold"] = alternative
        part["alternative_upper_threshold"] = alternative
        parts.append(part)

    vote_2way = base.loc[
        base["decision_mode"].eq("2way")
        & base["decision_scope"].eq("pixel_to_object")
    ].copy()
    if not vote_2way.empty:
        part = _cross_join_values(
            vote_2way,
            two_way_vote_deltas,
            column="perturbation_value",
        )
        part["perturbation_type"] = "vote_threshold_delta"
        alternative = (
            part["lower_threshold"] + part["perturbation_value"]
        ).clip(0.0, 1.0)
        part["alternative_lower_threshold"] = alternative
        part["alternative_upper_threshold"] = alternative
        parts.append(part)

    three_way = base.loc[base["decision_mode"].eq("3way")].copy()
    if not three_way.empty:
        width = three_way["upper_threshold"] - three_way["lower_threshold"]
        if (
            ~np.isfinite(width.to_numpy(dtype=float))
        ).any() or width.le(0).any():
            raise RuntimeError("A frozen 3-way interval is invalid.")
        center = 0.5 * (
            three_way["upper_threshold"] + three_way["lower_threshold"]
        )
        three_way = three_way.assign(_center=center, _width=width)

        shift = _cross_join_values(
            three_way,
            three_way_center_shift_fractions,
            column="perturbation_value",
        )
        shift["perturbation_type"] = "three_way_center_shift_fraction"
        shifted_center = (
            shift["_center"] + shift["perturbation_value"] * shift["_width"]
        )
        shift["alternative_lower_threshold"] = (
            shifted_center - 0.5 * shift["_width"]
        )
        shift["alternative_upper_threshold"] = (
            shifted_center + 0.5 * shift["_width"]
        )
        parts.append(shift)

        scale = _cross_join_values(
            three_way,
            three_way_width_scales,
            column="perturbation_value",
        )
        if scale["perturbation_value"].le(0).any():
            raise ValueError("Three-way width scales must be strictly positive.")
        scale["perturbation_type"] = "three_way_width_scale"
        scaled_width = scale["_width"] * scale["perturbation_value"]
        scale["alternative_lower_threshold"] = (
            scale["_center"] - 0.5 * scaled_width
        )
        scale["alternative_upper_threshold"] = (
            scale["_center"] + 0.5 * scaled_width
        )
        parts.append(scale)

    if not parts:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_PLAN_COLUMNS
        )

    plan = pd.concat(parts, ignore_index=True, sort=False)
    lower = pd.to_numeric(plan["alternative_lower_threshold"], errors="coerce")
    upper = pd.to_numeric(plan["alternative_upper_threshold"], errors="coerce")
    valid = np.isfinite(lower.to_numpy(dtype=float)) & np.isfinite(
        upper.to_numpy(dtype=float)
    )
    two_way = plan["decision_mode"].astype(str).eq("2way").to_numpy()
    valid &= np.where(
        two_way,
        np.isclose(lower, upper, rtol=0.0, atol=1e-12),
        lower < upper,
    )
    unchanged = np.isclose(
        lower,
        pd.to_numeric(plan["lower_threshold"], errors="coerce"),
        rtol=0.0,
        atol=1e-12,
    ) & np.isclose(
        upper,
        pd.to_numeric(plan["upper_threshold"], errors="coerce"),
        rtol=0.0,
        atol=1e-12,
    )
    plan["plan_status"] = np.select(
        [~valid, unchanged],
        ["invalid_after_domain_constraint", "no_effect_after_domain_constraint"],
        default="estimable",
    )
    plan["selection_influence"] = False

    key = [
        "model_id",
        "random_state",
        "decision_scope",
        "perturbation_type",
        "perturbation_value",
    ]
    if plan.duplicated(key).any():
        raise RuntimeError("Threshold sensitivity plan duplicates its natural key.")
    return (
        plan.reindex(
            columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_PLAN_COLUMNS
        )
        .sort_values(key, kind="mergesort")
        .reset_index(drop=True)
    )

def _overall_metric_view(
    metrics: pd.DataFrame,
    *,
    metric_names: Sequence[str],
) -> pd.DataFrame:
    require_columns(
        metrics,
        expcfg.SIMCA_VALIDATION_METRIC_COLUMNS,
        "validation metrics",
    )
    wanted = set(map(str, metric_names))
    out = metrics.loc[
        metrics["map_variant"].astype(str).eq(
            str(expcfg.SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT)
        )
        & metrics["aggregation_level"].astype(str).eq("overall")
        & metrics["group_id"].astype(str).eq("all")
        & metrics["status"].astype(str).eq("calculable")
        & metrics["metric"].astype(str).isin(wanted),
        [
            "model_id",
            "random_state",
            "track_id",
            "decision_scope",
            "metric",
            "value",
        ],
    ].copy()
    out["model_id"] = out["model_id"].astype(str)
    out["random_state"] = pd.to_numeric(
        out["random_state"], errors="raise"
    ).astype(int)
    out["track_id"] = out["track_id"].astype(str)
    out["decision_scope"] = out["decision_scope"].astype(str)
    out["metric"] = out["metric"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    key = ["model_id", "random_state", "track_id", "decision_scope", "metric"]
    if out.duplicated(key).any():
        raise RuntimeError("Validation overall metrics duplicate their natural key.")
    return out

def _prediction_groups(frame: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if frame is None or frame.empty:
        return {}
    require_columns(frame, ("projection_id",), "validation predictions")
    return {
        str(projection_id): group.reset_index(drop=True)
        for projection_id, group in frame.groupby(
            "projection_id", sort=False, dropna=False
        )
    }

def _scope_observations(
    *,
    projection_id: str,
    projection_level: str,
    decision_scope: str,
    object_predictions_by_projection: dict[str, pd.DataFrame],
    pixel_predictions_by_projection: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    if projection_level == "object_projection":
        observations = object_predictions_by_projection.get(projection_id)
    elif projection_level == "pixel_projection":
        observations = pixel_predictions_by_projection.get(projection_id)
    else:
        raise RuntimeError(f"Unknown projection level: {projection_level!r}.")
    if observations is None or observations.empty:
        raise RuntimeError(
            f"No saved validation predictions for projection_id={projection_id!r}."
        )
    if decision_scope == "direct":
        return observations, "simca_margin"
    if decision_scope == "pixel_to_object":
        if projection_level != "pixel_projection":
            raise RuntimeError("pixel_to_object is only valid for pixel projections.")
        return (
            build_pixel_vote_table(
                observations,
                group_columns=("source_image", "object_id"),
            ),
            "pixel_target_ratio",
        )
    raise RuntimeError(f"Unknown decision scope: {decision_scope!r}.")

def _apply_numeric_thresholds(
    scores: np.ndarray,
    *,
    decision_mode: str,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(decision_mode)
    if mode == "2way":
        if not np.isclose(lower, upper, rtol=0.0, atol=1e-12):
            raise RuntimeError("A 2-way threshold must satisfy lower == upper.")
        return apply_locked_margin_decision(
            scores,
            mode,
            direct_2way_threshold=float(lower),
            three_way_lower_threshold=np.nan,
            three_way_upper_threshold=np.nan,
        )
    if mode == "3way":
        if not float(lower) < float(upper):
            raise RuntimeError("A 3-way threshold must satisfy lower < upper.")
        return apply_locked_margin_decision(
            scores,
            mode,
            direct_2way_threshold=np.nan,
            three_way_lower_threshold=float(lower),
            three_way_upper_threshold=float(upper),
        )
    raise RuntimeError(f"Unknown decision mode: {mode!r}.")

def compare_threshold_registries_on_validation(
    reference_thresholds: pd.DataFrame,
    alternative_thresholds: pd.DataFrame,
    validation_executions: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    *,
    reference_validation_metrics: pd.DataFrame | None = None,
    metric_names: Sequence[str] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRICS
    ),
) -> dict[str, pd.DataFrame]:
    """Compare two fixed threshold registries on saved batch-3 margins.

    Scientific metrics are computed exclusively by the canonical 04C
    ``evaluate_locked_validation_predictions`` kernel.

    The complete 04C execution registry is therefore preserved for metric
    evaluation.  A compact execution view is created only for the local
    paired decision-flip diagnostic.

    No model, execution or threshold-policy identifier is created here.
    """

    # ------------------------------------------------------------------
    # Full canonical 04C execution contract.
    #
    # IMPORTANT:
    # evaluate_locked_validation_predictions() requires the complete
    # SIMCA_VALIDATION_EXECUTION_COLUMNS contract.  Do not reduce this
    # dataframe to _EXECUTION_VIEW before calling the 04C kernel.
    # ------------------------------------------------------------------
    executions = normalize_validation_executions(
        validation_executions,
        name="validation_executions",
    )

    # Compact view only for the local decision-level comparison below.
    execution_view = executions.loc[
        :,
        list(_EXECUTION_VIEW),
    ].copy()

    # ------------------------------------------------------------------
    # Canonical threshold registries.
    # ------------------------------------------------------------------
    reference = normalize_threshold_registry(
        reference_thresholds,
        name="reference_thresholds",
    )

    alternative = normalize_threshold_registry(
        alternative_thresholds,
        name="alternative_thresholds",
    )

    # ------------------------------------------------------------------
    # Reference and alternative registries must cover exactly the same
    # natural threshold scopes.
    # ------------------------------------------------------------------
    ref_keys = (
        reference.loc[
            :,
            list(_SCOPE_KEY),
        ]
        .sort_values(
            list(_SCOPE_KEY),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    alt_keys = (
        alternative.loc[
            :,
            list(_SCOPE_KEY),
        ]
        .sort_values(
            list(_SCOPE_KEY),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if not ref_keys.equals(alt_keys):
        raise RuntimeError(
            "Reference and alternative threshold registries cover "
            "different natural decision scopes."
        )

    # ------------------------------------------------------------------
    # Scientific metric evaluation.
    #
    # Reuse the exact 04C kernel; do not duplicate metric definitions.
    # ------------------------------------------------------------------
    alternative_metrics = evaluate_locked_validation_predictions(
        executions,
        alternative,
        object_predictions,
        pixel_predictions,
    )

    if reference_validation_metrics is None:
        reference_validation_metrics = (
            evaluate_locked_validation_predictions(
                executions,
                reference,
                object_predictions,
                pixel_predictions,
            )
        )

    # ------------------------------------------------------------------
    # Overall metric comparison.
    # ------------------------------------------------------------------
    ref_metric = _overall_metric_view(
        reference_validation_metrics,
        metric_names=metric_names,
    ).rename(
        columns={
            "value": "reference_value",
        }
    )

    alt_metric = _overall_metric_view(
        alternative_metrics,
        metric_names=metric_names,
    ).rename(
        columns={
            "value": "alternative_value",
        }
    )

    metric_key = [
        "model_id",
        "random_state",
        "track_id",
        "decision_scope",
        "metric",
    ]

    metric_effects = ref_metric.merge(
        alt_metric,
        on=metric_key,
        how="inner",
        validate="one_to_one",
    )

    metric_effects["delta"] = (
        pd.to_numeric(
            metric_effects["alternative_value"],
            errors="coerce",
        )
        - pd.to_numeric(
            metric_effects["reference_value"],
            errors="coerce",
        )
    )

    # ------------------------------------------------------------------
    # Threshold pairs for paired decision-flip diagnostics.
    #
    # Only the compact execution metadata needed for locating predictions
    # and applying decisions are joined here.
    # ------------------------------------------------------------------
    threshold_pairs = (
        reference[
            [
                *_SCOPE_KEY,
                "lower_threshold",
                "upper_threshold",
            ]
        ]
        .merge(
            alternative[
                [
                    *_SCOPE_KEY,
                    "lower_threshold",
                    "upper_threshold",
                ]
            ],
            on=list(_SCOPE_KEY),
            how="inner",
            suffixes=(
                "_reference",
                "_alternative",
            ),
            validate="one_to_one",
        )
        .merge(
            execution_view,
            on=[
                "model_id",
                "random_state",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    if threshold_pairs[
        [
            "projection_id",
            "track_id",
            "decision_mode",
            "projection_level",
        ]
    ].isna().any().any():
        bad = threshold_pairs.loc[
            threshold_pairs[
                [
                    "projection_id",
                    "track_id",
                    "decision_mode",
                    "projection_level",
                ]
            ].isna().any(axis=1),
            [
                "model_id",
                "random_state",
                "decision_scope",
            ],
        ].head(10)

        raise RuntimeError(
            "Could not attach canonical execution metadata to threshold "
            f"pairs: {bad.to_dict('records')}."
        )

    object_groups = _prediction_groups(
        object_predictions
    )

    pixel_groups = _prediction_groups(
        pixel_predictions
    )

    decision_rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Paired decision comparison.
    #
    # This loop is intentionally at execution/scope level.  Prediction
    # matrices are grouped once above, so no repeated dataframe scanning
    # occurs here.
    # ------------------------------------------------------------------
    for row in threshold_pairs.itertuples(index=False):

        observations, score_col = _scope_observations(
            projection_id=str(
                row.projection_id
            ),
            projection_level=str(
                row.projection_level
            ),
            decision_scope=str(
                row.decision_scope
            ),
            object_predictions_by_projection=(
                object_groups
            ),
            pixel_predictions_by_projection=(
                pixel_groups
            ),
        )

        scores = pd.to_numeric(
            observations[score_col],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        truth = (
            pd.to_numeric(
                observations["truth"],
                errors="raise",
            )
            .astype(bool)
            .to_numpy()
        )

        if not np.isfinite(scores).all():
            raise RuntimeError(
                "Saved validation scores contain non-finite values."
            )

        # --------------------------------------------------------------
        # Reference decisions.
        # --------------------------------------------------------------
        ref_target, ref_uncertain = (
            _apply_numeric_thresholds(
                scores,
                decision_mode=str(
                    row.decision_mode
                ),
                lower=float(
                    row.lower_threshold_reference
                ),
                upper=float(
                    row.upper_threshold_reference
                ),
            )
        )

        # --------------------------------------------------------------
        # Perturbed decisions.
        # --------------------------------------------------------------
        alt_target, alt_uncertain = (
            _apply_numeric_thresholds(
                scores,
                decision_mode=str(
                    row.decision_mode
                ),
                lower=float(
                    row.lower_threshold_alternative
                ),
                upper=float(
                    row.upper_threshold_alternative
                ),
            )
        )

        # Canonical local decision encoding:
        # 0 = non-target
        # 1 = target
        # 2 = uncertain
        ref_code = np.where(
            ref_uncertain,
            2,
            np.where(
                ref_target,
                1,
                0,
            ),
        ).astype(
            np.int8
        )

        alt_code = np.where(
            alt_uncertain,
            2,
            np.where(
                alt_target,
                1,
                0,
            ),
        ).astype(
            np.int8
        )

        flip = (
            ref_code
            != alt_code
        )

        target_flip = flip[
            truth
        ]

        decision_rows.append(
            {
                "model_id": str(
                    row.model_id
                ),
                "random_state": int(
                    row.random_state
                ),
                "track_id": str(
                    row.track_id
                ),
                "decision_scope": str(
                    row.decision_scope
                ),
                "n_entities": int(
                    flip.size
                ),
                "n_target_entities": int(
                    truth.sum()
                ),
                "decision_flip_rate": (
                    float(
                        flip.mean()
                    )
                    if flip.size
                    else np.nan
                ),
                "target_decision_flip_rate": (
                    float(
                        target_flip.mean()
                    )
                    if target_flip.size
                    else np.nan
                ),
            }
        )

    decision_effects = pd.DataFrame(
        decision_rows
    )

    return {
        "metrics": metric_effects.reset_index(
            drop=True
        ),
        "decisions": decision_effects.reset_index(
            drop=True
        ),
    }


def _materialize_variant_registry(
    reference_thresholds: pd.DataFrame,
    variant_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize one supporting numeric-threshold perturbation.

    The canonical policy coordinates are left unchanged.  Only the numerical
    lower/upper decision boundaries are replaced for the affected natural
    threshold scopes.

    No new identifier is created and the input registry is never mutated.
    """
    reference = normalize_threshold_registry(
        reference_thresholds,
        name="reference_thresholds",
    )

    require_columns(
        variant_rows,
        (
            *_SCOPE_KEY,
            "alternative_lower_threshold",
            "alternative_upper_threshold",
        ),
        "threshold-sensitivity variant rows",
    )

    changes = variant_rows[
        [
            *_SCOPE_KEY,
            "alternative_lower_threshold",
            "alternative_upper_threshold",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # One and only one perturbation per natural threshold scope.
    # ------------------------------------------------------------------
    if changes.duplicated(list(_SCOPE_KEY)).any():
        duplicates = (
            changes.loc[
                changes.duplicated(
                    list(_SCOPE_KEY),
                    keep=False,
                ),
                list(_SCOPE_KEY),
            ]
            .drop_duplicates()
            .head(10)
        )

        raise RuntimeError(
            "One threshold-sensitivity variant changes a scope more "
            "than once: "
            f"{duplicates.to_dict('records')}."
        )

    # Explicit float64 here too.  This makes the function independent of
    # whatever physical dtype was used in an upstream Parquet file.
    for column in (
        "alternative_lower_threshold",
        "alternative_upper_threshold",
    ):
        original = changes[column]

        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"{column} contains non-numeric perturbation values."
            )

        changes[column] = numeric.astype("float64")

    lower_present = changes[
        "alternative_lower_threshold"
    ].notna()

    upper_present = changes[
        "alternative_upper_threshold"
    ].notna()

    # A perturbation must always specify the complete numerical decision
    # boundary.  Never silently modify only one side.
    if not lower_present.equals(upper_present):
        bad = changes.loc[
            lower_present.ne(upper_present),
            [
                *_SCOPE_KEY,
                "alternative_lower_threshold",
                "alternative_upper_threshold",
            ],
        ].head(10)

        raise RuntimeError(
            "Threshold perturbations must define lower and upper "
            "boundaries together: "
            f"{bad.to_dict('records')}."
        )

    if lower_present.any():
        alternative_values = changes.loc[
            lower_present,
            [
                "alternative_lower_threshold",
                "alternative_upper_threshold",
            ],
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(alternative_values).all():
            raise RuntimeError(
                "An estimable threshold perturbation contains a "
                "non-finite numerical boundary."
            )

    # ------------------------------------------------------------------
    # Merge changes onto the complete reference registry.
    # ------------------------------------------------------------------
    out = reference.merge(
        changes,
        on=list(_SCOPE_KEY),
        how="left",
        validate="one_to_one",
    )

    changed = out[
        "alternative_lower_threshold"
    ].notna()

    reference_lower = pd.to_numeric(
        out["lower_threshold"],
        errors="raise",
    ).astype("float64")

    reference_upper = pd.to_numeric(
        out["upper_threshold"],
        errors="raise",
    ).astype("float64")

    alternative_lower = pd.to_numeric(
        out["alternative_lower_threshold"],
        errors="coerce",
    ).astype("float64")

    alternative_upper = pd.to_numeric(
        out["alternative_upper_threshold"],
        errors="coerce",
    ).astype("float64")

    # Series.where constructs complete float64 columns instead of assigning
    # float64 values into an existing possibly-float32 block.
    out["lower_threshold"] = reference_lower.where(
        ~changed,
        alternative_lower,
    )

    out["upper_threshold"] = reference_upper.where(
        ~changed,
        alternative_upper,
    )

    result = out.reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
    ).copy()

    # ------------------------------------------------------------------
    # Defensive postconditions.
    # ------------------------------------------------------------------
    if result.duplicated(list(_THRESHOLD_KEY)).any():
        raise RuntimeError(
            "Materialized threshold variant duplicates its natural key."
        )

    boundaries = result[
        [
            "lower_threshold",
            "upper_threshold",
        ]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(boundaries).all():
        raise RuntimeError(
            "Materialized threshold variant contains non-finite boundaries."
        )

    return result.reset_index(drop=True)


def evaluate_threshold_sensitivity(
    sensitivity_plan: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    validation_executions: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    *,
    reference_validation_metrics: pd.DataFrame,
    metric_names: Sequence[str] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRICS
    ),
) -> dict[str, pd.DataFrame]:
    """Evaluate the frozen local perturbation plan with the exact 04C kernel."""
    require_columns(
        sensitivity_plan,
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_PLAN_COLUMNS,
        "threshold sensitivity plan",
    )
    if sensitivity_plan.empty:
        return {
            "metrics": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRIC_COLUMNS
            ),
            "decisions": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DECISION_COLUMNS
            ),
        }
    assert_supporting_only(sensitivity_plan, name="threshold sensitivity")

    estimable = sensitivity_plan.loc[
        sensitivity_plan["plan_status"].astype(str).eq("estimable")
    ].copy()
    if estimable.empty:
        return {
            "metrics": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRIC_COLUMNS
            ),
            "decisions": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DECISION_COLUMNS
            ),
        }

    metric_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    variant_columns = ["decision_scope", "perturbation_type", "perturbation_value"]

    for variant_key, variant_rows in estimable.groupby(
        variant_columns,
        sort=False,
        dropna=False,
    ):
        scope, perturbation_type, perturbation_value = variant_key
        affected = variant_rows[
            ["model_id", "random_state", "track_id"]
        ].drop_duplicates()
        affected_runs = affected[["model_id", "random_state"]].drop_duplicates()

        execution_subset = validation_executions.merge(
            affected_runs,
            on=["model_id", "random_state"],
            how="inner",
            validate="many_to_one",
        )
        threshold_subset = selected_thresholds.merge(
            affected_runs,
            on=["model_id", "random_state"],
            how="inner",
            validate="many_to_one",
        )
        alternative = _materialize_variant_registry(
            threshold_subset,
            variant_rows,
        )
        reference_metrics_subset = reference_validation_metrics.merge(
            affected_runs,
            on=["model_id", "random_state"],
            how="inner",
            validate="many_to_one",
        )
        compared = compare_threshold_registries_on_validation(
            threshold_subset,
            alternative,
            execution_subset,
            object_predictions,
            pixel_predictions,
            reference_validation_metrics=reference_metrics_subset,
            metric_names=metric_names,
        )

        metrics = compared["metrics"].loc[
            lambda frame: frame["decision_scope"].astype(str).eq(str(scope))
        ].copy()
        metrics["perturbation_type"] = str(perturbation_type)
        metrics["perturbation_value"] = float(perturbation_value)
        metrics = annotate_practical_effects(metrics)
        metrics["selection_influence"] = False
        metric_parts.append(metrics)

        decisions = compared["decisions"].loc[
            lambda frame: frame["decision_scope"].astype(str).eq(str(scope))
        ].copy()
        decisions["perturbation_type"] = str(perturbation_type)
        decisions["perturbation_value"] = float(perturbation_value)
        decisions["selection_influence"] = False
        decision_parts.append(decisions)

    metrics_out = (
        pd.concat(metric_parts, ignore_index=True, sort=False)
        if metric_parts
        else pd.DataFrame()
    ).reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_METRIC_COLUMNS
    )
    decisions_out = (
        pd.concat(decision_parts, ignore_index=True, sort=False)
        if decision_parts
        else pd.DataFrame()
    ).reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DECISION_COLUMNS
    )

    metric_key = [
        "model_id",
        "random_state",
        "decision_scope",
        "perturbation_type",
        "perturbation_value",
        "metric",
    ]
    decision_key = metric_key[:-1]
    if len(metrics_out) and metrics_out.duplicated(metric_key).any():
        raise RuntimeError(
            "Threshold sensitivity metrics duplicate their natural key."
        )
    if len(decisions_out) and decisions_out.duplicated(decision_key).any():
        raise RuntimeError(
            "Threshold sensitivity decisions duplicate their natural key."
        )
    return {"metrics": metrics_out, "decisions": decisions_out}

def build_threshold_stability_diagnostics(
    seed_thresholds: pd.DataFrame,
    validation_executions: pd.DataFrame,
    *,
    warning_limits: dict[str, float] = (
        expcfg.SIMCA_ROBUSTNESS_THRESHOLD_STABILITY_WARNING_LIMITS
    ),
    raise_on_policy_drift: bool = True,
) -> pd.DataFrame:
    """Summarize calibration-threshold drift across the seed panel.

    Changing a frozen policy coordinate is a protocol error. For 2-way rules,
    changing the fixed numeric threshold is also a protocol error. Numeric
    drift of 3-way boundaries is a supporting calibration-stability diagnostic.
    """
    thresholds = normalize_threshold_registry(
        seed_thresholds,
        name="seed_thresholds",
    )
    executions = normalize_validation_executions(
        validation_executions,
        columns=("model_id", "random_state", "track_id", "decision_mode"),
    )
    merged = thresholds.merge(
        executions,
        on=["model_id", "random_state"],
        how="left",
        validate="many_to_one",
    )
    if merged[["track_id", "decision_mode"]].isna().any().any():
        raise RuntimeError(
            "Threshold stability could not resolve execution metadata."
        )

    policy_columns = ("lower_quantile", "upper_quantile", "vote_threshold")
    numeric_columns = ("lower_threshold", "upper_threshold")
    for column in (*policy_columns, *numeric_columns):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["threshold_center"] = 0.5 * (
        merged["lower_threshold"] + merged["upper_threshold"]
    )
    merged["uncertainty_band_width"] = (
        merged["upper_threshold"] - merged["lower_threshold"]
    )

    key = ["model_id", "track_id", "decision_scope", "decision_mode"]
    grouped = merged.groupby(key, sort=False, dropna=False)
    summary = grouped.agg(
        n_random_states=("random_state", "nunique"),
        lower_threshold_mean=("lower_threshold", "mean"),
        lower_threshold_std=("lower_threshold", finite_std),
        lower_threshold_min=("lower_threshold", "min"),
        lower_threshold_max=("lower_threshold", "max"),
        upper_threshold_mean=("upper_threshold", "mean"),
        upper_threshold_std=("upper_threshold", finite_std),
        upper_threshold_min=("upper_threshold", "min"),
        upper_threshold_max=("upper_threshold", "max"),
        threshold_center_mean=("threshold_center", "mean"),
        threshold_center_std=("threshold_center", finite_std),
        threshold_center_min=("threshold_center", "min"),
        threshold_center_max=("threshold_center", "max"),
        uncertainty_band_width_mean=("uncertainty_band_width", "mean"),
        uncertainty_band_width_std=("uncertainty_band_width", finite_std),
        uncertainty_band_width_min=("uncertainty_band_width", "min"),
        uncertainty_band_width_max=("uncertainty_band_width", "max"),
    ).reset_index()

    one_seed = summary["n_random_states"].eq(1)
    for column in (
        "lower_threshold_std",
        "upper_threshold_std",
        "threshold_center_std",
        "uncertainty_band_width_std",
    ):
        summary.loc[one_seed, column] = 0.0

    summary["lower_threshold_range"] = (
        summary.pop("lower_threshold_max") - summary.pop("lower_threshold_min")
    )
    summary["upper_threshold_range"] = (
        summary.pop("upper_threshold_max") - summary.pop("upper_threshold_min")
    )
    summary["threshold_center_range"] = (
        summary.pop("threshold_center_max") - summary.pop("threshold_center_min")
    )
    summary["uncertainty_band_width_range"] = (
        summary.pop("uncertainty_band_width_max")
        - summary.pop("uncertainty_band_width_min")
    )

    policy_nunique = grouped[list(policy_columns)].nunique(dropna=False).max(axis=1)
    numeric_nunique = grouped[list(numeric_columns)].nunique(dropna=False).max(axis=1)
    invariants = pd.DataFrame(
        {
            **{
                column: policy_nunique.index.get_level_values(column)
                for column in key
            },
            "policy_coordinates_invariant": policy_nunique.to_numpy() <= 1,
            "fixed_numeric_threshold_invariant": numeric_nunique.to_numpy() <= 1,
        }
    )
    summary = summary.merge(
        invariants,
        on=key,
        how="left",
        validate="one_to_one",
    )

    width = summary["uncertainty_band_width_mean"].abs()
    summary["band_width_cv"] = np.where(
        width.gt(0),
        summary["uncertainty_band_width_std"] / width,
        np.nan,
    )
    summary["center_range_over_mean_width"] = np.where(
        width.gt(0),
        summary["threshold_center_range"] / width,
        np.nan,
    )

    is_three_way = summary["decision_mode"].astype(str).eq("3way")
    numeric_warning = is_three_way & (
        summary["center_range_over_mean_width"].gt(
            float(warning_limits["max_center_range_over_mean_width"])
        )
        | summary["band_width_cv"].gt(
            float(warning_limits["max_band_width_cv"])
        )
    )
    policy_error = ~summary["policy_coordinates_invariant"].astype(bool)
    fixed_2way_error = (
        summary["decision_mode"].astype(str).eq("2way")
        & ~summary["fixed_numeric_threshold_invariant"].astype(bool)
    )
    summary["numeric_stability_warning"] = numeric_warning
    summary["stability_status"] = np.select(
        [policy_error, fixed_2way_error, numeric_warning],
        [
            "protocol_error_policy_coordinate_drift",
            "protocol_error_fixed_2way_threshold_drift",
            "supporting_numeric_threshold_warning",
        ],
        default="stable",
    )
    summary["selection_influence"] = False

    if raise_on_policy_drift and (policy_error | fixed_2way_error).any():
        examples = summary.loc[
            policy_error | fixed_2way_error,
            ["model_id", "track_id", "decision_scope", "stability_status"],
        ].head(20)
        raise RuntimeError(
            "Frozen threshold invariants changed during notebook 05: "
            f"{examples.to_dict('records')}."
        )

    return summary.reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_THRESHOLD_STABILITY_COLUMNS
    ).reset_index(drop=True)



# ---------------------------------------------------------------------------
# 6. Supporting validation-data composition sensitivity
# ---------------------------------------------------------------------------


def build_source_image_influence_diagnostics(
    validation_metrics: pd.DataFrame,
    *,
    model_ids: Sequence[str] | None = None,
    metric_names: Sequence[str] = expcfg.SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_METRICS,
    map_variant: str = expcfg.SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT,
) -> pd.DataFrame:
    """Compute leave-one-source-image-out macro summaries without row loops.

    Source images are the independent units.  The function works entirely from
    the persisted 04C source-image metric rows and keeps an explicit class-
    support check after each omission.
    """
    require_columns(
        validation_metrics,
        expcfg.SIMCA_VALIDATION_METRIC_COLUMNS,
        "validation_metrics",
    )
    metrics = validation_metrics.copy()
    metrics["model_id"] = metrics["model_id"].astype(str)
    metrics["random_state"] = pd.to_numeric(
        metrics["random_state"], errors="raise"
    ).astype(int)
    metrics["track_id"] = metrics["track_id"].astype(str)
    metrics["decision_scope"] = metrics["decision_scope"].astype(str)
    metrics["group_id"] = metrics["group_id"].astype(str)
    metrics["metric"] = metrics["metric"].astype(str)
    metrics["value"] = pd.to_numeric(metrics["value"], errors="coerce")

    if model_ids is not None:
        metrics = metrics.loc[
            metrics["model_id"].isin(set(map(str, model_ids)))
        ].copy()

    source = metrics.loc[
        metrics["map_variant"].astype(str).eq(str(map_variant))
        & metrics["aggregation_level"].astype(str).eq("source_image")
        & metrics["status"].astype(str).eq("calculable")
    ].copy()
    if source.empty:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_COLUMNS
        )

    natural_source_key = [*_SOURCE_IMAGE_GROUP_KEY, "group_id", "metric"]
    if source.duplicated(natural_source_key).any():
        raise RuntimeError("04C source-image metrics duplicate their natural key.")

    # Per-image target/non-target support is recovered from the same long 04C
    # table.  No raw pixel/object rows are treated as independent replicates.
    support = source.loc[
        source["metric"].isin({"n_target", "n_non_target"}),
        [*_SOURCE_IMAGE_GROUP_KEY, "group_id", "metric", "value"],
    ].pivot(
        index=[*_SOURCE_IMAGE_GROUP_KEY, "group_id"],
        columns="metric",
        values="value",
    ).reset_index()
    support.columns.name = None
    for column in ("n_target", "n_non_target"):
        if column not in support:
            support[column] = 0.0
        support[column] = pd.to_numeric(support[column], errors="coerce").fillna(0.0)

    support_totals = support.groupby(
        list(_SOURCE_IMAGE_GROUP_KEY), as_index=False, sort=False, dropna=False
    ).agg(
        total_n_target=("n_target", "sum"),
        total_n_non_target=("n_non_target", "sum"),
        n_source_images=("group_id", "nunique"),
    )
    support = support.merge(
        support_totals,
        on=list(_SOURCE_IMAGE_GROUP_KEY),
        how="left",
        validate="many_to_one",
    )
    support["retained_n_target"] = support["total_n_target"] - support["n_target"]
    support["retained_n_non_target"] = (
        support["total_n_non_target"] - support["n_non_target"]
    )

    work = source.loc[
        source["metric"].isin(set(map(str, metric_names))),
        [*_SOURCE_IMAGE_GROUP_KEY, "group_id", "metric", "value"],
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_COLUMNS
        )

    work["is_finite"] = np.isfinite(work["value"].to_numpy(dtype=float)).astype(int)
    work["finite_value"] = work["value"].where(work["is_finite"].eq(1), 0.0)
    metric_group = [*_SOURCE_IMAGE_GROUP_KEY, "metric"]
    totals = work.groupby(
        metric_group, as_index=False, sort=False, dropna=False
    ).agg(
        finite_sum=("finite_value", "sum"),
        n_finite_source_images_full=("is_finite", "sum"),
    )
    work = work.merge(
        totals,
        on=metric_group,
        how="left",
        validate="many_to_one",
    )
    work["n_finite_source_images_retained"] = (
        work["n_finite_source_images_full"] - work["is_finite"]
    ).astype(int)
    work["full_macro_image_value"] = np.where(
        work["n_finite_source_images_full"].gt(0),
        work["finite_sum"] / work["n_finite_source_images_full"],
        np.nan,
    )
    retained_sum = work["finite_sum"] - work["finite_value"]
    work["leave_one_image_out_value"] = np.where(
        work["n_finite_source_images_retained"].gt(0),
        retained_sum / work["n_finite_source_images_retained"],
        np.nan,
    )

    work = work.merge(
        support[
            [
                *_SOURCE_IMAGE_GROUP_KEY,
                "group_id",
                "n_source_images",
                "retained_n_target",
                "retained_n_non_target",
            ]
        ],
        on=[*_SOURCE_IMAGE_GROUP_KEY, "group_id"],
        how="left",
        validate="many_to_one",
    )
    class_support = (
        work["retained_n_target"].gt(0)
        & work["retained_n_non_target"].gt(0)
    )
    finite_support = work["n_finite_source_images_retained"].gt(0)
    work.loc[~class_support | ~finite_support, "leave_one_image_out_value"] = np.nan
    work["delta"] = (
        work["leave_one_image_out_value"] - work["full_macro_image_value"]
    )
    work["absolute_delta"] = work["delta"].abs()
    work["influence_status"] = np.select(
        [~class_support, class_support & ~finite_support],
        [
            "not_estimable_class_coverage_lost",
            "not_estimable_metric_support_lost",
        ],
        default="estimable_leave_one_image_out",
    )
    work["omitted_source_image"] = work["group_id"].astype(str)
    work["selection_influence"] = False

    out = work.reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_COLUMNS
    )
    key = [*_SOURCE_IMAGE_GROUP_KEY, "metric", "omitted_source_image"]
    if out.duplicated(key).any():
        raise RuntimeError("Source-image influence diagnostics duplicate their natural key.")
    return out.sort_values(key, kind="mergesort").reset_index(drop=True)



# ---------------------------------------------------------------------------
# 7. Supporting calibration-fold sensitivity
# ---------------------------------------------------------------------------


def _canonical_partition_signature(folds: pd.DataFrame) -> str:
    require_columns(folds, ("source_image", "fold_id"), "folds")
    groups = folds[["source_image", "fold_id"]].drop_duplicates().copy()
    groups["source_image"] = groups["source_image"].astype(str)
    if groups["source_image"].duplicated().any():
        raise RuntimeError("One source image belongs to more than one calibration fold.")
    blocks = [
        tuple(sorted(group["source_image"].astype(str)))
        for _, group in groups.groupby("fold_id", sort=False, dropna=False)
    ]
    return sha256_payload(sorted(blocks))

def build_calibration_fold_sensitivity(
    reference_folds: pd.DataFrame,
    *,
    candidate_random_states: Sequence[int] = (
        expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_RANDOM_STATES
    ),
    max_unique_alternatives: int = (
        expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_MAX_UNIQUE_ALTERNATIVES
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate unique valid partitions, deduplicated modulo fold labels.

    ``alternative_partition_sha256`` is the partition identity; no additional
    fold-sensitivity ID is created. The full alternative assignments are
    returned as a normalized table so the hash can be audited later.
    """
    require_columns(
        reference_folds,
        expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS,
        "reference calibration folds",
    )
    if int(max_unique_alternatives) < 1:
        raise ValueError("max_unique_alternatives must be positive.")

    reference = reference_folds.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS)
    ].copy()
    reference["source_image"] = reference["source_image"].astype(str)
    reference_hash = _canonical_partition_signature(reference)
    reference_rows = reference[
        ["source_image", "object_id", "class_name", "batch", "object_area"]
    ].drop_duplicates("object_id")

    observed_hashes = {reference_hash}
    plan_rows: list[dict[str, object]] = []
    assignment_parts: list[pd.DataFrame] = []

    for generator_state in map(int, candidate_random_states):
        try:
            folds, diagnostics = build_grouped_folds(
                reference_rows,
                group_col=expcfg.INTERNAL_CALIBRATION_GROUP_COL,
                label_col=expcfg.INTERNAL_CALIBRATION_LABEL_COL,
                batch_col=expcfg.INTERNAL_CALIBRATION_BATCH_COL,
                size_col=expcfg.INTERNAL_CALIBRATION_OBJECT_SIZE_COL,
                n_size_bins=expcfg.INTERNAL_CALIBRATION_SIZE_N_BINS,
                n_splits=expcfg.INTERNAL_CALIBRATION_N_SPLITS,
                random_state=generator_state,
                require_complete_coverage=True,
            )
        except Exception:
            continue

        folds = folds.reindex(columns=expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS)
        alternative_hash = _canonical_partition_signature(folds)
        if alternative_hash in observed_hashes:
            continue
        observed_hashes.add(alternative_hash)

        coverage_complete = (
            bool(diagnostics["coverage_complete"].fillna(False).astype(bool).all())
            if "coverage_complete" in diagnostics
            else True
        )
        if not coverage_complete:
            continue

        plan_rows.append(
            {
                "generator_random_state": generator_state,
                "reference_partition_sha256": reference_hash,
                "alternative_partition_sha256": alternative_hash,
                "n_source_images": int(folds["source_image"].nunique()),
                "n_folds": int(folds["fold_id"].nunique()),
                "coverage_complete": True,
                "plan_status": "estimable_unique_valid_partition",
                "selection_influence": False,
            }
        )
        assignment_parts.append(
            folds.assign(alternative_partition_sha256=alternative_hash).reindex(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_ASSIGNMENT_COLUMNS
            )
        )
        if len(plan_rows) >= int(max_unique_alternatives):
            break

    if not plan_rows:
        plan_rows.append(
            {
                "generator_random_state": pd.NA,
                "reference_partition_sha256": reference_hash,
                "alternative_partition_sha256": "",
                "n_source_images": int(reference["source_image"].nunique()),
                "n_folds": int(reference["fold_id"].nunique()),
                "coverage_complete": True,
                "plan_status": "not_estimable_no_alternative_valid_group_split",
                "selection_influence": False,
            }
        )

    plan = pd.DataFrame(plan_rows).reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_PLAN_COLUMNS
    )
    assignments = (
        pd.concat(assignment_parts, ignore_index=True, sort=False)
        if assignment_parts
        else pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_ASSIGNMENT_COLUMNS
        )
    )
    if len(assignments):
        key = ["alternative_partition_sha256", "object_id"]
        if assignments.duplicated(key).any():
            raise RuntimeError("Alternative fold assignments duplicate their natural key.")
    return plan, assignments

def _fixed_execution_registry(
    model_catalog: pd.DataFrame,
    validation_executions: pd.DataFrame,
    model_ids: Sequence[str],
) -> pd.DataFrame:
    """Reuse the frozen 03B/04C technical IDs for fold-sensitivity reruns.

    ``model_id``, ``fit_id`` and ``projection_id`` are never recomputed here.
    Only the OOF partition changes; scientific and technical execution identity
    remains the one already frozen upstream.
    """
    require_columns(
        model_catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        "model_catalog",
    )
    require_columns(
        validation_executions,
        ("model_id", "random_state", "fit_id", "projection_id"),
        "validation_executions",
    )

    requested = set(map(str, model_ids))
    runs = validation_executions.loc[
        validation_executions["model_id"].astype(str).isin(requested),
        ["model_id", "random_state", "fit_id", "projection_id"],
    ].drop_duplicates()
    runs["model_id"] = runs["model_id"].astype(str)
    runs["random_state"] = pd.to_numeric(
        runs["random_state"], errors="raise"
    ).astype(int)

    if runs.duplicated(["model_id", "random_state"]).any():
        raise RuntimeError(
            "Frozen validation executions duplicate (model_id, random_state)."
        )
    if runs[["fit_id", "projection_id"]].isna().any().any():
        raise RuntimeError(
            "Fold sensitivity requires the frozen fit_id/projection_id values."
        )

    catalog = model_catalog.loc[
        model_catalog["model_id"].astype(str).isin(requested),
        list(expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS),
    ].copy()
    catalog["model_id"] = catalog["model_id"].astype(str)
    if catalog["model_id"].duplicated().any():
        raise RuntimeError("model_catalog.model_id must be unique.")
    if set(catalog["model_id"]) != requested:
        missing = sorted(requested - set(catalog["model_id"]))
        raise RuntimeError(
            f"Fold sensitivity cannot resolve model_id values: {missing}."
        )

    executions = catalog.merge(
        runs,
        on="model_id",
        how="inner",
        validate="one_to_many",
    ).reindex(columns=expcfg.INTERNAL_CALIBRATION_EXECUTION_COLUMNS)

    if executions[["fit_id", "projection_id"]].isna().any().any():
        raise RuntimeError(
            "Fold sensitivity lost a frozen technical execution identifier."
        )
    if executions.duplicated(["model_id", "random_state"]).any():
        raise RuntimeError(
            "Fold-sensitivity execution registry duplicates its natural key."
        )
    return executions.reset_index(drop=True)

def evaluate_calibration_fold_sensitivity(
    *,
    fold_plan: pd.DataFrame,
    alternative_fold_assignments: pd.DataFrame,
    object_db: Mapping,
    wavelengths,
    model_catalog: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    validation_executions: pd.DataFrame,
    frozen_selected_thresholds: pd.DataFrame,
    reference_validation_metrics: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    checkpoint_root: str | Path,
    protocol_hash: str,
    checkpoint_context_base: Mapping[str, object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Recompute OOF calibration under each admissible alternative partition."""
    require_columns(
        fold_plan,
        expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_PLAN_COLUMNS,
        "fold sensitivity plan",
    )
    require_columns(
        pareto_candidates,
        ("model_id", "track_id", "is_protocol_pareto"),
        "pareto_candidates",
    )
    estimable = fold_plan.loc[
        fold_plan["plan_status"].astype(str).eq("estimable_unique_valid_partition")
    ].copy()
    if estimable.empty:
        return {
            "thresholds": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_THRESHOLD_COLUMNS
            ),
            "metrics": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRIC_COLUMNS
            ),
            "decisions": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_DECISION_COLUMNS
            ),
            "technical_events": pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_TECHNICAL_EVENT_COLUMNS
            ),
        }

    require_columns(
        alternative_fold_assignments,
        expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_ASSIGNMENT_COLUMNS,
        "alternative fold assignments",
    )
    model_ids = tuple(
        sorted(
            pareto_candidates.loc[
                pareto_candidates["is_protocol_pareto"].fillna(False).astype(bool),
                "model_id",
            ].astype(str).unique()
        )
    )
    if not model_ids:
        raise RuntimeError("No protocol-Pareto model is available for fold sensitivity.")

    validation_subset = validation_executions.loc[
        validation_executions["model_id"].astype(str).isin(model_ids)
    ].copy()
    configurations = _fixed_execution_registry(
        model_catalog,
        validation_subset,
        model_ids,
    )
    random_states = tuple(
        sorted(pd.to_numeric(configurations["random_state"], errors="raise").astype(int).unique())
    )
    reference_thresholds = frozen_selected_thresholds.loc[
        frozen_selected_thresholds["model_id"].astype(str).isin(model_ids)
    ].copy()
    reference_metrics = reference_validation_metrics.loc[
        reference_validation_metrics["model_id"].astype(str).isin(model_ids)
    ].copy()

    checkpoint_root = Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    base_context = dict(checkpoint_context_base or {})
    configuration_hash = sha256_dataframe(configurations)

    threshold_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []

    for plan_row in estimable.itertuples(index=False):
        partition_hash = str(plan_row.alternative_partition_sha256)
        folds = alternative_fold_assignments.loc[
            alternative_fold_assignments["alternative_partition_sha256"].astype(str).eq(
                partition_hash
            ),
            list(expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS),
        ].copy()
        if folds.empty:
            raise RuntimeError(
                f"No persisted fold assignment matches partition {partition_hash}."
            )
        if _canonical_partition_signature(folds) != partition_hash:
            raise RuntimeError("Alternative fold assignments do not match their hash.")

        context = {
            "protocol_hash": str(protocol_hash),
            "pca_selection_fingerprint": str(
                base_context.get(
                    "pca_selection_fingerprint",
                    "fold_sensitivity_fixed_pareto_models",
                )
            ),
            "track_contract_hash": str(
                base_context.get(
                    "track_contract_hash",
                    "fold_sensitivity_fixed_track_contract",
                )
            ),
            "fold_contract_hash": partition_hash,
            "configuration_hash": configuration_hash,
        }
        outputs = run_internal_calibration_8tracks(
            object_db=object_db,
            folds=folds,
            configurations=configurations,
            wavelengths=wavelengths,
            target_class=expcfg.TARGET_CLASS,
            non_target_label=expcfg.NON_TARGET_LABEL,
            under_m_policy=expcfg.INTERNAL_CALIBRATION_UNDER_M_POLICY,
            verbose=False,
            checkpoint_dir=checkpoint_root / partition_hash[:20],
            checkpoint_context=context,
            resume_from_checkpoint=expcfg.INTERNAL_CALIBRATION_RESUME_FROM_CHECKPOINT,
            keep_oof_in_memory=False,
            keep_threshold_metrics_in_memory=True,
        )
        threshold_metrics = outputs["threshold_metrics"]
        alternative_thresholds = materialize_fixed_threshold_policy_for_runs(
            threshold_metrics,
            reference_thresholds,
            model_catalog,
            expected_random_states=random_states,
            expected_model_ids=model_ids,
            expected_source_random_states=random_states,
        )
        threshold_parts.append(
            alternative_thresholds.assign(
                alternative_partition_sha256=partition_hash
            ).reindex(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_THRESHOLD_COLUMNS
            )
        )

        technical = outputs.get("technical_events", pd.DataFrame()).copy()
        if not technical.empty:
            technical.insert(0, "alternative_partition_sha256", partition_hash)
            event_parts.append(
                technical.reindex(
                    columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_TECHNICAL_EVENT_COLUMNS
                )
            )

        compared = compare_threshold_registries_on_validation(
            reference_thresholds,
            alternative_thresholds,
            validation_subset,
            object_predictions,
            pixel_predictions,
            reference_validation_metrics=reference_metrics,
            metric_names=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRICS,
        )
        metrics = annotate_practical_effects(compared["metrics"])
        metrics.insert(0, "alternative_partition_sha256", partition_hash)
        metrics["selection_influence"] = False
        metric_parts.append(
            metrics.reindex(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRIC_COLUMNS
            )
        )

        decisions = compared["decisions"].copy()
        decisions.insert(0, "alternative_partition_sha256", partition_hash)
        decisions["selection_influence"] = False
        decision_parts.append(
            decisions.reindex(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_DECISION_COLUMNS
            )
        )

    return {
        "thresholds": (
            pd.concat(threshold_parts, ignore_index=True, sort=False)
            if threshold_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_THRESHOLD_COLUMNS
            )
        ),
        "metrics": (
            pd.concat(metric_parts, ignore_index=True, sort=False)
            if metric_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRIC_COLUMNS
            )
        ),
        "decisions": (
            pd.concat(decision_parts, ignore_index=True, sort=False)
            if decision_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_DECISION_COLUMNS
            )
        ),
        "technical_events": (
            pd.concat(event_parts, ignore_index=True, sort=False)
            if event_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_TECHNICAL_EVENT_COLUMNS
            )
        ),
    }



# ---------------------------------------------------------------------------
# 8. Supporting Pareto-front jackknife robustness
# ---------------------------------------------------------------------------


def build_pareto_front_robustness(
    selection_members: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    *,
    base_random_states: Sequence[int] = expcfg.SIMCA_ROBUSTNESS_BASE_RANDOM_STATES,
    epsilon: float = expcfg.SIMCA_ROBUSTNESS_PARETO_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Jackknife the official front by omitting one base seed at a time.

    The official candidate universe is frozen. Only stochastic executions lose
    the omitted seed; deterministic 04C executions remain unchanged. The
    function is supporting-only and never replaces the official front.
    """
    require_columns(
        selection_members,
        expcfg.SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS,
        "selection_members",
    )
    require_columns(
        pareto_candidates,
        (
            "model_id",
            "track_id",
            "protocol_pareto_eligible",
            "is_protocol_pareto",
        ),
        "pareto_candidates",
    )
    members = selection_members.copy()
    members["model_id"] = members["model_id"].astype(str)
    members["track_id"] = members["track_id"].astype(str)
    members["random_state"] = pd.to_numeric(
        members["random_state"], errors="raise"
    ).astype(int)

    universe = pareto_candidates.loc[
        pareto_candidates["protocol_pareto_eligible"].fillna(False).astype(bool),
        ["model_id", "track_id", "is_protocol_pareto"],
    ].drop_duplicates()
    universe["model_id"] = universe["model_id"].astype(str)
    universe["track_id"] = universe["track_id"].astype(str)
    if universe.empty:
        return (
            pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_REPLICATE_COLUMNS),
            pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_SUMMARY_COLUMNS),
            pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_AUDIT_COLUMNS),
        )

    members = members.merge(
        universe[["model_id", "track_id"]],
        on=["model_id", "track_id"],
        how="inner",
        validate="many_to_one",
    )
    base_states = tuple(dict.fromkeys(map(int, base_random_states)))
    if len(base_states) < 2:
        raise ValueError("Pareto jackknife needs at least two base random states.")

    # Defensive contract: post-Pareto additional seeds can never enter this
    # diagnostic, even if a caller accidentally supplies the expanded member
    # table instead of the frozen base-04C table.
    members = members.loc[members["random_state"].isin(base_states)].copy()

    replicate_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for omitted_state in base_states:
        keep = ~(
            members["is_stochastic"].fillna(False).astype(bool)
            & members["random_state"].eq(int(omitted_state))
        )
        reduced = members.loc[keep].copy()
        aggregated = aggregate_repeated_execution_metrics(reduced)

        for track_id, track_universe in universe.groupby("track_id", sort=False):
            track_id = str(track_id)
            spec = expcfg.SIMCA_ROBUSTNESS_PARETO_OBJECTIVES[track_id]
            minimize = tuple(map(str, spec.get("minimize", ())))
            maximize = tuple(map(str, spec.get("maximize", ())))
            objectives = [*minimize, *maximize]
            track = aggregated.loc[
                aggregated["track_id"].astype(str).eq(track_id)
                & aggregated["model_id"].astype(str).isin(
                    set(track_universe["model_id"].astype(str))
                )
            ].copy()
            if track.empty:
                continue
            missing_objectives = [column for column in objectives if column not in track]
            if missing_objectives:
                raise KeyError(
                    f"{track_id}: missing Pareto objectives in jackknife aggregate: "
                    f"{missing_objectives}."
                )
            finite = np.isfinite(track[objectives].to_numpy(dtype=float)).all(axis=1)
            valid = track.loc[finite].copy()
            if valid.empty:
                replicate_front: set[str] = set()
                witness = pd.Series(dtype="object")
                is_front = pd.Series(dtype=bool)
            else:
                is_front, witness = pareto_front_with_witness(
                    valid,
                    minimize_cols=minimize,
                    maximize_cols=maximize,
                    epsilon=float(epsilon),
                )
                replicate_front = set(
                    valid.loc[is_front, "model_id"].astype(str)
                )

            reference_front = set(
                track_universe.loc[
                    track_universe["is_protocol_pareto"].fillna(False).astype(bool),
                    "model_id",
                ].astype(str)
            )
            union = reference_front | replicate_front
            intersection = reference_front & replicate_front
            jaccard = float(len(intersection) / len(union)) if union else 1.0

            valid_status = pd.DataFrame(
                {
                    "model_id": valid["model_id"].astype(str).to_numpy(),
                    "replicate_is_pareto": is_front.to_numpy(dtype=bool),
                    "dominated_by_model_id": witness.astype(str).to_numpy(),
                }
            ) if len(valid) else pd.DataFrame(
                columns=["model_id", "replicate_is_pareto", "dominated_by_model_id"]
            )
            track_rows = track_universe.merge(
                valid_status,
                on="model_id",
                how="left",
                validate="one_to_one",
            )
            track_rows["replicate_is_pareto"] = (
                track_rows["replicate_is_pareto"].fillna(False).astype(bool)
            )
            track_rows["dominated_by_model_id"] = (
                track_rows["dominated_by_model_id"].fillna("").astype(str)
            )
            counts = reduced.loc[
                reduced["track_id"].astype(str).eq(track_id)
            ].groupby("model_id")["random_state"].nunique()
            track_rows["omitted_random_state"] = int(omitted_state)
            track_rows["n_random_states_used"] = (
                track_rows["model_id"].map(counts).fillna(0).astype(int)
            )
            track_rows = track_rows.rename(
                columns={"is_protocol_pareto": "reference_is_protocol_pareto"}
            )
            track_rows["selection_influence"] = False
            replicate_parts.append(
                track_rows.reindex(
                    columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_REPLICATE_COLUMNS
                )
            )
            audit_rows.append(
                {
                    "track_id": track_id,
                    "omitted_random_state": int(omitted_state),
                    "n_candidate_models": int(len(track_universe)),
                    "n_reference_pareto": int(len(reference_front)),
                    "n_replicate_pareto": int(len(replicate_front)),
                    "pareto_jaccard_vs_reference": jaccard,
                    "selection_influence": False,
                }
            )

    replicates = (
        pd.concat(replicate_parts, ignore_index=True, sort=False)
        if replicate_parts
        else pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_REPLICATE_COLUMNS)
    )
    audit = pd.DataFrame(audit_rows).reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_AUDIT_COLUMNS
    )

    if replicates.empty:
        summary = pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_SUMMARY_COLUMNS
        )
    else:
        summary = (
            replicates.groupby(
                ["model_id", "track_id"],
                as_index=False,
                sort=False,
                dropna=False,
            )
            .agg(
                reference_is_protocol_pareto=("reference_is_protocol_pareto", "first"),
                n_replicates=("omitted_random_state", "nunique"),
                n_pareto_replicates=("replicate_is_pareto", "sum"),
            )
        )
        summary["pareto_membership_frequency"] = (
            summary["n_pareto_replicates"] / summary["n_replicates"]
        )
        summary["front_stability_status"] = np.select(
            [
                summary["pareto_membership_frequency"].eq(1.0),
                summary["pareto_membership_frequency"].eq(0.0),
            ],
            ["stable_member", "stable_non_member"],
            default="membership_sensitive_to_base_seed",
        )
        summary["selection_influence"] = False
        summary = summary.reindex(
            columns=expcfg.SIMCA_ROBUSTNESS_PARETO_ROBUSTNESS_SUMMARY_COLUMNS
        )

    replicate_key = ["model_id", "track_id", "omitted_random_state"]
    audit_key = ["track_id", "omitted_random_state"]
    if len(replicates) and replicates.duplicated(replicate_key).any():
        raise RuntimeError("Pareto robustness replicates duplicate their natural key.")
    if len(audit) and audit.duplicated(audit_key).any():
        raise RuntimeError("Pareto robustness audit duplicates its natural key.")
    return replicates, summary, audit



# ---------------------------------------------------------------------------
# 9. Supporting local spatial-lock sensitivity
# ---------------------------------------------------------------------------


def _adjacent_values(value: int, ordered_values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(map(int, ordered_values))))
    if int(value) not in values:
        return ()
    index = values.index(int(value))
    out: list[int] = []
    if index > 0:
        out.append(values[index - 1])
    if index + 1 < len(values):
        out.append(values[index + 1])
    return tuple(out)

def build_spatial_sensitivity_plan(
    spatial_lock: Mapping,
    *,
    candidate_grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one-factor local neighbors from the original 03C candidate grid.

    Existing ``spatial_candidate_id`` values are reused. No sensitivity-specific
    surrogate ID is created.
    """
    parameters_by_track = spatial_lock.get("selected_parameters_by_track")
    if not isinstance(parameters_by_track, Mapping):
        raise RuntimeError("Spatial lock has no selected_parameters_by_track mapping.")
    grid = build_spatial_candidate_grid() if candidate_grid is None else candidate_grid.copy()
    required = {
        "spatial_candidate_id",
        "connectivity",
        "morphology_operation",
        "morphology_radius",
        "min_area_pixels",
    }
    missing = sorted(required - set(grid.columns))
    if missing:
        raise KeyError(f"Spatial candidate grid is missing columns: {missing}.")

    grid = grid.loc[:, sorted(required)].copy()
    grid["spatial_candidate_id"] = grid["spatial_candidate_id"].astype(str)
    for column in ("connectivity", "morphology_radius", "min_area_pixels"):
        grid[column] = pd.to_numeric(grid[column], errors="raise").astype(int)
    grid["morphology_operation"] = grid["morphology_operation"].astype(str)

    min_area_values = tuple(sorted(grid["min_area_pixels"].unique()))
    rows: list[dict[str, object]] = []

    for track_id, raw in sorted(parameters_by_track.items()):
        base = dict(raw)
        base_key = (
            int(base["connectivity"]),
            str(base["morphology_operation"]),
            int(base["morphology_radius"]),
            int(base["min_area_pixels"]),
        )
        variants: list[tuple[str, str, tuple[int, str, int, int]]] = []

        for value in sorted(set(grid["connectivity"])):
            value = int(value)
            if value != base_key[0]:
                variants.append(
                    ("connectivity", "connectivity_neighbor", (value, base_key[1], base_key[2], base_key[3]))
                )

        for value in _adjacent_values(base_key[3], min_area_values):
            variants.append(
                ("min_area_pixels", "adjacent_min_area", (base_key[0], base_key[1], base_key[2], int(value)))
            )

        same_operation = grid.loc[
            grid["morphology_operation"].eq(base_key[1]), "morphology_radius"
        ]
        for value in _adjacent_values(base_key[2], same_operation.tolist()):
            variants.append(
                ("morphology", "adjacent_radius", (base_key[0], base_key[1], int(value), base_key[3]))
            )

        # Operation sensitivity is treated as one conceptual morphology factor;
        # the alternative must be an actual grid member at the same min area.
        operation_candidates = grid.loc[
            grid["min_area_pixels"].eq(base_key[3])
            & ~grid["morphology_operation"].eq(base_key[1])
        ].copy()
        for operation, group in operation_candidates.groupby(
            "morphology_operation", sort=False
        ):
            operation = str(operation)
            if operation == "none":
                radius = 0
            else:
                possible = sorted(set(map(int, group["morphology_radius"])))
                radius = base_key[2] if base_key[2] in possible else min(possible)
            variants.append(
                ("morphology", "operation_neighbor", (base_key[0], operation, int(radius), base_key[3]))
            )

        seen: set[tuple[str, tuple[int, str, int, int]]] = set()
        for factor, variant_type, key in variants:
            if key == base_key or (factor, key) in seen:
                continue
            seen.add((factor, key))
            match = grid.loc[
                grid["connectivity"].eq(key[0])
                & grid["morphology_operation"].eq(key[1])
                & grid["morphology_radius"].eq(key[2])
                & grid["min_area_pixels"].eq(key[3])
            ]
            if len(match) != 1:
                continue
            candidate_id = str(match.iloc[0]["spatial_candidate_id"])
            rows.append(
                {
                    "track_id": str(track_id),
                    "factor": str(factor),
                    "variant_type": str(variant_type),
                    "reference_spatial_candidate_id": str(base["spatial_candidate_id"]),
                    "alternative_spatial_candidate_id": candidate_id,
                    "connectivity": int(key[0]),
                    "morphology_operation": str(key[1]),
                    "morphology_radius": int(key[2]),
                    "min_area_pixels": int(key[3]),
                    "selection_influence": False,
                }
            )

    plan = pd.DataFrame(rows)
    if plan.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_PLAN_COLUMNS)
    key = ["track_id", "factor", "alternative_spatial_candidate_id"]
    plan = plan.drop_duplicates(key).reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_PLAN_COLUMNS
    )
    if plan.duplicated(key).any():
        raise RuntimeError("Spatial sensitivity plan duplicates its natural key.")
    return plan.sort_values(key, kind="mergesort").reset_index(drop=True)

def _derived_single_track_lock(
    spatial_lock: Mapping,
    plan_row: Mapping[str, object],
    *,
    n_models: int,
    n_executions: int,
) -> dict:
    """Build an in-memory diagnostic lock without changing the official 03C lock."""
    derived = copy.deepcopy(dict(spatial_lock))
    derived.pop("lock_sha256", None)
    track_id = str(plan_row["track_id"])
    raw = derived.get("selected_parameters_by_track")
    if not isinstance(raw, Mapping) or track_id not in raw:
        raise KeyError(f"Track {track_id!r} is absent from the spatial lock.")
    params = dict(raw[track_id])
    params.update(
        {
            "spatial_candidate_id": str(plan_row["alternative_spatial_candidate_id"]),
            "connectivity": int(plan_row["connectivity"]),
            "morphology_operation": str(plan_row["morphology_operation"]),
            "morphology_radius": int(plan_row["morphology_radius"]),
            "min_area_pixels": int(plan_row["min_area_pixels"]),
        }
    )
    derived["selected_parameters_by_track"] = {track_id: params}
    derived["spatial_track_ids"] = [track_id]
    if "selected_counts_by_track" in derived:
        derived["selected_counts_by_track"] = {
            track_id: {"models": int(n_models), "executions": int(n_executions)}
        }
    derived["n_selected_models"] = int(n_models)
    derived["n_selected_executions"] = int(n_executions)
    derived["lock_sha256"] = spatial_cal._payload_hash(derived)  # preserve 03C hash semantics
    return derived

def evaluate_spatial_sensitivity(
    sensitivity_plan: pd.DataFrame,
    validation_executions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    image_db: Mapping,
    spatial_lock: Mapping,
    reference_spatial_metrics: pd.DataFrame,
    *,
    metric_names: Sequence[str] = (
        expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_METRICS
    ),
) -> pd.DataFrame:
    """Evaluate local 03C-lock neighbors with the canonical spatial evaluator.

    A unique ``(track_id, alternative_spatial_candidate_id)`` is evaluated once,
    even if the same grid point is reachable through several diagnostic factor
    labels. The official 03C lock is never mutated or persisted.
    """
    require_columns(
        sensitivity_plan,
        expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_PLAN_COLUMNS,
        "spatial sensitivity plan",
    )
    require_columns(
        reference_spatial_metrics,
        expcfg.SIMCA_SPATIAL_COMPONENT_METRIC_COLUMNS,
        "reference spatial metrics",
    )
    if sensitivity_plan.empty:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_COLUMNS
        )
    assert_supporting_only(sensitivity_plan, name="spatial sensitivity plan")

    metric_names = tuple(map(str, metric_names))
    reference = reference_spatial_metrics.loc[
        reference_spatial_metrics["aggregation_level"].astype(str).eq("overall")
        & reference_spatial_metrics["map_variant"].astype(str).eq(
            str(expcfg.SIMCA_ROBUSTNESS_SPATIAL_MAP_VARIANT)
        )
    ].copy()

    # One heavy spatial evaluation per unique candidate/track.
    candidate_columns = [
        "track_id",
        "alternative_spatial_candidate_id",
        "connectivity",
        "morphology_operation",
        "morphology_radius",
        "min_area_pixels",
    ]
    candidate_plan = (
        sensitivity_plan[candidate_columns]
        .drop_duplicates(["track_id", "alternative_spatial_candidate_id"])
        .reset_index(drop=True)
    )

    candidate_results: list[pd.DataFrame] = []
    for plan_row in candidate_plan.to_dict("records"):
        track_id = str(plan_row["track_id"])
        track_executions = validation_executions.loc[
            validation_executions["track_id"].astype(str).eq(track_id)
        ].copy()
        if track_executions.empty:
            continue

        run_keys = track_executions[
            ["model_id", "random_state"]
        ].drop_duplicates()
        track_thresholds = selected_thresholds.merge(
            run_keys,
            on=["model_id", "random_state"],
            how="inner",
            validate="many_to_one",
        )
        projection_ids = set(track_executions["projection_id"].astype(str))
        track_pixels = pixel_predictions.loc[
            pixel_predictions["projection_id"].astype(str).isin(projection_ids)
        ].copy()

        derived_lock = _derived_single_track_lock(
            spatial_lock,
            plan_row,
            n_models=int(track_executions["model_id"].astype(str).nunique()),
            n_executions=int(len(run_keys)),
        )
        outputs = build_locked_spatial_validation_outputs(
            track_executions,
            track_thresholds,
            track_pixels,
            image_db,
            derived_lock,
        )
        alternative = outputs["spatial_component_metrics"].loc[
            lambda frame: (
                frame["aggregation_level"].astype(str).eq("overall")
                & frame["map_variant"].astype(str).eq(
                    str(expcfg.SIMCA_ROBUSTNESS_SPATIAL_MAP_VARIANT)
                )
            )
        ].copy()
        ref = reference.loc[
            reference["track_id"].astype(str).eq(track_id)
        ].copy()

        key = ["model_id", "random_state", "track_id"]
        columns = [
            metric
            for metric in metric_names
            if metric in ref.columns and metric in alternative.columns
        ]
        if not columns:
            continue

        compared = ref[key + columns].merge(
            alternative[key + columns],
            on=key,
            how="inner",
            suffixes=("_reference", "_alternative"),
            validate="one_to_one",
        )
        if compared.empty:
            continue

        long_parts = []
        for metric in columns:
            part = compared[key].copy()
            part["alternative_spatial_candidate_id"] = str(
                plan_row["alternative_spatial_candidate_id"]
            )
            part["metric"] = str(metric)
            part["reference_value"] = pd.to_numeric(
                compared[f"{metric}_reference"], errors="coerce"
            )
            part["alternative_value"] = pd.to_numeric(
                compared[f"{metric}_alternative"], errors="coerce"
            )
            long_parts.append(part)
        candidate_results.append(
            pd.concat(long_parts, ignore_index=True, sort=False)
        )

    if not candidate_results:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_COLUMNS
        )

    effects = pd.concat(candidate_results, ignore_index=True, sort=False)
    labels = sensitivity_plan[
        ["track_id", "factor", "alternative_spatial_candidate_id"]
    ].drop_duplicates()
    effects = effects.merge(
        labels,
        on=["track_id", "alternative_spatial_candidate_id"],
        how="left",
        validate="many_to_many",
    )
    effects["delta"] = (
        effects["alternative_value"] - effects["reference_value"]
    )
    effects = annotate_practical_effects(
        effects,
        directional_status_col="directional_status",
        alternative_label="alternative",
        reference_label="reference",
    )
    effects["selection_influence"] = False

    out = effects.reindex(
        columns=expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_COLUMNS
    )
    key = [
        "model_id",
        "random_state",
        "track_id",
        "factor",
        "alternative_spatial_candidate_id",
        "metric",
    ]
    if out.duplicated(key).any():
        raise RuntimeError(
            "Spatial sensitivity metrics duplicate their natural key."
        )
    return out.sort_values(key, kind="mergesort").reset_index(drop=True)



# ---------------------------------------------------------------------------
# 10. Supporting exact one-factor ablations
# ---------------------------------------------------------------------------


def build_robustness_ablation_plan(
    model_catalog: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    evaluated_models: pd.DataFrame | None = None,
    *,
    factor_columns: Mapping[str, Sequence[str]] = (
        expcfg.SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS
    ),
) -> pd.DataFrame:
    """Find exact one-factor counterfactuals with vectorized self-joins.

    Reference models are protocol-Pareto models. Counterfactuals may be any
    model actually evaluated in 04C inside the same track. No ablation-specific
    surrogate identifier is created.
    """
    validate_simca_table_columns(
        model_catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        table_name="03B model_catalog",
    )
    validate_simca_table_columns(
        pareto_candidates,
        ("model_id", "track_id", "is_protocol_pareto"),
        table_name="validation Pareto candidates",
    )
    if evaluated_models is None:
        # Backward-compatible fallback for the current notebook-05 draft.
        # The improved notebook should pass the complete 04C evaluated model
        # table so exact counterfactuals may be found outside the Pareto front.
        evaluated_models = pareto_candidates
    validate_simca_table_columns(
        evaluated_models,
        ("model_id", "track_id"),
        table_name="04C evaluated models",
    )

    evaluated_ids = set(evaluated_models["model_id"].astype(str))
    pareto_ids = set(
        pareto_candidates.loc[
            pareto_candidates["is_protocol_pareto"].fillna(False).astype(bool),
            "model_id",
        ].astype(str)
    )
    catalog = model_catalog.loc[
        model_catalog["model_id"].astype(str).isin(evaluated_ids)
    ].copy()
    catalog["model_id"] = catalog["model_id"].astype(str)
    catalog["track_id"] = catalog["track_id"].astype(str)
    if catalog["model_id"].duplicated().any():
        raise RuntimeError("model_catalog.model_id must be unique.")

    parameter_columns = [
        column
        for column in expcfg.SIMCA_MODEL_PARAMETER_COLUMNS
        if column not in {"evaluation_track", "track_id", "parent_track"}
        and column in catalog.columns
    ]
    references = catalog.loc[catalog["model_id"].isin(pareto_ids)].copy()
    if references.empty:
        return pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_PLAN_COLUMNS
        )

    parts: list[pd.DataFrame] = []
    for factor, raw_columns in factor_columns.items():
        factor = str(factor)
        factor_cols = tuple(
            column
            for column in map(str, raw_columns)
            if column in parameter_columns
        )
        if not factor_cols:
            continue

        non_factor = [
            column for column in parameter_columns if column not in factor_cols
        ]
        join_columns = ["track_id", *non_factor]

        left = references[
            ["model_id", "track_id", *non_factor, *factor_cols]
        ].rename(columns={"model_id": "reference_model_id"})
        right = catalog[
            ["model_id", "track_id", *non_factor, *factor_cols]
        ].rename(columns={"model_id": "ablated_model_id"})

        pairs = left.merge(
            right,
            on=join_columns,
            how="inner",
            suffixes=("__reference", "__alternative"),
            validate="many_to_many",
        )
        pairs = pairs.loc[
            ~pairs["reference_model_id"].astype(str).eq(
                pairs["ablated_model_id"].astype(str)
            )
        ].copy()
        if pairs.empty:
            continue

        changed = np.zeros(len(pairs), dtype=bool)
        for column in factor_cols:
            reference_value = (
                pairs[f"{column}__reference"]
                .astype("string")
                .fillna("<NA>")
            )
            alternative_value = (
                pairs[f"{column}__alternative"]
                .astype("string")
                .fillna("<NA>")
            )
            changed |= ~reference_value.eq(alternative_value).to_numpy()
        pairs = pairs.loc[changed].copy()
        if pairs.empty:
            continue

        # If both endpoints are Pareto, keep one deterministic orientation.
        alternative_is_pareto = pairs["ablated_model_id"].astype(str).isin(
            pareto_ids
        )
        pairs = pairs.loc[
            ~alternative_is_pareto
            | pairs["reference_model_id"].astype(str).lt(
                pairs["ablated_model_id"].astype(str)
            )
        ].copy()
        if pairs.empty:
            continue

        pairs["factor"] = factor
        pairs["selection_influence"] = False
        parts.append(
            pairs.reindex(
                columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_PLAN_COLUMNS
            )
        )

    out = (
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else pd.DataFrame(
            columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_PLAN_COLUMNS
        )
    )
    key = ["track_id", "reference_model_id", "ablated_model_id", "factor"]
    out = out.drop_duplicates(key).reset_index(drop=True)
    if len(out) and out.duplicated(key).any():
        raise RuntimeError("Ablation plan duplicates its natural key.")
    return out

def build_ablation_diagnostics(
    ablation_plan: pd.DataFrame,
    selection_members: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate exact counterfactuals as paired effects on common base seeds."""
    if ablation_plan is None or ablation_plan.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS)
    validate_simca_table_columns(
        ablation_plan,
        expcfg.SIMCA_ROBUSTNESS_ABLATION_PLAN_COLUMNS,
        table_name="robustness ablation plan",
    )
    member_required = set(expcfg.SIMCA_ROBUSTNESS_SELECTION_MEMBER_COLUMNS)
    unit_required = set(expcfg.SIMCA_ROBUSTNESS_SELECTION_UNIT_COLUMNS)

    if member_required.issubset(selection_members.columns):
        long = selection_members_to_long_metrics(selection_members)
    elif unit_required.issubset(selection_members.columns):
        # Backward-compatible bridge for the current notebook-05 draft.
        # Updated notebook code should pass selection_members_df so effects are
        # paired on the real common random states.
        available_metrics = [
            metric
            for metric in expcfg.SIMCA_ROBUSTNESS_ABLATION_METRICS
            if metric in selection_members.columns
        ]
        if not available_metrics:
            return pd.DataFrame(
                columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS
            )
        long = selection_members[
            ["model_id", "track_id", *available_metrics]
        ].melt(
            id_vars=["model_id", "track_id"],
            value_vars=available_metrics,
            var_name="metric",
            value_name="value",
        )
        long["random_state"] = -1
        long = long[
            ["model_id", "track_id", "random_state", "metric", "value"]
        ]
    else:
        raise KeyError(
            "Ablation diagnostics require either the notebook-05 "
            "selection-member or selection-unit contract."
        )

    long = long.loc[
        long["metric"].astype(str).isin(
            set(map(str, expcfg.SIMCA_ROBUSTNESS_ABLATION_METRICS))
        )
    ].copy()
    if long.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS)

    plan_key = ["track_id", "reference_model_id", "ablated_model_id", "factor"]
    ref = long.rename(
        columns={"model_id": "reference_model_id", "value": "reference_seed_value"}
    )
    alt = long.rename(
        columns={"model_id": "ablated_model_id", "value": "ablated_seed_value"}
    )
    paired = ablation_plan[plan_key].merge(
        ref[["track_id", "reference_model_id", "random_state", "metric", "reference_seed_value"]],
        on=["track_id", "reference_model_id"],
        how="left",
        validate="one_to_many",
    ).merge(
        alt[["track_id", "ablated_model_id", "random_state", "metric", "ablated_seed_value"]],
        on=["track_id", "ablated_model_id", "random_state", "metric"],
        how="inner",
        validate="many_to_one",
    )
    paired["effect_seed"] = paired["ablated_seed_value"] - paired["reference_seed_value"]
    paired = paired.loc[
        np.isfinite(paired["reference_seed_value"].to_numpy(float))
        & np.isfinite(paired["ablated_seed_value"].to_numpy(float))
    ].copy()
    if paired.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS)

    group_key = [*plan_key, "metric"]
    out = (
        paired.groupby(group_key, as_index=False, sort=False, dropna=False)
        .agg(
            n_paired_random_states=("random_state", "nunique"),
            reference_value=("reference_seed_value", finite_mean),
            ablated_value=("ablated_seed_value", finite_mean),
            effect=("effect_seed", finite_mean),
            effect_std=("effect_seed", lambda values: finite_std(values, ddof=0)),
            effect_min=("effect_seed", finite_min),
            effect_max=("effect_seed", finite_max),
        )
    )
    out = annotate_practical_effects(
        out,
        delta_col="effect",
        directional_status_col="directional_status",
        alternative_label="ablated",
        reference_label="reference",
    )
    out["selection_influence"] = False
    return _reindex_contract(out, expcfg.SIMCA_ROBUSTNESS_ABLATION_DIAGNOSTIC_COLUMNS)

def build_ablation_coverage(
    ablation_plan: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    *,
    factor_columns: Mapping[str, Sequence[str]] = expcfg.SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Report identifiability of exact one-factor counterfactuals."""
    validate_simca_table_columns(
        pareto_candidates,
        ("model_id", "track_id", "is_protocol_pareto"),
        table_name="pareto_candidates",
    )
    pareto = pareto_candidates.loc[
        pareto_candidates["is_protocol_pareto"].fillna(False).astype(bool),
        ["model_id", "track_id"],
    ].copy()
    tracks = pd.DataFrame({"track_id": list(map(str, expcfg.SIMCA_ROBUSTNESS_TRACK_IDS))})
    factors = pd.DataFrame({"factor": list(map(str, factor_columns))})
    universe = tracks.merge(factors, how="cross")
    reference_counts = pareto.groupby("track_id")["model_id"].nunique().rename(
        "n_reference_pareto_models"
    )
    universe["n_reference_pareto_models"] = (
        universe["track_id"].map(reference_counts).fillna(0).astype(int)
    )

    if ablation_plan is None or ablation_plan.empty:
        pair_counts = pd.DataFrame(
            columns=["track_id", "factor", "n_exact_counterfactual_pairs", "n_reference_models_with_counterfactual"]
        )
    else:
        pair_counts = (
            ablation_plan.groupby(["track_id", "factor"], as_index=False, sort=False)
            .agg(
                n_exact_counterfactual_pairs=("reference_model_id", "size"),
                n_reference_models_with_counterfactual=("reference_model_id", "nunique"),
            )
        )
    out = universe.merge(
        pair_counts,
        on=["track_id", "factor"],
        how="left",
        validate="one_to_one",
    )
    for column in ("n_exact_counterfactual_pairs", "n_reference_models_with_counterfactual"):
        out[column] = out[column].fillna(0).astype(int)
    out["reference_coverage_rate"] = np.where(
        out["n_reference_pareto_models"].gt(0),
        out["n_reference_models_with_counterfactual"] / out["n_reference_pareto_models"],
        np.nan,
    )
    out["coverage_status"] = np.select(
        [
            out["n_reference_pareto_models"].eq(0),
            out["n_reference_models_with_counterfactual"].eq(out["n_reference_pareto_models"]) & out["n_reference_pareto_models"].gt(0),
            out["n_reference_models_with_counterfactual"].gt(0),
        ],
        [
            "not_applicable_no_pareto_reference",
            "complete",
            "partial",
        ],
        default="not_estimable_no_exact_counterfactual",
    )
    out["selection_influence"] = False
    return _reindex_contract(out, expcfg.SIMCA_ROBUSTNESS_ABLATION_COVERAGE_COLUMNS)



# ---------------------------------------------------------------------------
# 11. Descriptive uncertainty envelope and risk/coverage
# ---------------------------------------------------------------------------


def build_descriptive_uncertainty_envelope(
    review_models: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Envelope the persisted 04C per-execution intervals; never call it a cluster CI."""
    if review_models is None or review_models.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_STATISTICAL_UNCERTAINTY_COLUMNS)
    validate_simca_table_columns(review_models, ("model_id", "track_id"), table_name="review models")
    validate_simca_table_columns(
        validation_metrics,
        expcfg.SIMCA_VALIDATION_METRIC_COLUMNS,
        table_name="04C validation_metrics",
    )
    model_ids = set(review_models["model_id"].astype(str))
    metrics = validation_metrics.loc[
        validation_metrics["model_id"].astype(str).isin(model_ids)
        & validation_metrics["map_variant"].astype(str).eq(
            expcfg.SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT
        )
    ].copy()
    overall = metrics.loc[
        metrics["aggregation_level"].astype(str).eq("overall")
        & metrics["group_id"].astype(str).eq("all")
        & metrics["status"].astype(str).eq("calculable")
    ].copy()
    image = metrics.loc[
        metrics["aggregation_level"].astype(str).eq("source_image")
        & metrics["status"].astype(str).eq("calculable")
    ].copy()
    for column in ("value", "ci_low", "ci_high"):
        overall[column] = pd.to_numeric(overall[column], errors="coerce")

    group_key = ["model_id", "track_id", "decision_scope", "metric"]
    summary = (
        overall.groupby(group_key, as_index=False, sort=False, dropna=False)
        .agg(
            estimate=("value", finite_mean),
            interval_envelope_low=("ci_low", finite_min),
            interval_envelope_high=("ci_high", finite_max),
            n_random_states=("random_state", "nunique"),
        )
    )
    image_counts = (
        image.groupby(
            ["model_id", "track_id", "decision_scope"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(n_independent_images=("group_id", "nunique"))
    )
    summary = summary.merge(
        image_counts,
        on=["model_id", "track_id", "decision_scope"],
        how="left",
        validate="many_to_one",
    )
    summary["n_independent_images"] = summary["n_independent_images"].fillna(0).astype(int)
    interval_available = np.isfinite(summary["interval_envelope_low"].to_numpy(float)) & np.isfinite(
        summary["interval_envelope_high"].to_numpy(float)
    )
    summary["interval_status"] = np.select(
        [
            ~interval_available,
            summary["n_independent_images"].lt(
                int(expcfg.SIMCA_ROBUSTNESS_MIN_IMAGES_FOR_CLUSTER_INTERVAL)
            ),
        ],
        [
            "descriptive_interval_envelope_unavailable",
            "descriptive_interval_envelope_low_independent_image_count",
        ],
        default="descriptive_interval_envelope_available",
    )
    summary["selection_influence"] = False
    return _reindex_contract(
        summary,
        expcfg.SIMCA_ROBUSTNESS_STATISTICAL_UNCERTAINTY_COLUMNS,
    )

def build_statistical_uncertainty(
    review_models: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-compatible function name; output semantics are descriptive only."""
    return build_descriptive_uncertainty_envelope(review_models, validation_metrics)

def build_risk_coverage_curves(
    review_models: pd.DataFrame,
    validation_executions: pd.DataFrame,
    thresholds: pd.DataFrame,
    object_predictions: pd.DataFrame,
    pixel_predictions: pd.DataFrame,
    *,
    coverage_grid: Sequence[float] = expcfg.SIMCA_ROBUSTNESS_RISK_COVERAGE_GRID,
) -> pd.DataFrame:
    """Build descriptive direct-scope selective-risk curves from saved margins."""
    if review_models is None or review_models.empty:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_RISK_COVERAGE_COLUMNS)
    model_ids = set(review_models["model_id"].astype(str))
    executions = validation_executions.loc[
        validation_executions["model_id"].astype(str).isin(model_ids)
    ].copy()
    policy = _threshold_registry(thresholds)
    direct = policy.loc[policy["decision_scope"].astype(str).eq("direct")]
    executions = executions.merge(
        direct[["model_id", "random_state", "lower_threshold", "upper_threshold"]],
        on=["model_id", "random_state"],
        how="left",
        validate="one_to_one",
    )
    requested = np.asarray(tuple(map(float, coverage_grid)), dtype=float)
    if requested.size == 0 or not np.isfinite(requested).all() or ((requested <= 0) | (requested > 1)).any():
        raise ValueError("coverage_grid must contain finite values in (0, 1].")

    seed_rows: list[dict[str, Any]] = []
    for execution in executions.to_dict("records"):
        source = (
            object_predictions
            if str(execution["projection_level"]) == "object_projection"
            else pixel_predictions
        )
        observations = source.loc[
            source["projection_id"].astype(str).eq(str(execution["projection_id"]))
        ].copy()
        if observations.empty:
            continue
        margin = pd.to_numeric(observations["simca_margin"], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(observations["truth"], errors="raise").astype(bool).to_numpy()
        if not np.isfinite(margin).all():
            continue
        lower = float(execution["lower_threshold"])
        upper = float(execution["upper_threshold"])
        if str(execution["decision_mode"]) == "2way":
            target = margin >= lower
            confidence = np.abs(margin - lower)
        else:
            target = margin >= upper
            confidence = np.minimum(np.abs(margin - lower), np.abs(margin - upper))
        cutoffs = np.quantile(confidence, np.clip(1.0 - requested, 0.0, 1.0))
        decided = confidence[:, None] >= cutoffs[None, :]

        def rates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            y = truth[mask, None]
            pred = target[mask, None]
            dec = decided[mask]
            target_den = np.count_nonzero(y & dec, axis=0)
            non_target_den = np.count_nonzero((~y) & dec, axis=0)
            miss = np.divide(
                np.count_nonzero(y & dec & (~pred), axis=0),
                target_den,
                out=np.full(len(requested), np.nan),
                where=target_den > 0,
            )
            false_accept = np.divide(
                np.count_nonzero((~y) & dec & pred, axis=0),
                non_target_den,
                out=np.full(len(requested), np.nan),
                where=non_target_den > 0,
            )
            return miss, false_accept

        if str(execution["projection_level"]) == "pixel_projection":
            image_codes, _ = pd.factorize(observations["source_image"], sort=False)
            image_miss: list[np.ndarray] = []
            image_false: list[np.ndarray] = []
            for image_code in np.unique(image_codes):
                miss_i, false_i = rates(image_codes == image_code)
                image_miss.append(miss_i)
                image_false.append(false_i)
            miss = np.nanmean(np.vstack(image_miss), axis=0)
            false_accept = np.nanmean(np.vstack(image_false), axis=0)
        else:
            miss, false_accept = rates(np.ones(len(observations), dtype=bool))

        for index, requested_coverage in enumerate(requested):
            seed_rows.append(
                {
                    "model_id": str(execution["model_id"]),
                    "track_id": str(execution["track_id"]),
                    "random_state": int(execution["random_state"]),
                    "requested_coverage": float(requested_coverage),
                    "attained_coverage": float(decided[:, index].mean()),
                    "target_miss_rate": float(miss[index]),
                    "false_accept_rate": float(false_accept[index]),
                    "n_decided": int(decided[:, index].sum()),
                }
            )
    if not seed_rows:
        return pd.DataFrame(columns=expcfg.SIMCA_ROBUSTNESS_RISK_COVERAGE_COLUMNS)
    seed_curves = pd.DataFrame(seed_rows)
    keys = ["model_id", "track_id", "requested_coverage"]
    curves = (
        seed_curves.groupby(keys, as_index=False, sort=False, dropna=False)
        .agg(
            attained_coverage=("attained_coverage", finite_mean),
            target_miss_rate=("target_miss_rate", finite_mean),
            false_accept_rate=("false_accept_rate", finite_mean),
            mean_n_decided=("n_decided", "mean"),
            n_random_states=("random_state", "nunique"),
        )
    )
    summary_rows: list[dict[str, Any]] = []

    # Use the same target-miss limit already frozen upstream, never a new limit.
    target_miss_limit = float(
        expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS["2way"]["max_fn_rate"]
    )

    for (model_id, track_id), group in curves.groupby(
        ["model_id", "track_id"],
        sort=False,
    ):
        ordered = group.loc[
            np.isfinite(group["attained_coverage"])
            & np.isfinite(group["target_miss_rate"])
        ].sort_values(
            "attained_coverage",
            kind="mergesort",
        )

        coverage = ordered[
            "attained_coverage"
        ].to_numpy(
            dtype=float
        )

        risk = ordered[
            "target_miss_rate"
        ].to_numpy(
            dtype=float
        )

        span = (
            float(
                coverage[-1]
                - coverage[0]
            )
            if len(coverage) > 1
            else 0.0
        )

        if span > 0:
            trapezoid = getattr(
                np,
                "trapezoid",
                None,
            )

            if trapezoid is None:
                trapezoid = getattr(
                    np,
                    "trapz",
                )

            auc = float(
                trapezoid(
                    risk,
                    coverage,
                )
                / span
            )
        else:
            auc = np.nan

        acceptable = ordered.loc[
            ordered[
                "target_miss_rate"
            ].le(
                target_miss_limit
            )
        ]

        summary_rows.append(
            {
                "model_id": str(
                    model_id
                ),
                "track_id": str(
                    track_id
                ),
                "selective_risk_auc": auc,
                "coverage_at_target_miss_guardrail": (
                    finite_max(
                        acceptable[
                            "attained_coverage"
                        ]
                    )
                    if not acceptable.empty
                    else np.nan
                ),
            }
        )
    curves = curves.merge(
        pd.DataFrame(summary_rows),
        on=["model_id", "track_id"],
        how="left",
        validate="many_to_one",
    )
    curves["curve_role"] = "descriptive_only_no_threshold_or_selection_influence"
    return _reindex_contract(curves, expcfg.SIMCA_ROBUSTNESS_RISK_COVERAGE_COLUMNS)



# ---------------------------------------------------------------------------
# 12. Notebook-05 child-contract hash
# ---------------------------------------------------------------------------


def robustness_contract_payload() -> dict[str, Any]:
    """Return every setting that can alter notebook-05 review/diagnostics."""
    return {
        "contract_version": str(expcfg.SIMCA_ROBUSTNESS_CONTRACT_VERSION),
        "protocol_version": str(expcfg.PROTOCOL_VERSION),
        "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
        "contract_role": str(expcfg.SIMCA_ROBUSTNESS_CONTRACT_ROLE),
        "selection_scope": str(expcfg.SIMCA_ROBUSTNESS_SELECTION_SCOPE),
        "allow_cross_track_selection": bool(expcfg.SIMCA_ROBUSTNESS_ALLOW_CROSS_TRACK_SELECTION),
        "allow_batch4_inputs": bool(expcfg.SIMCA_ROBUSTNESS_ALLOW_BATCH4_INPUTS),
        "final_model_selection_performed": bool(expcfg.SIMCA_ROBUSTNESS_FINAL_MODEL_SELECTION_PERFORMED),
        "base_random_states": list(map(int, expcfg.SIMCA_ROBUSTNESS_BASE_RANDOM_STATES)),
        "robustness_random_states": list(map(int, expcfg.SIMCA_ROBUSTNESS_RANDOM_STATES)),
        "additional_random_states": list(map(int, expcfg.SIMCA_ROBUSTNESS_ADDITIONAL_RANDOM_STATES)),
        "run_additional_seeds": bool(expcfg.SIMCA_ROBUSTNESS_RUN_ADDITIONAL_SEEDS),
        "recompute_pareto_after_additional_seeds": bool(expcfg.SIMCA_ROBUSTNESS_RECOMPUTE_PARETO_AFTER_ADDITIONAL_SEEDS),
        "reselect_threshold_policy_for_additional_seeds": bool(expcfg.SIMCA_ROBUSTNESS_RESELECT_THRESHOLD_POLICY_FOR_ADDITIONAL_SEEDS),
        "recalibrate_numeric_thresholds_for_additional_seeds": bool(expcfg.SIMCA_ROBUSTNESS_RECALIBRATE_NUMERIC_THRESHOLDS_FOR_ADDITIONAL_SEEDS),
        "stochastic_matrix_methods": list(map(str, expcfg.SIMCA_STOCHASTIC_MATRIX_METHODS)),
        "stochastic_sampling_strategies": list(map(str, expcfg.SIMCA_STOCHASTIC_SAMPLING_STRATEGIES)),
        "supported_eligibility_statuses": list(map(str, expcfg.SIMCA_ROBUSTNESS_SUPPORTED_ELIGIBILITY_STATUSES)),
        "supported_downstream_statuses": list(map(str, expcfg.SIMCA_ROBUSTNESS_SUPPORTED_DOWNSTREAM_STATUSES)),
        "protocol_candidate_statuses": list(map(str, expcfg.SIMCA_ROBUSTNESS_PROTOCOL_CANDIDATE_STATUSES)),
        "validation_map_variant": str(expcfg.SIMCA_ROBUSTNESS_VALIDATION_MAP_VARIANT),
        "spatial_map_variant": str(expcfg.SIMCA_ROBUSTNESS_SPATIAL_MAP_VARIANT),
        "pareto_epsilon": float(expcfg.SIMCA_ROBUSTNESS_PARETO_EPSILON),
        "pareto_objectives": expcfg.SIMCA_ROBUSTNESS_PARETO_OBJECTIVES,
        "pareto_seed_aggregation": "worst_observed_seed_bymetric_direction",
        "metric_directions": expcfg.SIMCA_ROBUSTNESS_METRIC_DIRECTIONS,
        "stability_limits": expcfg.SIMCA_ROBUSTNESS_STABILITY_LIMITS,
        "blocking_stability_metrics_by_track": expcfg.SIMCA_ROBUSTNESS_BLOCKING_STABILITY_METRICS_BY_TRACK,
        "decision_disagreement_limits": expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_LIMITS,
        "decision_disagreement_is_blocking": bool(expcfg.SIMCA_ROBUSTNESS_DECISION_DISAGREEMENT_IS_BLOCKING),
        "stability_registration_status": str(expcfg.SIMCA_ROBUSTNESS_STABILITY_REGISTRATION_STATUS),
        "supporting_diagnostic_rule_version": str(
            expcfg.SIMCA_ROBUSTNESS_SUPPORTING_DIAGNOSTIC_RULE_VERSION
        ),
        "supporting_diagnostic_registration_status": str(
            expcfg.SIMCA_ROBUSTNESS_SUPPORTING_DIAGNOSTIC_REGISTRATION_STATUS
        ),
        "uncertainty_summary_semantics": str(
            expcfg.SIMCA_ROBUSTNESS_UNCERTAINTY_SUMMARY_SEMANTICS
        ),
        "require_stability_for_pure_test": bool(expcfg.SIMCA_ROBUSTNESS_REQUIRE_STABILITY_FOR_PURE_TEST),
        "pure_test_stability_statuses": list(map(str, expcfg.SIMCA_ROBUSTNESS_PURE_TEST_STABILITY_STATUSES)),
        "threshold_sensitivity_registration_status": str(expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_REGISTRATION_STATUS),
        "threshold_sensitivity_direct_2way_deltas": list(map(float, expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_DIRECT_2WAY_DELTAS)),
        "threshold_sensitivity_vote_2way_deltas": list(map(float, expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_VOTE_2WAY_DELTAS)),
        "threshold_sensitivity_center_shift_fractions": list(map(float, expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_CENTER_SHIFT_FRACTIONS)),
        "threshold_sensitivity_width_scales": list(map(float, expcfg.SIMCA_ROBUSTNESS_THRESHOLD_SENSITIVITY_WIDTH_SCALES)),
        "threshold_stability_warning_limits": expcfg.SIMCA_ROBUSTNESS_THRESHOLD_STABILITY_WARNING_LIMITS,
        "source_image_influence_metrics": list(map(str, expcfg.SIMCA_ROBUSTNESS_SOURCE_IMAGE_INFLUENCE_METRICS)),
        "run_fold_sensitivity": bool(expcfg.SIMCA_ROBUSTNESS_RUN_FOLD_SENSITIVITY),
        "fold_sensitivity_generator_random_states": list(map(int, expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_RANDOM_STATES)),
        "fold_sensitivity_max_unique_alternatives": int(expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_MAX_UNIQUE_ALTERNATIVES),
        "fold_sensitivity_metrics": list(map(str, expcfg.SIMCA_ROBUSTNESS_FOLD_SENSITIVITY_METRICS)),
        "run_pareto_front_sensitivity": bool(expcfg.SIMCA_ROBUSTNESS_RUN_PARETO_FRONT_SENSITIVITY),
        "run_spatial_sensitivity": bool(expcfg.SIMCA_ROBUSTNESS_RUN_SPATIAL_SENSITIVITY),
        "spatial_sensitivity_metrics": list(map(str, expcfg.SIMCA_ROBUSTNESS_SPATIAL_SENSITIVITY_METRICS)),
        "ablation_factor_columns": expcfg.SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS,
        "ablation_metrics": list(map(str, expcfg.SIMCA_ROBUSTNESS_ABLATION_METRICS)),
        "ablation_registration_status": str(expcfg.SIMCA_ROBUSTNESS_ABLATION_REGISTRATION_STATUS),
        "sensitivity_tolerances": expcfg.SIMCA_ROBUSTNESS_SENSITIVITY_TOLERANCES,
    }

def hash_robustness_contract() -> str:
    return sha256_payload(robustness_contract_payload())

__all__ = [
    "validate_robustness_inputs",
    "validate_simca_robustness_inputs",
    "build_seed_metrics",
    "selection_members_to_long_metrics",
    "aggregate_repeated_execution_metrics",
    "build_selection_unit_metrics",
    "build_pareto_diagnostics",
    "build_robustness_seed_execution_registry",
    "compute_seed_decision_disagreement",
    "summarize_random_state_stability_metrics",
    "build_robustness_review_guardrails",
    "build_track_scoring_table",
    "build_threshold_sensitivity_plan",
    "compare_threshold_registries_on_validation",
    "evaluate_threshold_sensitivity",
    "build_threshold_stability_diagnostics",
    "build_source_image_influence_diagnostics",
    "build_calibration_fold_sensitivity",
    "evaluate_calibration_fold_sensitivity",
    "build_pareto_front_robustness",
    "build_spatial_sensitivity_plan",
    "evaluate_spatial_sensitivity",
    "build_robustness_ablation_plan",
    "build_ablation_diagnostics",
    "build_ablation_coverage",
    "build_descriptive_uncertainty_envelope",
    "build_statistical_uncertainty",
    "build_risk_coverage_curves",
    "robustness_contract_payload",
    "hash_robustness_contract",
]
