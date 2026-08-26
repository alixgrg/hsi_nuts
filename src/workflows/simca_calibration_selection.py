"""Auditable selection of SIMCA thresholds and calibrated models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import experiment_config as expcfg
from src.protocol_governance import canonical_json
from src.workflows.simca_internal_calibration import (
    INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS,
    attach_internal_calibration_runner_group_ids,
    iter_internal_calibration_checkpoint_shards_8tracks,
)
from src.workflows.simca_selection_utils import (
    pareto_front_with_witness,
)
from src.utils import require_columns, normalize_integer_sequence


_POLICY_COLUMNS = (
    "model_id",
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
_POLICY_VALUE_ATOL = 1e-7
_POLICY_CANONICAL_VALUES = {
    "lower_quantile": tuple(
        map(
            float,
            expcfg.INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES,
        )
    ),
    "upper_quantile": tuple(
        map(
            float,
            expcfg.INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES,
        )
    ),
    "vote_threshold": tuple(
        map(float, expcfg.INTERNAL_CALIBRATION_OBJECT_THRESHOLDS)
    ),
}

_SELECTION_AUDIT_STRING_COLUMNS = (
    "selection_level",
    "model_id",
    "decision_scope",
    "stage",
    "decision",
    "reason_code",
    "metric",
    "operator",
    "related_model_id",
)

_SELECTION_AUDIT_FLOAT_COLUMNS = (
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
    "observed_value",
    "reference_value",
)

_SELECTION_AUDIT_ARROW_SCHEMA = pa.schema(
    [
        *(
            pa.field(column, pa.string())
            for column in _SELECTION_AUDIT_STRING_COLUMNS[:3]
        ),
        *(
            pa.field(column, pa.float64())
            for column in _SELECTION_AUDIT_FLOAT_COLUMNS[:3]
        ),
        *(
            pa.field(column, pa.string())
            for column in _SELECTION_AUDIT_STRING_COLUMNS[3:7]
        ),
        pa.field("observed_value", pa.float64()),
        pa.field("operator", pa.string()),
        pa.field("reference_value", pa.float64()),
        pa.field("related_model_id", pa.string()),
    ]
)

_THRESHOLD_CANDIDATE_CACHE_METADATA_KEY = (
    b"hsi_nuts_threshold_candidate_cache"
)
_THRESHOLD_CANDIDATE_CACHE_FORMAT = "threshold_candidates_v1"

_MODEL_METRICS = (
    "target_miss_rate",
    "false_accept_rate",
    "uncertain_rate",
    "target_uncertain_rate",
    "non_target_uncertain_rate",
    "coverage_rate",
    "balanced_accuracy",
    "decided_balanced_accuracy",
    "worst_target_miss_rate",
    "worst_false_accept_rate",
    "worst_uncertain_rate",
    "worst_target_uncertain_rate",
    "worst_non_target_uncertain_rate",
    "minimum_coverage_rate",
    "minimum_balanced_accuracy",
    "minimum_decided_balanced_accuracy",
    "worst_unit_target_miss_rate",
    "worst_unit_false_accept_rate",
)


class _AtomicParquetStreamWriter:
    """Write bounded-size Arrow tables and publish only a complete file."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema: pa.Schema | None = None,
        schema_metadata: Mapping[bytes, bytes] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        self.schema = (
            None if schema is None else schema.remove_metadata()
        )
        self.schema_metadata = dict(schema_metadata or {})
        self._writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        normalized = table.replace_schema_metadata(None)
        if self.schema is None:
            self.schema = normalized.schema
        elif normalized.schema != self.schema:
            try:
                normalized = normalized.cast(self.schema, safe=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise RuntimeError(
                    f"Incompatible streamed parquet schema for {self.path}."
                ) from exc

        if self._writer is None:
            writer_schema = self.schema.with_metadata(
                self.schema_metadata or None
            )
            self._writer = pq.ParquetWriter(
                self.temporary_path,
                writer_schema,
                compression="zstd",
            )
        self._writer.write_table(
            normalized.replace_schema_metadata(
                self.schema_metadata or None
            )
        )
        self.row_count += int(normalized.num_rows)

    def finish(self) -> Path:
        if self._writer is None:
            raise RuntimeError(f"No rows were streamed to {self.path}.")
        self._writer.close()
        self._writer = None
        self.temporary_path.replace(self.path)
        return self.path

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.temporary_path.unlink(missing_ok=True)


def _threshold_candidate_cache_payload(
    cache_context: Mapping[str, object],
) -> dict[str, object]:
    return {
        "cache_format": _THRESHOLD_CANDIDATE_CACHE_FORMAT,
        "context": dict(cache_context),
    }


def save_threshold_candidate_cache(
    threshold_candidates: pd.DataFrame,
    path: str | Path,
    *,
    cache_context: Mapping[str, object],
) -> Path:
    """Atomically persist aggregated policies without a surrogate ID."""
    if threshold_candidates.empty:
        raise ValueError("Threshold candidates are empty.")
    require_columns(
        threshold_candidates,
        (*_POLICY_COLUMNS, "n_seeds", "n_folds", "n_run_folds"),
        "threshold candidates",
    )
    normalized = _fill_policy_values(threshold_candidates)
    if normalized.duplicated(list(_POLICY_COLUMNS)).any():
        raise RuntimeError("Threshold candidate policies are not unique.")

    ordered = _restore_policy_values(
        normalized.sort_values(
            list(_POLICY_COLUMNS),
            kind="mergesort",
        )
    ).reset_index(drop=True)
    table = pa.Table.from_pandas(ordered, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[_THRESHOLD_CANDIDATE_CACHE_METADATA_KEY] = canonical_json(
        _threshold_candidate_cache_payload(cache_context)
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _threshold_candidate_cache_writer(
    path: str | Path,
    *,
    cache_context: Mapping[str, object],
) -> _AtomicParquetStreamWriter:
    metadata = {
        _THRESHOLD_CANDIDATE_CACHE_METADATA_KEY: canonical_json(
            _threshold_candidate_cache_payload(cache_context)
        ).encode("utf-8")
    }
    return _AtomicParquetStreamWriter(
        path,
        schema_metadata=metadata,
    )


def load_threshold_candidate_cache(
    path: str | Path,
    *,
    expected_context: Mapping[str, object],
) -> pd.DataFrame:
    """Load a candidate cache only when its scientific context matches."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    parquet = pq.ParquetFile(source)
    metadata = parquet.schema_arrow.metadata or {}
    raw_payload = metadata.get(_THRESHOLD_CANDIDATE_CACHE_METADATA_KEY)
    if raw_payload is None:
        raise RuntimeError("Threshold candidate cache has no context metadata.")
    observed = raw_payload.decode("utf-8")
    expected = canonical_json(
        _threshold_candidate_cache_payload(expected_context)
    )
    if observed != expected:
        raise RuntimeError(
            "Threshold candidate cache context mismatch. Rebuild the cache."
        )

    candidates = parquet.read().to_pandas()
    require_columns(
        candidates,
        (*_POLICY_COLUMNS, "n_seeds", "n_folds", "n_run_folds"),
        "threshold candidate cache",
    )
    normalized = _fill_policy_values(candidates)
    if normalized.duplicated(list(_POLICY_COLUMNS)).any():
        raise RuntimeError("Cached threshold policies are not unique.")
    return _restore_policy_values(normalized).reset_index(drop=True)


def _selection_audit_to_arrow(audit: pd.DataFrame) -> pa.Table:
    out = audit.reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS
    ).copy()
    for column in _SELECTION_AUDIT_STRING_COLUMNS:
        out[column] = out[column].astype("string")
    for column in _SELECTION_AUDIT_FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return pa.Table.from_pandas(
        out,
        schema=_SELECTION_AUDIT_ARROW_SCHEMA,
        preserve_index=False,
        safe=True,
    )


def summarize_selection_audit(
    audit: pd.DataFrame,
    model_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build a small plotting summary without retaining the full audit."""
    require_columns(
        audit,
        (
            "selection_level",
            "model_id",
            "stage",
            "decision",
            "reason_code",
        ),
        "selection audit",
    )
    require_columns(
        model_catalog,
        ("model_id", "track_id"),
        "model catalog",
    )
    links = model_catalog[["model_id", "track_id"]].drop_duplicates()
    if links["model_id"].duplicated().any():
        raise RuntimeError("A model_id maps to multiple tracks.")
    work = audit.merge(
        links,
        on="model_id",
        how="left",
        validate="many_to_one",
    )
    group_columns = [
        "selection_level",
        "track_id",
        "stage",
        "decision",
        "reason_code",
    ]
    return (
        work.groupby(
            group_columns,
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            n_rows=("model_id", "size"),
            n_models=("model_id", "nunique"),
        )
    )


def _canonicalize_policy_values(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize checkpoint policy values onto the configured numeric grid."""
    out = frame.copy()
    for column in _POLICY_VALUE_COLUMNS:
        if column not in out:
            continue

        source = out[column]
        numeric = pd.to_numeric(
            source,
            errors="coerce",
        )
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            unexpected = sorted(
                source.loc[invalid].astype(str).drop_duplicates().tolist()
            )
            raise RuntimeError(
                f"Non-numeric {column} values in threshold policies: "
                f"{unexpected}"
            )

        # Pandas Copy-on-Write and Arrow-backed columns may expose a read-only
        # NumPy view. Canonicalization intentionally mutates this local array,
        # so always detach it from the source Series first.
        values = numeric.to_numpy(
            dtype=float,
            na_value=np.nan,
        ).copy()
        non_finite = ~np.isnan(values) & ~np.isfinite(values)
        if non_finite.any():
            raise RuntimeError(
                f"Non-finite {column} values in threshold policies."
            )

        finite_positions = np.flatnonzero(np.isfinite(values))
        if finite_positions.size:
            configured = np.asarray(
                _POLICY_CANONICAL_VALUES[column],
                dtype=float,
            )
            observed = values[finite_positions]
            distances = np.abs(
                observed[:, None] - configured[None, :]
            )
            nearest_indices = distances.argmin(axis=1)
            nearest = configured[nearest_indices]
            matches = np.isclose(
                observed,
                nearest,
                rtol=0.0,
                atol=_POLICY_VALUE_ATOL,
            )
            if not matches.all():
                unexpected = sorted(set(observed[~matches].tolist()))
                raise RuntimeError(
                    f"Non-configured {column} values in threshold policies: "
                    f"{unexpected}"
                )
            values[finite_positions] = nearest

        out[column] = values
    return out


def _fill_policy_values(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = _canonicalize_policy_values(frame)
    for column in _POLICY_VALUE_COLUMNS:
        out[column] = out[column].fillna(_POLICY_SENTINEL)
    return out


def _restore_policy_values(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    for column in _POLICY_VALUE_COLUMNS:
        out[column] = out[column].mask(
            out[column].eq(_POLICY_SENTINEL)
        )
    return out


def compare(
    values: pd.Series,
    operator: str,
    reference: float | pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(reference, pd.Series):
        reference_values = pd.to_numeric(
            reference.reindex(values.index),
            errors="coerce",
        )
    else:
        reference_values = pd.Series(
            float(reference),
            index=values.index,
            dtype=float,
        )
    finite = np.isfinite(numeric) & np.isfinite(reference_values)
    if operator == "<=":
        return finite & numeric.le(reference_values)
    if operator == ">=":
        return finite & numeric.ge(reference_values)
    if operator == "==":
        return finite & numeric.eq(reference_values)
    raise ValueError(f"Unsupported operator: {operator!r}")


def aggregate_threshold_candidates(
    threshold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate cross-fitted metrics by natural threshold policy."""
    require_columns(
        threshold_metrics,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS,
        "threshold metrics",
    )
    crossfit = threshold_metrics.loc[
        pd.to_numeric(
            threshold_metrics["evaluation_fold"],
            errors="coerce",
        ).ge(0)
    ].copy()
    base_columns = [
        *_POLICY_COLUMNS,
        "n_seeds",
        "n_folds",
        "n_run_folds",
    ]
    if crossfit.empty:
        return pd.DataFrame(columns=base_columns)

    crossfit = _fill_policy_values(crossfit)
    crossfit["value"] = pd.to_numeric(
        crossfit["value"],
        errors="raise",
    ).astype(float)
    execution_columns = [
        *_POLICY_COLUMNS,
        "random_state",
        "evaluation_fold",
    ]
    duplicate_key = [*execution_columns, "metric"]
    if crossfit.duplicated(duplicate_key).any():
        raise RuntimeError(
            "Duplicate metric rows for one threshold execution."
        )

    wide = crossfit.pivot(
        index=execution_columns,
        columns="metric",
        values="value",
    ).reset_index()
    wide.columns.name = None
    wide["_run_fold"] = list(
        zip(wide["random_state"], wide["evaluation_fold"])
    )

    aggregations = {}
    lower_metrics = (
        "target_miss_rate",
        "false_accept_rate",
        "uncertain_rate",
        "target_uncertain_rate",
        "non_target_uncertain_rate",
    )
    higher_metrics = (
        "coverage_rate",
        "balanced_accuracy",
        "decided_balanced_accuracy",
    )
    for metric in lower_metrics:
        if metric in wide:
            aggregations[metric] = (metric, "mean")
            aggregations[f"worst_{metric}"] = (metric, "max")
    for metric in higher_metrics:
        if metric in wide:
            aggregations[metric] = (metric, "mean")
            aggregations[f"minimum_{metric}"] = (metric, "min")
    if "max_unit_target_miss_rate" in wide:
        aggregations["worst_unit_target_miss_rate"] = (
            "max_unit_target_miss_rate",
            "max",
        )
    if "max_unit_false_accept_rate" in wide:
        aggregations["worst_unit_false_accept_rate"] = (
            "max_unit_false_accept_rate",
            "max",
        )

    summary = (
        wide.groupby(
            list(_POLICY_COLUMNS),
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            **aggregations,
            n_seeds=("random_state", "nunique"),
            n_folds=("evaluation_fold", "nunique"),
            n_run_folds=("_run_fold", "nunique"),
        )
    )
    return _restore_policy_values(summary)


def _value_series(
    value: float | pd.Series | None,
    index: pd.Index,
) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _threshold_audit_rows(
    frame: pd.DataFrame,
    keep: pd.Series,
    *,
    stage: str,
    reason_code: str,
    metric: str = "",
    operator: str = "",
    reference: float | pd.Series | None = None,
    kept_reason_code: str = "passed",
) -> pd.DataFrame:
    out = frame.reindex(columns=_POLICY_COLUMNS).copy()
    keep = keep.reindex(frame.index).fillna(False).astype(bool)
    out["selection_level"] = "threshold"
    out["stage"] = stage
    out["decision"] = np.where(keep, "kept", "eliminated")
    out["reason_code"] = np.where(
        keep,
        kept_reason_code,
        reason_code,
    )
    out["metric"] = metric
    out["observed_value"] = (
        pd.to_numeric(frame[metric], errors="coerce")
        if metric and metric in frame
        else np.nan
    )
    out["operator"] = operator
    out["reference_value"] = _value_series(
        reference,
        frame.index,
    ).to_numpy()
    out["related_model_id"] = ""
    return out.reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS
    )


def _rule_applicability_mask(
    frame: pd.DataFrame,
    rule: Mapping[str, object],
) -> pd.Series:
    """Return a vectorized OR-of-contexts applicability mask."""
    contexts = tuple(rule.get("applies_when", ()))
    if not contexts:
        return pd.Series(True, index=frame.index, dtype=bool)

    applicable = pd.Series(False, index=frame.index, dtype=bool)
    for context in contexts:
        if not isinstance(context, Mapping):
            raise TypeError("Each applies_when context must be a mapping.")
        context_mask = pd.Series(True, index=frame.index, dtype=bool)
        for column, expected in context.items():
            if column not in frame:
                raise KeyError(
                    f"Missing applicability column: {column!r}"
                )
            if isinstance(expected, (tuple, list, set, frozenset)):
                context_mask &= frame[column].isin(tuple(expected))
            else:
                context_mask &= frame[column].eq(expected)
        applicable |= context_mask
    return applicable


def _constraint_reference(
    frame: pd.DataFrame,
    rule: Mapping[str, object],
) -> pd.Series:
    """Resolve generic and track/scope-specific constraint references."""
    reference = pd.Series(
        float(rule["value"]),
        index=frame.index,
        dtype=float,
    )
    metric = str(rule["metric"])
    overrides = expcfg.INTERNAL_CALIBRATION_THRESHOLD_OVERRIDES
    for track_id, scope_overrides in overrides.items():
        for decision_scope, metric_overrides in scope_overrides.items():
            if metric not in metric_overrides:
                continue
            mask = (
                frame["track_id"].eq(str(track_id))
                & frame["decision_scope"].eq(str(decision_scope))
            )
            reference.loc[mask] = float(metric_overrides[metric])
    return reference


def _expected_threshold_scopes(
    configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = (
        "model_id",
        "random_state",
        "track_id",
        "decision_mode",
        "projection_level",
    )
    require_columns(configurations, required, "configurations")
    consistency = configurations.groupby(
        "model_id",
        dropna=False,
    )[["track_id", "decision_mode", "projection_level"]].nunique(
        dropna=False
    )
    if consistency.gt(1).any(axis=None):
        raise RuntimeError(
            "A model_id maps to multiple tracks, decisions or projections."
        )

    model_info = configurations[
        ["model_id", "track_id", "decision_mode", "projection_level"]
    ].drop_duplicates("model_id")
    seed_counts = configurations.groupby(
        "model_id",
        as_index=False,
    ).agg(expected_n_seeds=("random_state", "nunique"))
    model_info = model_info.merge(
        seed_counts,
        on="model_id",
        validate="one_to_one",
    )

    direct = model_info[["model_id"]].copy()
    direct["decision_scope"] = "direct"
    pixel = model_info.loc[
        model_info["projection_level"].eq("pixel_projection"),
        ["model_id"],
    ].copy()
    pixel["decision_scope"] = "pixel_to_object"
    scopes = pd.concat([direct, pixel], ignore_index=True)
    return model_info, scopes


def select_threshold_policies(
    threshold_metrics: pd.DataFrame,
    configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one threshold policy per model and decision scope."""
    candidates = aggregate_threshold_candidates(threshold_metrics)
    return select_threshold_policy_candidates(candidates, configurations)


def select_threshold_policy_candidates(
    candidates: pd.DataFrame,
    configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select policies from an already aggregated candidate table."""
    model_info, expected_scopes = _expected_threshold_scopes(
        configurations
    )
    observed_scopes = candidates[
        ["model_id", "decision_scope"]
    ].drop_duplicates()
    availability = expected_scopes.merge(
        observed_scopes.assign(_available=True),
        on=["model_id", "decision_scope"],
        how="left",
        validate="one_to_one",
    )
    available = availability["_available"].fillna(False)
    audit_parts = [
        _threshold_audit_rows(
            availability,
            available,
            stage="candidate_scope",
            reason_code="no_evaluable_threshold_policy",
        )
    ]

    if candidates.empty:
        audit = pd.concat(audit_parts, ignore_index=True)
        return candidates, audit

    candidates = candidates.merge(
        model_info,
        on="model_id",
        how="left",
        validate="many_to_one",
    )
    current = candidates.copy()
    audit_parts.append(
        _threshold_audit_rows(
            current,
            pd.Series(True, index=current.index),
            stage="candidate_policy",
            reason_code="",
        )
    )

    completeness = (
        (
            "n_seeds",
            current["expected_n_seeds"],
            "incomplete_seed_coverage",
        ),
        (
            "n_folds",
            pd.Series(
                expcfg.INTERNAL_CALIBRATION_N_SPLITS,
                index=current.index,
            ),
            "incomplete_fold_coverage",
        ),
        (
            "n_run_folds",
            current["expected_n_seeds"].mul(
                expcfg.INTERNAL_CALIBRATION_N_SPLITS
            ),
            "incomplete_run_fold_coverage",
        ),
    )
    for metric, reference, reason in completeness:
        reference = reference.reindex(current.index)
        passed = compare(current[metric], "==", reference)
        audit_parts.append(
            _threshold_audit_rows(
                current,
                passed,
                stage=f"completeness:{metric}",
                reason_code=reason,
                metric=metric,
                operator="==",
                reference=reference,
            )
        )
        current = current.loc[passed].copy()

    for decision_mode, rules in (
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_CONSTRAINTS.items()
    ):
        for rule in rules:
            subset = current.loc[
                current["decision_mode"].eq(decision_mode)
            ].copy()
            if subset.empty:
                continue
            metric = str(rule["metric"])
            if metric not in subset:
                raise KeyError(
                    f"Missing configured threshold metric: {metric}"
                )

            applicable = _rule_applicability_mask(subset, rule)
            not_applicable = subset.loc[~applicable]
            if not not_applicable.empty:
                audit_parts.append(
                    _threshold_audit_rows(
                        not_applicable,
                        pd.Series(True, index=not_applicable.index),
                        stage=f"constraint_applicability:{metric}",
                        reason_code="",
                        metric=metric,
                        kept_reason_code=(
                            "metric_not_identifiable_for_context"
                        ),
                    )
                )

            evaluated = subset.loc[applicable].copy()
            if evaluated.empty:
                continue
            reference = _constraint_reference(evaluated, rule)
            passed = compare(
                evaluated[metric],
                str(rule["operator"]),
                reference,
            )
            audit_parts.append(
                _threshold_audit_rows(
                    evaluated,
                    passed,
                    stage=f"constraint:{metric}",
                    reason_code=str(rule["reason"]),
                    metric=metric,
                    operator=str(rule["operator"]),
                    reference=reference,
                )
            )
            current = current.drop(
                index=evaluated.index[~passed]
            )

    group_columns = ["model_id", "decision_scope"]
    for rule in expcfg.INTERNAL_CALIBRATION_THRESHOLD_PRIORITY:
        applicable_modes = tuple(
            rule.get("applies_to", ("2way", "3way"))
        )
        subset = current.loc[
            current["decision_mode"].isin(applicable_modes)
        ].copy()
        if subset.empty:
            continue

        metric = str(rule["metric"])
        if metric not in subset:
            raise KeyError(
                f"Missing configured priority metric: {metric}"
            )

        applicable = _rule_applicability_mask(subset, rule)
        not_applicable = subset.loc[~applicable]
        if not not_applicable.empty:
            audit_parts.append(
                _threshold_audit_rows(
                    not_applicable,
                    pd.Series(True, index=not_applicable.index),
                    stage=f"priority_applicability:{metric}",
                    reason_code="",
                    metric=metric,
                    kept_reason_code=(
                        "metric_not_identifiable_for_context"
                    ),
                )
            )
        subset = subset.loc[applicable].copy()
        if subset.empty:
            continue

        direction = str(rule["direction"])
        tolerance = float(rule["tolerance"])
        grouped = subset.groupby(
            group_columns,
            dropna=False,
        )[metric]
        if direction == "min":
            reference = grouped.transform("min") + tolerance
            operator = "<="
        elif direction == "max":
            reference = grouped.transform("max") - tolerance
            operator = ">="
        else:
            raise ValueError(
                f"Unsupported priority direction: {direction!r}"
            )

        passed = compare(
            subset[metric],
            operator,
            reference,
        )
        audit_parts.append(
            _threshold_audit_rows(
                subset,
                passed,
                stage=f"priority:{metric}",
                reason_code=str(rule["reason"]),
                metric=metric,
                operator=operator,
                reference=reference,
            )
        )
        current = current.drop(index=subset.index[~passed])

    tie_columns = [
        str(rule["column"])
        for rule in expcfg.INTERNAL_CALIBRATION_THRESHOLD_TIEBREAK
    ]
    missing_tie_columns = sorted(set(tie_columns) - set(current.columns))
    if missing_tie_columns:
        raise KeyError(
            f"Missing threshold tie-break columns: {missing_tie_columns}"
        )
    ascending = [
        str(rule["direction"]) == "min"
        for rule in expcfg.INTERNAL_CALIBRATION_THRESHOLD_TIEBREAK
    ]
    ordered = current.sort_values(
        [*group_columns, *tie_columns],
        ascending=[True, True, *ascending],
        na_position="last",
        kind="mergesort",
    )
    selected_index = ordered.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ).head(1).index
    selected_mask = pd.Series(
        current.index.isin(selected_index),
        index=current.index,
    )
    audit_parts.append(
        _threshold_audit_rows(
            current,
            selected_mask,
            stage="deterministic_tiebreak",
            reason_code="not_first_deterministic_policy",
        )
    )
    selected = current.loc[selected_mask].copy()
    selected = selected.drop(
        columns=["expected_n_seeds", "track_id"],
        errors="ignore",
    )

    audit = pd.concat(
        audit_parts,
        ignore_index=True,
        sort=False,
    ).reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS
    )
    return selected.reset_index(drop=True), audit


def materialize_selected_thresholds(
    threshold_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize final numeric thresholds for every selected model run."""
    output_columns = (
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
    )
    if selected_policy_metrics.empty:
        return pd.DataFrame(columns=output_columns)

    required_selected = (
        *_POLICY_COLUMNS,
        "decision_mode",
        "n_seeds",
    )
    require_columns(
        selected_policy_metrics,
        required_selected,
        "selected threshold policies",
    )
    full = threshold_metrics.loc[
        pd.to_numeric(
            threshold_metrics["evaluation_fold"],
            errors="coerce",
        ).eq(-1)
        & threshold_metrics["metric"].eq("target_miss_rate"),
        [
            "model_id",
            "random_state",
            "decision_scope",
            "lower_quantile",
            "upper_quantile",
            "vote_threshold",
            "lower_threshold",
            "upper_threshold",
        ],
    ].copy()
    if full.empty:
        raise RuntimeError(
            "No full-OOF rows are available to materialize thresholds."
        )

    selected = selected_policy_metrics[
        [*_POLICY_COLUMNS, "decision_mode", "n_seeds"]
    ].drop_duplicates(list(_POLICY_COLUMNS))
    full = _fill_policy_values(full)
    selected = _fill_policy_values(selected)
    merge_columns = list(_POLICY_COLUMNS)

    chosen = full.merge(
        selected,
        on=merge_columns,
        how="inner",
        validate="many_to_one",
    )
    if chosen.duplicated(
        ["model_id", "random_state", "decision_scope"]
    ).any():
        raise RuntimeError(
            "More than one selected policy was materialized for a run."
        )

    observed = (
        chosen.groupby(
            merge_columns,
            as_index=False,
            dropna=False,
        )
        .agg(observed_n_seeds=("random_state", "nunique"))
    )
    coverage = selected.merge(
        observed,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )
    coverage["observed_n_seeds"] = coverage[
        "observed_n_seeds"
    ].fillna(0)
    if not coverage["observed_n_seeds"].eq(
        coverage["n_seeds"]
    ).all():
        raise RuntimeError(
            "Incomplete final-threshold coverage across random seeds."
        )

    lower = pd.to_numeric(
        chosen["lower_threshold"],
        errors="coerce",
    )
    upper = pd.to_numeric(
        chosen["upper_threshold"],
        errors="coerce",
    )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError("Selected numeric thresholds must be finite.")

    two_way = chosen["decision_mode"].eq("2way")
    three_way = chosen["decision_mode"].eq("3way")
    if not np.isclose(
        lower[two_way],
        upper[two_way],
    ).all():
        raise RuntimeError(
            "A 2-way policy has distinct lower and upper thresholds."
        )
    if not lower[three_way].lt(upper[three_way]).all():
        raise RuntimeError(
            "A 3-way policy has an invalid uncertainty interval."
        )

    vote_values = chosen.loc[
        chosen["vote_threshold"].ne(_POLICY_SENTINEL),
        "vote_threshold",
    ].to_numpy(dtype=float)
    allowed_votes = np.asarray(
        expcfg.INTERNAL_CALIBRATION_OBJECT_THRESHOLDS,
        dtype=float,
    )
    if vote_values.size:
        allowed = np.isclose(
            vote_values[:, None],
            allowed_votes[None, :],
        ).any(axis=1)
        if not allowed.all():
            raise RuntimeError(
                "A non-configured object vote threshold was selected."
            )

    chosen = _restore_policy_values(chosen)
    result = chosen.reindex(columns=output_columns)
    return result.sort_values(
        ["model_id", "random_state", "decision_scope"],
        kind="mergesort",
    ).reset_index(drop=True)


def materialize_selected_thresholds_from_parquet(
    threshold_metrics_path: str | Path,
    selected_policy_metrics: pd.DataFrame,
    *,
    random_states: Sequence[int] | None = None,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Materialize selected thresholds from streamed full-OOF rows.

    Existing 03B behaviour is unchanged when ``random_states`` is None.

    The optional random-state filter is used by notebook 05 to materialize
    thresholds only for the requested additional executions.
    """
    if selected_policy_metrics.empty:
        return pd.DataFrame(
            columns=(
                expcfg
                .INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
            )
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    states: tuple[int, ...] | None

    if random_states is None:
        states = None
    else:
        states = normalize_integer_sequence(
            random_states,
            name="random_states",
            allow_empty=False,
        )

    selected_keys = (
        _fill_policy_values(
            selected_policy_metrics[
                list(
                    _POLICY_COLUMNS
                )
            ]
        )
        .drop_duplicates(
            list(
                _POLICY_COLUMNS
            )
        )
    )

    selected_model_ids = set(
        selected_keys[
            "model_id"
        ].astype(str)
    )

    columns = [
        "model_id",
        "random_state",
        "evaluation_fold",
        "decision_scope",
        "lower_quantile",
        "upper_quantile",
        "vote_threshold",
        "lower_threshold",
        "upper_threshold",
        "metric",
    ]

    matched_parts: list[
        pd.DataFrame
    ] = []

    parquet = pq.ParquetFile(
        threshold_metrics_path
    )

    try:
        for batch in parquet.iter_batches(
            columns=columns,
            batch_size=int(
                batch_size
            ),
        ):
            frame = batch.to_pandas()

            evaluation_fold = pd.to_numeric(
                frame[
                    "evaluation_fold"
                ],
                errors="coerce",
            )

            keep = (
                evaluation_fold.eq(-1)
                &
                frame["metric"]
                .astype(str)
                .eq(
                    "target_miss_rate"
                )
                &
                frame["model_id"]
                .astype(str)
                .isin(
                    selected_model_ids
                )
            )
            if states is not None:
                observed_state = pd.to_numeric(
                    frame[
                        "random_state"
                    ],
                    errors="coerce",
                )

                keep &= observed_state.isin(
                    states
                )
            full = frame.loc[
                keep
            ].copy()
            if full.empty:
                continue
            full = _fill_policy_values(
                full
            )
            matched = full.merge(
                selected_keys,
                on=list(
                    _POLICY_COLUMNS
                ),
                how="inner",
                validate="many_to_one",
            )
            if not matched.empty:
                matched_parts.append(
                    _restore_policy_values(
                        matched
                    )
                )

    finally:
        parquet.close()
    if not matched_parts:
        raise RuntimeError(
            "No selected full-OOF threshold rows were found in Parquet."
        )
    full_rows = pd.concat(
        matched_parts,
        ignore_index=True,
        sort=False,
    )
    return materialize_selected_thresholds(
        full_rows,
        selected_policy_metrics,
    )

def _frozen_threshold_policy_coordinates(
    frozen_selected_thresholds: pd.DataFrame,
    *,
    expected_model_ids: Sequence[str] | None = None,
    expected_source_random_states: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Reduce 03B thresholds to one frozen policy per model and scope.

    Numeric lower/upper thresholds are intentionally ignored here because
    they are execution-specific. Only the scientific policy coordinates
    must be invariant across the original 03B seeds.
    """
    require_columns(
        frozen_selected_thresholds,
        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
        "frozen_selected_thresholds",
    )

    work = frozen_selected_thresholds[
        [
            "model_id",
            "random_state",
            "decision_scope",
            *_POLICY_VALUE_COLUMNS,
        ]
    ].copy()

    work["model_id"] = (
        work["model_id"]
        .astype(str)
    )

    work["decision_scope"] = (
        work["decision_scope"]
        .astype(str)
    )

    work["random_state"] = (
        pd.to_numeric(
            work["random_state"],
            errors="raise",
        )
        .astype(int)
    )

    # --------------------------------------------------------------
    # Restrict to the explicitly requested scientific models.
    # --------------------------------------------------------------
    if expected_model_ids is not None:
        raw_ids = pd.Series(
            list(expected_model_ids),
            dtype="object",
        )

        if raw_ids.isna().any():
            raise ValueError(
                "expected_model_ids cannot contain missing values."
            )

        model_ids = (
            raw_ids
            .astype(str)
            .tolist()
        )

        if len(model_ids) != len(set(model_ids)):
            raise ValueError(
                "expected_model_ids contains duplicates."
            )

        missing_models = sorted(
            set(model_ids)
            - set(work["model_id"])
        )

        if missing_models:
            raise KeyError(
                "Frozen thresholds do not contain requested model_id "
                f"values: {missing_models}."
            )

        work = work.loc[
            work["model_id"].isin(
                model_ids
            )
        ].copy()

    if work.empty:
        return pd.DataFrame(
            columns=_POLICY_COLUMNS
        )

    # --------------------------------------------------------------
    # Optional strict coverage check of the source/base seed panel.
    #
    # For notebook 05 this should be supplied for stochastic models:
    # expected_source_random_states =
    #     SIMCA_ROBUSTNESS_BASE_RANDOM_STATES
    #
    # The argument remains optional so deterministic one-seed models can
    # still use the helper if required elsewhere.
    # --------------------------------------------------------------
    if expected_source_random_states is not None:
        source_states = (
            normalize_integer_sequence(
                expected_source_random_states,
                name="expected_source_random_states",
                allow_empty=False,
            )
        )

        unexpected_states = sorted(
            set(work["random_state"])
            - set(source_states)
        )

        if unexpected_states:
            raise RuntimeError(
                "Frozen thresholds contain random states outside the "
                "expected source panel: "
                f"{unexpected_states}."
            )

        scope_keys = (
            work[
                [
                    "model_id",
                    "decision_scope",
                ]
            ]
            .drop_duplicates()
        )

        expected_keys = scope_keys.merge(
            pd.DataFrame(
                {
                    "random_state": source_states,
                }
            ),
            how="cross",
        )

        observed_keys = (
            work[
                [
                    "model_id",
                    "decision_scope",
                    "random_state",
                ]
            ]
            .drop_duplicates()
        )

        coverage = expected_keys.merge(
            observed_keys.assign(
                _present=True
            ),
            on=[
                "model_id",
                "decision_scope",
                "random_state",
            ],
            how="left",
            validate="one_to_one",
        )

        missing_source = coverage.loc[
            ~coverage["_present"]
            .fillna(False),
            [
                "model_id",
                "decision_scope",
                "random_state",
            ],
        ]

        if not missing_source.empty:
            raise RuntimeError(
                "Frozen threshold-policy coverage is incomplete across "
                "the expected source seeds: "
                f"{missing_source.to_dict('records')[:30]}."
            )

    # Use the existing policy canonicalization. This only corrects tiny
    # floating representation differences onto the prespecified grid; it
    # does not perform nearest-policy selection.
    normalized = _fill_policy_values(
        work
    )

    group_columns = [
        "model_id",
        "decision_scope",
    ]

    # --------------------------------------------------------------
    # Scientific lock:
    #
    # lower_quantile, upper_quantile and vote_threshold must not vary
    # between seeds of the same model/scope.
    # --------------------------------------------------------------
    policy_variation = (
        normalized
        .groupby(
            group_columns,
            sort=False,
            dropna=False,
        )[
            list(
                _POLICY_VALUE_COLUMNS
            )
        ]
        .nunique(
            dropna=False
        )
    )

    inconsistent = (
        policy_variation
        .gt(1)
        .any(axis=1)
    )

    if inconsistent.any():
        bad = [
            {
                "model_id": str(model_id),
                "decision_scope": str(decision_scope),
            }
            for model_id, decision_scope
            in policy_variation.index[
                inconsistent
            ]
        ]

        raise RuntimeError(
            "The 03B threshold policy is not invariant across seeds for: "
            f"{bad[:20]}. "
            "Numeric thresholds may vary by seed, but policy coordinates "
            "may not."
        )

    policies = (
        normalized[
            [
                "model_id",
                "decision_scope",
                *_POLICY_VALUE_COLUMNS,
            ]
        ]
        .drop_duplicates(
            group_columns
        )
    )

    if policies.duplicated(
        group_columns
    ).any():
        raise RuntimeError(
            "More than one frozen threshold policy exists for a "
            "model/scope."
        )

    # Every model must at least have its direct decision policy.
    model_ids = (
        policies["model_id"]
        .drop_duplicates()
    )

    direct_counts = (
        policies.loc[
            policies["decision_scope"].eq(
                "direct"
            )
        ]
        .groupby(
            "model_id",
            sort=False,
        )
        .size()
        .reindex(
            model_ids,
            fill_value=0,
        )
    )

    if not direct_counts.eq(1).all():
        bad_models = (
            direct_counts.loc[
                ~direct_counts.eq(1)
            ]
            .index
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Every frozen model must contain exactly one direct "
            f"threshold policy: {bad_models}."
        )

    unknown_scopes = sorted(
        set(
            policies[
                "decision_scope"
            ]
        )
        - {
            "direct",
            "pixel_to_object",
        }
    )

    if unknown_scopes:
        raise RuntimeError(
            "Unknown frozen decision scopes: "
            f"{unknown_scopes}."
        )

    return (
        _restore_policy_values(
            policies
        )
        .sort_values(
            [
                "model_id",
                "decision_scope",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _build_frozen_policy_metrics(
    frozen_selected_thresholds: pd.DataFrame,
    model_catalog: pd.DataFrame,
    *,
    expected_random_states: Sequence[int],
    expected_model_ids: Sequence[str] | None = None,
    expected_source_random_states: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Adapt the frozen 03B policy to the existing threshold materializer.

    This function does not select a new policy. It only reconstructs the
    minimal ``selected_policy_metrics`` contract expected by
    ``materialize_selected_thresholds``:

    - model_id
    - decision_scope
    - frozen policy coordinates
    - decision_mode
    - expected number of seeds
    """
    states = normalize_integer_sequence(
        expected_random_states,
        name="expected_random_states",
        allow_empty=False,
    )

    policies = (
        _frozen_threshold_policy_coordinates(
            frozen_selected_thresholds,
            expected_model_ids=(
                expected_model_ids
            ),
            expected_source_random_states=(
                expected_source_random_states
            ),
        )
    )

    output_columns = (
        *_POLICY_COLUMNS,
        "decision_mode",
        "n_seeds",
    )

    if policies.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    require_columns(
        model_catalog,
        (
            "model_id",
            "decision_mode",
            "projection_level",
        ),
        "model_catalog",
    )

    metadata = (
        model_catalog[
            [
                "model_id",
                "decision_mode",
                "projection_level",
            ]
        ]
        .copy()
    )

    metadata["model_id"] = (
        metadata["model_id"]
        .astype(str)
    )

    if metadata[
        "model_id"
    ].duplicated().any():
        raise RuntimeError(
            "model_catalog.model_id must be unique."
        )

    work = policies.merge(
        metadata,
        on="model_id",
        how="left",
        validate="many_to_one",
    )

    if work[
        [
            "decision_mode",
            "projection_level",
        ]
    ].isna().any().any():
        missing = sorted(
            set(
                policies[
                    "model_id"
                ].astype(str)
            )
            - set(
                metadata[
                    "model_id"
                ]
            )
        )

        raise RuntimeError(
            "Frozen threshold policies contain models absent from "
            f"model_catalog: {missing}."
        )

    # --------------------------------------------------------------
    # Exact decision-scope contract.
    # --------------------------------------------------------------
    model_info = (
        work[
            [
                "model_id",
                "projection_level",
            ]
        ]
        .drop_duplicates(
            "model_id"
        )
    )

    direct = (
        model_info[
            [
                "model_id",
            ]
        ]
        .copy()
    )
    direct[
        "decision_scope"
    ] = "direct"

    pixel = (
        model_info.loc[
            model_info[
                "projection_level"
            ]
            .astype(str)
            .eq(
                "pixel_projection"
            ),
            [
                "model_id",
            ],
        ]
        .copy()
    )
    pixel[
        "decision_scope"
    ] = "pixel_to_object"

    expected_scopes = pd.concat(
        [
            direct,
            pixel,
        ],
        ignore_index=True,
    )

    observed_scopes = (
        work[
            [
                "model_id",
                "decision_scope",
            ]
        ]
        .drop_duplicates()
    )

    scope_check = expected_scopes.merge(
        observed_scopes,
        on=[
            "model_id",
            "decision_scope",
        ],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not scope_check[
        "_merge"
    ].eq("both").all():
        bad = scope_check.loc[
            ~scope_check[
                "_merge"
            ].eq("both")
        ]

        raise RuntimeError(
            "Frozen threshold scopes do not match the model projection "
            f"contract: {bad.to_dict('records')[:30]}."
        )

    work["n_seeds"] = int(
        len(states)
    )

    return (
        work
        .reindex(
            columns=output_columns
        )
        .sort_values(
            [
                "model_id",
                "decision_scope",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _validate_materialized_threshold_coverage(
    thresholds: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    *,
    expected_random_states: Sequence[int],
) -> None:
    """Require exact model/scope/seed coverage."""
    states = normalize_integer_sequence(
        expected_random_states,
        name="expected_random_states",
        allow_empty=False,
    )

    natural_key = [
        "model_id",
        "random_state",
        "decision_scope",
    ]

    require_columns(
        thresholds,
        natural_key,
        "materialized thresholds",
    )

    if thresholds.duplicated(
        natural_key
    ).any():
        duplicates = (
            thresholds.loc[
                thresholds.duplicated(
                    natural_key,
                    keep=False,
                ),
                natural_key,
            ]
            .drop_duplicates()
        )

        raise RuntimeError(
            "Materialized threshold execution keys are duplicated: "
            f"{duplicates.to_dict('records')[:30]}."
        )

    scopes = (
        policy_metrics[
            [
                "model_id",
                "decision_scope",
            ]
        ]
        .drop_duplicates()
    )

    expected = scopes.merge(
        pd.DataFrame(
            {
                "random_state": states,
            }
        ),
        how="cross",
    )

    observed = (
        thresholds[
            natural_key
        ]
        .copy()
    )

    observed["model_id"] = (
        observed[
            "model_id"
        ]
        .astype(str)
    )

    observed["decision_scope"] = (
        observed[
            "decision_scope"
        ]
        .astype(str)
    )

    observed["random_state"] = (
        pd.to_numeric(
            observed[
                "random_state"
            ],
            errors="raise",
        )
        .astype(int)
    )

    expected["model_id"] = (
        expected[
            "model_id"
        ]
        .astype(str)
    )

    expected["decision_scope"] = (
        expected[
            "decision_scope"
        ]
        .astype(str)
    )

    expected["random_state"] = (
        pd.to_numeric(
            expected[
                "random_state"
            ],
            errors="raise",
        )
        .astype(int)
    )

    coverage = expected.merge(
        observed,
        on=natural_key,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not coverage[
        "_merge"
    ].eq("both").all():
        bad = coverage.loc[
            ~coverage[
                "_merge"
            ].eq("both")
        ]

        raise RuntimeError(
            "Materialized threshold coverage differs from the expected "
            f"model/scope/seed panel: {bad.to_dict('records')[:30]}."
        )


def materialize_fixed_threshold_policy_for_runs(
    threshold_metrics: pd.DataFrame,
    frozen_selected_thresholds: pd.DataFrame,
    model_catalog: pd.DataFrame,
    *,
    expected_random_states: Sequence[int],
    expected_model_ids: Sequence[str] | None = None,
    expected_source_random_states: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Materialize new-seed thresholds under the frozen 03B policy.

    No policy selection occurs.

    The existing 03B materialization function is reused, so threshold
    validation has one implementation only.
    """
    states = normalize_integer_sequence(
        expected_random_states,
        name="expected_random_states",
        allow_empty=False,
    )

    policy_metrics = (
        _build_frozen_policy_metrics(
            frozen_selected_thresholds,
            model_catalog,
            expected_random_states=states,
            expected_model_ids=(
                expected_model_ids
            ),
            expected_source_random_states=(
                expected_source_random_states
            ),
        )
    )

    if policy_metrics.empty:
        return pd.DataFrame(
            columns=(
                expcfg
                .INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
            )
        )

    require_columns(
        threshold_metrics,
        (
            "random_state",
        ),
        "threshold_metrics",
    )

    random_state = pd.to_numeric(
        threshold_metrics[
            "random_state"
        ],
        errors="coerce",
    )

    filtered_metrics = (
        threshold_metrics.loc[
            random_state.isin(
                states
            )
        ]
        .copy()
    )

    if filtered_metrics.empty:
        raise RuntimeError(
            "No threshold metrics exist for the requested robustness "
            "random states."
        )

    # Reuse the canonical 03B threshold materializer.
    result = materialize_selected_thresholds(
        filtered_metrics,
        policy_metrics,
    )

    _validate_materialized_threshold_coverage(
        result,
        policy_metrics,
        expected_random_states=states,
    )

    return result


def materialize_fixed_threshold_policy_for_runs_from_parquet(
    threshold_metrics_path: str | Path,
    frozen_selected_thresholds: pd.DataFrame,
    model_catalog: pd.DataFrame,
    *,
    expected_random_states: Sequence[int],
    expected_model_ids: Sequence[str] | None = None,
    expected_source_random_states: Sequence[int] | None = None,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Streaming materialization under the frozen 03B threshold policy."""
    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    source = Path(
        threshold_metrics_path
    )

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    states = normalize_integer_sequence(
        expected_random_states,
        name="expected_random_states",
        allow_empty=False,
    )

    policy_metrics = (
        _build_frozen_policy_metrics(
            frozen_selected_thresholds,
            model_catalog,
            expected_random_states=states,
            expected_model_ids=(
                expected_model_ids
            ),
            expected_source_random_states=(
                expected_source_random_states
            ),
        )
    )

    if policy_metrics.empty:
        return pd.DataFrame(
            columns=(
                expcfg
                .INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
            )
        )

    # Reuse the existing streamed 03B materializer.
    result = (
        materialize_selected_thresholds_from_parquet(
            source,
            policy_metrics,
            random_states=states,
            batch_size=batch_size,
        )
    )

    _validate_materialized_threshold_coverage(
        result,
        policy_metrics,
        expected_random_states=states,
    )

    return result


def reduce_threshold_policies_from_checkpoint_8tracks(
    checkpoint_run_dir: str | Path,
    configurations: pd.DataFrame,
    *,
    threshold_metrics_output_path: str | Path,
    threshold_audit_output_path: str | Path,
    threshold_candidates_output_path: str | Path | None = None,
    threshold_candidate_cache_context: Mapping[str, object] | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Reduce threshold shards by seed-family with bounded peak memory.

    Each scientific model belongs to exactly one runner family. All random
    seeds for that family are loaded together, selected, and released before
    the next family. The complete long metrics and threshold audit are written
    to Parquet streams rather than accumulated as pandas DataFrames.
    """
    required = {
        "model_id",
        "track_id",
        *INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS,
    }
    missing = sorted(required - set(configurations.columns))
    if missing:
        raise KeyError(
            f"Missing checkpoint-reduction configuration columns: {missing}"
        )
    if configurations.empty:
        raise ValueError("Configurations are empty.")

    work = attach_internal_calibration_runner_group_ids(
        configurations.reset_index(drop=True)
    )
    family_columns = tuple(
        column for column in INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS
        if column != "random_state"
    )
    model_family_identity = (
        work.groupby("model_id", dropna=False)[list(family_columns)]
        .nunique(dropna=False)
        .max(axis=1)
    )
    if model_family_identity.gt(1).any():
        raise RuntimeError("A model_id spans multiple checkpoint families.")

    shard_by_runner: dict[str, Path] = {}
    for runner_group_id, path in (
        iter_internal_calibration_checkpoint_shards_8tracks(
            checkpoint_run_dir,
            "threshold_metrics",
        )
    ):
        if runner_group_id in shard_by_runner:
            raise RuntimeError(
                f"Duplicate threshold shard for {runner_group_id}."
            )
        shard_by_runner[runner_group_id] = path

    expected_runner_ids = set(work["_runner_group_id"].astype(str))
    unexpected_runner_ids = set(shard_by_runner) - expected_runner_ids
    if unexpected_runner_ids:
        raise RuntimeError(
            "Threshold checkpoints do not match configurations: "
            f"unexpected={sorted(unexpected_runner_ids)}"
        )

    metrics_writer = _AtomicParquetStreamWriter(
        threshold_metrics_output_path
    )
    audit_writer = _AtomicParquetStreamWriter(
        threshold_audit_output_path,
        schema=_SELECTION_AUDIT_ARROW_SCHEMA,
    )
    candidate_writer: _AtomicParquetStreamWriter | None = None
    if threshold_candidates_output_path is not None:
        if threshold_candidate_cache_context is None:
            raise ValueError(
                "A threshold candidate cache context is required."
            )
        candidate_writer = _threshold_candidate_cache_writer(
            threshold_candidates_output_path,
            cache_context=threshold_candidate_cache_context,
        )
    candidate_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    selected_threshold_parts: list[pd.DataFrame] = []
    audit_summary_parts: list[pd.DataFrame] = []

    grouped = work.groupby(
        list(family_columns),
        sort=False,
        dropna=False,
    )
    try:
        for group_index, (_, group_configurations) in enumerate(
            grouped,
            start=1,
        ):
            runner_ids = (
                group_configurations.sort_values(
                    "random_state",
                    kind="mergesort",
                )["_runner_group_id"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            metric_parts: list[pd.DataFrame] = []
            for runner_group_id in runner_ids:
                path = shard_by_runner.get(runner_group_id)
                if path is None:
                    continue
                table = pq.read_table(path)
                metrics_writer.write_table(table)
                metric_parts.append(table.to_pandas())

            group_metrics = (
                pd.concat(
                    metric_parts,
                    ignore_index=True,
                    sort=False,
                )
                if metric_parts
                else pd.DataFrame(
                    columns=(
                        expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
                    )
                )
            )
            group_candidates = aggregate_threshold_candidates(group_metrics)
            selected, audit = select_threshold_policy_candidates(
                group_candidates,
                group_configurations,
            )
            selected_thresholds = (
                materialize_selected_thresholds(group_metrics, selected)
                if not selected.empty
                else pd.DataFrame(
                    columns=(
                        expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
                    )
                )
            )

            if not group_candidates.empty:
                candidate_parts.append(group_candidates)
                if candidate_writer is not None:
                    candidate_writer.write_table(
                        pa.Table.from_pandas(
                            group_candidates,
                            preserve_index=False,
                        )
                    )
            if not selected.empty:
                selected_parts.append(selected)
            if not selected_thresholds.empty:
                selected_threshold_parts.append(selected_thresholds)

            audit_writer.write_table(_selection_audit_to_arrow(audit))
            audit_summary_parts.append(
                summarize_selection_audit(
                    audit,
                    group_configurations[
                        ["model_id", "track_id"]
                    ].drop_duplicates(),
                )
            )

            if verbose:
                print(
                    f"[03B thresholds {group_index}/{grouped.ngroups}] "
                    f"runners={len(runner_ids)} "
                    f"models={group_configurations['model_id'].nunique()}"
                )

            del metric_parts
            del group_metrics
            del group_candidates
            del selected
            del selected_thresholds
            del audit
            gc.collect()

        threshold_metrics_path = metrics_writer.finish()
        threshold_audit_path = audit_writer.finish()
        threshold_candidates_path = (
            candidate_writer.finish()
            if candidate_writer is not None
            else None
        )
    except Exception:
        metrics_writer.abort()
        audit_writer.abort()
        if candidate_writer is not None:
            candidate_writer.abort()
        raise

    threshold_candidates = (
        pd.concat(candidate_parts, ignore_index=True, sort=False)
        if candidate_parts
        else pd.DataFrame()
    )
    selected_policy_metrics = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )
    selected_thresholds = (
        pd.concat(
            selected_threshold_parts,
            ignore_index=True,
            sort=False,
        )
        if selected_threshold_parts
        else pd.DataFrame(
            columns=expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS
        )
    )
    audit_summary = (
        pd.concat(audit_summary_parts, ignore_index=True, sort=False)
        .groupby(
            [
                "selection_level",
                "track_id",
                "stage",
                "decision",
                "reason_code",
            ],
            as_index=False,
            sort=False,
            dropna=False,
        )[["n_rows", "n_models"]]
        .sum()
        if audit_summary_parts
        else pd.DataFrame()
    )

    return {
        "threshold_candidates": threshold_candidates,
        "selected_policy_metrics": selected_policy_metrics,
        "selected_thresholds": selected_thresholds,
        "threshold_audit_summary": audit_summary,
        "threshold_metrics_path": threshold_metrics_path,
        "threshold_audit_path": threshold_audit_path,
        "threshold_candidates_path": threshold_candidates_path,
    }


def select_threshold_policies_from_candidate_cache_8tracks(
    threshold_candidates_path: str | Path,
    configurations: pd.DataFrame,
    *,
    threshold_metrics_path: str | Path,
    threshold_audit_output_path: str | Path,
    threshold_candidate_cache_context: Mapping[str, object],
    verbose: bool = True,
) -> dict[str, object]:
    """Re-select policies without recomputing fits or candidate aggregates."""
    required = {
        "model_id",
        "track_id",
        *INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS,
    }
    missing = sorted(required - set(configurations.columns))
    if missing:
        raise KeyError(
            f"Missing cached-selection configuration columns: {missing}"
        )
    if configurations.empty:
        raise ValueError("Configurations are empty.")

    candidates = load_threshold_candidate_cache(
        threshold_candidates_path,
        expected_context=threshold_candidate_cache_context,
    )
    work = configurations.reset_index(drop=True).copy()
    family_columns = tuple(
        column
        for column in INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS
        if column != "random_state"
    )
    model_families = work[
        ["model_id", *family_columns]
    ].drop_duplicates()
    if model_families["model_id"].duplicated().any():
        raise RuntimeError("A model_id spans multiple cache families.")
    model_families["_cache_group"] = model_families.groupby(
        list(family_columns),
        sort=False,
        dropna=False,
    ).ngroup()
    group_by_model = model_families.set_index("model_id")[
        "_cache_group"
    ]
    candidate_groups = candidates["model_id"].map(group_by_model)
    if candidate_groups.isna().any():
        unexpected = sorted(
            candidates.loc[
                candidate_groups.isna(),
                "model_id",
            ].astype(str).unique()
        )
        raise RuntimeError(
            "Cached candidates do not match configurations: "
            f"unexpected_models={unexpected}"
        )

    candidate_work = candidates.assign(
        _cache_group=candidate_groups.astype(int).to_numpy()
    )
    work["_cache_group"] = work["model_id"].map(group_by_model).astype(int)
    audit_writer = _AtomicParquetStreamWriter(
        threshold_audit_output_path,
        schema=_SELECTION_AUDIT_ARROW_SCHEMA,
    )
    selected_parts: list[pd.DataFrame] = []
    audit_summary_parts: list[pd.DataFrame] = []
    grouped = candidate_work.groupby(
        "_cache_group",
        sort=False,
        dropna=False,
    )
    try:
        for group_index, (group_id, group_candidates) in enumerate(
            grouped,
            start=1,
        ):
            group_candidates = group_candidates.drop(
                columns="_cache_group"
            )
            group_configurations = work.loc[
                work["_cache_group"].eq(int(group_id))
            ].drop(columns="_cache_group")
            selected, audit = select_threshold_policy_candidates(
                group_candidates,
                group_configurations,
            )
            if not selected.empty:
                selected_parts.append(selected)
            audit_writer.write_table(_selection_audit_to_arrow(audit))
            audit_summary_parts.append(
                summarize_selection_audit(
                    audit,
                    group_configurations[
                        ["model_id", "track_id"]
                    ].drop_duplicates(),
                )
            )
            if verbose:
                print(
                    f"[03B cached thresholds {group_index}/"
                    f"{grouped.ngroups}] "
                    f"models={group_configurations['model_id'].nunique()}"
                )
            del group_candidates
            del group_configurations
            del selected
            del audit
            gc.collect()
        threshold_audit_path = audit_writer.finish()
    except Exception:
        audit_writer.abort()
        raise

    selected_policy_metrics = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )
    selected_thresholds = materialize_selected_thresholds_from_parquet(
        threshold_metrics_path,
        selected_policy_metrics,
    )
    audit_summary = (
        pd.concat(audit_summary_parts, ignore_index=True, sort=False)
        .groupby(
            [
                "selection_level",
                "track_id",
                "stage",
                "decision",
                "reason_code",
            ],
            as_index=False,
            sort=False,
            dropna=False,
        )[["n_rows", "n_models"]]
        .sum()
        if audit_summary_parts
        else pd.DataFrame()
    )
    return {
        "threshold_candidates": candidates,
        "selected_policy_metrics": selected_policy_metrics,
        "selected_thresholds": selected_thresholds,
        "threshold_audit_summary": audit_summary,
        "threshold_metrics_path": Path(threshold_metrics_path),
        "threshold_audit_path": threshold_audit_path,
        "threshold_candidates_path": Path(threshold_candidates_path),
    }


def finalize_streamed_selection_audit(
    threshold_audit_path: str | Path,
    model_selection_audit: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Combine streamed threshold audit with the compact model audit."""
    source = Path(threshold_audit_path)
    destination = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        raise ValueError("Threshold and final audit paths must differ.")

    writer = _AtomicParquetStreamWriter(
        destination,
        schema=_SELECTION_AUDIT_ARROW_SCHEMA,
    )
    try:
        parquet = pq.ParquetFile(source)
        try:
            for batch in parquet.iter_batches(batch_size=100_000):
                writer.write_table(pa.Table.from_batches([batch]))
        finally:
            parquet.close()
        if not model_selection_audit.empty:
            writer.write_table(
                _selection_audit_to_arrow(model_selection_audit)
            )
        result = writer.finish()
    except Exception:
        writer.abort()
        raise

    source.unlink()
    return result


def sample_threshold_candidates_for_plot(
    threshold_candidates: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    model_catalog: pd.DataFrame,
    *,
    max_rows_per_track_scope: int,
) -> pd.DataFrame:
    """Return a deterministic visual sample while always keeping selections."""
    if max_rows_per_track_scope < 1:
        raise ValueError("max_rows_per_track_scope must be positive.")
    if threshold_candidates.empty:
        return threshold_candidates.copy()
    require_columns(
        threshold_candidates,
        (*_POLICY_COLUMNS, "target_miss_rate", "false_accept_rate"),
        "threshold candidates",
    )
    require_columns(
        model_catalog,
        ("model_id", "track_id"),
        "model catalog",
    )

    original_columns = list(threshold_candidates.columns)
    links = model_catalog[["model_id", "track_id"]].drop_duplicates()
    if links["model_id"].duplicated().any():
        raise RuntimeError("A model_id maps to multiple tracks.")
    work = threshold_candidates.merge(
        links,
        on="model_id",
        how="left",
        validate="many_to_one",
    )
    order_columns = [
        "model_id",
        "lower_quantile",
        "upper_quantile",
        "vote_threshold",
    ]
    sampled_parts: list[pd.DataFrame] = []
    for _, group in work.groupby(
        ["track_id", "decision_scope"],
        sort=True,
        dropna=False,
    ):
        ordered = group.sort_values(
            order_columns,
            kind="mergesort",
            na_position="last",
        )
        if len(ordered) > max_rows_per_track_scope:
            positions = np.linspace(
                0,
                len(ordered) - 1,
                num=max_rows_per_track_scope,
                dtype=int,
            )
            ordered = ordered.iloc[np.unique(positions)]
        sampled_parts.append(ordered[original_columns])

    selected_rows = selected_policy_metrics.reindex(
        columns=original_columns
    )
    combined = pd.concat(
        [*sampled_parts, selected_rows],
        ignore_index=True,
        sort=False,
    )
    normalized = _fill_policy_values(combined)
    normalized = normalized.drop_duplicates(
        list(_POLICY_COLUMNS),
        keep="last",
    )
    return _restore_policy_values(normalized).reset_index(drop=True)


def build_model_metrics(
    selected_policy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact long table of model-level metrics."""
    output_columns = expcfg.INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS
    if selected_policy_metrics.empty:
        return pd.DataFrame(columns=output_columns)

    present = [
        metric for metric in _MODEL_METRICS
        if metric in selected_policy_metrics
    ]
    metrics = selected_policy_metrics[
        ["model_id", "decision_scope", *present]
    ].melt(
        id_vars=["model_id", "decision_scope"],
        value_vars=present,
        var_name="_metric",
        value_name="value",
    )
    metrics["metric"] = (
        metrics["decision_scope"].astype(str)
        + "."
        + metrics["_metric"].astype(str)
    )
    metrics = metrics[["model_id", "metric", "value"]]

    parts = [metrics]
    for source_metric, safety_metric in (
        (
            "target_miss_rate",
            "safety.target_miss_rate",
        ),
        (
            "false_accept_rate",
            "safety.false_accept_rate",
        ),
    ):
        scope_metric = f".{source_metric}"
        source = metrics.loc[
            metrics["metric"].str.endswith(scope_metric)
        ]
        safety = (
            source.groupby("model_id", as_index=False)
            .agg(value=("value", "max"))
        )
        safety["metric"] = safety_metric
        parts.append(safety)

    result = pd.concat(parts, ignore_index=True).reindex(
        columns=output_columns
    )
    if result.duplicated(["model_id", "metric"]).any():
        raise RuntimeError("Duplicate model-level metric rows.")
    return result.sort_values(
        ["model_id", "metric"],
        kind="mergesort",
    ).reset_index(drop=True)


def _model_audit_rows(
    frame: pd.DataFrame,
    keep: pd.Series,
    *,
    stage: str,
    reason_code: str,
    metric: str = "",
    observed: str | pd.Series | None = None,
    operator: str = "",
    reference: float | pd.Series | None = None,
    related_model_id: pd.Series | None = None,
) -> pd.DataFrame:
    keep = keep.reindex(frame.index).fillna(False).astype(bool)
    out = pd.DataFrame(
        {
            "selection_level": "model",
            "model_id": frame["model_id"].astype(str),
            "decision_scope": "",
            "lower_quantile": np.nan,
            "upper_quantile": np.nan,
            "vote_threshold": np.nan,
            "stage": stage,
            "decision": np.where(keep, "kept", "eliminated"),
            "reason_code": np.where(
                keep,
                "passed",
                reason_code,
            ),
            "metric": metric,
            "operator": operator,
        },
        index=frame.index,
    )
    if isinstance(observed, str) and observed in frame:
        out["observed_value"] = pd.to_numeric(
            frame[observed],
            errors="coerce",
        )
    elif isinstance(observed, pd.Series):
        out["observed_value"] = pd.to_numeric(
            observed.reindex(frame.index),
            errors="coerce",
        )
    else:
        out["observed_value"] = np.nan
    out["reference_value"] = _value_series(
        reference,
        frame.index,
    ).to_numpy()
    out["related_model_id"] = (
        ""
        if related_model_id is None
        else related_model_id.reindex(frame.index).fillna("").astype(str)
    )
    return out.reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS
    )


def _metric_matrix(
    model_metrics: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        model_metrics,
        expcfg.INTERNAL_CALIBRATION_MODEL_METRIC_COLUMNS,
        "model metrics",
    )
    if model_metrics.duplicated(["model_id", "metric"]).any():
        raise RuntimeError("Duplicate model metric rows.")
    wide = model_metrics.pivot(
        index="model_id",
        columns="metric",
        values="value",
    ).reset_index()
    wide.columns.name = None
    return wide


def _group_reference(
    frame: pd.DataFrame,
    metric: str,
    group_columns: Sequence[str],
    direction: str,
    tolerance: float,
) -> tuple[pd.Series, str]:
    grouped = frame.groupby(
        list(group_columns),
        dropna=False,
    )[metric]
    if direction == "min":
        return grouped.transform("min") + tolerance, "<="
    if direction == "max":
        return grouped.transform("max") - tolerance, ">="
    raise ValueError(f"Unsupported priority direction: {direction!r}")


def select_calibrated_models(
    *,
    model_catalog: pd.DataFrame,
    candidate_runs: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    rule_diagnostics: pd.DataFrame,
    expected_n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select calibrated models using constraints, plateaus and Pareto."""
    require_columns(
        model_catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        "model catalog",
    )
    require_columns(
        candidate_runs,
        expcfg.INTERNAL_CALIBRATION_CANDIDATE_RUN_COLUMNS,
        "candidate runs",
    )
    require_columns(
        rule_diagnostics,
        expcfg.INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS,
        "rule diagnostics",
    )
    if model_catalog["model_id"].duplicated().any():
        raise RuntimeError("model_catalog contains duplicate model_id rows.")
    if candidate_runs.duplicated(
        ["model_id", "random_state"]
    ).any():
        raise RuntimeError(
            "candidate_runs contains duplicate model executions."
        )
    if expected_n_folds < 1:
        raise ValueError("expected_n_folds must be positive.")

    current = model_catalog.copy()
    audit_parts = [
        _model_audit_rows(
            current,
            pd.Series(True, index=current.index),
            stage="initial_candidate",
            reason_code="",
        )
    ]

    run_links = candidate_runs[
        ["model_id", "projection_id"]
    ].drop_duplicates()
    diagnostic_links = rule_diagnostics.merge(
        run_links,
        on="projection_id",
        how="inner",
        validate="many_to_many",
    )
    bad_counts = (
        diagnostic_links.loc[
            ~diagnostic_links["status"].astype(str).eq("ok")
        ]
        .groupby("model_id")
        .size()
    )
    current["_technical_error_count"] = (
        current["model_id"].map(bad_counts).fillna(0).astype(int)
    )
    passed = current["_technical_error_count"].eq(0)
    audit_parts.append(
        _model_audit_rows(
            current,
            passed,
            stage="technical_status",
            reason_code="technical_projection_error",
            metric="technical_error_count",
            observed="_technical_error_count",
            operator="==",
            reference=0.0,
        )
    )
    current = current.loc[passed].copy()

    expected_projection_folds = (
        run_links.groupby("model_id")
        .size()
        .mul(expected_n_folds)
    )
    successful_pairs = (
        diagnostic_links.loc[
            diagnostic_links["status"].astype(str).eq("ok"),
            ["model_id", "projection_id", "fold_id"],
        ]
        .drop_duplicates()
        .groupby("model_id")
        .size()
    )
    current["_observed_projection_folds"] = (
        current["model_id"].map(successful_pairs).fillna(0).astype(int)
    )
    current["_expected_projection_folds"] = (
        current["model_id"]
        .map(expected_projection_folds)
        .fillna(0)
        .astype(int)
    )
    passed = current["_observed_projection_folds"].eq(
        current["_expected_projection_folds"]
    )
    audit_parts.append(
        _model_audit_rows(
            current,
            passed,
            stage="projection_fold_coverage",
            reason_code="incomplete_projection_fold_coverage",
            metric="projection_fold_count",
            observed="_observed_projection_folds",
            operator="==",
            reference=current["_expected_projection_folds"],
        )
    )
    current = current.loc[passed].copy()

    observed_scopes = (
        selected_policy_metrics.groupby("model_id")["decision_scope"]
        .agg(lambda values: frozenset(map(str, values)))
    )
    expected_scopes = current["projection_level"].map(
        {
            "object_projection": frozenset(("direct",)),
            "pixel_projection": frozenset(
                ("direct", "pixel_to_object")
            ),
        }
    )
    observed_scope_sets = current["model_id"].map(
        observed_scopes
    ).apply(
        lambda value: (
            value if isinstance(value, frozenset) else frozenset()
        )
    )
    passed = observed_scope_sets.eq(expected_scopes)
    audit_parts.append(
        _model_audit_rows(
            current,
            passed,
            stage="required_decision_scopes",
            reason_code="missing_required_decision_scope",
            metric="decision_scope_count",
            observed=observed_scope_sets.map(len),
            operator="==",
            reference=expected_scopes.map(len),
        )
    )
    current = current.loc[passed].copy()

    run_counts = (
        candidate_runs.groupby("model_id", as_index=False)
        .agg(expected_n_seeds=("random_state", "nunique"))
    )
    policy_coverage = (
        selected_policy_metrics.groupby("model_id", as_index=False)
        .agg(
            observed_n_seeds=("n_seeds", "min"),
            observed_n_folds=("n_folds", "min"),
            observed_n_run_folds=("n_run_folds", "min"),
        )
        .merge(
            run_counts,
            on="model_id",
            how="outer",
            validate="one_to_one",
        )
    )
    current = current.merge(
        policy_coverage,
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    current["_expected_run_folds"] = (
        current["expected_n_seeds"].mul(expected_n_folds)
    )

    coverage_rules = (
        (
            "observed_n_seeds",
            current["expected_n_seeds"],
            "incomplete_selected_seed_coverage",
        ),
        (
            "observed_n_folds",
            pd.Series(expected_n_folds, index=current.index),
            "incomplete_selected_fold_coverage",
        ),
        (
            "observed_n_run_folds",
            current["_expected_run_folds"],
            "incomplete_selected_run_fold_coverage",
        ),
    )
    for metric, reference, reason in coverage_rules:
        reference = reference.reindex(current.index)
        passed = compare(current[metric], "==", reference)
        audit_parts.append(
            _model_audit_rows(
                current,
                passed,
                stage=f"policy_coverage:{metric}",
                reason_code=reason,
                metric=metric,
                observed=metric,
                operator="==",
                reference=reference,
            )
        )
        current = current.loc[passed].copy()

    metrics = _metric_matrix(model_metrics)
    current = current.merge(
        metrics,
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    safety_metrics = (
        "safety.target_miss_rate",
        "safety.false_accept_rate",
    )
    missing_safety = sorted(set(safety_metrics) - set(current.columns))
    if missing_safety:
        raise KeyError(f"Missing safety metrics: {missing_safety}")

    for metric in safety_metrics:
        passed = pd.Series(
            np.isfinite(
                pd.to_numeric(
                    current[metric],
                    errors="coerce",
                )
            ),
            index=current.index,
        )
        audit_parts.append(
            _model_audit_rows(
                current,
                passed,
                stage=f"finite_metric:{metric}",
                reason_code="non_finite_safety_metric",
                metric=metric,
                observed=metric,
            )
        )
        current = current.loc[passed].copy()

    for rule in expcfg.INTERNAL_CALIBRATION_MODEL_PRIORITY:
        if current.empty:
            break
        metric = str(rule["metric"])
        reference, operator = _group_reference(
            current,
            metric,
            ("track_id",),
            str(rule["direction"]),
            float(rule["tolerance"]),
        )
        passed = compare(current[metric], operator, reference)
        audit_parts.append(
            _model_audit_rows(
                current,
                passed,
                stage=f"model_priority:{metric}",
                reason_code=str(rule["reason"]),
                metric=metric,
                observed=metric,
                operator=operator,
                reference=reference,
            )
        )
        current = current.loc[passed].copy()

    parameter_order = tuple(
        expcfg.INTERNAL_CALIBRATION_COMPLEXITY_SELECTION[
            "parameter_order"
        ]
    )
    parameter_columns = list(expcfg.SIMCA_MODEL_PARAMETER_COLUMNS)

    for parameter in parameter_order:
        if current.empty or parameter not in current:
            continue
        numeric_parameter = pd.to_numeric(
            current[parameter],
            errors="coerce",
        )
        applicable_index = current.index[numeric_parameter.notna()]
        if not len(applicable_index):
            continue

        group_columns = [
            column for column in parameter_columns
            if column != parameter and column in current
        ]
        for rule in expcfg.INTERNAL_CALIBRATION_MODEL_PRIORITY:
            subset = current.loc[
                current.index.intersection(applicable_index)
            ].copy()
            if subset.empty:
                break
            metric = str(rule["metric"])
            reference, operator = _group_reference(
                subset,
                metric,
                group_columns,
                str(rule["direction"]),
                float(rule["tolerance"]),
            )
            passed = compare(subset[metric], operator, reference)
            audit_parts.append(
                _model_audit_rows(
                    subset,
                    passed,
                    stage=(
                        f"complexity:{parameter}:{metric}"
                    ),
                    reason_code=(
                        f"outside_{parameter}_risk_plateau"
                    ),
                    metric=metric,
                    observed=metric,
                    operator=operator,
                    reference=reference,
                )
            )
            current = current.drop(index=subset.index[~passed])
            applicable_index = applicable_index.intersection(
                current.index
            )

        subset = current.loc[
            current.index.intersection(applicable_index)
        ].copy()
        if subset.empty:
            continue
        numeric = pd.to_numeric(
            subset[parameter],
            errors="coerce",
        )
        reference = numeric.groupby(
            [
                subset[column]
                for column in group_columns
            ],
            dropna=False,
        ).transform("min")
        passed = numeric.eq(reference)
        audit_parts.append(
            _model_audit_rows(
                subset,
                passed,
                stage=f"complexity:{parameter}:smallest",
                reason_code=f"larger_{parameter}",
                metric=parameter,
                observed=numeric,
                operator="==",
                reference=reference,
            )
        )
        current = current.drop(index=subset.index[~passed])

    pareto_parts = []
    for track_id, subset in current.groupby(
        "track_id",
        sort=False,
        dropna=False,
    ):
        track_id = str(track_id)
        if track_id not in (
            expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES
        ):
            raise KeyError(
                f"No Pareto objectives configured for {track_id}."
            )
        objectives = (
            expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES[track_id]
        )
        minimize = tuple(objectives["minimize"])
        maximize = tuple(objectives["maximize"])
        objective_columns = [*minimize, *maximize]
        missing = sorted(
            set(objective_columns) - set(subset.columns)
        )
        if missing:
            raise KeyError(
                f"Missing Pareto metrics for {track_id}: {missing}"
            )

        finite = pd.Series(
            np.isfinite(
                subset[objective_columns].to_numpy(dtype=float)
            ).all(axis=1),
            index=subset.index,
        )
        audit_parts.append(
            _model_audit_rows(
                subset,
                finite,
                stage="pareto_metric_completeness",
                reason_code="non_finite_pareto_metric",
            )
        )
        eligible = subset.loc[finite].copy()
        if eligible.empty:
            continue

        front, witness = pareto_front_with_witness(
            eligible,
            minimize_cols=minimize,
            maximize_cols=maximize,
        )
        audit_parts.append(
            _model_audit_rows(
                eligible,
                front,
                stage="pareto_front",
                reason_code="pareto_dominated",
                related_model_id=witness,
            )
        )
        pareto_parts.append(eligible.loc[front])

    current = (
        pd.concat(pareto_parts, ignore_index=True, sort=False)
        if pareto_parts
        else current.iloc[0:0].copy()
    )
    selected_models = current[["model_id"]].drop_duplicates()
    selected_models["selection_status"] = "selected"
    selected_models = selected_models.reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS
    )
    selected_runs = candidate_runs.loc[
        candidate_runs["model_id"].isin(
            selected_models["model_id"]
        )
    ].reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS
    )
    audit = pd.concat(
        audit_parts,
        ignore_index=True,
        sort=False,
    ).reindex(
        columns=expcfg.INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS
    )
    return (
        selected_models.reset_index(drop=True),
        selected_runs.reset_index(drop=True),
        audit.reset_index(drop=True),
    )
