"""Freeze and validate the scientific protocol defined by tasks 01-02.

The module deliberately separates:
- four parent tracks, used to share model fits;
- eight evaluation tracks, used for projection-aware decisions and Pareto;
- the versioned configuration manifest;
- the prespecified H1-H4 inference plan and contrast table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src import experiment_config as expcfg
from src.utils import save_parquet


PROTOCOL_MANIFEST_COLUMNS = (
    "protocol_version",
    "schema_version",
    "protocol_status",
    "protocol_section",
    "parameter",
    "value_json",
    "parameter_sha256",
    "configuration_sha256",
)

PROTOCOL_CHECK_COLUMNS = ("check", "passed", "detail")

PLANNED_CONTRAST_COLUMNS = (
    "protocol_version",
    "contrast_id",
    "hypothesis_id",
    "scope",
    "contrast_type",
    "left_track",
    "left_track_id",
    "right_track",
    "right_track_id",
    "component_tracks_json",
    "contrast_expression",
    "metric",
    "metric_role",
    "direction",
    "primary_unit",
    "primary_analysis_stage",
    "supporting_analysis_stage",
    "estimator",
    "bootstrap_group_col",
    "confidence_level",
    "multiplicity_family",
    "multiplicity_method",
    "practical_tolerance_type",
    "practical_tolerance",
    "decision_rule",
    "status",
)


class ProtocolValidationError(RuntimeError):
    """Raised when the frozen scientific protocol is internally inconsistent."""


def validate_selection_only_protocol_lineage(
    *,
    current_protocol_hash: str,
    execution_protocol_hash: str,
    expected_parent_protocol_hash: str,
    amendment_scope: str,
    selection_profile_id: str,
    selection_parent_profile_id: str,
    checkpoint_manifest: Mapping[str, Any],
    expected_execution_context: Mapping[str, str],
    strict: bool = True,
) -> pd.DataFrame:
    """Validate reuse of immutable fits under a selection-only amendment.

    The current frozen protocol owns the amended selection, while PCA and the
    completed OOF fits retain the hash under which they were executed. Reuse is
    accepted only when every execution-defining fingerprint still matches the
    parent checkpoint exactly.
    """
    current_hash = str(current_protocol_hash)
    execution_hash = str(execution_protocol_hash)
    parent_hash = str(expected_parent_protocol_hash)
    scope = str(amendment_scope)
    profile_id = str(selection_profile_id)
    parent_profile_id = str(selection_parent_profile_id)
    context = {
        str(key): str(value)
        for key, value in dict(expected_execution_context).items()
    }
    manifest = dict(checkpoint_manifest)
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": str(check),
                "passed": bool(passed),
                "detail": str(detail),
            }
        )

    add(
        "protocol_amendment_scope_is_selection_only",
        scope == "selection_only",
        f"scope={scope}",
    )
    add(
        "selection_amendment_has_distinct_current_protocol",
        bool(current_hash) and current_hash != execution_hash,
        f"current={current_hash}, execution_parent={execution_hash}",
    )
    add(
        "execution_protocol_matches_declared_parent",
        bool(parent_hash) and execution_hash == parent_hash,
        f"expected={parent_hash}, observed={execution_hash}",
    )
    add(
        "selection_profile_has_declared_parent",
        bool(profile_id)
        and bool(parent_profile_id)
        and profile_id != parent_profile_id,
        f"profile={profile_id}, parent_profile={parent_profile_id}",
    )
    add(
        "checkpoint_protocol_matches_execution_parent",
        str(manifest.get("protocol_hash", "")) == execution_hash,
        (
            f"expected={execution_hash}, "
            f"observed={manifest.get('protocol_hash')}"
        ),
    )
    add(
        "checkpoint_protocol_version_matches_current",
        str(manifest.get("protocol_version", ""))
        == str(expcfg.PROTOCOL_VERSION),
        (
            f"expected={expcfg.PROTOCOL_VERSION}, "
            f"observed={manifest.get('protocol_version')}"
        ),
    )
    add(
        "checkpoint_schema_version_matches_current",
        str(manifest.get("schema_version", ""))
        == str(expcfg.RESULTS_SCHEMA_VERSION),
        (
            f"expected={expcfg.RESULTS_SCHEMA_VERSION}, "
            f"observed={manifest.get('schema_version')}"
        ),
    )

    required_context = (
        "protocol_hash",
        "pca_selection_fingerprint",
        "track_contract_hash",
        "fold_contract_hash",
        "configuration_hash",
    )
    for key in required_context:
        expected = context.get(key, "")
        observed = str(manifest.get(key, ""))
        add(
            f"checkpoint_{key}_matches_current_execution_context",
            bool(expected) and observed == expected,
            f"expected={expected}, observed={observed}",
        )

    result = pd.DataFrame(checks, columns=PROTOCOL_CHECK_COLUMNS)
    if strict and not bool(result["passed"].all()):
        failed = result.loc[~result["passed"]].to_dict(orient="records")
        raise ProtocolValidationError(
            "Selection-only protocol lineage validation failed: "
            f"{failed}"
        )
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_jsonable(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Protocol values must be JSON serializable; "
        f"received {type(value).__name__}: {value!r}"
    )


def canonical_json(value: Any) -> str:
    """Return the canonical UTF-8 JSON representation used by every hash."""
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_payload(value: Any) -> str:
    """Hash a JSON-compatible scientific payload."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_ndarray(value: Any) -> str:
    """Hash an array from its dtype, shape and canonical C-order bytes."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {"dtype": array.dtype.str, "shape": tuple(array.shape)}
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_dataframe(value: pd.DataFrame) -> str:
    """Hash a table including its schema, row order and null pattern."""
    frame = value.reset_index(drop=True).copy()
    header = canonical_json(
        {
            "columns": list(map(str, frame.columns)),
            "dtypes": [str(dtype) for dtype in frame.dtypes],
        }
    ).encode("utf-8")
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(
        dtype="<u8", copy=False
    )
    return hashlib.sha256(header + row_hashes.tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it entirely in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_simca_track_contracts() -> pd.DataFrame:
    """Materialise the projection-aware contract of evaluation tracks E1-E8."""
    rows: list[dict[str, Any]] = []
    for evaluation_track in expcfg.SIMCA_EVALUATION_TRACKS:
        spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[evaluation_track]
        rows.append(
            {
                "track_id": str(spec["track_id"]),
                "evaluation_track": str(evaluation_track),
                "parent_track": str(spec["parent_track"]),
                "training_matrix_family": str(
                    spec["training_matrix_family"]
                ),
                "projection_level": str(spec["projection_level"]),
                "projection_matrix_policy": str(
                    spec["projection_matrix_policy"]
                ),
                "allowed_projection_methods_json": canonical_json(
                    tuple(spec["allowed_projection_methods"])
                ),
                "primary_unit": str(spec["primary_unit"]),
                "decision_mode": str(spec["decision_mode"]),
                "decision_score_type": str(spec["decision_score_type"]),
                "higher_is_target": bool(spec["higher_is_target"]),
                "direct_2way_threshold": float(
                    spec["direct_2way_threshold"]
                ),
                "constraint_profile_id": str(
                    spec["constraint_profile_id"]
                ),
                "calibration_primary_metrics_json": canonical_json(
                    tuple(spec["calibration_primary_metrics"])
                ),
                "final_evaluation_metrics_json": canonical_json(
                    tuple(spec["final_evaluation_metrics"])
                ),
                "pareto_minimize_json": canonical_json(
                    tuple(spec["pareto_minimize"])
                ),
                "pareto_maximize_json": canonical_json(
                    tuple(spec["pareto_maximize"])
                ),
                "protocol_version": str(expcfg.PROTOCOL_VERSION),
                "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
            }
        )
    contracts = pd.DataFrame(rows).loc[
        :, list(expcfg.INTERNAL_CALIBRATION_TRACK_CONTRACT_COLUMNS)
    ]
    expected_tracks = set(expcfg.SIMCA_EVALUATION_TRACKS)
    observed_tracks = set(contracts["evaluation_track"])
    if observed_tracks != expected_tracks or len(contracts) != 8:
        raise ProtocolValidationError(
            "The SIMCA track contract must contain exactly E1-E8: "
            f"missing={sorted(expected_tracks - observed_tracks)}, "
            f"extra={sorted(observed_tracks - expected_tracks)}."
        )
    if not contracts["evaluation_track"].is_unique:
        raise ProtocolValidationError("Evaluation tracks must be unique.")
    if set(contracts["track_id"]) != {
        f"E{index}" for index in range(1, 9)
    }:
        raise ProtocolValidationError("Track IDs must be exactly E1-E8.")
    if not contracts["direct_2way_threshold"].eq(0.0).all():
        raise ProtocolValidationError(
            "The direct SIMCA threshold must be the zero margin."
        )
    return contracts.sort_values("track_id", kind="mergesort").reset_index(
        drop=True
    )


def _track_name(track_id: str) -> str:
    reverse = {
        identifier: track
        for track, identifier in expcfg.SIMCA_EVALUATION_TRACK_IDS.items()
    }
    return reverse[track_id]


def _track_id(track: str) -> str:
    return expcfg.SIMCA_EVALUATION_TRACK_IDS[track]


def build_protocol_configuration() -> dict[str, Any]:
    """Collect only the curated scientific settings that define the run."""
    missing = [
        name
        for name in expcfg.PROTOCOL_CONFIGURATION_KEYS
        if not hasattr(expcfg, name)
    ]
    if missing:
        raise ProtocolValidationError(
            f"Missing central protocol settings: {missing}"
        )
    return {
        name: _jsonable(getattr(expcfg, name))
        for name in expcfg.PROTOCOL_CONFIGURATION_KEYS
    }


def _manifest_section(parameter: str) -> str:
    if parameter.startswith("PROTOCOL_") or parameter == "RESULTS_SCHEMA_VERSION":
        return "protocol"
    if parameter.startswith(("TARGET_", "NON_TARGET", "REFERENCE_")):
        return "identity"
    if "BATCH" in parameter or parameter == "CV_GROUP_COL":
        return "data_roles"
    if parameter.startswith(("SEGMENTATION_", "QC_")):
        return "quality_control"
    if parameter.startswith(("BALANCED_", "PCA_")):
        return "matrices_pca"
    if parameter.startswith(("SIMCA_PARENT_", "SIMCA_EVALUATION_")):
        return "tracks"
    if parameter.startswith(("INTERNAL_CALIBRATION_", "SIMCA_")):
        return "simca_search"
    return "spectral"


def build_scientific_protocol_manifest() -> pd.DataFrame:
    """Build the task-01 parameter manifest with a stable semantic hash."""
    configuration = build_protocol_configuration()
    configuration_sha256 = sha256_payload(configuration)
    rows = []
    for parameter in sorted(configuration):
        value_json = canonical_json(configuration[parameter])
        rows.append(
            {
                "protocol_version": expcfg.PROTOCOL_VERSION,
                "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
                "protocol_status": expcfg.PROTOCOL_STATUS,
                "protocol_section": _manifest_section(parameter),
                "parameter": parameter,
                "value_json": value_json,
                "parameter_sha256": hashlib.sha256(
                    value_json.encode("utf-8")
                ).hexdigest(),
                "configuration_sha256": configuration_sha256,
            }
        )
    return pd.DataFrame(rows, columns=PROTOCOL_MANIFEST_COLUMNS)


def _contrast_row(
    *,
    contrast_id: str,
    hypothesis_id: str,
    left_track_id: str | None,
    right_track_id: str | None,
    metric: str,
    metric_role: str,
    direction: str,
    primary_unit: str,
    practical_tolerance: float,
    contrast_type: str = "pairwise",
    component_track_ids: Sequence[str] = (),
    contrast_expression: str = "",
) -> dict[str, Any]:
    left_track = "" if left_track_id is None else _track_name(left_track_id)
    right_track = "" if right_track_id is None else _track_name(right_track_id)
    components = [_track_name(track_id) for track_id in component_track_ids]
    if direction == "two_sided":
        decision_rule = (
            "Report left-minus-right and its grouped-bootstrap interval. A "
            "practical difference is claimed only when the interval lies "
            "entirely outside the prespecified negligible interval."
        )
    else:
        decision_rule = (
            "Report the grouped-bootstrap interval and the complete "
            "risk-coverage result. A directional benefit requires an effect "
            "at least as large as the practical tolerance without failure of "
            "the co-primary guardrail."
        )
    return {
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "contrast_id": contrast_id,
        "hypothesis_id": hypothesis_id,
        "scope": "primary",
        "contrast_type": contrast_type,
        "left_track": left_track,
        "left_track_id": left_track_id or "",
        "right_track": right_track,
        "right_track_id": right_track_id or "",
        "component_tracks_json": canonical_json(components),
        "contrast_expression": contrast_expression,
        "metric": metric,
        "metric_role": metric_role,
        "direction": direction,
        "primary_unit": primary_unit,
        "primary_analysis_stage": "external_test_batch4",
        "supporting_analysis_stage": "selection_validation_batch3",
        "estimator": "clustered_source_image_bootstrap_difference",
        "bootstrap_group_col": expcfg.PROTOCOL_BOOTSTRAP_GROUP_COL,
        "confidence_level": expcfg.PROTOCOL_CONFIDENCE_LEVEL,
        "multiplicity_family": hypothesis_id,
        "multiplicity_method": expcfg.PROTOCOL_MULTIPLICITY_METHOD,
        "practical_tolerance_type": (
            "standardized_absolute_difference"
            if metric == "standardized_projection_shift"
            else "absolute_rate_difference"
        ),
        "practical_tolerance": float(practical_tolerance),
        "decision_rule": decision_rule,
        "status": "frozen",
    }


def build_planned_contrasts() -> pd.DataFrame:
    """Return the complete prespecified H1-H4 contrast table."""
    rate_tolerance = float(expcfg.PROTOCOL_RATE_PRACTICAL_TOLERANCE)
    shift_tolerance = float(
        expcfg.PROTOCOL_STANDARDIZED_SHIFT_TOLERANCE
    )
    rows: list[dict[str, Any]] = []

    # H1: training family at fixed object projection and decision mode.
    h1_specs = (
        ("E1", "E5", "target_miss_rate", "co_primary"),
        ("E1", "E5", "false_accept_rate", "co_primary"),
        ("E1", "E5", "balanced_accuracy", "secondary"),
        ("E2", "E6", "target_miss_rate", "co_primary"),
        ("E2", "E6", "false_accept_rate", "co_primary"),
        ("E2", "E6", "decided_balanced_accuracy", "secondary"),
    )
    for index, (left, right, metric, role) in enumerate(h1_specs, start=1):
        rows.append(
            _contrast_row(
                contrast_id=f"H1_C{index:02d}",
                hypothesis_id="H1",
                left_track_id=left,
                right_track_id=right,
                metric=metric,
                metric_role=role,
                direction="two_sided",
                primary_unit="object",
                practical_tolerance=rate_tolerance,
            )
        )

    # H2: training family at fixed pixel projection and decision mode.
    h2_specs = (
        ("E3", "E7", "small_fragment_recall", "co_primary"),
        ("E3", "E7", "fragment_false_discovery_rate", "co_primary"),
        ("E3", "E7", "macro_image_balanced_accuracy", "secondary"),
        ("E4", "E8", "small_fragment_recall", "co_primary"),
        ("E4", "E8", "fragment_false_discovery_rate", "co_primary"),
        ("E4", "E8", "macro_image_decided_balanced_accuracy", "secondary"),
    )
    for index, (left, right, metric, role) in enumerate(h2_specs, start=1):
        rows.append(
            _contrast_row(
                contrast_id=f"H2_C{index:02d}",
                hypothesis_id="H2",
                left_track_id=left,
                right_track_id=right,
                metric=metric,
                metric_role=role,
                direction="two_sided",
                primary_unit="pixel_fragment",
                practical_tolerance=rate_tolerance,
            )
        )

    # H3: difference-in-differences on a common standardized shift metric.
    h3_specs = (
        (
            ("E3", "E1", "E7", "E5"),
            "(E3-E1)-(E7-E5)",
            "2way",
        ),
        (
            ("E4", "E2", "E8", "E6"),
            "(E4-E2)-(E8-E6)",
            "3way",
        ),
    )
    for index, (components, expression, decision_mode) in enumerate(
        h3_specs,
        start=1,
    ):
        rows.append(
            _contrast_row(
                contrast_id=f"H3_C{index:02d}",
                hypothesis_id="H3",
                left_track_id=None,
                right_track_id=None,
                metric="standardized_projection_shift",
                metric_role="primary",
                direction="two_sided",
                primary_unit="standardized_projection_observation",
                practical_tolerance=shift_tolerance,
                contrast_type="difference_in_differences",
                component_track_ids=components,
                contrast_expression=f"{expression}[{decision_mode}]",
            )
        )

    # H4: 3-way versus 2-way at fixed training family and projection.
    h4_pairs = (("E2", "E1"), ("E4", "E3"), ("E6", "E5"), ("E8", "E7"))
    index = 1
    for three_way, two_way in h4_pairs:
        for metric, direction in (
            ("selective_risk_auc", "lower"),
            ("coverage_at_target_miss_guardrail", "higher"),
        ):
            rows.append(
                _contrast_row(
                    contrast_id=f"H4_C{index:02d}",
                    hypothesis_id="H4",
                    left_track_id=three_way,
                    right_track_id=two_way,
                    metric=metric,
                    metric_role="co_primary",
                    direction=direction,
                    primary_unit=(
                        expcfg.SIMCA_EVALUATION_TRACK_SPECS[
                            _track_name(three_way)
                        ]["primary_unit"]
                    ),
                    practical_tolerance=rate_tolerance,
                )
            )
            index += 1

    return pd.DataFrame(rows, columns=PLANNED_CONTRAST_COLUMNS)


def build_inference_plan(
    planned_contrasts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the task-02 preregistered inference plan."""
    if planned_contrasts is None:
        planned_contrasts = build_planned_contrasts()
    contrast_records = planned_contrasts.to_dict(orient="records")
    return {
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
        "status": expcfg.PROTOCOL_STATUS,
        "freeze_date": expcfg.PROTOCOL_FREEZE_DATE,
        "registration_mode": expcfg.PROTOCOL_REGISTRATION_MODE,
        "amendment_justification": expcfg.PROTOCOL_AMENDMENT_JUSTIFICATION,
        "amendment_policy": expcfg.PROTOCOL_AMENDMENT_POLICY,
        "prior_evidence": {
            "status": expcfg.PROTOCOL_PRIOR_RESULTS_STATUS,
            "scope": (
                "All outputs generated before this freeze are supporting or "
                "exploratory and cannot redefine the primary contrasts."
            ),
            "batch4_blinding_claim": expcfg.PROTOCOL_TEST_BLINDING_CLAIM,
        },
        "data_roles": {
            "calibration_batches": list(
                expcfg.PROTOCOL_CALIBRATION_BATCHES
            ),
            "selection_validation_batches": list(
                expcfg.PROTOCOL_VALIDATION_BATCHES
            ),
            "external_test_batches": list(expcfg.PROTOCOL_TEST_BATCHES),
            "external_test_opening_rule": (
                "Open only after the eight-track panel and all thresholds are "
                "locked; never retune after opening."
            ),
        },
        "hypotheses": _jsonable(expcfg.PROTOCOL_PRIMARY_HYPOTHESES),
        "spectral_data_validity": {
            "raw_spectral_range_nm": [
                expcfg.SPECTRAL_START_NM,
                expcfg.SPECTRAL_END_NM,
            ],
            "raw_band_count": expcfg.N_BANDS_RAW,
            "n_remove_start": expcfg.N_REMOVE_START,
            "n_stop_end": expcfg.N_STOP_END,
            "terminal_band_policy": _jsonable(
                expcfg.TERMINAL_BAND_QC_POLICY
            ),
            "pixel_validity_policy": _jsonable(
                expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY
            ),
            "invalid_pixel_stage": (
                "identified during QC and excluded before "
                "matrix representation construction"
            ),
            "object_aggregation_policy": (
                "object mean and median spectra are recomputed "
                "from analysis-valid pixels only"
            ),
            "balanced_pixel_sampling_policy": (
                "sampling occurs only after pixel validity filtering"
            ),
            "absorbance_domain_policy": (
                "R must remain strictly positive; clipping and "
                "imputation are forbidden"
            ),
        },
        "primary_inference_unit": expcfg.PROTOCOL_BOOTSTRAP_GROUP_COL,
        "bootstrap": {
            "method": "clustered_nonparametric_bootstrap",
            "group_column": expcfg.PROTOCOL_BOOTSTRAP_GROUP_COL,
            "n_resamples": expcfg.PROTOCOL_BOOTSTRAP_N_RESAMPLES,
            "random_state": expcfg.PROTOCOL_BOOTSTRAP_RANDOM_STATE,
            "confidence_level": expcfg.PROTOCOL_CONFIDENCE_LEVEL,
            "objects_and_pixels_remain_grouped": True,
            "report_batches_separately": True,
        },
        "multiplicity": {
            "method": expcfg.PROTOCOL_MULTIPLICITY_METHOD,
            "families": ["H1", "H2", "H3", "H4"],
            "report_all_planned_estimates_and_intervals": True,
            "exploratory_analyses_must_be_labelled": True,
        },
        "practical_tolerances": {
            "absolute_rate_difference": (
                expcfg.PROTOCOL_RATE_PRACTICAL_TOLERANCE
            ),
            "standardized_projection_shift": (
                expcfg.PROTOCOL_STANDARDIZED_SHIFT_TOLERANCE
            ),
        },
        "selection": _jsonable(expcfg.PROTOCOL_SELECTION_POLICY),
        "threshold_policy": {
            "direct_object_or_pixel_decision": "simca_margin",
            "fixed_2way_pixel_to_object_vote": list(
                map(float, expcfg.SIMCA_OBJECT_THRESHOLDS)
            ),
            "fixed_vote_scope": ["E3", "E7"],
            "three_way_threshold_source": (
                "track-specific grouped OOF calibration on batches 1-2"
            ),
            "batch3_or_batch4_threshold_learning_forbidden": True,
        },
        "truth_policy": {
            "object_truth": "object_exact",
            "pixel_fragment_primary_truth": "pixel_annotated",
            "target_class": expcfg.SPATIAL_GT_TARGET_CLASS,
            "positive_class": expcfg.SPATIAL_GT_POSITIVE_CLASS,
            "mask_semantics_id": expcfg.SPATIAL_GT_MASK_SEMANTICS_ID,
            "boundary_policy_id": expcfg.SPATIAL_GT_BOUNDARY_POLICY_ID,
            "ambiguity_policy_id": expcfg.SPATIAL_GT_AMBIGUITY_POLICY_ID,
            "double_annotation_policy": (
                expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_POLICY
            ),
            "weak_or_indirect_truth_is_not_primary": True,
        },
        "n_planned_contrasts": int(len(planned_contrasts)),
        "planned_contrasts_sha256": sha256_payload(contrast_records),
    }


def _cartesian_evaluation_tracks() -> set[str]:
    training_tokens = {
        "object_matrix": "object_train",
        "pixel_matrix": "pixel_train",
    }
    return {
        "__".join(
            (
                training_tokens[family],
                projection,
                decision,
            )
        )
        for family in expcfg.SIMCA_MATRIX_FAMILIES
        for projection in expcfg.SIMCA_PROJECTION_LEVELS
        for decision in expcfg.SIMCA_DECISION_MODES
    }


def validate_protocol_contract(
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Validate tasks 01-02 and return an auditable check table."""
    contrasts = build_planned_contrasts()
    configuration = build_protocol_configuration()
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": check, "passed": bool(passed), "detail": str(detail)}
        )

    add(
        "protocol_is_frozen",
        expcfg.PROTOCOL_STATUS == "frozen",
        f"status={expcfg.PROTOCOL_STATUS}",
    )
    batch_sets = [
        set(expcfg.PROTOCOL_CALIBRATION_BATCHES),
        set(expcfg.PROTOCOL_VALIDATION_BATCHES),
        set(expcfg.PROTOCOL_TEST_BATCHES),
    ]
    disjoint = all(
        batch_sets[left].isdisjoint(batch_sets[right])
        for left in range(len(batch_sets))
        for right in range(left + 1, len(batch_sets))
    )
    add("batch_roles_are_disjoint", disjoint, f"roles={batch_sets}")

    expected_parent_count = (
        len(expcfg.SIMCA_MATRIX_FAMILIES)
        * len(expcfg.SIMCA_DECISION_MODES)
    )
    add(
        "four_parent_tracks",
        len(expcfg.SIMCA_PARENT_TRACKS) == expected_parent_count == 4,
        f"observed={list(expcfg.SIMCA_PARENT_TRACKS)}",
    )
    expected_evaluation_tracks = _cartesian_evaluation_tracks()
    observed_evaluation_tracks = set(expcfg.SIMCA_EVALUATION_TRACKS)
    add(
        "eight_evaluation_tracks_are_cartesian",
        observed_evaluation_tracks == expected_evaluation_tracks
        and len(expcfg.SIMCA_EVALUATION_TRACKS) == 8,
        (
            f"missing={sorted(expected_evaluation_tracks - observed_evaluation_tracks)}, "
            f"extra={sorted(observed_evaluation_tracks - expected_evaluation_tracks)}"
        ),
    )
    add(
        "evaluation_track_specs_are_complete",
        set(expcfg.SIMCA_EVALUATION_TRACK_SPECS)
        == observed_evaluation_tracks,
        f"n_specs={len(expcfg.SIMCA_EVALUATION_TRACK_SPECS)}",
    )
    add(
        "evaluation_track_ids_are_E1_to_E8",
        set(expcfg.SIMCA_EVALUATION_TRACK_IDS.values())
        == {f"E{index}" for index in range(1, 9)},
        f"ids={sorted(expcfg.SIMCA_EVALUATION_TRACK_IDS.values())}",
    )

    threshold_tracks = {
        str(spec["track_id"])
        for spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.values()
        if tuple(spec["secondary_object_aggregation_thresholds"])
    }
    configured_vote_thresholds = tuple(map(float, expcfg.SIMCA_OBJECT_THRESHOLDS))
    threshold_values_are_valid = (
        len(configured_vote_thresholds) > 0
        and all(0.0 <= value <= 1.0 for value in configured_vote_thresholds)
        and all(
            left < right
            for left, right in zip(
                configured_vote_thresholds[:-1],
                configured_vote_thresholds[1:],
            )
        )
    )
    threshold_specs_are_consistent = all(
        tuple(map(float, spec["secondary_object_aggregation_thresholds"]))
        in {(), configured_vote_thresholds}
        for spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.values()
    )
    add(
        "fixed_2way_vote_only_on_pixel_projection_tracks",
        (
            threshold_tracks == {"E3", "E7"}
            and threshold_values_are_valid
            and threshold_specs_are_consistent
        ),
        (
            f"tracks={sorted(threshold_tracks)}, "
            f"thresholds={configured_vote_thresholds}"
        ),
    )
    direct_scores = {
        spec["decision_score_type"]
        for spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.values()
    }
    add(
        "direct_decisions_use_simca_margin",
        direct_scores == {"simca_margin"},
        f"score_types={sorted(direct_scores)}",
    )
    add(
        "pareto_is_scoped_by_evaluation_track",
        expcfg.PROTOCOL_SELECTION_POLICY["pareto_scope"]
        == "evaluation_track",
        str(expcfg.PROTOCOL_SELECTION_POLICY),
    )
    add(
        "weighted_scores_are_forbidden",
        not expcfg.PROTOCOL_SELECTION_POLICY["weighted_scores_allowed"],
        (
            "forbidden_columns="
            f"{list(expcfg.ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS)}"
        ),
    )

    hypothesis_ids = {
        hypothesis["hypothesis_id"]
        for hypothesis in expcfg.PROTOCOL_PRIMARY_HYPOTHESES
    }
    add(
        "hypotheses_H1_to_H4_are_prespecified",
        hypothesis_ids == {"H1", "H2", "H3", "H4"},
        f"ids={sorted(hypothesis_ids)}",
    )
    contrast_hypotheses = set(contrasts["hypothesis_id"])
    add(
        "every_hypothesis_has_planned_contrasts",
        contrast_hypotheses == hypothesis_ids,
        f"ids={sorted(contrast_hypotheses)}",
    )
    simple_tracks = set(contrasts["left_track"]) | set(
        contrasts["right_track"]
    )
    simple_tracks.discard("")
    component_tracks = {
        track
        for value in contrasts["component_tracks_json"]
        for track in json.loads(value)
    }
    contrast_tracks = simple_tracks | component_tracks
    add(
        "all_contrast_tracks_are_registered",
        contrast_tracks.issubset(observed_evaluation_tracks),
        f"unknown={sorted(contrast_tracks - observed_evaluation_tracks)}",
    )
    add(
        "no_pending_contrast_status",
        not contrasts["status"].astype(str).str.lower().eq("pending").any(),
        f"statuses={sorted(contrasts['status'].unique())}",
    )
    add(
        "all_primary_contrasts_have_practical_tolerances",
        bool(
            contrasts["practical_tolerance"].notna().all()
            and (contrasts["practical_tolerance"] > 0).all()
        ),
        (
            "range="
            f"{contrasts['practical_tolerance'].min()}-"
            f"{contrasts['practical_tolerance'].max()}"
        ),
    )
    add(
        "source_image_is_primary_bootstrap_group",
        expcfg.PROTOCOL_BOOTSTRAP_GROUP_COL == "source_image",
        f"group={expcfg.PROTOCOL_BOOTSTRAP_GROUP_COL}",
    )
    add(
        "spatial_truth_is_explicit_peanut_presence",
        expcfg.SPATIAL_GT_TARGET_CLASS == expcfg.TARGET_CLASS == "peanut"
        and expcfg.SPATIAL_GT_ANNOTATED_CLASS == "peanut"
        and expcfg.SPATIAL_GT_POSITIVE_CLASS == "peanut"
        and expcfg.SPATIAL_GT_POSITIVE_VALUE == 1
        and expcfg.SPATIAL_GT_NEGATIVE_VALUE == 0,
        (
            f"target={expcfg.SPATIAL_GT_TARGET_CLASS}, "
            f"semantics={expcfg.SPATIAL_GT_MASK_SEMANTICS_ID}"
        ),
    )
    add(
        "all_selected_spatial_images_are_double_annotated",
        expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_POLICY == "all_selected_images"
        and float(expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_FRACTION) == 1.0,
        (
            f"policy={expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_POLICY}, "
            f"fraction={expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_FRACTION}"
        ),
    )
    spectral_stop = expcfg.N_STOP_END
    valid_stop = (
        spectral_stop is not None
        and int(expcfg.N_REMOVE_START)
        < int(spectral_stop)
        <= int(expcfg.N_BANDS_RAW)
    )
    add(
        "spectral_retained_interval_is_valid",
        valid_stop,
        (
            f"N_REMOVE_START={expcfg.N_REMOVE_START}, "
            f"N_STOP_END={expcfg.N_STOP_END}, "
            f"N_BANDS_RAW={expcfg.N_BANDS_RAW}"
        ),
    )
    pixel_policy = expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY
    add(
        "all_zero_pixels_are_excluded",
        bool(pixel_policy["exclude_all_zero"]),
        str(pixel_policy),
    )
    add(
        "common_preprocessing_population_requires_positive_reflectance",
        (
            bool(pixel_policy["require_strictly_positive"])
            and
            expcfg.PREPROCESSING_ABSORBANCE_NONPOSITIVE_POLICY
            == "error"
        ),
        (
            f"pixel_policy={pixel_policy}, "
            "absorbance_policy="
            f"{expcfg.PREPROCESSING_ABSORBANCE_NONPOSITIVE_POLICY}"
        ),
    )
    expected_n_bands = (
        int(expcfg.N_STOP_END)
        - int(expcfg.N_REMOVE_START)
    )
    add(
        "retained_band_count_is_61",
        expected_n_bands == 61,
        f"retained_n_bands={expected_n_bands}",
    )
    add(
        "configuration_key_list_is_complete",
        len(configuration) == len(expcfg.PROTOCOL_CONFIGURATION_KEYS),
        f"n={len(configuration)}",
    )

    checks_df = pd.DataFrame(checks, columns=PROTOCOL_CHECK_COLUMNS)
    if strict and not bool(checks_df["passed"].all()):
        failed = checks_df.loc[~checks_df["passed"]].to_dict(
            orient="records"
        )
        raise ProtocolValidationError(
            f"Frozen protocol validation failed: {failed}"
        )
    return checks_df


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def freeze_protocol(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate, write and hash all frozen task-01/task-02 artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = expcfg.PROTOCOL_OUTPUT_FILENAMES
    paths = {
        key: output_dir / filename for key, filename in filenames.items()
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Protocol artifacts already exist. Use an explicit new protocol "
            "version or pass overwrite=True only to regenerate the identical "
            f"declared version: {[str(path) for path in existing]}"
        )

    checks = validate_protocol_contract(strict=True)
    manifest = build_scientific_protocol_manifest()
    contrasts = build_planned_contrasts()
    inference_plan = build_inference_plan(contrasts)

    configuration_sha256 = manifest["configuration_sha256"].iloc[0]
    inference_plan_sha256 = sha256_payload(inference_plan)
    contrasts_sha256 = sha256_payload(
        contrasts.to_dict(orient="records")
    )

    save_parquet(manifest, paths["manifest"], optimize=False)
    save_parquet(checks, paths["checks"], optimize=False)
    _write_json(paths["inference_plan"], inference_plan)
    save_parquet(contrasts, paths["planned_contrasts"], optimize=False)

    artifact_sha256 = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key != "lock"
    }
    lock_payload = {
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
        "status": expcfg.PROTOCOL_STATUS,
        "freeze_date": expcfg.PROTOCOL_FREEZE_DATE,
        "configuration_sha256": configuration_sha256,
        "inference_plan_sha256": inference_plan_sha256,
        "planned_contrasts_sha256": contrasts_sha256,
        "artifact_sha256": artifact_sha256,
        "immutable": True,
        "amendment_justification": expcfg.PROTOCOL_AMENDMENT_JUSTIFICATION,
        "amendment_policy": expcfg.PROTOCOL_AMENDMENT_POLICY,
    }
    lock_payload["lock_sha256"] = sha256_payload(lock_payload)
    _write_json(paths["lock"], lock_payload)

    return {
        "paths": paths,
        "checks": checks,
        "configuration_sha256": configuration_sha256,
        "inference_plan_sha256": inference_plan_sha256,
        "planned_contrasts_sha256": contrasts_sha256,
        "lock_sha256": lock_payload["lock_sha256"],
    }


def verify_frozen_protocol(
    output_dir: str | Path,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Verify that the on-disk bundle still matches the central configuration."""
    output_dir = Path(output_dir)
    paths = {
        key: output_dir / filename
        for key, filename in expcfg.PROTOCOL_OUTPUT_FILENAMES.items()
    }
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": check, "passed": bool(passed), "detail": str(detail)}
        )

    missing = [str(path) for path in paths.values() if not path.exists()]
    add("all_frozen_artifacts_exist", not missing, f"missing={missing}")
    if missing:
        result = pd.DataFrame(checks, columns=PROTOCOL_CHECK_COLUMNS)
        if strict:
            raise ProtocolValidationError(
                f"Frozen protocol artifacts are missing: {missing}"
            )
        return result

    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    manifest = build_scientific_protocol_manifest()
    contrasts = build_planned_contrasts()
    inference_plan = build_inference_plan(contrasts)
    expected_semantic_hashes = {
        "configuration_sha256": manifest["configuration_sha256"].iloc[0],
        "inference_plan_sha256": sha256_payload(inference_plan),
        "planned_contrasts_sha256": sha256_payload(
            contrasts.to_dict(orient="records")
        ),
    }
    for key, expected in expected_semantic_hashes.items():
        observed = lock.get(key)
        add(
            f"{key}_matches_current_protocol",
            observed == expected,
            f"expected={expected}, observed={observed}",
        )

    for key, expected in lock.get("artifact_sha256", {}).items():
        observed = sha256_file(paths[key])
        add(
            f"{key}_file_checksum_matches_lock",
            observed == expected,
            f"expected={expected}, observed={observed}",
        )

    lock_without_hash = dict(lock)
    observed_lock_hash = lock_without_hash.pop("lock_sha256", None)
    expected_lock_hash = sha256_payload(lock_without_hash)
    add(
        "lock_checksum_is_valid",
        observed_lock_hash == expected_lock_hash,
        (
            f"expected={expected_lock_hash}, "
            f"observed={observed_lock_hash}"
        ),
    )

    result = pd.DataFrame(checks, columns=PROTOCOL_CHECK_COLUMNS)
    if strict and not bool(result["passed"].all()):
        failed = result.loc[~result["passed"]].to_dict(orient="records")
        raise ProtocolValidationError(
            f"Frozen protocol verification failed: {failed}"
        )
    return result


def make_selection_id(
    entity_type: str,
    payload: Mapping,
    *,
    length: int = 20,
) -> str:
    """Build one deterministic scientific entity identifier."""
    entity_type = str(entity_type)

    if entity_type not in expcfg.SELECTION_ID_PREFIXES:
        raise KeyError(
            f"Unknown selection entity type: {entity_type!r}. "
            f"Allowed={sorted(expcfg.SELECTION_ID_PREFIXES)}"
        )

    prefix = expcfg.SELECTION_ID_PREFIXES[entity_type]
    return f"{prefix}_{sha256_payload(dict(payload))[:int(length)]}"


def make_audit_id(
    *,
    stage: str,
    substage: str,
    entity_type: str,
    entity_id: str,
    metric: str = "",
    related_entity_id: str = "",
    ordinal: int = 0,
) -> str:
    """Build one deterministic audit-event identifier."""
    return make_selection_id(
        "audit_event",
        {
            "stage": str(stage),
            "substage": str(substage),
            "entity_type": str(entity_type),
            "entity_id": str(entity_id),
            "metric": str(metric),
            "related_entity_id": str(related_entity_id),
            "ordinal": int(ordinal),
        },
    )


__all__ = [
    "PLANNED_CONTRAST_COLUMNS",
    "PROTOCOL_CHECK_COLUMNS",
    "PROTOCOL_MANIFEST_COLUMNS",
    "ProtocolValidationError",
    "build_inference_plan",
    "build_planned_contrasts",
    "build_protocol_configuration",
    "build_simca_track_contracts",
    "build_scientific_protocol_manifest",
    "canonical_json",
    "freeze_protocol",
    "sha256_payload",
    "sha256_ndarray",
    "sha256_dataframe",
    "sha256_file",
    "validate_protocol_contract",
    "validate_selection_only_protocol_lineage",
    "verify_frozen_protocol",
    "make_selection_id",
    "make_audit_id",
]
