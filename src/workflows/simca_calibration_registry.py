"""Canonical SIMCA model/run registry for notebook 03B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.protocol_governance import sha256_file
from src.workflows.simca import (
    valid_sg_parameter_pairs,
)
from src.matrices.matrix_registry import matrix_family_from_method
from src.workflows.simca_candidates import (
    build_pca_preprocessing_configs_by_matrix_family,
)
from src.models.simca_rules import rule_family_from_variant
from src.utils import require_columns, normalize_integer_sequence


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items())
        }
    return value


def stable_id(
    prefix: str,
    row: Mapping[str, Any],
    columns: Sequence[str],
    *,
    length: int = 20,
) -> str:
    payload = {
        column: _json_value(row.get(column))
        for column in columns
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[: int(length)]
    return f"{prefix}_{digest}"


def stochastic_model_mask(
    models: pd.DataFrame,
) -> pd.Series:
    """Return the protocol-defined stochastic SIMCA models.

    Stochasticity is a property of the SIMCA execution mechanism, not of
    notebook 05 itself. A model is stochastic only when both its matrix
    construction and sampling strategy are declared stochastic globally.
    """
    require_columns(
        models,
        (
            "matrix_method",
            "balanced_pixel_strategy",
        ),
        "SIMCA model table",
    )
    stochastic_methods = tuple(
        map(
            str,
            expcfg.SIMCA_STOCHASTIC_MATRIX_METHODS,
        )
    )
    stochastic_strategies = tuple(
        map(
            str,
            expcfg.SIMCA_STOCHASTIC_SAMPLING_STRATEGIES,
        )
    )
    return (
        models["matrix_method"]
        .astype("string")
        .isin(stochastic_methods)
        &
        models["balanced_pixel_strategy"]
        .astype("string")
        .isin(stochastic_strategies)
    ).astype(bool)


def attach_execution_ids(
    executions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical fit_id and projection_id values.

    The exact stable_id definitions used by notebook 03B are preserved.

    Scientific identity
    -------------------
    model_id

    Execution identity
    ------------------
    (model_id, random_state)

    Technical identities
    --------------------
    fit_id and projection_id

    IDs are computed once per unique technical definition, while input row
    order is explicitly restored after the merge operations.
    """
    required = (
        "model_id",
        *expcfg.SIMCA_FIT_ID_COLUMNS,
        *expcfg.SIMCA_PROJECTION_ID_COLUMNS[1:],
    )
    require_columns(
        executions,
        required,
        "SIMCA execution rows",
    )

    out = executions.drop(
        columns=(
            "fit_id",
            "projection_id",
        ),
        errors="ignore",
    ).copy()

    order_column = "__simca_execution_order__"
    if order_column in out.columns:
        raise RuntimeError(
            f"Reserved temporary column already exists: {order_column!r}."
        )

    out[order_column] = np.arange(
        len(out),
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # fit_id
    # ------------------------------------------------------------------
    fit_columns = list(
        expcfg.SIMCA_FIT_ID_COLUMNS
    )

    fit_keys = (
        out[fit_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    fit_keys["fit_id"] = [
        stable_id(
            "fit",
            row,
            expcfg.SIMCA_FIT_ID_COLUMNS,
        )
        for row in fit_keys.to_dict("records")
    ]

    if fit_keys["fit_id"].duplicated().any():
        raise RuntimeError(
            "A fit_id hash collision was detected."
        )

    out = out.merge(
        fit_keys,
        on=fit_columns,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # projection_id
    # ------------------------------------------------------------------
    projection_columns = list(
        expcfg.SIMCA_PROJECTION_ID_COLUMNS
    )

    projection_keys = (
        out[projection_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    projection_keys["projection_id"] = [
        stable_id(
            "projection",
            row,
            expcfg.SIMCA_PROJECTION_ID_COLUMNS,
        )
        for row in projection_keys.to_dict("records")
    ]

    if projection_keys["projection_id"].duplicated().any():
        raise RuntimeError(
            "A projection_id hash collision was detected."
        )

    out = out.merge(
        projection_keys,
        on=projection_columns,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if out[
        [
            "fit_id",
            "projection_id",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Could not materialize fit_id/projection_id."
        )

    return (
        out
        .sort_values(
            order_column,
            kind="mergesort",
        )
        .drop(
            columns=order_column
        )
        .reset_index(drop=True)
    )

def build_additional_seed_execution_registry(
    model_catalog: pd.DataFrame,
    model_ids: Sequence[str],
    random_states: Sequence[int] = (
        expcfg.SIMCA_ROBUSTNESS_ADDITIONAL_RANDOM_STATES
    ),
    *,
    stochastic_only: bool = True,
    existing_executions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build notebook-05 repeated executions without creating a new model ID.

    Scientific identity
    -------------------
    ``model_id`` remains unchanged.

    Execution identity
    ------------------
    One execution is ``(model_id, random_state)``.

    Technical identity
    ------------------
    ``fit_id`` and ``projection_id`` are regenerated with exactly the same
    stable-ID definitions as notebook 03B.

    By default, only protocol-defined stochastic models are accepted.
    Deterministic models are never cloned unless ``stochastic_only=False`` is
    supplied explicitly for a diagnostic use case.
    """
    require_columns(
        model_catalog,
        expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
        "model_catalog",
    )

    catalog = model_catalog.loc[
        :,
        list(
            expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS
        ),
    ].copy()

    catalog["model_id"] = (
        catalog["model_id"]
        .astype(str)
    )

    catalog["track_id"] = (
        catalog["track_id"]
        .astype(str)
    )

    if catalog["model_id"].duplicated().any():
        raise RuntimeError(
            "model_catalog.model_id must be unique."
        )

    # ------------------------------------------------------------------
    # Requested models
    # ------------------------------------------------------------------
    raw_model_ids = pd.Series(
        list(model_ids),
        dtype="object",
    )

    if raw_model_ids.empty:
        return pd.DataFrame(
            columns=(
                expcfg.INTERNAL_CALIBRATION_EXECUTION_COLUMNS
            )
        )

    if raw_model_ids.isna().any():
        raise ValueError(
            "model_ids cannot contain missing values."
        )

    requested_model_ids = (
        raw_model_ids
        .astype(str)
        .tolist()
    )

    if len(requested_model_ids) != len(
        set(requested_model_ids)
    ):
        raise ValueError(
            "model_ids contains duplicates."
        )

    # ------------------------------------------------------------------
    # Requested new seeds
    # ------------------------------------------------------------------
    states = normalize_integer_sequence(
        random_states,
        name="random_states",
        allow_empty=False,
    )

    base_states = set(
        map(
            int,
            expcfg.SIMCA_ROBUSTNESS_BASE_RANDOM_STATES,
        )
    )

    additional_states = set(
        map(
            int,
            expcfg
            .SIMCA_ROBUSTNESS_ADDITIONAL_RANDOM_STATES,
        )
    )

    overlap_with_base = sorted(
        set(states).intersection(base_states)
    )

    if overlap_with_base:
        raise ValueError(
            "Additional seed registry cannot recreate the base 04C seeds: "
            f"{overlap_with_base}."
        )

    unconfigured_states = sorted(
        set(states) - additional_states
    )

    if unconfigured_states:
        raise ValueError(
            "Random states are outside the frozen notebook-05 robustness "
            f"panel: {unconfigured_states}."
        )

    # ------------------------------------------------------------------
    # Select existing scientific models.
    # ------------------------------------------------------------------
    selected = catalog.loc[
        catalog["model_id"].isin(
            requested_model_ids
        )
    ].copy()

    observed_model_ids = set(
        selected["model_id"]
    )

    missing_model_ids = sorted(
        set(requested_model_ids)
        - observed_model_ids
    )

    if missing_model_ids:
        raise KeyError(
            "Requested model_id values are absent from model_catalog: "
            f"{missing_model_ids}."
        )

    # Every scientific model must belong to one of the frozen E1-E8 tracks.
    expected_tracks = set(
        map(
            str,
            expcfg.SIMCA_ROBUSTNESS_TRACK_IDS,
        )
    )

    unknown_tracks = sorted(
        set(selected["track_id"])
        - expected_tracks
    )

    if unknown_tracks:
        raise RuntimeError(
            "Additional seed executions contain unknown tracks: "
            f"{unknown_tracks}."
        )

    # ------------------------------------------------------------------
    # Stochasticity contract
    # ------------------------------------------------------------------
    is_stochastic = stochastic_model_mask(
        selected
    )

    if stochastic_only and not is_stochastic.all():
        deterministic_model_ids = (
            selected.loc[
                ~is_stochastic,
                "model_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Additional seeds were requested for deterministic models. "
            "Pass stochastic_only=False only for an explicit diagnostic: "
            f"{deterministic_model_ids}."
        )

    # ------------------------------------------------------------------
    # Vectorized model x seed expansion.
    # ------------------------------------------------------------------
    seed_frame = pd.DataFrame(
        {
            "random_state": states,
        }
    )

    executions = selected.merge(
        seed_frame,
        how="cross",
    )

    executions["random_state"] = (
        executions["random_state"]
        .astype(int)
    )

    # Existing 03B ID machinery is reused. No new identifier is introduced.
    executions = attach_execution_ids(
        executions
    )

    executions = executions.reindex(
        columns=(
            expcfg.INTERNAL_CALIBRATION_EXECUTION_COLUMNS
        )
    )

    natural_key = [
        "model_id",
        "random_state",
    ]

    if executions.duplicated(
        natural_key
    ).any():
        raise RuntimeError(
            "Additional (model_id, random_state) execution keys "
            "are duplicated."
        )

    # A model_id must still represent one and only one scientific
    # configuration after seed expansion.
    identity = (
        executions
        .groupby(
            "model_id",
            dropna=False,
        )[
            list(
                expcfg.SIMCA_MODEL_PARAMETER_COLUMNS
            )
        ]
        .nunique(dropna=False)
        .max(axis=1)
    )

    if identity.gt(1).any():
        raise RuntimeError(
            "A model_id maps to multiple scientific configurations."
        )

    # ------------------------------------------------------------------
    # Optional collision check against the frozen 03B/04C executions.
    # ------------------------------------------------------------------
    if (
        existing_executions is not None
        and not existing_executions.empty
    ):
        require_columns(
            existing_executions,
            natural_key,
            "existing_executions",
        )

        existing_keys = (
            existing_executions[
                natural_key
            ]
            .copy()
        )

        existing_keys["model_id"] = (
            existing_keys["model_id"]
            .astype(str)
        )

        existing_keys["random_state"] = (
            pd.to_numeric(
                existing_keys["random_state"],
                errors="raise",
            )
            .astype(int)
        )

        overlap = (
            executions[
                natural_key
            ]
            .merge(
                existing_keys.drop_duplicates(),
                on=natural_key,
                how="inner",
            )
        )

        if not overlap.empty:
            raise RuntimeError(
                "Additional executions overlap existing execution keys: "
                f"{overlap.to_dict('records')[:10]}."
            )

    # Sorting is only for deterministic persistence/hashing.
    # No ranking or comparison occurs across tracks here.
    return (
        executions
        .sort_values(
            [
                "track_id",
                "model_id",
                "random_state",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def validate_internal_calibration_manifest(
    manifest: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path],
    *,
    required_artifacts: Sequence[str],
    protocol_hash: str,
) -> None:
    """Verify the exact 03B files consumed by a downstream notebook.

    The 03B checkpoint manifest is the lineage boundary. Downstream notebooks
    must not accept a file that merely has the expected name: every consumed
    artifact is checked against its frozen SHA-256 entry.
    """
    if str(manifest.get("protocol_version")) != str(expcfg.PROTOCOL_VERSION):
        raise RuntimeError("03B manifest uses another protocol version.")
    if str(manifest.get("schema_version")) != str(expcfg.RESULTS_SCHEMA_VERSION):
        raise RuntimeError("03B manifest uses another results schema.")
    if str(manifest.get("protocol_hash")) != str(protocol_hash):
        raise RuntimeError("03B manifest does not match the frozen protocol.")

    raw_entries = manifest.get("artifacts")
    if not isinstance(raw_entries, list):
        raise RuntimeError("03B manifest has no artifact registry.")
    entries = {
        str(entry.get("name")): entry
        for entry in raw_entries
        if isinstance(entry, Mapping) and entry.get("name") is not None
    }
    required = tuple(dict.fromkeys(map(str, required_artifacts)))
    missing_entries = sorted(set(required) - set(entries))
    missing_paths = sorted(set(required) - set(map(str, artifact_paths)))
    if missing_entries or missing_paths:
        raise RuntimeError(
            "03B manifest contract is incomplete: "
            f"missing_entries={missing_entries}, missing_paths={missing_paths}."
        )
    for name in required:
        path = Path(artifact_paths[name])
        if not path.is_file():
            raise FileNotFoundError(f"Missing 03B artifact: {path}")
        expected = str(entries[name].get("sha256", ""))
        observed = sha256_file(path)
        if not expected or observed != expected:
            raise RuntimeError(
                f"03B artifact {name!r} does not match checkpoint_manifest.json."
            )


def build_selected_execution_registry(
    model_catalog: pd.DataFrame,
    selected_models: pd.DataFrame,
    selected_runs: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    *,
    track_contracts: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize selected 03B tables without introducing another identifier.

    Returns one compact row per ``(model_id, random_state)`` and the matching
    long threshold rows. ``selected_thresholds.parquet`` intentionally also
    contains intermediate policies, so filtering it by the selected natural
    execution key is mandatory before downstream use.
    """
    table_contracts = (
        (
            model_catalog,
            expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS,
            "model_catalog",
        ),
        (
            selected_models,
            expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS,
            "selected_models",
        ),
        (
            selected_runs,
            expcfg.INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS,
            "selected_runs",
        ),
        (
            selected_thresholds,
            expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS,
            "selected_thresholds",
        ),
    )
    for frame, required, name in table_contracts:
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing columns: {missing}")

    catalog = model_catalog.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS)
    ].copy()
    if catalog["model_id"].astype(str).duplicated().any():
        raise RuntimeError("model_catalog.model_id must be unique.")

    models = selected_models.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS)
    ].copy()
    if models["model_id"].astype(str).duplicated().any():
        raise RuntimeError("selected_models.model_id must be unique.")
    if models.empty or not models["selection_status"].astype(str).eq("selected").all():
        raise RuntimeError("selected_models must contain selected rows only.")

    runs = selected_runs.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS)
    ].copy()
    run_keys = ["model_id", "random_state"]
    if runs.duplicated(run_keys).any():
        raise RuntimeError("Selected (model_id, random_state) keys must be unique.")
    if runs.empty or runs[["fit_id", "projection_id"]].isna().any().any():
        raise RuntimeError("Every selected execution needs fit_id and projection_id.")

    selected_ids = set(models["model_id"].astype(str))
    run_ids = set(runs["model_id"].astype(str))
    catalog_ids = set(catalog["model_id"].astype(str))
    if run_ids != selected_ids:
        raise RuntimeError(
            "selected_runs must cover every selected model, and no other model."
        )
    if not selected_ids.issubset(catalog_ids):
        raise RuntimeError("A selected model is absent from model_catalog.")

    model_metadata = models[["model_id"]].merge(
        catalog[
            [
                "model_id",
                "track_id",
                "decision_mode",
                "projection_level",
            ]
        ],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    executions = runs.merge(
        model_metadata,
        on="model_id",
        how="left",
        validate="many_to_one",
    ).reindex(columns=expcfg.DOMAIN_SPATIAL_SELECTED_EXECUTION_COLUMNS)
    if executions.isna().any().any():
        raise RuntimeError("Selected execution metadata is incomplete.")

    if track_contracts is not None:
        required_contract = {"track_id", "decision_mode", "projection_level"}
        missing = sorted(required_contract - set(track_contracts.columns))
        if missing:
            raise KeyError(f"track_contracts is missing columns: {missing}")
        contracts = track_contracts[
            ["track_id", "decision_mode", "projection_level"]
        ].drop_duplicates()
        if contracts["track_id"].astype(str).duplicated().any():
            raise RuntimeError("track_contracts.track_id must be unique.")
        checked = executions.merge(
            contracts,
            on="track_id",
            how="left",
            suffixes=("", "_contract"),
            validate="many_to_one",
        )
        for column in ("decision_mode", "projection_level"):
            if checked[f"{column}_contract"].isna().any() or not checked[
                column
            ].astype(str).eq(checked[f"{column}_contract"].astype(str)).all():
                raise RuntimeError(
                    f"Selected models conflict with the track contract for {column}."
                )

    thresholds = selected_thresholds.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_THRESHOLD_COLUMNS)
    ].merge(
        executions[run_keys],
        on=run_keys,
        how="inner",
        validate="many_to_one",
    )
    threshold_keys = [*run_keys, "decision_scope"]
    if thresholds.duplicated(threshold_keys).any():
        raise RuntimeError("A selected execution has duplicate decision scopes.")
    allowed_scopes = {"direct", "pixel_to_object"}
    unknown_scopes = sorted(
        set(thresholds["decision_scope"].astype(str)) - allowed_scopes
    )
    if unknown_scopes:
        raise RuntimeError(f"Unknown selected threshold scopes: {unknown_scopes}")

    scope_counts = (
        thresholds.assign(__present=1)
        .pivot_table(
            index=run_keys,
            columns="decision_scope",
            values="__present",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=["direct", "pixel_to_object"], fill_value=0)
        .reset_index()
    )
    checked_scopes = executions.merge(
        scope_counts,
        on=run_keys,
        how="left",
        validate="one_to_one",
    )
    checked_scopes[["direct", "pixel_to_object"]] = checked_scopes[
        ["direct", "pixel_to_object"]
    ].fillna(0).astype(int)
    expected_secondary = checked_scopes["projection_level"].astype(str).eq(
        "pixel_projection"
    )
    if not checked_scopes["direct"].eq(1).all() or not checked_scopes[
        "pixel_to_object"
    ].eq(expected_secondary.astype(int)).all():
        raise RuntimeError(
            "Selected threshold scopes do not match the projection-level contract."
        )

    direct = thresholds.loc[
        thresholds["decision_scope"].astype(str).eq("direct")
    ].merge(
        executions[[*run_keys, "decision_mode"]],
        on=run_keys,
        how="left",
        validate="one_to_one",
    )
    lower = pd.to_numeric(direct["lower_threshold"], errors="coerce")
    upper = pd.to_numeric(direct["upper_threshold"], errors="coerce")
    if not np.isfinite(np.column_stack([lower, upper])).all():
        raise RuntimeError("Every selected direct threshold must be finite.")
    two_way = direct["decision_mode"].astype(str).eq("2way")
    three_way = direct["decision_mode"].astype(str).eq("3way")
    if (~(two_way | three_way)).any():
        raise RuntimeError("Selected models contain an unknown decision mode.")
    if not np.isclose(lower[two_way], upper[two_way]).all():
        raise RuntimeError("A selected 2-way direct threshold is inconsistent.")
    if not (lower[three_way] < upper[three_way]).all():
        raise RuntimeError("A selected 3-way direct threshold is inconsistent.")

    secondary = thresholds.loc[
        thresholds["decision_scope"].astype(str).eq("pixel_to_object")
    ].merge(
        executions.loc[
            executions["projection_level"].astype(str).eq("pixel_projection"),
            [*run_keys, "decision_mode"],
        ],
        on=run_keys,
        how="left",
        validate="one_to_one",
    )
    secondary_lower = pd.to_numeric(
        secondary["lower_threshold"], errors="coerce"
    )
    secondary_upper = pd.to_numeric(
        secondary["upper_threshold"], errors="coerce"
    )
    if not np.isfinite(
        np.column_stack([secondary_lower, secondary_upper])
    ).all():
        raise RuntimeError("Every selected pixel-to-object threshold must be finite.")
    secondary_two_way = secondary["decision_mode"].astype(str).eq("2way")
    secondary_three_way = secondary["decision_mode"].astype(str).eq("3way")
    vote = pd.to_numeric(secondary["vote_threshold"], errors="coerce")
    if (
        not np.isclose(
            secondary_lower[secondary_two_way],
            secondary_upper[secondary_two_way],
        ).all()
        or not np.isfinite(vote[secondary_two_way]).all()
        or not np.isclose(
            vote[secondary_two_way], secondary_lower[secondary_two_way]
        ).all()
    ):
        raise RuntimeError("A selected 2-way pixel-to-object threshold is inconsistent.")
    if not (
        secondary_lower[secondary_three_way]
        < secondary_upper[secondary_three_way]
    ).all():
        raise RuntimeError("A selected 3-way pixel-to-object threshold is inconsistent.")

    thresholds = thresholds.sort_values(
        [*run_keys, "decision_scope"], kind="mergesort"
    ).reset_index(drop=True)
    executions = executions.sort_values(run_keys, kind="mergesort").reset_index(
        drop=True
    )
    return executions, thresholds


def _projection_methods(
    matrix_method: str,
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    allowed = tuple(
        map(
            str,
            json.loads(
                str(contract["allowed_projection_methods_json"])
            ),
        )
    )
    if contract["projection_matrix_policy"] == (
        "match_object_training_method"
    ):
        if matrix_method not in allowed:
            raise RuntimeError(
                f"Projection method {matrix_method!r} is not allowed."
            )
        return (matrix_method,)
    return allowed


def build_internal_calibration_candidate_runs(
    pca_selected: pd.DataFrame,
    track_contracts: pd.DataFrame,
    *,
    matrix_methods: Sequence[str] = (
        expcfg.INTERNAL_CALIBRATION_MATRIX_METHODS
    ),
    m_values: Sequence[int] = (
        expcfg.INTERNAL_CALIBRATION_M_VALUES
    ),
    pixel_strategies: Sequence[str] = (
        expcfg.INTERNAL_CALIBRATION_PIXEL_STRATEGIES
    ),
    n_components_values: Sequence[int] = (
        expcfg.INTERNAL_CALIBRATION_N_COMPONENTS_VALUES
    ),
    rule_variants: Sequence[str] = (
        expcfg.INTERNAL_CALIBRATION_RULE_VARIANTS
    ),
    alpha_values: Sequence[float] = (
        expcfg.INTERNAL_CALIBRATION_ALPHA_VALUES
    ),
    sg_windows: Sequence[int] = (
        expcfg.INTERNAL_CALIBRATION_SG_WINDOWS
    ),
    sg_polyorders: Sequence[int] = (
        expcfg.INTERNAL_CALIBRATION_SG_POLYORDERS
    ),
    dilation_radii: Sequence[int] = (
        expcfg.INTERNAL_CALIBRATION_DILATION_RADII
    ),
    random_seeds: Sequence[int] = (
        expcfg.INTERNAL_CALIBRATION_RANDOM_SEEDS
    ),
) -> pd.DataFrame:
    """Build canonical 03B model executions.

    model_id identifies the scientific model and therefore excludes the seed.
    random_state identifies a repeated execution.

    fit_id and projection_id are materialized centrally by
    attach_execution_ids(), using the same stable-ID contract as before.
    """
    preprocessing_by_family = (
        build_pca_preprocessing_configs_by_matrix_family(
            pca_selected
        )
    )

    random_seeds = tuple(
        map(
            int,
            random_seeds,
        )
    )
    default_seed = int(
        random_seeds[0]
    )

    rule_variants = tuple(
        map(
            str,
            rule_variants,
        )
    )

    rule_specs = tuple(
        (
            rule_variant,
            rule_family_from_variant(
                rule_variant
            ),
            (
                "calibration_train_only"
                if rule_variant.endswith("_emp_cv")
                else "theoretical_train_fit"
            ),
        )
        for rule_variant in rule_variants
    )

    rows: list[dict[str, Any]] = []

    for matrix_method in map(
        str,
        matrix_methods,
    ):
        matrix_family = (
            matrix_family_from_method(
                matrix_method
            )
        )

        preprocessing_configs = (
            preprocessing_by_family.get(
                matrix_family,
                {},
            )
        )

        if not preprocessing_configs:
            raise RuntimeError(
                f"No PCA preprocessing for {matrix_family!r}."
            )

        if matrix_method == "balanced_pixels":
            sampling_options = tuple(
                (
                    int(m),
                    str(strategy),
                )
                for m in m_values
                for strategy in pixel_strategies
            )
        else:
            sampling_options = (
                (
                    np.nan,
                    "not_applicable",
                ),
            )

        contracts = track_contracts.loc[
            track_contracts[
                "training_matrix_family"
            ]
            .astype(str)
            .eq(matrix_family)
        ]

        contract_records = tuple(
            contracts.to_dict(
                "records"
            )
        )

        for (
            preprocessing,
            raw_steps,
        ) in preprocessing_configs.items():
            steps = tuple(
                map(
                    str,
                    raw_steps,
                )
            )

            preprocessing_steps = (
                "+".join(steps)
            )

            sg_pairs = (
                valid_sg_parameter_pairs(
                    preprocessing_steps=steps,
                    sg_window_length_values=sg_windows,
                    sg_polyorder_values=sg_polyorders,
                    default_sg_window_length=(
                        expcfg.SG_DEFAULT_WINDOW
                    ),
                    default_sg_polyorder=(
                        expcfg.SG_POLYORDER
                    ),
                )
            )

            for (
                m,
                strategy,
            ) in sampling_options:
                if (
                    matrix_method
                    == "balanced_pixels"
                    and strategy == "random"
                ):
                    seeds = random_seeds
                else:
                    seeds = (
                        default_seed,
                    )

                for (
                    sg_window,
                    sg_polyorder,
                ) in sg_pairs:
                    for n_components in (
                        n_components_values
                    ):
                        for alpha in (
                            alpha_values
                        ):
                            for dilation_radius in (
                                dilation_radii
                            ):
                                for (
                                    rule_variant,
                                    rule_family,
                                    limit_source,
                                ) in rule_specs:
                                    for contract in (
                                        contract_records
                                    ):
                                        projection_methods = (
                                            _projection_methods(
                                                matrix_method,
                                                contract,
                                            )
                                        )

                                        for projection_method in (
                                            projection_methods
                                        ):
                                            model = {
                                                "evaluation_track": str(
                                                    contract[
                                                        "evaluation_track"
                                                    ]
                                                ),
                                                "track_id": str(
                                                    contract[
                                                        "track_id"
                                                    ]
                                                ),
                                                "parent_track": str(
                                                    contract[
                                                        "parent_track"
                                                    ]
                                                ),
                                                "decision_mode": str(
                                                    contract[
                                                        "decision_mode"
                                                    ]
                                                ),
                                                "matrix_family": (
                                                    matrix_family
                                                ),
                                                "matrix_method": (
                                                    matrix_method
                                                ),
                                                "projection_level": str(
                                                    contract[
                                                        "projection_level"
                                                    ]
                                                ),
                                                "projection_matrix_method": (
                                                    projection_method
                                                ),
                                                "m": m,
                                                "balanced_pixel_strategy": (
                                                    strategy
                                                ),
                                                "preprocessing": str(
                                                    preprocessing
                                                ),
                                                "preprocessing_steps": (
                                                    preprocessing_steps
                                                ),
                                                "rule_family": (
                                                    rule_family
                                                ),
                                                "rule_variant": (
                                                    rule_variant
                                                ),
                                                "limit_source": (
                                                    limit_source
                                                ),
                                                "n_components": int(
                                                    n_components
                                                ),
                                                "alpha": float(
                                                    alpha
                                                ),
                                                "sg_window_length": int(
                                                    sg_window
                                                ),
                                                "sg_polyorder": int(
                                                    sg_polyorder
                                                ),
                                                "position_dilation_radius": int(
                                                    dilation_radius
                                                ),
                                            }

                                            model_id = stable_id(
                                                "model",
                                                model,
                                                expcfg
                                                .SIMCA_MODEL_ID_COLUMNS,
                                            )

                                            for random_state in seeds:
                                                rows.append(
                                                    {
                                                        **model,
                                                        "model_id": (
                                                            model_id
                                                        ),
                                                        "random_state": int(
                                                            random_state
                                                        ),
                                                    }
                                                )

    out = (
        pd.DataFrame(rows)
        .drop_duplicates(
            [
                "model_id",
                "random_state",
            ]
        )
        .reset_index(drop=True)
    )

    # One canonical implementation for technical IDs.
    out = attach_execution_ids(
        out
    )

    out = out.reindex(
        columns=(
            expcfg
            .INTERNAL_CALIBRATION_EXECUTION_COLUMNS
        )
    )

    natural_key = [
        "model_id",
        "random_state",
    ]

    if out.duplicated(
        natural_key
    ).any():
        raise RuntimeError(
            "(model_id, random_state) must be unique."
        )

    identity = (
        out.groupby(
            "model_id",
            dropna=False,
        )[
            list(
                expcfg
                .SIMCA_MODEL_PARAMETER_COLUMNS
            )
        ]
        .nunique(
            dropna=False
        )
        .max(axis=1)
    )

    if identity.gt(1).any():
        raise RuntimeError(
            "A model_id maps to multiple scientific configurations."
        )

    return out.reset_index(
        drop=True
    )

def build_validation_execution_registry(
    model_catalog: pd.DataFrame,
    selected_models: pd.DataFrame,
    selected_runs: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    projection_eligibility: pd.DataFrame,
    model_reference: pd.DataFrame,
    *,
    track_contracts: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical 04C execution registry without new identifiers.

    The scientific identity remains ``model_id``. One execution is identified
    by ``(model_id, random_state)`` and reuses the 03B ``fit_id`` and
    ``projection_id``. 03C contributes only track-level eligibility and 04A
    contributes only the downstream support/diagnostic status; neither stage
    is allowed to add or remove a selected 03B model.
    """
    # Reuse the strict 03B normalization/threshold checks first.  The compact
    # execution table returned here is intentionally rebuilt from selected_runs
    # + model_catalog because the 03C registry omits fit_id and fit parameters.
    _, thresholds = build_selected_execution_registry(
        model_catalog=model_catalog,
        selected_models=selected_models,
        selected_runs=selected_runs,
        selected_thresholds=selected_thresholds,
        track_contracts=track_contracts,
    )

    required_catalog = set(expcfg.INTERNAL_CALIBRATION_MODEL_CATALOG_COLUMNS)
    required_models = set(expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS)
    required_runs = set(expcfg.INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS)
    required_eligibility = {
        "track_id",
        "eligibility_status",
    }
    required_reference = set(expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS)

    for frame, required, name in (
        (model_catalog, required_catalog, "model_catalog"),
        (selected_models, required_models, "selected_models"),
        (selected_runs, required_runs, "selected_runs"),
        (projection_eligibility, required_eligibility, "projection_eligibility"),
        (model_reference, required_reference, "model_reference"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{name} is missing columns: {missing}")

    if projection_eligibility.empty:
        raise RuntimeError("projection_eligibility is empty.")
    eligibility = projection_eligibility[
        ["track_id", "eligibility_status"]
    ].drop_duplicates()
    eligibility["track_id"] = eligibility["track_id"].astype(str)
    if eligibility["track_id"].duplicated().any():
        raise RuntimeError("03C eligibility must contain one row per track_id.")

    expected_track_ids = set(map(str, expcfg.SIMCA_EVALUATION_TRACK_IDS.values()))
    observed_eligibility_tracks = set(eligibility["track_id"])
    if observed_eligibility_tracks != expected_track_ids:
        missing = sorted(expected_track_ids - observed_eligibility_tracks)
        extra = sorted(observed_eligibility_tracks - expected_track_ids)
        raise RuntimeError(
            "03C eligibility must cover exactly E1-E8: "
            f"missing={missing}, extra={extra}."
        )

    models = selected_models.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_MODEL_COLUMNS)
    ].copy()
    models["model_id"] = models["model_id"].astype(str)
    if models.empty or not models["selection_status"].astype(str).eq("selected").all():
        raise RuntimeError("selected_models must contain selected rows only.")
    if models["model_id"].duplicated().any():
        raise RuntimeError("selected_models.model_id must be unique.")

    runs = selected_runs.loc[
        :, list(expcfg.INTERNAL_CALIBRATION_SELECTED_RUN_COLUMNS)
    ].copy()
    runs["model_id"] = runs["model_id"].astype(str)
    runs["random_state"] = pd.to_numeric(
        runs["random_state"], errors="raise"
    ).astype(int)
    run_keys = ["model_id", "random_state"]
    if runs.empty or runs.duplicated(run_keys).any():
        raise RuntimeError("Selected (model_id, random_state) keys must be unique.")
    if runs[["fit_id", "projection_id"]].isna().any().any():
        raise RuntimeError("Every selected execution needs fit_id and projection_id.")

    catalog_columns = [
        "model_id",
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
    ]
    catalog = model_catalog[catalog_columns].copy()
    catalog["model_id"] = catalog["model_id"].astype(str)
    catalog["track_id"] = catalog["track_id"].astype(str)
    if catalog["model_id"].duplicated().any():
        raise RuntimeError("model_catalog.model_id must be unique.")

    selected_ids = set(models["model_id"])
    run_ids = set(runs["model_id"])
    if run_ids != selected_ids:
        raise RuntimeError(
            "selected_runs must cover every selected model, and no other model."
        )
    if not selected_ids.issubset(set(catalog["model_id"])):
        raise RuntimeError("A selected model is absent from model_catalog.")

    reference = model_reference.loc[
        :, list(expcfg.SIMCA_GRID_MODEL_REFERENCE_COLUMNS)
    ].copy()
    reference["model_id"] = reference["model_id"].astype(str)
    reference["track_id"] = reference["track_id"].astype(str)
    if reference.empty or reference["model_id"].duplicated().any():
        raise RuntimeError("04A model_reference must contain one row per model_id.")
    if set(reference["model_id"]) != selected_ids:
        missing = sorted(selected_ids - set(reference["model_id"]))
        extra = sorted(set(reference["model_id"]) - selected_ids)
        raise RuntimeError(
            "04A changed the 03B selected-model universe: "
            f"missing={missing}, extra={extra}."
        )

    expected_run_counts = runs.groupby("model_id", sort=False).size().rename(
        "expected_n_selected_runs"
    )
    checked_counts = reference.merge(
        expected_run_counts,
        left_on="model_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    observed_counts = pd.to_numeric(
        checked_counts["n_selected_runs"], errors="raise"
    ).astype(int)
    if not observed_counts.eq(checked_counts["expected_n_selected_runs"].astype(int)).all():
        raise RuntimeError("04A n_selected_runs disagrees with selected_runs.")

    catalog_track_check = reference[["model_id", "track_id"]].merge(
        catalog[["model_id", "track_id"]],
        on="model_id",
        how="left",
        suffixes=("_04a", "_03b"),
        validate="one_to_one",
    )
    if not catalog_track_check["track_id_04a"].eq(
        catalog_track_check["track_id_03b"]
    ).all():
        raise RuntimeError("04A track_id disagrees with the 03B model catalog.")

    # 03C is track-level. 04A repeats that status per model. Verify equality
    # before keeping one canonical eligibility_status column.
    reference = reference.merge(
        eligibility.rename(columns={"eligibility_status": "eligibility_status_03c"}),
        on="track_id",
        how="left",
        validate="many_to_one",
    )
    if reference["eligibility_status_03c"].isna().any():
        raise RuntimeError("A selected model has no 03C eligibility status.")
    if not reference["eligibility_status"].astype(str).eq(
        reference["eligibility_status_03c"].astype(str)
    ).all():
        raise RuntimeError("04A and 03C eligibility statuses disagree.")

    expected_scope_counts = (
        thresholds.groupby("model_id", sort=False)["decision_scope"]
        .nunique()
        .rename("expected_n_decision_scopes")
    )
    checked_scope_counts = reference.merge(
        expected_scope_counts,
        left_on="model_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    observed_scope_counts = pd.to_numeric(
        checked_scope_counts["n_decision_scopes"], errors="raise"
    ).astype(int)
    if not observed_scope_counts.eq(
        checked_scope_counts["expected_n_decision_scopes"].astype(int)
    ).all():
        raise RuntimeError("04A n_decision_scopes disagrees with selected_thresholds.")

    allowed_downstream = {"supported", "diagnostic_only"}
    unknown_downstream = sorted(
        set(reference["downstream_status"].astype(str)) - allowed_downstream
    )
    if unknown_downstream:
        raise RuntimeError(
            f"Unknown 04A downstream statuses: {unknown_downstream}"
        )
    expected_downstream = np.where(
        reference["eligibility_status_03c"].astype(str).isin(
            expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
        ),
        "supported",
        "diagnostic_only",
    )
    if not reference["downstream_status"].astype(str).eq(
        pd.Series(expected_downstream, index=reference.index)
    ).all():
        raise RuntimeError(
            "04A downstream_status is inconsistent with 03C eligibility."
        )

    executions = runs.merge(
        catalog,
        on="model_id",
        how="left",
        validate="many_to_one",
    ).merge(
        reference[
            [
                "model_id",
                "track_id",
                "eligibility_status_03c",
                "downstream_status",
            ]
        ].rename(columns={"eligibility_status_03c": "eligibility_status"}),
        on=["model_id", "track_id"],
        how="left",
        validate="many_to_one",
    )

    if executions.duplicated(run_keys).any():
        raise RuntimeError("04C execution natural keys are duplicated.")
    if executions[["fit_id", "projection_id"]].astype(str).eq("").any().any():
        raise RuntimeError("04C fit_id and projection_id must be non-empty.")

    balanced = executions["matrix_method"].astype(str).eq("balanced_pixels")
    balanced_m = pd.to_numeric(
        executions.loc[balanced, "m"], errors="coerce"
    )
    if balanced_m.isna().any():
        raise RuntimeError("balanced_pixels executions require a finite m value.")
    if not executions.loc[balanced, "balanced_pixel_strategy"].astype(str).isin(
        expcfg.INTERNAL_CALIBRATION_PIXEL_STRATEGIES
    ).all():
        raise RuntimeError(
            "balanced_pixels executions contain an unknown sampling strategy."
        )
    if not executions.loc[~balanced, "balanced_pixel_strategy"].astype(str).eq(
        "not_applicable"
    ).all():
        raise RuntimeError(
            "Non-balanced executions must use balanced_pixel_strategy='not_applicable'."
        )

    required_complete = [
        column
        for column in expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS
        if column != "m"  # m is intentionally NA for non-balanced matrices.
    ]
    if executions[required_complete].isna().any().any():
        bad_columns = executions[required_complete].columns[
            executions[required_complete].isna().any()
        ].tolist()
        raise RuntimeError(
            f"04C execution metadata is incomplete: {bad_columns}"
        )

    observed_execution_tracks = set(executions["track_id"].astype(str))
    if observed_execution_tracks != expected_track_ids:
        missing = sorted(expected_track_ids - observed_execution_tracks)
        extra = sorted(observed_execution_tracks - expected_track_ids)
        raise RuntimeError(
            "Selected 03B executions must cover exactly E1-E8: "
            f"missing={missing}, extra={extra}."
        )

    executions = executions.reindex(
        columns=expcfg.SIMCA_VALIDATION_EXECUTION_COLUMNS
    ).sort_values(run_keys, kind="mergesort").reset_index(drop=True)

    thresholds = thresholds.sort_values(
        [*run_keys, "decision_scope"], kind="mergesort"
    ).reset_index(drop=True)
    return executions, thresholds