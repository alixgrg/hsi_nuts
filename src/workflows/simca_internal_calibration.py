"""Strict out-of-fold SIMCA calibration on pure batches 1-2.

Notebook 03B uses this module to keep every learned operation inside the
training part of each outer fold. Batch 3 and batch 4 are not accepted by the
workflow.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import uuid

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import experiment_config as expcfg
from src.decision.aggregation import aggregate_pixel_predictions_to_objects
from src.decision.labels import (
    pixel_ratio_col,
    predicted_col,
    true_col,
)
from src.decision.metrics import (
    binary_detection_metrics,
    coerce_binary_series,
    summarize_binary_metrics_vectorized,
)
from src.matrices.matrix_registry import build_matrix, matrix_family_from_method
from src.models.simca import SIMCAClassModel
from src.models.simca_rules import compute_rule_variant_stat_limit
from src.spectra.preprocessing import SpectralPreprocessor
from src.workflows.protocol_split import build_grouped_folds
from src.workflows.simca import (
    fit_simca_bundle_from_matrix,
    prepare_simca_projection,
    project_simca_bundle,
    valid_sg_parameter_pairs,
)
from src.workflows.simca_candidates import (
    build_pca_preprocessing_configs_by_matrix_family,
)
from src.workflows.projection_domain_audit import summarize_projection_shift
from src.workflows.simca_calibration_registry import (
    build_internal_calibration_candidate_runs,
)
from src.workflows.simca_thresholds_calibration import (
    evaluate_calibration_thresholds,
)
from src.utils import save_parquet


_INTERNAL_CALIBRATION_RUNNER_CONTRACT = (
    "8tracks_v5_compact_crossfit_shared_projection"
)

INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS = (
    "matrix_family",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "sg_window_length",
    "sg_polyorder",
    "random_state",
)


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [_json_scalar(item) for item in value]
    if isinstance(value, list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def hash_internal_calibration_configuration(
    configuration: Mapping[str, Any],
    *,
    exclude: Sequence[str] = (),
    prefix: str = "ic",
) -> str:
    """Return a stable short identifier for one calibration configuration."""
    excluded = set(map(str, exclude))
    payload = {
        str(key): _json_scalar(value)
        for key, value in sorted(configuration.items())
        if str(key) not in excluded
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"{prefix}_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _rule_family(rule_variant: str) -> str:
    token = str(rule_variant)
    if token.startswith("simple_"):
        return "simple"
    if token.startswith("alternative_"):
        return "alternative"
    if token.startswith("combined_index_"):
        return "combined_index"
    if token.startswith("data_driven_"):
        return "data_driven"
    raise ValueError(f"Unknown internal-calibration rule variant: {rule_variant!r}")


def _limit_source(rule_variant: str) -> str:
    return (
        "calibration_train_only"
        if str(rule_variant).endswith("_emp_cv")
        else "theoretical_train_fit"
    )


def _with_schema(
    df: pd.DataFrame | None,
    columns: Sequence[str],
    *,
    copy: bool = True,
) -> pd.DataFrame:
    expected = list(columns)
    if df is not None and list(df.columns) == expected:
        return df.copy() if copy else df
    out = pd.DataFrame() if df is None else df.copy(deep=copy)
    for column in columns:
        if column not in out:
            out[column] = pd.Series(pd.NA, index=out.index)
    return out.loc[:, expected]


# def _internal_calibration_run_signature(
#     configurations: pd.DataFrame,
#     folds: pd.DataFrame,
#     *,
#     wavelengths: np.ndarray | None,
#     calibration_batches: Sequence[int],
#     forbidden_batches: Sequence[int],
#     target_class: str,
#     non_target_label: str,
#     reference_object_threshold: float,
#     under_m_policy: str,
#     keep_oof_pixels: bool,
#     keep_oof_objects: bool,
#     error_granularity: str = "scope",
# ) -> str:
#     """Fingerprint the exact grid/folds contract used by checkpoint shards."""
#     fold_columns = [
#         column
#         for column in (
#             "source_image",
#             "object_id",
#             "class_name",
#             "batch",
#             "fold_id",
#         )
#         if column in folds
#     ]
#     fold_records = (
#         folds.loc[:, fold_columns]
#         .sort_values(fold_columns, kind="mergesort")
#         .to_dict("records")
#     )
#     wavelength_values = (
#         []
#         if wavelengths is None
#         else np.asarray(wavelengths, dtype=float).round(12).tolist()
#     )
#     payload = {
#         "checkpoint_format_version": 3,
#         "config_ids": sorted(configurations["config_id"].astype(str).tolist()),
#         "folds": fold_records,
#         "wavelengths": wavelength_values,
#         "calibration_batches": list(map(int, calibration_batches)),
#         "forbidden_batches": list(map(int, forbidden_batches)),
#         "target_class": str(target_class),
#         "non_target_label": str(non_target_label),
#         "reference_object_threshold": float(reference_object_threshold),
#         "under_m_policy": str(under_m_policy),
#         "keep_oof_pixels": bool(keep_oof_pixels),
#         "keep_oof_objects": bool(keep_oof_objects),
#     }
#     if str(error_granularity) != "scope":
#         payload["error_granularity"] = str(error_granularity)
#     canonical = json.dumps(
#         _json_scalar(payload),
#         sort_keys=True,
#         separators=(",", ":"),
#         ensure_ascii=True,
#     )
#     return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not repeat the destination filename in the temporary name.  Marker
    # paths already include a run signature and a data-configuration id; the
    # former scheme could exceed the legacy Windows MAX_PATH limit even when
    # the final marker path itself was valid.
    temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.tmp")
    temporary.write_text(
        json.dumps(
            _json_scalar(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary filename deliberately short: checkpoint directories
    # already contain a content signature and Windows still commonly enforces
    # the legacy MAX_PATH limit for Python file handles.
    temporary = path.with_name(f".{uuid.uuid4().hex[:12]}.parquet")
    save_parquet(df, temporary)
    temporary.replace(path)


def _streaming_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_8track_checkpoint_shard(
    path: Path,
    shard: Mapping[str, Any],
    schema: Sequence[str],
) -> None:
    """Validate one Parquet shard from metadata without materializing rows."""
    if not path.exists():
        raise FileNotFoundError(path)
    if _streaming_sha256(path) != str(shard["file_sha256"]):
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {path}")
    parquet = pq.ParquetFile(path)
    expected_columns = list(map(str, shard["columns"]))
    if int(parquet.metadata.num_rows) != int(shard["row_count"]):
        raise RuntimeError(f"Checkpoint row-count mismatch: {path}")
    if list(map(str, parquet.schema_arrow.names)) != expected_columns:
        raise RuntimeError(f"Checkpoint schema mismatch: {path}")
    if expected_columns != list(map(str, schema)):
        raise RuntimeError(f"Checkpoint configured schema mismatch: {path}")


def build_reference_object_table(
    object_db: Mapping[str, Mapping[str, Any]],
    *,
    allowed_object_ids: Sequence[str] | None = None,
    batches: Sequence[int] = expcfg.INTERNAL_CALIBRATION_BATCHES,
    classes: Sequence[str] = expcfg.REFERENCE_CLASSES,
) -> pd.DataFrame:
    """Build the one-row-per-object table used to create calibration folds."""
    allowed_batches = set(map(int, batches))
    allowed_classes = set(map(str, classes))
    allowed_ids = (
        None
        if allowed_object_ids is None
        else set(map(str, allowed_object_ids))
    )
    rows = []
    for object_id, obj in object_db.items():
        if allowed_ids is not None and str(object_id) not in allowed_ids:
            continue
        batch = obj.get("batch")
        class_name = obj.get("object_nut_type")
        if obj.get("sample_kind") != "pure":
            continue
        if batch is None or int(batch) not in allowed_batches:
            continue
        if class_name is None or str(class_name) not in allowed_classes:
            continue
        rows.append(
            {
                "source_image": str(
                    obj.get("source_clean_key", obj.get("source_image", ""))
                ),
                "object_id": str(object_id),
                "class_name": str(class_name),
                "batch": int(batch),
                "object_area": float(
                    obj.get(
                        "area_pixels",
                        obj.get("n_pixels", len(obj.get("spectra", ()))),
                    )
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No pure calibration objects were found in batches 1-2.")
    if out["source_image"].eq("").any():
        raise ValueError("Every calibration object must have a source image.")
    if not out["object_id"].is_unique:
        raise ValueError("Calibration object_id values must be unique.")
    if allowed_ids is not None:
        unauthorized = set(out["object_id"].astype(str)) - allowed_ids
        if unauthorized:
            raise RuntimeError(
                "Objects outside the QC calibration manifest were selected: "
                f"{sorted(unauthorized)[:10]}"
            )
    return out.sort_values(["source_image", "object_id"]).reset_index(drop=True)


def _size_bins(values: pd.Series, n_bins: int = 3) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Object sizes must be finite to construct folds.")
    n_unique = int(numeric.nunique())
    if n_unique <= 1:
        return pd.Series(0, index=values.index, dtype=int)
    q = min(int(n_bins), n_unique, len(values))
    ranked = numeric.rank(method="first")
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype(int)


def _fold_diagnostics(
    folds: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    batch_col: str,
    object_size_col: str,
    target_class: str,
    non_target_label: str,
) -> pd.DataFrame:
    rows = []
    for fold_id, group in folds.groupby("fold_id", sort=True):
        sizes = pd.to_numeric(group[object_size_col], errors="coerce")
        labels = group[label_col].astype(str)
        batches = pd.to_numeric(group[batch_col], errors="coerce")
        rows.append(
            {
                "fold_id": int(fold_id),
                "n_groups": int(group[group_col].nunique()),
                "n_objects": int(group["object_id"].nunique()),
                "n_target_objects": int(labels.eq(str(target_class)).sum()),
                "n_non_target_objects": int(
                    labels.eq(str(non_target_label)).sum()
                ),
                "n_batch_1_objects": int(batches.eq(1).sum()),
                "n_batch_2_objects": int(batches.eq(2).sum()),
                "median_object_size": float(sizes.median()),
                "coverage_complete": bool(
                    labels.eq(str(target_class)).any()
                    and labels.eq(str(non_target_label)).any()
                    and batches.eq(1).any()
                    and batches.eq(2).any()
                ),
            }
        )
    return _with_schema(
        pd.DataFrame(rows),
        expcfg.INTERNAL_CALIBRATION_FOLD_DIAGNOSTIC_COLUMNS,
    )


def build_calibration_folds(
    reference_df: pd.DataFrame,
    *,
    group_col: str = expcfg.INTERNAL_CALIBRATION_GROUP_COL,
    label_col: str = expcfg.INTERNAL_CALIBRATION_LABEL_COL,
    batch_col: str = expcfg.INTERNAL_CALIBRATION_BATCH_COL,
    object_size_col: str = expcfg.INTERNAL_CALIBRATION_OBJECT_SIZE_COL,
    n_splits: int = expcfg.INTERNAL_CALIBRATION_N_SPLITS,
    n_size_bins: int = expcfg.INTERNAL_CALIBRATION_SIZE_N_BINS,
    random_state: int = expcfg.INTERNAL_CALIBRATION_FOLD_RANDOM_STATE,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    require_complete_coverage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign whole images to balanced folds using class, batch and size."""
    required = (
        group_col,
        "object_id",
        label_col,
        batch_col,
        object_size_col,
    )
    missing = [column for column in required if column not in reference_df]
    if missing:
        raise KeyError(f"Missing fold-construction columns: {missing}")

    df = reference_df.loc[:, list(required)].copy()
    df[group_col] = df[group_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    df[batch_col] = pd.to_numeric(df[batch_col], errors="raise").astype(int)
    df[object_size_col] = pd.to_numeric(
        df[object_size_col],
        errors="raise",
    ).astype(float)

    n_groups = int(df[group_col].nunique())
    if int(n_splits) < 2 or n_groups < int(n_splits):
        raise ValueError(
            f"Need at least n_splits={n_splits} distinct groups, got {n_groups}."
        )

    common_assignments, _ = build_grouped_folds(
        df,
        group_col=group_col,
        label_col=label_col,
        batch_col=batch_col,
        size_col=object_size_col,
        n_size_bins=int(n_size_bins),
        n_splits=int(n_splits),
        random_state=int(random_state),
        require_complete_coverage=require_complete_coverage,
    )
    group_assignments = common_assignments[
        [group_col, "fold_id"]
    ].drop_duplicates(group_col)
    group_sizes = df.groupby(group_col, as_index=False).agg(
        median_object_size=(object_size_col, "median")
    )
    group_sizes["size_bin"] = _size_bins(
        group_sizes["median_object_size"],
        n_bins=int(n_size_bins),
    )
    folds = df.merge(
        group_assignments.merge(
            group_sizes[[group_col, "size_bin"]],
            on=group_col,
            how="left",
            validate="one_to_one",
        ),
        on=group_col,
        how="left",
        validate="many_to_one",
    )
    if folds["fold_id"].isna().any():
        raise RuntimeError("At least one calibration group has no fold.")
    folds["fold_id"] = folds["fold_id"].astype(int)
    if folds.groupby(group_col)["fold_id"].nunique().max() != 1:
        raise RuntimeError("A calibration group occurs in several folds.")
    if not folds["object_id"].is_unique:
        raise RuntimeError("Each calibration object must occur exactly once.")

    diagnostics = _fold_diagnostics(
        folds,
        group_col=group_col,
        label_col=label_col,
        batch_col=batch_col,
        object_size_col=object_size_col,
        target_class=target_class,
        non_target_label=non_target_label,
    )
    if require_complete_coverage:
        failed = diagnostics.loc[
            ~diagnostics["coverage_complete"].astype(bool),
            ["fold_id", "coverage_complete"],
        ]
        if len(failed):
            raise RuntimeError(
                "Fold coverage is incomplete; class/batch preservation failed: "
                f"{failed.to_dict('records')}"
            )

    rename_map = {
        group_col: "source_image",
        label_col: "class_name",
        batch_col: "batch",
        object_size_col: "object_area",
    }
    folds = folds.rename(columns=rename_map)
    return (
        _with_schema(
            folds,
            expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS,
        ).sort_values(["fold_id", "source_image", "object_id"]).reset_index(
            drop=True
        ),
        diagnostics,
    )


def build_internal_calibration_configurations(
    pca_selected_preprocessings_df: pd.DataFrame,
    *,
    matrix_methods: Sequence[str] = expcfg.INTERNAL_CALIBRATION_MATRIX_METHODS,
    m_values: Sequence[int] = expcfg.INTERNAL_CALIBRATION_M_VALUES,
    pixel_strategies: Sequence[str] = expcfg.INTERNAL_CALIBRATION_PIXEL_STRATEGIES,
    n_components_values: Sequence[int] = expcfg.INTERNAL_CALIBRATION_N_COMPONENTS_VALUES,
    rule_variants: Sequence[str] = expcfg.INTERNAL_CALIBRATION_RULE_VARIANTS,
    alpha_values: Sequence[float] = expcfg.INTERNAL_CALIBRATION_ALPHA_VALUES,
    sg_windows: Sequence[int] = expcfg.INTERNAL_CALIBRATION_SG_WINDOWS,
    sg_polyorders: Sequence[int] = expcfg.INTERNAL_CALIBRATION_SG_POLYORDERS,
    dilation_radii: Sequence[int] = expcfg.INTERNAL_CALIBRATION_DILATION_RADII,
    random_seeds: Sequence[int] = expcfg.INTERNAL_CALIBRATION_RANDOM_SEEDS,
    minimum_pixels_per_object: int | None = None,
    max_configs: int | None = expcfg.INTERNAL_CALIBRATION_MAX_CONFIGS,
) -> pd.DataFrame:
    """Expand the family-specific PCA preprocessing set into the 03B grid."""
    preprocessing_by_family = (
        build_pca_preprocessing_configs_by_matrix_family(
            pca_selected_preprocessings_df
        )
    )
    rows = []
    default_seed = int(tuple(random_seeds)[0])

    # ``under_m_policy='exclude'`` makes m a matrix-level eligibility rule:
    # small objects are omitted from the corresponding training matrix but
    # remain present in every OOF projection.  Do not shrink the protocol grid
    # according to the single smallest object.
    feasible_m_values = tuple(map(int, m_values))
    del minimum_pixels_per_object

    for matrix_method in map(str, matrix_methods):
        family = matrix_family_from_method(matrix_method)
        preprocessing_configs = preprocessing_by_family.get(family, {})
        if not preprocessing_configs:
            raise RuntimeError(
                f"No PCA preprocessing remains for matrix family {family!r}."
            )
        sampling = (
            [
                (int(m), str(strategy))
                for m in feasible_m_values
                for strategy in pixel_strategies
            ]
            if matrix_method == "balanced_pixels"
            else [(np.nan, "not_applicable")]
        )
        for preprocessing, steps in preprocessing_configs.items():
            steps = tuple(map(str, steps))
            sg_pairs = valid_sg_parameter_pairs(
                preprocessing_steps=steps,
                sg_window_length_values=sg_windows,
                sg_polyorder_values=sg_polyorders,
                default_sg_window_length=expcfg.SG_DEFAULT_WINDOW,
                default_sg_polyorder=expcfg.SG_POLYORDER,
            )
            for m, strategy in sampling:
                seeds = (
                    tuple(map(int, random_seeds))
                    if matrix_method == "balanced_pixels" and strategy == "random"
                    else (default_seed,)
                )
                for seed in seeds:
                    for sg_window, sg_polyorder in sg_pairs:
                        data_base = {
                            "matrix_family": family,
                            "matrix_method": matrix_method,
                            "m": m,
                            "balanced_pixel_strategy": strategy,
                            "preprocessing": str(preprocessing),
                            "preprocessing_steps": "+".join(steps),
                            "sg_window_length": int(sg_window),
                            "sg_polyorder": int(sg_polyorder),
                            "random_state": int(seed),
                        }
                        data_config_id = (
                            hash_internal_calibration_configuration(
                                data_base,
                                prefix="icdata",
                            )
                        )
                        for n_components in n_components_values:
                            for alpha in alpha_values:
                                for dilation_radius in dilation_radii:
                                    base = {
                                        **data_base,
                                        "data_config_id": data_config_id,
                                        "n_components": int(n_components),
                                        "alpha": float(alpha),
                                        "position_dilation_radius": int(
                                            dilation_radius
                                        ),
                                    }
                                    fit_id = (
                                        hash_internal_calibration_configuration(
                                            base,
                                            # Pure-reference truth is invariant
                                            # to dilation, so all radii reuse
                                            # the same fitted train-fold model.
                                            exclude=(
                                                "position_dilation_radius",
                                            ),
                                            prefix="icfit",
                                        )
                                    )
                                    for variant in map(str, rule_variants):
                                        row = {
                                            **base,
                                            "fit_id": fit_id,
                                            "rule_family": _rule_family(variant),
                                            "rule_variant": variant,
                                            "limit_source": _limit_source(variant),
                                        }
                                        row["config_id"] = (
                                            hash_internal_calibration_configuration(
                                                row,
                                                prefix="ic",
                                            )
                                        )
                                        rows.append(row)

    out = pd.DataFrame(rows).drop_duplicates("config_id").reset_index(drop=True)
    if max_configs is not None:
        out = out.head(int(max_configs)).copy()
    if out.empty:
        raise RuntimeError("The internal-calibration configuration grid is empty.")
    return out


def _internal_calibration_checkpoint_schemas() -> dict[str, Sequence[str]]:
    return {
        "fit_diagnostics": (
            expcfg.INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_COLUMNS
        ),
        "rule_diagnostics": (
            expcfg.INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS
        ),
        "projection_shift": (
            expcfg.INTERNAL_CALIBRATION_PROJECTION_SHIFT_COLUMNS
        ),
        "oof_object_predictions": (
            expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_COLUMNS
        ),
        "oof_pixel_predictions": (
            expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS
        ),
        "threshold_metrics": (
            expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
        ),
        "technical_events": (
            expcfg.INTERNAL_CALIBRATION_TECHNICAL_EVENT_COLUMNS
        ),
    }


def attach_internal_calibration_runner_group_ids(
    configurations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the private checkpoint runner-group identity deterministically."""
    missing = sorted(
        set(INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS)
        - set(configurations.columns)
    )
    if missing:
        raise KeyError(f"Missing runner-group columns: {missing}")

    out = configurations.copy()
    out["_runner_group_id"] = [
        hash_internal_calibration_configuration(
            {
                column: row[column]
                for column in INTERNAL_CALIBRATION_RUNNER_GROUP_COLUMNS
            },
            prefix="runner",
        )
        for row in out.to_dict("records")
    ]
    return out


def expand_projection_configurations(
    fit_configurations: pd.DataFrame,
    track_contracts: pd.DataFrame,
) -> pd.DataFrame:
    """Expand reusable fits into projection and evaluation identities."""
    required_contract = {
        "track_id",
        "evaluation_track",
        "training_matrix_family",
        "projection_level",
        "projection_matrix_policy",
        "allowed_projection_methods_json",
        "decision_mode",
        "decision_score_type",
    }
    missing_contract = sorted(required_contract - set(track_contracts.columns))
    if missing_contract:
        raise KeyError(f"Missing track-contract columns: {missing_contract}")
    if len(track_contracts) != 8 or set(track_contracts["track_id"]) != {
        f"E{index}" for index in range(1, 9)
    }:
        raise RuntimeError("track_contracts must contain exactly E1-E8.")
    required_fit = {
        "fit_id",
        "matrix_family",
        "matrix_method",
        "rule_variant",
        "limit_source",
    }
    missing_fit = sorted(required_fit - set(fit_configurations.columns))
    if missing_fit:
        raise KeyError(f"Missing fit-configuration columns: {missing_fit}")
    rows: list[dict[str, Any]] = []
    for fit_row in fit_configurations.to_dict("records"):
        matching_tracks = track_contracts.loc[
            track_contracts["training_matrix_family"].astype(str).eq(
                str(fit_row["matrix_family"])
            )
        ]
        for contract in matching_tracks.to_dict("records"):
            allowed = tuple(
                map(str, json.loads(contract["allowed_projection_methods_json"]))
            )
            if contract["projection_matrix_policy"] == (
                "match_object_training_method"
            ):
                methods = (str(fit_row["matrix_method"]),)
                if methods[0] not in allowed:
                    raise RuntimeError(
                        "Object projection cannot match training method "
                        f"{methods[0]!r}."
                    )
            else:
                methods = allowed
            for projection_method in methods:
                projection_payload = {
                    "fit_id": str(fit_row["fit_id"]),
                    "projection_level": str(contract["projection_level"]),
                    "projection_matrix_method": str(projection_method),
                    "rule_variant": str(fit_row["rule_variant"]),
                    "limit_source": str(fit_row["limit_source"]),
                    "protocol_version": str(expcfg.PROTOCOL_VERSION),
                    "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
                }
                projection_id = hash_internal_calibration_configuration(
                    projection_payload,
                    prefix="icproj",
                )
                evaluation_payload = {
                    **projection_payload,
                    "projection_id": projection_id,
                    "evaluation_track": str(contract["evaluation_track"]),
                    "decision_mode": str(contract["decision_mode"]),
                    "decision_score_type": str(
                        contract["decision_score_type"]
                    ),
                }
                model_id = hash_internal_calibration_configuration(
                    evaluation_payload,
                    prefix="iceval",
                )
                rows.append(
                    {
                        **fit_row,
                        "source_config_id": str(
                            fit_row.get("config_id", fit_row["fit_id"])
                        ),
                        "projection_id": projection_id,
                        "model_id": model_id,
                        "evaluation_track": str(contract["evaluation_track"]),
                        "track_id": str(contract["track_id"]),
                        "parent_track": str(contract.get("parent_track", "")),
                        "projection_level": str(contract["projection_level"]),
                        "projection_matrix_method": str(projection_method),
                        "decision_mode": str(contract["decision_mode"]),
                        "decision_score_type": str(
                            contract["decision_score_type"]
                        ),
                        "protocol_version": str(expcfg.PROTOCOL_VERSION),
                        "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
                    }
                )
    expanded = pd.DataFrame(rows).drop_duplicates(
        "model_id"
    ).reset_index(drop=True)
    observed = set(expanded["evaluation_track"])
    expected = set(track_contracts["evaluation_track"])
    if observed != expected:
        raise RuntimeError(
            "Projection expansion failed to cover all tracks: "
            f"missing={sorted(expected - observed)}"
        )
    if not expanded["model_id"].is_unique:
        raise RuntimeError("model_id must be unique.")
    return expanded


def validate_simca_configuration(
    configuration: Mapping[str, Any],
    X_train: np.ndarray | None = None,
    y_train: Sequence[Any] = (),
    n_target_observations: int | None = None,
    n_features: int | None = None,
    numeric_rank: int | None = None,
    n_pixels_by_object: Sequence[int] | None = None,
    available_classes: Sequence[str] | None = None,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
) -> dict[str, Any]:
    """Return homogeneous static and fold-specific technical diagnostics."""
    errors: list[dict[str, str]] = []

    def add_error(code: str, parameter: str, message: str) -> None:
        if any(item["code"] == code for item in errors):
            return
        errors.append(
            {
                "code": str(code),
                "message": str(message),
                "parameter": str(parameter),
            }
        )

    X = None if X_train is None else np.asarray(X_train)
    k_value = configuration.get("n_components")
    k = int(k_value) if k_value is not None and not pd.isna(k_value) else 0
    m_value = configuration.get("m")
    steps = str(configuration.get("preprocessing_steps", "")).split("+")
    window = configuration.get("sg_window_length")
    polyorder = configuration.get("sg_polyorder")
    dilation = configuration.get("position_dilation_radius")

    if k <= 0:
        add_error(
            "N_COMPONENTS_NOT_POSITIVE",
            "n_components",
            "n_components must be strictly positive.",
        )
    if X is not None:
        if X.ndim != 2 or X.shape[0] == 0:
            add_error(
                "EMPTY_TRAINING_MATRIX",
                "X_train",
                "The training matrix is empty or not two-dimensional.",
            )
        elif X.shape[1] == 0:
            add_error(
                "EMPTY_FEATURE_MATRIX",
                "X_train",
                "The training matrix contains no feature.",
            )
        elif not np.isfinite(X).all():
            add_error(
                "NON_FINITE_TRAINING_MATRIX",
                "X_train",
                "The training matrix contains non-finite values.",
            )

    class_values = (
        available_classes
        if available_classes is not None
        else y_train
    )
    classes = set(map(str, class_values))
    if classes:
        if str(target_class) not in classes:
            add_error(
                "MISSING_TARGET_CLASS",
                "available_classes",
                f"Missing target class {target_class!r}.",
            )
        if str(non_target_label) not in classes:
            add_error(
                "MISSING_NON_TARGET_CLASS",
                "available_classes",
                f"Missing non-target class {non_target_label!r}.",
            )
    if n_features is not None and k > int(n_features):
        add_error(
            "N_COMPONENTS_EXCEED_DIMENSION",
            "n_components",
            "n_components exceeds the spectral dimension.",
        )
    if (
        n_target_observations is not None
        and k >= int(n_target_observations)
    ):
        add_error(
            "N_COMPONENTS_EXCEED_TARGET_N",
            "n_components",
            "n_components must be smaller than target observations.",
        )
    if numeric_rank is not None and k > int(numeric_rank):
        add_error(
            "N_COMPONENTS_EXCEED_RANK",
            "n_components",
            "n_components exceeds the numerical rank.",
        )

    if any(step.startswith("sg_") for step in steps):
        if window is None or int(window) <= 0 or int(window) % 2 == 0:
            add_error(
                "SG_WINDOW_NOT_ODD",
                "sg_window_length",
                "The SG window must be a strictly positive odd integer.",
            )
        if (
            window is not None
            and polyorder is not None
            and int(window) <= int(polyorder)
        ):
            add_error(
                "SG_WINDOW_NOT_GREATER_THAN_POLYORDER",
                "sg_window_length",
                "The SG window must exceed the polynomial order.",
            )
        if (
            window is not None
            and n_features is not None
            and int(window) > int(n_features)
        ):
            add_error(
                "SG_WINDOW_EXCEEDS_SPECTRUM",
                "sg_window_length",
                "The SG window exceeds the spectral dimension.",
            )

    if (
        m_value is not None
        and not pd.isna(m_value)
        and str(configuration.get("matrix_method")) == "balanced_pixels"
    ):
        m = int(m_value)
        if m <= 0:
            add_error(
                "M_NOT_POSITIVE",
                "m",
                "m must be strictly positive for balanced pixels.",
            )
        if n_pixels_by_object is not None:
            pixels = np.asarray(n_pixels_by_object, dtype=float)
            pixels = pixels[np.isfinite(pixels)]
            if pixels.size == 0 or m > int(np.min(pixels)):
                add_error(
                    "M_EXCEEDS_AVAILABLE_PIXELS",
                    "m",
                    "m exceeds the available pixels in at least one object.",
                )

    if dilation is not None and int(dilation) < 0:
        add_error(
            "NEGATIVE_DILATION_RADIUS",
            "position_dilation_radius",
            "The dilation radius cannot be negative.",
        )

    decision_mode = str(configuration.get("decision_mode", "")).lower()
    object_threshold = configuration.get("object_threshold")
    lower_threshold = configuration.get("three_way_lower_threshold")
    upper_threshold = configuration.get("three_way_upper_threshold")
    if decision_mode == "2way":
        if (
            object_threshold is None
            or pd.isna(object_threshold)
            or not 0.0 <= float(object_threshold) <= 1.0
        ):
            add_error(
                "INVALID_OBJECT_THRESHOLD",
                "object_threshold",
                "A 2-way threshold must be finite and within [0, 1].",
            )
    elif decision_mode == "3way":
        if (
            lower_threshold is None
            or upper_threshold is None
            or pd.isna(lower_threshold)
            or pd.isna(upper_threshold)
            or not (
                0.0
                <= float(lower_threshold)
                < float(upper_threshold)
                <= 1.0
            )
        ):
            add_error(
                "INVALID_THREE_WAY_THRESHOLDS",
                "three_way_lower_threshold",
                "3-way thresholds must satisfy 0 <= lower < upper <= 1.",
            )

    return {
        "valid": len(errors) == 0,
        "is_valid": len(errors) == 0,
        "technical_errors": tuple(errors),
        "technical_error_codes": tuple(item["code"] for item in errors),
        "n_target_observations": (
            None
            if n_target_observations is None
            else int(n_target_observations)
        ),
        "n_features": None if n_features is None else int(n_features),
        "numeric_rank": None if numeric_rank is None else int(numeric_rank),
    }


def _empirical_quantile(values: Sequence[float], q: float) -> float:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return np.nan
    try:
        return float(np.quantile(numeric, q, method="higher"))
    except TypeError:
        return float(np.quantile(numeric, q, interpolation="higher"))


def compute_train_only_rule_thresholds(
    model: SIMCAClassModel,
    *,
    alpha: float,
) -> dict[str, float]:
    """Compute every empirical rule limit from the current fold's train only."""
    H = np.asarray(model.H_train_, dtype=float)
    Q = np.asarray(model.Q_train_, dtype=float)
    eps = float(model.eps)
    simple = np.maximum(H / model.H_limit_, Q / model.Q_limit_)
    alternative = H / model.H_limit_ + Q / model.Q_limit_
    data_driven = (
        model.NQ_ * Q / max(model.Q0_, eps)
        + model.NH_ * H / max(model.H0_, eps)
    )
    q = 1.0 - float(alpha)
    H_empirical = _empirical_quantile(H, q)
    Q_empirical = _empirical_quantile(Q, q)
    alternative_empirical_hq = (
        H / max(H_empirical, float(model.eps))
        + Q / max(Q_empirical, float(model.eps))
    )
    return {
        "alpha": float(alpha),
        "quantile": float(q),
        "H_emp_cv": H_empirical,
        "Q_emp_cv": Q_empirical,
        "simple_emp_cv": _empirical_quantile(simple, q),
        "alternative_chi2_emp_cv": _empirical_quantile(alternative, q),
        "alternative_empHQ_emp_cv": _empirical_quantile(
            alternative_empirical_hq,
            q,
        ),
        "combined_index_emp_cv": _empirical_quantile(alternative, q),
        "data_driven_emp_cv": _empirical_quantile(data_driven, q),
    }


def _record_group_filter(group_col: str) -> str:
    return "source_clean_key" if str(group_col) == "source_image" else str(group_col)


def _base_filters(
    *,
    batches: Sequence[int],
    classes: Sequence[str],
    groups: Sequence[str],
    group_col: str,
) -> dict[str, Any]:
    return {
        "sample_kind": ["pure"],
        "batch": list(map(int, batches)),
        "object_nut_type": list(map(str, classes)),
        _record_group_filter(group_col): list(map(str, groups)),
    }


def _training_pixels_by_object(
    object_db: Mapping[str, Mapping[str, Any]],
    filters: Mapping[str, Any],
) -> np.ndarray:
    group_field = next(
        (
            field
            for field in ("source_clean_key", "object_id")
            if field in filters
        ),
        None,
    )
    groups = set(map(str, filters.get(group_field, ()))) if group_field else None
    batches = set(map(int, filters.get("batch", ())))
    classes = set(map(str, filters.get("object_nut_type", ())))
    values = []
    for object_id, obj in object_db.items():
        group_value = object_id if group_field == "object_id" else obj.get(group_field)
        if groups is not None and str(group_value) not in groups:
            continue
        if batches and int(obj.get("batch", -1)) not in batches:
            continue
        if classes and str(obj.get("object_nut_type")) not in classes:
            continue
        values.append(int(obj.get("n_pixels", len(obj.get("spectra", ())))))
    return np.asarray(values, dtype=int)


def _sampling_minhash(
    metadata: Mapping[str, Sequence[Any]],
    *,
    n_hashes: int = 8,
) -> str:
    """Return a compact MinHash sketch of selected training pixels."""
    required = ("object_id", "row", "col")
    if any(column not in metadata for column in required):
        return ""
    pixel_ids = {
        f"{object_id}|{row}|{col}"
        for object_id, row, col in zip(
            metadata["object_id"],
            metadata["row"],
            metadata["col"],
        )
    }
    if not pixel_ids:
        return ""
    minima = []
    for salt in range(int(n_hashes)):
        minima.append(
            min(
                int.from_bytes(
                    hashlib.sha256(
                        f"{salt}|{pixel_id}".encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                for pixel_id in pixel_ids
            )
        )
    return ".".join(f"{value:016x}" for value in minima)


def run_internal_calibration_8tracks(
    *,
    object_db: Mapping[str, Mapping[str, Any]],
    folds: pd.DataFrame,
    configurations: pd.DataFrame,
    wavelengths: np.ndarray | None = None,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    under_m_policy: str = expcfg.INTERNAL_CALIBRATION_UNDER_M_POLICY,
    verbose: bool = expcfg.INTERNAL_CALIBRATION_VERBOSE,
    checkpoint_dir: str | Path | None = None,
    checkpoint_context: Mapping[str, str] | None = None,
    resume_from_checkpoint: bool = (
        expcfg.INTERNAL_CALIBRATION_RESUME_FROM_CHECKPOINT
    ),
    keep_oof_in_memory: bool = False,
    keep_threshold_metrics_in_memory: bool = True,
) -> dict[str, Any]:
    """Run shared OOF fits/projections for the eight SIMCA tracks.

    Scientific models are identified by ``model_id``. Random seeds represent
    repeated executions and therefore remain outside that identity.

    Candidate OOF predictions are checkpointed by a private runner-group key.
    Only compact diagnostics are always retained in memory. Threshold metrics
    can also remain checkpoint-only by setting
    ``keep_threshold_metrics_in_memory=False``; this is required for the full
    03B grid, whose long threshold table does not fit safely in pandas.
    """
    required_configuration_columns = {
        "model_id",
        "fit_id",
        "projection_id",
        "random_state",
        "evaluation_track",
        "track_id",
        "decision_mode",
        "matrix_family",
        "matrix_method",
        "projection_level",
        "projection_matrix_method",
        "m",
        "balanced_pixel_strategy",
        "preprocessing",
        "preprocessing_steps",
        "rule_variant",
        "limit_source",
        "n_components",
        "alpha",
        "sg_window_length",
        "sg_polyorder",
    }
    missing = sorted(
        required_configuration_columns - set(configurations.columns)
    )
    if missing:
        raise KeyError(
            f"Missing 8-track execution columns: {missing}"
        )
    if configurations.empty:
        raise ValueError("The 8-track execution table is empty.")
    if not keep_oof_in_memory and checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir is required when OOF predictions are not "
            "retained in memory."
        )

    required_fold_columns = set(expcfg.INTERNAL_CALIBRATION_FOLD_COLUMNS)
    missing_fold_columns = sorted(
        required_fold_columns - set(folds.columns)
    )
    if missing_fold_columns:
        raise KeyError(
            f"Missing calibration-fold columns: {missing_fold_columns}"
        )
    if folds.empty:
        raise ValueError("The calibration-fold table is empty.")
    if not folds["object_id"].astype(str).is_unique:
        raise RuntimeError(
            "Each calibration object must occur in exactly one fold."
        )

    observed_batches = set(
        pd.to_numeric(folds["batch"], errors="raise").astype(int)
    )
    allowed_batches = set(
        map(int, expcfg.INTERNAL_CALIBRATION_BATCHES)
    )
    forbidden_batches = set(
        map(int, expcfg.INTERNAL_CALIBRATION_FORBIDDEN_BATCHES)
    )
    if observed_batches - allowed_batches:
        raise RuntimeError(
            "Only the configured calibration batches may enter 03B."
        )
    if observed_batches & forbidden_batches:
        raise RuntimeError(
            "A forbidden validation/test batch entered notebook 03B."
        )

    work = configurations.copy().reset_index(drop=True)

    if work.duplicated(["model_id", "random_state"]).any():
        raise RuntimeError(
            "(model_id, random_state) must uniquely identify one execution."
        )

    projection_identity = (
        work.groupby("projection_id", dropna=False)[
            [
                "fit_id",
                "projection_level",
                "projection_matrix_method",
                "rule_variant",
                "limit_source",
                "random_state",
            ]
        ]
        .nunique(dropna=False)
        .max(axis=1)
    )
    if projection_identity.gt(1).any():
        raise RuntimeError(
            "A projection_id maps to multiple technical projections."
        )

    work = attach_internal_calibration_runner_group_ids(work)

    fold_table = folds.copy()
    fold_table["_object_key"] = (
        fold_table["object_id"].astype(str)
    )
    fold_metadata = fold_table.set_index("_object_key")[
        ["object_area", "size_bin"]
    ]

    missing_database_objects = (
        set(fold_table["_object_key"])
        - set(map(str, object_db))
    )
    if missing_database_objects:
        raise RuntimeError(
            "Calibration objects are absent from object_db: "
            f"{sorted(missing_database_objects)[:10]}"
        )

    schemas = _internal_calibration_checkpoint_schemas()
    compact_names = {
        "fit_diagnostics",
        "rule_diagnostics",
        "projection_shift",
        "threshold_metrics",
        "technical_events",
    }
    retained_compact_names = compact_names - (
        set()
        if keep_threshold_metrics_in_memory
        else {"threshold_metrics"}
    )
    oof_names = {
        "oof_object_predictions",
        "oof_pixel_predictions",
    }

    retained_parts: dict[str, list[pd.DataFrame]] = {
        name: [] for name in schemas
    }

    def add_technical_event(
        parts: dict[str, list[pd.DataFrame]],
        *,
        fit_id: str = "",
        projection_id: str = "",
        fold_id: int,
        stage: str,
        status: str,
        reason_code: str,
        n_initial: int | None = None,
        n_valid: int | None = None,
        n_filtered: int | None = None,
        exc: Exception | None = None,
    ) -> None:
        parts["technical_events"].append(
            pd.DataFrame(
                [
                    {
                        "fit_id": str(fit_id),
                        "projection_id": str(projection_id),
                        "fold_id": int(fold_id),
                        "stage": str(stage),
                        "status": str(status),
                        "reason_code": str(reason_code),
                        "n_initial": n_initial,
                        "n_valid": n_valid,
                        "n_filtered": n_filtered,
                        "error_type": (
                            "" if exc is None else type(exc).__name__
                        ),
                        "error_message": (
                            "" if exc is None else str(exc)
                        ),
                    }
                ]
            )
        )

    def compact_projection(
        projected: pd.DataFrame,
        *,
        projection_id: str,
        fold_id: int,
    ) -> pd.DataFrame:
        out = projected.copy()
        out.attrs = {}

        object_ids = out["object_id"].astype(str)
        out["object_area"] = object_ids.map(
            fold_metadata["object_area"]
        )
        out["size_bin"] = object_ids.map(
            fold_metadata["size_bin"]
        )
        if out[["object_area", "size_bin"]].isna().any(axis=None):
            raise RuntimeError(
                "An OOF projection contains an object outside the fold table."
            )

        out["projection_id"] = str(projection_id)
        out["fold_id"] = int(fold_id)
        return out

    context = {
        str(key): str(value)
        for key, value in dict(checkpoint_context or {}).items()
    }
    required_context = {
        "protocol_hash",
        "pca_selection_fingerprint",
        "track_contract_hash",
        "fold_contract_hash",
        "configuration_hash",
    }

    checkpoint_run_dir: Path | None = None
    checkpoint_signature = ""

    if checkpoint_dir is not None:
        missing_context = sorted(
            key for key in required_context if not context.get(key)
        )
        if missing_context:
            raise ValueError(
                f"Incomplete checkpoint context: {missing_context}"
            )

        signature_payload = {
            **context,
            "runner_contract": _INTERNAL_CALIBRATION_RUNNER_CONTRACT,
            "protocol_version": str(expcfg.PROTOCOL_VERSION),
            "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
            "model_ids": sorted(
                work["model_id"].astype(str).unique()
            ),
            "fit_ids": sorted(
                work["fit_id"].astype(str).unique()
            ),
            "projection_ids": sorted(
                work["projection_id"].astype(str).unique()
            ),
            "fold_ids": sorted(
                map(
                    int,
                    pd.to_numeric(folds["fold_id"]).unique(),
                )
            ),
            "under_m_policy": str(under_m_policy),
        }
        checkpoint_signature = hashlib.sha256(
            json.dumps(
                _json_scalar(signature_payload),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        checkpoint_run_dir = (
            Path(checkpoint_dir)
            / f"run_{checkpoint_signature[:20]}"
        )
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)

        _atomic_write_json(
            {
                "signature": checkpoint_signature,
                **signature_payload,
                "expected_runner_group_ids": sorted(
                    work["_runner_group_id"].astype(str).unique()
                ),
            },
            checkpoint_run_dir / "manifest.json",
        )

    runner_groups = list(
        work.groupby("_runner_group_id", sort=False)
    )
    fold_ids = sorted(
        map(int, pd.to_numeric(folds["fold_id"]).unique())
    )

    for group_index, (runner_group_id, group_rows) in enumerate(
        runner_groups,
        start=1,
    ):
        runner_group_id = str(runner_group_id)
        marker_path = (
            None
            if checkpoint_run_dir is None
            else checkpoint_run_dir
            / "markers"
            / f"{runner_group_id}.json"
        )

        if marker_path is not None and marker_path.exists():
            if not resume_from_checkpoint:
                raise RuntimeError(
                    "A compatible checkpoint exists but resume is disabled."
                )

            marker = json.loads(
                marker_path.read_text(encoding="utf-8")
            )
            if marker.get("signature") != checkpoint_signature:
                raise RuntimeError(
                    f"Checkpoint signature mismatch: {marker_path}"
                )

            for shard in marker.get("shards", ()):
                name = str(shard["name"])
                if name not in schemas:
                    raise RuntimeError(
                        f"Unknown checkpoint shard type: {name}"
                    )
                path = checkpoint_run_dir / shard["relative_path"]
                _validate_8track_checkpoint_shard(
                    path,
                    shard,
                    schemas[name],
                )
                if name in retained_compact_names or keep_oof_in_memory:
                    retained_parts[name].append(
                        _with_schema(
                            pd.read_parquet(path),
                            schemas[name],
                            copy=False,
                        )
                    )
            continue

        if verbose:
            base = group_rows.iloc[0]
            print(
                f"[03B {group_index}/{len(runner_groups)}] "
                f"{base['matrix_method']} | "
                f"{base['preprocessing']} | "
                f"m={base['m']} | "
                f"seed={base['random_state']}"
            )

        local_parts: dict[str, list[pd.DataFrame]] = {
            name: [] for name in schemas
        }

        for fold_id in fold_ids:
            valid_ids = set(
                fold_table.loc[
                    fold_table["fold_id"].astype(int).eq(fold_id),
                    "_object_key",
                ]
            )
            train_ids = set(fold_table["_object_key"]) - valid_ids

            if train_ids & valid_ids:
                raise RuntimeError(
                    "An object is shared between train and OOF validation."
                )

            train_db = {
                object_id: object_db[object_id]
                for object_id in sorted(train_ids)
            }
            valid_db = {
                object_id: object_db[object_id]
                for object_id in sorted(valid_ids)
            }

            train_filters = {
                "sample_kind": ["pure"],
                "object_nut_type": [str(target_class)],
            }
            projection_filters = {
                "sample_kind": ["pure"],
                "object_nut_type": [
                    str(target_class),
                    str(non_target_label),
                ],
            }

            base = group_rows.iloc[0]
            m_value = (
                int(base["m"])
                if not pd.isna(base["m"])
                else int(expcfg.PCA_BALANCED_M_VALUES[0])
            )
            strategy = str(base["balanced_pixel_strategy"])
            if strategy == "not_applicable":
                strategy = "random"

            matrix_start = perf_counter()
            try:
                X_train_raw, y_train, train_metadata = build_matrix(
                    object_db=train_db,
                    matrix_method=str(base["matrix_method"]),
                    filters=train_filters,
                    m=m_value,
                    random_state=int(base["random_state"]),
                    replace=False,
                    balanced_pixel_strategy=strategy,
                    under_m_policy=str(under_m_policy),
                )
                matrix_seconds = perf_counter() - matrix_start

                projection_data = {
                    method: build_matrix(
                        object_db=valid_db,
                        matrix_method=str(method),
                        filters=projection_filters,
                        under_m_policy=str(under_m_policy),
                    )
                    for method in sorted(
                        group_rows[
                            "projection_matrix_method"
                        ].astype(str).unique()
                    )
                }
            except Exception as exc:
                for fit_id in sorted(
                    group_rows["fit_id"].astype(str).unique()
                ):
                    add_technical_event(
                        local_parts,
                        fit_id=fit_id,
                        fold_id=fold_id,
                        stage="matrix_build",
                        status="error",
                        reason_code="MATRIX_BUILD_FAILED",
                        exc=exc,
                    )
                continue

            fitted_preprocessor = None
            X_train_preprocessed = None

            fit_groups = sorted(
                group_rows.groupby("fit_id", sort=False),
                key=lambda item: int(
                    item[1].iloc[0]["n_components"]
                ),
            )

            for fit_id, fit_rows in fit_groups:
                fit_id = str(fit_id)
                fit_base = fit_rows.iloc[0]

                try:
                    bundle = fit_simca_bundle_from_matrix(
                        X_train_raw,
                        y_train,
                        train_metadata,
                        preprocessing_spec={
                            "steps": tuple(
                                str(
                                    fit_base["preprocessing_steps"]
                                ).split("+")
                            ),
                            "sg_window_length": int(
                                fit_base["sg_window_length"]
                            ),
                            "sg_polyorder": int(
                                fit_base["sg_polyorder"]
                            ),
                        },
                        n_components=int(
                            fit_base["n_components"]
                        ),
                        alpha=float(fit_base["alpha"]),
                        wavelengths=wavelengths,
                        target_class=str(target_class),
                        fitted_preprocessor=fitted_preprocessor,
                        X_train_preprocessed=X_train_preprocessed,
                    )

                    if fitted_preprocessor is None:
                        fitted_preprocessor = bundle.preprocessor
                        X_train_preprocessed = bundle.X_train

                    local_parts["fit_diagnostics"].append(
                        pd.DataFrame(
                            [
                                {
                                    "fit_id": fit_id,
                                    "fold_id": fold_id,
                                    "raw_rank": bundle.raw_rank,
                                    "preprocessed_rank": (
                                        bundle.preprocessed_rank
                                    ),
                                    "n_train_target": len(
                                        bundle.X_train
                                    ),
                                    "n_features": (
                                        bundle.X_train.shape[1]
                                    ),
                                    "n_components": int(
                                        fit_base["n_components"]
                                    ),
                                    "matrix_build_seconds": (
                                        matrix_seconds
                                    ),
                                    "preprocessing_seconds": (
                                        bundle.preprocessing_seconds
                                    ),
                                    "fit_seconds": (
                                        bundle.fit_seconds
                                    ),
                                    "status": "ok",
                                    "error_code": "",
                                    "error_message": "",
                                }
                            ]
                        )
                    )

                    train_only_thresholds = (
                        compute_train_only_rule_thresholds(
                            bundle.model,
                            alpha=float(fit_base["alpha"]),
                        )
                    )
                except Exception as exc:
                    local_parts["fit_diagnostics"].append(
                        pd.DataFrame(
                            [
                                {
                                    "fit_id": fit_id,
                                    "fold_id": fold_id,
                                    "n_components": int(
                                        fit_base["n_components"]
                                    ),
                                    "matrix_build_seconds": (
                                        matrix_seconds
                                    ),
                                    "status": "error",
                                    "error_code": "FIT_FAILED",
                                    "error_message": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                }
                            ]
                        )
                    )
                    add_technical_event(
                        local_parts,
                        fit_id=fit_id,
                        fold_id=fold_id,
                        stage="fit",
                        status="error",
                        reason_code="FIT_FAILED",
                        exc=exc,
                    )
                    continue

                projection_caches: dict[str, dict[str, Any]] = {}
                projection_cache_errors: dict[str, Exception] = {}

                for method in sorted(
                    fit_rows[
                        "projection_matrix_method"
                    ].astype(str).unique()
                ):
                    try:
                        cache = prepare_simca_projection(
                            bundle,
                            object_db=valid_db,
                            projection_matrix_method=method,
                            projection_filters=projection_filters,
                            projection_data=projection_data[method],
                            under_m_policy=str(under_m_policy),
                        )
                        projection_caches[method] = cache

                        validity = dict(
                            cache.get("input_validity", {})
                        )
                        n_filtered = int(
                            validity.get("n_filtered_rows", 0)
                        )
                        if n_filtered:
                            affected = fit_rows.loc[
                                fit_rows[
                                    "projection_matrix_method"
                                ].astype(str).eq(method),
                                "projection_id",
                            ].astype(str).unique()
                            for projection_id in affected:
                                add_technical_event(
                                    local_parts,
                                    fit_id=fit_id,
                                    projection_id=projection_id,
                                    fold_id=fold_id,
                                    stage="projection_input",
                                    status="filtered",
                                    reason_code=(
                                        "INVALID_PREPROCESSING_INPUT"
                                    ),
                                    n_initial=int(
                                        validity.get(
                                            "n_input_rows", 0
                                        )
                                    ),
                                    n_valid=int(
                                        validity.get(
                                            "n_valid_rows", 0
                                        )
                                    ),
                                    n_filtered=n_filtered,
                                )
                    except Exception as exc:
                        projection_cache_errors[method] = exc

                for projection_id, projection_rows in fit_rows.groupby(
                    "projection_id",
                    sort=False,
                ):
                    projection_id = str(projection_id)
                    projection_base = projection_rows.iloc[0]
                    method = str(
                        projection_base[
                            "projection_matrix_method"
                        ]
                    )

                    if method in projection_cache_errors:
                        exc = projection_cache_errors[method]
                        local_parts["rule_diagnostics"].append(
                            pd.DataFrame(
                                [
                                    {
                                        "projection_id": projection_id,
                                        "fold_id": fold_id,
                                        "status": "error",
                                        "error_code": (
                                            "PROJECTION_PREPARATION_FAILED"
                                        ),
                                    }
                                ]
                            )
                        )
                        add_technical_event(
                            local_parts,
                            fit_id=fit_id,
                            projection_id=projection_id,
                            fold_id=fold_id,
                            stage="projection_preparation",
                            status="error",
                            reason_code=(
                                "PROJECTION_PREPARATION_FAILED"
                            ),
                            exc=exc,
                        )
                        continue

                    try:
                        projected = project_simca_bundle(
                            bundle,
                            object_db=valid_db,
                            projection_matrix_method=method,
                            projection_filters=projection_filters,
                            projection_cache=projection_caches[method],
                            rule_variant=str(
                                projection_base["rule_variant"]
                            ),
                            train_only_thresholds=(
                                train_only_thresholds
                            ),
                            target_class=str(target_class),
                            under_m_policy=str(under_m_policy),
                        )
                        compact = compact_projection(
                            projected,
                            projection_id=projection_id,
                            fold_id=fold_id,
                        )

                        output_name = (
                            "oof_pixel_predictions"
                            if str(
                                projection_base["projection_level"]
                            )
                            == "pixel_projection"
                            else "oof_object_predictions"
                        )
                        local_parts[output_name].append(compact)

                        train_stat, train_limit = (
                            compute_rule_variant_stat_limit(
                                H=bundle.model.H_train_,
                                Q=bundle.model.Q_train_,
                                model=bundle.model,
                                variant_name=str(
                                    projection_base[
                                        "rule_variant"
                                    ]
                                ),
                                cv_thresholds=(
                                    train_only_thresholds
                                ),
                            )
                        )
                        train_stat = np.asarray(
                            train_stat,
                            dtype=float,
                        )
                        train_limit = float(train_limit)

                        if (
                            not np.isfinite(train_limit)
                            or train_limit <= 0.0
                        ):
                            raise RuntimeError(
                                "The train-only SIMCA limit is invalid."
                            )

                        target_projection = compact.loc[
                            compact["truth"].astype(bool)
                        ]
                        local_parts["rule_diagnostics"].append(
                            pd.DataFrame(
                                [
                                    {
                                        "projection_id": projection_id,
                                        "fold_id": fold_id,
                                        "rule_limit": train_limit,
                                        "q_limit": float(
                                            bundle.model.Q_limit_
                                        ),
                                        "t2_limit": float(
                                            bundle.model.H_limit_
                                        ),
                                        "train_rejection_rate": float(
                                            np.mean(
                                                train_stat
                                                / train_limit
                                                >= 1.0
                                            )
                                        ),
                                        "oof_target_rejection_rate": (
                                            float(
                                                target_projection[
                                                    "simca_margin"
                                                ].lt(0.0).mean()
                                            )
                                            if len(target_projection)
                                            else np.nan
                                        ),
                                        "status": "ok",
                                        "error_code": "",
                                    }
                                ]
                            )
                        )

                        train_scores = bundle.train_scores.assign(
                            fold_id=fold_id,
                            rule_limit=train_limit,
                            normalized_ratio=(
                                train_stat / train_limit
                            ),
                            simca_margin=(
                                1.0 - train_stat / train_limit
                            ),
                        )
                        shift_input = compact.assign(
                            projection_level=str(
                                projection_base[
                                    "projection_level"
                                ]
                            ),
                            projection_matrix_method=method,
                        )
                        shift = summarize_projection_shift(
                            train_scores,
                            shift_input,
                            group_keys=(
                                "projection_id",
                                "projection_level",
                                "projection_matrix_method",
                                "fold_id",
                            ),
                        )
                        local_parts["projection_shift"].append(
                            shift
                        )
                    except Exception as exc:
                        local_parts["rule_diagnostics"].append(
                            pd.DataFrame(
                                [
                                    {
                                        "projection_id": projection_id,
                                        "fold_id": fold_id,
                                        "status": "error",
                                        "error_code": (
                                            "PROJECTION_RULE_FAILED"
                                        ),
                                    }
                                ]
                            )
                        )
                        add_technical_event(
                            local_parts,
                            fit_id=fit_id,
                            projection_id=projection_id,
                            fold_id=fold_id,
                            stage="projection_rule",
                            status="error",
                            reason_code="PROJECTION_RULE_FAILED",
                            exc=exc,
                        )

        local_tables = {
            name: _with_schema(
                (
                    pd.concat(
                        parts,
                        ignore_index=True,
                        sort=False,
                    )
                    if parts
                    else None
                ),
                schema,
                copy=False,
            )
            for name, schema in schemas.items()
            for parts in (local_parts[name],)
        }

        threshold_metrics = evaluate_calibration_thresholds(
            local_tables["oof_object_predictions"],
            local_tables["oof_pixel_predictions"],
            group_rows,
        )
        local_tables["threshold_metrics"] = _with_schema(
            threshold_metrics,
            schemas["threshold_metrics"],
            copy=False,
        )

        shard_rows: list[dict[str, Any]] = []
        completed_fit_ids = sorted(
            group_rows["fit_id"].astype(str).unique()
        )

        if checkpoint_run_dir is not None:
            for name, table in local_tables.items():
                if table.empty:
                    continue

                relative_path = (
                    Path("chunks")
                    / f"{runner_group_id}_{name}.parquet"
                )
                absolute_path = (
                    checkpoint_run_dir / relative_path
                )
                _atomic_save_parquet(table, absolute_path)

                shard_rows.append(
                    {
                        "name": name,
                        "relative_path": relative_path.as_posix(),
                        "schema_version": str(
                            expcfg.RESULTS_SCHEMA_VERSION
                        ),
                        "protocol_version": str(
                            expcfg.PROTOCOL_VERSION
                        ),
                        "row_count": int(len(table)),
                        "columns": list(map(str, table.columns)),
                        "file_sha256": _streaming_sha256(
                            absolute_path
                        ),
                        "completed_fit_ids": completed_fit_ids,
                    }
                )

            _atomic_write_json(
                {
                    "signature": checkpoint_signature,
                    "runner_group_id": runner_group_id,
                    **context,
                    "completed_fit_ids": completed_fit_ids,
                    "shards": shard_rows,
                },
                marker_path,
            )

        for name in retained_compact_names:
            if not local_tables[name].empty:
                retained_parts[name].append(
                    local_tables[name]
                )

        if keep_oof_in_memory:
            for name in oof_names:
                if not local_tables[name].empty:
                    retained_parts[name].append(
                        local_tables[name]
                    )

        del local_parts
        del local_tables
        gc.collect()

    if checkpoint_run_dir is not None:
        markers = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(
                (checkpoint_run_dir / "markers").glob("*.json")
            )
        ]
        expected_groups = set(
            work["_runner_group_id"].astype(str)
        )
        observed_groups = {
            str(marker["runner_group_id"])
            for marker in markers
        }
        if observed_groups != expected_groups:
            raise RuntimeError(
                "Incomplete checkpoint runner-group coverage: "
                f"missing={sorted(expected_groups - observed_groups)}"
            )

        expected_fit_ids = set(work["fit_id"].astype(str))
        observed_fit_ids = {
            str(fit_id)
            for marker in markers
            for fit_id in marker.get("completed_fit_ids", ())
        }
        if observed_fit_ids != expected_fit_ids:
            raise RuntimeError(
                "Incomplete checkpoint fit coverage: "
                f"missing={sorted(expected_fit_ids - observed_fit_ids)}"
            )

    results = {
        name: _with_schema(
            (
                pd.concat(
                    retained_parts[name],
                    ignore_index=True,
                    sort=False,
                )
                if retained_parts[name]
                else None
            ),
            schema,
        )
        for name, schema in schemas.items()
    }
    results["checkpoint_run_dir"] = checkpoint_run_dir
    return results


def iter_internal_calibration_checkpoint_shards_8tracks(
    checkpoint_run_dir: str | Path,
    table_name: str,
) -> Iterator[tuple[str, Path]]:
    """Yield validated checkpoint shards for one table without loading rows."""
    schemas = _internal_calibration_checkpoint_schemas()
    name = str(table_name)
    if name not in schemas:
        raise KeyError(f"Unknown internal-calibration checkpoint table: {name}")

    run_dir = Path(checkpoint_run_dir)
    manifest_path = run_dir / "manifest.json"
    marker_dir = run_dir / "markers"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not marker_dir.exists():
        raise FileNotFoundError(marker_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = str(manifest.get("signature", ""))
    if not signature:
        raise RuntimeError("Checkpoint run manifest has no signature.")

    markers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(marker_dir.glob("*.json"))
    ]
    expected_groups = set(
        map(str, manifest.get("expected_runner_group_ids", ()))
    )
    observed_groups = {
        str(marker.get("runner_group_id", "")) for marker in markers
    }
    if observed_groups != expected_groups:
        raise RuntimeError(
            "Incomplete checkpoint runner-group coverage: "
            f"missing={sorted(expected_groups - observed_groups)}, "
            f"unexpected={sorted(observed_groups - expected_groups)}"
        )

    for marker in markers:
        runner_group_id = str(marker.get("runner_group_id", ""))
        if str(marker.get("signature", "")) != signature:
            raise RuntimeError(
                f"Checkpoint signature mismatch for {runner_group_id}."
            )
        matches = [
            shard for shard in marker.get("shards", ())
            if str(shard.get("name", "")) == name
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"Duplicate {name} shards for {runner_group_id}."
            )
        if not matches:
            continue

        shard = matches[0]
        path = run_dir / str(shard["relative_path"])
        _validate_8track_checkpoint_shard(
            path,
            shard,
            schemas[name],
        )
        yield runner_group_id, path


def load_selected_oof_predictions_from_checkpoint_8tracks(
    checkpoint_run_dir: str | Path,
    selected_execution_domain: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only OOF projections required by selected model runs."""
    required = (
        "projection_id",
        "fit_id",
        "projection_level",
    )
    missing = sorted(
        set(required) - set(selected_execution_domain.columns)
    )
    if missing:
        raise KeyError(
            f"Missing selected-execution columns: {missing}"
        )
    if selected_execution_domain.empty:
        raise RuntimeError("The selected execution domain is empty.")

    selected = selected_execution_domain[
        list(required)
    ].drop_duplicates()
    level_counts = selected.groupby(
        "projection_id",
        dropna=False,
    )["projection_level"].nunique(dropna=False)
    if level_counts.gt(1).any():
        raise RuntimeError(
            "A projection_id maps to multiple projection levels."
        )

    level_by_projection = selected.set_index("projection_id")[
        "projection_level"
    ].astype(str)
    selected_ids = {
        "oof_object_predictions": set(
            level_by_projection.loc[
                level_by_projection.eq("object_projection")
            ].index.astype(str)
        ),
        "oof_pixel_predictions": set(
            level_by_projection.loc[
                level_by_projection.eq("pixel_projection")
            ].index.astype(str)
        ),
    }
    selected_fit_ids = set(selected["fit_id"].astype(str))
    schemas = {
        "oof_object_predictions": (
            expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_COLUMNS
        ),
        "oof_pixel_predictions": (
            expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS
        ),
    }
    parts = {name: [] for name in schemas}
    run_dir = Path(checkpoint_run_dir)
    marker_dir = run_dir / "markers"
    if not marker_dir.exists():
        raise FileNotFoundError(marker_dir)

    for marker_path in sorted(marker_dir.glob("*.json")):
        marker = json.loads(
            marker_path.read_text(encoding="utf-8")
        )
        marker_fit_ids = set(
            map(str, marker.get("completed_fit_ids", ()))
        )
        if marker_fit_ids.isdisjoint(selected_fit_ids):
            continue

        for shard in marker.get("shards", ()):
            name = str(shard["name"])
            wanted = selected_ids.get(name, set())
            if not wanted:
                continue

            path = run_dir / str(shard["relative_path"])
            _validate_8track_checkpoint_shard(
                path,
                shard,
                schemas[name],
            )
            table = pq.read_table(
                path,
                filters=[
                    (
                        "projection_id",
                        "in",
                        sorted(wanted),
                    )
                ],
            )
            if table.num_rows:
                parts[name].append(
                    _with_schema(
                        table.to_pandas(),
                        schemas[name],
                        copy=False,
                    )
                )

    outputs = {
        name: _with_schema(
            (
                pd.concat(
                    parts[name],
                    ignore_index=True,
                    sort=False,
                )
                if parts[name]
                else None
            ),
            schema,
        )
        for name, schema in schemas.items()
    }
    for name, expected_ids in selected_ids.items():
        observed_ids = set(
            outputs[name]["projection_id"].astype(str)
        )
        if observed_ids != expected_ids:
            raise RuntimeError(
                f"Incomplete selected OOF coverage for {name}: "
                f"missing={sorted(expected_ids - observed_ids)}, "
                f"unexpected={sorted(observed_ids - expected_ids)}"
            )

    return (
        outputs["oof_object_predictions"],
        outputs["oof_pixel_predictions"],
    )


def _safe_ratio_array(
    numerator: np.ndarray,
    denominator: int | np.ndarray,
) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    out = np.full(
        np.broadcast_shapes(
            numerator_array.shape,
            denominator_array.shape,
        ),
        np.nan,
        dtype=float,
    )
    np.divide(
        numerator_array,
        denominator_array,
        out=out,
        where=np.broadcast_to(denominator_array, out.shape) > 0,
    )
    return out


def _binary_metric_arrays(
    true_target: np.ndarray,
    predicted_target: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = np.asarray(true_target, dtype=bool)
    predictions = np.asarray(predicted_target, dtype=bool)
    if predictions.ndim != 2:
        raise ValueError("Binary prediction arrays must be two-dimensional.")
    n_thresholds = predictions.shape[1]
    target = truth[:, None]
    non_target = ~target
    tp = np.sum(target & predictions, axis=0, dtype=np.int64)
    fn = np.sum(target & ~predictions, axis=0, dtype=np.int64)
    fp = np.sum(non_target & predictions, axis=0, dtype=np.int64)
    tn = np.sum(non_target & ~predictions, axis=0, dtype=np.int64)
    sensitivity = _safe_ratio_array(tp, tp + fn)
    specificity = _safe_ratio_array(tn, tn + fp)
    balanced_accuracy = np.where(
        np.isfinite(sensitivity) & np.isfinite(specificity),
        0.5 * (sensitivity + specificity),
        np.nan,
    )
    return {
        "n": np.full(n_thresholds, len(truth), dtype=np.int64),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "target_sensitivity": sensitivity,
        "non_target_specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "fn_rate": _safe_ratio_array(fn, tp + fn),
        "fp_rate": _safe_ratio_array(fp, fp + tn),
    }


def _three_way_metric_arrays(
    true_target: np.ndarray,
    ratios: np.ndarray,
    lower_thresholds: np.ndarray,
    upper_thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = np.asarray(true_target, dtype=bool)
    ratio_values = np.asarray(ratios, dtype=float)
    lower = np.asarray(lower_thresholds, dtype=float)
    upper = np.asarray(upper_thresholds, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("Lower and upper threshold arrays must align.")

    target_decision = ratio_values[:, None] >= upper[None, :]
    non_target_decision = ratio_values[:, None] <= lower[None, :]
    uncertain = ~(target_decision | non_target_decision)
    target = truth[:, None]
    non_target = ~target

    n_thresholds = len(lower)
    n_target = int(np.sum(truth))
    n_non_target = int(np.sum(~truth))
    n_uncertain = np.sum(uncertain, axis=0, dtype=np.int64)
    target_missed = np.sum(
        target & non_target_decision,
        axis=0,
        dtype=np.int64,
    )
    false_accept = np.sum(
        non_target & target_decision,
        axis=0,
        dtype=np.int64,
    )
    target_uncertain = np.sum(
        target & uncertain,
        axis=0,
        dtype=np.int64,
    )
    non_target_uncertain = np.sum(
        non_target & uncertain,
        axis=0,
        dtype=np.int64,
    )

    decided_tp = np.sum(
        target & target_decision,
        axis=0,
        dtype=np.int64,
    )
    decided_fn = target_missed
    decided_fp = false_accept
    decided_tn = np.sum(
        non_target & non_target_decision,
        axis=0,
        dtype=np.int64,
    )
    decided_sensitivity = _safe_ratio_array(
        decided_tp,
        decided_tp + decided_fn,
    )
    decided_specificity = _safe_ratio_array(
        decided_tn,
        decided_tn + decided_fp,
    )
    decided_balanced_accuracy = np.where(
        np.isfinite(decided_sensitivity)
        & np.isfinite(decided_specificity),
        0.5 * (decided_sensitivity + decided_specificity),
        np.nan,
    )
    uncertain_rate = _safe_ratio_array(n_uncertain, len(truth))
    return {
        "n": np.full(n_thresholds, len(truth), dtype=np.int64),
        "target_miss_rate": _safe_ratio_array(
            target_missed,
            n_target,
        ),
        "non_target_false_accept_rate": _safe_ratio_array(
            false_accept,
            n_non_target,
        ),
        "uncertain_rate": uncertain_rate,
        "coverage_rate": 1.0 - uncertain_rate,
        "target_uncertain_rate": _safe_ratio_array(
            target_uncertain,
            n_target,
        ),
        "non_target_uncertain_rate": _safe_ratio_array(
            non_target_uncertain,
            n_non_target,
        ),
        "decided_balanced_accuracy": decided_balanced_accuracy,
    }


def _coerced_truth(
    frame: pd.DataFrame,
    *,
    target_class: str,
    non_target_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    truth = coerce_binary_series(
        frame["true_target_object"],
        target_class=target_class,
        non_target_class=non_target_label,
    )
    valid = truth.notna().to_numpy()
    values = np.zeros(len(frame), dtype=bool)
    if valid.any():
        values[valid] = truth.loc[truth.notna()].astype(bool).to_numpy()
    return values, valid


def resolve_internal_calibration_checkpoint_run_8tracks(
    checkpoint_dir: str | Path,
    *,
    checkpoint_context: Mapping[str, str],
    expected_fit_config_ids: Sequence[str],
) -> Path:
    """Locate the single complete compatible 03B checkpoint run."""
    required_signature = {
        "runner_contract": _INTERNAL_CALIBRATION_RUNNER_CONTRACT,
        "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
        "protocol_version": str(expcfg.PROTOCOL_VERSION),
        **{
            str(key): str(value)
            for key, value in dict(checkpoint_context).items()
        },
    }
    required_context = {
        "protocol_hash",
        "pca_selection_fingerprint",
        "track_contract_hash",
        "fold_contract_hash",
        "configuration_hash",
    }
    missing_context = sorted(
        key for key in required_context if not required_signature.get(key)
    )
    if missing_context:
        raise ValueError(
            f"Incomplete checkpoint context: {missing_context}"
        )
    expected_fit_ids = set(map(str, expected_fit_config_ids))
    matching_run_dirs: list[Path] = []
    for run_dir in sorted(Path(checkpoint_dir).glob("run_*")):
        run_manifest_path = run_dir / "manifest.json"
        if not run_manifest_path.exists():
            continue
        run_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        if not all(
            str(run_manifest.get(key)) == str(value)
            for key, value in required_signature.items()
        ):
            continue
        marker_paths = sorted((run_dir / "markers").glob("*.json"))
        observed_fit_ids = {
            str(fit_id)
            for marker_path in marker_paths
            for fit_id in json.loads(
                marker_path.read_text(encoding="utf-8")
            ).get("completed_fit_ids", ())
        }
        if observed_fit_ids == expected_fit_ids:
            matching_run_dirs.append(run_dir)
    if len(matching_run_dirs) != 1:
        raise RuntimeError(
            "Expected exactly one complete compatible 8-track checkpoint "
            f"run, found {len(matching_run_dirs)}."
        )
    return matching_run_dirs[0]


def build_internal_calibration_checkpoint_manifest(
    output_paths: Mapping[str, str | Path],
    *,
    checkpoint_dir: str | Path | None = None,
    protocol_hash: str,
    pca_selection_fingerprint: str,
    track_contract_hash: str,
    fold_contract_hash: str,
    configuration_hash: str,
    completed_fit_config_ids: Sequence[str],
) -> dict[str, Any]:
    """Build an integrity manifest for every canonical 03B table shard."""
    required_signature = {
        "runner_contract": _INTERNAL_CALIBRATION_RUNNER_CONTRACT,
        "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
        "protocol_version": str(expcfg.PROTOCOL_VERSION),
        "protocol_hash": str(protocol_hash),
        "pca_selection_fingerprint": str(pca_selection_fingerprint),
        "track_contract_hash": str(track_contract_hash),
        "fold_contract_hash": str(fold_contract_hash),
        "configuration_hash": str(configuration_hash),
    }
    if any(not value for value in required_signature.values()):
        raise ValueError("Checkpoint signature fields must all be non-empty.")
    completed = sorted(set(map(str, completed_fit_config_ids)))
    shards = []
    for name, raw_path in sorted(output_paths.items()):
        path = Path(raw_path)
        if name == "checkpoint_manifest":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing 03B output shard: {path}")
        if path.suffix.lower() != ".parquet":
            continue
        parquet = pq.ParquetFile(path)
        digest = _streaming_sha256(path)
        shards.append(
            {
                "kind": "canonical_output",
                "name": str(name),
                "path": str(path),
                **required_signature,
                "row_count": int(parquet.metadata.num_rows),
                "columns": list(map(str, parquet.schema_arrow.names)),
                "file_sha256": digest,
                "completed_fit_config_ids": completed,
            }
        )
    if checkpoint_dir is not None:
        run_dir = resolve_internal_calibration_checkpoint_run_8tracks(
            checkpoint_dir,
            checkpoint_context={
                key: required_signature[key]
                for key in (
                    "protocol_hash",
                    "pca_selection_fingerprint",
                    "track_contract_hash",
                    "fold_contract_hash",
                    "configuration_hash",
                )
            },
            expected_fit_config_ids=completed,
        )
        for marker_path in sorted((run_dir / "markers").glob("*.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for shard in marker.get("shards", []):
                path = run_dir / str(shard["relative_path"])
                if not path.exists():
                    raise FileNotFoundError(path)
                shards.append(
                    {
                        "kind": "checkpoint_shard",
                        "name": str(shard["name"]),
                        "data_config_id": str(marker["data_config_id"]),
                        "path": str(path),
                        **required_signature,
                        "row_count": int(shard["row_count"]),
                        "columns": list(map(str, shard["columns"])),
                        "file_sha256": str(shard["file_sha256"]),
                        "completed_fit_config_ids": list(
                            map(str, shard["completed_fit_config_ids"])
                        ),
                    }
                )
    if not shards:
        raise RuntimeError("No Parquet shard was found for the manifest.")
    payload = {**required_signature, "shards": shards}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return payload


def validate_internal_calibration_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_fit_config_ids: Sequence[str],
) -> None:
    """Block incomplete, stale, overlapping or modified 03B shards."""
    required = {
        "runner_contract",
        "schema_version",
        "protocol_version",
        "protocol_hash",
        "pca_selection_fingerprint",
        "track_contract_hash",
        "fold_contract_hash",
        "configuration_hash",
        "shards",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"Incomplete checkpoint manifest: {missing}")
    expected = set(map(str, expected_fit_config_ids))
    observed_fit_ids: set[str] = set()
    covered_by_table: dict[str, set[str]] = {}
    for shard in manifest["shards"]:
        path = Path(shard["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        if _streaming_sha256(path) != shard["file_sha256"]:
            raise RuntimeError(f"Checkpoint SHA256 mismatch: {path}")
        parquet = pq.ParquetFile(path)
        if int(shard["row_count"]) != int(parquet.metadata.num_rows):
            raise RuntimeError(f"Checkpoint row-count mismatch: {path}")
        if list(map(str, parquet.schema_arrow.names)) != list(shard["columns"]):
            raise RuntimeError(f"Checkpoint schema mismatch: {path}")
        shard_fit_ids = set(map(str, shard["completed_fit_config_ids"]))
        if not shard_fit_ids.issubset(expected):
            raise RuntimeError(f"Checkpoint has unexpected fit IDs: {path}")
        observed_fit_ids.update(shard_fit_ids)
        if shard.get("kind") == "checkpoint_shard":
            table_coverage = covered_by_table.setdefault(
                str(shard["name"]), set()
            )
            overlap = table_coverage.intersection(shard_fit_ids)
            if overlap:
                raise RuntimeError(
                    "Checkpoint shards overlap within table "
                    f"{shard['name']}: {sorted(overlap)[:5]}"
                )
            table_coverage.update(shard_fit_ids)
    if observed_fit_ids != expected:
        raise RuntimeError(
            "Checkpoint fit coverage is incomplete: "
            f"missing={sorted(expected - observed_fit_ids)}"
        )
