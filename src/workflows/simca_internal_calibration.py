"""Strict out-of-fold SIMCA calibration on pure batches 1-2.

Notebook 03B uses this module to keep every learned operation inside the
training part of each outer fold. Batch 3 and batch 4 are not accepted by the
workflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from src.matrices.matrix_registry import build_matrix
from src.models.simca import SIMCAClassModel
from src.models.simca_rules import compute_rule_variant_stat_limit
from src.spectra.preprocessing import SpectralPreprocessor
from src.workflows.protocol_split import build_grouped_folds
from src.workflows.simca import (
    fit_simca_bundle_from_matrix,
    matrix_family_from_method,
    prepare_simca_projection,
    project_simca_bundle,
    valid_sg_parameter_pairs,
)
from src.workflows.simca_candidates import (
    build_pca_preprocessing_configs_by_matrix_family,
    selection_track_from_parts,
)
from src.workflows.projection_domain_audit import summarize_projection_shift
from src.workflows.simca_selection_utils import pareto_front_by_group
from src.utils import save_parquet


_INTERNAL_CALIBRATION_RUNNER_CONTRACT = (
    "8tracks_projection_input_filter_pca_shift_v2"
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


_INTERNAL_DATA_CONFIGURATION_COLUMNS = (
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

_INTERNAL_PARAMETER_COLUMNS = (
    "matrix_family",
    "matrix_method",
    "m",
    "balanced_pixel_strategy",
    "preprocessing",
    "preprocessing_steps",
    "rule_family",
    "rule_variant",
    "limit_source",
    "n_components",
    "alpha",
    "sg_window_length",
    "sg_polyorder",
    "position_dilation_radius",
    "random_state",
)

_INTERNAL_RESULT_SCHEMAS = {
    "oof_pixels": expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_COLUMNS,
    "oof_objects": expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_COLUMNS,
    "fold_metrics": expcfg.INTERNAL_CALIBRATION_FOLD_METRIC_COLUMNS,
    "rule_diagnostics": expcfg.INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_COLUMNS,
    "sampling_diagnostics": (
        expcfg.INTERNAL_CALIBRATION_SAMPLING_DIAGNOSTIC_COLUMNS
    ),
    "errors": expcfg.INTERNAL_CALIBRATION_ERROR_COLUMNS,
}


def _attach_data_configuration_ids(configurations: pd.DataFrame) -> pd.DataFrame:
    """Identify rows that share the same raw matrix and fitted preprocessing."""
    out = configurations.copy()
    missing = [
        column
        for column in _INTERNAL_DATA_CONFIGURATION_COLUMNS
        if column not in out
    ]
    if missing:
        raise KeyError(
            "Missing data-configuration columns for internal calibration: "
            f"{missing}"
        )
    out["data_config_id"] = [
        hash_internal_calibration_configuration(
            {
                column: row[column]
                for column in _INTERNAL_DATA_CONFIGURATION_COLUMNS
            },
            prefix="icdata",
        )
        for row in out.to_dict("records")
    ]
    sampling_columns = [
        column
        for column in _INTERNAL_DATA_CONFIGURATION_COLUMNS
        if column != "random_state"
    ]
    out["sampling_group_id"] = [
        hash_internal_calibration_configuration(
            {column: row[column] for column in sampling_columns},
            prefix="icsampling",
        )
        for row in out.to_dict("records")
    ]
    return out


def _internal_calibration_run_signature(
    configurations: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    wavelengths: np.ndarray | None,
    calibration_batches: Sequence[int],
    forbidden_batches: Sequence[int],
    target_class: str,
    non_target_label: str,
    reference_object_threshold: float,
    under_m_policy: str,
    keep_oof_pixels: bool,
    keep_oof_objects: bool,
    error_granularity: str = "scope",
) -> str:
    """Fingerprint the exact grid/folds contract used by checkpoint shards."""
    fold_columns = [
        column
        for column in (
            "source_image",
            "object_id",
            "class_name",
            "batch",
            "fold_id",
        )
        if column in folds
    ]
    fold_records = (
        folds.loc[:, fold_columns]
        .sort_values(fold_columns, kind="mergesort")
        .to_dict("records")
    )
    wavelength_values = (
        []
        if wavelengths is None
        else np.asarray(wavelengths, dtype=float).round(12).tolist()
    )
    payload = {
        "checkpoint_format_version": 3,
        "config_ids": sorted(configurations["config_id"].astype(str).tolist()),
        "folds": fold_records,
        "wavelengths": wavelength_values,
        "calibration_batches": list(map(int, calibration_batches)),
        "forbidden_batches": list(map(int, forbidden_batches)),
        "target_class": str(target_class),
        "non_target_label": str(non_target_label),
        "reference_object_threshold": float(reference_object_threshold),
        "under_m_policy": str(under_m_policy),
        "keep_oof_pixels": bool(keep_oof_pixels),
        "keep_oof_objects": bool(keep_oof_objects),
    }
    if str(error_granularity) != "scope":
        payload["error_granularity"] = str(error_granularity)
    canonical = json.dumps(
        _json_scalar(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


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


def _read_checkpoint_markers(
    run_dir: Path,
    *,
    signature: str,
) -> tuple[set[str], list[dict[str, Any]]]:
    completed: set[str] = set()
    markers: list[dict[str, Any]] = []
    marker_dir = run_dir / "markers"
    if not marker_dir.exists():
        return completed, markers
    for marker_path in sorted(marker_dir.glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if str(marker.get("signature")) != str(signature):
            raise RuntimeError(
                f"Checkpoint signature mismatch in {marker_path}."
            )
        data_ids = tuple(map(str, marker.get("data_config_ids", ())))
        overlap = completed.intersection(data_ids)
        if overlap:
            raise RuntimeError(
                "Checkpoint data groups occur in more than one completed "
                f"chunk: {sorted(overlap)[:5]}"
            )
        completed.update(data_ids)
        markers.append(marker)
    return completed, markers


def _save_checkpoint_batch(
    *,
    run_dir: Path,
    signature: str,
    data_config_ids: Sequence[str],
    result_parts: Mapping[str, Sequence[pd.DataFrame]],
) -> None:
    data_ids = tuple(map(str, data_config_ids))
    if not data_ids:
        return
    token = hashlib.sha256(
        "|".join(sorted(data_ids)).encode("utf-8")
    ).hexdigest()[:16]
    files: dict[str, str] = {}
    for table_name, schema in _INTERNAL_RESULT_SCHEMAS.items():
        parts = [
            part
            for part in result_parts.get(table_name, ())
            if part is not None and not part.empty
        ]
        if not parts:
            continue
        table = _with_schema(
            pd.concat(parts, ignore_index=True, sort=False),
            schema,
        )
        relative_path = Path("chunks") / f"{token}_{table_name}.parquet"
        _atomic_save_parquet(table, run_dir / relative_path)
        files[table_name] = relative_path.as_posix()
    marker = {
        "signature": signature,
        "data_config_ids": list(data_ids),
        "files": files,
    }
    _atomic_write_json(marker, run_dir / "markers" / f"{token}.json")


def _load_checkpoint_results(
    run_dir: Path,
    *,
    signature: str,
) -> dict[str, pd.DataFrame]:
    _, markers = _read_checkpoint_markers(run_dir, signature=signature)
    parts: dict[str, list[pd.DataFrame]] = {
        table_name: []
        for table_name in _INTERNAL_RESULT_SCHEMAS
    }
    for marker in markers:
        for table_name, relative_path in marker.get("files", {}).items():
            if table_name not in parts:
                raise RuntimeError(
                    f"Unknown checkpoint table {table_name!r}."
                )
            path = run_dir / str(relative_path)
            if not path.exists():
                raise RuntimeError(
                    f"Checkpoint marker references a missing file: {path}"
                )
            parts[table_name].append(pd.read_parquet(path))
    return {
        table_name: _with_schema(
            pd.concat(
                parts[table_name],
                ignore_index=True,
                sort=False,
            )
            if parts[table_name]
            else None,
            schema,
        )
        for table_name, schema in _INTERNAL_RESULT_SCHEMAS.items()
    }


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
                                    fit_config_id = (
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
                                            "fit_config_id": fit_config_id,
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
        "fit_config_id",
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
                    "fit_config_id": str(fit_row["fit_config_id"]),
                    "projection_level": str(contract["projection_level"]),
                    "projection_matrix_method": str(projection_method),
                    "rule_variant": str(fit_row["rule_variant"]),
                    "limit_source": str(fit_row["limit_source"]),
                    "protocol_version": str(expcfg.PROTOCOL_VERSION),
                    "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
                }
                projection_config_id = hash_internal_calibration_configuration(
                    projection_payload,
                    prefix="icproj",
                )
                evaluation_payload = {
                    **projection_payload,
                    "projection_config_id": projection_config_id,
                    "evaluation_track": str(contract["evaluation_track"]),
                    "decision_mode": str(contract["decision_mode"]),
                    "decision_score_type": str(
                        contract["decision_score_type"]
                    ),
                }
                evaluation_config_id = hash_internal_calibration_configuration(
                    evaluation_payload,
                    prefix="iceval",
                )
                rows.append(
                    {
                        **fit_row,
                        "source_config_id": str(
                            fit_row.get("config_id", fit_row["fit_config_id"])
                        ),
                        "projection_config_id": projection_config_id,
                        "evaluation_config_id": evaluation_config_id,
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
        "evaluation_config_id"
    ).reset_index(drop=True)
    observed = set(expanded["evaluation_track"])
    expected = set(track_contracts["evaluation_track"])
    if observed != expected:
        raise RuntimeError(
            "Projection expansion failed to cover all tracks: "
            f"missing={sorted(expected - observed)}"
        )
    if not expanded["evaluation_config_id"].is_unique:
        raise RuntimeError("evaluation_config_id must be unique.")
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


def run_internal_calibration(
    *,
    object_db: Mapping[str, Mapping[str, Any]],
    image_db: Mapping[str, Mapping[str, Any]],
    folds: pd.DataFrame,
    configurations: pd.DataFrame,
    wavelengths: np.ndarray | None = None,
    group_col: str = expcfg.INTERNAL_CALIBRATION_GROUP_COL,
    calibration_batches: Sequence[int] = expcfg.INTERNAL_CALIBRATION_BATCHES,
    forbidden_batches: Sequence[int] = expcfg.INTERNAL_CALIBRATION_FORBIDDEN_BATCHES,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    reference_object_threshold: float = expcfg.INTERNAL_CALIBRATION_REFERENCE_OBJECT_THRESHOLD,
    under_m_policy: str = expcfg.INTERNAL_CALIBRATION_UNDER_M_POLICY,
    keep_oof_pixels: bool = False,
    keep_oof_objects: bool = False,
    error_granularity: str = "scope",
    verbose: bool = expcfg.INTERNAL_CALIBRATION_VERBOSE,
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: bool = (
        expcfg.INTERNAL_CALIBRATION_RESUME_FROM_CHECKPOINT
    ),
    checkpoint_every_n_data_configs: int = (
        expcfg.INTERNAL_CALIBRATION_CHECKPOINT_EVERY_N_DATA_CONFIGS
    ),
) -> dict[str, pd.DataFrame]:
    """Run strict outer-fold calibration with data reuse and safe checkpoints.

    The raw train matrix, fitted preprocessing and transformed validation
    pixels are built once per data configuration and fold, then reused for all
    requested component counts. A checkpoint unit contains complete data
    configurations only, so resuming can never reuse a partially evaluated
    fold or ``k`` value.
    """
    del image_db  # Kept in the public API for notebook/database symmetry.
    if set(map(int, folds["batch"].unique())) - set(map(int, calibration_batches)):
        raise RuntimeError("Calibration folds contain a batch outside batches 1-2.")
    if set(map(int, folds["batch"].unique())) & set(map(int, forbidden_batches)):
        raise RuntimeError("Batch 3 or 4 entered internal calibration.")
    if int(checkpoint_every_n_data_configs) < 1:
        raise ValueError("checkpoint_every_n_data_configs must be at least 1.")
    if str(error_granularity) not in {"scope", "configuration"}:
        raise ValueError(
            "error_granularity must be 'scope' or 'configuration'."
        )

    required_configs = {
        "config_id",
        "fit_config_id",
        "matrix_family",
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "preprocessing",
        "preprocessing_steps",
        "rule_family",
        "rule_variant",
        "limit_source",
        "n_components",
        "alpha",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
        "random_state",
    }
    missing = sorted(required_configs - set(configurations.columns))
    if missing:
        raise KeyError(f"Missing internal-calibration configuration columns: {missing}")
    active_radii = set(
        pd.to_numeric(
            configurations["position_dilation_radius"],
            errors="raise",
        ).astype(int)
    )
    if active_radii != {0}:
        raise ValueError(
            "Internal calibration on pure references only supports "
            "position_dilation_radius=0. Test non-zero radii later."
        )

    configurations = _attach_data_configuration_ids(configurations)

    def finalize_results(
        results: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        fixed_schema = expcfg.SIMCA_GRID_FOLD_METRIC_COLUMNS
        if (
            not keep_oof_objects
            or "decision_mode" not in configurations
            or results.get("oof_objects") is None
            or results["oof_objects"].empty
        ):
            results["fixed_fold_metrics"] = _with_schema(
                pd.DataFrame(),
                fixed_schema,
            )
            results["fixed_threshold_metrics"] = _with_schema(
                pd.DataFrame(),
                expcfg.SIMCA_GRID_THRESHOLD_METRIC_COLUMNS,
            )
            return results
        mapping_columns = [
            column
            for column in (
                "config_id",
                "domain_config_id",
                "calibration_track",
                "decision_mode",
                "random_state",
                "object_threshold",
                "three_way_lower_threshold",
                "three_way_upper_threshold",
            )
            if column in configurations
        ]
        mapping = configurations[mapping_columns].drop_duplicates(
            "config_id"
        )
        oof = results["oof_objects"].merge(
            mapping,
            on="config_id",
            how="inner",
            validate="many_to_one",
        )
        metric_parts = []
        for mode in ("2way", "3way"):
            mode_oof = oof.loc[oof["decision_mode"].eq(mode)]
            if mode_oof.empty:
                continue
            if mode == "2way":
                evaluated = evaluate_internal_object_thresholds(
                    mode_oof,
                    thresholds=None,
                    threshold_mode="fixed",
                    target_class=target_class,
                    non_target_label=non_target_label,
                )
            else:
                evaluated = evaluate_internal_three_way_thresholds(
                    mode_oof,
                    lower_thresholds=None,
                    upper_thresholds=None,
                    threshold_mode="fixed",
                    target_class=target_class,
                    non_target_label=non_target_label,
                )
            metadata = mapping[
                [
                    "config_id",
                    "domain_config_id",
                    "calibration_track",
                    "random_state",
                ]
            ]
            evaluated = evaluated.merge(
                metadata,
                on="config_id",
                how="left",
                validate="many_to_one",
            )
            metric_parts.append(evaluated)
        fixed = (
            pd.concat(metric_parts, ignore_index=True, sort=False)
            if metric_parts
            else pd.DataFrame()
        )
        results["fixed_fold_metrics"] = _with_schema(
            fixed,
            fixed_schema,
        )
        results["fixed_threshold_metrics"] = (
            _summarize_fixed_fold_metrics(
                results["fixed_fold_metrics"]
            )
        )
        return results

    fold_ids = sorted(map(int, folds["fold_id"].unique()))
    fold_groups = (
        folds[["source_image", "fold_id"]]
        .drop_duplicates()
        .set_index("source_image")["fold_id"]
    )

    data_groups = list(configurations.groupby("data_config_id", sort=False))
    checkpoint_run_dir: Path | None = None
    checkpoint_signature: str | None = None
    completed_data_ids: set[str] = set()
    if checkpoint_dir is not None:
        checkpoint_signature = _internal_calibration_run_signature(
            configurations,
            folds,
            wavelengths=wavelengths,
            calibration_batches=calibration_batches,
            forbidden_batches=forbidden_batches,
            target_class=target_class,
            non_target_label=non_target_label,
            reference_object_threshold=reference_object_threshold,
            under_m_policy=under_m_policy,
            keep_oof_pixels=keep_oof_pixels,
            keep_oof_objects=keep_oof_objects,
            error_granularity=error_granularity,
        )
        checkpoint_run_dir = (
            Path(checkpoint_dir)
            / f"run_{checkpoint_signature}"
        )
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_run_dir / "manifest.json"
        manifest = {
            "signature": checkpoint_signature,
            "n_data_configurations": len(data_groups),
            "n_rule_configurations": len(configurations),
            "checkpoint_every_n_data_configs": int(
                checkpoint_every_n_data_configs
            ),
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(existing.get("signature")) != checkpoint_signature:
                raise RuntimeError(
                    f"Checkpoint manifest mismatch: {manifest_path}"
                )
        else:
            _atomic_write_json(manifest, manifest_path)
        existing_completed, _ = _read_checkpoint_markers(
            checkpoint_run_dir,
            signature=checkpoint_signature,
        )
        if resume_from_checkpoint:
            completed_data_ids = existing_completed
            if verbose and completed_data_ids:
                print(
                    "Checkpoint resume: "
                    f"{len(completed_data_ids)}/{len(data_groups)} "
                    "data configurations already complete."
                )
        elif existing_completed:
            raise RuntimeError(
                "Compatible checkpoint shards already exist. Set "
                "resume_from_checkpoint=True or use another checkpoint_dir."
            )

    in_memory_parts: dict[str, list[pd.DataFrame]] = {
        table_name: []
        for table_name in _INTERNAL_RESULT_SCHEMAS
    }
    pending_checkpoint_parts: dict[str, list[pd.DataFrame]] = {
        table_name: []
        for table_name in _INTERNAL_RESULT_SCHEMAS
    }
    pending_checkpoint_ids: list[str] = []

    def append_errors(
        rows: pd.DataFrame,
        target: list[dict[str, Any]],
        *,
        fold_id: int,
        status: str,
        technical_errors: str = "",
        exception: Exception | None = None,
    ) -> None:
        if rows.empty:
            return
        if str(error_granularity) == "configuration":
            for config in rows.drop_duplicates("config_id").itertuples(
                index=False
            ):
                target.append(
                    {
                        "scope_id": str(config.config_id),
                        "fold_id": int(fold_id),
                        "status": str(status),
                        "technical_errors": str(technical_errors),
                        "error_type": (
                            type(exception).__name__
                            if exception is not None
                            else ""
                        ),
                        "error_message": (
                            str(exception)
                            if exception is not None
                            else ""
                        ),
                        "n_affected_configurations": 1,
                    }
                )
            return
        scope_column = next(
            (
                column
                for column in ("data_config_id", "fit_config_id", "config_id")
                if column in rows and rows[column].nunique(dropna=False) == 1
            ),
            "config_id",
        )
        target.append(
            {
                "scope_id": str(rows.iloc[0][scope_column]),
                "fold_id": int(fold_id),
                "status": str(status),
                "technical_errors": str(technical_errors),
                "error_type": (
                    type(exception).__name__
                    if exception is not None
                    else ""
                ),
                "error_message": (
                    str(exception)
                    if exception is not None
                    else ""
                ),
                "n_affected_configurations": int(rows["config_id"].nunique()),
            }
        )

    def make_chunk(
        *,
        pixel_parts: Sequence[pd.DataFrame],
        object_parts: Sequence[pd.DataFrame],
        fold_metric_rows: Sequence[Mapping[str, Any]],
        rule_rows: Sequence[Mapping[str, Any]],
        sampling_rows: Sequence[Mapping[str, Any]],
        error_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, pd.DataFrame]:
        rule_diagnostics = pd.DataFrame(rule_rows)
        raw_tables: dict[str, pd.DataFrame | None] = {
            "oof_pixels": (
                pd.concat(pixel_parts, ignore_index=True, sort=False)
                if pixel_parts
                else None
            ),
            "oof_objects": (
                pd.concat(object_parts, ignore_index=True, sort=False)
                if object_parts
                else None
            ),
            "fold_metrics": pd.DataFrame(fold_metric_rows),
            "rule_diagnostics": rule_diagnostics,
            "sampling_diagnostics": pd.DataFrame(sampling_rows),
            "errors": pd.DataFrame(error_rows),
        }
        return {
            table_name: _with_schema(raw_tables[table_name], schema)
            for table_name, schema in _INTERNAL_RESULT_SCHEMAS.items()
        }

    n_total_data_groups = len(data_groups)
    for data_counter, (data_config_id, data_rows) in enumerate(
        data_groups,
        start=1,
    ):
        data_config_id = str(data_config_id)
        if data_config_id in completed_data_ids:
            continue
        base = data_rows.iloc[0]
        if verbose:
            print(
                f"[data {data_counter}/{n_total_data_groups}] "
                f"{base['matrix_method']} | {base['preprocessing']} | "
                f"m={base['m']} | seed={base['random_state']} | "
                f"k={data_rows['n_components'].nunique()} values"
            )

        pixel_parts: list[pd.DataFrame] = []
        object_parts: list[pd.DataFrame] = []
        fold_metric_rows: list[dict[str, Any]] = []
        rule_rows: list[dict[str, Any]] = []
        sampling_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []

        for fold_id in fold_ids:
            valid_groups = fold_groups.index[fold_groups.eq(fold_id)].astype(str)
            train_groups = fold_groups.index[~fold_groups.eq(fold_id)].astype(str)
            train_reference = folds.loc[
                folds["source_image"].astype(str).isin(train_groups)
            ]
            available_classes = tuple(
                sorted(train_reference["class_name"].astype(str).unique())
            )
            train_filters = _base_filters(
                batches=calibration_batches,
                classes=(target_class,),
                groups=train_groups,
                group_col=group_col,
            )
            valid_filters = _base_filters(
                batches=calibration_batches,
                classes=(target_class, non_target_label),
                groups=valid_groups,
                group_col=group_col,
            )
            m_effective = (
                int(base["m"])
                if not pd.isna(base["m"])
                else expcfg.M_BALANCED_PIXELS
            )
            try:
                pixels_per_object = _training_pixels_by_object(
                    object_db,
                    train_filters,
                )
                if (
                    str(base["matrix_method"]) == "balanced_pixels"
                    and (
                        pixels_per_object.size == 0
                        or m_effective > int(np.min(pixels_per_object))
                    )
                ):
                    append_errors(
                        data_rows,
                        error_rows,
                        fold_id=fold_id,
                        status="technical_invalid",
                        technical_errors="M_EXCEEDS_AVAILABLE_PIXELS",
                    )
                    continue
                X_train_raw, y_train, meta_train, matrix_wavelengths = build_matrix(
                    object_db=object_db,
                    matrix_method=str(base["matrix_method"]),
                    filters=train_filters,
                    m=m_effective,
                    random_state=int(base["random_state"]),
                    replace=False,
                    balanced_pixel_strategy=str(
                        base["balanced_pixel_strategy"]
                        if base["balanced_pixel_strategy"] != "not_applicable"
                        else "random"
                    ),
                    under_m_policy=under_m_policy,
                    return_wavelengths=True,
                )
                sampling_rows.append(
                    {
                        "data_config_id": data_config_id,
                        "sampling_group_id": str(base["sampling_group_id"]),
                        "fold_id": int(fold_id),
                        "random_state": int(base["random_state"]),
                        "sampling_minhash": _sampling_minhash(meta_train),
                    }
                )
                X_train_raw_centered = (
                    X_train_raw - np.mean(X_train_raw, axis=0, keepdims=True)
                )
                raw_rank = int(
                    np.linalg.matrix_rank(X_train_raw_centered)
                )

                preprocessor = SpectralPreprocessor(
                    steps=tuple(str(base["preprocessing_steps"]).split("+")),
                    sg_window_length=int(base["sg_window_length"]),
                    sg_polyorder=int(base["sg_polyorder"]),
                )
                active_wavelengths = (
                    wavelengths
                    if wavelengths is not None
                    else matrix_wavelengths
                )
                X_train = preprocessor.fit_transform(
                    X_train_raw,
                    wavelengths=active_wavelengths,
                )
                transformed_rank = int(
                    np.linalg.matrix_rank(
                        X_train - np.mean(X_train, axis=0, keepdims=True)
                    )
                )
                X_valid_raw, y_valid, meta_valid = build_matrix(
                    object_db=object_db,
                    matrix_method="pixel",
                    filters=valid_filters,
                    balanced_pixel_strategy="random",
                )
                X_valid = preprocessor.transform(X_valid_raw)

                base_pixel = pd.DataFrame(meta_valid)
                base_pixel["label"] = np.asarray(y_valid).astype(str)
                true_pixel_col = true_col(target_class, "pixel")
                # Batches 1-2 contain pure references: their object label is the
                # exact pixel truth. Dilation is therefore non-identifiable here
                # and is not activated in the default 03B grid.
                base_pixel[true_pixel_col] = base_pixel["label"].eq(
                    str(target_class)
                )
                base_pixel["truth_available"] = True
                pred_pixel_col = predicted_col(target_class, "pixel")
                true_object_col = true_col(target_class, "object")
                ratio_column = pixel_ratio_col(target_class)

                for _, model_rows in data_rows.groupby(
                    "fit_config_id",
                    sort=False,
                ):
                    model_base = model_rows.iloc[0]
                    raw_validation = validate_simca_configuration(
                        model_base,
                        X_train_raw,
                        y_train,
                        n_target_observations=len(X_train_raw),
                        n_features=X_train_raw.shape[1],
                        numeric_rank=raw_rank,
                        n_pixels_by_object=pixels_per_object,
                        available_classes=available_classes,
                        target_class=target_class,
                        non_target_label=non_target_label,
                    )
                    technical_errors = list(
                        raw_validation["technical_error_codes"]
                    )
                    if (
                        int(model_base["n_components"])
                        > transformed_rank
                    ):
                        technical_errors.append(
                            "N_COMPONENTS_EXCEED_PREPROCESSED_RANK"
                        )
                    if technical_errors:
                        append_errors(
                            model_rows,
                            error_rows,
                            fold_id=fold_id,
                            status="technical_invalid",
                            technical_errors=";".join(
                                dict.fromkeys(technical_errors)
                            ),
                        )
                        continue

                    try:
                        model = SIMCAClassModel(
                            class_name=target_class,
                            n_components=int(
                                model_base["n_components"]
                            ),
                            alpha=float(model_base["alpha"]),
                        ).fit(X_train)
                        train_only_thresholds = (
                            compute_train_only_rule_thresholds(
                                model,
                                alpha=float(model_base["alpha"]),
                            )
                        )
                        values = model.decision_values(X_valid)
                        H = np.asarray(values["H"], dtype=float)
                        Q = np.asarray(values["Q"], dtype=float)
                    except Exception as exc:
                        append_errors(
                            model_rows,
                            error_rows,
                            fold_id=fold_id,
                            status="fit_or_projection_error",
                            exception=exc,
                        )
                        continue

                    model_pixel = base_pixel.copy()
                    model_pixel["H"] = H
                    model_pixel["Q"] = Q

                    for rule_variant, variant_rows in model_rows.groupby(
                        "rule_variant",
                        sort=False,
                    ):
                        rule_base = variant_rows.iloc[0]
                        try:
                            stat, limit = compute_rule_variant_stat_limit(
                                H=H,
                                Q=Q,
                                model=model,
                                variant_name=str(rule_variant),
                                cv_thresholds=train_only_thresholds,
                            )
                            stat = np.asarray(stat, dtype=float)
                            accepted = stat < float(limit)
                            pixel = model_pixel.assign(
                                **{
                                    pred_pixel_col: accepted,
                                    "rule_statistic": stat,
                                    "rule_limit": float(limit),
                                }
                            )
                            pixel_compact_base = pd.DataFrame(
                                {
                                    "fold_id": int(fold_id),
                                    "rule_variant": str(rule_variant),
                                    "limit_source": str(
                                        rule_base["limit_source"]
                                    ),
                                    "source_image": pixel[
                                        "source_image"
                                    ].astype(str),
                                    "object_id": pixel["object_id"].astype(
                                        str
                                    ),
                                    "batch": pd.to_numeric(pixel["batch"]),
                                    "row": pd.to_numeric(pixel["row"]),
                                    "col": pd.to_numeric(pixel["col"]),
                                    "true_target_pixel": pixel[
                                        true_pixel_col
                                    ].astype(bool),
                                    "predicted_target_pixel": (
                                        accepted.astype(bool)
                                    ),
                                    "H": H,
                                    "Q": Q,
                                    "rule_statistic": stat,
                                    "rule_limit": float(limit),
                                }
                            )

                            objects = (
                                aggregate_pixel_predictions_to_objects(
                                    pixel_df=pixel,
                                    object_db=object_db,
                                    target_class=target_class,
                                    non_target_label=non_target_label,
                                    object_threshold=float(
                                        reference_object_threshold
                                    ),
                                )
                            )
                            object_compact_base = pd.DataFrame(
                                {
                                    "fold_id": int(fold_id),
                                    "rule_variant": str(rule_variant),
                                    "source_image": objects[
                                        "source_image"
                                    ].astype(str),
                                    "object_id": objects[
                                        "object_id"
                                    ].astype(str),
                                    "batch": pd.to_numeric(
                                        objects["batch"]
                                    ),
                                    "target_pixel_ratio": pd.to_numeric(
                                        objects[ratio_column]
                                    ),
                                    "true_target_object": objects[
                                        true_object_col
                                    ].astype(bool),
                                    "n_pixels_projected": pd.to_numeric(
                                        objects["n_pixels_projected"]
                                    ),
                                }
                            )
                            object_eval = object_compact_base.assign(
                                predicted_target_object=(
                                    object_compact_base[
                                        "target_pixel_ratio"
                                    ]
                                    >= float(reference_object_threshold)
                                )
                            )
                            object_metrics = binary_detection_metrics(
                                object_eval,
                                true_col="true_target_object",
                                pred_col="predicted_target_object",
                                target_class=target_class,
                                non_target_class=non_target_label,
                            )
                            pixel_metrics = binary_detection_metrics(
                                pixel_compact_base,
                                true_col="true_target_pixel",
                                pred_col="predicted_target_pixel",
                                target_class=target_class,
                                non_target_class=non_target_label,
                            )
                            train_stat, train_limit = (
                                compute_rule_variant_stat_limit(
                                    H=model.H_train_,
                                    Q=model.Q_train_,
                                    model=model,
                                    variant_name=str(rule_variant),
                                    cv_thresholds=train_only_thresholds,
                                )
                            )
                            valid_target = pixel_compact_base[
                                "true_target_pixel"
                            ].astype(bool).to_numpy()
                            rejected_valid_target = ~accepted[valid_target]
                        except Exception as exc:
                            append_errors(
                                variant_rows,
                                error_rows,
                                fold_id=fold_id,
                                status="fit_or_projection_error",
                                exception=exc,
                            )
                            continue

                        for rule in variant_rows.itertuples(index=False):
                            if keep_oof_pixels:
                                pixel_parts.append(
                                    pixel_compact_base.assign(
                                        config_id=rule.config_id
                                    )
                                )
                            if keep_oof_objects:
                                object_parts.append(
                                    object_compact_base.assign(
                                        config_id=rule.config_id
                                    )
                                )
                            fold_metric_rows.append(
                                {
                                    "config_id": rule.config_id,
                                    "fold_id": int(fold_id),
                                    "matrix_family": rule.matrix_family,
                                    "matrix_method": rule.matrix_method,
                                    "preprocessing": rule.preprocessing,
                                    "rule_variant": rule.rule_variant,
                                    "n_components": int(
                                        rule.n_components
                                    ),
                                    "alpha": float(rule.alpha),
                                    "m": rule.m,
                                    "balanced_pixel_strategy": (
                                        rule.balanced_pixel_strategy
                                    ),
                                    "sg_window_length": int(
                                        rule.sg_window_length
                                    ),
                                    "position_dilation_radius": int(
                                        rule.position_dilation_radius
                                    ),
                                    "random_state": int(
                                        rule.random_state
                                    ),
                                    "n_objects": int(
                                        object_metrics["n"]
                                    ),
                                    "fn_rate": object_metrics["fn_rate"],
                                    "fp_rate": object_metrics["fp_rate"],
                                    "balanced_accuracy": object_metrics[
                                        "balanced_accuracy"
                                    ],
                                    "pixel_fn_rate": pixel_metrics[
                                        "fn_rate"
                                    ],
                                    "pixel_fp_rate": pixel_metrics[
                                        "fp_rate"
                                    ],
                                    "pixel_balanced_accuracy": (
                                        pixel_metrics[
                                            "balanced_accuracy"
                                        ]
                                    ),
                                }
                            )
                            rule_rows.append(
                                {
                                    "config_id": rule.config_id,
                                    "fold_id": int(fold_id),
                                    "rule_family": rule.rule_family,
                                    "rule_variant": rule.rule_variant,
                                    "limit_source": rule.limit_source,
                                    "rule_limit": float(train_limit),
                                    "train_rejection_rate": float(
                                        np.mean(
                                            np.asarray(
                                                train_stat,
                                                dtype=float,
                                            )
                                            >= float(train_limit)
                                        )
                                    ),
                                    "validation_target_rejection_rate": (
                                        float(
                                            np.mean(
                                                rejected_valid_target
                                            )
                                        )
                                        if rejected_valid_target.size
                                        else np.nan
                                    ),
                                    "n_train_target": int(
                                        len(train_stat)
                                    ),
                                    "n_validation_target": int(
                                        rejected_valid_target.size
                                    ),
                                }
                            )
            except Exception as exc:
                append_errors(
                    data_rows,
                    error_rows,
                    fold_id=fold_id,
                    status="fit_or_projection_error",
                    exception=exc,
                )

        chunk = make_chunk(
            pixel_parts=pixel_parts,
            object_parts=object_parts,
            fold_metric_rows=fold_metric_rows,
            rule_rows=rule_rows,
            sampling_rows=sampling_rows,
            error_rows=error_rows,
        )
        if checkpoint_run_dir is None:
            for table_name, table in chunk.items():
                if not table.empty:
                    in_memory_parts[table_name].append(table)
        else:
            pending_checkpoint_ids.append(data_config_id)
            for table_name, table in chunk.items():
                if not table.empty:
                    pending_checkpoint_parts[table_name].append(table)
            if (
                len(pending_checkpoint_ids)
                >= int(checkpoint_every_n_data_configs)
            ):
                _save_checkpoint_batch(
                    run_dir=checkpoint_run_dir,
                    signature=str(checkpoint_signature),
                    data_config_ids=pending_checkpoint_ids,
                    result_parts=pending_checkpoint_parts,
                )
                pending_checkpoint_ids = []
                pending_checkpoint_parts = {
                    table_name: []
                    for table_name in _INTERNAL_RESULT_SCHEMAS
                }

    if checkpoint_run_dir is not None:
        if pending_checkpoint_ids:
            _save_checkpoint_batch(
                run_dir=checkpoint_run_dir,
                signature=str(checkpoint_signature),
                data_config_ids=pending_checkpoint_ids,
                result_parts=pending_checkpoint_parts,
            )
        completed_data_ids, _ = _read_checkpoint_markers(
            checkpoint_run_dir,
            signature=str(checkpoint_signature),
        )
        expected_data_ids = {
            str(data_config_id)
            for data_config_id, _ in data_groups
        }
        missing_data_ids = expected_data_ids - completed_data_ids
        if missing_data_ids:
            raise RuntimeError(
                "Internal calibration ended with incomplete checkpoint "
                f"coverage: {len(missing_data_ids)} data configurations."
            )
        return finalize_results(
            _load_checkpoint_results(
                checkpoint_run_dir,
                signature=str(checkpoint_signature),
            )
        )

    results = {
        table_name: _with_schema(
            pd.concat(
                in_memory_parts[table_name],
                ignore_index=True,
                sort=False,
            )
            if in_memory_parts[table_name]
            else None,
            schema,
        )
        for table_name, schema in _INTERNAL_RESULT_SCHEMAS.items()
    }
    return finalize_results(results)


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
    materialize_checkpoint_results: bool = True,
) -> dict[str, Any]:
    """Fit once per fold/k and emit shared object/pixel OOF projections."""
    required = {
        "fit_config_id",
        "projection_config_id",
        "evaluation_config_id",
        "evaluation_track",
        "matrix_family",
        "matrix_method",
        "projection_level",
        "projection_matrix_method",
        "rule_variant",
        "limit_source",
        "n_components",
        "alpha",
        "preprocessing_steps",
        "sg_window_length",
        "sg_polyorder",
        "random_state",
    }
    missing = sorted(required - set(configurations.columns))
    if missing:
        raise KeyError(f"Missing 8-track configuration columns: {missing}")
    if not materialize_checkpoint_results and checkpoint_dir is None:
        raise ValueError(
            "A checkpoint_dir is required for out-of-core calibration."
        )
    if set(pd.to_numeric(folds["batch"]).astype(int)) - set(
        expcfg.INTERNAL_CALIBRATION_BATCHES
    ):
        raise RuntimeError("Only calibration batches 1-2 are allowed in 03B.")
    if not folds["object_id"].astype(str).is_unique:
        raise RuntimeError("Each QC-eligible object must occur in one fold.")
    configurations = _attach_data_configuration_ids(configurations)
    fold_metadata = folds.set_index(folds["object_id"].astype(str))[
        ["object_area", "size_bin"]
    ]
    schemas = {
        "fit_diagnostics": expcfg.INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_V2_COLUMNS,
        "rule_diagnostics": expcfg.INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_V2_COLUMNS,
        "oof_object_predictions": expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_V2_COLUMNS,
        "oof_pixel_predictions": expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_V2_COLUMNS,
        "projection_shift": expcfg.INTERNAL_CALIBRATION_PROJECTION_SHIFT_COLUMNS,
        "technical_errors": expcfg.INTERNAL_CALIBRATION_AUDIT_V2_COLUMNS,
    }
    result_parts: dict[str, list[pd.DataFrame]] = {
        "fit_diagnostics": [],
        "rule_diagnostics": [],
        "oof_object_predictions": [],
        "oof_pixel_predictions": [],
        "projection_shift": [],
        "technical_errors": [],
    }
    reported_projection_filters: set[tuple[str, int, str]] = set()

    def compact_projection(
        projected: pd.DataFrame,
        *,
        row: Mapping[str, Any],
        fold_id: int,
    ) -> pd.DataFrame:
        out = projected.copy()
        out.attrs = {}
        object_ids = out["object_id"].astype(str)
        out["object_area"] = object_ids.map(fold_metadata["object_area"])
        out["size_bin"] = object_ids.map(fold_metadata["size_bin"])
        if out[["object_area", "size_bin"]].isna().any(axis=None):
            raise RuntimeError("OOF projection contains an object outside folds.")
        out["projection_config_id"] = str(row["projection_config_id"])
        out["fit_config_id"] = str(row["fit_config_id"])
        out["fold_id"] = int(fold_id)
        out["random_state"] = int(row["random_state"])
        out["training_matrix_family"] = str(row["matrix_family"])
        out["projection_level"] = str(row["projection_level"])
        return out

    def append_rule_error(
        *,
        projection_config_id: str,
        fit_config_id: str,
        fold_id: int,
        projection_base: Mapping[str, Any],
        exc: Exception,
        error_code: str,
    ) -> None:
        result_parts["rule_diagnostics"].append(
            pd.DataFrame(
                [{
                    "projection_config_id": str(projection_config_id),
                    "fit_config_id": str(fit_config_id),
                    "fold_id": int(fold_id),
                    "random_state": int(projection_base["random_state"]),
                    "rule_variant": str(projection_base["rule_variant"]),
                    "limit_method": str(projection_base["limit_source"]),
                    "limit_alpha": float(projection_base["alpha"]),
                    "status": "error",
                    "error_code": (
                        f"{error_code}: {type(exc).__name__}: {exc}"
                    ),
                }]
            )
        )

    data_groups = list(configurations.groupby("data_config_id", sort=False))
    checkpoint_run_dir: Path | None = None
    checkpoint_signature = ""
    checkpoint_fields: dict[str, str] = {}
    if checkpoint_dir is not None:
        required_context = {
            "protocol_hash",
            "pca_shortlist_id",
            "track_contract_hash",
            "fold_contract_hash",
            "configuration_hash",
        }
        checkpoint_fields = {
            str(key): str(value)
            for key, value in dict(checkpoint_context or {}).items()
        }
        missing_context = sorted(
            key
            for key in required_context
            if not checkpoint_fields.get(key)
        )
        if missing_context:
            raise ValueError(
                "Incomplete 8-track checkpoint context: "
                f"{missing_context}"
            )
        signature_payload = {
            **checkpoint_fields,
            "runner_contract": _INTERNAL_CALIBRATION_RUNNER_CONTRACT,
            "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
            "protocol_version": str(expcfg.PROTOCOL_VERSION),
            "evaluation_config_ids": sorted(
                configurations["evaluation_config_id"].astype(str).unique()
            ),
            "fold_ids": sorted(
                map(int, folds["fold_id"].astype(int).unique())
            ),
            "under_m_policy": str(under_m_policy),
        }
        checkpoint_signature = hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # The complete SHA-256 remains stored and checked in the manifest, but
        # using all 64 hexadecimal characters in the directory name makes
        # otherwise valid shard paths exceed Windows MAX_PATH.  Twenty hex
        # characters provide an 80-bit directory key; a hypothetical prefix
        # collision is still rejected by the full manifest signature check.
        checkpoint_run_dir = (
            Path(checkpoint_dir) / f"run_{checkpoint_signature[:20]}"
        )
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            {
                "signature": checkpoint_signature,
                **signature_payload,
                "expected_data_config_ids": sorted(
                    configurations["data_config_id"].astype(str).unique()
                ),
                "expected_fit_config_ids": sorted(
                    configurations["fit_config_id"].astype(str).unique()
                ),
            },
            checkpoint_run_dir / "manifest.json",
        )

    for data_index, (data_config_id, data_rows) in enumerate(
        data_groups, start=1
    ):
        data_config_id = str(data_config_id)
        marker_path = (
            None
            if checkpoint_run_dir is None
            else checkpoint_run_dir / "markers" / f"{data_config_id}.json"
        )
        if marker_path is not None and marker_path.exists():
            if not resume_from_checkpoint:
                raise RuntimeError(
                    "Compatible 8-track checkpoint exists but resume is disabled."
                )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if str(marker.get("signature")) != checkpoint_signature:
                raise RuntimeError(f"Checkpoint signature mismatch: {marker_path}")
            for key, expected_value in checkpoint_fields.items():
                if str(marker.get(key)) != str(expected_value):
                    raise RuntimeError(
                        f"Checkpoint {key} mismatch: {marker_path}"
                    )
            for shard in marker.get("shards", []):
                name = str(shard["name"])
                if name not in schemas:
                    raise RuntimeError(f"Unknown checkpoint shard type: {name}")
                path = checkpoint_run_dir / str(shard["relative_path"])
                _validate_8track_checkpoint_shard(
                    path,
                    shard,
                    schemas[name],
                )
                if materialize_checkpoint_results:
                    table = pd.read_parquet(path)
                    result_parts[name].append(
                        _with_schema(table, schemas[name])
                    )
            continue
        part_offsets = {
            name: len(parts) for name, parts in result_parts.items()
        }
        base = data_rows.iloc[0]
        if verbose:
            print(
                f"[03B data {data_index}/{len(data_groups)}] "
                f"{base['matrix_method']} | {base['preprocessing']} | "
                f"m={base['m']} | seed={base['random_state']}"
            )
        for fold_id in sorted(pd.to_numeric(folds["fold_id"]).astype(int).unique()):
            valid_ids = set(
                folds.loc[folds["fold_id"].eq(fold_id), "object_id"].astype(str)
            )
            train_ids = set(folds["object_id"].astype(str)) - valid_ids
            if valid_ids & train_ids:
                raise RuntimeError("An object is shared by train and OOF projection.")
            train_db = {
                str(object_id): object_db[str(object_id)]
                for object_id in sorted(train_ids)
            }
            valid_db = {
                str(object_id): object_db[str(object_id)]
                for object_id in sorted(valid_ids)
            }
            train_filters = {
                "sample_kind": ["pure"],
                "object_nut_type": [str(target_class)],
            }
            projection_filters = {
                "sample_kind": ["pure"],
                "object_nut_type": [str(target_class), str(non_target_label)],
            }
            m_value = (
                int(base["m"])
                if not pd.isna(base.get("m"))
                else int(expcfg.PCA_BALANCED_M_VALUES[0])
            )
            matrix_start = perf_counter()
            try:
                X_train_raw, y_train, train_metadata = build_matrix(
                    object_db=train_db,
                    matrix_method=str(base["matrix_method"]),
                    filters=train_filters,
                    m=m_value,
                    random_state=int(base["random_state"]),
                    replace=False,
                    balanced_pixel_strategy=str(
                        base["balanced_pixel_strategy"]
                        if base["balanced_pixel_strategy"] != "not_applicable"
                        else "random"
                    ),
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
                        data_rows["projection_matrix_method"].astype(str).unique()
                    )
                }
            except Exception as exc:
                result_parts["technical_errors"].append(
                    pd.DataFrame(
                        [{
                            "audit_type": "technical_error",
                            "evaluation_track": "",
                            "track_id": "",
                            "track_status": "matrix_build_failed",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }]
                    )
                )
                continue

            fitted_preprocessor = None
            X_train_preprocessed = None
            fit_groups = sorted(
                data_rows.groupby("fit_config_id", sort=False),
                key=lambda item: int(item[1].iloc[0]["n_components"]),
            )
            for fit_config_id, fit_rows in fit_groups:
                fit_base = fit_rows.iloc[0]
                try:
                    bundle = fit_simca_bundle_from_matrix(
                        X_train_raw,
                        y_train,
                        train_metadata,
                        preprocessing_spec={
                            "steps": tuple(
                                str(fit_base["preprocessing_steps"]).split("+")
                            ),
                            "sg_window_length": int(
                                fit_base["sg_window_length"]
                            ),
                            "sg_polyorder": int(fit_base["sg_polyorder"]),
                        },
                        n_components=int(fit_base["n_components"]),
                        alpha=float(fit_base["alpha"]),
                        wavelengths=wavelengths,
                        target_class=str(target_class),
                        fitted_preprocessor=fitted_preprocessor,
                        X_train_preprocessed=X_train_preprocessed,
                    )
                    if fitted_preprocessor is None:
                        fitted_preprocessor = bundle.preprocessor
                        X_train_preprocessed = bundle.X_train
                    result_parts["fit_diagnostics"].append(
                        pd.DataFrame(
                            [{
                                "fit_config_id": str(fit_config_id),
                                "fold_id": int(fold_id),
                                "random_state": int(fit_base["random_state"]),
                                "raw_rank": int(bundle.raw_rank),
                                "preprocessed_rank": int(
                                    bundle.preprocessed_rank
                                ),
                                "n_train_target": int(len(bundle.X_train)),
                                "n_features": int(bundle.X_train.shape[1]),
                                "n_components": int(fit_base["n_components"]),
                                "matrix_build_seconds": float(matrix_seconds),
                                "preprocessing_seconds": float(
                                    bundle.preprocessing_seconds
                                ),
                                "fit_seconds": float(bundle.fit_seconds),
                                "status": "ok",
                                "error_code": "",
                                "error_message": "",
                            }]
                        )
                    )
                    train_only_thresholds = compute_train_only_rule_thresholds(
                        bundle.model,
                        alpha=float(fit_base["alpha"]),
                    )
                except Exception as exc:
                    result_parts["fit_diagnostics"].append(
                        pd.DataFrame(
                            [{
                                "fit_config_id": str(fit_config_id),
                                "fold_id": int(fold_id),
                                "random_state": int(fit_base["random_state"]),
                                "n_components": int(fit_base["n_components"]),
                                "matrix_build_seconds": float(matrix_seconds),
                                "status": "error",
                                "error_code": "FIT_FAILED",
                                "error_message": f"{type(exc).__name__}: {exc}",
                            }]
                        )
                    )
                    continue

                projection_caches: dict[str, dict[str, Any]] = {}
                projection_cache_errors: dict[str, Exception] = {}
                for method in sorted(
                    fit_rows["projection_matrix_method"].astype(str).unique()
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
                        validity = dict(cache.get("input_validity", {}))
                        filter_audit_key = (
                            str(data_config_id),
                            int(fold_id),
                            str(method),
                        )
                        if (
                            int(validity.get("n_filtered_rows", 0)) > 0
                            and filter_audit_key
                            not in reported_projection_filters
                        ):
                            method_rows = data_rows.loc[
                                data_rows["projection_matrix_method"]
                                .astype(str)
                                .eq(str(method))
                            ]
                            result_parts["technical_errors"].append(
                                pd.DataFrame(
                                    [{
                                        "audit_type": "projection_input_filter",
                                        "evaluation_track": ",".join(
                                            sorted(
                                                method_rows[
                                                    "evaluation_track"
                                                ].astype(str).unique()
                                            )
                                        ),
                                        "track_id": ",".join(
                                            sorted(
                                                method_rows["track_id"]
                                                .astype(str).unique()
                                            )
                                        ),
                                        "n_initial": int(
                                            validity["n_input_rows"]
                                        ),
                                        "n_technical_valid": int(
                                            validity["n_valid_rows"]
                                        ),
                                        "track_status": (
                                            "valid_after_projection_input_filter"
                                        ),
                                        "failure_reason": (
                                            f"data_config_id={data_config_id};"
                                            f"fit_config_id={fit_config_id};"
                                            f"fold_id={fold_id};"
                                            f"projection_matrix_method={method};"
                                            f"filtered_rows="
                                            f"{validity['n_filtered_rows']};"
                                            f"nonfinite_rows="
                                            f"{validity['n_nonfinite_rows']};"
                                            f"nonpositive_absorbance_rows="
                                            f"{validity['n_nonpositive_absorbance_rows']}"
                                        ),
                                    }]
                                )
                            )
                            reported_projection_filters.add(filter_audit_key)
                    except Exception as exc:
                        projection_cache_errors[method] = exc
                for projection_config_id, projection_rows in fit_rows.groupby(
                    "projection_config_id", sort=False
                ):
                    projection_base = projection_rows.iloc[0]
                    method = str(projection_base["projection_matrix_method"])
                    if method in projection_cache_errors:
                        append_rule_error(
                            projection_config_id=str(projection_config_id),
                            fit_config_id=str(fit_config_id),
                            fold_id=int(fold_id),
                            projection_base=projection_base,
                            exc=projection_cache_errors[method],
                            error_code="PROJECTION_PREPARATION_FAILED",
                        )
                        continue
                    try:
                        projected = project_simca_bundle(
                            bundle,
                            object_db=valid_db,
                            projection_matrix_method=method,
                            projection_filters=projection_filters,
                            projection_cache=projection_caches[method],
                            rule_variant=str(projection_base["rule_variant"]),
                            train_only_thresholds=train_only_thresholds,
                            target_class=str(target_class),
                            under_m_policy=str(under_m_policy),
                        )
                        compact = compact_projection(
                            projected,
                            row=projection_base,
                            fold_id=fold_id,
                        )
                        output_key = (
                            "oof_pixel_predictions"
                            if str(projection_base["projection_level"])
                            == "pixel_projection"
                            else "oof_object_predictions"
                        )
                        result_parts[output_key].append(compact)
                        train_stat, train_limit = compute_rule_variant_stat_limit(
                            H=bundle.model.H_train_,
                            Q=bundle.model.Q_train_,
                            model=bundle.model,
                            variant_name=str(projection_base["rule_variant"]),
                            cv_thresholds=train_only_thresholds,
                        )
                        train_stat = np.asarray(train_stat, dtype=float)
                        train_limit = float(train_limit)
                        if not np.isfinite(train_limit) or train_limit <= 0:
                            raise RuntimeError("Non-finite train-only rule limit.")
                        target_projected = compact.loc[
                            compact["truth"].astype(bool)
                        ]
                        result_parts["rule_diagnostics"].append(
                            pd.DataFrame(
                                [{
                                    "projection_config_id": str(
                                        projection_config_id
                                    ),
                                    "fit_config_id": str(fit_config_id),
                                    "fold_id": int(fold_id),
                                    "random_state": int(
                                        projection_base["random_state"]
                                    ),
                                    "rule_variant": str(
                                        projection_base["rule_variant"]
                                    ),
                                    "limit_method": str(
                                        projection_base["limit_source"]
                                    ),
                                    "limit_alpha": float(
                                        projection_base["alpha"]
                                    ),
                                    "q_limit": float(bundle.model.Q_limit_),
                                    "t2_limit": float(bundle.model.H_limit_),
                                    "rule_limit": train_limit,
                                    "train_rejection_rate": float(
                                        np.mean(train_stat / train_limit >= 1.0)
                                    ),
                                    "oof_target_rejection_rate": (
                                        float(
                                            target_projected[
                                                "simca_margin"
                                            ].lt(0.0).mean()
                                        )
                                        if len(target_projected)
                                        else np.nan
                                    ),
                                    "status": "ok",
                                    "error_code": "",
                                }]
                            )
                        )
                        train_scores = bundle.train_scores.assign(
                            fit_config_id=str(fit_config_id),
                            fold_id=int(fold_id),
                            rule_limit=train_limit,
                            normalized_ratio=train_stat / train_limit,
                            simca_margin=1.0 - train_stat / train_limit,
                        )
                        shift_input = compact.assign(
                            projection_config_id=str(projection_config_id)
                        )
                        result_parts["projection_shift"].append(
                            summarize_projection_shift(
                                train_scores,
                                shift_input,
                            )
                        )
                    except Exception as exc:
                        append_rule_error(
                            projection_config_id=str(projection_config_id),
                            fit_config_id=str(fit_config_id),
                            fold_id=int(fold_id),
                            projection_base=projection_base,
                            exc=exc,
                            error_code="PROJECTION_RULE_FAILED",
                        )

        if marker_path is not None and checkpoint_run_dir is not None:
            shard_rows = []
            completed_fit_ids = sorted(
                data_rows["fit_config_id"].astype(str).unique()
            )
            table = None
            new_parts = None
            for name, parts in result_parts.items():
                new_parts = parts[part_offsets[name]:]
                if not new_parts:
                    continue
                table = _with_schema(
                    pd.concat(new_parts, ignore_index=True, sort=False),
                    schemas[name],
                    copy=False,
                )
                relative_path = (
                    Path("chunks") / f"{data_config_id}_{name}.parquet"
                )
                absolute_path = checkpoint_run_dir / relative_path
                _atomic_save_parquet(table, absolute_path)
                shard_rows.append(
                    {
                        "name": name,
                        "relative_path": relative_path.as_posix(),
                        "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
                        "protocol_version": str(expcfg.PROTOCOL_VERSION),
                        **checkpoint_fields,
                        "row_count": int(len(table)),
                        "columns": list(map(str, table.columns)),
                        "file_sha256": _streaming_sha256(absolute_path),
                        "completed_fit_config_ids": completed_fit_ids,
                    }
                )
            _atomic_write_json(
                {
                    "signature": checkpoint_signature,
                    "data_config_id": data_config_id,
                    "schema_version": str(expcfg.RESULTS_SCHEMA_VERSION),
                    "protocol_version": str(expcfg.PROTOCOL_VERSION),
                    **checkpoint_fields,
                    "completed_fit_config_ids": completed_fit_ids,
                    "shards": shard_rows,
                },
                marker_path,
            )
            if not materialize_checkpoint_results:
                for name, offset in part_offsets.items():
                    del result_parts[name][offset:]
                table = None
                new_parts = None
                gc.collect()

    if checkpoint_run_dir is not None:
        markers = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((checkpoint_run_dir / "markers").glob("*.json"))
        ]
        observed_data_ids = {
            str(marker["data_config_id"]) for marker in markers
        }
        expected_data_ids = {
            str(data_config_id) for data_config_id, _ in data_groups
        }
        if observed_data_ids != expected_data_ids:
            raise RuntimeError(
                "Incomplete 8-track checkpoint data coverage: "
                f"missing={sorted(expected_data_ids - observed_data_ids)}"
            )
        observed_fit_ids = {
            str(fit_id)
            for marker in markers
            for fit_id in marker["completed_fit_config_ids"]
        }
        expected_fit_ids = set(configurations["fit_config_id"].astype(str))
        if observed_fit_ids != expected_fit_ids:
            raise RuntimeError(
                "Incomplete 8-track checkpoint fit coverage: "
                f"missing={sorted(expected_fit_ids - observed_fit_ids)}"
            )
    if not materialize_checkpoint_results:
        return {
            **{
                name: pd.DataFrame(columns=schema)
                for name, schema in schemas.items()
            },
            "checkpoint_run_dir": checkpoint_run_dir,
        }
    return {
        name: _with_schema(
            pd.concat(parts, ignore_index=True, sort=False) if parts else None,
            schemas[name],
        )
        for name, parts in result_parts.items()
    }


def evaluate_internal_2way_tracks(
    oof_object_predictions: pd.DataFrame,
    oof_pixel_predictions: pd.DataFrame,
    configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate zero-margin 2-way decisions and secondary pixel votes."""
    config_rows = configurations.loc[
        configurations["decision_mode"].astype(str).eq("2way")
    ].drop_duplicates("evaluation_config_id")
    metric_parts: list[pd.DataFrame] = []
    vote_rows: list[dict[str, Any]] = []
    for config in config_rows.to_dict("records"):
        source = (
            oof_pixel_predictions
            if config["projection_level"] == "pixel_projection"
            else oof_object_predictions
        )
        projected = source.loc[
            source["projection_config_id"].astype(str).eq(
                str(config["projection_config_id"])
            )
        ].copy()
        if projected.empty:
            continue
        if not projected["direct_2way_decision"].eq(
            projected["simca_margin"].ge(0.0)
        ).all():
            raise RuntimeError("Direct 2-way decisions must use margin >= 0.")
        for fold_id, fold in projected.groupby("fold_id", sort=True):
            summary = summarize_binary_metrics_vectorized(
                fold,
                truth_col="truth",
                prediction_col="direct_2way_decision",
                group_levels=("object_id", "source_image"),
            )
            summary["evaluation_config_id"] = str(
                config["evaluation_config_id"]
            )
            summary["evaluation_track"] = str(config["evaluation_track"])
            summary["track_id"] = str(config["track_id"])
            summary["fold_id"] = int(fold_id)
            summary["random_state"] = int(config["random_state"])
            summary["metric_role"] = np.where(
                summary["aggregation_level"].eq("macro_source_image"),
                "calibration_primary",
                "diagnostic",
            )
            metric_parts.append(summary)
            if config["projection_level"] != "pixel_projection":
                continue
            objects = (
                fold.groupby(
                    ["source_image", "object_id", "batch"],
                    as_index=False,
                    sort=False,
                )
                .agg(
                    truth=("truth", "first"),
                    pixel_target_ratio=("direct_2way_decision", "mean"),
                )
            )
            thresholds = np.asarray(
                expcfg.INTERNAL_CALIBRATION_OBJECT_THRESHOLDS,
                dtype=float,
            )
            prediction_matrix = (
                objects["pixel_target_ratio"].to_numpy()[:, None]
                >= thresholds[None, :]
            )
            for threshold_index, threshold in enumerate(thresholds):
                summary_object = summarize_binary_metrics_vectorized(
                    prediction_matrix[:, threshold_index],
                    objects["truth"].to_numpy(dtype=bool),
                ).iloc[0]
                vote_rows.append(
                    {
                        "evaluation_config_id": str(
                            config["evaluation_config_id"]
                        ),
                        "evaluation_track": str(config["evaluation_track"]),
                        "track_id": str(config["track_id"]),
                        "fold_id": int(fold_id),
                        "random_state": int(config["random_state"]),
                        "secondary_object_threshold": float(threshold),
                        "n_objects": int(len(objects)),
                        "target_miss_rate": float(
                            summary_object["target_miss_rate"]
                        ),
                        "false_accept_rate": float(
                            summary_object["false_accept_rate"]
                        ),
                        "balanced_accuracy": float(
                            summary_object["balanced_accuracy"]
                        ),
                    }
                )
    metrics = _with_schema(
        pd.concat(metric_parts, ignore_index=True, sort=False)
        if metric_parts
        else None,
        expcfg.INTERNAL_CALIBRATION_2WAY_METRIC_V2_COLUMNS,
    )
    votes = _with_schema(
        pd.DataFrame(vote_rows),
        expcfg.INTERNAL_CALIBRATION_PIXEL_TO_OBJECT_2WAY_COLUMNS,
    )
    return metrics, votes


def _three_way_metric_grid(
    scores: np.ndarray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    score = np.asarray(scores, dtype=float).reshape(-1, 1)
    target = np.asarray(truth, dtype=bool).reshape(-1, 1)
    lower = np.asarray(lower, dtype=float).reshape(1, -1)
    upper = np.asarray(upper, dtype=float).reshape(1, -1)
    predicted_target = score >= upper
    predicted_non_target = score <= lower
    uncertain = ~(predicted_target | predicted_non_target)
    n_target = int(target.sum())
    n_non_target = int((~target).sum())
    miss = np.sum(target & predicted_non_target, axis=0) / max(n_target, 1)
    false_accept = np.sum(~target & predicted_target, axis=0) / max(
        n_non_target, 1
    )
    uncertain_rate = uncertain.mean(axis=0)
    target_decided = np.sum(target & ~uncertain, axis=0)
    non_target_decided = np.sum(~target & ~uncertain, axis=0)
    sensitivity = np.divide(
        np.sum(target & predicted_target, axis=0),
        target_decided,
        out=np.full(lower.shape[1], np.nan),
        where=target_decided > 0,
    )
    specificity = np.divide(
        np.sum(~target & predicted_non_target, axis=0),
        non_target_decided,
        out=np.full(lower.shape[1], np.nan),
        where=non_target_decided > 0,
    )
    return pd.DataFrame(
        {
            "three_way_lower_threshold": lower.ravel(),
            "three_way_upper_threshold": upper.ravel(),
            "target_miss_rate": miss,
            "false_accept_rate": false_accept,
            "uncertain_rate": uncertain_rate,
            "coverage_rate": 1.0 - uncertain_rate,
            "decided_balanced_accuracy": np.nanmean(
                np.vstack([sensitivity, specificity]), axis=0
            ),
        }
    )


def _margin_threshold_pairs(
    scores: Sequence[float],
    *,
    lower_quantiles: Sequence[float],
    upper_quantiles: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    negative = values[np.isfinite(values) & (values < 0.0)]
    positive = values[np.isfinite(values) & (values > 0.0)]
    if negative.size == 0 or positive.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    lower = np.unique(np.quantile(negative, lower_quantiles))
    upper = np.unique(np.quantile(positive, upper_quantiles))
    lower_grid, upper_grid = np.meshgrid(lower, upper, indexing="ij")
    valid = (lower_grid < 0.0) & (upper_grid > 0.0)
    return lower_grid[valid], upper_grid[valid]


def _centered_threshold_pairs(
    scores: Sequence[float],
    *,
    center: float,
    lower_quantiles: Sequence[float],
    upper_quantiles: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    lower_support = values[np.isfinite(values) & (values < float(center))]
    upper_support = values[np.isfinite(values) & (values > float(center))]
    if lower_support.size == 0 or upper_support.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    lower = np.unique(np.quantile(lower_support, lower_quantiles))
    upper = np.unique(np.quantile(upper_support, upper_quantiles))
    lower_grid, upper_grid = np.meshgrid(lower, upper, indexing="ij")
    valid = lower_grid < upper_grid
    return lower_grid[valid], upper_grid[valid]


def _crossfit_three_way_scope(
    projected: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    score_col: str,
    score_type: str,
    decision_scope: str,
    lower_quantiles: Sequence[float],
    upper_quantiles: Sequence[float],
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    fold_ids = sorted(projected["fold_id"].astype(int).unique())
    objective_cols = [
        "target_miss_rate",
        "false_accept_rate",
        "uncertain_rate",
        "coverage_rate",
        "decided_balanced_accuracy",
    ]
    for evaluation_fold in [*fold_ids, -1]:
        threshold_train = (
            projected.loc[~projected["fold_id"].eq(evaluation_fold)]
            if evaluation_fold >= 0
            else projected
        )
        threshold_eval = (
            projected.loc[projected["fold_id"].eq(evaluation_fold)]
            if evaluation_fold >= 0
            else projected
        )
        if score_type == "simca_margin":
            lower, upper = _margin_threshold_pairs(
                threshold_train[score_col],
                lower_quantiles=lower_quantiles,
                upper_quantiles=upper_quantiles,
            )
            missing_support = "missing_signed_margin_support"
        else:
            lower, upper = _centered_threshold_pairs(
                threshold_train[score_col],
                center=expcfg.INTERNAL_CALIBRATION_PIXEL_VOTE_CENTER,
                lower_quantiles=lower_quantiles,
                upper_quantiles=upper_quantiles,
            )
            missing_support = "missing_pixel_vote_support"
        if lower.size == 0:
            rows.append(
                pd.DataFrame(
                    [{
                        "evaluation_config_id": config["evaluation_config_id"],
                        "evaluation_track": config["evaluation_track"],
                        "track_id": config["track_id"],
                        "evaluation_fold": int(evaluation_fold),
                        "random_state": int(config["random_state"]),
                        "decision_scope": decision_scope,
                        "score_type": score_type,
                        "feasible": False,
                        "pareto_front": False,
                        "selected": False,
                        "failure_reason": missing_support,
                    }]
                )
            )
            continue
        training_grid = _three_way_metric_grid(
            threshold_train[score_col].to_numpy(),
            threshold_train["truth"].to_numpy(dtype=bool),
            lower,
            upper,
        )
        finite = np.isfinite(
            training_grid[objective_cols].to_numpy(dtype=float)
        ).all(axis=1)
        training_grid = training_grid.loc[finite].copy()
        if training_grid.empty:
            raise ValueError("Pareto objectives must all be finite.")
        feasible = (
            training_grid["target_miss_rate"].le(
                expcfg.INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE
            )
            & training_grid["false_accept_rate"].le(
                expcfg.INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE
            )
            & training_grid["uncertain_rate"].le(
                expcfg.INTERNAL_CALIBRATION_MAX_UNCERTAIN_RATE
            )
            & training_grid["coverage_rate"].ge(
                expcfg.INTERNAL_CALIBRATION_MIN_COVERAGE
            )
        )
        feasible_grid = training_grid.loc[feasible].copy()
        if feasible_grid.empty:
            evaluated = training_grid.head(1).copy()
            evaluated[objective_cols] = np.nan
            evaluated["feasible"] = False
            evaluated["pareto_front"] = False
            evaluated["selected"] = False
            evaluated["failure_reason"] = "no_feasible_threshold_pair"
        else:
            pareto = pareto_front_by_group(
                feasible_grid.assign(_group="thresholds"),
                group_cols=["_group"],
                minimize_cols=[
                    "target_miss_rate",
                    "false_accept_rate",
                    "uncertain_rate",
                ],
                maximize_cols=[
                    "coverage_rate",
                    "decided_balanced_accuracy",
                ],
            ).drop(columns="_group")
            evaluated = _three_way_metric_grid(
                threshold_eval[score_col].to_numpy(),
                threshold_eval["truth"].to_numpy(dtype=bool),
                pareto["three_way_lower_threshold"].to_numpy(),
                pareto["three_way_upper_threshold"].to_numpy(),
            )
            evaluated["feasible"] = True
            evaluated["pareto_front"] = True
            evaluated["selected"] = True
            evaluated["failure_reason"] = ""
        evaluated["evaluation_config_id"] = str(
            config["evaluation_config_id"]
        )
        evaluated["evaluation_track"] = str(config["evaluation_track"])
        evaluated["track_id"] = str(config["track_id"])
        evaluated["evaluation_fold"] = int(evaluation_fold)
        evaluated["random_state"] = int(config["random_state"])
        evaluated["decision_scope"] = decision_scope
        evaluated["score_type"] = score_type
        rows.append(evaluated)
    return rows


def _summarize_crossfitted_three_way_thresholds(
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    study_keys = [
        "evaluation_track",
        "track_id",
        "random_state",
        "decision_scope",
        "score_type",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    ]
    if thresholds.empty:
        return pd.DataFrame()
    crossfit = (
        thresholds.loc[thresholds["evaluation_fold"].ge(0)]
        .groupby(study_keys, dropna=False, as_index=False)
        .agg(
            n_evaluation_folds=("evaluation_fold", "nunique"),
            crossfit_target_miss_rate=("target_miss_rate", "mean"),
            crossfit_false_accept_rate=("false_accept_rate", "mean"),
            crossfit_uncertain_rate=("uncertain_rate", "mean"),
        )
    )
    all_oof = (
        thresholds.loc[thresholds["evaluation_fold"].eq(-1)]
        .groupby(study_keys, dropna=False, as_index=False)
        .agg(all_oof_selected=("selected", "max"))
    )
    return crossfit.merge(
        all_oof,
        on=study_keys,
        how="outer",
        validate="one_to_one",
    )


def evaluate_crossfitted_three_way_thresholds(
    oof_object_predictions: pd.DataFrame,
    oof_pixel_predictions: pd.DataFrame,
    configurations: pd.DataFrame,
    *,
    lower_quantiles: Sequence[float] = (
        expcfg.INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES
    ),
    upper_quantiles: Sequence[float] = (
        expcfg.INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit signed-margin thresholds, then recalibrate on all OOF rows."""
    rows: list[pd.DataFrame] = []
    config_rows = configurations.loc[
        configurations["decision_mode"].astype(str).eq("3way")
    ].drop_duplicates("evaluation_config_id")
    for config in config_rows.to_dict("records"):
        source = (
            oof_pixel_predictions
            if config["projection_level"] == "pixel_projection"
            else oof_object_predictions
        )
        projected = source.loc[
            source["projection_config_id"].astype(str).eq(
                str(config["projection_config_id"])
            )
        ].copy()
        if projected.empty:
            continue
        rows.extend(
            _crossfit_three_way_scope(
                projected,
                config,
                score_col="simca_margin",
                score_type="simca_margin",
                decision_scope="direct",
                lower_quantiles=lower_quantiles,
                upper_quantiles=upper_quantiles,
            )
        )
        if config["projection_level"] == "pixel_projection":
            object_votes = (
                projected.groupby(
                    ["fold_id", "source_image", "object_id"],
                    as_index=False,
                    sort=False,
                )
                .agg(
                    truth=("truth", "first"),
                    pixel_vote_ratio=("direct_2way_decision", "mean"),
                )
            )
            rows.extend(
                _crossfit_three_way_scope(
                    object_votes,
                    config,
                    score_col="pixel_vote_ratio",
                    score_type="pixel_vote_ratio",
                    decision_scope="derived_pixel_to_object",
                    lower_quantiles=lower_quantiles,
                    upper_quantiles=upper_quantiles,
                )
            )
    thresholds = _with_schema(
        pd.concat(rows, ignore_index=True, sort=False) if rows else None,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_V2_COLUMNS,
    )
    return thresholds, _summarize_crossfitted_three_way_thresholds(
        thresholds
    )


def summarize_internal_calibration_checkpoint_8tracks(
    checkpoint_run_dir: str | Path,
    configurations: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Reduce OOF checkpoint shards one at a time with bounded memory.

    Raw candidate-level OOF rows remain in the checksum-protected checkpoint
    shards.  Only the compact diagnostics and calibration tables are retained
    in memory, which is sufficient for notebook 03B stages 6--9.
    """
    run_dir = Path(checkpoint_run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = str(run_manifest.get("signature", ""))
    if not signature:
        raise RuntimeError("Checkpoint run manifest has no signature.")

    configurations = _attach_data_configuration_ids(configurations)
    expected_data_ids = set(configurations["data_config_id"].astype(str))
    marker_paths = sorted((run_dir / "markers").glob("*.json"))
    markers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in marker_paths
    ]
    observed_data_ids = {
        str(marker.get("data_config_id", "")) for marker in markers
    }
    if observed_data_ids != expected_data_ids:
        raise RuntimeError(
            "Incomplete checkpoint coverage for streaming reduction: "
            f"missing={sorted(expected_data_ids - observed_data_ids)}"
        )

    shard_schemas = {
        "fit_diagnostics": expcfg.INTERNAL_CALIBRATION_FIT_DIAGNOSTIC_V2_COLUMNS,
        "rule_diagnostics": expcfg.INTERNAL_CALIBRATION_RULE_DIAGNOSTIC_V2_COLUMNS,
        "oof_object_predictions": expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_V2_COLUMNS,
        "oof_pixel_predictions": expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_V2_COLUMNS,
        "projection_shift": expcfg.INTERNAL_CALIBRATION_PROJECTION_SHIFT_COLUMNS,
        "technical_errors": expcfg.INTERNAL_CALIBRATION_AUDIT_V2_COLUMNS,
    }
    retained_names = (
        "fit_diagnostics",
        "rule_diagnostics",
        "projection_shift",
        "technical_errors",
    )
    retained_parts: dict[str, list[pd.DataFrame]] = {
        name: [] for name in retained_names
    }
    metric_parts: list[pd.DataFrame] = []
    vote_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []

    for marker_index, marker in enumerate(markers, start=1):
        if str(marker.get("signature")) != signature:
            raise RuntimeError("Checkpoint marker signature mismatch.")
        data_config_id = str(marker["data_config_id"])
        tables = {
            name: pd.DataFrame(columns=schema)
            for name, schema in shard_schemas.items()
        }
        for shard in marker.get("shards", []):
            name = str(shard["name"])
            if name not in shard_schemas:
                raise RuntimeError(f"Unknown checkpoint shard type: {name}")
            path = run_dir / str(shard["relative_path"])
            _validate_8track_checkpoint_shard(
                path,
                shard,
                shard_schemas[name],
            )
            tables[name] = _with_schema(
                pd.read_parquet(path),
                shard_schemas[name],
                copy=False,
            )
        for name in retained_names:
            if not tables[name].empty:
                retained_parts[name].append(tables[name])

        data_configurations = configurations.loc[
            configurations["data_config_id"].astype(str).eq(data_config_id)
        ]
        if data_configurations.empty:
            raise RuntimeError(
                f"No configuration rows for checkpoint {data_config_id}."
            )
        metrics, votes = evaluate_internal_2way_tracks(
            tables["oof_object_predictions"],
            tables["oof_pixel_predictions"],
            data_configurations,
        )
        thresholds, _ = evaluate_crossfitted_three_way_thresholds(
            tables["oof_object_predictions"],
            tables["oof_pixel_predictions"],
            data_configurations,
            lower_quantiles=(
                expcfg.INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES
            ),
            upper_quantiles=(
                expcfg.INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES
            ),
        )
        if not metrics.empty:
            metric_parts.append(metrics)
        if not votes.empty:
            vote_parts.append(votes)
        if not thresholds.empty:
            threshold_parts.append(thresholds)
        if verbose:
            print(
                f"[03B reduce {marker_index}/{len(markers)}] "
                f"{data_config_id}"
            )
        del tables, metrics, votes, thresholds
        gc.collect()

    thresholds_3way = _with_schema(
        pd.concat(threshold_parts, ignore_index=True, sort=False)
        if threshold_parts else None,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_V2_COLUMNS,
    )
    result = {
        name: _with_schema(
            pd.concat(parts, ignore_index=True, sort=False) if parts else None,
            shard_schemas[name],
        )
        for name, parts in retained_parts.items()
    }
    result.update(
        {
            "oof_object_predictions": pd.DataFrame(
                columns=expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_V2_COLUMNS
            ),
            "oof_pixel_predictions": pd.DataFrame(
                columns=expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_V2_COLUMNS
            ),
            "metrics_2way": _with_schema(
                pd.concat(metric_parts, ignore_index=True, sort=False)
                if metric_parts else None,
                expcfg.INTERNAL_CALIBRATION_2WAY_METRIC_V2_COLUMNS,
            ),
            "pixel_votes_2way": _with_schema(
                pd.concat(vote_parts, ignore_index=True, sort=False)
                if vote_parts else None,
                expcfg.INTERNAL_CALIBRATION_PIXEL_TO_OBJECT_2WAY_COLUMNS,
            ),
            "thresholds_3way": thresholds_3way,
            "thresholds_3way_study": (
                _summarize_crossfitted_three_way_thresholds(
                    thresholds_3way
                )
            ),
        }
    )
    return result


def load_selected_oof_predictions_from_checkpoint_8tracks(
    checkpoint_run_dir: str | Path,
    calibration_domain: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the calibrated OOF projections from candidate checkpoints.

    The full candidate grid remains split across the per-data-configuration
    checkpoint shards.  Once the eight tracks have been calibrated, this
    loader filters those shards by ``projection_config_id`` before converting
    Arrow data to pandas, so downstream notebooks receive only the selected
    projections instead of the complete search grid.
    """
    required_domain = {
        "projection_config_id",
        "fit_config_id",
        "projection_level",
    }
    missing = sorted(required_domain - set(calibration_domain.columns))
    if missing:
        raise KeyError(f"Missing calibration-domain columns: {missing}")
    if calibration_domain.empty:
        raise RuntimeError("The calibration domain is empty.")

    selected = calibration_domain.loc[
        :,
        ["projection_config_id", "fit_config_id", "projection_level"],
    ].drop_duplicates()
    level_by_projection = selected.set_index("projection_config_id")[
        "projection_level"
    ].astype(str)
    if level_by_projection.index.duplicated().any():
        raise RuntimeError(
            "A selected projection_config_id has multiple projection levels."
        )
    selected_ids = {
        "oof_object_predictions": set(
            level_by_projection.loc[
                ~level_by_projection.eq("pixel_projection")
            ].index.astype(str)
        ),
        "oof_pixel_predictions": set(
            level_by_projection.loc[
                level_by_projection.eq("pixel_projection")
            ].index.astype(str)
        ),
    }
    selected_fit_ids = set(selected["fit_config_id"].astype(str))
    schemas = {
        "oof_object_predictions": (
            expcfg.INTERNAL_CALIBRATION_OOF_OBJECT_V2_COLUMNS
        ),
        "oof_pixel_predictions": (
            expcfg.INTERNAL_CALIBRATION_OOF_PIXEL_V2_COLUMNS
        ),
    }
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in schemas}

    run_dir = Path(checkpoint_run_dir)
    marker_dir = run_dir / "markers"
    if not marker_dir.exists():
        raise FileNotFoundError(marker_dir)
    for marker_path in sorted(marker_dir.glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_fit_ids = set(
            map(str, marker.get("completed_fit_config_ids", ()))
        )
        if not marker_fit_ids:
            marker_fit_ids = {
                str(fit_id)
                for shard in marker.get("shards", ())
                for fit_id in shard.get("completed_fit_config_ids", ())
            }
        if marker_fit_ids.isdisjoint(selected_fit_ids):
            continue
        for shard in marker.get("shards", ()):
            name = str(shard["name"])
            wanted = selected_ids.get(name, set())
            if not wanted:
                continue
            path = run_dir / str(shard["relative_path"])
            _validate_8track_checkpoint_shard(path, shard, schemas[name])
            table = pq.read_table(
                path,
                filters=[("projection_config_id", "in", sorted(wanted))],
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
            pd.concat(table_parts, ignore_index=True, sort=False)
            if table_parts else None,
            schema,
        )
        for name, schema in schemas.items()
        for table_parts in (parts[name],)
    }
    for name, expected_ids in selected_ids.items():
        observed_ids = set(outputs[name]["projection_config_id"].astype(str))
        if observed_ids != expected_ids:
            raise RuntimeError(
                f"Selected OOF checkpoint coverage is incomplete for {name}: "
                f"missing={sorted(expected_ids - observed_ids)}, "
                f"unexpected={sorted(observed_ids - expected_ids)}"
            )
    return (
        outputs["oof_object_predictions"],
        outputs["oof_pixel_predictions"],
    )


def select_smallest_plateau_components(
    fold_metrics: pd.DataFrame,
    configurations: pd.DataFrame,
    *,
    rule_diagnostics: pd.DataFrame | None = None,
    fn_column: str = "fn_rate",
    fp_column: str = "fp_rate",
    balanced_accuracy_column: str = "balanced_accuracy",
    expected_n_folds: int | None = None,
    max_fn_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FN_RATE,
    max_fp_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FP_RATE,
    min_balanced_accuracy: float = expcfg.INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY,
    max_fold_balanced_accuracy_std: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_FOLD_BALANCED_ACCURACY_STD
    ),
    max_limit_relative_std: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_LIMIT_RELATIVE_STD
    ),
    max_train_validation_rejection_gap: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_TRAIN_VALIDATION_REJECTION_GAP
    ),
    max_train_rejection_alpha_gap: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_TRAIN_REJECTION_ALPHA_GAP
    ),
    tolerance: float = expcfg.INTERNAL_CALIBRATION_PERFORMANCE_PLATEAU_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the smallest robust ``k`` whose future gains are immaterial.

    Random seeds are evaluation replicates, not hyperparameters: they are
    aggregated before the component decision. The returned summary remains at
    configuration-row level so every seed belonging to a retained ``k`` can be
    rerun for threshold calibration.
    """
    if fold_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_metric_columns = {
        "config_id",
        fn_column,
        fp_column,
        balanced_accuracy_column,
    }
    missing_metrics = sorted(required_metric_columns - set(fold_metrics.columns))
    if missing_metrics:
        raise KeyError(f"Missing component-selection metrics: {missing_metrics}")

    config = configurations.drop_duplicates("config_id").copy()
    missing_parameters = [
        column for column in _INTERNAL_PARAMETER_COLUMNS if column not in config
    ]
    if missing_parameters:
        raise KeyError(
            "Missing component-selection configuration columns: "
            f"{missing_parameters}"
        )

    metrics = fold_metrics.copy()
    for column in (fn_column, fp_column, balanced_accuracy_column):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    if "fold_id" in metrics:
        per_config = (
            metrics.groupby("config_id", as_index=False)
            .agg(
                mean_fn_rate=(fn_column, "mean"),
                mean_fp_rate=(fp_column, "mean"),
                mean_balanced_accuracy=(balanced_accuracy_column, "mean"),
                worst_fold_fn_rate=(fn_column, "max"),
                worst_fold_fp_rate=(fp_column, "max"),
                worst_fold_balanced_accuracy=(
                    balanced_accuracy_column,
                    "min",
                ),
                std_fold_fn_rate=(fn_column, "std"),
                std_fold_fp_rate=(fp_column, "std"),
                std_fold_balanced_accuracy=(
                    balanced_accuracy_column,
                    "std",
                ),
                n_completed_folds=("fold_id", "nunique"),
            )
        )
    else:
        per_config = metrics.loc[
            :,
            ["config_id", fn_column, fp_column, balanced_accuracy_column],
        ].rename(
            columns={
                fn_column: "mean_fn_rate",
                fp_column: "mean_fp_rate",
                balanced_accuracy_column: "mean_balanced_accuracy",
            }
        )
        fallback_columns = {
            "worst_fold_fn_rate": ("max_fold_fn_rate", "mean_fn_rate"),
            "worst_fold_fp_rate": ("max_fold_fp_rate", "mean_fp_rate"),
            "worst_fold_balanced_accuracy": (
                "min_fold_balanced_accuracy",
                "mean_balanced_accuracy",
            ),
            "std_fold_fn_rate": ("std_fold_fn_rate",),
            "std_fold_fp_rate": ("std_fold_fp_rate",),
            "std_fold_balanced_accuracy": (
                "std_fold_balanced_accuracy",
                "std_fold_decided_balanced_accuracy",
            ),
            "n_completed_folds": ("n_folds",),
        }
        indexed_metrics = metrics.set_index("config_id")
        per_config = per_config.set_index("config_id")
        for output_column, source_columns in fallback_columns.items():
            values = pd.Series(np.nan, index=per_config.index, dtype=float)
            for source_column in source_columns:
                if source_column in indexed_metrics:
                    source = pd.to_numeric(
                        indexed_metrics[source_column],
                        errors="coerce",
                    )
                elif source_column in per_config:
                    source = pd.to_numeric(
                        per_config[source_column],
                        errors="coerce",
                    )
                else:
                    continue
                values = values.where(values.notna(), source)
            per_config[output_column] = values
        per_config = per_config.reset_index()

    if expected_n_folds is None:
        if "fold_id" in metrics:
            expected_n_folds = int(metrics["fold_id"].nunique())
        elif "n_folds" in metrics and metrics["n_folds"].notna().any():
            expected_n_folds = int(
                pd.to_numeric(metrics["n_folds"], errors="coerce").max()
            )
        else:
            expected_n_folds = 1
    per_config["n_completed_folds"] = pd.to_numeric(
        per_config["n_completed_folds"],
        errors="coerce",
    ).fillna(0).astype(int)
    for column in (
        "std_fold_fn_rate",
        "std_fold_fp_rate",
        "std_fold_balanced_accuracy",
    ):
        per_config[column] = pd.to_numeric(
            per_config[column],
            errors="coerce",
        ).fillna(0.0)

    if rule_diagnostics is not None and not rule_diagnostics.empty:
        diagnostics = rule_diagnostics.copy()
        for column in (
            "rule_limit",
            "train_rejection_rate",
            "validation_target_rejection_rate",
        ):
            diagnostics[column] = pd.to_numeric(
                diagnostics[column],
                errors="coerce",
            )
        per_rule = (
            diagnostics.groupby("config_id", as_index=False)
            .agg(
                calibrated_rule_limit=("rule_limit", "median"),
                limit_mean=("rule_limit", "mean"),
                limit_std=("rule_limit", lambda values: float(
                    np.nanstd(pd.to_numeric(values, errors="coerce"), ddof=0)
                )),
                mean_train_rejection_rate=("train_rejection_rate", "mean"),
                mean_validation_target_rejection_rate=(
                    "validation_target_rejection_rate",
                    "mean",
                ),
            )
        )
        per_rule["limit_relative_std"] = (
            per_rule["limit_std"]
            / per_rule["limit_mean"].abs().clip(lower=1e-12)
        )
        per_rule["train_validation_rejection_gap"] = (
            per_rule["mean_validation_target_rejection_rate"]
            - per_rule["mean_train_rejection_rate"]
        ).abs()
        per_config = per_config.merge(
            per_rule.drop(columns=["limit_mean", "limit_std"]),
            on="config_id",
            how="left",
            validate="one_to_one",
        )
    else:
        for column in (
            "calibrated_rule_limit",
            "limit_relative_std",
            "mean_train_rejection_rate",
            "mean_validation_target_rejection_rate",
            "train_validation_rejection_gap",
        ):
            per_config[column] = 0.0

    per_config = config.merge(
        per_config,
        on="config_id",
        how="left",
        validate="one_to_one",
    )
    per_config["train_rejection_alpha_gap"] = (
        pd.to_numeric(per_config["mean_train_rejection_rate"], errors="coerce")
        - pd.to_numeric(per_config["alpha"], errors="coerce")
    ).abs()
    group_columns = [
        column
        for column in _INTERNAL_PARAMETER_COLUMNS
        if column not in {"n_components", "random_state"}
    ]
    aggregate_columns = [*group_columns, "n_components"]
    numeric_aggregate_columns = (
        "mean_fn_rate",
        "mean_fp_rate",
        "mean_balanced_accuracy",
        "worst_fold_fn_rate",
        "worst_fold_fp_rate",
        "worst_fold_balanced_accuracy",
        "std_fold_fn_rate",
        "std_fold_fp_rate",
        "std_fold_balanced_accuracy",
        "limit_relative_std",
        "train_validation_rejection_gap",
        "train_rejection_alpha_gap",
        "calibrated_rule_limit",
        "n_completed_folds",
    )
    for column in numeric_aggregate_columns:
        per_config[column] = pd.to_numeric(
            per_config[column],
            errors="coerce",
        )
    per_config["_component_metrics_finite"] = per_config[
        ["mean_fn_rate", "mean_fp_rate", "mean_balanced_accuracy"]
    ].notna().all(axis=1)
    per_config["_component_fold_complete"] = (
        per_config["n_completed_folds"]
        .fillna(0)
        .eq(int(expected_n_folds))
    )
    aggregate = (
        per_config.groupby(
            aggregate_columns,
            dropna=False,
            sort=False,
            as_index=False,
        )
        .agg(
            mean_fn_rate=("mean_fn_rate", "mean"),
            mean_fp_rate=("mean_fp_rate", "mean"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
            worst_fold_fn_rate=("worst_fold_fn_rate", "max"),
            worst_fold_fp_rate=("worst_fold_fp_rate", "max"),
            worst_fold_balanced_accuracy=(
                "worst_fold_balanced_accuracy",
                "min",
            ),
            std_fold_fn_rate=("std_fold_fn_rate", "max"),
            std_fold_fp_rate=("std_fold_fp_rate", "max"),
            std_fold_balanced_accuracy=(
                "std_fold_balanced_accuracy",
                "max",
            ),
            limit_relative_std=("limit_relative_std", "max"),
            train_validation_rejection_gap=(
                "train_validation_rejection_gap",
                "max",
            ),
            train_rejection_alpha_gap=(
                "train_rejection_alpha_gap",
                "max",
            ),
            calibrated_rule_limit=("calibrated_rule_limit", "median"),
            n_completed_folds=("n_completed_folds", "min"),
            n_seeds_evaluated=("random_state", "nunique"),
            _all_metrics_finite=("_component_metrics_finite", "all"),
            _all_folds_complete=("_component_fold_complete", "all"),
        )
    )
    aggregate["n_completed_folds"] = (
        pd.to_numeric(
            aggregate["n_completed_folds"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    aggregate["component_complete"] = (
        aggregate.pop("_all_metrics_finite").astype(bool)
        & aggregate.pop("_all_folds_complete").astype(bool)
    )
    aggregate["component_group_id"] = [
        hash_internal_calibration_configuration(
            {column: row[column] for column in group_columns},
            prefix="ick",
        )
        for row in aggregate.to_dict("records")
    ]
    for column in (
        "limit_relative_std",
        "train_validation_rejection_gap",
        "train_rejection_alpha_gap",
    ):
        aggregate[column] = pd.to_numeric(
            aggregate[column],
            errors="coerce",
        ).fillna(0.0)
    aggregate["component_acceptable"] = (
        aggregate["component_complete"]
        & aggregate["mean_fn_rate"].le(float(max_fn_rate))
        & aggregate["mean_fp_rate"].le(float(max_fp_rate))
        & aggregate["mean_balanced_accuracy"].ge(float(min_balanced_accuracy))
        & aggregate["std_fold_balanced_accuracy"].le(
            float(max_fold_balanced_accuracy_std)
        )
        & aggregate["limit_relative_std"].le(float(max_limit_relative_std))
        & aggregate["train_validation_rejection_gap"].le(
            float(max_train_validation_rejection_gap)
        )
        & aggregate["train_rejection_alpha_gap"].le(
            float(max_train_rejection_alpha_gap)
        )
    )
    aggregate["component_plateau"] = False
    aggregate["component_selected"] = False
    aggregate["component_selection_status"] = "not_acceptable"
    aggregate["max_future_balanced_accuracy_gain"] = np.nan
    aggregate["max_future_fn_reduction"] = np.nan
    aggregate["max_future_fp_reduction"] = np.nan

    reverse_order = aggregate.sort_values(
        [*group_columns, "n_components"],
        ascending=[*[True] * len(group_columns), False],
        na_position="first",
        kind="mergesort",
    ).copy()
    acceptable = reverse_order["component_acceptable"].astype(bool)
    future_ba = (
        reverse_order["mean_balanced_accuracy"]
        .where(acceptable)
        .groupby(
            [reverse_order[column] for column in group_columns],
            dropna=False,
            sort=False,
        )
        .cummax()
    )
    future_fn = (
        reverse_order["mean_fn_rate"]
        .where(acceptable)
        .groupby(
            [reverse_order[column] for column in group_columns],
            dropna=False,
            sort=False,
        )
        .cummin()
    )
    future_fp = (
        reverse_order["mean_fp_rate"]
        .where(acceptable)
        .groupby(
            [reverse_order[column] for column in group_columns],
            dropna=False,
            sort=False,
        )
        .cummin()
    )
    ba_gain = (
        future_ba - reverse_order["mean_balanced_accuracy"]
    ).clip(lower=0.0)
    fn_reduction = (
        reverse_order["mean_fn_rate"] - future_fn
    ).clip(lower=0.0)
    fp_reduction = (
        reverse_order["mean_fp_rate"] - future_fp
    ).clip(lower=0.0)
    aggregate.loc[
        reverse_order.index,
        "max_future_balanced_accuracy_gain",
    ] = ba_gain.to_numpy()
    aggregate.loc[
        reverse_order.index,
        "max_future_fn_reduction",
    ] = fn_reduction.to_numpy()
    aggregate.loc[
        reverse_order.index,
        "max_future_fp_reduction",
    ] = fp_reduction.to_numpy()
    aggregate["component_plateau"] = (
        aggregate["component_acceptable"].astype(bool)
        & aggregate["max_future_balanced_accuracy_gain"].le(
            float(tolerance)
        )
        & aggregate["max_future_fn_reduction"].le(float(tolerance))
        & aggregate["max_future_fp_reduction"].le(float(tolerance))
    )
    selected_indices = (
        aggregate.loc[aggregate["component_plateau"]]
        .sort_values(
            [*group_columns, "n_components"],
            na_position="first",
            kind="mergesort",
        )
        .drop_duplicates(group_columns, keep="first")
        .index
    )
    aggregate.loc[selected_indices, "component_selected"] = True
    aggregate.loc[
        selected_indices,
        "component_selection_status",
    ] = "smallest_acceptable_plateau"
    other = (
        aggregate["component_acceptable"]
        & ~aggregate["component_selected"]
    )
    aggregate.loc[
        other,
        "component_selection_status",
    ] = "acceptable_but_not_selected"

    annotation_columns = [
        "component_group_id",
        "mean_fn_rate",
        "mean_fp_rate",
        "mean_balanced_accuracy",
        "worst_fold_fn_rate",
        "worst_fold_fp_rate",
        "worst_fold_balanced_accuracy",
        "std_fold_fn_rate",
        "std_fold_fp_rate",
        "std_fold_balanced_accuracy",
        "limit_relative_std",
        "train_validation_rejection_gap",
        "train_rejection_alpha_gap",
        "calibrated_rule_limit",
        "n_completed_folds",
        "n_seeds_evaluated",
        "component_complete",
        "component_acceptable",
        "component_plateau",
        "component_selected",
        "component_selection_status",
        "max_future_balanced_accuracy_gain",
        "max_future_fn_reduction",
        "max_future_fp_reduction",
    ]
    summary = config.merge(
        aggregate[[*group_columns, "n_components", *annotation_columns]],
        on=[*group_columns, "n_components"],
        how="left",
        validate="many_to_one",
    )
    selected = summary.loc[summary["component_selected"]].copy()
    return summary.reset_index(drop=True), selected.reset_index(drop=True)


_OOF_PREDICTION_EQUIVALENCE_COLUMNS = (
    "config_id",
    "representative_config_id",
    "prediction_signature",
    "equivalence_size",
)


def _canonical_oof_prediction_group(group: pd.DataFrame) -> pd.DataFrame:
    identity_columns = (
        "fold_id",
        "source_image",
        "object_id",
        "batch",
        "true_target_object",
    )
    missing = [
        column
        for column in (*identity_columns, "target_pixel_ratio")
        if column not in group
    ]
    if missing:
        raise KeyError(
            "Missing columns for exact OOF prediction signatures: "
            f"{missing}"
        )
    canonical = (
        group.loc[:, [*identity_columns, "target_pixel_ratio"]]
        .sort_values(
            ["fold_id", "source_image", "object_id"],
            kind="mergesort",
            na_position="first",
        )
        .reset_index(drop=True)
    )
    duplicated = canonical.duplicated(
        ["fold_id", "source_image", "object_id"],
        keep=False,
    )
    if duplicated.any():
        raise ValueError(
            "OOF prediction signatures require one row per "
            "(fold_id, source_image, object_id)."
        )
    return canonical


def build_exact_oof_prediction_equivalence(
    oof_objects: pd.DataFrame,
) -> pd.DataFrame:
    """Map strictly identical OOF ratio vectors to one representative.

    Object identity is canonicalized first. Ratios are then hashed from their
    exact little-endian IEEE-754 float64 bytes, without rounding. Hash buckets
    are verified against the exact identity token and ratio bytes before two
    configurations are declared equivalent.
    """
    required = {"config_id", "target_pixel_ratio"}
    missing = sorted(required - set(oof_objects.columns))
    if missing:
        raise KeyError(
            "Missing columns for OOF prediction equivalence: "
            f"{missing}"
        )
    if oof_objects.empty:
        return pd.DataFrame(columns=_OOF_PREDICTION_EQUIVALENCE_COLUMNS)
    if oof_objects["config_id"].isna().any():
        raise ValueError("OOF config_id values must not be missing.")

    identity_columns = [
        "fold_id",
        "source_image",
        "object_id",
        "batch",
        "true_target_object",
    ]
    missing_identity = [
        column
        for column in identity_columns
        if column not in oof_objects
    ]
    if missing_identity:
        raise KeyError(
            "Missing columns for exact OOF prediction signatures: "
            f"{missing_identity}"
        )

    identity_payloads: dict[str, bytes] = {}
    exact_payload_by_signature: dict[str, tuple[str, bytes]] = {}
    representative_by_signature: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    def identity_token_for(canonical: pd.DataFrame) -> str:
        identity_records = [
            [_json_scalar(value) for value in record]
            for record in canonical[identity_columns].itertuples(
                index=False,
                name=None,
            )
        ]
        identity_payload = json.dumps(
            identity_records,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        identity_digest = hashlib.sha256(identity_payload).hexdigest()
        identity_token = identity_digest
        collision_index = 1
        while (
            identity_token in identity_payloads
            and identity_payloads[identity_token] != identity_payload
        ):
            identity_token = f"{identity_digest}:{collision_index}"
            collision_index += 1
        identity_payloads.setdefault(identity_token, identity_payload)
        return identity_token

    def register_prediction(
        config_id: Any,
        identity_token: str,
        ratio_bytes: bytes,
    ) -> None:
        signature_digest = hashlib.sha256(
            identity_token.encode("ascii") + ratio_bytes
        ).hexdigest()
        signature = f"oofsha256_{signature_digest}"
        collision_index = 1
        exact_payload = (identity_token, ratio_bytes)
        while (
            signature in exact_payload_by_signature
            and exact_payload_by_signature[signature] != exact_payload
        ):
            signature = (
                f"oofsha256_{signature_digest}:{collision_index}"
            )
            collision_index += 1
        if signature not in exact_payload_by_signature:
            exact_payload_by_signature[signature] = exact_payload
            representative_by_signature[signature] = config_id
        rows.append(
            {
                "config_id": config_id,
                "representative_config_id": (
                    representative_by_signature[signature]
                ),
                "prediction_signature": signature,
            }
        )

    # Fast path for the 03B contract: every configuration predicts the exact
    # same OOF object universe. Map all rows to one canonical object index once
    # instead of sorting and serializing the 209 identities 11,000 times.
    config_order = pd.Index(oof_objects["config_id"].drop_duplicates())
    first_config_id = config_order[0]
    first_group = oof_objects.loc[
        oof_objects["config_id"].eq(first_config_id)
    ]
    canonical_template = _canonical_oof_prediction_group(first_group)
    object_key_columns = ["fold_id", "source_image", "object_id"]
    template_keys = pd.MultiIndex.from_frame(
        canonical_template[object_key_columns]
    )
    observed_keys = pd.MultiIndex.from_frame(
        oof_objects[object_key_columns]
    )
    object_positions = template_keys.get_indexer(observed_keys)
    config_codes = pd.Categorical(
        oof_objects["config_id"],
        categories=config_order,
        ordered=True,
    ).codes
    n_configurations = len(config_order)
    n_objects = len(canonical_template)
    counts_complete = np.array_equal(
        np.bincount(
            config_codes,
            minlength=n_configurations,
        ),
        np.full(n_configurations, n_objects, dtype=int),
    )
    keys_complete = bool(np.all(object_positions >= 0))
    metadata_complete = False
    positions_unique = False
    if counts_complete and keys_complete:
        flat_positions = (
            config_codes.astype(np.int64) * int(n_objects)
            + object_positions.astype(np.int64)
        )
        positions_unique = bool(
            np.all(
                np.bincount(
                    flat_positions,
                    minlength=n_configurations * n_objects,
                )
                == 1
            )
        )
        metadata_complete = True
        for column in ("batch", "true_target_object"):
            observed = oof_objects[column].astype(object).to_numpy()
            expected = (
                canonical_template[column]
                .astype(object)
                .to_numpy()[object_positions]
            )
            equal = np.asarray(observed == expected, dtype=bool)
            both_missing = pd.isna(observed) & pd.isna(expected)
            if not np.all(equal | both_missing):
                metadata_complete = False
                break

    if (
        counts_complete
        and keys_complete
        and positions_unique
        and metadata_complete
    ):
        identity_token = identity_token_for(canonical_template)
        ratio_values = pd.to_numeric(
            oof_objects["target_pixel_ratio"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        ratio_matrix = np.empty(
            (n_configurations, n_objects),
            dtype="<f8",
        )
        ratio_matrix[config_codes, object_positions] = ratio_values
        for position, config_id in enumerate(config_order):
            register_prediction(
                config_id,
                identity_token,
                np.ascontiguousarray(ratio_matrix[position]).tobytes(),
            )

        equivalence = pd.DataFrame(rows)
        sizes = equivalence.groupby(
            "representative_config_id",
            dropna=False,
        )["config_id"].transform("size")
        equivalence["equivalence_size"] = sizes.astype(int)
        return equivalence.loc[
            :,
            list(_OOF_PREDICTION_EQUIVALENCE_COLUMNS),
        ].reset_index(drop=True)

    for config_id, group in oof_objects.groupby("config_id", sort=False):
        canonical = _canonical_oof_prediction_group(group)
        identity_token = identity_token_for(canonical)
        ratios = pd.to_numeric(
            canonical["target_pixel_ratio"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        ratio_bytes = np.ascontiguousarray(
            ratios.astype("<f8", copy=False)
        ).tobytes()
        register_prediction(config_id, identity_token, ratio_bytes)

    equivalence = pd.DataFrame(rows)
    sizes = equivalence.groupby(
        "representative_config_id",
        dropna=False,
    )["config_id"].transform("size")
    equivalence["equivalence_size"] = sizes.astype(int)
    return equivalence.loc[
        :,
        list(_OOF_PREDICTION_EQUIVALENCE_COLUMNS),
    ].reset_index(drop=True)


def _resolve_oof_prediction_equivalence(
    oof_objects: pd.DataFrame,
    prediction_equivalence: pd.DataFrame | None,
) -> pd.DataFrame:
    equivalence = (
        build_exact_oof_prediction_equivalence(oof_objects)
        if prediction_equivalence is None
        else prediction_equivalence.copy()
    )
    required = {"config_id", "representative_config_id"}
    missing = sorted(required - set(equivalence.columns))
    if missing:
        raise KeyError(
            "Missing OOF prediction-equivalence columns: "
            f"{missing}"
        )
    if equivalence["config_id"].duplicated().any():
        raise ValueError(
            "OOF prediction equivalence must contain one row per config_id."
        )
    observed_ids = set(oof_objects["config_id"].dropna())
    mapped_ids = set(equivalence["config_id"].dropna())
    if observed_ids != mapped_ids:
        missing_ids = len(observed_ids - mapped_ids)
        extra_ids = len(mapped_ids - observed_ids)
        raise ValueError(
            "OOF prediction equivalence does not match the input "
            f"configurations (missing={missing_ids}, extra={extra_ids})."
        )
    representative_ids = set(equivalence["representative_config_id"])
    if not representative_ids.issubset(observed_ids):
        raise ValueError(
            "Every representative_config_id must exist in the OOF table."
        )
    representative_rows = equivalence.loc[
        equivalence["config_id"].isin(representative_ids),
        ["config_id", "representative_config_id"],
    ]
    if not representative_rows["config_id"].eq(
        representative_rows["representative_config_id"]
    ).all():
        raise ValueError(
            "Every OOF representative must map to itself."
        )
    return equivalence.reset_index(drop=True)


def _expand_representative_threshold_metrics(
    representative_metrics: pd.DataFrame,
    equivalence: pd.DataFrame,
) -> pd.DataFrame:
    if representative_metrics.empty:
        return representative_metrics.copy()
    metrics = representative_metrics.rename(
        columns={"config_id": "_representative_config_id"}
    )
    expanded = equivalence[
        ["config_id", "representative_config_id"]
    ].merge(
        metrics,
        left_on="representative_config_id",
        right_on="_representative_config_id",
        how="left",
        validate="many_to_many",
        sort=False,
    )
    return expanded.drop(
        columns=["representative_config_id", "_representative_config_id"]
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


def _position_groups(values: np.ndarray) -> list[np.ndarray]:
    series = pd.Series(values).reset_index(drop=True)
    return [
        np.asarray(positions, dtype=int)
        for positions in series.groupby(
            series,
            dropna=False,
            sort=False,
        ).indices.values()
    ]


def _finite_extreme(
    arrays: Sequence[np.ndarray],
    *,
    reducer: str,
    n_values: int,
) -> np.ndarray:
    if not arrays:
        return np.full(n_values, np.nan, dtype=float)
    values = np.vstack(arrays).astype(float, copy=False)
    finite = np.isfinite(values)
    available = finite.any(axis=0)
    fill_value = -np.inf if reducer == "max" else np.inf
    filled = np.where(finite, values, fill_value)
    reduced = (
        np.max(filled, axis=0)
        if reducer == "max"
        else np.min(filled, axis=0)
    )
    return np.where(available, reduced, np.nan)


def _finite_population_std(
    arrays: Sequence[np.ndarray],
    *,
    n_values: int,
) -> np.ndarray:
    if not arrays:
        return np.full(n_values, np.nan, dtype=float)
    values = np.vstack(arrays).astype(float, copy=False)
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    sums = np.sum(np.where(finite, values, 0.0), axis=0)
    means = _safe_ratio_array(sums, counts)
    squared = np.where(finite, (values - means) ** 2, 0.0)
    variance = _safe_ratio_array(np.sum(squared, axis=0), counts)
    return np.sqrt(variance)


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


def _evaluate_fixed_thresholds_by_fold(
    oof_objects: pd.DataFrame,
    *,
    decision_mode: str,
    object_threshold_column: str = "object_threshold",
    lower_threshold_column: str = "three_way_lower_threshold",
    upper_threshold_column: str = "three_way_upper_threshold",
    target_class: str,
    non_target_label: str,
) -> pd.DataFrame:
    """Evaluate one already-calibrated threshold set per config and fold."""
    mode = str(decision_mode).lower()
    required = {
        "config_id",
        "fold_id",
        "target_pixel_ratio",
        "true_target_object",
    }
    if mode == "2way":
        required.add(object_threshold_column)
    elif mode == "3way":
        required.update(
            {lower_threshold_column, upper_threshold_column}
        )
    else:
        raise ValueError("decision_mode must be '2way' or '3way'.")
    missing = sorted(required - set(oof_objects.columns))
    if missing:
        raise KeyError(f"Missing fixed-threshold columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (config_id, fold_id), group in oof_objects.groupby(
        ["config_id", "fold_id"],
        sort=False,
        dropna=False,
    ):
        ratios = pd.to_numeric(
            group["target_pixel_ratio"],
            errors="coerce",
        ).to_numpy(dtype=float)
        truth, truth_valid = _coerced_truth(
            group,
            target_class=target_class,
            non_target_label=non_target_label,
        )
        valid = truth_valid & np.isfinite(ratios)
        if not valid.any():
            rows.append(
                {
                    "config_id": config_id,
                    "fold_id": fold_id,
                    "decision_mode": mode,
                    "n_objects": 0,
                    "fn_rate": np.nan,
                    "fp_rate": np.nan,
                    "uncertain_rate": np.nan,
                    "coverage_rate": np.nan,
                    "balanced_accuracy": np.nan,
                    "status": "not_calculable",
                }
            )
            continue
        if mode == "2way":
            threshold = float(group[object_threshold_column].iloc[0])
            metrics = _binary_metric_arrays(
                truth[valid],
                (ratios[valid] >= threshold)[:, None],
            )
            fn_rate = float(metrics["fn_rate"][0])
            fp_rate = float(metrics["fp_rate"][0])
            balanced_accuracy = float(
                metrics["balanced_accuracy"][0]
            )
            uncertain_rate = 0.0
            coverage_rate = 1.0
        else:
            lower = float(group[lower_threshold_column].iloc[0])
            upper = float(group[upper_threshold_column].iloc[0])
            metrics = _three_way_metric_arrays(
                truth[valid],
                ratios[valid],
                np.asarray([lower], dtype=float),
                np.asarray([upper], dtype=float),
            )
            fn_rate = float(metrics["target_miss_rate"][0])
            fp_rate = float(
                metrics["non_target_false_accept_rate"][0]
            )
            uncertain_rate = float(metrics["uncertain_rate"][0])
            coverage_rate = float(metrics["coverage_rate"][0])
            balanced_accuracy = float(
                metrics["decided_balanced_accuracy"][0]
            )
        rows.append(
            {
                "config_id": config_id,
                "fold_id": int(fold_id),
                "decision_mode": mode,
                "n_objects": int(np.sum(valid)),
                "fn_rate": fn_rate,
                "fp_rate": fp_rate,
                "uncertain_rate": uncertain_rate,
                "coverage_rate": coverage_rate,
                "balanced_accuracy": balanced_accuracy,
                "status": (
                    "calculable"
                    if np.isfinite(
                        [
                            fn_rate,
                            fp_rate,
                            uncertain_rate,
                            coverage_rate,
                            balanced_accuracy,
                        ]
                    ).all()
                    else "not_calculable"
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_fixed_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold/seed metrics and apply the configured risk contract."""
    schema = expcfg.SIMCA_GRID_THRESHOLD_METRIC_COLUMNS
    if fold_metrics is None or fold_metrics.empty:
        return _with_schema(pd.DataFrame(), schema)
    metrics = fold_metrics.copy()
    numeric_columns = (
        "fn_rate",
        "fp_rate",
        "uncertain_rate",
        "coverage_rate",
        "balanced_accuracy",
    )
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    summary = (
        metrics.groupby(
            [
                "domain_config_id",
                "calibration_track",
                "decision_mode",
            ],
            as_index=False,
            dropna=False,
            sort=False,
        )
        .agg(
            n_seeds=("random_state", "nunique"),
            n_folds=("fold_id", "nunique"),
            fn_rate_mean=("fn_rate", "mean"),
            fn_rate_max=("fn_rate", "max"),
            fp_rate_mean=("fp_rate", "mean"),
            fp_rate_max=("fp_rate", "max"),
            uncertain_rate_mean=("uncertain_rate", "mean"),
            uncertain_rate_max=("uncertain_rate", "max"),
            coverage_rate_mean=("coverage_rate", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            fold_metric_std=("balanced_accuracy", lambda values: float(
                np.nanstd(
                    pd.to_numeric(values, errors="coerce"),
                    ddof=0,
                )
            )),
            _all_calculable=("status", lambda values: bool(
                pd.Series(values).eq("calculable").all()
            )),
        )
    )
    summary["status"] = "acceptable"
    for mode, constraints in expcfg.SIMCA_SEARCH_CONSTRAINTS.items():
        mask = summary["decision_mode"].eq(mode)
        acceptable = (
            summary["fn_rate_max"].le(
                float(constraints["max_fn_rate"])
            )
            & summary["fp_rate_max"].le(
                float(constraints["max_fp_rate"])
            )
            & summary["balanced_accuracy_mean"].ge(
                float(constraints["min_balanced_accuracy"])
            )
            & summary["fold_metric_std"].le(
                float(constraints["max_fold_metric_std"])
            )
            & summary["_all_calculable"].astype(bool)
        )
        if mode == "3way":
            acceptable &= (
                summary["uncertain_rate_max"].le(
                    float(constraints["max_uncertain_rate"])
                )
                & summary["coverage_rate_mean"].ge(
                    float(constraints["min_coverage"])
                )
            )
        summary.loc[mask & ~acceptable, "status"] = (
            "calculable_but_not_acceptable"
        )
    return _with_schema(summary, schema)


def evaluate_internal_object_thresholds(
    oof_objects: pd.DataFrame,
    *,
    thresholds: Sequence[float] | None = (
        expcfg.INTERNAL_CALIBRATION_OBJECT_THRESHOLDS
    ),
    threshold_mode: str = "grid",
    threshold_column: str = "object_threshold",
    prediction_equivalence: pd.DataFrame | None = None,
    max_fn_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FN_RATE,
    max_fp_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FP_RATE,
    min_balanced_accuracy: float = expcfg.INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY,
    min_decision_rate: float = expcfg.INTERNAL_CALIBRATION_MIN_DECISION_RATE,
    max_image_fn_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_IMAGE_FN_RATE,
    max_image_fp_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_IMAGE_FP_RATE,
    max_fold_fn_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FOLD_FN_RATE,
    max_fold_fp_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FOLD_FP_RATE,
    min_fold_balanced_accuracy: float = (
        expcfg.INTERNAL_CALIBRATION_MIN_FOLD_BALANCED_ACCURACY
    ),
    preferred_thresholds: Sequence[float] = (
        expcfg.INTERNAL_CALIBRATION_PREFERRED_OBJECT_THRESHOLDS
    ),
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
) -> pd.DataFrame:
    """Evaluate 2-way thresholds once per exact OOF prediction vector."""
    if str(threshold_mode) == "fixed":
        return _evaluate_fixed_thresholds_by_fold(
            oof_objects,
            decision_mode="2way",
            object_threshold_column=threshold_column,
            target_class=target_class,
            non_target_label=non_target_label,
        )
    if str(threshold_mode) != "grid":
        raise ValueError("threshold_mode must be 'grid' or 'fixed'.")
    if thresholds is None:
        raise ValueError("thresholds are required in grid mode.")
    threshold_values = np.asarray(tuple(thresholds), dtype=float)
    equivalence = _resolve_oof_prediction_equivalence(
        oof_objects,
        prediction_equivalence,
    )
    if oof_objects.empty or threshold_values.size == 0:
        return _with_schema(
            pd.DataFrame(),
            expcfg.INTERNAL_CALIBRATION_THRESHOLD_2WAY_COLUMNS,
        )

    representative_ids = set(equivalence["representative_config_id"])
    representative_oof = oof_objects.loc[
        oof_objects["config_id"].isin(representative_ids)
    ]
    parts: list[pd.DataFrame] = []
    n_thresholds = len(threshold_values)
    for config_id, group in representative_oof.groupby(
        "config_id",
        sort=False,
    ):
        ratios = pd.to_numeric(
            group["target_pixel_ratio"],
            errors="coerce",
        ).to_numpy(dtype=float)
        ratio_valid = np.isfinite(ratios)
        if not ratio_valid.any():
            continue

        truth, truth_valid = _coerced_truth(
            group,
            target_class=target_class,
            non_target_label=non_target_label,
        )
        evaluated_ratios = ratios[ratio_valid]
        evaluated_truth = truth[ratio_valid]
        evaluated_truth_valid = truth_valid[ratio_valid]
        predictions = (
            evaluated_ratios[:, None] >= threshold_values[None, :]
        )
        metrics = _binary_metric_arrays(
            evaluated_truth[evaluated_truth_valid],
            predictions[evaluated_truth_valid],
        )

        def grouped_binary_metrics(
            column: str,
        ) -> list[dict[str, np.ndarray]]:
            values = group[column].to_numpy()[ratio_valid]
            grouped = []
            for positions in _position_groups(values):
                valid_positions = positions[
                    evaluated_truth_valid[positions]
                ]
                grouped.append(
                    _binary_metric_arrays(
                        evaluated_truth[valid_positions],
                        predictions[valid_positions],
                    )
                )
            return grouped

        image_metrics = grouped_binary_metrics("source_image")
        fold_metrics = grouped_binary_metrics("fold_id")
        image_fn = [item["fn_rate"] for item in image_metrics]
        image_fp = [item["fp_rate"] for item in image_metrics]
        fold_fn = [item["fn_rate"] for item in fold_metrics]
        fold_fp = [item["fp_rate"] for item in fold_metrics]
        fold_ba = [
            item["balanced_accuracy"]
            for item in fold_metrics
        ]
        parts.append(
            pd.DataFrame(
                {
                    "config_id": np.repeat(config_id, n_thresholds),
                    "object_threshold": threshold_values,
                    "n": metrics["n"],
                    "n_folds": np.repeat(
                        pd.Series(
                            group["fold_id"].to_numpy()[ratio_valid]
                        ).nunique(),
                        n_thresholds,
                    ),
                    "fn": metrics["fn"],
                    "fp": metrics["fp"],
                    "target_sensitivity": metrics[
                        "target_sensitivity"
                    ],
                    "non_target_specificity": metrics[
                        "non_target_specificity"
                    ],
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "fn_rate": metrics["fn_rate"],
                    "fp_rate": metrics["fp_rate"],
                    "decision_rate": np.repeat(
                        float(np.mean(ratio_valid)),
                        n_thresholds,
                    ),
                    "max_image_fn_rate": _finite_extreme(
                        image_fn,
                        reducer="max",
                        n_values=n_thresholds,
                    ),
                    "max_image_fp_rate": _finite_extreme(
                        image_fp,
                        reducer="max",
                        n_values=n_thresholds,
                    ),
                    "max_fold_fn_rate": _finite_extreme(
                        fold_fn,
                        reducer="max",
                        n_values=n_thresholds,
                    ),
                    "max_fold_fp_rate": _finite_extreme(
                        fold_fp,
                        reducer="max",
                        n_values=n_thresholds,
                    ),
                    "min_fold_balanced_accuracy": _finite_extreme(
                        fold_ba,
                        reducer="min",
                        n_values=n_thresholds,
                    ),
                    "std_fold_balanced_accuracy": (
                        _finite_population_std(
                            fold_ba,
                            n_values=n_thresholds,
                        )
                    ),
                }
            )
        )

    out = (
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else pd.DataFrame()
    )
    if out.empty:
        return _with_schema(
            out,
            expcfg.INTERNAL_CALIBRATION_THRESHOLD_2WAY_COLUMNS,
        )
    out["threshold_sensitivity"] = 0.0
    for _, group in out.groupby("config_id", sort=False):
        ordered = group.sort_values("object_threshold")
        performance = pd.to_numeric(
            ordered["balanced_accuracy"],
            errors="coerce",
        )
        neighbor_range = pd.concat(
            [
                (performance - performance.shift(1)).abs(),
                (performance - performance.shift(-1)).abs(),
            ],
            axis=1,
        ).max(axis=1).fillna(0.0)
        out.loc[ordered.index, "threshold_sensitivity"] = (
            neighbor_range.to_numpy()
        )

    out["feasible"] = (
        out["fn_rate"].le(float(max_fn_rate))
        & out["fp_rate"].le(float(max_fp_rate))
        & out["balanced_accuracy"].ge(float(min_balanced_accuracy))
        & out["decision_rate"].ge(float(min_decision_rate))
        & out["max_image_fn_rate"].le(float(max_image_fn_rate))
        & out["max_image_fp_rate"].le(float(max_image_fp_rate))
        & out["max_fold_fn_rate"].le(float(max_fold_fn_rate))
        & out["max_fold_fp_rate"].le(float(max_fold_fp_rate))
        & out["min_fold_balanced_accuracy"].ge(
            float(min_fold_balanced_accuracy)
        )
    )
    out["selected"] = False
    out["selection_status"] = ""
    for _, group in out.groupby("config_id", sort=False):
        feasible = group.loc[group["feasible"]]
        if feasible.empty:
            out.loc[
                group.index,
                "selection_status",
            ] = "technically_calculable_but_not_acceptable"
            continue
        preferred = tuple(map(float, preferred_thresholds))
        feasible = feasible.assign(
            _preferred_distance=feasible["object_threshold"].map(
                lambda value: min(abs(float(value) - item) for item in preferred)
                if preferred
                else 0.0
            )
        )
        selected_index = feasible.sort_values(
            [
                "fn_rate",
                "fp_rate",
                "max_image_fn_rate",
                "max_image_fp_rate",
                "decision_rate",
                "balanced_accuracy",
                "threshold_sensitivity",
                "_preferred_distance",
                "object_threshold",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
            ],
            kind="mergesort",
        ).index[0]
        out.loc[group.index, "selection_status"] = "acceptable"
        out.loc[selected_index, "selected"] = True
    expanded = _expand_representative_threshold_metrics(
        out,
        equivalence,
    )
    result = _with_schema(
        expanded,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_2WAY_COLUMNS,
    )
    result.attrs["n_input_configurations"] = int(
        equivalence["config_id"].nunique()
    )
    result.attrs["n_prediction_signatures"] = int(
        equivalence["representative_config_id"].nunique()
    )
    return result


def evaluate_internal_three_way_thresholds(
    oof_objects: pd.DataFrame,
    *,
    lower_thresholds: Sequence[float] | None = (
        expcfg.INTERNAL_CALIBRATION_THREE_WAY_LOWER_THRESHOLDS
    ),
    upper_thresholds: Sequence[float] | None = (
        expcfg.INTERNAL_CALIBRATION_THREE_WAY_UPPER_THRESHOLDS
    ),
    threshold_mode: str = "grid",
    lower_threshold_column: str = "three_way_lower_threshold",
    upper_threshold_column: str = "three_way_upper_threshold",
    prediction_equivalence: pd.DataFrame | None = None,
    max_target_miss_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE,
    max_false_accept_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE,
    max_uncertain_rate: float = expcfg.INTERNAL_CALIBRATION_MAX_UNCERTAIN_RATE,
    min_coverage: float = expcfg.INTERNAL_CALIBRATION_MIN_COVERAGE,
    max_image_target_miss_rate: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_IMAGE_TARGET_MISS_RATE
    ),
    max_image_false_accept_rate: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_IMAGE_FALSE_ACCEPT_RATE
    ),
    max_image_uncertain_rate: float = (
        expcfg.INTERNAL_CALIBRATION_MAX_IMAGE_UNCERTAIN_RATE
    ),
    min_image_coverage: float = expcfg.INTERNAL_CALIBRATION_MIN_IMAGE_COVERAGE,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
) -> pd.DataFrame:
    """Evaluate the full 3-way grid once per exact OOF ratio vector."""
    if str(threshold_mode) == "fixed":
        return _evaluate_fixed_thresholds_by_fold(
            oof_objects,
            decision_mode="3way",
            lower_threshold_column=lower_threshold_column,
            upper_threshold_column=upper_threshold_column,
            target_class=target_class,
            non_target_label=non_target_label,
        )
    if str(threshold_mode) != "grid":
        raise ValueError("threshold_mode must be 'grid' or 'fixed'.")
    if lower_thresholds is None or upper_thresholds is None:
        raise ValueError(
            "lower_thresholds and upper_thresholds are required in grid mode."
        )
    threshold_pairs = [
        (float(lower), float(upper))
        for lower in lower_thresholds
        for upper in upper_thresholds
        if float(lower) < float(upper)
    ]
    equivalence = _resolve_oof_prediction_equivalence(
        oof_objects,
        prediction_equivalence,
    )
    if oof_objects.empty or not threshold_pairs:
        return _with_schema(
            pd.DataFrame(),
            expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_COLUMNS,
        )

    lower_values = np.asarray(
        [pair[0] for pair in threshold_pairs],
        dtype=float,
    )
    upper_values = np.asarray(
        [pair[1] for pair in threshold_pairs],
        dtype=float,
    )
    n_pairs = len(lower_values)
    representative_ids = set(equivalence["representative_config_id"])
    representative_oof = oof_objects.loc[
        oof_objects["config_id"].isin(representative_ids)
    ]
    parts: list[pd.DataFrame] = []
    for config_id, group in representative_oof.groupby(
        "config_id",
        sort=False,
    ):
        ratios = pd.to_numeric(
            group["target_pixel_ratio"],
            errors="coerce",
        ).to_numpy(dtype=float)
        truth, truth_valid = _coerced_truth(
            group,
            target_class=target_class,
            non_target_label=non_target_label,
        )
        if "truth_available_ratio" in group:
            truth_available = pd.to_numeric(
                group["truth_available_ratio"],
                errors="coerce",
            ).to_numpy(dtype=float)
            truth_valid &= truth_available >= 0.50
        metrics = _three_way_metric_arrays(
            truth[truth_valid],
            ratios[truth_valid],
            lower_values,
            upper_values,
        )

        def grouped_three_way_metrics(
            column: str,
        ) -> list[dict[str, np.ndarray]]:
            grouped = []
            for positions in _position_groups(group[column].to_numpy()):
                valid_positions = positions[truth_valid[positions]]
                grouped.append(
                    _three_way_metric_arrays(
                        truth[valid_positions],
                        ratios[valid_positions],
                        lower_values,
                        upper_values,
                    )
                )
            return grouped

        image_metrics = grouped_three_way_metrics("source_image")
        fold_metrics = grouped_three_way_metrics("fold_id")

        def metric_arrays(
            grouped: Sequence[dict[str, np.ndarray]],
            column: str,
        ) -> list[np.ndarray]:
            return [item[column] for item in grouped]

        parts.append(
            pd.DataFrame(
                {
                    "config_id": np.repeat(config_id, n_pairs),
                    "three_way_lower_threshold": lower_values,
                    "three_way_upper_threshold": upper_values,
                    "n": metrics["n"],
                    "n_folds": np.repeat(
                        group["fold_id"].nunique(),
                        n_pairs,
                    ),
                    "target_miss_rate": metrics[
                        "target_miss_rate"
                    ],
                    "non_target_false_accept_rate": metrics[
                        "non_target_false_accept_rate"
                    ],
                    "uncertain_rate": metrics["uncertain_rate"],
                    "coverage_rate": metrics["coverage_rate"],
                    "target_uncertain_rate": metrics[
                        "target_uncertain_rate"
                    ],
                    "non_target_uncertain_rate": metrics[
                        "non_target_uncertain_rate"
                    ],
                    "decided_balanced_accuracy": metrics[
                        "decided_balanced_accuracy"
                    ],
                    "max_image_target_miss_rate": _finite_extreme(
                        metric_arrays(
                            image_metrics,
                            "target_miss_rate",
                        ),
                        reducer="max",
                        n_values=n_pairs,
                    ),
                    "max_image_false_accept_rate": _finite_extreme(
                        metric_arrays(
                            image_metrics,
                            "non_target_false_accept_rate",
                        ),
                        reducer="max",
                        n_values=n_pairs,
                    ),
                    "max_image_uncertain_rate": _finite_extreme(
                        metric_arrays(
                            image_metrics,
                            "uncertain_rate",
                        ),
                        reducer="max",
                        n_values=n_pairs,
                    ),
                    "min_image_coverage": _finite_extreme(
                        metric_arrays(
                            image_metrics,
                            "coverage_rate",
                        ),
                        reducer="min",
                        n_values=n_pairs,
                    ),
                    "max_fold_target_miss_rate": _finite_extreme(
                        metric_arrays(
                            fold_metrics,
                            "target_miss_rate",
                        ),
                        reducer="max",
                        n_values=n_pairs,
                    ),
                    "max_fold_false_accept_rate": _finite_extreme(
                        metric_arrays(
                            fold_metrics,
                            "non_target_false_accept_rate",
                        ),
                        reducer="max",
                        n_values=n_pairs,
                    ),
                    "max_fold_uncertain_rate": _finite_extreme(
                        metric_arrays(
                            fold_metrics,
                            "uncertain_rate",
                        ),
                        reducer="max",
                        n_values=n_pairs,
                    ),
                    "min_fold_coverage": _finite_extreme(
                        metric_arrays(
                            fold_metrics,
                            "coverage_rate",
                        ),
                        reducer="min",
                        n_values=n_pairs,
                    ),
                    "std_fold_decided_balanced_accuracy": (
                        _finite_population_std(
                            metric_arrays(
                                fold_metrics,
                                "decided_balanced_accuracy",
                            ),
                            n_values=n_pairs,
                        )
                    ),
                    "uncertain_zone_width": (
                        upper_values - lower_values
                    ),
                }
            )
        )

    out = (
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else pd.DataFrame()
    )
    if out.empty:
        return _with_schema(
            out,
            expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_COLUMNS,
        )
    out["feasible"] = (
        out["target_miss_rate"].le(float(max_target_miss_rate))
        & out["non_target_false_accept_rate"].le(float(max_false_accept_rate))
        & out["uncertain_rate"].le(float(max_uncertain_rate))
        & out["coverage_rate"].ge(float(min_coverage))
        & out["max_image_target_miss_rate"].le(
            float(max_image_target_miss_rate)
        )
        & out["max_image_false_accept_rate"].le(
            float(max_image_false_accept_rate)
        )
        & out["max_image_uncertain_rate"].le(
            float(max_image_uncertain_rate)
        )
        & out["min_image_coverage"].ge(float(min_image_coverage))
    )
    out["selected"] = False
    out["selection_status"] = ""
    for _, group in out.groupby("config_id", sort=False):
        feasible = group.loc[group["feasible"]]
        if feasible.empty:
            out.loc[
                group.index,
                "selection_status",
            ] = "technically_calculable_but_not_acceptable"
            continue
        selected_index = feasible.sort_values(
            [
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
                "decided_balanced_accuracy",
                "uncertain_zone_width",
                "target_uncertain_rate",
                "non_target_uncertain_rate",
                "three_way_upper_threshold",
                "three_way_lower_threshold",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
            ],
            kind="mergesort",
        ).index[0]
        out.loc[group.index, "selection_status"] = "acceptable"
        out.loc[selected_index, "selected"] = True
    expanded = _expand_representative_threshold_metrics(
        out,
        equivalence,
    )
    result = _with_schema(
        expanded,
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_COLUMNS,
    )
    result.attrs["n_input_configurations"] = int(
        equivalence["config_id"].nunique()
    )
    result.attrs["n_prediction_signatures"] = int(
        equivalence["representative_config_id"].nunique()
    )
    return result


def summarize_internal_three_way_threshold_study(
    thresholds_3way: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the 3-way sensitivity grid without a weighted score."""
    schema = expcfg.INTERNAL_CALIBRATION_THRESHOLD_3WAY_STUDY_COLUMNS
    if thresholds_3way is None or thresholds_3way.empty:
        return pd.DataFrame(columns=schema)
    required = {
        "config_id",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
        "feasible",
        "selected",
        "target_miss_rate",
        "non_target_false_accept_rate",
        "uncertain_rate",
        "coverage_rate",
        "decided_balanced_accuracy",
    }
    missing = sorted(required.difference(thresholds_3way.columns))
    if missing:
        raise KeyError(
            "3-way threshold study is missing columns: "
            f"{missing}"
        )

    data = thresholds_3way.copy()
    data["feasible"] = data["feasible"].fillna(False).astype(bool)
    data["selected"] = data["selected"].fillna(False).astype(bool)
    metric_columns = (
        "target_miss_rate",
        "non_target_false_accept_rate",
        "uncertain_rate",
        "coverage_rate",
        "decided_balanced_accuracy",
    )
    for column in metric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    rows: list[dict[str, object]] = []
    keys = (
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    )
    for (lower, upper), group in data.groupby(list(keys), sort=True):
        n_configurations = int(group["config_id"].nunique())
        feasible_ids = group.loc[group["feasible"], "config_id"].nunique()
        selected_ids = group.loc[group["selected"], "config_id"].nunique()
        if selected_ids:
            pair_status = "selected_by_at_least_one_configuration"
        elif feasible_ids:
            pair_status = "feasible_not_selected"
        else:
            pair_status = "not_feasible_for_any_configuration"
        rows.append(
            {
                "three_way_lower_threshold": float(lower),
                "three_way_upper_threshold": float(upper),
                "n_configurations": n_configurations,
                "n_feasible_configurations": int(feasible_ids),
                "feasible_configuration_rate": (
                    float(feasible_ids / n_configurations)
                    if n_configurations
                    else np.nan
                ),
                "n_selected_configurations": int(selected_ids),
                "selected_configuration_rate": (
                    float(selected_ids / n_configurations)
                    if n_configurations
                    else np.nan
                ),
                "median_target_miss_rate": float(
                    group["target_miss_rate"].median()
                ),
                "p90_target_miss_rate": float(
                    group["target_miss_rate"].quantile(0.90)
                ),
                "median_non_target_false_accept_rate": float(
                    group["non_target_false_accept_rate"].median()
                ),
                "p90_non_target_false_accept_rate": float(
                    group["non_target_false_accept_rate"].quantile(0.90)
                ),
                "median_uncertain_rate": float(
                    group["uncertain_rate"].median()
                ),
                "p90_uncertain_rate": float(
                    group["uncertain_rate"].quantile(0.90)
                ),
                "median_coverage_rate": float(
                    group["coverage_rate"].median()
                ),
                "median_decided_balanced_accuracy": float(
                    group["decided_balanced_accuracy"].median()
                ),
                "pair_status": pair_status,
            }
        )
    return (
        pd.DataFrame(rows)
        .loc[:, list(schema)]
        .sort_values(list(keys), kind="mergesort")
        .reset_index(drop=True)
    )


def build_internal_calibrated_hyperparameters(
    component_selected: pd.DataFrame,
    thresholds_2way: pd.DataFrame,
    thresholds_3way: pd.DataFrame,
    *,
    rule_diagnostics: pd.DataFrame | None = None,
    oof_objects: pd.DataFrame | None = None,
    sampling_diagnostics: pd.DataFrame | None = None,
    tolerance: float = expcfg.INTERNAL_CALIBRATION_PERFORMANCE_PLATEAU_TOLERANCE,
) -> pd.DataFrame:
    """Calibrate seed-agnostic ``k``, ``m`` and object thresholds.

    Notebook 03B is a calibration step, not a final model-selection step. The
    function therefore selects a common threshold across evaluation seeds,
    verifies ``k`` at that threshold, selects the smallest robust ``m`` plateau
    and applies the declared stability constraints. It deliberately preserves
    every distinct model configuration that survives those groupwise
    calibrations: no metric deduplication, Pareto filter, diversity filter,
    cross-model ranking or per-track quota is applied here.
    """
    if component_selected.empty:
        return _with_schema(
            pd.DataFrame(),
            expcfg.INTERNAL_CALIBRATION_CALIBRATED_HYPERPARAMETER_COLUMNS,
        )
    configurations = component_selected.copy()
    if {"component_acceptable", "component_plateau"}.issubset(
        configurations.columns
    ):
        configurations = configurations.loc[
            configurations["component_acceptable"].fillna(False).astype(bool)
            & configurations["component_plateau"].fillna(False).astype(bool)
        ].copy()
    configurations = configurations.drop_duplicates("config_id")
    if configurations.empty:
        return _with_schema(
            pd.DataFrame(),
            expcfg.INTERNAL_CALIBRATION_CALIBRATED_HYPERPARAMETER_COLUMNS,
        )

    semantic_without_seed = [
        column
        for column in _INTERNAL_PARAMETER_COLUMNS
        if column != "random_state"
    ]
    configurations["model_group_id"] = [
        hash_internal_calibration_configuration(
            {column: row[column] for column in semantic_without_seed},
            prefix="icmodel",
        )
        for row in configurations.to_dict("records")
    ]
    model_map = configurations[
        ["config_id", "model_group_id", "random_state"]
    ].copy()
    model_metadata = (
        configurations.sort_values("random_state", kind="mergesort")
        .drop_duplicates("model_group_id")
        .loc[
            :,
            [
                "config_id",
                "model_group_id",
                *semantic_without_seed,
                "random_state",
            ],
        ]
        .rename(columns={"config_id": "source_config_id"})
    )
    seed_metadata = (
        configurations.groupby("model_group_id", as_index=False)
        .agg(
            n_seeds_evaluated=("random_state", "nunique"),
            random_states_json=(
                "random_state",
                lambda values: json.dumps(
                    sorted(
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                ),
            ),
        )
    )
    model_metadata = model_metadata.merge(
        seed_metadata,
        on="model_group_id",
        how="left",
        validate="one_to_one",
    )

    grouped_rule_diagnostics = None
    if rule_diagnostics is not None and not rule_diagnostics.empty:
        grouped_rule_diagnostics = rule_diagnostics.merge(
            model_map[["config_id", "model_group_id"]],
            on="config_id",
            how="inner",
            validate="many_to_one",
        )
        grouped_rule_diagnostics["config_id"] = grouped_rule_diagnostics[
            "model_group_id"
        ]
        grouped_rule_diagnostics = grouped_rule_diagnostics.drop(
            columns=["model_group_id"]
        )

    consensus_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for decision_mode, threshold_grid in (
        ("2way", thresholds_2way),
        ("3way", thresholds_3way),
    ):
        if threshold_grid is None or threshold_grid.empty:
            continue
        grid = threshold_grid.merge(
            model_map[["config_id", "model_group_id"]],
            on="config_id",
            how="inner",
            validate="many_to_one",
        )
        if grid.empty:
            continue
        threshold_columns = (
            ["object_threshold"]
            if decision_mode == "2way"
            else [
                "three_way_lower_threshold",
                "three_way_upper_threshold",
            ]
        )
        if decision_mode == "2way":
            mean_columns = (
                "fn_rate",
                "fp_rate",
                "balanced_accuracy",
                "decision_rate",
                "target_sensitivity",
                "non_target_specificity",
            )
            max_columns = (
                "max_image_fn_rate",
                "max_image_fp_rate",
                "max_fold_fn_rate",
                "max_fold_fp_rate",
                "std_fold_balanced_accuracy",
                "threshold_sensitivity",
            )
            min_columns = ("min_fold_balanced_accuracy",)
        else:
            mean_columns = (
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
                "coverage_rate",
                "target_uncertain_rate",
                "non_target_uncertain_rate",
                "decided_balanced_accuracy",
            )
            max_columns = (
                "max_image_target_miss_rate",
                "max_image_false_accept_rate",
                "max_image_uncertain_rate",
                "max_fold_target_miss_rate",
                "max_fold_false_accept_rate",
                "max_fold_uncertain_rate",
                "std_fold_decided_balanced_accuracy",
            )
            min_columns = (
                "min_image_coverage",
                "min_fold_coverage",
            )
        numeric_columns = [
            "n_folds",
            *mean_columns,
            *max_columns,
            *min_columns,
        ]
        for column in numeric_columns:
            grid[column] = pd.to_numeric(
                grid[column],
                errors="coerce",
            )
        grid["_seed_feasible"] = (
            grid["feasible"].fillna(False).astype(bool)
        )
        aggregation = {
            "n_seeds_evaluated": ("config_id", "nunique"),
            "_all_seeds_feasible": ("_seed_feasible", "all"),
            "n_folds": ("n_folds", "min"),
            **{
                column: (column, "mean")
                for column in mean_columns
            },
            **{
                column: (column, "max")
                for column in max_columns
            },
            **{
                column: (column, "min")
                for column in min_columns
            },
        }
        consensus_grid = (
            grid.groupby(
                ["model_group_id", *threshold_columns],
                dropna=False,
                sort=False,
                as_index=False,
            )
            .agg(**aggregation)
        )
        expected_seeds = model_metadata.set_index(
            "model_group_id"
        )["n_seeds_evaluated"]
        consensus_grid["_expected_n_seeds"] = (
            consensus_grid["model_group_id"]
            .map(expected_seeds)
            .astype(int)
        )
        consensus_grid["feasible"] = (
            consensus_grid["n_seeds_evaluated"].eq(
                consensus_grid["_expected_n_seeds"]
            )
            & consensus_grid["_all_seeds_feasible"].astype(bool)
        )
        if decision_mode == "3way":
            consensus_grid["uncertain_zone_width"] = (
                consensus_grid["three_way_upper_threshold"]
                - consensus_grid["three_way_lower_threshold"]
            )
        consensus_grid["selected"] = False
        feasible_consensus = consensus_grid.loc[
            consensus_grid["feasible"]
        ].copy()
        if decision_mode == "2way":
            preferred = np.asarray(
                expcfg.INTERNAL_CALIBRATION_PREFERRED_OBJECT_THRESHOLDS,
                dtype=float,
            )
            if preferred.size:
                distances = np.abs(
                    feasible_consensus[
                        "object_threshold"
                    ].to_numpy(dtype=float)[:, None]
                    - preferred[None, :]
                )
                feasible_consensus["_preferred_distance"] = (
                    distances.min(axis=1)
                )
            else:
                feasible_consensus["_preferred_distance"] = 0.0
            order = [
                "fn_rate",
                "fp_rate",
                "max_image_fn_rate",
                "max_image_fp_rate",
                "decision_rate",
                "balanced_accuracy",
                "std_fold_balanced_accuracy",
                "threshold_sensitivity",
                "_preferred_distance",
                "object_threshold",
            ]
            ascending = [
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
            ]
        else:
            order = [
                "decided_balanced_accuracy",
                "uncertain_zone_width",
                "target_miss_rate",
                "non_target_false_accept_rate",
                "uncertain_rate",
                "coverage_rate",
            ]
            ascending = [False, True, True, True, True, False]
        selected_indices = (
            feasible_consensus.sort_values(
                ["model_group_id", *order],
                ascending=[True, *ascending],
                kind="mergesort",
            )
            .drop_duplicates("model_group_id", keep="first")
            .index
        )
        consensus_grid.loc[selected_indices, "selected"] = True

        selected_consensus = consensus_grid.loc[
            consensus_grid["selected"]
        ].copy()
        audit_rows.append(
            {
                "stage": f"{decision_mode}_seed_consensus_threshold",
                "n_candidates": int(len(selected_consensus)),
            }
        )
        if selected_consensus.empty:
            continue
        selected_consensus["config_id"] = selected_consensus[
            "model_group_id"
        ]
        candidate_config = model_metadata.copy()
        candidate_config["config_id"] = candidate_config["model_group_id"]
        final_component_summary, final_component_selected = (
            select_smallest_plateau_components(
                selected_consensus,
                candidate_config,
                rule_diagnostics=grouped_rule_diagnostics,
                fn_column=(
                    "fn_rate"
                    if decision_mode == "2way"
                    else "target_miss_rate"
                ),
                fp_column=(
                    "fp_rate"
                    if decision_mode == "2way"
                    else "non_target_false_accept_rate"
                ),
                balanced_accuracy_column=(
                    "balanced_accuracy"
                    if decision_mode == "2way"
                    else "decided_balanced_accuracy"
                ),
                expected_n_folds=(
                    int(selected_consensus["n_folds"].max())
                    if not pd.isna(
                        selected_consensus["n_folds"].max()
                    )
                    else None
                ),
                max_fn_rate=(
                    expcfg.INTERNAL_CALIBRATION_MAX_FN_RATE
                    if decision_mode == "2way"
                    else expcfg.INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE
                ),
                max_fp_rate=(
                    expcfg.INTERNAL_CALIBRATION_MAX_FP_RATE
                    if decision_mode == "2way"
                    else expcfg.INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE
                ),
                min_balanced_accuracy=(
                    expcfg.INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY
                ),
                tolerance=tolerance,
            )
        )
        del final_component_summary
        if final_component_selected.empty:
            continue
        final_rows = final_component_selected.merge(
            selected_consensus.drop(columns=["config_id"]),
            on="model_group_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_threshold"),
        )
        final_rows["decision_mode"] = decision_mode
        consensus_parts.append(final_rows)
        audit_rows.append(
            {
                "stage": f"{decision_mode}_final_k",
                "n_candidates": int(len(final_rows)),
            }
        )

    if not consensus_parts:
        return _with_schema(
            pd.DataFrame(),
            expcfg.INTERNAL_CALIBRATION_CALIBRATED_HYPERPARAMETER_COLUMNS,
        )
    candidates = pd.concat(consensus_parts, ignore_index=True, sort=False)
    candidates["selection_track"] = [
        selection_track_from_parts(family, mode)
        for family, mode in zip(
            candidates["matrix_family"],
            candidates["decision_mode"],
        )
    ]

    # Select m only for balanced pixels. Each m has already received its own
    # final threshold and final k; seeds have already been aggregated.
    candidates["m_plateau"] = candidates["matrix_method"].ne(
        "balanced_pixels"
    )
    candidates["m_selected"] = candidates["matrix_method"].ne(
        "balanced_pixels"
    )
    m_group_columns = [
        "selection_track",
        "matrix_family",
        "matrix_method",
        "balanced_pixel_strategy",
        "preprocessing",
        "preprocessing_steps",
        "rule_family",
        "rule_variant",
        "limit_source",
        "alpha",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
    ]
    pixel_mask = candidates["matrix_method"].eq("balanced_pixels")
    for _, group in candidates.loc[pixel_mask].groupby(
        m_group_columns,
        dropna=False,
        sort=False,
    ):
        ordered = group.sort_values("m")
        for position, index in enumerate(ordered.index):
            future = ordered.iloc[position:]
            if str(candidates.loc[index, "decision_mode"]) == "2way":
                performance_column = "balanced_accuracy"
                target_risk_column = "fn_rate"
                non_target_risk_column = "fp_rate"
            else:
                performance_column = "decided_balanced_accuracy"
                target_risk_column = "target_miss_rate"
                non_target_risk_column = "non_target_false_accept_rate"
            gains = (
                float(
                    pd.to_numeric(
                        future[performance_column],
                        errors="coerce",
                    ).max()
                )
                - float(candidates.loc[index, performance_column]),
                float(candidates.loc[index, target_risk_column])
                - float(
                    pd.to_numeric(
                        future[target_risk_column],
                        errors="coerce",
                    ).min()
                ),
                float(candidates.loc[index, non_target_risk_column])
                - float(
                    pd.to_numeric(
                        future[non_target_risk_column],
                        errors="coerce",
                    ).min()
                ),
            )
            candidates.loc[index, "m_plateau"] = all(
                max(0.0, gain) <= float(tolerance) for gain in gains
            )
        eligible = ordered.loc[
            candidates.loc[ordered.index, "m_plateau"]
        ]
        if not eligible.empty:
            selected_index = eligible.sort_values("m").index[0]
            candidates.loc[selected_index, "m_selected"] = True
    candidates = candidates.loc[candidates["m_selected"]].copy()
    audit_rows.append(
        {"stage": "after_m_plateau", "n_candidates": int(len(candidates))}
    )

    # Object-prediction agreement is the seed robustness measure used for the
    # random balanced-pixel strategy. A deterministic configuration scores 1.
    candidates["seed_prediction_agreement"] = 1.0
    if oof_objects is not None and not oof_objects.empty:
        candidate_model_ids = pd.Index(
            candidates["model_group_id"].dropna().unique()
        )
        member_map = configurations.loc[
            configurations["model_group_id"].isin(candidate_model_ids),
            ["config_id", "model_group_id", "random_state"],
        ]
        object_predictions = oof_objects.merge(
            member_map,
            on="config_id",
            how="inner",
            validate="many_to_one",
        )
        object_predictions = object_predictions.drop_duplicates(
            [
                "model_group_id",
                "source_image",
                "object_id",
                "random_state",
            ],
            keep="first",
        )
        object_group_positions = object_predictions.groupby(
            "model_group_id",
            sort=False,
        ).indices
        object_codes = pd.factorize(
            pd.MultiIndex.from_frame(
                object_predictions[["source_image", "object_id"]]
            ),
            sort=False,
        )[0]
        ratio_values = pd.to_numeric(
            object_predictions["target_pixel_ratio"],
            errors="coerce",
        ).to_numpy(dtype=float)
        for index, row in candidates.iterrows():
            positions = object_group_positions.get(row["model_group_id"])
            if positions is None:
                continue
            seed_count = object_predictions.iloc[positions][
                "random_state"
            ].nunique()
            if seed_count <= 1:
                continue
            ratios = ratio_values[positions]
            if row["decision_mode"] == "2way":
                decisions = ratios >= float(row["object_threshold"])
                n_decisions = 2
            else:
                lower = float(row["three_way_lower_threshold"])
                upper = float(row["three_way_upper_threshold"])
                decisions = np.select(
                    [ratios >= upper, ratios <= lower],
                    [2, 0],
                    default=1,
                )
                n_decisions = 3
            local_object_codes = pd.factorize(
                object_codes[positions],
                sort=False,
            )[0]
            counts = np.zeros(
                (
                    int(local_object_codes.max()) + 1,
                    n_decisions,
                ),
                dtype=np.int16,
            )
            np.add.at(
                counts,
                (local_object_codes, decisions.astype(np.int8)),
                1,
            )
            observed = counts.sum(axis=1)
            valid = observed > 0
            if valid.any():
                agreement = np.mean(
                    counts[valid].max(axis=1) / observed[valid]
                )
                candidates.loc[index, "seed_prediction_agreement"] = float(
                    agreement
                )

    candidates["seed_sampling_agreement"] = 1.0
    if sampling_diagnostics is not None and not sampling_diagnostics.empty:
        sampling_map = configurations.loc[
            configurations["model_group_id"].isin(
                candidates["model_group_id"].dropna().unique()
            ),
            ["model_group_id", "data_config_id"],
        ].drop_duplicates()
        sketches = sampling_diagnostics.merge(
            sampling_map,
            on="data_config_id",
            how="inner",
            validate="many_to_many",
        )
        sketch_group_positions = sketches.groupby(
            "model_group_id",
            sort=False,
        ).indices
        for index, row in candidates.iterrows():
            positions = sketch_group_positions.get(row["model_group_id"])
            if positions is None:
                continue
            group = sketches.iloc[positions]
            if group.empty or group["random_state"].nunique() <= 1:
                continue
            fold_agreements = []
            for _, fold_group in group.groupby("fold_id", sort=False):
                tokens = [
                    str(value).split(".")
                    for value in fold_group["sampling_minhash"]
                    if str(value)
                ]
                if len(tokens) <= 1:
                    continue
                pair_agreements = []
                for left_index in range(len(tokens)):
                    for right_index in range(left_index + 1, len(tokens)):
                        n_tokens = min(
                            len(tokens[left_index]),
                            len(tokens[right_index]),
                        )
                        if n_tokens:
                            pair_agreements.append(
                                float(
                                    np.mean(
                                        np.asarray(
                                            tokens[left_index][:n_tokens]
                                        )
                                        == np.asarray(
                                            tokens[right_index][:n_tokens]
                                        )
                                    )
                                )
                            )
                if pair_agreements:
                    fold_agreements.append(float(np.mean(pair_agreements)))
            if fold_agreements:
                candidates.loc[index, "seed_sampling_agreement"] = float(
                    np.mean(fold_agreements)
                )

    candidates["fold_balanced_accuracy_std"] = pd.to_numeric(
        candidates.get(
            "std_fold_balanced_accuracy",
            candidates.get(
                "std_fold_decided_balanced_accuracy",
                pd.Series(0.0, index=candidates.index),
            ),
        ),
        errors="coerce",
    ).fillna(
        pd.to_numeric(
            candidates.get(
                "std_fold_decided_balanced_accuracy",
                pd.Series(0.0, index=candidates.index),
            ),
            errors="coerce",
        )
    )
    candidates["stability_passed"] = (
        candidates["fold_balanced_accuracy_std"].le(
            float(expcfg.INTERNAL_CALIBRATION_MAX_FOLD_BALANCED_ACCURACY_STD)
        )
        & pd.to_numeric(
            candidates["limit_relative_std"],
            errors="coerce",
        ).le(float(expcfg.INTERNAL_CALIBRATION_MAX_LIMIT_RELATIVE_STD))
        & pd.to_numeric(
            candidates["train_validation_rejection_gap"],
            errors="coerce",
        ).le(
            float(
                expcfg.INTERNAL_CALIBRATION_MAX_TRAIN_VALIDATION_REJECTION_GAP
            )
        )
    )
    candidates = candidates.loc[candidates["stability_passed"]].copy()
    audit_rows.append(
        {"stage": "after_stability", "n_candidates": int(len(candidates))}
    )
    if candidates.empty:
        return _with_schema(
            pd.DataFrame(),
            expcfg.INTERNAL_CALIBRATION_CALIBRATED_HYPERPARAMETER_COLUMNS,
        )

    calibrated = candidates.copy()
    three_way_mask = calibrated["decision_mode"].eq("3way")
    calibrated.loc[three_way_mask, "fn_rate"] = calibrated.loc[
        three_way_mask,
        "target_miss_rate",
    ]
    calibrated.loc[three_way_mask, "fp_rate"] = calibrated.loc[
        three_way_mask,
        "non_target_false_accept_rate",
    ]
    calibrated.loc[three_way_mask, "balanced_accuracy"] = calibrated.loc[
        three_way_mask,
        "decided_balanced_accuracy",
    ]
    calibrated.loc[~three_way_mask, "coverage_rate"] = calibrated.loc[
        ~three_way_mask,
        "decision_rate",
    ]
    calibrated.loc[~three_way_mask, "uncertain_rate"] = 0.0
    calibrated["calibration_status"] = "calibrated_for_downstream_search"
    calibrated["calibration_id"] = [
        hash_internal_calibration_configuration(
            {
                "model_group": row["model_group_id"],
                "decision_mode": row["decision_mode"],
                "object_threshold": row.get("object_threshold"),
                "three_way_lower_threshold": row.get(
                    "three_way_lower_threshold"
                ),
                "three_way_upper_threshold": row.get(
                    "three_way_upper_threshold"
                ),
            },
            prefix="iccal",
        )
        for row in calibrated.to_dict("records")
    ]
    calibrated["calibration_track"] = calibrated["selection_track"]
    calibrated = calibrated.sort_values(
        [
            "calibration_track",
            "matrix_method",
            "preprocessing",
            "rule_variant",
            "n_components",
            "m",
        ],
        na_position="first",
        kind="mergesort",
    )
    audit_rows.append(
        {
            "stage": "calibrated_for_downstream_search",
            "n_candidates": int(len(calibrated)),
        }
    )
    result = _with_schema(
        calibrated,
        expcfg.INTERNAL_CALIBRATION_CALIBRATED_HYPERPARAMETER_COLUMNS,
    )
    # Keep attrs JSON-compatible so direct pandas/pyarrow serialization also
    # works. The canonical parquet writer strips attrs because diagnostics are
    # not part of the tabular output contract.
    result.attrs["calibration_audit"] = audit_rows
    return result


def build_calibration_domain_from_03b(
    calibrated_hyperparameters: pd.DataFrame,
    *,
    pca_selected_preprocessings: pd.DataFrame | None = None,
    random_seeds: Sequence[int] = expcfg.SIMCA_SEARCH_RANDOM_SEEDS,
    allowed_dilation_radii: Sequence[int] = (
        expcfg.SIMCA_SEARCH_ALLOWED_DILATION_RADII
    ),
) -> pd.DataFrame:
    """Build the sole seed-expanded search domain consumed by 04A and 04B.

    Thresholds are copied from 03B and are part of ``domain_config_id``. They
    are never suggested or recalibrated downstream. Rows differing only by an
    evaluation seed share one domain identifier and receive distinct runtime
    ``config_id`` values.
    """
    required = {
        "calibration_id",
        "calibration_track",
        "decision_mode",
        "matrix_method",
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
        "object_threshold",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
        "calibration_status",
    }
    missing = sorted(required - set(calibrated_hyperparameters.columns))
    if missing:
        raise KeyError(f"Missing calibrated-domain columns: {missing}")

    calibrated = calibrated_hyperparameters.loc[
        calibrated_hyperparameters["calibration_status"].eq(
            "calibrated_for_downstream_search"
        )
    ].copy()
    if calibrated.empty:
        raise RuntimeError("The calibrated 03B search domain is empty.")

    # Backward compatibility for 03B Parquet files produced before provenance
    # was added to the compact schema. Internal calibration has always imposed
    # the neutral dilation radius, so this default does not alter a model.
    if "position_dilation_radius" not in calibrated:
        calibrated["position_dilation_radius"] = 0
    calibrated["matrix_family"] = calibrated["matrix_method"].map(
        matrix_family_from_method
    )
    calibrated["rule_family"] = calibrated["rule_variant"].map(_rule_family)
    calibrated["position_dilation_radius"] = pd.to_numeric(
        calibrated["position_dilation_radius"],
        errors="raise",
    ).astype(int)
    allowed_radii = set(map(int, allowed_dilation_radii))
    calibrated = calibrated.loc[
        calibrated["position_dilation_radius"].isin(allowed_radii)
    ].copy()
    if calibrated.empty:
        raise RuntimeError(
            "No calibrated configuration uses an allowed dilation radius."
        )

    if (
        pca_selected_preprocessings is not None
        and not pca_selected_preprocessings.empty
    ):
        shortlist = (
            pca_selected_preprocessings[
                ["matrix_family", "preprocessing"]
            ]
            .drop_duplicates()
            .assign(_pca_authorized=True)
        )
        calibrated = calibrated.merge(
            shortlist,
            on=["matrix_family", "preprocessing"],
            how="left",
            validate="many_to_one",
        )
        missing_pca = calibrated["_pca_authorized"].ne(True)
        if missing_pca.any():
            rejected = calibrated.loc[
                missing_pca,
                ["matrix_family", "preprocessing"],
            ].drop_duplicates()
            raise ValueError(
                "03B domain contains preprocessing values outside the PCA "
                f"shortlist: {rejected.to_dict('records')}"
            )
        calibrated = calibrated.drop(columns="_pca_authorized")

    if "source_config_id" not in calibrated:
        calibrated["source_config_id"] = calibrated["calibration_id"]
    if "model_group_id" not in calibrated:
        calibrated["model_group_id"] = calibrated["calibration_id"]
    default_seed_values = sorted(set(map(int, random_seeds)))
    if not default_seed_values:
        raise ValueError("At least one search evaluation seed is required.")

    canonical_columns = [
        "calibration_track",
        "decision_mode",
        "matrix_family",
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "preprocessing",
        "preprocessing_steps",
        "rule_family",
        "rule_variant",
        "limit_source",
        "n_components",
        "alpha",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
        "object_threshold",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    ]
    calibrated = calibrated.drop_duplicates(
        canonical_columns,
        keep="first",
    ).copy()
    calibrated["domain_config_id"] = [
        hash_internal_calibration_configuration(
            {column: row[column] for column in canonical_columns},
            prefix="icdomain",
        )
        for row in calibrated.to_dict("records")
    ]

    runtime_rows: list[dict[str, Any]] = []
    for row in calibrated.to_dict("records"):
        validation = validate_simca_configuration(row)
        if not validation["is_valid"]:
            raise ValueError(
                f"Invalid calibrated configuration {row['calibration_id']}: "
                f"{validation['technical_errors']}"
            )
        raw_seeds = row.get("random_states_json")
        try:
            seed_values = (
                sorted(set(map(int, json.loads(str(raw_seeds)))))
                if raw_seeds is not None and not pd.isna(raw_seeds)
                else []
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            seed_values = []
        if not seed_values:
            seed_values = (
                default_seed_values
                if (
                    str(row["matrix_method"]) == "balanced_pixels"
                    and str(row["balanced_pixel_strategy"]) == "random"
                )
                else default_seed_values[:1]
            )
        row["random_states_json"] = json.dumps(seed_values)
        fit_payload = {
            column: row[column]
            for column in canonical_columns
            if column
            not in {
                "calibration_track",
                "decision_mode",
                "rule_family",
                "rule_variant",
                "limit_source",
                "object_threshold",
                "three_way_lower_threshold",
                "three_way_upper_threshold",
            }
        }
        fit_config_id = hash_internal_calibration_configuration(
            fit_payload,
            prefix="icsearchfit",
        )
        for seed in seed_values:
            runtime = {
                **row,
                "fit_config_id": fit_config_id,
                "random_state": int(seed),
            }
            runtime["config_id"] = (
                hash_internal_calibration_configuration(
                    {
                        "domain_key": row["domain_config_id"],
                        "random_state": int(seed),
                    },
                    prefix="icsearch",
                )
            )
            runtime_rows.append(runtime)

    domain = pd.DataFrame(runtime_rows)
    if domain.empty:
        raise RuntimeError("The seed-expanded calibrated domain is empty.")
    domain = _attach_data_configuration_ids(domain)
    domain = domain.sort_values(
        ["calibration_track", "domain_config_id", "random_state"],
        kind="mergesort",
    ).reset_index(drop=True)
    domain.attrs["domain_audit"] = [
        {
            "stage": "calibrated_rows",
            "n_rows": int(len(calibrated)),
        },
        {
            "stage": "seed_expanded_runtime_rows",
            "n_rows": int(len(domain)),
        },
    ]
    return domain


def build_internal_calibrated_hyperparameters_8tracks(
    configurations: pd.DataFrame,
    metrics_2way: pd.DataFrame,
    thresholds_3way: pd.DataFrame,
    *,
    tolerance: float = expcfg.INTERNAL_CALIBRATION_PERFORMANCE_PLATEAU_TOLERANCE,
    allowed_unsupported_track_ids: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply risk, k/m plateau, seed consensus and Pareto by track.

    A track may be declared unsupported only when it has technically valid
    metrics but no candidate satisfying the frozen risk constraints.  Missing
    metrics and later selection failures remain hard errors in the notebook.
    """
    allowed_unsupported = set(map(str, allowed_unsupported_track_ids))
    known_track_ids = set(expcfg.SIMCA_EVALUATION_TRACK_IDS.values())
    unknown_unsupported = sorted(allowed_unsupported - known_track_ids)
    if unknown_unsupported:
        raise ValueError(
            "Unknown allowed unsupported track IDs: "
            f"{unknown_unsupported}"
        )
    config_map = configurations.drop_duplicates("evaluation_config_id").copy()
    primary_2way = metrics_2way.loc[
        metrics_2way["aggregation_level"].eq("macro_source_image")
    ]
    two_way = (
        primary_2way.groupby("evaluation_config_id", as_index=False)
        .agg(
            target_miss_rate=("target_miss_rate", "max"),
            false_accept_rate=("false_accept_rate", "max"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
    )
    if not two_way.empty:
        two_way["three_way_lower_threshold"] = np.nan
        two_way["three_way_upper_threshold"] = np.nan
        two_way["uncertain_rate"] = 0.0
        two_way["coverage_rate"] = 1.0
        two_way["decided_balanced_accuracy"] = two_way[
            "balanced_accuracy"
        ]
        two_way["risk_feasible"] = (
            two_way["target_miss_rate"].le(
                expcfg.INTERNAL_CALIBRATION_MAX_FN_RATE
            )
            & two_way["false_accept_rate"].le(
                expcfg.INTERNAL_CALIBRATION_MAX_FP_RATE
            )
            & two_way["balanced_accuracy"].ge(
                expcfg.INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY
            )
        )
    three_way = thresholds_3way.loc[
        thresholds_3way["evaluation_fold"].eq(-1)
        & thresholds_3way["decision_scope"].astype(str).eq("direct")
        & thresholds_3way["score_type"].astype(str).eq("simca_margin")
        & thresholds_3way["selected"].fillna(False).astype(bool)
    ].copy()
    if not three_way.empty:
        three_way["balanced_accuracy"] = three_way[
            "decided_balanced_accuracy"
        ]
        three_way["risk_feasible"] = three_way["feasible"].astype(bool)
    metric_columns = [
        "evaluation_config_id",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
        "target_miss_rate",
        "false_accept_rate",
        "uncertain_rate",
        "coverage_rate",
        "balanced_accuracy",
        "decided_balanced_accuracy",
        "risk_feasible",
    ]
    candidates = pd.concat(
        [
            two_way.reindex(columns=metric_columns),
            three_way.reindex(columns=metric_columns),
        ],
        ignore_index=True,
        sort=False,
    ).merge(
        config_map,
        on="evaluation_config_id",
        how="left",
        validate="many_to_one",
    )
    audit_rows: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    for evaluation_track in expcfg.SIMCA_EVALUATION_TRACKS:
        initial_track_configurations = config_map.loc[
            config_map["evaluation_track"].astype(str).eq(evaluation_track),
            "evaluation_config_id",
        ].nunique()
        track = candidates.loc[
            candidates["evaluation_track"].astype(str).eq(evaluation_track)
        ].copy()
        track_id = expcfg.SIMCA_EVALUATION_TRACK_IDS[evaluation_track]
        audit = {
            "audit_type": "selection_funnel",
            "evaluation_track": evaluation_track,
            "track_id": track_id,
            "n_initial": int(initial_track_configurations),
            "n_technical_valid": int(
                track["evaluation_config_id"].nunique()
            ),
            "n_risk_feasible": 0,
            "n_k_plateau": 0,
            "n_m_plateau": 0,
            "n_seed_consensus": 0,
            "n_pareto": 0,
            "track_status": "no_feasible_configuration",
            "failure_reason": "no_metrics",
        }
        if track.empty:
            audit_rows.append(audit)
            continue
        track = track.loc[track["risk_feasible"].fillna(False).astype(bool)]
        audit["n_risk_feasible"] = int(len(track))
        if track.empty:
            audit["failure_reason"] = "risk_constraints"
            audit_rows.append(audit)
            continue
        semantic = [
            column
            for column in (
                "evaluation_track",
                "matrix_method",
                "projection_matrix_method",
                "preprocessing",
                "preprocessing_steps",
                "rule_variant",
                "limit_source",
                "alpha",
                "sg_window_length",
                "sg_polyorder",
                "balanced_pixel_strategy",
                "m",
                "three_way_lower_threshold",
                "three_way_upper_threshold",
            )
            if column in track
        ]
        k_group = [column for column in semantic if column != "m"]
        best_by_group = track.groupby(k_group, dropna=False)[
            "balanced_accuracy"
        ].transform("max")
        track = track.loc[
            track["balanced_accuracy"].ge(best_by_group - float(tolerance))
        ]
        minimum_k = track.groupby(k_group, dropna=False)[
            "n_components"
        ].transform("min")
        track = track.loc[track["n_components"].eq(minimum_k)]
        audit["n_k_plateau"] = int(len(track))
        balanced = track["matrix_method"].astype(str).eq("balanced_pixels")
        if balanced.any():
            m_group = [column for column in semantic if column != "m"]
            minimum_m = track.loc[balanced].groupby(
                m_group, dropna=False
            )["m"].transform("min")
            keep_balanced = track.loc[balanced, "m"].eq(minimum_m)
            track = pd.concat(
                [track.loc[~balanced], track.loc[balanced].loc[keep_balanced]],
                ignore_index=True,
            )
        audit["n_m_plateau"] = int(len(track))
        seed_group = [
            column
            for column in semantic + ["n_components"]
            if column not in {"random_state"}
        ]
        consensus_rows = []
        for _, seed_rows in track.groupby(seed_group, dropna=False):
            base = seed_rows.iloc[0].to_dict()
            observed_seeds = sorted(
                map(
                    int,
                    pd.to_numeric(seed_rows["random_state"])
                    .astype(int)
                    .unique(),
                )
            )
            expected_seeds = (
                list(expcfg.INTERNAL_CALIBRATION_RANDOM_SEEDS)
                if base["matrix_method"] == "balanced_pixels"
                and base["balanced_pixel_strategy"] == "random"
                else observed_seeds[:1]
            )
            if observed_seeds != expected_seeds:
                continue
            for metric in (
                "target_miss_rate",
                "false_accept_rate",
                "uncertain_rate",
            ):
                base[metric] = float(seed_rows[metric].max())
            for metric in (
                "coverage_rate",
                "balanced_accuracy",
                "decided_balanced_accuracy",
            ):
                base[metric] = float(seed_rows[metric].min())
            base["random_states_json"] = json.dumps(observed_seeds)
            base["n_seeds_evaluated"] = int(len(observed_seeds))
            base["member_evaluation_config_ids_json"] = json.dumps(
                sorted(seed_rows["evaluation_config_id"].astype(str).unique())
            )
            consensus_rows.append(base)
        consensus = pd.DataFrame(consensus_rows)
        audit["n_seed_consensus"] = int(len(consensus))
        if consensus.empty:
            audit["failure_reason"] = "seed_consensus"
            audit_rows.append(audit)
            continue
        objectives = [
            "target_miss_rate",
            "false_accept_rate",
            "uncertain_rate",
            "coverage_rate",
            "decided_balanced_accuracy",
        ]
        if not np.isfinite(consensus[objectives].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite calibration objective before Pareto.")
        pareto = pareto_front_by_group(
            consensus,
            group_cols=["evaluation_track"],
            minimize_cols=[
                "target_miss_rate",
                "false_accept_rate",
                "uncertain_rate",
            ],
            maximize_cols=["coverage_rate", "decided_balanced_accuracy"],
        )
        pareto["pareto_front"] = True
        pareto["selection_method"] = "hard_constraints|pareto"
        pareto["selection_objectives_json"] = json.dumps(objectives)
        pareto["direct_2way_threshold"] = np.where(
            pareto["decision_mode"].eq("2way"), 0.0, np.nan
        )
        pareto["secondary_object_threshold"] = np.nan
        pareto["calibration_status"] = "calibrated_8tracks"
        pareto["calibration_id"] = [
            hash_internal_calibration_configuration(
                {
                    "evaluation_track": row["evaluation_track"],
                    "members": row["member_evaluation_config_ids_json"],
                    "lower": row["three_way_lower_threshold"],
                    "upper": row["three_way_upper_threshold"],
                    "protocol_version": expcfg.PROTOCOL_VERSION,
                    "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
                },
                prefix="iccal8",
            )
            for row in pareto.to_dict("records")
        ]
        audit["n_pareto"] = int(len(pareto))
        audit["track_status"] = "calibrated"
        audit["failure_reason"] = ""
        audit_rows.append(audit)
        selected_parts.append(pareto)
    calibrated = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )
    audit = _with_schema(
        pd.DataFrame(audit_rows),
        expcfg.INTERNAL_CALIBRATION_AUDIT_V2_COLUMNS,
    )
    unsupported = (
        audit["track_id"].astype(str).isin(allowed_unsupported)
        & audit["track_status"].astype(str).eq(
            "no_feasible_configuration"
        )
        & audit["failure_reason"].astype(str).eq("risk_constraints")
    )
    audit.loc[unsupported, "track_status"] = "unsupported"
    return calibrated, audit


def build_calibration_domain_8tracks(
    calibrated_hyperparameters: pd.DataFrame,
    configurations: pd.DataFrame,
    *,
    pca_shortlist_id: str,
    protocol_hash: str,
    unsupported_track_ids: Sequence[str] = (),
) -> pd.DataFrame:
    """Build the executable domain, excluding explicitly unsupported tracks."""
    unsupported_ids = set(map(str, unsupported_track_ids))
    track_by_id = {
        str(track_id): str(track)
        for track, track_id in expcfg.SIMCA_EVALUATION_TRACK_IDS.items()
    }
    unknown_unsupported = sorted(unsupported_ids - set(track_by_id))
    if unknown_unsupported:
        raise ValueError(
            f"Unknown unsupported track IDs: {unknown_unsupported}"
        )
    unsupported_tracks = {track_by_id[track_id] for track_id in unsupported_ids}
    rows: list[dict[str, Any]] = []
    config_map = configurations.set_index("evaluation_config_id", drop=False)
    for calibrated in calibrated_hyperparameters.to_dict("records"):
        member_ids = json.loads(
            str(calibrated["member_evaluation_config_ids_json"])
        )
        for member_id in member_ids:
            if str(member_id) not in config_map.index:
                raise RuntimeError(
                    f"Calibrated member is absent from configurations: {member_id}"
                )
            runtime = config_map.loc[str(member_id)].to_dict()
            runtime.update(
                calibration_id=str(calibrated["calibration_id"]),
                calibration_status=str(calibrated["calibration_status"]),
                direct_2way_threshold=calibrated[
                    "direct_2way_threshold"
                ],
                secondary_object_threshold=calibrated[
                    "secondary_object_threshold"
                ],
                three_way_lower_threshold=calibrated[
                    "three_way_lower_threshold"
                ],
                three_way_upper_threshold=calibrated[
                    "three_way_upper_threshold"
                ],
                protocol_version=str(expcfg.PROTOCOL_VERSION),
                schema_version=str(expcfg.RESULTS_SCHEMA_VERSION),
                protocol_hash=str(protocol_hash),
                pca_shortlist_id=str(pca_shortlist_id),
            )
            runtime["domain_config_id"] = hash_internal_calibration_configuration(
                {
                    "evaluation_config_id": str(member_id),
                    "calibration_id": str(calibrated["calibration_id"]),
                    "protocol_hash": str(protocol_hash),
                    "pca_shortlist_id": str(pca_shortlist_id),
                },
                prefix="icdomain8",
            )
            rows.append(runtime)
    domain = pd.DataFrame(rows)
    if domain.empty:
        raise RuntimeError("The 8-track calibration domain is empty.")
    if not domain["domain_config_id"].is_unique:
        raise RuntimeError("domain_config_id must be unique.")
    observed_tracks = set(domain["evaluation_track"].astype(str))
    expected_tracks = set(expcfg.SIMCA_EVALUATION_TRACKS)
    if observed_tracks.intersection(unsupported_tracks):
        raise RuntimeError(
            "An explicitly unsupported track entered the executable domain."
        )
    missing_tracks = expected_tracks - observed_tracks
    if missing_tracks != unsupported_tracks:
        raise RuntimeError(
            "The executable calibration domain has undeclared missing tracks: "
            f"missing={sorted(missing_tracks)}, "
            f"declared_unsupported={sorted(unsupported_tracks)}"
        )
    if "object_threshold" in domain:
        domain = domain.drop(columns="object_threshold")
    missing = [
        column
        for column in expcfg.SIMCA_CALIBRATION_DOMAIN_COLUMNS
        if column not in domain
    ]
    if missing:
        raise RuntimeError(f"Incomplete 8-track domain schema: {missing}")
    domain = domain.loc[:, list(expcfg.SIMCA_CALIBRATION_DOMAIN_COLUMNS)]
    return domain.sort_values(
        ["track_id", "domain_config_id"], kind="mergesort"
    ).reset_index(drop=True)


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
        "pca_shortlist_id",
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
            ).get("completed_fit_config_ids", ())
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
    pca_shortlist_id: str,
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
        "pca_shortlist_id": str(pca_shortlist_id),
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
                    "pca_shortlist_id",
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
        "pca_shortlist_id",
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
