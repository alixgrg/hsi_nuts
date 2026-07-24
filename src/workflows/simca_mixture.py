from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.utils import filter_records
from src.workflows.simca_candidates import (
    normalize_simca_candidate_columns,
    validate_simca_candidate_contract,
)
from src.workflows.simca_pure_test import run_pure_test_refit_batches
from src.workflows.simca_tables import (
    concat_nonempty_tables,
    read_simca_table,
    resolve_merge_suffix_columns,
    write_simca_table,
)


DEFAULT_MIXTURE_EVALUATION_STAGE = expcfg.MIXTURE_APPLICATION_EVALUATION_STAGE

MIXTURE_REQUIRED_OUTPUT_KEYS = (
    "selected_configs",
    "2way_object_metrics",
    "2way_pixel_metrics",
    "3way_object_metrics",
    "metrics_long",
)

MIXTURE_CORE_SAVE_KEYS = (
    "selected_configs",
    "2way_object_metrics",
    "2way_pixel_metrics",
    "3way_object_metrics",
    "metrics_long",
    "object_image_diagnostics",
    "pixel_image_diagnostics",
    "3way_object_image_diagnostics",
    "pixel_errors_by_image",
    "errors",
    "batch_manifest",
    "summary",
    "guardrails",
    "protocol",
)


def _json_ready(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str)
    return str(value)


def _track_decision_mode(track: Any) -> str:
    token = str(track)
    return "3way" if token.endswith("_3way") else "2way"


def build_mixture_projection_filters(
    *,
    mixture_keys: Sequence[str] | None = None,
) -> dict[str, list[Any]]:
    """Return the canonical projection filter for mixture objects."""
    filters: dict[str, list[Any]] = {"sample_kind": ["mixture"]}
    if mixture_keys is not None:
        filters["source_clean_key"] = [str(key) for key in mixture_keys]
    return filters


def restore_mixture_selected_configs(
    *,
    final_selected_models_df: pd.DataFrame,
    candidate_panel_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    max_models_per_track: int | None = expcfg.MIXTURE_APPLICATION_MAX_MODELS_PER_TRACK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restore full refit configs for models selected in notebook 06B.

    `final_selected_models.parquet` is intentionally compact. This function
    keeps the 06B track assignment and rank, then joins the full 06A candidate
    panel required for final refit/application.
    """
    if final_selected_models_df is None or len(final_selected_models_df) == 0:
        raise ValueError("final_selected_models_df is empty.")
    if candidate_panel_df is None or len(candidate_panel_df) == 0:
        raise ValueError("candidate_panel_df is empty.")

    selected = final_selected_models_df.copy()
    selected["selected_config_id"] = selected["selected_config_id"].astype(str)
    selected["assigned_selection_track"] = selected.get(
        "assigned_selection_track",
        selected["selection_track"],
    )
    selected["assigned_selection_track"] = selected["assigned_selection_track"].fillna(selected["selection_track"])
    selected["_selection_order"] = np.arange(len(selected), dtype=int)

    if max_models_per_track is not None:
        sort_cols = [
            col
            for col in ["selection_track", "final_rank_in_track", "pareto_rank_in_track", "_selection_order"]
            if col in selected.columns
        ]
        selected = (
            selected.sort_values(sort_cols)
            .groupby("selection_track", dropna=False, group_keys=False)
            .head(int(max_models_per_track))
            .reset_index(drop=True)
        )
        selected["_selection_order"] = np.arange(len(selected), dtype=int)

    candidates = normalize_simca_candidate_columns(candidate_panel_df).copy()
    candidates["selected_config_id"] = candidates["selected_config_id"].astype(str)
    candidates = candidates.drop_duplicates("selected_config_id")

    restored = selected.merge(
        candidates,
        on="selected_config_id",
        how="left",
        validate="many_to_one",
        suffixes=("_selected", ""),
    )
    restored = resolve_merge_suffix_columns(restored, prefer="y")

    missing = restored.loc[restored["candidate_id"].isna(), "selected_config_id"].astype(str).tolist()
    if missing:
        raise RuntimeError(
            "Some 06B selected_config_id values are missing from the 06A candidate panel. "
            f"Preview: {missing[:10]}"
        )

    restored["selection_track"] = restored["assigned_selection_track"].astype(str)
    restored["decision_mode"] = restored["selection_track"].map(_track_decision_mode)
    restored = restored.sort_values("_selection_order").drop(columns=["_selection_order"]).reset_index(drop=True)

    if thresholds_df is None or len(thresholds_df) == 0:
        raise ValueError("thresholds_df is empty.")
    thresholds = thresholds_df.copy()
    thresholds["selected_config_id"] = thresholds["selected_config_id"].astype(str)
    threshold_cols = [
        col
        for col in [
            "selected_config_id",
            "three_way_lower_threshold",
            "three_way_upper_threshold",
        ]
        if col in thresholds.columns
    ]
    selected_thresholds = (
        thresholds[threshold_cols]
        .drop_duplicates("selected_config_id")
        .loc[lambda df: df["selected_config_id"].isin(restored["selected_config_id"].astype(str))]
        .reset_index(drop=True)
    )
    missing_threshold_ids = sorted(
        set(restored["selected_config_id"].astype(str)) - set(selected_thresholds["selected_config_id"].astype(str))
    )
    if missing_threshold_ids:
        raise RuntimeError(
            "Missing fixed 3-way thresholds for selected mixture model(s). "
            f"Preview: {missing_threshold_ids[:10]}"
        )

    restored = restored.merge(
        selected_thresholds,
        on="selected_config_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_threshold"),
    )
    for col in ("three_way_lower_threshold", "three_way_upper_threshold"):
        suffixed = f"{col}_threshold"
        if suffixed not in restored.columns:
            continue
        if col in restored.columns:
            restored[col] = restored[col].combine_first(restored[suffixed])
        else:
            restored[col] = restored[suffixed]
        restored = restored.drop(columns=[suffixed])
    restored = resolve_merge_suffix_columns(restored, prefer="x")
    validate_simca_candidate_contract(restored)
    return restored.reset_index(drop=True), selected_thresholds.reset_index(drop=True)


def build_mixture_guardrails(
    *,
    selected_configs_df: pd.DataFrame,
    final_selected_models_df: pd.DataFrame,
    candidate_panel_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    object_db: Mapping[str, Mapping[str, Any]] | None,
    train_batches: Sequence[int],
    projection_filters: Mapping[str, Any],
    expected_tracks: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
    target_class: str = expcfg.TARGET_CLASS,
) -> pd.DataFrame:
    """Build notebook-07 guardrails before mixture application."""
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "", severity: str = "error", n_records: int | None = None) -> None:
        rows.append(
            {
                "check_name": check_name,
                "passed": bool(passed),
                "status": "passed" if passed else "failed",
                "severity": severity,
                "details": _json_ready(details),
                "n_records": np.nan if n_records is None else int(n_records),
            }
        )

    selected_tracks = (
        set(final_selected_models_df.get("selection_track", pd.Series(dtype=str)).dropna().astype(str))
        if final_selected_models_df is not None
        else set()
    )
    add(
        "final_selection_tracks_available",
        set(map(str, expected_tracks)).issubset(selected_tracks),
        sorted(set(map(str, expected_tracks)) - selected_tracks),
        n_records=0 if final_selected_models_df is None else len(final_selected_models_df),
    )
    add(
        "selected_configs_restored",
        selected_configs_df is not None and len(selected_configs_df) > 0,
        n_records=0 if selected_configs_df is None else len(selected_configs_df),
    )
    add(
        "selected_configs_match_final_selection",
        selected_configs_df is not None
        and final_selected_models_df is not None
        and 0 < len(selected_configs_df) <= len(final_selected_models_df),
        {
            "restored": 0 if selected_configs_df is None else len(selected_configs_df),
            "final_selected": 0 if final_selected_models_df is None else len(final_selected_models_df),
        },
    )
    add(
        "candidate_panel_available",
        candidate_panel_df is not None and len(candidate_panel_df) > 0,
        n_records=0 if candidate_panel_df is None else len(candidate_panel_df),
    )
    add(
        "three_way_thresholds_available",
        thresholds_df is not None and len(thresholds_df) > 0,
        n_records=0 if thresholds_df is None else len(thresholds_df),
    )
    add(
        "train_batches_are_final_pure_batches",
        list(map(int, train_batches)) == list(map(int, expcfg.MIXTURE_FINAL_TRAIN_BATCHES)),
        list(map(int, train_batches)),
    )
    add(
        "projection_filters_select_mixtures",
        list(projection_filters.get("sample_kind", [])) == ["mixture"],
        dict(projection_filters),
    )

    if object_db is not None:
        n_train_target = len(
            filter_records(
                object_db,
                return_items=False,
                sample_kind="pure",
                object_nut_type=target_class,
                batch=list(map(int, train_batches)),
            )
        )
        n_mixture = len(
            filter_records(
                object_db,
                return_items=False,
                **dict(projection_filters),
            )
        )
        add("final_train_target_objects_available", n_train_target > 0, f"{n_train_target} target objects", n_records=n_train_target)
        add("mixture_objects_available", n_mixture > 0, f"{n_mixture} mixture objects", n_records=n_mixture)

    return pd.DataFrame(rows)


def validate_mixture_guardrails(guardrails_df: pd.DataFrame) -> pd.DataFrame:
    """Raise if an error-level notebook-07 guardrail failed."""
    if guardrails_df is None or len(guardrails_df) == 0:
        raise RuntimeError("Mixture guardrail table is empty.")
    failed = guardrails_df.loc[
        ~guardrails_df["passed"].astype(bool)
        & guardrails_df.get("severity", "error").astype(str).eq("error")
    ]
    if len(failed) > 0:
        raise RuntimeError(
            "Mixture guardrail failure(s): "
            + ", ".join(failed["check_name"].astype(str).tolist())
        )
    return guardrails_df


def _filter_to_assigned_track(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()
    required = {"selection_track", "assigned_selection_track"}
    if not required.issubset(df.columns):
        return df.copy()
    assigned = df["assigned_selection_track"].astype(str)
    observed = df["selection_track"].astype(str)
    return df.loc[assigned.eq(observed)].reset_index(drop=True)


def prepare_mixture_outputs(
    outputs: Mapping[str, pd.DataFrame],
    *,
    keep_only_assigned_track_metrics: bool = expcfg.MIXTURE_APPLICATION_KEEP_ONLY_ASSIGNED_TRACK_METRICS,
) -> dict[str, pd.DataFrame]:
    """Materialize standard mixture outputs and optionally keep only assigned final tracks."""
    out = {key: (value.copy() if value is not None else pd.DataFrame()) for key, value in outputs.items()}
    metric_keys = [
        "2way_object_metrics",
        "2way_pixel_metrics",
        "3way_object_metrics",
        "object_image_diagnostics",
        "pixel_image_diagnostics",
        "3way_object_image_diagnostics",
    ]
    if keep_only_assigned_track_metrics:
        for key in metric_keys:
            out[key] = _filter_to_assigned_track(out.get(key, pd.DataFrame()))

    out["metrics_long"] = concat_nonempty_tables(
        [
            out.get("2way_object_metrics"),
            out.get("2way_pixel_metrics"),
            out.get("3way_object_metrics"),
        ]
    )
    out["summary"] = summarize_mixture_outputs(out)
    return out


def run_mixture_application_batches(
    *,
    selected_configs_df: pd.DataFrame,
    object_db: Mapping[str, Mapping[str, Any]],
    image_db: Mapping[str, Mapping[str, Any]],
    train_filters: Mapping[str, Any],
    projection_filters: Mapping[str, Any],
    preprocessing_configs: Mapping[str, Mapping[str, Sequence[str]]],
    thresholds_df: pd.DataFrame,
    evaluation_stage: str = DEFAULT_MIXTURE_EVALUATION_STAGE,
    wavelengths: Sequence[float] | None = None,
    random_state: int = expcfg.RANDOM_STATE,
    replace: bool = expcfg.REPLACE_BALANCED_PIXELS,
    cv_n_splits: int | None = expcfg.CV_N_SPLITS,
    cv_group_col: str = expcfg.CV_GROUP_COL,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    batch_size: int = expcfg.MIXTURE_APPLICATION_BATCH_SIZE,
    batch_metrics_dir: str | Path | None = None,
    batch_objects_dir: str | Path | None = None,
    batch_pixels_dir: str | Path | None = None,
    batch_3way_objects_dir: str | Path | None = None,
    save_batch_metric_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_BATCH_METRIC_TABLES,
    save_batch_object_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_BATCH_OBJECT_TABLES,
    save_batch_pixel_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_BATCH_PIXEL_TABLES,
    save_batch_3way_object_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_BATCH_3WAY_OBJECT_TABLES,
    save_combined_object_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_COMBINED_OBJECT_TABLES,
    save_combined_pixel_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_COMBINED_PIXEL_TABLES,
    save_combined_3way_object_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_COMBINED_3WAY_OBJECT_TABLES,
    keep_only_assigned_track_metrics: bool = expcfg.MIXTURE_APPLICATION_KEEP_ONLY_ASSIGNED_TRACK_METRICS,
    fixed_thresholds_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Refit final selected configs on pure batches 1-4 and project mixture objects."""
    outputs = run_pure_test_refit_batches(
        selected_configs_df=selected_configs_df,
        object_db=object_db,
        image_db=image_db,
        train_filters=dict(train_filters),
        projection_filters=dict(projection_filters),
        preprocessing_configs=preprocessing_configs,
        thresholds_df=thresholds_df,
        evaluation_stage=evaluation_stage,
        wavelengths=wavelengths,
        random_state=random_state,
        replace=replace,
        cv_n_splits=cv_n_splits,
        cv_group_col=cv_group_col,
        target_class=target_class,
        non_target_label=non_target_label,
        batch_size=batch_size,
        batch_metrics_dir=batch_metrics_dir,
        batch_objects_dir=batch_objects_dir,
        batch_pixels_dir=batch_pixels_dir,
        batch_3way_objects_dir=batch_3way_objects_dir,
        save_batch_metric_tables=save_batch_metric_tables,
        save_batch_object_tables=save_batch_object_tables,
        save_batch_pixel_tables=save_batch_pixel_tables,
        save_batch_3way_object_tables=save_batch_3way_object_tables,
        save_combined_object_tables=save_combined_object_tables,
        save_combined_pixel_tables=save_combined_pixel_tables,
        save_combined_3way_object_tables=save_combined_3way_object_tables,
        fixed_thresholds_path=fixed_thresholds_path,
    )
    return prepare_mixture_outputs(
        outputs,
        keep_only_assigned_track_metrics=keep_only_assigned_track_metrics,
    )


def validate_mixture_outputs(outputs: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Validate required notebook-07 outputs and return a mutable copy."""
    out = {key: (value.copy() if value is not None else pd.DataFrame()) for key, value in outputs.items()}
    missing_or_empty = [
        key
        for key in MIXTURE_REQUIRED_OUTPUT_KEYS
        if key not in out or out[key] is None or len(out[key]) == 0
    ]
    if missing_or_empty:
        raise RuntimeError(
            "Mixture outputs are incomplete. Empty required table(s): "
            + ", ".join(missing_or_empty)
        )
    out.setdefault("errors", pd.DataFrame(columns=["selected_config_id", "evaluation_split", "error"]))
    out.setdefault("objects", pd.DataFrame())
    out.setdefault("pixels", pd.DataFrame())
    out.setdefault("3way_objects", pd.DataFrame())
    out.setdefault("batch_manifest", pd.DataFrame())
    out.setdefault("summary", summarize_mixture_outputs(out))
    return out


def summarize_mixture_outputs(outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact descriptive summary of mixture metrics."""
    metrics_long_df = outputs.get("metrics_long", pd.DataFrame())
    if metrics_long_df is None or len(metrics_long_df) == 0:
        return pd.DataFrame()
    group_cols = [
        col
        for col in [
            "selection_track",
            "assigned_selection_track",
            "matrix_family",
            "decision_mode",
            "metric_level",
        ]
        if col in metrics_long_df.columns
    ]
    d = metrics_long_df.copy()
    for col in [
        "fn_rate",
        "fp_rate",
        "balanced_accuracy",
        "target_miss_rate",
        "non_target_false_accept_rate",
        "uncertain_rate",
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = np.nan
    return (
        d.groupby(group_cols, dropna=False)
        .agg(
            n_models=("selected_config_id", "nunique"),
            n_rows=("selected_config_id", "size"),
            best_fn_rate=("fn_rate", "min"),
            best_fp_rate=("fp_rate", "min"),
            best_balanced_accuracy=("balanced_accuracy", "max"),
            median_fn_rate=("fn_rate", "median"),
            median_fp_rate=("fp_rate", "median"),
            median_balanced_accuracy=("balanced_accuracy", "median"),
            best_target_miss_rate=("target_miss_rate", "min"),
            best_non_target_false_accept_rate=("non_target_false_accept_rate", "min"),
            best_uncertain_rate=("uncertain_rate", "min"),
            median_target_miss_rate=("target_miss_rate", "median"),
            median_non_target_false_accept_rate=("non_target_false_accept_rate", "median"),
            median_uncertain_rate=("uncertain_rate", "median"),
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_mixture_protocol(settings: Mapping[str, Any], outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the notebook-07 protocol table."""
    row = dict(settings)
    for key, value in list(row.items()):
        if isinstance(value, Path):
            row[key] = str(value)
        elif isinstance(value, (dict, list, tuple, set)):
            row[key] = json.dumps(value, default=str)
    row.update(
        {
            "n_selected_models": int(len(outputs.get("final_selected_models", pd.DataFrame()))),
            "n_restored_configs": int(len(outputs.get("selected_configs", pd.DataFrame()))),
            "n_2way_object_metrics": int(len(outputs.get("2way_object_metrics", pd.DataFrame()))),
            "n_2way_pixel_metrics": int(len(outputs.get("2way_pixel_metrics", pd.DataFrame()))),
            "n_3way_object_metrics": int(len(outputs.get("3way_object_metrics", pd.DataFrame()))),
            "n_metrics_long": int(len(outputs.get("metrics_long", pd.DataFrame()))),
            "n_object_image_diagnostics": int(len(outputs.get("object_image_diagnostics", pd.DataFrame()))),
            "n_pixel_image_diagnostics": int(len(outputs.get("pixel_image_diagnostics", pd.DataFrame()))),
            "n_3way_object_image_diagnostics": int(len(outputs.get("3way_object_image_diagnostics", pd.DataFrame()))),
            "n_errors": int(len(outputs.get("errors", pd.DataFrame()))),
        }
    )
    return pd.DataFrame([row])


def save_mixture_outputs(
    outputs: Mapping[str, pd.DataFrame],
    paths: Mapping[str, str | Path],
    *,
    save_combined_object_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_COMBINED_OBJECT_TABLES,
    save_combined_pixel_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_COMBINED_PIXEL_TABLES,
    save_combined_3way_object_tables: bool = expcfg.MIXTURE_APPLICATION_SAVE_COMBINED_3WAY_OBJECT_TABLES,
) -> list[Path]:
    """Save the standard notebook-07 output set."""
    saved: list[Path] = []
    for key in MIXTURE_CORE_SAVE_KEYS:
        if key in outputs and key in paths:
            saved.append(write_simca_table(outputs[key], paths[key]))

    optional = []
    if save_combined_object_tables:
        optional.append("objects")
    if save_combined_pixel_tables:
        optional.append("pixels")
    if save_combined_3way_object_tables:
        optional.append("3way_objects")

    for key in optional:
        table = outputs.get(key, pd.DataFrame())
        if key in paths and table is not None and len(table) > 0:
            saved.append(write_simca_table(table, paths[key]))
    return saved


def missing_existing_mixture_paths(paths: Mapping[str, str | Path]) -> list[Path]:
    """Return missing required mixture output paths for reload mode."""
    return [
        Path(paths[key])
        for key in MIXTURE_REQUIRED_OUTPUT_KEYS
        if key not in paths or not Path(paths[key]).exists()
    ]


def load_existing_mixture_outputs(paths: Mapping[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Load already generated notebook-07 outputs."""
    missing = missing_existing_mixture_paths(paths)
    if missing:
        raise FileNotFoundError(
            "Required mixture output file(s) are missing:\n"
            + "\n".join(map(str, missing))
        )
    return {
        key: read_simca_table(path)
        for key, path in paths.items()
        if Path(path).exists()
    }


def choose_mixture_diagnostic_images(
    image_metrics_df: pd.DataFrame,
    *,
    config_id: str | None = None,
    n_images: int = expcfg.MIXTURE_APPLICATION_DIAGNOSTIC_TOP_IMAGES,
    image_col: str = "source_image",
) -> pd.DataFrame:
    """Choose a small set of difficult mixture images for notebook-level plots."""
    if image_metrics_df is None or len(image_metrics_df) == 0:
        return pd.DataFrame()
    d = image_metrics_df.copy()
    if config_id is not None and "selected_config_id" in d.columns:
        d = d.loc[d["selected_config_id"].astype(str).eq(str(config_id))].copy()
    if d.empty or image_col not in d.columns:
        return pd.DataFrame()
    sort_cols = [
        col
        for col in [
            "fn_rate",
            "fp_rate",
            "target_miss_rate",
            "non_target_false_accept_rate",
            "uncertain_rate",
            "balanced_accuracy",
        ]
        if col in d.columns
    ]
    if sort_cols:
        for col in sort_cols:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        ascending = [False] * len(sort_cols)
        if "balanced_accuracy" in sort_cols:
            ascending[sort_cols.index("balanced_accuracy")] = True
        d = d.sort_values(sort_cols, ascending=ascending)
    return d.drop_duplicates(image_col).head(int(n_images)).reset_index(drop=True)
