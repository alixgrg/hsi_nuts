from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.decision.metrics import (
    summarize_object_errors_by_image,
    summarize_pixel_errors_by_image,
)
from src.decision.uncertainty import (
    add_three_way_confidence,
    evaluate_three_way_by_config,
)
from src.utils import filter_records
from src.workflows.simca import refit_selected_simca_configs
from src.workflows.simca_candidates import (
    add_selection_track,
    normalize_simca_candidate_columns,
    validate_simca_candidate_contract,
    validate_simca_evaluation_contract,
)
from src.workflows.simca_robustness import validate_no_pure_test_inputs
from src.workflows.simca_selection_utils import materialize_selection_metrics
from src.workflows.simca_tables import (
    concat_nonempty_tables,
    iter_dataframe_batches,
    read_simca_table,
    write_simca_table,
)


DEFAULT_PURE_TEST_EVALUATION_STAGE = "pure_test_batch_4"

PURE_TEST_REQUIRED_OUTPUT_KEYS = (
    "2way_object_metrics",
    "2way_pixel_metrics",
    "3way_object_metrics",
    "metrics_long",
)

PURE_TEST_CORE_SAVE_KEYS = (
    "candidate_panel",
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
    "guardrails",
    "diagnostics",
    "protocol",
)


def exact_int_list(values: Sequence[Any]) -> list[int]:
    """Normalize batch identifiers to a list of Python integers."""
    return [int(value) for value in list(values)]


def build_pure_test_projection_filters(
    reference_classes: Sequence[str],
    test_batches: Sequence[int],
) -> dict[str, list[Any]]:
    """Return the canonical projection filter for held-out pure-test objects."""
    return {
        "sample_kind": ["pure"],
        "object_nut_type": list(reference_classes),
        "batch": exact_int_list(test_batches),
    }


def build_pure_test_guardrails(
    *,
    train_batches: Sequence[int],
    test_batches: Sequence[int],
    train_filters: Mapping[str, Any],
    projection_filters: Mapping[str, Any],
    thresholds_path: str | Path,
    object_db: Mapping[str, Mapping[str, Any]] | None = None,
    target_class: str = expcfg.TARGET_CLASS,
    reference_classes: Sequence[str] = expcfg.REFERENCE_CLASSES,
) -> pd.DataFrame:
    """Build a guardrail table proving that 06A uses the pure-test split only as test data."""
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "", **extra: Any) -> None:
        row = {
            "check_name": check_name,
            "passed": bool(passed),
            "status": "passed" if passed else "failed",
            "details": str(details),
        }
        row.update(extra)
        rows.append(row)

    train_batches_i = exact_int_list(train_batches)
    test_batches_i = exact_int_list(test_batches)
    thresholds_path = Path(thresholds_path)

    add("train_batches_are_1_2_3", train_batches_i == [1, 2, 3], train_batches_i)
    add("test_batches_are_4", test_batches_i == [4], test_batches_i)
    add(
        "train_test_batches_disjoint",
        set(train_batches_i).isdisjoint(set(test_batches_i)),
        {"train": train_batches_i, "test": test_batches_i},
    )
    add(
        "projection_filters_are_pure_batch_4",
        projection_filters.get("sample_kind") == ["pure"]
        and exact_int_list(projection_filters.get("batch", [])) == [4],
        projection_filters,
    )
    add("train_filters_exclude_batch_4", 4 not in set(train_batches_i), train_filters)
    add("fixed_3way_thresholds_from_04c", thresholds_path.exists(), thresholds_path)

    if object_db is not None:
        n_train_target = len(
            filter_records(
                object_db,
                return_items=False,
                sample_kind="pure",
                object_nut_type=target_class,
                batch=train_batches_i,
            )
        )
        n_test_reference = len(
            filter_records(
                object_db,
                return_items=False,
                sample_kind="pure",
                object_nut_type=list(reference_classes),
                batch=test_batches_i,
            )
        )
        add(
            "train_target_objects_available",
            n_train_target > 0,
            f"{n_train_target} target objects",
            n_records=int(n_train_target),
        )
        add(
            "pure_test_reference_objects_available",
            n_test_reference > 0,
            f"{n_test_reference} reference objects",
            n_records=int(n_test_reference),
        )

    return pd.DataFrame(rows)


def validate_pure_test_guardrails(guardrails_df: pd.DataFrame) -> pd.DataFrame:
    """Raise if any pure-test protocol guardrail failed."""
    failed = guardrails_df.loc[~guardrails_df["passed"].astype(bool)]
    if len(failed) > 0:
        raise RuntimeError(
            "Pure-test guardrail failure(s): "
            + ", ".join(failed["check_name"].astype(str).tolist())
        )
    return guardrails_df


def validate_upstream_tables_are_pre_test(*tables: pd.DataFrame) -> None:
    """Reject candidate/review/threshold tables that already contain pure-test results."""
    for table in tables:
        validate_no_pure_test_inputs(table)


def select_pure_test_candidate_panel(
    candidate_panel_df: pd.DataFrame,
    track_scoring_flags_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    *,
    max_candidates: int | None = None,
    max_candidates_per_track: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the 06A candidate panel from 05 reviewed IDs and 04C full candidate configs.

    The 05 table decides which configurations have passed review, but the full
    refit parameters are restored from the 04C candidate panel.
    """
    candidate_panel_df = normalize_simca_candidate_columns(candidate_panel_df)
    validate_no_pure_test_inputs(candidate_panel_df)
    validate_no_pure_test_inputs(track_scoring_flags_df)
    validate_no_pure_test_inputs(thresholds_df)
    validate_simca_candidate_contract(candidate_panel_df)

    reviewed_config_ids = (
        track_scoring_flags_df["selected_config_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    reviewed_order_df = pd.DataFrame(
        {
            "selected_config_id": reviewed_config_ids,
            "_review_order": np.arange(len(reviewed_config_ids), dtype=int),
        }
    )

    out = (
        candidate_panel_df.assign(
            selected_config_id=lambda df: df["selected_config_id"].astype(str)
        )
        .merge(reviewed_order_df, on="selected_config_id", how="inner", validate="one_to_one")
        .sort_values("_review_order")
        .drop(columns=["_review_order"])
        .reset_index(drop=True)
    )

    if max_candidates_per_track is not None:
        ranked = track_scoring_flags_df.copy()
        ranked["selected_config_id"] = ranked["selected_config_id"].astype(str)
        sort_cols = [
            col
            for col in [
                "selection_track",
                "review_rank_in_track",
                "review_flag_count",
                "selected_config_id",
            ]
            if col in ranked.columns
        ]
        if sort_cols:
            ranked = ranked.sort_values(
                sort_cols,
                ascending=True,
                kind="mergesort",
            )
        limited_ids = (
            ranked.groupby("selection_track", dropna=False, group_keys=False)
            .head(int(max_candidates_per_track))["selected_config_id"]
            .drop_duplicates()
        )
        out = out.loc[out["selected_config_id"].isin(limited_ids)].reset_index(drop=True)

    if max_candidates is not None:
        out = out.head(int(max_candidates)).copy()

    missing_candidate_ids = sorted(
        set(reviewed_config_ids) - set(candidate_panel_df["selected_config_id"].astype(str))
    )
    if missing_candidate_ids:
        raise RuntimeError(
            "Some 05 reviewed selected_config_id values are missing from the 04C candidate panel. "
            f"Preview: {missing_candidate_ids[:10]}"
        )

    missing_threshold_ids = sorted(
        set(out["selected_config_id"].astype(str))
        - set(thresholds_df["selected_config_id"].astype(str))
    )
    if missing_threshold_ids:
        raise RuntimeError(
            "Missing fixed 3-way validation thresholds for pure-test candidate(s). "
            f"Preview: {missing_threshold_ids[:10]}"
        )

    selected_thresholds_df = thresholds_df.loc[
        thresholds_df["selected_config_id"].astype(str).isin(
            out["selected_config_id"].astype(str)
        )
    ].copy()

    validate_simca_candidate_contract(out)
    return out.reset_index(drop=True), selected_thresholds_df.reset_index(drop=True)


def simca_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Return model/config columns useful for metric grouping."""
    cols = [
        "selected_config_id",
        "candidate_id",
        "model_candidate_id",
        "assigned_selection_track",
        "final_rank_in_track",
        "pareto_tier",
        "pareto_rank_in_track",
        "matrix_family",
        "matrix_method",
        "training_matrix_id",
        "preprocessing",
        "rule_variant",
        "selected_rule_name",
        "rule_for_refit",
        "n_components",
        "alpha",
        "object_threshold",
        "m_effective",
        "balanced_pixel_strategy_effective",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
    ]
    return [col for col in cols if col in df.columns]


def three_way_extra_group_cols(df: pd.DataFrame) -> list[str]:
    """Return 3-way grouping columns, excluding the config id added by the evaluator."""
    return [col for col in simca_metadata_columns(df) if col != "selected_config_id"]


def finalize_simca_metric_table(
    df: pd.DataFrame,
    *,
    decision_mode: str,
    metric_level: str,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
) -> pd.DataFrame:
    """Attach standard SIMCA metric metadata and validate the evaluation contract."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out["decision_mode"] = decision_mode
    out["evaluation_stage"] = evaluation_stage
    out["evaluation_split"] = evaluation_stage
    out["metric_level"] = metric_level
    out = add_selection_track(out)
    out = materialize_selection_metrics(
        out,
        keep_source_columns=False,
    )
    out = out.drop(
        columns=expcfg.ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS,
        errors="ignore",
    )
    validate_simca_evaluation_contract(out)
    return out


def build_2way_object_metrics(
    refit_metrics_df: pd.DataFrame,
    *,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
) -> pd.DataFrame:
    """Build standardized object-level 2-way metrics from SIMCA refit rows."""
    metrics_df = materialize_selection_metrics(refit_metrics_df)
    return finalize_simca_metric_table(
        metrics_df,
        decision_mode="2way",
        metric_level="object",
        evaluation_stage=evaluation_stage,
    )


def build_2way_pixel_metrics(
    pixel_df: pd.DataFrame,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
) -> pd.DataFrame:
    """Build standardized pixel-level 2-way metrics grouped by model configuration."""
    if pixel_df is None or len(pixel_df) == 0:
        return pd.DataFrame()
    metrics_df = summarize_pixel_errors_by_image(
        pixel_df,
        target_class=target_class,
        non_target_label=non_target_label,
        group_cols=simca_metadata_columns(pixel_df),
        sort_worst_first=False,
    )
    return finalize_simca_metric_table(
        metrics_df,
        decision_mode="2way",
        metric_level="pixel",
        evaluation_stage=evaluation_stage,
    )


def build_2way_object_image_diagnostics(
    object_df: pd.DataFrame,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
) -> pd.DataFrame:
    """Build object-level 2-way diagnostics by source image."""
    if object_df is None or len(object_df) == 0:
        return pd.DataFrame()
    out = summarize_object_errors_by_image(
        object_df,
        target_class=target_class,
        non_target_label=non_target_label,
        group_cols=simca_metadata_columns(object_df) + ["source_image"],
        sort_worst_first=False,
    )
    return finalize_simca_metric_table(
        out,
        decision_mode="2way",
        metric_level="object_image",
        evaluation_stage=evaluation_stage,
    )


def build_2way_pixel_image_diagnostics(
    pixel_df: pd.DataFrame,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
) -> pd.DataFrame:
    """Build pixel-level 2-way diagnostics by source image."""
    if pixel_df is None or len(pixel_df) == 0:
        return pd.DataFrame()
    out = summarize_pixel_errors_by_image(
        pixel_df,
        target_class=target_class,
        non_target_label=non_target_label,
        group_cols=simca_metadata_columns(pixel_df) + ["source_image"],
        sort_worst_first=False,
    )
    return finalize_simca_metric_table(
        out,
        decision_mode="2way",
        metric_level="pixel_image",
        evaluation_stage=evaluation_stage,
    )


def build_3way_outputs(
    object_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply fixed thresholds and build 3-way object metrics plus image diagnostics."""
    if object_df is None or len(object_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    threshold_cols = [
        "selected_config_id",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    ]

    metrics_df, objects_3way_df = evaluate_three_way_by_config(
        object_df=object_df,
        thresholds_df=thresholds_df,
        config_id_col="selected_config_id",
        extra_group_cols=three_way_extra_group_cols(object_df),
        target_class=target_class,
        non_target_label=non_target_label,
    )

    if len(objects_3way_df) > 0:
        objects_3way_df = add_three_way_confidence(
            objects_3way_df,
            target_class=target_class,
            non_target_label=non_target_label,
        )

    if len(metrics_df) > 0:
        metrics_df = metrics_df.merge(
            thresholds_df[threshold_cols].drop_duplicates("selected_config_id"),
            on="selected_config_id",
            how="left",
            validate="many_to_one",
        )
        metrics_df["fn_rate"] = metrics_df["target_miss_rate"]
        metrics_df["fp_rate"] = metrics_df["non_target_false_accept_rate"]
        metrics_df["balanced_accuracy"] = metrics_df["decided_balanced_accuracy"]
        metrics_df = finalize_simca_metric_table(
            metrics_df,
            decision_mode="3way",
            metric_level="object",
            evaluation_stage=evaluation_stage,
        )

    image_metrics_df = pd.DataFrame()
    if len(objects_3way_df) > 0:
        image_metrics_df, _ = evaluate_three_way_by_config(
            object_df=objects_3way_df,
            thresholds_df=thresholds_df,
            config_id_col="selected_config_id",
            extra_group_cols=three_way_extra_group_cols(objects_3way_df) + ["source_image"],
            target_class=target_class,
            non_target_label=non_target_label,
        )
        if len(image_metrics_df) > 0:
            image_metrics_df = image_metrics_df.merge(
                thresholds_df[threshold_cols].drop_duplicates("selected_config_id"),
                on="selected_config_id",
                how="left",
                validate="many_to_one",
            )
            image_metrics_df["fn_rate"] = image_metrics_df["target_miss_rate"]
            image_metrics_df["fp_rate"] = image_metrics_df["non_target_false_accept_rate"]
            image_metrics_df["balanced_accuracy"] = image_metrics_df["decided_balanced_accuracy"]
            image_metrics_df = finalize_simca_metric_table(
                image_metrics_df,
                decision_mode="3way",
                metric_level="object_image",
                evaluation_stage=evaluation_stage,
            )

    return metrics_df, image_metrics_df, objects_3way_df


def _save_batch_table(
    df: pd.DataFrame,
    directory: str | Path | None,
    batch_id: str,
    stem: str,
) -> Path | None:
    if directory is None or df is None or len(df) == 0:
        return None
    return write_simca_table(df, Path(directory) / f"{batch_id}_{stem}.parquet")


def run_pure_test_refit_batches(
    *,
    selected_configs_df: pd.DataFrame,
    object_db: Mapping[str, Mapping[str, Any]],
    image_db: Mapping[str, Mapping[str, Any]],
    train_filters: Mapping[str, Any],
    projection_filters: Mapping[str, Any],
    preprocessing_configs: Mapping[str, Mapping[str, Sequence[str]]],
    thresholds_df: pd.DataFrame,
    evaluation_stage: str = DEFAULT_PURE_TEST_EVALUATION_STAGE,
    wavelengths: Sequence[float] | None = None,
    random_state: int = expcfg.RANDOM_STATE,
    replace: bool = expcfg.REPLACE_BALANCED_PIXELS,
    cv_n_splits: int | None = expcfg.CV_N_SPLITS,
    cv_group_col: str = expcfg.CV_GROUP_COL,
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    batch_size: int = 50,
    batch_metrics_dir: str | Path | None = None,
    batch_objects_dir: str | Path | None = None,
    batch_pixels_dir: str | Path | None = None,
    batch_3way_objects_dir: str | Path | None = None,
    save_batch_metric_tables: bool = True,
    save_batch_object_tables: bool = False,
    save_batch_pixel_tables: bool = False,
    save_batch_3way_object_tables: bool = False,
    save_combined_object_tables: bool = True,
    save_combined_pixel_tables: bool = False,
    save_combined_3way_object_tables: bool = True,
    fixed_thresholds_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Refit selected configs on pure train batches and project held-out pure-test objects."""
    metric_parts = []
    object_metric_parts = []
    pixel_metric_parts = []
    three_way_metric_parts = []
    object_image_diag_parts = []
    pixel_image_diag_parts = []
    three_way_object_image_diag_parts = []
    pixel_error_parts = []
    error_parts = []
    object_parts = []
    pixel_parts = []
    three_way_object_parts = []
    batch_manifest_rows = []

    for batch_id, start, stop, batch_df in iter_dataframe_batches(
        selected_configs_df,
        batch_size,
    ):
        print(f"[{evaluation_stage}] {batch_id}: candidates {start + 1}-{stop} / {len(selected_configs_df)}")
        batch_paths: dict[str, Path | None] = {}

        try:
            (
                batch_refit_metrics_df,
                batch_objects_df,
                batch_pixels_df,
                batch_pixel_errors_df,
                batch_errors_df,
            ) = refit_selected_simca_configs(
                selected_configs_df=batch_df,
                object_db=object_db,
                image_db=image_db,
                train_filters=dict(train_filters),
                projection_filters=dict(projection_filters),
                preprocessing_configs=preprocessing_configs,
                evaluation_split=evaluation_stage,
                wavelengths=wavelengths,
                random_state=random_state,
                replace=replace,
                cv_n_splits=cv_n_splits,
                cv_group_col=cv_group_col,
                target_class=target_class,
                non_target_label=non_target_label,
            )
        except Exception as exc:
            batch_refit_metrics_df = pd.DataFrame()
            batch_objects_df = pd.DataFrame()
            batch_pixels_df = pd.DataFrame()
            batch_pixel_errors_df = pd.DataFrame()
            batch_errors_df = batch_df.copy()
            batch_errors_df["evaluation_split"] = evaluation_stage
            batch_errors_df["error"] = repr(exc)
            print("  -> BATCH ERROR:", repr(exc))

        batch_2way_object_metrics_df = (
            build_2way_object_metrics(
                batch_refit_metrics_df,
                evaluation_stage=evaluation_stage,
            )
            if len(batch_refit_metrics_df)
            else pd.DataFrame()
        )
        batch_2way_pixel_metrics_df = (
            build_2way_pixel_metrics(
                batch_pixels_df,
                target_class=target_class,
                non_target_label=non_target_label,
                evaluation_stage=evaluation_stage,
            )
            if len(batch_pixels_df)
            else pd.DataFrame()
        )
        batch_object_image_diag_df = (
            build_2way_object_image_diagnostics(
                batch_objects_df,
                target_class=target_class,
                non_target_label=non_target_label,
                evaluation_stage=evaluation_stage,
            )
            if len(batch_objects_df)
            else pd.DataFrame()
        )
        batch_pixel_image_diag_df = (
            build_2way_pixel_image_diagnostics(
                batch_pixels_df,
                target_class=target_class,
                non_target_label=non_target_label,
                evaluation_stage=evaluation_stage,
            )
            if len(batch_pixels_df)
            else pd.DataFrame()
        )
        (
            batch_3way_object_metrics_df,
            batch_3way_object_image_diag_df,
            batch_3way_objects_df,
        ) = build_3way_outputs(
            batch_objects_df,
            thresholds_df,
            target_class=target_class,
            non_target_label=non_target_label,
            evaluation_stage=evaluation_stage,
        )

        if len(batch_refit_metrics_df):
            metric_parts.append(batch_refit_metrics_df)
        if len(batch_2way_object_metrics_df):
            object_metric_parts.append(batch_2way_object_metrics_df)
        if len(batch_2way_pixel_metrics_df):
            pixel_metric_parts.append(batch_2way_pixel_metrics_df)
        if len(batch_3way_object_metrics_df):
            three_way_metric_parts.append(batch_3way_object_metrics_df)
        if len(batch_object_image_diag_df):
            object_image_diag_parts.append(batch_object_image_diag_df)
        if len(batch_pixel_image_diag_df):
            pixel_image_diag_parts.append(batch_pixel_image_diag_df)
        if len(batch_3way_object_image_diag_df):
            three_way_object_image_diag_parts.append(batch_3way_object_image_diag_df)
        if len(batch_pixel_errors_df):
            pixel_error_parts.append(batch_pixel_errors_df)
        if len(batch_errors_df):
            error_parts.append(batch_errors_df)

        if save_combined_object_tables and len(batch_objects_df):
            object_parts.append(batch_objects_df.copy())
        if save_combined_pixel_tables and len(batch_pixels_df):
            pixel_parts.append(batch_pixels_df.copy())
        if save_combined_3way_object_tables and len(batch_3way_objects_df):
            three_way_object_parts.append(batch_3way_objects_df.copy())

        if save_batch_metric_tables:
            batch_paths["2way_object_metrics_path"] = _save_batch_table(
                batch_2way_object_metrics_df,
                batch_metrics_dir,
                batch_id,
                "2way_object_metrics",
            )
            batch_paths["2way_pixel_metrics_path"] = _save_batch_table(
                batch_2way_pixel_metrics_df,
                batch_metrics_dir,
                batch_id,
                "2way_pixel_metrics",
            )
            batch_paths["3way_object_metrics_path"] = _save_batch_table(
                batch_3way_object_metrics_df,
                batch_metrics_dir,
                batch_id,
                "3way_object_metrics",
            )
        if save_batch_object_tables:
            batch_paths["objects_path"] = _save_batch_table(
                batch_objects_df,
                batch_objects_dir,
                batch_id,
                "objects",
            )
        if save_batch_pixel_tables:
            batch_paths["pixels_path"] = _save_batch_table(
                batch_pixels_df,
                batch_pixels_dir,
                batch_id,
                "pixels",
            )
        if save_batch_3way_object_tables:
            batch_paths["objects_3way_path"] = _save_batch_table(
                batch_3way_objects_df,
                batch_3way_objects_dir,
                batch_id,
                "objects_3way",
            )

        manifest_row = {
            "batch_id": batch_id,
            "row_start": int(start),
            "row_stop": int(stop),
            "n_candidates": int(len(batch_df)),
            "n_refit_metric_rows": int(len(batch_refit_metrics_df)),
            "n_2way_object_metric_rows": int(len(batch_2way_object_metrics_df)),
            "n_2way_pixel_metric_rows": int(len(batch_2way_pixel_metrics_df)),
            "n_3way_threshold_grid_rows": 0,
            "n_3way_selected_threshold_rows": 0,
            "n_3way_object_metric_rows": int(len(batch_3way_object_metrics_df)),
            "n_object_rows": int(len(batch_objects_df)),
            "n_pixel_rows": int(len(batch_pixels_df)),
            "n_3way_object_rows": int(len(batch_3way_objects_df)),
            "n_pixel_error_rows": int(len(batch_pixel_errors_df)),
            "n_error_rows": int(len(batch_errors_df)),
            "3way_threshold_grid_path": None,
            "3way_selected_thresholds_path": str(fixed_thresholds_path) if fixed_thresholds_path is not None else None,
        }
        manifest_row.update(
            {
                key: str(value) if value is not None else None
                for key, value in batch_paths.items()
            }
        )
        batch_manifest_rows.append(manifest_row)

        del batch_refit_metrics_df
        del batch_objects_df
        del batch_pixels_df
        del batch_pixel_errors_df
        del batch_errors_df
        del batch_2way_object_metrics_df
        del batch_2way_pixel_metrics_df
        del batch_3way_object_metrics_df
        del batch_object_image_diag_df
        del batch_pixel_image_diag_df
        del batch_3way_object_image_diag_df
        del batch_3way_objects_df
        gc.collect()

    return {
        "refit_metrics": concat_nonempty_tables(metric_parts),
        "2way_object_metrics": concat_nonempty_tables(object_metric_parts),
        "2way_pixel_metrics": concat_nonempty_tables(pixel_metric_parts),
        "3way_object_metrics": concat_nonempty_tables(three_way_metric_parts),
        "object_image_diagnostics": concat_nonempty_tables(object_image_diag_parts),
        "pixel_image_diagnostics": concat_nonempty_tables(pixel_image_diag_parts),
        "3way_object_image_diagnostics": concat_nonempty_tables(three_way_object_image_diag_parts),
        "pixel_errors_by_image": concat_nonempty_tables(pixel_error_parts),
        "errors": concat_nonempty_tables(error_parts),
        "objects": concat_nonempty_tables(object_parts),
        "pixels": concat_nonempty_tables(pixel_parts),
        "3way_objects": concat_nonempty_tables(three_way_object_parts),
        "batch_manifest": pd.DataFrame(batch_manifest_rows),
    }


def missing_existing_pure_test_paths(paths: Mapping[str, str | Path]) -> list[Path]:
    """Return missing required pure-test metric outputs for reload mode."""
    return [
        Path(paths[key])
        for key in PURE_TEST_REQUIRED_OUTPUT_KEYS
        if key not in paths or not Path(paths[key]).exists()
    ]


def load_existing_pure_test_outputs(paths: Mapping[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Load already generated 06A output tables from parquet files."""
    missing = missing_existing_pure_test_paths(paths)
    if missing:
        raise FileNotFoundError(
            "Required pure-test output file(s) are missing:\n"
            + "\n".join(map(str, missing))
        )

    outputs = {
        key: read_simca_table(path)
        for key, path in paths.items()
        if Path(path).exists()
    }
    return outputs


def validate_pure_test_outputs(
    outputs: Mapping[str, pd.DataFrame],
    *,
    expected_tracks: Sequence[str] = expcfg.SIMCA_SELECTION_TRACKS,
    allow_track_subset: bool = False,
) -> dict[str, pd.DataFrame]:
    """Validate required 06A metric outputs and materialize `metrics_long`."""
    out = dict(outputs)

    required_metric_keys = (
        "2way_object_metrics",
        "2way_pixel_metrics",
        "3way_object_metrics",
    )
    empty_required = [
        key
        for key in required_metric_keys
        if key not in out or out[key] is None or len(out[key]) == 0
    ]
    if empty_required:
        raise RuntimeError(
            "Pure-test outputs are incomplete. Empty required table(s): "
            + ", ".join(empty_required)
        )

    out.setdefault(
        "errors",
        pd.DataFrame(columns=["selected_config_id", "evaluation_split", "error"]),
    )
    out.setdefault(
        "pixel_errors_by_image",
        pd.DataFrame(columns=["selected_config_id", "evaluation_split", "source_image"]),
    )
    out.setdefault("object_image_diagnostics", pd.DataFrame())
    out.setdefault("pixel_image_diagnostics", pd.DataFrame())
    out.setdefault("3way_object_image_diagnostics", pd.DataFrame())
    out.setdefault("objects", pd.DataFrame())
    out.setdefault("pixels", pd.DataFrame())
    out.setdefault("3way_objects", pd.DataFrame())
    out.setdefault("batch_manifest", pd.DataFrame())

    out["metrics_long"] = pd.concat(
        [
            out["2way_object_metrics"],
            out["2way_pixel_metrics"],
            out["3way_object_metrics"],
        ],
        ignore_index=True,
        sort=False,
    )

    expected = set(map(str, expected_tracks))
    observed = set(out["metrics_long"]["selection_track"].dropna().astype(str))
    missing_tracks = sorted(expected - observed)
    if missing_tracks and not allow_track_subset:
        raise RuntimeError(f"Missing expected SIMCA pure-test selection tracks: {missing_tracks}")

    for key in ("2way_object_metrics", "2way_pixel_metrics", "3way_object_metrics", "metrics_long"):
        validate_simca_evaluation_contract(out[key])

    return out


def summarize_pure_test_outputs(outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact descriptive summary of pure-test metrics."""
    metrics_long_df = outputs.get("metrics_long", pd.DataFrame())
    if metrics_long_df is None or len(metrics_long_df) == 0:
        return pd.DataFrame()
    return (
        metrics_long_df.groupby(
            ["selection_track", "matrix_family", "decision_mode", "metric_level"],
            dropna=False,
        )
        .agg(
            n_rows=("candidate_id", "size"),
            n_candidates=("candidate_id", "nunique"),
            best_fn_rate=("fn_rate", "min"),
            best_fp_rate=("fp_rate", "min"),
            best_balanced_accuracy=("balanced_accuracy", "max"),
            median_fn_rate=("fn_rate", "median"),
            median_fp_rate=("fp_rate", "median"),
            median_balanced_accuracy=("balanced_accuracy", "median"),
        )
        .reset_index()
        .sort_values(["selection_track", "metric_level"])
        .reset_index(drop=True)
    )


def build_pure_test_protocol(
    settings: Mapping[str, Any],
    outputs: Mapping[str, pd.DataFrame],
    *,
    final_selection_performed: bool = False,
) -> pd.DataFrame:
    """Build the 06A protocol table from settings and output row counts."""
    row = dict(settings)
    for key, value in list(row.items()):
        if isinstance(value, (dict, list, tuple, set)):
            row[key] = json.dumps(value, default=str)
        elif isinstance(value, Path):
            row[key] = str(value)

    row.update(
        {
            "n_candidate_panel": int(len(outputs.get("candidate_panel", pd.DataFrame()))),
            "n_2way_object_metrics": int(len(outputs.get("2way_object_metrics", pd.DataFrame()))),
            "n_2way_pixel_metrics": int(len(outputs.get("2way_pixel_metrics", pd.DataFrame()))),
            "n_3way_object_metrics": int(len(outputs.get("3way_object_metrics", pd.DataFrame()))),
            "n_metrics_long": int(len(outputs.get("metrics_long", pd.DataFrame()))),
            "n_errors": int(len(outputs.get("errors", pd.DataFrame()))),
            "final_selection_performed": bool(final_selection_performed),
        }
    )
    return pd.DataFrame([row])


def save_pure_test_outputs(
    outputs: Mapping[str, pd.DataFrame],
    paths: Mapping[str, str | Path],
    *,
    save_combined_object_tables: bool = True,
    save_combined_pixel_tables: bool = False,
    save_combined_3way_object_tables: bool = True,
) -> list[Path]:
    """Save standard 06A outputs and optional heavy projection tables."""
    saved_paths: list[Path] = []

    for key in PURE_TEST_CORE_SAVE_KEYS:
        if key in outputs and key in paths:
            saved_paths.append(write_simca_table(outputs[key], paths[key]))

    optional_keys = []
    if save_combined_object_tables:
        optional_keys.append("objects")
    if save_combined_pixel_tables:
        optional_keys.append("pixels")
    if save_combined_3way_object_tables:
        optional_keys.append("3way_objects")

    for key in optional_keys:
        table = outputs.get(key, pd.DataFrame())
        if key in paths and table is not None and len(table) > 0:
            saved_paths.append(write_simca_table(table, paths[key]))

    return saved_paths
