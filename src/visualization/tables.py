from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def build_database_inventory_table(
    image_db: Mapping,
    object_db: Mapping,
) -> pd.DataFrame:
    """Summarise images, objects and object pixels by class, batch and sample kind."""
    rows = []
    for image_key, image in image_db.items():
        rows.append(
            {
                "source_image": str(image_key),
                "nut_type": image.get("nut_type", "unknown"),
                "batch": image.get("batch", "unknown"),
                "sample_kind": image.get("sample_kind", "unknown"),
                "is_pure": bool(image.get("is_pure", False)),
                "is_mixture": bool(image.get("is_mixture", False)),
                "n_image_pixels": int(np.prod(np.asarray(image.get("image_ref", image.get("labels"))).shape))
                if image.get("image_ref", image.get("labels")) is not None
                else np.nan,
            }
        )
    images = pd.DataFrame(rows)

    object_rows = []
    for object_id, obj in object_db.items():
        object_rows.append(
            {
                "object_id": str(object_id),
                "source_image": str(obj.get("source_clean_key", obj.get("source_image", "unknown"))),
                "nut_type": obj.get("object_nut_type", obj.get("label", "unknown")),
                "batch": obj.get("batch", "unknown"),
                "sample_kind": obj.get("sample_kind", "unknown"),
                "area_pixels": pd.to_numeric(obj.get("area_pixels", np.nan), errors="coerce"),
            }
        )
    objects = pd.DataFrame(object_rows)
    if objects.empty:
        return pd.DataFrame()

    object_summary = (
        objects.groupby(["nut_type", "batch", "sample_kind"], dropna=False)
        .agg(
            n_objects=("object_id", "nunique"),
            n_object_pixels=("area_pixels", "sum"),
            median_object_area=("area_pixels", "median"),
            mean_object_area=("area_pixels", "mean"),
            n_source_images=("source_image", "nunique"),
        )
        .reset_index()
    )
    if not images.empty:
        image_summary = (
            images.groupby(["nut_type", "batch", "sample_kind"], dropna=False)
            .agg(n_images=("source_image", "nunique"))
            .reset_index()
        )
        object_summary = object_summary.merge(
            image_summary,
            on=["nut_type", "batch", "sample_kind"],
            how="outer",
        )
    return object_summary.sort_values(["sample_kind", "nut_type", "batch"]).reset_index(drop=True)


def build_preprocessing_shortlist_table(
    pca_summary_df: pd.DataFrame,
    matrix_col: str = "matrix_method",
    preprocessing_col: str = "preprocessing",
    ranking_metric: str = "class_over_batch_ratio",
    top_n: int = 3,
    extra_cols: Sequence[str] = (
        "class_trace_ratio",
        "batch_trace_ratio",
        "ncomp_90",
        "ncomp_95",
    ),
) -> pd.DataFrame:
    """Return the top preprocessing methods within each matrix representation."""
    required = [matrix_col, preprocessing_col, ranking_metric]
    missing = [column for column in required if column not in pca_summary_df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    columns = required + [column for column in extra_cols if column in pca_summary_df.columns]
    return (
        pca_summary_df[columns]
        .sort_values([matrix_col, ranking_metric], ascending=[True, False])
        .groupby(matrix_col, group_keys=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def build_candidate_model_table(
    df: pd.DataFrame,
    id_col: str = "selected_config_id",
    include_cols: Sequence[str] = (
        "matrix_family",
        "matrix_method",
        "preprocessing",
        "rule",
        "n_components",
        "alpha",
        "object_threshold",
        "fn_rate",
        "fp_rate",
        "balanced_accuracy",
        "uncertain_rate",
        "coverage_rate",
    ),
    sort_cols: Sequence[str] = ("fn_rate", "fp_rate", "balanced_accuracy"),
    ascending: Sequence[bool] = (True, True, False),
) -> pd.DataFrame:
    """Build a compact candidate-model table for reports and presentations."""
    columns = [column for column in [id_col, *include_cols] if column in df.columns]
    out = df[columns].drop_duplicates(id_col if id_col in columns else None).copy()
    existing_sort = [column for column in sort_cols if column in out.columns]
    if existing_sort:
        direction = list(ascending)[: len(existing_sort)]
        out = out.sort_values(existing_sort, ascending=direction)
    return out.reset_index(drop=True)


def build_frozen_reference_table(
    df: pd.DataFrame,
    id_col: str = "selected_config_id",
) -> pd.DataFrame:
    """Compact validation/test table for the frozen reference panel."""
    preferred = [
        id_col,
        "matrix_family",
        "matrix_method",
        "preprocessing",
        "rule",
        "n_components",
        "alpha",
        "object_threshold",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
        "validation_fn_rate",
        "validation_fp_rate",
        "validation_balanced_accuracy",
        "pure_test_fn_rate",
        "pure_test_fp_rate",
        "pure_test_balanced_accuracy",
        "uncertain_rate",
        "coverage_rate",
        "final_rank",
    ]
    columns = [column for column in preferred if column in df.columns]
    out = df[columns].drop_duplicates(id_col if id_col in columns else None)
    sort = [column for column in ("final_rank", "pure_test_fn_rate", "pure_test_fp_rate") if column in out.columns]
    if sort:
        out = out.sort_values(sort, ascending=[True] * len(sort))
    return out.reset_index(drop=True)


def build_per_image_error_table(
    df: pd.DataFrame,
    image_col: str = "source_image",
    config_col: str | None = "selected_config_id",
) -> pd.DataFrame:
    """Select and order the most useful per-image error columns."""
    preferred = [
        config_col,
        image_col,
        "n",
        "n_truth_objects",
        "n_truth_pixels",
        "tp",
        "tn",
        "fp",
        "fn",
        "fn_rate",
        "fp_rate",
        "balanced_accuracy",
        "accuracy",
        "uncertain_rate",
        "coverage_rate",
    ]
    columns = [column for column in preferred if column and column in df.columns]
    out = df[columns].copy()
    sort_cols = [column for column in (config_col, "fn_rate", "fp_rate") if column and column in out.columns]
    if sort_cols:
        ascending = [True] + [False] * (len(sort_cols) - 1) if config_col in sort_cols else [False] * len(sort_cols)
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.reset_index(drop=True)


def build_presentation_summary_table(
    df: pd.DataFrame,
    rows: int = 10,
    id_col: str = "selected_config_id",
    priority_cols: Sequence[str] = (
        "pure_test_fn_rate",
        "pure_test_fp_rate",
        "pure_test_balanced_accuracy",
    ),
) -> pd.DataFrame:
    """Return a short, presentation-ready top-model table."""
    out = build_frozen_reference_table(df, id_col=id_col)
    sort_cols = [column for column in priority_cols if column in out.columns]
    if sort_cols:
        ascending = [True, True, False][: len(sort_cols)]
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.head(int(rows)).reset_index(drop=True)
