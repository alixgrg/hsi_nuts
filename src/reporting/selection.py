from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def choose_diagnostic_configs(
    df: pd.DataFrame,
    config_col: str = "selected_config_id",
    fn_col: str = "fn_rate",
    fp_col: str = "fp_rate",
    score_col: str | None = "balanced_accuracy",
    family_col: str | None = "matrix_family",
    n_total: int = 6,
    max_per_family: int | None = None,
) -> pd.DataFrame:
    """Select a compact, diverse set of configurations for diagnostics.

    Priority is low FN rate, then low FP rate, then high score. Optional family
    caps prevent all selected models from coming from one representation.
    """
    required = [config_col, fn_col, fp_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = df.copy().drop_duplicates(config_col)
    sort_cols = [fn_col, fp_col]
    ascending = [True, True]
    if score_col is not None and score_col in d.columns:
        sort_cols.append(score_col)
        ascending.append(False)
    d = d.sort_values(sort_cols, ascending=ascending)

    if family_col is None or family_col not in d.columns or max_per_family is None:
        return d.head(int(n_total)).reset_index(drop=True)

    selected = []
    counts: dict[str, int] = {}
    for _, row in d.iterrows():
        family = str(row[family_col])
        if counts.get(family, 0) >= int(max_per_family):
            continue
        selected.append(row)
        counts[family] = counts.get(family, 0) + 1
        if len(selected) >= int(n_total):
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def choose_images_for_config(
    image_metrics_df: pd.DataFrame,
    config_id,
    config_col: str = "selected_config_id",
    image_col: str = "source_image",
    fn_col: str = "fn_rate",
    fp_col: str = "fp_rate",
    target_count_col: str = "n_true_target_objects",
    n_images: int = 3,
    max_single_target_images: int = 1,
) -> pd.DataFrame:
    """Select best, difficult and representative images for one configuration.

    At most ``max_single_target_images`` images with exactly one target object are
    retained, matching the reporting constraint used for the mixture analysis.
    """
    required = [config_col, image_col, fn_col, fp_col]
    missing = [column for column in required if column not in image_metrics_df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = image_metrics_df[image_metrics_df[config_col].astype(str).eq(str(config_id))].copy()
    if d.empty:
        return d
    d[fn_col] = pd.to_numeric(d[fn_col], errors="coerce")
    d[fp_col] = pd.to_numeric(d[fp_col], errors="coerce")
    d["_difficulty"] = d[fn_col].fillna(1.0) * 3.0 + d[fp_col].fillna(1.0)

    candidates = []
    # Hardest, best, then closest to median difficulty.
    candidates.append(d.sort_values([fn_col, fp_col], ascending=[False, False]).iloc[0])
    candidates.append(d.sort_values([fn_col, fp_col], ascending=[True, True]).iloc[0])
    median = d["_difficulty"].median()
    candidates.extend(
        row for _, row in d.assign(_median_distance=(d["_difficulty"] - median).abs())
        .sort_values("_median_distance")
        .iterrows()
    )

    selected = []
    seen = set()
    n_single = 0
    for row in candidates:
        image = str(row[image_col])
        if image in seen:
            continue
        single_target = (
            target_count_col in row.index
            and pd.notna(row[target_count_col])
            and int(row[target_count_col]) == 1
        )
        if single_target and n_single >= int(max_single_target_images):
            continue
        selected.append(row)
        seen.add(image)
        if single_target:
            n_single += 1
        if len(selected) >= int(n_images):
            break

    out = pd.DataFrame(selected).drop(columns=["_difficulty", "_median_distance"], errors="ignore")
    return out.reset_index(drop=True)


def choose_images_for_config_2way(*args, **kwargs) -> pd.DataFrame:
    """Compatibility alias for binary image selection."""
    return choose_images_for_config(*args, **kwargs)


def choose_images_for_config_3way(
    image_metrics_df: pd.DataFrame,
    config_id,
    uncertain_col: str = "uncertain_rate",
    miss_col: str = "target_miss_rate",
    false_accept_col: str = "non_target_false_accept_rate",
    **kwargs,
) -> pd.DataFrame:
    """Select 3-way images using miss rate first, then uncertainty/false accepts."""
    d = image_metrics_df.copy()
    if miss_col in d.columns:
        d["_three_way_fn"] = pd.to_numeric(d[miss_col], errors="coerce")
    else:
        d["_three_way_fn"] = 0.0
    components = []
    if false_accept_col in d.columns:
        components.append(pd.to_numeric(d[false_accept_col], errors="coerce").fillna(1.0))
    if uncertain_col in d.columns:
        components.append(pd.to_numeric(d[uncertain_col], errors="coerce").fillna(1.0))
    d["_three_way_fp"] = sum(components) if components else 0.0
    return choose_images_for_config(
        d,
        config_id=config_id,
        fn_col="_three_way_fn",
        fp_col="_three_way_fp",
        **kwargs,
    ).drop(columns=["_three_way_fn", "_three_way_fp"], errors="ignore")


def sample_for_qt2_plot(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("decision_3way",),
    n_per_group: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Balanced sampling for large Q/T² pixel tables."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()
    valid_groups = [column for column in group_cols if column in df.columns]
    if not valid_groups:
        return df.sample(
            n=min(len(df), int(n_per_group)),
            random_state=random_state,
        ).reset_index(drop=True)
    parts = []
    for _, group in df.groupby(valid_groups, dropna=False, sort=False):
        parts.append(
            group.sample(
                n=min(len(group), int(n_per_group)),
                random_state=random_state,
            )
        )
    return pd.concat(parts, ignore_index=True)


# Notebook compatibility name.
_sample_for_qt2_plot = sample_for_qt2_plot
