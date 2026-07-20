from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


OBJECT_MATRIX_METHODS = {"object_mean", "object_median"}
PIXEL_MATRIX_METHODS = {"balanced_pixels", "all_pixels", "pixel"}


@dataclass(frozen=True)
class PCASelectionProfile:
    """Scoring weights and QC metrics for one PCA matrix family."""

    positive_weights: Mapping[str, float]
    negative_weights: Mapping[str, float]
    separation_metric: str
    batch_metric: str
    projection_metric: str = "projection_q_deviation"
    validation_metric: str = "mean_train_projection_shift_norm"


@dataclass(frozen=True)
class PCASelectionConfig:
    """Configuration for PCA preprocessing ranking and strict shortlisting."""

    profiles: Mapping[str, PCASelectionProfile] = field(
        default_factory=lambda: {
            "object_matrix": PCASelectionProfile(
                positive_weights={
                    "class_trace_ratio": 3.0,
                    "mahalanobis_pc1_pc2_pc3": 1.0,
                },
                negative_weights={
                    "batch_trace_ratio": 2.0,
                    "mean_train_projection_shift_norm": 1.5,
                    "projection_q_deviation": 1.5,
                    "ncomp_95": 0.3,
                },
                separation_metric="class_trace_ratio",
                batch_metric="batch_trace_ratio",
            ),
            "pixel_matrix": PCASelectionProfile(
                positive_weights={
                    "object_class_trace_ratio": 3.0,
                    "object_over_intra_ratio": 1.0,
                },
                negative_weights={
                    "object_batch_trace_ratio": 2.0,
                    "mean_intra_object_trace": 1.0,
                    "mean_train_projection_shift_norm": 1.2,
                    "projection_q_deviation": 1.2,
                    "ncomp_95": 0.3,
                },
                separation_metric="object_class_trace_ratio",
                batch_metric="object_batch_trace_ratio",
            ),
        }
    )
    score_col: str = "selection_score"
    matrix_method_col: str = "matrix_method"
    family_col: str = "matrix_family"
    variant_col: str = "matrix_variant"
    preprocessing_col: str = "preprocessing"
    preprocessing_steps_col: str = "preprocessing_steps"
    group_cols: tuple[str, ...] = ("matrix_variant",)
    max_preprocessings_per_family: int = 5
    expected_families: tuple[str, ...] = ("object_matrix", "pixel_matrix")
    robust: bool = True
    clip_quantiles: tuple[float, float] = (0.05, 0.95)
    quality_lower_quantile: float = 0.25
    quality_upper_quantile: float = 0.75
    validation_upper_quantile: float = 0.75
    stability_bootstrap_iterations: int = 100
    stability_penalty_weight: float = 0.25
    random_state: int = 42
    eps: float = 1e-12


DEFAULT_PCA_SELECTION_CONFIG = PCASelectionConfig()


def make_pca_selection_config(**overrides) -> PCASelectionConfig:
    """Return the default PCA selection configuration with explicit overrides."""
    return replace(DEFAULT_PCA_SELECTION_CONFIG, **overrides)


def pca_matrix_family_from_method(matrix_method: str) -> str:
    """Return the matrix family used by the PCA selection score."""
    matrix_method = str(matrix_method)
    if matrix_method in OBJECT_MATRIX_METHODS:
        return "object_matrix"
    if matrix_method in PIXEL_MATRIX_METHODS:
        return "pixel_matrix"
    return "unknown_matrix_family"


def pca_matrix_variant_from_method(
    matrix_method: str,
    balanced_pixel_strategy: str = "random",
) -> str:
    """Return a stable matrix variant label for PCA selection."""
    matrix_method = str(matrix_method)
    if matrix_method == "balanced_pixels":
        return f"balanced_pixels_{balanced_pixel_strategy}"
    return matrix_method


def _as_group_cols(group_cols) -> tuple[str, ...]:
    if group_cols is None:
        return ()
    if isinstance(group_cols, str):
        return (group_cols,)
    return tuple(group_cols)


def _groupby_indices(df: pd.DataFrame, group_cols: tuple[str, ...]):
    by = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    return df.groupby(by, dropna=False, sort=False).groups.items()


def _ensure_selection_metadata(
    df: pd.DataFrame,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    out = df.copy()
    matrix_col = config.matrix_method_col
    if config.family_col not in out.columns and matrix_col in out.columns:
        out[config.family_col] = out[matrix_col].apply(pca_matrix_family_from_method)
    if "balanced_pixel_strategy" not in out.columns:
        out["balanced_pixel_strategy"] = np.where(
            out[matrix_col].astype(str).eq("balanced_pixels"),
            "random",
            "not_applicable",
        )
    if "balanced_pixel_strategy_effective" not in out.columns:
        out["balanced_pixel_strategy_effective"] = np.where(
            out[matrix_col].astype(str).eq("balanced_pixels"),
            out["balanced_pixel_strategy"].astype(str),
            "random",
        )
    if config.variant_col not in out.columns and matrix_col in out.columns:
        out[config.variant_col] = out.apply(
            lambda row: pca_matrix_variant_from_method(
                matrix_method=row[matrix_col],
                balanced_pixel_strategy=row.get(
                    "balanced_pixel_strategy_effective",
                    row.get("balanced_pixel_strategy", "random"),
                ),
            ),
            axis=1,
        )
    return out


def _scale_metric(
    values,
    reference_values=None,
    *,
    robust: bool = True,
    clip_quantiles=(0.05, 0.95),
    eps: float = 1e-12,
) -> pd.Series:
    values = pd.Series(values, dtype="float64")
    reference = values if reference_values is None else pd.Series(reference_values, dtype="float64")
    reference = reference.replace([np.inf, -np.inf], np.nan).dropna()
    if len(reference) <= 1:
        return pd.Series(np.zeros(len(values)), index=values.index)
    lo, hi = reference.quantile(clip_quantiles[0]), reference.quantile(clip_quantiles[1])
    values_clip = values.clip(lo, hi)
    reference_clip = reference.clip(lo, hi)
    if robust:
        center = reference_clip.median()
        scale = reference_clip.quantile(0.75) - reference_clip.quantile(0.25)
    else:
        center = reference_clip.mean()
        scale = reference_clip.std(ddof=0)
    if not np.isfinite(scale) or scale < eps:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values_clip - center) / (scale + eps)


def _profile_for_family(
    family: str,
    config: PCASelectionConfig,
) -> PCASelectionProfile:
    family = str(family)
    if family in config.profiles:
        return config.profiles[family]
    return config.profiles["object_matrix"]


def _score_group(
    group: pd.DataFrame,
    config: PCASelectionConfig,
    reference_group: pd.DataFrame | None = None,
) -> pd.Series:
    reference_group = group if reference_group is None else reference_group
    scaled = {}
    all_metrics: set[str] = set()
    for profile in config.profiles.values():
        all_metrics.update(profile.positive_weights)
        all_metrics.update(profile.negative_weights)

    for metric in sorted(all_metrics):
        scaled[metric] = (
            _scale_metric(
                group[metric],
                reference_group[metric],
                robust=config.robust,
                clip_quantiles=config.clip_quantiles,
                eps=config.eps,
            )
            if metric in group.columns and metric in reference_group.columns
            else pd.Series(np.zeros(len(group)), index=group.index)
        )

    scores = pd.Series(np.zeros(len(group)), index=group.index, dtype="float64")
    for idx in group.index:
        family = str(group.loc[idx, config.family_col])
        profile = _profile_for_family(family, config)
        score = 0.0
        for metric, weight in profile.positive_weights.items():
            score += float(weight) * float(scaled[metric].loc[idx])
        for metric, weight in profile.negative_weights.items():
            score -= float(weight) * float(scaled[metric].loc[idx])
        scores.loc[idx] = score
    return scores


def _add_raw_scores_for_group(
    group: pd.DataFrame,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    out = group.copy()
    active_scores = _score_group(out, config)

    for family in config.profiles:
        out[f"{family}_score_raw"] = np.nan
        out[f"{family}_score"] = np.nan

    all_metrics: set[str] = set()
    for profile in config.profiles.values():
        all_metrics.update(profile.positive_weights)
        all_metrics.update(profile.negative_weights)

    scaled = {
        metric: (
            _scale_metric(
                out[metric],
                robust=config.robust,
                clip_quantiles=config.clip_quantiles,
                eps=config.eps,
            )
            if metric in out.columns
            else pd.Series(np.zeros(len(out)), index=out.index)
        )
        for metric in sorted(all_metrics)
    }

    for idx in out.index:
        family = str(out.loc[idx, config.family_col])
        profile = _profile_for_family(family, config)
        raw_score = float(active_scores.loc[idx])
        if family in config.profiles:
            out.loc[idx, f"{family}_score_raw"] = raw_score
            out.loc[idx, f"{family}_score"] = raw_score
        for metric, weight in profile.positive_weights.items():
            out.loc[idx, f"contrib_plus_{metric}"] = float(weight) * float(scaled[metric].loc[idx])
        for metric, weight in profile.negative_weights.items():
            out.loc[idx, f"contrib_minus_{metric}"] = -float(weight) * float(scaled[metric].loc[idx])
        out.loc[idx, "active_family_score_raw"] = raw_score
        out.loc[idx, "selection_score_without_stability"] = raw_score
        out.loc[idx, config.score_col] = raw_score

    return out


def _bootstrap_stability_for_group(
    group: pd.DataFrame,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    out = group.copy()
    n_iter = int(config.stability_bootstrap_iterations)
    if n_iter <= 1 or len(out) <= 2:
        out["selection_score_stability_mean"] = out[config.score_col]
        out["selection_score_stability_std"] = 0.0
        out["selection_score_rank_std"] = 0.0
        return out

    rng = np.random.default_rng(config.random_state)
    scores = np.zeros((n_iter, len(out)), dtype=float)
    ranks = np.zeros((n_iter, len(out)), dtype=float)
    indices = np.asarray(out.index)

    for iteration in range(n_iter):
        sampled_index = rng.choice(indices, size=len(indices), replace=True)
        reference_group = out.loc[sampled_index]
        sample_scores = _score_group(out, config, reference_group=reference_group)
        scores[iteration, :] = sample_scores.to_numpy(dtype=float)
        ranks[iteration, :] = sample_scores.rank(method="average", ascending=False).to_numpy(dtype=float)

    out["selection_score_stability_mean"] = scores.mean(axis=0)
    out["selection_score_stability_std"] = scores.std(axis=0, ddof=1)
    out["selection_score_rank_std"] = ranks.std(axis=0, ddof=1)
    return out


def _add_bootstrap_stability(
    df: pd.DataFrame,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    group_cols = _as_group_cols(config.group_cols)
    if not group_cols:
        out = _bootstrap_stability_for_group(df, config)
    else:
        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing group column(s) for PCA score stability: {missing}")
        parts = []
        for _, group_index in _groupby_indices(df, group_cols):
            parts.append(_bootstrap_stability_for_group(df.loc[group_index], config))
        out = pd.concat(parts, ignore_index=False, sort=False).sort_index().reset_index(drop=True)

    out["selection_score_stability_penalty"] = (
        float(config.stability_penalty_weight)
        * out["selection_score_stability_std"].fillna(0.0)
    )
    out[config.score_col] = (
        out["selection_score_without_stability"]
        - out["selection_score_stability_penalty"]
    )
    for family in config.profiles:
        mask = out[config.family_col].astype(str).eq(family)
        out.loc[mask, f"{family}_score"] = out.loc[mask, config.score_col]
    return out


def _quantile_thresholds(
    group: pd.DataFrame,
    metric: str,
    quantile: float,
) -> float:
    if metric not in group.columns:
        return np.nan
    values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return np.nan
    return float(values.quantile(quantile))


def add_pca_relative_quality_flags(
    df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Add data-relative PCA quality flags instead of fixed hard-coded thresholds."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    out = df.copy()
    threshold_rows = []
    for family, group in out.groupby(config.family_col, dropna=False):
        profile = _profile_for_family(str(family), config)
        threshold_rows.append(
            {
                config.family_col: family,
                "separation_low_threshold": _quantile_thresholds(
                    group,
                    profile.separation_metric,
                    config.quality_lower_quantile,
                ),
                "batch_high_threshold": _quantile_thresholds(
                    group,
                    profile.batch_metric,
                    config.quality_upper_quantile,
                ),
                "projection_high_threshold": _quantile_thresholds(
                    group,
                    profile.projection_metric,
                    config.quality_upper_quantile,
                ),
                "validation_shift_high_threshold": _quantile_thresholds(
                    group,
                    profile.validation_metric,
                    config.validation_upper_quantile,
                ),
                "score_stability_high_threshold": _quantile_thresholds(
                    group,
                    "selection_score_stability_std",
                    config.quality_upper_quantile,
                ),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    out = out.merge(thresholds, on=config.family_col, how="left")

    flags = []
    warnings = []
    passes = []
    for _, row in out.iterrows():
        family = str(row.get(config.family_col, "unknown_matrix_family"))
        profile = _profile_for_family(family, config)
        row_flags = []
        if (
            profile.separation_metric in row
            and pd.notna(row.get("separation_low_threshold"))
            and pd.notna(row.get(profile.separation_metric))
            and float(row.get(profile.separation_metric)) <= float(row.get("separation_low_threshold"))
        ):
            row_flags.append("weak_relative_separation")
        if (
            profile.batch_metric in row
            and pd.notna(row.get("batch_high_threshold"))
            and pd.notna(row.get(profile.batch_metric))
            and float(row.get(profile.batch_metric)) >= float(row.get("batch_high_threshold"))
        ):
            row_flags.append("batch_sensitive")
        if (
            profile.projection_metric in row
            and pd.notna(row.get("projection_high_threshold"))
            and pd.notna(row.get(profile.projection_metric))
            and float(row.get(profile.projection_metric)) >= float(row.get("projection_high_threshold"))
        ):
            row_flags.append("unstable_projection")
        if (
            profile.validation_metric in row
            and pd.notna(row.get("validation_shift_high_threshold"))
            and pd.notna(row.get(profile.validation_metric))
            and float(row.get(profile.validation_metric)) >= float(row.get("validation_shift_high_threshold"))
        ):
            row_flags.append("high_projection_shift")
        if (
            "selection_score_stability_std" in row
            and pd.notna(row.get("score_stability_high_threshold"))
            and pd.notna(row.get("selection_score_stability_std"))
            and float(row.get("selection_score_stability_std")) >= float(row.get("score_stability_high_threshold"))
        ):
            row_flags.append("score_unstable")
        flags.append(row_flags[0] if row_flags else "candidate")
        warnings.append("; ".join(row_flags))
        passes.append(len(row_flags) == 0)

    out["selection_flag"] = flags
    out["pca_validation_warning"] = warnings
    out["pca_validation_pass"] = passes
    return out


def add_pca_selection_scores(
    summary_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
    *,
    group_col=None,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Add configurable, family-specific PCA selection scores and diagnostics."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    if group_col is not None:
        config = replace(config, group_cols=_as_group_cols(group_col))
    if score_col is not None:
        config = replace(config, score_col=score_col)

    df = _ensure_selection_metadata(summary_df, config)
    group_cols = _as_group_cols(config.group_cols)
    if group_cols:
        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing group column(s) in PCA summary: {missing}")
        parts = []
        for _, group_index in _groupby_indices(df, group_cols):
            parts.append(_add_raw_scores_for_group(df.loc[group_index], config))
        scored = pd.concat(parts, ignore_index=False, sort=False).sort_index().reset_index(drop=True)
    else:
        scored = _add_raw_scores_for_group(df, config)

    scored = _add_bootstrap_stability(scored, config)
    scored = add_pca_relative_quality_flags(scored, config)
    return scored


def format_pca_selection_reason(
    row: pd.Series,
    config: PCASelectionConfig | None = None,
) -> str:
    """Build a compact human-readable reason for a selected preprocessing row."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    family = str(row.get(config.family_col, "unknown_family"))
    variant = str(row.get(config.variant_col, "unknown_variant"))
    flag = str(row.get("selection_flag", "unknown"))
    warning = str(row.get("pca_validation_warning", "") or "none")
    stability = row.get("selection_score_stability_std", np.nan)
    stability_text = "unknown" if pd.isna(stability) else f"{float(stability):.4g}"
    return (
        f"top_{config.max_preprocessings_per_family}_preprocessing_within_{family}; "
        f"best_variant={variant}; quality_flag={flag}; "
        f"validation_warning={warning}; score_stability_std={stability_text}"
    )


def build_pca_scoring_diagnostics(
    scored_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Return a compact scoring-diagnostic table separate from the full PCA summary."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    base_cols = [
        "rank",
        config.family_col,
        config.variant_col,
        config.matrix_method_col,
        "balanced_pixel_strategy",
        config.preprocessing_col,
        config.preprocessing_steps_col,
        config.score_col,
        "selection_score_without_stability",
        "selection_score_stability_mean",
        "selection_score_stability_std",
        "selection_score_rank_std",
        "selection_score_stability_penalty",
        "object_matrix_score",
        "pixel_matrix_score",
        "object_matrix_score_raw",
        "pixel_matrix_score_raw",
        "active_family_score_raw",
        "selection_flag",
        "pca_validation_warning",
        "pca_validation_pass",
        "separation_low_threshold",
        "batch_high_threshold",
        "projection_high_threshold",
        "validation_shift_high_threshold",
        "score_stability_high_threshold",
    ]
    metric_cols: list[str] = []
    for profile in config.profiles.values():
        for metric in [*profile.positive_weights, *profile.negative_weights]:
            if metric not in metric_cols:
                metric_cols.append(metric)
    contribution_cols = sorted(
        col
        for col in scored_df.columns
        if col.startswith("contrib_plus_") or col.startswith("contrib_minus_")
    )
    ordered_cols = []
    for col in [*base_cols, *metric_cols, *contribution_cols]:
        if col in scored_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    out = scored_df.loc[:, ordered_cols].copy()
    if "rank" in out.columns:
        out = out.sort_values("rank", ascending=True)
    elif config.score_col in out.columns:
        out = out.sort_values(config.score_col, ascending=False)
    return out.reset_index(drop=True)


def validate_pca_preprocessing_shortlist(
    df: pd.DataFrame,
    *,
    max_per_family: int,
    expected_families: Sequence[str] | None = None,
    family_col: str = "matrix_family",
    context: str = "PCA shortlist",
) -> pd.Series:
    """Raise when a PCA preprocessing shortlist violates the family-size contract."""
    if df is None or len(df) == 0:
        raise RuntimeError(f"{context}: PCA selection is empty.")
    if family_col not in df.columns:
        raise KeyError(f"{context}: missing family column: {family_col}")
    family_counts = df.groupby(family_col, dropna=False).size().sort_index()
    too_many = family_counts[family_counts > int(max_per_family)]
    if len(too_many) > 0:
        raise RuntimeError(
            f"{context}: max {max_per_family} rows per {family_col}, "
            f"got {too_many.to_dict()}"
        )
    if expected_families is not None:
        missing = sorted(set(map(str, expected_families)) - set(df[family_col].astype(str)))
        if missing:
            raise RuntimeError(f"{context}: missing expected matrix families: {missing}")
    return family_counts


def select_pca_preprocessing_shortlist(
    scored_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Select at most N preprocessing rows per matrix family from a scored PCA table."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    required = [
        config.family_col,
        config.variant_col,
        config.matrix_method_col,
        "balanced_pixel_strategy",
        config.preprocessing_col,
        config.preprocessing_steps_col,
        config.score_col,
    ]
    missing = [col for col in required if col not in scored_df.columns]
    if missing:
        raise KeyError(f"Missing columns for strict PCA selection: {missing}")

    df = scored_df.copy()
    if "rank" not in df.columns:
        df = df.sort_values(config.score_col, ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

    candidate_pool = (
        df.sort_values(
            [
                config.family_col,
                config.preprocessing_col,
                config.score_col,
                "rank",
            ],
            ascending=[True, True, False, True],
        )
        .drop_duplicates([config.family_col, config.preprocessing_col], keep="first")
        .copy()
    )
    candidate_pool["selection_reason"] = candidate_pool.apply(
        lambda row: format_pca_selection_reason(row, config),
        axis=1,
    )
    candidate_pool["best_selection_score"] = candidate_pool[config.score_col]
    candidate_pool["best_rank"] = candidate_pool["rank"]
    candidate_pool["best_matrix_variant"] = candidate_pool[config.variant_col]
    candidate_pool["selected_from_variants"] = candidate_pool[config.variant_col].astype(str)
    candidate_pool["selected_from_methods"] = candidate_pool[config.matrix_method_col].astype(str)
    candidate_pool["selected_from_strategies"] = candidate_pool["balanced_pixel_strategy"].astype(str)
    candidate_pool["selection_reasons"] = candidate_pool["selection_reason"]
    candidate_pool["best_selection_flag"] = candidate_pool["selection_flag"] if "selection_flag" in candidate_pool.columns else "unknown"

    selected = (
        candidate_pool.sort_values(
            [config.family_col, config.score_col, "rank"],
            ascending=[True, False, True],
        )
        .groupby(config.family_col, group_keys=False, dropna=False)
        .head(config.max_preprocessings_per_family)
        .copy()
    )
    selected["family_selection_rank"] = (
        selected.groupby(config.family_col, dropna=False).cumcount() + 1
    )
    selected = (
        selected.sort_values([config.family_col, "family_selection_rank"])
        .reset_index(drop=True)
    )
    family_counts = validate_pca_preprocessing_shortlist(
        selected,
        max_per_family=config.max_preprocessings_per_family,
        expected_families=config.expected_families,
        family_col=config.family_col,
        context="strict PCA preprocessing selection",
    )
    return selected, candidate_pool.reset_index(drop=True), family_counts
