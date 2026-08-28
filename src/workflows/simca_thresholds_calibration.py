"""Vectorized and cross-fitted SIMCA threshold calibration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.utils import require_columns

_POLICY_COLUMNS = (
    "lower_quantile",
    "upper_quantile",
    "vote_threshold",
    "lower_threshold",
    "upper_threshold",
)

_COUNT_METRICS = {
    "n_observations",
    "n_target",
    "n_non_target",
}

def _prepare_observations(
    frame: pd.DataFrame,
    *,
    score_col: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        ("fold_id", "source_image", "truth", score_col),
        "OOF observations",
    )
    out = frame.copy().reset_index(drop=True)
    score = pd.to_numeric(out[score_col], errors="coerce")
    truth = pd.to_numeric(out["truth"], errors="coerce")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError(f"{score_col!r} contains non-finite values.")
    if not truth.isin((0, 1)).all():
        raise ValueError("truth must contain only binary values.")
    out[score_col] = score.astype(float)
    out["truth"] = truth.astype(bool)
    out["fold_id"] = pd.to_numeric(
        out["fold_id"],
        errors="raise",
    ).astype(int)
    return out


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray | float,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _binary_metric_arrays(
    scores: Sequence[float],
    truth: Sequence[bool],
    thresholds: Sequence[float],
) -> dict[str, np.ndarray]:
    score = np.asarray(scores, dtype=float).reshape(-1, 1)
    target = np.asarray(truth, dtype=bool).reshape(-1, 1)
    threshold = np.asarray(thresholds, dtype=float).reshape(1, -1)
    predicted_target = score >= threshold
    n_policies = threshold.shape[1]
    n_target = int(target.sum())
    n_non_target = int((~target).sum())

    target_miss = _safe_divide(
        np.sum(target & ~predicted_target, axis=0),
        n_target,
    )
    false_accept = _safe_divide(
        np.sum((~target) & predicted_target, axis=0),
        n_non_target,
    )
    balanced_accuracy = 1.0 - 0.5 * (
        target_miss + false_accept
    )

    return {
        "target_miss_rate": target_miss,
        "false_accept_rate": false_accept,
        "balanced_accuracy": balanced_accuracy,
        "n_observations": np.full(n_policies, len(target)),
        "n_target": np.full(n_policies, n_target),
        "n_non_target": np.full(n_policies, n_non_target),
    }


def _three_way_metric_arrays(
    scores: Sequence[float],
    truth: Sequence[bool],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, np.ndarray]:
    score = np.asarray(scores, dtype=float).reshape(-1, 1)
    target = np.asarray(truth, dtype=bool).reshape(-1, 1)
    lower_array = np.asarray(lower, dtype=float).reshape(1, -1)
    upper_array = np.asarray(upper, dtype=float).reshape(1, -1)
    if np.any(lower_array >= upper_array):
        raise ValueError("Every lower threshold must be below upper.")

    predicted_target = score >= upper_array
    predicted_non_target = score <= lower_array
    uncertain = ~(predicted_target | predicted_non_target)
    n_policies = lower_array.shape[1]
    n_target = int(target.sum())
    n_non_target = int((~target).sum())

    target_failure = target & predicted_non_target
    if expcfg.INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY == (
        "counts_as_miss"
    ):
        target_failure |= target & uncertain

    target_miss = _safe_divide(
        np.sum(target_failure, axis=0),
        n_target,
    )
    false_accept = _safe_divide(
        np.sum((~target) & predicted_target, axis=0),
        n_non_target,
    )
    target_uncertain = _safe_divide(
        np.sum(target & uncertain, axis=0),
        n_target,
    )
    non_target_uncertain = _safe_divide(
        np.sum((~target) & uncertain, axis=0),
        n_non_target,
    )

    target_decided = np.sum(target & ~uncertain, axis=0)
    non_target_decided = np.sum((~target) & ~uncertain, axis=0)
    sensitivity = _safe_divide(
        np.sum(target & predicted_target, axis=0),
        target_decided,
    )
    specificity = _safe_divide(
        np.sum((~target) & predicted_non_target, axis=0),
        non_target_decided,
    )
    uncertain_rate = uncertain.mean(axis=0)

    return {
        "target_miss_rate": target_miss,
        "false_accept_rate": false_accept,
        "uncertain_rate": uncertain_rate,
        "target_uncertain_rate": target_uncertain,
        "non_target_uncertain_rate": non_target_uncertain,
        "coverage_rate": 1.0 - uncertain_rate,
        "decided_balanced_accuracy": 0.5 * (
            sensitivity + specificity
        ),
        "n_observations": np.full(n_policies, len(target)),
        "n_target": np.full(n_policies, n_target),
        "n_non_target": np.full(n_policies, n_non_target),
    }


def _group_positions(
    frame: pd.DataFrame,
    column: str,
) -> list[np.ndarray]:
    return [
        np.asarray(indices, dtype=int)
        for indices in frame.groupby(
            column,
            sort=False,
            dropna=False,
        ).indices.values()
    ]


def _reduce_metric_parts(
    parts: Sequence[dict[str, np.ndarray]],
    metric: str,
    reduction: str,
) -> np.ndarray:
    values = np.vstack([part[metric] for part in parts])
    finite = np.isfinite(values)
    if reduction == "mean":
        return _safe_divide(
            np.nansum(values, axis=0),
            finite.sum(axis=0),
        )
    if reduction == "max":
        result = np.max(
            np.where(finite, values, -np.inf),
            axis=0,
        )
        result[~finite.any(axis=0)] = np.nan
        return result
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def _macro_object_target_miss_rate(
    frame: pd.DataFrame,
    *,
    score_col: str,
    decision_mode: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Mean pixel-level target miss rate across target objects.

    Each target object receives the same weight, independently of its
    number of pixels.

    For 3-way decisions, the definition follows the frozen uncertainty
    policy:
    - safe_reject: uncertain target pixels are not counted as misses;
    - counts_as_miss: uncertain target pixels are counted as misses.
    """
    require_columns(
        frame,
        (
            "source_image",
            "object_id",
            "truth",
            score_col,
        ),
        "pixel observations for macro-object metric",
    )

    work = frame[
        [
            "source_image",
            "object_id",
            "truth",
            score_col,
        ]
    ].copy()

    truth = pd.to_numeric(
        work["truth"],
        errors="coerce",
    )
    if not truth.isin((0, 1)).all():
        raise ValueError(
            "truth must contain only binary values."
        )
    work["truth"] = truth.astype(bool)

    score = pd.to_numeric(
        work[score_col],
        errors="coerce",
    )
    if not np.isfinite(
        score.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{score_col!r} contains non-finite values."
        )
    work[score_col] = score.astype(float)

    # One object must have one unique class label.
    truth_counts = (
        work.groupby(
            ["source_image", "object_id"],
            sort=False,
            dropna=False,
        )["truth"]
        .nunique(dropna=False)
    )
    if truth_counts.gt(1).any():
        raise RuntimeError(
            "Pixels from one object have inconsistent truth labels."
        )

    # Only target objects contribute to a target-miss metric.
    target_work = (
        work.loc[
            work["truth"].astype(bool),
            [
                "source_image",
                "object_id",
                score_col,
            ],
        ]
        .reset_index(drop=True)
    )

    lower_array = np.asarray(
        lower,
        dtype=float,
    ).reshape(1, -1)
    upper_array = np.asarray(
        upper,
        dtype=float,
    ).reshape(1, -1)

    n_policies = lower_array.shape[1]

    if target_work.empty:
        return np.full(
            n_policies,
            np.nan,
            dtype=float,
        )

    scores = target_work[
        score_col
    ].to_numpy(dtype=float)

    object_rates = []

    grouped_positions = (
        target_work.groupby(
            ["source_image", "object_id"],
            sort=False,
            dropna=False,
        )
        .indices
        .values()
    )

    for positions in grouped_positions:
        positions = np.asarray(
            positions,
            dtype=int,
        )
        object_scores = scores[
            positions
        ].reshape(-1, 1)

        if decision_mode == "2way":
            # predicted_target = score >= threshold
            missed = (
                object_scores < lower_array
            )

        elif decision_mode == "3way":
            policy = str(
                expcfg.INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY
            )

            if policy == "safe_reject":
                # Only an explicit non-target decision is a miss.
                # predicted_non_target = score <= lower
                missed = (
                    object_scores <= lower_array
                )

            elif policy == "counts_as_miss":
                # Non-target + uncertain = anything not predicted target.
                # predicted_target = score >= upper
                missed = (
                    object_scores < upper_array
                )

            else:
                raise ValueError(
                    "Unsupported "
                    "INTERNAL_CALIBRATION_TARGET_UNCERTAIN_POLICY: "
                    f"{policy!r}"
                )

        else:
            raise ValueError(
                f"Unsupported decision mode: {decision_mode!r}"
            )

        object_rates.append(
            missed.mean(axis=0)
        )

    return np.mean(
        np.vstack(object_rates),
        axis=0,
    )


def _evaluate_threshold_block(
    frame: pd.DataFrame,
    *,
    score_col: str,
    decision_mode: str,
    lower: np.ndarray,
    upper: np.ndarray,
    primary_unit: str,
) -> pd.DataFrame:
    score = frame[
        score_col
    ].to_numpy(
        dtype=float
    )

    truth = frame[
        "truth"
    ].to_numpy(
        dtype=bool
    )

    def evaluate(
        positions: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        selected_score = (
            score
            if positions is None
            else score[positions]
        )

        selected_truth = (
            truth
            if positions is None
            else truth[positions]
        )

        if decision_mode == "2way":
            return _binary_metric_arrays(
                selected_score,
                selected_truth,
                lower,
            )

        if decision_mode == "3way":
            return _three_way_metric_arrays(
                selected_score,
                selected_truth,
                lower,
                upper,
            )

        raise ValueError(
            f"Unsupported decision mode: {decision_mode!r}"
        )

    metrics = evaluate()

    image_parts = [
        evaluate(
            positions
        )
        for positions in _group_positions(
            frame,
            "source_image",
        )
    ]

    if primary_unit == "source_image":
        # --------------------------------------------------------------
        # Direct pixel-projection tracks:
        #
        # Primary protocol unit remains the source image.
        # Therefore the existing macro-image metrics are unchanged.
        # --------------------------------------------------------------

        for metric in (
            set(metrics)
            - _COUNT_METRICS
        ):
            metrics[
                metric
            ] = _reduce_metric_parts(
                image_parts,
                metric,
                "mean",
            )

        # --------------------------------------------------------------
        # Additional diagnostic requested for pixel-direct tracks:
        #
        # Compute one target miss rate per object from its pixels,
        # then average equally across target objects.
        #
        # Non-target objects have an undefined target miss rate (NaN)
        # and are naturally ignored by _reduce_metric_parts.
        # --------------------------------------------------------------

        object_parts = [
            evaluate(
                np.asarray(
                    indices,
                    dtype=int,
                )
            )
            for indices in frame.groupby(
                [
                    "source_image",
                    "object_id",
                ],
                sort=False,
                dropna=False,
            ).indices.values()
        ]

        metrics[
            "macro_object_target_miss_rate"
        ] = _reduce_metric_parts(
            object_parts,
            "target_miss_rate",
            "mean",
        )

    elif primary_unit != "object":
        raise ValueError(
            f"Unsupported primary unit: {primary_unit!r}"
        )

    metrics[
        "max_unit_target_miss_rate"
    ] = _reduce_metric_parts(
        image_parts,
        "target_miss_rate",
        "max",
    )

    metrics[
        "max_unit_false_accept_rate"
    ] = _reduce_metric_parts(
        image_parts,
        "false_accept_rate",
        "max",
    )

    return pd.DataFrame(metrics)

def build_quantile_policies(
    scores: Sequence[float],
    *,
    center: float,
    lower_quantiles: Sequence[float],
    upper_quantiles: Sequence[float],
) -> pd.DataFrame:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    lower_support = values[values < float(center)]
    upper_support = values[values > float(center)]
    columns = (
        "lower_quantile",
        "upper_quantile",
        "lower_threshold",
        "upper_threshold",
    )
    if not len(lower_support) or not len(upper_support):
        return pd.DataFrame(columns=columns)

    lower_q = np.asarray(lower_quantiles, dtype=float)
    upper_q = np.asarray(upper_quantiles, dtype=float)
    if np.any((lower_q < 0.0) | (lower_q > 1.0)):
        raise ValueError("Lower quantiles must be in [0, 1].")
    if np.any((upper_q < 0.0) | (upper_q > 1.0)):
        raise ValueError("Upper quantiles must be in [0, 1].")

    lower_values = np.quantile(lower_support, lower_q)
    upper_values = np.quantile(upper_support, upper_q)
    policies = pd.DataFrame(
        {
            "lower_quantile": np.repeat(lower_q, len(upper_q)),
            "upper_quantile": np.tile(upper_q, len(lower_q)),
            "lower_threshold": np.repeat(
                lower_values,
                len(upper_q),
            ),
            "upper_threshold": np.tile(
                upper_values,
                len(lower_q),
            ),
        }
    )
    valid = (
        policies["lower_threshold"].lt(float(center))
        & policies["upper_threshold"].gt(float(center))
        & policies["lower_threshold"].lt(
            policies["upper_threshold"]
        )
    )
    return policies.loc[valid].reset_index(drop=True)


def metrics_to_long(
    evaluated: pd.DataFrame,
    *,
    model_id: str,
    random_state: int,
    evaluation_fold: int,
    decision_scope: str,
) -> pd.DataFrame:
    policy_columns = [
        column for column in _POLICY_COLUMNS
        if column in evaluated
    ]
    metric_columns = [
        column for column in evaluated
        if column not in policy_columns
    ]
    long = evaluated.melt(
        id_vars=policy_columns,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    long["model_id"] = str(model_id)
    long["random_state"] = int(random_state)
    long["evaluation_fold"] = int(evaluation_fold)
    long["decision_scope"] = str(decision_scope)
    return long.reindex(
        columns=expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
    )


def _evaluate_fixed_scope(
    frame: pd.DataFrame,
    *,
    score_col: str,
    thresholds: Sequence[float],
    model_id: str,
    random_state: int,
    decision_scope: str,
    primary_unit: str,
    thresholds_are_votes: bool,
) -> pd.DataFrame:
    observations = _prepare_observations(
        frame,
        score_col=score_col,
    )
    values = np.asarray(thresholds, dtype=float)
    policies = pd.DataFrame(
        {
            "lower_quantile": np.nan,
            "upper_quantile": np.nan,
            "vote_threshold": (
                values if thresholds_are_votes else np.nan
            ),
            "lower_threshold": values,
            "upper_threshold": values,
        }
    )
    parts = []
    fold_ids = sorted(observations["fold_id"].unique())
    for evaluation_fold in (*fold_ids, -1):
        evaluation = (
            observations
            if evaluation_fold == -1
            else observations.loc[
                observations["fold_id"].eq(evaluation_fold)
            ]
        )
        metrics = _evaluate_threshold_block(
            evaluation,
            score_col=score_col,
            decision_mode="2way",
            lower=values,
            upper=values,
            primary_unit=primary_unit,
        )
        evaluated = pd.concat(
            [
                policies.reset_index(drop=True),
                metrics.reset_index(drop=True),
            ],
            axis=1,
        )
        parts.append(
            metrics_to_long(
                evaluated,
                model_id=model_id,
                random_state=random_state,
                evaluation_fold=evaluation_fold,
                decision_scope=decision_scope,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _evaluate_quantile_scope(
    frame: pd.DataFrame,
    *,
    score_col: str,
    center: float,
    model_id: str,
    random_state: int,
    decision_scope: str,
    primary_unit: str,
) -> pd.DataFrame:
    observations = _prepare_observations(
        frame,
        score_col=score_col,
    )
    fold_ids = sorted(observations["fold_id"].unique())
    block_size = int(
        expcfg.INTERNAL_CALIBRATION_THRESHOLD_BLOCK_SIZE
    )
    if block_size < 1:
        raise ValueError("Threshold block size must be positive.")

    parts = []
    for evaluation_fold in (*fold_ids, -1):
        evaluation = (
            observations
            if evaluation_fold == -1
            else observations.loc[
                observations["fold_id"].eq(evaluation_fold)
            ]
        )
        if (
            evaluation_fold == -1
            or not expcfg.INTERNAL_CALIBRATION_THRESHOLD_CROSSFIT
        ):
            calibration = observations
        else:
            calibration = observations.loc[
                ~observations["fold_id"].eq(evaluation_fold)
            ]

        policies = build_quantile_policies(
            calibration[score_col].to_numpy(dtype=float),
            center=center,
            lower_quantiles=(
                expcfg
                .INTERNAL_CALIBRATION_THREE_WAY_LOWER_QUANTILES
            ),
            upper_quantiles=(
                expcfg
                .INTERNAL_CALIBRATION_THREE_WAY_UPPER_QUANTILES
            ),
        )
        if policies.empty:
            continue
        policies["vote_threshold"] = np.nan

        for start in range(0, len(policies), block_size):
            block = policies.iloc[start:start + block_size].copy()
            metrics = _evaluate_threshold_block(
                evaluation,
                score_col=score_col,
                decision_mode="3way",
                lower=block["lower_threshold"].to_numpy(dtype=float),
                upper=block["upper_threshold"].to_numpy(dtype=float),
                primary_unit=primary_unit,
            )
            evaluated = pd.concat(
                [
                    block.reset_index(drop=True),
                    metrics.reset_index(drop=True),
                ],
                axis=1,
            )
            parts.append(
                metrics_to_long(
                    evaluated,
                    model_id=model_id,
                    random_state=random_state,
                    evaluation_fold=evaluation_fold,
                    decision_scope=decision_scope,
                )
            )

    if not parts:
        return pd.DataFrame(
            columns=expcfg
            .INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
        )
    return pd.concat(parts, ignore_index=True)


def build_pixel_vote_table(
    pixel_predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (
        "fold_id",
        "source_image",
        "object_id",
    ),
) -> pd.DataFrame:
    """Aggregate binary pixel membership to one target ratio per object/group.

    The binary pixel membership definition is exactly the one used during 03B
    threshold calibration: ``simca_margin >= INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD``.
    ``group_columns`` allows the same implementation to be reused for OOF data
    (with ``fold_id``) and locked batch-3 validation data (without ``fold_id``).
    """
    group_columns = tuple(map(str, group_columns))
    required = {*group_columns, "truth", "simca_margin"}
    require_columns(pixel_predictions, required, "pixel predictions")

    work = pixel_predictions[
        list(dict.fromkeys([*group_columns, "truth", "simca_margin"]))
    ].copy()
    margin = pd.to_numeric(work["simca_margin"], errors="coerce")
    if not np.isfinite(margin.to_numpy(dtype=float)).all():
        raise ValueError("simca_margin contains non-finite values.")
    truth = pd.to_numeric(work["truth"], errors="coerce")
    if not truth.isin((0, 1)).all():
        raise ValueError("truth must contain only binary values.")
    work["truth"] = truth.astype(bool)
    work["simca_margin"] = margin.astype(float)

    truth_counts = work.groupby(
        list(group_columns),
        sort=False,
        dropna=False,
    )["truth"].nunique(dropna=False)
    if truth_counts.gt(1).any():
        raise RuntimeError(
            "Pixels from one aggregation group have inconsistent truth labels."
        )

    work["_pixel_target"] = work["simca_margin"].ge(
        expcfg.INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD
    )
    return (
        work.groupby(
            list(group_columns),
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            truth=("truth", "first"),
            pixel_target_ratio=("_pixel_target", "mean"),
        )
    )


def _projection_groups(
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {}
    require_columns(frame, ("projection_id",), "OOF predictions")
    return {
        str(projection_id): group.reset_index(drop=True)
        for projection_id, group in frame.groupby(
            "projection_id",
            sort=False,
            dropna=False,
        )
    }


def evaluate_calibration_thresholds(
    oof_object_predictions: pd.DataFrame,
    oof_pixel_predictions: pd.DataFrame,
    configurations: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate cross-fitted threshold policies without adding policy IDs."""
    required = (
        "model_id",
        "random_state",
        "projection_id",
        "evaluation_track",
        "decision_mode",
        "projection_level",
    )
    require_columns(configurations, required, "configurations")
    models = configurations.loc[:, required].drop_duplicates()
    if models.duplicated(["model_id", "random_state"]).any():
        raise RuntimeError(
            "(model_id, random_state) must identify one execution."
        )

    object_groups = _projection_groups(oof_object_predictions)
    pixel_groups = _projection_groups(oof_pixel_predictions)
    parts = []

    for row in models.itertuples(index=False):
        evaluation_track = str(row.evaluation_track)
        if evaluation_track not in expcfg.SIMCA_EVALUATION_TRACK_SPECS:
            raise KeyError(
                f"Unknown evaluation track: {evaluation_track!r}"
            )
        spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[evaluation_track]
        decision_mode = str(row.decision_mode)
        projection_level = str(row.projection_level)
        if decision_mode != str(spec["decision_mode"]):
            raise RuntimeError(
                f"Decision-mode mismatch for {evaluation_track}."
            )
        if projection_level != str(spec["projection_level"]):
            raise RuntimeError(
                f"Projection-level mismatch for {evaluation_track}."
            )

        projection_id = str(row.projection_id)
        source_groups = (
            pixel_groups
            if projection_level == "pixel_projection"
            else object_groups
        )
        if projection_id not in source_groups:
            continue

        predictions = source_groups[projection_id]
        common = {
            "model_id": str(row.model_id),
            "random_state": int(row.random_state),
            "decision_scope": "direct",
            "primary_unit": str(spec["primary_unit"]),
        }

        if decision_mode == "2way":
            parts.append(
                _evaluate_fixed_scope(
                    predictions,
                    score_col="simca_margin",
                    thresholds=(
                        expcfg
                        .INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD,
                    ),
                    thresholds_are_votes=False,
                    **common,
                )
            )
        else:
            parts.append(
                _evaluate_quantile_scope(
                    predictions,
                    score_col="simca_margin",
                    center=(
                        expcfg
                        .INTERNAL_CALIBRATION_DIRECT_2WAY_THRESHOLD
                    ),
                    **common,
                )
            )

        if projection_level != "pixel_projection":
            continue

        object_votes = build_pixel_vote_table(predictions)
        object_common = {
            "model_id": str(row.model_id),
            "random_state": int(row.random_state),
            "decision_scope": "pixel_to_object",
            "primary_unit": "object",
        }
        if decision_mode == "2way":
            parts.append(
                _evaluate_fixed_scope(
                    object_votes,
                    score_col="pixel_target_ratio",
                    thresholds=(
                        expcfg
                        .INTERNAL_CALIBRATION_OBJECT_THRESHOLDS
                    ),
                    thresholds_are_votes=True,
                    **object_common,
                )
            )
        else:
            parts.append(
                _evaluate_quantile_scope(
                    object_votes,
                    score_col="pixel_target_ratio",
                    center=(
                        expcfg
                        .INTERNAL_CALIBRATION_PIXEL_VOTE_CENTER
                    ),
                    **object_common,
                )
            )

    columns = expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
    if not parts:
        return pd.DataFrame(columns=columns)

    result = pd.concat(parts, ignore_index=True).reindex(
        columns=columns
    )
    natural_key = (
        "model_id",
        "random_state",
        "evaluation_fold",
        "decision_scope",
        "lower_quantile",
        "upper_quantile",
        "vote_threshold",
        "metric",
    )
    if result.duplicated(list(natural_key)).any():
        raise RuntimeError(
            "Duplicate natural keys in threshold metrics."
        )
    return result.sort_values(
        list(natural_key),
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)