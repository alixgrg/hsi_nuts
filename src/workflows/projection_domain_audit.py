"""Train-to-projection diagnostics shared by notebooks 03B and 03C."""

from __future__ import annotations

from collections.abc import Sequence
import json

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.decision.border import add_border_flags_to_pixel_df
from src.decision.metrics import coerce_binary_series


_FLOAT_COMPARISON_RTOL = 8.0 * np.finfo(float).eps
_FLOAT_COMPARISON_ATOL = np.finfo(float).tiny


def _is_numerically_equal(left, right):
    """Compare values at floating-point precision, without an absolute 1e-12 floor."""
    return np.isclose(
        left,
        right,
        rtol=_FLOAT_COMPARISON_RTOL,
        atol=_FLOAT_COMPARISON_ATOL,
    )


def _mean_preserving_unbounded(values) -> float:
    """Mean that keeps an unbounded shift explicit and never reduces +inf with -inf."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if numeric.size == 0 or np.isnan(numeric).any():
        return np.nan
    has_positive_infinity = bool(np.isposinf(numeric).any())
    has_negative_infinity = bool(np.isneginf(numeric).any())
    if has_positive_infinity and has_negative_infinity:
        # The direction is undefined, but the magnitude is unbounded. Returning
        # +inf keeps the crossing scientifically blocking instead of hiding it
        # behind a NaN created by (+inf) + (-inf).
        return np.inf
    if has_positive_infinity:
        return np.inf
    if has_negative_infinity:
        return -np.inf
    return float(numeric.mean())


def _pca_shift_norm(shifts) -> float:
    """Return the two-axis norm without silently dropping a missing/infinite PC."""
    values = np.asarray(shifts, dtype=float)
    if values.shape != (2,) or np.isnan(values).any():
        return np.nan
    if np.isinf(values).any():
        return np.inf
    return float(np.linalg.norm(values))


def _standardized_shift(train, projection) -> float:
    train_values = pd.to_numeric(pd.Series(train), errors="coerce").dropna()
    projection_values = pd.to_numeric(
        pd.Series(projection), errors="coerce"
    ).dropna()
    if train_values.empty or projection_values.empty:
        return np.nan
    scale = float(train_values.std(ddof=1))
    if not np.isfinite(scale):
        return np.nan
    if scale < 0.0:
        raise ValueError("A standard deviation cannot be negative.")
    train_mean = float(train_values.mean())
    projection_mean = float(projection_values.mean())
    if scale == 0.0:
        if _is_numerically_equal(train_mean, projection_mean):
            return 0.0
        return float(np.sign(projection_mean - train_mean) * np.inf)
    return float((projection_mean - train_mean) / scale)


def _location_scale(values) -> tuple[float, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return np.nan, np.nan
    return float(numeric.mean()), float(numeric.std(ddof=1))


def summarize_projection_shift(
    train_scores: pd.DataFrame,
    projection_scores: pd.DataFrame,
    train_margin=None,
    projection_margin=None,
    *,
    group_keys: Sequence[str] = (
        "projection_id",
        "projection_level",
        "projection_matrix_method",
        "fold_id",
    ),
) -> pd.DataFrame:
    """Store compact train references and train-to-projection shifts.

    The train location/scale values are intentionally persisted here. 03C can
    then stratify OOF projections without saving every training observation or
    refitting a model.
    """
    projection = projection_scores.copy()
    train = train_scores.copy()
    if train_margin is not None:
        train["simca_margin"] = np.asarray(train_margin, dtype=float)
    if projection_margin is not None:
        projection["simca_margin"] = np.asarray(
            projection_margin, dtype=float
        )
    required = {"H", "Q", "simca_margin"}
    for name, table in (("train", train), ("projection", projection)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise KeyError(f"Missing {name} score columns: {missing}")
    optional = (
        "pca_score_pc1",
        "pca_score_pc2",
        "rule_limit",
        "normalized_ratio",
    )
    for column in optional:
        if column not in train:
            train[column] = np.nan
        if column not in projection:
            projection[column] = np.nan
    active_keys = [key for key in group_keys if key in projection]
    grouped = (
        projection.groupby(active_keys, dropna=False, sort=False)
        if active_keys
        else [((), projection)]
    )
    rows = []
    for key, projected_group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(active_keys, key))
        train_group = train
        for column in ("fit_config_id", "fold_id"):
            if column in base and column in train_group:
                train_group = train_group.loc[
                    train_group[column].astype(str).eq(str(base[column]))
                ]
        target_projection = projected_group.loc[
            projected_group.get(
                "truth",
                pd.Series(True, index=projected_group.index),
            ).astype(bool)
        ]
        reference = {}
        for column, prefix in (
            ("pca_score_pc1", "pc1"),
            ("pca_score_pc2", "pc2"),
            ("H", "h"),
            ("Q", "q"),
            ("rule_limit", "rule_limit"),
            ("normalized_ratio", "normalized_ratio"),
            ("simca_margin", "margin"),
        ):
            mean, std = _location_scale(train_group[column])
            reference[f"train_{prefix}_mean"] = mean
            reference[f"train_{prefix}_std"] = std
        pc_shifts = np.asarray(
            [
                _standardized_shift(
                    train_group["pca_score_pc1"],
                    projected_group["pca_score_pc1"],
                ),
                _standardized_shift(
                    train_group["pca_score_pc2"],
                    projected_group["pca_score_pc2"],
                ),
            ],
            dtype=float,
        )
        rows.append(
            {
                **base,
                "n_train": int(len(train_group)),
                "n_projection": int(len(projected_group)),
                **reference,
                "pca_pc1_standardized_shift": pc_shifts[0],
                "pca_pc2_standardized_shift": pc_shifts[1],
                "pca_centroid_shift": _pca_shift_norm(pc_shifts),
                "h_standardized_shift": _standardized_shift(
                    train_group["H"], projected_group["H"]
                ),
                "q_standardized_shift": _standardized_shift(
                    train_group["Q"], projected_group["Q"]
                ),
                "rule_limit_standardized_shift": _standardized_shift(
                    train_group["rule_limit"],
                    projected_group["rule_limit"],
                ),
                "normalized_ratio_standardized_shift": _standardized_shift(
                    train_group["normalized_ratio"],
                    projected_group["normalized_ratio"],
                ),
                "margin_standardized_shift": _standardized_shift(
                    train_group["simca_margin"],
                    projected_group["simca_margin"],
                ),
                "projection_out_of_domain_rate": float(
                    pd.to_numeric(
                        projected_group["normalized_ratio"], errors="coerce"
                    ).ge(1.0).mean()
                ),
                "projection_target_rejection_rate": (
                    float(target_projection["simca_margin"].lt(0.0).mean())
                    if len(target_projection)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _standardize_against_reference(
    values: pd.Series,
    means: pd.Series,
    scales: pd.Series,
) -> np.ndarray:
    value = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(means, errors="coerce").to_numpy(dtype=float)
    scale = pd.to_numeric(scales, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(value), np.nan, dtype=float)
    if np.any(np.isfinite(scale) & (scale < 0.0)):
        raise ValueError("A standard-deviation reference cannot be negative.")
    finite = np.isfinite(value) & np.isfinite(mean) & np.isfinite(scale)
    regular = finite & (scale > 0.0)
    out[regular] = (value[regular] - mean[regular]) / scale[regular]
    constant = finite & (scale == 0.0)
    equal = _is_numerically_equal(value, mean)
    out[constant & equal] = 0.0
    out[constant & ~equal] = np.sign(value[constant & ~equal] - mean[constant & ~equal]) * np.inf
    return out


def _projection_rows_with_references(
    oof_objects: pd.DataFrame,
    oof_pixels: pd.DataFrame,
    selected_executions: pd.DataFrame,
    projection_shift: pd.DataFrame,
    *,
    object_db: dict | None,
    border_width: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_columns = list(expcfg.DOMAIN_SPATIAL_SELECTED_EXECUTION_COLUMNS)
    missing = sorted(set(mapping_columns) - set(selected_executions.columns))
    if missing:
        raise KeyError(f"Missing selected-execution columns: {missing}")
    mapping = selected_executions[mapping_columns].copy()
    execution_keys = ["model_id", "random_state"]
    if mapping.duplicated(execution_keys).any():
        raise RuntimeError("Selected (model_id, random_state) keys must be unique.")
    if mapping.empty:
        raise RuntimeError("03C received no selected execution from 03B.")
    allowed_levels = {"object_projection", "pixel_projection"}
    unknown_levels = sorted(
        set(mapping["projection_level"].astype(str)) - allowed_levels
    )
    if unknown_levels:
        raise RuntimeError(f"Unknown selected projection levels: {unknown_levels}")

    level_by_projection = mapping[
        ["projection_id", "projection_level"]
    ].drop_duplicates()
    if level_by_projection["projection_id"].astype(str).duplicated().any():
        raise RuntimeError("A projection_id maps to several projection levels.")
    object_projection_ids = set(
        level_by_projection.loc[
            level_by_projection["projection_level"].astype(str).eq(
                "object_projection"
            ),
            "projection_id",
        ].astype(str)
    )
    pixel_projection_ids = set(
        level_by_projection.loc[
            level_by_projection["projection_level"].astype(str).eq(
                "pixel_projection"
            ),
            "projection_id",
        ].astype(str)
    )

    tables = []
    observed_projection_ids: set[str] = set()
    if oof_objects is not None and not oof_objects.empty:
        if "projection_id" not in oof_objects:
            raise KeyError("OOF object predictions are missing projection_id.")
        objects = oof_objects.loc[
            oof_objects["projection_id"].astype(str).isin(object_projection_ids)
        ].copy()
        if not objects.empty:
            objects["border_core"] = "not_applicable"
            tables.append(objects)
            observed_projection_ids.update(objects["projection_id"].astype(str))
    if oof_pixels is not None and not oof_pixels.empty:
        if "projection_id" not in oof_pixels:
            raise KeyError("OOF pixel predictions are missing projection_id.")
        pixels = oof_pixels.loc[
            oof_pixels["projection_id"].astype(str).isin(pixel_projection_ids)
        ].copy()
        if not pixels.empty:
            if object_db is None:
                raise ValueError("object_db is required for border/core diagnostics.")
            pixels = add_border_flags_to_pixel_df(
                pixels,
                object_db,
                border_width=int(border_width),
            )
            pixels["border_core"] = np.select(
                [pixels["is_border_pixel"], pixels["is_core_pixel"]],
                ["border", "core"],
                default="unresolved",
            )
            tables.append(pixels)
            observed_projection_ids.update(pixels["projection_id"].astype(str))
    if not tables:
        raise RuntimeError("03C received no selected OOF prediction row from 03B.")
    expected_projection_ids = set(mapping["projection_id"].astype(str))
    missing_predictions = sorted(expected_projection_ids - observed_projection_ids)
    if missing_predictions:
        raise RuntimeError(
            "Selected 03B executions have no matching OOF predictions: "
            f"{missing_predictions[:10]}"
        )

    oof = pd.concat(tables, ignore_index=True, sort=False)
    oof = oof.merge(
        mapping,
        on="projection_id",
        how="inner",
        validate="many_to_many",
    )
    if oof.empty:
        raise RuntimeError("No selected 03B execution has matching OOF predictions.")

    reference_keys = ["projection_id", "fold_id"]
    reference_numeric_columns = [
        "train_pc1_mean",
        "train_pc1_std",
        "train_pc2_mean",
        "train_pc2_std",
        "train_h_mean",
        "train_h_std",
        "train_q_mean",
        "train_q_std",
        "train_rule_limit_mean",
        "train_rule_limit_std",
        "train_normalized_ratio_mean",
        "train_normalized_ratio_std",
        "train_margin_mean",
        "train_margin_std",
    ]
    required_reference = set(reference_keys + reference_numeric_columns)
    missing_reference = sorted(required_reference - set(projection_shift.columns))
    if missing_reference:
        raise RuntimeError(
            "projection_shift.parquet predates the selected-run 03C contract; "
            f"rerun 03B. Missing columns: {missing_reference}"
        )
    references = projection_shift.loc[
        projection_shift["projection_id"].astype(str).isin(expected_projection_ids),
        reference_keys + reference_numeric_columns,
    ].drop_duplicates()
    if references.duplicated(reference_keys).any():
        raise RuntimeError(
            "Conflicting train references exist for one projection/fold crossing."
        )
    out = oof.merge(
        references,
        on=reference_keys,
        how="left",
        validate="many_to_one",
    )
    reference_values = out[reference_numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(reference_values.to_numpy(dtype=float)).all():
        raise RuntimeError(
            "At least one selected OOF crossing has a missing/non-finite train reference."
        )
    reference_scales = [
        column for column in reference_numeric_columns if column.endswith("_std")
    ]
    if reference_values[reference_scales].lt(0.0).any().any():
        raise RuntimeError("A train-reference standard deviation is negative.")

    projection_numeric_columns = [
        "pca_score_pc1",
        "pca_score_pc2",
        "H",
        "Q",
        "rule_limit",
        "normalized_ratio",
        "simca_margin",
    ]
    missing_projection_values = sorted(
        set(projection_numeric_columns + ["truth"]) - set(out.columns)
    )
    if missing_projection_values:
        raise KeyError(f"Missing OOF diagnostic columns: {missing_projection_values}")
    projection_values = out[projection_numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(projection_values.to_numpy(dtype=float)).all():
        raise RuntimeError("OOF diagnostic values must all be finite.")
    out[projection_numeric_columns] = projection_values
    truth = coerce_binary_series(out["truth"], target_class=expcfg.TARGET_CLASS)
    if truth.isna().any():
        raise RuntimeError("OOF truth contains an unknown or missing binary label.")
    out["truth"] = truth.astype(bool)
    out["truth_class"] = np.where(out["truth"], "target", "non_target")
    standardizations = {
        "pc1_z": ("pca_score_pc1", "train_pc1_mean", "train_pc1_std"),
        "pc2_z": ("pca_score_pc2", "train_pc2_mean", "train_pc2_std"),
        "t2_z": ("H", "train_h_mean", "train_h_std"),
        "q_z": ("Q", "train_q_mean", "train_q_std"),
        "rule_limit_z": (
            "rule_limit",
            "train_rule_limit_mean",
            "train_rule_limit_std",
        ),
        "ratio_z": (
            "normalized_ratio",
            "train_normalized_ratio_mean",
            "train_normalized_ratio_std",
        ),
        "margin_z": (
            "simca_margin",
            "train_margin_mean",
            "train_margin_std",
        ),
    }
    for output, (value, mean, scale) in standardizations.items():
        out[output] = _standardize_against_reference(out[value], out[mean], out[scale])
    out["target_margin_delta"] = (
        out["simca_margin"].to_numpy(dtype=float)
        - reference_values["train_margin_mean"].to_numpy(dtype=float)
    )
    return out, mapping


def build_projection_shift_diagnostics(
    oof_objects: pd.DataFrame,
    oof_pixels: pd.DataFrame,
    selected_executions: pd.DataFrame,
    projection_shift: pd.DataFrame,
    *,
    object_db: dict | None,
    protocol_hash: str,
    border_width: int = expcfg.PROJECTION_DOMAIN_BORDER_WIDTH,
    min_stratum_n: int = expcfg.PROJECTION_DOMAIN_MIN_STRATUM_N,
    allowed_batches: Sequence[int] = (
        expcfg.PROJECTION_DOMAIN_AUDIT_ALLOWED_BATCHES
    ),
) -> pd.DataFrame:
    """Build vectorized diagnostics for every selected 03B execution."""
    rows, _ = _projection_rows_with_references(
        oof_objects,
        oof_pixels,
        selected_executions,
        projection_shift,
        object_db=object_db,
        border_width=int(border_width),
    )
    batches = set(pd.to_numeric(rows["batch"], errors="raise").astype(int))
    if not batches.issubset(set(map(int, allowed_batches))):
        raise RuntimeError(f"Forbidden batch in 03C domain audit: {sorted(batches)}")

    identifiers = ["model_id", "random_state", "track_id"]
    dimension_columns = {
        "overall": None,
        "fold": "fold_id",
        "size_bin": "size_bin",
        "border_core": "border_core",
        "truth_class": "truth_class",
        "source_image": "source_image",
    }
    dimensions = tuple(map(str, expcfg.PROJECTION_DOMAIN_DIAGNOSTIC_DIMENSIONS))
    unknown_dimensions = sorted(set(dimensions) - set(dimension_columns))
    if unknown_dimensions:
        raise ValueError(f"Unknown projection diagnostic dimensions: {unknown_dimensions}")

    rows = rows[
        [
            *identifiers,
            "fold_id",
            "size_bin",
            "border_core",
            "truth_class",
            "source_image",
            "truth",
            "pc1_z",
            "pc2_z",
            "t2_z",
            "q_z",
            "rule_limit_z",
            "ratio_z",
            "margin_z",
            "normalized_ratio",
            "simca_margin",
            "target_margin_delta",
        ]
    ].copy()
    rows["__out_of_domain"] = rows["normalized_ratio"].ge(1.0).astype(float)
    rows["__target_rejected"] = np.where(
        rows["truth"].to_numpy(dtype=bool),
        rows["simca_margin"].lt(0.0).to_numpy(dtype=bool).astype(float),
        np.nan,
    )
    rows["__target_margin_delta"] = np.where(
        rows["truth"].to_numpy(dtype=bool),
        rows["target_margin_delta"].to_numpy(dtype=float),
        np.nan,
    )

    parts: list[pd.DataFrame] = []
    eligibility_dimensions = set(
        map(str, expcfg.PROJECTION_DOMAIN_ELIGIBILITY_DIMENSIONS)
    )
    for dimension in dimensions:
        column = dimension_columns[dimension]
        dimension_rows = rows.copy()
        dimension_rows["stratum_type"] = dimension
        dimension_rows["stratum_value"] = (
            "all"
            if column is None
            else dimension_rows[column].astype("string").fillna("<missing>")
        )
        dimension_rows["__diagnostic_fold_id"] = (
            pd.to_numeric(dimension_rows["fold_id"], errors="raise").astype(int)
            if dimension == "fold"
            else -1
        )
        group_columns = [
            *identifiers,
            "__diagnostic_fold_id",
            "stratum_type",
            "stratum_value",
        ]
        skeleton = dimension_rows[group_columns].drop_duplicates()
        population = (
            dimension_rows.loc[dimension_rows["truth"].astype(bool)]
            if dimension in eligibility_dimensions
            else dimension_rows
        )
        aggregate = population.groupby(
            group_columns,
            dropna=False,
            sort=False,
            as_index=False,
        ).agg(
            n_observations=("truth", "size"),
            n_target=("truth", "sum"),
            __pc1_shift=("pc1_z", _mean_preserving_unbounded),
            __pc2_shift=("pc2_z", _mean_preserving_unbounded),
            t2_standardized_shift=("t2_z", _mean_preserving_unbounded),
            q_standardized_shift=("q_z", _mean_preserving_unbounded),
            rule_limit_standardized_shift=(
                "rule_limit_z",
                _mean_preserving_unbounded,
            ),
            normalized_ratio_standardized_shift=(
                "ratio_z",
                _mean_preserving_unbounded,
            ),
            simca_margin_standardized_shift=(
                "margin_z",
                _mean_preserving_unbounded,
            ),
            out_of_domain_rate=("__out_of_domain", "mean"),
            target_rejection_rate=("__target_rejected", "mean"),
            target_margin_displacement=(
                "__target_margin_delta",
                _mean_preserving_unbounded,
            ),
        )
        part = skeleton.merge(
            aggregate,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
        part[["n_observations", "n_target"]] = part[
            ["n_observations", "n_target"]
        ].fillna(0).astype(int)
        pc_values = part[["__pc1_shift", "__pc2_shift"]].to_numpy(dtype=float)
        part["pca_centroid_shift"] = np.where(
            np.isnan(pc_values).any(axis=1),
            np.nan,
            np.hypot(pc_values[:, 0], pc_values[:, 1]),
        )
        supported = part["n_observations"].ge(int(min_stratum_n))
        eligibility_population = dimension in eligibility_dimensions
        part["diagnostic_status"] = np.where(
            supported,
            "ok",
            (
                "insufficient_target_support"
                if eligibility_population
                else "descriptive_small_n"
            ),
        )
        part["fold_id"] = part.pop("__diagnostic_fold_id").astype(int)
        part["protocol_version"] = expcfg.PROTOCOL_VERSION
        part["protocol_hash"] = str(protocol_hash)
        parts.append(part)

    result = pd.concat(parts, ignore_index=True, sort=False)
    result = result.reindex(columns=expcfg.PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS)
    diagnostic_keys = [
        "model_id",
        "random_state",
        "fold_id",
        "stratum_type",
        "stratum_value",
    ]
    if result.duplicated(diagnostic_keys).any():
        raise RuntimeError("Projection diagnostic natural keys are not unique.")
    return result.sort_values(
        ["track_id", "model_id", "random_state", "stratum_type", "stratum_value"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_projection_eligibility(
    diagnostics: pd.DataFrame,
    selected_executions: pd.DataFrame,
    *,
    protocol_hash: str,
    expected_track_ids: Sequence[str] | None = None,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Assign one auditable status per track from selected natural run keys."""
    thresholds = dict(
        expcfg.PROJECTION_DOMAIN_ELIGIBILITY_THRESHOLDS
        if thresholds is None
        else thresholds
    )
    required = set(expcfg.PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS)
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise KeyError(f"Missing projection diagnostics: {missing}")
    mapping_columns = ["model_id", "random_state", "track_id"]
    missing_mapping = sorted(set(mapping_columns) - set(selected_executions.columns))
    if missing_mapping:
        raise KeyError(f"Missing selected-execution columns: {missing_mapping}")
    mapping = selected_executions[mapping_columns].drop_duplicates()
    if mapping.duplicated(["model_id", "random_state"]).any():
        raise RuntimeError("Selected execution keys must be unique.")
    tracks = (
        list(dict.fromkeys(mapping["track_id"].astype(str)))
        if expected_track_ids is None
        else list(dict.fromkeys(map(str, expected_track_ids)))
    )
    unknown_observed = sorted(
        set(mapping["track_id"].astype(str)) - set(tracks)
    )
    if unknown_observed:
        raise RuntimeError(
            f"Selected executions contain tracks outside the contract: {unknown_observed}"
        )
    observed_pairs = diagnostics[
        ["model_id", "random_state"]
    ].drop_duplicates()
    missing_pairs = mapping.merge(
        observed_pairs,
        on=["model_id", "random_state"],
        how="left",
        indicator=True,
    )
    missing_pairs = missing_pairs.loc[missing_pairs["_merge"].eq("left_only")]
    if not missing_pairs.empty:
        raise RuntimeError(
            "A selected execution has no projection diagnostic: "
            f"{missing_pairs.drop(columns='_merge').to_dict('records')[:10]}"
        )

    eligibility = diagnostics.loc[
        diagnostics["stratum_type"].isin(
            expcfg.PROJECTION_DOMAIN_ELIGIBILITY_DIMENSIONS
        )
    ].copy()
    shift_columns = [
        "pca_centroid_shift",
        "t2_standardized_shift",
        "q_standardized_shift",
        "rule_limit_standardized_shift",
        "normalized_ratio_standardized_shift",
        "simca_margin_standardized_shift",
    ]
    rows_out: list[dict] = []
    for track_id in tracks:
        track_mapping = mapping.loc[mapping["track_id"].astype(str).eq(track_id)]
        if track_mapping.empty:
            rows_out.append(
                {
                    "track_id": track_id,
                    "n_selected_models": 0,
                    "n_selected_runs": 0,
                    "n_diagnostics": 0,
                    "max_abs_standardized_shift": np.nan,
                    "max_out_of_domain_rate": np.nan,
                    "max_target_rejection_rate": np.nan,
                    "eligibility_status": (
                        "unsupported_internal_calibration"
                    ),
                    "eligibility_reason": "no_selected_model_in_03b",
                    "rule_version": (
                        expcfg.PROJECTION_DOMAIN_AUDIT_RULE_VERSION
                    ),
                    "thresholds_json": json.dumps(
                        thresholds, sort_keys=True, separators=(",", ":")
                    ),
                    "protocol_version": expcfg.PROTOCOL_VERSION,
                    "protocol_hash": str(protocol_hash),
                }
            )
            continue
        group_all = eligibility.loc[
            eligibility["track_id"].astype(str).eq(track_id)
        ]
        if group_all.empty:
            raise RuntimeError(f"Track {track_id} has no eligibility diagnostic.")
        incomplete_target_support = not group_all["diagnostic_status"].eq("ok").all()
        group = group_all.loc[group_all["diagnostic_status"].eq("ok")]
        shift_values = group[shift_columns].apply(pd.to_numeric, errors="coerce")
        absolute_shifts = np.abs(shift_values.to_numpy(dtype=float))
        max_shift = (
            np.nan
            if not absolute_shifts.size or np.isnan(absolute_shifts).any()
            else float(absolute_shifts.max())
        )
        ood_values = pd.to_numeric(
            group["out_of_domain_rate"], errors="coerce"
        ).to_numpy(dtype=float)
        rejection_values = pd.to_numeric(
            group["target_rejection_rate"], errors="coerce"
        ).to_numpy(dtype=float)
        max_ood = (
            float(ood_values.max())
            if ood_values.size and np.isfinite(ood_values).all()
            else np.nan
        )
        max_reject = (
            float(rejection_values.max())
            if rejection_values.size and np.isfinite(rejection_values).all()
            else np.nan
        )
        unsupported_reasons = (
            ["insufficient_target_support"]
            if incomplete_target_support
            else []
        )
        warning_reasons = []
        for value, stem in (
            (max_shift, "standardized_shift"),
            (max_ood, "out_of_domain_rate"),
            (max_reject, "target_rejection_rate"),
        ):
            unsupported_key = f"unsupported_{'max_abs_' if stem == 'standardized_shift' else ''}{stem}"
            warning_key = f"warning_{'max_abs_' if stem == 'standardized_shift' else ''}{stem}"
            if not np.isfinite(value):
                unsupported_reasons.append(f"{stem}_not_finite")
            elif value > float(thresholds[unsupported_key]):
                unsupported_reasons.append(stem)
            elif value > float(thresholds[warning_key]):
                warning_reasons.append(stem)
        if unsupported_reasons:
            status = "unsupported_domain_shift"
            reasons = unsupported_reasons
        elif warning_reasons:
            status = "eligible_with_warning"
            reasons = warning_reasons
        else:
            status = "eligible"
            reasons = ["all_predeclared_limits_satisfied"]
        rows_out.append(
            {
                "track_id": track_id,
                "n_selected_models": int(track_mapping["model_id"].nunique()),
                "n_selected_runs": int(len(track_mapping)),
                "n_diagnostics": int(len(group_all)),
                "max_abs_standardized_shift": max_shift,
                "max_out_of_domain_rate": max_ood,
                "max_target_rejection_rate": max_reject,
                "eligibility_status": status,
                "eligibility_reason": ";".join(reasons),
                "rule_version": expcfg.PROJECTION_DOMAIN_AUDIT_RULE_VERSION,
                "thresholds_json": json.dumps(
                    thresholds, sort_keys=True, separators=(",", ":")
                ),
                "protocol_version": expcfg.PROTOCOL_VERSION,
                "protocol_hash": str(protocol_hash),
            }
        )
    return pd.DataFrame(rows_out).reindex(
        columns=expcfg.PROJECTION_ELIGIBILITY_COLUMNS
    )


__all__ = [
    "summarize_projection_shift",
    "build_projection_shift_diagnostics",
    "build_projection_eligibility",
]
