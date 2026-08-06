"""Train-to-projection diagnostics shared by notebooks 03B and 03C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        "projection_config_id",
        "fit_config_id",
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
    calibration_domain: pd.DataFrame,
    projection_shift: pd.DataFrame,
    *,
    object_db: dict | None,
    border_width: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_columns = [
        "evaluation_track",
        "track_id",
        "projection_config_id",
        "fit_config_id",
        "projection_level",
        "projection_matrix_method",
    ]
    missing = sorted(set(mapping_columns) - set(calibration_domain.columns))
    if missing:
        raise KeyError(f"Missing calibration-domain columns: {missing}")
    mapping = calibration_domain[mapping_columns].drop_duplicates()
    duplicated = mapping.duplicated(
        ["evaluation_track", "projection_config_id"], keep=False
    )
    if duplicated.any():
        raise RuntimeError(
            "A projection configuration has conflicting 03B domain metadata."
        )

    tables = []
    if oof_objects is not None and not oof_objects.empty:
        objects = oof_objects.copy()
        objects["border_core"] = "not_applicable"
        tables.append(objects)
    if oof_pixels is not None and not oof_pixels.empty:
        if object_db is None:
            raise ValueError("object_db is required for border/core diagnostics.")
        pixels = add_border_flags_to_pixel_df(
            oof_pixels,
            object_db,
            border_width=int(border_width),
        )
        pixels["border_core"] = np.select(
            [pixels["is_border_pixel"], pixels["is_core_pixel"]],
            ["border", "core"],
            default="unresolved",
        )
        tables.append(pixels)
    if not tables:
        raise RuntimeError("03C received no OOF prediction row from 03B.")
    oof = pd.concat(tables, ignore_index=True, sort=False)
    oof = oof.merge(
        mapping,
        on=[
            "projection_config_id",
            "fit_config_id",
            "projection_level",
            "projection_matrix_method",
        ],
        how="inner",
        validate="many_to_many",
    )
    if oof.empty:
        raise RuntimeError(
            "No calibrated 03B projection has matching OOF predictions."
        )
    reference_keys = ["projection_config_id", "fit_config_id", "fold_id"]
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
            "projection_shift.parquet predates the 03C contract; rerun 03B. "
            f"Missing columns: {missing_reference}"
        )
    references = projection_shift[
        reference_keys + reference_numeric_columns
    ].drop_duplicates()
    conflicting_references = references.duplicated(reference_keys, keep=False)
    if conflicting_references.any():
        raise RuntimeError(
            "Conflicting train references exist for one projection/fold crossing."
        )
    out = oof.merge(
        references,
        on=["projection_config_id", "fit_config_id", "fold_id"],
        how="left",
        validate="many_to_one",
    )
    reference_values = out[reference_numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(reference_values.to_numpy(dtype=float)).all():
        raise RuntimeError("At least one OOF crossing has a missing/non-finite train reference.")
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
        out[output] = _standardize_against_reference(
            out[value], out[mean], out[scale]
        )
    out["target_margin_delta"] = (
        pd.to_numeric(out["simca_margin"], errors="coerce")
        - pd.to_numeric(out["train_margin_mean"], errors="coerce")
    )
    return out, mapping


def build_projection_shift_diagnostics(
    oof_objects: pd.DataFrame,
    oof_pixels: pd.DataFrame,
    calibration_domain: pd.DataFrame,
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
    """Build compact, explainable task-25 diagnostics for every crossing."""
    rows, _ = _projection_rows_with_references(
        oof_objects,
        oof_pixels,
        calibration_domain,
        projection_shift,
        object_db=object_db,
        border_width=int(border_width),
    )
    batches = set(pd.to_numeric(rows["batch"], errors="raise").astype(int))
    if not batches.issubset(set(map(int, allowed_batches))):
        raise RuntimeError(f"Forbidden batch in 03C domain audit: {sorted(batches)}")

    identifiers = [
        "evaluation_track",
        "track_id",
        "projection_config_id",
        "fit_config_id",
        "projection_level",
        "projection_matrix_method",
    ]
    dimensions: list[tuple[str, str | None]] = [
        ("overall", None),
        ("fold", "fold_id"),
        ("size_bin", "size_bin"),
        ("border_core", "border_core"),
        ("truth_class", "truth_class"),
        ("source_image", "source_image"),
    ]
    diagnostics: list[dict] = []
    for dimension, column in dimensions:
        group_columns = identifiers + ([] if column is None else [column])
        for key, group in rows.groupby(group_columns, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(group_columns, key))
            # Train references describe the SIMCA target class. Eligibility
            # therefore compares target OOF projections with target training;
            # including intentional non-target rejections would mechanically
            # make every useful classifier look out of domain. Other strata
            # remain fully descriptive, including truth_class=non_target.
            eligibility_population = dimension in set(
                expcfg.PROJECTION_DOMAIN_ELIGIBILITY_DIMENSIONS
            )
            target_all = group["truth"].to_numpy(dtype=bool)
            diagnostic_group = (
                group.loc[target_all] if eligibility_population else group
            )
            target = diagnostic_group["truth"].to_numpy(dtype=bool)
            pc_means = np.asarray(
                [
                    _mean_preserving_unbounded(diagnostic_group["pc1_z"]),
                    _mean_preserving_unbounded(diagnostic_group["pc2_z"]),
                ],
                dtype=float,
            )
            diagnostics.append(
                {
                    **{name: base[name] for name in identifiers},
                    "fold_id": int(base["fold_id"]) if dimension == "fold" else -1,
                    "stratum_type": dimension,
                    "stratum_value": "all" if column is None else str(base[column]),
                    "n_observations": int(len(diagnostic_group)),
                    "n_target": int(target.sum()),
                    "pca_centroid_shift": _pca_shift_norm(pc_means),
                    "t2_standardized_shift": _mean_preserving_unbounded(
                        diagnostic_group["t2_z"]
                    ),
                    "q_standardized_shift": _mean_preserving_unbounded(
                        diagnostic_group["q_z"]
                    ),
                    "rule_limit_standardized_shift": _mean_preserving_unbounded(
                        diagnostic_group["rule_limit_z"]
                    ),
                    "normalized_ratio_standardized_shift": _mean_preserving_unbounded(
                        diagnostic_group["ratio_z"]
                    ),
                    "simca_margin_standardized_shift": _mean_preserving_unbounded(
                        diagnostic_group["margin_z"]
                    ),
                    "out_of_domain_rate": float(
                        pd.to_numeric(
                            diagnostic_group["normalized_ratio"], errors="coerce"
                        )
                        .ge(1.0)
                        .mean()
                    ) if len(diagnostic_group) else np.nan,
                    "target_rejection_rate": (
                        float(
                            diagnostic_group.loc[
                                target, "simca_margin"
                            ].lt(0.0).mean()
                        )
                        if target.any()
                        else np.nan
                    ),
                    "target_margin_displacement": (
                        _mean_preserving_unbounded(
                            diagnostic_group.loc[target, "target_margin_delta"]
                        )
                        if target.any()
                        else np.nan
                    ),
                    "diagnostic_status": (
                        "ok"
                        if len(diagnostic_group) >= int(min_stratum_n)
                        else "missing_target_support"
                        if eligibility_population and not len(diagnostic_group)
                        else "descriptive_small_n"
                    ),
                    "protocol_version": expcfg.PROTOCOL_VERSION,
                    "protocol_hash": str(protocol_hash),
                }
            )
    result = pd.DataFrame(diagnostics)
    return result.reindex(columns=expcfg.PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS)


def build_projection_eligibility(
    diagnostics: pd.DataFrame,
    calibration_domain: pd.DataFrame,
    *,
    protocol_hash: str,
    expected_tracks: Sequence[str] | None = None,
    unsupported_tracks: Mapping[str, str] | None = None,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Assign one explicit status per calibrated or unsupported track."""
    unsupported_tracks = {
        str(track): str(reason)
        for track, reason in dict(unsupported_tracks or {}).items()
    }
    thresholds = dict(
        expcfg.PROJECTION_DOMAIN_ELIGIBILITY_THRESHOLDS
        if thresholds is None
        else thresholds
    )
    required = set(expcfg.PROJECTION_SHIFT_DIAGNOSTIC_COLUMNS)
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise KeyError(f"Missing projection diagnostics: {missing}")
    mapping = calibration_domain[
        ["evaluation_track", "track_id", "projection_config_id"]
    ].drop_duplicates()
    tracks = sorted(
        set(mapping["evaluation_track"].astype(str))
        if expected_tracks is None
        else set(map(str, expected_tracks))
    )
    unknown_unsupported = sorted(set(unsupported_tracks) - set(tracks))
    if unknown_unsupported:
        raise ValueError(
            "Unsupported tracks are absent from the expected track contract: "
            f"{unknown_unsupported}"
        )
    missing_tracks = sorted(
        set(tracks)
        - set(diagnostics["evaluation_track"].astype(str))
        - set(unsupported_tracks)
    )
    if missing_tracks:
        raise RuntimeError(
            "A track cannot be accepted without a domain diagnostic: "
            f"{missing_tracks}"
        )
    observed_pairs = diagnostics[
        ["evaluation_track", "projection_config_id"]
    ].drop_duplicates()
    missing_pairs = mapping.merge(
        observed_pairs,
        on=["evaluation_track", "projection_config_id"],
        how="left",
        indicator=True,
    )
    missing_pairs = missing_pairs.loc[missing_pairs["_merge"].eq("left_only")]
    if not missing_pairs.empty:
        raise RuntimeError(
            "A calibrated crossing has no domain diagnostic: "
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
    rows_out = []
    for track in tracks:
        if track in unsupported_tracks:
            rows_out.append(
                {
                    "evaluation_track": str(track),
                    "track_id": str(
                        expcfg.SIMCA_EVALUATION_TRACK_IDS[str(track)]
                    ),
                    "n_projection_configurations": 0,
                    "n_diagnostics": 0,
                    "max_abs_standardized_shift": np.nan,
                    "max_out_of_domain_rate": np.nan,
                    "max_target_rejection_rate": np.nan,
                    "eligibility_status": (
                        "unsupported_internal_calibration"
                    ),
                    "eligibility_reason": unsupported_tracks[track],
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
            eligibility["evaluation_track"].astype(str).eq(str(track))
        ]
        if group_all.empty:
            raise RuntimeError(f"Track {track} has no eligibility diagnostic.")
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
        track_mapping = mapping.loc[
            mapping["evaluation_track"].astype(str).eq(str(track))
        ]
        rows_out.append(
            {
                "evaluation_track": str(track),
                "track_id": str(track_mapping["track_id"].iloc[0]),
                "n_projection_configurations": int(
                    track_mapping["projection_config_id"].nunique()
                ),
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
