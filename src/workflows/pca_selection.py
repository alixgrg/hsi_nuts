"""Robust preprocessing-level Pareto selection for notebook 03.

Candidate PCA fits are reviewed individually, but the scientific decision unit
is ``(matrix_family, preprocessing)``. Matrix construction variants are treated
as robustness conditions: every expected variant must be admissible, and Pareto
objectives use the least favourable value observed across variants. No weighted
score, relative-quantile filter, projection target, or diversity filter is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.protocol_governance import sha256_file, sha256_payload, make_selection_id


_PCA_SCIENTIFIC_METRICS = (
    "n_observations",
    "n_bands",
    "ncomp_90",
    "ncomp_95",
    "ncomp_99",
    "centroid_distance_pc1_pc2_pc3",
    "mahalanobis_pc1_pc2_pc3",
    "class_trace_ratio",
    "within_class_trace",
    "mean_distance_to_class_centroid",
    "q95_distance_to_class_centroid",
    "batch_trace_ratio",
    "mean_batch_centroid_shift_norm",
    "max_batch_centroid_shift_norm",
    "object_class_trace_ratio",
    "object_batch_trace_ratio",
    "mean_intra_object_trace",
    "object_over_intra_ratio",
    "train_q_mean",
    "train_q_q95",
    "projection_q_mean",
    "projection_q_q95",
    "projection_q_deviation",
    "mean_train_projection_shift_norm",
    "loading_abs_correlation_mean",
    "loading_angle_mean_deg",
    "group_fold_subspace_instability",
    "seed_subspace_instability",
    "bootstrap_subspace_instability",
    "group_fold_projection_shift_std",
    "score_stability_std",
    "instability_metric",
)


@dataclass(frozen=True)
class PCASelectionProfile:
    """Minimal family-specific preprocessing Pareto objectives."""

    maximize_metrics: tuple[str, ...] = ()
    minimize_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PCASelectionConfig:
    """Candidate-review and preprocessing-level Pareto configuration."""

    profiles: Mapping[str, PCASelectionProfile] = field(
        default_factory=lambda: {
            family: PCASelectionProfile(
                maximize_metrics=tuple(
                    profile["maximize_metrics"]
                ),
                minimize_metrics=tuple(
                    profile["minimize_metrics"]
                ),
            )
            for family, profile
            in expcfg.PCA_SELECTION_PROFILES.items()
        }
    )
    family_col: str = "matrix_family"
    variant_col: str = "matrix_variant"
    matrix_method_col: str = "matrix_method"
    preprocessing_col: str = "preprocessing"
    preprocessing_steps_col: str = "preprocessing_steps"
    expected_families: tuple[str, ...] = tuple(
        expcfg.PCA_SELECTION_EXPECTED_FAMILIES
    )
    strict_variant_coverage: bool = (
        expcfg.PCA_SELECTION_STRICT_VARIANT_COVERAGE
    )
    max_preprocessings_per_family: int | None = (
        expcfg.MAX_PCA_PREPROCESSINGS_PER_FAMILY
    )


def make_pca_selection_config(
    **overrides,
) -> PCASelectionConfig:
    """Build a selection configuration while preserving the public API."""
    return PCASelectionConfig(**overrides)


DEFAULT_PCA_SELECTION_CONFIG = (
    make_pca_selection_config()
)


def _profile_for_family(
    family: str,
    config: PCASelectionConfig,
) -> PCASelectionProfile:
    try:
        return config.profiles[str(family)]
    except KeyError as exc:
        raise KeyError(f"No PCA selection profile for family {family!r}.") from exc


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy()), index=series.index)


def build_pca_artifact_review_table(
    diagnostics: pd.DataFrame,
    *,
    run_fingerprint: str = "",
    review_pdf_path: str = "",
    review_pdf_sha256: str = "",
    page_by_candidate: Mapping[str, int] | None = None,
    existing_review: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a fingerprinted editable review row for every valid candidate."""
    id_cols = [
        column
        for column in (
            "candidate_id",
            "matrix_family",
            "matrix_variant",
            "matrix_method",
            "m",
            "balanced_pixel_strategy",
            "preprocessing",
        )
        if column in diagnostics
    ]
    out = diagnostics[id_cols].drop_duplicates().reset_index(drop=True)
    out["review_status"] = "pending"
    out["review_decision"] = ""
    out["artifact_codes"] = ""
    out["critical_artifact"] = False
    out["review_comment"] = ""
    out["reviewer"] = ""
    out["review_date"] = ""
    page_by_candidate = {} if page_by_candidate is None else page_by_candidate
    out["review_evidence"] = out.get(
        "candidate_id", pd.Series("", index=out.index)
    ).map(
        lambda candidate_id: (
            f"{review_pdf_path}#page={page_by_candidate.get(str(candidate_id), '')}"
            f";sha256={review_pdf_sha256}"
        )
    )
    out["run_fingerprint"] = str(run_fingerprint)

    if existing_review is not None and len(existing_review):
        if "candidate_id" not in out or "candidate_id" not in existing_review:
            return out
        previous = existing_review.loc[
            existing_review.get(
                "run_fingerprint",
                pd.Series("", index=existing_review.index),
            ).astype(str).eq(str(run_fingerprint))
        ].drop_duplicates("candidate_id", keep="last")
        editable = [
            column
            for column in (
                "review_status",
                "review_decision",
                "artifact_codes",
                "critical_artifact",
                "review_comment",
                "reviewer",
                "review_date",
            )
            if column in previous
        ]
        out = out.merge(
            previous[["candidate_id", *editable]],
            on="candidate_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_previous"),
        )
        for column in editable:
            previous_column = f"{column}_previous"
            if previous_column in out:
                out[column] = out[previous_column].where(
                    out[previous_column].notna(), out[column]
                )
                out = out.drop(columns=previous_column)
    return out


def apply_pca_artifact_review_decisions(
    review: pd.DataFrame,
    *,
    decision_groups: Sequence[Mapping[str, Any]],
    reviewed_pdf_sha256: str,
    reviewer: str,
    review_date: str,
    default_decision: str = "accept",
    default_artifact_codes: str = "none",
    default_critical_artifact: bool = False,
    default_review_comment: str,
) -> pd.DataFrame:
    """Bind documented visual decisions to the current, byte-identical PDF.

    ``run_fingerprint`` deliberately remains the fingerprint generated by the
    current run.  Human decisions may cross a protocol-lock-only change solely
    when the generated PDF has exactly the reviewed SHA-256.  Any visual byte
    change therefore returns the workflow to a blocking review state.

    Candidate decisions are supplied by groups and applied with indexed column
    mappings, so the notebook does not duplicate row-wise mutation logic.
    """
    required_review_columns = {
        "candidate_id",
        "review_evidence",
        "run_fingerprint",
    }
    missing = sorted(required_review_columns.difference(review.columns))
    if missing:
        raise KeyError(f"PCA artifact review is missing columns: {missing}")

    pdf_sha256 = str(reviewed_pdf_sha256).strip().lower()
    if not pd.Series([pdf_sha256]).str.fullmatch(r"[0-9a-f]{64}").iloc[0]:
        raise ValueError("reviewed_pdf_sha256 must be a lowercase SHA-256.")
    evidence_sha256 = review["review_evidence"].astype(str).str.extract(
        r";sha256=([0-9a-f]{64})$",
        expand=False,
    )
    if evidence_sha256.isna().any() or not evidence_sha256.eq(pdf_sha256).all():
        observed = sorted(evidence_sha256.dropna().astype(str).unique())
        raise RuntimeError(
            "The generated PCA review PDF differs from the reviewed evidence: "
            f"expected={pdf_sha256}, observed={observed}. Re-review is required."
        )

    reviewer = str(reviewer).strip()
    review_date = str(review_date).strip()
    default_review_comment = str(default_review_comment).strip()
    if not reviewer or not review_date or not default_review_comment:
        raise ValueError(
            "reviewer, review_date and default_review_comment are required."
        )
    if str(default_decision) not in expcfg.PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS:
        raise ValueError(f"Invalid default PCA review decision: {default_decision!r}")

    required_group_columns = {
        "candidate_ids",
        "review_decision",
        "artifact_codes",
        "critical_artifact",
        "review_comment",
    }
    decision_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(decision_groups):
        missing_group = sorted(required_group_columns.difference(group))
        if missing_group:
            raise KeyError(
                f"PCA review group {group_index} is missing: {missing_group}"
            )
        candidate_ids = sorted(set(map(str, group["candidate_ids"])))
        if not candidate_ids:
            raise ValueError(f"PCA review group {group_index} has no candidates.")
        for candidate_id in candidate_ids:
            decision_rows.append(
                {
                    "candidate_id": candidate_id,
                    "review_decision": str(group["review_decision"]),
                    "artifact_codes": str(group["artifact_codes"]),
                    "critical_artifact": bool(group["critical_artifact"]),
                    "review_comment": str(group["review_comment"]).strip(),
                }
            )

    decisions = pd.DataFrame(
        decision_rows,
        columns=(
            "candidate_id",
            "review_decision",
            "artifact_codes",
            "critical_artifact",
            "review_comment",
        ),
    )
    if decisions["candidate_id"].duplicated().any():
        duplicated = sorted(
            decisions.loc[
                decisions["candidate_id"].duplicated(keep=False),
                "candidate_id",
            ].unique()
        )
        raise RuntimeError(
            "Candidates occur in more than one PCA review group: "
            f"{duplicated}"
        )
    invalid_decisions = ~decisions["review_decision"].isin(
        expcfg.PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS
    )
    if invalid_decisions.any():
        raise ValueError(
            "Invalid PCA review group decisions: "
            f"{sorted(decisions.loc[invalid_decisions, 'review_decision'].unique())}"
        )
    undocumented = decisions["review_comment"].eq("")
    if undocumented.any():
        raise ValueError("Every PCA review group must have a comment.")

    candidate_ids = review["candidate_id"].astype(str)
    unknown = sorted(set(decisions["candidate_id"]) - set(candidate_ids))
    if unknown:
        raise RuntimeError(
            "PCA review decisions do not match the current candidate universe: "
            f"{unknown}"
        )

    out = review.copy()
    out["review_status"] = expcfg.PCA_ARTIFACT_REVIEW_REQUIRED_STATUS
    out["review_decision"] = str(default_decision)
    out["artifact_codes"] = str(default_artifact_codes)
    out["critical_artifact"] = bool(default_critical_artifact)
    out["review_comment"] = default_review_comment
    out["reviewer"] = reviewer
    out["review_date"] = review_date

    overrides = decisions.set_index("candidate_id")
    for column in (
        "review_decision",
        "artifact_codes",
        "critical_artifact",
        "review_comment",
    ):
        mapped = candidate_ids.map(overrides[column])
        mask = mapped.notna()
        out.loc[mask, column] = mapped.loc[mask].to_numpy()

    fingerprints = out["run_fingerprint"].astype(str).unique()
    if len(fingerprints) != 1 or not str(fingerprints[0]).strip():
        raise RuntimeError("PCA review must contain one current run_fingerprint.")
    validate_pca_artifact_review(
        out,
        expected_candidate_ids=candidate_ids,
        expected_run_fingerprint=str(fingerprints[0]),
    )
    return out


def validate_pca_artifact_review(
    review: pd.DataFrame,
    *,
    expected_candidate_ids: Sequence[str] | None = None,
    expected_run_fingerprint: str | None = None,
) -> None:
    """Block pending, stale, undocumented or logically invalid reviews."""
    required = set(expcfg.PCA_ARTIFACT_REVIEW_COLUMNS)
    missing = sorted(required.difference(review.columns))
    if missing:
        raise KeyError(f"PCA artifact review is missing columns: {missing}")
    if review["candidate_id"].duplicated().any():
        raise RuntimeError("PCA artifact review contains duplicate candidates.")
    if expected_candidate_ids is not None:
        expected = set(map(str, expected_candidate_ids))
        observed = set(review["candidate_id"].astype(str))
        if observed != expected:
            raise RuntimeError(
                "PCA artifact review candidate mismatch: "
                f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
    review_complete = (
        review["review_status"]
        .fillna("")
        .astype(str)
        .eq(expcfg.PCA_ARTIFACT_REVIEW_REQUIRED_STATUS)
    )
    if not review_complete.all():
        raise RuntimeError(
            "PCA artifact review is pending for at least one candidate."
        )
    decisions = review["review_decision"].astype(str)
    invalid = ~decisions.isin(expcfg.PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS)
    if invalid.any():
        raise RuntimeError(
            "Invalid PCA review decisions: "
            f"{sorted(decisions.loc[invalid].unique())}"
        )
    if expected_run_fingerprint is not None and not review[
        "run_fingerprint"
    ].astype(str).eq(str(expected_run_fingerprint)).all():
        raise RuntimeError("PCA artifact review fingerprint is stale.")
    documented = review[
        ["review_comment", "reviewer", "review_date", "review_evidence"]
    ].fillna("").astype(str).apply(lambda column: column.str.strip().ne(""))
    if not documented.all(axis=1).all():
        raise RuntimeError("Every PCA review decision must be documented.")
    evidence = review["review_evidence"].astype(str)
    if not (
        evidence.str.contains(r"#page=\d+", regex=True)
        & evidence.str.contains(r";sha256=[0-9a-f]{64}$", regex=True)
    ).all():
        raise RuntimeError("PCA review evidence must contain PDF page and SHA-256.")
    critical = review["critical_artifact"].fillna(False).astype(bool)
    if (critical & decisions.ne("reject")).any():
        raise RuntimeError("A critical PCA artifact must be rejected.")


def _merge_artifact_review(
    diagnostics: pd.DataFrame,
    artifact_review: pd.DataFrame | None,
) -> pd.DataFrame:
    out = diagnostics.copy()
    if artifact_review is None:
        artifact_review = build_pca_artifact_review_table(out)
    keys = (
        ["candidate_id"]
        if "candidate_id" in out and "candidate_id" in artifact_review
        else [
            column
            for column in (
                "matrix_family",
                "matrix_variant",
                "matrix_method",
                "m",
                "balanced_pixel_strategy",
                "preprocessing",
            )
            if column in out and column in artifact_review
        ]
    )
    if not keys:
        raise KeyError("PCA artifact review has no identifier columns in common.")
    review_cols = [
        *keys,
        *[column for column in expcfg.PCA_ARTIFACT_COLUMNS if column in artifact_review],
        *[
            column
            for column in (
                "review_status",
                "review_comment",
                *expcfg.PCA_REVIEW_METADATA_COLUMNS,
            )
            if column in artifact_review
        ],
    ]
    out = out.drop(
        columns=[
            column
            for column in (
                *expcfg.PCA_ARTIFACT_COLUMNS,
                "review_status",
                "review_comment",
                *expcfg.PCA_REVIEW_METADATA_COLUMNS,
            )
            if column in out
        ],
        errors="ignore",
    )
    out = out.merge(
        artifact_review[review_cols].drop_duplicates(keys, keep="last"),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    for column in expcfg.PCA_ARTIFACT_COLUMNS:
        if column not in out:
            out[column] = False
        out[column] = out[column].fillna(False).astype(bool)
    if "review_status" not in out:
        out["review_status"] = "pending"
    else:
        out["review_status"] = out["review_status"].fillna("pending")
    if "review_comment" not in out:
        out["review_comment"] = ""
    else:
        out["review_comment"] = out["review_comment"].fillna("")
    for column in expcfg.PCA_REVIEW_METADATA_COLUMNS:
        if column not in out:
            out[column] = ""
        else:
            out[column] = out[column].fillna("")
    return out


def _pareto_dominance_matrix(
    df: pd.DataFrame,
    *,
    maximize_metrics: Sequence[str],
    minimize_metrics: Sequence[str],
) -> tuple[pd.Index, np.ndarray]:
    """Return valid row indices and competitor-by-candidate dominance."""
    metrics = [
        *maximize_metrics,
        *minimize_metrics,
    ]
    missing = [
        metric
        for metric in metrics
        if metric not in df
    ]
    if missing:
        raise KeyError(
            f"Missing Pareto metric columns: {missing}"
        )

    if df.empty:
        return (
            df.index[:0],
            np.zeros((0, 0), dtype=bool),
        )

    numeric = df.loc[:, metrics].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid_mask = np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all(axis=1)
    valid_index = df.index[valid_mask]

    values = numeric.loc[
        valid_index
    ].to_numpy(dtype=float)
    directions = np.asarray(
        [1.0] * len(maximize_metrics)
        + [-1.0] * len(minimize_metrics),
        dtype=float,
    )
    utilities = values * directions

    greater_or_equal = (
        utilities[:, None, :]
        >= utilities[None, :, :]
    )
    strictly_greater = (
        utilities[:, None, :]
        > utilities[None, :, :]
    )
    dominates = (
        greater_or_equal.all(axis=2)
        & strictly_greater.any(axis=2)
    )
    np.fill_diagonal(dominates, False)

    return valid_index, dominates


def select_pca_pareto_front(
    diagnostics: pd.DataFrame,
    maximize_metrics: Sequence[str],
    minimize_metrics: Sequence[str],
) -> pd.DataFrame:
    """Return rows not dominated across the requested objectives."""
    valid_index, dominates = _pareto_dominance_matrix(
        diagnostics,
        maximize_metrics=maximize_metrics,
        minimize_metrics=minimize_metrics,
    )
    front_mask = ~dominates.any(axis=0)
    return diagnostics.loc[valid_index[front_mask]].copy()


def _base_technical_validity(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return pre-review technical validity and exact failing flag names.

    Artifact review is intentionally excluded here. Candidates that fail a
    technical check are never sent to human review, so ``review_status=pending``
    must not be reported as an additional reason for their rejection.
    """
    flags: dict[str, pd.Series] = {}
    for column in expcfg.PCA_TECHNICAL_FLAG_COLUMNS:
        if column == "stability_valid" and column not in df:
            flags[column] = (
                _finite_numeric(df["instability_metric"])
                if "instability_metric" in df
                else pd.Series(False, index=df.index)
            )
        elif column in df:
            flags[column] = df[column].fillna(False).astype(bool)
        else:
            flags[column] = pd.Series(False, index=df.index)

    valid = pd.Series(True, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=object)
    for column, flag in flags.items():
        valid &= flag
        failed = ~flag
        reasons.loc[failed] = reasons.loc[failed].map(
            lambda value, name=column: f"{value};{name}" if value else name
        )
    return valid, reasons


def _technical_validity(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return final admissibility after technical and visual review."""
    base_valid, reasons = _base_technical_validity(df)

    review_complete = (
        df.get(
            "review_status",
            pd.Series("pending", index=df.index),
        )
        .fillna("")
        .astype(str)
        .eq(expcfg.PCA_ARTIFACT_REVIEW_REQUIRED_STATUS)
    )
    review_decision = (
        df.get(
            "review_decision",
            pd.Series("", index=df.index),
        )
        .fillna("")
        .astype(str)
    )
    valid_decision = review_decision.isin(
        expcfg.PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS
    )

    needs_review = base_valid
    review_accepted = (
        review_complete
        & valid_decision
        & review_decision.isin({"accept", "warning"})
    )
    valid = base_valid & (~needs_review | review_accepted)

    pending = needs_review & ~review_complete
    invalid_decision = (
        needs_review
        & review_complete
        & ~valid_decision
    )
    rejected = (
        needs_review
        & review_complete
        & review_decision.eq("reject")
    )
    critical = (
        needs_review
        & df.get(
            "critical_artifact",
            pd.Series(False, index=df.index),
        )
        .fillna(False)
        .astype(bool)
    )

    reason_updates = (
        (pending, "artifact_review_pending"),
        (invalid_decision, "artifact_review_invalid"),
        (rejected, "artifact_review_reject"),
        (critical, "critical_artifact"),
    )
    for mask, reason in reason_updates:
        reasons.loc[mask] = np.where(
            reasons.loc[mask].astype(str).ne(""),
            reasons.loc[mask].astype(str) + ";" + reason,
            reason,
        )

    valid &= ~critical
    return valid, reasons


def build_pca_selection_diagnostics(
    summary_df: pd.DataFrame,
    *,
    artifact_review_df: pd.DataFrame | None = None,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Attach the visual-review and technical status to every PCA candidate.

    This table deliberately performs no quantitative preselection. Candidate
    metrics are aggregated only afterwards at preprocessing level.
    """
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    required = [
        config.family_col,
        config.variant_col,
        config.matrix_method_col,
        config.preprocessing_col,
        config.preprocessing_steps_col,
    ]
    missing = [column for column in required if column not in summary_df]
    if missing:
        raise KeyError(f"Missing PCA diagnostic identifier columns: {missing}")

    out = _merge_artifact_review(summary_df, artifact_review_df)
    (
        out["technical_fit_valid"],
        out["technical_fit_blocking_reason"],
    ) = _base_technical_validity(out)
    out["technical_valid"], out["blocking_reason"] = _technical_validity(out)
    out["selection_status"] = np.where(
        out["technical_valid"],
        "candidate_admissible",
        "candidate_blocked",
    )
    out["selection_reason"] = np.where(
        out["technical_valid"],
        "candidate_admissible_for_strict_preprocessing_coverage",
        "blocked:" + out["blocking_reason"].astype(str),
    )
    return out


def _json_values(values: Sequence[object]) -> str:
    return json.dumps(sorted({str(value) for value in values}), ensure_ascii=False)


def _validate_pca_selection_unit_identity(
    diagnostics: pd.DataFrame,
    *,
    config: PCASelectionConfig,
) -> None:
    identity_columns = [
        config.family_col,
        config.preprocessing_col,
        config.preprocessing_steps_col,
        "sg_window_length",
        "sg_polyorder",
        "wavelength_axis_id",
    ]
    missing = [
        column
        for column in identity_columns
        if column not in diagnostics
    ]
    if missing:
        raise KeyError(
            "Missing PCA selection-unit identity columns: "
            f"{missing}"
        )

    counts = diagnostics.groupby(
        "selection_unit_id",
        dropna=False,
        sort=True,
    )[identity_columns].nunique(dropna=False)

    ambiguous = counts.gt(1)
    if ambiguous.any(axis=None):
        violations = [
            {
                "selection_unit_id": str(unit_id),
                "column": str(column),
                "n_unique": int(counts.loc[unit_id, column]),
            }
            for unit_id, column
            in ambiguous.stack()[
                lambda values: values
            ].index
        ]
        raise RuntimeError(
            "A selection_unit_id maps to inconsistent scientific "
            f"parameters: {violations[:20]}"
        )


def _aggregate_pca_family_diagnostics(
    family_df: pd.DataFrame,
    *,
    family: str,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    profile = _profile_for_family(family, config)
    maximize = tuple(profile.maximize_metrics)
    minimize = tuple(profile.minimize_metrics)
    metrics = (*maximize, *minimize)

    missing_metrics = [
        metric
        for metric in metrics
        if metric not in family_df
    ]
    if missing_metrics:
        raise KeyError(
            f"Missing Pareto source metrics for {family!r}: "
            f"{missing_metrics}"
        )

    unit_key = family_df["selection_unit_id"]
    grouped = family_df.groupby(
        "selection_unit_id",
        dropna=False,
        sort=True,
    )

    identity_columns = [
        config.family_col,
        config.preprocessing_col,
        config.preprocessing_steps_col,
        "sg_window_length",
        "sg_polyorder",
        "wavelength_axis_id",
    ]
    out = grouped[identity_columns].first()

    expected_variants = tuple(
        sorted(
            family_df[config.variant_col]
            .astype(str)
            .unique()
        )
    )
    expected_set = set(expected_variants)

    observed_variants = grouped[
        config.variant_col
    ].agg(
        lambda values: tuple(
            sorted(values.astype(str).unique())
        )
    )
    missing_variants = observed_variants.map(
        lambda values: sorted(
            expected_set.difference(values)
        )
    )
    extra_variants = observed_variants.map(
        lambda values: sorted(
            set(values).difference(expected_set)
        )
    )

    technical = (
        family_df["technical_valid"]
        .fillna(False)
        .astype(bool)
    )
    technical_grouped = technical.groupby(
        unit_key,
        dropna=False,
        sort=True,
    )

    review_decisions = (
        family_df.get(
            "review_decision",
            pd.Series("", index=family_df.index),
        )
        .fillna("")
        .astype(str)
    )
    review_grouped = review_decisions.groupby(
        unit_key,
        dropna=False,
        sort=True,
    )

    candidate_ids = family_df.get(
        "candidate_id",
        pd.Series(
            family_df.index.astype(str),
            index=family_df.index,
        ),
    )

    numeric = family_df.loc[:, metrics].apply(
        pd.to_numeric,
        errors="coerce",
    )
    numeric["selection_unit_id"] = unit_key.to_numpy()
    numeric_grouped = numeric.groupby(
        "selection_unit_id",
        dropna=False,
        sort=True,
    )

    minimum = numeric_grouped[list(metrics)].min()
    median = numeric_grouped[list(metrics)].median()
    maximum = numeric_grouped[list(metrics)].max()
    q25 = numeric_grouped[list(metrics)].quantile(0.25)
    q75 = numeric_grouped[list(metrics)].quantile(0.75)

    finite_rows = pd.Series(
        np.isfinite(
            numeric.loc[:, metrics].to_numpy(dtype=float)
        ).all(axis=1),
        index=family_df.index,
    )
    metrics_complete = finite_rows.groupby(
        unit_key,
        dropna=False,
        sort=True,
    ).all()

    out["n_candidates"] = grouped.size()
    out["n_expected_variants"] = len(expected_variants)
    out["n_observed_variants"] = observed_variants.map(len)
    out["expected_variants_json"] = _json_values(
        expected_variants
    )
    out["observed_variants_json"] = (
        observed_variants.map(_json_values)
    )
    out["missing_variants_json"] = (
        missing_variants.map(_json_values)
    )
    out["extra_variants_json"] = (
        extra_variants.map(_json_values)
    )
    out["candidate_ids_json"] = (
        candidate_ids.groupby(
            unit_key,
            dropna=False,
            sort=True,
        ).agg(_json_values)
    )

    out["n_accept"] = review_grouped.apply(
        lambda values: int(values.eq("accept").sum())
    )
    out["n_warning"] = review_grouped.apply(
        lambda values: int(values.eq("warning").sum())
    )
    out["n_reject"] = review_grouped.apply(
        lambda values: int(values.eq("reject").sum())
    )

    out["n_blocked_candidates"] = (
        (~technical).groupby(
            unit_key,
            dropna=False,
            sort=True,
        ).sum().astype(int)
    )
    out["coverage_complete"] = (
        missing_variants.map(len).eq(0)
        & extra_variants.map(len).eq(0)
    )
    out["all_candidates_admissible"] = (
        technical_grouped.all()
    )

    if config.strict_variant_coverage:
        out["strict_coverage_pass"] = (
            out["coverage_complete"]
            & out["all_candidates_admissible"]
        )
    else:
        out["strict_coverage_pass"] = (
            technical_grouped.any()
        )

    out["objective_metrics_complete"] = metrics_complete
    out["preprocessing_eligible"] = (
        out["strict_coverage_pass"]
        & out["objective_metrics_complete"]
    )
    out["pareto_front"] = False
    out["dominated_by"] = ""

    out["selection_status"] = np.select(
        [
            out["preprocessing_eligible"],
            out["strict_coverage_pass"],
        ],
        [
            "pareto_eligible",
            "ineligible_incomplete_metrics",
        ],
        default="ineligible_strict_coverage",
    )

    for metric in metrics:
        out[f"{metric}_min"] = minimum[metric]
        out[f"{metric}_median"] = median[metric]
        out[f"{metric}_max"] = maximum[metric]
        out[f"{metric}_iqr"] = q75[metric] - q25[metric]
        out[f"{metric}_worst"] = (
            minimum[metric]
            if metric in maximize
            else maximum[metric]
        )

    return out.reset_index()


def aggregate_pca_preprocessing_diagnostics(
    candidate_diagnostics_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Aggregate candidate diagnostics at selection-unit level."""
    config = (
        DEFAULT_PCA_SELECTION_CONFIG
        if config is None
        else config
    )

    required = [
        "selection_unit_id",
        config.family_col,
        config.variant_col,
        config.preprocessing_col,
        config.preprocessing_steps_col,
        "technical_valid",
    ]
    missing = [
        column
        for column in required
        if column not in candidate_diagnostics_df
    ]
    if missing:
        raise KeyError(
            f"Missing preprocessing aggregation columns: {missing}"
        )

    _validate_pca_selection_unit_identity(
        candidate_diagnostics_df,
        config=config,
    )

    frames = []
    for family, family_df in candidate_diagnostics_df.groupby(
        config.family_col,
        dropna=False,
        sort=True,
    ):
        frames.append(
            _aggregate_pca_family_diagnostics(
                family_df,
                family=str(family),
                config=config,
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _resolved_pca_technical_flags(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    flags = pd.DataFrame(index=diagnostics.index)

    for flag in expcfg.PCA_TECHNICAL_FLAG_COLUMNS:
        if flag in diagnostics:
            flags[flag] = (
                diagnostics[flag]
                .fillna(False)
                .astype(bool)
            )
        elif (
            flag == "stability_valid"
            and "instability_metric" in diagnostics
        ):
            values = pd.to_numeric(
                diagnostics["instability_metric"],
                errors="coerce",
            )
            flags[flag] = np.isfinite(
                values.to_numpy(dtype=float)
            )
        else:
            flags[flag] = False

    return flags


def build_pca_scoring_diagnostics(
    diagnostics_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
    *,
    preprocessing_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return long-form scientific, technical, review and selection diagnostics."""
    config = (
        DEFAULT_PCA_SELECTION_CONFIG
        if config is None
        else config
    )
    diagnostics = diagnostics_df.copy()

    if preprocessing_summary_df is not None:
        selection_columns = [
            "selection_unit_id",
            "selection_status",
            "selection_reason",
            "pareto_front",
        ]
        missing = [
            column
            for column in selection_columns
            if column not in preprocessing_summary_df
        ]
        if missing:
            raise KeyError(
                f"Missing preprocessing selection columns: {missing}"
            )

        diagnostics = diagnostics.drop(
            columns=[
                "selection_status",
                "selection_reason",
                "pareto_front",
            ],
            errors="ignore",
        ).merge(
            preprocessing_summary_df.loc[
                :,
                selection_columns,
            ],
            on="selection_unit_id",
            how="left",
            validate="many_to_one",
        )

    id_columns = list(
        expcfg.PCA_DIAGNOSTIC_ID_COLUMNS
    )
    for column in id_columns:
        if column not in diagnostics:
            diagnostics[column] = "not_applicable"

    if "selection_status" not in diagnostics:
        diagnostics["selection_status"] = "not_evaluated"
    else:
        diagnostics["selection_status"] = (
            diagnostics["selection_status"]
            .fillna("not_evaluated")
            .astype(str)
        )

    if "selection_reason" not in diagnostics:
        diagnostics["selection_reason"] = ""
    if "blocking_reason" not in diagnostics:
        diagnostics["blocking_reason"] = ""

    goal_rows = []
    for family, profile in config.profiles.items():
        goal_rows.extend(
            {
                config.family_col: str(family),
                "metric": metric,
                "pareto_goal": "maximize",
            }
            for metric in profile.maximize_metrics
        )
        goal_rows.extend(
            {
                config.family_col: str(family),
                "metric": metric,
                "pareto_goal": "minimize",
            }
            for metric in profile.minimize_metrics
        )
    goal_table = pd.DataFrame(
        goal_rows,
        columns=(
            config.family_col,
            "metric",
            "pareto_goal",
        ),
    )

    profile_metrics = tuple(
        dict.fromkeys(
            metric
            for profile in config.profiles.values()
            for metric in (
                *profile.maximize_metrics,
                *profile.minimize_metrics,
            )
        )
    )
    scientific_columns = [
        metric
        for metric in dict.fromkeys(
            (
                *_PCA_SCIENTIFIC_METRICS,
                *profile_metrics,
            )
        )
        if metric in diagnostics
    ]

    scientific = diagnostics.melt(
        id_vars=[
            *id_columns,
            "selection_status",
        ],
        value_vars=scientific_columns,
        var_name="metric",
        value_name="value",
    )
    if len(goal_table):
        scientific = scientific.merge(
            goal_table,
            on=[
                config.family_col,
                "metric",
            ],
            how="left",
            validate="many_to_one",
        )
    else:
        scientific["pareto_goal"] = ""

    scientific["pareto_goal"] = (
        scientific["pareto_goal"]
        .fillna("")
        .astype(str)
    )
    scientific["diagnostic_group"] = np.where(
        scientific["pareto_goal"].ne(""),
        "preprocessing_pareto_source_metric",
        "scientific_metric",
    )
    scientific["value"] = pd.to_numeric(
        scientific["value"],
        errors="coerce",
    )
    scientific["value"] = scientific["value"].where(
        np.isfinite(scientific["value"]),
        np.nan,
    )
    scientific["threshold"] = np.nan
    scientific["constraint"] = ""
    scientific["passed"] = pd.NA
    scientific["detail"] = ""

    technical_source = diagnostics.loc[
        :,
        [
            *id_columns,
            "selection_status",
            "blocking_reason",
        ],
    ].copy()
    technical_flags = _resolved_pca_technical_flags(
        diagnostics
    )
    technical_source = pd.concat(
        [
            technical_source,
            technical_flags,
        ],
        axis=1,
    )
    technical_source["technical_valid"] = (
        diagnostics.get(
            "technical_valid",
            pd.Series(False, index=diagnostics.index),
        )
        .fillna(False)
        .astype(bool)
    )

    technical = technical_source.melt(
        id_vars=[
            *id_columns,
            "selection_status",
            "blocking_reason",
        ],
        value_vars=[
            *expcfg.PCA_TECHNICAL_FLAG_COLUMNS,
            "technical_valid",
        ],
        var_name="metric",
        value_name="passed",
    )
    technical["passed"] = (
        technical["passed"]
        .fillna(False)
        .astype(bool)
    )
    technical["diagnostic_group"] = "technical_blocker"
    technical["value"] = technical["passed"].astype(float)
    technical["threshold"] = 1.0
    technical["constraint"] = "=="
    technical["pareto_goal"] = ""
    technical["detail"] = np.where(
        technical["metric"].eq("technical_valid"),
        technical["blocking_reason"].fillna("").astype(str),
        "",
    )
    technical = technical.drop(
        columns="blocking_reason"
    )

    artifact_source = diagnostics.loc[
        :,
        [
            *id_columns,
            "selection_status",
        ],
    ].copy()
    for column in expcfg.PCA_ARTIFACT_COLUMNS:
        artifact_source[column] = (
            diagnostics.get(
                column,
                pd.Series(False, index=diagnostics.index),
            )
            .fillna(False)
            .astype(bool)
        )

    review_status = diagnostics.get(
        "review_status",
        pd.Series("pending", index=diagnostics.index),
    ).fillna("pending").astype(str)
    review_decision = diagnostics.get(
        "review_decision",
        pd.Series("", index=diagnostics.index),
    ).fillna("").astype(str)
    review_comment = diagnostics.get(
        "review_comment",
        pd.Series("", index=diagnostics.index),
    ).fillna("").astype(str)

    artifact_source["_detail"] = (
        review_status
        + ": "
        + review_decision
        + "; "
        + review_comment
    ).str.strip()

    artifact = artifact_source.melt(
        id_vars=[
            *id_columns,
            "selection_status",
            "_detail",
        ],
        value_vars=list(expcfg.PCA_ARTIFACT_COLUMNS),
        var_name="metric",
        value_name="_artifact_present",
    )
    artifact["_artifact_present"] = (
        artifact["_artifact_present"]
        .fillna(False)
        .astype(bool)
    )
    artifact["diagnostic_group"] = "artifact_review"
    artifact["value"] = (
        artifact["_artifact_present"].astype(float)
    )
    artifact["threshold"] = 0.0
    artifact["constraint"] = "=="
    artifact["passed"] = ~artifact["_artifact_present"]
    artifact["pareto_goal"] = ""
    artifact["detail"] = artifact["_detail"]
    artifact = artifact.drop(
        columns=[
            "_detail",
            "_artifact_present",
        ]
    )

    selection = diagnostics.loc[
        :,
        [
            *id_columns,
            "selection_status",
            "selection_reason",
        ],
    ].copy()
    selection["diagnostic_group"] = "selection"
    selection["metric"] = (
        "candidate_retained_after_pareto"
    )
    selection["passed"] = (
        selection["selection_status"].eq("selected")
    )
    selection["value"] = selection["passed"].astype(float)
    selection["threshold"] = np.nan
    selection["constraint"] = ""
    selection["pareto_goal"] = ""
    selection["detail"] = (
        selection["selection_reason"]
        .fillna("")
        .astype(str)
    )
    selection = selection.drop(
        columns="selection_reason"
    )

    output = pd.concat(
        [
            scientific,
            technical,
            artifact,
            selection,
        ],
        ignore_index=True,
        sort=False,
    )
    return output.loc[:,list(expcfg.PCA_SCORING_DIAGNOSTIC_COLUMNS)]

def build_pca_selection_flow_tables(
    diagnostics_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize preprocessing retention and its first elimination stage."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    required = (
        config.family_col,
        config.preprocessing_col,
        config.preprocessing_steps_col,
        "strict_coverage_pass",
        "objective_metrics_complete",
        "pareto_front",
    )
    missing = [column for column in required if column not in diagnostics_df]
    if missing:
        raise KeyError(f"Missing PCA selection-flow columns: {missing}")

    stage_rows = []
    outcome_rows = []
    for family, family_df in diagnostics_df.groupby(
        config.family_col,
        dropna=False,
        sort=True,
    ):
        coverage_mask = family_df["strict_coverage_pass"].fillna(False).astype(bool)
        metrics_mask = (
            coverage_mask
            & family_df["objective_metrics_complete"].fillna(False).astype(bool)
        )
        pareto_mask = (
            metrics_mask
            & family_df["pareto_front"].fillna(False).astype(bool)
        )
        stages = (
            ("input", pd.Series(True, index=family_df.index)),
            ("strict_family_coverage", coverage_mask),
            ("complete_pareto_metrics", metrics_mask),
            ("pareto_front", pareto_mask),
        )
        previous_retained = len(family_df)
        for stage, mask in stages:
            retained = int(mask.sum())
            entering = (
                len(family_df)
                if stage == "input"
                else int(previous_retained)
            )
            eliminated = max(entering - retained, 0)
            stage_rows.append(
                {
                    config.family_col: family,
                    "stage": stage,
                    "n_entering": int(entering),
                    "n_retained": retained,
                    "n_eliminated": int(eliminated),
                    "retention_rate": (
                        float(retained / entering)
                        if entering
                        else np.nan
                    ),
                }
            )
            previous_retained = retained

        for index, preprocessing in family_df.iterrows():
            if not coverage_mask.loc[index]:
                first_failed_stage = "strict_family_coverage"
                reason = "missing_variant_or_blocked_candidate"
            elif not metrics_mask.loc[index]:
                first_failed_stage = "complete_pareto_metrics"
                reason = "non_finite_pareto_source_metric"
            elif not pareto_mask.loc[index]:
                first_failed_stage = "pareto_dominance"
                reason = str(preprocessing.get("selection_reason", "pareto_dominated"))
            else:
                first_failed_stage = "retained_after_pareto"
                reason = "strict_coverage_and_preprocessing_pareto_pass"
            outcome_rows.append(
                {
                    config.family_col: family,
                    config.preprocessing_col: preprocessing.get(
                        config.preprocessing_col
                    ),
                    config.preprocessing_steps_col: preprocessing.get(
                        config.preprocessing_steps_col
                    ),
                    "strict_coverage_pass": bool(coverage_mask.loc[index]),
                    "objective_metrics_complete": bool(metrics_mask.loc[index]),
                    "pareto_front": bool(pareto_mask.loc[index]),
                    "retained_after_pareto": bool(pareto_mask.loc[index]),
                    "first_failed_stage": first_failed_stage,
                    "elimination_reason": reason,
                }
            )

    return (
        pd.DataFrame(
            stage_rows,
            columns=(
                config.family_col,
                "stage",
                "n_entering",
                "n_retained",
                "n_eliminated",
                "retention_rate",
            ),
        ),
        pd.DataFrame(outcome_rows),
    )


def _new_selection_events(
    source: pd.DataFrame,
    *,
    substage: str,
    entity_type: str,
    entity_col: str,
    related_col: str | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)

    out["stage"] = expcfg.PCA_AUDIT_STAGE
    out["substage"] = str(substage)
    out["entity_type"] = str(entity_type)
    out["entity_id"] = (
        source[entity_col]
        .fillna("")
        .astype(str)
    )
    out["related_entity_id"] = (
        ""
        if related_col is None
        else source[related_col]
        .fillna("")
        .astype(str)
    )
    out["track_id"] = ""
    out["decision"] = ""
    out["reason_code"] = ""
    out["metric"] = ""
    out["observed_value"] = np.nan
    out["operator"] = ""
    out["reference_value"] = np.nan
    out["reference_source"] = ""
    out["mechanism"] = ""
    out["detail"] = ""

    return out


def _validate_pca_selection_audit_inputs(
    candidate_registry: pd.DataFrame,
    diagnostics: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
    *,
    config: PCASelectionConfig,
) -> None:
    contracts = {
        "candidate_registry_df": (
            candidate_registry,
            {
                "candidate_id",
                "selection_unit_id",
            },
        ),
        "candidate_diagnostics_df": (
            diagnostics,
            {
                "candidate_id",
                "selection_unit_id",
                "technical_fit_valid",
                "technical_valid",
                "blocking_reason",
            },
        ),
        "preprocessing_summary_df": (
            preprocessing_summary,
            {
                "selection_unit_id",
                config.family_col,
                config.preprocessing_col,
                "strict_coverage_pass",
                "objective_metrics_complete",
                "pareto_front",
                "selection_status",
                "n_expected_variants",
                "n_observed_variants",
                "n_blocked_candidates",
                "expected_variants_json",
                "observed_variants_json",
                "missing_variants_json",
                "dominated_by",
            },
        ),
    }

    for name, (frame, required) in contracts.items():
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(
                f"{name} is missing required columns: {missing}"
            )

    uniqueness_contracts = (
        (
            "candidate registry",
            candidate_registry,
            "candidate_id",
        ),
        (
            "candidate diagnostics",
            diagnostics,
            "candidate_id",
        ),
        (
            "preprocessing summary",
            preprocessing_summary,
            "selection_unit_id",
        ),
    )
    for name, frame, column in uniqueness_contracts:
        duplicated = frame[column].duplicated(keep=False)
        if duplicated.any():
            values = sorted(
                frame.loc[duplicated, column]
                .astype(str)
                .unique()
            )
            raise RuntimeError(
                f"{name} contains duplicate {column} values: "
                f"{values[:10]}"
            )

    registry_candidates = set(
        candidate_registry["candidate_id"].astype(str)
    )
    diagnostic_candidates = set(
        diagnostics["candidate_id"].astype(str)
    )
    if registry_candidates != diagnostic_candidates:
        raise RuntimeError(
            "Candidate universe mismatch between registry and "
            "diagnostics: "
            f"missing_in_diagnostics="
            f"{sorted(registry_candidates-diagnostic_candidates)}, "
            f"extra_in_diagnostics="
            f"{sorted(diagnostic_candidates-registry_candidates)}"
        )

    registry_units = set(
        candidate_registry["selection_unit_id"].astype(str)
    )
    summary_units = set(
        preprocessing_summary[
            "selection_unit_id"
        ].astype(str)
    )
    if registry_units != summary_units:
        raise RuntimeError(
            "PCA selection-unit universe mismatch: "
            f"missing_in_summary="
            f"{sorted(registry_units-summary_units)}, "
            f"extra_in_summary="
            f"{sorted(summary_units-registry_units)}"
        )

    registry_links = (
        candidate_registry[
            ["candidate_id", "selection_unit_id"]
        ]
        .astype(str)
        .set_index("candidate_id")["selection_unit_id"]
        .sort_index()
    )
    diagnostic_links = (
        diagnostics[
            ["candidate_id", "selection_unit_id"]
        ]
        .astype(str)
        .set_index("candidate_id")["selection_unit_id"]
        .sort_index()
    )
    if not registry_links.equals(diagnostic_links):
        raise RuntimeError(
            "candidate_id -> selection_unit_id mapping differs "
            "between registry and diagnostics."
        )


def _build_candidate_generation_events(
    candidate_registry: pd.DataFrame,
) -> pd.DataFrame:
    out = _new_selection_events(
        candidate_registry,
        substage="candidate_generation",
        entity_type="pca_candidate",
        entity_col="candidate_id",
        related_col="selection_unit_id",
    )
    out["decision"] = "entered"
    out["reason_code"] = "candidate_from_notebook02"
    out["mechanism"] = "candidate_generation"
    out["detail"] = (
        "PCA candidate generated from accepted notebook-02 "
        "matrix/preprocessing outputs."
    )
    return out


def _resolve_pca_technical_rule(
    flag: str,
) -> dict[str, object]:
    rule = dict(
        expcfg.PCA_TECHNICAL_AUDIT_FALLBACK_RULE
    )
    rule.update(
        expcfg.PCA_TECHNICAL_AUDIT_RULES.get(
            flag,
            {},
        )
    )
    rule.setdefault("metric", flag)
    rule.setdefault("reason_code_prefix", flag)

    if "reference_config" in rule:
        config_name = str(rule["reference_config"])
        if not hasattr(expcfg, config_name):
            raise AttributeError(
                f"Unknown PCA technical reference config: "
                f"{config_name!r}"
            )
        rule["reference_value"] = getattr(
            expcfg,
            config_name,
        )

    return rule


def _build_technical_check_events(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    id_columns = [
        "candidate_id",
        "selection_unit_id",
    ]
    resolved_flags = _resolved_pca_technical_flags(
        diagnostics
    )

    special_flags = tuple(
        flag
        for flag in expcfg.PCA_TECHNICAL_FLAG_COLUMNS
        if flag in expcfg.PCA_TECHNICAL_AUDIT_RULES
    )
    boolean_flags = tuple(
        flag
        for flag in expcfg.PCA_TECHNICAL_FLAG_COLUMNS
        if flag not in special_flags
    )

    frames = []

    if boolean_flags:
        flags = pd.concat(
            [
                diagnostics.loc[:, id_columns],
                resolved_flags.loc[:, boolean_flags],
            ],
            axis=1,
        ).melt(
            id_vars=id_columns,
            value_vars=list(boolean_flags),
            var_name="metric",
            value_name="passed",
        )
        flags["passed"] = (
            flags["passed"]
            .fillna(False)
            .astype(bool)
        )

        events = _new_selection_events(
            flags,
            substage="technical_check",
            entity_type="pca_candidate",
            entity_col="candidate_id",
            related_col="selection_unit_id",
        )
        events["decision"] = np.where(
            flags["passed"],
            "kept",
            "eliminated",
        )
        events["reason_code"] = (
            flags["metric"].astype(str)
            + np.where(
                flags["passed"],
                "_pass",
                "_fail",
            )
        )
        events["metric"] = flags["metric"].astype(str)
        events["observed_value"] = (
            flags["passed"].astype(float)
        )
        events["operator"] = "=="
        events["reference_value"] = 1.0
        events["reference_source"] = (
            expcfg.PCA_TECHNICAL_AUDIT_FALLBACK_RULE[
                "reference_source"
            ]
        )
        events["mechanism"] = (
            expcfg.PCA_TECHNICAL_AUDIT_FALLBACK_RULE[
                "mechanism"
            ]
        )
        frames.append(events)

    for flag in special_flags:
        rule = _resolve_pca_technical_rule(flag)
        source = diagnostics.loc[:, id_columns].copy()
        source["passed"] = resolved_flags[flag]

        events = _new_selection_events(
            source,
            substage="technical_check",
            entity_type="pca_candidate",
            entity_col="candidate_id",
            related_col="selection_unit_id",
        )
        passed = source["passed"].astype(bool)
        metric = str(rule["metric"])
        prefix = str(rule["reason_code_prefix"])

        observed = (
            pd.to_numeric(
                diagnostics[metric],
                errors="coerce",
            )
            if metric in diagnostics
            else pd.Series(
                np.nan,
                index=diagnostics.index,
            )
        )

        events["decision"] = np.where(
            passed,
            "kept",
            "eliminated",
        )
        events["reason_code"] = (
            prefix
            + np.where(
                passed,
                "_pass",
                "_fail",
            )
        )
        events["metric"] = metric
        events["observed_value"] = observed
        events["operator"] = str(rule["operator"])
        events["reference_value"] = float(
            rule["reference_value"]
        )
        events["reference_source"] = str(
            rule["reference_source"]
        )
        events["mechanism"] = str(rule["mechanism"])
        frames.append(events)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _build_technical_fit_events(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    out = _new_selection_events(
        diagnostics,
        substage="technical_fit_outcome",
        entity_type="pca_candidate",
        entity_col="candidate_id",
        related_col="selection_unit_id",
    )
    fit_valid = (
        diagnostics["technical_fit_valid"]
        .fillna(False)
        .astype(bool)
    )

    out["decision"] = np.where(
        fit_valid,
        "kept",
        "eliminated",
    )
    out["reason_code"] = np.where(
        fit_valid,
        "technical_fit_valid",
        "technical_fit_blocked",
    )
    out["metric"] = "technical_fit_valid"
    out["observed_value"] = fit_valid.astype(float)
    out["operator"] = "=="
    out["reference_value"] = 1.0
    out["reference_source"] = (
        "PCA pre-review technical contract"
    )
    out["mechanism"] = "technical_eligibility"
    out["detail"] = (
        diagnostics.get(
            "technical_fit_blocking_reason",
            pd.Series("", index=diagnostics.index),
        )
        .fillna("")
        .astype(str)
    )
    return out


def _build_artifact_review_events(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    out = _new_selection_events(
        diagnostics,
        substage="artifact_review",
        entity_type="pca_candidate",
        entity_col="candidate_id",
        related_col="selection_unit_id",
    )

    fit_valid = (
        diagnostics["technical_fit_valid"]
        .fillna(False)
        .astype(bool)
    )
    review_status = (
        diagnostics.get(
            "review_status",
            pd.Series("pending", index=diagnostics.index),
        )
        .fillna("pending")
        .astype(str)
    )
    review_decision = (
        diagnostics.get(
            "review_decision",
            pd.Series("", index=diagnostics.index),
        )
        .fillna("")
        .astype(str)
    )
    review_complete = review_status.eq(
        expcfg.PCA_ARTIFACT_REVIEW_REQUIRED_STATUS
    )
    valid_decision = review_decision.isin(
        expcfg.PCA_ARTIFACT_REVIEW_ALLOWED_DECISIONS
    )
    critical = (
        diagnostics.get(
            "critical_artifact",
            pd.Series(False, index=diagnostics.index),
        )
        .fillna(False)
        .astype(bool)
    )

    conditions = [
        ~fit_valid,
        fit_valid & ~review_complete,
        fit_valid & review_complete & ~valid_decision,
        fit_valid & critical,
        fit_valid & review_decision.eq("reject"),
        fit_valid & review_decision.eq("warning"),
    ]
    decisions = [
        "not_applicable",
        "eliminated",
        "eliminated",
        "eliminated",
        "eliminated",
        "warning",
    ]
    reasons = [
        "technical_failure_before_review",
        "artifact_review_pending",
        "artifact_review_invalid",
        "critical_artifact",
        "artifact_review_reject",
        "artifact_review_warning",
    ]

    out["decision"] = np.select(
        conditions,
        decisions,
        default="kept",
    )
    out["reason_code"] = np.select(
        conditions,
        reasons,
        default="artifact_review_accept",
    )
    out["metric"] = "critical_artifact"
    out["observed_value"] = (
        critical.astype(float).where(
            fit_valid,
            np.nan,
        )
    )
    out["operator"] = np.where(
        fit_valid,
        "==",
        "",
    )
    out["reference_value"] = (
        pd.Series(0.0, index=diagnostics.index)
        .where(fit_valid, np.nan)
    )
    out["reference_source"] = np.where(
        fit_valid,
        "PCA human artifact review protocol",
        "",
    )
    out["mechanism"] = "human_review"

    artifact_codes = diagnostics.get(
        "artifact_codes",
        pd.Series("", index=diagnostics.index),
    ).fillna("").astype(str)
    review_comment = diagnostics.get(
        "review_comment",
        pd.Series("", index=diagnostics.index),
    ).fillna("").astype(str)

    out["detail"] = [
        "; ".join(
            part
            for part in (
                f"review_status={status}",
                f"review_decision={decision}",
                (
                    f"artifact_codes={codes}"
                    if codes
                    else ""
                ),
                (
                    f"comment={comment}"
                    if comment
                    else ""
                ),
            )
            if part
        )
        for status, decision, codes, comment
        in zip(
            review_status,
            review_decision,
            artifact_codes,
            review_comment,
        )
    ]
    return out


def _build_candidate_admissibility_events(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    out = _new_selection_events(
        diagnostics,
        substage="candidate_admissibility",
        entity_type="pca_candidate",
        entity_col="candidate_id",
        related_col="selection_unit_id",
    )
    admissible = (
        diagnostics["technical_valid"]
        .fillna(False)
        .astype(bool)
    )

    out["decision"] = np.where(
        admissible,
        "kept",
        "eliminated",
    )
    out["reason_code"] = np.where(
        admissible,
        "candidate_admissible",
        "candidate_blocked",
    )
    out["metric"] = "technical_valid"
    out["observed_value"] = admissible.astype(float)
    out["operator"] = "=="
    out["reference_value"] = 1.0
    out["reference_source"] = (
        "PCA candidate admissibility contract"
    )
    out["mechanism"] = "eligibility"
    out["detail"] = (
        diagnostics["blocking_reason"]
        .fillna("")
        .astype(str)
    )
    return out


def _build_variant_coverage_events(
    preprocessing_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    out = _new_selection_events(
        preprocessing_summary,
        substage="strict_variant_coverage",
        entity_type="pca_preprocessing",
        entity_col="selection_unit_id",
    )
    strict_pass = (
        preprocessing_summary["strict_coverage_pass"]
        .fillna(False)
        .astype(bool)
    )
    n_expected = pd.to_numeric(
        preprocessing_summary["n_expected_variants"],
        errors="coerce",
    ).fillna(0).astype(int)
    n_observed = pd.to_numeric(
        preprocessing_summary["n_observed_variants"],
        errors="coerce",
    ).fillna(0).astype(int)
    n_blocked = pd.to_numeric(
        preprocessing_summary["n_blocked_candidates"],
        errors="coerce",
    ).fillna(0).astype(int)
    n_admissible = (
        n_observed - n_blocked
    ).clip(lower=0)

    blocked = diagnostics.loc[
        ~diagnostics["technical_valid"]
        .fillna(False)
        .astype(bool),
        [
            "candidate_id",
            "selection_unit_id",
            "blocking_reason",
        ],
    ].copy()
    blocked["_detail"] = (
        blocked["candidate_id"].astype(str)
        + ":"
        + blocked["blocking_reason"]
        .fillna("")
        .astype(str)
    )
    blocked_by_unit = (
        blocked.groupby(
            "selection_unit_id",
            dropna=False,
            sort=True,
        )["_detail"]
        .agg("|".join)
    )

    unit_ids = (
        preprocessing_summary["selection_unit_id"]
        .astype(str)
    )
    blocked_detail = unit_ids.map(
        blocked_by_unit.rename(
            index=lambda value: str(value)
        )
    ).fillna("")

    expected_text = (
        preprocessing_summary["expected_variants_json"]
        .fillna("")
        .astype(str)
    )
    observed_text = (
        preprocessing_summary["observed_variants_json"]
        .fillna("")
        .astype(str)
    )
    missing_text = (
        preprocessing_summary["missing_variants_json"]
        .fillna("")
        .astype(str)
    )

    out["decision"] = np.where(
        strict_pass,
        "kept",
        "eliminated",
    )
    out["reason_code"] = np.where(
        strict_pass,
        "complete_admissible_variant_coverage",
        "incomplete_or_blocked_variant_coverage",
    )
    out["metric"] = "n_admissible_variants"
    out["observed_value"] = n_admissible.astype(float)
    out["operator"] = "=="
    out["reference_value"] = n_expected.astype(float)
    out["reference_source"] = (
        "PCASelectionConfig.strict_variant_coverage"
    )
    out["mechanism"] = "hard_constraint"
    out["detail"] = [
        "; ".join(
            [
                f"expected_variants={expected}",
                f"observed_variants={observed}",
                f"missing_variants={missing}",
                *(
                    [f"blocked_candidates={blocked_values}"]
                    if blocked_values
                    else []
                ),
            ]
        )
        for expected, observed, missing, blocked_values
        in zip(
            expected_text,
            observed_text,
            missing_text,
            blocked_detail,
        )
    ]
    return out


def _build_pareto_metric_events(
    preprocessing_summary: pd.DataFrame,
    *,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    frames = []

    for family, family_df in preprocessing_summary.groupby(
        config.family_col,
        dropna=False,
        sort=True,
    ):
        profile = _profile_for_family(
            str(family),
            config,
        )
        metrics = (
            *profile.maximize_metrics,
            *profile.minimize_metrics,
        )
        source_columns = [
            f"{metric}_worst"
            for metric in metrics
        ]
        numeric = family_df.reindex(
            columns=source_columns
        ).apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite = np.isfinite(
            numeric.to_numpy(dtype=float)
        )
        incomplete = [
            ",".join(
                metric
                for metric, is_finite
                in zip(metrics, row)
                if not is_finite
            )
            for row in finite
        ]

        events = _new_selection_events(
            family_df,
            substage="pareto_metric_completeness",
            entity_type="pca_preprocessing",
            entity_col="selection_unit_id",
        )
        complete = (
            family_df["objective_metrics_complete"]
            .fillna(False)
            .astype(bool)
        )

        events["decision"] = np.where(
            complete,
            "kept",
            "eliminated",
        )
        events["reason_code"] = np.where(
            complete,
            "pareto_metrics_complete",
            "non_finite_pareto_source_metric",
        )
        events["metric"] = "objective_metrics_complete"
        events["observed_value"] = complete.astype(float)
        events["operator"] = "=="
        events["reference_value"] = 1.0
        events["reference_source"] = (
            "PCASelectionProfile objective contract"
        )
        events["mechanism"] = "hard_constraint"
        events["detail"] = np.where(
            complete,
            "",
            "non_finite_metrics="
            + pd.Series(
                incomplete,
                index=family_df.index,
            ),
        )
        frames.append(events)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _build_pareto_objective_events(
    preprocessing_summary: pd.DataFrame,
    *,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    frames = []

    for family, family_df in preprocessing_summary.groupby(
        config.family_col,
        dropna=False,
        sort=True,
    ):
        profile = _profile_for_family(
            str(family),
            config,
        )

        for goal, metrics in (
            ("maximize", profile.maximize_metrics),
            ("minimize", profile.minimize_metrics),
        ):
            if not metrics:
                continue

            source_columns = [
                f"{metric}_worst"
                for metric in metrics
            ]
            source = family_df[
                ["selection_unit_id", *source_columns]
            ].melt(
                id_vars="selection_unit_id",
                value_vars=source_columns,
                var_name="_source_metric",
                value_name="observed_value",
            )
            source["metric"] = source[
                "_source_metric"
            ].str.removesuffix("_worst")
            source["observed_value"] = pd.to_numeric(
                source["observed_value"],
                errors="coerce",
            )
            finite = np.isfinite(
                source["observed_value"].to_numpy(dtype=float)
            )

            events = _new_selection_events(
                source,
                substage="pareto_objective",
                entity_type="pca_preprocessing",
                entity_col="selection_unit_id",
            )
            events["decision"] = np.where(
                finite,
                "evaluated",
                "not_evaluated",
            )
            events["reason_code"] = np.where(
                finite,
                "pareto_objective",
                "pareto_objective_non_finite",
            )
            events["metric"] = source["metric"]
            events["observed_value"] = (
                source["observed_value"]
            )
            events["mechanism"] = f"pareto_{goal}"
            events["detail"] = (
                "worst_across_matrix_variants"
            )
            frames.append(events)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _build_pareto_dominance_events(
    preprocessing_summary: pd.DataFrame,
    *,
    config: PCASelectionConfig,
) -> pd.DataFrame:
    summary = preprocessing_summary.copy()
    summary["selection_unit_id"] = (
        summary["selection_unit_id"].astype(str)
    )
    summary_by_id = summary.set_index(
        "selection_unit_id",
        drop=False,
    )
    dominated = summary.loc[
        summary["strict_coverage_pass"]
        .fillna(False)
        .astype(bool)
        & summary["objective_metrics_complete"]
        .fillna(False)
        .astype(bool)
        & ~summary["pareto_front"]
        .fillna(False)
        .astype(bool)
    ].copy()
    if dominated.empty:
        return pd.DataFrame(
            columns=expcfg.SELECTION_AUDIT_COLUMNS
        )
    pairs = dominated[
        ["selection_unit_id", "dominated_by"]
    ].rename(
        columns={
            "selection_unit_id": "entity_id",
        }
    )
    pairs["related_entity_id"] = (
        pairs["dominated_by"]
        .fillna("")
        .astype(str)
        .str.split(";")
    )
    pairs = pairs.explode(
        "related_entity_id"
    )
    pairs["related_entity_id"] = (
        pairs["related_entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    pairs = pairs.loc[
        pairs["related_entity_id"].ne("")
    ].drop(
        columns="dominated_by"
    ).drop_duplicates()
    missing_sources = (
        set(dominated["selection_unit_id"])
        - set(pairs["entity_id"])
    )
    if missing_sources:
        raise RuntimeError(
            "Pareto-dominated preprocessings have no recorded "
            f"dominator: {sorted(missing_sources)}"
        )
    unknown = (
        set(pairs["related_entity_id"])
        - set(summary_by_id.index)
    )
    if unknown:
        raise RuntimeError(
            "Unknown Pareto dominator selection_unit_id values: "
            f"{sorted(unknown)}"
        )
    family_by_id = summary_by_id[
        config.family_col
    ].astype(str)
    pairs["_source_family"] = (
        pairs["entity_id"].map(family_by_id)
    )
    pairs["_dominator_family"] = (
        pairs["related_entity_id"].map(family_by_id)
    )
    cross_family = pairs[
        "_source_family"
    ].ne(pairs["_dominator_family"])
    if cross_family.any():
        raise RuntimeError(
            "Pareto dominance cannot cross PCA matrix families: "
            f"{pairs.loc[cross_family].to_dict('records')[:10]}"
        )
    frames = []
    for family, family_pairs in pairs.groupby(
        "_source_family",
        sort=True,
    ):
        profile = _profile_for_family(
            str(family),
            config,
        )
        for goal, metrics, operator in (
            ("maximize", profile.maximize_metrics, "<="),
            ("minimize", profile.minimize_metrics, ">="),
        ):
            for metric in metrics:
                column = f"{metric}_worst"
                values_by_id = pd.to_numeric(
                    summary_by_id[column],
                    errors="coerce",
                )
                source = family_pairs.copy()
                source["observed_value"] = (
                    source["entity_id"].map(values_by_id)
                )
                source["reference_value"] = (
                    source["related_entity_id"].map(
                        values_by_id
                    )
                )
                finite = (
                    np.isfinite(
                        source["observed_value"]
                        .to_numpy(dtype=float)
                    )
                    & np.isfinite(
                        source["reference_value"]
                        .to_numpy(dtype=float)
                    )
                )
                if not finite.all():
                    raise RuntimeError(
                        "Pareto dominance contains non-finite "
                        f"{metric!r} values."
                    )
                strict = (
                    source["reference_value"]
                    > source["observed_value"]
                    if goal == "maximize"
                    else source["reference_value"]
                    < source["observed_value"]
                )
                events = _new_selection_events(
                    source,
                    substage="pareto_dominance",
                    entity_type="pca_preprocessing",
                    entity_col="entity_id",
                    related_col="related_entity_id",
                )
                events["decision"] = "eliminated"
                events["reason_code"] = (
                    "pareto_dominated_by"
                )
                events["metric"] = metric
                events["observed_value"] = (
                    source["observed_value"]
                )
                events["operator"] = operator
                events["reference_value"] = (
                    source["reference_value"]
                )
                events["reference_source"] = (
                    source["related_entity_id"]
                )
                events["mechanism"] = "pareto"
                events["detail"] = (
                    f"goal={goal}; strict_improvement="
                    + strict.astype(str)
                )
                frames.append(events)
    return pd.concat(
        frames,
        ignore_index=True,
    )


def _build_pareto_selection_events(
    preprocessing_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = _new_selection_events(
        preprocessing_summary,
        substage="pareto_selection",
        entity_type="pca_preprocessing",
        entity_col="selection_unit_id",
    )
    strict_pass = (
        preprocessing_summary["strict_coverage_pass"]
        .fillna(False)
        .astype(bool)
    )
    metrics_complete = (
        preprocessing_summary["objective_metrics_complete"]
        .fillna(False)
        .astype(bool)
    )
    pareto_front = (
        preprocessing_summary["pareto_front"]
        .fillna(False)
        .astype(bool)
    )
    dominated_by = (
        preprocessing_summary["dominated_by"]
        .fillna("")
        .astype(str)
    )
    selection_reason = (
        preprocessing_summary.get(
            "selection_reason",
            pd.Series(
                "",
                index=preprocessing_summary.index,
            ),
        )
        .fillna("")
        .astype(str)
    )
    out["decision"] = np.where(
        pareto_front,
        "kept",
        "eliminated",
    )
    out["reason_code"] = np.select(
        [
            pareto_front,
            ~strict_pass,
            ~metrics_complete,
        ],
        [
            "pareto_non_dominated",
            "not_pareto_eligible_strict_coverage",
            "not_pareto_eligible_incomplete_metrics",
        ],
        default="pareto_dominated",
    )
    out["mechanism"] = "pareto"
    out["detail"] = np.select(
        [
            pareto_front,
            ~strict_pass,
            ~metrics_complete,
        ],
        [
            "",
            selection_reason,
            selection_reason,
        ],
        default="dominated_by=" + dominated_by,
    )
    return out


def _finalize_selection_audit(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    expected_columns = list(
        expcfg.SELECTION_AUDIT_COLUMNS
    )
    missing = [
        column
        for column in expected_columns
        if column not in audit
    ]
    if missing:
        raise RuntimeError(
            "PCA selection audit builder did not produce the "
            f"complete schema: {missing}"
        )
    out = audit.loc[:, expected_columns].copy()
    out["observed_value"] = pd.to_numeric(
        out["observed_value"],
        errors="coerce",
    ).astype(float)
    out["reference_value"] = pd.to_numeric(
        out["reference_value"],
        errors="coerce",
    ).astype(float)
    text_columns = [
        column
        for column in expected_columns
        if column not in {
            "observed_value",
            "reference_value",
        }
    ]
    for column in text_columns:
        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
        )
    audit_key = [
        "stage",
        "substage",
        "entity_type",
        "entity_id",
        "related_entity_id",
        "metric",
    ]
    duplicated = out.duplicated(audit_key, keep=False)
    if duplicated.any():
        raise RuntimeError(
            "Duplicate PCA audit events for their natural key."
        )
    return out.reset_index(drop=True)


def build_pca_selection_audit(
    candidate_registry_df: pd.DataFrame,
    candidate_diagnostics_df: pd.DataFrame,
    preprocessing_summary_df: pd.DataFrame,
    *,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Build notebook-03's normalized longitudinal selection audit."""
    config = (
        DEFAULT_PCA_SELECTION_CONFIG
        if config is None
        else config
    )

    _validate_pca_selection_audit_inputs(
        candidate_registry_df,
        candidate_diagnostics_df,
        preprocessing_summary_df,
        config=config,
    )

    frames = [
        _build_candidate_generation_events(
            candidate_registry_df
        ),
        _build_technical_check_events(
            candidate_diagnostics_df
        ),
        _build_technical_fit_events(
            candidate_diagnostics_df
        ),
        _build_artifact_review_events(
            candidate_diagnostics_df
        ),
        _build_candidate_admissibility_events(
            candidate_diagnostics_df
        ),
        _build_variant_coverage_events(
            preprocessing_summary_df,
            candidate_diagnostics_df,
        ),
        _build_pareto_metric_events(
            preprocessing_summary_df,
            config=config,
        ),
        _build_pareto_objective_events(
            preprocessing_summary_df,
            config=config,
        ),
        _build_pareto_dominance_events(
            preprocessing_summary_df,
            config=config,
        ),
        _build_pareto_selection_events(
            preprocessing_summary_df
        ),
    ]

    audit = pd.concat(
        frames,
        ignore_index=True,
    )
    return _finalize_selection_audit(audit)


def _canonical_frame_records(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[dict]:
    available = [column for column in columns if column in frame]
    ordered = frame.loc[:, available].copy()
    sort_columns = [
        column
        for column in ("candidate_id", "matrix_family", "preprocessing")
        if column in ordered
    ]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, kind="mergesort")
    records = []
    for values in ordered.itertuples(index=False, name=None):
        record = {}
        for key, value in zip(available, values):
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def build_pca_run_fingerprint(
    candidate_plan: pd.DataFrame,
    *,
    protocol_hash: str,
    input_hashes: Mapping[str, str],
) -> str:
    """Fingerprint the exact task-15 candidate universe and its inputs."""
    records = _canonical_frame_records(
        candidate_plan,
        (
            "candidate_id",
            "training_matrix_id",
            "matrix_method",
            "m",
            "balanced_pixel_strategy",
            "preprocessing",
            "preprocessing_steps",
            "sg_window_length",
            "sg_polyorder",
            "wavelength_axis_id",
        ),
    )
    return sha256_payload(
        {
            "protocol_hash": str(protocol_hash),
            "input_hashes": dict(
                sorted((str(key), str(value)) for key, value in input_hashes.items())
            ),
            "candidate_plan": records,
        }
    )


def pca_input_artifact_paths(
    project_root: str | Path,
    *,
    results_tag: str,
) -> dict[str, Path]:
    """Return the one central task-15 upstream artifact map."""
    root = Path(project_root)
    matrix_dir = root / "results" / f"{expcfg.MATRIX_RESULTS_DIR_PREFIX}_{results_tag}"
    return {
        "database_manifest": root.joinpath(
            *expcfg.DATABASE_RESULTS_RELATIVE_DIR,
            expcfg.DATABASE_OUTPUT_FILENAMES["manifest"],
        ),
        "protocol_split_manifest": root.joinpath(
            *expcfg.QC_RESULTS_RELATIVE_DIR,
            expcfg.QC_OUTPUT_FILENAMES["split_manifest"],
        ),
        "wavelength_config": matrix_dir
        / expcfg.MATRIX_OUTPUT_FILENAMES["wavelength_config"],
        "matrix_summary": matrix_dir
        / expcfg.MATRIX_OUTPUT_FILENAMES["matrix_summary"],
        "m_feasibility": matrix_dir
        / expcfg.MATRIX_OUTPUT_FILENAMES["m_feasibility"],
        "preprocessing_validation": matrix_dir
        / expcfg.MATRIX_OUTPUT_FILENAMES["preprocessing_validation"],
    }


def hash_pca_input_artifacts(
    project_root: str | Path,
    *,
    results_tag: str,
) -> dict[str, str]:
    """Hash every upstream artifact used by notebooks 03 and 03B."""
    paths = pca_input_artifact_paths(project_root, results_tag=results_tag)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PCA input artifacts: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def pca_input_fingerprint(input_hashes: Mapping[str, str]) -> str:
    """Hash a named upstream-artifact hash map deterministically."""
    return sha256_payload(
        dict(
            sorted((str(key), str(value)) for key, value in input_hashes.items())
        )
    )


def hash_pca_review_table(review: pd.DataFrame) -> str:
    """Hash the complete reviewed decisions in candidate order."""
    return sha256_payload(
        _canonical_frame_records(
            review,
            (
                "candidate_id",
                "review_status",
                "review_decision",
                "artifact_codes",
                "critical_artifact",
                "review_comment",
                "reviewer",
                "review_date",
                "review_evidence",
                "run_fingerprint",
            ),
        )
    )


def freeze_pca_shortlist(
    shortlist: pd.DataFrame,
    *,
    protocol_hash: str,
    review_hash: str,
    input_hashes: Mapping[str, str],
) -> pd.DataFrame:
    """Attach provenance hashes without creating a redundant set ID."""
    out = shortlist.copy()
    out["protocol_hash"] = str(protocol_hash)
    out["input_fingerprint"] = pca_input_fingerprint(input_hashes)
    out["review_hash"] = str(review_hash)
    out["selection_status"] = "selected"

    if out["selection_unit_id"].astype(str).duplicated().any():
        raise RuntimeError(
            "The PCA shortlist contains duplicate selection_unit_id values."
        )

    return out.reindex(
        columns=expcfg.PCA_SELECTED_PREPROCESSING_COLUMNS
    )


def validate_pca_preprocessing_shortlist(
    df: pd.DataFrame,
    *,
    max_per_family: int | None,
    expected_families: Sequence[str] | None = None,
    expected_protocol_hash: str | None = None,
    expected_input_fingerprint: str | None = None,
    expected_review_hash: str | None = None,
    family_col: str = "matrix_family",
    context: str = "PCA shortlist",
) -> pd.Series:
    """Validate a preprocessing shortlist and an optional family limit."""
    if df is None or len(df) == 0:
        raise RuntimeError(f"{context}: PCA selection is empty.")
    required = ("selection_unit_id", family_col, "preprocessing", "preprocessing_steps")
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"{context}: missing columns: {missing}")
    if "selection_status" in df and not df["selection_status"].eq("selected").all():
        raise RuntimeError(f"{context}: contains rows not marked selected.")
    counts = df.groupby(family_col, dropna=False).size().sort_index()
    if max_per_family is not None:
        overflow = counts[counts > int(max_per_family)]
        if len(overflow):
            raise RuntimeError(
                f"{context}: max {max_per_family} rows per {family_col}, "
                f"got {overflow.to_dict()}"
            )
    duplicates = df['selection_unit_id'].duplicated(keep=False)
    if duplicates.any():
        raise RuntimeError(
            f"{context}: duplicate preprocessing within a matrix family."
        )
    if expected_families is not None:
        missing_families = sorted(
            set(map(str, expected_families)) - set(df[family_col].astype(str))
        )
        if missing_families:
            raise RuntimeError(
                f"{context}: missing expected matrix families: {missing_families}"
            )
    for column, expected in (
        ("protocol_hash", expected_protocol_hash),
        ("input_fingerprint", expected_input_fingerprint),
        ("review_hash", expected_review_hash),
    ):
        if expected is None:
            continue
        if column not in df or not df[column].astype(str).eq(str(expected)).all():
            raise RuntimeError(f"{context}: stale or missing {column}.")
    return counts


def select_pca_preprocessing_shortlist(
    diagnostics_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select every non-dominated preprocessing after strict variant coverage.

    The returned preprocessing diagnostics contain the exact Pareto dominator
    set and an objective-wise JSON comparison for every dominated unit.
    """
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    if "preprocessing_eligible" in diagnostics_df:
        diagnostics = diagnostics_df.copy()
    else:
        candidate_diagnostics = (
            diagnostics_df.copy()
            if "technical_valid" in diagnostics_df
            else build_pca_selection_diagnostics(diagnostics_df, config=config)
        )
        diagnostics = aggregate_pca_preprocessing_diagnostics(
            candidate_diagnostics,
            config=config,
        )

    diagnostics["pareto_front"] = False
    diagnostics["dominated_by"] = ""
    retained_indices: list[object] = []

    for family, family_indices in diagnostics.groupby(config.family_col, dropna=False, sort=True).groups.items():
        profile = _profile_for_family(str(family),config)
        maximize = tuple(
            f"{metric}_worst"
            for metric in profile.maximize_metrics
        )
        minimize = tuple(
            f"{metric}_worst"
            for metric in profile.minimize_metrics
        )
        eligible = diagnostics.loc[family_indices]
        eligible = eligible.loc[
            eligible["preprocessing_eligible"]
            .fillna(False)
            .astype(bool)
        ].copy()
        if eligible.empty:
            continue
        valid_index, dominates = _pareto_dominance_matrix(
            eligible,
            maximize_metrics=maximize,
            minimize_metrics=minimize,
        )
        if len(valid_index) != len(eligible):
            raise RuntimeError(
                "A preprocessing marked Pareto-eligible contains "
                "non-finite objective values."
            )

        front_mask = ~dominates.any(axis=0)
        front_index = valid_index[front_mask]
        retained_indices.extend(front_index.tolist())
        labels = (
            diagnostics.loc[
                valid_index,
                "selection_unit_id",
            ]
            .astype(str)
            .to_numpy()
        )
        dominated_by = [
            ";".join(
                sorted(
                    labels[
                        np.flatnonzero(
                            dominates[:, candidate_position]
                        )
                    ].tolist()
                )
            )
            for candidate_position
            in range(len(valid_index))
        ]

        diagnostics.loc[
            valid_index,
            "selection_status",
        ] = "pareto_dominated"
        diagnostics.loc[
            valid_index,
            "dominated_by",
        ] = dominated_by
        diagnostics.loc[
            front_index,
            "pareto_front",
        ] = True
        diagnostics.loc[
            front_index,
            "selection_status",
        ] = "selected"

    if not retained_indices:
        raise RuntimeError("PCA Pareto selection is empty.")

    diagnostics["selection_reason"] = np.select(
        [
            diagnostics["pareto_front"].fillna(False),
            ~diagnostics["strict_coverage_pass"].fillna(False),
            ~diagnostics["objective_metrics_complete"].fillna(False),
        ],
        [
            "strict_family_coverage;preprocessing_level_pareto_front",
            "ineligible:missing_variant_or_blocked_candidate",
            "ineligible:non_finite_pareto_source_metric",
        ],
        default=(
            "dominated_on_preprocessing_pareto_by:"
            + diagnostics["dominated_by"].astype(str)
        ),
    )
    retained = diagnostics.loc[retained_indices].copy()

    max_per_family = config.max_preprocessings_per_family
    if max_per_family is not None:
        counts = retained.groupby(config.family_col, dropna=False).size()
        overflow = counts[counts > int(max_per_family)]
        if len(overflow):
            raise RuntimeError(
                "The preprocessing Pareto front exceeds the configured family limit "
                f"({max_per_family}): {overflow.to_dict()}. No automatic crowding or "
                "diversity truncation is applied; define an explicit scientific rule "
                "before reactivating the limit."
            )

    stage_summary, _ = build_pca_selection_flow_tables(
        diagnostics,
        config=config,
    )
    validate_pca_preprocessing_shortlist(
        retained,
        max_per_family=config.max_preprocessings_per_family,
        expected_families=config.expected_families,
    )
    return (
        retained.reset_index(drop=True),
        diagnostics.reset_index(drop=True),
        stage_summary,
    )


def add_pca_relative_quality_flags(
    df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Compatibility alias for the relative/Pareto diagnostic builder."""
    return build_pca_selection_diagnostics(df, config=config)


def format_pca_selection_reason(row: pd.Series) -> str:
    """Return the already-audited non-score selection reason."""
    return str(row.get("selection_reason", "not_evaluated"))
