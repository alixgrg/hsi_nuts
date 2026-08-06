"""Robust preprocessing-level Pareto selection for notebook 03.

Candidate PCA fits are reviewed individually, but the scientific decision unit
is ``(matrix_family, preprocessing)``. Matrix construction variants are treated
as robustness conditions: every expected variant must be admissible, and Pareto
objectives use the least favourable value observed across variants. No weighted
score, relative-quantile filter, projection target, or diversity filter is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.protocol_governance import sha256_file, sha256_payload


PCA_TECHNICAL_FLAG_COLUMNS = (
    "matrix_nonempty",
    "finite_values",
    "sg_valid",
    "variance_valid",
    "pca_fit_valid",
    "projection_valid",
    "residuals_valid",
    "stability_valid",
)

PCA_ARTIFACT_COLUMNS = ("critical_artifact",)
PCA_REQUIRED_REVIEW_STATUS = "reviewed"
PCA_ALLOWED_REVIEW_DECISIONS = ("accept", "warning", "reject")
PCA_REVIEW_METADATA_COLUMNS = (
    "review_decision",
    "artifact_codes",
    "reviewer",
    "review_date",
    "review_evidence",
    "run_fingerprint",
)


@dataclass(frozen=True)
class PCASelectionProfile:
    """Minimal family-specific preprocessing Pareto objectives."""

    maximize_metrics: tuple[str, ...] = ()
    minimize_metrics: tuple[str, ...] = ()


def _default_profiles() -> dict[str, PCASelectionProfile]:
    return {
        "object_matrix": PCASelectionProfile(
            maximize_metrics=("class_trace_ratio",),
            minimize_metrics=(
                "batch_trace_ratio",
                "instability_metric",
                "ncomp_95",
            ),
        ),
        "pixel_matrix": PCASelectionProfile(
            maximize_metrics=("object_class_trace_ratio",),
            minimize_metrics=(
                "object_batch_trace_ratio",
                "instability_metric",
                "ncomp_95",
            ),
        ),
    }


@dataclass(frozen=True)
class PCASelectionConfig:
    """Configuration for candidate review and preprocessing-level Pareto."""

    profiles: Mapping[str, PCASelectionProfile] = field(default_factory=_default_profiles)
    family_col: str = "matrix_family"
    variant_col: str = "matrix_variant"
    matrix_method_col: str = "matrix_method"
    preprocessing_col: str = "preprocessing"
    preprocessing_steps_col: str = "preprocessing_steps"
    expected_families: tuple[str, ...] = ("object_matrix", "pixel_matrix")
    strict_variant_coverage: bool = True
    max_preprocessings_per_family: int | None = None


DEFAULT_PCA_SELECTION_CONFIG = PCASelectionConfig()


def make_pca_selection_config(**overrides) -> PCASelectionConfig:
    """Return a PCA selection configuration without weighted score settings."""
    return replace(DEFAULT_PCA_SELECTION_CONFIG, **overrides)


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
    if str(default_decision) not in PCA_ALLOWED_REVIEW_DECISIONS:
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
        PCA_ALLOWED_REVIEW_DECISIONS
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
    out["review_status"] = PCA_REQUIRED_REVIEW_STATUS
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
    required = {
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
    }
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
    if not review["review_status"].astype(str).eq(
        PCA_REQUIRED_REVIEW_STATUS
    ).all():
        raise RuntimeError("PCA artifact review is pending for at least one candidate.")
    decisions = review["review_decision"].astype(str)
    invalid = ~decisions.isin(PCA_ALLOWED_REVIEW_DECISIONS)
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
        *[column for column in PCA_ARTIFACT_COLUMNS if column in artifact_review],
        *[
            column
            for column in (
                "review_status",
                "review_comment",
                *PCA_REVIEW_METADATA_COLUMNS,
            )
            if column in artifact_review
        ],
    ]
    out = out.drop(
        columns=[
            column
            for column in (
                *PCA_ARTIFACT_COLUMNS,
                "review_status",
                "review_comment",
                *PCA_REVIEW_METADATA_COLUMNS,
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
    for column in PCA_ARTIFACT_COLUMNS:
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
    for column in PCA_REVIEW_METADATA_COLUMNS:
        if column not in out:
            out[column] = ""
        else:
            out[column] = out[column].fillna("")
    return out


def select_pca_pareto_front(
    diagnostics: pd.DataFrame,
    maximize_metrics: Sequence[str],
    minimize_metrics: Sequence[str],
) -> pd.DataFrame:
    """Return candidates not dominated across the requested objectives."""
    metrics = [*maximize_metrics, *minimize_metrics]
    missing = [column for column in metrics if column not in diagnostics]
    if missing:
        raise KeyError(f"Missing Pareto metric columns: {missing}")
    if len(diagnostics) == 0:
        return diagnostics.copy()

    values = diagnostics[metrics].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy()).all(axis=1)
    candidates = diagnostics.loc[valid].copy()
    numeric = values.loc[valid].to_numpy(dtype=float)
    directions = np.asarray(
        [1.0] * len(maximize_metrics) + [-1.0] * len(minimize_metrics)
    )
    utilities = numeric * directions
    # competitor x candidate x objective; this is the complete vectorized
    # non-domination relation and contains no weighted aggregation.
    greater_or_equal = utilities[:, None, :] >= utilities[None, :, :]
    strictly_greater = utilities[:, None, :] > utilities[None, :, :]
    dominates = greater_or_equal.all(axis=2) & strictly_greater.any(axis=2)
    np.fill_diagonal(dominates, False)
    return candidates.loc[~dominates.any(axis=0)].copy()


def _technical_validity(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    flags: dict[str, pd.Series] = {}
    for column in PCA_TECHNICAL_FLAG_COLUMNS:
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
    review_complete = df.get(
        "review_status",
        pd.Series("pending", index=df.index),
    ).eq(PCA_REQUIRED_REVIEW_STATUS)
    review_decision = df.get(
        "review_decision",
        pd.Series("", index=df.index),
    ).astype(str)
    review_accepted = review_complete & review_decision.isin(
        {"accept", "warning"}
    )
    valid &= review_accepted
    reasons.loc[~review_complete] = reasons.loc[~review_complete].map(
        lambda value: (
            f"{value};artifact_review_pending"
            if value
            else "artifact_review_pending"
        )
    )
    rejected = review_complete & review_decision.eq("reject")
    reasons.loc[rejected] = reasons.loc[rejected].map(
        lambda value: f"{value};artifact_review_reject" if value else "artifact_review_reject"
    )
    invalid_decision = review_complete & ~review_decision.isin(
        PCA_ALLOWED_REVIEW_DECISIONS
    )
    reasons.loc[invalid_decision] = reasons.loc[invalid_decision].map(
        lambda value: f"{value};artifact_review_invalid" if value else "artifact_review_invalid"
    )
    if "critical_artifact" in df:
        critical = df["critical_artifact"].fillna(False).astype(bool)
        valid &= ~critical
        reasons.loc[critical] = reasons.loc[critical].map(
            lambda value: f"{value};critical_artifact" if value else "critical_artifact"
        )
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


def aggregate_pca_preprocessing_diagnostics(
    candidate_diagnostics_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
) -> pd.DataFrame:
    """Aggregate candidate diagnostics into robust preprocessing decisions.

    The expected variant universe is inferred once per matrix family from the
    complete candidate plan. A preprocessing is eligible only when it covers
    that universe, every candidate is admissible, and every Pareto source metric
    is finite. Pareto objectives are aggregated in the least favourable
    direction; medians and ranges are retained for audit only.
    """
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    group_cols = [
        config.family_col,
        config.preprocessing_col,
        config.preprocessing_steps_col,
    ]
    required = [*group_cols, config.variant_col, "technical_valid"]
    missing = [column for column in required if column not in candidate_diagnostics_df]
    if missing:
        raise KeyError(f"Missing preprocessing aggregation columns: {missing}")

    step_counts = candidate_diagnostics_df.groupby(
        [config.family_col, config.preprocessing_col],
        dropna=False,
    )[config.preprocessing_steps_col].nunique(dropna=False)
    ambiguous = step_counts[step_counts > 1]
    if len(ambiguous):
        raise RuntimeError(
            "A preprocessing name maps to multiple step definitions within a family: "
            f"{ambiguous.index.tolist()}"
        )

    expected_variants = {
        str(family): tuple(sorted(group[config.variant_col].astype(str).unique()))
        for family, group in candidate_diagnostics_df.groupby(
            config.family_col,
            dropna=False,
            sort=True,
        )
    }
    rows: list[dict[str, object]] = []
    for keys, group in candidate_diagnostics_df.groupby(
        group_cols,
        dropna=False,
        sort=True,
    ):
        family, preprocessing, preprocessing_steps = keys
        profile = _profile_for_family(str(family), config)
        maximize = tuple(profile.maximize_metrics)
        minimize = tuple(profile.minimize_metrics)
        objective_metrics = (*maximize, *minimize)
        missing_metrics = [metric for metric in objective_metrics if metric not in group]
        if missing_metrics:
            raise KeyError(
                f"Missing Pareto source metrics for {family!r}: {missing_metrics}"
            )

        family_expected = expected_variants[str(family)]
        observed = tuple(sorted(group[config.variant_col].astype(str).unique()))
        missing_variants = sorted(set(family_expected).difference(observed))
        extra_variants = sorted(set(observed).difference(family_expected))
        coverage_complete = not missing_variants and not extra_variants
        candidate_admissible = group["technical_valid"].fillna(False).astype(bool)
        all_candidates_admissible = bool(candidate_admissible.all())
        review_decisions = group.get(
            "review_decision",
            pd.Series("", index=group.index),
        ).fillna("").astype(str)

        numeric_metrics = group.loc[:, list(objective_metrics)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        objective_metrics_complete = bool(
            np.isfinite(numeric_metrics.to_numpy(dtype=float)).all()
        )
        strict_coverage_pass = (
            coverage_complete and all_candidates_admissible
            if config.strict_variant_coverage
            else bool(candidate_admissible.any())
        )
        eligible = strict_coverage_pass and objective_metrics_complete
        row: dict[str, object] = {
            config.family_col: family,
            config.preprocessing_col: preprocessing,
            config.preprocessing_steps_col: preprocessing_steps,
            "n_candidates": int(len(group)),
            "n_expected_variants": int(len(family_expected)),
            "n_observed_variants": int(len(observed)),
            "expected_variants_json": _json_values(family_expected),
            "observed_variants_json": _json_values(observed),
            "missing_variants_json": _json_values(missing_variants),
            "extra_variants_json": _json_values(extra_variants),
            "candidate_ids_json": _json_values(
                group.get("candidate_id", pd.Series(group.index, index=group.index))
            ),
            "n_accept": int(review_decisions.eq("accept").sum()),
            "n_warning": int(review_decisions.eq("warning").sum()),
            "n_reject": int(review_decisions.eq("reject").sum()),
            "n_blocked_candidates": int((~candidate_admissible).sum()),
            "coverage_complete": bool(coverage_complete),
            "all_candidates_admissible": all_candidates_admissible,
            "strict_coverage_pass": bool(strict_coverage_pass),
            "objective_metrics_complete": objective_metrics_complete,
            "preprocessing_eligible": bool(eligible),
            "pareto_front": False,
            "dominated_by": "",
            "selection_status": (
                "pareto_eligible"
                if eligible
                else (
                    "ineligible_incomplete_metrics"
                    if strict_coverage_pass
                    else "ineligible_strict_coverage"
                )
            ),
        }
        for metric in objective_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_min"] = float(finite.min()) if len(finite) else np.nan
            row[f"{metric}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )
            row[f"{metric}_max"] = float(finite.max()) if len(finite) else np.nan
            row[f"{metric}_iqr"] = (
                float(finite.quantile(0.75) - finite.quantile(0.25))
                if len(finite)
                else np.nan
            )
            row[f"{metric}_worst"] = (
                row[f"{metric}_min"] if metric in maximize else row[f"{metric}_max"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_pca_scoring_diagnostics(
    diagnostics_df: pd.DataFrame,
    config: PCASelectionConfig | None = None,
    *,
    preprocessing_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return candidate metrics and their preprocessing-level Pareto outcome."""
    config = DEFAULT_PCA_SELECTION_CONFIG if config is None else config
    diagnostics_df = diagnostics_df.copy()
    if preprocessing_summary_df is not None:
        merge_keys = [config.family_col, config.preprocessing_col]
        selection_columns = [
            *merge_keys,
            "selection_status",
            "selection_reason",
            "pareto_front",
        ]
        missing = [
            column for column in selection_columns if column not in preprocessing_summary_df
        ]
        if missing:
            raise KeyError(f"Missing preprocessing selection columns: {missing}")
        diagnostics_df = diagnostics_df.drop(
            columns=["selection_status", "selection_reason", "pareto_front"],
            errors="ignore",
        ).merge(
            preprocessing_summary_df.loc[:, selection_columns],
            on=merge_keys,
            how="left",
            validate="many_to_one",
        )
    id_columns = [
        "candidate_id",
        "training_matrix_id",
        "wavelength_axis_id",
        config.family_col,
        config.variant_col,
        config.matrix_method_col,
        "m",
        "balanced_pixel_strategy",
        config.preprocessing_col,
        config.preprocessing_steps_col,
    ]
    scientific_metrics = [
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
    ]
    output_columns = [
        *id_columns,
        "diagnostic_group",
        "metric",
        "value",
        "threshold",
        "constraint",
        "passed",
        "pareto_goal",
        "selection_status",
        "detail",
    ]
    output_rows = []

    def numeric_value(value):
        if value is None or pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return float(bool(value))
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return np.nan
        return converted if np.isfinite(converted) else np.nan

    for _, candidate in diagnostics_df.iterrows():
        identifiers = {
            column: candidate.get(column, "not_applicable")
            for column in id_columns
        }
        family = str(candidate.get(config.family_col))
        profile = _profile_for_family(family, config)
        pareto_goals = {
            **{metric: "maximize" for metric in profile.maximize_metrics},
            **{metric: "minimize" for metric in profile.minimize_metrics},
        }
        metrics = list(dict.fromkeys(
            [
                *scientific_metrics,
                *profile.maximize_metrics,
                *profile.minimize_metrics,
            ]
        ))
        for metric in metrics:
            if metric not in candidate:
                continue
            threshold = np.nan
            constraint = ""
            passed = None
            diagnostic_group = (
                "preprocessing_pareto_source_metric"
                if metric in pareto_goals
                else "scientific_metric"
            )
            output_rows.append(
                {
                    **identifiers,
                    "diagnostic_group": diagnostic_group,
                    "metric": metric,
                    "value": numeric_value(candidate.get(metric)),
                    "threshold": threshold,
                    "constraint": constraint,
                    "passed": passed,
                    "pareto_goal": pareto_goals.get(metric, ""),
                    "selection_status": candidate.get(
                        "selection_status",
                        "not_evaluated",
                    ),
                    "detail": "",
                }
            )

        for metric in (*PCA_TECHNICAL_FLAG_COLUMNS, "technical_valid"):
            passed = bool(candidate.get(metric, False))
            output_rows.append(
                {
                    **identifiers,
                    "diagnostic_group": "technical_blocker",
                    "metric": metric,
                    "value": float(passed),
                    "threshold": 1.0,
                    "constraint": "==",
                    "passed": passed,
                    "pareto_goal": "",
                    "selection_status": candidate.get(
                        "selection_status",
                        "not_evaluated",
                    ),
                    "detail": (
                        str(candidate.get("blocking_reason", ""))
                        if metric == "technical_valid"
                        else ""
                    ),
                }
            )

        for metric in PCA_ARTIFACT_COLUMNS:
            value = bool(candidate.get(metric, False))
            output_rows.append(
                {
                    **identifiers,
                    "diagnostic_group": "artifact_review",
                    "metric": metric,
                    "value": float(value),
                    "threshold": 0.0,
                    "constraint": "==",
                    "passed": not value,
                    "pareto_goal": "",
                    "selection_status": candidate.get(
                        "selection_status",
                        "not_evaluated",
                    ),
                    "detail": (
                        f"{candidate.get('review_status', 'pending')}: "
                        f"{candidate.get('review_decision', '')}; "
                        f"{candidate.get('review_comment', '')}".strip()
                    ),
                }
            )

        retained = candidate.get("selection_status") in {
            "selected",
        }
        output_rows.append(
            {
                **identifiers,
                "diagnostic_group": "selection",
                "metric": "candidate_retained_after_pareto",
                "value": float(retained),
                "threshold": np.nan,
                "constraint": "",
                "passed": retained,
                "pareto_goal": "",
                "selection_status": candidate.get(
                    "selection_status",
                    "not_evaluated",
                ),
                "detail": str(candidate.get("selection_reason", "")),
            }
        )

    return pd.DataFrame(output_rows, columns=output_columns)


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
    """Attach one immutable shortlist identity and all upstream hashes."""
    out = shortlist.copy()
    input_fingerprint = pca_input_fingerprint(input_hashes)
    payload = {
        "protocol_hash": str(protocol_hash),
        "review_hash": str(review_hash),
        "input_fingerprint": input_fingerprint,
        "selection": _canonical_frame_records(
            out,
            (
                "candidate_id",
                "matrix_family",
                "preprocessing",
                "preprocessing_steps",
            ),
        ),
    }
    out["shortlist_id"] = "pca_shortlist_" + sha256_payload(payload)[:20]
    out["protocol_hash"] = str(protocol_hash)
    out["input_fingerprint"] = input_fingerprint
    out["review_hash"] = str(review_hash)
    out["selection_status"] = "selected"
    return out


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
    required = (family_col, "preprocessing", "preprocessing_steps")
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
    duplicates = df.duplicated([family_col, "preprocessing"], keep=False)
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
    if "shortlist_id" in df and df["shortlist_id"].astype(str).nunique() != 1:
        raise RuntimeError(f"{context}: expected exactly one shortlist_id.")
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
    """Select every non-dominated preprocessing after strict variant coverage."""
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
    for family, family_indices in diagnostics.groupby(
        config.family_col,
        dropna=False,
        sort=True,
    ).groups.items():
        profile = _profile_for_family(str(family), config)
        maximize = tuple(f"{metric}_worst" for metric in profile.maximize_metrics)
        minimize = tuple(f"{metric}_worst" for metric in profile.minimize_metrics)
        eligible = diagnostics.loc[family_indices]
        eligible = eligible.loc[eligible["preprocessing_eligible"]].copy()
        if eligible.empty:
            continue
        front = select_pca_pareto_front(
            eligible,
            maximize_metrics=maximize,
            minimize_metrics=minimize,
        )
        retained_indices.extend(front.index.tolist())
        diagnostics.loc[eligible.index, "selection_status"] = "pareto_dominated"
        diagnostics.loc[front.index, "pareto_front"] = True
        diagnostics.loc[front.index, "selection_status"] = "selected"

        metric_columns = [*maximize, *minimize]
        numeric = eligible[metric_columns].apply(pd.to_numeric, errors="coerce")
        directions = np.asarray([1.0] * len(maximize) + [-1.0] * len(minimize))
        utilities = numeric.to_numpy(dtype=float) * directions
        greater_or_equal = utilities[:, None, :] >= utilities[None, :, :]
        strictly_greater = utilities[:, None, :] > utilities[None, :, :]
        dominates = greater_or_equal.all(axis=2) & strictly_greater.any(axis=2)
        np.fill_diagonal(dominates, False)
        labels = eligible[config.preprocessing_col].astype(str).to_numpy()
        for position, index in enumerate(eligible.index):
            if diagnostics.at[index, "pareto_front"]:
                continue
            diagnostics.at[index, "dominated_by"] = ";".join(
                sorted(labels[dominates[:, position]].tolist())
            )

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
