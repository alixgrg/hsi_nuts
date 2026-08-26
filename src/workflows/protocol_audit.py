"""Cross-notebook scientific-governance checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.utils import clean_text_series, finalize_checks, numeric_comparison_mask


# ---------------------------------------------------------------------------
# Generic audit constants
# ---------------------------------------------------------------------------

_NUMERIC_COMPARATORS = {
    "<",
    "<=",
    ">",
    ">=",
    "==",
}

_CHECK_COLUMNS = (
    "check",
    "passed",
    "detail",
)

# ---------------------------------------------------------------------------
# Generic score-governance audit
# ---------------------------------------------------------------------------


def assert_no_forbidden_score_columns(
    tables: Mapping[str, pd.DataFrame],
    *,
    forbidden_columns: tuple[str, ...] = (
        expcfg.ACTIVE_PROTOCOL_FORBIDDEN_SCORE_COLUMNS
    ),
    strict: bool = True,
) -> pd.DataFrame:
    """Check that active protocol tables contain no arbitrary score column."""
    forbidden = {
        str(column).lower()
        for column in forbidden_columns
    }

    rows = []
    for table_name, table in tables.items():
        columns = (
            []
            if table is None
            else list(table.columns)
        )
        matches = sorted(
            str(column)
            for column in columns
            if str(column).lower() in forbidden
        )
        rows.append(
            {
                "table": str(table_name),
                "n_columns": len(columns),
                "forbidden_score_columns": ",".join(matches),
                "score_free": not matches,
            }
        )

    audit = pd.DataFrame(rows)
    failures = audit.loc[~audit["score_free"]]

    if strict and len(failures):
        raise RuntimeError(
            "Arbitrary weighted-score columns are forbidden "
            f"in the active protocol: "
            f"{failures.to_dict('records')}"
        )

    return audit


def _check(
    name: str,
    passed: bool,
    detail: object = "",
) -> dict[str, object]:
    return {
        "check": str(name),
        "passed": bool(passed),
        "detail": str(detail),
    }


def _check_frame(
    rows,
) -> pd.DataFrame:
    return pd.DataFrame(
        list(rows),
        columns=_CHECK_COLUMNS,
    )


def _check_selection_audit_schema(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    expected = tuple(expcfg.SELECTION_AUDIT_COLUMNS)
    observed = tuple(audit.columns)

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    exact = not missing and not extra

    rows.append(
        _check(
            "audit_schema_exact",
            exact,
            f"missing={missing}, extra={extra}",
        )
    )
    if not exact:
        return _check_frame(rows)

    rows.append(
        _check(
            "audit_column_order_matches_contract",
            observed == expected,
            f"observed={list(observed)}, "
            f"expected={list(expected)}",
        )
    )

    stages = clean_text_series(audit["stage"])
    track_ids = clean_text_series(audit["track_id"])
    substages = clean_text_series(audit["substage"])
    decisions = clean_text_series(audit["decision"])
    entity_types = clean_text_series(
        audit["entity_type"]
    )
    operators = clean_text_series(audit["operator"])

    rows.extend(
        [
            _check(
                "audit_natural_keys_unique",
                not audit.duplicated(
                    [
                        "stage",
                        "substage",
                        "entity_type",
                        "entity_id",
                        "related_entity_id",
                        "metric",
                    ]
                ).any(),
                "duplicates="
                f"{int(audit.duplicated([
                    'stage',
                    'substage',
                    'entity_type',
                    'entity_id',
                    'related_entity_id',
                    'metric',
                ]).sum())}",
            ),
            _check(
                "audit_stage_is_notebook03",
                stages.eq(expcfg.PCA_AUDIT_STAGE).all(),
                f"observed={sorted(stages.unique())}",
            ),
            _check(
                "pca_audit_is_track_independent",
                track_ids.eq("").all(),
                "nonempty_track_rows="
                f"{int(track_ids.ne('').sum())}",
            ),
        ]
    )

    observed_substages = set(substages)
    allowed_substages = set(
        expcfg.PCA_AUDIT_ALLOWED_SUBSTAGES
    )
    required_substages = set(
        expcfg.PCA_AUDIT_REQUIRED_SUBSTAGES
    )

    rows.append(
        _check(
            "pca_substages_are_valid",
            not observed_substages.difference(
                allowed_substages
            ),
            "unknown="
            f"{sorted(observed_substages-allowed_substages)}",
        )
    )
    rows.append(
        _check(
            "required_pca_substages_are_present",
            not required_substages.difference(
                observed_substages
            ),
            "missing="
            f"{sorted(required_substages-observed_substages)}",
        )
    )

    allowed_pairs = pd.MultiIndex.from_tuples(
        [
            (substage, decision)
            for substage, allowed
            in expcfg.PCA_AUDIT_ALLOWED_DECISIONS.items()
            for decision in allowed
        ]
    )
    observed_pairs = pd.MultiIndex.from_arrays(
        [substages, decisions]
    )
    known_substage = substages.isin(
        expcfg.PCA_AUDIT_ALLOWED_DECISIONS
    )
    invalid_decisions = (
        known_substage
        & ~observed_pairs.isin(allowed_pairs)
    )
    rows.append(
        _check(
            "decisions_match_substage_contract",
            not invalid_decisions.any(),
            "invalid="
            f"{audit.loc[
                invalid_decisions,
                ['substage', 'decision'],
            ].head(10).to_dict('records')}",
        )
    )

    candidate_mask = substages.isin(
        expcfg.PCA_AUDIT_CANDIDATE_SUBSTAGES
    )
    preprocessing_mask = substages.isin(
        expcfg.PCA_AUDIT_PREPROCESSING_SUBSTAGES
    )
    invalid_entity = (
        candidate_mask
        & ~entity_types.eq("pca_candidate")
    ) | (
        preprocessing_mask
        & ~entity_types.eq("pca_preprocessing")
    )
    rows.append(
        _check(
            "entity_types_match_substages",
            not invalid_entity.any(),
            "invalid_rows="
            f"{audit.index[invalid_entity].tolist()[:10]}",
        )
    )

    allowed_operators = {
        "",
        *_NUMERIC_COMPARATORS,
    }
    invalid_operators = sorted(
        set(operators).difference(allowed_operators)
    )
    rows.append(
        _check(
            "numeric_operators_are_valid",
            not invalid_operators,
            f"invalid={invalid_operators}",
        )
    )

    for column in (
        "observed_value",
        "reference_value",
    ):
        numeric = pd.to_numeric(
            audit[column],
            errors="coerce",
        )
        raw = clean_text_series(audit[column])
        explicitly_missing = (
            raw.eq("")
            | raw.str.lower().isin(
                {"nan", "none", "<na>"}
            )
        )
        invalid_numeric = (
            ~explicitly_missing
            & numeric.isna()
        )
        rows.append(
            _check(
                f"{column}_is_numeric_or_missing",
                not invalid_numeric.any(),
                "invalid_rows="
                f"{audit.index[
                    invalid_numeric
                ].tolist()[:10]}",
            )
        )

    eliminated = audit.loc[
        decisions.eq("eliminated")
    ]
    reasons = clean_text_series(
        eliminated["reason_code"]
    )
    missing_reason_rows = eliminated.index[
        reasons.eq("")
    ].tolist()
    rows.append(
        _check(
            "every_elimination_has_reason",
            not missing_reason_rows,
            f"missing_reason_rows={missing_reason_rows[:10]}",
        )
    )

    return _check_frame(rows)


def _check_pca_entity_contracts(
    audit: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    candidate_required = {
        "candidate_id",
        "selection_unit_id",
    }
    preprocessing_required = {
        "selection_unit_id",
        "strict_coverage_pass",
        "objective_metrics_complete",
        "pareto_front",
        "selection_status",
    }

    candidate_missing = sorted(
        candidate_required.difference(
            candidate_registry.columns
        )
    )
    preprocessing_missing = sorted(
        preprocessing_required.difference(
            preprocessing_summary.columns
        )
    )

    rows.extend(
        [
            _check(
                "candidate_registry_contract_complete",
                not candidate_missing,
                f"missing={candidate_missing}",
            ),
            _check(
                "preprocessing_summary_contract_complete",
                not preprocessing_missing,
                f"missing={preprocessing_missing}",
            ),
        ]
    )
    if candidate_missing or preprocessing_missing:
        return _check_frame(rows)

    candidate_ids = clean_text_series(
        candidate_registry["candidate_id"]
    )
    candidate_units = clean_text_series(
        candidate_registry["selection_unit_id"]
    )
    summary_units_series = clean_text_series(
        preprocessing_summary["selection_unit_id"]
    )

    rows.extend(
        [
            _check(
                "candidate_registry_ids_nonempty",
                candidate_ids.ne("").all(),
                f"empty={int(candidate_ids.eq('').sum())}",
            ),
            _check(
                "candidate_registry_ids_unique",
                not candidate_ids.duplicated().any(),
                "duplicates="
                f"{int(candidate_ids.duplicated().sum())}",
            ),
            _check(
                "candidate_registry_selection_units_nonempty",
                candidate_units.ne("").all(),
                f"empty={int(candidate_units.eq('').sum())}",
            ),
            _check(
                "preprocessing_summary_ids_nonempty",
                summary_units_series.ne("").all(),
                f"empty={int(summary_units_series.eq('').sum())}",
            ),
            _check(
                "preprocessing_summary_ids_unique",
                not summary_units_series.duplicated().any(),
                "duplicates="
                f"{int(summary_units_series.duplicated().sum())}",
            ),
        ]
    )

    registry_units = set(candidate_units)
    summary_units = set(summary_units_series)
    rows.append(
        _check(
            "registry_and_summary_selection_units_match",
            registry_units == summary_units,
            "missing_in_summary="
            f"{sorted(registry_units-summary_units)}, "
            "extra_in_summary="
            f"{sorted(summary_units-registry_units)}",
        )
    )

    entity_types = clean_text_series(
        audit["entity_type"]
    )
    preprocessing_events = audit.loc[
        entity_types.eq("pca_preprocessing")
    ].copy()
    preprocessing_event_ids = clean_text_series(
        preprocessing_events["entity_id"]
    )
    unknown = set(
        preprocessing_event_ids
    ).difference(summary_units)

    rows.append(
        _check(
            "all_preprocessing_events_reference_known_units",
            not unknown,
            f"unknown={sorted(unknown)}",
        )
    )

    non_dominance = preprocessing_events.loc[
        ~preprocessing_events["substage"].eq(
            "pareto_dominance"
        )
    ]
    related = clean_text_series(
        non_dominance["related_entity_id"]
    )
    invalid_related = non_dominance.index[
        related.ne("")
    ].tolist()
    rows.append(
        _check(
            "non_dominance_preprocessing_events_have_no_related_entity",
            not invalid_related,
            f"invalid_rows={invalid_related[:10]}",
        )
    )

    return _check_frame(rows)


def _check_pca_candidate_lineage(
    audit: pd.DataFrame,
    candidate_registry: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "selection_unit_id",
    }
    if required.difference(candidate_registry.columns):
        return _check_frame([])

    rows = []
    candidate_table = candidate_registry[
        ["candidate_id", "selection_unit_id"]
    ].copy()
    candidate_table["candidate_id"] = clean_text_series(
        candidate_table["candidate_id"]
    )
    candidate_table["selection_unit_id"] = (
        clean_text_series(
            candidate_table["selection_unit_id"]
        )
    )
    candidate_to_unit = (
        candidate_table
        .drop_duplicates("candidate_id", keep="last")
        .set_index("candidate_id")["selection_unit_id"]
    )

    candidate_ids = set(
        candidate_table["candidate_id"]
    )
    candidate_events = audit.loc[
        clean_text_series(
            audit["entity_type"]
        ).eq("pca_candidate")
    ].copy()
    event_ids = clean_text_series(
        candidate_events["entity_id"]
    )

    unknown = set(event_ids).difference(candidate_ids)
    rows.append(
        _check(
            "all_candidate_events_reference_known_candidates",
            not unknown,
            f"unknown={sorted(unknown)}",
        )
    )

    expected_related = event_ids.map(candidate_to_unit)
    observed_related = clean_text_series(
        candidate_events["related_entity_id"]
    )
    lineage_valid = (
        expected_related.notna()
        & observed_related.eq(
            expected_related.fillna("")
        )
    )
    rows.append(
        _check(
            "candidate_events_preserve_selection_unit_lineage",
            len(candidate_events) > 0
            and lineage_valid.all(),
            "invalid_rows="
            f"{candidate_events.index[
                ~lineage_valid
            ].tolist()[:10]}",
        )
    )

    substages = clean_text_series(
        candidate_events["substage"]
    )
    generation = candidate_events.loc[
        substages.eq("candidate_generation")
    ]
    generated_ids = clean_text_series(
        generation["entity_id"]
    )
    generated_set = set(generated_ids)

    rows.extend(
        [
            _check(
                "candidate_generation_universe_complete",
                generated_set == candidate_ids,
                "missing="
                f"{sorted(candidate_ids-generated_set)}, "
                "extra="
                f"{sorted(generated_set-candidate_ids)}",
            ),
            _check(
                "one_generation_event_per_candidate",
                len(generation) == len(candidate_ids)
                and not generated_ids.duplicated().any(),
                f"rows={len(generation)}, "
                f"expected={len(candidate_ids)}",
            ),
        ]
    )

    outcomes = candidate_events.loc[
        substages.eq("candidate_admissibility")
    ]
    outcome_ids = clean_text_series(
        outcomes["entity_id"]
    )
    outcome_set = set(outcome_ids)

    rows.extend(
        [
            _check(
                "candidate_outcome_universe_complete",
                outcome_set == candidate_ids,
                "missing="
                f"{sorted(candidate_ids-outcome_set)}, "
                "extra="
                f"{sorted(outcome_set-candidate_ids)}",
            ),
            _check(
                "one_candidate_outcome_per_candidate",
                len(outcomes) == len(candidate_ids)
                and not outcome_ids.duplicated().any(),
                f"rows={len(outcomes)}, "
                f"expected={len(candidate_ids)}",
            ),
        ]
    )

    return _check_frame(rows)


def _check_pca_outcomes(
    audit: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
    selected_preprocessings: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    selected_missing = (
        []
        if "selection_unit_id" in selected_preprocessings
        else ["selection_unit_id"]
    )
    rows.append(
        _check(
            "selected_preprocessings_contract_complete",
            not selected_missing,
            f"missing={selected_missing}",
        )
    )

    required = {
        "selection_unit_id",
        "strict_coverage_pass",
        "objective_metrics_complete",
        "pareto_front",
    }
    if (
        selected_missing
        or required.difference(
            preprocessing_summary.columns
        )
    ):
        return _check_frame(rows)

    summary = preprocessing_summary.copy()
    summary["_unit_id"] = clean_text_series(
        summary["selection_unit_id"]
    )
    summary["_expected_reason"] = np.select(
        [
            summary["pareto_front"]
            .fillna(False)
            .astype(bool),
            ~summary["strict_coverage_pass"]
            .fillna(False)
            .astype(bool),
            ~summary["objective_metrics_complete"]
            .fillna(False)
            .astype(bool),
        ],
        [
            "pareto_non_dominated",
            "not_pareto_eligible_strict_coverage",
            "not_pareto_eligible_incomplete_metrics",
        ],
        default="pareto_dominated",
    )

    reason_by_unit = (
        summary.drop_duplicates(
            "_unit_id",
            keep="last",
        )
        .set_index("_unit_id")["_expected_reason"]
    )
    expected_selected = set(
        summary.loc[
            summary["pareto_front"]
            .fillna(False)
            .astype(bool),
            "_unit_id",
        ]
    )
    summary_units = set(summary["_unit_id"])

    pareto_outcomes = audit.loc[
        clean_text_series(
            audit["entity_type"]
        ).eq("pca_preprocessing")
        & clean_text_series(
            audit["substage"]
        ).eq("pareto_selection")
    ].copy()
    outcome_ids = clean_text_series(
        pareto_outcomes["entity_id"]
    )
    outcome_set = set(outcome_ids)

    rows.extend(
        [
            _check(
                "preprocessing_outcome_universe_complete",
                outcome_set == summary_units,
                "missing="
                f"{sorted(summary_units-outcome_set)}, "
                "extra="
                f"{sorted(outcome_set-summary_units)}",
            ),
            _check(
                "one_pareto_outcome_per_selection_unit",
                len(pareto_outcomes) == len(summary_units)
                and not outcome_ids.duplicated().any(),
                f"rows={len(pareto_outcomes)}, "
                f"expected={len(summary_units)}",
            ),
        ]
    )

    expected_reasons = outcome_ids.map(reason_by_unit)
    observed_reasons = clean_text_series(
        pareto_outcomes["reason_code"]
    )
    reason_mismatch = (
        expected_reasons.isna()
        | observed_reasons.ne(
            expected_reasons.fillna("")
        )
    )
    rows.append(
        _check(
            "pareto_outcome_reasons_match_summary_state",
            not reason_mismatch.any(),
            "mismatches="
            f"{pareto_outcomes.loc[
                reason_mismatch,
                ['entity_id', 'reason_code'],
            ].head(10).to_dict('records')}",
        )
    )

    audit_selected = set(
        clean_text_series(
            pareto_outcomes.loc[
                pareto_outcomes["decision"].eq("kept"),
                "entity_id",
            ]
        )
    )
    rows.append(
        _check(
            "audit_selected_matches_preprocessing_summary",
            audit_selected == expected_selected,
            "missing="
            f"{sorted(expected_selected-audit_selected)}, "
            "extra="
            f"{sorted(audit_selected-expected_selected)}",
        )
    )

    shortlist_ids = clean_text_series(
        selected_preprocessings["selection_unit_id"]
    )
    shortlist_set = set(shortlist_ids)

    rows.extend(
        [
            _check(
                "selected_preprocessings_ids_nonempty",
                shortlist_ids.ne("").all(),
                f"empty={int(shortlist_ids.eq('').sum())}",
            ),
            _check(
                "selected_preprocessings_ids_unique",
                not shortlist_ids.duplicated().any(),
                "duplicates="
                f"{int(shortlist_ids.duplicated().sum())}",
            ),
            _check(
                "shortlist_matches_preprocessing_summary",
                shortlist_set == expected_selected,
                "missing="
                f"{sorted(expected_selected-shortlist_set)}, "
                "extra="
                f"{sorted(shortlist_set-expected_selected)}",
            ),
            _check(
                "audit_selected_matches_shortlist",
                audit_selected == shortlist_set,
                "missing="
                f"{sorted(shortlist_set-audit_selected)}, "
                "extra="
                f"{sorted(audit_selected-shortlist_set)}",
            ),
        ]
    )

    if "selection_set_id" in selected_preprocessings:
        selection_set_ids = clean_text_series(
            selected_preprocessings[
                "selection_set_id"
            ]
        )
        rows.append(
            _check(
                "shortlist_has_one_selection_set_id",
                selection_set_ids.ne("").all()
                and selection_set_ids.nunique() == 1,
                f"n_unique={selection_set_ids.nunique()}, "
                f"empty={int(selection_set_ids.eq('').sum())}",
            )
        )

    return _check_frame(rows)


def _check_numeric_audit_rules(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    operators = clean_text_series(audit["operator"])
    thresholded = audit.loc[
        operators.isin(_NUMERIC_COMPARATORS)
    ].copy()

    observed = pd.to_numeric(
        thresholded["observed_value"],
        errors="coerce",
    )
    reference = pd.to_numeric(
        thresholded["reference_value"],
        errors="coerce",
    )
    sources = clean_text_series(
        thresholded["reference_source"]
    )

    observed_valid = np.isfinite(
        observed.to_numpy(dtype=float)
    )
    reference_valid = np.isfinite(
        reference.to_numpy(dtype=float)
    )
    traceable = bool(
        observed_valid.all()
        and reference_valid.all()
        and sources.ne("").all()
    )
    rows.append(
        _check(
            "numeric_reference_rules_are_traceable",
            traceable,
            f"rows={len(thresholded)}, "
            f"missing_observed={int((~observed_valid).sum())}, "
            f"missing_reference={int((~reference_valid).sum())}, "
            f"missing_source={int(sources.eq('').sum())}",
        )
    )

    hard_rules = thresholded.loc[
        thresholded["mechanism"].isin(
            {
                "hard_constraint",
                "hard_threshold",
                "technical_eligibility",
                "eligibility",
            }
        )
    ].copy()

    relation_holds = numeric_comparison_mask(
        hard_rules["observed_value"],
        hard_rules["operator"],
        hard_rules["reference_value"],
        atol=expcfg.PCA_PARETO_ATOL,
    )
    expected_pass = (
        hard_rules["decision"]
        .astype(str)
        .eq("kept")
        .to_numpy()
    )
    mismatch = relation_holds != expected_pass

    mismatch_details = hard_rules.loc[
        mismatch,
        [
            "entity_id",
            "substage",
            "metric",
            "decision",
        ],
    ].copy()
    if len(mismatch_details):
        mismatch_details["relation_holds"] = (
            relation_holds[mismatch]
        )

    rows.append(
        _check(
            "hard_rule_decisions_match_numeric_relations",
            not mismatch.any(),
            "mismatches="
            f"{mismatch_details.head(10).to_dict('records')}",
        )
    )

    return _check_frame(rows)


def _check_pca_pareto_relations(
    audit: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
) -> pd.DataFrame:
    if "selection_unit_id" not in preprocessing_summary:
        return _check_frame([])

    rows = []
    summary_units = set(
        clean_text_series(
            preprocessing_summary["selection_unit_id"]
        )
    )

    pareto_outcomes = audit.loc[
        clean_text_series(
            audit["substage"]
        ).eq("pareto_selection")
    ].copy()
    dominated_final = set(
        clean_text_series(
            pareto_outcomes.loc[
                pareto_outcomes["reason_code"].eq(
                    "pareto_dominated"
                ),
                "entity_id",
            ]
        )
    )
    audit_selected = set(
        clean_text_series(
            pareto_outcomes.loc[
                pareto_outcomes["decision"].eq("kept"),
                "entity_id",
            ]
        )
    )

    dominance = audit.loc[
        clean_text_series(
            audit["substage"]
        ).eq("pareto_dominance")
    ].copy()
    source_ids = clean_text_series(
        dominance["entity_id"]
    )
    dominator_ids = clean_text_series(
        dominance["related_entity_id"]
    )
    source_set = set(source_ids)

    rows.extend(
        [
            _check(
                "every_pareto_dominated_unit_has_dominance_events",
                dominated_final == source_set,
                "missing_events="
                f"{sorted(dominated_final-source_set)}, "
                "unexpected_events="
                f"{sorted(source_set-dominated_final)}",
            ),
            _check(
                "pareto_dominators_are_nonempty",
                dominator_ids.ne("").all(),
                "empty_rows="
                f"{dominance.index[
                    dominator_ids.eq('')
                ].tolist()[:10]}",
            ),
            _check(
                "pareto_dominators_are_known_selection_units",
                not set(
                    dominator_ids.loc[
                        dominator_ids.ne("")
                    ]
                ).difference(summary_units),
                "unknown="
                f"{sorted(
                    set(dominator_ids).difference(summary_units)
                )}",
            ),
            _check(
                "selected_units_have_no_dominance_events",
                not audit_selected.intersection(source_set),
                "violations="
                f"{sorted(audit_selected.intersection(source_set))}",
            ),
        ]
    )

    if (
        "matrix_family" in preprocessing_summary
        and len(dominance)
    ):
        family_table = preprocessing_summary[
            ["selection_unit_id", "matrix_family"]
        ].copy()
        family_table["selection_unit_id"] = (
            clean_text_series(
                family_table["selection_unit_id"]
            )
        )
        family_table["matrix_family"] = (
            clean_text_series(
                family_table["matrix_family"]
            )
        )
        family_by_unit = (
            family_table
            .drop_duplicates("selection_unit_id")
            .set_index("selection_unit_id")[
                "matrix_family"
            ]
        )

        source_families = source_ids.map(family_by_unit)
        dominator_families = dominator_ids.map(
            family_by_unit
        )
        cross_family = (
            source_families.notna()
            & dominator_families.notna()
            & source_families.ne(dominator_families)
        )
        rows.append(
            _check(
                "pareto_dominance_stays_within_matrix_family",
                not cross_family.any(),
                "invalid_rows="
                f"{dominance.index[
                    cross_family
                ].tolist()[:10]}",
            )
        )

    observed = pd.to_numeric(
        dominance["observed_value"],
        errors="coerce",
    )
    reference = pd.to_numeric(
        dominance["reference_value"],
        errors="coerce",
    )
    operator = clean_text_series(
        dominance["operator"]
    )

    valid_contract = (
        operator.isin({"<=", ">="})
        & np.isfinite(observed.to_numpy(dtype=float))
        & np.isfinite(reference.to_numpy(dtype=float))
    )
    relation_holds = numeric_comparison_mask(
        observed,
        operator,
        reference,
        atol=expcfg.PCA_PARETO_ATOL,
    )
    relation_failure = (
        ~valid_contract.to_numpy()
        | ~relation_holds
    )

    rows.append(
        _check(
            "pareto_dominance_numeric_relations_hold",
            not relation_failure.any(),
            "failures="
            f"{dominance.loc[
                relation_failure,
                [
                    'entity_id',
                    'related_entity_id',
                    'metric',
                    'operator',
                ],
            ].head(10).to_dict('records')}",
        )
    )

    if len(dominance):
        strict = (
            valid_contract.to_numpy()
            & ~np.isclose(
                observed.to_numpy(dtype=float),
                reference.to_numpy(dtype=float),
                atol=expcfg.PCA_PARETO_ATOL,
                rtol=0.0,
            )
        )
        strict_by_pair = pd.Series(
            strict,
            index=dominance.index,
        ).groupby(
            [
                source_ids,
                dominator_ids,
            ],
            dropna=False,
            sort=True,
        ).any()
        non_strict = strict_by_pair.loc[
            ~strict_by_pair
        ]
        non_strict_pairs = [
            {
                "entity_id": str(source),
                "related_entity_id": str(dominator),
            }
            for source, dominator
            in non_strict.index
        ]
    else:
        non_strict_pairs = []

    rows.append(
        _check(
            "pareto_dominance_has_at_least_one_strict_objective",
            not non_strict_pairs,
            f"non_strict_pairs={non_strict_pairs[:10]}",
        )
    )

    if len(dominance):
        event_contract = (
            dominance["decision"].eq("eliminated")
            & dominance["reason_code"].eq(
                "pareto_dominated_by"
            )
            & dominance["mechanism"].eq("pareto")
            & operator.isin({"<=", ">="})
        )
        invalid_contract = dominance.index[
            ~event_contract
        ].tolist()
        contract_passed = not invalid_contract
        contract_detail = (
            f"invalid_rows={invalid_contract[:10]}"
        )
    else:
        contract_passed = not dominated_final
        contract_detail = (
            "No dominance events; "
            f"dominated_final_units={sorted(dominated_final)}"
        )

    rows.append(
        _check(
            "pareto_dominance_event_contract",
            contract_passed,
            contract_detail,
        )
    )

    return _check_frame(rows)


def _check_score_free_selection(
    audit: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
    selected_preprocessings: pd.DataFrame,
) -> pd.DataFrame:
    score_audit = assert_no_forbidden_score_columns(
        {
            "pca_candidate_registry": candidate_registry,
            "pca_preprocessing_summary": preprocessing_summary,
            "pca_selected_preprocessings": (
                selected_preprocessings
            ),
            "pca_selection_audit": audit,
        },
        strict=False,
    )
    return _check_frame(
        [
            _check(
                "pca_selection_tables_are_score_free",
                bool(score_audit["score_free"].all()),
                score_audit.to_dict("records"),
            )
        ]
    )


def assert_pca_selection_audit_consistency(
    audit: pd.DataFrame,
    *,
    candidate_registry: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
    selected_preprocessings: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """Validate notebook-03's compact PCA selection audit."""
    if not hasattr(expcfg, "SELECTION_AUDIT_COLUMNS"):
        raise AttributeError(
            "experiment_config.SELECTION_AUDIT_COLUMNS is required."
        )

    schema_checks = _check_selection_audit_schema(
        audit
    )
    schema_exact = schema_checks.loc[
        schema_checks["check"].eq("audit_schema_exact"),
        "passed",
    ].all()

    if not schema_exact:
        return finalize_checks(
            schema_checks,
            strict=strict,
            context="PCA selection audit",
        )

    check_frames = [
        schema_checks,
        _check_pca_entity_contracts(
            audit,
            candidate_registry,
            preprocessing_summary,
        ),
        _check_pca_candidate_lineage(
            audit,
            candidate_registry,
        ),
        _check_pca_outcomes(
            audit,
            preprocessing_summary,
            selected_preprocessings,
        ),
        _check_numeric_audit_rules(audit),
        _check_pca_pareto_relations(
            audit,
            preprocessing_summary,
        ),
        _check_score_free_selection(
            audit,
            candidate_registry,
            preprocessing_summary,
            selected_preprocessings,
        ),
    ]

    checks = pd.concat(
        check_frames,
        ignore_index=True,
    )
    return finalize_checks(
        checks,
        strict=strict,
        context="PCA selection audit",
    )

def assert_no_test_stage_inputs(
    df: pd.DataFrame,
    stage_cols: Sequence[str] = ("evaluation_stage", "evaluation_split"),
) -> pd.DataFrame:
    """Reject pure-test or held-out-test tables from the robustness notebook."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    forbidden = ("pure_test", "heldout_test", "held_out_test", "external_test")
    for col in stage_cols:
        if col not in out.columns:
            continue
        values = out[col].astype("string").str.lower().fillna("")
        bad = values.apply(lambda value: any(token in value for token in forbidden))
        if bool(bad.any()):
            examples = sorted(values.loc[bad].dropna().unique().tolist())[:5]
            raise ValueError(
                "Notebook 05 must not consume pure-test tables. "
                f"Forbidden values found in {col}: {examples}"
            )
    return out

__all__ = [
    "assert_no_forbidden_score_columns",
    "assert_pca_selection_audit_consistency",
]
