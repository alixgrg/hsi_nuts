from __future__ import annotations

import json

import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.protocol_governance import (
    build_inference_plan,
    build_planned_contrasts,
    build_scientific_protocol_manifest,
    freeze_protocol,
    ProtocolValidationError,
    validate_protocol_contract,
    validate_selection_only_protocol_lineage,
    verify_frozen_protocol,
)


def test_eight_evaluation_tracks_cover_the_three_protocol_axes():
    assert len(expcfg.SIMCA_PARENT_TRACKS) == 4
    assert len(expcfg.SIMCA_EVALUATION_TRACKS) == 8
    assert set(expcfg.SIMCA_EVALUATION_TRACK_IDS.values()) == {
        f"E{index}" for index in range(1, 9)
    }

    observed = {
        (
            spec["training_matrix_family"],
            spec["projection_level"],
            spec["decision_mode"],
        )
        for spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.values()
    }
    expected = {
        (matrix_family, projection_level, decision_mode)
        for matrix_family in expcfg.SIMCA_MATRIX_FAMILIES
        for projection_level in expcfg.SIMCA_PROJECTION_LEVELS
        for decision_mode in expcfg.SIMCA_DECISION_MODES
    }
    assert observed == expected


def test_object_vote_thresholds_are_secondary_and_pixel_projection_only():
    tracks_with_fixed_votes = {
        spec["track_id"]
        for spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.values()
        if spec["secondary_object_aggregation_thresholds"]
    }
    assert tracks_with_fixed_votes == {"E3", "E7"}
    for spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.values():
        assert spec["decision_score_type"] == "simca_margin"
        if spec["track_id"] in tracks_with_fixed_votes:
            assert spec["projection_level"] == "pixel_projection"
            assert spec["decision_mode"] == "2way"
            assert tuple(
                spec["secondary_object_aggregation_thresholds"]
            ) == (0.75, 0.80)


def test_protocol_manifest_is_stable_and_excludes_legacy_score_weights():
    left = build_scientific_protocol_manifest()
    right = build_scientific_protocol_manifest()
    pd.testing.assert_frame_equal(left, right)
    assert left["configuration_sha256"].nunique() == 1
    assert not left["parameter"].str.contains("SCORE_WEIGHTS").any()
    assert (
        left.loc[
            left["parameter"].eq("PROTOCOL_SELECTION_POLICY"),
            "value_json",
        ]
        .item()
        .find('"weighted_scores_allowed":false')
        >= 0
    )


def test_h1_h4_inference_plan_is_frozen_and_complete():
    contrasts = build_planned_contrasts()
    plan = build_inference_plan(contrasts)

    assert plan["status"] == "frozen"
    assert {item["hypothesis_id"] for item in plan["hypotheses"]} == {
        "H1",
        "H2",
        "H3",
        "H4",
    }
    assert set(contrasts["hypothesis_id"]) == {"H1", "H2", "H3", "H4"}
    assert not contrasts["status"].eq("pending").any()
    assert contrasts["practical_tolerance"].gt(0).all()
    assert set(contrasts["bootstrap_group_col"]) == {"source_image"}
    assert set(contrasts["multiplicity_method"]) == {
        "holm_within_hypothesis_family"
    }


def test_protocol_contract_checks_are_all_blocking_and_green():
    checks = validate_protocol_contract(strict=True)
    assert len(checks) >= 10
    assert checks["passed"].all()
    assert checks["check"].is_unique


def test_tasks_03_14_output_budgets_and_public_api_are_canonical():
    assert len(expcfg.DATABASE_OUTPUT_FILENAMES) == 7
    assert len(expcfg.QC_OUTPUT_FILENAMES) == 11  # 10 Parquet + 1 PDF
    assert len(expcfg.SPATIAL_GT_OUTPUT_FILENAMES) == 6
    assert len(expcfg.MATRIX_OUTPUT_FILENAMES) == 8
    all_names = {
        *expcfg.DATABASE_OUTPUT_FILENAMES.values(),
        *expcfg.QC_OUTPUT_FILENAMES.values(),
        *expcfg.SPATIAL_GT_OUTPUT_FILENAMES.values(),
        *expcfg.MATRIX_OUTPUT_FILENAMES.values(),
    }
    assert not {
        "qc_flags.parquet",
        "balanced_sampling_summary.parquet",
        "preprocessing_summary.parquet",
        "matrix_preprocessing_errors.parquet",
    }.intersection(all_names)
    assert not hasattr(expcfg, "QC_MANUAL_REVIEW_OVERRIDES")

    import src
    import src.decision

    assert "add_detection_score" not in src.__all__
    assert "add_detection_score" not in src.decision.__all__


def test_freeze_protocol_writes_a_hashed_immutable_bundle(tmp_path):
    result = freeze_protocol(tmp_path)
    for path in result["paths"].values():
        assert path.exists()

    manifest = pd.read_parquet(result["paths"]["manifest"])
    contrasts = pd.read_parquet(result["paths"]["planned_contrasts"])
    checks = pd.read_parquet(result["paths"]["checks"])
    inference_plan = json.loads(
        result["paths"]["inference_plan"].read_text(encoding="utf-8")
    )
    lock = json.loads(result["paths"]["lock"].read_text(encoding="utf-8"))

    assert manifest["configuration_sha256"].iloc[0] == (
        result["configuration_sha256"]
    )
    assert len(contrasts) == inference_plan["n_planned_contrasts"]
    assert checks["passed"].all()
    assert lock["immutable"] is True
    assert lock["lock_sha256"] == result["lock_sha256"]
    verification = verify_frozen_protocol(tmp_path)
    assert verification["passed"].all()


def test_selection_only_lineage_requires_exact_execution_context():
    parent_hash = "parent-protocol"
    context = {
        "protocol_hash": parent_hash,
        "pca_selection_fingerprint": "pca",
        "track_contract_hash": "tracks",
        "fold_contract_hash": "folds",
        "configuration_hash": "configurations",
    }
    manifest = {
        **context,
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
    }

    checks = validate_selection_only_protocol_lineage(
        current_protocol_hash="amended-protocol",
        execution_protocol_hash=parent_hash,
        expected_parent_protocol_hash=parent_hash,
        amendment_scope="selection_only",
        selection_profile_id="amended-selection",
        selection_parent_profile_id="parent-selection",
        checkpoint_manifest=manifest,
        expected_execution_context=context,
    )
    assert checks["passed"].all()

    stale_manifest = {**manifest, "fold_contract_hash": "stale-folds"}
    with pytest.raises(
        ProtocolValidationError,
        match="fold_contract_hash",
    ):
        validate_selection_only_protocol_lineage(
            current_protocol_hash="amended-protocol",
            execution_protocol_hash=parent_hash,
            expected_parent_protocol_hash=parent_hash,
            amendment_scope="selection_only",
            selection_profile_id="amended-selection",
            selection_parent_profile_id="parent-selection",
            checkpoint_manifest=stale_manifest,
            expected_execution_context=context,
        )
