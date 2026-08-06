from __future__ import annotations

import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.utils import save_parquet_if_nonempty
from src.workflows.quality_check import (
    IMAGE_QC_COLUMNS,
    OBJECT_QC_COLUMNS,
    QC_FLAG_COLUMNS,
    QC_REVIEW_COLUMNS,
    apply_qc_reviews,
    build_image_qc_table,
    build_image_qc_warnings,
    build_object_qc_table,
    build_object_qc_warnings,
    build_object_shape_check_tables,
    build_qc_exclusion_report,
    build_qc_flags_table,
    build_qc_review_table,
    build_segmentation_diagnostics_table,
    build_spectral_integrity_table,
    check_missing_required_fields,
    qc_requires_new_cycle,
    validate_qc_review_closure,
)
from src.workflows.protocol_split import build_protocol_manifest


def test_quality_check_tables_are_generated_on_mini_fixture(mini_hsi_db):
    object_db, image_db = mini_hsi_db

    image_qc_df = build_image_qc_table(image_db)
    object_qc_df = build_object_qc_table(
        object_db,
        image_db=image_db,
        merge_warning_thresholds={
            "min_fill_ratio": 0.1,
            "min_separation_pixels": 0.0,
            "min_area_pixels": 1,
        },
    )

    assert list(image_qc_df.columns) == list(IMAGE_QC_COLUMNS)
    assert list(object_qc_df.columns) == list(OBJECT_QC_COLUMNS)
    assert len(image_qc_df) == 2
    assert len(object_qc_df) == 2
    assert image_qc_df["n_objects"].tolist() == [1, 1]
    assert object_qc_df["n_pixels"].tolist() == [4, 4]
    assert image_qc_df["image_status"].tolist() == ["accepted", "accepted"]
    assert object_qc_df["object_status"].tolist() == ["accepted", "accepted"]


def test_quality_check_empty_flags_do_not_require_output_file(tmp_path, mini_hsi_db):
    object_db, image_db = mini_hsi_db
    image_qc_df = build_image_qc_table(image_db)
    object_qc_df = build_object_qc_table(
        object_db,
        image_db=image_db,
        merge_warning_thresholds={
            "min_fill_ratio": 0.1,
            "min_separation_pixels": 0.0,
            "min_area_pixels": 1,
        },
    )
    image_warnings_df = build_image_qc_warnings(image_qc_df)
    object_warnings_df = build_object_qc_warnings(object_qc_df)
    missing_fields_df = check_missing_required_fields(image_db, object_db)
    _, bad_shape_df = build_object_shape_check_tables(object_db, image_db)

    qc_flags_df = build_qc_flags_table(
        image_warnings_df=image_warnings_df,
        object_warnings_df=object_warnings_df,
        missing_fields_df=missing_fields_df,
        bad_shape_df=bad_shape_df,
    )
    saved_path = save_parquet_if_nonempty(qc_flags_df, tmp_path / "qc_flags.parquet")

    assert image_warnings_df.empty
    assert object_warnings_df.empty
    assert missing_fields_df.empty
    assert bad_shape_df.empty
    assert qc_flags_df.empty
    assert list(qc_flags_df.columns) == QC_FLAG_COLUMNS
    assert saved_path is None
    assert not (tmp_path / "qc_flags.parquet").exists()


def test_quality_check_flags_combine_non_empty_inputs():
    image_warnings_df = pd.DataFrame(
        [{"clean_key": "almond1", "warning": "No object detected"}]
    )
    object_warnings_df = pd.DataFrame(
        [{"object_id": "almond1_obj001", "warning": "Invalid object area"}]
    )

    qc_flags_df = build_qc_flags_table(
        image_warnings_df=image_warnings_df,
        object_warnings_df=object_warnings_df,
    )

    assert list(qc_flags_df.columns) == QC_FLAG_COLUMNS
    assert qc_flags_df["flag_type"].tolist() == ["image_warning", "object_warning"]
    assert qc_flags_df["record_id"].tolist() == ["almond1", "almond1_obj001"]


def test_spectral_integrity_and_segmentation_diagnostics_detect_risk(mini_hsi_db):
    object_db, image_db = mini_hsi_db
    reference_axis = image_db["almond1"]["wavelengths"]
    bad_image_db = {key: value.copy() for key, value in image_db.items()}
    bad_image_db["peanut1"]["cube"] = bad_image_db["peanut1"]["cube"].copy()
    bad_image_db["peanut1"]["cube"][0, 0, 0] = np.nan

    integrity_df = build_spectral_integrity_table(
        bad_image_db,
        record_type="image",
        array_getter=lambda image: image["cube"],
        axis_getter=lambda image: image["wavelengths"],
        expected_ndim=3,
        reference_axis=reference_axis,
    )
    assert integrity_df.set_index("record_id").loc["peanut1", "is_valid"] == False
    assert integrity_df.set_index("record_id").loc["peanut1", "n_nan"] == 1

    diagnostics_df = build_segmentation_diagnostics_table(
        object_db,
        image_db,
        merge_warning_thresholds={
            "min_fill_ratio": 1.1,
            "min_separation_pixels": 0.0,
            "min_area_pixels": 1,
        },
    )
    assert list(diagnostics_df.columns) == list(
        expcfg.SEGMENTATION_DIAGNOSTIC_COLUMNS
    )
    assert diagnostics_df["segmentation_action"].eq(
        "review_possible_merge"
    ).all()


def test_qc_exclusions_and_cycle_signal_are_explicit():
    flags = pd.DataFrame(
        [
            {
                "record_type": "object",
                "record_id": "obj1",
                "flag_type": "too_small",
                "severity": "error",
                "qc_status": "excluded",
                "exclusion_reason": "area_below_minimum",
                "requires_segmentation_review": True,
                "warning": "Too small",
            }
        ],
        columns=QC_FLAG_COLUMNS,
    )
    exclusions = build_qc_exclusion_report(flags)
    assert exclusions["record_id"].tolist() == ["obj1"]
    assert qc_requires_new_cycle(flags) is True


def test_qc_reviews_are_auditable_and_pending_is_blocking():
    flags = pd.DataFrame(
        [
            {
                "record_type": "object",
                "record_id": "obj1",
                "flag_type": "possible_merged_object",
                "severity": "warning",
                "qc_status": "warning",
                "exclusion_reason": "",
                "requires_segmentation_review": True,
                "warning": "Review required",
            }
        ],
        columns=QC_FLAG_COLUMNS,
    )
    pending = build_qc_review_table(flags)
    assert list(pending.columns) == list(QC_REVIEW_COLUMNS)
    with np.testing.assert_raises_regex(RuntimeError, "pending"):
        apply_qc_reviews(flags, pending, require_complete=True)

    review = build_qc_review_table(
        flags,
        overrides=(
            {
                "record_type": "object",
                "record_id": "obj1",
                "flag_type": "possible_merged_object",
                "review_status": "reviewed",
                "review_decision": "accept_as_is",
                "reviewer": "reviewer",
                "review_date": "2026-07-29",
                "review_comment": "Single object.",
                "review_evidence": "review.png",
            },
        ),
        require_complete=True,
    )
    resolved = apply_qc_reviews(flags, review)
    assert resolved.loc[0, "qc_status"] == "accepted_after_review"
    assert not bool(resolved.loc[0, "requires_segmentation_review"])
    assert qc_requires_new_cycle(resolved) is False


def test_review_requires_documentation_and_exclusion_reaches_split(mini_hsi_db):
    object_db, image_db = mini_hsi_db
    alert = {
        "alert_id": "stable",
        "record_type": "object",
        "record_id": "almond1_obj001",
        "flag_type": "possible_merged_object",
        "severity": "warning",
        "qc_status": "warning",
        "exclusion_reason": "",
        "requires_segmentation_review": True,
        "warning": "Review",
        "evidence_json": '{"fill_ratio":0.2}',
    }
    flags = pd.DataFrame([alert], columns=QC_FLAG_COLUMNS)
    incomplete = build_qc_review_table(flags)
    incomplete.loc[0, [
        "review_status", "review_decision", "reviewer", "review_date",
        "review_comment", "review_evidence",
    ]] = ["reviewed", "exclude", "", "2026-07-30", "reason", "report.pdf"]
    with np.testing.assert_raises_regex(RuntimeError, "blocking"):
        validate_qc_review_closure(incomplete)

    exclusion = pd.DataFrame(
        [
            {
                "record_type": "object",
                "record_id": "almond1_obj001",
                "qc_status": "excluded",
                "exclusion_reason": "manual_review",
                "requires_segmentation_review": False,
            }
        ]
    )
    split, _ = build_protocol_manifest(
        image_db,
        object_db,
        exclusion_manifest=exclusion,
        strict=False,
    )
    row = split.set_index("object_id").loc["almond1_obj001"]
    assert row["qc_eligibility"] == "excluded"
    assert row["protocol_role"] == "excluded"
