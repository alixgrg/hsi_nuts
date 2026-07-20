from __future__ import annotations

import pandas as pd

from src.utils import save_parquet_if_nonempty
from src.workflows.quality_check import (
    IMAGE_QC_COLUMNS,
    OBJECT_QC_COLUMNS,
    QC_FLAG_COLUMNS,
    build_image_qc_table,
    build_image_qc_warnings,
    build_object_qc_table,
    build_object_qc_warnings,
    build_object_shape_check_tables,
    build_qc_flags_table,
    check_missing_required_fields,
)


def test_quality_check_tables_are_generated_on_mini_fixture(mini_hsi_db):
    object_db, image_db = mini_hsi_db

    image_qc_df = build_image_qc_table(image_db)
    object_qc_df = build_object_qc_table(object_db)

    assert list(image_qc_df.columns) == list(IMAGE_QC_COLUMNS)
    assert list(object_qc_df.columns) == list(OBJECT_QC_COLUMNS)
    assert len(image_qc_df) == 2
    assert len(object_qc_df) == 2
    assert image_qc_df["n_objects_recorded"].tolist() == [1, 1]
    assert object_qc_df["n_pixels"].tolist() == [4, 4]


def test_quality_check_empty_flags_do_not_require_output_file(tmp_path, mini_hsi_db):
    object_db, image_db = mini_hsi_db
    image_qc_df = build_image_qc_table(image_db)
    object_qc_df = build_object_qc_table(object_db)
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
