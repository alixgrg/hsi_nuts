from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.matrices.matrix_registry import MatrixOutput, build_matrix, build_matrix_output
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.preprocessing_configs import (
    DEFAULT_PREPROCESSING_CONFIGS,
    SIMCA_SEARCH_PREPROCESSING_CONFIGS,
    normalize_preprocessing_configs,
    preprocessing_name_from_steps,
)
from src.utils import filter_dataframe_by_values, filter_records
from src.workflows.matrix_preprocessing import (
    summarize_matrix_output,
    summarize_preprocessing_output,
    validate_required_columns,
)


PURE_FILTERS = {
    "sample_kind": "pure",
    "object_nut_type": ["almond", "peanut"],
    "batch": [1],
}


def test_build_matrix_supports_object_pixel_and_balanced_outputs(mini_hsi_db):
    object_db, _ = mini_hsi_db

    X_mean, y_mean, meta_mean = build_matrix(
        object_db,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
    )
    X_median, y_median, _ = build_matrix(
        object_db,
        matrix_method="object_median",
        filters=PURE_FILTERS,
    )
    X_pixels, y_pixels, meta_pixels = build_matrix(
        object_db,
        matrix_method="all_pixels",
        filters=PURE_FILTERS,
    )
    X_balanced, y_balanced, meta_balanced = build_matrix(
        object_db,
        matrix_method="balanced_pixels",
        filters=PURE_FILTERS,
        m=2,
        random_state=123,
        balanced_pixel_strategy="center",
    )

    assert X_mean.shape == (2, 4)
    assert X_median.shape == (2, 4)
    assert X_pixels.shape == (8, 4)
    assert X_balanced.shape == (4, 4)
    assert set(y_mean) == {"almond", "peanut"}
    assert set(y_median) == {"almond", "peanut"}
    assert set(y_pixels) == {"almond", "peanut"}
    assert set(y_balanced) == {"almond", "peanut"}
    assert "object_id" in meta_mean
    assert "pixel_index" in meta_pixels
    assert "pixel_index" in meta_balanced


def test_build_matrix_output_formal_contract_includes_wavelengths(mini_hsi_db):
    object_db, _ = mini_hsi_db

    output = build_matrix_output(
        object_db,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
    )
    X, y, metadata, wavelengths = build_matrix(
        object_db,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
        return_wavelengths=True,
    )

    assert isinstance(output, MatrixOutput)
    assert output.validate() is output
    assert output.X.shape == (2, 4)
    np.testing.assert_allclose(output.wavelengths, [900.0, 910.0, 920.0, 930.0])
    np.testing.assert_allclose(wavelengths, output.wavelengths)
    np.testing.assert_allclose(X, output.X)
    assert y.tolist() == output.y.tolist()
    assert metadata["object_id"].tolist() == output.metadata["object_id"].tolist()
    assert "source_clean_key" in metadata
    assert "source_image_id" in metadata


def test_dynamic_filters_work_for_records_dataframes_and_matrices(mini_hsi_db):
    object_db, _ = mini_hsi_db

    selected_ids = filter_records(
        object_db,
        return_items=False,
        sample_kind="pure",
        object_nut_type="almond",
    )
    assert selected_ids == ["almond1_obj001"]

    df = pd.DataFrame(object_db.values())
    filtered_df = filter_dataframe_by_values(
        df,
        {"sample_kind": "pure", "object_nut_type": ["almond"], "batch": 1},
    )
    assert filtered_df["object_id"].tolist() == ["almond1_obj001"]

    X, y, meta = build_matrix(
        object_db,
        matrix_method="object_mean",
        filters={"source_clean_key": "almond1"},
    )
    assert X.shape == (1, 4)
    assert y.tolist() == ["almond"]
    assert meta["object_id"].tolist() == ["almond1_obj001"]

    with pytest.raises(ValueError, match="No objects found"):
        build_matrix(
            object_db,
            matrix_method="object_mean",
            filters={"object_nut_type": "walnut"},
        )


def test_normalize_preprocessing_configs_accepts_aliases_and_blocks_invalid_chains():
    configs = normalize_preprocessing_configs(["raw", "absorbance_snv", ("snv", "sg_d1")])

    assert configs["raw"] == ("raw",)
    assert configs["absorbance_snv"] == ("absorbance", "snv")
    assert configs["snv_sg_d1"] == ("snv", "sg_d1")

    with pytest.raises(ValueError, match="Unknown preprocessing"):
        normalize_preprocessing_configs(["does_not_exist"])

    with pytest.raises(ValueError, match="raw"):
        normalize_preprocessing_configs({"bad": ("raw", "snv")})


def test_preprocessing_config_names_are_stable_for_default_and_simca_sets():
    for configs in [DEFAULT_PREPROCESSING_CONFIGS, SIMCA_SEARCH_PREPROCESSING_CONFIGS]:
        normalized = normalize_preprocessing_configs(configs)
        for name, steps in normalized.items():
            assert preprocessing_name_from_steps(steps) == name


def test_matrix_and_preprocessing_summary_contracts(mini_hsi_db):
    object_db, _ = mini_hsi_db
    X, y, meta = build_matrix(
        object_db,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
    )

    matrix_row, meta_df = summarize_matrix_output(
        X,
        y,
        meta,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
    )
    matrix_summary_df = pd.DataFrame([matrix_row])
    validate_required_columns(
        matrix_summary_df,
        expcfg.MATRIX_SUMMARY_REQUIRED_COLUMNS,
        table_name="matrix_summary.parquet",
    )

    preprocessor = SpectralPreprocessor(steps=("snv",))
    X_preprocessed = preprocessor.fit_transform(X)
    preprocessing_row = summarize_preprocessing_output(
        X_preprocessed,
        preprocessing_name="snv",
        steps=("snv",),
        sg_window_length=9,
        sg_polyorder=2,
    )
    preprocessing_summary_df = pd.DataFrame([preprocessing_row])
    validate_required_columns(
        preprocessing_summary_df,
        expcfg.PREPROCESSING_SUMMARY_REQUIRED_COLUMNS,
        table_name="preprocessing_summary.parquet",
    )

    assert len(meta_df) == len(y)
    assert bool(matrix_summary_df.loc[0, "has_metadata"]) is True
    assert preprocessing_summary_df.loc[0, "n_features"] == X.shape[1]
    assert np.isfinite(preprocessing_summary_df.loc[0, "global_std"])
