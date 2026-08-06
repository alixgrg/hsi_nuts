from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.matrices.matrix_registry import MatrixOutput, build_matrix, build_matrix_output
from src.matrices.redim_matrix import (
    select_balanced_pixel_indices,
    stable_object_seed,
)
from src.spectra.preprocessing import (
    SpectralPreprocessor,
    msc_transform,
    reflectance_to_absorbance,
)
from src.spectra.preprocessing_configs import (
    DEFAULT_PREPROCESSING_CONFIGS,
    SIMCA_SEARCH_PREPROCESSING_CONFIGS,
    normalize_preprocessing_configs,
    preprocessing_name_from_steps,
    validate_preprocessing_steps,
)
from src.utils import filter_dataframe_by_values, filter_records
from src.workflows.matrix_preprocessing import (
    assert_wavelength_lock,
    build_matrix_coverage_table,
    build_protocol_manifest,
    build_wavelength_config,
    evaluate_balanced_sampling_grid,
    evaluate_preprocessing_grid,
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
    assert set(metadata) == set(expcfg.MATRIX_REQUIRED_METADATA)


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

    rng = np.random.default_rng(123)
    X_preprocessed = SpectralPreprocessor(steps=("snv",)).fit_transform(
        0.2 + rng.random((8, X.shape[1]))
    )
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
    assert matrix_summary_df.loc[0, "status"] == "accepted"
    assert matrix_summary_df.loc[0, "n_objects"] == len(y)
    assert preprocessing_summary_df.loc[0, "n_features_after"] == X.shape[1]
    assert preprocessing_summary_df.loc[0, "status"] == "accepted"


def test_matrix_validation_rejects_nonfinite_missing_classes_and_coverage(mini_hsi_db):
    object_db, _ = mini_hsi_db
    output = build_matrix_output(
        object_db,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
    )
    assert output.validate(
        expected_classes=("almond", "peanut"),
        expected_object_ids=object_db,
        require_two_classes=True,
    ) is output

    bad = MatrixOutput(
        X=output.X.copy(),
        y=output.y,
        metadata=output.metadata,
        wavelengths=output.wavelengths,
        matrix_method=output.matrix_method,
        matrix_spec=output.matrix_spec,
    )
    bad.X[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        bad.validate()
    with pytest.raises(ValueError, match="coverage mismatch"):
        output.validate(expected_object_ids=["missing_obj"])


def test_under_m_policy_is_explicit(mini_hsi_db):
    object_db, _ = mini_hsi_db
    obj = object_db["almond1_obj001"]

    assert len(
        select_balanced_pixel_indices(
            obj,
            m=6,
            under_m_policy="short",
        )
    ) == 4
    assert (
        select_balanced_pixel_indices(
            obj,
            m=6,
            under_m_policy="exclude",
        )
        is None
    )
    assert len(
        select_balanced_pixel_indices(
            obj,
            m=6,
            under_m_policy="replace",
        )
    ) == 6
    with pytest.raises(ValueError, match="fewer than requested"):
        select_balanced_pixel_indices(obj, m=6, under_m_policy="error")


def test_balanced_sampling_uses_stable_independent_object_seeds(mini_hsi_db):
    object_db, _ = mini_hsi_db
    almond = object_db["almond1_obj001"]
    first, first_diag = select_balanced_pixel_indices(
        almond,
        m=2,
        random_state=42,
        object_id="almond1_obj001",
        return_diagnostics=True,
    )
    repeated, repeated_diag = select_balanced_pixel_indices(
        almond,
        m=2,
        random_state=42,
        object_id="almond1_obj001",
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(first, repeated)
    assert first_diag["selection_hash"] == repeated_diag["selection_hash"]
    assert stable_object_seed(42, "almond1_obj001") != stable_object_seed(
        42, "peanut1_obj001"
    )


def test_protocol_manifest_wavelength_lock_and_coverage(mini_hsi_db):
    object_db, image_db = mini_hsi_db
    manifest, checks = build_protocol_manifest(
        image_db,
        object_db,
        strict=False,
    )
    assert set(manifest["protocol_role"]) == {"calibration"}
    assert checks.loc[
        checks["check"].eq("calibration_contains_expected_classes"),
        "passed",
    ].item()

    wavelength_config = build_wavelength_config(
        image_db,
        object_db,
        wavelength_mode="test",
    )
    assert wavelength_config.loc[0, "all_axes_match"]
    assert len(wavelength_config.loc[0, "wavelength_axis_id"]) == 64
    changed = wavelength_config.copy()
    changed.loc[0, "spectral_config_id"] = "changed"
    with pytest.raises(RuntimeError, match="without a new protocol"):
        assert_wavelength_lock(wavelength_config, changed)

    output = build_matrix_output(
        object_db,
        matrix_method="object_mean",
        filters=PURE_FILTERS,
    )
    coverage = build_matrix_coverage_table(
        output.metadata,
        matrix_id="calibration_object_mean",
    )
    assert list(coverage.columns) == list(
        (
            "matrix_id",
            "object_id",
            "source_image",
            "batch",
            "label",
            "sample_kind",
            "n_rows",
        )
    )
    assert coverage["n_rows"].tolist() == [1, 1]


def test_balanced_sampling_and_preprocessing_grids_are_audited(mini_hsi_db):
    object_db, _ = mini_hsi_db
    sampling = evaluate_balanced_sampling_grid(
        object_db,
        filters=PURE_FILTERS,
        m_values=(2, 6),
        strategies=("random",),
        seeds=(1, 2),
        under_m_policy="exclude",
        min_eligible_rate=1.0,
    )
    assert sampling.set_index("m").loc[2, "status"] == "accepted"
    assert sampling.set_index("m").loc[6, "status"] == "invalid"
    assert sampling.set_index("m").loc[6, "n_objects_under_m"] == 2

    rng = np.random.default_rng(321)
    X = 0.2 + rng.random((8, 21))
    summary, outputs, errors = evaluate_preprocessing_grid(
        X[:4],
        X[4:],
        preprocessing_configs={
            "raw": ("raw",),
            "sg_d2": ("sg_d2",),
        },
        sg_windows=(5, 7),
        sg_polyorder=2,
        wavelengths=np.linspace(900.0, 1100.0, 21),
    )
    assert errors.empty
    assert summary["status"].eq("accepted").all()
    assert {"raw", "sg_d2__sg5", "sg_d2__sg7"}.issubset(outputs)
    assert summary.loc[summary["preprocessing"].eq("sg_d2"), "deriv"].eq(2).all()


def test_savgol_validation_uses_protocol_constraints():
    assert validate_preprocessing_steps(
        ("sg_d2",),
        n_features=21,
        sg_window_length=5,
        sg_polyorder=2,
    ) == ("sg_d2",)
    with pytest.raises(ValueError, match="odd"):
        validate_preprocessing_steps(
            ("sg_d1",),
            n_features=21,
            sg_window_length=6,
            sg_polyorder=2,
        )


def test_vectorized_msc_and_absorbance_policy():
    rng = np.random.default_rng(17)
    reference = 0.2 + rng.random(12)
    X = 0.1 + rng.random((7, 12))
    expected = np.vstack(
        [
            (row - np.linalg.lstsq(
                np.column_stack([np.ones_like(reference), reference]),
                row,
                rcond=None,
            )[0][0])
            / np.linalg.lstsq(
                np.column_stack([np.ones_like(reference), reference]),
                row,
                rcond=None,
            )[0][1]
            for row in X
        ]
    )
    np.testing.assert_allclose(msc_transform(X, reference), expected)
    with pytest.raises(ValueError, match="non-positive"):
        reflectance_to_absorbance(np.array([[0.2, 0.0, -0.1]]))
    with pytest.raises(ValueError, match="exceeds"):
        validate_preprocessing_steps(
            ("sg_d1",),
            n_features=7,
            sg_window_length=9,
            sg_polyorder=2,
        )
