import numpy as np
import pytest

from src.spectra.preprocessing import (
    SpectralPreprocessor,
    preprocessing_input_validity_report,
)
from src.spectra.preprocessing_configs import PREPROCESSING_ALIASES


def _sample_spectra():
    wavelengths = np.linspace(900.0, 1700.0, 63)
    base = 0.45 + 0.05 * np.sin(wavelengths / 90.0)
    trend = np.linspace(-0.04, 0.04, wavelengths.size)

    spectra = []
    for idx in range(8):
        scale = 1.0 + 0.03 * idx
        offset = 0.005 * idx
        spectra.append(scale * base + offset + ((-1) ** idx) * trend)

    return np.asarray(spectra, dtype=float), wavelengths


_PREPROCESSING_CASES = list(PREPROCESSING_ALIASES.items())


@pytest.mark.parametrize(
    "preprocessing_name,steps",
    _PREPROCESSING_CASES,
    ids=[name for name, _ in _PREPROCESSING_CASES],
)
def test_fit_transform_matches_fit_then_transform_for_each_preprocessing(
    preprocessing_name,
    steps,
):
    X, wavelengths = _sample_spectra()

    direct = SpectralPreprocessor(
        steps=steps,
        sg_window_length=9,
        sg_polyorder=2,
    ).fit_transform(X, wavelengths=wavelengths)

    preprocessor = SpectralPreprocessor(
        steps=steps,
        sg_window_length=9,
        sg_polyorder=2,
    )
    sequential = preprocessor.fit(X, wavelengths=wavelengths).transform(X)

    np.testing.assert_allclose(
        direct,
        sequential,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Mismatch for preprocessing={preprocessing_name}",
    )


@pytest.mark.parametrize(
    "step,expected",
    [
        ("sg_smooth", {"window_length": 9, "polyorder": 2, "deriv": 0}),
        ("sg_d1", {"window_length": 9, "polyorder": 2, "deriv": 1}),
        ("sg_d2", {"window_length": 9, "polyorder": 2, "deriv": 2}),
    ],
)
def test_savgol_params_are_resolved_once_and_stored(step, expected):
    X, wavelengths = _sample_spectra()
    preprocessor = SpectralPreprocessor(
        steps=(step,),
        sg_window_length=9,
        sg_polyorder=2,
    )

    assert preprocessor._resolve_savgol_params(step) == expected

    preprocessor.fit(X, wavelengths=wavelengths)
    fitted_params = preprocessor.fitted_params_[step]

    for key, value in expected.items():
        assert fitted_params[key] == value
    assert fitted_params["delta"] == pytest.approx(float(np.mean(np.diff(wavelengths))))


@pytest.mark.parametrize(
    "steps",
    [
        ("absorbance", "msc"),
        ("absorbance", "snv"),
        ("absorbance", "snv", "sg_smooth"),
        ("absorbance", "msc", "sg_d1"),
        ("snv", "sg_d2"),
    ],
)
def test_combined_preprocessing_chains_are_finite_and_reusable(steps):
    X, wavelengths = _sample_spectra()

    preprocessor = SpectralPreprocessor(
        steps=steps,
        sg_window_length=9,
        sg_polyorder=2,
    )
    X_fit_transform = preprocessor.fit_transform(X, wavelengths=wavelengths)
    X_transform = preprocessor.transform(X)

    assert X_fit_transform.shape == X.shape
    assert X_transform.shape == X.shape
    assert np.isfinite(X_fit_transform).all()
    assert np.isfinite(X_transform).all()
    np.testing.assert_allclose(X_fit_transform, X_transform, rtol=1e-12, atol=1e-12)

    if "msc" in steps:
        assert "msc_reference" in preprocessor.fitted_params_


def test_input_validity_is_preprocessing_aware_without_silent_clipping():
    X = np.asarray(
        [
            [0.4, 0.5, 0.6],
            [0.0, 0.5, 0.6],
            [np.nan, 0.5, 0.6],
        ]
    )
    raw = preprocessing_input_validity_report(X, steps=("raw",))
    absorbance = preprocessing_input_validity_report(
        X,
        steps=("absorbance",),
    )
    assert raw["valid_mask"].tolist() == [True, True, False]
    assert absorbance["valid_mask"].tolist() == [True, False, False]
    assert absorbance["n_nonpositive_absorbance_rows"] == 1
    assert absorbance["n_nonfinite_rows"] == 1
    clipped = preprocessing_input_validity_report(
        X,
        steps=("absorbance",),
        absorbance_nonpositive_policy="clip",
    )
    assert clipped["valid_mask"].tolist() == [True, True, False]
