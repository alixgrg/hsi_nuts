import numpy as np
from scipy.signal import savgol_filter

from src.spectra.preprocessing_configs import validate_preprocessing_steps


_SAVGOL_DERIV_BY_STEP = {
    "sg_smooth": 0,
    "sg_d1": 1,
    "sg_d2": 2,
}


def center_X(X):
    """
    Center the data matrix X by subtracting the mean spectrum.

    Parameters
    ----------
    X : ndarray, shape (N, B)
        Data matrix of N spectra with B bands.

    Returns
    -------
    X_centered : ndarray, shape (N, B)
        Centered data matrix.
    mu : ndarray, shape (B,)
        Mean spectrum that was subtracted from each row of X.
    """
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    return X_centered, mu


def snv(X, eps=1e-12):
    """
    Apply Standard Normal Variate (SNV) normalization to the data matrix X.

    Parameters
    ----------
    X : ndarray, shape (N, B)
        Data matrix of N spectra with B bands.
    eps : float
        Small constant to avoid division by zero when standard deviation is zero.

    Returns
    -------
    X_snv : ndarray, shape (N, B)
        SNV-normalized data matrix.
    """
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, ddof=1.0, keepdims=True)
    sigma = np.where(sigma < eps, 1.0, sigma)  # avoid division by zero
    X_snv = (X - mu) / sigma
    return X_snv


def vector_normalize(X, eps=1e-12):
    """
    Normalize each spectrum by its L2 norm.
    """
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def msc_fit(X):
    """
    Fit MSC reference spectrum.
    Usually the mean spectrum of the training set.
    """
    X = np.asarray(X, dtype=float)
    ref = np.mean(X, axis=0)
    return ref

def msc_transform(X, ref, eps=1e-12):
    """
    Apply Multiplicative Scatter Correction.
    """
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float)

    if X.ndim != 2 or ref.ndim != 1 or X.shape[1] != ref.size:
        raise ValueError(
            f"MSC requires X=(n, bands) and ref=(bands,), got {X.shape} "
            f"and {ref.shape}."
        )
    ref_centered = ref - ref.mean()
    denominator = float(ref_centered @ ref_centered)
    if denominator <= float(eps):
        raise ValueError("MSC reference spectrum has zero variance.")
    means = X.mean(axis=1)
    slopes = ((X - means[:, None]) @ ref_centered) / denominator
    safe_slopes = np.where(np.abs(slopes) < float(eps), 1.0, slopes)
    intercepts = means - safe_slopes * ref.mean()
    return (X - intercepts[:, None]) / safe_slopes[:, None]


def savgol_derivative(
    X,
    window_length=9,
    polyorder=2,
    deriv=1,
    delta=1.0,
):
    """
    Savitzky-Golay derivative.
    
    deriv=0: smoothing
    deriv=1: first derivative
    deriv=2: second derivative
    delta: wavelength spacing.
    """
    X = np.asarray(X, dtype=float)
    return savgol_filter(
        X,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=1,
        mode="interp",
    )


def reflectance_to_absorbance(X, eps=1e-8, nonpositive_policy="error"):
    """
    Convert reflectance to absorbance:
        A = log10(1 / R) = -log10(R)
    """
    X = np.asarray(X, dtype=float)
    nonpositive = X <= 0
    policy = str(nonpositive_policy)
    if np.any(nonpositive) and policy == "error":
        raise ValueError(
            "Absorbance is undefined for non-positive reflectance; "
            f"found {int(nonpositive.sum())} value(s)."
        )
    if policy not in {"error", "clip"}:
        raise ValueError(
            "nonpositive_policy must be either 'error' or 'clip'."
        )
    X_safe = np.clip(X, eps, None) if policy == "clip" else X
    return np.log10(1.0 / X_safe)


def preprocessing_input_validity_report(
    X,
    *,
    steps=("raw",),
    absorbance_nonpositive_policy="error",
):
    """Return the row mask required before applying a preprocessing chain.

    General QC removes non-finite spectra. Strict absorbance adds a
    preprocessing-dependent requirement: every reflectance value in a row
    must be strictly positive. Keeping this rule next to the transform avoids
    duplicating it in PCA/SIMCA projection workflows and prevents silent
    clipping to extreme absorbance values.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={values.shape}")
    resolved_steps = [steps] if isinstance(steps, str) else list(steps)
    finite_mask = np.isfinite(values).all(axis=1)
    strict_absorbance = (
        "absorbance" in resolved_steps
        and str(absorbance_nonpositive_policy) == "error"
    )
    nonpositive_absorbance_mask = (
        finite_mask & np.any(values <= 0.0, axis=1)
        if strict_absorbance
        else np.zeros(values.shape[0], dtype=bool)
    )
    valid_mask = finite_mask & ~nonpositive_absorbance_mask
    return {
        "valid_mask": valid_mask,
        "n_input_rows": int(values.shape[0]),
        "n_valid_rows": int(valid_mask.sum()),
        "n_filtered_rows": int((~valid_mask).sum()),
        "n_nonfinite_rows": int((~finite_mask).sum()),
        "n_nonpositive_absorbance_rows": int(
            nonpositive_absorbance_mask.sum()
        ),
    }


class SpectralPreprocessor:
    def __init__(
        self,
        steps=("raw",),
        sg_window_length=9,
        sg_polyorder=2,
        eps=1e-12,
        absorbance_nonpositive_policy="error",
    ):
        resolved_steps = [steps] if isinstance(steps, str) else list(steps)
        self.steps = list(validate_preprocessing_steps(resolved_steps))
        self.sg_window_length = int(sg_window_length)
        self.sg_polyorder = int(sg_polyorder)
        self.eps = eps
        self.absorbance_nonpositive_policy = str(
            absorbance_nonpositive_policy
        )
        self.fitted_params_ = {}
        self.wavelengths_ = None
        self.is_fitted_ = False

    def input_validity_report(self, X):
        """Return preprocessing-aware row validity without transforming X."""
        return preprocessing_input_validity_report(
            X,
            steps=self.steps,
            absorbance_nonpositive_policy=(
                self.absorbance_nonpositive_policy
            ),
        )

    def _resolve_savgol_params(self, step):
        if step not in _SAVGOL_DERIV_BY_STEP:
            raise ValueError(f"Unknown Savitzky-Golay preprocessing step: {step}")

        deriv = _SAVGOL_DERIV_BY_STEP[step]
        return {
            "window_length": self.sg_window_length,
            "polyorder": self.sg_polyorder,
            "deriv": int(deriv),
        }

    def fit(self, X, wavelengths=None):
        X_work = np.asarray(X, dtype=float)
        if X_work.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_work.shape}")
        if any(step in _SAVGOL_DERIV_BY_STEP for step in self.steps):
            validate_preprocessing_steps(
                self.steps,
                n_features=X_work.shape[1],
                sg_window_length=self.sg_window_length,
                sg_polyorder=self.sg_polyorder,
            )
        self.wavelengths_ = None if wavelengths is None else np.asarray(
            wavelengths,
            dtype=float,
        )
        if self.wavelengths_ is not None:
            if (
                self.wavelengths_.ndim != 1
                or len(self.wavelengths_) != X_work.shape[1]
                or not np.all(np.diff(self.wavelengths_) > 0)
            ):
                raise ValueError(
                    "The wavelength axis must be one-dimensional, aligned "
                    "with X, and strictly increasing."
                )

        for step in self.steps:
            if step == "raw":
                continue
            if step == "msc":
                ref = msc_fit(X_work)
                self.fitted_params_["msc_reference"] = ref
                X_work = msc_transform(X_work, ref, eps=self.eps)
            elif step in {"sg_smooth", "sg_d1", "sg_d2"}:
                params = self._resolve_savgol_params(step)
                delta = 1.0 if self.wavelengths_ is None else float(np.mean(np.diff(self.wavelengths_)))
                params["delta"] = delta
                self.fitted_params_[step] = params
                X_work = savgol_derivative(
                    X_work,
                    **params,
                )
            else:
                X_work = self._apply_stateless_step(X_work, step)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("SpectralPreprocessor must be fitted before transform().")
        X_work = np.asarray(X, dtype=float)
        for step in self.steps:
            if step == "raw":
                continue
            if step == "msc":
                X_work = msc_transform(
                    X_work,
                    self.fitted_params_["msc_reference"],
                    eps=self.eps,
                )
            elif step in {"sg_smooth", "sg_d1", "sg_d2"}:
                params = self._resolve_savgol_params(step)
                params.update(self.fitted_params_[step])
                X_work = savgol_derivative(
                    X_work,
                    **params,
                )
            else:
                X_work = self._apply_stateless_step(X_work, step)
        return X_work

    def fit_transform(self, X, wavelengths=None):
        return self.fit(X, wavelengths=wavelengths).transform(X)

    def _apply_stateless_step(self, X, step):
        if step == "absorbance":
            return reflectance_to_absorbance(
                X,
                eps=self.eps,
                nonpositive_policy=self.absorbance_nonpositive_policy,
            )
        if step == "snv":
            return snv(X, eps=self.eps)
        if step == "vector_norm":
            return vector_normalize(X, eps=self.eps)
        raise ValueError(f"Unknown preprocessing step: {step}")
