import numpy as np
from scipy.signal import savgol_filter

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

    X_corr = np.zeros_like(X)

    # Design matrix: [1, ref]
    A = np.vstack([np.ones_like(ref), ref]).T

    for i in range(X.shape[0]):
        coef, _, _, _ = np.linalg.lstsq(A, X[i, :], rcond=None)
        a_i, b_i = coef

        if abs(b_i) < eps:
            b_i = 1.0

        X_corr[i, :] = (X[i, :] - a_i) / b_i

    return X_corr


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


def reflectance_to_absorbance(X, eps=1e-8):
    """
    Convert reflectance to absorbance:
        A = log10(1 / R) = -log10(R)
    """
    X = np.asarray(X, dtype=float)
    X_safe = np.clip(X, eps, None)
    return np.log10(1.0 / X_safe)


class SpectralPreprocessor:
    def __init__(
        self,
        steps=("raw",),
        sg_window_length=9,
        sg_polyorder=2,
        eps=1e-12,
    ):
        self.steps = [steps] if isinstance(steps, str) else list(steps)
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder
        self.eps = eps
        self.fitted_params_ = {}
        self.wavelengths_ = None
        self.is_fitted_ = False

    def fit(self, X, wavelengths=None):
        X_work = np.asarray(X, dtype=float)
        self.wavelengths_ = None if wavelengths is None else np.asarray(wavelengths)

        for step in self.steps:
            if step == "raw":
                continue
            if step == "msc":
                ref = msc_fit(X_work)
                self.fitted_params_["msc_reference"] = ref
                X_work = msc_transform(X_work, ref, eps=self.eps)
            elif step in {"sg_smooth", "sg_d1", "sg_d2"}:
                deriv_map = {
                    "sg_smooth": 0,
                    "sg_d1": 1,
                    "sg_d2": 2,
                }
                deriv = deriv_map[step]
                delta = 1.0 if self.wavelengths_ is None else float(np.mean(np.diff(self.wavelengths_)))
                self.fitted_params_[step] = {
                    "deriv": deriv,
                    "delta": delta,
                }
                X_work = savgol_derivative(
                    X_work,
                    window_length=self.sg_window_length if deriv in {0, 1} else max(self.sg_window_length, 11),
                    polyorder=self.sg_polyorder if deriv in {0, 1} else max(self.sg_polyorder, 3),
                    deriv=deriv,
                    delta=delta,
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
                params = self.fitted_params_[step]
                deriv = params["deriv"]
                X_work = savgol_derivative(
                    X_work,
                    window_length=self.sg_window_length if deriv == 1 else max(self.sg_window_length, 11),
                    polyorder=self.sg_polyorder if deriv == 1 else max(self.sg_polyorder, 3),
                    deriv=deriv,
                    delta=params["delta"],
                )
            else:
                X_work = self._apply_stateless_step(X_work, step)
        return X_work

    def fit_transform(self, X, wavelengths=None):
        return self.fit(X, wavelengths=wavelengths).transform(X)

    def _apply_stateless_step(self, X, step):
        if step == "absorbance":
            return reflectance_to_absorbance(X, eps=self.eps)
        if step == "snv":
            return snv(X, eps=self.eps)
        if step == "vector_norm":
            return vector_normalize(X, eps=self.eps)
        raise ValueError(f"Unknown preprocessing step: {step}")