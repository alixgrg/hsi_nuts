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


