import numpy as np
from scipy.stats import chi2

def mean_spectrum(cube):
    pixels = cube.reshape(-1, cube.shape[2])
    return np.nanmean(pixels, axis=0)

def hotelling_t2(pca_res, n_components=None):
    """
    Compute Hotelling's T² statistic for each observation in the PCA scores.

    Parameters
    ----------
    pca_res : dict
        Result of PCA containing "scores" and "eigenvalues".
    n_components : int or None
        Number of principal components to use for T² calculation. If None, use all.

    Returns
    -------
    T2 : ndarray, shape (N,)
        Hotelling's T² statistic for each observation.
    """
    T = np.asarray(pca_res["scores"])
    eigvals = np.asarray(pca_res["eigenvalues"])
    if n_components is None:
        n_components = T.shape[1]
    
    scores = T[:, :n_components]
    lambdas = eigvals[:n_components]
    
    # Avoid division by zero in case of zero variance
    lambdas = eigvals[:n_components].copy()
    lambdas[lambdas == 0] = 1e-10
    
    T2 = np.sum((scores ** 2) / lambdas, axis=1)
    
    return T2


def q_residuals(X_c, pca_res, n_components=None):
    """
    Compute Q residuals (squared reconstruction error) for each observation in X_c.
    E = Xc - T P^T
    Q_i = sum_b E_i,b^2

    Parameters
    ----------
    X_c : ndarray, shape (N, B)
        Centered data matrix.
    pca_res : dict
        Result of PCA containing "loadings" and "scores".
    n_components : int or None
        Number of principal components to use for reconstruction. If None, use all.

    Returns
    -------
    Q : ndarray, shape (N,)
        Q residuals for each observation.
    E : ndarray, shape (N, B)
        Residuals (reconstruction error) for each observation and band.
    """
    Xc = np.asarray(X_c)
    P = np.asarray(pca_res["loadings"])
    T = np.asarray(pca_res["scores"])
    if n_components is None:
        n_components = P.shape[1]
    
    # Reconstruct the data using the selected number of components
    X_hat = T[:, :n_components] @ P[:, :n_components].T
    # Compute residuals
    E = Xc - X_hat
    # Compute Q residuals as squared reconstruction error
    Q = np.sum(E ** 2, axis=1)
    
    return Q, E


