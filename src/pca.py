import numpy as np
from sklearn.decomposition import PCA


def pca_from_cov(X, n_components=None):
    """
    Perform PCA on the data matrix X using the covariance method.
    X must be centered (zero mean) before calling this function.

    Parameters
    ----------
    X : ndarray, shape (N, B)
        Data matrix (centered) of N spectra with B bands.
    n_components : int or None
        Number of principal components to keep. If None, keep all.

    Returns
    -------
    dict :
        "covariance": ndarray, shape (B, B)
            Covariance matrix of X.
        "eigenvalues": ndarray, shape (B,)
            Eigenvalues sorted in decreasing order.
        "eigenvectors": ndarray, shape (B, B)
            Corresponding eigenvectors (loadings), columns sorted by eigenvalue.
        "loadings": ndarray, shape (B, n_components)
            Loadings of the top n_components principal components.
        "scores": ndarray, shape (N, n_components)
            Projection of X onto the top n_components principal components.
        "explained_variance_ratio": ndarray, shape (B,)
            Proportion of variance explained by each principal component.
        "cumulative_explained_variance_ratio": ndarray, shape (B,)
            Cumulative proportion of variance explained up to each component.
    """
    # Cov matrix
    N = X.shape[0]
    S = (X.T @ X) / (N - 1)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(S) # eigh for symmetric matrices
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Loadings
    P = eigvecs[:, :n_components]
    # Scores
    T = X @ P
    # eigenvalues such that lambda_a = sum_i t_ia^2
    eigvals_scores = np.sum((X @ eigvecs)**2, axis=0)  #formula from paper, check if /(N-1) needed
    # Explained variance
    total = np.sum(eigvals)
    if total <= 0:
        explained_variance = np.zeros_like(eigvals)
    else:
        explained_variance = eigvals / total
    cumulative = np.cumsum(explained_variance)
    return {
        "covariance": S,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "eigenvalues_score": eigvals_scores,
        "loadings": P,
        "scores": T,
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance_ratio": cumulative,
    }


def pca_sklearn(X, n_components=None):
    """
    Perform PCA on the data matrix X using scikit-learn.

    Parameters
    ----------
    X : ndarray, shape (N, B)
        Data matrix of N spectra with B bands.
    n_components : int or None
        Number of principal components to keep. If None, keep all.

    Returns
    -------
    dict :
        "loadings": ndarray, shape (B, n_components)
            Loadings of the top n_components principal components.
        "scores": ndarray, shape (N, n_components)
            Projection of X onto the top n_components principal components.
        "explained_variance_ratio": ndarray, shape (n_components,)
            Proportion of variance explained by each principal component.
        "cumulative_explained_variance_ratio": ndarray, shape (n_components,)
            Cumulative proportion of variance explained up to each component.
    """
    pca = PCA(n_components=n_components)
    T = pca.fit_transform(X) # scores
    P = pca.components_.T # loadings
    explained_variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance)
    return {
        "loadings": P,
        "scores": T,
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance_ratio": cumulative,
    }