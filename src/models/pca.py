import numpy as np

from src.utils import as_2d_array, safe_positive


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
    X = as_2d_array(X)
    N = X.shape[0]
    # Cov matrix
    S = (X.T @ X) / (N - 1)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(S) # eigh for symmetric matrices
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if n_components is None:
        n_components = X.shape[1]
    # Loadings
    P = eigvecs[:, :n_components]
    # Scores
    T = X @ P
    # eigenvalues such that lambda_a = sum_i t_ia^2
    #eigvals_scores = np.sum((X @ eigvecs)**2, axis=0)  #formula from paper, check if /(N-1) needed
    # Explained variance
    total = np.sum(eigvals)
    explained_variance = eigvals / total if total > 0 else np.zeros_like(eigvals)
    cumulative = np.cumsum(explained_variance)
    return {
        "covariance": S,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        #"eigenvalues_score": eigvals_scores,
        "loadings": P,
        "scores": T,
        "explained_variance_ratio": explained_variance,
        "cumulative_explained_variance_ratio": cumulative,
    }



class PCAModel:
    def __init__(self, n_components=None, center=True, eps=1e-12):
        self.n_components = n_components
        self.center = center
        self.eps = eps
        self.Xc_ = None
        self.mean_ = None
        self.covariance_ = None
        self.loadings_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None
        self.scores_ = None

    def fit(self, X):
        X = as_2d_array(X)
        self.mean_ = np.mean(X, axis=0) if self.center else np.zeros(X.shape[1])
        Xc = X - self.mean_
        self.Xc_ = Xc
        res = pca_from_cov(Xc, n_components=self.n_components)
        self.covariance_ = res["covariance"]
        self.scores_ = res["scores"]
        self.loadings_ = res["loadings"]
        self.eigenvalues_ = res["eigenvalues"]
        self.eigenvectors_ = res["eigenvectors"]
        self.explained_variance_ratio_ = res["explained_variance_ratio"]
        self.cumulative_explained_variance_ratio_ = res["cumulative_explained_variance_ratio"]
        return self

    def transform(self, X):
        self._check_fitted()
        X = as_2d_array(X)
        Xc = X - self.mean_
        return Xc @ self.loadings_

    def fit_transform(self, X):
        return self.fit(X).scores_

    def inverse_transform(self, scores):
        self._check_fitted()
        scores = as_2d_array(scores)
        return scores @ self.loadings_.T + self.mean_

    def reconstruct(self, X, n_components=None):
        self._check_fitted()
        X = as_2d_array(X)
        if n_components is None:
            n_components = self.loadings_.shape[1]
        scores = self.transform(X)
        P = self.loadings_[:, :n_components]
        T = scores[:, :n_components]
        X_hat = T @ P.T + self.mean_
        residuals = X - X_hat
        return X_hat, residuals

    def q_residuals(self, X, n_components=None):
        _, residuals = self.reconstruct(X, n_components=n_components)
        return np.sum(residuals ** 2, axis=1), residuals

    def hotelling_t2(self, X=None, n_components=None):
        self._check_fitted()
        if X is None:
            scores = self.scores_
        else:
            scores = self.transform(X)
        if n_components is None:
            n_components = scores.shape[1]
        lambdas = safe_positive(
            self.eigenvalues_[:n_components],
            eps=self.eps,
        )
        return np.sum((scores[:, :n_components] ** 2) / lambdas, axis=1)

    def distances(self, X, n_components=None):
        if X is None:
            X = self.inverse_transform(self.scores_)
        T2 = self.hotelling_t2(X, n_components=n_components)
        Q, _ = self.q_residuals(X, n_components=n_components)
        return T2, Q

    def _check_fitted(self):
        if self.loadings_ is None:
            raise RuntimeError("PCAModel must be fitted before use.")