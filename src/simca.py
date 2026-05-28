import numpy as np
from scipy.stats import chi2

from .pca import pca_from_cov


def _as_2d_array(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


# def _pca_from_covariance(X_centered, n_components):
#     """
#     PCA from covariance matrix.

#     Parameters
#     ----------
#     X_centered : ndarray, shape (N, B)
#         Centered data matrix.
#     n_components : int
#         Number of PCs to keep.

#     Returns
#     -------
#     dict
#         PCA results.
#     """
#     X_centered = np.asarray(X_centered, dtype=float)

#     n_samples = X_centered.shape[0]

#     S = (X_centered.T @ X_centered) / (n_samples - 1)

#     eigvals, eigvecs = np.linalg.eigh(S)

#     idx = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[idx]
#     eigvecs = eigvecs[:, idx]

#     loadings = eigvecs[:, :n_components]
#     scores = X_centered @ loadings

#     explained_variance_ratio = eigvals / np.sum(eigvals)
#     cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

#     return {
#         "covariance": S,
#         "eigenvalues": eigvals,
#         "eigenvectors": eigvecs,
#         "loadings": loadings,
#         "scores": scores,
#         "explained_variance_ratio": explained_variance_ratio,
#         "cumulative_explained_variance_ratio": cumulative_explained_variance_ratio,
#     }

def _safe_positive(x, eps=1e-12):
    """
    Replace too-small positive values by eps.
    Useful to avoid division by zero.
    """
    return np.maximum(np.asarray(x, dtype=float), eps)



class SIMCAClassModel:
    """
    PCA model for one target class in SIMCA.

    This model learns:
    - class mean
    - PCA loadings
    - PCA eigenvalues
    - training H and Q distances
    - empirical H and Q limits
    """

    def __init__(
        self,
        class_name,
        n_components=3,
        alpha=0.05,
        eps=1e-12,
    ):
        self.class_name = class_name
        self.n_components = n_components
        self.alpha = float(alpha)
        self.eps = eps

        self.mean_ = None
        self.loadings_ = None
        self.eigenvalues_score_ = None
        self.eigenvalues_score_full_ = None
        self.pca_ = None

        self.H_train_ = None
        self.Q_train_ = None
        self.H0_ = None
        self.Q0_ = None
        self.H_limit_ = None
        self.Q_limit_ = None

        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X):
        """
        Fit one SIMCA class model.

        Parameters
        ----------
        X : ndarray, shape (N, B)
            Training spectra from the target class only.
        """
        X = _as_2d_array(X)
        self.n_samples_, self.n_features_ = X.shape

        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if self.n_samples_ <= self.n_components:
            raise ValueError(
                f"Class {self.class_name}: n_samples={self.n_samples_} "
                f"must be > n_components={self.n_components}."
            )
        if self.n_components > self.n_features_:
            raise ValueError(
                f"class={self.class_name}: n_components={self.n_components} "
                f"cannot exceed n_features={self.n_features_}."
            )
        
        # target class mean and PCA model
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        self.pca_ = pca_from_cov(
            Xc,
            n_components=self.n_components,
        )
        self.loadings_ = self.pca_["loadings"]
        self.eigenvalues_score_full_ = self.pca_["eigenvalues_score"]
        self.eigenvalues_score_ = self.pca_["eigenvalues_score"][: self.n_components]
        H, Q, _, _ = self.compute_distances(X)
        self.H_train_ = H
        self.Q_train_ = Q
        # empirical limits
        self._fit_distribution_parameters()
        self._fit_individual_limits()
        #self.H_limit_ = np.quantile(H, 1.0 - self.alpha)
        #self.Q_limit_ = np.quantile(Q, 1.0 - self.alpha)

        return self

    def transform(self, X):
        """
        Project X into this class PCA model.

        Returns
        -------
        scores : ndarray, shape (N, A)
        residuals : ndarray, shape (N, B)
        X_reconstructed_centered : ndarray, shape (N, B)
        """
        X = _as_2d_array(X)

        if self.mean_ is None:
            raise RuntimeError("Model must be fitted before transform.")
        
        Xc = X - self.mean_
        scores = Xc @ self.loadings_
        X_hat_c = scores @ self.loadings_.T
        residuals = Xc - X_hat_c
        return scores, residuals, X_hat_c

    def compute_distances(self, X):
        """
        Compute SIMCA H and Q distances.

        H = score distance, similar to Hotelling T².
        Q = orthogonal distance, squared residual norm.
        """
        #X = _as_2d_array(X)
        scores, residuals, _ = self.transform(X)
        lambdas = _safe_positive(self.eigenvalues_score_, self.eps)
        #lambdas = np.where(lambdas < self.eps, self.eps, lambdas)
        H = np.sum((scores ** 2) / lambdas, axis=1)
        Q = np.sum(residuals ** 2, axis=1)
        return H, Q, scores, residuals
    
    def _fit_distribution_parameters(self):
        """
        Estimate H0, Q0, NH and NQ for DD-SIMCA and CI-SIMCA.

        Paper formulas:
            H0 = A / I
            Q0 = mean(Q)
            NH = 2 H0^2 / Var(H)
            NQ = 2 Q0^2 / Var(Q)

        Variances are computed with denominator I, as in the paper.
        """
        I = self.n_samples_
        A = self.n_components
        H = self.H_train_
        Q = self.Q_train_
        self.H0_ = A / I
        self.Q0_ = np.mean(Q)
        var_H = np.mean((H - self.H0_) ** 2)
        var_Q = np.mean((Q - self.Q0_) ** 2)
        if var_H <= self.eps:
            self.NH_ = float(A)
        else:
            self.NH_ = float(2.0 * self.H0_ ** 2 / var_H)
        if var_Q <= self.eps or self.Q0_ <= self.eps:
            # If residuals are almost zero, set a large effective DoF.
            self.NQ_ = float(max(self.n_features_ - A, 1))
        else:
            self.NQ_ = float(2.0 * self.Q0_ ** 2 / var_Q)
        # Keep DoF numerically valid
        self.NH_ = max(self.NH_, self.eps)
        self.NQ_ = max(self.NQ_, self.eps)

    def _fit_individual_limits(self):
        """
        Compute H_alpha and Q_alpha using scaled chi-square distributions.

        If:
            NH * H / H0 ~ chi2(NH)
        then:
            H_alpha = H0 / NH * chi2.ppf(1-alpha, NH)

        Same for Q.
        """
        self.H_limit_ = (
            self.H0_ / self.NH_
            * chi2.ppf(1.0 - self.alpha, self.NH_)
        )

        self.Q_limit_ = (
            self.Q0_ / self.NQ_
            * chi2.ppf(1.0 - self.alpha, self.NQ_)
        )
        self.H_limit_ = max(float(self.H_limit_), self.eps)
        self.Q_limit_ = max(float(self.Q_limit_), self.eps)

    def decision_values(self, X):
        """
        Compute all useful quantities for decision rules.
        """
        H, Q, scores, residuals = self.compute_distances(X)
        H_norm_limit = H / self.H_limit_
        Q_norm_limit = Q / self.Q_limit_
        H_norm_mean = H / max(self.H0_, self.eps)
        Q_norm_mean = Q / max(self.Q0_, self.eps)
        return {
            "H": H,
            "Q": Q,
            "scores": scores,
            "residuals": residuals,
            "H_norm_limit": H_norm_limit,
            "Q_norm_limit": Q_norm_limit,
            "H_norm_mean": H_norm_mean,
            "Q_norm_mean": Q_norm_mean,
            "H_limit": self.H_limit_,
            "Q_limit": self.Q_limit_,
            "H0": self.H0_,
            "Q0": self.Q0_,
            "NH": self.NH_,
            "NQ": self.NQ_,
        }




class BaseSIMCARule:
    """
    Base interface for SIMCA decision rules.
    """
    name = "base"

    def fit(self, model):
        """
        Optional rule-specific fitting using the fitted class model.
        """
        return self

    def statistic(self, H, Q, model):
        """
        Return the rule statistic.
        """
        raise NotImplementedError

    def limit(self, model):
        """
        Return the rule threshold.
        """
        raise NotImplementedError

    def accept(self, H, Q, model):
        """
        Return boolean array: True if accepted by the target class.
        """
        raise NotImplementedError
    

class SimpleSIMCARule(BaseSIMCARule):
    """
    Simple SIMCA rule.

    Accept if:
        H < H_limit
    and:
        Q < Q_limit
    """
    name = "simple"
    def statistic(self, H, Q, model):
        H_norm = H / model.H_limit_
        Q_norm = Q / model.Q_limit_
        return np.maximum(H_norm, Q_norm)

    def limit(self, model):
        return 1.0

    def accept(self, H, Q, model):
        return (H < model.H_limit_) & (Q < model.Q_limit_)


class AltSIMCARule(BaseSIMCARule):
    """
    Alternative SIMCA rule.

    Accept if:
        H / H_limit + Q / Q_limit < 2
    """
    name = "alternative"

    def __init__(self, threshold=2.0, eps=1e-12):
        self.threshold = threshold
        self.eps = eps

    def statistic(self, H, Q, model):
        C_alt = H / model.H_limit_ + Q / model.Q_limit_
        return C_alt

    def limit(self,model):
        return self.threshold

    def accept(self, H, Q, model):
        return self.statistic(H, Q, model) < self.threshold


class CombinedIndexSIMCARule(BaseSIMCARule):
    """
    Combined Index SIMCA.

    C = H / H_alpha + Q / Q_alpha

    Accept if:
        C < C_alpha

    Here C_alpha is estimated by a scaled chi-square approximation
    fitted on the training C values.
    """

    name = "combined_index"

    def __init__(self, eps=1e-12):
        self.eps = eps
        self.C0_by_class_ = {}
        self.NC_by_class_ = {}
        self.C_limit_by_class_ = {}

    def fit(self, model):
        H = model.H_train_
        Q = model.Q_train_
        C = H / model.H_limit_ + Q / model.Q_limit_
        C0 = float(np.mean(C))
        var_C = float(np.mean((C - C0) ** 2))
        if var_C <= self.eps:
            NC = 1e6
        else:
            NC = 2.0 * C0 ** 2 / var_C
        NC = max(float(NC), self.eps)
        C_limit = C0 / NC * chi2.ppf(1.0 - model.alpha, NC)
        self.C0_by_class_[model.class_name] = C0
        self.NC_by_class_[model.class_name] = NC
        self.C_limit_by_class_[model.class_name] = float(C_limit)
        return self

    def statistic(self, H, Q, model):
        return H / model.H_limit_ + Q / model.Q_limit_

    def limit(self, model):
        return self.C_limit_by_class_[model.class_name]

    def accept(self, H, Q, model):
        C = self.statistic(H, Q, model)
        C_limit = self.limit(model)
        return C < C_limit


class DataDrivenSIMCARule(BaseSIMCARule):
    """
    Data Driven SIMCA.

    D = NQ * Q / Q0 + NH * H / H0

    Accept if:
        D < chi2.ppf(1-alpha, NQ + NH)
    """

    name = "data_driven"

    def statistic(self, H, Q, model):
        D = (
            model.NQ_ * Q / max(model.Q0_, model.eps)
            +
            model.NH_ * H / max(model.H0_, model.eps)
        )
        return D

    def limit(self, model):
        ND = model.NQ_ + model.NH_
        return chi2.ppf(1.0 - model.alpha, ND)

    def accept(self, H, Q, model):
        D = self.statistic(H, Q, model)
        D_limit = self.limit(model)
        return D < D_limit


class SIMCAClassifier:
    """
    Multi-class SIMCA classifier.

    It fits one independent SIMCAClassModel per class.
    """

    def __init__(
        self,
        class_names,
        n_components_by_class=None,
        alpha=0.05,
        rule=None,
    ):
        self.class_names = list(class_names)
        self.n_components_by_class = n_components_by_class or {}
        self.alpha = float(alpha)
        self.rule = rule if rule is not None else SimpleSIMCARule()
        self.models_ = {}

    def fit(self, X, y):
        """
        Fit one SIMCA model per class.

        Parameters
        ----------
        X : ndarray, shape (N, B)
        y : ndarray, shape (N,)
        """
        X = _as_2d_array(X)
        y = np.asarray(y)
        self.models_ = {}
        for class_name in self.class_names:
            X_class = X[y == class_name]
            if X_class.shape[0] == 0:
                raise ValueError(f"No samples found for class {class_name}.")
            n_components = self.n_components_by_class.get(class_name, 3)
            model = SIMCAClassModel(
                class_name=class_name,
                n_components=n_components,
                alpha=self.alpha,
            )
            model.fit(X_class)
            self.rule.fit(model)
            self.models_[class_name] = model
        return self

    def decision_function(self, X):
        """
        Compute H, Q and acceptance for each class.

        Returns
        -------
        results : dict
            results[class_name] contains H, Q, H_norm, Q_norm, accepted.
        """
        X = _as_2d_array(X)
        results = {}
        for class_name, model in self.models_.items():
            values = model.decision_values(X)
            H = values["H"]
            Q = values["Q"]
            accepted = self.rule.accept(H, Q, model)
            rule_statistic = self.rule.statistic(H, Q, model)
            rule_limit = self.rule.limit(model)
            values["accepted"] = accepted
            values["rule_name"] = self.rule.name
            values["rule_statistic"] = rule_statistic
            values["rule_limit"] = rule_limit
            values["combined_limit_distance"] = (
                values["H_norm_limit"] + values["Q_norm_limit"]
            )
            results[class_name] = values
        return results

    def predict(self, X):
        """
        Predict final label.

        Possible outputs:
        - class name if accepted by one class only
        - 'ambiguous' if accepted by several classes
        - 'unknown' if accepted by no class

        If accepted by several classes, the method also chooses the closest
        class internally by combined normalized distance, but final label remains
        'ambiguous' to preserve SIMCA's soft decision.
        """
        X = _as_2d_array(X)
        results = self.decision_function(X)
        n = X.shape[0]
        predictions = []
        closest_classes = []

        for i in range(n):
            accepted_classes = []
            distances = {}
            for class_name in self.class_names:
                accepted = results[class_name]["accepted"][i]
                # smaller H/Hlim + Q/Qlim means closer to class acceptance center
                dist = results[class_name]["combined_limit_distance"][i]
                distances[class_name] = dist
                if accepted:
                    accepted_classes.append(class_name)
            closest = min(distances, key=distances.get)
            closest_classes.append(closest)
            if len(accepted_classes) == 0:
                predictions.append("unknown")
            elif len(accepted_classes) == 1:
                predictions.append(accepted_classes[0])
            else:
                predictions.append("ambiguous")
        return np.array(predictions), np.array(closest_classes), results