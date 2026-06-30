import numpy as np
from scipy.stats import chi2

from src.models.pca import PCAModel
from src.utils import as_2d_array, safe_positive
from src.models.simca_rules import (
    BaseSIMCARule,
    SimpleSIMCARule,
    AltSIMCARule,
    CombinedIndexSIMCARule,
    DataDrivenSIMCARule,
)


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
        h_dof_method="theoretical",
        q_dof_method="box",
    ):
        self.class_name = class_name
        self.n_components = n_components
        self.alpha = float(alpha)
        self.eps = eps

        self.h_dof_method = h_dof_method
        self.q_dof_method = q_dof_method

        self.mean_ = None
        self.loadings_ = None
        self.eigenvalues_score_ = None
        self.eigenvalues_score_full_ = None
        self.pca_ = None

        self.H_train_ = None
        self.Q_train_ = None
        self.H0_ = None
        self.Q0_ = None
        self.NH_ = None
        self.NQ_ = None
        self.H_limit_ = None
        self.Q_limit_ = None

        self.var_H_ = None
        self.var_Q_ = None
        self.q1_residual_ = None
        self.q2_residual_ = None

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
        X = as_2d_array(X)
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
        self.pca_ = PCAModel(n_components=self.n_components, center=True, eps=self.eps).fit(X)
        self.mean_ = self.pca_.mean_
        self.loadings_ = self.pca_.loadings_
        self.eigenvalues_score_full_ = self.pca_.eigenvalues_
        self.eigenvalues_score_ = self.pca_.eigenvalues_[: self.n_components]
        H, Q, _, _ = self.compute_distances(X)
        self.H_train_ = H
        self.Q_train_ = Q
        # chi2 based distribution parameters and theoretical limits
        self._fit_distribution_parameters()
        self._fit_individual_limits()
        self.fit_empirical_limits()
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
        X = as_2d_array(X)

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
        #X = as_2d_array(X)
        scores, residuals, _ = self.transform(X)
        lambdas = safe_positive(self.eigenvalues_score_, self.eps)
        #lambdas = np.where(lambdas < self.eps, self.eps, lambdas)
        H = np.sum((scores ** 2) / lambdas, axis=1)
        Q = np.sum(residuals ** 2, axis=1)
        return H, Q, scores, residuals
    
    def _fit_distribution_parameters(self):
        """
        Estimate H0, Q0, NH and NQ for SIMCA decision rules.

        Important convention
        --------------------
        PCAModel stores covariance eigenvalues:

            lambda_cov = sum_i(t_ia^2) / (I - 1)

        Therefore the score distance used here,

            H = sum_a t_ia^2 / lambda_cov_a

        is on the covariance scale.

        With this convention:

            mean(H) = A * (I - 1) / I

        instead of A / I in the paper convention.

        For H:
            default uses theoretical NH = A.

        For Q:
            default uses Box/Qin moments from residual covariance eigenvalues:

                Q0 = sum residual eigenvalues
                NQ = q1^2 / q2

            where:
                q1 = sum(lambda_residual)
                q2 = sum(lambda_residual^2)
        """
        I = int(self.n_samples_)
        A = int(self.n_components)

        H = np.asarray(self.H_train_, dtype=float)
        Q = np.asarray(self.Q_train_, dtype=float)

        # ------------------------------------------------------------
        # H distribution
        # ------------------------------------------------------------
        # Because H is computed with covariance eigenvalues, not score
        # eigenvalues, the expected mean is A * (I - 1) / I.
        self.H0_ = float(A * (I - 1) / I)

        if self.h_dof_method == "theoretical":
            # Under approximately normal scores, NH = A.
            # This avoids unstable empirical variance estimation.
            self.NH_ = float(A)
            self.var_H_ = float(2.0 * self.H0_**2 / self.NH_)

        elif self.h_dof_method == "moment":
            # Optional empirical method of moments.
            # Use only for diagnostics or if you explicitly want data-driven H DoF.
            var_H = float(np.mean((H - self.H0_) ** 2))
            self.var_H_ = var_H

            if var_H <= self.eps:
                self.NH_ = float(A)
            else:
                self.NH_ = float(2.0 * self.H0_**2 / var_H)

        else:
            raise ValueError(
                "h_dof_method must be either 'theoretical' or 'moment'."
            )

        # ------------------------------------------------------------
        # Q distribution
        # ------------------------------------------------------------
        eigenvalues = np.asarray(self.pca_.eigenvalues_, dtype=float)

        # Residual covariance eigenvalues after the retained A components.
        residual_eigenvalues = eigenvalues[A:]
        residual_eigenvalues = residual_eigenvalues[
            residual_eigenvalues > self.eps
        ]

        if self.q_dof_method == "box" and residual_eigenvalues.size > 0:
            q1 = float(np.sum(residual_eigenvalues))
            q2 = float(np.sum(residual_eigenvalues ** 2))

            self.q1_residual_ = q1
            self.q2_residual_ = q2

            self.Q0_ = q1

            if q2 <= self.eps:
                self.NQ_ = float(max(self.n_features_ - A, 1))
            else:
                self.NQ_ = float(q1**2 / q2)

            self.var_Q_ = float(2.0 * self.Q0_**2 / self.NQ_)

        elif self.q_dof_method == "moment":
            # Empirical fallback based on the observed Q distribution.
            self.Q0_ = float(np.mean(Q))
            var_Q = float(np.mean((Q - self.Q0_) ** 2))
            self.var_Q_ = var_Q

            if var_Q <= self.eps or self.Q0_ <= self.eps:
                self.NQ_ = float(max(self.n_features_ - A, 1))
            else:
                self.NQ_ = float(2.0 * self.Q0_**2 / var_Q)

        else:
            # Degenerate case: residual eigenvalues are numerically zero.
            # This can happen when the retained PCs reconstruct almost all variation.
            self.Q0_ = float(np.mean(Q))
            self.var_Q_ = float(np.mean((Q - self.Q0_) ** 2))

            if self.Q0_ <= self.eps:
                self.NQ_ = float(max(self.n_features_ - A, 1))
            elif self.var_Q_ <= self.eps:
                self.NQ_ = float(max(self.n_features_ - A, 1))
            else:
                self.NQ_ = float(2.0 * self.Q0_**2 / self.var_Q_)

        # ------------------------------------------------------------
        # Safety
        # ------------------------------------------------------------
        self.H0_ = max(float(self.H0_), self.eps)
        self.Q0_ = max(float(self.Q0_), self.eps)

        self.NH_ = max(float(self.NH_), self.eps)
        self.NQ_ = max(float(self.NQ_), self.eps)


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

    def fit_empirical_limits(self, alpha=None):
        """
        Compute empirical H and Q limits from training distances.

        This is useful for comparing chi-square theoretical limits
        with empirical quantile-based limits.
        """
        if alpha is None:
            alpha = self.alpha
        H = np.asarray(self.H_train_, dtype=float)
        Q = np.asarray(self.Q_train_, dtype=float)
        self.H_empirical_limit_ = float(np.quantile(H, 1.0 - alpha))
        self.Q_empirical_limit_ = float(np.quantile(Q, 1.0 - alpha))
        C_chi2 = H / self.H_limit_ + Q / self.Q_limit_
        self.C_alt_empirical_limit_chi2_HQ_ = float(
            np.quantile(C_chi2, 1.0 - alpha)
        )
        C_emp = H / self.H_empirical_limit_ + Q / self.Q_empirical_limit_
        self.C_alt_empirical_limit_empirical_HQ_ = float(
            np.quantile(C_emp, 1.0 - alpha)
        )
        return self

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
            "var_H": self.var_H_,
            "var_Q": self.var_Q_,
            "NH": self.NH_,
            "NQ": self.NQ_,
        }


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
        X = as_2d_array(X)
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
        X = as_2d_array(X)
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
        X = as_2d_array(X)
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