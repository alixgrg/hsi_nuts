from __future__ import annotations

import numpy as np
from scipy.stats import chi2


class BaseSIMCARule:
    """Base interface for SIMCA decision rules."""

    name = "base"

    def fit(self, model):
        return self

    def statistic(self, H, Q, model):
        raise NotImplementedError

    def limit(self, model):
        raise NotImplementedError

    def accept(self, H, Q, model):
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
        H / H_limit + Q / Q_limit < threshold
    """

    name = "alternative"

    def __init__(
        self,
        threshold=2.0,
        limit_mode="chi2",
        threshold_mode="fixed",
        eps=1e-12,
    ):
        self.threshold = threshold
        self.limit_mode = limit_mode
        self.threshold_mode = threshold_mode
        self.eps = eps

    def _get_HQ_limits(self, model):
        if self.limit_mode == "chi2":
            return model.H_limit_, model.Q_limit_

        if self.limit_mode == "empirical":
            if not hasattr(model, "H_empirical_limit_"):
                model.fit_empirical_limits()
            return model.H_empirical_limit_, model.Q_empirical_limit_

        raise ValueError("limit_mode must be 'chi2' or 'empirical'.")

    def statistic(self, H, Q, model):
        H_limit, Q_limit = self._get_HQ_limits(model)
        return H / max(H_limit, self.eps) + Q / max(Q_limit, self.eps)

    def fit(self, model):
        if not hasattr(model, "H_empirical_limit_"):
            model.fit_empirical_limits()

        if self.threshold_mode == "empirical":
            H = model.H_train_
            Q = model.Q_train_
            C = self.statistic(H, Q, model)
            self.empirical_threshold_ = float(np.quantile(C, 1.0 - model.alpha))

        return self

    def limit(self, model):
        if self.threshold_mode == "fixed":
            return float(self.threshold)

        if self.threshold_mode == "empirical":
            if not hasattr(self, "empirical_threshold_"):
                self.fit(model)
            return float(self.empirical_threshold_)

        raise ValueError("threshold_mode must be 'fixed' or 'empirical'.")

    def accept(self, H, Q, model):
        return self.statistic(H, Q, model) < self.limit(model)


class CombinedIndexSIMCARule(BaseSIMCARule):
    """
    Combined-index SIMCA.

    C = H / H_limit + Q / Q_limit

    C_limit is estimated with a scaled chi-square approximation.
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
            NC = 2.0 * C0**2 / var_C

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
        return self.statistic(H, Q, model) < self.limit(model)


class DataDrivenSIMCARule(BaseSIMCARule):
    """
    Data-driven SIMCA rule.

    D = NQ * Q / Q0 + NH * H / H0
    """

    name = "data_driven"

    def statistic(self, H, Q, model):
        return (
            model.NQ_ * Q / max(model.Q0_, model.eps)
            + model.NH_ * H / max(model.H0_, model.eps)
        )

    def limit(self, model):
        ND = model.NQ_ + model.NH_
        return chi2.ppf(1.0 - model.alpha, ND)

    def accept(self, H, Q, model):
        return self.statistic(H, Q, model) < self.limit(model)


def make_simca_rule(rule_name: str = "alternative", **kwargs):
    """
    Factory for standard SIMCA rule objects.
    """
    rule_name = str(rule_name).lower()

    if rule_name == "simple":
        return SimpleSIMCARule(**kwargs)

    if rule_name in {"alternative", "alt"}:
        return AltSIMCARule(**kwargs)

    if rule_name == "combined_index":
        return CombinedIndexSIMCARule(**kwargs)

    if rule_name == "data_driven":
        return DataDrivenSIMCARule(**kwargs)

    raise ValueError(
        "rule_name must be one of: "
        "'simple', 'alternative', 'alt', 'combined_index', 'data_driven'."
    )


def _require_cv_thresholds(cv_thresholds: dict | None, required_keys: list[str]):
    if cv_thresholds is None:
        raise ValueError("cv_thresholds must be provided for this empirical rule variant.")

    missing = [key for key in required_keys if key not in cv_thresholds]
    if missing:
        raise KeyError(f"Missing CV threshold(s): {missing}")


def compute_rule_variant_stat_limit(
    H,
    Q,
    model,
    variant_name: str,
    cv_thresholds: dict | None = None,
):
    """
    Compute statistic and limit for one SIMCA rule variant.

    This is used for empirical-CV variants such as:
    - simple_emp_cv
    - alternative_chi2_emp_cv
    - alternative_empHQ_emp_cv
    - data_driven_emp_cv
    """
    variant_name = str(variant_name)

    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if variant_name == "simple_chi2":
        stat = np.maximum(H / model.H_limit_, Q / model.Q_limit_)
        limit = 1.0
        return stat, limit

    if variant_name == "simple_emp_cv":
        _require_cv_thresholds(cv_thresholds, ["simple_emp_cv"])
        stat = np.maximum(H / model.H_limit_, Q / model.Q_limit_)
        limit = cv_thresholds["simple_emp_cv"]
        return stat, limit

    if variant_name == "alternative_chi2_fixed2":
        stat = H / model.H_limit_ + Q / model.Q_limit_
        limit = 2.0
        return stat, limit

    if variant_name == "alternative_chi2_emp_cv":
        _require_cv_thresholds(cv_thresholds, ["alternative_chi2_emp_cv"])
        stat = H / model.H_limit_ + Q / model.Q_limit_
        limit = cv_thresholds["alternative_chi2_emp_cv"]
        return stat, limit

    if variant_name == "alternative_empHQ_fixed2":
        _require_cv_thresholds(cv_thresholds, ["H_emp_cv", "Q_emp_cv"])
        stat = H / cv_thresholds["H_emp_cv"] + Q / cv_thresholds["Q_emp_cv"]
        limit = 2.0
        return stat, limit

    if variant_name == "alternative_empHQ_emp_cv":
        _require_cv_thresholds(
            cv_thresholds,
            ["H_emp_cv", "Q_emp_cv", "alternative_empHQ_emp_cv"],
        )
        stat = H / cv_thresholds["H_emp_cv"] + Q / cv_thresholds["Q_emp_cv"]
        limit = cv_thresholds["alternative_empHQ_emp_cv"]
        return stat, limit

    if variant_name == "data_driven_chi2":
        stat = (
            model.NQ_ * Q / max(model.Q0_, model.eps)
            + model.NH_ * H / max(model.H0_, model.eps)
        )
        limit = chi2.ppf(1.0 - model.alpha, model.NQ_ + model.NH_)
        return stat, limit

    if variant_name == "data_driven_emp_cv":
        _require_cv_thresholds(cv_thresholds, ["data_driven_emp_cv"])
        stat = (
            model.NQ_ * Q / max(model.Q0_, model.eps)
            + model.NH_ * H / max(model.H0_, model.eps)
        )
        limit = cv_thresholds["data_driven_emp_cv"]
        return stat, limit

    if variant_name == "combined_index_chi2":
        rule = CombinedIndexSIMCARule()
        rule.fit(model)
        stat = rule.statistic(H, Q, model)
        limit = rule.limit(model)
        return stat, limit

    if variant_name == "combined_index_emp_cv":
        _require_cv_thresholds(cv_thresholds, ["combined_index_emp_cv"])
        stat = H / model.H_limit_ + Q / model.Q_limit_
        limit = cv_thresholds["combined_index_emp_cv"]
        return stat, limit

    raise ValueError(f"Unknown rule variant: {variant_name}")


def accept_rule_variant(
    H,
    Q,
    model,
    variant_name: str,
    cv_thresholds: dict | None = None,
):
    """Return boolean acceptance for a rule variant."""
    stat, limit = compute_rule_variant_stat_limit(
        H=H,
        Q=Q,
        model=model,
        variant_name=variant_name,
        cv_thresholds=cv_thresholds,
    )
    return stat < limit
