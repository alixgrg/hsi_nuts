from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
import gc
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from src import experiment_config as expcfg
from src.decision.aggregation import object_threshold_grid
from src.decision.border import add_border_flags_to_pixel_df
from src.decision.labels import (
    DEFAULT_NON_TARGET_LABEL,
    DEFAULT_TARGET_CLASS,
    pixel_ratio_col,
    predicted_col,
    true_col,
)
from src.decision.metrics import binary_detection_metrics, summarize_pixel_errors_by_image
from src.decision.truth import add_pixel_truth_labels
from src.matrices.matrix_registry import build_matrix, matrix_family_from_method
from src.models.simca import SIMCAClassModel
from src.models.simca_rules import compute_rule_variant_stat_limit, make_simca_rule
from src.protocol_governance import sha256_file
from src.spectra.preprocessing import SpectralPreprocessor
from src.spectra.preprocessing_configs import normalize_preprocessing_configs
from src.utils import (
    as_list,
    load_parquet,
    parse_preprocessing_steps,
    row_float,
    row_int,
    row_str,
    row_value,
    save_parquet,
)
from src.workflows.simca_selection_utils import (
    materialize_selection_metrics,
    sort_detection_selection,
)

# -----------------------------------------------------------------------------
# Small SIMCA workflow helpers
# -----------------------------------------------------------------------------


def uses_sg(preprocessing_steps) -> bool:
    """Return True when a preprocessing pipeline contains a Savitzky-Golay step."""
    return any(str(step).startswith("sg_") for step in tuple(preprocessing_steps))


def valid_sg_parameter_pairs(
    preprocessing_steps,
    sg_window_length_values=(11,),
    sg_polyorder_values=(2,),
    default_sg_window_length: int = 11,
    default_sg_polyorder: int = 2,
) -> list[tuple[int, int]]:
    """Return valid SG parameter pairs without expanding grids when SG is unused."""
    if not uses_sg(preprocessing_steps):
        return [(int(default_sg_window_length), int(default_sg_polyorder))]

    pairs: list[tuple[int, int]] = []
    for window in as_list(sg_window_length_values):
        for poly in as_list(sg_polyorder_values):
            window = int(window)
            poly = int(poly)
            if window <= 0 or window % 2 == 0:
                continue
            if poly >= window:
                continue
            pairs.append((window, poly))

    if not pairs:
        raise ValueError(
            "No valid Savitzky-Golay parameter pair. "
            "Require odd sg_window_length and sg_polyorder < sg_window_length."
        )
    return pairs


def _is_preprocessing_configs_by_family(preprocessing_configs) -> bool:
    if not isinstance(preprocessing_configs, Mapping):
        return False
    known_families = {"object_matrix", "pixel_matrix", "unknown_matrix_family"}
    return any(str(key) in known_families for key in preprocessing_configs.keys())


def _normalize_preprocessing_configs_by_family(preprocessing_configs) -> dict[str, dict[str, tuple[str, ...]]]:
    """Normalize flat or family-specific preprocessing configs."""
    if _is_preprocessing_configs_by_family(preprocessing_configs):
        out: dict[str, dict[str, tuple[str, ...]]] = {}
        for family, configs in preprocessing_configs.items():
            out[str(family)] = normalize_preprocessing_configs(configs)
        return out

    flat = normalize_preprocessing_configs(preprocessing_configs)
    return {
        "object_matrix": flat,
        "pixel_matrix": flat,
        "unknown_matrix_family": flat,
    }


def _preprocessing_configs_for_family(
    preprocessing_configs_by_family: Mapping[str, Mapping[str, Sequence[str]]],
    matrix_family: str,
) -> Mapping[str, Sequence[str]]:
    return preprocessing_configs_by_family.get(str(matrix_family), {})


def _resolve_preprocessing_steps_for_row(
    row: pd.Series,
    preprocessing_configs,
) -> tuple[str, ...]:
    preprocessing_name = str(row["preprocessing"])
    configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)
    matrix_family = row_str(
        row,
        "matrix_family",
        matrix_family_from_method(row_str(row, "matrix_method", "")),
    )
    configs = _preprocessing_configs_for_family(configs_by_family, matrix_family)
    if preprocessing_name in configs:
        return tuple(configs[preprocessing_name])
    return tuple(parse_preprocessing_steps(row_value(row, "preprocessing_steps", preprocessing_name)))


def balanced_strategy_grid_for_matrix(
    matrix_method: str,
    m_values=(40,),
    balanced_pixel_strategy_values=("random",),
    default_m: int = 40,
) -> list[dict[str, Any]]:
    """Return valid pixel-sampling configurations for a matrix method."""
    matrix_method = str(matrix_method)

    if matrix_method == "balanced_pixels":
        configs = []
        for m in as_list(m_values):
            for strategy in as_list(balanced_pixel_strategy_values):
                configs.append(
                    {
                        "m": int(m),
                        "m_effective": int(m),
                        "balanced_pixel_strategy": str(strategy),
                        "balanced_pixel_strategy_effective": str(strategy),
                        "training_matrix_id": f"balanced_pixel_{strategy}_m{int(m)}",
                    }
                )
        return configs

    return [
        {
            "m": np.nan,
            "m_effective": int(default_m),
            "balanced_pixel_strategy": "not_applicable",
            "balanced_pixel_strategy_effective": "random",
            "training_matrix_id": matrix_method,
        }
    ]


def standard_grid_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort grid-search results lexicographically, with FN first."""
    return sort_detection_selection(df)


def make_target_train_filters(
    target_class: str = DEFAULT_TARGET_CLASS,
    train_batches=None,
    split=None,
    class_col: str = "object_nut_type",
) -> dict[str, Any]:
    """Build filters selecting pure target-class objects for one-class SIMCA training."""
    filters: dict[str, Any] = {
        "sample_kind": ["pure"],
        class_col: [target_class],
    }
    if train_batches is not None:
        filters["batch"] = list(train_batches)
    if split is not None:
        filters["split"] = [split] if isinstance(split, str) else list(split)
    return filters


def _empirical_quantile(values, q: float) -> float:
    """Conservative empirical quantile using method='higher' when available."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def _make_group_splitter(groups, n_splits: int | None = None):
    """Use LeaveOneGroupOut unless a smaller GroupKFold split count is requested."""
    groups = np.asarray(groups).astype(str)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Need at least two groups for group cross-validation.")
    if n_splits is None or int(n_splits) >= n_groups:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=int(n_splits))


# -----------------------------------------------------------------------------
# Standard SIMCA fit / projection
# -----------------------------------------------------------------------------


@dataclass
class SimcaFitBundle:
    """Reusable train-only preprocessing and SIMCA fit."""

    target_class: str
    preprocessing_steps: tuple[str, ...]
    preprocessor: SpectralPreprocessor
    model: SIMCAClassModel
    X_train_raw: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    metadata: dict[str, np.ndarray]
    wavelengths: np.ndarray | None
    raw_rank: int
    preprocessed_rank: int
    preprocessing_seconds: float
    fit_seconds: float
    train_scores: pd.DataFrame


def fit_simca_bundle_from_matrix(
    X_train_raw,
    y_train,
    metadata,
    *,
    preprocessing_spec,
    n_components: int,
    alpha: float,
    wavelengths=None,
    target_class: str = DEFAULT_TARGET_CLASS,
    fitted_preprocessor: SpectralPreprocessor | None = None,
    X_train_preprocessed=None,
) -> SimcaFitBundle:
    """Fit one reusable SIMCA bundle from a raw train matrix only."""
    X_raw = np.asarray(X_train_raw, dtype=float)
    y = np.asarray(y_train).astype(str)
    if X_raw.ndim != 2 or X_raw.shape[0] != len(y) or X_raw.size == 0:
        raise ValueError("The SIMCA train matrix must be non-empty and aligned.")
    if not np.isfinite(X_raw).all():
        raise ValueError("The SIMCA train matrix contains non-finite values.")
    target_mask = y == str(target_class)
    if not target_mask.any():
        raise ValueError("The SIMCA train matrix contains no target observation.")
    X_target_raw = X_raw[target_mask]
    target_metadata = {
        str(key): np.asarray(values)[target_mask]
        for key, values in dict(metadata).items()
    }
    if isinstance(preprocessing_spec, Mapping):
        steps = tuple(preprocessing_spec.get("steps", ("raw",)))
        sg_window_length = int(preprocessing_spec.get("sg_window_length", 9))
        sg_polyorder = int(preprocessing_spec.get("sg_polyorder", 2))
    else:
        steps = tuple(preprocessing_spec)
        sg_window_length = 9
        sg_polyorder = 2
    raw_rank = int(
        np.linalg.matrix_rank(
            X_target_raw - X_target_raw.mean(axis=0, keepdims=True)
        )
    )
    if fitted_preprocessor is None:
        preprocessor = SpectralPreprocessor(
            steps=steps,
            sg_window_length=sg_window_length,
            sg_polyorder=sg_polyorder,
        )
        start = perf_counter()
        X_target = preprocessor.fit_transform(
            X_target_raw,
            wavelengths=wavelengths,
        )
        preprocessing_seconds = perf_counter() - start
    else:
        if X_train_preprocessed is None:
            raise ValueError(
                "X_train_preprocessed is required with fitted_preprocessor."
            )
        preprocessor = fitted_preprocessor
        X_target = np.asarray(X_train_preprocessed, dtype=float)
        if X_target.shape[0] != X_target_raw.shape[0]:
            raise ValueError("Reused preprocessed train matrix is misaligned.")
        preprocessing_seconds = 0.0
    if not np.isfinite(X_target).all():
        raise ValueError("Preprocessing produced non-finite train values.")
    preprocessed_rank = int(
        np.linalg.matrix_rank(
            X_target - X_target.mean(axis=0, keepdims=True)
        )
    )
    k = int(n_components)
    if k < 1 or k > min(preprocessed_rank, len(X_target) - 1):
        raise ValueError(
            "n_components is incompatible with target count/rank: "
            f"k={k}, n_target={len(X_target)}, rank={preprocessed_rank}."
        )
    model = SIMCAClassModel(
        class_name=str(target_class),
        n_components=k,
        alpha=float(alpha),
    )
    start = perf_counter()
    model.fit(X_target)
    fit_seconds = perf_counter() - start
    pca_scores = np.asarray(model.pca_.scores_, dtype=float)
    train_scores = pd.DataFrame(
        {
            "pca_score_pc1": pca_scores[:, 0],
            "pca_score_pc2": (
                pca_scores[:, 1]
                if pca_scores.shape[1] > 1
                else np.full(len(pca_scores), np.nan)
            ),
            "H": np.asarray(model.H_train_, dtype=float),
            "Q": np.asarray(model.Q_train_, dtype=float),
        }
    )
    return SimcaFitBundle(
        target_class=str(target_class),
        preprocessing_steps=steps,
        preprocessor=preprocessor,
        model=model,
        X_train_raw=X_target_raw,
        X_train=X_target,
        y_train=y[target_mask],
        metadata=target_metadata,
        wavelengths=(
            None if wavelengths is None else np.asarray(wavelengths, dtype=float)
        ),
        raw_rank=raw_rank,
        preprocessed_rank=preprocessed_rank,
        preprocessing_seconds=float(preprocessing_seconds),
        fit_seconds=float(fit_seconds),
        train_scores=train_scores,
    )


def project_simca_bundle(
    bundle: SimcaFitBundle,
    *,
    object_db,
    projection_matrix_method: str,
    projection_filters: Mapping[str, Any] | None,
    projection_data: tuple[Any, Any, Mapping[str, Any]] | None = None,
    projection_cache: Mapping[str, Any] | None = None,
    rule_variant: str | None = None,
    rule: Any | None = None,
    train_only_thresholds: Mapping[str, float] | None = None,
    target_class: str | None = None,
    m: int = 10,
    random_state: int = 0,
    balanced_pixel_strategy: str = "random",
    under_m_policy: str = "exclude",
) -> pd.DataFrame:
    """Project a reusable fit and return canonical signed SIMCA margins."""
    if projection_cache is None:
        projection_cache = prepare_simca_projection(
            bundle,
            object_db=object_db,
            projection_matrix_method=projection_matrix_method,
            projection_filters=projection_filters,
            projection_data=projection_data,
            m=m,
            random_state=random_state,
            balanced_pixel_strategy=balanced_pixel_strategy,
            under_m_policy=under_m_policy,
        )
    X_raw = projection_cache["X_raw"]
    y = projection_cache["y"]
    metadata = projection_cache["metadata"]
    X = projection_cache["X"]
    values = projection_cache["values"]
    if not np.isfinite(X).all():
        raise ValueError("Projection preprocessing produced non-finite values.")
    H = np.asarray(values["H"], dtype=float)
    Q = np.asarray(values["Q"], dtype=float)
    pca_scores = np.asarray(values["scores"], dtype=float)
    if rule_variant is None:
        if rule is None:
            raise ValueError("Provide rule_variant or a fitted legacy rule.")
        statistic = rule.statistic(H, Q, bundle.model)
        limit = rule.limit(bundle.model)
    else:
        statistic, limit = compute_rule_variant_stat_limit(
            H=H,
            Q=Q,
            model=bundle.model,
            variant_name=str(rule_variant),
            cv_thresholds=train_only_thresholds,
        )
    statistic = np.asarray(statistic, dtype=float)
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            f"SIMCA rule limit must be finite and positive, got {limit}."
        )
    if not np.isfinite(statistic).all():
        raise ValueError("SIMCA rule statistics contain non-finite values.")
    ratio = statistic / limit
    margin = 1.0 - ratio
    target = str(bundle.target_class if target_class is None else target_class)
    out = pd.DataFrame(metadata)
    out["truth"] = np.asarray(y).astype(str) == target
    out["truth_level"] = (
        "object_label_inherited_weak"
        if str(projection_matrix_method) in {"pixel", "all_pixels"}
        else "pure_reference_object"
    )
    out["pca_score_pc1"] = pca_scores[:, 0]
    out["pca_score_pc2"] = (
        pca_scores[:, 1]
        if pca_scores.shape[1] > 1
        else np.full(len(pca_scores), np.nan)
    )
    out["H"] = H
    out["Q"] = Q
    out["rule_statistic"] = statistic
    out["rule_limit"] = limit
    out["normalized_ratio"] = ratio
    out["simca_margin"] = margin
    out["direct_2way_decision"] = margin >= 0.0
    out["projection_matrix_method"] = str(projection_matrix_method)
    out.attrs["simca_values"] = values
    out.attrs["X_transformed"] = X
    return out


def prepare_simca_projection(
    bundle: SimcaFitBundle,
    *,
    object_db,
    projection_matrix_method: str,
    projection_filters: Mapping[str, Any] | None,
    projection_data: tuple[Any, Any, Mapping[str, Any]] | None = None,
    m: int = 10,
    random_state: int = 0,
    balanced_pixel_strategy: str = "random",
    under_m_policy: str = "exclude",
) -> dict[str, Any]:
    """Build and transform one projection matrix once for all SIMCA rules."""
    if projection_data is None:
        X_raw, y, metadata = build_matrix(
            object_db=object_db,
            matrix_method=str(projection_matrix_method),
            filters=dict(projection_filters or {}),
            m=int(m),
            random_state=int(random_state),
            balanced_pixel_strategy=str(balanced_pixel_strategy),
            under_m_policy=str(under_m_policy),
        )
    else:
        X_raw, y, metadata = projection_data
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y)
    metadata = {
        str(key): np.asarray(value)
        for key, value in dict(metadata).items()
    }
    validity = bundle.preprocessor.input_validity_report(X_raw)
    valid_mask = np.asarray(validity["valid_mask"], dtype=bool)
    if not valid_mask.any():
        raise ValueError(
            "Projection contains no row compatible with the fitted "
            "preprocessing chain."
        )
    if len(y) != len(X_raw):
        raise ValueError("Projection labels are not aligned with X rows.")
    for key, values in metadata.items():
        if values.ndim == 0 or len(values) != len(X_raw):
            raise ValueError(
                f"Projection metadata[{key!r}] are not aligned with X rows."
            )
    X_raw_valid = X_raw[valid_mask]
    y_valid = y[valid_mask]
    metadata_valid = {
        key: values[valid_mask]
        for key, values in metadata.items()
    }
    X = bundle.preprocessor.transform(X_raw_valid)
    if not np.isfinite(X).all():
        raise ValueError("Projection preprocessing produced non-finite values.")
    return {
        "X_raw": X_raw_valid,
        "y": y_valid,
        "metadata": metadata_valid,
        "X": X,
        "values": bundle.model.decision_values(X),
        "input_validity": {
            key: value
            for key, value in validity.items()
            if key != "valid_mask"
        },
    }


def run_locked_simca_validation_refit(
    validation_executions: pd.DataFrame,
    *,
    object_db: Mapping[str, Mapping[str, Any]],
    wavelengths=None,
    train_batches: Sequence[int] = expcfg.SIMCA_CONCAT_REFIT_TRAIN_BATCHES,
    projection_batches: Sequence[int] = (
        expcfg.SIMCA_CONCAT_REFIT_PROJECTION_BATCHES
    ),
    target_class: str = expcfg.TARGET_CLASS,
    non_target_label: str = expcfg.NON_TARGET_LABEL,
    under_m_policy: str = expcfg.INTERNAL_CALIBRATION_UNDER_M_POLICY,
    border_width: int = expcfg.SIMCA_CONCAT_REFIT_BORDER_WIDTH,
    verbose: bool = expcfg.SIMCA_CONCAT_REFIT_VERBOSE,
) -> dict[str, pd.DataFrame]:
    """Refit the selected 03B executions and project batch 3 once per projection.

    No 04C-specific scientific identifier is created. ``fit_id`` and
    ``projection_id`` are reused exactly from 03B. Continuous predictions are
    persisted once per ``projection_id``; model-specific thresholds are applied
    later by ``evaluate_locked_validation_predictions``.
    """
    required = {
        "model_id",
        "random_state",
        "fit_id",
        "projection_id",
        "track_id",
        "decision_mode",
        "projection_level",
        "matrix_method",
        "projection_matrix_method",
        "m",
        "balanced_pixel_strategy",
        "preprocessing_steps",
        "rule_variant",
        "limit_source",
        "n_components",
        "alpha",
        "sg_window_length",
        "sg_polyorder",
    }
    missing = sorted(required - set(validation_executions.columns))
    if missing:
        raise KeyError(f"Missing locked validation-execution columns: {missing}")

    empty_outputs = {
        "object_predictions": pd.DataFrame(
            columns=expcfg.SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS
        ),
        "pixel_predictions": pd.DataFrame(
            columns=expcfg.SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS
        ),
        "technical_events": pd.DataFrame(
            columns=expcfg.SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS
        ),
    }
    if validation_executions.empty:
        return empty_outputs

    executions = validation_executions.copy()
    executions["model_id"] = executions["model_id"].astype(str)
    executions["fit_id"] = executions["fit_id"].astype(str)
    executions["projection_id"] = executions["projection_id"].astype(str)
    executions["random_state"] = pd.to_numeric(
        executions["random_state"], errors="raise"
    ).astype(int)
    run_keys = ["model_id", "random_state"]
    if executions.duplicated(run_keys).any():
        raise RuntimeError("Validation executions duplicate (model_id, random_state).")
    if executions[["fit_id", "projection_id"]].eq("").any().any():
        raise RuntimeError("fit_id and projection_id must be non-empty.")

    train_batch_set = set(map(int, train_batches))
    projection_batch_set = set(map(int, projection_batches))
    if train_batch_set & projection_batch_set:
        raise RuntimeError("Training and validation batches must be disjoint.")
    forbidden = set(map(int, expcfg.SIMCA_CONCAT_REFIT_FORBIDDEN_BATCHES))
    if forbidden & (train_batch_set | projection_batch_set):
        raise RuntimeError("04C cannot load a forbidden protocol batch.")

    # A fit_id must map to one exact technical fit definition.
    fit_definition_columns = (
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "preprocessing_steps",
        "n_components",
        "alpha",
        "sg_window_length",
        "sg_polyorder",
        "random_state",
    )
    fit_identity = executions.groupby("fit_id", dropna=False)[
        list(fit_definition_columns)
    ].nunique(dropna=False)
    if len(fit_identity) and fit_identity.max(axis=1).gt(1).any():
        raise RuntimeError("A fit_id maps to multiple technical fit definitions.")

    # A projection_id must map to one exact continuous projection definition.
    projection_definition_columns = (
        "fit_id",
        "projection_level",
        "projection_matrix_method",
        "rule_variant",
        "limit_source",
    )
    projection_identity = executions.groupby("projection_id", dropna=False)[
        list(projection_definition_columns)
    ].nunique(dropna=False)
    if len(projection_identity) and projection_identity.max(axis=1).gt(1).any():
        raise RuntimeError(
            "A projection_id maps to multiple continuous projection definitions."
        )

    # Import locally to avoid a module-level cycle: 03B itself imports the
    # reusable fit/projection primitives from this module.
    from src.workflows.simca_internal_calibration import (
        compute_train_only_rule_thresholds,
    )

    object_area = {
        str(object_id): float(
            record.get(
                "area_pixels",
                record.get("n_pixels", len(record.get("spectra", ()))),
            )
        )
        for object_id, record in object_db.items()
    }

    object_parts: list[pd.DataFrame] = []
    pixel_parts: list[pd.DataFrame] = []
    technical_events: list[dict[str, Any]] = []

    def record_error(rows: pd.DataFrame, stage: str, exc: Exception) -> None:
        pairs = rows[["fit_id", "projection_id"]].drop_duplicates()
        for pair in pairs.itertuples(index=False):
            technical_events.append(
                {
                    "fit_id": str(pair.fit_id),
                    "projection_id": str(pair.projection_id),
                    "stage": str(stage),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    # Raw matrix construction depends only on these columns. Preprocessing,
    # n_components and rule variants are deliberately handled downstream so
    # that no surrogate data_config_id is needed.
    matrix_group_columns = (
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "random_state",
    )
    matrix_groups = list(
        executions.groupby(
            list(matrix_group_columns),
            sort=False,
            dropna=False,
        )
    )

    train_filters = {
        "sample_kind": ["pure"],
        "object_nut_type": [str(target_class)],
        "batch": sorted(train_batch_set),
    }
    projection_filters = {
        "sample_kind": ["pure"],
        "object_nut_type": [str(target_class), str(non_target_label)],
        "batch": sorted(projection_batch_set),
    }

    for group_index, (_, data_rows) in enumerate(matrix_groups, start=1):
        base = data_rows.iloc[0]
        if verbose:
            print(
                f"[04C matrix {group_index}/{len(matrix_groups)}] "
                f"{base['matrix_method']} | m={base['m']} | "
                f"strategy={base['balanced_pixel_strategy']} | "
                f"seed={base['random_state']}"
            )

        m_value = (
            int(base["m"])
            if not pd.isna(base["m"])
            else int(expcfg.PCA_BALANCED_M_VALUES[0])
        )
        strategy = str(base["balanced_pixel_strategy"])
        if strategy == "not_applicable":
            strategy = "random"

        try:
            X_train_raw, y_train, train_metadata = build_matrix(
                object_db=object_db,
                matrix_method=str(base["matrix_method"]),
                filters=train_filters,
                m=m_value,
                random_state=int(base["random_state"]),
                replace=False,
                balanced_pixel_strategy=strategy,
                under_m_policy=str(under_m_policy),
            )
        except Exception as exc:
            record_error(data_rows, "train_matrix_build", exc)
            continue

        # Raw validation matrices are independent of the fitted preprocessor.
        projection_data: dict[str, tuple[Any, Any, Mapping[str, Any]]] = {}
        projection_matrix_errors: dict[str, Exception] = {}
        for method in sorted(
            data_rows["projection_matrix_method"].astype(str).unique()
        ):
            method_rows = data_rows.loc[
                data_rows["projection_matrix_method"].astype(str).eq(method)
            ]
            try:
                projection_data[method] = build_matrix(
                    object_db=object_db,
                    matrix_method=method,
                    filters=projection_filters,
                    under_m_policy=str(under_m_policy),
                )
            except Exception as exc:
                projection_matrix_errors[method] = exc
                record_error(method_rows, "projection_matrix_build", exc)

        # Reuse preprocessing inside one raw-matrix group whenever several
        # selected fits differ only by PCA dimension / alpha.
        preprocessing_group_columns = (
            "preprocessing_steps",
            "sg_window_length",
            "sg_polyorder",
        )
        for _, preprocessing_rows in data_rows.groupby(
            list(preprocessing_group_columns),
            sort=False,
            dropna=False,
        ):
            fitted_preprocessor = None
            X_train_preprocessed = None
            fit_groups = list(
                preprocessing_rows.groupby("fit_id", sort=False, dropna=False)
            )
            fit_groups.sort(key=lambda item: int(item[1].iloc[0]["n_components"]))

            for fit_id, fit_rows in fit_groups:
                fit_base = fit_rows.iloc[0]
                try:
                    bundle = fit_simca_bundle_from_matrix(
                        X_train_raw,
                        y_train,
                        train_metadata,
                        preprocessing_spec={
                            "steps": tuple(
                                parse_preprocessing_steps(
                                    fit_base["preprocessing_steps"]
                                )
                            ),
                            "sg_window_length": int(
                                fit_base["sg_window_length"]
                            ),
                            "sg_polyorder": int(fit_base["sg_polyorder"]),
                        },
                        n_components=int(fit_base["n_components"]),
                        alpha=float(fit_base["alpha"]),
                        wavelengths=wavelengths,
                        target_class=str(target_class),
                        fitted_preprocessor=fitted_preprocessor,
                        X_train_preprocessed=X_train_preprocessed,
                    )
                    if fitted_preprocessor is None:
                        fitted_preprocessor = bundle.preprocessor
                        X_train_preprocessed = bundle.X_train
                    train_only_thresholds = compute_train_only_rule_thresholds(
                        bundle.model,
                        alpha=float(fit_base["alpha"]),
                    )
                except Exception as exc:
                    record_error(fit_rows, "fit", exc)
                    continue

                projection_caches: dict[str, dict[str, Any]] = {}
                cache_errors: dict[str, Exception] = {}
                for method in sorted(
                    fit_rows["projection_matrix_method"].astype(str).unique()
                ):
                    method_rows = fit_rows.loc[
                        fit_rows["projection_matrix_method"].astype(str).eq(method)
                    ]
                    if method in projection_matrix_errors:
                        continue
                    try:
                        projection_caches[method] = prepare_simca_projection(
                            bundle,
                            object_db=object_db,
                            projection_matrix_method=method,
                            projection_filters=projection_filters,
                            projection_data=projection_data[method],
                            under_m_policy=str(under_m_policy),
                        )
                    except Exception as exc:
                        cache_errors[method] = exc
                        record_error(method_rows, "projection_preparation", exc)

                # Several scientific models may share one projection_id. The
                # group is projected exactly once and written exactly once.
                for projection_id, projection_rows in fit_rows.groupby(
                    "projection_id", sort=False, dropna=False
                ):
                    projection_base = projection_rows.iloc[0]
                    method = str(projection_base["projection_matrix_method"])
                    if method in projection_matrix_errors or method in cache_errors:
                        continue
                    try:
                        projected = project_simca_bundle(
                            bundle,
                            object_db=object_db,
                            projection_matrix_method=method,
                            projection_filters=projection_filters,
                            projection_cache=projection_caches[method],
                            rule_variant=str(projection_base["rule_variant"]),
                            train_only_thresholds=train_only_thresholds,
                            target_class=str(target_class),
                            under_m_policy=str(under_m_policy),
                        )
                        projected.attrs = {}
                        projected["projection_id"] = str(projection_id)
                        projected["object_area"] = (
                            projected["object_id"].astype(str).map(object_area)
                        )
                        if projected["object_area"].isna().any():
                            raise RuntimeError(
                                "A validation observation has no object-area provenance."
                            )
                        projected["batch"] = pd.to_numeric(
                            projected["batch"], errors="raise"
                        ).astype(int)
                        if not set(projected["batch"]).issubset(
                            projection_batch_set
                        ):
                            raise RuntimeError(
                                "A validation projection contains a forbidden batch."
                            )
                        projected = projected.drop(
                            columns=[
                                "direct_2way_decision",
                                "projection_matrix_method",
                            ],
                            errors="ignore",
                        )

                        if str(projection_base["projection_level"]) == "pixel_projection":
                            projected["truth_level"] = (
                                expcfg.SIMCA_CONCAT_REFIT_TRUTH_SOURCE
                            )
                            projected = add_border_flags_to_pixel_df(
                                projected,
                                object_db=dict(object_db),
                                border_width=int(border_width),
                            )
                            pixel_parts.append(
                                projected.reindex(
                                    columns=(
                                        expcfg.SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS
                                    )
                                )
                            )
                        elif str(projection_base["projection_level"]) == "object_projection":
                            object_parts.append(
                                projected.reindex(
                                    columns=(
                                        expcfg.SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS
                                    )
                                )
                            )
                        else:
                            raise RuntimeError(
                                "Unknown projection_level in validation registry: "
                                f"{projection_base['projection_level']!r}."
                            )
                    except Exception as exc:
                        record_error(projection_rows, "projection", exc)

    object_predictions = (
        pd.concat(object_parts, ignore_index=True, sort=False)
        if object_parts
        else empty_outputs["object_predictions"].copy()
    )
    pixel_predictions = (
        pd.concat(pixel_parts, ignore_index=True, sort=False)
        if pixel_parts
        else empty_outputs["pixel_predictions"].copy()
    )
    events = pd.DataFrame(
        technical_events,
        columns=expcfg.SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS,
    ).drop_duplicates().reset_index(drop=True)

    for name, frame, entity_columns in (
        (
            "object",
            object_predictions,
            ("source_image", "object_id"),
        ),
        (
            "pixel",
            pixel_predictions,
            ("source_image", "object_id", "row", "col"),
        ),
    ):
        if frame.empty:
            continue
        keys = ["projection_id", *entity_columns]
        if frame.duplicated(keys).any():
            raise RuntimeError(f"Duplicated {name} validation predictions.")

    return {
        "object_predictions": object_predictions.reindex(
            columns=expcfg.SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS
        ),
        "pixel_predictions": pixel_predictions.reindex(
            columns=expcfg.SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS
        ),
        "technical_events": events.reindex(
            columns=expcfg.SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS
        ),
    }


def run_locked_simca_validation_refit_checkpointed(
    validation_executions: pd.DataFrame,
    *,
    object_db: Mapping[str, Mapping[str, Any]],
    checkpoint_dir: str | Path,
    checkpoint_context: Mapping[str, str],
    resume: bool = expcfg.SIMCA_CONCAT_REFIT_RESUME_FROM_CHECKPOINT,
    **refit_kwargs,
) -> dict[str, pd.DataFrame]:
    """Run 04C with hash-verified shards and no persisted surrogate data ID.

    Shards are organized by a deterministic hash of the raw matrix-build
    parameters. That hash is used only in filenames/markers and is never added
    to scientific result tables.
    """
    import hashlib

    required_context = {
        "protocol_hash",
        "validation_plan_hash",
        "execution_registry_hash",
        "database_sha256",
    }
    context = {str(key): str(value) for key, value in checkpoint_context.items()}
    missing = sorted(key for key in required_context if not context.get(key))
    if missing:
        raise ValueError(f"Incomplete 04C checkpoint context: {missing}")

    required_execution_columns = {
        "model_id",
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "random_state",
        "fit_id",
        "projection_id",
    }
    missing_columns = sorted(
        required_execution_columns - set(validation_executions.columns)
    )
    if missing_columns:
        raise KeyError(
            f"Missing validation-execution columns: {missing_columns}"
        )

    executions = validation_executions.copy()
    executions["model_id"] = executions["model_id"].astype(str)
    executions["fit_id"] = executions["fit_id"].astype(str)
    executions["projection_id"] = executions["projection_id"].astype(str)
    executions["random_state"] = pd.to_numeric(
        executions["random_state"], errors="raise"
    ).astype(int)
    if executions.duplicated(["model_id", "random_state"]).any():
        raise RuntimeError(
            "Validation executions duplicate (model_id, random_state)."
        )

    # A canonical fit/projection may belong to only one raw matrix shard. This
    # is checked before sharding so a corrupted registry cannot silently split
    # one identity across two checkpoint files.
    matrix_group_columns = (
        "matrix_method",
        "m",
        "balanced_pixel_strategy",
        "random_state",
    )
    for identity_column in ("fit_id", "projection_id"):
        membership = executions.groupby(identity_column, dropna=False)[
            list(matrix_group_columns)
        ].nunique(dropna=False)
        if len(membership) and membership.max(axis=1).gt(1).any():
            raise RuntimeError(
                f"A {identity_column} maps to multiple checkpoint matrix groups."
            )

    root = Path(checkpoint_dir) / (
        f"run_{context['validation_plan_hash'][:12]}_"
        f"{context['execution_registry_hash'][:12]}"
    )
    root.mkdir(parents=True, exist_ok=True)

    object_parts: list[pd.DataFrame] = []
    pixel_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []

    def json_scalar(value: Any) -> Any:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    grouped = executions.groupby(
        list(matrix_group_columns),
        sort=False,
        dropna=False,
    )
    for _, group in grouped:
        first = group.iloc[0]
        payload = {
            column: json_scalar(first[column])
            for column in matrix_group_columns
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        group_key = "matrix_" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:16]

        paths = {
            "object_predictions": root / f"{group_key}_objects.parquet",
            "pixel_predictions": root / f"{group_key}_pixels.parquet",
            "technical_events": root / f"{group_key}_events.parquet",
        }
        marker_path = root / f"{group_key}.json"
        loaded: dict[str, pd.DataFrame] | None = None

        if marker_path.exists():
            if not resume:
                raise RuntimeError(
                    "A compatible checkpoint exists but 04C resume is disabled."
                )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for key, expected in context.items():
                if str(marker.get(key)) != expected:
                    raise RuntimeError(
                        f"04C checkpoint {key} mismatch: {marker_path}"
                    )
            if str(marker.get("matrix_group_key")) != group_key:
                raise RuntimeError(
                    f"04C checkpoint matrix-group mismatch: {marker_path}"
                )
            if marker.get("matrix_group_payload") != payload:
                raise RuntimeError(
                    f"04C checkpoint matrix payload mismatch: {marker_path}"
                )
            for name, path in paths.items():
                if not path.exists():
                    raise RuntimeError(f"Incomplete 04C checkpoint: {path}")
                expected_sha = str(marker.get("file_sha256", {}).get(name, ""))
                if not expected_sha or sha256_file(path) != expected_sha:
                    raise RuntimeError(f"Modified 04C checkpoint shard: {path}")
            loaded = {name: load_parquet(path) for name, path in paths.items()}

        if loaded is None:
            loaded = run_locked_simca_validation_refit(
                group,
                object_db=object_db,
                **refit_kwargs,
            )
            for name, path in paths.items():
                save_parquet(loaded[name], path)
            marker = {
                **context,
                "matrix_group_key": group_key,
                "matrix_group_payload": payload,
                "n_execution_rows": int(len(group)),
                "n_fit_ids": int(group["fit_id"].astype(str).nunique()),
                "n_projection_ids": int(
                    group["projection_id"].astype(str).nunique()
                ),
                "file_sha256": {
                    name: sha256_file(path) for name, path in paths.items()
                },
            }
            temporary_marker = marker_path.with_name(
                f".{marker_path.name}.tmp"
            )
            temporary_marker.write_text(
                json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_marker.replace(marker_path)

        object_parts.append(loaded["object_predictions"])
        pixel_parts.append(loaded["pixel_predictions"])
        event_parts.append(loaded["technical_events"])

    outputs = {
        "object_predictions": (
            pd.concat(object_parts, ignore_index=True, sort=False)
            if object_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS
            )
        ),
        "pixel_predictions": (
            pd.concat(pixel_parts, ignore_index=True, sort=False)
            if pixel_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS
            )
        ),
        "technical_events": (
            pd.concat(event_parts, ignore_index=True, sort=False)
            .drop_duplicates()
            .reset_index(drop=True)
            if event_parts
            else pd.DataFrame(
                columns=expcfg.SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS
            )
        ),
        "checkpoint_run_dir": root,
    }

    for name, entity_columns in (
        ("object_predictions", ("source_image", "object_id")),
        (
            "pixel_predictions",
            ("source_image", "object_id", "row", "col"),
        ),
    ):
        frame = outputs[name]
        if frame.empty:
            continue
        keys = ["projection_id", *entity_columns]
        if frame.duplicated(keys).any():
            raise RuntimeError(
                f"Hash-valid 04C checkpoints contain duplicated {name}."
            )
        outputs[name] = frame.reindex(
            columns=(
                expcfg.SIMCA_VALIDATION_OBJECT_PREDICTION_COLUMNS
                if name == "object_predictions"
                else expcfg.SIMCA_VALIDATION_PIXEL_PREDICTION_COLUMNS
            )
        )

    outputs["technical_events"] = outputs["technical_events"].reindex(
        columns=expcfg.SIMCA_VALIDATION_TECHNICAL_EVENT_COLUMNS
    )
    return outputs


def fit_one_class_simca(
    object_db,
    matrix_method: str,
    train_filters: dict,
    target_class: str = DEFAULT_TARGET_CLASS,
    preprocessing_steps=("absorbance", "snv", "sg_d1"),
    n_components: int = 5,
    alpha: float = 0.01,
    rule_name: str = "alternative",
    wavelengths=None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    balanced_pixel_strategy: str = "random",
) -> dict[str, Any]:
    """Fit preprocessing, one-class SIMCA and one decision rule."""
    X_train_raw, y_train, meta_train = build_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )

    fit_bundle = fit_simca_bundle_from_matrix(
        X_train_raw,
        y_train,
        meta_train,
        preprocessing_spec={
            "steps": tuple(preprocessing_steps),
            "sg_window_length": int(sg_window_length),
            "sg_polyorder": int(sg_polyorder),
        },
        n_components=int(n_components),
        alpha=float(alpha),
        wavelengths=wavelengths,
        target_class=target_class,
    )
    preprocessor = fit_bundle.preprocessor
    X_train = fit_bundle.X_train
    model = fit_bundle.model

    rule = make_simca_rule(rule_name)
    rule.fit(model)

    return {
        "target_class": target_class,
        "matrix_method": matrix_method,
        "preprocessing_steps": tuple(preprocessing_steps),
        "preprocessor": preprocessor,
        "model": model,
        "rule": rule,
        "X_train_raw": X_train_raw,
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "fit_bundle": fit_bundle,
    }


def predict_pixels_with_simca(
    object_db,
    simca_bundle: dict,
    projection_filters: dict | None = None,
    target_class: str | None = None,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    balanced_pixel_strategy: str = "random",
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Apply a fitted one-class SIMCA model to selected object pixels."""
    fit_bundle = simca_bundle.get("fit_bundle")
    if fit_bundle is None:
        model = simca_bundle["model"]
        X_train = np.asarray(simca_bundle["X_train"], dtype=float)
        fit_bundle = SimcaFitBundle(
            target_class=str(simca_bundle.get("target_class", model.class_name)),
            preprocessing_steps=tuple(simca_bundle["preprocessing_steps"]),
            preprocessor=simca_bundle["preprocessor"],
            model=model,
            X_train_raw=np.asarray(simca_bundle["X_train_raw"], dtype=float),
            X_train=X_train,
            y_train=np.asarray(simca_bundle["y_train"]),
            metadata={
                str(key): np.asarray(value)
                for key, value in dict(simca_bundle["meta_train"]).items()
            },
            wavelengths=None,
            raw_rank=int(np.linalg.matrix_rank(simca_bundle["X_train_raw"])),
            preprocessed_rank=int(np.linalg.matrix_rank(X_train)),
            preprocessing_seconds=np.nan,
            fit_seconds=np.nan,
            train_scores=pd.DataFrame(
                {"H": model.H_train_, "Q": model.Q_train_}
            ),
        )
    projected = project_simca_bundle(
        fit_bundle,
        object_db=object_db,
        projection_matrix_method="all_pixels",
        projection_filters=projection_filters or {},
        rule=simca_bundle["rule"],
        target_class=target_class,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )
    values = projected.attrs["simca_values"]
    X_pixel = projected.attrs["X_transformed"]
    model = fit_bundle.model
    rule = simca_bundle["rule"]

    if target_class is None:
        target_class = simca_bundle.get("target_class", model.class_name)

    accepted = projected["direct_2way_decision"].to_numpy(dtype=bool)
    rule_statistic = projected["rule_statistic"].to_numpy(dtype=float)
    rule_limit = float(projected["rule_limit"].iloc[0])

    pred_col = predicted_col(target_class, "pixel")
    df = projected.copy()
    df["label"] = np.where(
        df["truth"].astype(bool), target_class, non_target_label
    )
    df[pred_col] = accepted.astype(bool)
    df["predicted_label_pixel"] = np.where(accepted, target_class, non_target_label)
    df["H"] = values["H"]
    df["Q"] = values["Q"]
    df["H_norm_limit"] = values["H_norm_limit"]
    df["Q_norm_limit"] = values["Q_norm_limit"]
    df["rule_statistic"] = rule_statistic
    df["rule_limit"] = float(rule_limit)
    df["rule_name"] = rule.name
    df["matrix_method"] = simca_bundle["matrix_method"]
    df["target_class"] = target_class
    df["non_target_label"] = non_target_label

    for k in range(values["scores"].shape[1]):
        df[f"T{k + 1}"] = values["scores"][:, k]

    return df, values, X_pixel


class _SimcaRefitStageError(RuntimeError):
    """Internal error carrying the technical stage that failed."""

    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = str(failure_type)


def run_single_simca_pixel_projection(
    object_db,
    image_db,
    matrix_method: str,
    preprocessing_name: str,
    preprocessing_steps,
    rule_name: str,
    train_filters: dict,
    projection_filters: dict,
    object_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    n_components: int = 5,
    alpha: float = 0.01,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    position_dilation_radius: int = 3,
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> dict[str, Any]:
    """Fit one-class SIMCA, project pixels, add truth, and aggregate to objects."""
    try:
        bundle = fit_one_class_simca(
            object_db=object_db,
            matrix_method=matrix_method,
            train_filters=train_filters,
            target_class=target_class,
            preprocessing_steps=tuple(preprocessing_steps),
            n_components=n_components,
            alpha=alpha,
            rule_name=rule_name,
            wavelengths=wavelengths,
            m=m,
            random_state=random_state,
            replace=replace,
            sg_window_length=sg_window_length,
            sg_polyorder=sg_polyorder,
            balanced_pixel_strategy=balanced_pixel_strategy,
        )
    except Exception as exc:
        raise _SimcaRefitStageError("training_error", str(exc)) from exc

    if bundle.get("X_train") is None or len(bundle["X_train"]) == 0:
        raise _SimcaRefitStageError("empty_training_matrix", "Empty training matrix.")

    try:
        pixel_df, simca_values, X_pixel = predict_pixels_with_simca(
            object_db=object_db,
            simca_bundle=bundle,
            projection_filters=projection_filters,
            target_class=target_class,
            non_target_label=non_target_label,
            balanced_pixel_strategy=balanced_pixel_strategy,
        )
        if pixel_df is None or len(pixel_df) == 0:
            raise ValueError("No pixel prediction was produced.")

        pixel_df = add_pixel_truth_labels(
            pixel_df=pixel_df,
            image_db=image_db,
            object_db=object_db,
            target_class=target_class,
            dilation_radius=position_dilation_radius,
        )

        threshold_df, object_tables = object_threshold_grid(
            pixel_df=pixel_df,
            object_db=object_db,
            target_class=target_class,
            non_target_label=non_target_label,
            thresholds=object_thresholds,
        )
        if threshold_df is None or len(threshold_df) == 0 or not object_tables:
            raise ValueError("No object prediction was produced.")
    except _SimcaRefitStageError:
        raise
    except Exception as exc:
        raise _SimcaRefitStageError("projection_error", str(exc)) from exc

    if threshold_df is not None and len(threshold_df) > 0:
        threshold_df = materialize_selection_metrics(
            threshold_df,
            keep_source_columns=False,
        )
        threshold_df["matrix_method"] = matrix_method
        threshold_df["matrix_family"] = matrix_family_from_method(matrix_method)
        threshold_df["preprocessing"] = preprocessing_name
        threshold_df["preprocessing_steps"] = "+".join(tuple(preprocessing_steps))
        threshold_df["rule"] = rule_name
        threshold_df["rule_variant"] = rule_name
        threshold_df["n_components"] = int(n_components)
        threshold_df["alpha"] = float(alpha)
        threshold_df["m"] = int(m) if matrix_method == "balanced_pixels" else np.nan
        threshold_df["n_train_observations"] = int(bundle["X_train"].shape[0])
        threshold_df["n_projected_pixels"] = int(len(pixel_df))
        threshold_df["balanced_pixel_strategy"] = balanced_pixel_strategy
        threshold_df["target_class"] = target_class
        threshold_df["non_target_label"] = non_target_label
        threshold_df["sg_window_length"] = int(sg_window_length)
        threshold_df["sg_polyorder"] = int(sg_polyorder)
        threshold_df["position_dilation_radius"] = int(position_dilation_radius)

    return {
        "bundle": bundle,
        "pixel_df": pixel_df,
        "simca_values": simca_values,
        "X_pixel": X_pixel,
        "threshold_df": threshold_df,
        "object_tables": object_tables,
    }


def run_simca_pixel_projection_grid(
    object_db,
    image_db,
    matrix_methods=("object_mean", "balanced_pixels"),
    preprocessing_configs=None,
    rule_names=("simple", "alternative", "combined_index", "data_driven"),
    train_filters=None,
    projection_filters=None,
    object_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    n_components_values=(5,),
    alpha_values=(0.01,),
    m_values=(40,),
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length_values=(9,),
    sg_polyorder_values=(2,),
    position_dilation_radius_values=(3,),
    balanced_pixel_strategy_values=("random",),
    default_m: int = 40,
    default_sg_window_length: int = 11,
    default_sg_polyorder: int = 2,
    keep_pixel_tables: bool = False,
    verbose: bool = True,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Grid search for standard SIMCA rules with pixel-level projection."""
    preprocessing_configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)
    train_filters = train_filters or make_target_train_filters(target_class=target_class)
    projection_filters = projection_filters or {"sample_kind": ["mixture"]}

    summary_parts = []
    stored_results = {}
    errors = []
    grid_configs = []

    for matrix_method in matrix_methods:
        matrix_method = str(matrix_method)
        matrix_family = matrix_family_from_method(matrix_method)
        matrix_param_configs = balanced_strategy_grid_for_matrix(
            matrix_method=matrix_method,
            m_values=m_values,
            balanced_pixel_strategy_values=balanced_pixel_strategy_values,
            default_m=default_m,
        )
        current_preprocessing_configs = _preprocessing_configs_for_family(
            preprocessing_configs_by_family,
            matrix_family,
        )

        for matrix_params in matrix_param_configs:
            for preprocessing_name, preprocessing_steps in current_preprocessing_configs.items():
                preprocessing_steps = tuple(preprocessing_steps)
                for sg_window_length, sg_polyorder in valid_sg_parameter_pairs(
                    preprocessing_steps=preprocessing_steps,
                    sg_window_length_values=sg_window_length_values,
                    sg_polyorder_values=sg_polyorder_values,
                    default_sg_window_length=default_sg_window_length,
                    default_sg_polyorder=default_sg_polyorder,
                ):
                    for rule_name in rule_names:
                        for n_components in n_components_values:
                            for alpha in alpha_values:
                                for position_dilation_radius in position_dilation_radius_values:
                                    grid_configs.append(
                                        {
                                            "matrix_family": matrix_family,
                                            "matrix_method": matrix_method,
                                            "training_matrix_id": matrix_params["training_matrix_id"],
                                            "m": matrix_params["m"],
                                            "m_effective": matrix_params["m_effective"],
                                            "balanced_pixel_strategy": matrix_params["balanced_pixel_strategy"],
                                            "balanced_pixel_strategy_effective": matrix_params["balanced_pixel_strategy_effective"],
                                            "preprocessing": str(preprocessing_name),
                                            "preprocessing_steps": preprocessing_steps,
                                            "rule": str(rule_name),
                                            "n_components": int(n_components),
                                            "alpha": float(alpha),
                                            "sg_window_length": int(sg_window_length),
                                            "sg_polyorder": int(sg_polyorder),
                                            "position_dilation_radius": int(position_dilation_radius),
                                        }
                                    )

    total = len(grid_configs)
    for k, cfg in enumerate(grid_configs, start=1):
        if verbose:
            print(
                f"[{k}/{total}] standard | matrix={cfg['training_matrix_id']} | "
                f"preproc={cfg['preprocessing']} | rule={cfg['rule']} | "
                f"A={cfg['n_components']} | alpha={cfg['alpha']} | "
                f"SG=({cfg['sg_window_length']},{cfg['sg_polyorder']}) | "
                f"dilation={cfg['position_dilation_radius']}"
            )

        key = (
            "standard_rule",
            cfg["training_matrix_id"],
            cfg["preprocessing"],
            cfg["rule"],
            cfg["n_components"],
            cfg["alpha"],
            cfg["sg_window_length"],
            cfg["sg_polyorder"],
            cfg["position_dilation_radius"],
        )

        try:
            res = run_single_simca_pixel_projection(
                object_db=object_db,
                image_db=image_db,
                matrix_method=cfg["matrix_method"],
                preprocessing_name=cfg["preprocessing"],
                preprocessing_steps=cfg["preprocessing_steps"],
                rule_name=cfg["rule"],
                train_filters=train_filters,
                projection_filters=projection_filters,
                object_thresholds=object_thresholds,
                n_components=cfg["n_components"],
                alpha=cfg["alpha"],
                m=cfg["m_effective"],
                random_state=random_state,
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=cfg["sg_window_length"],
                sg_polyorder=cfg["sg_polyorder"],
                position_dilation_radius=cfg["position_dilation_radius"],
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
                non_target_label=non_target_label,
            )

            threshold_df = res["threshold_df"].copy()
            if len(threshold_df) > 0:
                threshold_df["search_method"] = "grid_standard_rules"
                threshold_df["model_family"] = "standard_rule"
                threshold_df["matrix_family"] = cfg["matrix_family"]
                threshold_df["training_matrix_id"] = cfg["training_matrix_id"]
                threshold_df["m"] = cfg["m"]
                threshold_df["m_effective"] = cfg["m_effective"]
                threshold_df["balanced_pixel_strategy_effective"] = cfg["balanced_pixel_strategy_effective"]
                summary_parts.append(threshold_df)

            stored_results[key] = res if keep_pixel_tables else {
                "threshold_df": threshold_df,
                "object_tables": res["object_tables"],
                "bundle": res["bundle"],
            }

        except Exception as exc:
            err = dict(cfg)
            err.update(
                {
                    "search_method": "grid_standard_rules",
                    "model_family": "standard_rule",
                    "target_class": target_class,
                    "non_target_label": non_target_label,
                    "error": repr(exc),
                }
            )
            errors.append(err)
            if verbose:
                print("  -> ERROR:", repr(exc))

    summary_df = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    errors_df = pd.DataFrame(errors)
    return standard_grid_sort(summary_df), stored_results, errors_df


# -----------------------------------------------------------------------------
# Empirical-CV SIMCA thresholds
# -----------------------------------------------------------------------------


def _fit_fold_simca(
    X_train_raw,
    preprocessing_steps,
    n_components,
    alpha,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    target_class: str = DEFAULT_TARGET_CLASS,
):
    preprocessor = SpectralPreprocessor(
        steps=tuple(preprocessing_steps),
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )
    X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)
    model = SIMCAClassModel(class_name=target_class, n_components=int(n_components), alpha=float(alpha))
    model.fit(X_train)
    return preprocessor, model


def _fold_statistics_from_values(values, model) -> dict[str, Any]:
    H = np.asarray(values["H"], dtype=float)
    Q = np.asarray(values["Q"], dtype=float)

    H_norm_chi2 = H / model.H_limit_
    Q_norm_chi2 = Q / model.Q_limit_
    simple_chi2_stat = np.maximum(H_norm_chi2, Q_norm_chi2)
    alternative_chi2_stat = H_norm_chi2 + Q_norm_chi2
    combined_index_stat = alternative_chi2_stat.copy()
    data_driven_stat = model.NQ_ * Q / max(model.Q0_, model.eps) + model.NH_ * H / max(model.H0_, model.eps)
    data_driven_limit = chi2.ppf(1.0 - model.alpha, model.NQ_ + model.NH_)

    return {
        "H": H,
        "Q": Q,
        "H_norm_chi2": H_norm_chi2,
        "Q_norm_chi2": Q_norm_chi2,
        "simple_chi2_stat": simple_chi2_stat,
        "alternative_chi2_stat": alternative_chi2_stat,
        "combined_index_stat": combined_index_stat,
        "data_driven_stat": data_driven_stat,
        "data_driven_chi2_limit": float(data_driven_limit),
        "H_chi2_limit_fold": float(model.H_limit_),
        "Q_chi2_limit_fold": float(model.Q_limit_),
        "H0_fold": float(model.H0_),
        "Q0_fold": float(model.Q0_),
        "NH_fold": float(model.NH_),
        "NQ_fold": float(model.NQ_),
    }


def calibrate_simca_thresholds_cv(
    object_db,
    train_filters: dict,
    matrix_method: str = "balanced_pixels",
    preprocessing_steps=("absorbance", "sg_d1"),
    n_components: int = 10,
    alpha: float = 0.05,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    group_col: str = "object_id",
    n_splits: int | None = None,
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cross-validated empirical calibration of SIMCA thresholds."""
    X_raw, y, meta = build_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )
    if group_col not in meta:
        raise ValueError(f"metadata does not contain group_col={group_col!r}.")

    groups = np.asarray(meta[group_col]).astype(str)
    splitter = _make_group_splitter(groups, n_splits=n_splits)
    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X_raw, y, groups=groups), start=1):
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        if X_train_raw.shape[0] <= int(n_components):
            continue

        preprocessor, model = _fit_fold_simca(
            X_train_raw=X_train_raw,
            preprocessing_steps=preprocessing_steps,
            n_components=n_components,
            alpha=alpha,
            wavelengths=wavelengths,
            sg_window_length=sg_window_length,
            sg_polyorder=sg_polyorder,
            target_class=target_class,
        )
        X_test = preprocessor.transform(X_test_raw)
        values = model.decision_values(X_test)
        stats = _fold_statistics_from_values(values, model)

        for j, row_index in enumerate(test_idx):
            row = {
                "fold": int(fold_id),
                "row_index": int(row_index),
                "label": str(y[row_index]),
                "group": str(groups[row_index]),
            }
            for key, val in meta.items():
                if len(val) == len(X_raw):
                    row[key] = val[row_index]
            for stat_name, stat_values in stats.items():
                row[stat_name] = float(stat_values if np.ndim(stat_values) == 0 else stat_values[j])
            rows.append(row)

    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        raise RuntimeError("No CV distances were produced. Check n_components and groups.")

    q = 1.0 - float(alpha)
    H_emp_cv = _empirical_quantile(cv_df["H"], q)
    Q_emp_cv = _empirical_quantile(cv_df["Q"], q)
    cv_df["alternative_empHQ_stat"] = cv_df["H"] / max(H_emp_cv, 1e-12) + cv_df["Q"] / max(Q_emp_cv, 1e-12)

    thresholds = {
        "alpha": float(alpha),
        "quantile": float(q),
        "H_emp_cv": H_emp_cv,
        "Q_emp_cv": Q_emp_cv,
        "simple_emp_cv": _empirical_quantile(cv_df["simple_chi2_stat"], q),
        "alternative_chi2_emp_cv": _empirical_quantile(cv_df["alternative_chi2_stat"], q),
        "alternative_empHQ_emp_cv": _empirical_quantile(cv_df["alternative_empHQ_stat"], q),
        "combined_index_emp_cv": _empirical_quantile(cv_df["combined_index_stat"], q),
        "data_driven_emp_cv": _empirical_quantile(cv_df["data_driven_stat"], q),
        "H_chi2_limit_fold_median": float(np.median(cv_df["H_chi2_limit_fold"])),
        "Q_chi2_limit_fold_median": float(np.median(cv_df["Q_chi2_limit_fold"])),
        "data_driven_chi2_limit_fold_median": float(np.median(cv_df["data_driven_chi2_limit"])),
        "n_cv_observations": int(len(cv_df)),
        "n_cv_groups": int(cv_df["group"].nunique()),
    }
    return cv_df, thresholds


def fit_final_simca_model(
    object_db,
    train_filters: dict,
    matrix_method: str = "balanced_pixels",
    preprocessing_steps=("absorbance", "sg_d1"),
    n_components: int = 10,
    alpha: float = 0.05,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
) -> dict[str, Any]:
    """Fit final SIMCA model on all target training data."""
    X_train_raw, y_train, meta_train = build_matrix(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=train_filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )
    preprocessor = SpectralPreprocessor(
        steps=tuple(preprocessing_steps),
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
    )
    X_train = preprocessor.fit_transform(X_train_raw, wavelengths=wavelengths)
    model = SIMCAClassModel(class_name=target_class, n_components=int(n_components), alpha=float(alpha))
    model.fit(X_train)
    return {
        "target_class": target_class,
        "matrix_method": matrix_method,
        "preprocessing_steps": tuple(preprocessing_steps),
        "preprocessor": preprocessor,
        "model": model,
        "X_train_raw": X_train_raw,
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
    }


def project_pixels_with_rule_variants(
    object_db,
    final_bundle: dict,
    projection_filters: dict,
    cv_thresholds: dict | None = None,
    rule_variants: Sequence[str] = (
        "simple_chi2",
        "alternative_chi2_fixed2",
        "alternative_chi2_emp_cv",
        "alternative_empHQ_emp_cv",
        "data_driven_chi2",
        "data_driven_emp_cv",
        "combined_index_chi2",
        "combined_index_emp_cv",
    ),
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Project pixels and compute predictions for several SIMCA rule variants."""
    X_pixel_raw, y_pixel, meta_pixel = build_matrix(
        object_db=object_db,
        matrix_method="pixel",
        filters=projection_filters,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )
    X_pixel = final_bundle["preprocessor"].transform(X_pixel_raw)
    model = final_bundle["model"]
    values = model.decision_values(X_pixel)
    H = np.asarray(values["H"], dtype=float)
    Q = np.asarray(values["Q"], dtype=float)

    df = pd.DataFrame(meta_pixel)
    df["label"] = y_pixel.astype(str)
    df["H"] = H
    df["Q"] = Q
    df["H_norm_limit"] = H / model.H_limit_
    df["Q_norm_limit"] = Q / model.Q_limit_
    df["matrix_method"] = final_bundle["matrix_method"]
    df["target_class"] = target_class

    for k in range(values["scores"].shape[1]):
        df[f"T{k + 1}"] = values["scores"][:, k]

    for variant in rule_variants:
        stat, limit = compute_rule_variant_stat_limit(
            H=H,
            Q=Q,
            model=model,
            variant_name=variant,
            cv_thresholds=cv_thresholds,
        )
        df[f"stat_{variant}"] = stat
        df[f"limit_{variant}"] = float(limit)
        df[f"pred_{variant}"] = stat < float(limit)

    return df, values, X_pixel


def summarize_cv_calibration(cv_df: pd.DataFrame, cv_thresholds: dict) -> pd.DataFrame:
    """Summarize target-class CV rejection rate for each threshold variant."""
    alpha = float(cv_thresholds["alpha"])
    variants = {
        "simple_chi2": ("simple_chi2_stat", 1.0),
        "simple_emp_cv": ("simple_chi2_stat", cv_thresholds["simple_emp_cv"]),
        "alternative_chi2_fixed2": ("alternative_chi2_stat", 2.0),
        "alternative_chi2_emp_cv": ("alternative_chi2_stat", cv_thresholds["alternative_chi2_emp_cv"]),
        "alternative_empHQ_emp_cv": ("alternative_empHQ_stat", cv_thresholds["alternative_empHQ_emp_cv"]),
        "data_driven_emp_cv": ("data_driven_stat", cv_thresholds["data_driven_emp_cv"]),
        "combined_index_emp_cv": (
            "combined_index_stat",
            cv_thresholds["combined_index_emp_cv"],
        ),
    }

    rows = []
    for name, (stat_col, limit) in variants.items():
        if stat_col not in cv_df.columns or limit is None or not np.isfinite(float(limit)):
            continue
        stat = cv_df[stat_col].to_numpy(dtype=float)
        rejected = stat >= float(limit)
        rows.append(
            {
                "rule_variant": name,
                "stat_col": stat_col,
                "limit": float(limit),
                "n": int(len(stat)),
                "n_rejected_target_cv": int(np.sum(rejected)),
                "rejection_rate_target_cv": float(np.mean(rejected)),
                "acceptance_rate_target_cv": float(1.0 - np.mean(rejected)),
                "expected_rejection_rate": alpha,
                "expected_acceptance_rate": 1.0 - alpha,
                "abs_rejection_error": float(abs(np.mean(rejected) - alpha)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_rejection_error").reset_index(drop=True)


def _cv_rule_diagnostics(cv_calibration_summary: pd.DataFrame, rule_variant: str) -> dict:
    if cv_calibration_summary is None or len(cv_calibration_summary) == 0:
        return {}
    sub = cv_calibration_summary[cv_calibration_summary["rule_variant"].astype(str).eq(str(rule_variant))]
    if len(sub) == 0:
        return {}
    row = sub.iloc[0]
    return {
        "cv_target_rejection_rate": float(row["rejection_rate_target_cv"]),
        "cv_target_acceptance_rate": float(row["acceptance_rate_target_cv"]),
        "cv_expected_rejection_rate": float(row["expected_rejection_rate"]),
        "cv_abs_rejection_error": float(row["abs_rejection_error"]),
        "cv_rule_limit": float(row["limit"]),
    }


def run_simca_rule_variant_grid(
    object_db,
    image_db,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    matrix_methods=("balanced_pixels",),
    rule_variants=(
        "simple_chi2",
        "simple_emp_cv",
        "alternative_chi2_fixed2",
        "alternative_chi2_emp_cv",
        "alternative_empHQ_fixed2",
        "alternative_empHQ_emp_cv",
        "data_driven_chi2",
        "data_driven_emp_cv",
        "combined_index_chi2",
        "combined_index_emp_cv",
    ),
    n_components_values=(5, 8, 10, 12, 15, 20),
    alpha_values=(0.05,),
    object_thresholds=(0.75,),
    m_values=(40,),
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length_values=(11,),
    sg_polyorder_values=(2,),
    position_dilation_radius_values=(3,),
    cv_n_splits: int | None = 5,
    group_col: str = "object_id",
    keep_pixel_tables: bool = False,
    keep_cv_tables: bool = False,
    verbose: bool = True,
    balanced_pixel_strategy_values=("random",),
    default_m: int = 40,
    default_sg_window_length: int = 11,
    default_sg_polyorder: int = 2,
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Grid search for empirical-CV SIMCA rule variants."""
    preprocessing_configs_by_family = _normalize_preprocessing_configs_by_family(preprocessing_configs)
    summary_rows = []
    results = {}
    errors = []
    base_configs = []

    for matrix_method in matrix_methods:
        matrix_method = str(matrix_method)
        matrix_family = matrix_family_from_method(matrix_method)
        matrix_param_configs = balanced_strategy_grid_for_matrix(
            matrix_method=matrix_method,
            m_values=m_values,
            balanced_pixel_strategy_values=balanced_pixel_strategy_values,
            default_m=default_m,
        )
        current_preprocessing_configs = _preprocessing_configs_for_family(
            preprocessing_configs_by_family,
            matrix_family,
        )
        for matrix_params in matrix_param_configs:
            for preprocessing_name, preprocessing_steps in current_preprocessing_configs.items():
                preprocessing_steps = tuple(preprocessing_steps)
                for sg_window_length, sg_polyorder in valid_sg_parameter_pairs(
                    preprocessing_steps=preprocessing_steps,
                    sg_window_length_values=sg_window_length_values,
                    sg_polyorder_values=sg_polyorder_values,
                    default_sg_window_length=default_sg_window_length,
                    default_sg_polyorder=default_sg_polyorder,
                ):
                    for n_components in n_components_values:
                        for alpha in alpha_values:
                            for position_dilation_radius in position_dilation_radius_values:
                                base_configs.append(
                                    {
                                        "matrix_family": matrix_family,
                                        "matrix_method": matrix_method,
                                        "training_matrix_id": matrix_params["training_matrix_id"],
                                        "m": matrix_params["m"],
                                        "m_effective": matrix_params["m_effective"],
                                        "balanced_pixel_strategy": matrix_params["balanced_pixel_strategy"],
                                        "balanced_pixel_strategy_effective": matrix_params["balanced_pixel_strategy_effective"],
                                        "preprocessing": str(preprocessing_name),
                                        "preprocessing_steps": preprocessing_steps,
                                        "n_components": int(n_components),
                                        "alpha": float(alpha),
                                        "sg_window_length": int(sg_window_length),
                                        "sg_polyorder": int(sg_polyorder),
                                        "position_dilation_radius": int(position_dilation_radius),
                                    }
                                )

    total = len(base_configs)
    for base_counter, cfg in enumerate(base_configs, start=1):
        base_key = (
            "empirical_cv_rule",
            cfg["training_matrix_id"],
            cfg["preprocessing"],
            cfg["n_components"],
            cfg["alpha"],
            cfg["sg_window_length"],
            cfg["sg_polyorder"],
            cfg["position_dilation_radius"],
        )
        if verbose:
            print(
                f"\n[{base_counter}/{total}] empirical_cv | matrix={cfg['training_matrix_id']} | "
                f"preprocessing={cfg['preprocessing']} | A={cfg['n_components']} | alpha={cfg['alpha']} | "
                f"SG=({cfg['sg_window_length']},{cfg['sg_polyorder']}) | dilation={cfg['position_dilation_radius']}"
            )

        training_complete = False
        try:
            cv_df, cv_thresholds = calibrate_simca_thresholds_cv(
                object_db=object_db,
                train_filters=train_filters,
                matrix_method=cfg["matrix_method"],
                preprocessing_steps=cfg["preprocessing_steps"],
                n_components=cfg["n_components"],
                alpha=cfg["alpha"],
                m=cfg["m_effective"],
                random_state=random_state,
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=cfg["sg_window_length"],
                sg_polyorder=cfg["sg_polyorder"],
                group_col=group_col,
                n_splits=cv_n_splits,
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
            )
            cv_calibration_summary = summarize_cv_calibration(cv_df=cv_df, cv_thresholds=cv_thresholds)
            final_bundle = fit_final_simca_model(
                object_db=object_db,
                train_filters=train_filters,
                matrix_method=cfg["matrix_method"],
                preprocessing_steps=cfg["preprocessing_steps"],
                n_components=cfg["n_components"],
                alpha=cfg["alpha"],
                m=cfg["m_effective"],
                random_state=random_state,
                replace=replace,
                wavelengths=wavelengths,
                sg_window_length=cfg["sg_window_length"],
                sg_polyorder=cfg["sg_polyorder"],
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
            )
            if final_bundle.get("X_train") is None or len(final_bundle["X_train"]) == 0:
                raise ValueError("Empty training matrix.")
            training_complete = True
            pixel_variants_df, simca_values, X_pixel = project_pixels_with_rule_variants(
                object_db=object_db,
                final_bundle=final_bundle,
                projection_filters=projection_filters,
                cv_thresholds=cv_thresholds,
                rule_variants=rule_variants,
                balanced_pixel_strategy=cfg["balanced_pixel_strategy_effective"],
                target_class=target_class,
            )
            pixel_variants_df = add_pixel_truth_labels(
                pixel_df=pixel_variants_df,
                image_db=image_db,
                object_db=object_db,
                target_class=target_class,
                dilation_radius=cfg["position_dilation_radius"],
            )

            object_tables_by_rule = {}
            for rule_variant in rule_variants:
                pred_variant_col = f"pred_{rule_variant}"
                stat_col = f"stat_{rule_variant}"
                limit_col = f"limit_{rule_variant}"
                if pred_variant_col not in pixel_variants_df.columns:
                    errors.append({**cfg, "search_method": "grid_empirical_cv_rules", "model_family": "empirical_cv_rule", "rule_variant": rule_variant, "technical_failure_type": "missing_predictions", "error": f"Missing prediction column: {pred_variant_col}", "target_class": target_class, "non_target_label": non_target_label})
                    continue

                tmp_pixel_df = pixel_variants_df.copy()
                target_pred_col = predicted_col(target_class, "pixel")
                tmp_pixel_df[target_pred_col] = tmp_pixel_df[pred_variant_col].astype(bool)
                tmp_pixel_df["predicted_label_pixel"] = np.where(tmp_pixel_df[target_pred_col], target_class, non_target_label)
                tmp_pixel_df["rule_statistic"] = tmp_pixel_df[stat_col]
                tmp_pixel_df["rule_limit"] = tmp_pixel_df[limit_col]
                tmp_pixel_df["rule_name"] = str(rule_variant)
                tmp_pixel_df["non_target_label"] = non_target_label

                threshold_df, object_tables = object_threshold_grid(
                    pixel_df=tmp_pixel_df,
                    object_db=object_db,
                    target_class=target_class,
                    non_target_label=non_target_label,
                    thresholds=object_thresholds,
                )
                if threshold_df is None or len(threshold_df) == 0:
                    errors.append({**cfg, "search_method": "grid_empirical_cv_rules", "model_family": "empirical_cv_rule", "rule_variant": rule_variant, "technical_failure_type": "missing_predictions", "error": "Empty threshold_df.", "target_class": target_class, "non_target_label": non_target_label})
                    continue

                object_tables_by_rule[str(rule_variant)] = object_tables
                threshold_df = materialize_selection_metrics(
                    threshold_df,
                    keep_source_columns=False,
                )
                threshold_df["search_method"] = "grid_empirical_cv_rules"
                threshold_df["model_family"] = "empirical_cv_rule"
                threshold_df["matrix_family"] = cfg["matrix_family"]
                threshold_df["training_matrix_id"] = cfg["training_matrix_id"]
                threshold_df["matrix_method"] = cfg["matrix_method"]
                threshold_df["m"] = cfg["m"]
                threshold_df["m_effective"] = cfg["m_effective"]
                threshold_df["balanced_pixel_strategy"] = cfg["balanced_pixel_strategy"]
                threshold_df["balanced_pixel_strategy_effective"] = cfg["balanced_pixel_strategy_effective"]
                threshold_df["preprocessing"] = cfg["preprocessing"]
                threshold_df["preprocessing_steps"] = "+".join(cfg["preprocessing_steps"])
                threshold_df["rule_variant"] = str(rule_variant)
                threshold_df["rule"] = str(rule_variant)
                threshold_df["n_components"] = cfg["n_components"]
                threshold_df["alpha"] = cfg["alpha"]
                threshold_df["target_class"] = target_class
                threshold_df["non_target_label"] = non_target_label
                threshold_df["sg_window_length"] = cfg["sg_window_length"]
                threshold_df["sg_polyorder"] = cfg["sg_polyorder"]
                threshold_df["position_dilation_radius"] = cfg["position_dilation_radius"]
                threshold_df["cv_n_splits"] = cv_n_splits if cv_n_splits is not None else "leave_one_group_out"
                threshold_df["n_cv_observations"] = int(cv_thresholds.get("n_cv_observations", len(cv_df)))
                threshold_df["n_cv_groups"] = int(cv_thresholds.get("n_cv_groups", cv_df["group"].nunique()))

                for name in ["H_emp_cv", "Q_emp_cv", "simple_emp_cv", "alternative_chi2_emp_cv", "alternative_empHQ_emp_cv", "combined_index_emp_cv", "data_driven_emp_cv"]:
                    threshold_df[f"{name}_limit" if not name.endswith("_cv") else name] = float(cv_thresholds.get(name, np.nan))
                for key_diag, value_diag in _cv_rule_diagnostics(cv_calibration_summary, rule_variant).items():
                    threshold_df[key_diag] = value_diag

                summary_rows.append(threshold_df)

            stored = {
                "cv_thresholds": cv_thresholds,
                "cv_calibration_summary": cv_calibration_summary,
                "object_tables_by_rule": object_tables_by_rule,
                "final_bundle": final_bundle,
            }
            if keep_cv_tables:
                stored["cv_df"] = cv_df
            if keep_pixel_tables:
                stored["pixel_variants_df"] = pixel_variants_df
                stored["simca_values"] = simca_values
                stored["X_pixel"] = X_pixel
            results[base_key] = stored

        except Exception as exc:
            failure_type = "projection_error" if training_complete else "training_error"
            errors.append({**cfg, "search_method": "grid_empirical_cv_rules", "model_family": "empirical_cv_rule", "rule_variant": "ALL", "technical_failure_type": failure_type, "error": str(exc), "target_class": target_class, "non_target_label": non_target_label})
            if verbose:
                print("  -> ERROR:", repr(exc))

    summary_df = pd.concat(summary_rows, ignore_index=True, sort=False) if summary_rows else pd.DataFrame()
    errors_df = pd.DataFrame(errors)
    return standard_grid_sort(summary_df), results, errors_df


# -----------------------------------------------------------------------------
# Refit utilities
# -----------------------------------------------------------------------------


def refit_best_grid_row(
    object_db,
    image_db,
    best_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs=None,
    object_thresholds=None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    wavelengths=None,
    sg_window_length: int = 9,
    sg_polyorder: int = 2,
    position_dilation_radius: int = 3,
    balanced_pixel_strategy: str = "random",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> dict[str, Any]:
    """Refit one standard-rule configuration from a grid summary row."""
    preprocessing_name = str(best_row["preprocessing"])
    steps = _resolve_preprocessing_steps_for_row(best_row, preprocessing_configs)

    if object_thresholds is None:
        object_thresholds = [row_float(best_row, "object_threshold", 0.75)]

    m = row_int(best_row, "m_effective", row_int(best_row, "m", m))
    sg_window_length = row_int(best_row, "sg_window_length", sg_window_length)
    sg_polyorder = row_int(best_row, "sg_polyorder", sg_polyorder)
    position_dilation_radius = row_int(best_row, "position_dilation_radius", position_dilation_radius)
    balanced_pixel_strategy = row_str(best_row, "balanced_pixel_strategy_effective", row_str(best_row, "balanced_pixel_strategy", balanced_pixel_strategy))
    if balanced_pixel_strategy == "not_applicable":
        balanced_pixel_strategy = "random"
    target_class = row_str(best_row, "target_class", target_class)
    non_target_label = row_str(best_row, "non_target_label", non_target_label)
    rule_name = row_str(best_row, "rule_for_refit", row_str(best_row, "rule", "alternative"))

    return run_single_simca_pixel_projection(
        object_db=object_db,
        image_db=image_db,
        matrix_method=str(best_row["matrix_method"]),
        preprocessing_name=preprocessing_name,
        preprocessing_steps=steps,
        rule_name=rule_name,
        train_filters=train_filters,
        projection_filters=projection_filters,
        object_thresholds=object_thresholds,
        n_components=int(best_row["n_components"]),
        alpha=float(best_row["alpha"]),
        m=m,
        random_state=random_state,
        replace=replace,
        wavelengths=wavelengths,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        position_dilation_radius=position_dilation_radius,
        balanced_pixel_strategy=balanced_pixel_strategy,
        target_class=target_class,
        non_target_label=non_target_label,
    )


def refit_empirical_cv_rule_row(
    object_db,
    image_db,
    best_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    keep_cv_tables: bool = False,
) -> dict[str, Any]:
    """Refit one empirical-CV SIMCA configuration from a selected row."""
    preprocessing_name = str(best_row["preprocessing"])
    steps = _resolve_preprocessing_steps_for_row(best_row, preprocessing_configs)

    rule_variant = str(row_value(best_row, "rule_variant", row_value(best_row, "rule_for_refit", None)))
    if rule_variant in {"None", "nan"}:
        raise ValueError("Could not infer rule_variant for empirical CV refit.")

    matrix_method = str(best_row["matrix_method"])
    object_threshold = float(row_value(best_row, "object_threshold", 0.75))
    n_components = int(row_value(best_row, "n_components", 5))
    alpha = float(row_value(best_row, "alpha", 0.05))
    m_effective = int(row_value(best_row, "m_effective", row_value(best_row, "m", 40)))
    sg_window_length = int(row_value(best_row, "sg_window_length", 11))
    sg_polyorder = int(row_value(best_row, "sg_polyorder", 2))
    position_dilation_radius = int(row_value(best_row, "position_dilation_radius", 3))
    balanced_pixel_strategy_effective = str(row_value(best_row, "balanced_pixel_strategy_effective", row_value(best_row, "balanced_pixel_strategy", "random")))
    if balanced_pixel_strategy_effective == "not_applicable":
        balanced_pixel_strategy_effective = "random"
    target_class = row_str(best_row, "target_class", target_class)
    non_target_label = row_str(best_row, "non_target_label", non_target_label)

    summary_df, results, errors_df = run_simca_rule_variant_grid(
        object_db=object_db,
        image_db=image_db,
        train_filters=train_filters,
        projection_filters=projection_filters,
        preprocessing_configs={preprocessing_name: steps},
        matrix_methods=[matrix_method],
        rule_variants=[rule_variant],
        n_components_values=[n_components],
        alpha_values=[alpha],
        object_thresholds=[object_threshold],
        m_values=[m_effective],
        random_state=random_state,
        replace=replace,
        wavelengths=wavelengths,
        sg_window_length_values=[sg_window_length],
        sg_polyorder_values=[sg_polyorder],
        position_dilation_radius_values=[position_dilation_radius],
        cv_n_splits=cv_n_splits,
        group_col=cv_group_col,
        keep_pixel_tables=True,
        keep_cv_tables=keep_cv_tables,
        verbose=False,
        balanced_pixel_strategy_values=[balanced_pixel_strategy_effective],
        default_m=m_effective,
        default_sg_window_length=sg_window_length,
        default_sg_polyorder=sg_polyorder,
        target_class=target_class,
        non_target_label=non_target_label,
    )
    if len(results) == 0:
        if errors_df is not None and len(errors_df):
            error_row = errors_df.iloc[0]
            raise _SimcaRefitStageError(
                str(error_row.get("technical_failure_type", "training_error")),
                str(error_row.get("error", "No result returned by empirical CV refit.")),
            )
        raise _SimcaRefitStageError(
            "training_error",
            "No result returned by empirical CV refit.",
        )

    first_key = next(iter(results.keys()))
    stored = results[first_key]
    if rule_variant not in stored.get("object_tables_by_rule", {}):
        raise _SimcaRefitStageError(
            "missing_predictions",
            f"No object predictions for rule {rule_variant!r}.",
        )
    if object_threshold not in stored["object_tables_by_rule"][rule_variant]:
        raise _SimcaRefitStageError(
            "missing_predictions",
            f"No object predictions for threshold {object_threshold!r}.",
        )
    object_df = stored["object_tables_by_rule"][rule_variant][object_threshold].copy()
    pixel_df = stored["pixel_variants_df"].copy()

    pred_variant_col = f"pred_{rule_variant}"
    stat_col = f"stat_{rule_variant}"
    limit_col = f"limit_{rule_variant}"
    target_pred_col = predicted_col(target_class, "pixel")
    pixel_df[target_pred_col] = pixel_df[pred_variant_col].astype(bool)
    pixel_df["predicted_label_pixel"] = np.where(pixel_df[target_pred_col], target_class, non_target_label)
    pixel_df["rule_statistic"] = pixel_df[stat_col]
    pixel_df["rule_limit"] = pixel_df[limit_col]
    pixel_df["rule_name"] = rule_variant
    pixel_df["non_target_label"] = non_target_label

    return {
        "summary_df": summary_df,
        "results": results,
        "errors_df": errors_df,
        "object_df": object_df,
        "pixel_df": pixel_df,
        "rule_variant": rule_variant,
    }


def _attach_selected_metadata(df: pd.DataFrame, row: pd.Series, evaluation_split: str) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "candidate_id",
        "model_candidate_id",
        "candidate_source",
        "candidate_sources",
        "selected_config_id",
        "selection_track",
        "assigned_selection_track",
        "final_rank_in_track",
        "pareto_tier",
        "pareto_rank_in_track",
        "matrix_family",
        "decision_mode",
        "metric_level",
        "training_matrix_id",
        "matrix_method",
        "balanced_pixel_strategy",
        "balanced_pixel_strategy_effective",
        "model_family",
        "preprocessing",
        "preprocessing_steps",
        "rule",
        "rule_variant",
        "selected_rule_name",
        "rule_for_refit",
        "target_class",
        "non_target_label",
        "n_components",
        "alpha",
        "object_threshold",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
        "m",
        "m_effective",
        "sg_window_length",
        "sg_polyorder",
        "position_dilation_radius",
    ]:
        if col in row.index:
            out[col] = row[col]
    out["evaluation_split"] = evaluation_split
    return out


def refit_selected_simca_row(
    object_db,
    image_db,
    selected_row: pd.Series,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> dict[str, Any]:
    """Refit one selected SIMCA row from a standard or empirical-CV summary."""
    model_family = str(selected_row["model_family"])
    target_class = row_str(selected_row, "target_class", target_class)
    non_target_label = row_str(selected_row, "non_target_label", non_target_label)
    decision_mode = row_str(selected_row, "decision_mode", "2way")
    refit_row = selected_row.copy()
    if decision_mode == "3way":
        upper_threshold = row_value(
            selected_row,
            "three_way_upper_threshold",
            None,
        )
        if upper_threshold is None or not np.isfinite(float(upper_threshold)):
            raise _SimcaRefitStageError(
                "impossible_threshold",
                "A 3-way refit requires its calibrated upper threshold.",
            )
        # Object aggregation needs a binary cut to materialize the table, but
        # the subsequent 3-way decision uses both fixed 03B thresholds.
        refit_row["object_threshold"] = float(upper_threshold)

    if model_family == "standard_rule":
        res = refit_best_grid_row(
            object_db=object_db,
            image_db=image_db,
            best_row=refit_row,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            object_thresholds=[float(refit_row["object_threshold"])],
            random_state=random_state,
            replace=replace,
            wavelengths=wavelengths,
            target_class=target_class,
            non_target_label=non_target_label,
        )
        object_df = res["object_tables"][float(refit_row["object_threshold"])].copy()
        pixel_df = res["pixel_df"].copy()
    elif model_family in {"empirical_cv_rule", "rule_variant_grid"}:
        res = refit_empirical_cv_rule_row(
            object_db=object_db,
            image_db=image_db,
            best_row=refit_row,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            wavelengths=wavelengths,
            random_state=random_state,
            replace=replace,
            cv_n_splits=cv_n_splits,
            cv_group_col=cv_group_col,
            target_class=target_class,
            non_target_label=non_target_label,
        )
        object_df = res["object_df"].copy()
        pixel_df = res["pixel_df"].copy()
    else:
        raise ValueError(f"Unknown model_family={model_family!r}")

    return {
        "result": res,
        "object_df": _attach_selected_metadata(object_df, selected_row, "projection"),
        "pixel_df": _attach_selected_metadata(pixel_df, selected_row, "projection"),
    }


def refit_selected_simca_configs(
    selected_configs_df: pd.DataFrame,
    object_db,
    image_db,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    evaluation_split: str = "projection",
    wavelengths=None,
    random_state: int = 42,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit several selected SIMCA configurations and collect object/pixel outputs."""
    metric_rows = []
    object_parts = []
    pixel_parts = []
    pixel_error_parts = []
    errors = []

    for _, row in selected_configs_df.iterrows():
        config_id = row.get(
            "candidate_id",
            row.get("selected_config_id", "unknown_config"),
        )
        if verbose:
            print(f"[{evaluation_split}] {config_id}")
        try:
            out = refit_selected_simca_row(
                object_db=object_db,
                image_db=image_db,
                selected_row=row,
                train_filters=train_filters,
                projection_filters=projection_filters,
                preprocessing_configs=preprocessing_configs,
                wavelengths=wavelengths,
                random_state=random_state,
                replace=replace,
                cv_n_splits=cv_n_splits,
                cv_group_col=cv_group_col,
                target_class=target_class,
                non_target_label=non_target_label,
            )

            object_df = out["object_df"].copy()
            pixel_df = out["pixel_df"].copy()
            object_df["evaluation_split"] = evaluation_split
            pixel_df["evaluation_split"] = evaluation_split

            target_class_i = row_str(row, "target_class", target_class)
            non_target_label_i = row_str(row, "non_target_label", non_target_label)
            true_object_col = true_col(target_class_i, "object")
            pred_object_col = predicted_col(target_class_i, "object")
            ratio_col = pixel_ratio_col(target_class_i)
            pred_pixel_col = predicted_col(target_class_i, "pixel")

            if len(object_df) == 0 or len(pixel_df) == 0:
                raise _SimcaRefitStageError(
                    "missing_predictions",
                    "The refit produced an empty object or pixel prediction table.",
                )
            required_object_cols = {ratio_col, pred_object_col}
            required_pixel_cols = {pred_pixel_col, "rule_statistic", "rule_limit"}
            missing_output_cols = sorted(
                (required_object_cols - set(object_df.columns))
                | (required_pixel_cols - set(pixel_df.columns))
            )
            if missing_output_cols:
                raise _SimcaRefitStageError(
                    "missing_predictions",
                    f"Missing output columns: {missing_output_cols}",
                )
            finite_cols = (
                (object_df, (ratio_col,)),
                (pixel_df, ("rule_statistic", "rule_limit")),
            )
            for frame, columns in finite_cols:
                for col in columns:
                    values = pd.to_numeric(frame[col], errors="coerce")
                    if not np.isfinite(values).all():
                        failure_type = (
                            "nonfinite_simca_limit"
                            if col == "rule_limit"
                            else "nonfinite_output"
                        )
                        raise _SimcaRefitStageError(
                            failure_type,
                            f"{col} contains NaN or Inf.",
                        )

            metrics = {}
            if {true_object_col, pred_object_col}.issubset(object_df.columns):
                metrics = binary_detection_metrics(
                    object_df,
                    true_col=true_object_col,
                    pred_col=pred_object_col,
                    target_class=target_class_i,
                    non_target_class=non_target_label_i,
                )

            metric_row = row.to_dict()
            metric_row.update(metrics)
            metric_row["evaluation_split"] = evaluation_split
            metric_row["n_projected_objects"] = int(len(object_df))
            metric_row["n_projected_pixels"] = int(len(pixel_df))
            metric_rows.append(metric_row)
            object_parts.append(object_df)
            pixel_parts.append(pixel_df)

            true_pixel_col = true_col(target_class_i, "pixel")
            if {true_pixel_col, pred_pixel_col}.issubset(pixel_df.columns):
                pixel_err = summarize_pixel_errors_by_image(
                    pixel_df,
                    target_class=target_class_i,
                    non_target_label=non_target_label_i,
                    group_cols=("source_image",),
                )
                if len(pixel_err) > 0:
                    pixel_err["selected_config_id"] = config_id
                    pixel_err["evaluation_split"] = evaluation_split
                    pixel_error_parts.append(pixel_err)

        except Exception as exc:
            failure_type = (
                exc.failure_type
                if isinstance(exc, _SimcaRefitStageError)
                else "training_error"
            )
            errors.append(
                {
                    "candidate_id": str(config_id),
                    "evaluation_split": evaluation_split,
                    "technical_failure_type": failure_type,
                    "technical_failure_message": str(exc),
                }
            )
            if verbose:
                print("  -> ERROR:", repr(exc))

    return (
        pd.DataFrame(metric_rows),
        pd.concat(object_parts, ignore_index=True, sort=False) if object_parts else pd.DataFrame(),
        pd.concat(pixel_parts, ignore_index=True, sort=False) if pixel_parts else pd.DataFrame(),
        pd.concat(pixel_error_parts, ignore_index=True, sort=False) if pixel_error_parts else pd.DataFrame(),
        pd.DataFrame(errors),
    )


def run_selected_simca_random_state_stability(
    selected_configs_df: pd.DataFrame,
    object_db,
    image_db,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 10, 20, 42, 100),
    evaluation_split: str = "random_state_stability",
    wavelengths=None,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Refit selected SIMCA configs over several random seeds.

    Returns
    -------
    metrics_df:
        One row per selected configuration and random seed.

    pixel_errors_by_image_df:
        Lightweight pixel-error summaries by image/config/seed.

    errors_df:
        Refit errors, if any.

    Notes
    -----
    This is mainly useful for balanced_pixels models.
    """
    metric_parts = []
    pixel_error_parts = []
    error_parts = []

    for seed in seeds:
        print(f"[random_state_stability] seed={seed}")

        (
            metrics_df,
            _objects_df,
            _pixels_df,
            pixel_errors_df,
            errors_df,
        ) = refit_selected_simca_configs(
            selected_configs_df=selected_configs_df,
            object_db=object_db,
            image_db=image_db,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            evaluation_split=evaluation_split,
            wavelengths=wavelengths,
            random_state=int(seed),
            replace=replace,
            cv_n_splits=cv_n_splits,
            cv_group_col=cv_group_col,
            target_class=target_class,
            non_target_label=non_target_label,
        )

        if metrics_df is not None and len(metrics_df) > 0:
            metrics_df = metrics_df.copy()
            metrics_df["random_state"] = int(seed)
            metric_parts.append(metrics_df)

        if pixel_errors_df is not None and len(pixel_errors_df) > 0:
            pixel_errors_df = pixel_errors_df.copy()
            pixel_errors_df["random_state"] = int(seed)
            pixel_error_parts.append(pixel_errors_df)

        if errors_df is not None and len(errors_df) > 0:
            errors_df = errors_df.copy()
            errors_df["random_state"] = int(seed)
            error_parts.append(errors_df)

        del metrics_df, _objects_df, _pixels_df, pixel_errors_df, errors_df
        gc.collect()

    metrics_out = (
        pd.concat(metric_parts, ignore_index=True, sort=False)
        if metric_parts
        else pd.DataFrame()
    )

    pixel_errors_out = (
        pd.concat(pixel_error_parts, ignore_index=True, sort=False)
        if pixel_error_parts
        else pd.DataFrame()
    )

    errors_out = (
        pd.concat(error_parts, ignore_index=True, sort=False)
        if error_parts
        else pd.DataFrame()
    )

    return metrics_out, pixel_errors_out, errors_out


def run_selected_simca_random_state_stability_full(
    selected_configs_df: pd.DataFrame,
    object_db,
    image_db,
    train_filters: dict,
    projection_filters: dict,
    preprocessing_configs,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 10, 20, 42, 100),
    evaluation_split: str = "random_state_stability",
    wavelengths=None,
    replace: bool = False,
    cv_n_splits: int | None = 5,
    cv_group_col: str = "object_id",
    target_class: str = DEFAULT_TARGET_CLASS,
    non_target_label: str = DEFAULT_NON_TARGET_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Refit selected SIMCA configs over several random seeds.

    Returns
    -------
    metrics_df
    objects_df
    pixel_errors_by_image_df
    errors_df
    """
    metric_parts = []
    object_parts = []
    pixel_error_parts = []
    error_parts = []

    for seed in seeds:
        print(f"[random_state_stability_full] seed={seed}")

        (
            metrics_df,
            objects_df,
            _pixels_df,
            pixel_errors_df,
            errors_df,
        ) = refit_selected_simca_configs(
            selected_configs_df=selected_configs_df,
            object_db=object_db,
            image_db=image_db,
            train_filters=train_filters,
            projection_filters=projection_filters,
            preprocessing_configs=preprocessing_configs,
            evaluation_split=evaluation_split,
            wavelengths=wavelengths,
            random_state=int(seed),
            replace=replace,
            cv_n_splits=cv_n_splits,
            cv_group_col=cv_group_col,
            target_class=target_class,
            non_target_label=non_target_label,
        )

        if len(metrics_df) > 0:
            metrics_df = metrics_df.copy()
            metrics_df["random_state"] = int(seed)
            metric_parts.append(metrics_df)

        if len(objects_df) > 0:
            objects_df = objects_df.copy()
            objects_df["random_state"] = int(seed)
            object_parts.append(objects_df)

        if len(pixel_errors_df) > 0:
            pixel_errors_df = pixel_errors_df.copy()
            pixel_errors_df["random_state"] = int(seed)
            pixel_error_parts.append(pixel_errors_df)

        if len(errors_df) > 0:
            errors_df = errors_df.copy()
            errors_df["random_state"] = int(seed)
            error_parts.append(errors_df)

        del metrics_df, objects_df, _pixels_df, pixel_errors_df, errors_df
        gc.collect()

    return (
        pd.concat(metric_parts, ignore_index=True, sort=False) if metric_parts else pd.DataFrame(),
        pd.concat(object_parts, ignore_index=True, sort=False) if object_parts else pd.DataFrame(),
        pd.concat(pixel_error_parts, ignore_index=True, sort=False) if pixel_error_parts else pd.DataFrame(),
        pd.concat(error_parts, ignore_index=True, sort=False) if error_parts else pd.DataFrame(),
    )
