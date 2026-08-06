"""Canonical QC-aware scientific split construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
import pandas as pd

from src import experiment_config as expcfg


PROTOCOL_SPLIT_MANIFEST_COLUMNS = expcfg.PROTOCOL_SPLIT_MANIFEST_COLUMNS
PROTOCOL_SPLIT_CHECK_COLUMNS = expcfg.PROTOCOL_SPLIT_CHECK_COLUMNS
SPLIT_DIAGNOSTIC_COLUMNS = expcfg.SPLIT_DIAGNOSTIC_COLUMNS


def _base_protocol_role(
    image,
    *,
    calibration_batches,
    validation_batches,
    test_batches,
) -> str:
    if image.get("image_status") == "excluded":
        return "excluded"
    if bool(image.get("is_mixture")) or image.get("sample_kind") == "mixture":
        return "mixture_application"
    if not bool(image.get("is_pure")) and image.get("sample_kind") != "pure":
        return "excluded"
    batch = image.get("batch")
    if batch in calibration_batches:
        return "calibration"
    if batch in validation_batches:
        return "validation"
    if batch in test_batches:
        return "test"
    return "excluded"


def _exclusion_sets(exclusion_manifest):
    if exclusion_manifest is None or len(exclusion_manifest) == 0:
        return set(), set()
    required = {"record_type", "record_id"}
    missing = required.difference(exclusion_manifest.columns)
    if missing:
        raise KeyError(
            f"Exclusion manifest is missing columns: {sorted(missing)}"
        )
    excluded_objects = set(
        exclusion_manifest.loc[
            exclusion_manifest["record_type"].eq("object"),
            "record_id",
        ].astype(str)
    )
    excluded_images = set(
        exclusion_manifest.loc[
            exclusion_manifest["record_type"].eq("image"),
            "record_id",
        ].astype(str)
    )
    return excluded_objects, excluded_images


def build_protocol_manifest(
    image_db: Mapping,
    object_db: Mapping | None = None,
    *,
    exclusion_manifest: pd.DataFrame | None = None,
    calibration_batches=expcfg.PROTOCOL_CALIBRATION_BATCHES,
    validation_batches=expcfg.PROTOCOL_VALIDATION_BATCHES,
    test_batches=expcfg.PROTOCOL_TEST_BATCHES,
    expected_classes=expcfg.PROTOCOL_EXPECTED_CLASSES,
    strict: bool = True,
):
    """Build the one authoritative object split after QC closure."""
    calibration_batches = tuple(calibration_batches)
    validation_batches = tuple(validation_batches)
    test_batches = tuple(test_batches)
    excluded_objects, excluded_images = _exclusion_sets(exclusion_manifest)
    rows = []

    for source_image, image in image_db.items():
        source_image = str(source_image)
        role = _base_protocol_role(
            image,
            calibration_batches=calibration_batches,
            validation_batches=validation_batches,
            test_batches=test_batches,
        )
        image_excluded = (
            source_image in excluded_images
            or str(image.get("image_id", "")) in excluded_images
        )
        object_ids = list(image.get("object_ids", [])) or [None]
        for object_id in object_ids:
            object_text = None if object_id is None else str(object_id)
            obj = (
                None
                if object_db is None or object_text is None
                else object_db.get(object_text)
            )
            object_excluded = (
                object_text in excluded_objects
                if object_text is not None
                else False
            )
            qc_eligibility = (
                "excluded"
                if image_excluded or object_excluded
                else "accepted"
            )
            effective_role = (
                "excluded" if qc_eligibility == "excluded" else role
            )
            rows.append(
                {
                    "source_image": source_image,
                    "object_id": object_text,
                    "batch": image.get("batch"),
                    "label": (
                        image.get("nut_type")
                        if obj is None
                        else obj.get("object_nut_type")
                    ),
                    "sample_kind": image.get("sample_kind"),
                    "protocol_role": effective_role,
                    "cv_group": source_image,
                    "qc_eligibility": qc_eligibility,
                }
            )

    manifest = pd.DataFrame(
        rows,
        columns=PROTOCOL_SPLIT_MANIFEST_COLUMNS,
    )
    checks = assert_no_split_leakage(
        manifest,
        expected_classes=expected_classes,
        calibration_batches=calibration_batches,
        validation_batches=validation_batches,
        test_batches=test_batches,
        strict=False,
    )
    if strict and not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"]].to_dict("records")
        raise ValueError(f"Invalid protocol split assignment: {failed}")
    return manifest, checks


def assert_no_split_leakage(
    manifest: pd.DataFrame,
    *,
    expected_classes=expcfg.PROTOCOL_EXPECTED_CLASSES,
    calibration_batches=expcfg.PROTOCOL_CALIBRATION_BATCHES,
    validation_batches=expcfg.PROTOCOL_VALIDATION_BATCHES,
    test_batches=expcfg.PROTOCOL_TEST_BATCHES,
    strict: bool = True,
) -> pd.DataFrame:
    """Audit grouped roles, exclusions and batch separation."""
    missing = set(PROTOCOL_SPLIT_MANIFEST_COLUMNS).difference(
        manifest.columns
    )
    if missing:
        raise KeyError(
            f"Protocol split manifest is missing columns: {sorted(missing)}"
        )
    checks = []

    def add(check, passed, detail):
        checks.append(
            {"check": check, "passed": bool(passed), "detail": str(detail)}
        )

    image_roles = manifest.groupby("source_image")["protocol_role"].nunique()
    add(
        "source_images_have_one_role",
        (image_roles <= 1).all(),
        f"violations={int((image_roles > 1).sum())}",
    )
    object_rows = manifest.dropna(subset=["object_id"])
    object_roles = object_rows.groupby("object_id")["protocol_role"].nunique()
    add(
        "objects_have_one_role",
        (object_roles <= 1).all(),
        f"violations={int((object_roles > 1).sum())}",
    )
    add(
        "object_ids_are_unique",
        not object_rows["object_id"].duplicated().any(),
        f"duplicates={int(object_rows['object_id'].duplicated().sum())}",
    )
    add(
        "cv_group_is_source_image",
        manifest["cv_group"].astype(str).eq(
            manifest["source_image"].astype(str)
        ).all(),
        "Pixels inherit their object's unique image-level group.",
    )

    expected_classes = {str(value) for value in expected_classes}
    for role in ("calibration", "validation", "test"):
        observed = set(
            manifest.loc[
                manifest["protocol_role"].eq(role),
                "label",
            ].dropna().astype(str)
        )
        add(
            f"{role}_contains_expected_classes",
            expected_classes.issubset(observed),
            f"expected={sorted(expected_classes)}, observed={sorted(observed)}",
        )

    role_batches = {
        "calibration": set(calibration_batches),
        "validation": set(validation_batches),
        "test": set(test_batches),
    }
    for role, allowed in role_batches.items():
        observed = set(
            pd.to_numeric(
                manifest.loc[
                    manifest["protocol_role"].eq(role),
                    "batch",
                ],
                errors="coerce",
            ).dropna().astype(int)
        )
        add(
            f"{role}_batch_separation",
            observed.issubset(allowed),
            f"allowed={sorted(allowed)}, observed={sorted(observed)}",
        )

    excluded_in_analysis = manifest[
        manifest["qc_eligibility"].eq("excluded")
        & manifest["protocol_role"].isin(
            {"calibration", "validation", "test"}
        )
    ]
    add(
        "qc_exclusions_are_propagated",
        excluded_in_analysis.empty,
        f"violations={len(excluded_in_analysis)}",
    )
    batch4_calibration = manifest[
        manifest["protocol_role"].eq("calibration")
        & pd.to_numeric(manifest["batch"], errors="coerce").isin(
            test_batches
        )
    ]
    add(
        "no_test_batch_in_calibration",
        batch4_calibration.empty,
        f"violations={len(batch4_calibration)}",
    )
    mixture_in_pure = manifest[
        manifest["sample_kind"].eq("mixture")
        & manifest["protocol_role"].isin(
            {"calibration", "validation", "test"}
        )
    ]
    add(
        "no_mixture_in_pure_roles",
        mixture_in_pure.empty,
        f"violations={len(mixture_in_pure)}",
    )

    result = pd.DataFrame(checks, columns=PROTOCOL_SPLIT_CHECK_COLUMNS)
    if strict and not bool(result["passed"].all()):
        raise RuntimeError(
            "Protocol split leakage audit failed: "
            f"{result.loc[~result['passed']].to_dict('records')}"
        )
    return result


def build_split_diagnostics(
    split_manifest: pd.DataFrame,
    object_db: Mapping | None = None,
) -> pd.DataFrame:
    """Summarize class counts and object-area distributions by role/batch."""
    rows = split_manifest.dropna(subset=["object_id"]).copy()
    if object_db is None:
        rows["area_pixels"] = np.nan
    else:
        rows["area_pixels"] = rows["object_id"].map(
            {
                str(object_id): obj.get("area_pixels", np.nan)
                for object_id, obj in object_db.items()
            }
        )
    return (
        rows.groupby(
            ["protocol_role", "label", "batch"],
            dropna=False,
        )
        .agg(
            n_objects=("object_id", "nunique"),
            n_images=("source_image", "nunique"),
            area_min=("area_pixels", "min"),
            area_median=("area_pixels", "median"),
            area_max=("area_pixels", "max"),
        )
        .reset_index()
        .loc[:, SPLIT_DIAGNOSTIC_COLUMNS]
    )


def eligible_object_ids(
    split_manifest: pd.DataFrame,
    role: str,
) -> tuple[str, ...]:
    """Return the only object IDs allowed to enter one matrix role."""
    rows = split_manifest[
        split_manifest["protocol_role"].eq(str(role))
        & split_manifest["qc_eligibility"].eq("accepted")
    ]
    return tuple(rows["object_id"].dropna().astype(str).unique())


def build_grouped_folds(
    reference_df: pd.DataFrame,
    *,
    group_col: str = "source_image",
    label_col: str = "label",
    batch_col: str = "batch",
    size_col: str | None = None,
    n_size_bins: int = 3,
    n_splits: int = 2,
    random_state: int = 0,
    require_complete_coverage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign immutable groups to common class/batch-balanced folds.

    The returned assignment is independent of PCA sampling seeds.  A group is
    the validation group in exactly one fold and belongs to the training set of
    every other fold.  Coverage is audited for both sides of every split.
    """
    required = (group_col, label_col, batch_col)
    missing = [column for column in required if column not in reference_df]
    if missing:
        raise KeyError(f"Missing grouped-fold columns: {missing}")

    selected_columns = list(required) + (
        [str(size_col)] if size_col is not None else []
    )
    rows = reference_df.loc[:, selected_columns].dropna().copy()
    rows[group_col] = rows[group_col].astype(str)
    rows[label_col] = rows[label_col].astype(str)
    rows[batch_col] = rows[batch_col].astype(str)
    if rows.empty:
        raise ValueError("Cannot build grouped folds from an empty table.")
    group_nunique = rows.groupby(group_col)[[label_col, batch_col]].nunique()
    if group_nunique.gt(1).any(axis=None):
        raise ValueError("Each group must have exactly one class and one batch.")

    aggregations: dict[str, tuple[str, str]] = {
        label_col: (label_col, "first"),
        batch_col: (batch_col, "first"),
        "n_objects": (group_col, "size"),
    }
    if size_col is not None:
        rows[str(size_col)] = pd.to_numeric(
            rows[str(size_col)], errors="raise"
        ).astype(float)
        aggregations["median_object_size"] = (str(size_col), "median")
    groups = (
        rows.groupby(group_col, as_index=False)
        .agg(**aggregations)
        .sort_values(group_col, kind="mergesort")
        .reset_index(drop=True)
    )
    if size_col is None:
        groups["median_object_size"] = 0.0
    ranked_size = groups["median_object_size"].rank(method="first")
    groups["size_bin"] = pd.qcut(
        ranked_size,
        q=min(max(1, int(n_size_bins)), len(groups)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    n_splits = int(n_splits)
    if n_splits < 2 or len(groups) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} groups, got {len(groups)}."
        )
    expected_labels = set(groups[label_col])
    expected_batches = set(groups[batch_col])
    def assignment_score(assignment: np.ndarray) -> tuple:
        incompleteness = 0
        object_counts = []
        median_sizes = []
        bin_counts = []
        for fold in range(n_splits):
            valid = groups.loc[assignment == fold]
            train = groups.loc[assignment != fold]
            incompleteness += len(expected_labels - set(valid[label_col]))
            incompleteness += len(expected_batches - set(valid[batch_col]))
            incompleteness += len(expected_labels - set(train[label_col]))
            incompleteness += len(expected_batches - set(train[batch_col]))
            object_counts.append(int(valid["n_objects"].sum()))
            median_sizes.append(float(valid["median_object_size"].median()))
            bin_counts.append(
                valid["size_bin"].value_counts().reindex(
                    range(int(groups["size_bin"].max()) + 1), fill_value=0
                ).to_numpy(dtype=int)
            )
        size_bin_imbalance = int(
            np.ptp(np.vstack(bin_counts), axis=0).sum()
        )
        median_size_imbalance = float(
            np.nanmax(median_sizes) - np.nanmin(median_sizes)
        )
        return (
            int(incompleteness),
            int(np.ptp(object_counts)),
            size_bin_imbalance,
            median_size_imbalance,
            tuple(map(int, assignment)),
        )

    candidates: list[np.ndarray] = []
    if n_splits == 2 and len(groups) <= 12:
        # Fix the first group in fold 0 to remove symmetric duplicates.
        for suffix in product(range(n_splits), repeat=len(groups) - 1):
            assignment = np.asarray((0, *suffix), dtype=int)
            if set(assignment) == set(range(n_splits)):
                candidates.append(assignment)
    else:
        rng = np.random.default_rng(int(random_state))
        n_attempts = max(256, min(8192, 128 * len(groups)))
        for _ in range(n_attempts):
            assignment = np.arange(len(groups), dtype=int) % n_splits
            rng.shuffle(assignment)
            candidates.append(assignment)
    if not candidates:
        raise RuntimeError("No grouped-fold partition could be generated.")
    best_assignment = min(candidates, key=assignment_score).copy()

    groups["fold_id"] = best_assignment.astype(int)
    assignment_df = rows.merge(
        groups[[group_col, "fold_id"]],
        on=group_col,
        how="left",
        validate="many_to_one",
    )
    diagnostic_rows = []
    for fold in range(n_splits):
        valid = groups.loc[groups["fold_id"].eq(fold)]
        train = groups.loc[~groups["fold_id"].eq(fold)]
        shared = set(valid[group_col]) & set(train[group_col])
        checks = {
            "validation_has_all_classes": set(valid[label_col]) == expected_labels,
            "validation_has_all_batches": set(valid[batch_col]) == expected_batches,
            "training_has_all_classes": set(train[label_col]) == expected_labels,
            "training_has_all_batches": set(train[batch_col]) == expected_batches,
            "no_shared_groups": not shared,
        }
        diagnostic_rows.append(
            {
                "fold_id": fold,
                "n_validation_groups": int(valid[group_col].nunique()),
                "n_training_groups": int(train[group_col].nunique()),
                "n_validation_objects": int(valid["n_objects"].sum()),
                "median_validation_object_size": float(
                    valid["median_object_size"].median()
                ),
                **checks,
                "coverage_complete": bool(all(checks.values())),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    if require_complete_coverage and not diagnostics["coverage_complete"].all():
        raise RuntimeError(
            "Grouped folds cannot preserve all classes and batches in both "
            f"training and validation: {diagnostics.to_dict('records')}"
        )
    return assignment_df, diagnostics


__all__ = [
    "PROTOCOL_SPLIT_CHECK_COLUMNS",
    "PROTOCOL_SPLIT_MANIFEST_COLUMNS",
    "SPLIT_DIAGNOSTIC_COLUMNS",
    "assert_no_split_leakage",
    "build_protocol_manifest",
    "build_grouped_folds",
    "build_split_diagnostics",
    "eligible_object_ids",
]
