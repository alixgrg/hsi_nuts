import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflows.pca import build_pca_candidate_plan, compare_aligned_loadings
from src.workflows.pca_selection import (
    apply_pca_artifact_review_decisions,
    build_pca_artifact_review_table,
    freeze_pca_shortlist,
    make_pca_selection_config,
    pca_input_fingerprint,
    select_pca_preprocessing_shortlist,
    validate_pca_artifact_review,
    validate_pca_preprocessing_shortlist,
)
from src.workflows.protocol_split import build_grouped_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _task14_tables():
    variants = [
        ("object_mean", "object_mean"),
        ("object_median", "object_median"),
        ("all_pixels", "all_pixels"),
        ("balanced_pixels_random_m10", "balanced_pixels"),
        ("balanced_pixels_center_m10", "balanced_pixels"),
        ("balanced_pixels_random_m20", "balanced_pixels"),
        ("balanced_pixels_center_m20", "balanced_pixels"),
    ]
    matrix_summary = pd.DataFrame(
        [
            {
                "matrix_id": f"calibration_{matrix_id}",
                "protocol_role": "calibration",
                "matrix_method": method,
                "balanced_pixel_strategy": None,
                "wavelength_axis_id": "axis",
                "status": "accepted",
            }
            for matrix_id, method in variants
        ]
        + [
            {
                "matrix_id": "calibration_balanced_pixels_random_m40",
                "protocol_role": "calibration",
                "matrix_method": "balanced_pixels",
                "balanced_pixel_strategy": "random",
                "wavelength_axis_id": "axis",
                "status": "warning",
            },
            {
                "matrix_id": "validation_object_mean",
                "protocol_role": "validation",
                "matrix_method": "object_mean",
                "balanced_pixel_strategy": None,
                "wavelength_axis_id": "axis",
                "status": "accepted",
            },
        ]
    )
    feasibility = pd.DataFrame(
        [
            {"m": m, "strategy": strategy, "status": "accepted"}
            for m in (10, 20)
            for strategy in ("random", "center")
        ]
        + [{"m": 40, "strategy": "random", "status": "warning"}]
    )
    preprocessing = pd.DataFrame(
        [
            {
                "matrix_id": matrix_id,
                "fit_role": "calibration",
                "preprocessing": preprocessing_name,
                "steps": steps,
                "sg_window_length": window,
                "sg_polyorder": polyorder,
                "status": status,
            }
            for matrix_id, _ in variants
            for preprocessing_name, steps, window, polyorder, status in (
                ("raw", "raw", np.nan, np.nan, "accepted"),
                ("sg_d1", "sg_d1", 11, 2, "accepted"),
                ("sg_d1", "sg_d1", 9, 2, "accepted"),
                ("snv", "snv", np.nan, np.nan, "rejected"),
            )
        ]
    )
    return matrix_summary, feasibility, preprocessing


def test_candidate_plan_is_driven_only_by_accepted_task14_outputs():
    plan = build_pca_candidate_plan(
        *_task14_tables(),
        allowed_m=(10, 20),
        sg_window_length=11,
    )
    expected_variants = {
        "object_mean",
        "object_median",
        "all_pixels",
        "balanced_pixels_random_m10",
        "balanced_pixels_center_m10",
        "balanced_pixels_random_m20",
        "balanced_pixels_center_m20",
    }
    assert set(plan["matrix_variant"]) == expected_variants
    assert set(plan["preprocessing"]) == {"raw", "sg_d1"}
    assert plan["candidate_id"].is_unique
    assert plan.loc[plan["matrix_method"].eq("all_pixels"), "m"].isna().all()
    assert not plan["matrix_variant"].str.contains("m40").any()


def test_subspace_stability_is_invariant_to_sign_and_component_permutation():
    reference, _ = np.linalg.qr(
        np.asarray([[1.0, 0.2], [0.4, 1.0], [0.1, 0.3], [0.2, -0.4]])
    )
    permuted = reference[:, [1, 0]] * np.asarray([-1.0, 1.0])
    compared = compare_aligned_loadings(
        [
            {"run_type": "reference", "loadings": reference},
            {"run_type": "group_fold", "loadings": permuted},
        ],
        n_components=2,
    )
    fold = compared.loc[compared["run_type"].eq("group_fold")]
    assert np.allclose(fold["subspace_instability"], 0.0, atol=1e-12)
    assert np.allclose(fold["max_principal_angle_deg"], 0.0, atol=1e-6)


def test_common_two_folds_preserve_classes_batches_and_images():
    reference = pd.DataFrame(
        {
            "source_image": ["almond1", "peanut1", "almond2", "peanut2"],
            "label": ["almond", "peanut", "almond", "peanut"],
            "batch": [1, 1, 2, 2],
        }
    )
    folds, diagnostics = build_grouped_folds(reference, n_splits=2)
    assert folds.groupby("source_image")["fold_id"].nunique().max() == 1
    assert diagnostics["coverage_complete"].all()


def test_review_requires_decision_documentation_fingerprint_and_evidence():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "matrix_family": ["object_matrix", "pixel_matrix"],
            "matrix_variant": ["object_mean", "all_pixels"],
            "matrix_method": ["object_mean", "all_pixels"],
            "m": [np.nan, np.nan],
            "balanced_pixel_strategy": ["not_applicable", "not_applicable"],
            "preprocessing": ["raw", "raw"],
        }
    )
    review = build_pca_artifact_review_table(
        candidates,
        run_fingerprint="run",
        review_pdf_path="review.pdf",
        review_pdf_sha256="a" * 64,
        page_by_candidate={"c1": 1, "c2": 2},
    )
    with pytest.raises(RuntimeError, match="pending"):
        validate_pca_artifact_review(review, expected_run_fingerprint="run")
    review["review_status"] = "reviewed"
    review["review_decision"] = ["accept", "warning"]
    review["review_comment"] = "documented"
    review["reviewer"] = "reviewer"
    review["review_date"] = "2026-08-03"
    validate_pca_artifact_review(review, expected_run_fingerprint="run")
    review.loc[1, "critical_artifact"] = True
    with pytest.raises(RuntimeError, match="critical"):
        validate_pca_artifact_review(review, expected_run_fingerprint="run")


def test_review_decisions_follow_current_run_when_reviewed_pdf_is_identical():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "matrix_family": ["object_matrix", "pixel_matrix"],
            "matrix_variant": ["object_mean", "all_pixels"],
            "matrix_method": ["object_mean", "all_pixels"],
            "m": [np.nan, np.nan],
            "balanced_pixel_strategy": ["not_applicable", "not_applicable"],
            "preprocessing": ["raw", "raw"],
        }
    )
    reviewed_sha256 = "a" * 64
    review = build_pca_artifact_review_table(
        candidates,
        run_fingerprint="new-protocol-run",
        review_pdf_path="review.pdf",
        review_pdf_sha256=reviewed_sha256,
        page_by_candidate={"c1": 1, "c2": 2},
    )
    completed = apply_pca_artifact_review_decisions(
        review,
        decision_groups=(
            {
                "candidate_ids": {"c2"},
                "review_decision": "reject",
                "artifact_codes": "critical_outlier",
                "critical_artifact": True,
                "review_comment": "Critical visual outlier.",
            },
        ),
        reviewed_pdf_sha256=reviewed_sha256,
        reviewer="reviewer",
        review_date="2026-08-03",
        default_review_comment="No critical visual artifact.",
    )
    assert completed["run_fingerprint"].eq("new-protocol-run").all()
    assert completed.set_index("candidate_id")["review_decision"].to_dict() == {
        "c1": "accept",
        "c2": "reject",
    }
    with pytest.raises(RuntimeError, match="Re-review is required"):
        apply_pca_artifact_review_decisions(
            review,
            decision_groups=(),
            reviewed_pdf_sha256="b" * 64,
            reviewer="reviewer",
            review_date="2026-08-03",
            default_review_comment="No critical visual artifact.",
        )


def test_pareto_shortlist_keeps_all_non_dominated_preprocessings_and_freezes_hashes():
    rows = []
    for family in ("object_matrix", "pixel_matrix"):
        for index in range(8):
            row = {
                "matrix_family": family,
                "preprocessing": f"prep_{index}",
                "preprocessing_steps": f"prep_{index}",
                "strict_coverage_pass": True,
                "objective_metrics_complete": True,
                "preprocessing_eligible": True,
                "pareto_front": False,
                "selection_status": "pareto_eligible",
            }
            if family == "object_matrix":
                row.update(
                    {
                        "class_trace_ratio_worst": float(index),
                        "batch_trace_ratio_worst": 0.1,
                    }
                )
            else:
                row.update(
                    {
                        "object_class_trace_ratio_worst": float(index),
                        "object_batch_trace_ratio_worst": 0.1,
                    }
                )
            row["instability_metric_worst"] = 0.05
            row["ncomp_95_worst"] = float(index + 1)
            rows.append(row)
    selected, _, _ = select_pca_preprocessing_shortlist(
        pd.DataFrame(rows),
        config=make_pca_selection_config(max_preprocessings_per_family=None),
    )
    assert selected.groupby("matrix_family")["preprocessing"].nunique().eq(8).all()
    frozen = freeze_pca_shortlist(
        selected,
        protocol_hash="protocol",
        review_hash="review",
        input_hashes={"b": "2", "a": "1"},
    )
    validate_pca_preprocessing_shortlist(
        frozen,
        max_per_family=None,
        expected_families=("object_matrix", "pixel_matrix"),
        expected_protocol_hash="protocol",
        expected_input_fingerprint=pca_input_fingerprint({"a": "1", "b": "2"}),
        expected_review_hash="review",
    )
    assert frozen["shortlist_id"].nunique() == 1

    with pytest.raises(RuntimeError, match="No automatic crowding"):
        select_pca_preprocessing_shortlist(
            pd.DataFrame(rows),
            config=make_pca_selection_config(max_preprocessings_per_family=5),
        )


def test_notebooks_enforce_task15_16_order_and_downstream_hash_lock():
    notebook03 = json.loads(
        (PROJECT_ROOT / "notebooks" / "03_pca_exploration_selection.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source03 = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook03["cells"]
    )
    assert "build_pca_candidate_plan(" in source03
    assert "build_pca_visual_review_pdf(" in source03
    assert source03.index("build_pca_visual_review_pdf(") < source03.index(
        "select_pca_preprocessing_shortlist("
    )
    assert "projection_object_db" not in source03
    assert "PCA_CONFIRMATION_BATCHES" not in source03
    assert "apply_pca_artifact_review_decisions(" in source03
    assert "expected_review_run_fingerprint" not in source03
    assert "reviewed_pdf_sha256=REVIEWED_PDF_SHA256" in source03

    notebook03b = json.loads(
        (PROJECT_ROOT / "notebooks" / "03B_internal_calibration.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source03b = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook03b["cells"]
    )
    assert "expected_protocol_hash=execution_protocol_hash" in source03b
    assert "validate_selection_only_protocol_lineage(" in source03b
    assert "checkpoint_context=checkpoint_execution_context" in source03b
    assert "expected_input_fingerprint=pca_input_hash" in source03b
    assert "expected_review_hash=pca_review_hash" in source03b
    assert "validate_pca_preprocessing_shortlist(" in source03b
