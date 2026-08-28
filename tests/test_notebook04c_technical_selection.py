import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.decision.metrics import component_detection_metrics
from src.workflows.simca_candidates import build_locked_validation_candidate_pool
from src.workflows.simca_candidates import hash_locked_validation_evaluation_rule
from src.workflows.simca_grid_evaluation import (
    build_validation_guardrails,
    evaluate_locked_validation_predictions,
)
from src.workflows.spatial_postprocessing_calibration import (
    build_locked_spatial_validation_outputs,
    decode_boolean_map,
    encode_boolean_map,
)


def test_notebook_04c_is_clean_and_implements_tasks_31_to_33():
    path = Path("notebooks/04C_simca_concat_refit.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for required in (
        "build_locked_validation_candidate_pool",
        "run_locked_simca_validation_refit_checkpointed",
        "evaluate_locked_validation_predictions",
        "build_locked_spatial_validation_outputs",
        "build_validation_guardrails",
        "hash_locked_validation_evaluation_rule",
        "VALIDATION_EVALUATION_RULE_HASH",
        "single_class_image_balanced_accuracy",
        "verify_spatial_postprocessing_lock",
        "VALIDATION_PLAN_HASH",
        "expected_ablation_hash",
        "diagnostic_only",
        "validation_object_predictions.parquet",
        "validation_pixel_predictions.parquet",
        "pixel_maps_manifest.parquet",
        "validation_guardrails.parquet",
        '"tasks": [31, 32, 33]',
    ):
        assert required in source
    for forbidden in (
        "selection_score",
        "composite_score",
        "select_top_by_score",
        "evaluate_three_way_by_config",
        "refit_selected_simca_configs",
        "SIMCA_CONCAT_REFIT_RANDOM_STATE",
    ):
        assert forbidden not in source
    assert source.index("expected_ablation_hash") < source.index("load_nir_uco_h5(")
    assert source.index("verify_spatial_postprocessing_lock(") < source.index(
        "load_nir_uco_h5("
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        code = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(code, filename=f"{path.name}:cell-{index}")
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")


def _domain_row(calibration_id, track, **overrides):
    spec = expcfg.SIMCA_EVALUATION_TRACK_SPECS[track]
    row = {
        "validation_candidate_id": f"validation_{calibration_id}",
        "calibration_id": calibration_id,
        "domain_config_id": f"domain_{calibration_id}",
        "evaluation_config_id": f"eval_{calibration_id}",
        "data_config_id": "data_a",
        "fit_config_id": "fit_a",
        "projection_config_id": f"projection_{calibration_id}",
        "evaluation_track": track,
        "track_id": spec["track_id"],
        "parent_track": spec["parent_track"],
        "decision_mode": spec["decision_mode"],
        "matrix_family": spec["training_matrix_family"],
        "matrix_method": (
            "object_mean"
            if spec["training_matrix_family"] == "object_matrix"
            else "balanced_pixels"
        ),
        "projection_level": spec["projection_level"],
        "projection_matrix_method": (
            "all_pixels"
            if spec["projection_level"] == "pixel_projection"
            else "object_mean"
        ),
        "m": np.nan,
        "balanced_pixel_strategy": "not_applicable",
        "preprocessing": "raw",
        "preprocessing_steps": "raw",
        "rule_variant": "simple_chi2",
        "limit_source": "theoretical_train_fit",
        "n_components": 2,
        "alpha": 0.01,
        "random_state": 0,
        "sg_window_length": 9,
        "sg_polyorder": 2,
        "direct_2way_threshold": 0.0,
        "secondary_object_threshold": np.nan,
        "three_way_lower_threshold": (-0.2 if spec["decision_mode"] == "3way" else np.nan),
        "three_way_upper_threshold": (0.2 if spec["decision_mode"] == "3way" else np.nan),
        "position_dilation_radius": 0,
        "calibration_status": "calibrated_8tracks",
    }
    row.update(overrides)
    return row


def test_candidate_pool_uses_protocol_and_diagnostic_fronts_without_optuna_addition():
    supported_track = expcfg.SIMCA_EVALUATION_TRACKS[0]
    unsupported_track = expcfg.SIMCA_EVALUATION_TRACKS[3]
    domain = pd.DataFrame(
        [
            _domain_row("supported", supported_track),
            _domain_row("not_front", supported_track),
            _domain_row("unsupported", unsupported_track),
        ]
    )
    calibrated = domain.drop(columns="domain_config_id").copy()
    pareto = pd.DataFrame(
        [
            {
                "row_type": "configuration",
                "calibration_id": "supported",
                "evaluation_track": supported_track,
                "eligibility_status": "eligible",
                "technical_status": "calculable",
                "protocol_pareto_front": True,
                "diagnostic_pareto_front": True,
            },
            {
                "row_type": "configuration",
                "calibration_id": "not_front",
                "evaluation_track": supported_track,
                "eligibility_status": "eligible",
                "technical_status": "calculable",
                "protocol_pareto_front": False,
                "diagnostic_pareto_front": False,
            },
            {
                "row_type": "configuration",
                "calibration_id": "unsupported",
                "evaluation_track": unsupported_track,
                "eligibility_status": "unsupported_domain_shift",
                "technical_status": "calculable",
                "protocol_pareto_front": False,
                "diagnostic_pareto_front": True,
            },
        ]
    )
    eligibility = pd.DataFrame(
        {
            "evaluation_track": [supported_track, unsupported_track],
            "eligibility_status": ["eligible", "unsupported_domain_shift"],
        }
    )
    trials = pd.DataFrame({"calibration_id": ["not_front", "supported"]})
    optuna_front = pd.DataFrame({"calibration_id": ["not_front"]})

    pool = build_locked_validation_candidate_pool(
        calibrated,
        domain,
        pareto,
        eligibility,
        optuna_trials=trials,
        optuna_pareto_candidates=optuna_front,
    )

    assert pool["calibration_id"].tolist() == ["supported", "unsupported"]
    assert pool["candidate_front"].tolist() == [
        "protocol_pareto",
        "diagnostic_pareto_unsupported_domain_shift",
    ]
    assert pool["visited_by_optuna"].tolist() == [True, False]
    assert not pool["optuna_pareto"].any()


def test_candidate_pool_blocks_a_parameter_changed_after_03b():
    track = expcfg.SIMCA_EVALUATION_TRACKS[0]
    domain = pd.DataFrame([_domain_row("a", track)])
    calibrated = domain.drop(columns="domain_config_id").copy()
    calibrated.loc[0, "n_components"] = 3
    pareto = pd.DataFrame(
        [{
            "row_type": "configuration",
            "calibration_id": "a",
            "evaluation_track": track,
            "eligibility_status": "eligible",
            "technical_status": "calculable",
            "protocol_pareto_front": True,
            "diagnostic_pareto_front": True,
        }]
    )
    eligibility = pd.DataFrame(
        [{"evaluation_track": track, "eligibility_status": "eligible"}]
    )
    with pytest.raises(RuntimeError, match="differ from frozen 03B"):
        build_locked_validation_candidate_pool(
            calibrated, domain, pareto, eligibility
        )


def _object_predictions(projection_id):
    return pd.DataFrame(
        {
            "projection_config_id": [projection_id] * 4,
            "fit_config_id": ["fit"] * 4,
            "random_state": [0] * 4,
            "training_matrix_family": ["object_matrix"] * 4,
            "projection_level": ["object_projection"] * 4,
            "projection_matrix_method": ["object_mean"] * 4,
            "source_image": ["target", "target", "other", "other"],
            "object_id": ["t1", "t2", "n1", "n2"],
            "batch": [3] * 4,
            "object_area": [10.0] * 4,
            "truth": [True, True, False, False],
            "truth_level": ["pure_reference_object"] * 4,
            "pca_score_pc1": [1.0, 1.1, -1.0, -1.1],
            "pca_score_pc2": [0.0] * 4,
            "H": [0.1] * 4,
            "Q": [0.1] * 4,
            "rule_statistic": [0.1, 0.2, 2.0, 1.5],
            "rule_limit": [1.0] * 4,
            "normalized_ratio": [0.1, 0.2, 2.0, 1.5],
            "simca_margin": [0.9, 0.8, -1.0, -0.5],
        }
    )


def test_validation_metrics_tag_equivalence_without_dropping_candidates():
    track = expcfg.SIMCA_EVALUATION_TRACKS[0]
    candidates = pd.DataFrame(
        [
            _domain_row("a", track, projection_config_id="shared"),
            _domain_row("b", track, projection_config_id="shared"),
        ]
    ).assign(eligibility_status="eligible")
    metrics = evaluate_locked_validation_predictions(
        candidates,
        _object_predictions("shared"),
        pd.DataFrame(),
    )
    overall = metrics.query("aggregation_level == 'overall' and status == 'calculable'")
    assert set(overall["calibration_id"]) == {"a", "b"}
    assert overall["prediction_equivalence_group_id"].nunique() == 1
    assert overall["decision_equivalence_group_id"].nunique() == 1
    assert overall["target_miss_rate"].eq(0.0).all()
    assert overall["false_accept_rate"].eq(0.0).all()
    assert overall["target_miss_rate_ci_high"].notna().all()


def test_pure_single_class_images_use_class_conditional_macro_balanced_accuracy():
    track = expcfg.SIMCA_EVALUATION_TRACKS[0]
    candidate = pd.DataFrame(
        [_domain_row("pure_2way", track, projection_config_id="pure_2way")]
    ).assign(eligibility_status="eligible")
    metrics = evaluate_locked_validation_predictions(
        candidate,
        _object_predictions("pure_2way"),
        pd.DataFrame(),
    )
    by_image = metrics.query("aggregation_level == 'source_image'")
    overall = metrics.query("aggregation_level == 'overall'").iloc[0]

    assert by_image["balanced_accuracy"].isna().all()
    assert overall["macro_image_target_miss_rate"] == pytest.approx(0.0)
    assert overall["macro_image_false_accept_rate"] == pytest.approx(0.0)
    assert overall["macro_image_balanced_accuracy"] == pytest.approx(1.0)
    assert overall["macro_image_decided_balanced_accuracy"] == pytest.approx(1.0)
    assert np.isfinite(overall["macro_image_balanced_accuracy_ci_low"])
    assert np.isfinite(overall["macro_image_balanced_accuracy_ci_high"])

    guardrails = build_validation_guardrails(candidate, metrics)
    candidate_guardrails = guardrails.loc[
        guardrails["calibration_id"].eq("pure_2way")
    ]
    assert set(candidate_guardrails["candidate_status"]) == {"pass"}
    assert not (
        candidate_guardrails["scope"].eq("worst_image")
        & candidate_guardrails["metric"].eq("balanced_accuracy")
    ).any()


def test_pure_single_class_images_compute_decided_macro_balanced_accuracy():
    track = expcfg.SIMCA_EVALUATION_TRACKS[1]
    candidate = pd.DataFrame(
        [_domain_row("pure_3way", track, projection_config_id="pure_3way")]
    ).assign(eligibility_status="eligible")
    predictions = _object_predictions("pure_3way")
    predictions["simca_margin"] = [0.8, 0.0, -0.8, 0.0]

    metrics = evaluate_locked_validation_predictions(
        candidate,
        predictions,
        pd.DataFrame(),
    )
    by_image = metrics.query("aggregation_level == 'source_image'")
    overall = metrics.query("aggregation_level == 'overall'").iloc[0]

    assert by_image["decided_balanced_accuracy"].isna().all()
    assert overall["macro_image_target_miss_rate"] == pytest.approx(0.0)
    assert overall["macro_image_false_accept_rate"] == pytest.approx(0.0)
    assert overall["macro_image_uncertain_rate"] == pytest.approx(0.5)
    assert overall["macro_image_coverage_rate"] == pytest.approx(0.5)
    assert overall["macro_image_decided_balanced_accuracy"] == pytest.approx(1.0)

    guardrails = build_validation_guardrails(candidate, metrics)
    candidate_guardrails = guardrails.loc[
        guardrails["calibration_id"].eq("pure_3way")
    ]
    assert set(candidate_guardrails["candidate_status"]) == {"pass"}
    assert not (
        candidate_guardrails["scope"].eq("worst_image")
        & candidate_guardrails["metric"].eq("decided_balanced_accuracy")
    ).any()


def test_worst_image_uses_scope_specific_uncertainty_limit():
    track = expcfg.SIMCA_EVALUATION_TRACKS[1]
    candidate = pd.DataFrame(
        [_domain_row("image_limit", track, projection_config_id="image_limit")]
    ).assign(eligibility_status="eligible")
    base = _object_predictions("image_limit")
    duplicated = base.copy()
    duplicated["object_id"] = duplicated["object_id"].astype(str) + "_copy"
    predictions = pd.concat([base, duplicated], ignore_index=True)
    target = predictions["truth"].astype(bool)
    predictions.loc[target, "simca_margin"] = [0.8, 0.0, 0.0, 0.0]

    metrics = evaluate_locked_validation_predictions(
        candidate,
        predictions,
        pd.DataFrame(),
    )
    overall = metrics.query("aggregation_level == 'overall'").iloc[0]
    assert overall["uncertain_rate"] == pytest.approx(0.375)
    assert overall["coverage_rate"] == pytest.approx(0.625)

    guardrails = build_validation_guardrails(candidate, metrics)
    candidate_guardrails = guardrails.loc[
        guardrails["calibration_id"].eq("image_limit")
    ]
    assert set(candidate_guardrails["candidate_status"]) == {"pass"}
    worst_uncertainty = candidate_guardrails.loc[
        candidate_guardrails["scope"].eq("worst_image")
        & candidate_guardrails["metric"].eq("uncertain_rate")
    ].iloc[0]
    assert worst_uncertainty["observed_value"] == pytest.approx(0.75)
    assert worst_uncertainty["threshold"] == pytest.approx(
        expcfg.INTERNAL_CALIBRATION_MAX_IMAGE_UNCERTAIN_RATE
    )
    assert "coverage_rate" not in set(candidate_guardrails["metric"])


def test_guardrails_reject_inconsistent_uncertainty_and_coverage():
    track = expcfg.SIMCA_EVALUATION_TRACKS[1]
    candidate = pd.DataFrame(
        [_domain_row("bad_coverage", track, projection_config_id="bad_coverage")]
    ).assign(eligibility_status="eligible")
    metrics = evaluate_locked_validation_predictions(
        candidate,
        _object_predictions("bad_coverage"),
        pd.DataFrame(),
    )
    overall = metrics["aggregation_level"].eq("overall")
    metrics.loc[overall, "coverage_rate"] = 0.123

    with pytest.raises(ValueError, match="coverage_rate must equal"):
        build_validation_guardrails(candidate, metrics)


def test_guardrails_still_reject_a_real_false_acceptance_violation():
    track = expcfg.SIMCA_EVALUATION_TRACKS[0]
    candidate = pd.DataFrame(
        [_domain_row("bad_fp", track, projection_config_id="bad_fp")]
    ).assign(eligibility_status="eligible")
    predictions = _object_predictions("bad_fp")
    predictions.loc[predictions["truth"].eq(False), "simca_margin"] = 0.8
    metrics = evaluate_locked_validation_predictions(
        candidate,
        predictions,
        pd.DataFrame(),
    )
    guardrails = build_validation_guardrails(candidate, metrics)
    candidate_guardrails = guardrails.loc[
        guardrails["calibration_id"].eq("bad_fp")
    ]

    assert set(candidate_guardrails["candidate_status"]) == {
        "calculable_but_not_acceptable"
    }
    failed = candidate_guardrails.loc[
        candidate_guardrails["check_status"].eq("fail")
    ]
    assert "false_accept_rate" in set(failed["metric"])


def test_non_finite_required_guardrail_metric_is_a_technical_error():
    track = expcfg.SIMCA_EVALUATION_TRACKS[0]
    candidate = pd.DataFrame(
        [_domain_row("missing_image", track)]
    ).assign(eligibility_status="eligible")
    metric = pd.DataFrame(
        [{
            "validation_candidate_id": "validation_missing_image",
            "calibration_id": "missing_image",
            "evaluation_track": track,
            "track_id": "E1",
            "decision_mode": "2way",
            "projection_level": "object_projection",
            "random_state": 0,
            "aggregation_level": "overall",
            "status": "calculable",
            "target_miss_rate": 0.0,
            "false_accept_rate": 0.0,
            "balanced_accuracy": 1.0,
        }]
    ).reindex(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    guardrails = build_validation_guardrails(candidate, metric)
    candidate_guardrails = guardrails.loc[
        guardrails["calibration_id"].eq("missing_image")
    ]

    assert set(candidate_guardrails["candidate_status"]) == {
        "technical_failure"
    }
    technical = candidate_guardrails.loc[
        candidate_guardrails["check_status"].eq("technical_error")
    ]
    assert len(technical) == 2
    assert technical["reason"].eq(
        "required_guardrail_metric_non_finite"
    ).all()


def test_validation_evaluation_rule_hash_is_deterministic_and_sensitive(
    monkeypatch,
):
    original_hash = hash_locked_validation_evaluation_rule()
    assert original_hash == hash_locked_validation_evaluation_rule()
    assert len(original_hash) == 64

    changed = {
        mode: tuple(dict(spec) for spec in specs)
        for mode, specs in expcfg.SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS.items()
    }
    changed["2way"] += (
        {
            "rule_id": "hash_sensitivity_probe",
            "scope": "overall",
            "metric": "target_miss_rate",
            "limit_key": "max_fn_rate",
            "comparator": "<=",
            "severity": "warning",
        },
    )
    monkeypatch.setattr(
        expcfg,
        "SIMCA_CONCAT_REFIT_GUARDRAIL_CHECK_SPECS",
        changed,
    )
    assert hash_locked_validation_evaluation_rule() != original_hash


def test_guardrails_keep_unsupported_track_as_diagnostic():
    track = expcfg.SIMCA_EVALUATION_TRACKS[3]
    candidate = pd.DataFrame(
        [_domain_row("unsupported", track)]
    ).assign(eligibility_status="unsupported_domain_shift")
    metric = pd.DataFrame(
        [{
            "validation_candidate_id": "validation_unsupported",
            "calibration_id": "unsupported",
            "evaluation_track": track,
            "track_id": "E4",
            "decision_mode": "3way",
            "projection_level": "pixel_projection",
            "random_state": 0,
            "aggregation_level": "overall",
            "status": "calculable",
            "target_miss_rate": 0.0,
            "false_accept_rate": 0.0,
            "uncertain_rate": 0.0,
            "coverage_rate": 1.0,
            "decided_balanced_accuracy": 1.0,
            "macro_image_target_miss_rate": 0.0,
            "macro_image_false_accept_rate": 0.0,
            "macro_image_uncertain_rate": 0.0,
            "macro_image_coverage_rate": 1.0,
            "macro_image_decided_balanced_accuracy": 1.0,
            "prediction_equivalence_group_id": "p",
            "decision_equivalence_group_id": "d",
        }]
    ).reindex(columns=expcfg.SIMCA_VALIDATION_METRIC_COLUMNS)
    guardrails = build_validation_guardrails(candidate, metric)
    assert set(
        guardrails.loc[
            guardrails["calibration_id"].eq("unsupported"), "candidate_status"
        ]
    ) == {"diagnostic_only"}


def test_boolean_map_encoding_roundtrip():
    mask = np.asarray([[True, False, True], [False, True, False]])
    encoded = encode_boolean_map(mask)
    assert decode_boolean_map(encoded, mask.shape).tolist() == mask.tolist()


def test_explicit_truth_labels_keep_adjacent_objects_distinct():
    truth = np.ones((1, 2), dtype=bool)
    labels = np.asarray([[1, 2]], dtype=int)
    metrics = component_detection_metrics(
        truth,
        truth,
        truth_component_labels=labels,
        connectivity=1,
    )
    assert metrics["n_truth_components"] == 2
    assert metrics["component_recall"] == 1.0


def test_spatial_validation_keeps_raw_post_and_uncertainty_layers():
    track = expcfg.SIMCA_EVALUATION_TRACKS[6]
    candidate = pd.DataFrame(
        [_domain_row("pixel", track, projection_config_id="pixel_projection")]
    )
    pixels = pd.DataFrame(
        {
            "projection_config_id": ["pixel_projection"] * 8,
            "random_state": [0] * 8,
            "source_image": ["peanut3"] * 4 + ["almond3"] * 4,
            "object_id": ["p1"] * 4 + ["a1"] * 4,
            "batch": [3] * 8,
            "row": [0, 0, 1, 1] * 2,
            "col": [0, 1, 0, 1] * 2,
            "simca_margin": [0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8],
        }
    )
    image_db = {
        "peanut3": {
            "is_pure": True,
            "sample_kind": "pure",
            "batch": 3,
            "nut_type": "peanut",
            "labels": np.asarray([[1, 1], [2, 2]], dtype=int),
        },
        "almond3": {
            "is_pure": True,
            "sample_kind": "pure",
            "batch": 3,
            "nut_type": "almond",
            "labels": np.asarray([[1, 1], [1, 1]], dtype=int),
        },
    }
    lock = {
        "selected_parameters": {
            "spatial_candidate_id": "spatial_test",
            "connectivity": 1,
            "morphology_operation": "none",
            "morphology_radius": 0,
            "min_area_pixels": 0,
        },
        "uncertain_pixel_policy": "preserve_as_distinct_immutable_layer",
    }
    encoded = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
    lock["lock_sha256"] = hashlib.sha256(encoded).hexdigest()

    outputs = build_locked_spatial_validation_outputs(
        candidate, pixels, image_db, lock
    )
    manifest = outputs["pixel_maps_manifest"]
    metrics = outputs["spatial_component_metrics"]
    assert len(manifest) == 2
    assert set(metrics["map_variant"]) == {"raw", "locked_postprocessed"}
    assert manifest["truth_level"].eq("pure_image_class_exact").all()
    peanut = manifest.loc[manifest["source_image"].eq("peanut3")].iloc[0]
    uncertain = decode_boolean_map(
        peanut["uncertain_mask"], (peanut["height"], peanut["width"])
    )
    assert not uncertain.any()
