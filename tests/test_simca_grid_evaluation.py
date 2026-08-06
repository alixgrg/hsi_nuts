import numpy as np
import pandas as pd

from src import experiment_config as expcfg
from src.decision.metrics import apply_locked_margin_decision
from src.workflows.simca_grid_evaluation import (
    run_exhaustive_locked_grid_evaluation,
)


def test_locked_margin_boundaries_match_03c_contract():
    target, uncertain = apply_locked_margin_decision(
        np.array([-1.0, 0.0, 1.0]),
        "3way",
        three_way_lower_threshold=0.0,
        three_way_upper_threshold=1.0,
    )

    assert target.tolist() == [False, False, True]
    assert uncertain.tolist() == [False, True, False]


def _domain_row(domain_id, calibration_id, projection_id, seed, n_components=2):
    return {
        "domain_config_id": domain_id,
        "calibration_id": calibration_id,
        "evaluation_config_id": f"eval_{domain_id}",
        "projection_config_id": projection_id,
        "fit_config_id": f"fit_{domain_id}",
        "source_config_id": f"source_{domain_id}",
        "track_id": "E1",
        "evaluation_track": "object_train__object_projection__2way",
        "parent_track": "object_matrix_2way",
        "decision_mode": "2way",
        "decision_score_type": "simca_margin",
        "matrix_family": "object_matrix",
        "matrix_method": "object_mean",
        "projection_level": "object_projection",
        "projection_matrix_method": "object_mean",
        "m": np.nan,
        "balanced_pixel_strategy": "not_applicable",
        "preprocessing": "raw",
        "preprocessing_steps": "raw",
        "rule_family": "simple",
        "rule_variant": "simple_chi2",
        "limit_source": "chi2",
        "n_components": n_components,
        "alpha": 0.01,
        "sg_window_length": 11,
        "sg_polyorder": 2,
        "position_dilation_radius": 0,
        "direct_2way_threshold": 0.0,
        "secondary_object_threshold": np.nan,
        "three_way_lower_threshold": np.nan,
        "three_way_upper_threshold": np.nan,
        "random_state": seed,
        "calibration_status": "calibrated",
        "schema_version": "test",
        "protocol_version": "test",
        "protocol_hash": "hash",
        "pca_shortlist_id": "pca_test",
    }


def _oof_rows(projection_id, seed, *, finite=True):
    margins = np.array([-2.0, -1.0, 1.0, 2.0])
    rows = []
    for index, (truth, margin) in enumerate(zip([False, False, True, True], margins)):
        rows.append(
            {
                "projection_config_id": projection_id,
                "fold_id": index % 2,
                "random_state": seed,
                "source_image": f"image_{index}",
                "object_id": f"object_{index}",
                "truth": truth,
                "H": 1.0,
                "Q": 1.0 if finite else np.nan,
                "rule_statistic": 1.0,
                "rule_limit": 2.0,
                "normalized_ratio": 0.5,
                "simca_margin": margin,
            }
        )
    return rows


def test_exhaustive_grid_keeps_seed_repetitions_errors_and_all_tracks():
    domain = pd.DataFrame(
        [
            _domain_row("d1", "c1", "p1", 0),
            _domain_row("d2", "c1", "p1", 1),
            _domain_row("d3", "c2", "p2", 0),
            _domain_row("d4", "c3", "p3", 0, n_components=3),
        ]
    )
    object_predictions = pd.DataFrame(
        [
            *_oof_rows("p1", 0),
            *_oof_rows("p1", 1),
            *_oof_rows("p2", 0),
            *_oof_rows("p3", 0, finite=False),
        ]
    )
    pixel_predictions = pd.DataFrame(columns=object_predictions.columns)
    eligibility = pd.DataFrame(
        {
            "evaluation_track": expcfg.SIMCA_EVALUATION_TRACKS,
            "eligibility_status": [
                "eligible",
                "eligible_with_warning",
                "unsupported_internal_calibration",
                "unsupported_domain_shift",
                "eligible",
                "eligible",
                "eligible_with_warning",
                "unsupported_domain_shift",
            ],
        }
    )

    outputs = run_exhaustive_locked_grid_evaluation(
        domain,
        object_predictions,
        pixel_predictions,
        eligibility,
    )

    audit = outputs["technical_audit"]
    assert len(audit) == len(domain)
    assert audit["domain_config_id"].is_unique
    assert audit.set_index("domain_config_id").loc["d4", "technical_status"] == "technical_error"
    metrics = outputs["threshold_metrics"].set_index("calibration_id")
    assert metrics.loc["c1", "n_seeds"] == 2
    assert metrics.loc["c1", "n_domain_configurations"] == 2
    summaries = outputs["pareto_reference"].query("row_type == 'track_summary'")
    assert set(summaries["evaluation_track"]) == set(expcfg.SIMCA_EVALUATION_TRACKS)
    exact_groups = outputs["duplicate_groups"].query(
        "duplicate_kind == 'exact_configuration'"
    )
    assert len(exact_groups) == 1
    assert outputs["calculable_not_acceptable"].empty
