from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import experiment_config as expcfg
from src.workflows.simca_calibration_selection import (
    aggregate_threshold_candidates,
    finalize_streamed_selection_audit,
    reduce_threshold_policies_from_checkpoint_8tracks,
    sample_threshold_candidates_for_plot,
    select_threshold_policies_from_candidate_cache_8tracks,
    select_threshold_policy_candidates,
)
from src.workflows.simca_internal_calibration import (
    attach_internal_calibration_runner_group_ids,
    resolve_internal_calibration_checkpoint_run_8tracks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold_rows(model_id: str) -> pd.DataFrame:
    rows = []
    crossfit_metrics = {
        "target_miss_rate": 0.01,
        "false_accept_rate": 0.02,
        "balanced_accuracy": 0.98,
        "max_unit_target_miss_rate": 0.02,
        "max_unit_false_accept_rate": 0.03,
    }
    for evaluation_fold in (0, 1):
        for metric, value in crossfit_metrics.items():
            rows.append(
                {
                    "model_id": model_id,
                    "random_state": 0,
                    "evaluation_fold": evaluation_fold,
                    "decision_scope": "direct",
                    "lower_quantile": None,
                    "upper_quantile": None,
                    "vote_threshold": None,
                    "lower_threshold": 0.0,
                    "upper_threshold": 0.0,
                    "metric": metric,
                    # Existing 03B checkpoints persisted this mixed long value
                    # column as strings. The reducer must remain compatible.
                    "value": str(value),
                }
            )
    rows.append(
        {
            "model_id": model_id,
            "random_state": 0,
            "evaluation_fold": -1,
            "decision_scope": "direct",
            "lower_quantile": None,
            "upper_quantile": None,
            "vote_threshold": None,
            "lower_threshold": 0.0,
            "upper_threshold": 0.0,
            "metric": "target_miss_rate",
            "value": "0.01",
        }
    )
    return pd.DataFrame(rows).reindex(
        columns=expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
    )


def _checkpoint(tmp_path: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    configurations = pd.DataFrame(
        [
            {
                "model_id": "model-1",
                "track_id": "E1",
                "decision_mode": "2way",
                "projection_level": "object_projection",
                "matrix_family": "object_matrix",
                "matrix_method": "object_mean",
                "m": None,
                "balanced_pixel_strategy": "not_applicable",
                "preprocessing": "raw",
                "preprocessing_steps": "raw",
                "sg_window_length": 5,
                "sg_polyorder": 2,
                "random_state": 0,
            }
        ]
    )
    configurations = attach_internal_calibration_runner_group_ids(
        configurations
    )
    runner_group_id = str(configurations["_runner_group_id"].iloc[0])

    run_dir = tmp_path / "run_test"
    chunk_dir = run_dir / "chunks"
    marker_dir = run_dir / "markers"
    chunk_dir.mkdir(parents=True)
    marker_dir.mkdir(parents=True)

    threshold_metrics = _threshold_rows("model-1")
    relative_path = Path("chunks") / f"{runner_group_id}_threshold.parquet"
    shard_path = run_dir / relative_path
    threshold_metrics.to_parquet(shard_path, index=False, compression="zstd")
    signature = "test-signature"
    shard = {
        "name": "threshold_metrics",
        "relative_path": relative_path.as_posix(),
        "row_count": len(threshold_metrics),
        "columns": list(threshold_metrics.columns),
        "file_sha256": _sha256(shard_path),
        "completed_fit_ids": ["fit-1"],
    }
    (marker_dir / f"{runner_group_id}.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "runner_group_id": runner_group_id,
                "completed_fit_ids": ["fit-1"],
                "shards": [shard],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "expected_runner_group_ids": [runner_group_id],
            }
        ),
        encoding="utf-8",
    )
    return (
        run_dir,
        configurations.drop(columns="_runner_group_id"),
        threshold_metrics,
    )


def test_checkpoint_threshold_reduction_is_streamed_and_string_compatible(
    tmp_path,
):
    run_dir, configurations, source_metrics = _checkpoint(tmp_path)
    metrics_path = tmp_path / "threshold_metrics.parquet"
    threshold_audit_path = tmp_path / "threshold_audit.parquet"
    candidate_cache_path = tmp_path / "threshold_candidates.parquet"
    cache_context = {"configuration_hash": "test-configuration"}

    reduced = reduce_threshold_policies_from_checkpoint_8tracks(
        run_dir,
        configurations,
        threshold_metrics_output_path=metrics_path,
        threshold_audit_output_path=threshold_audit_path,
        threshold_candidates_output_path=candidate_cache_path,
        threshold_candidate_cache_context=cache_context,
        verbose=False,
    )

    assert pq.ParquetFile(metrics_path).metadata.num_rows == len(source_metrics)
    assert not reduced["threshold_candidates"].empty
    assert len(reduced["selected_policy_metrics"]) == 1
    assert len(reduced["selected_thresholds"]) == 1
    assert threshold_audit_path.exists()
    assert candidate_cache_path.exists()
    assert not reduced["threshold_audit_summary"].empty

    cached_audit_path = tmp_path / "cached_threshold_audit.parquet"
    cached = select_threshold_policies_from_candidate_cache_8tracks(
        candidate_cache_path,
        configurations,
        threshold_metrics_path=metrics_path,
        threshold_audit_output_path=cached_audit_path,
        threshold_candidate_cache_context=cache_context,
        verbose=False,
    )
    pd.testing.assert_frame_equal(
        cached["selected_policy_metrics"],
        reduced["selected_policy_metrics"],
        check_like=True,
    )
    pd.testing.assert_frame_equal(
        cached["selected_thresholds"],
        reduced["selected_thresholds"],
        check_like=True,
    )
    assert cached_audit_path.exists()

    with pytest.raises(RuntimeError, match="cache context mismatch"):
        select_threshold_policies_from_candidate_cache_8tracks(
            candidate_cache_path,
            configurations,
            threshold_metrics_path=metrics_path,
            threshold_audit_output_path=(
                tmp_path / "invalid_context_audit.parquet"
            ),
            threshold_candidate_cache_context={
                "configuration_hash": "different"
            },
            verbose=False,
        )

    model_audit = pd.DataFrame(
        [
            {
                "selection_level": "model",
                "model_id": "model-1",
                "decision_scope": None,
                "lower_quantile": None,
                "upper_quantile": None,
                "vote_threshold": None,
                "stage": "test",
                "decision": "kept",
                "reason_code": "passed",
                "metric": "",
                "observed_value": None,
                "operator": "",
                "reference_value": None,
                "related_model_id": "",
            }
        ],
        columns=expcfg.INTERNAL_CALIBRATION_SELECTION_AUDIT_COLUMNS,
    )
    final_audit_path = tmp_path / "selection_audit.parquet"
    threshold_rows = pq.ParquetFile(threshold_audit_path).metadata.num_rows
    finalize_streamed_selection_audit(
        threshold_audit_path,
        model_audit,
        final_audit_path,
    )

    assert not threshold_audit_path.exists()
    assert pq.ParquetFile(final_audit_path).metadata.num_rows == (
        threshold_rows + 1
    )


def test_complete_checkpoint_resolution_uses_persisted_fit_ids(tmp_path):
    run_dir, _, _ = _checkpoint(tmp_path)
    context = {
        "protocol_hash": "protocol",
        "pca_selection_fingerprint": "pca",
        "track_contract_hash": "tracks",
        "fold_contract_hash": "folds",
        "configuration_hash": "configurations",
    }
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "runner_contract": (
                "8tracks_v5_compact_crossfit_shared_projection"
            ),
            "schema_version": expcfg.RESULTS_SCHEMA_VERSION,
            "protocol_version": expcfg.PROTOCOL_VERSION,
            **context,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_internal_calibration_checkpoint_run_8tracks(
        tmp_path,
        checkpoint_context=context,
        expected_fit_config_ids=("fit-1",),
    )

    assert resolved == run_dir


def test_policy_values_are_canonicalized_before_seed_aggregation():
    rows = []
    metrics = {
        "target_miss_rate": 0.01,
        "false_accept_rate": 0.02,
        "balanced_accuracy": 0.98,
    }
    for random_state, vote_threshold in enumerate(
        (0.8, "0.8", np.float32(0.8))
    ):
        for evaluation_fold in (0, 1):
            for metric, value in metrics.items():
                rows.append(
                    {
                        "model_id": "model-mixed-vote-dtype",
                        "random_state": random_state,
                        "evaluation_fold": evaluation_fold,
                        "decision_scope": "pixel_to_object",
                        "lower_quantile": None,
                        "upper_quantile": None,
                        "vote_threshold": vote_threshold,
                        "lower_threshold": 0.8,
                        "upper_threshold": 0.8,
                        "metric": metric,
                        "value": value,
                    }
                )

    threshold_metrics = pd.DataFrame(rows).reindex(
        columns=expcfg.INTERNAL_CALIBRATION_THRESHOLD_METRIC_COLUMNS
    )
    # Copy-on-Write makes Series.to_numpy() return a read-only view in the
    # environment used by the notebook. The reducer must never mutate it.
    with pd.option_context("mode.copy_on_write", True):
        candidates = aggregate_threshold_candidates(threshold_metrics)

    assert len(candidates) == 1
    assert candidates.loc[0, "vote_threshold"] == 0.8
    assert candidates.loc[0, "n_seeds"] == 3
    assert candidates.loc[0, "n_folds"] == 2
    assert candidates.loc[0, "n_run_folds"] == 6

    invalid = threshold_metrics.copy()
    invalid["vote_threshold"] = 0.79
    with pytest.raises(RuntimeError, match="Non-configured vote_threshold"):
        aggregate_threshold_candidates(invalid)


def _selection_configuration(
    model_id: str,
    *,
    track_id: str,
    decision_mode: str,
    projection_level: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "random_state": 0,
                "track_id": track_id,
                "decision_mode": decision_mode,
                "projection_level": projection_level,
            }
        ]
    )


def _selection_candidate(
    model_id: str,
    *,
    decision_scope: str,
    decision_mode: str,
    target_miss_rate: float,
    worst_target_miss_rate: float,
    worst_unit_target_miss_rate: float,
    balanced_accuracy: float = np.nan,
    decided_balanced_accuracy: float = np.nan,
) -> pd.DataFrame:
    row = {
        "model_id": model_id,
        "decision_scope": decision_scope,
        "lower_quantile": None,
        "upper_quantile": None,
        "vote_threshold": (
            0.75 if decision_scope == "pixel_to_object" else None
        ),
        "n_seeds": 1,
        "n_folds": expcfg.INTERNAL_CALIBRATION_N_SPLITS,
        "n_run_folds": expcfg.INTERNAL_CALIBRATION_N_SPLITS,
        "target_miss_rate": target_miss_rate,
        "worst_target_miss_rate": worst_target_miss_rate,
        "worst_unit_target_miss_rate": worst_unit_target_miss_rate,
        "false_accept_rate": 0.10,
        "worst_false_accept_rate": 0.20,
        "balanced_accuracy": balanced_accuracy,
    }
    if decision_mode == "3way":
        row.update(
            {
                "target_uncertain_rate": 0.10,
                "uncertain_rate": 0.20,
                "coverage_rate": 0.80,
                "decided_balanced_accuracy": decided_balanced_accuracy,
            }
        )
    return pd.DataFrame([row])


def test_e3_fn_amendment_is_scoped_and_audited():
    configurations = _selection_configuration(
        "model-e3",
        track_id="E3",
        decision_mode="2way",
        projection_level="pixel_projection",
    )
    candidates = pd.concat(
        [
            _selection_candidate(
                "model-e3",
                decision_scope="direct",
                decision_mode="2way",
                target_miss_rate=0.17,
                worst_target_miss_rate=0.24,
                worst_unit_target_miss_rate=0.24,
            ),
            _selection_candidate(
                "model-e3",
                decision_scope="pixel_to_object",
                decision_mode="2way",
                target_miss_rate=0.24,
                worst_target_miss_rate=0.34,
                worst_unit_target_miss_rate=0.34,
                balanced_accuracy=0.80,
            ),
        ],
        ignore_index=True,
    )

    selected, audit = select_threshold_policy_candidates(
        candidates,
        configurations,
    )

    assert set(selected["decision_scope"]) == {
        "direct",
        "pixel_to_object",
    }
    mean_fn = audit.loc[
        audit["stage"].eq("constraint:target_miss_rate")
    ]
    references = mean_fn.set_index("decision_scope")["reference_value"]
    assert references["direct"] == pytest.approx(0.18)
    assert references["pixel_to_object"] == pytest.approx(0.25)
    assert (
        audit["reason_code"]
        .eq("metric_not_identifiable_for_context")
        .any()
    )


def test_e3_amendment_does_not_relax_e7():
    configurations = _selection_configuration(
        "model-e7",
        track_id="E7",
        decision_mode="2way",
        projection_level="pixel_projection",
    )
    candidates = pd.concat(
        [
            _selection_candidate(
                "model-e7",
                decision_scope="direct",
                decision_mode="2way",
                target_miss_rate=0.17,
                worst_target_miss_rate=0.24,
                worst_unit_target_miss_rate=0.24,
            ),
            _selection_candidate(
                "model-e7",
                decision_scope="pixel_to_object",
                decision_mode="2way",
                target_miss_rate=0.24,
                worst_target_miss_rate=0.34,
                worst_unit_target_miss_rate=0.34,
                balanced_accuracy=0.80,
            ),
        ],
        ignore_index=True,
    )

    selected, audit = select_threshold_policy_candidates(
        candidates,
        configurations,
    )

    assert selected.empty
    eliminated = audit.loc[
        audit["reason_code"].eq("mean_fn_above_limit")
    ]
    assert set(eliminated["decision_scope"]) == {
        "direct",
        "pixel_to_object",
    }
    assert eliminated["reference_value"].eq(0.05).all()


def test_pixel_projection_direct_accuracy_is_skipped_only_by_context():
    pixel_configurations = _selection_configuration(
        "model-e4",
        track_id="E4",
        decision_mode="3way",
        projection_level="pixel_projection",
    )
    pixel_candidates = pd.concat(
        [
            _selection_candidate(
                "model-e4",
                decision_scope="direct",
                decision_mode="3way",
                target_miss_rate=0.01,
                worst_target_miss_rate=0.02,
                worst_unit_target_miss_rate=0.02,
            ),
            _selection_candidate(
                "model-e4",
                decision_scope="pixel_to_object",
                decision_mode="3way",
                target_miss_rate=0.01,
                worst_target_miss_rate=0.02,
                worst_unit_target_miss_rate=0.02,
                decided_balanced_accuracy=0.80,
            ),
        ],
        ignore_index=True,
    )
    selected, audit = select_threshold_policy_candidates(
        pixel_candidates,
        pixel_configurations,
    )
    assert set(selected["decision_scope"]) == {
        "direct",
        "pixel_to_object",
    }
    applicability = audit.loc[
        audit["reason_code"].eq(
            "metric_not_identifiable_for_context"
        )
    ]
    assert applicability["decision_scope"].eq("direct").all()

    object_configurations = _selection_configuration(
        "model-e1",
        track_id="E1",
        decision_mode="2way",
        projection_level="object_projection",
    )
    object_candidates = _selection_candidate(
        "model-e1",
        decision_scope="direct",
        decision_mode="2way",
        target_miss_rate=0.01,
        worst_target_miss_rate=0.02,
        worst_unit_target_miss_rate=0.02,
    )
    object_selected, object_audit = select_threshold_policy_candidates(
        object_candidates,
        object_configurations,
    )
    assert object_selected.empty
    assert object_audit["reason_code"].eq(
        "balanced_accuracy_below_limit"
    ).any()


def test_pareto_accuracy_matches_metric_identifiability():
    for track_id in ("E4", "E8"):
        maximize = expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES[
            track_id
        ]["maximize"]
        assert "direct.decided_balanced_accuracy" not in maximize
        assert "pixel_to_object.decided_balanced_accuracy" in maximize
    for track_id in ("E2", "E6"):
        maximize = expcfg.INTERNAL_CALIBRATION_PARETO_OBJECTIVES[
            track_id
        ]["maximize"]
        assert "direct.decided_balanced_accuracy" in maximize


def test_visual_sampling_is_bounded_and_keeps_selected_policy():
    candidates = pd.DataFrame(
        {
            "model_id": [f"model-{index}" for index in range(20)],
            "decision_scope": ["direct"] * 20,
            "lower_quantile": [None] * 20,
            "upper_quantile": [None] * 20,
            "vote_threshold": [None] * 20,
            "target_miss_rate": [index / 100 for index in range(20)],
            "false_accept_rate": [index / 200 for index in range(20)],
        }
    )
    selected = candidates.iloc[[7]].copy()
    catalog = pd.DataFrame(
        {
            "model_id": candidates["model_id"],
            "track_id": ["E1"] * len(candidates),
        }
    )

    sampled = sample_threshold_candidates_for_plot(
        candidates,
        selected,
        catalog,
        max_rows_per_track_scope=5,
    )

    assert len(sampled) <= 6
    assert "model-7" in set(sampled["model_id"])


def test_notebook_03b_does_not_materialize_full_threshold_tables():
    notebook = json.loads(
        Path("notebooks/03B_internal_calibration.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "keep_threshold_metrics_in_memory=False" in source
    assert "reduce_threshold_policies_from_checkpoint_8tracks" in source
    assert (
        "select_threshold_policies_from_candidate_cache_8tracks"
        in source
    )
    assert "threshold_candidate_cache_context" in source
    assert "selection_profile_hash" in source
    assert '"selection_profile_sha256"' in source
    assert "execution_parent_protocol_hash" in source
    assert (
        "resolve_internal_calibration_checkpoint_run_8tracks"
        in source
    )
    assert "checkpoint_context=checkpoint_execution_context" in source
    assert "finalize_streamed_selection_audit" in source
    assert "threshold_metrics = calibration_results" not in source
    assert "pd.read_parquet(output_paths[key])" not in source
    assert "candidate_threshold_matches = np.isclose(" in source
    assert (
        "candidate_object_thresholds != expected_object_thresholds"
        not in source
    )
    assert ".issubset({0.75, 0.80})" not in source


def test_selection_amendment_is_explicit_and_narrow():
    assert expcfg.INTERNAL_CALIBRATION_SELECTION_PARENT_PROFILE_ID == (
        expcfg.INTERNAL_CALIBRATION_CONSTRAINT_PROFILE_ID
    )
    assert expcfg.INTERNAL_CALIBRATION_SELECTION_PROFILE_ID != (
        expcfg.INTERNAL_CALIBRATION_SELECTION_PARENT_PROFILE_ID
    )
    assert expcfg.INTERNAL_CALIBRATION_SELECTION_AMENDMENT_SCOPE == (
        "selection_only"
    )
    assert (
        expcfg.INTERNAL_CALIBRATION_SELECTION_PARENT_PROTOCOL_HASH
        == "5d66e659d7da4e69fa647123058bcea08d33fe0155b6c59d71001820dbc78f9e"
    )
    assert (
        "INTERNAL_CALIBRATION_SELECTION_PARENT_PROTOCOL_HASH"
        not in expcfg.PROTOCOL_CONFIGURATION_KEYS
    )
    assert (
        "INTERNAL_CALIBRATION_SELECTION_AMENDMENT_SCOPE"
        not in expcfg.PROTOCOL_CONFIGURATION_KEYS
    )
    assert set(expcfg.INTERNAL_CALIBRATION_THRESHOLD_OVERRIDES) == {"E3"}
    assert expcfg.INTERNAL_CALIBRATION_THRESHOLD_OVERRIDES["E3"] == {
        "direct": {
            "target_miss_rate": 0.18,
            "worst_target_miss_rate": 0.25,
            "worst_unit_target_miss_rate": 0.25,
        },
        "pixel_to_object": {
            "target_miss_rate": 0.25,
            "worst_target_miss_rate": 0.35,
            "worst_unit_target_miss_rate": 0.35,
        },
    }
