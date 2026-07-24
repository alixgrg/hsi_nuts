import numpy as np
import pandas as pd

from src.workflows.simca_tables import (
    SIMCA_TABLE_COLUMNS,
    compact_simca_table,
    compact_simca_table_for_path,
    concat_nonempty_tables,
    iter_dataframe_batches,
    read_simca_table,
    schema_diagnostics,
    write_simca_table,
)


def test_compact_candidate_panel_resolves_suffixes_and_aliases():
    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "candidate_id": "cand_001",
                "target_class": "peanut",
                "non_target_class": "almond",
                "matrix_family": "object_matrix",
                "matrix_method": "object_mean",
                "training_matrix_id": "object_mean",
                "m": 40,
                "balanced_pixel_strategy": "not_applicable",
                "preprocessing": "snv",
                "rule_variant": "simple_emp_cv",
                "n_components": 4,
                "alpha": 0.01,
                "object_threshold": 0.75,
                "n_candidate_sources_x": 1,
                "n_candidate_sources_y": 2,
                "n_duplicate_rows_x": 1,
                "n_duplicate_rows_y": 2,
                "all_empty": np.nan,
            }
        ]
    )

    out = compact_simca_table(raw, table_kind="candidate_panel")

    assert out.loc[0, "non_target_label"] == "almond"
    assert out.loc[0, "m_effective"] == 40
    assert out.loc[0, "matrix_family"] == "object_matrix"
    assert out.loc[0, "balanced_pixel_strategy_effective"] == "not_applicable"
    assert out.loc[0, "n_candidate_sources"] == 2
    assert out.loc[0, "n_duplicate_rows"] == 2
    assert "non_target_class" not in out.columns
    assert "m" not in out.columns
    assert "balanced_pixel_strategy" not in out.columns
    assert "n_candidate_sources_x" not in out.columns
    assert "n_candidate_sources_y" not in out.columns
    assert "all_empty" not in out.columns


def test_compact_candidate_panel_recovers_missing_matrix_family_from_method():
    raw = pd.DataFrame(
        [
            {
                "candidate_id": "cand_001",
                "candidate_sources": "grid",
                "target_class": "peanut",
                "non_target_label": "almond",
                "matrix_method": "balanced_pixels",
                "training_matrix_id": "balanced_pixel_random_m40",
                "preprocessing": "snv",
                "preprocessing_steps": "snv",
                "model_family": "empirical_cv_rule",
                "rule_variant": "simple_emp_cv",
                "n_components": 4,
                "alpha": 0.01,
                "object_threshold": 0.75,
            }
        ]
    )

    out = compact_simca_table(raw, table_kind="candidate_panel")

    assert out.loc[0, "matrix_family"] == "pixel_matrix"


def test_compact_known_empty_path_returns_stable_schema_without_empty_sentinel():
    out = compact_simca_table_for_path(pd.DataFrame(), "validation_refit_errors.parquet")

    assert out.empty
    assert "_empty" not in out.columns
    assert {"selected_config_id", "evaluation_split", "error"}.issubset(out.columns)


def test_read_write_simca_table_round_trip_compacts_schema(tmp_path):
    path = tmp_path / "pure_test_pixel_diagnostics_by_image.parquet"
    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "candidate_id": "cand_001",
                "selection_track": "pixel_matrix_2way",
                "decision_mode": "2way",
                "metric_level": "pixel_image",
                "target_class": "peanut",
                "non_target_class": "almond",
                "source_image": "B4_img_001",
                "n": 10,
                "fn": 1,
                "fp": 0,
                "balanced_accuracy": 0.95,
                "all_empty": np.nan,
            }
        ]
    )

    saved_path = write_simca_table(raw, path)
    out = read_simca_table(saved_path, required=True)

    assert saved_path == path
    assert out.loc[0, "non_target_label"] == "almond"
    assert "non_target_class" not in out.columns
    assert "all_empty" not in out.columns


def test_concat_and_iter_dataframe_batches_are_empty_safe():
    df = pd.DataFrame({"x": [1, 2, 3]})

    assert concat_nonempty_tables([None, pd.DataFrame()]).empty
    assert concat_nonempty_tables([pd.DataFrame(), df])["x"].tolist() == [1, 2, 3]

    batches = list(iter_dataframe_batches(df, batch_size=2, batch_prefix="chunk"))

    assert [batch_id for batch_id, _, _, _ in batches] == ["chunk_0001", "chunk_0002"]
    assert [(start, stop) for _, start, stop, _ in batches] == [(0, 2), (2, 3)]


def test_compact_batch_metric_path_uses_suffix_schema():
    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "candidate_id": "cand_001",
                "matrix_family": "object_matrix",
                "decision_mode": "2way",
                "metric_level": "object",
                "target_class": "peanut",
                "non_target_class": "almond",
                "matrix_method": "object_mean",
                "preprocessing": "snv",
                "rule_variant": "simple_emp_cv",
                "n_components": 4,
                "alpha": 0.01,
                "object_threshold": 0.75,
                "n": 12,
                "tp": 10,
                "fn": 2,
                "fp": 1,
                "tn": 9,
                "balanced_accuracy": 0.85,
                "unrelated_empty": np.nan,
            }
        ]
    )

    out = compact_simca_table_for_path(raw, "0001_2way_object_metrics.parquet")

    assert "non_target_label" in out.columns
    assert "non_target_class" not in out.columns
    assert "unrelated_empty" not in out.columns
    assert ["selected_config_id", "candidate_id", "matrix_family"] == list(out.columns[:3])


def test_compact_3way_metrics_keeps_3way_columns_and_drops_binary_only_columns():
    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "candidate_id": "cand_001",
                "matrix_family": "pixel_matrix",
                "decision_mode": "3way",
                "metric_level": "object",
                "target_class": "peanut",
                "non_target_label": "almond",
                "matrix_method": "balanced_pixels",
                "preprocessing": "snv",
                "rule_variant": "simple_emp_cv",
                "n_components": 4,
                "alpha": 0.01,
                "object_threshold": 0.75,
                "tp": np.nan,
                "fn": np.nan,
                "fp": np.nan,
                "tn": np.nan,
                "target_miss_rate": 0.05,
                "non_target_false_accept_rate": 0.10,
                "uncertain_rate": 0.20,
                "coverage_rate": 0.80,
                "decided_balanced_accuracy": 0.90,
            }
        ]
    )

    out = compact_simca_table(raw, table_kind="simca_3way_metrics")

    assert "target_miss_rate" in out.columns
    assert "uncertain_rate" in out.columns
    assert "tp" not in out.columns
    assert "fn" not in out.columns


def test_compact_pure_test_image_diagnostics_preserves_source_image_and_track():
    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "candidate_id": "cand_001",
                "selection_track": "pixel_matrix_2way",
                "decision_mode": "2way",
                "metric_level": "pixel_image",
                "target_class": "peanut",
                "non_target_class": "almond",
                "matrix_method": "balanced_pixels",
                "preprocessing": "snv",
                "source_image": "B4_img_001",
                "n": 100,
                "tp": 95,
                "fn": 5,
                "fp": 2,
                "tn": 98,
                "balanced_accuracy": 0.965,
                "all_na_column": np.nan,
            }
        ]
    )

    out = compact_simca_table_for_path(raw, "pure_test_pixel_diagnostics_by_image.parquet")

    assert out.loc[0, "non_target_label"] == "almond"
    assert out.loc[0, "matrix_family"] == "pixel_matrix"
    assert out.loc[0, "source_image"] == "B4_img_001"
    assert out.loc[0, "selection_track"] == "pixel_matrix_2way"
    assert "all_na_column" not in out.columns


def test_compact_pure_test_3way_metrics_uses_3way_schema():
    raw = pd.DataFrame(
        [
            {
                "selected_config_id": "sel_001",
                "candidate_id": "cand_001",
                "selection_track": "object_matrix_3way",
                "decision_mode": "3way",
                "metric_level": "object",
                "target_class": "peanut",
                "non_target_label": "almond",
                "matrix_family": "object_matrix",
                "matrix_method": "object_mean",
                "preprocessing": "snv",
                "target_miss_rate": 0.0,
                "non_target_false_accept_rate": 0.05,
                "uncertain_rate": 0.10,
                "coverage_rate": 0.90,
                "decided_balanced_accuracy": 0.975,
                "tp": np.nan,
                "fn": np.nan,
            }
        ]
    )

    out = compact_simca_table_for_path(raw, "pure_test_3way_object_metrics.parquet")

    assert "target_miss_rate" in out.columns
    assert "non_target_false_accept_rate" in out.columns
    assert "uncertain_rate" in out.columns
    assert "tp" not in out.columns
    assert "fn" not in out.columns


def test_schema_diagnostics_counts_suffix_and_all_na_columns():
    df = pd.DataFrame({"a_x": [1], "a_y": [2], "empty": [np.nan]})

    diagnostics = schema_diagnostics(df)

    assert diagnostics["n_suffix_columns"] == 2
    assert diagnostics["n_all_na_columns"] == 1
    assert diagnostics["suffix_columns"] == "a_x,a_y"


def test_final_selection_tables_use_compact_contracts():
    selected_cols = SIMCA_TABLE_COLUMNS["final_selected_models"]
    pool_cols = SIMCA_TABLE_COLUMNS["final_selection_pool"]

    assert "selection_score" not in selected_cols
    assert "eligibility_status" not in selected_cols
    assert "target_class" not in selected_cols
    assert "candidate_sources" not in selected_cols
    assert len(selected_cols) < len(pool_cols)
    assert len(selected_cols) <= 30
    assert len(pool_cols) <= 42
