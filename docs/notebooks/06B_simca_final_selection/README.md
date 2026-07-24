# Notebook 06B - SIMCA Final Pareto Selection

## Role In The Workflow

`notebooks/06B_simca_final_selection.ipynb` selects several final SIMCA candidates after validation robustness review and pure-test evaluation.

This notebook does not refit models, tune thresholds, compute weighted scores, or run mixture application. Final selection is performed by Pareto ranking inside each track:

- `object_matrix_2way`
- `object_matrix_3way`
- `pixel_matrix_2way`
- `pixel_matrix_3way`

## Execution Order

Run this notebook after:

1. `05_simca_validation_robustness.ipynb`
2. `06A_simca_pure_test.ipynb`

Required inputs:

```text
results/05_simca_validation_robustness_<RESULTS_TAG>/track_scoring_flags.parquet
results/06A_simca_pure_test_<RESULTS_TAG>/pure_test_candidate_panel.parquet
results/06A_simca_pure_test_<RESULTS_TAG>/pure_test_metrics_long.parquet
results/06A_simca_pure_test_<RESULTS_TAG>/pure_test_guardrails.parquet
```

Optional input:

```text
results/06A_simca_pure_test_<RESULTS_TAG>/pure_test_errors.parquet
```

Outputs are written to:

```text
results/06B_simca_final_selection_<RESULTS_TAG>/
```

## Main Outputs

- `final_selection_pool.parquet`: compact Pareto pool with one primary pure-test metric row per candidate and track.
- `final_selected_models.parquet`: compact final multi-model selection list, ranked within each track.
- `final_selection_summary.parquet`: small count table by track, selection status, preselection status, and Pareto tier.
- `final_selection_guardrails.parquet`: input checks for notebook 05/06A consistency.
- `final_selection_protocol.parquet`: settings and output row counts.

The 06B tables are intentionally compact. They keep selection identifiers, track labels, key model descriptors, Pareto metrics, and selection statuses. Full model configuration details remain available in `pure_test_candidate_panel.parquet` and can be joined by `selected_config_id` when notebook 07 needs to refit or apply the models to mixtures.

## Selection Logic

The workflow is intentionally score-free:

1. Check that pure-test guardrails from notebook 06A passed.
2. Build a compact candidate pool from primary pure-test metrics.
3. Attach validation/robustness/stability flags from notebook 05.
4. Optionally filter candidates based on previous flags.
5. Exclude pure-test refit errors by default.
6. Assign Pareto tiers separately within each track.
7. Select up to `SIMCA_FINAL_TOP_N_PER_TRACK` candidates per track, following Pareto tier and deterministic rate tie-breakers.

The optional previous-flag filter is controlled by:

```python
SIMCA_FINAL_APPLY_PREVIOUS_FLAG_FILTER
SIMCA_FINAL_PREVIOUS_FLAGS_TO_FILTER
```

It is disabled by default. When enabled, it is a second-stage filter on explicit upstream flags, not a score and not a common arbitrary metric threshold.

## Pareto Objectives

For 2-way tracks, Pareto ranking minimizes:

```python
SIMCA_FINAL_2WAY_PARETO_MINIMIZE_COLUMNS = ("fn_rate", "fp_rate")
```

For 3-way tracks, Pareto ranking minimizes:

```python
SIMCA_FINAL_3WAY_PARETO_MINIMIZE_COLUMNS = (
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
)
```

Tie-breakers are deterministic and configured in `src/experiment_config.py`; they are used only to order candidates inside or after Pareto tiers, not to replace the Pareto analysis with a score.

## Optional Diversity Rule

The optional diversity rule is controlled by:

```python
SIMCA_FINAL_APPLY_DIVERSITY
SIMCA_FINAL_DIVERSITY_COLUMNS
```

When enabled, it follows Pareto order but tries to avoid selecting only one preprocessing, one SIMCA rule, or one pixel strategy when viable alternatives exist. It is disabled by default.

## Optional Cross-Track Deduplication

Cross-track deduplication is controlled by:

```python
SIMCA_FINAL_DEDUPLICATE_ACROSS_TRACKS
SIMCA_FINAL_CROSS_TRACK_DEDUP_COL
```

When enabled, the same model can be assigned to the first matching track and later tracks are refilled when possible. It is disabled by default.

## Associated Modules And Functions

From `src.workflows.simca_final_selection`:

- `build_final_selection_guardrails(...)`: checks that 05/06A inputs are available and complete.
- `validate_final_selection_guardrails(...)`: blocks final selection if critical guardrails fail.
- `build_final_selection_pool(...)`: builds the compact score-free Pareto pool.
- `pareto_front_mask(...)`: identifies non-dominated rows for a set of rates to minimize.
- `assign_pareto_tiers(...)`: repeatedly removes non-dominated fronts to create Pareto tiers.
- `select_final_models_by_track(...)`: selects final candidates separately for each track.
- `select_top_with_diversity(...)`: applies the optional greedy diversity rule after Pareto ordering.
- `summarize_final_selection(...)`: creates the compact status summary.
- `build_final_selection_protocol(...)`: records settings and output counts.
- `save_final_selection_outputs(...)`: saves the standard 06B output set.

From `src.workflows.simca_tables`:

- `read_simca_table(...)`: reads schema-aware parquet outputs.
- `write_simca_table(...)`: writes compact schema-aware parquet outputs.
- `compact_simca_table(...)`: keeps only the documented output columns for each table kind.

## Notes For Notebook 07

Notebook 07 should use `final_selected_models.parquet` as the official selected-model list.

When full model configuration is required, notebook 07 should join `final_selected_models.parquet` with `pure_test_candidate_panel.parquet` on `selected_config_id`.
