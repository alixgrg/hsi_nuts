# Notebook 06A - SIMCA Pure-Test Evaluation

## Role In The Workflow

`notebooks/06A_simca_pure_test.ipynb` is the first notebook that is allowed to evaluate the held-out pure-test split.

The notebook is intentionally thin: the reusable workflow logic lives in `src/workflows/simca_pure_test.py`, while schema-aware parquet I/O lives in `src/workflows/simca_tables.py`.

It does not select final models. It evaluates the candidates reviewed in notebook 05 on batch 4, using only decisions already fixed before the test:

- train pure target models on batches 1, 2, and 3;
- project pure reference objects from batch 4;
- compute object-level and pixel-level 2-way metrics;
- apply 3-way thresholds selected in notebook 04C validation;
- compute image-level diagnostics;
- write guardrails documenting the pure-test protocol.

Notebook 06B should consume these outputs for final multi-model selection.

## Execution Order

Run this notebook after:

1. `00_building_database.ipynb`
2. `01_database_quality_check.ipynb`
3. `02_matrices_preprocessing.ipynb`
4. `03_pca_exploration_selection.ipynb`
5. `04A_simca_grid_search.ipynb`
6. `04B_simca_optuna_search.ipynb`
7. `04C_simca_concat_refit.ipynb`
8. `05_simca_validation_robustness.ipynb`

Notebook 06A reads:

```text
results/03_pca_<RESULTS_TAG>/pca_selected_preprocessings.parquet
results/04C_simca_concat_refit_<RESULTS_TAG>/candidate_panel.parquet
results/04C_simca_concat_refit_<RESULTS_TAG>/validation_3way_selected_thresholds.parquet
results/05_simca_validation_robustness_<RESULTS_TAG>/track_scoring_flags.parquet
```

and writes:

```text
results/06A_simca_pure_test_<RESULTS_TAG>/
```

## Pure-Test Protocol

The notebook enforces these guardrails:

- `PURE_TEST_TRAIN_BATCHES` must be `[1, 2, 3]`;
- `PURE_TEST_BATCHES` must be `[4]`;
- train and test batches must be disjoint;
- projection filters must use `sample_kind=["pure"]` and `batch=[4]`;
- the 3-way threshold file must already exist before 06A starts;
- upstream 04C/05 inputs must not already contain a `pure_test` evaluation stage.

The guardrail table is saved as:

```text
pure_test_guardrails.parquet
```

## Main Outputs

Core outputs:

- `pure_test_candidate_panel.parquet`: full candidate definitions restored from 04C for the configurations reviewed in 05.
- `pure_test_2way_object_metrics.parquet`: object-level 2-way metrics on batch 4.
- `pure_test_2way_pixel_metrics.parquet`: pixel-level 2-way metrics on batch 4.
- `pure_test_3way_object_metrics.parquet`: object-level 3-way metrics using fixed 04C thresholds.
- `pure_test_metrics_long.parquet`: combined metric table for downstream review.
- `pure_test_object_diagnostics_by_image.parquet`: object-level diagnostics by source image.
- `pure_test_pixel_diagnostics_by_image.parquet`: pixel-level diagnostics by source image.
- `pure_test_3way_object_diagnostics_by_image.parquet`: object-level 3-way diagnostics by source image.
- `pure_test_batch_manifest.parquet`: batch-level execution manifest.
- `pure_test_protocol.parquet`: settings and row-count summary.
- `pure_test_errors.parquet`: candidate-level errors, if any.

Optional heavy outputs:

- `pure_test_objects.parquet`
- `pure_test_pixels.parquet`
- `pure_test_3way_objects.parquet`

By default, the notebook keeps only metric and diagnostic tables, and skips detailed object/pixel/3-way projection tables to limit RAM and disk usage.

For the normal downstream workflow, the indispensable outputs are:

- `pure_test_candidate_panel.parquet`
- `pure_test_metrics_long.parquet`
- `pure_test_2way_object_metrics.parquet`
- `pure_test_2way_pixel_metrics.parquet`
- `pure_test_3way_object_metrics.parquet`
- `pure_test_guardrails.parquet`
- `pure_test_protocol.parquet`
- `pure_test_errors.parquet`

The image-diagnostic tables are strongly recommended for auditability:

- `pure_test_object_diagnostics_by_image.parquet`
- `pure_test_pixel_diagnostics_by_image.parquet`
- `pure_test_3way_object_diagnostics_by_image.parquet`

The detailed projection tables are not required by notebook 06B final selection or notebook 07 mixture application. Keep them disabled for the full pure-test run unless you are debugging a small subset of final candidates.

## How To Use The Notebook

For a standard run, keep:

```python
RUN_PURE_TEST_REFIT = True
USE_EXISTING_PURE_TEST_OUTPUTS = True
PURE_TEST_BATCH_SIZE = 50
MAX_PURE_TEST_CANDIDATES = None
MAX_PURE_TEST_CANDIDATES_PER_TRACK = None
SAVE_BATCH_METRIC_TABLES = True
SAVE_BATCH_OBJECT_TABLES = False
SAVE_BATCH_PIXEL_TABLES = False
SAVE_BATCH_3WAY_OBJECT_TABLES = False
SAVE_COMBINED_OBJECT_TABLES = False
SAVE_COMBINED_PIXEL_TABLES = False
SAVE_COMBINED_3WAY_OBJECT_TABLES = False
```

If RAM is limited, reduce `PURE_TEST_BATCH_SIZE` to `10`, then `5`, then `1`. Batch size controls how many candidate projections are held before the per-batch summaries are returned.

To debug quickly:

```python
MAX_PURE_TEST_CANDIDATES = 20
```

or:

```python
MAX_PURE_TEST_CANDIDATES_PER_TRACK = 5
```

To reload existing outputs without refitting:

```python
RUN_PURE_TEST_REFIT = False
USE_EXISTING_PURE_TEST_OUTPUTS = True
```

## Scientific Logic

Notebook 06A is intentionally stricter than validation notebooks.

The test batch is not used to choose preprocessings, hyperparameters, SIMCA rules, thresholds, Pareto fronts, robustness flags, or final models. It is used only once the candidate panel and 3-way thresholds already exist.

The 2-way metrics answer: if the model must accept or reject each object/pixel, how often does it miss the target class or falsely accept non-target classes?

The 3-way metrics answer: after applying the preselected uncertainty thresholds, how often does the model miss targets, falsely accept non-targets, or defer decisions as uncertain?

Image-level diagnostics are descriptive. They are useful for identifying batch-4 images that are systematically harder, but they should not be used to tune thresholds or hyperparameters inside 06A.

## Associated Modules And Functions

Configuration:

- `src.experiment_config.PURE_TEST_TRAIN_BATCHES`: train batches, expected to be `[1, 2, 3]`.
- `src.experiment_config.PURE_TEST_BATCHES`: held-out pure-test batch, expected to be `[4]`.
- `src.experiment_config.SIMCA_SELECTION_TRACKS`: the four model tracks maintained through the SIMCA workflow.

Input/output:

- `src.io.database_h5.load_nir_uco_h5`: loads object and image databases from HDF5.
- `src.workflows.simca_tables.read_simca_table`: reads parquet outputs and applies the schema inferred from the file name.
- `src.workflows.simca_tables.write_simca_table`: writes compact, schema-normalized parquet outputs.
- `src.workflows.simca_tables.compact_simca_table_for_path`: normalizes output schemas and removes non-applicable columns.

Data filtering and preprocessing:

- `src.data.database.filter_records`: applies metadata filters to the object database.
- `src.data.database.select_wavelength_range_from_database`: optionally restricts the wavelength axis.
- `src.data.database.wavelength_selection_summary`: records wavelength-window metadata.
- `src.workflows.pca_selection.build_pca_preprocessing_configs_by_matrix_family`: rebuilds matrix-family-specific preprocessing dictionaries from notebook 03.

SIMCA refit and metrics:

- `src.workflows.simca.make_target_train_filters`: creates pure target training filters.
- `src.workflows.simca.refit_selected_simca_configs`: refits each candidate and projects the test split.
- `src.workflows.simca_pure_test.run_pure_test_refit_batches`: orchestrates batched 06A refit/projection and builds metric tables.
- `src.workflows.simca_pure_test.select_pure_test_candidate_panel`: combines 05 reviewed IDs with full 04C candidate definitions and checks fixed 3-way thresholds.
- `src.workflows.simca_pure_test.build_pure_test_guardrails`: records and validates the pure-test protocol.
- `src.workflows.simca_pure_test.validate_pure_test_outputs`: checks required 06A metric tables and materializes `pure_test_metrics_long.parquet`.
- `src.workflows.simca_pure_test.summarize_pure_test_outputs`: creates the descriptive pure-test diagnostic summary.
- `src.workflows.simca_pure_test.save_pure_test_outputs`: saves the standard 06A output set and optional heavy projection tables.
- `src.decision.metrics.summarize_object_errors_by_image`: aggregates object decisions by image.
- `src.decision.metrics.summarize_pixel_errors_by_image`: aggregates pixel decisions by image or by model configuration.
- `src.decision.uncertainty.evaluate_three_way_by_config`: applies fixed 3-way thresholds by configuration.
- `src.decision.uncertainty.add_three_way_confidence`: adds confidence diagnostics to 3-way object outputs.

Contracts and guardrails:

- `src.workflows.simca_candidates.normalize_simca_candidate_columns`: canonicalizes candidate columns.
- `src.workflows.simca_candidates.validate_simca_candidate_contract`: validates candidate-table identity/config columns.
- `src.workflows.simca_candidates.validate_simca_evaluation_contract`: validates metric-table identity/track/metric columns.
- `src.workflows.simca_robustness.validate_no_pure_test_inputs`: rejects upstream tables that already contain pure-test outputs.
- `src.workflows.simca_selection_utils.materialize_selection_metrics`: ensures 2-way metrics have consistent derived columns.

## Notes For Downstream Notebook 06B

Notebook 06B should perform final multi-model selection separately for:

- object matrix, 2-way;
- object matrix, 3-way;
- pixel matrix, 2-way;
- pixel matrix, 3-way.

It should use 06A metrics as pure-test evidence, not as a place to retune models.
