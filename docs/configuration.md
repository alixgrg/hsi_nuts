# Project Configuration

`src/experiment_config.py` is the central source of truth for the active experiment protocol used by the deliverable notebooks.

Notebook-level constants may still exist for readability, plotting, or local runtime switches, but split definitions, reduced search grids, and PCA scoring policy should be initialized from this module.

## Shared Experiment Settings

- `DEFAULT_WAVELENGTH_MODE`: spectral namespace used by the current workflow.
- `DEFAULT_RESULTS_TAG`: result-folder suffix used by the active notebooks.
- `TARGET_CLASS`: target nut class for one-class detection.
- `NON_TARGET_LABEL`: non-target class label.
- `REFERENCE_CLASSES`: pure reference classes used in PCA and SIMCA reference workflows.
- `RANDOM_STATE`: default seed for deterministic sampling and scoring diagnostics.

## Batch Protocol

- `PCA_ALLOWED_BATCHES = [1, 2, 3]`: pure batches allowed for PCA exploration and preprocessing selection.
- `SIMCA_TRAIN_BATCHES = [1, 2]`: pure batches used to train validation-stage SIMCA models.
- `SIMCA_VALIDATION_BATCHES = [3]`: pure validation batch.
- `PURE_TEST_TRAIN_BATCHES = [1, 2, 3]`: pure batches used before external pure-test evaluation.
- `PURE_TEST_BATCHES = [4]`: held-out pure test batch.
- `MIXTURE_FINAL_TRAIN_BATCHES = [1, 2, 3, 4]`: pure batches available before final mixture application.

Batch 4 must stay out of notebook 03 when it is used as the external pure-test batch in notebook 04C.

## SIMCA Grid Reduction

- `SIMCA_ALPHA_VALUES = [0.01]`
- `SIMCA_OBJECT_THRESHOLDS = [0.75, 0.80]`

These values encode the reduced grid agreed after the audit. Downstream notebooks should import them instead of redefining local grids.

## SIMCA Tracks And Candidate Contracts

The final model workflow is organized into four explicit tracks:

- `object_matrix_2way`
- `object_matrix_3way`
- `pixel_matrix_2way`
- `pixel_matrix_3way`

The track definitions live in `SIMCA_SELECTION_TRACKS` and `SIMCA_SELECTION_TRACK_SPECS`. Each track combines one matrix family (`object_matrix` or `pixel_matrix`) with one decision mode (`2way` or `3way`).

SIMCA candidate identity is defined by `SIMCA_CANDIDATE_ID_COLUMNS`. The stable id is created by `src.workflows.simca_candidates.simca_candidate_key(...)` and added to tables with `add_simca_candidate_ids(...)` or `deduplicate_simca_candidates(...)`.

Candidate and evaluation output schemas are documented by:

- `SIMCA_PCA_SHORTLIST_REQUIRED_COLUMNS`
- `SIMCA_CANDIDATE_CONFIG_REQUIRED_COLUMNS`
- `SIMCA_CANDIDATE_EVALUATION_REQUIRED_COLUMNS`
- `SIMCA_FINAL_MODEL_SELECTION_REQUIRED_COLUMNS`

The PCA shortlist from notebook 03 must remain scoped by matrix family. Use `build_pca_preprocessing_configs_by_matrix_family(...)` before running grid search or Optuna. This prevents preprocessings selected for `object_matrix` from being applied to `pixel_matrix`, and vice versa, unless the preprocessing appears in both PCA shortlist families.

## SIMCA Output Table Policy

SIMCA notebooks 04A to 05 write compact result tables through `src.workflows.simca_tables`.

The table policy is:

- resolve pandas merge suffixes such as `_x` and `_y`;
- normalize common aliases such as `non_target_class` to `non_target_label`;
- preserve and, when possible, re-infer `matrix_family` from `matrix_method`, `training_matrix_id`, or `selection_track`;
- keep model-defining parameters in their effective form, for example `m_effective` and `balanced_pixel_strategy_effective`;
- drop columns that are entirely non-applicable for a given output;
- preserve stable empty schemas for known empty support tables, such as error logs and optional duplicate-refit outputs.

Scientific metric tables have stricter schemas than detailed object or pixel tables. Detailed projection tables may keep additional diagnostic columns because they are consumed by border/core and downstream visual checks.

Use:

```python
from src.workflows.simca_tables import compact_simca_table_for_path

save_parquet(compact_simca_table_for_path(df, output_path), output_path)
```

for SIMCA result files. When adding a new SIMCA output file, register its file name or suffix in `TABLE_KIND_BY_FILE_NAME` or `TABLE_KIND_BY_FILE_SUFFIX` if the table should have a strict schema. Leave it unregistered only when all non-empty columns are intentionally diagnostic payload.

Notebook 04C can skip the expensive validation refit and reuse saved validation parquet outputs with:

```python
RUN_REFIT = False
USE_EXISTING_VALIDATION_REFIT_OUTPUTS = True
```

This mode still rebuilds the lightweight candidate and duplicate panels, then loads the existing validation metric outputs before optional duplicated-candidate refits. To run only the duplicated-candidate check after a previous full 04C run, use:

```python
RUN_REFIT = False
USE_EXISTING_VALIDATION_REFIT_OUTPUTS = True
REFIT_DUPLICATED = True
```

When `REFIT_DUPLICATED = False`, notebook 04C no longer overwrites existing duplicated-refit parquet files with empty tables.

## SIMCA Robustness Review Policy

Notebook 05 uses the following project-level settings from `src/experiment_config.py`:

- `SIMCA_ROBUSTNESS_RANDOM_STATES`: seeds used by the optional random-state stability refit.
- `SIMCA_ROBUSTNESS_MAX_STABILITY_CANDIDATES_PER_TRACK`: maximum candidates per track sent to the optional seed-stability panel.
- `SIMCA_ROBUSTNESS_PREFER_BALANCED_PIXELS_FOR_STABILITY`: prioritizes `balanced_pixels` candidates when the stability panel is limited.
- `SIMCA_ROBUSTNESS_BORDER_WIDTHS`: border widths tested when detailed pixel tables are available.
- `SIMCA_ROBUSTNESS_MIN_CORE_PIXELS`: minimum core pixels required before a core-only object decision is trusted.
- `SIMCA_ROBUSTNESS_PARETO_EPSILON`: numerical tolerance used by Pareto dominance checks.
- `SIMCA_ROBUSTNESS_WARNING_THRESHOLDS`: warning thresholds used to create notebook-05 flags.
- `SIMCA_ROBUSTNESS_2WAY_SCORE_WEIGHTS`: diagnostic score weights for 2-way metrics.
- `SIMCA_ROBUSTNESS_3WAY_SCORE_WEIGHTS`: diagnostic score weights for 3-way metrics.
- `SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS`: hyperparameters summarized in the ablation diagnostics.
- `SIMCA_ROBUSTNESS_2WAY_PARETO_MINIMIZE_COLUMNS` and `SIMCA_ROBUSTNESS_2WAY_PARETO_MAXIMIZE_COLUMNS`: 2-way Pareto objectives.
- `SIMCA_ROBUSTNESS_3WAY_PARETO_MINIMIZE_COLUMNS` and `SIMCA_ROBUSTNESS_3WAY_PARETO_MAXIMIZE_COLUMNS`: 3-way Pareto objectives.

Notebook 05 is a validation robustness review. It must not consume pure-test outputs and must not perform final model selection. Its outputs should be interpreted as diagnostic evidence for the later pure-test and final multi-model selection stages.

## Matrix And Pixel Sampling Defaults

- `M_BALANCED_PIXELS`: number of sampled pixels per object for balanced pixel matrices.
- `BALANCED_PIXEL_STRATEGIES`: allowed balanced-pixel sampling strategies.
- `REPLACE_BALANCED_PIXELS`: whether balanced pixel sampling uses replacement.
- `CV_N_SPLITS`: default number of grouped CV splits.
- `CV_GROUP_COL`: default grouping column for grouped validation.

Matrix construction is formalized by `src.matrices.matrix_registry.MatrixOutput`, whose contract is `X`, `y`, `metadata`, and `wavelengths`. Existing notebooks can keep using `build_matrix()` as `X, y, metadata`; new code can request wavelengths with `return_wavelengths=True` or use `build_matrix_output(...)`.

## Notebook 02 Result-Table Contracts

Notebook 02 validates required output schemas before saving key result tables.

- `MATRIX_SUMMARY_REQUIRED_COLUMNS`: required columns for `results/02_matrices_<RESULTS_TAG>/matrix_summary.parquet`.
- `PREPROCESSING_SUMMARY_REQUIRED_COLUMNS`: required columns for `results/02_matrices_<RESULTS_TAG>/preprocessing_summary.parquet`.

The contract is enforced with `src.workflows.matrix_preprocessing.validate_required_columns(...)`.

## PCA Shortlist Policy

- `MAX_PCA_PREPROCESSINGS_PER_FAMILY = 5`

Notebook 03 uses this limit through `expcfg.make_pca_selection_config()`. The saved shortlist must contain at most five rows for `object_matrix` and five rows for `pixel_matrix`.

## PCA Scoring Policy

The canonical PCA scoring configuration is built with:

```python
from src import experiment_config as expcfg

PCA_SELECTION_CONFIG = expcfg.make_pca_selection_config()
```

The scoring implementation lives in `src.workflows.pca_selection`, but the project-level parameters live in `src.experiment_config`.

### Family Profiles

`PCA_SELECTION_PROFILES` stores the metric weights and diagnostic metric names for each matrix family.

For `object_matrix`, the score rewards:

- `class_trace_ratio`
- `mahalanobis_pc1_pc2_pc3`

It penalizes:

- `batch_trace_ratio`
- `mean_train_projection_shift_norm`
- `projection_q_deviation`
- `ncomp_95`

For `pixel_matrix`, the score rewards:

- `object_class_trace_ratio`
- `object_over_intra_ratio`

It penalizes:

- `object_batch_trace_ratio`
- `mean_intra_object_trace`
- `mean_train_projection_shift_norm`
- `projection_q_deviation`
- `ncomp_95`

### Scaling And Stability

- `PCA_SELECTION_GROUP_COLS`: comparison groups used for robust metric scaling.
- `PCA_SELECTION_ROBUST_SCALING`: whether to use median/IQR scaling instead of mean/std scaling.
- `PCA_SELECTION_CLIP_QUANTILES`: lower and upper quantiles used before scaling.
- `PCA_SELECTION_BOOTSTRAP_ITERATIONS`: number of bootstrap resamples used to estimate score stability.
- `PCA_SELECTION_STABILITY_PENALTY_WEIGHT`: coefficient subtracted from the raw score using `selection_score_stability_std`.
- `PCA_SELECTION_EPS`: numerical tolerance used in scaling.

### Relative Quality Flags

- `PCA_SELECTION_QUALITY_LOWER_QUANTILE`: lower threshold for weak separation flags.
- `PCA_SELECTION_QUALITY_UPPER_QUANTILE`: upper threshold for high batch/projection/stability flags.
- `PCA_SELECTION_VALIDATION_UPPER_QUANTILE`: upper threshold for projection-shift validation warnings.

These thresholds are distribution-relative by matrix family, which makes the notebook more adaptable to new data than fixed hard-coded limits.

## Maintenance Rules

- Change PCA scoring parameters in `src/experiment_config.py`, not in notebook 03.
- Keep `src.workflows.pca_selection` focused on reusable scoring mechanics.
- Keep notebook 03 focused on orchestration, diagnostics, and saving the shortlist.
- When scoring parameters change, rerun notebook 03 and regenerate `pca_summary.parquet` and `pca_selected_preprocessings.parquet`.
- When the shortlist policy changes, rerun downstream notebooks that consume `pca_selected_preprocessings.parquet`.
